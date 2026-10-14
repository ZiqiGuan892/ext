#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Precompute Whisper audio embeddings and ViT vision embeddings offline.

Why this exists
---------------
MiniCPM-o 5.0 GSPO RL training on 16-card NPU crashes during actor forward with
``507001`` / ``EI0002`` device notify wait 300s timeout. Root cause: multimodal
load imbalance — ranks with 3000-frame audio spend 5+ min in Whisper, ranks without
audio reach LLM decoder layers in seconds. FSDP2's per-layer unshard AllGather
requires all ranks to sync at each layer → fast rank waits for slow rank → 300s
timeout.

Both Whisper (``self.apm``) and ViT (``self.vpm``) are frozen during RL training.
Their output is deterministic — same input → same embedding. This script precomputes
embeddings offline, storing per-chunk audio embeddings and per-image vision
embeddings keyed by SHA-256 of the input mel/pixel tensor content.

At training time, ``minicpmo_5_0_forward.py`` reads these cache files and injects
them directly, skipping the expensive encoder forward entirely.

Usage
-----

    python3 examples/gspo_trainer/data_process/precompute_embeddings.py \
        --parquet /opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8/train.parquet \
        --model_path /opt/huawei/dataset/downloaded_models/MiniCPM-O/MiniCPM-o-5_0_RL \
        --cache_dir /opt/huawei/dataset/ccb_data/data/omni/embed_cache \
        --sampling_rate 16000 --max_slice_nums 1

The script is idempotent: if a cache file already exists, it is skipped. Re-running
after adding new data only computes embeddings for the new files.

Parquet columns read
--------------------
- ``audios``: ``list[str]`` — bare file paths to ``.wav`` files
- ``images``: ``list[dict]`` — each ``{"image": "/path/to/xxx.jpg"}``
- ``videos``: ``list[dict]`` — each ``{"video": path, "fps": ..., "max_frames": ...}``
  (frames are sampled and treated as images)

Cache layout
------------
::

    <cache_dir>/
      audio/
        <sha256>.pt    # Tensor[num_audio_tokens, embed_dim], bfloat16 on CPU
      vision/
        <sha256>.pt    # Tensor[num_vision_tokens, embed_dim], bfloat16 on CPU

Embedding dim
-------------
Both audio and vision embeddings are projected to ``self.llm.config.hidden_size``
(``embed_dim``) by ``audio_projection_layer`` / ``resampler`` respectively. The
cached tensors are the final per-chunk / per-image slices that
``get_audio_embedding`` / ``get_vision_embedding`` would return.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path: resolve repo root for importing verl_omni helpers.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # .../verl-omni


# ---------------------------------------------------------------------------
# Parquet media extraction
# ---------------------------------------------------------------------------

def extract_media_paths(parquet_path: str):
    """Read parquet, return ``(audio_paths, image_paths)`` — unique, sorted.

    ``audio_paths``: all unique strings from the ``audios`` column.
    ``image_paths``: all unique strings from ``images[i]["image"]``.
    Video entries (``videos[i]["video"]``) are NOT included here — they require
    frame sampling and are handled separately by the caller.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)

    audio_paths: set[str] = set()
    image_paths: set[str] = set()

    rows = table.to_pylist()

    for row in rows:
        # audios: list[str]
        for audio_path in row.get("audios") or []:
            if isinstance(audio_path, str) and audio_path:
                audio_paths.add(audio_path)

        # images: list[dict] with {"image": path}
        for img_entry in row.get("images") or []:
            if isinstance(img_entry, dict):
                p = img_entry.get("image")
                if isinstance(p, str) and p:
                    image_paths.add(p)
            elif isinstance(img_entry, str) and img_entry:
                image_paths.add(img_entry)

    return sorted(audio_paths), sorted(image_paths)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def path_hash(path: str) -> str:
    """SHA-256 of a file path string. Used as cache key.

    Path-based keys are deterministic regardless of how the image/audio is
    loaded, which dtype the processor uses, or how slices/chunks are
    arranged — eliminating the content-hash mismatch problem (logs 16-28).
    """
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _load_zarr_audio(zarr_path: str, sampling_rate: int) -> "np.ndarray":
    """Load audio from a zarr.zip::uuid member path.

    Path format: ``/path/to/archive.zarr.zip::member-uuid``
    Uses zarr.storage.ZipStore (v3 API) to handle compressed chunk data.
    """
    import numpy as np
    import zarr

    archive_path, member_id = zarr_path.split("::", 1)

    store = zarr.storage.ZipStore(archive_path, mode="r")
    root = zarr.open(store=store, mode="r")
    member = root[f"{member_id}.wav"]
    audio_np = np.array(member["audio"], dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0 if audio_np.shape[0] < audio_np.shape[-1] else -1)

    return audio_np


# ---------------------------------------------------------------------------
# Audio embedding precompute
# ---------------------------------------------------------------------------

def precompute_audio_embeddings(
    model,
    processor,
    audio_paths: list[str],
    cache_dir: Path,
    sampling_rate: int,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Compute and cache Whisper + projection + pooler embeddings for each audio file.

    Uses multi-threaded IO (librosa/zarr loading) + batched NPU forward for speed.
    Audio files are loaded in parallel by a thread pool, then forwarded in batches
    through the Whisper encoder to maximize NPU utilization.

    Args:
        batch_size: Number of audio files to forward simultaneously through apm.
        num_workers: Number of threads for parallel audio loading.
    """
    import librosa
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    audio_cache = cache_dir / "audio"
    audio_cache.mkdir(parents=True, exist_ok=True)

    device = next(model.apm.parameters()).device
    dtype = next(model.apm.parameters()).dtype

    n_cached = 0
    n_skipped = 0
    n_failed = 0

    # Phase 1: Filter out missing and already-cached files
    to_process = []
    for i, audio_path in enumerate(audio_paths):
        archive_path = audio_path.split("::")[0] if "::" in audio_path else audio_path
        if not os.path.isfile(archive_path):
            n_failed += 1
            continue
        h = path_hash(audio_path)
        cache_file = audio_cache / f"{h}.pt"
        if cache_file.exists():
            n_skipped += 1
            continue
        to_process.append((i, audio_path, cache_file))

    print(f"[audio] {len(to_process)} to compute, {n_skipped} cached, {n_failed} missing", flush=True)

    # Phase 2: Load + process in batches with parallel IO
    def _load_and_extract(audio_path):
        """Load audio and extract mel features. Returns (path, cache_file, features, lens) or None."""
        if "::" in audio_path:
            wav = _load_zarr_audio(audio_path, sampling_rate)
        else:
            wav, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        if wav is None or len(wav) == 0:
            return None

        hop_length = 160
        pad_length = -wav.shape[-1] % hop_length
        if pad_length > 0:
            wav = np.pad(wav, (0, pad_length))

        result = processor.process_audio(audios=wav, sampling_rate=sampling_rate)

        if hasattr(result, "data"):
            audio_features = result.data.get("audio_features")
            audio_feature_lens = result.data.get("audio_feature_lens")
        elif isinstance(result, dict):
            audio_features = result.get("audio_features", result.get("input_features"))
            audio_feature_lens = result.get("audio_feature_lens")
        else:
            audio_features = result
            audio_feature_lens = None

        if audio_features is None or (
            hasattr(audio_features, "numel") and audio_features.numel() == 0
        ):
            return None

        # Normalize to [n_chunks, 80, T] tensor
        if isinstance(audio_features, torch.Tensor):
            pass
        elif isinstance(audio_features, list):
            parts = [v for v in audio_features if v is not None and len(v)]
            if not parts:
                return None
            parts = [v if v.dim() == 3 else v.unsqueeze(0) for v in parts]
            max_len = max(v.shape[-1] for v in parts)
            padded = [
                v if v.shape[-1] == max_len
                else torch.nn.functional.pad(v, (0, max_len - v.shape[-1]))
                for v in parts
            ]
            audio_features = torch.cat(padded, dim=0)
        else:
            audio_features = torch.as_tensor(audio_features)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)

        # Normalize lens
        if isinstance(audio_feature_lens, list):
            parts_lens = [
                torch.as_tensor(v).flatten()
                for v in audio_feature_lens
                if v is not None and (isinstance(v, torch.Tensor) or len(v) > 0)
            ]
            if parts_lens:
                audio_feature_lens = torch.cat(parts_lens, dim=0).long()
            else:
                audio_feature_lens = None
        elif audio_feature_lens is not None:
            if not isinstance(audio_feature_lens, torch.Tensor):
                audio_feature_lens = torch.as_tensor(audio_feature_lens)
            audio_feature_lens = audio_feature_lens.flatten().long()
        else:
            audio_feature_lens = None

        if audio_feature_lens is None:
            audio_feature_lens = torch.full(
                (audio_features.shape[0],), audio_features.shape[-1], dtype=torch.long
            )

        return audio_features, audio_feature_lens

    # Process in batches: load batch_size files in parallel, then forward together
    import time
    done = 0
    total_io_time = 0.0
    total_fwd_time = 0.0
    total_save_time = 0.0
    batch_count = 0
    t_start_all = time.time()

    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start : batch_start + batch_size]

        # Parallel IO loading
        t_io_start = time.time()
        loaded = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_load_and_extract, audio_path): (idx, audio_path, cache_file)
                for idx, audio_path, cache_file in batch
            }
            for future in as_completed(futures):
                idx, audio_path, cache_file = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        loaded[idx] = (audio_path, cache_file, result)
                    else:
                        n_failed += 1
                        done += 1
                except Exception as exc:
                    print(f"[audio] FAILED: {audio_path}: {type(exc).__name__}: {exc}", flush=True)
                    n_failed += 1
                    done += 1
        t_io = time.time() - t_io_start
        total_io_time += t_io

        if not loaded:
            continue

        # Forward each loaded file through apm individually (different chunk counts)
        # But batch the IO loading above — the real win is parallel file reads
        for idx in sorted(loaded.keys()):
            audio_path, cache_file, (audio_features, audio_feature_lens) = loaded[idx]
            try:
                n_chunks = audio_features.shape[0]
                batch_bs, _, max_mel_seq_len = audio_features.shape
                max_seq_len = (max_mel_seq_len - 1) // 2 + 1

                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype)
                    .unsqueeze(0)
                    .expand(batch_bs, max_seq_len)
                )
                lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_bs, max_seq_len)
                padding_mask = seq_range >= lengths_expand
                audio_attention_mask_ = padding_mask.view(
                    batch_bs, 1, 1, max_seq_len
                ).expand(batch_bs, 1, max_seq_len, max_seq_len)
                audio_attention_mask = audio_attention_mask_.to(dtype=dtype, device=device)
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                mel_on_device = audio_features.to(dtype=dtype, device=device)
                t_fwd_start = time.time()
                audio_states = model.apm(
                    mel_on_device,
                    output_hidden_states=True,
                    attention_mask=audio_attention_mask,
                ).hidden_states[model.audio_encoder_layer]

                audio_embeds = model.audio_projection_layer(audio_states)
                audio_embeds = audio_embeds.transpose(1, 2)
                audio_embeds = model.audio_avg_pooler(audio_embeds)
                audio_embeds = audio_embeds.transpose(1, 2)

                _, feature_lens_after_pooling = model._get_feat_extract_output_lengths(
                    audio_feature_lens.to(device)
                )
                num_audio_tokens = feature_lens_after_pooling
                if hasattr(num_audio_tokens, "cpu"):
                    num_audio_tokens = num_audio_tokens.cpu()

                chunk_embeds = []
                for chunk_idx in range(n_chunks):
                    n_tok = int(num_audio_tokens[chunk_idx].item())
                    emb = audio_embeds[chunk_idx, :n_tok, :].detach().cpu().to(torch.bfloat16)
                    chunk_embeds.append(emb)
                t_fwd = time.time() - t_fwd_start
                total_fwd_time += t_fwd

                t_save_start = time.time()
                torch.save(chunk_embeds, cache_file)
                t_save = time.time() - t_save_start
                total_save_time += t_save

                n_cached += 1
                done += 1

                if done % 100 == 0 or done <= 5:
                    elapsed = time.time() - t_start_all
                    files_per_sec = done / elapsed if elapsed > 0 else 0
                    print(
                        f"[audio {done}/{len(to_process)}] COMPUTED ({n_chunks} chunks): "
                        f"speed: {files_per_sec:.1f} files/s | "
                        f"IO: {total_io_time/done*1000:.0f}ms/f | "
                        f"NPU: {total_fwd_time/done*1000:.0f}ms/f | "
                        f"save: {total_save_time/done*1000:.0f}ms/f | "
                        f"elapsed: {elapsed:.0f}s | "
                        f"ETA: {(len(to_process)-done)/files_per_sec/60:.0f}min",
                        flush=True,
                    )

            except Exception as exc:
                print(f"[audio] FAILED: {audio_path}: {type(exc).__name__}: {exc}", flush=True)
                n_failed += 1
                done += 1

    elapsed_total = time.time() - t_start_all
    print(
        f"\n[audio] Timing: total={elapsed_total:.0f}s | "
        f"IO={total_io_time:.0f}s ({total_io_time/elapsed_total*100:.0f}%) | "
        f"NPU={total_fwd_time:.0f}s ({total_fwd_time/elapsed_total*100:.0f}%) | "
        f"save={total_save_time:.0f}s ({total_save_time/elapsed_total*100:.0f}%) | "
        f"throughput={done/elapsed_total:.1f} files/s",
        flush=True,
    )

    return n_cached, n_skipped, n_failed


# ---------------------------------------------------------------------------
# Vision embedding precompute
# ---------------------------------------------------------------------------

def precompute_vision_embeddings(
    model,
    processor,
    image_paths: list[str],
    cache_dir: Path,
    max_slice_nums: int,
):
    """Compute and cache ViT + resampler embeddings for each image file.

    For each image file:
    1. ``PIL.Image.open(path).convert("RGB")``
    2. ``processor.process_image(images=[image], return_tensors="pt", max_slice_nums=...)``
       → ``pixel_values`` + ``tgt_sizes``
    3. Hash pixel tensor → cache key
    4. ``model.vpm(pixel_values, tgt_sizes, ...)`` → ``vision_embedding``
    5. ``model.resampler(vision_embedding, tgt_sizes)`` → projected embedding
    6. Split per image → store ``<cache_dir>/vision/<hash>.pt``
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    vision_cache = cache_dir / "vision"
    vision_cache.mkdir(parents=True, exist_ok=True)

    device = next(model.vpm.parameters()).device
    dtype = next(model.vpm.parameters()).dtype

    downsample_mode = getattr(model.config, "downsample_mode", "16x")
    use_vit_merger = downsample_mode != "4x"

    n_cached = 0
    n_skipped = 0
    n_failed = 0

    for i, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            print(f"[vision {i+1}/{len(image_paths)}] MISSING: {image_path}", flush=True)
            n_failed += 1
            continue

        try:
            # 1. Load image — MUST use process_mm_info (same as training dataset
            # omni_rl_datasets.py:74) to ensure identical image loading/resize.
            # Image.open + process_image produces different pixel values than
            # process_mm_info + process_image (log 28-29, 2026-09-16).
            from qwen_omni_utils import process_mm_info

            messages = [{"role": "user", "content": [{"type": "image", "image": image_path}]}]
            _, images, _ = process_mm_info(messages, use_audio_in_video=False)
            if not images:
                print(f"[vision {i+1}/{len(image_paths)}] NO IMAGE LOADED: {image_path}", flush=True)
                n_failed += 1
                continue

            # 2. Process via checkpoint processor — same call as training's
            # processor.__call__ → process_image
            proc_result = processor.process_image(
                images=images, return_tensors="pt", max_slice_nums=max_slice_nums
            )

            # Extract pixel_values and tgt_sizes
            if isinstance(proc_result, dict):
                pixel_values_raw = proc_result.get("pixel_values")
                tgt_sizes_raw = proc_result.get("tgt_sizes")
            else:
                pixel_values_raw = getattr(proc_result, "pixel_values", None)
                tgt_sizes_raw = getattr(proc_result, "tgt_sizes", None)

            if pixel_values_raw is None:
                print(f"[vision {i+1}/{len(image_paths)}] NO PIXELS: {image_path}", flush=True)
                n_failed += 1
                continue

            # pixel_values: per-sample list of lists of [3, H, W] tensors,
            # matching what training sees (minicpmo_5_0_forward.py:740 —
            # "per-sample list of lists of [3, H, W] tensors"). We process
            # 1 image at a time → outer list has 1 entry → unwrap to that
            # sample's slice list before hashing/forward.
            if isinstance(pixel_values_raw, torch.Tensor):
                pixel_values_list = [pixel_values_raw]
            elif isinstance(pixel_values_raw, list):
                if pixel_values_raw and isinstance(pixel_values_raw[0], list):
                    # Per-sample wrapper present — take first sample's slices
                    pixel_values_list = list(pixel_values_raw[0])
                else:
                    # Already a flat slice list
                    pixel_values_list = list(pixel_values_raw)
            else:
                pixel_values_list = list(pixel_values_raw) if pixel_values_raw else []

            # Filter to actual tensors (skip sentinels like [])
            pixel_values_list = [
                pv for pv in pixel_values_list if isinstance(pv, torch.Tensor)
            ]

            # tgt_sizes: per-sample list of [N, 2] tensors — same structure
            if isinstance(tgt_sizes_raw, torch.Tensor):
                tgt_sizes_list = [tgt_sizes_raw]
            elif isinstance(tgt_sizes_raw, list):
                if tgt_sizes_raw and isinstance(tgt_sizes_raw[0], list):
                    tgt_sizes_list = list(tgt_sizes_raw[0])
                else:
                    tgt_sizes_list = list(tgt_sizes_raw)
            else:
                tgt_sizes_list = list(tgt_sizes_raw) if tgt_sizes_raw else []

            # Filter valid tensors
            tgt_sizes_list = [
                ts for ts in tgt_sizes_list if isinstance(ts, torch.Tensor)
            ]

            # 3. Compute hash for cache lookup — per-slice granularity.
            #
            # Training-side _lookup_vision_cache also hashes per-slice, because
            # a training sample may contain multiple images and hashing the
            # concatenated pixels would never match any single-image precompute
            # entry. Per-slice hashing makes the cache lookup independent of how
            # many images a sample carries.
            if not pixel_values_list or not tgt_sizes_list:
                print(
                    f"[vision {i+1}/{len(image_paths)}] NO TENSORS "
                    f"(pixels={len(pixel_values_list)}, tgt_sizes={len(tgt_sizes_list)}): {image_path}",
                    flush=True,
                )
                n_failed += 1
                continue

            # Normalize each slice to [3, H, W]
            normalized_slices = [
                pv if pv.dim() == 3 else pv.unsqueeze(0)
                for pv in pixel_values_list
            ]

            # Per-file cache key — path hash, not content hash
            h = path_hash(image_path)
            cache_file = vision_cache / f"{h}.pt"
            if cache_file.exists():
                n_skipped += 1
                if (i + 1) % 50 == 0 or i == 0:
                    print(
                        f"[vision {i+1}/{len(image_paths)}] CACHED: {image_path}",
                        flush=True,
                    )
                continue

            # 4. ViT forward (mirrors modeling_minicpmo.py:563-619)
            # Forward all slices together — Merger returns a list of per-slice embeddings
            tgt_sizes = torch.vstack(tgt_sizes_list).type(torch.int32)
            all_pixel_values = torch.concat(normalized_slices, dim=-1).unsqueeze(0).to(dtype)
            cu_seqlens = F.pad(
                torch.cumsum(tgt_sizes[:, 0] * tgt_sizes[:, 1], dim=0, dtype=torch.int32).to(device),
                (1, 0),
            )
            max_seqlen = int(torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item())

            all_pixel_values = all_pixel_values.to(device=device)

            vision_outputs, tgt_sizes_out = model.vpm(
                all_pixel_values,
                tgt_sizes=tgt_sizes,
                cu_seqlens=cu_seqlens,
                max_seqlens=max_seqlen,
                vit_merger=model.vit_merger if use_vit_merger else None,
                insert_layer_id=model.insert_layer_id if use_vit_merger else None,
                model_type=model.vision_model_type,
            )
            vision_embedding = vision_outputs.last_hidden_state
            vision_embedding = model.resampler(vision_embedding, tgt_sizes_out)

            # 5. Store per-file: list of per-slice embeddings (path-based key)
            if isinstance(vision_embedding, list):
                emb_list = [t.detach().cpu().to(torch.bfloat16) for t in vision_embedding]
            else:
                emb_list = [vision_embedding.detach().cpu().to(torch.bfloat16)]
            torch.save(emb_list, cache_file)

            n_cached += 1
            print(
                f"[vision {i+1}/{len(image_paths)}] COMPUTED: {image_path}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[vision {i+1}/{len(image_paths)}] FAILED: {image_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            n_failed += 1

    return n_cached, n_skipped, n_failed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str, device: str = "npu"):
    """Load MiniCPMO model + processor from checkpoint.

    We only need the encoders (apm, vpm, resampler, audio_projection_layer,
    audio_avg_pooler, vit_merger). The LLM backbone (34.66B params) is NOT
    needed for precompute — we skip loading it to save memory.

    However, the checkpoint's ``__init__`` creates the full model including LLM.
    On NPU with ~60GB free this should be fine (encoders are < 1B params; LLM
    is 34.66B but we move it to CPU immediately and never call it).
    """
    import torch
    import torch_npu  # noqa: F401 — registers NPU backend
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoImageProcessor,
    )

    # Monkey-patch AutoImageProcessor.register to tolerate the string key
    # passed by MiniCPM-o 5.0 RL remote code (processing_minicpmo.py:478).
    # Newer transformers (4.5x+) require a config class as the first arg and
    # accesses ``key.__module__``; passing a string raises AttributeError.
    # We only need the processor for tokenization/image preprocessing, not for
    # auto-loading ``MiniCPMVImageProcessor`` via the registry, so skipping the
    # string-key registration is safe here.
    _orig_register = AutoImageProcessor.register

    def _safe_register(*args, **kwargs):
        if args and isinstance(args[0], str):
            return
        return _orig_register(*args, **kwargs)

    AutoImageProcessor.register = _safe_register

    print(f"[load] Loading model from {model_path}...", flush=True)

    # Load processor first (small, CPU)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load full model — we need apm/vpm/resampler which are only created in __init__
    # with init_vision=True and init_audio=True. Loading with torch_dtype=bfloat16
    # to match training. The LLM backbone is loaded but will not be called.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Move encoders to NPU, keep LLM on CPU (saves NPU memory)
    npu_device = torch.device(device)
    for name in ("apm", "audio_projection_layer", "audio_avg_pooler",
                 "vpm", "resampler", "vit_merger"):
        sub = getattr(model, name, None)
        if sub is not None:
            sub.to(npu_device)
            print(f"[load] {name} → {device}", flush=True)

    # Verify encoders are on device
    for name in ("apm", "vpm"):
        sub = getattr(model, name, None)
        if sub is not None:
            first_param = next(sub.parameters(), None)
            if first_param is not None:
                print(
                    f"[load] {name} on {first_param.device}, "
                    f"dtype={first_param.dtype}, "
                    f"params={sum(p.numel() for p in sub.parameters())/1e6:.1f}M",
                    flush=True,
                )

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute Whisper + ViT embeddings for MiniCPM-o 5.0 RL training."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to train.parquet (or validation.parquet).",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to MiniCPM-o 5.0 checkpoint directory.",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Output cache directory. Will create audio/ and vision/ subdirs.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Audio sampling rate (Whisper requires 16000). Default: 16000.",
    )
    parser.add_argument(
        "--max_slice_nums",
        type=int,
        default=1,
        help="Image slice count for processor.process_image. Must match training config.",
    )
    parser.add_argument(
        "--device",
        default="npu",
        help="Device for encoder forward (npu or cpu). Default: npu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch_size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Audio sampling rate (Whisper requires 16000). Default: 16000.",
    )        
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip audio embedding precompute.",
    )
    parser.add_argument(
        "--skip_vision",
        action="store_true",
        help="Skip vision embedding precompute.",
    )

    args = parser.parse_args()

    # Validate parquet exists
    if not os.path.isfile(args.parquet):
        print(f"ERROR: parquet not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Ensure subdirs exist. Do NOT auto-clear — running validation precompute
    # after train would delete train's results. If hash function changed and
    # stale files need clearing, delete the cache dir manually first:
    #   rm -rf /opt/.../embed_cache/{audio,vision}
    for subdir in ("audio", "vision"):
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)
    # 1. Extract media paths from parquet
    print(f"[scan] Reading parquet: {args.parquet}", flush=True)
    audio_paths, image_paths = extract_media_paths(args.parquet)
    print(
        f"[scan] Found {len(audio_paths)} unique audio files, "
        f"{len(image_paths)} unique image files",
        flush=True,
    )

    if not audio_paths and not image_paths:
        print("[scan] No media files found in parquet. Nothing to do.", flush=True)
        sys.exit(0)

    # 2. Load model + processor
    model, processor = load_model_and_processor(args.model_path, args.device)

    # 3. Precompute audio embeddings
    if audio_paths and not args.skip_audio:
        print(f"\n[audio] Precomputing {len(audio_paths)} audio embeddings...", flush=True)
        a_cached, a_skipped, a_failed = precompute_audio_embeddings(
            model, processor, audio_paths, cache_dir, args.sampling_rate,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print(
            f"[audio] Done: {a_cached} computed, {a_skipped} cached (skipped), "
            f"{a_failed} failed",
            flush=True,
        )
    else:
        print("[audio] Skipped (no audio files or --skip_audio)", flush=True)

    # 4. Precompute vision embeddings
    if image_paths and not args.skip_vision:
        print(f"\n[vision] Precomputing {len(image_paths)} vision embeddings...", flush=True)
        v_cached, v_skipped, v_failed = precompute_vision_embeddings(
            model, processor, image_paths, cache_dir, args.max_slice_nums,
        )
        print(
            f"[vision] Done: {v_cached} computed, {v_skipped} cached (skipped), "
            f"{v_failed} failed",
            flush=True,
        )
    else:
        print("[vision] Skipped (no image files or --skip_vision)", flush=True)

    print("\n[done] Precompute complete.", flush=True)
    print(f"  Cache dir: {cache_dir}", flush=True)
    print(f"  Audio:  {len(list((cache_dir / 'audio').glob('*.pt')))} files" if (cache_dir / "audio").exists() else "  Audio:  0 files", flush=True)
    print(f"  Vision: {len(list((cache_dir / 'vision').glob('*.pt')))} files" if (cache_dir / "vision").exists() else "  Vision: 0 files", flush=True)
    print(
        "\nNext: set MINICPMO_EMBED_CACHE_DIR=" + str(cache_dir)
        + " in the launch script and re-run training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
