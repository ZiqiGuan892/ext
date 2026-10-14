"""Precompute Whisper + Qwen3-VL embeddings for Qwen3.6-35B-A3B-whisper RL training.

Adapted from MiniCPM-o 5.0's precompute_embeddings.py for the new model structure.

Key differences from MiniCPM-o 5.0:
  - Audio encoder: ``model.audio_encoder`` (WhisperEncoder from transformers) instead of ``model.apm``
  - Audio projection: ``model.audio_projection`` (2-layer MLP with GELU) instead of ``model.audio_projection_layer`` (with ReLU)
  - No ``audio_avg_pooler`` — the new model uses ``nn.AvgPool1d`` inline during training forward
  - Vision encoder: ``model.visual`` (Qwen3_VisionTransformer from vllm) instead of ``model.vpm`` (SiglipVisionTransformer)
  - Vision merger: ``model.visual.merger`` instead of ``model.vit_merger`` + ``model.resampler``
  - No ``model.resampler`` — Qwen3-VL merger is built into the vision transformer

Usage::

    python3 examples/data_process/precompute_embeddings_qwen36.py \\
        --parquet /path/to/train.parquet \\
        --model_path /path/to/Qwen3.6-35B-A3B-whisper-0814-rl \\
        --cache_dir /path/to/embed_cache_qwen36 \\
        --sampling_rate 16000 --max_slice_nums 1 --batch_size 16 --num_workers 16

Cache structure::

    <cache_dir>/
      audio/
        <sha256>.pt    # list[Tensor[n_audio_tokens, 2048]], bfloat16 on CPU
      vision/
        <sha256>.pt    # list[Tensor[n_vision_tokens, 2048]], bfloat16 on CPU

Both audio and vision embeddings are projected to ``hidden_size=2048`` (the LLM
embedding dim). Audio goes through Whisper → audio_projection → avg_pool. Vision
goes through Qwen3-VL → merger.
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
_REPO_ROOT = _HERE.parent.parent  # .../verl_omni_qwen36


# ---------------------------------------------------------------------------
# Parquet media extraction (identical to MiniCPM-o 5.0)
# ---------------------------------------------------------------------------

def extract_media_paths(parquet_path: str):
    """Read parquet, return ``(audio_paths, image_paths)`` — unique, sorted."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)

    audio_paths: set[str] = set()
    image_paths: set[str] = set()

    rows = table.to_pylist()

    for row in rows:
        for audio_path in row.get("audios") or []:
            if isinstance(audio_path, str) and audio_path:
                audio_paths.add(audio_path)

        for img_entry in row.get("images") or []:
            if isinstance(img_entry, dict):
                p = img_entry.get("image")
                if isinstance(p, str) and p:
                    image_paths.add(p)
            elif isinstance(img_entry, str) and img_entry:
                image_paths.add(img_entry)

    return sorted(audio_paths), sorted(image_paths)


# ---------------------------------------------------------------------------
# Hashing (identical to MiniCPM-o 5.0 — path-based, not content-based)
# ---------------------------------------------------------------------------

def path_hash(path: str) -> str:
    """SHA-256 of a file path string. Used as cache key."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _load_zarr_audio(audio_path: str, sampling_rate: int):
    """Load audio from zarr.zip archive (``path::key`` format).

    The zarr.zip contains groups keyed by UUID with .wav suffix, e.g.:
        6c7f8e9e-4d6b-4919-b6e3-c790f699cc0d.wav/audio
        6c7f8e9e-4d6b-4919-b6e3-c790f699cc0d.wav/sample_rate
    """
    import zarr.storage
    import zarr

    parts = audio_path.split("::")
    archive_path = parts[0]
    key = parts[1] if len(parts) > 1 else "audio"

    store = zarr.storage.ZipStore(archive_path, mode="r")
    root = zarr.open(store=store, mode="r")

    # Try exact key, key.wav, and common names
    candidates = [key, f"{key}.wav"]
    for candidate in candidates:
        try:
            group = root[candidate]
            # group is a zarr Group with 'audio' and 'sample_rate' arrays
            arr = group["audio"]
            wav = np.asarray(arr[:], dtype=np.float32)
            if wav.ndim == 0:
                wav = np.array([float(wav)], dtype=np.float32)
            if wav.ndim > 1:
                wav = wav[0]
            store.close()
            return wav
        except KeyError:
            continue
        except Exception:
            continue

    # Fallback: scan all keys for a match
    try:
        for k in root.keys():
            if key in k:
                group = root[k]
                arr = group["audio"]
                wav = np.asarray(arr[:], dtype=np.float32)
                if wav.ndim == 0:
                    wav = np.array([float(wav)], dtype=np.float32)
                if wav.ndim > 1:
                    wav = wav[0]
                store.close()
                return wav
    except Exception:
        pass

    store.close()
    return None


# ---------------------------------------------------------------------------
# Qwen3-VL image → patches + grid_thw (adapted from vllm-omni q36_frame_to_patches)
# ---------------------------------------------------------------------------

# Q36 vision config (from config.json vision_config)
_Q36_VISION_CONFIG = {
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
}
_Q36_IMAGE_MIN_PIXELS = 4096
_Q36_IMAGE_MAX_PIXELS = 4014080  # ~2000x2000, limits ViT patch count to avoid OOM


def _q36_round_up(x: int, factor: int) -> int:
    return (x + factor - 1) // factor * factor


def _q36_image_smart_resize(height: int, width: int) -> tuple[int, int]:
    """smart_resize: factor = patch * merge = 32."""
    factor = _Q36_VISION_CONFIG["patch_size"] * _Q36_VISION_CONFIG["spatial_merge_size"]
    min_pixels = _Q36_IMAGE_MIN_PIXELS
    max_pixels = _Q36_IMAGE_MAX_PIXELS
    if height * width > max_pixels:
        scale = max_pixels / (height * width)
        h_bar = _q36_round_up(int(height * scale**0.5), factor)
        w_bar = _q36_round_up(int(width * scale**0.5), factor)
    elif height * width < min_pixels:
        scale = min_pixels / (height * width)
        h_bar = _q36_round_up(int(height * scale**0.5), factor)
        w_bar = _q36_round_up(int(width * scale**0.5), factor)
    else:
        h_bar = _q36_round_up(height, factor)
        w_bar = _q36_round_up(width, factor)
    return h_bar, w_bar


def _q36_frame_to_patches(frame) -> tuple["torch.Tensor", "torch.Tensor"]:
    """PIL image → (patches (gh*gw, pc), grid_thw (1,3)).

    Adapted from vllm-omni's q36_frame_to_patches:
      smart_resize → BICUBIC → normalize → patch → grid_thw
    """
    from PIL import Image as PILImage
    import torch
    import numpy as np

    patch = _Q36_VISION_CONFIG["patch_size"]
    merge = _Q36_VISION_CONFIG["spatial_merge_size"]
    temporal = _Q36_VISION_CONFIG["temporal_patch_size"]
    c = _Q36_VISION_CONFIG["in_channels"]

    if isinstance(frame, PILImage.Image):
        img = frame.convert("RGB")
        w, h = img.size
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            h, w = arr.shape[:2]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            img = PILImage.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")

    resized_h, resized_w = _q36_image_smart_resize(h, w)
    img = img.resize((resized_w, resized_h), PILImage.BICUBIC)
    pixel = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)  # (H, W, 3)
    pixel = (pixel - 0.5) / 0.5
    pixel = pixel.permute(2, 0, 1)  # (3, H, W)

    gh, gw = resized_h // patch, resized_w // patch
    # Single frame copied to temporal frames; 10-dim view + qwen3vl permute.
    frames = pixel.unsqueeze(0).unsqueeze(0).expand(1, temporal, -1, -1, -1)  # (1,T,3,H,W)
    patches = frames.view(
        1, 1, temporal, c, gh // merge, merge, patch, gw // merge, merge, patch
    ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    patches = patches.reshape(gh * gw, -1)

    grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long)
    return patches, grid_thw


# ---------------------------------------------------------------------------
# Audio embedding precompute (adapted for Qwen3.6)
# ---------------------------------------------------------------------------

def precompute_audio_embeddings(
    model,
    processor,
    audio_paths: list[str],
    cache_dir: Path,
    sampling_rate: int,
    audio_pool_step: int = 5,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Compute and cache Whisper + projection + pool embeddings for each audio file.

    Adapted for Qwen3.6:
      - ``model.model.audio_encoder`` instead of ``model.apm``
      - ``model.model.audio_projection`` instead of ``model.audio_projection_layer``
      - Inline ``nn.AvgPool1d(audio_pool_step)`` instead of ``model.audio_avg_pooler``
      - ``_get_feat_extract_output_lengths`` computed inline (no model method)
    """
    import librosa
    import torch
    import torch.nn as nn
    from concurrent.futures import ThreadPoolExecutor, as_completed

    audio_cache = cache_dir / "audio"
    audio_cache.mkdir(parents=True, exist_ok=True)

    # Resolve audio encoder + projection from new model structure
    model_sub = getattr(model, "model", None)
    if model_sub is not None:
        audio_encoder = getattr(model_sub, "audio_encoder", None)
        audio_projection = getattr(model_sub, "audio_projection", None)
    else:
        audio_encoder = getattr(model, "audio_encoder", None)
        audio_projection = getattr(model, "audio_projection", None)

    if audio_encoder is None:
        print("[audio] ERROR: model.model.audio_encoder not found. Cannot precompute audio.", flush=True)
        return 0, 0, 0

    device = next(audio_encoder.parameters()).device
    dtype = next(audio_encoder.parameters()).dtype

    # Inline avg pooler (new model doesn't have model.audio_avg_pooler)
    avg_pooler = nn.AvgPool1d(audio_pool_step, stride=audio_pool_step).to(device)

    # Inline _get_feat_extract_output_lengths
    def _get_feat_extract_output_lengths(input_lengths):
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (input_lengths_after_cnn - audio_pool_step) // audio_pool_step + 1
        return input_lengths_after_cnn, input_lengths_after_pooling.to(dtype=torch.int32)

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
        """Load audio and extract mel features."""
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

        # Use process_audio (same as MiniCPM-o 5.0 baseline) — returns
        # MiniCPMOBatchFeature with data["audio_features"] + data["audio_feature_lens"].
        # Do NOT use audio_feature_extract (it calls get_audio_placeholder which
        # needs tokenizer.audio_start, not available on Qwen2Tokenizer).
        result = processor.process_audio(audios=wav, sampling_rate=sampling_rate)

        if hasattr(result, "data"):
            audio_features = result.data.get("audio_features")
            audio_feature_lens = result.data.get("audio_feature_lens")
        elif isinstance(result, dict):
            audio_features = result.get("audio_features")
            audio_feature_lens = result.get("audio_feature_lens")
        else:
            audio_features = result
            audio_feature_lens = None

        if audio_features is None:
            return None

        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.as_tensor(audio_features)

        # audio_feature_lens from process_audio is a List[torch.Tensor] (one
        # tensor per batch). Flatten by concatenating all tensors then flattening.
        if audio_feature_lens is not None:
            if isinstance(audio_feature_lens, list):
                # List of tensors → concatenate into one 1-D tensor
                flat = []
                for item in audio_feature_lens:
                    if isinstance(item, torch.Tensor):
                        flat.append(item.flatten())
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, torch.Tensor):
                                flat.append(sub.flatten())
                            else:
                                flat.append(torch.tensor([int(sub)]))
                    else:
                        flat.append(torch.tensor([int(item)]))
                if flat:
                    audio_feature_lens = torch.cat(flat).long()
                else:
                    audio_feature_lens = None
            elif not isinstance(audio_feature_lens, torch.Tensor):
                audio_feature_lens = torch.as_tensor(audio_feature_lens)

        if audio_feature_lens is not None and isinstance(audio_feature_lens, torch.Tensor):
            audio_feature_lens = audio_feature_lens.flatten().long()
        else:
            audio_feature_lens = torch.full(
                (audio_features.shape[0],), audio_features.shape[-1], dtype=torch.long
            )

        return audio_features, audio_feature_lens

    import time
    done = 0
    total_io_time = 0.0
    total_fwd_time = 0.0
    total_save_time = 0.0
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
                    import traceback
                    print(f"[audio] FAILED: {audio_path}: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    n_failed += 1
                    done += 1
        t_io = time.time() - t_io_start
        total_io_time += t_io

        if not loaded:
            continue

        # Forward each loaded file through audio encoder individually
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

                # Pad mel to 3000 frames (Whisper requires fixed 3000 input)
                # process_audio may return mel shorter than 3000 for short audio
                _, _, mel_len = mel_on_device.shape
                if mel_len < 3000:
                    pad_mel = torch.zeros(
                        mel_on_device.shape[0], mel_on_device.shape[1], 3000 - mel_len,
                        dtype=mel_on_device.dtype, device=mel_on_device.device,
                    )
                    mel_on_device = torch.cat([mel_on_device, pad_mel], dim=2)
                elif mel_len > 3000:
                    mel_on_device = mel_on_device[:, :, :3000]

                t_fwd_start = time.time()

                # Whisper encoder forward (output_hidden_states=False → use last_hidden_state)
                audio_outputs = audio_encoder(
                    mel_on_device,
                    attention_mask=audio_attention_mask,
                    output_hidden_states=False,
                )
                audio_states = audio_outputs.last_hidden_state

                # Audio projection (2-layer MLP: linear1 → GELU → linear2)
                audio_embeds = audio_projection(audio_states)

                # Avg pool (inline, same as training forward)
                audio_embeds = audio_embeds.transpose(1, 2)
                audio_embeds = avg_pooler(audio_embeds)
                audio_embeds = audio_embeds.transpose(1, 2)

                _, feature_lens_after_pooling = _get_feat_extract_output_lengths(
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
                import traceback
                print(f"[audio] FAILED: {audio_path}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                n_failed += 1
                done += 1

    elapsed_total = time.time() - t_start_all
    print(
        f"[audio] Done: {n_cached} computed, {n_skipped} cached (skipped), "
        f"{n_failed} failed | elapsed: {elapsed_total:.0f}s | "
        f"IO: {total_io_time:.0f}s ({total_io_time/elapsed_total*100:.0f}%) | "
        f"NPU: {total_fwd_time:.0f}s ({total_fwd_time/elapsed_total*100:.0f}%) | "
        f"save={total_save_time:.0f}s ({total_save_time/elapsed_total*100:.0f}%)",
        flush=True,
    )
    return n_cached, n_skipped, n_failed


# ---------------------------------------------------------------------------
# Vision embedding precompute (adapted for Qwen3.6 Qwen3-VL)
# ---------------------------------------------------------------------------

def precompute_vision_embeddings(
    model,
    processor,
    image_paths: list[str],
    cache_dir: Path,
    max_slice_nums: int,
):
    """Compute and cache Qwen3-VL + merger embeddings for each image file.

    Adapted for Qwen3.6:
      - ``model.model.visual`` (Qwen3_VisionTransformer) instead of ``model.vpm``
      - ``model.visual.merger`` instead of ``model.vit_merger`` + ``model.resampler``
      - Forward signature: ``visual(pixel_values, grid_thw=grid_thw)`` → embeddings
      - No ``cu_seqlens`` / ``tgt_sizes`` / ``vit_merger`` — Qwen3-VL uses grid_thw

    The Qwen3-VL processor produces ``pixel_values`` (flattened patches) and
    ``image_grid_thw`` (T, H, W grid). The vision transformer processes all patches
    at once and the merger projects to ``out_hidden_size=2048``.
    """
    from PIL import Image
    import torch

    vision_cache = cache_dir / "vision"
    vision_cache.mkdir(parents=True, exist_ok=True)

    # Resolve vision encoder from new model structure
    model_sub = getattr(model, "model", None)
    if model_sub is not None:
        visual = getattr(model_sub, "visual", None)
    else:
        visual = getattr(model, "visual", None)
    if visual is None:
        print("[vision] ERROR: model.model.visual not found. Cannot precompute vision.", flush=True)
        return 0, 0, 0

    device = next(visual.parameters()).device
    dtype = next(visual.parameters()).dtype
    spatial_merge_size = getattr(visual, "spatial_merge_size", 2)

    n_cached = 0
    n_skipped = 0
    n_failed = 0

    for i, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            print(f"[vision {i+1}/{len(image_paths)}] MISSING: {image_path}", flush=True)
            n_failed += 1
            continue

        try:
            # 1. Load image via PIL (no dependency on qwen_omni_utils)
            from PIL import Image
            img = Image.open(image_path).convert("RGB")

            # Per-file cache key — path hash
            h = path_hash(image_path)
            cache_file = vision_cache / f"{h}.pt"
            if cache_file.exists():
                n_skipped += 1
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"[vision {i+1}/{len(image_paths)}] CACHED: {image_path}", flush=True)
                continue

            # 2. Convert PIL image to Qwen3-VL patches + grid_thw
            # MiniCPMOProcessor.process_image returns SigLIP-style pixel_values +
            # tgt_sizes, NOT Qwen3-VL style pixel_values + image_grid_thw.
            # Use the same q36_frame_to_patches logic as vLLM-omni:
            #   smart_resize → BICUBIC → normalize → patch → grid_thw
            patches, grid_thw = _q36_frame_to_patches(img)

            # 3. Qwen3-VL forward: visual(pixel_values, grid_thw=grid_thw)
            pixel_values = patches.to(device=device, dtype=dtype)
            grid_thw = grid_thw.to(device=device)

            embeds = visual(pixel_values, grid_thw=grid_thw)

            # HF Qwen3_5MoeVisionModel.forward does proj but NOT spatial merge.
            # last_hidden_state has L = T*H*W rows (pre-merge), but we need
            # L_merged = T*H*W / merge^2 rows. Apply spatial merge by averaging
            # every merge^2 consecutive rows.
            if hasattr(embeds, "last_hidden_state"):
                embeds = embeds.last_hidden_state
            elif not isinstance(embeds, torch.Tensor):
                embeds = embeds[0] if isinstance(embeds, (list, tuple)) else embeds

            # Apply spatial merge: (L, C) → (L//merge^2, merge^2, C) → mean → (L//merge^2, C)
            L, C = embeds.shape
            merge_sq = spatial_merge_size * spatial_merge_size
            if L % merge_sq != 0:
                print(f"[vision {i+1}/{len(image_paths)}] WARNING: L={L} not divisible by merge^2={merge_sq}", flush=True)
            embeds = embeds.view(-1, merge_sq, C).mean(dim=1)

            # 4. Split per image by grid_thw
            # Qwen3-VL returns (L, out_hidden_size), need to split by
            # (T * H * W) / (merge * merge) per image
            sizes = (grid_thw.prod(-1) // (spatial_merge_size * spatial_merge_size)).tolist()
            if isinstance(embeds, torch.Tensor):
                if embeds.dim() == 1:
                    embeds = embeds.unsqueeze(0)
                emb_list = list(embeds.split(sizes))
                emb_list = [t.detach().cpu().to(torch.bfloat16) for t in emb_list]
            elif isinstance(embeds, (list, tuple)):
                emb_list = [t.detach().cpu().to(torch.bfloat16) for t in embeds]
            else:
                emb_list = [embeds.detach().cpu().to(torch.bfloat16)]

            torch.save(emb_list, cache_file)

            # Free NPU memory between images to avoid OOM on large images
            del embeds, emb_list, pixel_values, grid_thw, patches
            torch.npu.empty_cache()

            n_cached += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[vision {i+1}/{len(image_paths)}] COMPUTED: {image_path}", flush=True)

        except Exception as exc:
            import traceback
            print(
                f"[vision {i+1}/{len(image_paths)}] FAILED: {image_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            n_failed += 1

    return n_cached, n_skipped, n_failed


# ---------------------------------------------------------------------------
# Model loading (adapted for Qwen3.6)
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str, device: str = "npu"):
    """Load Qwen3.6 model + processor from checkpoint.

    The new model uses transformers' built-in ``Qwen3_5MoeForConditionalGeneration``,
    which includes ``model.visual`` and ``model.language_model`` but NOT
    ``model.audio_encoder`` or ``model.audio_projection``. We manually load the
    audio encoder from checkpoint (same logic as ``thinker_adapter._load_audio_encoder``).

    We only need the encoders (audio_encoder, audio_projection, visual). The LLM
    backbone is NOT needed for precompute — we skip loading it or move it to CPU.
    """
    import json
    import torch
    import torch_npu  # noqa: F401 — registers NPU backend
    import torch.nn as nn
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoImageProcessor,
        WhisperConfig,
    )
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    from safetensors.torch import load_file as load_safetensors

    # Monkey-patch AutoImageProcessor.register (same as MiniCPM-o 5.0)
    _orig_register = AutoImageProcessor.register

    def _safe_register(*args, **kwargs):
        if args and isinstance(args[0], str):
            return
        return _orig_register(*args, **kwargs)

    AutoImageProcessor.register = _safe_register

    print(f"[load] Loading model from {model_path}...", flush=True)

    # Load processor first (small, CPU)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load full model — try AutoModelForMultimodalLM first (gets
    # Qwen3_5MoeForConditionalGeneration with visual+audio+language_model),
    # then AutoModelForCausalLM (gets Qwen3_5MoeForCausalLM text-only, needs
    # manual audio_encoder loading).
    # Qwen3_5MoeForConditionalGeneration includes model.visual + model.language_model
    # + model.audio_encoder + model.audio_projection.
    # Qwen3_5MoeForCausalLM (text-only) only has model.model (Qwen3_5MoeModel).
    model = None
    load_errors = []
    for auto_cls_name, auto_cls in [
        ("AutoModelForMultimodalLM", None),  # resolved below
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModel", None),  # resolved below
    ]:
        try:
            if auto_cls_name == "AutoModelForMultimodalLM":
                from transformers import AutoModelForMultimodalLM as _cls
                auto_cls = _cls
            elif auto_cls_name == "AutoModel":
                from transformers import AutoModel as _cls
                auto_cls = _cls
            model = auto_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"[load] Loaded via {auto_cls_name} → {type(model).__name__}", flush=True)
            break
        except Exception as e:
            load_errors.append(f"{auto_cls_name}: {e}")
            print(f"[load] {auto_cls_name} failed: {e}", flush=True)

    if model is None:
        raise RuntimeError(
            f"Failed to load model with any Auto class:\n" + "\n".join(load_errors)
        )

    # --- Manually load audio encoder + projection (not in built-in class) ---
    model_sub = getattr(model, "model", None)
    if model_sub is not None and getattr(model_sub, "audio_encoder", None) is None:
        print("[load] Loading audio encoder from checkpoint...", flush=True)

        cfg = model.config
        audio_cfg = getattr(cfg, "audio_config", None)
        if audio_cfg is not None:
            # Build WhisperConfig + WhisperEncoder
            if hasattr(audio_cfg, "to_dict"):
                audio_cfg_dict = audio_cfg.to_dict()
            elif isinstance(audio_cfg, dict):
                audio_cfg_dict = dict(audio_cfg)
            else:
                audio_cfg_dict = dict(vars(audio_cfg))

            try:
                whisper_cfg = WhisperConfig(**audio_cfg_dict)
            except TypeError:
                import inspect
                valid = set(inspect.signature(WhisperConfig.__init__).parameters.keys())
                filtered = {k: v for k, v in audio_cfg_dict.items() if k in valid}
                whisper_cfg = WhisperConfig(**filtered)

            audio_encoder = WhisperEncoder(whisper_cfg)

            # Audio projection (2-layer MLP: linear1 → GELU → linear2)
            d_model = audio_cfg_dict.get("d_model", 1024)
            text_hidden = 2048
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                text_hidden = getattr(text_cfg, "hidden_size", None) or (
                    text_cfg.get("hidden_size", 2048) if isinstance(text_cfg, dict) else 2048
                )

            class AudioProjection(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    self.linear1 = nn.Linear(in_dim, out_dim)
                    self.linear2 = nn.Linear(out_dim, out_dim)
                def forward(self, x):
                    return self.linear2(torch.nn.functional.gelu(self.linear1(x)))

            audio_projection = AudioProjection(d_model, text_hidden)

            # Load weights from checkpoint safetensors
            idx_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                idx = json.load(open(idx_path))
                weight_map = idx.get("weight_map", {})
                audio_files = set()
                for key in weight_map:
                    if "audio_encoder" in key or "audio_projection" in key:
                        audio_files.add(weight_map[key])
                all_state = {}
                for f in sorted(audio_files):
                    all_state.update(load_safetensors(os.path.join(model_path, f)))
            else:
                all_state = load_safetensors(os.path.join(model_path, "model.safetensors"))

            # Load audio encoder weights
            encoder_state = {}
            for key, val in all_state.items():
                if key.startswith("model.audio_encoder."):
                    new_key = key.replace("model.audio_encoder.", "", 1)
                    encoder_state[new_key] = val
            if encoder_state:
                missing, unexpected = audio_encoder.load_state_dict(encoder_state, strict=False)
                print(f"[load] audio_encoder: {len(encoder_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

            # Load audio projection weights
            proj_state = {}
            for key, val in all_state.items():
                if key.startswith("model.audio_projection."):
                    new_key = key.replace("model.audio_projection.", "", 1)
                    proj_state[new_key] = val
            if proj_state:
                missing, unexpected = audio_projection.load_state_dict(proj_state, strict=False)
                print(f"[load] audio_projection: {len(proj_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

            # Attach to model
            model_sub.audio_encoder = audio_encoder
            model_sub.audio_projection = audio_projection

            n_enc = sum(p.numel() for p in audio_encoder.parameters())
            n_proj = sum(p.numel() for p in audio_projection.parameters())
            print(f"[load] audio_encoder ({n_enc / 1e6:.1f}M) + audio_projection ({n_proj / 1e6:.1f}M) attached", flush=True)

    # Move encoders to NPU, keep LLM on CPU (saves ~60GB NPU memory)
    npu_device = torch.device(device)
    for attr_path in (
        ("model", "audio_encoder"),
        ("model", "audio_projection"),
        ("model", "visual"),
    ):
        sub = model
        for attr in attr_path:
            sub = getattr(sub, attr, None)
            if sub is None:
                break
        if sub is not None:
            sub.to(npu_device)
            print(f"[load] {'.'.join(attr_path)} → {device}", flush=True)

    # Move LLM backbone to CPU to free NPU memory for vision forward
    for attr_name in ("language_model", "llm", "lm_head"):
        sub = getattr(model, attr_name, None)
        if sub is None:
            sub = getattr(getattr(model, "model", None), attr_name, None)
        if sub is not None:
            sub.to("cpu")
            n_params = sum(p.numel() for p in sub.parameters())
            print(f"[load] {attr_name} → cpu ({n_params / 1e9:.2f}B params freed from NPU)", flush=True)
    # Also move model.model (Qwen3_5MoeModel) if it's the text backbone wrapper
    model_model = getattr(model, "model", None)
    if model_model is not None:
        # model.model may contain language_model as a submodule
        lm = getattr(model_model, "language_model", None)
        if lm is not None:
            lm.to("cpu")
            n_params = sum(p.numel() for p in lm.parameters())
            print(f"[load] model.model.language_model → cpu ({n_params / 1e9:.2f}B params freed from NPU)", flush=True)
    # Force garbage collection
    import gc
    gc.collect()
    torch.npu.empty_cache()

    # Verify encoders are on device
    for attr_path in (
        ("model", "audio_encoder"),
        ("model", "visual"),
    ):
        sub = model
        for attr in attr_path:
            sub = getattr(sub, attr, None)
            if sub is None:
                break
        if sub is not None:
            first_param = next(sub.parameters(), None)
            if first_param is not None:
                print(
                    f"[load] {'.'.join(attr_path)} on {first_param.device}, "
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
        description="Precompute Whisper + Qwen3-VL embeddings for Qwen3.6 RL training."
    )
    parser.add_argument("--parquet", required=True, help="Path to train/validation parquet.")
    parser.add_argument("--model_path", required=True, help="Path to Qwen3.6 checkpoint directory.")
    parser.add_argument("--cache_dir", required=True, help="Output cache directory.")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--max_slice_nums", type=int, default=1, help="Must match training config.")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--audio_pool_step", type=int, default=5, help="Audio avg pool stride (config.audio_pool_step).")
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_vision", action="store_true")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index (0-based) for multi-GPU parallelism.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for multi-GPU parallelism.")

    args = parser.parse_args()

    if not os.path.isfile(args.parquet):
        print(f"ERROR: parquet not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("audio", "vision"):
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    # 1. Extract media paths
    print(f"[scan] Reading parquet: {args.parquet}", flush=True)
    audio_paths, image_paths = extract_media_paths(args.parquet)
    print(f"[scan] Found {len(audio_paths)} unique audio files, {len(image_paths)} unique image files", flush=True)

    # Shard paths for multi-GPU parallelism
    if args.num_shards > 1:
        audio_paths = audio_paths[args.shard_id::args.num_shards]
        image_paths = image_paths[args.shard_id::args.num_shards]
        print(f"[shard {args.shard_id}/{args.num_shards}] Processing {len(audio_paths)} audio + {len(image_paths)} vision files", flush=True)

    if not audio_paths and not image_paths:
        print("[scan] No media files found. Nothing to do.", flush=True)
        sys.exit(0)

    # 2. Load model + processor
    model, processor = load_model_and_processor(args.model_path, args.device)

    # 3. Precompute audio
    if audio_paths and not args.skip_audio:
        print(f"\n[audio] Precomputing {len(audio_paths)} audio embeddings...", flush=True)
        a_cached, a_skipped, a_failed = precompute_audio_embeddings(
            model, processor, audio_paths, cache_dir, args.sampling_rate,
            audio_pool_step=args.audio_pool_step,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print(f"[audio] Done: {a_cached} computed, {a_skipped} cached, {a_failed} failed", flush=True)
    else:
        print("[audio] Skipped", flush=True)

    # 4. Precompute vision
    if image_paths and not args.skip_vision:
        print(f"\n[vision] Precomputing {len(image_paths)} vision embeddings...", flush=True)
        v_cached, v_skipped, v_failed = precompute_vision_embeddings(
            model, processor, image_paths, cache_dir, args.max_slice_nums,
        )
        print(f"[vision] Done: {v_cached} computed, {v_skipped} cached, {v_failed} failed", flush=True)
    else:
        print("[vision] Skipped", flush=True)

    print("\n[done] Precompute complete.", flush=True)
    print(f"  Cache dir: {cache_dir}", flush=True)
    print(f"  Audio:  {len(list((cache_dir / 'audio').glob('*.pt')))} files", flush=True)
    print(f"  Vision: {len(list((cache_dir / 'vision').glob('*.pt')))} files", flush=True)
    print(f"\nNext: set MINICPMO_EMBED_CACHE_DIR={cache_dir} in the launch script.", flush=True)


if __name__ == "__main__":
    main()
