"""Insert Huanyu audio features while retaining the HF image/text forward.

The initial integration uses padded [B,S] inputs. Audio is encoded by the
installed vLLM-Omni implementation, then replaces only verified audio slots in
the token-embedding result. Keeping input_ids in the original HF forward also
keeps its image/video masks and mRoPE construction intact.
"""

from contextvars import ContextVar
from functools import wraps
from numbers import Integral
from types import MethodType

import torch
from torch.nn import functional as F

from .audio_alignment import find_audio_bounds, insert_audio_embeddings
from .aqa_position_ids import restore_huanyu_aqa_positions


_ACTIVE_AUDIO = ContextVar("huanyu_training_audio", default=None)
_FLAG = "_verl_huanyu_forward_installed"


def _unwrap(value):
    if type(value).__name__ in ("NonTensorData", "NonTensorStack"):
        return value.data if hasattr(value, "data") else value.tolist()
    return value


def _feature_chunks(value):
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return []
        if value.ndim == 2:
            return [value]
        if value.ndim in (3, 4):
            return [chunk for item in value for chunk in _feature_chunks(item)]
        raise ValueError("Huanyu audio features must be [F,T] chunks or a batch of chunks")
    if isinstance(value, (list, tuple)):
        return [chunk for item in value for chunk in _feature_chunks(item)]
    raise TypeError(f"Unsupported Huanyu audio feature container: {type(value).__name__}")


def _lengths(value):
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError("Huanyu audio lengths must be integers")
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [length for item in value for length in _lengths(item)]
    raise TypeError("Huanyu audio lengths must contain integers")


def _flatten_visual(value, name):
    value = _unwrap(value)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [_flatten_visual(item, name) for item in value]
        tensors = [item for item in tensors if item is not None and item.numel()]
        return torch.cat(tensors, dim=0) if tensors else None
    if value is None:
        return None
    raise TypeError(f"Unsupported {name} container: {type(value).__name__}")


def _prepare_features(features, lengths):
    chunks = _feature_chunks(features)
    lens = _lengths(lengths)
    if len(chunks) != len(lens):
        raise ValueError(f"Huanyu audio chunk count {len(chunks)} != length count {len(lens)}")
    if not chunks:
        return None, None
    width = chunks[0].shape[0]
    for chunk, length in zip(chunks, lens):
        if chunk.shape[0] != width or not chunk.is_floating_point():
            raise ValueError("Huanyu audio chunks must share a floating-point feature dimension")
        if not 0 < length <= chunk.shape[-1]:
            raise ValueError(f"Invalid Huanyu audio frame length {length} for shape {tuple(chunk.shape)}")
    max_length = max(lens)
    max_length += -max_length % 4
    batch = torch.stack([F.pad(chunk[:, :length], (0, max_length - length))
                         for chunk, length in zip(chunks, lens)])
    return batch, torch.tensor(lens, dtype=torch.long, device=batch.device)


def _image_spans(row, image_token_id):
    spans, start = [], None
    for i, token in enumerate(row.tolist()):
        if token == image_token_id:
            start = i if start is None else start
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(row)))
    return spans


def _cached_samples(ids, audio_chunks, vision_chunks, lengths, audio_counts, image_counts,
                    grids, marker_ids, image_token_id, spatial_merge_size, hidden_size):
    """Validate EACH sample before merging cached embeddings into token slots."""
    if len(audio_counts) != len(ids) or len(image_counts) != len(ids):
        raise ValueError("Huanyu cache requires per-sample audio/image counts")
    if any(n < 0 for n in audio_counts + image_counts):
        raise ValueError("Huanyu cached media counts cannot be negative")
    if sum(audio_counts) != len(audio_chunks) or len(lengths) != len(audio_chunks):
        raise ValueError("Huanyu cached audio counts/lengths disagree")
    if sum(image_counts) != len(vision_chunks):
        raise ValueError("Huanyu cached image counts disagree")
    if grids is None:
        if vision_chunks:
            raise ValueError("Huanyu cached images require image_grid_thw")
    elif grids.ndim != 2 or grids.shape != (len(vision_chunks), 3) or torch.any(grids <= 0):
        raise ValueError("Huanyu cache image_grid_thw does not match image count")
    for chunk in audio_chunks + vision_chunks:
        if chunk.ndim != 2 or chunk.shape[1] != hidden_size or chunk.shape[0] < 1:
            raise ValueError("Huanyu cached embedding shape must be [positive_tokens, hidden_size]")
        if chunk.dtype != torch.bfloat16 or chunk.requires_grad or not torch.isfinite(chunk).all():
            raise ValueError("Huanyu cached embeddings must be frozen finite BF16 tensors")
    for chunk, frames in zip(audio_chunks, lengths):
        if frames <= 0 or chunk.shape[0] != ((frames + 3) // 4) // 2:
            raise ValueError("Huanyu cached audio token length disagrees with real frame length")
    samples, ac, vc = [], 0, 0
    for row, na, nv in zip(ids, audio_counts, image_counts):
        audio = audio_chunks[ac:ac + na]
        vision = vision_chunks[vc:vc + nv]
        abounds = find_audio_bounds(row, [len(x) for x in audio], **marker_ids)
        vbounds = _image_spans(row, image_token_id)
        if len(vbounds) != nv:
            raise ValueError("Huanyu cached image spans do not match per-sample image counts")
        for i, (chunk, (begin, end)) in enumerate(zip(vision, vbounds)):
            patches = int(grids[vc + i].prod())
            if patches % spatial_merge_size ** 2 or patches // spatial_merge_size ** 2 != len(chunk) or end - begin != len(chunk):
                raise ValueError("Huanyu cached image span/grid/embedding length mismatch")
        ordered = sorted(zip(abounds + vbounds, audio + vision), key=lambda pair: pair[0][0])
        samples.append(([chunk for _, chunk in ordered], [span for span, _ in ordered]))
        ac += na
        vc += nv
    return samples


def install_huanyu_forward(module, *, start_token_id, end_token_id, placeholder_token_id, encode_audio=None,
                           use_precompute=False, image_token_id=None, spatial_merge_size=2,
                           require_audio_only=False):
    """Install a model-instance-only wrapper; no HF class or shared file changes."""
    if getattr(module, _FLAG, False):
        return module
    if encode_audio is None:
        from .audio_runtime import encode_huanyu_audio

        encode_audio = encode_huanyu_audio
    embedding = module.get_input_embeddings()
    original_forward = module.forward
    marker_ids = dict(start_token_id=start_token_id, end_token_id=end_token_id,
                      placeholder_token_id=placeholder_token_id)
    if use_precompute and (image_token_id is None or spatial_merge_size < 1):
        raise ValueError("Huanyu precompute requires image token ID and spatial merge size")
    if use_precompute and require_audio_only:
        raise ValueError("Trainable Huanyu audio requires online encoding, not cached embeddings")

    def inject_audio(embedding_module, args, output):
        active = _ACTIVE_AUDIO.get()
        if active is None or active["owner"] is not module:
            return None
        if not args or not torch.equal(args[0], active["ids"]):
            raise RuntimeError("Huanyu embedding lookup does not match the validated input_ids")
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError("Huanyu audio insertion requires token embeddings [B,S,H]")
        if output.shape[:2] != active["ids"].shape:
            raise ValueError("Huanyu token embedding batch/sequence shape does not match input_ids")
        active["calls"] += 1
        if active["calls"] != 1:
            raise RuntimeError("Huanyu audio forward unexpectedly looked up token embeddings more than once")
        rows = []
        for row, (chunks, bounds) in zip(output, active["samples"]):
            aligned = [chunk.to(device=row.device, dtype=row.dtype) for chunk in chunks]
            rows.append(insert_audio_embeddings(row, aligned, bounds) if chunks else row)
        return torch.stack(rows)

    embedding.register_forward_hook(inject_audio)

    @wraps(getattr(original_forward, "__func__", original_forward))
    def forward(self, *args, **kwargs):
        cache_enabled = kwargs.pop("huanyu_cache_enabled", None)
        cached_audio = kwargs.pop("huanyu_cached_audio_embeds", None)
        cached_vision = kwargs.pop("huanyu_cached_vision_embeds", None)
        image_counts = kwargs.pop("huanyu_cached_image_counts", None)
        features = kwargs.pop("huanyu_audio_features", None)
        lengths = kwargs.pop("huanyu_audio_lengths", None)
        chunk_counts = kwargs.pop("huanyu_audio_chunk_counts", None)
        # Stored prompt bounds may precede batch left-padding. Re-scan the actual
        # training ids rather than reusing stale offsets or decoded strings.
        kwargs.pop("huanyu_audio_bounds", None)
        kwargs.pop("image_bound", None)
        for name in ("_precomputed_audio_embeds", "_precomputed_vision_embeds"):
            if kwargs.pop(name, None) is not None:
                raise ValueError("Huanyu forward does not accept Whisper/unverified embedding caches")
        for name in ("audio_features", "audio_feature_lens", "audio_bounds"):
            value = kwargs.pop(name, None)
            if value is not None and (not hasattr(value, "__len__") or len(value)):
                raise ValueError(f"Unexpected legacy {name}; select the Huanyu processor/dataset")
        for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            if name in kwargs:
                kwargs[name] = _flatten_visual(kwargs[name], name)

        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None or not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("Huanyu training requires input_ids [B,S] and use_remove_padding=false")
        if "position_ids" in kwargs:
            kwargs["position_ids"] = restore_huanyu_aqa_positions(kwargs["position_ids"], ids)
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("Huanyu forward builds audio embeddings from input_ids; inputs_embeds is unsupported")
        if use_precompute:
            if _lengths(cache_enabled) != [1] * ids.shape[0]:
                raise ValueError("Huanyu strict cache mode requires cached inputs for every sample; no online fallback")
            if _feature_chunks(features) or any(kwargs.get(key) is not None for key in ("pixel_values", "pixel_values_videos", "video_grid_thw")):
                raise ValueError("Huanyu cached forward received raw encoder inputs; no online fallback")
            if kwargs.get("image_grid_thw") is not None and kwargs["image_grid_thw"].numel() and kwargs.get("position_ids") is None:
                raise ValueError("Huanyu cached images require processor-generated multimodal position_ids")
            samples = _cached_samples(
                ids, _feature_chunks(cached_audio), _feature_chunks(cached_vision),
                _lengths(lengths), _lengths(chunk_counts), _lengths(image_counts),
                kwargs.get("image_grid_thw"), marker_ids, image_token_id, spatial_merge_size,
                embedding.weight.shape[-1],
            )
            active = {"owner": self, "ids": ids, "samples": samples, "calls": 0}
            token = _ACTIVE_AUDIO.set(active)
            try:
                output = original_forward(*args, **kwargs)
                if active["calls"] != 1:
                    raise RuntimeError("HF forward did not consume Huanyu cached embeddings")
                return output
            finally:
                _ACTIVE_AUDIO.reset(token)
        if any(value is not None for value in (cache_enabled, cached_audio, cached_vision, image_counts)):
            raise ValueError("Huanyu cached inputs require HUANYU_USE_PRECOMPUTE=1 on training workers")
        batch, lens = _prepare_features(features, lengths)
        actual_counts = (ids == start_token_id).sum(dim=1).tolist()
        if require_audio_only:
            if batch is None or any(count == 0 for count in actual_counts):
                raise ValueError("Huanyu audio training requires audio in every AQA sample on every rank")
            if any(kwargs.get(key) is not None and kwargs[key].numel() for key in
                   ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")):
                raise ValueError("Huanyu audio training requires AQA audio-only inputs; image/video inputs are unsupported")
        if chunk_counts is not None:
            counts = _lengths(chunk_counts)
            if counts != actual_counts:
                raise ValueError(f"Huanyu per-sample audio counts {counts} != token-span counts {actual_counts}")
            if sum(counts) != (len(lens) if lens is not None else 0):
                raise ValueError("Huanyu per-sample audio counts do not match feature chunks")
        elif batch is not None and ids.shape[0] != 1:
            raise ValueError("Mixed-batch Huanyu audio requires per-sample huanyu_audio_chunk_counts")
        has_audio_markers = bool(((ids == start_token_id) | (ids == end_token_id)).any())
        if batch is None:
            if has_audio_markers:
                raise ValueError("Audio tokens are present but Huanyu audio features are missing")
            return original_forward(*args, **kwargs)
        if not has_audio_markers:
            raise ValueError("Huanyu audio features are present but audio tokens are missing")

        chunks = list(encode_audio(self, batch, lens))
        if len(chunks) != len(lens):
            raise ValueError("Huanyu encoder returned a different number of audio chunks")
        samples, cursor = [], 0
        for row in ids:
            count = int((row == start_token_id).sum())
            sample_chunks = chunks[cursor:cursor + count]
            bounds = find_audio_bounds(row, [chunk.shape[0] for chunk in sample_chunks], **marker_ids)
            samples.append((sample_chunks, bounds))
            cursor += count
        if cursor != len(chunks):
            raise ValueError(f"Huanyu audio token spans {cursor} != encoded chunks {len(chunks)}")
        active = {"owner": self, "ids": ids, "samples": samples, "calls": 0}
        token = _ACTIVE_AUDIO.set(active)
        try:
            output = original_forward(*args, **kwargs)
            if active["calls"] != 1:
                raise RuntimeError("HF forward did not consume Huanyu audio through the input embedding module")
            return output
        finally:
            _ACTIVE_AUDIO.reset(token)

    module.forward = MethodType(forward, module)
    setattr(module, _FLAG, True)
    return module
