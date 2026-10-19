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


def install_huanyu_forward(module, *, start_token_id, end_token_id, placeholder_token_id, encode_audio=None):
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
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("Huanyu forward builds audio embeddings from input_ids; inputs_embeds is unsupported")
        batch, lens = _prepare_features(features, lengths)
        actual_counts = (ids == start_token_id).sum(dim=1).tolist()
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
