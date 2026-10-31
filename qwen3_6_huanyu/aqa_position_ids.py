"""Keep audio-only AQA and text-only STEM positions 1D through TransferQueue.

AgentLoop normally adds a text axis to Qwen's three RoPE axes, even for audio
without images. Some TransferQueue/verl combinations pack each [4, S] sample
along axis 0, then change only _ragged_idx to 2. Offsets still describe four
axes instead of S tokens, so mini-batch selection fails before model forward.

In explicitly selected AQA/STEM modes all four axes must agree at valid tokens.
Transporting their common [S] sequence preserves position semantics and uses
verl's existing text-position batching path. Image/video mRoPE is never reduced.
"""

from functools import wraps
import importlib
import os

import torch


_FLAG = "_huanyu_aqa_position_ids"


def _enabled():
    return any(os.environ.get(key, "0") == "1" for key in (
        "HUANYU_AQA_AUDIO_TRAIN", "HUANYU_STEM_TEXT_TRAIN",
    ))


def _nonempty(value):
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return bool(len(value))


def _reject_vision(inputs):
    inputs = inputs or {}
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw",
                "video_grid_thw", "image_bound", "huanyu_cached_vision_embeds"):
        if _nonempty(inputs.get(key)):
            raise ValueError(f"Huanyu AQA/STEM position IDs require inputs without vision; got {key}")
    counts = inputs.get("huanyu_cached_image_counts")
    if counts is not None and bool(torch.as_tensor(counts).ne(0).any()):
        raise ValueError("Huanyu AQA/STEM position IDs require zero cached image counts")


def _collapse_positions(positions, input_ids, attention_mask):
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Huanyu AQA/STEM input_ids must be [batch, sequence]")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("Huanyu AQA/STEM attention_mask must match input_ids")
    if not isinstance(positions, torch.Tensor):
        raise TypeError("Huanyu AQA/STEM position IDs must be a tensor")
    if positions.ndim == 2 and positions.shape == input_ids.shape:
        return positions
    if positions.ndim != 3 or positions.shape != (input_ids.shape[0], 4, input_ids.shape[1]):
        raise ValueError("Huanyu AQA/STEM positions must be [batch, sequence] or [batch, 4, sequence]")
    valid = (torch.ones(input_ids.shape, device=positions.device, dtype=torch.bool) if attention_mask is None
             else attention_mask.to(device=positions.device, dtype=torch.bool))
    disagreement = positions.ne(positions[:, :1, :]) & valid.unsqueeze(1)
    if bool(disagreement.any()):
        raise ValueError("Huanyu AQA/STEM RoPE axes differ at valid tokens; refusing to discard spatial positions")
    # Return the contiguous common sequence for transport.
    return positions[:, 0, :].contiguous()


def restore_huanyu_aqa_positions(positions, input_ids):
    """Restore the engine's [text,t,h,w; B,S] contract at model entry.

    Only transport uses [B,S]. This avoids relying on the installed HF version
    accepting two-dimensional positions directly. ``expand`` allocates no
    four-axis copy, and all valid-token positions are unchanged.
    """
    if not _enabled() or positions is None:
        return positions
    if not isinstance(positions, torch.Tensor):
        raise TypeError("Huanyu AQA/STEM position IDs must be a tensor")
    if positions.ndim == 2 and positions.shape == input_ids.shape:
        return positions.unsqueeze(0).expand(4, -1, -1)
    if positions.ndim == 3 and positions.shape == (4, *input_ids.shape):
        return positions
    raise ValueError("Huanyu AQA/STEM model positions must be [batch, sequence] or [4, batch, sequence]")


def install_huanyu_aqa_position_ids():
    """Install for explicit AQA/STEM launchers, without modifying verl files."""
    if not _enabled():
        return 0
    candidates = []
    for module_name, class_name in (
        ("verl.experimental.agent_loop.agent_loop", "AgentLoopWorker"),
        ("verl.trainer.ppo.v1.agent_loop_tq", "AgentLoopWorkerTQ"),
    ):
        cls = getattr(importlib.import_module(module_name), class_name)
        candidates.append(cls)
        modified = getattr(getattr(cls, "__ray_metadata__", None), "modified_class", None)
        if modified is not None:
            candidates.append(modified)
    installed = 0
    for cls in dict.fromkeys(candidates):
        original = getattr(cls, "_compute_position_ids", None)
        if not callable(original):
            continue
        if getattr(original, _FLAG, False):
            installed += 1
            continue

        # Keep a local qualname so Ray serializes this wrapper, rather than
        # resolving an unpatched method by the upstream class's import path.
        @wraps(original, assigned=("__name__", "__doc__", "__annotations__"))
        def compute(self, input_ids, attention_mask, multi_modal_inputs=None, *args,
                    _original=original, **kwargs):
            processor = getattr(self, "processor", None)
            active = _enabled() and getattr(processor, "_huanyu_training_processor", False)
            if active:
                _reject_vision(multi_modal_inputs)
            positions = _original(self, input_ids, attention_mask, multi_modal_inputs, *args, **kwargs)
            if not active:
                return positions
            result = _collapse_positions(positions, input_ids, attention_mask)
            if not getattr(self, "_huanyu_aqa_positions_reported", False):
                stem = os.environ.get("HUANYU_STEM_TEXT_TRAIN", "0") == "1"
                tag = "huanyu-stem-positions" if stem else "huanyu-aqa-positions"
                mode = "text_only=true" if stem else "audio_only=true"
                print(f"[{tag}] pid={os.getpid()} "
                      f"shape={tuple(positions.shape)}->{tuple(result.shape)} "
                      f"{mode} equivalent_axes=true", flush=True)
                self._huanyu_aqa_positions_reported = True
            return result

        setattr(compute, _FLAG, True)
        setattr(cls, "_compute_position_ids", compute)
        installed += 1
    if not installed:
        raise RuntimeError("No AgentLoop position-ID method could be wired for Huanyu AQA/STEM")
    return installed
