"""Encode each media item at its own shape in training and serving.

Variable-size batches change BF16 encoder arithmetic. Audio padding also changes
the encoder boundary unless each chunk is cropped before convolution. Frozen
towers stay unsharded through these local calls; the outer actor hook releases
them uniformly across ranks.
"""
from functools import wraps
import os
from types import MethodType

import torch


def enabled():
    return os.environ.get("MINICPMO_ISOLATE_MEDIA_ENCODERS", "1") == "1"


def split_vision_inputs(pixels, grids):
    if grids.ndim == 1:
        grids = grids.reshape(1, 3)
    if pixels.ndim >= 3:
        pixels = pixels.flatten(0, 1)
    lengths = grids.prod(-1).detach().cpu().tolist()
    if sum(lengths) != pixels.shape[0]:
        raise ValueError(f"Vision patch/grid mismatch: patches={pixels.shape[0]}, grids={lengths}")
    return list(zip(pixels.split(lengths, dim=0), grids.split(1, dim=0)))


def install_actor(module):
    if not enabled():
        return False
    visual = getattr(getattr(module, "model", None), "visual", None)
    if visual is None or getattr(visual, "_q36_isolated_media", False):
        return False
    # Trainable tower FSDP/autograd needs its existing collective schedule.
    if any(parameter.requires_grad for parameter in visual.parameters()):
        return False
    original = visual.forward.__func__

    @wraps(original)
    def forward(self, hidden_states, grid_thw, **kwargs):
        items = split_vision_inputs(hidden_states, grid_thw)
        if len(items) <= 1:
            return original(self, hidden_states, grid_thw=grid_thw, **kwargs)
        outputs = [original(self, pixels, grid_thw=grid, **kwargs) for pixels, grid in items]
        return type(outputs[0])(
            last_hidden_state=torch.cat([item.last_hidden_state for item in outputs], dim=0),
            pooler_output=torch.cat([item.pooler_output for item in outputs], dim=0),
        )

    visual.forward = MethodType(forward, visual)
    visual._q36_isolated_media = True
    print("[qwen36-media] frozen actor vision encodes each item at its own shape", flush=True)
    return True
