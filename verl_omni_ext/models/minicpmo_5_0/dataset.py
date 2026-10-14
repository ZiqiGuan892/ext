"""Compatibility entry point for the singular dataset module name.

Launchers and the Qwen3.6-whisper adapter import ``minicpmo_5_0.dataset``;
the supplied implementation lives in ``datasets.py``. Re-export the same
objects, including the cache helpers used by Qwen36, without duplicating the
implementation or installing its module-level patches a second time.
"""

from .datasets import (  # noqa: F401
    MiniCPMOOmniRLHFDataset,
    MiniCPMOThinkerRLHFDataset,
    _EmbedList,
    _attach_precomputed,
    _load_precomputed,
    _path_hash,
)

__all__ = ["MiniCPMOThinkerRLHFDataset", "MiniCPMOOmniRLHFDataset"]
