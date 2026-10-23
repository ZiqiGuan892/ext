"""Explicit inference-worker entry point for Huanyu precision and IPC hooks.

Keep this filename distinct from the dependency package ``vllm_omni``:
direct Huanyu CLI scripts put this directory first on Python's import path.
"""

# Keep the existing native model and pipeline registration, then apply the
# Huanyu-specific hooks. No shared source files or checkpoint configs are edited.
from verl_omni_ext.models.qwen3_6_whisper import vllm_omni as _registration  # noqa: F401

from .cache_precision import install_huanyu_cache_precision
from .rollout_precision import install_huanyu_rollout_precision
from .weight_reload import install_huanyu_weight_reload

install_huanyu_rollout_precision()
install_huanyu_cache_precision()
install_huanyu_weight_reload()
