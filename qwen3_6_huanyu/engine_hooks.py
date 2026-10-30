"""Explicit inference-worker entry point for Huanyu precision and IPC hooks.

Keep this filename distinct from the dependency package ``vllm_omni``:
direct Huanyu CLI scripts put this directory first on Python's import path.
"""

# Keep the existing native model and pipeline registration, then apply the
# Huanyu-specific hooks. No shared source files or checkpoint configs are edited.
from verl_omni_ext.models.qwen3_6_whisper import vllm_omni as _registration  # noqa: F401

from .cache_precision import install_huanyu_cache_precision
from .processor_compat import install_huanyu_processor_compat
from .rollout_precision import install_huanyu_rollout_precision
from .weight_reload import install_huanyu_weight_reload

# Dummy multimodal budgeting calls the processor before model initialization.
install_huanyu_processor_compat()
install_huanyu_rollout_precision()
install_huanyu_cache_precision()
# Best-effort compatibility bind may defer during the native registry's
# circular import. The rollout adapter also selects an explicit worker class;
# that worker factory import is the mandatory, non-skippable binding point.
install_huanyu_weight_reload()

# Selected only by the text/embedding A/B launcher, before runner construction.
import os as _os
if _os.environ.get("HUANYU_EMBEDDING_AB_DIR", "").strip():
    from .prompt_embedding_probe import install_runtime_probe
    install_runtime_probe()
