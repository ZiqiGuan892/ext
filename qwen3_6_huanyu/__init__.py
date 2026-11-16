"""Explicit Huanyu registration, selected through model.external_lib.

No pyproject/entrypoint edits are needed. The training alias is independent of
Whisper/RM and does not replace the HF checkpoint architecture.

``apply_fsdp2_pad_sync_patch`` 与注册无关，放在这里只是为了**时序**：它必须在
任何对 FSDP2 模块的 ``.to("cpu")`` 之前打上，而包 import 发生在
``_build_fsdp_module`` 之前。见 fsdp2_pad_sync.py 的模块 docstring。
"""

from .thinker_adapter import Qwen36HuanyuThinkerAdapter
from .rollout_adapter import Qwen36HuanyuRolloutAdapter
from .fsdp2_pad_sync import apply_fsdp2_pad_sync_patch

__all__ = ["Qwen36HuanyuThinkerAdapter", "Qwen36HuanyuRolloutAdapter"]

apply_fsdp2_pad_sync_patch()
