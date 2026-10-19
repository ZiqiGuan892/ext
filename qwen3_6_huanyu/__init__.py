"""Explicit Huanyu registration, selected through model.external_lib.

No pyproject/entrypoint edits are needed. The training alias is independent of
Whisper/RM and does not replace the HF checkpoint architecture.
"""

from .thinker_adapter import Qwen36HuanyuThinkerAdapter
from .rollout_adapter import Qwen36HuanyuRolloutAdapter

__all__ = ["Qwen36HuanyuThinkerAdapter", "Qwen36HuanyuRolloutAdapter"]
