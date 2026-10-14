"""
L2: Qwen3.5-MoE monkey patch（ViT 位置编码设备修复）

打的对象：transformers 的 ``Qwen3_5MoeVisionModel``（第三方代码，L2 正确用法）。
原因：FSDP2 CPUOffload 下（verl 对所有 forward_only engine 无条件开启，即 GRPO/SGPO 的
ref 模型），参数报 cpu、激活在 npu，``fast_pos_embed_interpolate`` 用
``self.pos_embed.weight.device`` 建索引张量 → 建在 cpu → ``F.embedding`` device mismatch。

实现对齐 verl-omni 提交 1e6ffbc 的验证版 ``qwen3_5_moe_vision.py``：
- forward 里记录计算设备的真值（取自激活 ``hidden_states``，而非可能被 offload 的参数）
- fast_pos_embed_interpolate 里把 ``pos_embed`` 暂挂到计算设备，finally 恢复。
只改设备选择，运算/数值与上游逐字相同；offload 未生效时是 no-op。

注意：这是**类级**补丁（打在 Qwen3_5MoeVisionModel 类上，forward 期才生效），
所以放在 configure_model 里调用即可，不必在包 import 期。
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_verl_omni_ext_qwen3_5_vision_device_fixed"


def apply_qwen3_5_vision_device_fix(module: Any = None) -> bool:
    """Patch ``Qwen3_5MoeVisionModel`` 使位置编码查找设备安全（类级，幂等）。

    Args:
        module: 兼容旧调用约定，忽略——补丁目标是 transformers 的类，不是实例。
            保留参数仅为不破坏 ``thinker_adapter.configure_model`` 的现有调用。

    Returns:
        bool: True 若补丁已应用（或已存在）；False 若目标类不可 import
        （非 Qwen3.5 的 transformers build），属预期降级。
    """
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeVisionModel,
        )
    except ImportError as exc:
        logger.info("Qwen3.5 vision device fix skipped: %s", exc)
        return False

    if getattr(Qwen3_5MoeVisionModel, _PATCH_FLAG, False):
        return True

    original_interpolate = Qwen3_5MoeVisionModel.fast_pos_embed_interpolate
    original_forward = Qwen3_5MoeVisionModel.forward

    def _patched_interpolate(self, grid_thw):
        # 由 _patched_forward 在单次 forward 期间设置。
        target_device = getattr(self, "_verl_omni_ext_compute_device", None)
        if target_device is None:
            return original_interpolate(self, grid_thw)

        pos_embed = self.pos_embed
        if pos_embed.weight.device == target_device:
            # offload 未在本 rank 生效，无需修正。
            return original_interpolate(self, grid_thw)

        # 暂挂 pos_embed 到计算设备，使原实现里
        # `device = self.pos_embed.weight.device` 解析正确；finally 恢复，
        # 不打扰 FSDP 对 CPU 副本的 offload 记账。
        self.pos_embed = pos_embed.to(target_device)
        try:
            return original_interpolate(self, grid_thw)
        finally:
            self.pos_embed = pos_embed

    def _patched_forward(self, hidden_states, grid_thw, *args, **kwargs):
        # hidden_states 已由调用方在计算设备上生成，是可信的设备真值，
        # 区别于可能被 FSDP offload 的参数。
        self._verl_omni_ext_compute_device = hidden_states.device
        try:
            return original_forward(self, hidden_states, grid_thw, *args, **kwargs)
        finally:
            self._verl_omni_ext_compute_device = None

    Qwen3_5MoeVisionModel.fast_pos_embed_interpolate = _patched_interpolate
    Qwen3_5MoeVisionModel.forward = _patched_forward
    setattr(Qwen3_5MoeVisionModel, _PATCH_FLAG, True)

    logger.info("Applied Qwen3.5 vision positional-embedding device fix (FSDP2 CPU offload).")
    return True