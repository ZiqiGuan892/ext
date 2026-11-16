"""R3（Rollout Routing Replay）+ 训练侧融合 MoE kernel。

设计约束：
- 融合 MoE kernel（``MINICPMO_MOE_NPU_KERNEL=1``）整体替换 HF 的逐专家循
  环实现，是 replay 的唯一载体；``HUANYU_EXPERT_REPLAY=1`` 未开 kernel 时
  install 直接 raise（fail-closed）。
- 本包按 CPU import 契约保持轻量：import 本包不拉起 torch_npu/transformers/
  verl（trainer 侧重依赖全部延迟到 configure_model 内的 install 调用）。
"""

from .rollout_enable import expert_replay_engine_extras

__all__ = ["expert_replay_engine_extras"]
