"""
Qwen3.5-MoE Thinker 适配器

槽位① OmniModelBase.register  | 槽位② OmniRolloutPipelineBase | L2 vision device fix
"""
from . import thinker_adapter, rollout_adapter, patches  # noqa: F401