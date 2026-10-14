"""MiniCPM-o 5.0 插件包（槽位① thinker | 槽位② rollout | L2 补丁 | 槽位④ 数据集）。

模块级补丁必须在任何 remote code 被 import 之前执行——``from_pretrained`` 在
``configure_model`` 之前（fsdp/omni_impl.py），所以让 transformers 能 import 得动
remote code 的补丁放在这里，不是放在 ``configure_model`` 里。

顺序有讲究：
  1. ``apply_minicpmo_auto_register_guard`` 必须先落地——checkpoint 的
     processing_minicpmo.py 用 str 调 register，新版 transformers 拒收，连带
     modeling 的 import 都进不来。
  2. ``apply_minicpmo_automodel_fallback`` 在 config.json auto_map 缺
     "AutoModelForMultimodalLM" 时兜底；只在主调用失败后触发。
  3. ``apply_fsdp2_cpu_offload_pin_memory_patch`` 与 transformers 无关：verl
     硬编码 ``CPUOffloadPolicy(pin_memory=True)``（transformer_impl.py:443），
     Ascend 上 pin 掉整节点 ~200 GiB 主机内存、并把 sharded-grad 的 D2H 拷贝
     变非阻塞，正是第一个 loss.backward() 以 EH0002 "Argument [dst] must not be
     null" 崩掉的根源。见 fsdp_cpu_offload_pin.py 的模块 docstring。
"""
import logging

from verl_omni_ext.models.minicpmo_5_0.fsdp_cpu_offload_pin import apply_fsdp2_cpu_offload_pin_memory_patch
from verl_omni_ext.models.minicpmo_5_0.patches import (
    apply_minicpmo_auto_register_guard,
    apply_minicpmo_automodel_fallback,
)

logger = logging.getLogger(__name__)

from . import dataset  # noqa: F401
from . import thinker_adapter  # noqa: F401

apply_minicpmo_auto_register_guard()
apply_minicpmo_automodel_fallback()
apply_fsdp2_cpu_offload_pin_memory_patch()

__all__ = ["MiniCPMO50ThinkerAdapter"]

try:
    from verl_omni_ext.models.minicpmo_5_0.rollout_adapter import MiniCPMO50RolloutAdapter
    from verl_omni_ext.models.minicpmo_5_0 import vllm_omni  # noqa: F401
    __all__.append("MiniCPMO50RolloutAdapter")
except ImportError as e:
    logger.warning(
        "MiniCPM-o 5.0 rollout adapter unavailable: %s. Training-side loading still works; "
        "rollout does not.",
        e,
    )