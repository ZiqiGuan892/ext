"""
MiniCPM-o 5.0 vllm-omni 外部模块（通过 VLLM_OMNI_EXTERNAL_MODULES 加载）

加载方式：
  export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.minicpmo_5_0.vllm_omni

miniCPM-o 的 wrapper 模型类 ``MiniCPMO50OmniForConditionalGeneration`` 已由
vllm-omni 核心注册（registry.py `_OMNI_MODELS`），但 pipeline 定义走 gate patch
GP-004 外置：本模块 import 时把 ``MINICPMO_5_0_THINKER_ONLY_PIPELINE`` 注册进
``OMNI_PIPELINES``。这样 vllm-omni 引擎子进程（只 import vllm_omni，不碰 verl_omni）
靠 GP-004 的 ``VLLM_OMNI_EXTERNAL_MODULES`` 加载本模块时，就能在引擎侧解析到
``minicpmo_5_0_thinker_only``；driver 侧的 ensure_pipeline_registered 走同一入口，
幂等。
"""
from vllm_omni.config.pipeline_registry import register_pipeline
from .pipeline import MINICPMO_5_0_THINKER_ONLY_PIPELINE

register_pipeline(MINICPMO_5_0_THINKER_ONLY_PIPELINE)

from vllm.model_executor.models import ModelRegistry

if "MiniCPMO50OmniForConditionalGeneration" not in ModelRegistry.get_supported_archs():
    ModelRegistry.register_model(
        "MiniCPMO50OmniForConditionalGeneration",
        "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni:MiniCPMO50OmniForConditionalGeneration",
    )
