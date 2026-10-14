"""
Qwen3.6-35B-A3B-whisper vllm-omni 外部模块（通过 VLLM_OMNI_EXTERNAL_MODULES 加载）

加载方式：
  export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_whisper.vllm_omni

新模型的 wrapper 模型类 ``MiniCPMO50OmniForConditionalGeneration`` 已由
vllm-omni 核心注册（registry.py ``_OMNI_MODELS``），pipeline 定义走 gate patch
GP-004 外置：本模块 import 时把 ``QWEN36_WHISPER_THINKER_ONLY_PIPELINE`` 注册进
``OMNI_PIPELINES``。

方案 A（推荐）：训练侧 rollout ``pipeline_name`` 复用 vllm-omni-xy 已注册的
``minicpmo_5_0_thinker_only`` —— 推理 pipeline 已 ready，无需新增。此情况下：
  - examples 脚本中 ``+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_5_0_thinker_only``
  - 本模块的 ``register_pipeline`` 是 idempotent，与 minicpmo_5_0.vllm_omni 的注册
    不冲突（注册同一个 pipeline，幂等）。
"""
from vllm_omni.config.pipeline_registry import register_pipeline
from .pipeline import QWEN36_WHISPER_THINKER_ONLY_PIPELINE

register_pipeline(QWEN36_WHISPER_THINKER_ONLY_PIPELINE)

from vllm.model_executor.models import ModelRegistry

if "MiniCPMO50OmniForConditionalGeneration" not in ModelRegistry.get_supported_archs():
    ModelRegistry.register_model(
        "MiniCPMO50OmniForConditionalGeneration",
        "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni:MiniCPMO50OmniForConditionalGeneration",
    )