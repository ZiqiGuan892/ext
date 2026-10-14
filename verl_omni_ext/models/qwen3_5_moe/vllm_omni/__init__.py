"""
Qwen3.5-MoE vllm-omni 外部模块（通过 VLLM_OMNI_EXTERNAL_MODULES 加载）

加载链路（需要先给 vllm-omni 打 gate patch GP-004）：
  1. 启动脚本设 export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni
  2. vllm-omni 的 registry.py（打了 GP-004 补丁后）会 importlib.import_module 本模块
  3. 本模块 import pipeline.py，使 QWEN3_5_MOE_THINKER_ONLY_PIPELINE 可被
     Qwen35MoeRolloutAdapter.ensure_pipeline_registered 显式注册

注意：此处**不**往 _OMNI_MODELS 注册 Qwen3_5MoeForConditionalGeneration。
该架构直接加载上游 vLLM 的 Qwen3_5MoeForConditionalGeneration（verl-omni 提交
4e3d1c9 的 pipeline.py 明确 "loads upstream vLLM Qwen3_5 directly, no omni wrapper"），
vllm-omni 源码树里没有对应的 wrapper 目录。若注册，会生成指向不存在模块的 lazy model，
并在 _VLLM_OMNI_MODELS 里覆盖掉有效的上游 vLLM 条目。_OMNI_MODELS 只用于 vllm-omni
自有的 wrapper 模型；上游模型交给 vLLM 自己的 registry 处理。

本模块 import 期必须把 pipeline 注册进 ``OMNI_PIPELINES``：deploy config 里只带
``pipeline: <model_type>``，引擎子进程靠 GP-004 的 ``VLLM_OMNI_EXTERNAL_MODULES``
import 本模块时，若不在 import 期注册，引擎侧解析 ``qwen3_5_moe_thinker_only`` 会失败。
driver 侧 ``ensure_pipeline_registered`` 走同一入口，幂等。
"""
from vllm_omni.config.pipeline_registry import register_pipeline

from .pipeline import QWEN3_5_MOE_THINKER_ONLY_PIPELINE

register_pipeline(QWEN3_5_MOE_THINKER_ONLY_PIPELINE)