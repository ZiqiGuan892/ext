
"""
MiniCPM-o 5.0 vllm-omni thinker-only pipeline 拓扑定义（frozen，对齐 vllm-omni 验证值）

Stage 0: Thinker — 多模态输入（image/video/audio）+ 文本输出。
验证值来源：vllm-omni 未提交的 ``MINICPMO_5_0_THINKER_ONLY_PIPELINE``。

要点：
- ``model_arch="MiniCPMO50OmniForConditionalGeneration"``，与 _OMNI_MODELS 注册键一致。
- ``model_stage="llm"``（非 "thinker"）——匹配
  ``MiniCPMO50OmniForConditionalGeneration.__init__`` 的 model_stage 分派。
- ``engine_output_type="text"``——无下游 stage 消费 hidden states，让 AR runner 跳过
  构建/序列化 per-request 多模态载荷；若未来有 stage 要读 thinker 的 hidden states 再
  改回 "latent"。
"""
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MINICPMO_5_0_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="minicpmo_5_0_thinker_only",
    model_arch="MiniCPMO50OmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

# 与 rollout adapter 约定的旧名保持兼容
MINICPMO_5_0_THINKER_ONLY_STAGES = MINICPMO_5_0_THINKER_ONLY_PIPELINE.stages