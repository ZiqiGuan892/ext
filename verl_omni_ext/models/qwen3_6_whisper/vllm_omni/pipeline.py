"""
Qwen3.6-35B-A3B-whisper vllm-omni thinker-only pipeline 拓扑定义（frozen）

与 MiniCPM-o 5.0 ``MINICPMO_5_0_THINKER_ONLY_PIPELINE`` 完全相同：
  - 同一 ``model_arch="MiniCPMO50OmniForConditionalGeneration"``（推理侧 wrapper 类复用）
  - 同一 ``model_stage="llm"``
  - 同一 ``engine_output_type="text"``
  - 同一 stage 拓扑（单 stage 0，thinker-only）

差异：
  - ``model_type="qwen3_6_whisper_thinker_only"``（独立 id，避免与 MiniCPM-o 5.0
    注册键冲突，但实际复用 vllm-omni-xy 已 ready 的 ``minicpmo_5_0_thinker_only``
    pipeline，见方案 A）
"""
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN36_WHISPER_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="qwen3_6_whisper_thinker_only",
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
QWEN36_WHISPER_THINKER_ONLY_STAGES = QWEN36_WHISPER_THINKER_ONLY_PIPELINE.stages