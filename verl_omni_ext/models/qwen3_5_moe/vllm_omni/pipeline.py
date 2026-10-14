"""
Qwen3.5-MoE vllm-omni thinker-only pipeline 拓扑定义（frozen，对齐 vllm-omni 验证值）

这是真正的推理拓扑知识——stage 数量、execution_type、input_sources 等。
verl-omni 侧的 rollout adapter（槽位②）只是转发这里的定义。

验证值来源：vllm-omni 提交 4e3d1c9 ``qwen3_5_moe_thinker_only``。

schema 对齐 verl-omni pin 的上游 vllm-omni（``.github/vllm_omni_pin.txt``）——
其 ``PipelineConfig`` 无 ``default_deploy_config_name`` 字段（那是 NPU 镜像自加的），
RL 路径由 verl-omni ``_write_deploy_config`` 自生成 yaml，不依赖该字段，故这里不传。

Stage 0: Thinker — 多模态理解 + 文本生成（无 talker/code2wav，thinker-only 单 stage）。
"""
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_5_MOE_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="qwen3_5_moe_thinker_only",
    model_arch="Qwen3_5MoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
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
QWEN3_5_MOE_THINKER_ONLY_STAGES = QWEN3_5_MOE_THINKER_ONLY_PIPELINE.stages