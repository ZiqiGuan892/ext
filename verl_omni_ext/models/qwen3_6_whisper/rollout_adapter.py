"""
槽位②: Qwen3.6-35B-A3B-whisper 推理侧拓扑适配器

@OmniRolloutPipelineBase.register("qwen3_6_whisper")

拓扑定义在 ``verl_omni_ext.models.qwen3_6_whisper.vllm_omni.pipeline``（frozen）。
样板逻辑收敛在基类 ``PipelineFwdAdapter``。

与 MiniCPM-o 5.0 rollout_adapter 的差异：
  - 注册键 ``qwen3_6_whisper`` ≠ pipeline id ``minicpmo_5_0_thinker_only``
    （推理侧复用 vllm-omni-xy 已 ready 的 ``minicpmo_5_0_thinker_only`` pipeline，
     见开发文档方案 A）。
  - ``model_arch`` 仍是 ``MiniCPMO50OmniForConditionalGeneration``（推理侧 wrapper
    类复用，因为新模型与 MiniCPM-o 5.0 共享同一推理 wrapper）。
  - 多模态 extras 与 MiniCPM-o 5.0 相同：``limit_mm_per_prompt`` / ``media_io_kwargs``
    / ``interleave_mm_strings`` / NPU ``weight_nz_mode=0``。
  - 新增 video limit（``limit_mm_per_prompt.video``），但 MiniCPM-o 5.0 的 adapter
    已设 ``video: 1``，故直接复用。

开发文档方案 A：训练侧 rollout ``pipeline_name`` 复用 vllm-omni-xy 已注册的
``minicpmo_5_0_thinker_only`` —— 推理 pipeline 已 ready，无需新增。此文件在 verl
侧注册 ``qwen3_6_whisper`` rollout adapter，但实际使用的 vllm-omni pipeline 仍是
``minicpmo_5_0_thinker_only``（通过 examples 脚本中的
``+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_5_0_thinker_only``
显式覆盖）。
"""
from verl.utils.device import is_npu_available

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

from verl_omni_ext.models.common import PipelineFwdAdapter


@OmniRolloutPipelineBase.register("qwen3_6_whisper")
class Qwen36WhisperRolloutAdapter(PipelineFwdAdapter):
    """Qwen3.6-whisper thinker-only 推理拓扑。

    ``model_stage`` 在 vLLM 侧是 ``"llm"``，训练侧注册键是 ``"thinker"``。
    pipeline id 复用 ``minicpmo_5_0_thinker_only``（方案 A）。
    """

    _pipeline_module = "verl_omni_ext.models.qwen3_6_whisper.vllm_omni.pipeline"
    _pipeline_attr = "QWEN36_WHISPER_THINKER_ONLY_PIPELINE"
    _stages_attr = "QWEN36_WHISPER_THINKER_ONLY_STAGES"
    _pipeline_label = "qwen3_6_whisper"

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        """只允许 thinker_only。"""
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r} for qwen3_6_whisper. Only 'thinker_only' "
                "is supported: RL rollout needs text out."
            )
        stages = super().build_stage_configs(pipeline_mode)
        assert len(stages) == 1, (
            f"Expected 1 stage in qwen3_6_whisper thinker-only pipeline, got {len(stages)}. "
            "vLLM-Omni may have changed the pipeline definition."
        )
        return stages

    @classmethod
    def rollout_flags(cls, pipeline_mode: str = "thinker_only") -> dict:
        """单 stage 没有 inter-stage flag。"""
        return {}

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """无 HF config 覆盖。新模型无 TTS，无需 tts_config 处理。"""
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """stage-0 engine extras：多模态上限 + NPU weight 格式修正。

        与 MiniCPM-o 5.0 完全相同，因为新模型的多模态 extras 需求一致：
          - ``interleave_mm_strings=True``：媒体与文本交错
          - ``media_io_kwargs``：video fps/num_frames
          - ``limit_mm_per_prompt``：image 96 / audio 64 / video 1
          - ``weight_nz_mode=0``（NPU）
        """
        if stage_id != 0:
            return {}
        pipeline = cls._require()[0]
        extras = {
            "model_arch": pipeline.model_arch,
            # model_stage 必须显式传入 engine_extras：远程 verl-omni 新版代码
            # (2026-09-29) 不再从 PipelineConfig.stages[0].model_stage 向
            # vllm_config.model_config.model_stage 传播，导致默认 "thinker"
            # 被传给 minicpmo_5_0_omni.py:209，触发 ValueError。
            # MiniCPMO50OmniForConditionalGeneration 只接受 "llm" 或 "tts"。
            "model_stage": "llm",
            "interleave_mm_strings": True,
            "media_io_kwargs": {"video": {"fps": 1, "num_frames": 128}},
            "limit_mm_per_prompt": {"image": 96, "audio": 64, "video": 1},
        }
        if is_npu_available:
            extras["additional_config"] = {"weight_nz_mode": 0}
        return extras
