"""
槽位②: Qwen3.5-MoE 推理侧拓扑适配器

@OmniRolloutPipelineBase.register("qwen3_5_moe")

拓扑定义在 ``verl_omni_ext.models.qwen3_5_moe.vllm_omni.pipeline``（frozen，对齐
vllm-omni 提交 4e3d1c9 的验证值）。样板逻辑收敛在基类 ``PipelineFwdAdapter``：
子类只声明类属性名并覆写 Qwen3.5-MoE 特有的 extras，三件套方法（build_stage_configs
/ get_pipeline_id / ensure_pipeline_registered）由基类惰性转发——CPU import 路径
不碰 vllm_omni。

本文件整段对齐树内 ``pipelines/qwen35_moe/omni_rollout_adapter.py`` 的语义，
只是把对 vllm-omni 的直接 import 换成基类的 ``cls._require()`` 惰性解析。

与 minicpmo_5_0 的差异：
- 无多模态 engine extras：``model_arch`` 取自 pipeline 定义（Qwen3.5-MoE 没有
  独立 thinker-only model 类，pipeline 直接加载上游 vLLM 的
  ``Qwen3_5MoeForConditionalGeneration``）。interleave_mm_strings /
  media_io_kwargs / limit_mm_per_prompt 四行都不用。
- 只有 ``thinker_only`` 一种模式：无 talker/code2wav stage，没得选。
"""
from verl.utils.device import is_npu_available

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

from verl_omni_ext.models.common import PipelineFwdAdapter


@OmniRolloutPipelineBase.register("qwen3_5_moe")
class Qwen35MoeRolloutAdapter(PipelineFwdAdapter):
    """Qwen3.5-35B-A3B thinker-only 推理拓扑。

    ``model_stage`` 在 vLLM 侧是 ``"thinker"``；注册键 ``qwen3_5_moe`` ≠
    pipeline id ``qwen3_5_moe_thinker_only``，生成 deploy YAML 时取后者。
    """

    _pipeline_module = "verl_omni_ext.models.qwen3_5_moe.vllm_omni.pipeline"
    _pipeline_attr = "QWEN3_5_MOE_THINKER_ONLY_PIPELINE"
    _stages_attr = "QWEN3_5_MOE_THINKER_ONLY_STAGES"
    _pipeline_label = "qwen3_5_moe"

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        """只允许 thinker_only：Qwen3.5-MoE 无 talker/code2wav stage。

        同时断言只解析出 1 个 stage，防止上游改动静默加 stage。
        """
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r} for qwen3_5_moe. Only 'thinker_only' "
                "is supported: Qwen3.5-MoE has no talker/code2wav stages."
            )
        stages = super().build_stage_configs(pipeline_mode)
        assert len(stages) == 1, (
            f"Expected 1 stage in qwen3_5_moe thinker-only pipeline, got {len(stages)}. "
            "vLLM-Omni may have changed the pipeline definition."
        )
        return stages

    @classmethod
    def rollout_flags(cls, pipeline_mode: str = "thinker_only") -> dict:
        """单 stage 没有 inter-stage flag，标记为显式空。"""
        return {}

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """无 HF config 覆盖。

        Qwen3.5 的 config.json 是平铺结构（text_config / vision_config 在根，
        无 thinker_config），且没有音频输出开关要关。
        """
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """stage-0 engine extras：model_arch + NPU weight 格式修正。

        只放 verl 不知道的 key。verl 自己的 ``vllm serve`` 参数走 per-stage
        ``runtime_overrides``（会覆盖 YAML），所以把 ``max_model_len`` 等塞这里
        会被静默忽略，应从 trainer config 设。
        """
        if stage_id != 0:
            return {}
        pipeline = cls._require()[0]
        extras = {"model_arch": pipeline.model_arch}
        if is_npu_available:
            # vllm-ascend 在 FRACTAL_NZ 下拒绝 wake_up()：reorder 后的权重跨 RL
            # weight sync 丢精度。这个 dict 由 _write_deploy_config 提升到顶层
            # `additional_config`，因为 vllm-ascend 从 per-TP-worker 的全局单例
            # 读它，那个单例由 CLI 参数构建，不是从 stage 级 engine_extras。
            extras["additional_config"] = {"weight_nz_mode": 0}
        return extras
