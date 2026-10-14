"""
槽位②: MiniCPM-o 5.0 推理侧拓扑适配器

@OmniRolloutPipelineBase.register("minicpmo_5_0")

拓扑定义在 ``verl_omni_ext.models.minicpmo_5_0.vllm_omni.pipeline``（frozen，对齐
vllm-omni 验证值）。样板逻辑收敛在基类 ``PipelineFwdAdapter``：子类只声明类属性名并
覆写 MiniCPM-o 特有的 extras，三件套方法（build_stage_configs / get_pipeline_id /
ensure_pipeline_registered）由基类惰性转发——CPU import 路径不碰 vllm_omni。

与 qwen3_5_moe 的差异：
- ``model_arch`` 是 vllm-omni 自研的 ``MiniCPMO50OmniForConditionalGeneration``，
  不是上游 vLLM 类。注册键 ``minicpmo_5_0`` ≠ pipeline id ``minicpmo_5_0_thinker_only``。
- 多模态 engine extras（limit_mm_per_prompt / media_io_kwargs / interleave_mm_strings）
  必须有，Qwen3.5-MoE 一行都不用。
- vllm-omni 侧 ``minicpmo_5_0_thinker_only`` 不在核心 registry（已按 GP-004 回滚），
  engine 子进程靠 ``vllm_omni/__init__.py`` import 期 register_pipeline 补注册，
  driver 侧走基类 ensure_pipeline_registered，两条路幂等。
"""
from verl.utils.device import is_npu_available

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

from verl_omni_ext.models.common import PipelineFwdAdapter


@OmniRolloutPipelineBase.register("minicpmo_5_0")
class MiniCPMO50RolloutAdapter(PipelineFwdAdapter):
    """MiniCPM-o 5.0 thinker-only 推理拓扑。

    ``model_stage`` 在 vLLM 侧是 ``"llm"``（匹配
    ``MiniCPMO50OmniForConditionalGeneration.__init__`` 的分派）而训练侧注册键是
    ``"thinker"``——两个无关命名空间，互不需要知道对方。
    """

    _pipeline_module = "verl_omni_ext.models.minicpmo_5_0.vllm_omni.pipeline"
    _pipeline_attr = "MINICPMO_5_0_THINKER_ONLY_PIPELINE"
    _stages_attr = "MINICPMO_5_0_THINKER_ONLY_STAGES"
    _pipeline_label = "minicpmo_5_0"

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        """只允许 thinker_only：3-stage 变体会启 Talker/Code2Wav，RL 不需要。

        同时断言只解析出 1 个 stage，防止上游改动静默加 stage 或误解析到
        3-stage ``minicpmo_5_0``。
        """
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r} for minicpmo_5_0. Only 'thinker_only' "
                "is supported: RL rollout needs text out, and the Talker/Code2Wav stages would "
                "claim devices the trainer needs."
            )
        stages = super().build_stage_configs(pipeline_mode)
        assert len(stages) == 1, (
            f"Expected 1 stage in minicpmo_5_0 thinker-only pipeline, got {len(stages)}. "
            "vLLM-Omni may have changed the pipeline definition, or this resolved to the "
            "3-stage MINICPMO_5_0_PIPELINE."
        )
        return stages

    @classmethod
    def rollout_flags(cls, pipeline_mode: str = "thinker_only") -> dict:
        """单 stage 没有 inter-stage flag，标记为显式空。"""
        return {}

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """无 HF config 覆盖。

        丢掉 audio 输出靠选 thinker-only pipeline 表达，不是翻 config flag：
        ``tts.*`` 权重由 ``load_weights`` 跳过并打 warning；audio 输入保留
        （thinker 持有 Whisper encoder ``apm.*``/``audio_projection_layer.*``）。
        """
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """stage-0 engine extras：多模态上限 + NPU weight 格式修正。

        只放 verl 不知道的 key。verl 自己的 ``vllm serve`` 参数走 per-stage
        ``runtime_overrides``（stage_config.py:1014-1017 会覆盖 YAML），所以把
        ``max_model_len``/``enforce_eager`` 塞这里会被静默忽略，应从 trainer config 设。
        """
        if stage_id != 0:
            return {}
        pipeline = cls._require()[0]
        extras = {
            "model_arch": pipeline.model_arch,
            # MiniCPM-o 把媒体和文本交错放在同一个 prompt，不是前置全部媒体。
            "interleave_mm_strings": True,
            "media_io_kwargs": {"video": {"fps": 1, "num_frames": 128}},
            # vLLM 用这几个上限去 dummy-run KV cache 换存，既是校验 knob 也是内存 knob；
            # 若 profiling OOM 就降到数据集的真实上限。
            #
            # 刻意不用 `rollout.limit_images` 表达：verl 会把那个整体变成
            # `{"image": N}`（vllm_async_server.py:253），audio/video 预算会被整个丢掉。
            "limit_mm_per_prompt": {"image": 96, "audio": 64, "video": 1},
        }
        if is_npu_available:
            # vllm-ascend 在 FRACTAL_NZ 下拒绝 wake_up()：reorder 后的权重跨 RL weight
            # sync 丢精度。这个 dict 由 _write_deploy_config 提升到顶层
            # `additional_config`，因为 vllm-ascend 从 per-TP-worker 的全局单例读它，
            # 那个单例由 CLI 参数构建，不是从 stage 级 engine_extras。
            extras["additional_config"] = {"weight_nz_mode": 0}
        return extras





