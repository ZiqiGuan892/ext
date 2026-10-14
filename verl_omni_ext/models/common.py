"""
跨模型共享的 rollout 适配器基类。

多个 thinker-only 模型的 rollout 适配器此前逐字重复同一段样板：
  - 从自有 ``vllm_omni/pipeline.py`` 惰性 import PipelineConfig / stages
  - 实现 ``build_stage_configs`` / ``get_pipeline_id`` / ``ensure_pipeline_registered``
    三个方法，除模块路径与常量名外完全相同。

``PipelineFwdAdapter`` 把这段收敛成"子类声明类属性 + 基类方法惰性解析"，
每个模型的 rollout_adapter 从 ~35 行降到 ~10 行。惰性 import 保留在方法
内部，CPU import 路径不碰 vllm_omni（见 pipelines.md / testing.md 的 CPU
test 依赖）。
"""
from __future__ import annotations

import importlib
from typing import Any

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase


class PipelineFwdAdapter(OmniRolloutPipelineBase):
    """把 ext 包内 frozen 的 vllm-omni pipeline 定义转发给基类的适配器基类。

    子类只需声明四个类属性，即获得三个方法的标准实现：

        _pipeline_module: pipeline 定义所在模块（如
            "verl_omni_ext.models.qwen3_5_moe.vllm_omni.pipeline"）
        _pipeline_attr:   该模块里的 PipelineConfig 常量名
        _stages_attr:     该模块里的 stages 常量名
        _pipeline_label:  错误信息里的模型名（如 "qwen3_5_moe"）

    不注册本基类——每个模型子类各自 ``@OmniRolloutPipelineBase.register(key)``。
    """

    _pipeline_module: str = ""
    _pipeline_attr: str = ""
    _stages_attr: str = ""
    _pipeline_label: str = ""

    @classmethod
    def _spec(cls) -> tuple[Any, Any]:
        """惰性解析 pipeline 定义；不可用时返回 ``(None, None)``。

        vllm-omni / gate patch 未就绪时，pipeline 模块 import 失败属预期——
        返回 ``(None, None)`` 而非硬失败，保持与旧版 ``_VLLM_OMNI_AVAILABLE``
        一致的降级语义（见 pipelines.md 的 import 侧降级原则）。
        模块存在但缺常量（拼写/重命名）走 ``AttributeError``，同样视为不可用，
        避免返回 ``None`` 之后在 ``.model_type``/``.stages`` 上崩溃。
        """
        try:
            mod = importlib.import_module(cls._pipeline_module)
            return getattr(mod, cls._pipeline_attr), getattr(mod, cls._stages_attr)
        except (ImportError, AttributeError):
            return None, None

    @classmethod
    def _require(cls) -> tuple[Any, Any]:
        pipeline, stages = cls._spec()
        if pipeline is None or stages is None:
            raise ImportError(
                f"vllm_omni pipeline for "
                f"{cls._pipeline_label or cls._pipeline_module} not available"
            )
        return pipeline, stages

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        """返回 ext 包内 frozen 的 per-stage 拓扑。"""
        _, stages = cls._require()
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """返回 pipeline 定义的 ``model_type``（带 ``_thinker_only`` 后缀）。

        注册键 ≠ pipeline id：基类默认返回注册键（如 ``qwen3_5_moe``），
        会生成错误的 YAML。这里取 frozen pipeline 的 ``model_type``
        （如 ``qwen3_5_moe_thinker_only``），与 vllm-omni 验证值一致。
        """
        pipeline, _ = cls._require()
        return pipeline.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """确保 vLLM-Omni 的 pipeline registry 里有这个 model_type。

        不可用时静默返回（与旧版一致）——gate patch 没打时无法注册，也不该在此抛错。
        """
        pipeline, _ = cls._spec()
        if pipeline is None:
            return
        from vllm_omni.config.pipeline_registry import register_pipeline

        register_pipeline(pipeline)