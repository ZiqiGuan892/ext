"""Select the existing Huanyu-capable vLLM wrapper with an isolated verl alias."""

import os

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni_ext.models.qwen3_6_whisper.rollout_adapter import Qwen36WhisperRolloutAdapter


@OmniRolloutPipelineBase.register("qwen3_6_huanyu")
class Qwen36HuanyuRolloutAdapter(Qwen36WhisperRolloutAdapter):
    """Reuse topology, not Whisper audio code.

    The inherited single llm stage uses MiniCPMO50OmniForConditionalGeneration.
    In the selected vLLM-Omni checkout that implementation dispatches to Huanyu
    by audio_config.model_type. Retain the registered vLLM pipeline ID because
    the existing backend exports the corresponding external pipeline module.
    """

    _pipeline_label = "qwen3_6_huanyu"

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        extras = dict(super().get_stage_engine_extras(stage_id, pipeline_mode=pipeline_mode))
        if stage_id == 0:
            # Resolve/bind IPC handling through the actual worker factory, not
            # only the registry's best-effort external import (which may run
            # while the original extension module is still being imported).
            extras["worker_extension_cls"] = (
                "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
            )
            # Keep generated stage extras and outer CLI kwargs consistent:
            # native deploy precedence may replace the entire additional_config.
            # This adapter belongs to the actor, not the separate GenRM pipeline.
            if os.environ.get("HUANYU_TEXT_AB_BATCH_INVARIANT") == "1":
                additional = dict(extras.get("additional_config", {}))
                additional["rl_config"] = {"enabled": True, "enable_batch_invariant": True}
                extras["additional_config"] = additional
        return extras
