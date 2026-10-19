"""Select the existing Huanyu-capable vLLM wrapper with an isolated verl alias."""

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
