"""Independent Huanyu training adapter; keeps the checkpoint HF architecture."""

import logging
import os
from collections.abc import Mapping

from verl_omni.pipelines.model_base import OmniModelBase


logger = logging.getLogger(__name__)
TRAINING_ARCHITECTURE = "Qwen36HuanyuForTraining"


def _value(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_huanyu_config(config):
    audio = _value(config, "audio_config")
    if _value(audio, "model_type") != "huanyu_conformer_0_6b":
        raise ValueError("Huanyu training adapter requires audio_config.model_type=huanyu_conformer_0_6b")
    if _value(config, "model_type") != "qwen3_5_moe":
        raise ValueError("This Huanyu training adapter requires the Qwen3.5-MoE HF backbone")
    return audio


def _flatten_text_config(config):
    text = _value(config, "text_config")
    for name in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "vocab_size", "num_experts", "num_experts_per_tok",
        "router_aux_loss_coef", "shared_expert_intermediate_size", "moe_intermediate_size",
        "head_dim", "max_position_embeddings", "rms_norm_eps", "rope_theta",
        "partial_rotary_factor", "mrope_interleaved", "mrope_section", "layer_types",
        "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads",
        "linear_num_value_heads", "linear_value_head_dim", "mamba_ssm_dtype",
    ):
        value = _value(text, name)
        if value is not None and not hasattr(config, name):
            setattr(config, name, value)


def _audio_token_ids(tokenizer, config):
    if tokenizer is None:
        raise ValueError("Huanyu training requires the checkpoint tokenizer to resolve audio placeholder IDs")
    result = {}
    for token, field, config_field in (
        ("<|audio_start|>", "start_token_id", "audio_start_token_id"),
        ("<|audio_end|>", "end_token_id", "audio_end_token_id"),
        ("<unk>", "placeholder_token_id", None),
    ):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_tokens_to_ids(token) != ids[0]:
            raise ValueError(f"Huanyu marker {token!r} must be a single checkpoint token")
        expected = _value(config, config_field) if config_field else None
        if expected is not None and int(expected) != ids[0]:
            raise ValueError(f"Huanyu {token} ID {ids[0]} disagrees with config {expected}")
        result[field] = int(ids[0])
    if len(set(result.values())) != 3:
        raise ValueError("Huanyu audio marker IDs must be distinct")
    return result


@OmniModelBase.register(TRAINING_ARCHITECTURE, stage="thinker")
class Qwen36HuanyuThinkerAdapter(OmniModelBase):
    """Alias selects the adapter only; never rewrites hf_config.architectures."""

    @classmethod
    def get_strip_modules(cls, model_config):
        return []

    @classmethod
    def configure_tokenizer(cls, model_path, model_config):
        from verl.utils import hf_tokenizer

        validate_huanyu_config(model_config.hf_config)
        return hf_tokenizer(model_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_processor(cls, model_path, model_config):
        from .processor import build_huanyu_processor
        from .rollout_inputs import install_huanyu_rollout_inputs

        validate_huanyu_config(model_config.hf_config)
        processor = build_huanyu_processor(model_path, model_config)
        install_huanyu_rollout_inputs()
        return processor

    @classmethod
    def configure_model(cls, module, model_config):
        from .audio_runtime import attach_huanyu_audio
        from .forward import install_huanyu_forward

        validate_huanyu_config(module.config)
        if _value(model_config, "use_remove_padding", True):
            raise ValueError("Huanyu initial training integration requires actor_rollout_ref.model.use_remove_padding=false")
        if os.environ.get("MINICPMO_TUNE_AUDIO", "0") == "1":
            raise ValueError("Huanyu integration currently supports a frozen audio encoder/projector only")
        if os.environ.get("MINICPMO_USE_PRECOMPUTE", "0") == "1":
            raise ValueError("Set MINICPMO_USE_PRECOMPUTE=0 for Huanyu: Whisper caches are not compatible; use online encoding")
        module = super().configure_model(module, model_config)
        _flatten_text_config(module.config)
        token_ids = _audio_token_ids(_value(model_config, "tokenizer"), module.config)
        attach_huanyu_audio(module, model_config)

        visual = getattr(module.model, "visual", None)
        if visual is not None:
            tune_visual = os.environ.get("MINICPMO_TUNE_VISION", "0") == "1"
            visual.requires_grad_(tune_visual)
            visual.train(tune_visual)
            # Reuse only the existing visual FSDP hook, not the Whisper forward
            # builder (which also installs Whisper audio/cache behavior).
            from verl_omni_ext.models.qwen3_6_whisper.forward import patch_visual_fsdp_unshard

            patch_visual_fsdp_unshard(module)
        # Frozen audio must stay in eval mode even when the outer training
        # engine calls module.train(). The native network has no dropout, but
        # keep the intended state explicit at each actual encoder invocation.
        for child in (module.model.audio_encoder, module.model.audio_projection):
            child.requires_grad_(False)
            child.eval()
        install_huanyu_forward(module, **token_ids)
        logger.info("Huanyu training adapter selected: alias=%s, HF architecture=%s, audio=frozen, cache=off",
                    TRAINING_ARCHITECTURE, module.config.architectures)
        return module
