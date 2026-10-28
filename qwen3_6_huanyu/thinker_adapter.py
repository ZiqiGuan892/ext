"""Independent Huanyu training adapter; keeps the checkpoint HF architecture."""

import os
from collections.abc import Mapping

from verl_omni.pipelines.model_base import OmniModelBase

from .training_logging import get_training_logger


logger = get_training_logger(__name__)
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
        from .checkpoint_validation import install_checkpoint_validation

        validate_huanyu_config(model_config.hf_config)
        install_checkpoint_validation()
        return hf_tokenizer(model_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_processor(cls, model_path, model_config):
        from .aqa_position_ids import install_huanyu_aqa_position_ids
        from .audio_training import validate_audio_mode
        from .processor import build_huanyu_processor
        from .rollout_inputs import install_huanyu_rollout_inputs

        validate_huanyu_config(model_config.hf_config)
        validate_audio_mode(model_config)
        processor = build_huanyu_processor(model_path, model_config)
        install_huanyu_rollout_inputs()
        install_huanyu_aqa_position_ids()
        return processor

    @classmethod
    def configure_model(cls, module, model_config):
        from .audio_runtime import attach_huanyu_audio
        from .audio_training import (
            configure_trainable_audio,
            encode_trainable_huanyu_audio,
            resolve_audio_checkpointing,
            resolve_audio_train_components,
            validate_audio_mode,
        )
        from .entropy import install_huanyu_entropy_chunking
        from .forward import install_huanyu_forward
        from .precision import install_huanyu_actor_precision
        from .precision_policy import report_policy, install_fsdp_pin_audit

        validate_huanyu_config(module.config)
        if report_policy("actor/ref") == "full":
            install_fsdp_pin_audit()
        if _value(model_config, "use_remove_padding", True):
            raise ValueError("Huanyu initial training integration requires actor_rollout_ref.model.use_remove_padding=false")
        freeze_audio = validate_audio_mode(model_config)
        train_encoder, train_projector = resolve_audio_train_components(model_config)
        if os.environ.get("MINICPMO_USE_PRECOMPUTE", "0") == "1":
            raise ValueError("Set MINICPMO_USE_PRECOMPUTE=0 for Huanyu: Whisper caches are not compatible; use online encoding")
        use_cache = os.environ.get("HUANYU_USE_PRECOMPUTE", "0") == "1"
        if use_cache:
            if not os.environ.get("HUANYU_EMBED_CACHE_DIR"):
                raise ValueError("HUANYU_USE_PRECOMPUTE=1 requires HUANYU_EMBED_CACHE_DIR")
            if os.environ.get("MINICPMO_TUNE_VISION", "0") == "1":
                raise ValueError("Huanyu precompute requires frozen vision and audio")
            vision_config = _value(module.config, "vision_config")
            if _value(vision_config, "deepstack_visual_indexes", []):
                raise ValueError("Huanyu precompute does not support deepstack visual features")
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
        # Save the explicit mode in future checkpoints as well as the resolved
        # runtime config. Optimizer construction happens after this adapter.
        module.config.huanyu_freeze_audio = freeze_audio
        module.config.huanyu_train_audio_encoder = train_encoder
        module.config.huanyu_train_audio_projector = train_projector
        configure_trainable_audio(module, checkpoint_chunks=resolve_audio_checkpointing(model_config),
                                  train_encoder=train_encoder, train_projector=train_projector)
        cache_args = {}
        if use_cache:
            cache_args = dict(use_precompute=True, image_token_id=_value(module.config, "image_token_id"),
                              spatial_merge_size=_value(vision_config, "spatial_merge_size", 2))
        if not freeze_audio:
            cache_args = dict(encode_audio=encode_trainable_huanyu_audio, require_audio_only=True)
        install_huanyu_actor_precision(module)
        install_huanyu_forward(module, **token_ids, **cache_args)
        # old_log_prob may call the unchunked verl utility despite the engine
        # chunk flags. Install here, after Huanyu validation/loading succeeds.
        install_huanyu_entropy_chunking()
        # Opt in only for the long-context memory profile. Older verl builds
        # lack the engine's no-sync configuration field; use a Huanyu-scoped
        # compatibility guard rather than passing an unknown dataclass key.
        sync_each_microbatch = _value(module.config, "huanyu_sync_each_microbatch", False)
        if type(sync_each_microbatch) is not bool:
            raise ValueError("huanyu_sync_each_microbatch must be a boolean")
        if sync_each_microbatch:
            from .microbatch_sync import install_huanyu_microbatch_sync

            install_huanyu_microbatch_sync(module)
        logger.info("Huanyu training adapter selected: alias=%s, HF architecture=%s, huanyu_freeze_audio=%s, "
                    "train_encoder=%s, train_projector=%s, cache=%s",
                    TRAINING_ARCHITECTURE, module.config.architectures, freeze_audio,
                    train_encoder, train_projector, "strict" if use_cache else "off")
        return module
