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

        validate_huanyu_config(model_config.hf_config)
        try:
            from .checkpoint_validation import install_checkpoint_validation
        except ModuleNotFoundError as error:
            # 上游的 checkpoint_validation.py 被 .gitignore 的 *checkpoint* 吞掉，从未入库。
            # 只在没声明校验模式时放行；声明了模式却没有实现必须炸。
            if error.name != f"{__package__}.checkpoint_validation":
                raise
            mode = os.environ.get("HUANYU_CHECKPOINT_VALIDATION_MODE", "").strip()
            if mode:
                raise RuntimeError(
                    f"HUANYU_CHECKPOINT_VALIDATION_MODE={mode} requires checkpoint_validation.py, "
                    "which is not in this checkout (ignored by .gitignore '*checkpoint*')"
                ) from error
            print(
                "[huanyu-checkpoint-validation] WARNING module missing from checkout "
                "(.gitignore '*checkpoint*'); validation hook NOT installed",
                flush=True,
            )
        else:
            install_checkpoint_validation()
        return hf_tokenizer(model_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_processor(cls, model_path, model_config):
        from .aqa_position_ids import install_huanyu_aqa_position_ids
        from .audio_training import validate_audio_mode
        from .processor import build_huanyu_processor
        from .rollout_inputs import install_huanyu_rollout_inputs
        from .text_ab_capture import install_text_ab_capture

        validate_huanyu_config(model_config.hf_config)
        validate_audio_mode(model_config)
        processor = build_huanyu_processor(model_path, model_config)
        install_huanyu_rollout_inputs()
        install_huanyu_aqa_position_ids()
        install_text_ab_capture()
        # Retry after plugin discovery: early imports can see partial modules.
        from .gen_timing import install as install_gen_timing

        install_gen_timing()
        # R3 传输补丁：与 prompt 收缩钩子同处 Ray server actor（HUANYU_EXPERT_REPLAY=1 门控）。
        from .expert_replay.server_patch import maybe_install_server_patch

        maybe_install_server_patch()
        return processor

    @classmethod
    def configure_model(cls, module, model_config):
        if os.environ.get("HUANYU_JOB_DETERMINISM") == "1":
            from .job_determinism import configure_job_determinism

            configure_job_determinism()
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
        from .fsdp_offload import install_huanyu_fsdp_offload
        from .precision import install_huanyu_actor_precision
        from .precision_policy import report_policy, install_fsdp_pin_audit
        from verl_omni_ext.models.qwen3_6_whisper.position_ids import install_tq_position_ids_patch

        validate_huanyu_config(module.config)
        # HCCL 策略自检：每个训练 rank 都过 configure_model。声明的 profile 与
        # worker env（raylet 快照）不一致时立刻炸——worker 里 AIV 被静默回退成
        # AICPU RPC 曾是 trainer 69% kernel 时间的形态（2026-11-10 诊断）。
        # 未声明 HUANYU_HCCL_PROFILE 时为 no-op（CPU 契约安全）。
        from verl_omni_ext.probes.hccl_policy import assert_hccl_policy

        assert_hccl_policy()
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
        # This adapter inherits OmniModelBase directly, so it does not run
        # Whisper's TQ restore hook. Video/image positions must keep all four
        # axes, with nested offsets describing tokens rather than channels.
        install_tq_position_ids_patch()
        module = super().configure_model(module, model_config)
        _flatten_text_config(module.config)
        token_ids = _audio_token_ids(_value(model_config, "tokenizer"), module.config)
        attach_huanyu_audio(module, model_config)
        # FSDP2 re-pads partial shards on the CPU inside Module.to(). The D2H
        # transfer must finish before that host copy, including initial load.
        # 默认由包 import 期的 fsdp2_pad_sync 保护；这里是整次 offload 串行化的
        # 备选方案，仅 HUANYU_FSDP_BLOCKING_OFFLOAD=1 时生效。
        install_huanyu_fsdp_offload(module)
        if os.environ.get("HUANYU_WEIGHT_SYNC_AUDIT_DIR", "").strip():
            from .weight_sync_audit import start_actor_audit

            start_actor_audit(module, model_config)

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
                              video_token_id=_value(module.config, "video_token_id"),
                              spatial_merge_size=_value(vision_config, "spatial_merge_size", 2))
        if not freeze_audio:
            cache_args = dict(encode_audio=encode_trainable_huanyu_audio, require_audio_only=True)
        install_huanyu_actor_precision(module)
        install_huanyu_forward(module, **token_ids, **cache_args)
        if not use_cache:
            from .vision_fsdp import install_frozen_vision_fsdp

            install_frozen_vision_fsdp(module)
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
        # 融合 MoE kernel / R3 路由回放：env 默认全关，开启时整体替换
        # Qwen3_5MoeSparseMoeBlock.forward（replay 仅内建于该 kernel）。放在
        # microbatch_sync 之后：两者都可能包 FSDPEngine.forward_backward_batch。
        from .expert_replay.moe_kernel import install_huanyu_expert_replay

        install_huanyu_expert_replay(module)
        # NaN/Inf 溯源探针（HUANYU_FINITE_PROBE=1，默认关）：装在所有 kernel /
        # replay 替换之后，hook 才覆盖实际执行的实现。
        from .kernel_perf.finite_probe import install_finite_probe

        install_finite_probe(module)
        # 独立于 rollout profiler：只记录训练侧阶段、内存和超时线程栈。
        from .train_hang_probe import install_train_hang_probe

        install_train_hang_probe(module)
        logger.info("Huanyu training adapter selected: alias=%s, HF architecture=%s, huanyu_freeze_audio=%s, "
                    "train_encoder=%s, train_projector=%s, cache=%s",
                    TRAINING_ARCHITECTURE, module.config.architectures, freeze_audio,
                    train_encoder, train_projector, "strict" if use_cache else "off")
        return module
