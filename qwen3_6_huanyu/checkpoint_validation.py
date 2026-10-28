"""Huanyu validation-only A/B: disk-loaded rollout versus initial Actor sync.

Both runs keep rollout weights resident (free_cache_engine=False). Skipping
on_init_end alone is unsafe if sleep has discarded the original weights.
This mode must never continue to optimizer updates or checkpoint resume.
"""

from functools import wraps
import os


ENV = "HUANYU_CHECKPOINT_VALIDATION_MODE"
_MARKER = "_huanyu_checkpoint_validation"


def _get(obj, name, default=None):
    return obj.get(name, default) if hasattr(obj, "get") else getattr(obj, name, default)


def mode():
    value = os.environ.get(ENV, "").strip().lower()
    if value not in ("", "disk", "sync"):
        raise ValueError("%s must be disk, sync, or unset" % ENV)
    return value


def validate_config(trainer):
    config = trainer.config
    train, policy = config.trainer, config.actor_rollout_ref
    rollout, model = policy.rollout, policy.model
    errors = []
    required = (
        ("trainer.val_only", _get(train, "val_only"), True),
        ("trainer.val_before_train", _get(train, "val_before_train"), True),
        ("trainer.resume_mode", _get(train, "resume_mode"), "disable"),
        ("trainer.v1.trainer_mode", _get(_get(train, "v1"), "trainer_mode"), "omni_sync"),
        ("rollout.name", _get(rollout, "name"), "vllm_omni"),
        ("rollout.load_format", _get(rollout, "load_format"), "safetensors"),
        ("rollout.free_cache_engine", _get(rollout, "free_cache_engine"), False),
    )
    for key, actual, expected in required:
        if actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
            errors.append("%s=%r (required %r)" % (key, actual, expected))
    if _get(model, "lora_rank", 0) or _get(_get(model, "lora", {}), "rank", 0) or _get(model, "lora_adapter_path"):
        errors.append("LoRA is unsupported in the checkpoint validation experiment")
    if _get(trainer, "global_steps", 0) != 0:
        errors.append("global_steps must be 0")
    if not os.environ.get("HUANYU_AUDIO_PROBE_DIR"):
        errors.append("HUANYU_AUDIO_PROBE_DIR is required to verify the actual rollout weights")
    if getattr(trainer.reward_loop_manager, "reward_loop_worker_handles", None) is None:
        errors.append("a separate reward loop is required; colocated reward evaluation may reload policy weights")
    if errors:
        raise ValueError("Huanyu checkpoint validation configuration: " + "; ".join(errors))


def _set_rollout_version(trainer):
    # Normally set by Actor.update_weights. Retain version metadata without
    # moving any tensors, so replay-buffer validation does not see None.
    import ray

    replicas = trainer.llm_server_manager.get_replicas()
    if not replicas or any(replica.server_handle is None for replica in replicas):
        raise RuntimeError("Huanyu checkpoint validation requires initialized rollout server handles")
    ray.get([replica.server_handle.set_global_steps.remote(0) for replica in replicas])


def _bind(trainer_class):
    original = trainer_class.on_init_end
    if getattr(original, _MARKER, False):
        return
    original_train = trainer_class.on_train_begin

    @wraps(original)
    def on_init_end(trainer):
        selected = mode()
        model = trainer.config.actor_rollout_ref.model
        if not selected or _get(model, "architecture") != "Qwen36HuanyuForTraining":
            return original(trainer)
        validate_config(trainer)
        if selected == "sync":
            result = original(trainer)
        else:
            result = None
        _set_rollout_version(trainer)
        setattr(trainer, _MARKER, selected)

        def reject_update(*args, **kwargs):
            raise RuntimeError("Huanyu checkpoint validation forbids subsequent Actor-to-rollout weight updates")

        # Scope the guard to this manager instance, not other jobs or the RM.
        trainer.checkpoint_manager.update_weights = reject_update
        print("[huanyu-checkpoint-validation] mode=%s initial_actor_sync=%s "
              "checkpoint=%s free_cache_engine=False val_only=True global_steps=0" %
              (selected, "skipped" if selected == "disk" else "completed", _get(model, "path")), flush=True)
        return result

    @wraps(original_train)
    def on_train_begin(trainer):
        if getattr(trainer, _MARKER, None):
            raise RuntimeError("Huanyu checkpoint validation must exit after val_before_train; training is forbidden")
        return original_train(trainer)

    setattr(on_init_end, _MARKER, True)
    trainer_class.on_init_end = on_init_end
    trainer_class.on_train_begin = on_train_begin


def install_checkpoint_validation():
    if not mode():
        return False
    # configure_tokenizer runs in TaskRunner's _init_tokenizer, before worker
    # creation and on_init_end. Import here to avoid changing normal startup.
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    _bind(PPOTrainerSync)
    return True
