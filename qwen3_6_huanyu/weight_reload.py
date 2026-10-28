"""Huanyu IPC reload using the same vLLM lifecycle as the Whisper adapter.

Processed rollout parameters (notably Ascend MoE parameters) cannot safely be
overwritten as if they still had their checkpoint layout. vLLM's reload API
restores the canonical loaders and finalizes each layer into its kernel storage.
This hook only changes full-weight Huanyu updates; it does not reshape experts
or replace the LoRA protocol.
"""

from collections.abc import Mapping
from functools import wraps
import importlib
import inspect
import os
import socket
import sys


_EXTENSION_MODULE = "verl_omni.workers.rollout.vllm_rollout.utils"
_RECEIVER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
_RELOAD_MODULE = "vllm.model_executor.model_loader.reload"
_PATCH_MARKER = "_huanyu_layerwise_reload_patch"
_ENV_FLAG = "MINICPMO_HUANYU_LAYERWISE_RELOAD"
_KNOWN_ARGUMENTS = {"peft_config", "base_sync_done", "use_shm", "zmq_update_id"}
_REPORTED_FALLBACKS = set()


def _log(message, worker=None):
    rank = getattr(worker, "local_rank", os.environ.get("LOCAL_RANK", "-"))
    print(f"[huanyu-reload] host={socket.gethostname()} pid={os.getpid()} rank={rank} {message}", flush=True)


def _report_once(reason, worker=None):
    if reason not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(reason)
        _log(reason, worker)


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _enabled():
    value = os.environ.get(_ENV_FLAG, "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_ENV_FLAG} must be a boolean, got {value!r}")


def _standard_model_and_config(worker):
    getter = getattr(worker, "_get_standard_weight_model_and_config", None)
    if callable(getter):
        return getter()
    # Older verl-omni extensions expose the same model_runner without a helper.
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return None
    getter = getattr(runner, "get_model", None)
    model = getter() if callable(getter) else getattr(runner, "model", None)
    config = getattr(runner, "model_config", None)
    return (model, config) if model is not None and config is not None else None


def _is_huanyu(model, model_config):
    config = _value(model_config, "hf_config")
    if config is None:
        config = getattr(model, "config", None)
    return _value(_value(config, "audio_config"), "model_type") == "huanyu_conformer_0_6b"


def _load_audio_weights(model, weights):
    """Use live reload loaders instead of load_state_dict on temporary meta params.

    The Huanyu thinker's native names match Whisper's apm/projector convention.
    Resolve the parameter on every call: a completed layer may replace objects.
    """
    thinker = getattr(model, "thinker", None)
    remaining, count = [], 0
    prefixes = (
        ("model.audio_encoder.", "apm"),
        ("model.audio_projection.", "audio_projection_layer"),
        ("apm.", "apm"),
        ("audio_projection_layer.", "audio_projection_layer"),
    )
    for name, value in weights:
        for prefix, module_name in prefixes:
            if not name.startswith(prefix):
                continue
            module = getattr(thinker, module_name, None)
            if module is None:
                raise ValueError(f"Cannot reload {name}: thinker.{module_name} is absent")
            relative = name[len(prefix):]
            path, _, leaf = relative.rpartition(".")
            layer = module.get_submodule(path) if path else module
            param = getattr(layer, leaf)
            loader = getattr(param, "weight_loader", None)
            if not callable(loader):
                raise RuntimeError(f"No layerwise weight loader for Huanyu tensor {name}")
            loader(param, value)
            count += 1
            break
        else:
            remaining.append((name, value))
    return remaining, count


def receive_layerwise_weights(model, model_config, receiver, *, worker=None):
    """Receive one complete update; finalize once after all buckets succeeded."""
    from .audio_reload_guard import AudioReloadGuard

    try:
        reload_api = importlib.import_module(_RELOAD_MODULE)
        initialize = reload_api.initialize_layerwise_reload
        finalize = reload_api.finalize_layerwise_reload
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Huanyu IPC reload requires vLLM initialize_layerwise_reload and "
            "finalize_layerwise_reload; check the vLLM version on every rollout node"
        ) from exc

    guard = AudioReloadGuard(model)
    update_id = getattr(model, "_huanyu_weight_sync_update_id", 0) + 1
    audit = None
    audit_extra = {}
    if os.environ.get("HUANYU_WEIGHT_SYNC_AUDIT_DIR", "").strip():
        from . import weight_sync_audit

        if weight_sync_audit.enabled(update_id):
            audit = weight_sync_audit
            audit_extra = {"initialize": audit.source_info(initialize),
                           "finalize": audit.source_info(finalize),
                           "receiver": audit.source_info(receiver.receive_weights)}

    def record_model(stage, **extra):
        if audit is not None:
            audit.record(stage, model.named_parameters(), round_id=update_id,
                         extra={**audit_extra, **extra})

    # Optional evidence from the actual live model, before any reload loader
    # replaces its parameters. Constructor-time hashes would hash random/meta
    # tensors instead of the checkpoint. Limit this to the first IPC update.
    snapshot = getattr(model, "_huanyu_audio_probe_snapshot_weights", None)
    reload_count = getattr(model, "_huanyu_audio_probe_reload_count", 0)
    if callable(snapshot) and reload_count == 0:
        snapshot("before_first_reload")
    record_model("before_reload")
    initialize(model)
    record_model("after_initialize")
    audio_count, bucket_count, encoder_count, projector_count = 0, 0, 0, 0

    def load_bucket(weights, is_last=None):
        nonlocal audio_count, bucket_count, encoder_count, projector_count
        # Both callback(weights) and callback(weights, is_last) are used by verl
        # versions. Completion is receiver return, never a per-bucket finalize.
        items = list(weights.items() if isinstance(weights, Mapping) else weights)
        if audit is not None:
            audit.record("receiver_input", items, round_id=update_id,
                         extra={**audit_extra, "bucket": bucket_count + 1})
        # Deferred loaders may retain tensors until a later bucket completes a
        # layer. The IPC buffer is reused immediately after the callback's ACK.
        owned = [(name, value.clone()) for name, value in items]
        guard.ingest(owned)
        if audit is not None:
            audit.record("receiver_owned", owned, round_id=update_id,
                         extra={**audit_extra, "bucket": bucket_count + 1})
        encoder_count += sum(name.startswith(("model.audio_encoder.", "apm.")) for name, _ in owned)
        projector_count += sum(name.startswith(("model.audio_projection.", "audio_projection_layer.")) for name, _ in owned)
        remaining, count = _load_audio_weights(model, owned)
        audio_count += count
        bucket_count += 1
        if remaining:
            model.load_weights(remaining)
        record_model("after_load_bucket", bucket=bucket_count)

    receiver.receive_weights(on_bucket_received=load_bucket)
    # On receive/load failure, propagate it. Finalizing an incomplete update
    # cannot make it usable and must not emit a misleading completion message.
    guard.check_complete()
    finalize(model, model_config)
    record_model("after_finalize")
    meta = [name for name, value in model.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"Huanyu layerwise reload left meta parameters: {meta[:8]}")
    guard.verify(model)
    model._huanyu_weight_sync_update_id = update_id
    if callable(snapshot):
        model._huanyu_audio_probe_reload_count = reload_count + 1
        if reload_count == 0:
            snapshot("after_first_reload")
    freeze = os.environ.get("HUANYU_FREEZE_AUDIO", "unspecified").lower()
    audio_mode = "frozen" if freeze in {"1", "true", "yes", "on"} else (
        "trainable" if freeze in {"0", "false", "no", "off"} else "unspecified"
    )
    _log(f"layerwise IPC weight update completed; buckets={bucket_count} "
         f"audio_tensors={audio_count} meta_parameters=0 audio_mode={audio_mode} "
         f"audio_encoder_tensors={encoder_count} audio_projection_tensors={projector_count}", worker)


def _arguments(signature, worker, args, kwargs):
    bound = signature.bind(worker, *args, **kwargs)
    bound.apply_defaults()
    first = next(iter(signature.parameters))
    values, unknown = {}, []
    for name, param in signature.parameters.items():
        if name == first:
            continue
        value = bound.arguments.get(name)
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            if value:
                unknown.append(name)
        elif name in _KNOWN_ARGUMENTS:
            values[name] = value
        else:
            # Future defaults can alter the transfer contract too. Full Huanyu
            # updates must reject these rather than fall back to unsafe loading.
            unknown.append(name)
    return values, unknown


def bind_huanyu_weight_reload(extension):
    """Bind directly on the selected worker extension, before vLLM consumes it.

    Keeping the method in this class's own namespace supports both native
    workers that copy extension methods and workers that add it to their MRO.
    The subclass retains the base extension's __new__ and other lifecycle APIs.
    """
    original = extension.update_weights_from_ipc
    if getattr(original, _PATCH_MARKER, False):
        inherited = "update_weights_from_ipc" not in extension.__dict__
        extension.update_weights_from_ipc = original
        if inherited:
            _log(f"bound existing reload wrapper on explicit extension={extension.__module__}.{extension.__name__}")
        return True
    signature = inspect.signature(original)

    @wraps(original)
    def huanyu_update_weights(worker, *args, **kwargs):
        # A disk-only validation must never silently receive Actor weights,
        # including when the regular layerwise reload switch is disabled.
        if os.environ.get("HUANYU_CHECKPOINT_VALIDATION_MODE", "").strip().lower() == "disk":
            standard = _standard_model_and_config(worker)
            if standard is not None and _is_huanyu(*standard):
                raise RuntimeError("Huanyu disk checkpoint validation forbids Actor-to-rollout IPC updates")
        if not _enabled():
            _report_once(f"delegated to original IPC handler: {_ENV_FLAG}=0", worker)
            return original(worker, *args, **kwargs)
        standard = _standard_model_and_config(worker)
        if standard is None:
            config = getattr(worker, "model_config", None)
            if config is None:
                config = _value(getattr(worker, "vllm_config", None), "model_config")
            if _is_huanyu(None, config):
                raise RuntimeError("Huanyu IPC reload cannot resolve model_runner model/config; refusing legacy loading")
        if standard is None or not _is_huanyu(*standard):
            _report_once("delegated to original IPC handler: model is not a Huanyu AR model", worker)
            return original(worker, *args, **kwargs)
        values, unknown = _arguments(signature, worker, args, kwargs)
        if values.get("peft_config") is not None or getattr(worker, "_pending_lora_peft_config", None) is not None:
            # Original code owns pending-LoRA consumption/base_sync_done and
            # adapter IDs. Do not clear the stash or receive any buckets here.
            _report_once("delegated to original IPC handler: LoRA update", worker)
            return original(worker, *args, **kwargs)
        if unknown:
            raise RuntimeError(
                f"Huanyu layerwise IPC reload cannot safely handle unsupported arguments={unknown}; "
                f"worker extension signature={signature}. Refusing the legacy full-weight load path."
            )
        if getattr(worker, "device", None) is None:
            raise RuntimeError("Huanyu IPC reload requires an initialized worker device")
        handle = worker._get_zmq_handle()
        update_id = values.get("zmq_update_id")
        if update_id is not None:
            zmq_utils = importlib.import_module("verl_omni.workers.rollout.vllm_rollout.zmq_utils")
            handle = zmq_utils.make_update_zmq_handle(handle, update_id)
        receiver_class = importlib.import_module(_RECEIVER_MODULE).BucketedWeightReceiver
        receiver = receiver_class(
            zmq_handle=handle, device=worker.device, use_shm=values.get("use_shm", False),
        )
        _report_once(
            f"entered full Huanyu IPC update; worker={type(worker).__module__}.{type(worker).__name__}", worker,
        )
        return receive_layerwise_weights(*standard, receiver, worker=worker)

    setattr(huanyu_update_weights, _PATCH_MARKER, True)
    extension.update_weights_from_ipc = huanyu_update_weights
    _log(f"installed Huanyu-only layerwise IPC reload; extension={extension.__module__}.{extension.__name__} "
         f"enabled={int(_enabled())}")
    return True


def install_huanyu_weight_reload():
    """Compatibility installer for external engine imports.

    The registry can import engine_hooks while utils is still defining the
    original worker extension. Do not recursively request its not-yet-defined
    class: GP-004 catches that error and silently skips this last installer.
    Huanyu's stage config names worker_extension.HuanyuColocateWorkerExtension;
    importing that explicit class after utils completes is the mandatory bind.
    """
    if not _enabled():
        _report_once(f"disabled by {_ENV_FLAG}=0")
        return False
    pending = sys.modules.get(_EXTENSION_MODULE)
    if pending is not None and getattr(getattr(pending, "__spec__", None), "_initializing", False):
        if not hasattr(pending, "vLLMOmniColocateWorkerExtension"):
            _report_once("deferred: base worker module is initializing; explicit Huanyu worker extension will bind")
            return False
    try:
        extension = importlib.import_module(_EXTENSION_MODULE).vLLMOmniColocateWorkerExtension
        return bind_huanyu_weight_reload(extension)
    except Exception as error:
        # The native registry catches exceptions. Emit direct evidence before
        # propagating, and let the explicit worker extension fail startup too.
        _log(f"installation failed: {type(error).__name__}: {error}")
        raise
