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


_EXTENSION_MODULE = "verl_omni.workers.rollout.vllm_rollout.utils"
_RECEIVER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
_RELOAD_MODULE = "vllm.model_executor.model_loader.reload"
_PATCH_MARKER = "_huanyu_layerwise_reload_patch"
_ENV_FLAG = "MINICPMO_HUANYU_LAYERWISE_RELOAD"
_KNOWN_ARGUMENTS = {"peft_config", "base_sync_done", "use_shm", "zmq_update_id"}
_REPORTED_FALLBACKS = set()


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


def receive_layerwise_weights(model, model_config, receiver):
    """Receive one complete update; finalize once after all buckets succeeded."""
    try:
        reload_api = importlib.import_module(_RELOAD_MODULE)
        initialize = reload_api.initialize_layerwise_reload
        finalize = reload_api.finalize_layerwise_reload
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Huanyu IPC reload requires vLLM initialize_layerwise_reload and "
            "finalize_layerwise_reload; check the vLLM version on every rollout node"
        ) from exc

    initialize(model)
    audio_count, bucket_count = 0, 0

    def load_bucket(weights, is_last=None):
        nonlocal audio_count, bucket_count
        # Both callback(weights) and callback(weights, is_last) are used by verl
        # versions. Completion is receiver return, never a per-bucket finalize.
        items = weights.items() if isinstance(weights, Mapping) else weights
        # Deferred loaders may retain tensors until a later bucket completes a
        # layer. The IPC buffer is reused immediately after the callback's ACK.
        owned = [(name, value.clone()) for name, value in items]
        remaining, count = _load_audio_weights(model, owned)
        audio_count += count
        bucket_count += 1
        if remaining:
            model.load_weights(remaining)

    receiver.receive_weights(on_bucket_received=load_bucket)
    # On receive/load failure, propagate it. Finalizing an incomplete update
    # cannot make it usable and must not emit a misleading completion message.
    finalize(model, model_config)
    meta = [name for name, value in model.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"Huanyu layerwise reload left meta parameters: {meta[:8]}")
    print(
        f"[huanyu-reload] layerwise IPC weight update completed; "
        f"buckets={bucket_count} audio_tensors={audio_count} meta_parameters=0",
        flush=True,
    )


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
            # Future defaults can alter the transfer contract too. Delegate to
            # their implementation rather than silently discarding them.
            unknown.append(name)
    return values, unknown


def install_huanyu_weight_reload():
    """Install a process-local, Huanyu-only extension wrapper, idempotently.

    Dependencies are never edited on disk. Import the reload API only when a
    Huanyu full update occurs, so training-only and RM imports remain unaffected.
    Returns False when explicitly disabled, True when installed/already present.
    """
    if not _enabled():
        return False
    extension = importlib.import_module(_EXTENSION_MODULE).vLLMOmniColocateWorkerExtension
    original = extension.update_weights_from_ipc
    if getattr(original, _PATCH_MARKER, False):
        return True
    signature = inspect.signature(original)

    @wraps(original)
    def huanyu_update_weights(worker, *args, **kwargs):
        if not _enabled():
            return original(worker, *args, **kwargs)
        standard = _standard_model_and_config(worker)
        if standard is None or not _is_huanyu(*standard):
            return original(worker, *args, **kwargs)
        values, unknown = _arguments(signature, worker, args, kwargs)
        if unknown:
            reason = tuple(unknown)
            if reason not in _REPORTED_FALLBACKS:
                _REPORTED_FALLBACKS.add(reason)
                print(
                    f"[huanyu-reload] delegated to original IPC handler: "
                    f"unsupported arguments={unknown}", flush=True,
                )
            return original(worker, *args, **kwargs)
        if values.get("peft_config") is not None or getattr(worker, "_pending_lora_peft_config", None) is not None:
            # Original code owns pending-LoRA consumption/base_sync_done and
            # adapter IDs. Do not clear the stash or receive any buckets here.
            return original(worker, *args, **kwargs)
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
        return receive_layerwise_weights(*standard, receiver)

    setattr(huanyu_update_weights, _PATCH_MARKER, True)
    extension.update_weights_from_ipc = huanyu_update_weights
    print("[huanyu-reload] installed Huanyu-only layerwise IPC reload", flush=True)
    return True
