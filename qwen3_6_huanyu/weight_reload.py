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
# 逐字节校验只在「首次同步」有意义：首次同步发送 checkpoint 原值，跨 run 确定性可比；
# 其后各次同步是训练更新后的权重，受训练非确定性影响，不能跨 run 对比。一进程一次。
_VERIFY_DUMPED = False


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


def _wsync_env_active():
    """仅凭 env 判断权重同步优化是否开启，不 import 任何重模块（off 路径零新增 import）。"""
    profile = os.environ.get("HUANYU_WSYNC_PROFILE", "off").strip().lower()
    if profile not in {"", "off"}:
        return True
    trace = os.environ.get("HUANYU_WSYNC_TRACE", "off").strip().lower()
    if trace not in {"", "off", "0", "false", "no"}:
        return True
    verify = os.environ.get("HUANYU_WSYNC_VERIFY", "0").strip().lower()
    return verify in {"1", "true", "yes", "on"}


def _maybe_wsync_config():
    if not _wsync_env_active():
        return None
    from verl_omni_ext.features.weight_sync.config import WsyncConfig

    return WsyncConfig.from_env()


def _build_wsync_receiver(handle, device, use_shm, cfg):
    """按 cfg 构造本地 receiver（协议与 verl v1 一致）+ rollout 侧 trace。"""
    from verl_omni_ext.features.weight_sync.receiver import EarlyAckReceiver
    from verl_omni_ext.features.weight_sync.trace import SyncTrace

    sync_fn = None
    if cfg.trace == "sync":
        try:
            from verl.utils.device import get_torch_device

            sync_fn = get_torch_device().synchronize
        except Exception:  # noqa: BLE001
            sync_fn = None
    trace = SyncTrace("rollout", sync_fn=sync_fn, extra={"cfg": cfg.summary()}) if cfg.tracing else None
    return EarlyAckReceiver(
        zmq_handle=handle, device=device, use_shm=use_shm, early_ack=cfg.early_ack, trace=trace,
    )


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


_THINKER_NATIVE_PREFIXES = ("visual.", "llm.", "apm.", "audio_projection_layer.")


def make_direct_thinker_loader(model, *, strict=False):
    """IPC 每桶直达 ``model.thinker.load_weights``；不可用时返回 ``(None, 原因)``。

    omni 顶层 ``load_weights`` 是为「从磁盘首次加载」写的：批内没有 ``visual.*`` 时从
    ``Q36_CKPT_DIR`` 重读整套视觉权重、没有 ``apm.*`` 时探测 sidecar 音频权重，并把读到的
    checkpoint 原值注入本批。IPC 同步每桶都踩中这两条分支（每桶固定 ~0.2s，且视觉若
    解冻会被 checkpoint 值覆盖）。这里复用模型自己的 ``_map_q36_weights`` 前缀表，只跳过
    注入；映射不了的张量直接 raise，不像顶层那样 warning 后丢弃。
    """
    thinker = getattr(model, "thinker", None)
    load = getattr(thinker, "load_weights", None)
    mapper = getattr(model, "_map_q36_weights", None)
    missing = [name for name, fn in (("thinker.load_weights", load), ("_map_q36_weights", mapper))
               if not callable(fn)]
    if missing:
        reason = f"direct_load unavailable on {type(model).__name__}: missing {missing}"
        if strict:
            raise RuntimeError(reason + " (HUANYU_WSYNC_STRICT=1)")
        return None, reason

    def direct(weights):
        mapped, unknown = [], []
        for name, value in weights:
            if name.startswith(_THINKER_NATIVE_PREFIXES):
                mapped.append((name, value))
                continue
            out = list(mapper([(name, value)]))
            if out:
                mapped.extend(out)
            else:
                unknown.append(name)
        if unknown:
            raise RuntimeError(
                f"direct_load cannot map {len(unknown)} tensors into the thinker namespace "
                f"(e.g. {unknown[:4]}); refusing to drop them silently"
            )
        if mapped:
            load(mapped)

    return direct, None


def receive_layerwise_weights(model, model_config, receiver, *, worker=None, cfg=None):
    """Receive one complete update; finalize once after all buckets succeeded.

    ``cfg`` is None for the default (knob-off) path — behaviour is then byte-for
    -byte identical to before. When a :class:`WsyncConfig` is passed, the bucket
    callback additionally routes fused experts through the MoE fast consumer and
    (optionally) checksums the whole model before/after for the VERIFY A/B.
    """
    global _VERIFY_DUMPED
    from .audio_reload_guard import AudioReloadGuard

    if getattr(model, "_huanyu_full_weight_audit_started", False):
        raise RuntimeError("Full weight audit forbids subsequent weight updates; restart the engine")
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
    full_audit = None
    initial_disk = None
    if os.environ.get("HUANYU_FULL_WEIGHT_AUDIT_DIR", "").strip():
        if os.environ.get("HUANYU_CHECKPOINT_VALIDATION_MODE", "").strip() != "sync":
            raise RuntimeError("Full weight audit requires guarded sync validation mode")
        if (update_id != 1 or getattr(model, "_huanyu_full_weight_audit_complete", False)
                or getattr(model, "_huanyu_full_weight_audit_started", False)):
            raise RuntimeError("Full weight audit forbids subsequent weight updates")
        from . import full_weight_audit

        full_audit = full_weight_audit
        initial_disk = full_audit.snapshot_initial_disk(model, model_config, worker)
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
    do_verify = cfg is not None and cfg.verify and not _VERIFY_DUMPED
    verify_before = None
    if do_verify:
        from verl_omni_ext.features.weight_sync import verify as _verify

        verify_before = _verify.fingerprint_model(model)
        small_before = _verify.small_values(model)

    # 索引只持有模块引用（layerwise reload 换参数不换模块），放在 initialize 之前：
    # strict 下索引不可用时在模型仍完好时报错，而不是留下一堆 meta 参数。
    moe_consumer, moe_stats = None, None
    if cfg is not None and cfg.moe_fast:
        from .moe_fast_reload import make_moe_fast_consumer

        moe_consumer, moe_stats = make_moe_fast_consumer(model, strict=cfg.strict)
    load_remaining, direct_load = model.load_weights, False
    if cfg is not None and cfg.direct_load:
        direct, reason = make_direct_thinker_loader(model, strict=cfg.strict)
        if direct is not None:
            load_remaining, direct_load = direct, True
        else:
            _report_once(f"WARNING {reason}; falling back to model.load_weights", worker)
    trace = getattr(receiver, "trace", None)

    if callable(snapshot) and reload_count == 0:
        snapshot("before_first_reload")
    record_model("before_reload")
    initialize(model)
    record_model("after_initialize")
    audio_count, bucket_count, encoder_count, projector_count = 0, 0, 0, 0

    def load_bucket(weights, is_last=None, owned=False):
        nonlocal audio_count, bucket_count, encoder_count, projector_count
        # Both callback(weights) and callback(weights, is_last) are used by verl
        # versions. Completion is receiver return, never a per-bucket finalize.
        items = list(weights.items() if isinstance(weights, Mapping) else weights)
        if audit is not None:
            audit.record("receiver_input", items, round_id=update_id,
                         extra={**audit_extra, "bucket": bucket_count + 1})
        # Deferred loaders may retain tensors until a later bucket completes a
        # layer. The IPC buffer is reused immediately after the callback's ACK.
        # ``owned=True`` means the receiver already cloned (early-ack), so skip
        # the duplicate clone here. 本地名不能叫 owned：那是入参。
        cloned = items if owned else [(name, value.clone()) for name, value in items]
        guard.ingest(cloned)
        if audit is not None:
            audit.record("receiver_owned", cloned, round_id=update_id,
                         extra={**audit_extra, "bucket": bucket_count + 1})
        encoder_count += sum(name.startswith(("model.audio_encoder.", "apm.")) for name, _ in cloned)
        projector_count += sum(name.startswith(("model.audio_projection.", "audio_projection_layer.")) for name, _ in cloned)
        remaining, count = _load_audio_weights(model, cloned)
        audio_count += count
        if moe_consumer is not None:
            remaining = moe_consumer(remaining)
        if moe_stats is not None and trace is not None:
            # 累计值直接覆盖：trace 在 receive_weights 结束时 emit，此时已是全量。
            for key in ("fast_tensors", "fast_calls", "fallback_fused_tensors"):
                trace.counters[f"moe_{key}"] = moe_stats[key]
        bucket_count += 1
        if remaining:
            if trace is not None:
                trace.counters["model_load_calls"] = trace.counters.get("model_load_calls", 0) + 1
            load_remaining(remaining)
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
    if full_audit is not None:
        # Reference replay changes the live model even if it fails partway.
        # Keep the engine unusable if a caller catches the diagnostic exception.
        model._huanyu_full_weight_audit_started = True
        if worker is not None:
            worker._huanyu_full_weight_audit_started = True

        def reject_forward(*args, **kwargs):
            raise RuntimeError("Full weight audit replaced rollout weights; generation is forbidden")

        model.forward = reject_forward
        full_audit.audit_first_sync(
            model, model_config, worker, initialize, finalize, initial=initial_disk,
        )
        # The driver stops after every worker returns. Do not record a normal
        # sync completion or an audio-probe snapshot of disk-replayed weights.
        model._huanyu_weight_sync_update_id = update_id
        return
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
         f"audio_encoder_tensors={encoder_count} audio_projection_tensors={projector_count} "
         f"direct_load={int(direct_load)}", worker)
    if moe_stats is not None:
        _log(f"moe_fast reload stats: fast_tensors={moe_stats['fast_tensors']} "
             f"fast_calls={moe_stats['fast_calls']} "
             f"fallback_fused_tensors={moe_stats['fallback_fused_tensors']} "
             f"indexed_modules={moe_stats['indexed_modules']}/{moe_stats['candidate_modules']} "
             f"ineligible={moe_stats['ineligible']}", worker)
        if moe_stats["fast_tensors"] == 0 or moe_stats["fallback_fused_tensors"]:
            _report_once(
                "WARNING moe_fast is not covering the fused experts (every fallback costs 256 "
                "weight_loader calls per shard); set HUANYU_WSYNC_STRICT=1 to fail fast", worker,
            )
    if do_verify and verify_before is not None:
        from verl_omni_ext.features.weight_sync import verify as _verify

        after = _verify.fingerprint_model(model)
        changed = _verify.diff_fingerprints(verify_before, after)
        small_after = _verify.small_values(model)
        small_changed = {name: {"before": small_before.get(name), "after": small_after[name]}
                         for name in changed if name in small_after}
        path = _verify.dump_fingerprints(cfg.profile or "custom", after, changed,
                                         extra={"small_changed": small_changed})
        _VERIFY_DUMPED = True
        _log(f"VERIFY fingerprint: changed_tensors={len(changed)} dumped={path}", worker)


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
        if getattr(worker, "_huanyu_full_weight_audit_started", False):
            raise RuntimeError("Full weight audit engine cannot be reused; restart before sync or generation")
        # A disk-only validation must never silently receive Actor weights,
        # including when the regular layerwise reload switch is disabled.
        if os.environ.get("HUANYU_CHECKPOINT_VALIDATION_MODE", "").strip().lower() == "disk":
            standard = _standard_model_and_config(worker)
            if standard is not None and _is_huanyu(*standard):
                raise RuntimeError("Huanyu disk checkpoint validation forbids Actor-to-rollout IPC updates")
        if not _enabled():
            if os.environ.get("HUANYU_FULL_WEIGHT_AUDIT_DIR", "").strip():
                raise RuntimeError("Full weight audit requires Huanyu layerwise reload enabled")
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
            if os.environ.get("HUANYU_FULL_WEIGHT_AUDIT_DIR", "").strip():
                raise RuntimeError("Full weight audit does not support LoRA updates")
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
        # 默认路径（knob off）用上游 stock receiver，签名/行为不变；仅当权重同步
        # 优化或观测开启时，换用本地 EarlyAckReceiver（协议与 v1 一致）。
        cfg = _maybe_wsync_config()
        use_shm = values.get("use_shm", False)
        if cfg is not None and cfg.needs_receiver():
            receiver = _build_wsync_receiver(handle, worker.device, use_shm, cfg)
        else:
            cfg = None
            receiver_class = importlib.import_module(_RECEIVER_MODULE).BucketedWeightReceiver
            receiver = receiver_class(
                zmq_handle=handle, device=worker.device, use_shm=use_shm,
            )
        _report_once(
            f"entered full Huanyu IPC update; worker={type(worker).__module__}.{type(worker).__name__}", worker,
        )
        return receive_layerwise_weights(*standard, receiver, worker=worker, cfg=cfg)

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
