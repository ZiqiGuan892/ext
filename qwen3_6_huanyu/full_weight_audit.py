"""Destructive, validation-only audit of the first full Huanyu weight sync.

The reference is produced by the installed native checkpoint loader, so tensor
parallel slicing, fused projections and Ascend processing are not reimplemented
here. All live parameters are poisoned before reference loading. The caller MUST
stop the trainer after this audit: the model now contains the disk reference,
not the synchronized policy. Hashes cover every byte, in bounded CPU chunks.
"""

from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import struct
import uuid


ENV = "HUANYU_FULL_WEIGHT_AUDIT_DIR"
RUN_ENV = "HUANYU_FULL_WEIGHT_AUDIT_RUN_ID"
CHUNK_BYTES = 64 * 1024 * 1024


def enabled():
    return bool(os.environ.get(ENV, "").strip())


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _source(obj):
    result = {"name": getattr(obj, "__qualname__", str(type(obj)))}
    try:
        path = Path(inspect.getsourcefile(obj))
        result.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (OSError, TypeError):
        result["file"] = None
    return result


def _phase(name, **fields):
    print("[huanyu-full-weight-audit] phase=%s host=%s pid=%s %s" % (
        name, socket.gethostname(), os.getpid(),
        " ".join("%s=%s" % item for item in fields.items())), flush=True)


def _process_group_endpoint(group):
    """Best-effort local store inspection; never starts a collective."""
    try:
        api = importlib.import_module("torch.distributed.distributed_c10d")
        getter = getattr(api, "_get_process_group_store", None)
        process_group = getattr(group, "cpu_group", None)
        if process_group is None:
            process_group = getattr(group, "device_group", None)
        if not callable(getter) or process_group is None:
            return None
        store = getter(process_group)
        seen = set()
        for _ in range(8):
            if store is None or id(store) in seen:
                break
            seen.add(id(store))
            host, port = getattr(store, "host", None), getattr(store, "port", None)
            if isinstance(host, str) and isinstance(port, int):
                return "%s:%s" % (host, port)
            nested = getattr(store, "underlying_store", None)
            if nested is None:
                nested = getattr(store, "_underlying_non_prefix_store", None)
            store = nested
    except (ImportError, AttributeError, RuntimeError, AssertionError, KeyError, TypeError):
        pass
    return None


def _identity(worker):
    process = {"host": socket.gethostname(), "pid": os.getpid(),
               "rank": _get(worker, "rank"), "local_rank": _get(worker, "local_rank"),
               "tp_rank": None, "tp_world_size": None, "engine_group": None}
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            process["rank"] = dist.get_rank()
    except (ImportError, RuntimeError):
        pass
    try:
        api = importlib.import_module("vllm.distributed")
        group = api.get_tp_group()
        process["tp_rank"] = api.get_tensor_model_parallel_rank()
        process["tp_world_size"] = api.get_tensor_model_parallel_world_size()
        ranks = list(group.ranks)
        process["tp_global_ranks"] = ranks
        # A rank list alone is not unique: separate engine replicas reuse ranks.
        # Prefer the actual process-group rendezvous. Absent identity must
        # remain incomplete rather than grouping unrelated replicas together.
        store = getattr(group, "tcp_store", None)
        rendezvous = _get(worker, "distributed_init_method")
        address, port = os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")
        if store is not None and hasattr(store, "host") and hasattr(store, "port"):
            endpoint, source = "%s:%s" % (store.host, store.port), "tp_tcp_store"
        elif rendezvous:
            endpoint, source = str(rendezvous), "worker_distributed_init_method"
        elif (store_endpoint := _process_group_endpoint(group)) is not None:
            endpoint, source = store_endpoint, "tp_process_group_store"
        elif address and port:
            endpoint, source = "%s:%s" % (address, port), "MASTER_environment"
        else:
            endpoint, source = None, None
        if endpoint:
            process["engine_group"] = "%s/tp=%s" % (endpoint, ",".join(map(str, ranks)))
            process["engine_group_source"] = source
    except (ImportError, AttributeError, RuntimeError, AssertionError):
        pass
    return process


def _validate(model_config, worker):
    if not enabled():
        raise RuntimeError("%s is required" % ENV)
    if os.environ.get("HUANYU_CHECKPOINT_VALIDATION_MODE") != "sync":
        raise RuntimeError("Full weight audit requires checkpoint validation mode=sync")
    raw = _get(model_config, "model")
    if not raw or not Path(raw).is_dir():
        raise ValueError("Full weight audit requires an existing local checkpoint directory")
    checkpoint = Path(raw).resolve()
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping:
            raise ValueError("Safetensors index has no weight_map")
        files = sorted(set(mapping.values()))
    else:
        files = sorted(p.name for p in checkpoint.glob("*.safetensors"))
    if not files:
        raise ValueError("No local safetensors checkpoint files")
    headers = {}
    for name in files:
        relative = Path(name)
        path = checkpoint / relative
        # Local Hugging Face snapshots may legitimately symlink shard files to
        # a local blob cache. Reject index path traversal, not those symlinks.
        if relative.is_absolute() or ".." in relative.parts \
                or path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("Invalid or missing local checkpoint shard: %s" % name)
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Truncated safetensors header: %s" % path)
            length = struct.unpack("<Q", length_bytes)[0]
            if not 2 <= length <= min(path.stat().st_size - 8, 100 * 1024 * 1024):
                raise ValueError("Invalid safetensors header size: %s" % path)
            raw_header = handle.read(length)
        header = json.loads(raw_header)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        headers[name] = {"header_sha256": hashlib.sha256(raw_header).hexdigest(),
                         "file_bytes": path.stat().st_size, "tensor_count": len(tensors),
                         "dtype_counts": dict(Counter(value["dtype"] for value in tensors.values()))}
    config = _get(worker, "vllm_config") or _get(_get(worker, "model_runner"), "vllm_config")
    load = _get(config, "load_config")
    load_format = _get(load, "load_format")
    load_format = getattr(load_format, "value", load_format)
    if load_format != "safetensors":
        raise ValueError("Actual worker load_config.load_format must be safetensors, got %r" % load_format)
    if _get(model_config, "quantization") or _get(config, "quant_config"):
        raise ValueError("Full weight audit does not support quantized models")
    if _get(config, "lora_config") is not None:
        raise ValueError("Full weight audit does not support LoRA")
    return {"checkpoint": str(checkpoint), "load_format": load_format,
            "checkpoint_files": files, "checkpoint_headers": headers,
            "model_dtype": str(_get(model_config, "dtype")),
            "model_class_source": None}


def _chunks(tensor, budget=CHUNK_BYTES):
    """Yield logical C-order slices, including noncontiguous tensors, bounded."""
    if tensor.numel() * tensor.element_size() <= budget:
        yield tensor
        return
    if tensor.ndim == 0:
        raise ValueError("Chunk budget is smaller than one tensor element")
    row_bytes = tensor[0].numel() * tensor.element_size()
    if row_bytes <= budget:
        rows = max(1, budget // max(1, row_bytes))
        for start in range(0, tensor.shape[0], rows):
            yield tensor[start:start + rows]
    else:
        for row in tensor:
            yield from _chunks(row, budget)


def _fingerprint(tensor):
    import torch
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "numel": tensor.numel(), "is_meta": bool(tensor.is_meta),
              "device": str(tensor.device), "stride": list(tensor.stride()),
              "finite": None, "sha256": None}
    if tensor.is_meta:
        return result
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("Only dense, unquantized tensors can be fingerprinted")
    digest = hashlib.sha256()
    finite = True
    for part in _chunks(tensor.detach()):
        # Blocking D2H is required: CPU hashing must not read an unfinished copy.
        cpu = part.to(device="cpu", non_blocking=False).contiguous()
        if cpu.is_floating_point() or cpu.is_complex():
            finite = finite and bool(torch.isfinite(cpu).all())
        digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
    result.update(sha256=digest.hexdigest(), finite=finite)
    return result


def _snapshot(model, phase=None):
    import torch
    params = list(model.named_parameters(remove_duplicate=False))
    if not params:
        raise ValueError("Model has no registered parameters")
    memo, first, aliases, parameters, buffers = {}, {}, {}, {}, {}
    for index, (name, tensor) in enumerate(params, 1):
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) \
                or tensor.layout != torch.strided:
            raise ValueError("Unsupported parameter dtype/layout: %s %s" % (name, tensor.dtype))
        identity = id(tensor)
        if identity not in memo:
            memo[identity] = _fingerprint(tensor)
            first[identity] = name
        else:
            aliases[name] = first[identity]
        parameters[name] = dict(memo[identity])
        if phase and index % 100 == 0:
            _phase(phase + "_progress", parameters=index, total=len(params))
    persistent = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for leaf in module._buffers:
            key = module_name + "." + leaf if module_name else leaf
            persistent[key] = leaf not in module._non_persistent_buffers_set
    for name, tensor in model.named_buffers(remove_duplicate=False):
        if id(tensor) not in memo:
            memo[id(tensor)] = _fingerprint(tensor)
        buffers[name] = dict(memo[id(tensor)], persistent=persistent.get(name, False))
    return {"parameters": parameters, "buffers": buffers, "aliases": aliases,
            "unique_parameter_objects": len(first),
            "unique_parameter_numel": sum(memo[key]["numel"] for key in first)}


def snapshot_initial_disk(model, model_config, worker):
    """Capture the resident native safetensors model before first initialization."""
    _phase("initial_disk_snapshot_start")
    provenance = _validate(model_config, worker)
    if getattr(model, "_huanyu_weight_sync_update_id", 0):
        raise RuntimeError("Initial disk baseline must precede every Actor sync")
    provenance["model_class_source"] = _source(type(model))
    snapshot = _snapshot(model, "initial_disk_snapshot")
    if any(not p["finite"] for p in snapshot["parameters"].values()):
        raise ValueError("Initial disk baseline contains meta or nonfinite parameters")
    snapshot["provenance"] = provenance
    snapshot["process"] = _identity(worker)
    _phase("initial_disk_snapshot_complete", parameter_names=len(snapshot["parameters"]),
           unique_parameter_numel=snapshot["unique_parameter_numel"])
    return snapshot


def _compare(initial, synchronized, reference, *, allow_nonfinite=False):
    rows = {}
    for name in sorted(set(initial) | set(synchronized) | set(reference)):
        fingerprints = {"initial_disk": initial.get(name), "after_sync": synchronized.get(name),
                        "disk_replay": reference.get(name)}
        row = dict(fingerprints, status="PASS", differences=[])
        if any(value is None or value.get("is_meta") or type(value.get("finite")) is not bool
               or (not allow_nonfinite and value.get("finite") is not True)
               or not value.get("sha256") for value in fingerprints.values()):
            row.update(status="INCOMPLETE", reason="missing_meta_or_nonfinite")
        else:
            for left, right in (("initial_disk", "disk_replay"), ("after_sync", "disk_replay")):
                differing = [key for key in ("shape", "dtype", "sha256")
                             if fingerprints[left][key] != fingerprints[right][key]]
                if differing:
                    row["differences"].append({"left": left, "right": right, "fields": differing})
            if row["differences"]:
                row["status"] = "FAIL"
                row["reason"] = ("reference_reload_changes_disk_baseline"
                                 if row["differences"][0]["left"] == "initial_disk"
                                 else "sync_differs_from_disk")
        rows[name] = row
    return rows


def _status(rows):
    statuses = {row["status"] for row in rows.values()}
    if not rows or "INCOMPLETE" in statuses:
        return "INCOMPLETE"
    return "FAIL" if "FAIL" in statuses else "PASS"


def _write(report):
    root = Path(os.environ[ENV]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("full_weights-%s-%s-%s.json" % (socket.gethostname(), os.getpid(), uuid.uuid4().hex))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("[huanyu-full-weight-audit] status=%s report=%s audit_only=True generation_forbidden=True" %
          (report["status"], path), flush=True)
    return str(path)


def _checkpoint_iterator(loader, model_config, model, report):
    getter = getattr(loader, "get_all_weights", None)
    if not callable(getter):
        # Some installed vLLM versions retain the private name for the same API.
        getter = getattr(loader, "_get_all_weights", None)
    if not callable(getter):
        raise RuntimeError("Installed DefaultModelLoader lacks get_all_weights(model_config, model)")
    inspect.signature(getter).bind(model_config, model)
    report["native_sources"]["checkpoint_iterator"] = _source(getter)
    stats = report["checkpoint_stream"]
    if getattr(model, "secondary_weights", None):
        raise ValueError("Secondary weight sources are unsupported in a local-only full audit")
    seen, manifest = set(), {}
    import torch
    for name, tensor in getter(model_config, model):
        if name in seen:
            raise ValueError("Duplicate native checkpoint stream key: %s" % name)
        seen.add(name)
        stats["tensor_count"] += 1
        dtype = str(tensor.dtype)
        stats["dtype_tensor_counts"][dtype] = stats["dtype_tensor_counts"].get(dtype, 0) + 1
        stats["numel"] += tensor.numel()
        raw_digest = hashlib.sha256()
        changed, max_abs, nonfinite = 0, 0.0, 0
        for part in _chunks(tensor.detach()):
            cpu = part.to(device="cpu", non_blocking=False).contiguous()
            raw_digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
            if cpu.is_floating_point():
                finite = torch.isfinite(cpu)
                nonfinite += int((~finite).sum())
                if cpu.dtype == torch.bfloat16:
                    continue
                rounded = cpu.to(torch.bfloat16).to(cpu.dtype)
                changed += int(((cpu != rounded) & finite).sum())
                if bool(finite.any()):
                    maximum = float((cpu[finite].double() - rounded[finite].double()).abs().max())
                    max_abs = max(max_abs, maximum)
        stats["bf16_rounding_changed_elements"] += changed
        stats["nonfinite_elements"] += nonfinite
        if changed or nonfinite:
            stats["rounding_tensors"][name] = {
                "source_dtype": dtype, "changed_elements": changed,
                "max_abs_error": max_abs if math.isfinite(max_abs) else "inf",
                "nonfinite_elements": nonfinite}
        manifest[name] = {"dtype": dtype, "shape": list(tensor.shape), "sha256": raw_digest.hexdigest()}
        yield name, tensor
    # The digest remains absent if a broken model loader fails to consume its
    # complete iterator. Same path on different nodes does not prove same data.
    stats["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _reference_loader_weights(model, weights):
    """Audio uses the live deferred loaders; native loading owns all TP/fusion.

    Some native Huanyu loaders use load_state_dict for audio. After the reload
    initializer those parameters are meta, where that copy would be a no-op.
    Reuse the established bridge instead of bypassing the reload lifecycle.
    """
    audio_prefixes = ("model.audio_encoder.", "model.audio_projection.",
                      "apm.", "audio_projection_layer.")
    for name, tensor in weights:
        if name.startswith(audio_prefixes):
            from .weight_reload import _load_audio_weights
            remaining, count = _load_audio_weights(model, [(name, tensor)])
            if count != 1 or remaining:
                raise RuntimeError("Audio checkpoint bridge did not load %s" % name)
        else:
            yield name, tensor


def audit_first_sync(model, model_config, worker, initialize, finalize, *, initial):
    """Write the three-way report and leave a permanently audit-only model.

    Numerical mismatch returns a FAIL/INCOMPLETE report. API or loading failures
    also write INCOMPLETE, then raise. No model from either path may generate.
    """
    report = {"schema_version": 1, "status": "INCOMPLETE",
              "run_id": os.environ.get(RUN_ENV) or str(Path(os.environ[ENV]).resolve()),
              "checkpoint": _get(model_config, "model"), "process": _identity(worker),
              "parameters": {}, "buffers": {}, "summary": {}, "errors": [],
              "audit_only": True, "generation_forbidden": True,
              "comparison": "initial_native_disk_vs_first_sync_vs_poisoned_native_disk_replay",
              "native_sources": {"initialize": _source(initialize), "finalize": _source(finalize),
                                 "model_load_weights": _source(model.load_weights)},
              "checkpoint_stream": {"tensor_count": 0, "numel": 0, "dtype_tensor_counts": {},
                                    "bf16_rounding_changed_elements": 0, "rounding_tensors": {},
                                    "nonfinite_elements": 0, "manifest_sha256": None}}
    # Set before any possible mutation, so exception handlers cannot continue
    # serving with NaNs or reference weights after an unsuccessful audit.
    model._huanyu_full_weight_audit_generation_forbidden = True

    try:
        provenance = _validate(model_config, worker)
        report["checkpoint"] = provenance["checkpoint"]
        if not initial or initial.get("provenance", {}).get("checkpoint") != provenance["checkpoint"]:
            raise ValueError("Missing or different-checkpoint initial native disk baseline")
        if getattr(model, "_huanyu_weight_sync_update_id", 0) != 0:
            raise RuntimeError("Audit is permitted only at the first sync boundary")
        report["provenance"] = initial["provenance"]
        _phase("after_sync_snapshot_start")
        synchronized = _snapshot(model, "after_sync_snapshot")
        _phase("after_sync_snapshot_complete", parameter_names=len(synchronized["parameters"]))
        # Preserve the two available observations even if the native reference
        # loader subsequently errors, so INCOMPLETE reports remain diagnostic.
        report["parameters"] = _compare(initial["parameters"], synchronized["parameters"], {})
        report["buffers"] = _compare(initial["buffers"], synchronized["buffers"], {}, allow_nonfinite=True)
        # Native loader decides TP slicing, fused projection names and processing.
        default = importlib.import_module("vllm.model_executor.model_loader.default_loader")
        config = importlib.import_module("vllm.config")
        loader = default.DefaultModelLoader(config.LoadConfig(load_format="safetensors"))
        report["native_sources"]["loader"] = _source(type(loader))
        import torch
        _phase("poisoned_disk_replay_start")
        with torch.no_grad():
            for param in model.parameters():
                if param.is_meta or not param.is_floating_point():
                    raise ValueError("Cannot poison meta/nonfloating inference parameters")
                param.fill_(float("nan"))
            initialize(model)
            loaded = model.load_weights(_reference_loader_weights(
                model, _checkpoint_iterator(loader, model_config, model, report)))
            if loaded is not None:
                # Native loaders commonly return the loaded parameter-name set.
                report["native_loaded_parameter_names"] = sorted(map(str, loaded))
            finalize(model, model_config)
        _phase("poisoned_disk_replay_complete", checkpoint_tensors=report["checkpoint_stream"]["tensor_count"])
        _phase("disk_replay_snapshot_start")
        reference = _snapshot(model, "disk_replay_snapshot")
        _phase("disk_replay_snapshot_complete", parameter_names=len(reference["parameters"]))
        report["parameters"] = _compare(initial["parameters"], synchronized["parameters"], reference["parameters"])
        report["buffers"] = _compare(
            initial["buffers"], synchronized["buffers"], reference["buffers"], allow_nonfinite=True)
        for name, row in report["buffers"].items():
            row["comparison_scope"] = "registered_buffer_threeway_not_poisoned"
            row["persistent"] = reference["buffers"].get(name, {}).get("persistent", False)
        report["aliases"] = {"initial_disk": initial["aliases"], "after_sync": synchronized["aliases"],
                             "disk_replay": reference["aliases"]}
        report["summary"] = {
            "parameter_names": len(report["parameters"]),
            "parameter_numel_with_aliases": sum(p["numel"] for p in synchronized["parameters"].values()),
            "unique_parameter_objects": synchronized["unique_parameter_objects"],
            "unique_parameter_numel": synchronized["unique_parameter_numel"],
            "parameter_statuses": dict(Counter(p["status"] for p in report["parameters"].values())),
            "buffer_names": len(report["buffers"]),
            "buffer_statuses": dict(Counter(p["status"] for p in report["buffers"].values())),
            "buffers_independently_loaded_from_checkpoint": False,
        }
        report["status"] = _status(report["parameters"])
        if (report["checkpoint_stream"]["tensor_count"] == 0
                or report["checkpoint_stream"]["nonfinite_elements"]
                or not report["checkpoint_stream"]["manifest_sha256"]):
            report["status"] = "INCOMPLETE"
            report["errors"].append("Empty, nonfinite or not fully consumed checkpoint stream")
        # A changed registered buffer can also change model execution. It is
        # reported separately without claiming poisoning coverage for buffers.
        buffer_status = _status(report["buffers"]) if report["buffers"] else "PASS"
        if buffer_status == "INCOMPLETE":
            report["status"] = "INCOMPLETE"
        elif buffer_status == "FAIL" and report["status"] == "PASS":
            report["status"] = "FAIL"
        path = _write(report)
        model._huanyu_full_weight_audit_complete = True
        model._huanyu_full_weight_audit_report_path = path
        return path
    except Exception as exc:
        report["status"] = "INCOMPLETE"
        report["errors"].append("%s: %s" % (type(exc).__name__, exc))
        model._huanyu_full_weight_audit_report_path = _write(report)
        raise
