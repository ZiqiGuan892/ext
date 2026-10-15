"""Bounded observations of the installed GDN kernel, never a replacement kernel.

Only CPU snapshots (at most 64 values per tensor) survive a sampling call. The
real operator is called by its original wrapper exactly once. No reference
formula is assumed: state layout, normalization and active slots are backend
contracts. An all-finite sample does NOT establish numerical correctness.
"""

import math
import os
import sys
import threading

import torch

from . import diagnostic_io as diag


PROBE_REVISION = "gdn_runtime_coverage_20261015"
SAMPLE_CALLS = (1, 2, 16, 128, 1024, 8192, 32768, 131072)
_calls = {}
_lock = threading.Lock()
_INPUTS = ("state", "ssm_state", "query", "key", "value", "g", "beta")
_INDICES = ("ssm_state_indices", "state_indices", "actual_seq_lengths", "num_accepted_tokens")


def _capture_reason(value):
    if value.device.type == "cpu":
        return None
    backend = getattr(torch, value.device.type, None)
    check = getattr(backend, "is_current_stream_capturing", None)
    if not callable(check) and value.device.type == "npu":
        # Do not import/init a backend just to observe it.
        backend = sys.modules.get("torch_npu.npu")
        check = getattr(backend, "is_current_stream_capturing", None)
    if not callable(check):
        return "capture_status_unknown"
    try:
        # The check is for the current device; do not switch devices for a probe.
        current = getattr(backend, "current_device", None)
        if callable(current) and value.device.index is not None and current() != value.device.index:
            return "tensor_on_noncurrent_device"
        return "graph_capture_active" if check() else None
    except Exception:
        return "capture_status_unknown"


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _sample(value):
    try:
        return _sample_impl(value)
    except Exception as exc:
        # A failure to observe one tensor must not hide the other boundaries.
        return {"status": "observation_error", "error": type(exc).__name__}


def _sample_impl(value):
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        return {"status": "skipped", "reason": "unsupported_or_missing_tensor"}
    meta = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if value.layout != torch.strided or not value.is_contiguous():
        return dict(meta, status="skipped", reason="noncontiguous_or_nondense")
    if value.is_complex() or value.is_quantized:
        return dict(meta, status="skipped", reason="unsupported_dtype")
    reason = _capture_reason(value)
    if reason:
        return dict(meta, status="skipped", reason=reason)
    size = value.numel()
    count = min(64, size)
    positions = [i * (size - 1) // (count - 1) for i in range(count)] if count > 1 else list(range(count))
    with torch.no_grad():
        # Never flatten/contiguous-copy a noncontiguous cache. Only <=64 selected
        # elements move to CPU; copy=True also protects CPU in-place test inputs.
        indices = torch.tensor(positions, dtype=torch.long, device=value.device)
        values = value.detach().view(-1).index_select(0, indices).to(device="cpu", copy=True)
        values = values.to(torch.float64)
    finite = torch.isfinite(values)
    valid = values[finite]
    return dict(
        meta, status="sampled", numel=size, stride=list(value.stride()), count=count,
        positions=positions, finite=int(finite.sum()), nan=int(torch.isnan(values).sum()),
        posinf=int(torch.isposinf(values).sum()), neginf=int(torch.isneginf(values).sum()),
        min=_number(valid.min()) if valid.numel() else None,
        max=_number(valid.max()) if valid.numel() else None,
        max_abs=_number(valid.abs().max()) if valid.numel() else None,
        rms=_number(valid.square().mean().sqrt()) if valid.numel() else None,
        _values=values,
    )


def _public(snapshot):
    # Do not put sample values in JSON, or retain accelerator tensors in traces.
    record = {k: v for k, v in snapshot.items() if k not in ("_values", "positions")}
    if snapshot.get("status") == "sampled":
        record["sample_rule"] = "floor(i*(numel-1)/(count-1)); i=0..count-1"
    return record


def _compare(before, after):
    if before.get("status") != "sampled" or after.get("status") != "sampled":
        return {"status": "skipped", "reason": "sample_unavailable"}
    if before["shape"] != after["shape"] or before["positions"] != after["positions"]:
        return {"status": "skipped", "reason": "sample_locations_differ"}
    a, b = before["_values"], after["_values"]
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    both = fa & fb
    diff = b[both] - a[both]
    norm = a[both].norm().item()
    error = diff.norm().item()
    equal = (a == b) | (torch.isnan(a) & torch.isnan(b))
    return {
        "status": "compared", "count": a.numel(), "changed": int((~equal).sum()),
        "finite_to_nonfinite": int((fa & ~fb).sum()),
        "max_abs_diff": _number(diff.abs().max()) if diff.numel() else None,
        "relative_l2": _number(error / norm) if norm else (0.0 if not error else None),
    }


def _bind(raw_op, args, kwargs):
    bound = dict(kwargs)
    schema = getattr(raw_op, "_schema", None)
    if isinstance(schema, str):
        schema = torch._C.parse_schema(schema)
    arguments = getattr(schema, "arguments", None)
    if arguments is None:
        return bound, {}, "keyword_names_only; positional_binding_unavailable"
    positional = [arg for arg in arguments if not arg.kwarg_only]
    if len(args) > len(positional) or any(arg.name in kwargs for arg in positional[:len(args)]):
        return bound, {}, "ambiguous_arguments; raw_operator_validates_call"
    bound.update({arg.name: value for arg, value in zip(positional, args)})
    writes = {arg.name: bool(arg.alias_info and arg.alias_info.is_write) for arg in arguments}
    return bound, writes, "actual_operator_schema"


def _record_error(trace, exc):
    trace.setdefault("observation_errors", []).append(type(exc).__name__)


def begin(raw_op, args, before_kwargs, after_kwargs, native=False):
    """Sample before the ONE real call. Binding never changes its arguments."""
    if not diag.enabled() or os.environ.get("MINICPMO_DIAG_NUMERICS") != "1":
        return None
    with _lock:
        pid = os.getpid()
        call = min(_calls.get(pid, 0) + 1, SAMPLE_CALLS[-1] + 1)
        _calls[pid] = call
    if call not in SAMPLE_CALLS:
        return None
    trace = {"call": call, "native": native, "reference_verified": False,
             "scope": "recurrent operator only; 64 spread elements/tensor; no prefill, active-slot or kernel equivalence assertion"}
    try:
        original, _, binding = _bind(raw_op, args, before_kwargs)
        converted, writes, _ = _bind(raw_op, args, after_kwargs)
        trace["binding"] = binding
        trace["writes"] = writes
        trace["before"] = {k: _sample(original[k]) for k in _INPUTS if k in original}
        trace["after_cast"] = {k: _sample(converted[k]) for k in _INPUTS if k in converted}
        trace["cast_delta"] = {k: _compare(v, trace["after_cast"][k]) for k, v in trace["before"].items()}
        # Indices are diagnostic evidence only: never assume a cache layout.
        indices = {k: _sample(converted[k]) for k in _INDICES if k in converted}
        index_records = {}
        for k, sample in indices.items():
            record = _public(sample)
            if sample.get("status") == "sampled":
                record["sample_values"] = [_number(v) for v in sample["_values"][:16]]
            index_records[k] = record
        trace["indices"] = index_records
        controls = {k: (_number(v) if isinstance(v, float) else v) for k, v in converted.items()
                    if v is None or isinstance(v, (bool, int, float))}
        diag.emit("qwen36_gdn_numeric_begin", call=call, native=native, binding=binding,
                  probe_revision=PROBE_REVISION,
                  raw_schema=str(getattr(raw_op, "_schema", None)), scope=trace["scope"],
                  reference_verified=False, sample_calls=SAMPLE_CALLS,
                  controls=controls, schema_writable_arguments=writes,
                  inputs={k: _public(v) for k, v in trace["before"].items()},
                  after_cast={k: _public(v) for k, v in trace["after_cast"].items()},
                  cast_delta=trace["cast_delta"], indices=index_records,
                  whole_state_cast_back=(not native and original.get("state") is not converted.get("state")))
    except Exception as exc:
        _record_error(trace, exc)
    return trace


def after_kernel(trace, raw_op, args, kernel_kwargs, out, cast_back_list):
    if trace is None:
        return
    try:
        bound, writes, _ = _bind(raw_op, args, kernel_kwargs)
        trace["kernel_state"] = {k: _sample(bound[k]) for k in ("state", "ssm_state") if k in bound}
        trace["state_update"] = {k: _compare(trace.get("after_cast", {}).get(k, {}), v)
                                 for k, v in trace["kernel_state"].items()}
        # Observe changes in inputs declared readonly. A sampled change is
        # evidence to investigate (including aliasing), not proof of corruption.
        trace["readonly_changes"] = {}
        for k in _INPUTS[2:]:
            if k in bound and k in writes and not writes[k]:
                trace["readonly_changes"][k] = _compare(trace.get("after_cast", {}).get(k, {}), _sample(bound[k]))
        outputs = {}
        for k, value in bound.items():
            if k not in _INPUTS and (writes.get(k) or k in ("out", "output")) and isinstance(value, torch.Tensor):
                outputs["argument:" + k] = _sample(value)
                if len(outputs) >= 4:
                    break
        returned = list(out[:4]) if isinstance(out, (list, tuple)) else [out]
        for i, value in enumerate(returned):
            if isinstance(value, torch.Tensor) and len(outputs) < 4:
                outputs["return:" + str(i)] = _sample(value)
        trace["outputs"] = outputs
        trace["copy_sources"] = [_sample(source) for source, _ in cast_back_list[:1]]
    except Exception as exc:
        _record_error(trace, exc)


def finish(trace, cast_back_list):
    if trace is None:
        return
    try:
        copyback = []
        for index, (_, destination) in enumerate(cast_back_list[:1]):
            actual = _sample(destination)
            source = trace.get("copy_sources", [{}])[index]
            expected = dict(source)
            if source.get("status") == "sampled":
                # Match copy_ destination conversion, using only CPU snapshots.
                expected["_values"] = source["_values"].to(destination.dtype).to(torch.float64)
            original = trace.get("before", {}).get("state", {})
            cast_state = trace.get("after_cast", {}).get("state", {})
            cast_only_changes = None
            if all(v.get("status") == "sampled" for v in (source, cast_state, original, actual)):
                if all(v["positions"] == source["positions"] for v in (cast_state, original, actual)):
                    unchanged = source["_values"] == cast_state["_values"]
                    altered = original["_values"] != actual["_values"]
                    finite = torch.isfinite(original["_values"]) & torch.isfinite(actual["_values"])
                    cast_only_changes = int((unchanged & altered & finite).sum())
            copyback.append({"actual": _public(actual), "expected_vs_actual": _compare(expected, actual),
                             "original_vs_actual": _compare(original, actual),
                             "kernel_unchanged_but_writeback_changed": cast_only_changes})
        diag.emit("qwen36_gdn_numeric_result", call=trace["call"], native=trace["native"],
                  probe_revision=PROBE_REVISION,
                  scope=trace["scope"], reference_verified=False,
                  kernel_state={k: _public(v) for k, v in trace.get("kernel_state", {}).items()},
                  state_update=trace.get("state_update"), readonly_changes=trace.get("readonly_changes"),
                  outputs={k: _public(v) for k, v in trace.get("outputs", {}).items()},
                  copyback=copyback, observation_errors=trace.get("observation_errors", []))
    except Exception as exc:
        # Observation failures must not replace the model's return or exception.
        try:
            diag.emit("qwen36_gdn_numeric_observation_error", call=trace["call"], error=type(exc).__name__)
        except Exception:
            pass


def failed(trace, stage, exc):
    if trace is not None:
        try:
            diag.emit("qwen36_gdn_numeric_error", call=trace["call"], stage=stage,
                      error_type=type(exc).__name__, error=str(exc))
        except Exception:
            pass
