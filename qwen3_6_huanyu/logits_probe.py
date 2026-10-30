"""Observe real model logits and the real greedy decision, without rerunning either.

Opt-in companion to prompt_embedding_probe. Only synchronous, single-request,
non-speculative eager runs are supported. Copies are blocking and owned: native
samplers mutate logits in place. No model input, parameter or sampling setting
is changed. This is a diagnostic, not a performance measurement.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from functools import lru_cache, wraps
import dataclasses
import enum
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import MethodType
import uuid


FORMAT = "huanyu_logits_v1"
_ACTIVE = ContextVar("huanyu_logits_forward", default=None)
_SAMPLING = ContextVar("huanyu_logits_sampling", default=None)
_MARKER = "_huanyu_logits_probe_state"
STAGES = ("model_logits", "sampler_input", "greedy_input")
REQUIRED_PARAMS = ("temperature", "top_p", "top_k", "n", "seed", "max_tokens",
                   "min_tokens", "ignore_eos", "repetition_penalty",
                   "frequency_penalty", "presence_penalty", "stop", "stop_token_ids")
PARAM_DIAGNOSTICS = {"logprobs", "prompt_logprobs", "logprob_token_ids", "detokenize",
                     "skip_special_tokens", "spaces_between_special_tokens", "output_kind",
                     "include_stop_str_in_output", "skip_clone"}
METADATA_DIAGNOSTICS = {"max_num_logprobs", "logprob_token_ids"}


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json_hash(data):
    return _hash(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")).encode("utf-8"))


@lru_cache(maxsize=256)
def _source(fn):
    chain, current, seen = [], fn, set()
    while id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "__code__", None)
        filename = getattr(code, "co_filename", None)
        chain.append({"callable": getattr(current, "__module__", "") + "." + getattr(current, "__qualname__", type(current).__name__),
                      "file": filename, "line": getattr(code, "co_firstlineno", None),
                      "file_sha256": _hash(Path(filename).read_bytes()) if filename and Path(filename).is_file() else None})
        current = getattr(current, "__wrapped__", None)
        if current is None:
            break
    fn = inspect.unwrap(fn)
    try:
        text = inspect.getsource(fn)
        filename = inspect.getsourcefile(fn)
    except (OSError, TypeError):
        text, filename = None, None
    return {"callable": getattr(fn, "__module__", "") + "." + getattr(fn, "__qualname__", type(fn).__name__),
            "file": filename, "sha256": None if text is None else _hash(text.encode("utf-8")), "wrapper_chain": chain}


def _fields(obj):
    if isinstance(obj, Mapping):
        return dict(obj)
    names = getattr(type(obj), "__struct_fields__", None)
    if names is None and dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    if names is not None:
        return {name: getattr(obj, name) for name in names}
    return dict(vars(obj)) if hasattr(obj, "__dict__") else {}


def snapshot(value, depth=0):
    """JSON evidence, preserving tensor bytes; never serialize opaque repr addresses."""
    import numpy as np
    import torch
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, enum.Enum):
        return snapshot(value.value, depth + 1)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return snapshot(value.item(), depth + 1)
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to(device="cpu", non_blocking=False).contiguous().clone()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        result = {"dtype": str(cpu.dtype), "shape": list(cpu.shape), "sha256": _hash(raw)}
        if cpu.numel() <= 4096:
            result["values"] = snapshot(cpu.tolist(), depth + 1)
        return result
    if isinstance(value, torch.Generator):
        return {"generator_seed": value.initial_seed(), "state": snapshot(value.get_state(), depth + 1)}
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if depth > 8:
        return {"unserialized": type(value).__module__ + "." + type(value).__qualname__}
    if isinstance(value, Mapping):
        return {str(k): snapshot(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((snapshot(v, depth + 1) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    return {"unserialized": type(value).__module__ + "." + type(value).__qualname__}


def snapshot_metadata(metadata):
    fields = _fields(metadata)
    for key in ("all_greedy", "all_random", "temperature", "top_p", "top_k"):
        if key not in fields:
            raise ValueError("sampling metadata missing " + key)
    processors = fields.pop("logitsprocs", None)
    result = snapshot(fields)
    # Processor objects contain mutable device state and are not ordinary
    # dataclasses. Record their implementation identity; greedy_input observes
    # the actual result of applying them, not a guessed reconstruction.
    if processors is not None:
        result["logitsprocs"] = {
            key: [_source(p.apply) for p in getattr(processors, key)]
            for key in ("argmax_invariant", "non_argmax_invariant")
        }
    return result


def snapshot_params(params):
    fields = _fields(params)
    missing = [key for key in REQUIRED_PARAMS if key not in fields]
    if missing:
        raise ValueError("actual runner SamplingParams missing " + repr(missing))
    # Includes the model EOS IDs merged by the engine, where supported.
    if hasattr(params, "all_stop_token_ids"):
        fields["all_stop_token_ids"] = params.all_stop_token_ids
    return snapshot(fields)


def _top_indices(values, count):
    """Linear selection plus a small sort; break ties by ascending token ID."""
    import numpy as np
    count = min(count, values.size)
    if count == values.size:
        ids = np.arange(values.size)
    else:
        cutoff = np.partition(values, values.size - count)[values.size - count]
        above = np.flatnonzero(values > cutoff)
        ties = np.flatnonzero(values == cutoff)[:count - above.size]
        ids = np.concatenate((above, ties))
    return ids[np.lexsort((ids, -values[ids]))]


def capture_vector(logits, top_k=100):
    import numpy as np
    import torch
    if type(top_k) is not int or top_k < 0:
        raise ValueError("logits capture top_k must be a nonnegative integer (0=full)")
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1
            or logits.shape[1] < 1 or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)):
        raise ValueError("logits probe requires one real [1, vocab] floating-point row")
    cpu = logits.detach().to(device="cpu", non_blocking=False).contiguous().clone()
    values = cpu.double().numpy().reshape(-1)
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError("invalid logits (NaN, +inf or all masked)")
    raw = cpu.view(torch.uint8).numpy()
    order = _top_indices(values, max(2, top_k) if top_k else 10)
    best = int(order[0])
    summary = {"dtype": str(cpu.dtype), "shape": list(cpu.shape), "sha256": _hash(raw.tobytes()),
               "negative_inf_count": int(np.isneginf(values).sum()), "argmax_local": best,
               "max_tie_count": int(np.count_nonzero(values == values[best])),
               "top10": [{"token_id_local": int(i), "logit": float(values[i]) if np.isfinite(values[i]) else "-inf"}
                         for i in order[:min(10, top_k) if top_k else 10]]}
    summary["top1_top2_margin"] = (float(values[best] - values[order[1]])
                                  if len(order) > 1 and np.isfinite(values[order[1]]) else None)
    if top_k:
        ids = order[:min(top_k, values.size)]
        payload = {"token_ids_local": ids.tolist(),
                   "logits": [float(values[i]) if np.isfinite(values[i]) else "-inf" for i in ids]}
        summary.update(storage="topk_json", requested_top_k=top_k, stored_shape=[1, len(ids)],
                       topk=payload, topk_sha256=_json_hash(payload),
                       sha256_scope="full_vocab_original_dtype_bytes")
        return summary, None
    summary["storage"] = "full_npz"
    return summary, raw.copy()


def _capture(logits, state):
    return capture_vector(logits, top_k=state["case"]["logits_check"].get("top_k", 100))


def _token_ids(value, name):
    from numbers import Integral
    if not isinstance(value, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value):
        raise ValueError("missing or invalid " + name)
    return list(map(int, value))


def _identity(context):
    from .prompt_embedding_probe import parse_request_id
    runner, state = context["runner"], context["state"]
    batch = runner.input_batch
    ids = list(batch.req_ids)
    if len(ids) != 1:
        raise ValueError("logits probe requires exactly one active request")
    request_id = ids[0]
    case_id, sample = parse_request_id(request_id)
    if case_id != state["case"]["case_id"] or sample not in state["rows"]:
        raise ValueError("logits request is not in this case")
    request = runner.requests[request_id]
    prompt = _token_ids(_get(request, "prompt_token_ids"), "runner prompt_token_ids")
    prefix = _token_ids(_get(request, "output_token_ids"), "runner output_token_ids")
    # A stale CPU prefix from asynchronous scheduling would attach correct
    # logits to the wrong history. Reject it instead of claiming comparability.
    if context.get("scheduler") is not None:
        scheduled = _get(context["scheduler"], "num_scheduled_tokens", {})
        computed = int(batch.num_computed_tokens_cpu[0])
        if (set(scheduled) != {request_id}
                or computed + int(scheduled[request_id]) != len(prompt) + len(prefix)):
            raise ValueError("scheduled/computed tokens do not match the observed complete prefix")
    previous = state["history"].get(request_id, [])
    if prefix != previous:
        raise ValueError("actual prefix differs from previously observed sampler outputs")
    return {"request_id": request_id, "sample_id": sample, "step": len(prefix),
            "prompt_token_ids": prompt, "prefix_token_ids": prefix,
            "prompt_token_sha256": _json_hash(prompt), "prefix_token_sha256": _json_hash(prefix)}, request


def _write(state, record, arrays=None):
    import numpy as np
    folder = state["root"] / "logits" / state["side"]
    folder.mkdir(parents=True, exist_ok=True)
    stem = "%s-step%05d-tp%d-%s" % (record.get("sample_id", "runtime"), record.get("step", 0), state["tp_rank"], uuid.uuid4().hex)
    record.update(format=FORMAT, case_id=state["case"]["case_id"], side=state["side"],
                  tp_rank=state["tp_rank"], tp_size=state["tp_size"], runtime_id=state["runtime_id"],
                  host=socket.gethostname(), pid=os.getpid())
    if arrays:
        path = folder / (stem + ".npz")
        with path.open("xb") as handle:
            np.savez(handle, **arrays)
        record.update(tensor_file=path.name, tensor_file_sha256=_hash(path.read_bytes()))
    with (folder / (stem + ".json")).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _reduce_sample(sampler):
    fn = inspect.unwrap(sampler.greedy_sample)
    fn = getattr(fn, "__func__", fn)
    getter = getattr(fn, "__globals__", {}).get("get_ascend_config")
    if getter is not None:
        value = getattr(getter(), "enable_reduce_sample", None)
        if not isinstance(value, bool):
            raise ValueError("cannot resolve actual Ascend enable_reduce_sample")
        return value
    return False  # Upstream Sampler.greedy_sample has no reduce-sample branch.


def _determinism_state():
    import torch

    # Observe the already-initialized worker; importing the Ascend module here
    # could itself change execution. Unknown version markers stay unknown.
    module = sys.modules.get("vllm_ascend.batch_invariant")
    registered = None
    if module is not None and hasattr(module, "_batch_invariant_LIB"):
        registered = module._batch_invariant_LIB is not None
    return {
        "environment": {key: os.environ.get(key) for key in (
            "VLLM_BATCH_INVARIANT", "HCCL_DETERMINISTIC", "ASCEND_LAUNCH_BLOCKING",
            "NCCL_DETERMINISTIC", "LCCL_DETERMINISTIC")},
        "batch_invariant_registered": registered,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def _bind_sampler(runner, state):
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        raise RuntimeError("runner has no sampler; unsupported logits probe integration")
    if getattr(sampler, "_huanyu_logits_bound", False):
        return
    forward, greedy = sampler.forward, sampler.greedy_sample
    signature = inspect.signature(forward)
    if not {"logits", "sampling_metadata"}.issubset(signature.parameters):
        raise RuntimeError("unsupported sampler.forward signature: " + str(signature))
    reduce_sample = _reduce_sample(sampler)
    state["reduce_sample"] = reduce_sample
    runtime = {"kind": "runtime", "status": "complete", "config": state["config"],
               "enable_reduce_sample": reduce_sample,
               "sampler_forward": _source(forward), "greedy_sample": _source(greedy),
               "sampler_sample": _source(sampler.sample), "compute_logits": _source(runner.model.compute_logits),
               "runner_execute_model": _source(runner.execute_model),
               "precision_env": {k: v for k, v in os.environ.items()
                                 if k.startswith(("MINICPMO_", "HUANYU_PRECISION_", "VLLM_ASCEND_"))
                                 or k in ("HUANYU_GRAPH_MODE", "HCCL_OP_EXPANSION_MODE", "ASCEND_WEIGHT_NZ_MODE")}}
    llm = getattr(getattr(getattr(runner.model, "thinker", None), "llm", None), "model", None)
    runtime["llm_forward"] = _source(llm.forward) if llm is not None else None
    runtime["determinism"] = _determinism_state()
    _write(state, runtime)
    print("[huanyu-determinism] actual side=%s tp_rank=%s %s" % (
        state["side"], state["tp_rank"], json.dumps(runtime["determinism"], sort_keys=True)), flush=True)

    @wraps(greedy)
    def observe_greedy(self, logits):
        event = _SAMPLING.get()
        if event is None:
            return greedy(logits)
        if "greedy_input" in event["vectors"]:
            raise RuntimeError("greedy_sample called more than once for one token")
        event["vectors"]["greedy_input"] = _capture(logits, state)
        result = greedy(logits)
        event["greedy_token_ids"] = result.detach().to("cpu", non_blocking=False).reshape(-1).tolist()
        return result

    @wraps(forward)
    def observe_forward(self, *args, **kwargs):
        context = _ACTIVE.get()
        if context is None:
            return forward(*args, **kwargs)
        identity, request = _identity(context)
        if identity["step"] >= state["max_steps"]:
            # Continue observing the prefix, without storing further vectors.
            output = forward(*args, **kwargs)
            selected = output.sampled_token_ids.detach().to("cpu", non_blocking=False).reshape(-1).tolist()
            if len(selected) != 1:
                raise RuntimeError("unexpected number of sampled tokens")
            state["history"][identity["request_id"]] = identity["prefix_token_ids"] + selected
            return output
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        metadata = bound.arguments["sampling_metadata"]
        event = dict(identity, kind="step", phase="prefill" if identity["step"] == 0 else "decode",
                     status="running", vectors={})
        event["sampling_params"] = snapshot_params(_get(request, "sampling_params"))
        event["sampling_metadata"] = snapshot_metadata(metadata)
        event["sampler_call"] = snapshot({k: v for k, v in bound.arguments.items()
                                          if k not in ("logits", "sampling_metadata")})
        event["enable_reduce_sample"] = reduce_sample
        event["logits_layout"] = "tp_vocab_shard" if reduce_sample else "replicated_vocab"
        pending = getattr(runner, "_huanyu_pending_logits", None)
        if pending is None or pending[0] != identity:
            raise RuntimeError("sampler has no matching observed model.compute_logits result")
        event["vectors"]["model_logits"] = pending[1]
        event["vectors"]["sampler_input"] = _capture(bound.arguments["logits"], state)
        runner._huanyu_pending_logits = None
        token = _SAMPLING.set(event)
        try:
            output = forward(*args, **kwargs)
            selected = output.sampled_token_ids.detach().to("cpu", non_blocking=False).reshape(-1).tolist()
            if len(selected) != 1:
                raise RuntimeError("expected exactly one sampled token")
            event["sampled_token_ids"] = selected
            event["all_greedy"] = bool(_get(metadata, "all_greedy"))
            event["all_random"] = bool(_get(metadata, "all_random"))
            event["status"] = "complete" if "greedy_input" in event["vectors"] else "incomplete"
            if event["status"] != "complete":
                event["error"] = "real greedy boundary was not executed (possibly overridden to random sampling)"
            event["greedy_output_matches_sampled"] = event.get("greedy_token_ids") == selected
            params = event["sampling_params"]
            eos = params.get("all_stop_token_ids", params.get("stop_token_ids", [])) or []
            event["terminal_evidence"] = (
                "max_tokens" if identity["step"] + 1 >= params["max_tokens"] else
                "stop_token" if not params["ignore_eos"] and identity["step"] + 1 >= params["min_tokens"]
                and selected[0] in eos else None)
            state["history"][identity["request_id"]] = identity["prefix_token_ids"] + selected
            return output
        except BaseException as error:
            event.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            _SAMPLING.reset(token)
            vectors = event.pop("vectors")
            event["logits"] = {key: value[0] for key, value in vectors.items()}
            event["tensor_storage"] = "topk_json" if all(raw is None for _, raw in vectors.values()) else "full_npz"
            # Top-K lives inline on every rank. Explicit full storage keeps
            # replicated vectors on TP0, or every shard for reduce-sample.
            arrays = None
            if event["tensor_storage"] == "full_npz" and (reduce_sample or state["tp_rank"] == 0):
                arrays, aliases, seen = {}, {}, {}
                for key, (summary, raw) in vectors.items():
                    fingerprint = (summary["dtype"], tuple(summary["shape"]), summary["sha256"])
                    target = seen.setdefault(fingerprint, key)
                    aliases[key] = target
                    if target == key:
                        arrays[key] = raw
                event["tensor_keys"] = aliases
            _write(state, event, arrays)
            if state["case"].get("forward_check", {}).get("until_eos"):
                from .forward_probe import record_termination
                record_termination(state, event)
            if identity["step"] == 0:
                print("[huanyu-logits-ab] side=%s sample=%s tp=%s all_greedy=%s reduce_sample=%s params=%s" %
                      (state["side"], identity["sample_id"], state["tp_rank"], event.get("all_greedy"),
                       reduce_sample, json.dumps(event["sampling_params"], ensure_ascii=False)), flush=True)

    sampler.forward = MethodType(observe_forward, sampler)
    sampler.greedy_sample = MethodType(observe_greedy, sampler)
    sampler._huanyu_logits_bound = True


def _bind_model(runner):
    model = runner.model
    original = model.compute_logits
    if getattr(original, "_huanyu_logits_bound", False):
        return
    state = getattr(_native(runner), _MARKER)
    forward_probe = None
    if state["case"].get("forward_check", {}).get("enabled", False):
        from . import forward_probe
        forward_probe.bind(_native(runner), state)
        state["forward_manifest"]["compute_logits"] = {"forward": _source(original), "flags": {}}

    @wraps(original)
    def compute(self, *args, **kwargs):
        if forward_probe is not None:
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            forward_probe.observe("compute_logits", "input", dict(bound.arguments))
        result = original(*args, **kwargs)
        context = _ACTIVE.get()
        if context is not None:
            identity, _ = _identity(context)
            if identity["step"] < context["state"]["max_steps"]:
                if getattr(runner, "_huanyu_pending_logits", None) is not None:
                    raise RuntimeError("compute_logits result was never consumed by the instrumented sampler")
                runner._huanyu_pending_logits = (identity, _capture(result, context["state"]))
            if forward_probe is not None:
                forward_probe.observe("compute_logits", "output", result)
                forward_probe.finish(context)
        return result

    compute._huanyu_logits_bound = True
    model.compute_logits = MethodType(compute, model)


def _native(runner):
    model = getattr(runner, "model", None)
    if getattr(model, _MARKER, None) is not None:
        return model
    matches = [m for m in model.modules() if getattr(m, _MARKER, None) is not None] if hasattr(model, "modules") else []
    return matches[0] if len(matches) == 1 else None


def bind_runner_class(cls):
    for name in ("execute_model", "sample_tokens"):
        original = getattr(cls, name, None)
        if not callable(original) or getattr(original, "_huanyu_logits_runner", False):
            continue
        def bind(original, name):
            signature = inspect.signature(original)

            @wraps(original)
            def wrapped(self, *args, **kwargs):
                native = _native(self)
                if native is None or _ACTIVE.get() is not None:
                    return original(self, *args, **kwargs)
                state = getattr(native, _MARKER)
                scheduler = signature.bind(self, *args, **kwargs).arguments.get("scheduler_output")
                _bind_sampler(self, state)
                _bind_model(self)
                context = {"runner": self, "state": state, "scheduler": scheduler}
                token = _ACTIVE.set(context)
                try:
                    return original(self, *args, **kwargs)
                except BaseException as error:
                    _write(state, {"kind": "error", "status": "incomplete",
                                   "error": type(error).__name__ + ": " + str(error)})
                    raise
                finally:
                    _ACTIVE.reset(token)
            wrapped._huanyu_logits_runner = True
            return wrapped
        setattr(cls, name, bind(original, name))


def attach_model_probe(native, config, embedding_state):
    check = embedding_state["case"].get("logits_check", {})
    if not check.get("enabled", False) or getattr(native, _MARKER, None) is not None:
        return False
    max_steps = check.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("logits_check.max_steps must be a positive integer")
    top_k = check.get("top_k", 100)
    if type(top_k) is not int or top_k < 0:
        raise ValueError("logits_check.top_k must be a nonnegative integer (0=full)")
    state = dict(embedding_state, history={}, max_steps=max_steps, runtime_id=uuid.uuid4().hex)
    # Selected live worker config; model path is recorded separately in case.
    sections = {
        "model_config": ("dtype", "max_model_len", "enforce_eager", "generation_config", "quantization"),
        "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "enable_expert_parallel"),
        "cache_config": ("cache_dtype", "block_size", "enable_prefix_caching", "mamba_cache_dtype", "mamba_ssm_cache_dtype"),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "async_scheduling"),
        "compilation_config": ("mode", "cudagraph_mode"),
    }
    state["config"] = {section: {key: snapshot(_get(_get(config, section), key, "<absent>")) for key in keys}
                       for section, keys in sections.items()}
    if _get(_get(config, "scheduler_config"), "async_scheduling", False):
        raise ValueError("logits A/B requires async_scheduling=false to observe a complete CPU prefix")
    setattr(native, _MARKER, state)
    bound = False
    for module_name, class_name in (
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ("vllm_omni.platforms.npu.worker.npu_ar_model_runner", "NPUARModelRunner"),
    ):
        try:
            cls = getattr(importlib.import_module(module_name), class_name, None)
        except ImportError:
            continue
        if cls is not None and callable(getattr(cls, "execute_model", None)):
            bind_runner_class(cls)
            bound = True
    if not bound:
        raise RuntimeError("no supported actual runner for logits capture")
    return True
