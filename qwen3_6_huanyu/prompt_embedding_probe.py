"""Strict, opt-in capture at the real Huanyu language-model input boundary.

Only the single-request, full-prefill, eager A/B diagnostic is supported. This
observer never calls an embedding/encoder again and never changes model input.
Raw bytes preserve BF16 and signed-zero differences; NPZ contains no pickle.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
from types import MethodType
import uuid


DIRECTORY_ENV = "HUANYU_EMBEDDING_AB_DIR"
SIDE_ENV = "HUANYU_EMBEDDING_AB_SIDE"
FORMAT = "huanyu_prompt_embedding_v1"
_ACTIVE = ContextVar("huanyu_prompt_embedding_forward", default=None)
_MARKER = "_huanyu_embedding_probe_installed"
_TAG = re.compile(r"^(?:chatcmpl-)?huanyuab\.([0-9a-f]{32})\.(row_[0-9]{6,})(?:-[0-9a-f]{8})?$")


def _wraps_bound_call(original):
    """Preserve a callable's public signature when wrapping with MethodType.

    wraps(bound_method) alone makes inspect.signature remove a receiver twice:
    once through __wrapped__, and again for the new bound MethodType. Give the
    wrapper function exactly one explicit receiver, while retaining __wrapped__
    for source attribution. This also handles a static sampler callable.
    """
    signature = inspect.signature(original)
    receiver = "_huanyu_receiver"
    while receiver in signature.parameters:
        receiver = "_" + receiver
    method_signature = signature.replace(parameters=[
        inspect.Parameter(receiver, inspect.Parameter.POSITIONAL_ONLY),
        *signature.parameters.values(),
    ])

    def decorate(wrapper):
        wrapper = wraps(original)(wrapper)
        wrapper.__signature__ = method_signature
        return wrapper

    return decorate


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return _hash_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _enabled():
    return bool(os.environ.get(DIRECTORY_ENV, "").strip())


def parse_request_id(request_id):
    """Accept only an explicit case/sample tag plus vLLM's documented suffix."""
    match = _TAG.fullmatch(request_id) if isinstance(request_id, str) else None
    if match is None:
        raise ValueError("embedding A/B request lacks an unambiguous huanyuab.<case>.<sample> tag: %r" % request_id)
    return match.group(1), match.group(2)


def _load_case():
    root = Path(os.environ[DIRECTORY_ENV]).expanduser().resolve(strict=True)
    side = os.environ.get(SIDE_ENV)
    if side not in ("serve", "rollout"):
        raise ValueError("%s must be serve or rollout" % SIDE_ENV)
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    check = case.get("embedding_check", {})
    if (check.get("enabled") is not True or check.get("strict") is not True
            or check.get("scope") != "llm_inputs_embeds"):
        raise ValueError("case was not prepared for strict LLM input embedding capture")
    path = root / "manifest.jsonl"
    if _file_hash(path) != case.get("manifest_sha256"):
        raise ValueError("embedding A/B manifest changed")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = row.get("sample_id", "")
        if (re.fullmatch(r"row_[0-9]{6,}", sample) is None or sample in rows
                or row.get("case_id") != case.get("case_id")):
            raise ValueError("invalid, duplicate or foreign embedding A/B manifest row")
        row_hash = _json_hash(row["prompt"])
        if row.get("prompt_sha256", row_hash) != row_hash:
            raise ValueError("embedding A/B manifest prompt hash differs")
        rows[sample] = dict(row, prompt_sha256=row_hash)
    if not rows or len(rows) != case.get("sample_size"):
        raise ValueError("embedding A/B manifest sample count differs")
    return root, side, case, rows


def _source(obj):
    cls = type(obj)
    try:
        filename = inspect.getfile(cls)
        digest = _file_hash(filename)
    except (OSError, TypeError):
        filename, digest = None, None
    return {"class": cls.__module__ + "." + cls.__qualname__, "file": filename, "sha256": digest}


def _tp_identity():
    group = importlib.import_module("vllm.distributed.parallel_state").get_tp_group()
    return int(group.rank_in_group), int(group.world_size)


def _validate_config(config, case):
    actual_checkpoint = _get(_get(config, "model_config"), "model")
    if (not isinstance(actual_checkpoint, str)
            or Path(actual_checkpoint).expanduser().resolve() != Path(case["model_path"]).expanduser().resolve()):
        raise ValueError("embedding A/B actual model checkpoint differs from case")
    checks = (("model_config", "enforce_eager", True),
              ("scheduler_config", "max_num_seqs", 1),
              ("scheduler_config", "enable_chunked_prefill", False),
              ("cache_config", "enable_prefix_caching", False))
    for section, key, expected in checks:
        if _get(_get(config, section), key) != expected:
            raise ValueError("embedding A/B requires actual %s.%s=%r" % (section, key, expected))
    parallel = _get(config, "parallel_config")
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if _get(parallel, name, 1) != 1:
            raise ValueError("embedding A/B requires %s=1" % name)
    if _get(parallel, "tensor_parallel_size") != case["embedding_check"]["expected_tp_size"]:
        raise ValueError("embedding A/B actual tensor parallel size differs from case")
    if _get(config, "speculative_config") is not None:
        raise ValueError("embedding A/B does not support speculative decoding")


def _integer(value, name):
    import numpy as np
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % name)
    return int(value)


def _cpu_array(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", non_blocking=False).contiguous().numpy().copy()
    return np.asarray(value).copy()


def _audio_mask(features, count):
    import numpy as np
    mask, occupied, spans = np.zeros(count, dtype=np.bool_), np.zeros(count, dtype=np.bool_), []
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("actual request has no multimodal audio feature metadata")
    for feature in features:
        if _get(feature, "modality") != "audio":
            raise ValueError("embedding A/B supports audio feature metadata only")
        position = _get(feature, "mm_position")
        start, length = _integer(_get(position, "offset"), "audio offset"), _integer(_get(position, "length"), "audio length")
        if start < 0 or length <= 0 or start + length > count or occupied[start:start + length].any():
            raise ValueError("audio placeholder span is invalid or overlapping")
        occupied[start:start + length] = True
        is_embed = _get(position, "is_embed")
        if is_embed is None:
            chunk_mask = np.ones(length, dtype=np.bool_)
        else:
            chunk_mask = _cpu_array(is_embed)
            if chunk_mask.dtype != np.bool_ or chunk_mask.shape != (length,):
                raise ValueError("audio is_embed mask must be boolean with placeholder length")
        if not chunk_mask.any():
            raise ValueError("audio placeholder span has no embedding slots")
        mask[start:start + length] = chunk_mask
        spans.append({"offset": start, "length": length, "embedding_tokens": int(chunk_mask.sum()),
                      "mask_source": "mm_position.is_embed" if is_embed is not None else "mm_position.range"})
    return mask, spans


def _capture(native, llm, args, kwargs, signature):
    import numpy as np
    import torch
    context = _ACTIVE.get()
    if context is None:
        return  # Warmup/profile runs have no real scheduler request.
    state = native._huanyu_embedding_probe_state
    runner, scheduler = context["runner"], context["scheduler"]
    batch = _get(runner, "input_batch")
    req_ids = list(_get(batch, "req_ids", []))
    scheduled = _get(scheduler, "num_scheduled_tokens")
    if len(req_ids) != 1 or not isinstance(scheduled, Mapping) or set(req_ids) != set(scheduled):
        raise ValueError("embedding A/B requires exactly one active scheduled request")
    request_id = req_ids[0]
    case_id, sample = parse_request_id(request_id)
    if case_id != state["case"]["case_id"] or sample not in state["rows"]:
        raise ValueError("embedding A/B request belongs to another case or sample")
    context["sample_id"], context["request_id"] = sample, request_id
    request = _get(runner, "requests", {}).get(request_id)
    ids = _get(request, "prompt_token_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("actual runner request has no prompt token IDs")
    ids = np.asarray([_integer(x, "prompt token ID") for x in ids], dtype=np.int64)
    if (ids < 0).any():
        raise ValueError("negative actual prompt token ID")
    computed_array = _get(batch, "num_computed_tokens_cpu")
    if computed_array is None:
        raise ValueError("runner has no trustworthy computed-token metadata")
    computed = _integer(computed_array[0], "computed token count")
    if request_id in state["seen"]:
        if computed < len(ids):
            raise ValueError("embedding A/B observed repeated or partial prefill for one request")
        context["decode"] = True
        return
    count = _integer(scheduled[request_id], "scheduled token count")
    if computed != 0 or count != len(ids):
        raise ValueError("embedding A/B requires the entire uncached prompt in one prefill")
    if _get(scheduler, "total_num_scheduled_tokens") != count:
        raise ValueError("scheduler total differs from full prompt length")
    if context.get("captured"):
        raise ValueError("LLM input boundary executed twice during one prefill")
    bound = signature.bind(*args, **kwargs)
    values = bound.arguments
    embeds, positions = values.get("inputs_embeds"), values.get("positions")
    if values.get("intermediate_tensors") is not None:
        raise ValueError("embedding A/B cannot capture a non-first pipeline stage")
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 2 or embeds.shape[0] < count:
        raise ValueError("actual LLM inputs_embeds must be [tokens, hidden] with full prompt coverage")
    if embeds.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ValueError("unsupported actual input embedding dtype")
    # Padding is outside the logical request. Never include stale runner tail.
    cpu = embeds[:count].detach().to(device="cpu", non_blocking=False).contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError("actual LLM input embedding contains nonfinite values")
    raw = cpu.view(torch.uint8).reshape(count, cpu.shape[1], cpu.element_size()).numpy().copy()
    if not isinstance(positions, torch.Tensor) or positions.ndim not in (1, 2) or positions.shape[-1] < count:
        raise ValueError("actual LLM positions are missing or have unsupported shape")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("actual LLM positions must be integer")
    pos = positions[..., :count].detach().to(device="cpu", dtype=torch.int64, non_blocking=False).numpy().copy()
    if pos.ndim == 2 and pos.shape[0] != 3:
        raise ValueError("only one-dimensional or three-axis mRoPE positions are supported")
    actual_ids = values.get("input_ids")
    observed_ids = None
    if actual_ids is not None:
        observed_ids = _cpu_array(actual_ids)
        if (observed_ids.ndim != 1 or len(observed_ids) < count
                or not np.issubdtype(observed_ids.dtype, np.integer)):
            raise ValueError("actual LLM input_ids must be an integer vector covering the prompt")
        # Huanyu's native top-level forward creates dummy zero IDs when the
        # runner supplies inputs_embeds. Qwen consumes inputs_embeds instead;
        # preserve those observed IDs as diagnostics, not false prompt identity.
        observed_ids = observed_ids[:count].astype(np.int64, copy=False)
    audio_mask, spans = _audio_mask(_get(request, "mm_features"), count)
    row = state["rows"][sample]
    if len(spans) != len(row["wav_sha256"]):
        raise ValueError("actual audio feature count differs from sample WAV count")
    record = _base_record(state, sample, request_id)
    record.update(status="complete", capture_scope="llm_inputs_embeds", prefill_complete=True,
                  num_computed_tokens=computed, boundary="native.thinker.llm.model.forward",
                  prompt_token_count=count, captured_token_count=count,
                  embeddings={"dtype": str(cpu.dtype), "shape": list(cpu.shape),
                              "sha256": _hash_bytes(raw.tobytes())},
                  input_ids_source="runner.requests.prompt_token_ids", audio_spans=spans,
                  audio_token_count=int(audio_mask.sum()),
                  llm_source=_source(llm), runner_source=_source(runner),
                  llm_forward_signature=str(signature), padded_token_count=int(embeds.shape[0]),
                  observed_llm_input_ids=actual_ids is not None,
                  llm_input_ids=None if observed_ids is None else observed_ids.tolist(),
                  llm_input_ids_match_prompt=None if observed_ids is None else bool(np.array_equal(observed_ids, ids)),
                  weight_reload_count=getattr(native, "_huanyu_audio_probe_reload_count", 0))
    _write_record(state, record, {"embedding_bytes": raw, "input_ids": ids,
                                 "positions": pos, "audio_mask": audio_mask})
    state["seen"].add(request_id)
    context["captured"] = True


def _base_record(state, sample, request_id):
    row = state["rows"].get(sample, {})
    return {"format": FORMAT, "case_id": state["case"]["case_id"], "sample_id": sample,
            "side": state["side"], "request_id": request_id, "host": socket.gethostname(),
            "pid": os.getpid(), "tp_rank": state["tp_rank"], "tp_size": state["tp_size"],
            "prompt_sha256": row.get("prompt_sha256"), "wav_sha256": row.get("wav_sha256"),
            "checkpoint": state["case"]["model_path"]}


def _write_record(state, record, arrays=None):
    import numpy as np
    folder = state["root"] / "embeddings" / state["side"]
    folder.mkdir(parents=True, exist_ok=True)
    sample = record.get("sample_id") or "unknown"
    stem = "prompt_embeddings-%s-tp%d-%s" % (sample, state["tp_rank"], uuid.uuid4().hex)
    if arrays is not None:
        tensor_file = folder / (stem + ".npz")
        with tensor_file.open("xb") as handle:
            np.savez(handle, **arrays)
        record.update(tensor_file=tensor_file.name, tensor_file_sha256=_file_hash(tensor_file))
    with (folder / (stem + ".json")).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("[huanyu-embedding-ab] side=%s sample=%s tp=%s status=%s scope=llm_inputs_embeds" %
          (state["side"], sample, state["tp_rank"], record["status"]), flush=True)


def _native_for_runner(runner):
    model = _get(runner, "model")
    if getattr(model, _MARKER, False):
        return model
    if model is not None and callable(getattr(model, "modules", None)):
        matches = [module for module in model.modules() if getattr(module, _MARKER, False)]
        if len(matches) == 1:
            return matches[0]
    return None


def bind_runner_class(cls):
    original = cls.execute_model
    if getattr(original, "_huanyu_embedding_runner", False):
        return False
    signature = inspect.signature(original)
    if "scheduler_output" not in signature.parameters:
        raise RuntimeError("embedding A/B cannot bind runner execute_model without scheduler_output")

    @wraps(original)
    def execute(self, *args, **kwargs):
        native = _native_for_runner(self) if _enabled() else None
        if native is None or _ACTIVE.get() is not None:
            return original(self, *args, **kwargs)
        scheduler = signature.bind(self, *args, **kwargs).arguments["scheduler_output"]
        context = {"runner": self, "scheduler": scheduler, "captured": False}
        token = _ACTIVE.set(context)
        state = native._huanyu_embedding_probe_state
        try:
            scheduled = _get(scheduler, "num_scheduled_tokens")
            if not isinstance(scheduled, Mapping):
                raise ValueError("embedding A/B scheduler is missing scheduled-token metadata")
            # Establish identity before any native work so even a failure in
            # preprocessing is attributable to its exact request.
            if len(scheduled) == 1:
                request_id = next(iter(scheduled))
                context["request_id"] = request_id
                case_id, sample = parse_request_id(request_id)
                context["sample_id"] = sample
                if case_id != state["case"]["case_id"] or sample not in state["rows"]:
                    raise ValueError("scheduled request belongs to another case or sample")
            result = original(self, *args, **kwargs)
            for request_id in scheduled:
                case_id, sample = parse_request_id(request_id)
                context.update(request_id=request_id, sample_id=sample)
                if case_id != state["case"]["case_id"] or sample not in state["rows"]:
                    raise ValueError("scheduled request is not in this embedding case")
                if request_id not in state["seen"]:
                    raise RuntimeError("scheduled prompt never reached the instrumented actual LLM input boundary")
            return result
        except BaseException as error:
            record = _base_record(state, context.get("sample_id"), context.get("request_id"))
            record.update(status="incomplete", capture_scope="llm_inputs_embeds", prefill_complete=False,
                          error={"type": type(error).__name__, "message": str(error)})
            _write_record(state, record)
            raise
        finally:
            _ACTIVE.reset(token)

    execute._huanyu_embedding_runner = True
    cls.execute_model = execute
    print("[huanyu-embedding-ab] runner_hook=%s.%s signature=%s" %
          (cls.__module__, cls.__qualname__, signature), flush=True)
    return True


def install_runtime_probe():
    """Bind the real native runner; retry at model attach after circular imports."""
    if not _enabled():
        return False
    bound = False
    for module_name, class_name in (
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ("vllm_omni.platforms.npu.worker.npu_ar_model_runner", "NPUARModelRunner"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None and callable(getattr(cls, "execute_model", None)):
            bind_runner_class(cls)
            bound = True
    return bound


def attach_model_probe(native, vllm_config):
    if not _enabled() or getattr(native, _MARKER, False):
        return False
    hf = _get(_get(vllm_config, "model_config"), "hf_config")
    if _get(_get(hf, "audio_config"), "model_type") != "huanyu_conformer_0_6b":
        return False  # Judge models share worker environment but are not A/B inputs.
    root, side, case, rows = _load_case()
    _validate_config(vllm_config, case)
    rank, size = _tp_identity()
    if size != case["embedding_check"]["expected_tp_size"] or not 0 <= rank < size:
        raise ValueError("actual TP identity differs from embedding A/B case")
    causal_lm = getattr(getattr(native, "thinker", None), "llm", None)
    llm = getattr(causal_lm, "model", None)
    if llm is None or not callable(getattr(llm, "forward", None)):
        raise RuntimeError("embedding A/B requires the native thinker.llm.model.forward boundary")
    signature = inspect.signature(llm.forward)
    if not {"inputs_embeds", "positions"}.issubset(signature.parameters):
        raise RuntimeError("unsupported actual Huanyu LLM forward signature: %s" % signature)
    state = {"root": root, "side": side, "case": case, "rows": rows,
             "tp_rank": rank, "tp_size": size, "seen": set()}
    native._huanyu_embedding_probe_state = state
    original = llm.forward

    @_wraps_bound_call(original)
    def forward(self, *args, **kwargs):
        _capture(native, self, args, kwargs, signature)
        return original(*args, **kwargs)

    # Wrapping forward also observes native callers which invoke it directly.
    # Binding is based on the original signature, so positional inputs work.
    llm.forward = MethodType(forward, llm)
    setattr(native, _MARKER, True)
    if not install_runtime_probe():
        raise RuntimeError("embedding A/B could not install a supported native model-runner hook")
    if case.get("logits_check", {}).get("enabled", False):
        from .logits_probe import attach_model_probe as attach_logits_probe

        attach_logits_probe(native, vllm_config, state)
    print("[huanyu-embedding-ab] attached side=%s tp=%d/%d source=%s signature=%s "
          "scope=llm_inputs_embeds full_prefill_only=True" %
          (side, rank, size, _source(llm), signature), flush=True)
    return True
