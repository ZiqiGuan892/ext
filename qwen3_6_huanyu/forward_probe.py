"""Opt-in hashes of real LLM boundaries; never replay or replace arithmetic.

Uses the logits probe's request/prefix identity. Blocking CPU copies perturb
timing; this is for localization, not performance or proof of determinism.
"""

from functools import wraps
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import MethodType
import uuid

from . import logits_probe as lp


FORMAT = "huanyu_forward_v1"
FLAGS = ("use_sequence_parallel", "use_attn_reduce_scatter_for_moe", "layer_type",
         "layer_scale", "start_layer", "end_layer", "training")


def tensor_tree(value):
    """Hash ALL tensor elements without writing activations or casting dtype."""
    import torch
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to(device="cpu", non_blocking=False).contiguous().clone()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor": True, "dtype": str(cpu.dtype), "shape": list(cpu.shape),
                "sha256": lp._hash(raw)}
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(v) for v in value]
    return lp.snapshot(value)


def _targets(native):
    llm = getattr(getattr(getattr(native, "thinker", None), "llm", None), "model", None)
    if llm is None or not hasattr(llm, "layers") or not hasattr(llm, "norm"):
        raise ValueError("forward probe requires thinker.llm.model with layers and norm")
    targets = [("llm", llm)]
    start, end = getattr(llm, "start_layer", 0), getattr(llm, "end_layer", len(llm.layers))
    for i in range(start, end):
        layer = llm.layers[i]
        name = "layers.%d" % i
        targets.append((name, layer))
        attention = {"linear_attention": "linear_attn", "full_attention": "self_attn"}.get(
            getattr(layer, "layer_type", None))
        if attention is None:
            raise ValueError("unrecognized decoder layer_type at " + name)
        for child in ("input_layernorm", attention, "post_attention_layernorm", "mlp"):
            module = getattr(layer, child, None)
            if module is None or not callable(getattr(module, "forward", None)):
                raise ValueError("missing forward boundary " + name + "." + child)
            targets.append((name + "." + child, module))
        # Observe router/expert modules where exposed. Fused functional kernels
        # may bypass them; the outer MLP boundary remains mandatory.
        for child in ("gate", "experts", "shared_expert", "shared_experts"):
            module = getattr(layer.mlp, child, None)
            if module is not None and callable(getattr(module, "forward", None)):
                targets.append((name + ".mlp." + child, module))
    targets.append(("norm", llm.norm))
    return targets


def _context():
    context = lp._ACTIVE.get()
    if context is None or not context["state"]["case"].get("forward_check", {}).get("enabled", False):
        return None
    if "forward_trace" not in context:
        identity, _ = lp._identity(context)
        if identity["step"] >= context["state"]["case"]["forward_check"]["max_steps"]:
            context["forward_trace"] = None
        else:
            context["forward_trace"] = dict(identity, events=[], phase="prefill" if identity["step"] == 0 else "decode")
    return context if context["forward_trace"] is not None else None


def observe(name, boundary, value):
    context = _context()
    if context is not None:
        context["forward_trace"]["events"].append({"name": name, "boundary": boundary, "value": tensor_tree(value)})


def _attention_context():
    # Read only an already active native context. Physical cache slot numbers
    # are diagnostics, not equality requirements across processes.
    module = sys.modules.get("vllm.forward_context")
    if module is None or not hasattr(module, "get_forward_context"):
        return {"available": False}
    try:
        current = module.get_forward_context()
        metadata = getattr(current, "attn_metadata", None)
        def fields(value):
            if isinstance(value, dict):
                return {str(k): fields(v) for k, v in value.items()}
            return lp.snapshot(lp._fields(value))
        return {"available": True, "attn_metadata": fields(metadata)}
    except (AssertionError, RuntimeError) as error:
        return {"available": False, "error": str(error)}


def bind(native, state):
    check = state["case"].get("forward_check", {})
    if not check.get("enabled", False) or "forward_manifest" in state:
        return
    steps = check.get("max_steps")
    if type(steps) is not int or not 1 <= steps <= state["max_steps"]:
        raise ValueError("forward max_steps must be positive and <= logits max_steps")
    targets = _targets(native)
    manifest = {}
    for name, module in targets:
        manifest[name] = {"forward": lp._source(module.forward),
                          "flags": {k: lp.snapshot(getattr(module, k)) for k in FLAGS if hasattr(module, k)}}
        for attr, method in (("quant_method", "apply"), ("impl", "forward")):
            implementation = getattr(getattr(module, attr, None), method, None)
            if callable(implementation):
                manifest[name][attr + "." + method] = lp._source(implementation)
    manifest["compute_logits"] = {"forward": lp._source(native.compute_logits), "flags": {}}
    state["forward_manifest"] = manifest
    state["forward_required"] = [name for name, _ in targets if ".mlp." not in name] + ["compute_logits"]

    def wrap(name, original):
        signature = inspect.signature(original)
        @wraps(original)
        def forward(self, *args, **kwargs):
            context = _context()
            if context is None:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if name == "llm":
                context["forward_trace"]["attention_context"] = _attention_context()
            observe(name, "input", dict(bound.arguments))
            result = original(*args, **kwargs)
            observe(name, "output", result)
            return result
        return forward

    for name, module in targets:
        module.forward = MethodType(wrap(name, module.forward), module)


def finish(context):
    event = context.get("forward_trace")
    if event is None:
        return
    state = context["state"]
    observed = {(e["name"], e["boundary"]) for e in event["events"]}
    missing = [(name, boundary) for name in state["forward_required"] for boundary in ("input", "output")
               if (name, boundary) not in observed]
    event.update(format=FORMAT, case_id=state["case"]["case_id"], side=state["side"],
                 tp_rank=state["tp_rank"], tp_size=state["tp_size"], runtime_id=state["runtime_id"],
                 host=socket.gethostname(), pid=os.getpid(), manifest=state["forward_manifest"],
                 required=state["forward_required"], missing_boundaries=missing,
                 status="INCOMPLETE" if missing else "CAPTURED")
    folder = Path(state["root"]) / "forward" / state["side"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s-step%05d-tp%d-%s.json" % (event["sample_id"], event["step"], state["tp_rank"], uuid.uuid4().hex))
    with path.open("x", encoding="utf-8") as stream:
        json.dump(event, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    context["forward_trace"] = None


def record_termination(state, event):
    """Attach actual sampler termination to full-generation forward coverage.

    compute_logits runs before sampling, sometimes in another runner method.
    Store a separate final marker rather than guessing EOS from missing traces.
    """
    check = state["case"].get("forward_check", {})
    if (not check.get("enabled") or not check.get("until_eos")
            or event.get("status") != "complete" or not event.get("terminal_evidence")):
        return
    fields = ("request_id", "sample_id", "step", "phase", "prompt_token_ids", "prompt_token_sha256",
              "prefix_token_ids", "prefix_token_sha256", "sampled_token_ids", "terminal_evidence", "sampling_params")
    record = {key: event[key] for key in fields}
    record.update(format="huanyu_forward_termination_v1", kind="termination", status="CAPTURED",
                  case_id=state["case"]["case_id"], side=state["side"], tp_rank=state["tp_rank"],
                  tp_size=state["tp_size"], runtime_id=state["runtime_id"])
    folder = Path(state["root"]) / "forward" / state["side"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s-termination-tp%d-%s.json" % (record["sample_id"], state["tp_rank"], uuid.uuid4().hex))
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
