"""Opt-in wall-clock spans for synchronous Huanyu sampling.

No tensors, prompts or answers are inspected. Nested/concurrent durations are
inclusive and must NOT be added to estimate gen wall time. Framework imports
only occur in install(), which is a no-op unless HUANYU_GEN_TIMING=1.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import json
import os
import socket
import time
import uuid


_context = ContextVar("huanyu_gen_timing", default=None)
_reported = set()
_MARKER = "_huanyu_gen_timing_phase"
PREFIX = "[huanyu-gen-timing] "


def enabled():
    return os.environ.get("HUANYU_GEN_TIMING") == "1"


def _state():
    # Resolve process-local state by import. Ray must not pickle ContextVars.
    return importlib.import_module(__name__)


def get_context():
    return dict(_state()._context.get() or {})


def emit(event, **fields):
    """Observability failures must not replace the original result/exception."""
    try:
        record = {**fields, "event": event, "host": socket.gethostname(), "pid": os.getpid()}
        print(PREFIX + json.dumps(record, ensure_ascii=True, separators=(",", ":")), flush=True)
    except Exception:
        pass


@contextmanager
def span(phase, **fields):
    if not enabled():
        yield
        return
    state = _state()
    parent = state._context.get() or {}
    context = {k: v for k, v in parent.items() if k not in ("span_id", "parent_span_id", "phase")}
    context.update(fields)
    context.update(phase=phase, span_id=uuid.uuid4().hex,
                   parent_span_id=parent.get("span_id"))
    token = state._context.set(context)
    started_ns = time.time_ns()
    started = time.perf_counter()
    error_type = None
    try:
        yield
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        state._context.reset(token)
        record = {**context, "start_unix_ns": started_ns, "end_unix_ns": time.time_ns(),
                  "elapsed_s": elapsed, "clock": "host_wall", "inclusive": True,
                  "status": "error" if error_type else "ok", "error_type": error_type}
        record.pop("event", None)
        emit("span", **record)


def _identity(arguments):
    """Read identifiers only, never stringify tensors or user payloads."""
    fields = {}
    instance = arguments.get("self")
    step = getattr(instance, "global_steps", None)
    if type(step) is int:
        fields["step"] = step
    sources = [arguments]
    for key in ("kwargs", "trajectory", "prompt"):
        value = arguments.get(key)
        if isinstance(value, dict):
            sources.append(value)
    for source in sources:
        for key, target in (("global_steps", "step"), ("step", "step"),
                            ("uid", "uid"), ("session_id", "session_id"),
                            ("request_id", "request_id"), ("validate", "validate"),
                            ("partition_id", "partition")):
            value = source.get(key)
            if type(value) in (str, int, bool):
                fields[target] = value
    for key in ("replica_rank", "node_rank", "rank", "local_trigger_step"):
        value = getattr(instance, key, None)
        if type(value) is int:
            fields[key] = value
    model_config = getattr(instance, "model_config", None)
    for key in ("path", "model", "local_path"):
        value = getattr(model_config, key, None)
        if isinstance(value, str) and value:
            fields["model"] = value
            break
    return fields


def wrap_method(target, name, phase, *, require_context=False, asynchronous=None):
    """Wrap classes or module functions, preserving async/cancellation behavior."""
    original = getattr(target, name, None)
    if not callable(original):
        return False
    if getattr(original, _MARKER, None) == phase:
        return True
    if inspect.isasyncgenfunction(original) or inspect.isgeneratorfunction(original):
        return False  # A generator's construction is not its execution lifetime.
    signature = inspect.signature(original)

    def fields(args, kwargs):
        try:
            arguments = dict(signature.bind_partial(*args, **kwargs).arguments)
            nested = arguments.get("kwargs", {})
            if isinstance(nested, dict):
                arguments.update(nested)
            result = _identity(arguments)
            if phase.startswith("server.") and "step" in result:
                result["weight_step"] = result.pop("step")
            return result
        except Exception:
            return {}

    is_async = asynchronous if asynchronous is not None else inspect.iscoroutinefunction(original)
    if is_async:
        @wraps(original)
        async def wrapped(*args, **kwargs):
            if not enabled() or (require_context and not get_context()):
                return await original(*args, **kwargs)
            with span(phase, **fields(args, kwargs)):
                return await original(*args, **kwargs)
    else:
        @wraps(original)
        def wrapped(*args, **kwargs):
            if not enabled() or (require_context and not get_context()):
                return original(*args, **kwargs)
            with span(phase, **fields(args, kwargs)):
                return original(*args, **kwargs)

    setattr(wrapped, _MARKER, phase)
    setattr(target, name, wrapped)
    return True


def _bind(module_name, class_name, methods, *, asynchronous=None):
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, class_name) if class_name else module
        target = getattr(getattr(target, "__ray_metadata__", None), "modified_class", target)
    except Exception as exc:
        _report_install(module_name, class_name, "unavailable", type(exc).__name__)
        return
    for name, phase, scoped in methods:
        try:
            ok = wrap_method(target, name, phase, require_context=scoped, asynchronous=asynchronous)
            _report_install(module_name, name, "installed" if ok else "unavailable")
        except Exception as exc:
            _report_install(module_name, name, "unavailable", type(exc).__name__)


def _report_install(module, method, status, error_type=None):
    key = (module, method, status, error_type)
    state = _state()
    if key not in state._reported:
        state._reported.add(key)
        emit("hook", module=module, method=method, status=status, error_type=error_type)


def install():
    """Retryable across plugin import order; no global 'installed' latch."""
    if not enabled():
        return
    _bind("verl.trainer.ppo.v1.trainer_base", "PPOTrainer", (
        ("_step_once", "trainer.update_cycle", False),
        ("_submit_batch_to_rollout", "trainer.submit", False),
        ("on_sample_begin", "trainer.sample_begin", True),
    ))
    _bind("verl.trainer.ppo.v1.trainer_sync", "PPOTrainerSync", (
        ("on_sample_end", "trainer.sample_end", True),
    ))
    _bind("verl.trainer.ppo.v1.replay_buffer", "ReplayBuffer", (
        ("sample", "trainer.wait_batch", True),
    ))
    _bind("verl.experimental.agent_loop.agent_loop", "AgentLoopWorker", (
        ("_run_agent_loop", "agent.trajectory", False),
        ("_compute_score", "agent.reward", True),
        ("_compute_teacher_logprobs", "agent.teacher", True),
        ("_compute_multi_modal_inputs", "agent.media_inputs", True),
        ("_compute_position_ids", "agent.positions", True),
    ))
    _bind("verl.trainer.ppo.v1.agent_loop_tq", "AgentLoopWorkerTQ", (
        ("_run_agent_loop", "agent.trajectory", False),
        ("_agent_loop_postprocess", "agent.postprocess", True),
    ))
    _bind("verl.workers.rollout.llm_server", "LLMServerClient", (
        ("generate", "agent.generate_request", True),
        ("_acquire_server", "agent.acquire_server", True),
    ))
    # This API is awaited by AgentLoopWorkerTQ. Its decorators may hide the
    # coroutine-function flag; measure through await, not coroutine creation.
    _bind("transfer_queue", None, (
        ("async_kv_batch_put", "agent.queue_put", True),
    ), asynchronous=True)
    _bind("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server", "vLLMOmniHttpServer", (
        ("generate", "server.generate_request", False),
    ))


def summarize(log, step=None):
    """Aggregate call statistics; totals are work-seconds, NOT wall fractions."""
    groups = {}
    for line in log:
        if PREFIX not in line:
            continue
        try:
            item, _ = json.JSONDecoder().raw_decode(line.split(PREFIX, 1)[1])
        except (ValueError, TypeError):
            continue
        if item.get("event") != "span" or (step is not None and item.get("step") != step):
            continue
        key = (item["phase"], item.get("clock", "host_wall"), item.get("status", "ok"))
        group = groups.setdefault(key, [0, 0.0, 0.0])
        value = item["elapsed_s"]
        group[0] += 1
        group[1] += value
        group[2] = max(group[2], value)
    return groups


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize inclusive wall spans; never sum phases as gen time.")
    parser.add_argument("log")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    with open(args.log, encoding="utf-8", errors="replace") as stream:
        groups = summarize(stream, args.step)
    print("phase\tclock\tstatus\tcalls\tmean_s\tmax_s\twork_s (overlapping, not gen wall time)")
    for (phase, clock, status), (count, total, maximum) in sorted(groups.items()):
        print(f"{phase}\t{clock}\t{status}\t{count}\t{total/count:.6f}\t{maximum:.6f}\t{total:.6f}")
    if not groups:
        print("No matching spans. Check hook status and step; engine/RPC spans may have no training step.")
