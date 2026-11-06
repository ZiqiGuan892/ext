"""Sample Huanyu inference boundaries without synchronizing a device.

All durations are host wall time: asynchronous kernel execution may outlive a
call, while a blocking call may include device/communication waits. These spans
must not be interpreted as NPU kernel timings. Use a device profiler for that.

The first execute_model call and every HUANYU_GEN_TIMING_ENGINE_EVERY-th call
(default 100) are sampled. Nested spans are inclusive and cannot be added to
their parent duration. No model inputs, outputs, tensor values, or text are read.
"""

from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import os
import sys


_ACTIVE = ContextVar("huanyu_gen_timing_engine", default=None)
_NOTICES = set()
_MARKER = "_huanyu_gen_timing_engine"
_ITERATION_ATTR = "_huanyu_gen_timing_iteration"
_LAST_ATTR = "_huanyu_gen_timing_last_sample"


def _module():
    # Ray serializes nested method wrappers by value. Resolve process-local
    # ContextVars through a module, never capture the ContextVar in a closure.
    return sys.modules[__name__]


def _core():
    return importlib.import_module(__package__ + ".gen_timing")


def _notice(event, **fields):
    key = (event, tuple(sorted((name, str(value)) for name, value in fields.items())))
    if key not in _module()._NOTICES:
        _module()._NOTICES.add(key)
        _core().emit(event, **fields)


def _every():
    value = os.environ.get("HUANYU_GEN_TIMING_ENGINE_EVERY", "100")
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    _notice("engine_config_invalid", setting="HUANYU_GEN_TIMING_ENGINE_EVERY", fallback=100)
    return 100


def _identity(runner):
    fields = {"runner_class": type(runner).__name__}
    for name in ("rank", "local_rank", "global_rank"):
        value = getattr(runner, name, None)
        if type(value) is int:
            fields[name] = value
    config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    if config is None:
        config = getattr(runner, "model_config", None)
    model_path = getattr(config, "model", None)
    if isinstance(model_path, str):
        fields["model_path"] = model_path
    return fields


def _scheduler_fields(scheduler):
    """Read scheduler's Python metadata only; never coerce a tensor to int."""
    fields = {}
    scheduled = getattr(scheduler, "num_scheduled_tokens", None)
    if isinstance(scheduled, dict):
        fields["scheduled_requests"] = len(scheduled)
        if all(type(value) is int for value in scheduled.values()):
            fields["scheduled_tokens"] = sum(scheduled.values())
    total = getattr(scheduler, "total_num_scheduled_tokens", None)
    if type(total) is int:
        fields["scheduled_tokens"] = total
    encoder = getattr(scheduler, "scheduled_encoder_inputs", None)
    if isinstance(encoder, dict):
        fields["scheduled_encoder_requests"] = len(encoder)
    # Token counts alone do not prove prefill vs decode (speculative decoding
    # and mixed batches exist), so do not assign an inferred phase label.
    return fields


def _bind_nested(cls, name, phase, *, outside_execute=False):
    original = getattr(cls, name, None)
    target = f"{cls.__module__}.{cls.__qualname__}.{name}"
    if (not callable(original) or inspect.iscoroutinefunction(original)
            or inspect.isgeneratorfunction(original) or inspect.isasyncgenfunction(original)):
        _notice("engine_hook_unavailable", target=target, phase=phase)
        return False
    if getattr(original, _MARKER, False):
        return True

    @wraps(original)
    def measured(self, *args, **kwargs):
        if not _core().enabled():
            return original(self, *args, **kwargs)
        context = _module()._ACTIVE.get()
        if context is None and outside_execute:
            # Newer runners may execute sample_tokens in a separate RPC after
            # execute_model has returned. Keep that as an independent span.
            context = getattr(self, _LAST_ATTR, None)
        if context is None:
            return original(self, *args, **kwargs)
        _notice("engine_hook_executed", target=target, phase=phase, clock="host_wall")
        with _core().span(phase, **context, measurement="host_call"):
            return original(self, *args, **kwargs)

    setattr(measured, _MARKER, True)
    setattr(cls, name, measured)
    _notice("engine_hook_installed", target=target, phase=phase, clock="host_wall")
    return True


def _bind_model(runner):
    model = getattr(runner, "model", None)
    if model is None:
        _notice("engine_hook_unavailable", target=type(runner).__name__ + ".model", phase="engine.forward")
        return
    # The actual top-level callable is measured, rather than every transformer
    # layer. Sampling remains controlled by execute_model's context.
    _bind_nested(type(model), "forward", "engine.forward")
    _bind_nested(type(model), "compute_logits", "engine.compute_logits")


def bind_runner_class(cls):
    """Install once on the concrete Huanyu runner; useful with small CPU fakes."""
    original = getattr(cls, "execute_model", None)
    target = f"{cls.__module__}.{cls.__qualname__}.execute_model"
    if (not callable(original) or inspect.iscoroutinefunction(original)
            or inspect.isgeneratorfunction(original) or inspect.isasyncgenfunction(original)):
        _notice("engine_hook_unavailable", target=target, phase="engine.execute_model")
        return False
    if getattr(original, _MARKER, False):
        return True
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        _notice("engine_hook_unavailable", target=target, reason="unknown_signature")
        return False
    if "scheduler_output" not in signature.parameters:
        _notice("engine_hook_unavailable", target=target, reason="missing_scheduler_output")
        return False

    for method, phase in (
        ("_preprocess", "engine.preprocess"),
        ("_execute_mm_encoder", "engine.mm_encoder"),
    ):
        _bind_nested(cls, method, phase)
    _bind_nested(cls, "sample_tokens", "engine.sample_tokens", outside_execute=True)

    @wraps(original)
    def measured(self, *args, **kwargs):
        if not _core().enabled() or _module()._ACTIVE.get() is not None:
            return original(self, *args, **kwargs)
        iteration = getattr(self, _ITERATION_ATTR, 0) + 1
        setattr(self, _ITERATION_ATTR, iteration)
        every = _every()
        if iteration != 1 and iteration % every:
            setattr(self, _LAST_ATTR, None)
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        context = {
            **_identity(self),
            **_scheduler_fields(bound.arguments.get("scheduler_output")),
            "engine_iteration": iteration,
            "sample_every": every,
        }
        _bind_model(self)
        setattr(self, _LAST_ATTR, context)
        token = _module()._ACTIVE.set(context)
        _notice("engine_hook_executed", target=target, phase="engine.execute_model", clock="host_wall")
        try:
            with _core().span("engine.execute_model", **context, measurement="host_call"):
                return original(self, *args, **kwargs)
        except BaseException:
            # A failed execute must not label a later sampler as its success.
            setattr(self, _LAST_ATTR, None)
            raise
        finally:
            _module()._ACTIVE.reset(token)

    setattr(measured, _MARKER, True)
    cls.execute_model = measured
    _notice("engine_hook_installed", target=target, phase="engine.execute_model", clock="host_wall")
    return True


def install_engine():
    """Best-effort and repeatable during external-module and worker startup."""
    if not _core().enabled():
        return False
    module_name = "vllm_omni.platforms.npu.worker.npu_ar_model_runner"
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, "NPUARModelRunner")
    except Exception as error:
        _notice("engine_hook_unavailable", target=module_name + ".NPUARModelRunner",
                error_type=type(error).__name__)
        return False
    return bind_runner_class(cls)
