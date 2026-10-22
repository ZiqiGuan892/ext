"""Opt-in Huanyu FSDP synchronization without changing installed verl sources.

Small micro-batches must not retain the full MoE gradients until the final
backward. The HF config flag is intentionally separate from verl's versioned
engine dataclass: old engines may not accept its newer no-sync configuration.
"""

import ast
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
import importlib
import inspect
import os
import textwrap


_MARKER = "_huanyu_sync_each_microbatch"
_PATCH_MARKER = "_huanyu_microbatch_sync_patch"
_CONFIG_FLAG = "huanyu_sync_each_microbatch"
_REPORTED = set()


def _value(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _model_chain(module):
    """FSDP2 keeps the original object; FSDP1/compile may wrap that object."""
    seen = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        yield module
        wrapped = getattr(module, "_fsdp_wrapped_module", None)
        module = wrapped if wrapped is not None else getattr(module, "_orig_mod", None)


def _enabled_for_engine(engine):
    modules = tuple(_model_chain(getattr(engine, "module", None)))
    # An explicit disabled marker takes precedence over a copied HF config on
    # an outer FSDP/compile wrapper (including a config saved by an earlier run).
    for module in modules:
        marker = getattr(module, _MARKER, None)
        if marker is not None:
            return marker is True
    for module in modules:
        config = getattr(module, "config", None)
        audio = _value(config, "audio_config")
        if (
            _value(config, _CONFIG_FLAG) is True
            and _value(config, "model_type") == "qwen3_5_moe"
            and _value(audio, "model_type") == "huanyu_conformer_0_6b"
        ):
            return True
    return False


def _enable_sync(module):
    for candidate in _model_chain(module):
        setter = getattr(candidate, "set_requires_gradient_sync", None)
        if callable(setter):
            setter(True)
            return
        # FSDP1 synchronizes by default when its no_sync context is not entered.
        if callable(getattr(candidate, "no_sync", None)):
            return
    raise RuntimeError(
        "Huanyu microbatch synchronization requires a wrapped FSDP model; "
        "cannot verify gradient synchronization on this engine module"
    )


def _report_once(mode):
    if mode not in _REPORTED:
        _REPORTED.add(mode)
        print(
            f"[huanyu-memory] pid={os.getpid()} sync_each_microbatch=enabled "
            f"mode={mode}; gradients synchronize after every backward",
            flush=True,
        )


def _verify_legacy_loop(method):
    """Fail closed when an older engine's accumulation behavior is unknown."""
    try:
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError(
            "Cannot inspect this legacy FSDP forward_backward_batch; "
            "Huanyu cannot safely enable multiple micro-batches"
        ) from exc

    delayed_sync_names = {
        "no_sync", "_gradient_sync_context", "set_requires_gradient_sync",
        "set_requires_all_reduce", "require_backward_grad_sync",
        "requires_gradient_sync", "sync_gradients", "gradient_sync",
    }
    def safe_context_value(expr):
        if isinstance(expr, ast.IfExp):
            return safe_context_value(expr.body) and safe_context_value(expr.orelse)
        if not isinstance(expr, ast.Call):
            return False
        target = expr.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
        return name in {"nullcontext", "no_grad"}

    context_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ctx" for t in node.targets):
            context_assignments.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "ctx":
            context_assignments.append(node.value)
    verified_ctx = bool(context_assignments) and all(safe_context_value(value) for value in context_assignments)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        if any(name in delayed_sync_names for name in names):
            raise RuntimeError(
                "Legacy FSDP forward_backward_batch contains gradient-sync control "
                "but exposes no _gradient_sync_context hook; refusing an unsafe Huanyu microbatch policy"
            )
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Name) and expr.id == "ctx" and verified_ctx:
                    continue
                target = expr.func if isinstance(expr, ast.Call) else expr
                name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
                if name not in {"nullcontext", "no_grad", "record_function"}:
                    raise RuntimeError(
                        "Legacy FSDP forward_backward_batch uses an unverified context manager; "
                        "cannot establish per-microbatch gradient synchronization"
                    )
    has_backward_loop = any(
        isinstance(loop, (ast.For, ast.While))
        and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
            for node in ast.walk(loop)
        )
        for loop in ast.walk(tree)
    )
    if not has_backward_loop:
        raise RuntimeError(
            "Legacy FSDP forward_backward_batch has no inspectable backward loop; "
            "cannot establish per-microbatch gradient synchronization"
        )


def install_huanyu_microbatch_sync(module, enabled=True):
    """Mark one Huanyu model and install a gated, version-compatible engine hook.

    Call during Huanyu configure_model, before FSDP wrapping. Disabling the
    policy does not import the engine or change other model instances. The
    wrapper never changes optimizer stepping, loss scaling or tensor dtypes.
    """
    if type(enabled) is not bool:
        raise ValueError("huanyu_sync_each_microbatch must be a boolean")
    if not enabled:
        setattr(module, _MARKER, False)
        return module

    impl = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    engine_cls = getattr(impl, "FSDPEngine", None)
    if engine_cls is None:
        raise RuntimeError("Cannot install Huanyu microbatch synchronization: FSDPEngine is unavailable")
    original = getattr(engine_cls, "_gradient_sync_context", None)
    if callable(original):
        if not getattr(original, _PATCH_MARKER, False):
            @wraps(original)
            @contextmanager
            def huanyu_sync_context(engine, *args, **kwargs):
                if not _enabled_for_engine(engine):
                    with original(engine, *args, **kwargs):
                        yield
                    return
                _enable_sync(engine.module)
                _report_once("fsdp_context")
                try:
                    yield
                finally:
                    # Also reset after an exception or a nested operation that
                    # changed the flag; no subsequent backward may inherit False.
                    _enable_sync(engine.module)

            setattr(huanyu_sync_context, _PATCH_MARKER, True)
            engine_cls._gradient_sync_context = huanyu_sync_context
    else:
        original = getattr(engine_cls, "forward_backward_batch", None)
        if not callable(original):
            raise RuntimeError("Cannot find FSDP forward_backward_batch for Huanyu microbatch synchronization")
        if not getattr(original, _PATCH_MARKER, False):
            _verify_legacy_loop(original)

            @wraps(original)
            def huanyu_legacy_forward_backward(engine, *args, **kwargs):
                if not _enabled_for_engine(engine):
                    return original(engine, *args, **kwargs)
                _enable_sync(engine.module)
                _report_once("verified_legacy_loop")
                try:
                    return original(engine, *args, **kwargs)
                finally:
                    _enable_sync(engine.module)

            setattr(huanyu_legacy_forward_backward, _PATCH_MARKER, True)
            engine_cls.forward_backward_batch = huanyu_legacy_forward_backward
    setattr(module, _MARKER, True)
    return module
