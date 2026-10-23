"""Opt-in, fixed-reduction BF16 projections for Ascend consistency diagnostics.

Only the forward reduction changes. Autograd retains the ordinary linear
derivatives; FP32 tensors, CPU execution and quantized layers are untouched.
"""
import os
from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F

_actor_depth = ContextVar("huanyu_stable_linear_depth", default=0)


_EXECUTED = set()


def _status(component, phase, **details):
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[huanyu-precision] component={component} phase={phase} {fields}".rstrip(), flush=True)


def _executed(component, x, output):
    if component not in _EXECUTED:
        _EXECUTED.add(component)
        _status(component, "executed", input_dtype=x.dtype, output_dtype=output.dtype,
                shape=tuple(x.shape), grad_enabled=torch.is_grad_enabled())


def enabled():
    value = os.environ.get("MINICPMO_STABLE_LINEAR", "0")
    if value not in ("0", "1"):
        raise ValueError(f"MINICPMO_STABLE_LINEAR must be 0 or 1, got {value!r}")
    return value == "1"


def _eligible(x, weight):
    return x.device.type == "npu" and x.dtype == weight.dtype == torch.bfloat16


def _require_backend():
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        linear_persistent_kernel, matmul_persistent,
    )
    import torch_npu
    if not callable(matmul_persistent) or not hasattr(linear_persistent_kernel, "__getitem__"):
        raise RuntimeError("Huanyu stable linear requires the Ascend persistent matmul kernels")
    for name in ("get_npu_format", "npu_format_cast"):
        if not callable(getattr(torch_npu, name, None)):
            raise RuntimeError(f"Huanyu stable linear requires torch_npu.{name}")


def persistent_linear(x, weight, bias=None):
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        linear_persistent_kernel, matmul_persistent,
    )
    import torch_npu

    # A logical contiguous view can still use Ascend's tiled physical storage.
    if torch_npu.get_npu_format(x) != 2:
        x = torch_npu.npu_format_cast(x, 2)
    if torch_npu.get_npu_format(weight) != 2:
        weight = torch_npu.npu_format_cast(weight, 2)

    flat = x.reshape(-1, x.shape[-1])
    m, k = flat.shape
    n = weight.shape[0]
    if m == 0:
        return x.new_empty((*x.shape[:-1], n))
    if bias is not None:
        result = matmul_persistent(flat, weight.T, bias)
    else:
        # Keep the K reduction and dot tiles identical for prefill, decode and
        # different actor microbatches. Read [N,K] weights without copying them.
        result = x.new_empty((m, n))
        blocks_m, blocks_n = (m + 127) // 128, (n + 127) // 128
        grid = blocks_m * blocks_n
        linear_persistent_kernel[(grid,)](
            flat, weight, result, m, n, k,
            *flat.stride(), *weight.stride(), *result.stride(),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=256,
            NUM_BLOCKS_M=blocks_m, NUM_BLOCKS_N=blocks_n, GRID_SIZE=grid,
        )
    return result.reshape(*x.shape[:-1], n)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return persistent_linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, grad.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        grad_x = (flat_grad @ weight).reshape_as(x) if ctx.needs_input_grad[0] else None
        grad_w = flat_grad.T @ flat_x if ctx.needs_input_grad[1] else None
        grad_b = flat_grad.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_x, grad_w, grad_b


def install_actor(module=None):
    if not enabled():
        _status("stable_linear_actor", "disabled")
        return False
    lm = getattr(getattr(module, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers") or not hasattr(module, "lm_head"):
        raise RuntimeError("Huanyu stable linear requires an explicit actor language model and lm_head")
    # Validate the real NPU backend before announcing installation. CPU modules
    # remain supported for wiring tests and do not execute the Ascend kernel.
    if any(parameter.device.type == "npu" for parameter in lm.parameters()):
        _require_backend()
    targets = (lm, *lm.layers, module.lm_head)
    new_scopes = sum(_scope_forward(target) for target in targets)
    if getattr(F.linear, "_huanyu_stable_linear", False) is True:
        _status("stable_linear_actor", "already_installed", new_scopes=new_scopes)
        return False
    original = F.linear

    @wraps(original)
    def linear(x, weight, bias=None):
        if _actor_depth.get() > 0 and _eligible(x, weight):
            result = _Linear.apply(x, weight, bias)
            _executed("stable_linear_actor", x, result)
            return result
        return original(x, weight, bias)

    linear._huanyu_stable_linear = True
    F.linear = linear
    _status("stable_linear_actor", "installed", new_scopes=new_scopes)
    return True


def _scope_forward(module):
    original = module.forward
    if getattr(original, "_huanyu_linear_scope", False) is True:
        return False

    @wraps(getattr(original, "__func__", original))
    def forward(self, *args, **kwargs):
        token = _actor_depth.set(_actor_depth.get() + 1)
        try:
            return original(*args, **kwargs)
        finally:
            _actor_depth.reset(token)

    forward._huanyu_linear_scope = True
    module.forward = MethodType(forward, module)
    return True


def install_rollout(native):
    if not enabled():
        _status("stable_linear_rollout", "disabled")
        return False
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    language = getattr(native, "llm", None)
    if language is None:
        language = getattr(getattr(native, "thinker", None), "llm", None)
    if language is None or not callable(getattr(language, "modules", None)):
        raise RuntimeError("Huanyu stable linear requires an explicit native language model")
    layers = [layer for layer in language.modules()
              if isinstance(getattr(layer, "quant_method", None), AscendUnquantizedLinearMethod)]
    if not layers:
        raise RuntimeError("Huanyu stable linear found no Ascend unquantized language projections")
    if any(getattr(getattr(layer, "weight", None), "device", None).type == "npu" for layer in layers):
        _require_backend()
    new_scopes = 0
    for layer in layers:
        if getattr(layer, "_huanyu_stable_linear_layer", False) is not True:
            layer._huanyu_stable_linear_layer = True
            new_scopes += 1
    if getattr(AscendUnquantizedLinearMethod.apply, "_huanyu_stable_linear", False) is True:
        _status("stable_linear_rollout", "already_installed", new_scopes=new_scopes, total=len(layers))
        return False
    original = AscendUnquantizedLinearMethod.apply

    @wraps(original)
    def apply(self, layer, x, bias=None):
        if getattr(layer, "_huanyu_stable_linear_layer", False) is True and _eligible(x, layer.weight):
            result = persistent_linear(x, layer.weight, bias)
            _executed("stable_linear_rollout", x, result)
            return result
        return original(self, layer, x, bias)

    apply._huanyu_stable_linear = True
    AscendUnquantizedLinearMethod.apply = apply
    _status("stable_linear_rollout", "installed", new_scopes=new_scopes, total=len(layers))
    return True
