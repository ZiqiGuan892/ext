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

_actor_depth = ContextVar("q36_stable_linear_depth", default=0)


def enabled():
    return os.environ.get("MINICPMO_STABLE_LINEAR", "0") == "1"


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
    if not enabled() or getattr(F.linear, "_q36_stable_linear", False):
        return False
    original = F.linear
    scoped = module is not None
    if scoped:
        lm = module.model.language_model
        # Wrap forward itself: checkpoint wrappers can call forward directly,
        # bypassing Module.__call__ and its forward hooks during recomputation.
        for target in (lm, *lm.layers, module.lm_head):
            _scope_forward(target)

    def linear(x, weight, bias=None):
        if ((not scoped or _actor_depth.get() > 0) and x.device.type == "npu"
                and x.dtype == weight.dtype == torch.bfloat16):
            return _Linear.apply(x, weight, bias)
        return original(x, weight, bias)

    linear._q36_stable_linear = True
    F.linear = linear
    print("[q36-precision] actor fixed-tile BF16 language projections enabled", flush=True)
    return True


def _enter_actor():
    _actor_depth.set(_actor_depth.get() + 1)


def _leave_actor():
    _actor_depth.set(max(0, _actor_depth.get() - 1))


def _scope_forward(module):
    original = module.forward

    @wraps(getattr(original, "__func__", original))
    def forward(self, *args, **kwargs):
        token = _actor_depth.set(_actor_depth.get() + 1)
        try:
            return original(*args, **kwargs)
        finally:
            _actor_depth.reset(token)

    module.forward = MethodType(forward, module)


def install_rollout():
    if not enabled():
        return False
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    if getattr(AscendUnquantizedLinearMethod.apply, "_q36_stable_linear", False):
        return False
    original = AscendUnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        if ("thinker.llm" in getattr(layer, "prefix", "")
                and x.dtype == layer.weight.dtype == torch.bfloat16):
            return persistent_linear(x, layer.weight, bias)
        return original(self, layer, x, bias)

    apply._q36_stable_linear = True
    AscendUnquantizedLinearMethod.apply = apply
    print("[q36-precision] rollout fixed-tile BF16 language projections enabled", flush=True)
    return True
