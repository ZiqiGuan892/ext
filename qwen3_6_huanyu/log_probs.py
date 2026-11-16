"""Bound native NPU cross-entropy intermediates without changing the RL batch.

The padded FSDP engine calls logprobs_from_logits after the model returns. Its
native NPU helper otherwise submits all valid tokens to cross entropy at once.
Keep the original backend, labels, vocabulary and autocast policy, but submit
bounded token blocks. Recompute those blocks in backward instead of retaining
every block's native cross-entropy intermediates until the policy update.
"""

from functools import wraps
import inspect
from itertools import product
import os

import torch
from torch.autograd.function import once_differentiable


DEFAULT_CHUNK_SIZE = 512


def _validate_chunk_size(chunk_size):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Huanyu log-prob chunk_size must be a positive integer")


def _blocks(logits, chunk_size):
    # Do not flatten a noncontiguous [B,S,V] tensor into a full vocabulary copy.
    for prefix in product(*(range(size) for size in logits.shape[:-2])):
        for start in range(0, logits.shape[-2], chunk_size):
            index = prefix + (slice(start, start + chunk_size),)
            yield index, logits[index]


class _ChunkedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, backend, chunk_size):
        ctx.backend = backend
        ctx.chunk_size = chunk_size
        ctx.device_type = logits.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        ctx.save_for_backward(logits, labels)
        output = None
        for index, block in _blocks(logits, chunk_size):
            values = backend(block, labels[index])
            if output is None:
                # Native torch_npu determines the result dtype; do not force it
                # to either FP32 or the logits dtype.
                output = values.new_empty(logits.shape[:-1])
            output[index] = values
            del values
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        # The logits gradient is necessarily full sized. Only CE intermediates
        # and its recomputation graph are bounded; model logits are unchanged.
        # This supplies first-order RL gradients, not higher-order derivatives.
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        with torch.enable_grad(), torch.autocast(
            device_type=ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled
        ):
            for index, block in _blocks(logits, ctx.chunk_size):
                local_logits = block.detach().requires_grad_(True)
                values = ctx.backend(local_logits, labels[index])
                (gradient,) = torch.autograd.grad(values, local_logits, grad_outputs=grad_output[index])
                grad_logits[index].copy_(gradient)
                del values, gradient, local_logits
        return grad_logits, None, None, None


def chunked_logprobs_from_logits(logits, labels, backend, chunk_size=DEFAULT_CHUNK_SIZE):
    _validate_chunk_size(chunk_size)
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("Huanyu log-probs require floating-point logits")
    if logits.ndim < 2 or not logits.shape[-1]:
        raise ValueError("Huanyu log-probs expect [..., tokens, vocab] with non-empty vocab")
    if not isinstance(labels, torch.Tensor) or labels.numel() != logits.numel() // logits.shape[-1]:
        raise ValueError("Huanyu log-probs require one label per token")
    # Labels are small; preserve the original helper's acceptance of flat labels.
    labels = labels.reshape(logits.shape[:-1])
    if not labels.numel():
        return backend(logits, labels)
    return _ChunkedLogProbs.apply(logits, labels, backend, chunk_size)


def install_huanyu_logprob_chunking(chunk_size=DEFAULT_CHUNK_SIZE):
    """Patch the NPU helper reached by the engine's already-imported dispatcher.

Installed only when configuring a Huanyu training model, like the entropy
guard. CPU/Flash-Attention dispatcher branches remain unchanged. Does not edit
the installed verl or torch_npu sources and does not change model precision.
"""
    _validate_chunk_size(chunk_size)
    from verl.utils import torch_functional as verl_F

    target = "logprobs_from_logits_torch_npu"
    original = getattr(verl_F, target, None)
    if not callable(original):
        raise RuntimeError(f"Cannot install Huanyu log-prob chunking: {target} is not callable")
    if getattr(original, "_huanyu_logprobs_chunked", False):
        if original._huanyu_logprobs_chunk_size != chunk_size:
            raise RuntimeError("Huanyu log-prob chunk size is already fixed for this worker")
        return original
    if tuple(inspect.signature(original).parameters) != ("logits", "labels"):
        raise RuntimeError(f"Unsupported Huanyu NPU log-prob helper signature: {inspect.signature(original)}")
    dispatcher = getattr(verl_F, "logprobs_from_logits", None)
    if getattr(dispatcher, "__globals__", {}).get(target) is not original:
        raise RuntimeError("Cannot verify that verl log-prob dispatcher resolves the NPU helper dynamically")
    reported = set()

    @wraps(original)
    def huanyu_logprobs_from_logits(logits, labels):
        mode = "grad" if torch.is_grad_enabled() and logits.requires_grad else "no_grad"
        if mode not in reported:
            print(f"[huanyu-logprobs] active pid={os.getpid()} shape={tuple(logits.shape)} "
                  f"dtype={logits.dtype} device={logits.device} chunk_size={chunk_size} "
                  f"mode={mode} backend=native_npu", flush=True)
            reported.add(mode)
        return chunked_logprobs_from_logits(logits, labels, original, chunk_size)

    huanyu_logprobs_from_logits._huanyu_logprobs_chunked = True
    huanyu_logprobs_from_logits._huanyu_logprobs_chunk_size = chunk_size
    setattr(verl_F, target, huanyu_logprobs_from_logits)
    print(f"[huanyu-logprobs] installed pid={os.getpid()} chunk_size={chunk_size} "
          f"target=verl.utils.torch_functional.{target}", flush=True)
    return huanyu_logprobs_from_logits
