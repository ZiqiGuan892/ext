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


def _blocks(logits, chunk_size, lengths=None):
    if lengths is not None:
        offset = 0
        for batch, length in enumerate(lengths):
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                index = (batch, slice(start, end))
                yield index, slice(offset + start, offset + end), logits[index]
            offset += length
        return
    # Do not flatten a noncontiguous [B,S,V] tensor into a full vocabulary copy.
    for prefix in product(*(range(size) for size in logits.shape[:-2])):
        for start in range(0, logits.shape[-2], chunk_size):
            index = prefix + (slice(start, start + chunk_size),)
            yield index, index, logits[index]


class _ChunkedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, backend, chunk_size, lengths):
        ctx.backend = backend
        ctx.chunk_size = chunk_size
        ctx.lengths = lengths
        ctx.device_type = logits.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        ctx.save_for_backward(logits, labels)
        output = None
        for _, output_index, block in _blocks(logits, chunk_size, lengths):
            values = backend(block, labels[output_index])
            if output is None:
                # Native torch_npu determines the result dtype; do not force it
                # to either FP32 or the logits dtype.
                output = values.new_empty(labels.shape)
            output[output_index] = values
            del values
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        # The logits gradient is necessarily full sized. Only CE intermediates
        # and its recomputation graph are bounded; model logits are unchanged.
        # This supplies first-order RL gradients, not higher-order derivatives.
        allocate = torch.zeros_like if ctx.lengths is not None else torch.empty_like
        grad_logits = allocate(logits, memory_format=torch.contiguous_format)
        with torch.enable_grad(), torch.autocast(
            device_type=ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled
        ):
            for index, output_index, block in _blocks(logits, ctx.chunk_size, ctx.lengths):
                local_logits = block.detach().requires_grad_(True)
                values = ctx.backend(local_logits, labels[output_index])
                (gradient,) = torch.autograd.grad(values, local_logits, grad_outputs=grad_output[output_index])
                grad_logits[index].copy_(gradient)
                del values, gradient, local_logits
        return grad_logits, None, None, None, None


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
    return _ChunkedLogProbs.apply(logits, labels, backend, chunk_size, None)


def chunked_ragged_logprobs_from_logits(logits, labels, lengths, backend, chunk_size=DEFAULT_CHUNK_SIZE):
    """Select valid rows from dense logits without making a packed [T,V] copy.

    Labels are the engine's already-rolled, packed labels; do not shift again
    at sample boundaries. Only the scalar output is packed. Backward writes
    zero gradients to padding and never assembles a packed vocabulary tensor.
    """
    _validate_chunk_size(chunk_size)
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 3:
        raise ValueError("Huanyu ragged log-probs require floating-point [batch,seq,vocab] logits")
    lengths = tuple(lengths)
    if (not logits.shape[-1] or len(lengths) != logits.shape[0]
            or any(type(length) is not int or not 0 <= length <= logits.shape[1] for length in lengths)):
        raise ValueError("Huanyu ragged log-prob lengths must match the dense batch and sequence bounds")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != sum(lengths):
        raise ValueError("Huanyu ragged log-probs require one packed label per valid token")
    if not labels.numel():
        # A view with no elements preserves native empty-output dtype/autograd.
        return backend(logits[:, :0, :].reshape(0, logits.shape[-1]), labels)
    return _ChunkedLogProbs.apply(logits, labels, backend, chunk_size, lengths)


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
        if logits.ndim == 2 and logits.shape[0] <= chunk_size:
            # The padded-output hook already bounds this call. Native autograd
            # inside its block recomputation avoids nesting two recomputations.
            return original(logits, labels)
        return chunked_logprobs_from_logits(logits, labels, original, chunk_size)

    huanyu_logprobs_from_logits._huanyu_logprobs_chunked = True
    huanyu_logprobs_from_logits._huanyu_logprobs_chunk_size = chunk_size
    setattr(verl_F, target, huanyu_logprobs_from_logits)
    print(f"[huanyu-logprobs] installed pid={os.getpid()} chunk_size={chunk_size} "
          f"target=verl.utils.torch_functional.{target}", flush=True)
    return huanyu_logprobs_from_logits
