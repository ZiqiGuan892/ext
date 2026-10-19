"""Bound entropy intermediates in Huanyu training workers, including padded logits.

The server's old-log-prob path calls verl_F.entropy_from_logits directly even
with entropy_coeff=0. Install this replacement only when configuring a Huanyu
training model; do not import the Whisper forward/cache builder.
"""

import os

import torch
from torch.autograd.function import once_differentiable


DEFAULT_CHUNK_SIZE = 512


def _validate_chunk_size(chunk_size):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Huanyu entropy chunk_size must be a positive integer")


def _validate_logits(logits):
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("Huanyu entropy requires floating-point logits")
    if logits.ndim not in (2, 3) or logits.shape[-1] == 0:
        raise ValueError("Huanyu entropy expects [tokens,vocab] or [batch,seq,vocab] with non-empty vocab")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"Unsupported Huanyu entropy dtype: {logits.dtype}")


def _compute_dtype(logits):
    return torch.float64 if logits.dtype == torch.float64 else torch.float32


def _token_blocks(logits, chunk_size):
    # A response slice of [B,S,V] often is not contiguous. Flattening it can
    # allocate a second complete logits tensor, defeating the memory bound.
    if logits.ndim == 2:
        for start in range(0, logits.shape[0], chunk_size):
            index = slice(start, start + chunk_size)
            yield index, logits[index]
    else:
        for batch in range(logits.shape[0]):
            for start in range(0, logits.shape[1], chunk_size):
                index = (batch, slice(start, start + chunk_size))
                yield index, logits[index]


def _probability_terms(logits_chunk):
    work = logits_chunk.to(dtype=_compute_dtype(logits_chunk))
    # Translation invariance avoids cancellation for a large common offset.
    # Never mutate work in place: it may alias the original FP32/FP64 logits.
    shifted = work - work.amax(dim=-1, keepdim=True)
    probabilities = torch.softmax(shifted, dim=-1)
    expectation = (probabilities * shifted).sum(dim=-1, keepdim=True)
    return shifted, probabilities, expectation


def _entropy_block(logits_chunk):
    shifted, probabilities, expectation = _probability_terms(logits_chunk)
    return torch.logsumexp(shifted, dim=-1) - expectation.squeeze(-1)


class _ChunkedEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, chunk_size):
        ctx.chunk_size = chunk_size
        # Save only the existing logits storage, not a full softmax or a list
        # of chunk graphs. In no-grad inference there is no retained graph.
        ctx.save_for_backward(logits)
        entropy = torch.empty(logits.shape[:-1], device=logits.device, dtype=_compute_dtype(logits))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            for index, chunk in _token_blocks(logits, chunk_size):
                entropy[index] = _entropy_block(chunk)
        return entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        # The required output gradient is full size; only probability and
        # arithmetic intermediates are bounded. Supports ordinary RL backward,
        # not second-order derivatives.
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            for index, chunk in _token_blocks(logits, ctx.chunk_size):
                shifted, probabilities, expectation = _probability_terms(chunk)
                gradient = probabilities * (expectation - shifted)
                gradient.mul_(grad_output[index].to(gradient.dtype).unsqueeze(-1))
                grad_logits[index].copy_(gradient)
                # Release before allocating the next block (RHS evaluation
                # otherwise overlaps with the previous block's live tensors).
                del shifted, probabilities, expectation, gradient
        return grad_logits, None


def chunked_entropy_from_logits(logits, chunk_size=DEFAULT_CHUNK_SIZE):
    """Full-vocabulary entropy, chunked only over tokens; retains first gradients.

    FP16/BF16/FP32 inputs compute and return FP32; FP64 is preserved. Inputs are
    neither flattened into a copy nor cast in full. Return shape is logits.shape
    without its vocabulary axis, including empty batch/sequence dimensions.
    """
    _validate_chunk_size(chunk_size)
    _validate_logits(logits)
    return _ChunkedEntropy.apply(logits, chunk_size)


def install_huanyu_entropy_chunking(chunk_size=DEFAULT_CHUNK_SIZE):
    """Install in this Huanyu model worker; do not edit installed verl sources."""
    _validate_chunk_size(chunk_size)
    from verl.utils import torch_functional as verl_F

    original = getattr(verl_F, "entropy_from_logits", None)
    if not callable(original):
        raise RuntimeError("Cannot install Huanyu entropy chunking: verl_F.entropy_from_logits is not callable")
    if getattr(original, "_huanyu_entropy_chunked", False):
        if getattr(original, "_huanyu_entropy_chunk_size", None) != chunk_size:
            raise RuntimeError("Huanyu entropy chunk size is already fixed for this worker")
        return original

    first_call = True

    def huanyu_entropy_from_logits(logits):
        nonlocal first_call
        _validate_logits(logits)
        if first_call:
            print(f"[huanyu-entropy] active pid={os.getpid()} shape={tuple(logits.shape)} "
                  f"dtype={logits.dtype} device={logits.device} chunk_size={chunk_size} "
                  f"grad_enabled={torch.is_grad_enabled()} requires_grad={logits.requires_grad}", flush=True)
            first_call = False
        return chunked_entropy_from_logits(logits, chunk_size)

    huanyu_entropy_from_logits._huanyu_entropy_chunked = True
    huanyu_entropy_from_logits._huanyu_entropy_chunk_size = chunk_size
    verl_F.entropy_from_logits = huanyu_entropy_from_logits
    print(f"[huanyu-entropy] installed pid={os.getpid()} chunk_size={chunk_size} "
          "target=verl.utils.torch_functional.entropy_from_logits", flush=True)
    return huanyu_entropy_from_logits
