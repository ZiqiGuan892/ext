"""Bound the score matrix of frozen HF vision attention without dropping tokens.

Only query rows are tiled. Every row still attends to the full K/V sequence
supplied by HF, retaining its per-image/frame boundaries and eager arithmetic.
This context is for the single-threaded offline encoder, never training/rollout.
"""

from contextlib import contextmanager

import torch


QUERY_CHUNK_SIZE = 128


@contextmanager
def chunked_vision_attention(hf_module, vision_model, query_chunk_size=QUERY_CHUNK_SIZE):
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("Vision query chunk size must be a positive integer")
    original = getattr(hf_module, "eager_attention_forward", None)
    if not callable(original):
        raise RuntimeError("Installed HF vision implementation has no eager_attention_forward")
    selected = {id(module) for module in vision_model.modules()}
    stats = {"calls": 0, "chunks": 0, "max_query_tokens": 0,
             "max_key_tokens": 0, "max_chunk_tokens": 0}

    def bounded(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if id(module) not in selected:
            return original(module, query, key, value, attention_mask,
                            scaling=scaling, dropout=dropout, **kwargs)
        if module.training or torch.is_grad_enabled():
            raise RuntimeError("Chunked vision eager requires eval mode and no_grad")
        if dropout != 0.0 or kwargs.get("is_causal", False) or getattr(module, "is_causal", False):
            raise ValueError("Chunked vision eager requires noncausal attention and zero dropout")
        if any(tensor.ndim != 4 for tensor in (query, key, value)) or query.shape[2] <= 0:
            raise ValueError("Vision attention requires nonempty [batch,heads,tokens,dim] tensors")
        query_tokens, key_tokens = query.shape[2], key.shape[2]
        if (key_tokens <= 0 or key.shape[:3] != value.shape[:3]
                or key.shape[0] != query.shape[0] or key.shape[-1] != query.shape[-1]):
            raise ValueError("Vision attention Q/K/V batch, token or feature dimensions do not match")
        if attention_mask is not None and (
            attention_mask.ndim < 2 or attention_mask.shape[-2] not in (1, query_tokens)
            or attention_mask.shape[-1] not in (1, key_tokens)
        ):
            raise ValueError("Vision attention mask must broadcast over query/key token axes")
        stats["calls"] += 1
        stats["max_query_tokens"] = max(stats["max_query_tokens"], query_tokens)
        stats["max_key_tokens"] = max(stats["max_key_tokens"], key_tokens)
        outputs = []
        for start in range(0, query_tokens, query_chunk_size):
            stop = min(start + query_chunk_size, query_tokens)
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            output, weights = original(
                module, query[:, :, start:stop, :], key, value, mask,
                scaling=scaling, dropout=dropout, **kwargs,
            )
            # Do not keep quadratic attention weights across chunks or layers.
            del weights
            expected = (query.shape[0], stop - start, query.shape[1], value.shape[-1])
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
                raise RuntimeError("HF eager output must have [batch,query_tokens,heads,value_dim] layout")
            if output.dtype != query.dtype or output.device != query.device:
                raise RuntimeError("HF eager output must preserve query dtype and device")
            outputs.append(output)
            stats["chunks"] += 1
            stats["max_chunk_tokens"] = max(stats["max_chunk_tokens"], stop - start)
        return (outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)), None

    # Keep HF's own forward, RoPE, cu_seqlens split, scaling and softmax code.
    # The CLI runs one encoder call at a time and restores this binding on error.
    hf_module.eager_attention_forward = bounded
    try:
        yield stats
    finally:
        hf_module.eager_attention_forward = original
