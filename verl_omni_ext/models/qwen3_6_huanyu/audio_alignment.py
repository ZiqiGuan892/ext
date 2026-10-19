"""Strict, single-sample audio/token alignment helpers.

These helpers do not tokenize, encode, pad, truncate, or change attention masks.
They are not yet wired into the training adapter. The caller must supply the
actual Huanyu encoder/projector output lengths for each chunk, in prompt order.
"""

from numbers import Integral
from typing import Sequence

import torch


def _integers(values, *, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
            raise TypeError(f"{name} must be a one-dimensional integer sequence")
        values = values.tolist()
    result = list(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise TypeError(f"{name} must contain integers, not booleans or floats")
    return [int(value) for value in result]


def find_audio_bounds(
    prompt_ids: Sequence[int] | torch.Tensor,
    expected_chunk_lengths: Sequence[int] | torch.Tensor,
    *,
    start_token_id: int,
    end_token_id: int,
    placeholder_token_id: int,
) -> list[tuple[int, int]]:
    """Validate audio spans and return half-open interior bounds ``[start, end)``.

    ``prompt_ids`` is one unpadded or already correctly masked prompt ``[S]``.
    Each span must consist of a start token, exactly ``T > 0`` placeholder
    tokens, and an end token. ``expected_chunk_lengths`` contains one projected
    chunk length per span, in the same order. Boundary tokens are excluded from
    returned bounds. Nested, unclosed, empty, extra, or missing spans fail.

    Zero-token audio chunks are deliberately rejected: the caller must handle
    unsupported short audio/chunks before building prompt and feature lists.
    Placeholder tokens outside audio spans are left alone (they may be text).
    """
    ids = _integers(prompt_ids, name="prompt_ids")
    lengths = _integers(expected_chunk_lengths, name="expected_chunk_lengths")
    markers = _integers(
        [start_token_id, end_token_id, placeholder_token_id], name="audio token ids"
    )
    if min(markers) < 0 or len(set(markers)) != 3:
        raise ValueError("audio token ids must be distinct non-negative integers")
    if any(length <= 0 for length in lengths):
        raise ValueError("each expected audio chunk length must be positive")

    bounds = []
    interior_start = None
    for position, token in enumerate(ids):
        if token == start_token_id:
            if interior_start is not None:
                raise ValueError(f"nested audio start token at position {position}")
            interior_start = position + 1
        elif token == end_token_id:
            if interior_start is None:
                raise ValueError(f"audio end token without start at position {position}")
            if position == interior_start:
                raise ValueError(f"empty audio span ending at position {position}")
            bounds.append((interior_start, position))
            interior_start = None
        elif interior_start is not None and token != placeholder_token_id:
            raise ValueError(f"non-placeholder token {token} inside audio span at position {position}")
    if interior_start is not None:
        raise ValueError(f"unclosed audio span starting at position {interior_start - 1}")
    if len(bounds) != len(lengths):
        raise ValueError(f"audio span count {len(bounds)} != projected chunk count {len(lengths)}")
    for index, ((start, end), expected) in enumerate(zip(bounds, lengths)):
        if end - start != expected:
            raise ValueError(
                f"audio chunk {index}: placeholder length {end - start} != projected length {expected}"
            )
    return bounds


def insert_audio_embeddings(
    token_embeddings: torch.Tensor,
    audio_chunks: Sequence[torch.Tensor],
    bounds: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Return ``[S,H]`` embeddings with each span replaced by its ``[T,H]`` chunk.

    All bounds are half-open, positive-width, in sequence order, nonoverlapping,
    and within ``[0,S]``. Each chunk must exactly match its span length, hidden
    dimension, dtype and device. Validation completes before copying anything.
    The input is never mutated; gradients flow to audio features and to the
    token embeddings outside replaced spans. No masks or positions are changed.
    """
    if not isinstance(token_embeddings, torch.Tensor) or token_embeddings.ndim != 2:
        raise TypeError("token_embeddings must be a tensor with shape [S,H]")
    if not token_embeddings.is_floating_point() or token_embeddings.shape[1] == 0:
        raise ValueError("token_embeddings must be floating point with H > 0")
    chunks = list(audio_chunks)
    spans = list(bounds)
    if len(chunks) != len(spans):
        raise ValueError(f"audio chunk count {len(chunks)} != bound count {len(spans)}")
    sequence_length, hidden_size = token_embeddings.shape
    normalized = []
    previous_end = 0
    for index, (span, chunk) in enumerate(zip(spans, chunks)):
        pair = _integers(span, name=f"bounds[{index}]")
        if len(pair) != 2:
            raise ValueError(f"bounds[{index}] must contain exactly two positions")
        start, end = pair
        if not 0 <= start < end <= sequence_length:
            raise ValueError(f"invalid audio bound {index}: [{start}, {end}) for sequence length {sequence_length}")
        if start < previous_end:
            raise ValueError(f"audio bound {index} overlaps or is out of order")
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 2:
            raise TypeError(f"audio chunk {index} must be a tensor with shape [T,H]")
        if tuple(chunk.shape) != (end - start, hidden_size):
            raise ValueError(
                f"audio chunk {index} shape {tuple(chunk.shape)} != required {(end - start, hidden_size)}"
            )
        if chunk.dtype != token_embeddings.dtype or chunk.device != token_embeddings.device:
            raise ValueError(f"audio chunk {index} must match token embedding dtype and device")
        normalized.append((start, end))
        previous_end = end

    output = token_embeddings.clone()
    for (start, end), chunk in zip(normalized, chunks):
        output[start:end] = chunk
    return output
