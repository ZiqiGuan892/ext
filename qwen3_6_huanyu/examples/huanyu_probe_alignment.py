"""Align the known Huanyu native hook layout without matching calls by index.

The returned chunks follow ``audio_groups`` order. Unknown layouts fail closed:
this helper never guesses tensor axes or silently ignores unaccounted calls.
"""

from numbers import Integral

import numpy as np


def align_native_chunks(metadata, arrays):
    """Return ``(list[dict[str, ndarray]], None)`` or ``(None, reason)``.

    Arrays are keyed by logical hook paths, as returned by ``read_capture``.
    Valid regions are retained, while batching and cross-item padding are removed.
    """
    def positive(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(name + " must be a positive integer")
        return int(value)

    def tensor(path, ndim):
        value = arrays.get(path)
        if not isinstance(value, np.ndarray) or value.ndim != ndim:
            raise ValueError("unsupported or missing tensor layout: " + path)
        return value

    try:
        down = positive(metadata.get("encoder_downsample"), "encoder_downsample")
        fold = positive(metadata.get("projector_downsample"), "projector_downsample")
        groups = metadata.get("audio_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("audio_groups must be a non-empty list")
        lengths = []
        for group in groups:
            values = group.get("chunk_lengths")
            if not isinstance(values, list) or not values:
                raise ValueError("each audio group needs non-empty chunk_lengths")
            lengths.extend(positive(value, "chunk length") for value in values)
        counts = metadata.get("hook_counts", {})
        count = positive(counts.get("encoder"), "encoder hook count")
        if positive(counts.get("projector"), "projector hook count") != count:
            raise ValueError("encoder/projector call counts differ")
        for component in ("encoder", "projector"):
            found = {path.split("/")[1] for path in arrays if path.startswith(component + "/")}
            if found != {str(index) for index in range(count)}:
                raise ValueError(component + " hook paths disagree with hook_counts")
        chunks, offset = [], 0
        for call in range(count):
            enc = "encoder/%d/" % call
            proj = "projector/%d/" % call
            features = tensor(enc + "input/args/0", 3)
            lens = tensor(enc + "input/args/1", 1)
            encoded = tensor(enc + "output/0", 3)
            folded = tensor(proj + "input/args/0", 3)
            projected = tensor(proj + "output", 3)
            batch = features.shape[0]
            if batch <= 0 or any(value.shape[0] != batch for value in (lens, encoded, folded, projected)):
                raise ValueError("call %d has inconsistent or empty batch dimensions" % call)
            if lens.dtype.kind not in "iu" or lens.tolist() != lengths[offset:offset + batch]:
                raise ValueError("call %d lengths/order disagree with audio_groups" % call)
            if any(size <= 0 for value in (features, encoded, folded, projected) for size in value.shape):
                raise ValueError("call %d has empty tensor axes" % call)
            if folded.shape[2] != encoded.shape[2] * fold:
                raise ValueError("call %d projector width disagrees with fold factor" % call)
            if folded.shape[1] != encoded.shape[1] // fold or projected.shape[1] != folded.shape[1]:
                raise ValueError("call %d projector sequence shape disagrees with native fold" % call)
            for index, length in enumerate(lens.tolist()):
                enc_length = (length + down - 1) // down
                out_length = enc_length // fold
                if (out_length <= 0 or length > features.shape[2] or enc_length > encoded.shape[1]
                        or out_length > folded.shape[1] or out_length > projected.shape[1]):
                    raise ValueError("call %d valid lengths exceed captured tensors" % call)
                chunks.append({
                    "encoder_input": features[index, :, :length],
                    "encoder_output": encoded[index, :enc_length],
                    "projector_input": folded[index, :out_length],
                    "projector_output": projected[index, :out_length],
                })
            offset += batch
        if offset != len(lengths):
            raise ValueError("module calls do not account for every audio chunk")
        return chunks, None
    except (ValueError, TypeError, AttributeError, IndexError) as error:
        return None, str(error)
