"""Strict loading for a real Huanyu module supplied by the model implementation.

This file deliberately does not construct a Whisper encoder, infer an audio
network from tensor shapes, or select a projection activation. Those choices
must come from the Huanyu model's actual source.
"""

from collections.abc import Mapping


def validate_audio_state_dict(module, state_dict: Mapping, *, component: str) -> None:
    """Check every key and shape before mutating the target module.

    ``strict=False`` would silently accept the observed 290 unexpected Huanyu
    keys when the destination is a Whisper encoder. It also does not suppress
    projection shape errors. Reject either situation with a compact summary.
    """
    expected = module.state_dict()
    if not expected or not state_dict:
        raise ValueError(f"Huanyu {component}: empty target or checkpoint state_dict")
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatches = [
        (name, tuple(state_dict[name].shape), tuple(expected[name].shape))
        for name in sorted(set(expected) & set(state_dict))
        if tuple(state_dict[name].shape) != tuple(expected[name].shape)
    ]
    if missing or unexpected or mismatches:
        raise ValueError(
            f"Huanyu {component}: incompatible checkpoint; "
            f"missing={len(missing)} sample={missing[:5]}; "
            f"unexpected={len(unexpected)} sample={unexpected[:5]}; "
            f"shape_mismatches={len(mismatches)} "
            f"sample(name, checkpoint, target)={mismatches[:5]}. "
            "Use the actual Huanyu architecture; do not reuse WhisperEncoder "
            "or reshape/truncate weights to make them fit."
        )


def load_audio_state_dict(module, state_dict: Mapping, *, component: str):
    """Load a complete encoder or projector state after checking compatibility.

    Caller owns prefix extraction, module construction, dtype/device placement,
    and the forward path. Matching keys/shapes alone do not prove semantic or
    numerical equivalence; this is a loading check, not a complete adapter.
    """
    validate_audio_state_dict(module, state_dict, component=component)
    return module.load_state_dict(state_dict, strict=True)
