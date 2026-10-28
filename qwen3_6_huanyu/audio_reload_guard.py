"""Completeness and bounded value checks for a full Huanyu audio weight update.

Construct before vLLM replaces parameters with reload metadata, ingest each
owned bucket before loading it, and check completeness before finalization.
Partially materializing a layer can otherwise copy uninitialized parameters
back into the live model. After finalization, verify the eight small VGG
LayerNorm tensors against received values, without modifying any weights.
"""

from collections.abc import Mapping


_AUDIO_PREFIXES = (
    ("model.audio_encoder.", "model.audio_encoder."),
    ("apm.", "model.audio_encoder."),
    ("model.audio_projection.", "model.audio_projection."),
    ("audio_projection_layer.", "model.audio_projection."),
)
_VGG_NORMS = frozenset(
    f"model.audio_encoder.conv_layers.{block}.layers.{layer}.{leaf}"
    for block in (0, 1)
    for layer in (1, 4)
    for leaf in ("weight", "bias")
)


class AudioReloadGuard:
    """Reject incomplete, duplicate, or incompatible full audio updates.

    Keep metadata and independent CPU copies of at most eight VGG LayerNorm
    tensors (at most 256 elements each), never IPC buffers or the model itself.
    Non-audio weights are left to the native model loader. Omitting all audio
    weights is allowed: vLLM restores completely untouched layers. Once any
    audio weight arrives, the whole audio update must be complete; partial audio
    updates must not use this full-update protocol.
    """

    def __init__(self, model):
        self._expected = {}
        self._received = {}
        self._norm_values = {}
        thinker = getattr(model, "thinker", None)
        for module_name, prefix in (
            ("apm", "model.audio_encoder."),
            ("audio_projection_layer", "model.audio_projection."),
        ):
            module = getattr(thinker, module_name, None)
            if module is None or not callable(getattr(module, "named_parameters", None)):
                raise RuntimeError(f"Cannot validate Huanyu audio reload: thinker.{module_name} is absent")
            parameters = {
                prefix + name: tuple(param.shape)
                for name, param in module.named_parameters()
            }
            if not parameters:
                raise RuntimeError(f"Cannot validate Huanyu audio reload: thinker.{module_name} has no parameters")
            self._expected.update(parameters)

    def ingest(self, weights):
        """Validate one bucket before loading; return its audio tensor count."""
        items = weights.items() if isinstance(weights, Mapping) else weights
        pending = {}
        norm_values = {}
        for source_name, value in items:
            canonical_name = None
            for prefix, canonical_prefix in _AUDIO_PREFIXES:
                if source_name.startswith(prefix):
                    canonical_name = canonical_prefix + source_name[len(prefix):]
                    break
            if canonical_name is None:
                continue
            if canonical_name not in self._expected:
                raise ValueError(f"Unknown Huanyu audio reload parameter: {source_name}")
            previous = self._received.get(canonical_name, pending.get(canonical_name))
            if previous is not None:
                raise ValueError(
                    f"Duplicate Huanyu audio reload parameter: {canonical_name}; "
                    f"received as {previous!r} and {source_name!r}"
                )
            expected_shape = self._expected[canonical_name]
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Huanyu audio reload shape mismatch for {source_name}: "
                    f"expected {expected_shape}, received {actual_shape}"
                )
            if canonical_name in _VGG_NORMS:
                if len(actual_shape) != 1 or actual_shape[0] > 256:
                    raise ValueError(
                        f"Huanyu VGG norm reload verification requires a small 1D parameter: "
                        f"{canonical_name} has shape {actual_shape}"
                    )
                norm_values[canonical_name] = value.detach().to(device="cpu").clone()
            pending[canonical_name] = source_name
        self._received.update(pending)
        self._norm_values.update(norm_values)
        return len(pending)

    def check_complete(self):
        """Fail before finalization if any audio parameter was not delivered."""
        if not self._received:
            return
        missing = sorted(self._expected.keys() - self._received.keys())
        if missing:
            raise RuntimeError(
                f"Incomplete Huanyu audio reload: received {len(self._received)}/{len(self._expected)} "
                f"parameters; missing={missing[:8]}"
                + (f" (and {len(missing) - 8} more)" if len(missing) > 8 else "")
            )

    def verify(self, model):
        """Verify finalized VGG norms match received values after dtype casting.

        No checkpoint reads or repairs are performed. This tests the receiver's
        reload boundary; it cannot establish whether the sender's values were
        correct before they arrived.
        """
        if not self._norm_values:
            return
        import torch

        encoder = model.thinker.apm
        for name, received in self._norm_values.items():
            relative = name[len("model.audio_encoder."):]
            path, _, leaf = relative.rpartition(".")
            param = getattr(encoder.get_submodule(path), leaf)
            if param.is_meta:
                raise RuntimeError(f"Cannot verify Huanyu audio reload: {name} is still on meta")
            actual = param.detach().to(device="cpu")
            expected = received.to(dtype=actual.dtype)
            if actual.shape != expected.shape:
                raise RuntimeError(
                    f"Huanyu audio reload verification failed for {name}: "
                    f"received shape={tuple(expected.shape)}, finalized shape={tuple(actual.shape)}"
                )
            if not torch.equal(actual, expected):
                mismatch = torch.nonzero(actual.reshape(-1) != expected.reshape(-1)).flatten()
                raise RuntimeError(
                    f"Huanyu audio reload verification failed for {name}: "
                    f"finalized values differ from received values cast to {actual.dtype}; "
                    f"mismatched_elements={mismatch.numel()}/{actual.numel()} "
                    f"bad_indices={mismatch[:16].tolist()}"
                )
