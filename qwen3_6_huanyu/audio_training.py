"""Differentiable online Huanyu audio; frozen/cache execution stays unchanged.

The encoder and projector retain their checkpoint names and belong to the FSDP
root. No audio child may perform collectives inside a variable-length chunk
loop. BF16 compute views remain connected to the optimizer's master parameters.
"""

from collections.abc import Mapping
import os

import torch
from torch.utils.checkpoint import checkpoint

from .training_logging import get_training_logger


logger = get_training_logger(__name__)


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _option(model_config, name, default):
    overrides = _get(model_config, "override_config", {})
    value = _get(overrides, name, _get(_get(model_config, "hf_config"), name, default))
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def resolve_freeze_audio(model_config):
    """Public model override; environment variables never silently unfreeze."""
    return _option(model_config, "huanyu_freeze_audio", True)


def resolve_audio_checkpointing(model_config):
    return _option(model_config, "huanyu_audio_gradient_checkpointing", True)


def validate_audio_mode(model_config):
    frozen = resolve_freeze_audio(model_config)
    if not frozen and any(os.environ.get(key, "0") == "1" for key in
                          ("HUANYU_USE_PRECOMPUTE", "MINICPMO_USE_PRECOMPUTE")):
        raise ValueError("huanyu_freeze_audio=false requires online encoding; precomputed embeddings cannot train audio")
    if frozen and os.environ.get("MINICPMO_TUNE_AUDIO", "0") == "1":
        raise ValueError("MINICPMO_TUNE_AUDIO=1 conflicts with frozen audio: set model.override_config.huanyu_freeze_audio=false")
    if not frozen and os.environ.get("MINICPMO_TUNE_VISION", "0") == "1":
        raise ValueError("Huanyu audio training currently requires AQA audio-only data and frozen vision")
    return frozen


def _assert_root_owned(part):
    # Check descendants too: calling a wrapped Conformer repeatedly would add
    # rank-dependent collectives even if the enclosing audio module is plain.
    for name, child in part.named_modules():
        if ((hasattr(child, "unshard") and hasattr(child, "reshard"))
                or hasattr(child, "_fsdp_wrapped_module")):
            raise RuntimeError(
                f"Huanyu audio child {name or '<root>'} is independently FSDP-wrapped; "
                "keep audio under the outer FSDP root, wrap only language decoder/embedding modules"
            )
    if any(hasattr(value, "placements") for value in part.parameters()):
        raise RuntimeError("Huanyu audio parameters are still sharded; encode inside the outer FSDP forward")


def configure_trainable_audio(module, checkpoint_chunks=True):
    """Run before optimizer construction and FSDP wrapping."""
    model = module.model
    if getattr(model, "_huanyu_audio_runtime", None) is None:
        raise RuntimeError("Attach native Huanyu audio modules before configuring training")
    wrap_names = set(getattr(module, "_no_split_modules", ()) or ())
    for name in ("audio_encoder", "audio_projection"):
        part = getattr(model, name)
        _assert_root_owned(part)
        for child_name, child in part.named_modules():
            if (type(child).__name__ in wrap_names or isinstance(child, torch.nn.Embedding)
                    or child_name.rsplit(".", 1)[-1] in ("embed_tokens", "lm_head")):
                raise ValueError(
                    f"{name}.{child_name} would be independently FSDP-wrapped; "
                    "remove audio classes from the wrap policy (retain language decoder wrapping)"
                )
        part.requires_grad_(True)
        part.train(module.training)
        part._huanyu_train_component = name
    model._huanyu_audio_runtime["checkpoint_chunks"] = bool(checkpoint_chunks)
    model._huanyu_audio_runtime["trainable"] = True
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters())
              for name in ("audio_encoder", "audio_projection")}
    logger.info("[huanyu-audio-train] enabled encoder_params=%s projector_params=%s "
                "ownership=fsdp_root checkpoint_chunks=%s cache=off",
                counts["audio_encoder"], counts["audio_projection"], checkpoint_chunks)


def _trainable_bf16_call(part, *args):
    _assert_root_owned(part)
    # .to() has a differentiable cast; never replace master Parameters or detach.
    state = {name: value.to(torch.bfloat16) if value.is_floating_point() else value
             for name, value in part.named_parameters()}
    if torch.is_grad_enabled() and not getattr(part, "_huanyu_backward_observed", False):
        # Attach to the current full-parameter compute view, not the Parameter
        # object replaced by FSDP. No tensor values or large norm temporaries.
        name, value = next(iter(state.items()))
        if value.requires_grad:
            def observe(grad, parameter_name=name):
                if not getattr(part, "_huanyu_backward_observed", False):
                    logger.info("[huanyu-audio-train] backward component=%s parameter=%s shape=%s dtype=%s",
                                part._huanyu_train_component, parameter_name, tuple(grad.shape), grad.dtype)
                    part._huanyu_backward_observed = True
                return grad
            value.register_hook(observe)
    buffers = dict(part.named_buffers())
    for name, original in getattr(part, "_huanyu_fp32_buffers", {}).items():
        buffers[name] = original.to(device=buffers[name].device)
    state.update(buffers)
    return torch.func.functional_call(part, state, args, strict=True)


def _encode_chunk(module, features, lengths):
    model = module.model
    encoder, projector = model.audio_encoder, model.audio_projection
    runtime = model._huanyu_audio_runtime
    device = next(encoder.parameters()).device
    with torch.autocast(device_type=device.type, enabled=False):
        encoded, _ = _trainable_bf16_call(encoder, features, lengths)
        if torch.is_grad_enabled() and not encoded.requires_grad:
            raise RuntimeError("Huanyu native encoder detached its output; encoder training is impossible")
        projected, valid = runtime["fold"](
            encoded, lengths, lambda values: _trainable_bf16_call(projector, values),
            enc_downsample=encoder.fwd_downsample, downsample_rate=runtime["downsample_rate"],
        )
    expected = ((lengths + encoder.fwd_downsample - 1) // encoder.fwd_downsample
                // runtime["downsample_rate"])
    if torch.any(expected <= 0) or not torch.equal(valid.to(expected), expected):
        raise ValueError("Huanyu audio output lengths disagree with the encoder/fold contract")
    if (projected.ndim != 3 or projected.shape[0] != 1
            or projected.shape[2] != runtime["hidden_size"]
            or torch.any(expected > projected.shape[1])):
        raise ValueError("Huanyu projection output shape disagrees with the audio/text contract")
    return projected[0, :int(expected[0])]


def encode_trainable_huanyu_audio(module, features, lengths):
    """Encode real chunks with gradients, also usable in no_grad logprob passes."""
    runtime = getattr(module.model, "_huanyu_audio_runtime", {})
    if not runtime.get("trainable"):
        raise RuntimeError("Huanyu online audio training was not configured")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("Huanyu features must have shape [chunks, feature_bins, padded_frames]")
    lengths = torch.as_tensor(lengths, device=features.device)
    if (lengths.ndim != 1 or len(lengths) != len(features) or lengths.is_floating_point()
            or lengths.is_complex() or lengths.dtype == torch.bool):
        raise ValueError("Huanyu lengths must be one integer per feature chunk")
    if torch.any(lengths <= 0) or torch.any(lengths > features.shape[2]):
        raise ValueError("Huanyu lengths must fit the unpadded feature chunks")
    if features.shape[1] != runtime["input_features"]:
        raise ValueError(f"Huanyu fbank mismatch: expected {runtime['input_features']}, got {features.shape[1]}")
    encoder, projector = module.model.audio_encoder, module.model.audio_projection
    for part in (encoder, projector):
        _assert_root_owned(part)
        part.train(module.training)
    if torch.is_grad_enabled() and any(not p.requires_grad for part in (encoder, projector)
                                      for p in part.parameters()):
        raise RuntimeError("Huanyu trainable audio parameters were frozen after adapter configuration")
    device = next(encoder.parameters()).device
    outputs = []
    # Lengths are metadata. Moving them to a Python list does not detach features
    # or encoder activations, and adds no distributed operation.
    for index, length in enumerate(lengths.tolist()):
        chunk = features[index:index + 1, :, :length]
        chunk = torch.nn.functional.pad(chunk, (0, -length % 4)).to(device=device, dtype=torch.bfloat16)
        real_length = lengths[index:index + 1].to(device=device, dtype=torch.int64)
        if torch.is_grad_enabled() and runtime["checkpoint_chunks"]:
            # Non-reentrant checkpoint supports parameters requiring gradients
            # even when the raw fbank input itself does not require gradients.
            output = checkpoint(lambda x, n: _encode_chunk(module, x, n),
                                chunk, real_length, use_reentrant=False)
        else:
            output = _encode_chunk(module, chunk, real_length)
        if torch.is_grad_enabled() and not output.requires_grad:
            raise RuntimeError("Huanyu native audio forward detached its output; encoder training is impossible")
        outputs.append(output)
    mode = "grad" if torch.is_grad_enabled() else "no_grad"
    marker = f"_logged_{mode}"
    if not runtime.get(marker):
        logger.info("[huanyu-audio-train] forward mode=%s chunks=%s frames=%s "
                    "output_requires_grad=%s ownership=fsdp_root",
                    mode, len(outputs), lengths.tolist()[:8],
                    any(x.requires_grad for x in outputs))
        runtime[marker] = True
    return outputs
