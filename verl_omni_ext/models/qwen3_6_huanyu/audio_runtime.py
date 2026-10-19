"""Frozen Huanyu audio execution using the installed vLLM-Omni implementation.

Only the training integration lives here. The Conformer, projection activation,
and folding operation are imported from the same model implementation used by
rollout; no Whisper fallback, shape coercion, or replacement encoder is allowed.
"""

from collections.abc import Mapping
from importlib import import_module
import json
from pathlib import Path

import torch

from .audio_weights import validate_audio_state_dict


_NATIVE_PACKAGE = "vllm_omni.model_executor.models.minicpmo_5_0"
_PREFIXES = ("model.audio_encoder.", "model.audio_projection.")


def load_huanyu_runtime():
    """Return the installed native Huanyu module, shared by preprocessing/encoding."""
    return import_module(f"{_NATIVE_PACKAGE}.huanyu_audio")


def _mapping(value, name):
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"Huanyu requires {name} to be a configuration mapping")
    return dict(value)


def _field(config, name):
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


def _positive(config, name, default=None):
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Huanyu requires a positive integer {name}, got {value!r}")
    return value


def load_huanyu_audio_weights(model_path):
    """Read only audio tensors, even when a shard also contains LLM weights."""
    from safetensors import safe_open

    root = Path(model_path).resolve()
    index = root / "model.safetensors.index.json"
    expected = None
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        expected = {key: value for key, value in weight_map.items() if key.startswith(_PREFIXES)}
        files = sorted(set(expected.values()))
    else:
        files = sorted(path.name for path in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No Huanyu audio safetensors found in {root}")

    loaded = {}
    for name in files:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Checkpoint shard escapes model directory: {name}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith(_PREFIXES):
                    continue
                if key in loaded:
                    raise ValueError(f"Duplicate Huanyu audio tensor: {key}")
                if expected is not None and expected.get(key) != name:
                    raise ValueError(f"Huanyu audio tensor/index mismatch: {key} in {name}")
                loaded[key] = handle.get_tensor(key)
    if expected is not None and set(loaded) != set(expected):
        raise ValueError(f"Huanyu audio tensors missing from shards: {sorted(set(expected) - set(loaded))[:5]}")

    states = tuple({key[len(prefix):]: value for key, value in loaded.items() if key.startswith(prefix)}
                   for prefix in _PREFIXES)
    if not all(states):
        raise ValueError("Checkpoint must contain both model.audio_encoder.* and model.audio_projection.*")
    return states


def attach_huanyu_audio(module, model_config):
    """Attach and freeze native Huanyu modules before FSDP wraps the model.

    Names remain ``model.audio_encoder.*`` and ``model.audio_projection.*`` so
    saving and rollout weight transfer use the checkpoint's original namespace.
    """
    if _field(model_config, "path") is None:
        raise ValueError("Huanyu audio loading requires model_config.path")
    model = getattr(module, "model", None)
    config = getattr(module, "config", None)
    if model is None or config is None:
        raise ValueError("Huanyu expects an HF module with .model and .config")
    if getattr(model, "_huanyu_audio_runtime", None) is not None:
        return module
    if any(getattr(model, name, None) is not None for name in ("audio_encoder", "audio_projection")):
        raise ValueError("Audio modules already exist without the Huanyu contract; refusing to reuse them")

    audio = _mapping(_field(config, "audio_config"), "audio_config")
    text = _mapping(_field(config, "text_config"), "text_config")
    if audio.get("model_type") != "huanyu_conformer_0_6b" or audio.get("projector_type") != "firered_conv2d":
        raise ValueError("Expected huanyu_conformer_0_6b audio with firered_conv2d projector")
    indices = audio.get("loaded_transformer_module_indices")
    if not isinstance(indices, (list, tuple)) or len(indices) < 2 or list(indices) != list(range(len(indices))):
        raise ValueError("Huanyu needs explicit contiguous loaded_transformer_module_indices starting at 0")
    hidden = _positive(text, "hidden_size")
    embed = _positive(audio, "encoder_hidden_size")
    down = _positive(audio, "projector_downsample_rate")
    if _positive(audio, "projector_output_size") != hidden:
        raise ValueError("Huanyu projector_output_size does not match text hidden_size")

    native = load_huanyu_runtime()
    llm = import_module(f"{_NATIVE_PACKAGE}.minicpmo_5_0_omni_llm")
    # These signatures match the server constructor supplied for this checkpoint.
    with torch.device("cpu"):
        encoder = native.HuanyuAudioEncoder(
            input_feat=_positive(audio, "input_feat_per_channel"),
            embed_dim=embed,
            ffn_dim=_positive(audio, "encoder_ffn_dim", 6144),
            num_heads=_positive(audio, "encoder_num_heads", 8),
            num_conformer_layers=len(indices) - 1,
            flat_dim=_positive(audio, "vgg_flat_dim") if audio.get("vgg_flat_dim") else None,
        )
        projector = llm.MultiModalProjector(in_dim=embed * down, out_dim=hidden)
    encoder_state, projector_state = load_huanyu_audio_weights(_field(model_config, "path"))
    # Validate both modules before attachment; any omitted or mismatched key fails.
    validate_audio_state_dict(encoder, encoder_state, component="encoder")
    validate_audio_state_dict(projector, projector_state, component="projector")
    for part, state in ((encoder, encoder_state), (projector, projector_state)):
        # assign=True also materializes parameters when the enclosing HF loader
        # uses init_empty_weights; a copy into meta parameters would do nothing.
        part.load_state_dict(
            {name: value.to(torch.bfloat16) if value.is_floating_point() else value
             for name, value in state.items()},
            strict=True,
            assign=True,
        )
        if any(value.is_meta for value in part.buffers()):
            raise RuntimeError("Huanyu nonpersistent buffers were created on meta; construct them on CPU")
        # Preserve the native FP32 rotary constants through later module.to()
        # calls by the engine. These tiny CPU snapshots are not model weights.
        part._huanyu_fp32_buffers = {
            name: value.detach().float().cpu().clone()
            for name, value in part.named_buffers() if name.endswith("inv_freq")
        }
        part.requires_grad_(False)
        part.eval()
    model.audio_encoder = encoder
    model.audio_projection = projector
    model._huanyu_audio_runtime = {
        "input_features": audio["input_feat_per_channel"],
        "hidden_size": hidden,
        "downsample_rate": down,
        "fold": native.huanyu_project_and_fold,
    }
    print(
        "[huanyu] Loaded frozen audio encoder/projector strictly: "
        f"{len(encoder_state)}/{len(projector_state)} tensors; "
        f"features={audio['input_feat_per_channel']} encoder_hidden={embed} fold={down}; "
        f"native_source={native.__file__}",
        flush=True,
    )
    return module


def _frozen_bf16_call(part, *args):
    """Use temporary BF16 parameter views without changing FSDP master weights.

    Huanyu's native forward explicitly casts Conformer activations to BF16; the
    Actor may have converted its master parameters to FP32. An autocast context
    would also downcast the native FP32 VGG convolutions, so use functional_call
    instead. Parameter views live for this frozen forward only, never in a cache.
    The caller must enter after FSDP has unsharded the relevant parameters.
    """
    if hasattr(part, "unshard") and hasattr(part, "reshard"):
        raise RuntimeError("Huanyu audio must not be independently FSDP-wrapped for functional execution")
    parameters = dict(part.named_parameters())
    if any(value.requires_grad for value in parameters.values()):
        raise ValueError("Huanyu audio runtime currently supports frozen audio only")
    if any(hasattr(value, "placements") for value in parameters.values()):
        raise RuntimeError("Huanyu audio parameters are still sharded; call from an unsharded FSDP forward")
    state = {name: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
             for name, value in parameters.items()}
    # Keep RoPE inv_freq buffers in their current precision (normally FP32).
    buffers = dict(part.named_buffers())
    for name, original in getattr(part, "_huanyu_fp32_buffers", {}).items():
        buffers[name] = original.to(device=buffers[name].device)
    state.update(buffers)
    with torch.no_grad():
        return torch.func.functional_call(part, state, args, strict=True)


def encode_huanyu_audio(module, features, lengths):
    """Encode chunk-batched fbank features into valid [tokens, text_hidden] rows.

    ``features`` is [number_of_chunks, feature_bins, padded_frames]; ``lengths``
    contains actual, unpadded frame counts. Chunk order is preserved.
    """
    model = getattr(module, "model", None)
    runtime = getattr(model, "_huanyu_audio_runtime", None)
    if runtime is None:
        raise RuntimeError("Huanyu audio modules have not been attached")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("Huanyu features must have shape [chunks, feature_bins, padded_frames]")
    lengths = torch.as_tensor(lengths, device=features.device)
    if lengths.ndim != 1 or len(lengths) != features.shape[0] or lengths.is_floating_point():
        raise ValueError("Huanyu lengths must be one integer per feature chunk")
    if features.shape[1] != runtime["input_features"]:
        raise ValueError(f"Huanyu fbank mismatch: expected {runtime['input_features']}, got {features.shape[1]}")
    if torch.any(lengths <= 0) or torch.any(lengths > features.shape[2]):
        raise ValueError("Huanyu lengths must fit the unpadded feature chunks")
    if features.shape[0] == 0:
        return []
    encoder, projector = model.audio_encoder, model.audio_projection
    device = next(encoder.parameters()).device
    encoder.eval()
    projector.eval()
    features = features.to(device=device, dtype=torch.bfloat16)
    lengths = lengths.to(device=device, dtype=torch.int64)
    with torch.no_grad():
        # Disable any outer mixed-precision scope: VGG must perform explicit FP32
        # convolutions while the source encoder owns its BF16 Conformer casts.
        with torch.autocast(device_type=device.type, enabled=False):
            encoded, _ = _frozen_bf16_call(encoder, features, lengths)
            projected, valid = runtime["fold"](
                encoded,
                lengths,
                lambda values: _frozen_bf16_call(projector, values),
                enc_downsample=encoder.fwd_downsample,
                downsample_rate=runtime["downsample_rate"],
            )
    expected = ((lengths + encoder.fwd_downsample - 1) // encoder.fwd_downsample
                // runtime["downsample_rate"])
    if torch.any(expected <= 0) or not torch.equal(valid.to(expected), expected):
        raise ValueError("Huanyu audio output lengths disagree with the encoder/fold contract")
    if projected.ndim != 3 or projected.shape[0] != len(lengths) or projected.shape[2] != runtime["hidden_size"]:
        raise ValueError("Huanyu projection output shape disagrees with text hidden_size")
    if torch.any(expected > projected.shape[1]):
        raise ValueError("Huanyu projection produced fewer tokens than the real feature lengths require")
    return [projected[index, :int(count)].detach() for index, count in enumerate(expected.tolist())]
