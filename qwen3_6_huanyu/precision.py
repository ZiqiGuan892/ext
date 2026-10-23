"""Share Qwen language arithmetic with Whisper without its audio/packed forward.

Huanyu intentionally keeps padded inputs and its own differentiable audio
encoder. Only the common Qwen3.5 language backbone receives these patches.
Optional arithmetic references retain the Whisper switches and defaults.
"""

import importlib
import os


_SHARED = "verl_omni_ext.models.qwen3_6_huanyu.qwen_precision"
_LINEAR_SCOPE = "_huanyu_stable_linear_scoped"


def _shared(name):
    return importlib.import_module(f"{_SHARED}.{name}")


def _enabled(name, default="0"):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _validate_tp_references():
    requested = False
    for name in ("MINICPMO_MOE_TP_REFERENCE", "MINICPMO_ROW_TP_REFERENCE"):
        value = os.environ.get(name, "0")
        if value not in ("0", "4"):
            raise ValueError(f"{name} supports only 0 or 4, got {value!r}")
        requested = requested or value == "4"
    if requested and os.environ.get("MINICPMO_HUANYU_ROLLOUT_TP") != "4":
        raise ValueError(
            "Huanyu TP4 arithmetic references require actual rollout TP=4; "
            "set MINICPMO_HUANYU_ROLLOUT_TP=4 through the training launcher"
        )


def _validate_vision_references():
    if not _enabled("MINICPMO_VISION_POSITION_REFERENCE"):
        return
    if _enabled("MINICPMO_NATIVE_VISION_REFERENCE"):
        raise ValueError(
            "Choose only one direction of vision arithmetic alignment: "
            "MINICPMO_VISION_POSITION_REFERENCE and MINICPMO_NATIVE_VISION_REFERENCE conflict"
        )
    if os.environ.get("HUANYU_USE_PRECOMPUTE", "0") == "1":
        raise ValueError(
            "MINICPMO_VISION_POSITION_REFERENCE cannot consume existing Huanyu precompute caches; "
            "the cache producer has not recorded this arithmetic policy"
        )


def _install_linear(module, shared):
    """Let the Huanyu installer own its scope, including later actor/ref models."""
    shared.install_actor(module)


def _gdn_coverage(language):
    """Inspect bindings only: no tensor reads, device sync or kernel imports."""
    counts = dict(total=0, guarded=0, native=0, unguarded_torch=0, unrecognized=0)
    for layer in language.layers:
        attention = getattr(layer, "linear_attn", None)
        if attention is None:
            continue
        counts["total"] += 1
        kernel = getattr(attention, "chunk_gated_delta_rule", None)
        source = getattr(kernel, "__module__", "")
        name = getattr(kernel, "__name__", "")
        if getattr(kernel, "_huanyu_fp32_guard", False) is True:
            counts["guarded"] += 1
        elif source.startswith(("fla.", "flash_linear_attention.")):
            counts["native"] += 1
        elif source.startswith("transformers.") and name == "torch_chunk_gated_delta_rule":
            counts["unguarded_torch"] += 1
        else:
            counts["unrecognized"] += 1
    return counts


def install_huanyu_actor_precision(module):
    """Install language precision fixes after loading/freezing, before FSDP.

Do not call Whisper's install_packed_forward: Huanyu uses remove_padding=false
and preserves its separate audio injection and four-axis position bridge.
The HF torch GDN fallback still needs autocast disabled with padded inputs:
explicit .float() alone does not stop BF16 autocast on recurrence matmuls.
"""
    language = module.model.language_model
    _validate_tp_references()
    _validate_vision_references()
    if _enabled("MINICPMO_ISOLATE_FULL_ATTENTION"):
        raise ValueError(
            "MINICPMO_ISOLATE_FULL_ATTENTION requires packed forward; Huanyu uses padded inputs. "
            "Zero-valued padding positions must not be treated as packed sequence boundaries"
        )
    for layer in language.layers:
        if not hasattr(layer, "layer_type") and hasattr(layer, "block_type"):
            layer.layer_type = layer.block_type

    guarded = 0
    if _enabled("MINICPMO_HUANYU_GDN_FP32_TRAINING", "1"):
        guarded = _shared("packed").isolate_torch_gdn_fallbacks(module)

    # Keep the same operation order as the Whisper actor. A TP4 block reference
    # intentionally supersedes the simpler expert reference if both are chosen.
    if _enabled("MINICPMO_STABLE_LINEAR"):
        _install_linear(module, _shared("linear_precision"))
    if _enabled("MINICPMO_MOE_MATCH_ROLLOUT"):
        _shared("moe_precision").install_actor(language)
    if os.environ.get("MINICPMO_MOE_TP_REFERENCE", "0") == "4":
        _shared("moe_tp_reference").install_actor(language)
    if os.environ.get("MINICPMO_ROW_TP_REFERENCE", "0") == "4":
        _shared("row_tp_reference").install_actor(language)
    if _enabled("MINICPMO_VISION_POSITION_REFERENCE"):
        _shared("vision_position_precision").install_actor(module)
    if _enabled("MINICPMO_ISOLATE_MEDIA_ENCODERS", "1"):
        # The shared actor helper touches frozen visual only. Its Whisper
        # audio helper is deliberately never installed on Huanyu.
        _shared("media_precision").install_actor(module)

    coverage = _gdn_coverage(language)
    if os.environ.get("HUANYU_PRECISION_PROFILE") == "full" and (
        not coverage["total"] or coverage["guarded"] != coverage["total"]
    ):
        raise RuntimeError(f"Huanyu full precision requires verified HF torch GDN guards on every layer: {coverage}")
    print(
        f"[huanyu-precision] actor padded forward retained; "
        f"GDN coverage total={coverage['total']} guarded={coverage['guarded']} "
        f"new={guarded} already={coverage['guarded'] - guarded} "
        f"native={coverage['native']} unguarded_torch={coverage['unguarded_torch']} "
        f"unrecognized={coverage['unrecognized']}; "
        "coverage reports bindings only; execution is logged separately; "
        "Huanyu audio forward and gradients unchanged",
        flush=True,
    )
    return module
