"""Install the common Qwen precision fixes on Huanyu inference instances.

Importing this module is safe in a training process. The engine entry point
calls the installer only after loading the native vLLM-Omni registrations.
Huanyu audio keeps its native encoder/fold implementation.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from importlib import import_module
import os
from types import MethodType

import torch


_active = ContextVar("huanyu_precision_forward", default=False)
_SHARED = "verl_omni_ext.models.qwen3_6_huanyu.qwen_precision."
_NATIVE = "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni"


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def is_huanyu_config(vllm_config):
    model_config = _get(vllm_config, "model_config", vllm_config)
    hf = _get(model_config, "hf_config", model_config)
    return _get(_get(hf, "audio_config"), "model_type") == "huanyu_conformer_0_6b"


def _shared(name):
    return import_module(_SHARED + name)


def _install_audio(native):
    from .audio_rollout import install_audio_isolation

    return install_audio_isolation(native)


def install_recurrent_fp32():
    """Keep FP32 cache values, updating only the slots scheduled this step."""
    if os.environ.get("MINICPMO_GDN_FP32_REFERENCE", "1") != "1":
        print("[huanyu-precision] component=gdn_decode phase=disabled", flush=True)
        return False
    ops = torch.ops._C_ascend
    if not hasattr(ops, "npu_recurrent_gated_delta_rule"):
        return False  # Native model initialization retries once the ops exist.
    original = ops.npu_recurrent_gated_delta_rule
    if getattr(original, "_huanyu_fp32_state", False):
        return True
    reference = _shared("gdn_precision").recurrent_gdn_fp32
    reported = False

    @wraps(original)
    def recurrent(*args, **kwargs):
        nonlocal reported
        if not _active.get():
            return original(*args, **kwargs)
        state = kwargs.get("state", args[3] if len(args) > 3 else None)
        if state is None or state.dtype != torch.float32:
            raise RuntimeError(
                "Huanyu FP32 GDN requires a float32 recurrent cache; check "
                "text_config.mamba_ssm_dtype / mamba_ssm_cache_dtype. "
                "Set MINICPMO_GDN_FP32_REFERENCE=0 only for an explicit legacy comparison."
            )
        with torch.autocast(device_type=state.device.type, enabled=False):
            result = reference(*args, **kwargs)
        if not reported:
            reported = True
            print("[huanyu-precision] component=gdn_decode phase=executed "
                  "GDN decode executing FP32; only scheduled cache slots updated", flush=True)
        return result

    recurrent._huanyu_fp32_state = True
    # The inherited Whisper initializer retries its legacy BF16 wrapper. Mark
    # this scoped wrapper as handled so a later init cannot wrap it again.
    recurrent._is_bf16_patched = True
    recurrent.default = recurrent  # Some Ascend versions call the overload explicitly.
    ops.npu_recurrent_gated_delta_rule = recurrent
    # SafeGDN also replaced the saved OpOverloadPacket.default. Its raw callable
    # is an OpOverload (not that packet), so replacing this alias cannot recurse
    # when the non-Huanyu path delegates to SafeGDN.
    raw = vars(original).get("_raw_op") if hasattr(original, "__dict__") else None
    packet = getattr(raw, "_overloadpacket", None) if raw is not None else None
    if packet is not None and packet is not raw:
        packet.default = recurrent
    print("[huanyu-precision] component=gdn_decode phase=installed scope=huanyu cache_dtype=float32", flush=True)
    return True


def _scope_forward(native):
    if getattr(native, "_huanyu_precision_scope", False):
        return
    original = native.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        token = _active.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _active.reset(token)

    native.forward = MethodType(forward, native)
    native._huanyu_precision_scope = True


def _isolate_vision(native):
    if os.environ.get("MINICPMO_ISOLATE_MEDIA_ENCODERS", "1") != "1":
        return False
    thinker = getattr(native, "thinker", native)
    original = getattr(thinker, "_q36_encode_vision", None)
    if original is None or getattr(original, "_huanyu_isolated_vision", False):
        return False
    split = _shared("media_precision").split_vision_inputs
    reported = False

    @wraps(original)
    def vision(self, pixels, grids):
        nonlocal reported
        if getattr(self, "visual", None) is None:
            return original(pixels, grids)
        result = tuple(item for one_pixels, one_grid in split(pixels, grids)
                       for item in original(one_pixels, one_grid))
        if result and not reported:
            reported = True
            print(f"[huanyu-precision] component=vision_rollout phase=executed "
                  f"pid={os.getpid()} items={len(result)} per_item_shape=True", flush=True)
        return result

    vision._huanyu_isolated_vision = True
    thinker._q36_encode_vision = MethodType(vision, thinker)
    return True


def configure_huanyu_rollout(native, vllm_config):
    if not is_huanyu_config(vllm_config):
        return False
    if getattr(native, "_huanyu_precision_configured", False):
        return True
    if os.environ.get("MINICPMO_NATIVE_VISION_REFERENCE", "0") == "1":
        raise ValueError(
            "MINICPMO_NATIVE_VISION_REFERENCE uses Whisper's HF SDPA/FSDP vision "
            "reference, which is not compatible with Huanyu's eager-Q128 embedding "
            "cache. Keep it 0; changing vision math requires a matching cache producer."
        )
    _scope_forward(native)
    fp32 = install_recurrent_fp32()
    if os.environ.get("MINICPMO_GDN_FP32_REFERENCE", "1") == "1" and not fp32:
        raise RuntimeError("Huanyu GDN precision hook unavailable after native init: Ascend recurrent op was not loaded")
    gdn = _shared("gdn_precision")
    gdn.install_reference_prefill(native)
    gdn.install_mixed_cache_barrier(native)
    # These installers already restrict edits to the selected language model;
    # they are deliberately opt-in, just as in the Whisper implementation.
    _shared("norm_precision").install_rollout(native)
    _shared("text_rope_precision").install_rollout(native)
    _shared("router_precision").install_rollout(native)
    _shared("linear_precision").install_rollout(native)
    vision = _isolate_vision(native)
    audio = _install_audio(native)
    # Capture the real rollout audio boundary, after the active isolation
    # wrapper. The neutral probe also runs in standalone serve without
    # importing this precision policy. No tensor reads when it is disabled.
    if os.environ.get("HUANYU_AUDIO_PROBE_DIR"):
        from .examples.huanyu_audio_probe import attach_audio_probe

        native._huanyu_audio_probe_config = vllm_config
        attach_audio_probe(native)
    if os.environ.get("HUANYU_EMBEDDING_AB_DIR", "").strip():
        from .prompt_embedding_probe import attach_model_probe

        attach_model_probe(native, vllm_config)
    native._huanyu_precision_configured = True
    print(f"[huanyu-precision] rollout installed: gdn_fp32={fp32} isolated_vision={vision} "
          f"isolated_audio={audio}; native Huanyu audio retained", flush=True)
    return True


def install_huanyu_rollout_precision():
    """Retry precision setup AFTER the inherited native initializer runs."""
    native_cls = import_module(_NATIVE).MiniCPMO50OmniForConditionalGeneration
    original = native_cls.__init__
    if getattr(original, "_huanyu_precision_init", False):
        return True

    @wraps(original)
    def init(self, *args, **kwargs):
        config = kwargs.get("vllm_config", kwargs.get("config", args[0] if args else None))
        original(self, *args, **kwargs)
        if not is_huanyu_config(config):
            config = getattr(self, "vllm_config", getattr(self, "config", config))
        configure_huanyu_rollout(self, config)

    init._huanyu_precision_init = True
    native_cls.__init__ = init
    print("[huanyu-precision] native model-init hook installed (Huanyu checkpoint only)", flush=True)
    return True
