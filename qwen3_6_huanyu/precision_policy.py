"""Validate the requested Huanyu precision policy inside each real worker."""

from functools import wraps
import importlib
import inspect
import os
import socket


FULL_ON = (
    "MINICPMO_HUANYU_GDN_FP32_TRAINING", "MINICPMO_GDN_FP32_REFERENCE",
    "MINICPMO_HUANYU_LAYERWISE_RELOAD", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE",
    "MINICPMO_ISOLATE_MEDIA_ENCODERS", "MINICPMO_GDN_CACHE_BARRIER",
    "MINICPMO_GDN_PREFILL_REFERENCE", "MINICPMO_GDN_L2NORM_REFERENCE",
    "MINICPMO_GDN_CONV_REFERENCE", "MINICPMO_GDN_GATING_REFERENCE",
    "MINICPMO_PRIVATE_HYBRID_CACHE", "MINICPMO_STABLE_LINEAR",
)
_REPORTED = set()


def report_policy(role):
    # Direct Python integrations retain custom mode; all supported launchers
    # select full by default and forward its exact flags through runtime_env.
    profile = os.environ.get("HUANYU_PRECISION_PROFILE", "custom")
    if profile not in {"full", "custom"}:
        raise ValueError(f"Unknown HUANYU_PRECISION_PROFILE={profile!r}; choose full or custom")
    expected = dict.fromkeys(FULL_ON, "1")
    expected["MINICPMO_FSDP2_PIN_MEMORY"] = "0"
    if profile == "full":
        mismatches = {key: os.environ.get(key, "<unset>") for key, value in expected.items()
                      if os.environ.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Huanyu full precision policy was not propagated to {role}: {mismatches}")
        if os.environ.get("HUANYU_GRAPH_MODE", "0").lower() not in {"0", "false", "off", "no"}:
            raise RuntimeError("Huanyu full precision references require eager mode (HUANYU_GRAPH_MODE=0)")
    snapshot = tuple((key, os.environ.get(key, "<unset>")) for key in expected)
    communication = tuple((key, os.environ.get(key, "<unset>")) for key in (
        "HCCL_OP_EXPANSION_MODE", "HUANYU_HCCL_OP_EXPANSION_MODE",
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "ASCEND_RT_VISIBLE_DEVICES",
    ))
    marker = (role, profile, snapshot, communication)
    if marker not in _REPORTED:
        _REPORTED.add(marker)
        print(f"[huanyu-precision] phase=configured role={role} profile={profile} "
              f"host={socket.gethostname()} pid={os.getpid()} source={__file__} "
              + " ".join(f"{key}={value}" for key, value in snapshot), flush=True)
        # Report the worker's own environment, including reused Ray workers.
        # This confirms propagation, not successful HCCL communication.
        print(f"[huanyu-communication] event=environment role={role} "
              f"host={socket.gethostname()} pid={os.getpid()} "
              + " ".join(f"{key}={value}" for key, value in communication), flush=True)
    return profile


def install_fsdp_pin_audit():
    """Enforce and observe the shared MR's policy at actual construction time.

    Keep the genuine PyTorch policy object and the caller's offload selection.
    This also covers a shared installer that was skipped during package import.
    """
    if os.environ.get("MINICPMO_FSDP2_PIN_MEMORY", "0") == "1":
        print("[huanyu-precision] component=fsdp_pin_memory phase=disabled", flush=True)
        return False
    target = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    original = target.CPUOffloadPolicy
    if getattr(original, "_huanyu_pin_audit", False):
        return True
    # Shared patch factories retain the underlying class in .cls. Its signature
    # handles positional pin_memory too, without duplicate kwargs.
    signature = inspect.signature(getattr(original, "cls", original))
    if "pin_memory" not in signature.parameters:
        raise RuntimeError(f"Unsupported FSDP CPUOffloadPolicy signature: {signature}")
    reported = False

    @wraps(original)
    def policy(*args, **kwargs):
        nonlocal reported
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments["pin_memory"] = False
        result = original(*bound.args, **bound.kwargs)
        if getattr(result, "pin_memory", None) is not False:
            raise RuntimeError("FSDP CPUOffloadPolicy did not preserve pin_memory=False")
        if not reported:
            reported = True
            print(f"[huanyu-precision] component=fsdp_pin_memory phase=executed "
                  f"pid={os.getpid()} pin_memory=False policy={type(result).__name__}", flush=True)
        return result

    policy._huanyu_pin_audit = True
    policy.cls = getattr(original, "cls", original)
    target.CPUOffloadPolicy = policy
    print(f"[huanyu-precision] component=fsdp_pin_memory phase=installed pid={os.getpid()} "
          "pin_memory=False offload_selection=unchanged; execution requires CPUOffloadPolicy construction",
          flush=True)
    return True
