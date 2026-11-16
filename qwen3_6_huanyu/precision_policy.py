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
# kernel 档案：NPU 原生/融合 kernel 替换 FP32 参考（与 precision_mode.sh 一致）。
KERNEL_ON = (
    "MINICPMO_GDN_NPU_KERNEL", "MINICPMO_NPU_RMS_NORM",
    "MINICPMO_HUANYU_LAYERWISE_RELOAD", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE",
    "MINICPMO_ISOLATE_MEDIA_ENCODERS",
)
KERNEL_OFF = (
    "MINICPMO_HUANYU_GDN_FP32_TRAINING", "MINICPMO_GDN_FP32_REFERENCE",
    "MINICPMO_GDN_CACHE_BARRIER", "MINICPMO_GDN_PREFILL_REFERENCE",
    "MINICPMO_GDN_L2NORM_REFERENCE", "MINICPMO_GDN_CONV_REFERENCE",
    "MINICPMO_GDN_GATING_REFERENCE", "MINICPMO_STABLE_LINEAR",
    "MINICPMO_PRIVATE_HYBRID_CACHE",
)
# 诊断二分：kernel 档案默认同时开融合 GDN 与 npu_rms_norm，单独回退一个时
# 期望值随声明变化（见 examples/qwen3_6_huanyu/precision_mode.sh）。
KERNEL_BISECT = {
    "none": {},
    "gdn_off": {"MINICPMO_GDN_NPU_KERNEL": "0", "MINICPMO_HUANYU_GDN_FP32_TRAINING": "1"},
    "rms_off": {"MINICPMO_NPU_RMS_NORM": "0"},
}
_REPORTED = set()


def report_policy(role):
    # Direct Python integrations retain custom mode; all supported launchers
    # select full by default and forward its exact flags through runtime_env.
    profile = os.environ.get("HUANYU_PRECISION_PROFILE", "custom")
    if profile not in {"full", "custom", "kernel", "kernel_graph"}:
        raise ValueError(f"Unknown HUANYU_PRECISION_PROFILE={profile!r}; choose full, custom, kernel or kernel_graph")
    expected = dict.fromkeys(FULL_ON, "1")
    expected["MINICPMO_FSDP2_PIN_MEMORY"] = "0"
    if profile == "full":
        mismatches = {key: os.environ.get(key, "<unset>") for key, value in expected.items()
                      if os.environ.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Huanyu full precision policy was not propagated to {role}: {mismatches}")
        if os.environ.get("HUANYU_GRAPH_MODE", "0").lower() not in {"0", "false", "off", "no"}:
            raise RuntimeError("Huanyu full precision references require eager mode (HUANYU_GRAPH_MODE=0)")
    if profile in {"kernel", "kernel_graph"}:
        kernel_expected = dict.fromkeys(KERNEL_ON, "1")
        kernel_expected.update(dict.fromkeys(KERNEL_OFF, "0"))
        kernel_expected["MINICPMO_FSDP2_PIN_MEMORY"] = "0"
        # 诊断二分（precision_mode.sh 的 HUANYU_KERNEL_BISECT）：单独回退一个
        # 融合 kernel。断言不被绕过，而是改成断言声明的那一组值。
        bisect = os.environ.get("HUANYU_KERNEL_BISECT", "none")
        if bisect not in KERNEL_BISECT:
            raise ValueError(
                f"Unknown HUANYU_KERNEL_BISECT={bisect!r}; choose {sorted(KERNEL_BISECT)}")
        kernel_expected.update(KERNEL_BISECT[bisect])
        mismatches = {key: os.environ.get(key, "<unset>") for key, value in kernel_expected.items()
                      if os.environ.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Huanyu {profile} precision policy was not propagated to {role}: {mismatches}")
        graph_requested = os.environ.get("HUANYU_GRAPH_MODE", "0").lower() in {"1", "true", "on", "yes"}
        if profile == "kernel" and graph_requested:
            raise RuntimeError("Huanyu kernel precision requires eager mode (HUANYU_GRAPH_MODE=0; use kernel_graph for aclgraph)")
        if profile == "kernel_graph" and not graph_requested:
            raise RuntimeError("Huanyu kernel_graph precision requires graph mode (HUANYU_GRAPH_MODE=1; use kernel for eager)")
    snapshot = tuple((key, os.environ.get(key, "<unset>")) for key in expected)
    if profile in {"kernel", "kernel_graph"}:
        snapshot = tuple((key, os.environ.get(key, "<unset>"))
                         for key in tuple(KERNEL_ON) + tuple(KERNEL_OFF)
                         + ("MINICPMO_FSDP2_PIN_MEMORY", "HUANYU_KERNEL_BISECT"))
    marker = (role, profile, snapshot)
    if marker not in _REPORTED:
        _REPORTED.add(marker)
        print(f"[huanyu-precision] phase=configured role={role} profile={profile} "
              f"host={socket.gethostname()} pid={os.getpid()} source={__file__} "
              + " ".join(f"{key}={value}" for key, value in snapshot), flush=True)
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
