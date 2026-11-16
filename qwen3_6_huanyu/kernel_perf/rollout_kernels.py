"""Server（vLLM-Omni 推理 worker）NPU 原生 GDN kernel 安装器。

server 的原生路径本身即 fused kernel（vllm_ascend 的
``AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused`` /
``_C_ascend.npu_recurrent_gated_delta_rule`` / ``DeviceOperator.fused_gdn_gating``
/ ``l2norm_fwd`` / ``npu_causal_conv1d_custom``）。kernel 档案的职责是在我们
拥有控制权的包装点上：

1. 安装期校验 env 互斥（FP32 参考 flags 必须全 0，同开即 fail-fast）；
2. 原生模型初始化后断言上述原生算子齐备且未被 FP32 参考装回；
3. 断言 recurrent cache 为 BF16（910_93 原生 decode kernel 的硬约束，
   由 engine kwarg ``mamba_ssm_cache_dtype=bfloat16`` 提供）。

本模块不改任何上游 ``__init__.py``；由 ``engine_hooks.py`` 显式调用。
"""

import os
from functools import wraps
from importlib import import_module

import torch

_NATIVE = "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni"

# kernel 档案要求全 0 的 FP32 参考开关（与 fused kernel 互斥）。
_REFERENCE_FLAGS = (
    "MINICPMO_GDN_FP32_REFERENCE",
    "MINICPMO_GDN_PREFILL_REFERENCE",
    "MINICPMO_GDN_L2NORM_REFERENCE",
    "MINICPMO_GDN_CONV_REFERENCE",
    "MINICPMO_GDN_GATING_REFERENCE",
    "MINICPMO_GDN_CACHE_BARRIER",
    "MINICPMO_HF_NORM_REFERENCE",
)


def _enabled(name, default="0"):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _validate_kernel_env():
    """MINICPMO_GDN_NPU_KERNEL=1 时校验互斥 flags；未开 kernel 则直接跳过。"""
    if not _enabled("MINICPMO_GDN_NPU_KERNEL"):
        return False
    conflicts = [name for name in _REFERENCE_FLAGS if _enabled(name)]
    if conflicts:
        raise ValueError(
            f"MINICPMO_GDN_NPU_KERNEL 与 FP32 参考实现互斥，请置 0: {conflicts}"
        )
    return True


def _is_huanyu_config(config):
    from ..rollout_precision import is_huanyu_config

    try:
        return is_huanyu_config(config)
    except Exception:
        return False


def _check_native_instance(self, config):
    """原生初始化后的断言：只读检查，不吃异常。"""
    from .gdn_kernels import assert_native_rollout_gdn

    if not _is_huanyu_config(config):
        return False
    availability = assert_native_rollout_gdn()
    ops = torch.ops._C_ascend
    recurrent = ops.npu_recurrent_gated_delta_rule
    if getattr(recurrent, "_huanyu_fp32_state", False):
        raise RuntimeError(
            "kernel 档案要求原生 BF16 recurrent decode，但检测到 FP32 参考已包装 "
            "npu_recurrent_gated_delta_rule（MINICPMO_GDN_FP32_REFERENCE=1？）"
        )
    cache_dtype = getattr(getattr(config, "cache_config", None), "mamba_ssm_cache_dtype", None)
    if cache_dtype not in (None, "bfloat16"):
        raise RuntimeError(
            f"kernel 档案要求 mamba_ssm_cache_dtype=bfloat16（910_93 原生 decode kernel "
            f"仅接受 BF16 state），当前为 {cache_dtype!r}；请通过 rollout engine_kwargs "
            "设置 mamba_ssm_cache_dtype=bfloat16"
        )
    if os.environ.get("HUANYU_GRAPH_MODE", "0").lower() in {"1", "true", "on", "yes"}:
        compilation = getattr(config, "compilation_config", None)
        mode = getattr(compilation, "cudagraph_mode", None)
        mode_name = getattr(mode, "name", str(mode))
        if mode_name in (None, "NONE", "None"):
            raise RuntimeError(
                "HUANYU_GRAPH_MODE=1 请求 decode 图捕获，但 compilation_config."
                f"cudagraph_mode={mode_name!r}（静默回退 eager）。检查 enforce_eager/"
                "vllm-ascend 的编译模式解析。"
            )
        print(f"[huanyu-kernel] graph mode requested; resolved cudagraph_mode={mode_name}",
              flush=True)
    print(f"[huanyu-kernel] rollout native GDN verified: {availability} "
          f"cache_dtype={cache_dtype}", flush=True)
    return True


def install_huanyu_rollout_kernels():
    """在原生模型 ``__init__`` 上挂 kernel 档案校验（仅 Huanyu checkpoint 生效）。"""
    if not _validate_kernel_env():
        print("[huanyu-kernel] rollout kernels disabled", flush=True)
        return False
    native_cls = import_module(_NATIVE).MiniCPMO50OmniForConditionalGeneration
    original = native_cls.__init__
    if getattr(original, "_huanyu_kernel_init", False):
        return True

    @wraps(original)
    def init(self, *args, **kwargs):
        config = kwargs.get("vllm_config", kwargs.get("config", args[0] if args else None))
        original(self, *args, **kwargs)
        if not _is_huanyu_config(config):
            config = getattr(self, "vllm_config", getattr(self, "config", config))
        _check_native_instance(self, config)

    init._huanyu_kernel_init = True
    native_cls.__init__ = init
    print("[huanyu-kernel] rollout native GDN init hook installed (Huanyu checkpoint only)", flush=True)
    return True
