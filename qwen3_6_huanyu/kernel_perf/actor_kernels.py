"""训练侧（HF/FSDP actor/ref）NPU 融合 kernel 安装器。

包装点与 ``qwen_precision/packed.py`` 相同——HF 原生支持的
``layer.linear_attn.chunk_gated_delta_rule`` 属性注入——但绑定的是
``kernel_perf.gdn_kernels.chunk_gated_delta_rule_npu``（fwd+bwd 全融合），
而不是 FP32 torch 参考守护。RMSNorm 走 ``kernel_perf.rms_norm``。

与 FP32 参考互斥：``MINICPMO_GDN_NPU_KERNEL=1`` 要求
``MINICPMO_HUANYU_GDN_FP32_TRAINING=0``（两者绑定同一属性，同开即 raise）。

模型无关：``install_gdn_kernels(language)`` / ``install_rms_norm(module)``
接受显式模块句柄；huanyu 接线在 ``install_actor_kernels(module)``。
"""

import os

import torch


def _enabled(name, default="0"):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _language_layers(module):
    language = getattr(getattr(module, "model", module), "language_model", None)
    if language is None:
        language = getattr(module, "model", module)
    layers = getattr(language, "layers", None)
    if layers is None:
        raise RuntimeError("GDN fused kernel 安装需要显式的 language model layers")
    return layers


def install_gdn_kernels(module) -> int:
    """把每层 linear_attn.chunk_gated_delta_rule 绑定为 NPU 融合实现。幂等。"""
    from .gdn_kernels import chunk_gated_delta_rule_npu

    bound = already = 0
    for layer in _language_layers(module):
        attention = getattr(layer, "linear_attn", None)
        if attention is None:
            continue
        kernel = getattr(attention, "chunk_gated_delta_rule", None)
        if getattr(kernel, "_huanyu_npu_kernel", False) is True:
            already += 1
            continue
        if kernel is not None and getattr(kernel, "_huanyu_fp32_guard", False) is True:
            raise RuntimeError(
                "GDN fused kernel 与 FP32 torch 参考守护互斥：该层已绑定 FP32 参考，"
                "请设置 MINICPMO_HUANYU_GDN_FP32_TRAINING=0"
            )
        attention.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        bound += 1
    if bound + already == 0:
        raise RuntimeError("GDN fused kernel 安装失败：未找到任何 linear_attn 层")
    return bound


def install_actor_kernels(module):
    """训练侧总入口：按 env 开关安装 GDN 融合 kernel 与 npu_rms_norm。"""
    gdn = _enabled("MINICPMO_GDN_NPU_KERNEL")
    rms = _enabled("MINICPMO_NPU_RMS_NORM")
    if not (gdn or rms):
        print("[huanyu-kernel] actor kernels disabled", flush=True)
        return module
    if gdn:
        if _enabled("MINICPMO_HUANYU_GDN_FP32_TRAINING"):
            raise ValueError(
                "MINICPMO_GDN_NPU_KERNEL 与 MINICPMO_HUANYU_GDN_FP32_TRAINING 互斥："
                "fused kernel 与 FP32 参考绑定同一属性，kernel 档案请将后者置 0"
            )
        bound = install_gdn_kernels(module)
        print(f"[huanyu-kernel] actor GDN fused kernel installed: new={bound}", flush=True)
    if rms:
        from .rms_norm import install_npu_rms_norm

        count = install_npu_rms_norm(module)
        if count == 0:
            raise RuntimeError("MINICPMO_NPU_RMS_NORM=1 但未匹配到任何 Qwen3_5Moe RMSNorm 实例")
        print(f"[huanyu-kernel] actor npu_rms_norm installed: new={count}", flush=True)
    return module
