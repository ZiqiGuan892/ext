"""NPU GDN kernel（训练侧纯 torch 向量化实现 / 推理侧原生算子探测），模型无关可复用。

训练侧：``torch_chunk_gated_delta_rule_fast`` 与 HF
``torch_chunk_gated_delta_rule`` 签名/布局完全一致，直接绑定到
``layer.linear_attn.chunk_gated_delta_rule``（HF 原生支持该属性注入）。
全部使用 torch 内置算子（bmm / cumsum / exp / 分块倍增求逆 + gdn_glue
的安全 triton l2norm），bf16 计算 + FP32 state 累积，反向除 WY 求逆与
l2norm 走闭式自定义 Function 外**由 autograd 自动给出**——不依赖任何
自定义 AscendC 算子（fla_npu v26.1.0 的
npu_chunk_gated_delta_rule_fwd_h 存在值相关 AICore 越界 bug，已弃用该
路径；vendored triton 的 tl.make_block_ptr 在本机 triton-ascend 3.5.0
越界访存，同样弃用，详见 dev_log.md）。

推理侧：``assert_native_rollout_gdn`` 只读探测 vllm_ascend 原生 GDN 算子
是否齐备，不做任何包装（server 的原生路径本身即 fused kernel）。

CPU import 契约：模块级只 import torch/typing；triton 与 torch_npu 相关
导入全部延迟到函数内，CPU 环境 import 本模块无副作用。
"""

from typing import Optional

import torch
import torch.nn.functional as F


def torch_chunk_gated_delta_rule_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """HF ``torch_chunk_gated_delta_rule`` 的向量化 NPU 替代（训练侧绑定入口）。

    布局与 HF 完全一致：query/key/value 为 [B, T, H, D]（调用方已完成 GQA
    head 扩展），g/beta 为 [B, T, H]，initial_state 为 [B, H, Dk, Dv]；
    返回 (o [B, T, H, Dv], final_state [B, H, Dk, Dv] 或 None)。

    与 HF 参考的差异（精度单测逐 kernel 量化，报告误差与 Pearson）：
    - WY 表示（A、kkt、求逆、w/u）在 FP32 域计算（外层 autocast 显式关闭，
      否则这些 matmul 会被降回 BF16），求逆用 gdn_glue 的分块倍增
      （替代 HF 的 63 次前替换循环），反向为闭式解；
    - chunk 递推 scan 的 matmul 用输入 dtype（bf16），state 以 FP32 累积；
      exp/decay 保持 FP32。

    padded 训练（use_remove_padding=false）下 cu_seqlens 必须为 None；T 非
    chunk_size 整数倍时与 HF 参考同样在序列尾补零（零 padding 位状态直通、
    输出为零）。varlen 显式 raise，不静默降级。
    """
    if g is None or beta is None:
        raise ValueError("GDN kernel requires explicit g and beta")
    if cu_seqlens is not None:
        raise NotImplementedError(
            "torch GDN kernel 仅支持 uniform padded 输入（cu_seqlens=None）；"
            "Huanyu 训练要求 use_remove_padding=false，packed 序列请用 full 精度档案"
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"GDN kernel requires bf16/fp16 projections, got {query.dtype}; "
            "FP32 输入请使用 full 精度档案的 HF 参考实现"
        )
    from .gdn_glue import (
        chunk_local_cumsum,
        chunk_scaled_dot_kkt_fwd,
        l2norm_apply,
        solve_tril_npu,
    )

    batch, tokens, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    compute_dtype = query.dtype

    # FSDP 的 forward_step 把 actor 前向整个包在 BF16 autocast 里
    # （transformer_impl.py 的 autocast_ctx），而 autocast 只看算子不看调用方
    # 显式写的 .float()——WY 表示那几个 matmul 会被静默降回 BF16。这里显式
    # 关掉，与 qwen_precision/packed.py 的 FP32 参考守护、kernel_perf/rms_norm
    # 的自定义 Function 同约定。bf16 计算段靠下面的 compute_dtype 显式 cast
    # 维持，不受影响。
    with torch.autocast(device_type=query.device.type, enabled=False):
        # 1) q/k l2 归一化（HF eps=1e-6，先于 transpose/scale；可微分 triton 版）
        if use_qk_l2norm_in_kernel:
            query = l2norm_apply(query)
            key = l2norm_apply(key)

        # 2) T 补零到 chunk_size 整数倍（与 HF 参考同语义）
        pad = (chunk_size - tokens % chunk_size) % chunk_size
        if pad:
            query = F.pad(query, (0, 0, 0, 0, 0, pad))
            key = F.pad(key, (0, 0, 0, 0, 0, pad))
            value = F.pad(value, (0, 0, 0, 0, 0, pad))
            g = F.pad(g, (0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, pad))
        tokens_padded = tokens + pad
        n_chunks = tokens_padded // chunk_size

        # 3) 转 [B,H,T,D]；g chunk 内累加（FP32）
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        g_cumsum = chunk_local_cumsum(g.float(), chunk_size)

        # 4) WY 表示（FP32）：A = tril_{-1}(β·(KK^T)·exp(gi-gj))，Ai = (I+A)^{-1}
        A = chunk_scaled_dot_kkt_fwd(k, g_cumsum, beta, chunk_size=chunk_size,
                                     output_dtype=torch.float32)
        A_inv = solve_tril_npu(A, output_dtype=torch.float32)

        # 5) w = Ai @ (k·β·exp(g))，u = Ai @ (v·β)（FP32 bmm，公式与
        #    recompute_w_u_fwd 的 CPU golden 逐位核对一致）
        kc = k.float().reshape(batch, heads, n_chunks, chunk_size, key_dim)
        vc = v.float().reshape(batch, heads, n_chunks, chunk_size, value_dim)
        gc = g_cumsum.transpose(1, 2).reshape(batch, heads, n_chunks, chunk_size)
        bc = beta.float().transpose(1, 2).reshape(batch, heads, n_chunks, chunk_size)
        Aic = A_inv.transpose(1, 2).reshape(batch, heads, n_chunks, chunk_size, chunk_size)
        w = Aic @ (kc * (bc * gc.exp()).unsqueeze(-1))
        u = Aic @ (vc * bc.unsqueeze(-1))

        # 6) chunk 递推 scan：循环外把与 state 无关的量全部 batched 预算（一次
        #    bmm/elementwise 覆盖所有 chunk），循环内每 chunk 只剩 3 个 bf16
        #    matmul + 1 个 FP32 state matmul + 1 次 state cast。
        qc = q.reshape(batch, heads, n_chunks, chunk_size, key_dim)
        # 与 HF 同序：先 tril 掩掉上三角差值再 exp（否则 exp(+110)=inf，反向 0*inf=NaN）
        decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).tril().exp().tril()  # 含对角（=1）
        exp_g = gc.exp()
        # exp(g_last - g)：gc 单调递减故差值 ≤ 0、exp ≤ 1（误写成 exp(g_last)-g
        # 会在深度衰减时得到 exp(+大数)=inf）。
        exp_last_minus_g = (gc[..., -1:] - gc).exp()
        scale = key_dim ** -0.5
        # HF 对 q 的所有用途统一施加 Dk^-0.5 缩放（含 state 读出项）。
        q_scaled = (qc * scale).to(compute_dtype)
        q_exp = (qc.float() * (scale * exp_g).unsqueeze(-1)).to(compute_dtype)
        # intra-chunk attention 与 state 无关：全部 chunk 一次 bmm + 一次 decay 乘
        attn_all = (q_scaled @ kc.to(compute_dtype).transpose(-1, -2)).float() * decay
        attn_all = attn_all.to(compute_dtype)
        # state 更新项的 k·exp(g_last-g) 同样循环外预算（FP32，供 FP32 累积）
        k_decay = kc * exp_last_minus_g.unsqueeze(-1)
        w = w.to(compute_dtype)
        u = u.to(compute_dtype)
        if initial_state is not None:
            state = initial_state.float().contiguous()
        else:
            state = torch.zeros(batch, heads, key_dim, value_dim,
                                dtype=torch.float32, device=query.device)
        out = torch.empty(batch, heads, n_chunks, chunk_size, value_dim,
                          dtype=compute_dtype, device=query.device)
        for i in range(n_chunks):
            state_bf = state.to(compute_dtype)
            # v_new = u - w @ S
            v_new_i = u[:, :, i] - w[:, :, i] @ state_bf
            # o = (q·exp(g)) @ S + attn @ v_new
            out[:, :, i] = q_exp[:, :, i] @ state_bf + attn_all[:, :, i] @ v_new_i
            # S = S·exp(g_last) + (k·exp(g_last - g))^T @ v_new（FP32 累积）
            state = state * exp_g[:, :, i, -1, None, None] + (
                k_decay[:, :, i].transpose(-1, -2) @ v_new_i.float()
            )

        out = out.reshape(batch, heads, tokens_padded, value_dim).transpose(1, 2)[:, :tokens]
        final_state = state if output_final_state else None
        return out, final_state


# HF 绑定入口（precision._gdn_coverage 与单测据此识别 fused kernel 绑定）。
chunk_gated_delta_rule_npu = torch_chunk_gated_delta_rule_fast
chunk_gated_delta_rule_npu._huanyu_npu_kernel = True


def assert_native_rollout_gdn() -> dict:
    """探测 server（vllm_ascend）原生 GDN 算子是否齐备，缺失即 fail-fast。

    只读探测、不包装：server 的原生路径本身即 fused kernel，kernel 档案只需
    确认 FP32 参考实现没有被装回这些位置。返回各算子可用性字典。
    """
    availability: dict[str, bool] = {}
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("kernel 档案需要 torch_npu（NPU 环境）") from exc
    ops = torch.ops._C_ascend
    availability["recurrent_decode"] = hasattr(ops, "npu_recurrent_gated_delta_rule")
    availability["causal_conv1d"] = hasattr(ops, "npu_causal_conv1d_custom")
    import torch_npu as _torch_npu

    availability["chunk_prefill"] = hasattr(_torch_npu, "npu_chunk_gated_delta_rule")
    try:
        from vllm_ascend.device.device_op import DeviceOperator

        availability["fused_gdn_gating"] = callable(getattr(DeviceOperator, "fused_gdn_gating", None))
    except ImportError:
        availability["fused_gdn_gating"] = False
    try:
        from vllm_ascend.ops import gdn as _ascend_gdn

        availability["l2norm"] = callable(getattr(_ascend_gdn, "l2norm_fwd", None))
    except ImportError:
        availability["l2norm"] = False
    missing = [name for name, ok in availability.items() if not ok]
    if missing:
        raise RuntimeError(
            f"kernel 档案要求 vllm_ascend 原生 GDN 算子齐备，缺失: {missing}；"
            "请确认 _C_ascend 自定义算子已完成注册（原生模型初始化后重试）"
        )
    return availability
