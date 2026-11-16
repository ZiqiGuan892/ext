"""训练侧融合 MoE kernel：整体替换 HF ``Qwen3_5MoeSparseMoeBlock.forward``。

动机：HF 参考实现逐专家 Python 循环（`Qwen3_5MoeExperts.forward`，每层
256 专家逐个 F.linear + index_add），在 40 层 × top-8 下是 trainer 侧
GEMM 碎片的主要来源。本模块替换为 NPU 融合链路（参考 MindSpeed 的
``te/pytorch/permutation.py`` 与 ``lite/ops/grouped_matmul.py``，但全部
vendored 自写、零 megatron/mindspeed 依赖）：

    routing_map → npu_moe_token_permute_with_routing_map
    → npu_grouped_matmul(gate_up) → npu_swiglu → npu_grouped_matmul(down)
    → npu_moe_token_unpermute_with_routing_map（probs 在 op 内施加，梯度回流 router）

R3 路由回放内建于此：``HUANYU_EXPERT_REPLAY=1`` 时按 controller 目标替换
top-k 索引，gating 权重由训练侧 router logits 的 softmax 在记录索引上
gather + 归一化（论文 eq.8，梯度保留）；无记录行（重复/全零行）回退原生
路由。replay 仅在本 kernel 开启时有效。

router 数学保持 HF 精确（fp32 softmax → topk → 归一化），原生路由决策与
HF 逐位一致；shared expert 保留 HF 原路径。

env：
- ``MINICPMO_MOE_NPU_KERNEL=1`` 启用；与 ``MINICPMO_MOE_MATCH_ROLLOUT=1``
  （FP32 专家参考）互斥。
- ``HUANYU_EXPERT_REPLAY=1`` 要求本开关已开，否则 install 直接 raise。
"""

import os
from types import MethodType

import torch
import torch.nn.functional as F

from . import controller

_FLAG = "_huanyu_fused_moe"
_LAYER_ATTR = "_huanyu_moe_layer_idx"


def _enabled(name, default="0"):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def replay_requested() -> bool:
    return _enabled("HUANYU_EXPERT_REPLAY")


def kernel_requested() -> bool:
    return _enabled("MINICPMO_MOE_NPU_KERNEL")


def _torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "MINICPMO_MOE_NPU_KERNEL requires torch_npu (Ascend training worker); "
            "set MINICPMO_MOE_NPU_KERNEL=0 on non-NPU hosts"
        ) from exc
    return torch_npu


class _GroupedMatmul(torch.autograd.Function):
    """3D 堆叠权重的 grouped matmul（fwd + dx + dw K-split）。

    参考 MindSpeed ``lite/ops/grouped_matmul.py`` 的 _GroupedMatmul，扩展为
    单个 [E, K, N] 权重张量接口。weight 以 transpose view 传入，dw 经
    autograd 的 transpose 映射回原始参数存储。
    """

    @staticmethod
    def forward(ctx, x, weight, group_list):
        # x: [sum_M, K]；weight: [E, K, N]；group_list: 每专家 token 数（int64, NPU）
        torch_npu = _torch_npu()
        output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        ctx.save_for_backward(x, weight, group_list)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        torch_npu = _torch_npu()
        x, weight, group_list = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # dx = grad @ weight^T（逐专家 [E, N, K]）
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_output], [weight.transpose(1, 2)], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        # dw = x^T @ grad（K-split GMM → [E, K, N]，与 weight view 同形）
        grad_w = torch_npu.npu_grouped_matmul(
            [x.t()], [grad_output], bias=None, group_list=group_list,
            split_item=3, group_type=2, group_list_type=1,
        )[0]
        return grad_x, grad_w, None


class _MoePermute(torch.autograd.Function):
    """routing_map 变体 token permute（probs 不在此施加，留待 unpermute）。

    参考 MindSpeed ``te/pytorch/permutation.py`` 的 MoePermuteMaskMap。
    """

    @staticmethod
    def forward(ctx, tokens, routing_map, num_out_tokens):
        torch_npu = _torch_npu()
        permuted, _, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
            tokens, routing_map, probs=None, num_out_tokens=num_out_tokens, drop_and_pad=False,
        )
        ctx.save_for_backward(sorted_indices, routing_map)
        ctx.num_tokens = tokens.shape[0]
        return permuted, sorted_indices

    @staticmethod
    def backward(ctx, permuted_act_grad, _sorted_indices_grad):
        torch_npu = _torch_npu()
        sorted_indices, routing_map = ctx.saved_tensors
        act_grad, _ = torch_npu.npu_moe_token_permute_with_routing_map_grad(
            permuted_act_grad.contiguous(), None, sorted_indices, routing_map,
            routing_map.shape[1], ctx.num_tokens, False,
        )
        return act_grad, None, None


class _MoeUnpermute(torch.autograd.Function):
    """routing_map 变体 token unpermute，probs 在 op 内施加且梯度回流。

    参考 MindSpeed ``te/pytorch/permutation.py`` 的 MoeUnpermuteMaskMap
    （非 drop_and_pad 分支）。
    """

    @staticmethod
    def forward(ctx, permuted_tokens, sorted_indices, restore_shape, probs, routing_map):
        torch_npu = _torch_npu()
        unpermuted, _, _, _ = torch_npu._npu_moe_token_unpermute_with_routing_map(
            permuted_tokens, sorted_indices, restore_shape,
            probs=probs, routing_map=routing_map, drop_and_pad=False,
        )
        ctx.save_for_backward(permuted_tokens, probs, sorted_indices, routing_map)
        ctx.restore_shape = restore_shape
        return unpermuted

    @staticmethod
    def backward(ctx, unpermuted_grad):
        torch_npu = _torch_npu()
        permuted_tokens, probs, sorted_indices, routing_map = ctx.saved_tensors
        act_grad, probs_grad = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
            unpermuted_grad.contiguous(), sorted_indices, sorted_indices,
            routing_map=routing_map, permuted_tokens=permuted_tokens, probs=probs,
            drop_and_pad=False, restore_shape=ctx.restore_shape,
        )
        return act_grad, None, None, probs_grad, None


def _substitute_replay(block, router_logits, native_weights, native_indices):
    """R3 核心：用记录的 top-k 索引替换原生路由，权重 gather 自训练侧 softmax。

    论文 eq.8：g_replay,i = I_infer,i * exp(s_train,i) / Σ_j I_infer,j * exp(s_train,j)，
    即 mask 来自推理引擎、softmax 权重来自训练 logits —— 梯度照常回流 router。
    无记录行（重复值行，含全零）回退原生 (indices, weights)。
    """
    target, fallback = controller.layer_target(getattr(block, _LAYER_ATTR))  # fallback [T,1] True → 原生
    if target.shape != native_indices.shape:
        raise RuntimeError(
            f"expert_replay REPLAY: layer {getattr(block, _LAYER_ATTR)} target "
            f"shape {tuple(target.shape)} != router indices {tuple(native_indices.shape)}; "
            "layout.build_padded_targets 必须与 padded input_ids 的 [B,S]→[B*S] 行主序对齐"
        )
    debug = os.environ.get("HUANYU_REPLAY_DEBUG", "0") == "1"
    valid = ~fallback
    if debug:
        # baseline 一致率（论文 §3.2 的 router-level 指标）：有记录行中**原生**
        # top-k 集合与推理记录完全一致的比例，即 R3 要消除的那部分偏差，本就
        # 不该是 100%。每次 forward 每层一次同步，仅调试用。
        agree = _set_agreement(native_indices, target) & valid.squeeze(1)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    gathered = router_probs.gather(1, target)
    gathered = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    replay_weights = gathered.to(native_weights.dtype)
    indices = torch.where(fallback, native_indices, target)
    weights = torch.where(fallback, native_weights, replay_weights)
    if debug:
        # replayed 一致率：**实际送进 GMM 的** indices 与记录的一致比例。构造上
        # 必然 100%，但这是「replay 真的生效了」的唯一直接观测量（baseline 那个
        # 数字无论 replay 开关都一样），所以 debug 档显式测出来而不是靠推理。
        replayed = int((_set_agreement(indices, target) & valid.squeeze(1)).sum())
        total = int(valid.sum())
        layer = getattr(block, _LAYER_ATTR)
        print(f"[huanyu-replay-debug] layer={layer} rows={target.shape[0]} valid={total} "
              f"agreement={int(agree.sum())}/{total} replayed={replayed}/{total}", flush=True)
        if replayed != total:
            raise RuntimeError(
                f"expert_replay REPLAY: layer {layer} 回放后仍有 {total - replayed}/{total} "
                "行的 top-k 集合与记录不一致——replay 未生效或 target 被污染"
            )
    return indices, weights


def _set_agreement(indices, target):
    """逐行判断两组 top-k 的**集合**是否完全一致（顺序无关），返回 [T] bool。"""
    return (indices.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1).all(dim=-1)


_FINITE_PRINTS = {"left": 400}


def _finite_report(block, stages):
    """HUANYU_MOE_FINITE_DEBUG=1：逐层报告各阶段非有限行数（定位 NaN 首现位置，调试用）。"""
    if _FINITE_PRINTS["left"] <= 0:
        return
    counts = {name: int((~torch.isfinite(t.float())).reshape(t.shape[0], -1).any(-1).sum()) for name, t in stages}
    layer = getattr(block, _LAYER_ATTR, -1)
    if layer == 0 or any(counts.values()):
        _FINITE_PRINTS["left"] -= 1
        print(f"[huanyu-moe-finite] layer={layer} rows={stages[0][1].shape[0]} nonfinite={counts} "
              f"in_absmax={float(stages[0][1].detach().float().abs().max()):.4g} pid={os.getpid()}", flush=True)


def _fused_moe_forward(block, hidden_states):
    """替换 ``Qwen3_5MoeSparseMoeBlock.forward``；签名与返回保持一致。"""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden2d = hidden_states.reshape(-1, hidden_dim)
    shared_output = block.shared_expert(hidden2d)
    router_logits, routing_weights, selected_experts = block.gate(hidden2d)
    if controller.is_armed():
        selected_experts, routing_weights = _substitute_replay(
            block, router_logits, routing_weights, selected_experts)

    num_tokens, top_k = selected_experts.shape
    experts = block.experts
    num_experts = experts.num_experts
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device=hidden2d.device)
    routing_map.scatter_(1, selected_experts, True)
    # out-of-place scatter：保证 routing_weights → probs_map 的 autograd 边（router 梯度）。
    probs_map = torch.zeros(num_tokens, num_experts, dtype=routing_weights.dtype, device=hidden2d.device)
    probs_map = probs_map.scatter(1, selected_experts, routing_weights)

    permuted, sorted_indices = _MoePermute.apply(hidden2d, routing_map, num_tokens * top_k)
    tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int64)
    gate_up = _GroupedMatmul.apply(permuted, experts.gate_up_proj.transpose(1, 2), tokens_per_expert)
    activated = _torch_npu().npu_swiglu(gate_up, dim=-1)
    down = _GroupedMatmul.apply(activated, experts.down_proj.transpose(1, 2), tokens_per_expert)
    combined = _MoeUnpermute.apply(down, sorted_indices, (num_tokens, hidden_dim), probs_map, routing_map)

    shared_output = F.sigmoid(block.shared_expert_gate(hidden2d)) * shared_output
    output = combined.to(shared_output.dtype) + shared_output
    if os.environ.get("HUANYU_MOE_FINITE_DEBUG", "0") == "1":
        _finite_report(block, [("input", hidden2d), ("routing_weights", routing_weights),
                               ("permuted", permuted), ("gate_up", gate_up), ("activated", activated),
                               ("down", down), ("combined", combined), ("shared", shared_output)])
    return output.reshape(batch_size, sequence_length, hidden_dim)


def _language_layers(module):
    language = getattr(getattr(module, "model", module), "language_model", None)
    if language is None:
        language = getattr(module, "model", module)
    layers = getattr(language, "layers", None)
    if layers is None:
        raise RuntimeError("fused MoE kernel 安装需要显式的 language model layers")
    return layers


def install_fused_moe(module) -> int:
    """实例级替换每个 MoE block 的 forward；幂等；返回新安装层数。"""
    if _enabled("MINICPMO_MOE_MATCH_ROLLOUT"):
        raise ValueError(
            "MINICPMO_MOE_NPU_KERNEL 与 MINICPMO_MOE_MATCH_ROLLOUT 互斥："
            "FP32 专家参考与融合 kernel 替换同一 forward"
        )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    layers = list(_language_layers(module))
    bound = already = 0
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, Qwen3_5MoeSparseMoeBlock):
            continue
        if getattr(mlp, _FLAG, False):
            already += 1
            continue
        setattr(mlp, _LAYER_ATTR, layer_idx)
        mlp.forward = MethodType(_fused_moe_forward, mlp)
        setattr(mlp, _FLAG, True)
        bound += 1
    if bound + already == 0:
        raise RuntimeError("MINICPMO_MOE_NPU_KERNEL=1 但未匹配到任何 Qwen3_5MoeSparseMoeBlock")
    # R3 层序契约：vLLM 按绝对 decoder 层号填充 routed_experts，要求每个
    # decoder 层恰好一个 MoE block；出现 dense 层时层号错位会静默误训
    # （纯 kernel 无此约束，但 replay 必须 fail-closed）。
    if replay_requested() and bound + already != len(layers):
        raise RuntimeError(
            f"HUANYU_EXPERT_REPLAY 要求每个 decoder 层都是 MoE（对齐 vLLM 捕获的绝对层号），"
            f"实际 {bound + already}/{len(layers)}；该模型不支持 R3 路由回放"
        )
    print(f"[huanyu-kernel] actor fused MoE installed: new={bound} total={bound + already} "
          f"replay={'on' if replay_requested() else 'off'} pid={os.getpid()}", flush=True)
    return bound


def install_huanyu_expert_replay(module):
    """训练侧总入口：按 env 安装融合 MoE kernel 与 R3 replay。configure_model 调用。"""
    kernel = kernel_requested()
    replay = replay_requested()
    if replay and not kernel:
        raise ValueError(
            "HUANYU_EXPERT_REPLAY=1 要求 MINICPMO_MOE_NPU_KERNEL=1："
            "路由回放仅内建于融合 MoE kernel（HF 参考路径不提供 replay）"
        )
    if not kernel:
        print("[huanyu-kernel] fused MoE disabled (MINICPMO_MOE_NPU_KERNEL=0)", flush=True)
        return module
    _torch_npu()  # 早失败：非 NPU 环境在 configure_model 即报错
    install_fused_moe(module)
    if replay:
        from .engine_hook import install_engine_hook

        install_engine_hook(module)
    return module
