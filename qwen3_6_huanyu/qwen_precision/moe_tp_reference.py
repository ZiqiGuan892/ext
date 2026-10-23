"""Opt-in actor reference for the rollout's TP4 MoE rounding boundaries.

This retains trainable full weights and uses the installed differentiable NPU
GMM. It emulates measured four-rank BF16 trees in default HCCL and strict AIV;
it is a TP4 diagnostic without expert parallelism, not an HCCL replacement.
"""
import os
from types import MethodType

import torch
import torch.nn.functional as F


def reduce_tp4(parts):
    mode = os.environ.get('HCCL_OP_EXPANSION_MODE')
    deterministic = os.environ.get('HCCL_DETERMINISTIC')
    if mode == 'AIV' and deterministic == 'strict':
        return (parts[0] + parts[2]) + (parts[1] + parts[3])
    if not mode and not deterministic:
        return (parts[0] + parts[1]) + (parts[2] + parts[3])
    raise ValueError('TP4 reference requires measured default HCCL or AIV with HCCL_DETERMINISTIC=strict')


def tp4_forward(self, hidden_states):
    import torch_npu
    from verl.models.transformers.npu_patch import NPUGmmFunction

    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    _, weights, ids = self.gate(x)
    permuted, rows = torch_npu.npu_moe_token_permute(x, ids.to(torch.int32))
    experts = self.experts
    counts = torch.histc(ids, bins=experts.num_experts, min=0, max=experts.num_experts)
    width = experts.intermediate_dim
    shared = self.shared_expert
    shared_width = shared.gate_proj.weight.shape[0]
    if width % 4 or shared_width % 4:
        raise ValueError('TP4 MoE reference requires expert widths divisible by four')
    shared_gate = torch.sigmoid(self.shared_expert_gate(x))
    parts = []
    for rank in range(4):
        start, end = rank * (width // 4), (rank + 1) * (width // 4)
        w13 = torch.cat((experts.gate_up_proj[:, start:end],
                        experts.gate_up_proj[:, width + start:width + end]), 1)
        gu = NPUGmmFunction.apply(permuted, w13.transpose(1, 2).contiguous(), counts)
        activated = torch_npu.npu_swiglu(gu, dim=-1)
        down = experts.down_proj[:, :, start:end].transpose(1, 2).contiguous()
        values = NPUGmmFunction.apply(activated, down, counts)
        routed = torch_npu.npu_moe_token_unpermute(values.to(weights.dtype), rows, probs=weights)
        start, end = rank * (shared_width // 4), (rank + 1) * (shared_width // 4)
        shared_gu = F.linear(x, torch.cat((shared.gate_proj.weight[start:end],
                                          shared.up_proj.weight[start:end]), 0))
        shared_act = torch_npu.npu_swiglu(shared_gu, dim=-1)
        shared_value = F.linear(shared_act, shared.down_proj.weight[:, start:end])
        # The native runner adds the gated shared partial before all-reduce.
        parts.append(routed + shared_gate * shared_value)
    output = reduce_tp4(parts)
    return output.reshape(shape)


def install_actor(model):
    setting = os.environ.get('MINICPMO_MOE_TP_REFERENCE', '0')
    if setting == '0':
        return 0
    if setting != '4':
        raise ValueError('MINICPMO_MOE_TP_REFERENCE supports only 0 or 4')
    mode = os.environ.get('HCCL_OP_EXPANSION_MODE')
    deterministic = os.environ.get('HCCL_DETERMINISTIC')
    if not ((mode == 'AIV' and deterministic == 'strict') or (not mode and not deterministic)):
        raise ValueError('TP4 reference requires measured default HCCL or strict AIV')
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    count = 0
    for child in model.modules():
        if isinstance(child, Qwen3_5MoeSparseMoeBlock) and not getattr(child, '_q36_moe_tp_reference', False):
            child._q36_original_moe_forward = child.forward
            child.forward = MethodType(tp4_forward, child)
            child._q36_moe_tp_reference = True
            count += 1
    print(f'[q36-precision] actor TP4 MoE arithmetic reference: {count} blocks', flush=True)
    return count