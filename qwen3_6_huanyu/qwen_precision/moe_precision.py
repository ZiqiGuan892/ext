"""Match unquantized Ascend SwiGLU and expert-combine rounding in HF actors."""
import os
from types import MethodType

import torch
import torch.nn.functional as F


def mlp_forward(self, x):
    gate, up = self.gate_proj(x), self.up_proj(x)
    activated = (self.act_fn(gate.float()) * up.float()).to(gate.dtype)
    return self.down_proj(activated)


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    combined = torch.zeros_like(hidden_states, dtype=torch.float32)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, -1)
        activated = (self.act_fn(gate.float()) * up.float()).to(gate.dtype)
        value = F.linear(activated, self.down_proj[expert_idx])
        weighted = value.float() * top_k_weights[token_idx, top_k_pos, None].float()
        combined.index_add_(0, token_idx, weighted)
    return combined.to(hidden_states.dtype)


def install_actor(model):
    if os.environ.get("MINICPMO_MOE_MATCH_ROLLOUT", "0") != "1":
        return 0
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts, Qwen3_5MoeMLP

    count = 0
    for child in model.modules():
        if getattr(child, "_q36_moe_precision", False):
            continue
        if isinstance(child, Qwen3_5MoeExperts):
            child.forward = MethodType(experts_forward, child)
        elif isinstance(child, Qwen3_5MoeMLP):
            child.forward = MethodType(mlp_forward, child)
        else:
            continue
        child._q36_moe_precision = True
        count += 1
    print(f"[q36-precision] actor FP32 SwiGLU and expert combination: {count} modules", flush=True)
    return count