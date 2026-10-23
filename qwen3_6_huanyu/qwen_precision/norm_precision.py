"""Opt-in actor rounding boundaries for Qwen3.5 norms and QK/RoPE."""
import os
from types import MethodType

import torch
import torch.nn.functional as F


def gemma_reference(self, x, residual=None):
    # HF materializes the residual sum in the model dtype before its FP32 norm.
    summed = x if residual is None else x + residual
    xf = summed.float()
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)
    result = (normed * (1. + self.weight.float())).to(x.dtype)
    return result if residual is None else (result, summed)


def gated_reference(self, x, z=None):
    xf = x.float()
    if x.device.type == "npu":
        import torch_npu
        weighted = torch_npu.npu_rms_norm(xf, self.weight.float(), epsilon=self.eps)[0]
    else:
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        weighted = normalized * self.weight.float()
    # FSDPEngine applies verl's NPU patch to the actor: its norm, weight and
    # SiLU multiplication stay FP32 until the final cast. The unpatched HF
    # implementation has intermediate BF16 casts and is not our actor target.
    return (weighted * F.silu(z.float()) if z is not None else weighted).to(x.dtype)


def attention_reference(self, positions, hidden_states, output=None):
    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], -1)
        q, gate = q_gate.reshape(-1, self.num_heads, self.head_dim * 2).chunk(2, -1)
        q, gate = q.reshape(-1, self.q_size), gate.reshape(-1, self.q_size)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
    q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, self.q_size)
    k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(-1, self.kv_size)
    # Keep HF's BF16 norm output before applying the unchanged RoPE formula.
    q, k = self.rotary_emb.forward_native(positions, q, k)
    result = self.attn(q, k, v)
    if self.attn_output_gate:
        result = result * torch.sigmoid(gate)
    result, _ = self.o_proj(result)
    if output is not None:
        output.copy_(result)
    return result


def install_rollout(native):
    if os.environ.get("MINICPMO_HF_NORM_REFERENCE", "0") != "1":
        return False
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    if getattr(native, "_q36_hf_norm_reference", False):
        return False
    language = getattr(native, "llm", None)
    if language is None:
        language = getattr(getattr(native, "thinker", None), "llm", None)
    if language is None:
        raise RuntimeError("HF norm reference requires the initialized thinker language model")
    norms = attention = gated = 0
    for child in language.modules():
        if isinstance(child, GemmaRMSNorm):
            # CustomOp caches the bound dispatch method at construction time.
            child._forward_method = MethodType(gemma_reference, child)
            norms += 1
        if isinstance(child, RMSNormGated) and child.norm_before_gate and child.group_size is None:
            child._forward_method = MethodType(gated_reference, child)
            gated += 1
        # Ascend replaces Base.forward in place; existing instances retain the
        # upstream class identity rather than becoming Ascend's subclass.
        if isinstance(child, Qwen3NextAttention):
            child.forward = MethodType(attention_reference, child)
            attention += 1
    if norms == 0:
        raise RuntimeError("HF norm reference requested but no language Gemma norms found")
    native._q36_hf_norm_reference = True
    print(f"[q36-precision] HF norm rounding enabled: {norms} norms, {gated} gated norms, {attention} QK/RoPE paths", flush=True)
    return True