"""FP32 recurrent GDN for CANN builds whose recurrent op only accepts BF16.

The cache has layout [slots, value_heads, value_dim, key_dim]. Never cast the
whole cache: doing so rounds even slots belonging to unrelated requests on
every decode step. This implements the recurrent delta rule used by the HF
Qwen3.5 model, gathering and updating only the requested slots.
"""
import os
from contextvars import ContextVar
from types import MethodType

import torch

_policy_prefill = ContextVar("q36_policy_prefill", default=False)


def install_mixed_cache_barrier(native):
    """Diagnostic ordering barrier for policy GDN mixed prefill/decode."""
    if os.environ.get('MINICPMO_GDN_CACHE_BARRIER', '0') != '1':
        return 0
    language = getattr(native, 'llm', None)
    if language is None:
        language = getattr(getattr(native, 'thinker', None), 'llm', None)
    if language is None:
        raise RuntimeError('GDN cache barrier requires the policy language model')
    count = 0
    def wrap(module):
        original = module._forward_core
        reported = False
        def core(self, *args, **kwargs):
            nonlocal reported
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata
            current = metadata.get(self.prefix) if isinstance(metadata, dict) else None
            mixed = current is not None and current.num_prefills > 0 and current.num_decodes > 0
            if mixed:
                torch.npu.synchronize(self.kv_cache[1].device)
                if not reported and self.layer_idx == 0:
                    reported = True
                    print(f'[qwen36-gdn] mixed cache barrier executed: prefill={current.num_prefills}, decode={current.num_decodes}', flush=True)
            try:
                return original(*args, **kwargs)
            finally:
                if mixed:
                    torch.npu.synchronize(self.kv_cache[1].device)
        module._forward_core = MethodType(core, module)
        module._q36_mixed_cache_barrier = True
    for layer in language.model.layers:
        attention = getattr(layer, 'linear_attn', None)
        if attention is not None and not getattr(attention, '_q36_mixed_cache_barrier', False):
            wrap(attention)
            count += 1
    print(f'[qwen36-gdn] mixed cache ordering diagnostic installed: {count} layers', flush=True)
    return count


def actor_gdn_gating(A_log, a, b, dt_bias):
    """Use the actual actor's parameter cast and separate FP32 operations."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        alpha = A_log.to(a.dtype).float()
        bias = dt_bias.to(a.dtype)
        g = -alpha.exp() * torch.nn.functional.softplus(a.float() + bias)
        return g.unsqueeze(0), b.sigmoid().unsqueeze(0)


def install_gating_reference():
    if os.environ.get('MINICPMO_GDN_GATING_REFERENCE', '0') != '1':
        return False
    from vllm_ascend.ops.gdn import DeviceOperator
    original = DeviceOperator.fused_gdn_gating
    if getattr(original, '_q36_actor_gating', False):
        return True
    reported = False
    def gating(A_log, a, b, dt_bias):
        nonlocal reported
        if not _policy_prefill.get():
            return original(A_log, a, b, dt_bias)
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise RuntimeError('Actor GDN gate reference requires BF16 projections')
        if not reported:
            reported = True
            print(f'[qwen36-gdn] actual actor decay/beta math: shape={tuple(a.shape)}', flush=True)
        return actor_gdn_gating(A_log, a, b, dt_bias)
    gating._q36_actor_gating = True
    DeviceOperator.fused_gdn_gating = staticmethod(gating)
    return True


def install_conv_reference():
    """Round the convolution to BF16 before SiLU, as the actual HF actor does."""
    if os.environ.get('MINICPMO_GDN_CONV_REFERENCE','0')!='1':
        return False
    ops=torch.ops._C_ascend
    if not hasattr(ops,'npu_causal_conv1d_custom'):
        return False  # Retry after the native model loads Ascend's operators.
    raw=ops.npu_causal_conv1d_custom
    if getattr(raw,'_q36_conv_reference',False):
        return True
    reported_modes=set()
    def convolution(*args,**kwargs):
        activation=kwargs.get('activation_mode',args[9] if len(args)>9 else None)
        output=kwargs.get('output',args[0] if args else None)
        if not (_policy_prefill.get() and activation==1 and output is not None and output.dtype==torch.bfloat16):
            return raw(*args,**kwargs)
        if 'activation_mode' in kwargs:
            kwargs={**kwargs,'activation_mode':0}
        else:
            args=list(args);args[9]=0
        result=raw(*args,**kwargs)
        with torch.autocast(device_type=output.device.type,enabled=False):
            output.copy_(torch.nn.functional.silu(output))
        mode=kwargs.get('run_mode',args[11] if len(args)>11 else None)
        if mode not in reported_modes:
            reported_modes.add(mode)
            print(f'[qwen36-gdn] applied actor conv/SiLU boundary: run_mode={mode}, shape={tuple(output.shape)}',flush=True)
        return result
    convolution._q36_conv_reference=True
    convolution.__wrapped__=raw
    ops.npu_causal_conv1d_custom=convolution
    print('[qwen36-gdn] thinker convolution rounds to BF16 before SiLU',flush=True)
    return True


def actor_l2norm(x, eps=1e-6, output_dtype=None):
    """Match the autocast-disabled torch fallback's normalization boundaries."""
    result = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return result if output_dtype is None else result.to(output_dtype)


def reference_prefill(q, k, v, g, beta, initial_state, cu_seqlens, scale):
    """HF chunk algebra with FP32 boundary states and independent sequences.

    Diagnostic alternative to CANN's BF16-only prefill boundary. State layout
    at the API boundary remains [sequence, head, value, key].
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule

    if scale != k.shape[-1] ** -0.5:
        raise ValueError("Reference Qwen GDN requires its configured inverse-sqrt scale")
    offsets = cu_seqlens.detach().cpu().tolist()
    if (q.shape[0] != 1 or len(offsets) != len(initial_state) + 1
            or offsets[0] != 0 or offsets[-1] != q.shape[1]
            or any(a > b for a, b in zip(offsets[:-1], offsets[1:]))):
        raise ValueError("Invalid packed GDN prefill boundaries")
    ratio = v.shape[-2] // k.shape[-2]
    if ratio * k.shape[-2] != v.shape[-2] or q.shape[-2] != k.shape[-2]:
        raise ValueError("Invalid GDN query/key/value head counts")
    outputs, states = [], []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = initial_state[i:i+1].transpose(-1, -2).float().contiguous()
        if start == end:
            states.append(state)
            continue
        out, state = torch_chunk_gated_delta_rule(
            q[:, start:end].repeat_interleave(ratio, dim=2),
            k[:, start:end].repeat_interleave(ratio, dim=2), v[:, start:end],
            g=g[:, start:end], beta=beta[:, start:end], initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        outputs.append(out)
        states.append(state)
    output = torch.cat(outputs, dim=1) if outputs else torch.empty_like(v)
    return output, torch.cat(states).transpose(-1, -2).contiguous()


def install_reference_prefill():
    prefill_enabled = os.environ.get("MINICPMO_GDN_PREFILL_REFERENCE", "0") == "1"
    l2norm_enabled = os.environ.get("MINICPMO_GDN_L2NORM_REFERENCE", "0") == "1"
    conv_enabled = os.environ.get('MINICPMO_GDN_CONV_REFERENCE','0')=='1'
    gating_enabled = os.environ.get('MINICPMO_GDN_GATING_REFERENCE','0')=='1'
    if not (prefill_enabled or l2norm_enabled or conv_enabled or gating_enabled):
        return False
    install_conv_reference()
    install_gating_reference()
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention as GDN

    if getattr(GDN, "_q36_reference_prefill_installed", False):
        return True
    original_prefill, original_forward = GDN._chunk_gated_delta_rule_fused, GDN.forward

    def scoped_forward(self, *args, **kwargs):
        token = _policy_prefill.set("thinker.llm" in self.prefix)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _policy_prefill.reset(token)

    def prefill(*args, **kwargs):
        fn = reference_prefill if _policy_prefill.get() else original_prefill
        return fn(*args, **kwargs)

    GDN.forward = scoped_forward
    # Ascend also installs its methods on the upstream class in place. Handle
    # either import order so existing upstream-class instances enter the scope.
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    if QwenGatedDeltaNetAttention.forward is original_forward:
        QwenGatedDeltaNetAttention.forward = scoped_forward
    if prefill_enabled:
        GDN._chunk_gated_delta_rule_fused = staticmethod(prefill)
        GDN._fused_chunk_available = True
    if l2norm_enabled:
        from importlib import import_module
        gdn_module = import_module("vllm_ascend.ops.gdn")
        original_l2norm = gdn_module.l2norm_fwd

        def l2norm(*args, **kwargs):
            fn = actor_l2norm if _policy_prefill.get() else original_l2norm
            return fn(*args, **kwargs)

        gdn_module.l2norm_fwd = l2norm
    GDN._q36_reference_prefill_installed = True
    print(f"[qwen36-gdn] thinker reference prefill={prefill_enabled}, actor Q/K norm={l2norm_enabled}; FP32 boundary states", flush=True)
    return True


def recurrent_gdn_fp32(query, key, value, state, beta=None, scale=None,
                       actual_seq_lengths=None, ssm_state_indices=None,
                       num_accepted_tokens=None, g=None, gk=None):
    if gk is not None:
        raise ValueError("Qwen3.5 recurrent GDN does not use per-key decay gk")
    if state.dtype != torch.float32:
        raise ValueError("FP32 GDN requires a float32 recurrent cache")
    tokens, value_heads, _ = value.shape
    if tokens == 0:
        return torch.empty_like(value)
    if ssm_state_indices is None or actual_seq_lengths is None:
        raise ValueError("GDN requires sequence lengths and cache slot indices")
    indices = ssm_state_indices.reshape(-1).long()
    key_heads = key.shape[-2]
    if value_heads % key_heads:
        raise ValueError("value_heads must be divisible by key_heads")
    ratio = value_heads // key_heads
    q = query.float().repeat_interleave(ratio, dim=1)
    k = key.float().repeat_interleave(ratio, dim=1)
    q = q * (key.shape[-1] ** -0.5 if scale is None else scale)
    v = value.float()

    def step(h, token_indices):
        kt = k[token_indices]
        if g is not None:
            h = h * g[token_indices].float().exp()[..., None, None]
        residual = v[token_indices] - (h * kt[..., None, :]).sum(-1)
        if beta is not None:
            residual = residual * beta[token_indices].float()[..., None]
        h = h + residual[..., None] * kt[..., None, :]
        output = (h * q[token_indices][..., None, :]).sum(-1)
        return h, output.to(value.dtype)

    # Ordinary decode: one token per sequence, vectorized across requests.
    # The .item() check synchronizes with the host; -1 marks a padded cache slot.
    if (num_accepted_tokens is None and actual_seq_lengths.numel() == tokens + 1
            and bool((actual_seq_lengths[1:] == 1).all().item())):
        valid = indices >= 0
        slots = indices.clamp_min(0)
        h, output = step(state[slots], torch.arange(tokens, device=query.device))
        valid_slots = slots[valid]
        state.index_copy_(0, valid_slots, h[valid])
        return output.masked_fill(~valid[..., None, None], 0)

    # The Ascend API encodes lengths as [0, len(seq0), len(seq1), ...], not
    # cumulative offsets. Speculative decode stores each intermediate state.
    lengths = actual_seq_lengths.detach().cpu().tolist()
    if not lengths or lengths[0] != 0 or sum(lengths[1:]) != tokens:
        raise ValueError(f"Invalid Ascend GDN sequence lengths: {lengths}, tokens={tokens}")
    accepted = (num_accepted_tokens.detach().cpu().tolist()
                if num_accepted_tokens is not None else None)
    output = torch.zeros_like(value)
    start = 0
    for seq, length in enumerate(lengths[1:]):
        if length == 0:
            continue
        source = start if accepted is None else start + int(accepted[seq]) - 1
        source_slot = indices[source:source + 1]
        h = state[source_slot.clamp_min(0)]
        for token in range(start, start + length):
            token_index = torch.tensor([token], device=query.device)
            h, out = step(h, token_index)
            slot = indices[token:token + 1]
            valid = slot >= 0
            state.index_copy_(0, slot[valid], h[valid])
            output[token:token + 1] = out.masked_fill(~valid[..., None, None], 0)
        start += length
    return output


def install_fp32_gdn():
    """Install only in explicitly configured Qwen3.6 rollout workers."""
    if os.environ.get("MINICPMO_GDN_FP32_REFERENCE", "0") != "1":
        return False
    install_reference_prefill()
    ops = torch.ops._C_ascend
    if not hasattr(ops, "npu_recurrent_gated_delta_rule"):
        return False  # The model-init hook retries after Ascend loads its ops.
    raw = ops.npu_recurrent_gated_delta_rule
    if getattr(raw, "_q36_fp32_state", False):
        return True

    def recurrent(*args, **kwargs):
        state = kwargs.get("state", args[3] if len(args) > 3 else None)
        if state is not None and state.dtype == torch.float32:
            return recurrent_gdn_fp32(*args, **kwargs)
        return raw(*args, **kwargs)

    recurrent._q36_fp32_state = True
    ops.npu_recurrent_gated_delta_rule = recurrent
    # Keep the working CANN prefill path. The alternative Ascend chunk pipeline
    # requires aclnnChunkGatedDeltaRuleFwdH, absent from this CANN build. Prefill
    # may round its boundary state once, but decode must not round all slots on
    # every token (including slots for requests not scheduled in this batch).
    print("[qwen36-gdn] FP32 recurrent state enabled; existing prefill retained", flush=True)
    return True