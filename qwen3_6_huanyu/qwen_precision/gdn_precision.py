"""FP32 recurrent GDN for CANN builds whose recurrent op only accepts BF16.

The cache has layout [slots, value_heads, value_dim, key_dim]. Never cast the
whole cache: doing so rounds even slots belonging to unrelated requests on
every decode step. This implements the recurrent delta rule used by the HF
Qwen3.5 model, gathering and updating only the requested slots.
"""
import os
from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch

_policy_prefill = ContextVar("huanyu_policy_gdn", default=False)
_EXECUTED = set()


def _status(component, phase, **details):
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[huanyu-precision] component={component} phase={phase} {fields}".rstrip(), flush=True)


def _executed(component, **details):
    if component not in _EXECUTED:
        _EXECUTED.add(component)
        _status(component, "executed", **details)


def _enabled(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _attention_modules(native):
    language = getattr(native, "llm", None)
    if language is None:
        language = getattr(getattr(native, "thinker", None), "llm", None)
    layers = getattr(getattr(language, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Huanyu GDN reference requires the explicit native language model")
    modules = [layer.linear_attn for layer in layers if getattr(layer, "linear_attn", None) is not None]
    if not modules:
        raise RuntimeError("Huanyu GDN reference found no language linear-attention layers")
    return modules


def install_mixed_cache_barrier(native):
    """Diagnostic ordering barrier for policy GDN mixed prefill/decode."""
    if not _enabled('MINICPMO_GDN_CACHE_BARRIER'):
        _status('gdn_cache_barrier', 'disabled')
        return 0
    modules = _attention_modules(native)
    count = 0
    def wrap(module):
        original = getattr(module, '_forward_core', None)
        if not callable(original):
            raise RuntimeError('Huanyu GDN cache barrier requires _forward_core')
        @wraps(original)
        def core(self, *args, **kwargs):
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata
            current = metadata.get(self.prefix) if isinstance(metadata, dict) else None
            mixed = current is not None and current.num_prefills > 0 and current.num_decodes > 0
            if mixed:
                torch.npu.synchronize(self.kv_cache[1].device)
            try:
                result = original(*args, **kwargs)
            finally:
                if mixed:
                    torch.npu.synchronize(self.kv_cache[1].device)
            if mixed:
                _executed('gdn_cache_barrier', prefill=current.num_prefills, decode=current.num_decodes)
            return result
        core._huanyu_mixed_cache_barrier = True
        module._forward_core = MethodType(core, module)
    for attention in modules:
        if getattr(getattr(attention, '_forward_core', None), '_huanyu_mixed_cache_barrier', False) is not True:
            wrap(attention)
            count += 1
    _status('gdn_cache_barrier', 'installed' if count else 'already_installed', new=count, total=len(modules))
    return count


def actor_gdn_gating(A_log, a, b, dt_bias):
    """Use the actual actor's parameter cast and separate FP32 operations."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        alpha = A_log.to(a.dtype).float()
        bias = dt_bias.to(a.dtype)
        g = -alpha.exp() * torch.nn.functional.softplus(a.float() + bias)
        return g.unsqueeze(0), b.sigmoid().unsqueeze(0)


def install_gating_reference():
    if not _enabled('MINICPMO_GDN_GATING_REFERENCE'):
        _status('gdn_gating', 'disabled')
        return False
    from vllm_ascend.ops.gdn import DeviceOperator
    original = DeviceOperator.fused_gdn_gating
    if getattr(original, '_huanyu_actor_gating', False) is True:
        _status('gdn_gating', 'already_installed')
        return True
    @wraps(original)
    def gating(A_log, a, b, dt_bias):
        if not _policy_prefill.get():
            return original(A_log, a, b, dt_bias)
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise RuntimeError('Actor GDN gate reference requires BF16 projections')
        result = actor_gdn_gating(A_log, a, b, dt_bias)
        _executed('gdn_gating', input_dtype=a.dtype, decay_dtype=result[0].dtype,
                  beta_dtype=result[1].dtype, shape=tuple(a.shape), autocast=False)
        return result
    gating._huanyu_actor_gating = True
    DeviceOperator.fused_gdn_gating = staticmethod(gating)
    _status('gdn_gating', 'installed')
    return True


def install_conv_reference():
    """Round the convolution to BF16 before SiLU, as the actual HF actor does."""
    if not _enabled('MINICPMO_GDN_CONV_REFERENCE'):
        _status('gdn_conv', 'disabled')
        return False
    ops=torch.ops._C_ascend
    if not hasattr(ops,'npu_causal_conv1d_custom'):
        raise RuntimeError('Huanyu GDN conv reference enabled but npu_causal_conv1d_custom is unavailable; retry after native model initialization')
    raw=ops.npu_causal_conv1d_custom
    if getattr(raw,'_huanyu_conv_reference',False) is True:
        _status('gdn_conv', 'already_installed')
        return True
    @wraps(raw)
    def convolution(*args,**kwargs):
        activation=kwargs.get('activation_mode',args[9] if len(args)>9 else None)
        output=kwargs.get('output',args[0] if args else None)
        if not (_policy_prefill.get() and activation==1):
            return raw(*args,**kwargs)
        if output is None or output.dtype != torch.bfloat16:
            raise RuntimeError('Huanyu actor conv/SiLU reference requires a BF16 output buffer')
        if 'activation_mode' in kwargs:
            kwargs={**kwargs,'activation_mode':0}
        else:
            args=list(args);args[9]=0
        result=raw(*args,**kwargs)
        with torch.autocast(device_type=output.device.type,enabled=False):
            output.copy_(torch.nn.functional.silu(output))
        mode=kwargs.get('run_mode',args[11] if len(args)>11 else None)
        _executed(f'gdn_conv_{mode}', dtype=output.dtype, shape=tuple(output.shape), autocast=False)
        return result
    convolution._huanyu_conv_reference=True
    convolution.default=convolution
    ops.npu_causal_conv1d_custom=convolution
    _status('gdn_conv', 'installed')
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
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _reference_prefill(q, k, v, g, beta, initial_state, cu_seqlens, scale)


def _reference_prefill(q, k, v, g, beta, initial_state, cu_seqlens, scale):
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


def install_l2norm_reference():
    if not _enabled("MINICPMO_GDN_L2NORM_REFERENCE"):
        _status("gdn_l2norm", "disabled")
        return False
    from importlib import import_module
    module = import_module("vllm_ascend.ops.gdn")
    original = module.l2norm_fwd
    if getattr(original, "_huanyu_actor_l2norm", False) is True:
        _status("gdn_l2norm", "already_installed")
        return True

    @wraps(original)
    def l2norm(x, *args, **kwargs):
        if not _policy_prefill.get():
            return original(x, *args, **kwargs)
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = actor_l2norm(x, *args, **kwargs)
        _executed("gdn_l2norm", input_dtype=x.dtype, output_dtype=result.dtype,
                  shape=tuple(x.shape), autocast=False)
        return result

    l2norm._huanyu_actor_l2norm = True
    module.l2norm_fwd = l2norm
    _status("gdn_l2norm", "installed")
    return True


def install_reference_prefill(native):
    """Bind reference calls to explicit Huanyu layers, independently of Whisper.

    Invoke after native construction: Ascend registers its custom operators
    while constructing the model. An early failed attempt leaves no marker.
    """
    flags = {name: _enabled(f"MINICPMO_GDN_{name.upper()}_REFERENCE")
             for name in ("prefill", "l2norm", "conv", "gating")}
    if not any(flags.values()):
        for name in flags:
            _status(f"gdn_{name}", "disabled")
        return False
    modules = _attention_modules(native)
    for module in modules:
        if not callable(getattr(module, "forward", None)):
            raise RuntimeError("Huanyu GDN reference requires attention.forward")
        if flags["prefill"] and not callable(getattr(module, "_chunk_gated_delta_rule_fused", None)):
            raise RuntimeError("Huanyu GDN prefill reference requires _chunk_gated_delta_rule_fused")
    if flags["prefill"]:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule
        if not callable(torch_chunk_gated_delta_rule):
            raise RuntimeError("Huanyu GDN prefill requires the HF torch chunk reference")
    install_conv_reference()
    install_gating_reference()
    install_l2norm_reference()
    new_scope = new_prefill = 0

    def scope(module):
        original = module.forward

        @wraps(original)
        def forward(self, *args, **kwargs):
            token = _policy_prefill.set(True)
            try:
                return original(*args, **kwargs)
            finally:
                _policy_prefill.reset(token)

        forward._huanyu_gdn_scope = True
        module.forward = MethodType(forward, module)

    def bind_prefill(module):
        def prefill(*args, **kwargs):
            result = reference_prefill(*args, **kwargs)
            _executed("gdn_prefill", output_dtype=result[0].dtype, state_dtype=result[1].dtype,
                      output_shape=tuple(result[0].shape), autocast=False)
            return result
        prefill._huanyu_reference_prefill = True
        # Native's API is static. A function stored on the instance stays unbound.
        module._chunk_gated_delta_rule_fused = prefill
        module._fused_chunk_available = True

    for module in modules:
        if getattr(module.forward, "_huanyu_gdn_scope", False) is not True:
            scope(module)
            new_scope += 1
        if flags["prefill"] and getattr(module._chunk_gated_delta_rule_fused, "_huanyu_reference_prefill", False) is not True:
            bind_prefill(module)
            new_prefill += 1
    _status("gdn_scope", "installed" if new_scope else "already_installed", new=new_scope, total=len(modules))
    _status("gdn_prefill", ("installed" if new_prefill else "already_installed") if flags["prefill"] else "disabled",
            new=new_prefill, total=len(modules))
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
    """Reject the obsolete unscoped API; Huanyu's rollout entry owns decode."""
    raise RuntimeError("Use qwen3_6_huanyu.rollout_precision on an explicit Huanyu model; unscoped GDN patching is unsupported")
