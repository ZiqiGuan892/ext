"""Opt-in language RoPE matching the actual verl NPU actor operation/cache."""
import os
from types import MethodType

import torch


def actor_rotary(self, x, cos, sin):
    """Apply the actor's npu_rotary_mul in its actual [1, heads, tokens, dim] layout."""
    import torch_npu
    cos = torch.cat((cos, cos), dim=-1)[None, None]
    sin = torch.cat((sin, sin), dim=-1)[None, None]
    result = torch_npu.npu_rotary_mul(x.transpose(0, 1)[None], cos, sin)[0].transpose(0, 1)
    if not getattr(self, '_q36_actor_rotary_applied', False):
        self._q36_actor_rotary_applied = True
        print(f'[q36-rope] actual actor NPU rotation: x={tuple(x.shape)}, dtype={x.dtype}', flush=True)
    return result


def install_rotary(rotary, config):
    if getattr(rotary, '_q36_actor_text_rope', False):
        return False
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    if not rotary.mrope_interleaved or rotary.head_size != 256 or rotary.rotary_dim != 64:
        raise ValueError('Actor text RoPE reference requires the configured interleaved Qwen3.6 rotation')
    if config.rope_parameters.get('rope_type', 'default') != 'default':
        raise ValueError('Actor text RoPE cache reference requires default rope scaling')
    if list(config.rope_parameters['mrope_section']) != list(rotary.mrope_section):
        raise ValueError('Actor/native RoPE sections differ')
    device = rotary.cos_sin_cache.device
    # Construct on CPU as the actor does, then compute the cache using HF's
    # exact FP32 matmul/trigonometry. Native einsum/pow differs at a few BF16
    # rounding boundaries even when position numbering is identical.
    with torch.device('cpu'):
        generator = Qwen3_5MoeTextRotaryEmbedding(config)
    generator = generator.to(device)
    length = rotary.cos_sin_cache.shape[0]
    chunks = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        dummy = torch.empty(1, dtype=torch.bfloat16, device=device)
        for start in range(0, length, 8192):
            positions = torch.arange(start, min(start + 8192, length), device=device)[None, None].expand(3, 1, -1)
            cos, sin = generator(dummy, positions)
            chunks.append(torch.cat((cos[0, :, :32], sin[0, :, :32]), dim=-1))
    rotary.cos_sin_cache = torch.cat(chunks, dim=0).contiguous()
    rotary.apply_rotary_emb.forward_native = MethodType(actor_rotary, rotary.apply_rotary_emb)
    rotary._q36_actor_text_rope = True
    return True


def install_rollout(native):
    if os.environ.get('MINICPMO_HF_TEXT_ROPE_REFERENCE', '0') != '1':
        return False
    if not getattr(native, '_q36_hf_norm_reference', False):
        raise ValueError('Actor text RoPE reference requires MINICPMO_HF_NORM_REFERENCE=1')
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    cfg = native.vllm_config.model_config.hf_config.text_config
    config = Qwen3_5MoeTextConfig(**(cfg if isinstance(cfg, dict) else cfg.to_dict()))
    language = native.thinker.llm
    paths = caches = 0
    for child in language.modules():
        if isinstance(child, Qwen3NextAttention):
            paths += 1
            caches += install_rotary(child.rotary_emb, config)
    if paths != 10:
        raise RuntimeError(f'Expected 10 actual text attention paths, found {paths}')
    print(f'[q36-rope] actual actor NPU math/cache installed: paths={paths}, new_caches={caches}', flush=True)
    return True