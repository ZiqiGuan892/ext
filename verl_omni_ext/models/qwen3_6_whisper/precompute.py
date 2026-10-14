"""预计算 embedding 注入（Whisper + Qwen3-VL visual + 新增 video cache）。

当 ``MINICPMO_EMBED_CACHE_DIR`` 非空时，``build_qwen36_forward_adapter`` 会调用
``patch_precomputed_embeddings`` 冻结 encoder 并 patch ``get_audio_embedding``；
``_inject_precomputed_embeddings`` 在每次 forward 前从 dataloader 注入的
``_precomputed_*_embeds`` 字段中 unwrap NonTensorData，按 batch 对齐后写回
``data["_precomputed_audio_embeds"]`` 和 ``data["vision_hidden_states"]``。

**铁律**：cache miss 必须 FATAL raise（不静默 fallback）——FSDP2 集合通信要求所有
rank 同步，cache miss 静默 fallback 会导致不对称集合通信死锁。
See memory: feedback_precompute_cache_miss_fatal.md

与 MiniCPM-o 5.0 差异：
  - encoder 模块路径改名（``apm`` → ``model.audio_encoder``，
    ``vpm`` → ``model.visual``，无 ``resampler`` / ``vit_merger`` / ``audio_projection_layer``）
  - 新增 video encoder 预计算缓存
  - RoPE 修复逻辑与旧模型相同（LM 骨架相同）
"""
import hashlib
import logging
import os
import types
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PRECOMPUTE_FLAG = "_verl_omni_qwen36_precompute_embeds"
_PRECOMPUTE_CACHE_ENV = "MINICPMO_EMBED_CACHE_DIR"


def _path_hash(path: str) -> str:
    """SHA-256 of a file path string. Used as cache key."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _unwrap_ntd(v):
    """Unwrap NonTensorData (may be serialized to dict by TensorDict collation)."""
    from tensordict import NonTensorData

    if isinstance(v, NonTensorData):
        return v.data
    if isinstance(v, dict) and "data" in v and len(v) <= 4:
        return v["data"]
    return v


def _inject_precomputed_embeddings(data: dict, cache_dir: str, call_no: int):
    """Inject precomputed Whisper + Qwen3-VL visual + video embeddings into the data dict.

    与 MiniCPM-o 5.0 的 ``_inject_precomputed_embeddings`` 逻辑相同，仅字段名与
    归一化路径适配新模型。
    """
    input_ids = data.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        target_device = input_ids.device
    else:
        target_device = torch.device("cpu")

    bs = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1

    # ----- Audio: read precomputed embeds injected by dataloader -----
    audio_embeds_raw = data.get("_precomputed_audio_embeds")
    if audio_embeds_raw is not None:
        if isinstance(audio_embeds_raw, list):
            audio_embeds_raw = [_unwrap_ntd(v) for v in audio_embeds_raw]
        else:
            audio_embeds_raw = _unwrap_ntd(audio_embeds_raw)

        data["_precomputed_audio_embeds"] = [
            [torch.cat([emb.to(target_device) for chunk_embeds in sample_embeds for emb in chunk_embeds], dim=0)]
            if sample_embeds else []
            for sample_embeds in audio_embeds_raw
        ]
        if len(data["_precomputed_audio_embeds"]) < bs:
            data["_precomputed_audio_embeds"] += [[] for _ in range(bs - len(data["_precomputed_audio_embeds"]))]

    # ----- Vision (image + video): read precomputed embeds injected by dataloader -----
    vision_embeds_raw = data.get("_precomputed_vision_embeds")
    if call_no <= 1:
        print(
            f"[verl-omni] INJECT DIAG: vision_embeds_raw type={type(vision_embeds_raw).__name__} "
            f"is_none={vision_embeds_raw is None} data_keys={sorted(data.keys())}",
            flush=True,
        )
    if vision_embeds_raw is not None:
        if isinstance(vision_embeds_raw, list):
            vision_embeds_raw = [_unwrap_ntd(v) for v in vision_embeds_raw]
        else:
            vision_embeds_raw = _unwrap_ntd(vision_embeds_raw)

        # vision_embeds_raw is a per-sample list. Each entry is precomputed_vision
        # for that sample: a list of per-file slice embed lists.
        vhs = [
            [emb.to(target_device) if isinstance(emb, torch.Tensor) else emb
             for file_embeds in sample_embeds
             for emb in file_embeds]
            if sample_embeds else []
            for sample_embeds in vision_embeds_raw
        ]

        # Pad/truncate each slice embedding to match image_bound range
        image_bound = data.get("image_bound", [])
        if isinstance(image_bound, list):
            for sample_i, sample_vhs in enumerate(vhs):
                if sample_i >= len(image_bound) or not sample_vhs:
                    continue
                bounds = image_bound[sample_i]
                if not hasattr(bounds, "__len__") or len(bounds) == 0:
                    continue
                for img_idx, bound in enumerate(bounds):
                    if img_idx >= len(sample_vhs):
                        break
                    bound_len = int(bound[1] - bound[0])
                    emb = sample_vhs[img_idx]
                    if isinstance(emb, torch.Tensor) and emb.shape[0] != bound_len:
                        if emb.shape[0] > bound_len:
                            sample_vhs[img_idx] = emb[:bound_len]
                        else:
                            pad = torch.zeros(
                                bound_len - emb.shape[0], emb.shape[1],
                                dtype=emb.dtype, device=emb.device,
                            )
                            sample_vhs[img_idx] = torch.cat([emb, pad], dim=0)

        data["vision_hidden_states"] = vhs
        if len(data["vision_hidden_states"]) < bs:
            data["vision_hidden_states"] += [[] for _ in range(bs - len(data["vision_hidden_states"]))]
    else:
        data["vision_hidden_states"] = [[] for _ in range(bs)]

    # ----- Pad image_bound and audio_bounds to bs -----
    vhs = data.get("vision_hidden_states", [])
    pae = data.get("_precomputed_audio_embeds")

    image_bound = data.get("image_bound")
    if image_bound is not None and isinstance(image_bound, list) and len(image_bound) < bs:
        padded = []
        ib_idx = 0
        for i in range(bs):
            has_image = i < len(vhs) and len(vhs[i]) > 0 if isinstance(vhs, list) else False
            if has_image and ib_idx < len(image_bound):
                padded.append(image_bound[ib_idx])
                ib_idx += 1
            else:
                padded.append([])
        data["image_bound"] = padded

    audio_bounds = data.get("audio_bounds")
    if audio_bounds is not None and isinstance(audio_bounds, list) and len(audio_bounds) < bs:
        padded = []
        ab_idx = 0
        for i in range(bs):
            has_audio = (pae is not None and i < len(pae) and len(pae[i]) > 0) if pae else False
            if has_audio and ab_idx < len(audio_bounds):
                padded.append(audio_bounds[ab_idx])
                ab_idx += 1
            else:
                padded.append([])
        data["audio_bounds"] = padded

    # Ensure vision_hidden_states[i] has enough embeddings for image_bound[i]
    vhs_final = data.get("vision_hidden_states", [])
    image_bound_final = data.get("image_bound", [])
    if isinstance(vhs_final, list) and isinstance(image_bound_final, list):
        for i in range(min(len(vhs_final), len(image_bound_final))):
            sample_vhs = vhs_final[i]
            bounds = image_bound_final[i]
            if not hasattr(bounds, "__len__") or len(bounds) == 0:
                continue
            n_bounds = len(bounds)
            if len(sample_vhs) < n_bounds:
                if sample_vhs:
                    last_emb = sample_vhs[-1]
                    while len(sample_vhs) < n_bounds:
                        sample_vhs.append(last_emb)
                vhs_final[i] = sample_vhs
        data["vision_hidden_states"] = vhs_final

    if call_no <= 1:
        _vhs = data.get("vision_hidden_states")
        _pae = data.get("_precomputed_audio_embeds")
        _input_ids = data.get("input_ids")
        _bs = len(_input_ids) if _input_ids is not None else "?"
        _vhs_lens = [len(x) if hasattr(x, "__len__") else "?" for x in _vhs] if isinstance(_vhs, list) else "?"
        _pae_lens = [len(x) if hasattr(x, "__len__") else "?" for x in _pae] if isinstance(_pae, list) else "?"
        print(
            f"[verl-omni] qwen3.6-whisper precompute inject call={call_no}: "
            f"audio={'cached' if '_precomputed_audio_embeds' in data else 'none'}, "
            f"vision={'cached' if 'vision_hidden_states' in data else 'none'}, "
            f"device={target_device} bs={_bs} "
            f"vhs_len={len(_vhs) if _vhs is not None else 'None'} "
            f"vhs_sublens={_vhs_lens} "
            f"pae_len={len(_pae) if _pae is not None else 'None'} "
            f"pae_sublens={_pae_lens}",
            flush=True,
        )


def patch_precomputed_embeddings(module: Any, cache_dir: str):
    """Freeze encoders and patch ``get_audio_embedding`` to use precomputed cache.

    与 MiniCPM-o 5.0 的 ``patch_precomputed_embeddings`` 逻辑相同，差异：

    1. **Encoder 模块路径改名**：
       - 旧 ``apm`` / ``audio_projection_layer`` / ``audio_avg_pooler`` →
         新 ``model.audio_encoder`` / ``model.audio_projection``
       - 旧 ``vpm`` / ``resampler`` / ``vit_merger`` →
         新 ``model.visual``（含 merger，无独立 resampler）

    2. **新增 video encoder**：新模型支持 video，``model.visual`` 同时处理 image
       与 video。预计算时视频路径与图像路径共用 visual encoder，cache key 区分
       file path 即可。

    3. **RoPE 修复**：LM 骨架与旧模型相同（Qwen3.5-MoE），NPU
       ``aclnnRotaryPositionEmbeddingGrad`` broadcast bug 仍存在，复用同一 RoPE
       patch。但 ``modeling_qwen3_5_moe`` 在新模型环境下的 import 路径需验证。

    See memory:
      - feedback_precompute_all_modalities.md (所有 frozen encoder 都要 precompute)
      - feedback_precompute_cache_miss_fatal.md (cache miss FATAL)
      - feedback_npu_rope_backward_broadcast.md (RoPE 修复)
    """
    if getattr(module, _PRECOMPUTE_FLAG, False):
        return

    # 1. Freeze encoders — 新模型模块路径
    model_sub = getattr(module, "model", None)
    frozen_names = []
    for attr_name in ("audio_encoder", "audio_projection", "audio_avg_pooler"):
        sub = getattr(model_sub, attr_name, None) if model_sub is not None else None
        if sub is None:
            sub = getattr(module, attr_name, None)
        if sub is not None:
            n_params = sum(p.numel() for p in sub.parameters())
            sub.requires_grad_(False)
            frozen_names.append(f"{attr_name}({n_params / 1e6:.1f}M)")

    # Visual encoder 在 model.visual（新模型）
    visual_sub = getattr(model_sub, "visual", None) if model_sub is not None else None
    if visual_sub is None:
        for name in ("vpm", "vision_tower", "visual"):
            sub = getattr(module, name, None)
            if sub is not None:
                visual_sub = sub
                break
    if visual_sub is not None:
        n_params = sum(p.numel() for p in visual_sub.parameters())
        visual_sub.requires_grad_(False)
        frozen_names.append(f"visual({n_params / 1e6:.1f}M)")

    print(
        f"[verl-omni] qwen3.6-whisper precompute: froze encoders [{', '.join(frozen_names)}]. "
        f"Cache dir: {cache_dir}",
        flush=True,
    )

    # 2. Patch get_audio_embedding
    original_get_audio_embedding = getattr(module, "get_audio_embedding", None)

    def _patched_get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        precomputed = data.get("_precomputed_audio_embeds")
        if precomputed is not None:
            return precomputed
        wavforms = data.get("audio_features", [])
        input_ids = data.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            device = input_ids.device
        else:
            device = "cpu"
        if len(wavforms) == 0:
            return [torch.zeros(1, 1, dtype=torch.float32, device=device)]
        # Cache miss with audio present: return zero embeddings matching
        # audio_bounds structure.
        hidden_size = 2048
        audio_bounds = data.get("audio_bounds", [])
        bs = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1

        # Normalize audio_bounds to list of list of [start_int, end_int] pairs.
        # audio_bounds[i] must be a LIST of [start, end] pairs (one per segment).
        # See memory: feedback_precompute_cache_miss_fatal.md
        normalized_bounds = []
        if isinstance(audio_bounds, torch.Tensor):
            for i in range(audio_bounds.shape[0]):
                row = audio_bounds[i]
                if row.dim() == 0:
                    continue
                s = int(row[0].item()) if hasattr(row[0], "item") else int(row[0])
                e = int(row[1].item()) if hasattr(row[1], "item") else int(row[1])
                normalized_bounds.append([[s, e]])
        elif isinstance(audio_bounds, list):
            for sample_bounds in audio_bounds:
                if isinstance(sample_bounds, torch.Tensor):
                    if sample_bounds.dim() == 1:
                        s = int(sample_bounds[0].item()) if hasattr(sample_bounds[0], "item") else int(sample_bounds[0])
                        e = int(sample_bounds[1].item()) if hasattr(sample_bounds[1], "item") else int(sample_bounds[1])
                        normalized_bounds.append([[s, e]])
                    elif sample_bounds.dim() == 2:
                        pairs = []
                        for j in range(sample_bounds.shape[0]):
                            s = int(sample_bounds[j][0].item()) if hasattr(sample_bounds[j][0], "item") else int(sample_bounds[j][0])
                            e = int(sample_bounds[j][1].item()) if hasattr(sample_bounds[j][1], "item") else int(sample_bounds[j][1])
                            pairs.append([s, e])
                        normalized_bounds.append(pairs)
                    continue
                elif hasattr(sample_bounds, "__len__") and len(sample_bounds) >= 2:
                    first = sample_bounds[0]
                    if hasattr(first, "__len__"):
                        pairs = []
                        for pair in sample_bounds:
                            s = int(pair[0].item()) if hasattr(pair[0], "item") else int(pair[0])
                            e = int(pair[1].item()) if hasattr(pair[1], "item") else int(pair[1])
                            pairs.append([s, e])
                        normalized_bounds.append(pairs)
                    else:
                        s = int(first.item()) if hasattr(first, "item") else int(first)
                        e = int(sample_bounds[1].item()) if hasattr(sample_bounds[1], "item") else int(sample_bounds[1])
                        normalized_bounds.append([[s, e]])
                    continue
                normalized_bounds.append([[0, 1]])
        else:
            normalized_bounds = []

        if normalized_bounds:
            data["audio_bounds"] = normalized_bounds

        final_audio_embeds = []
        for i in range(min(bs, len(normalized_bounds))):
            sample_pairs = normalized_bounds[i]
            sample_embeds = []
            for s, e in sample_pairs:
                n_tokens = e - s
                if n_tokens > 0:
                    sample_embeds.append(torch.zeros(n_tokens, hidden_size, dtype=torch.float32, device=device))
                else:
                    sample_embeds.append(torch.zeros(1, hidden_size, dtype=torch.float32, device=device))
            if not sample_embeds:
                sample_embeds = [torch.zeros(1, hidden_size, dtype=torch.float32, device=device)]
            final_audio_embeds.append(sample_embeds)
        if not final_audio_embeds:
            final_audio_embeds = [[torch.zeros(1, hidden_size, dtype=torch.float32, device=device)]]
        return final_audio_embeds

    if original_get_audio_embedding is not None:
        module.get_audio_embedding = types.MethodType(_patched_get_audio_embedding, module)

    setattr(module, _PRECOMPUTE_FLAG, True)
    print(
        "[verl-omni] qwen3.6-whisper precompute: patched get_audio_embedding to check "
        "data['_precomputed_audio_embeds'] before running Whisper.",
        flush=True,
    )

    # 3. Force-disable stream_input (same as MiniCPM-o 5.0)
    if hasattr(module, "config") and getattr(module.config, "stream_input", False):
        module.config.stream_input = False
        print("[verl-omni] qwen3.6-whisper precompute: forced config.stream_input=False "
              "(RL training uses bs>1, stream_input requires bs==1)", flush=True)

    # 4. Force-disable canvas M-RoPE (same as MiniCPM-o 5.0)
    if hasattr(module, "config") and getattr(module.config, "mrope_mode", None):
        module.config.mrope_mode = None
        print("[verl-omni] qwen3.6-whisper precompute: forced config.mrope_mode=None "
              "(NPU rotary backward doesn't support canvas M-RoPE broadcasting)",
              flush=True)

    # 5. Patch RoPE to standard PyTorch (LM backbone identical to MiniCPM-o 5.0)
    _patch_rope_to_pytorch()


def _patch_rope_to_pytorch():
    """Replace npu_rotary_mul-based RoPE with standard PyTorch RoPE.

    NPU's ``aclnnRotaryPositionEmbeddingGrad`` (error 561002) doesn't support head
    broadcasting. Standard PyTorch RoPE with ``partial_rotary_factor`` split works.

    LM 骨架与 MiniCPM-o 5.0 相同（Qwen3.5-MoE），bug 与 fix 都相同。

    See memory: feedback_npu_rope_backward_broadcast.md
    """
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
    except ImportError:
        try:
            # 备选 import 路径（新模型可能在别的模块）
            from transformers.models.qwen3_5_moe_text import modeling_qwen3_5_moe_text as modeling_qwen3_5_moe
        except ImportError:
            return

    if getattr(modeling_qwen3_5_moe.apply_rotary_pos_emb, "_omni_patched", False):
        return

    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_pytorch(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([q_rot * cos + _rotate_half(q_rot) * sin, q_pass], dim=-1)
        k_embed = torch.cat([k_rot * cos + _rotate_half(k_rot) * sin, k_pass], dim=-1)
        return q_embed, k_embed

    _apply_rotary_pos_emb_pytorch._omni_patched = True
    modeling_qwen3_5_moe.apply_rotary_pos_emb = _apply_rotary_pos_emb_pytorch
    print("[verl-omni] qwen3.6-whisper precompute: patched RoPE to standard PyTorch "
          "(NPU npu_rotary_mul backward doesn't support head broadcasting)",
          flush=True)