# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forward-signature adapter for Qwen3.6-35B-A3B-whisper thinker training.

与新模型差异（vs MiniCPM-o 5.0）
------------------------------
MiniCPM-o 5.0 的 forward 签名是 ``MiniCPMO.forward(self, data, **kwargs)`` ——
MiniCPM-V/o family 的 single positional dict 约定，与 verl 的 keyword-only 调
用约定不兼容，需要 ``build_minicpmo_forward_adapter`` 重写 forward。

新模型 Qwen3.6-35B-A3B-whisper-0814-rl 的 forward 签名已是标准 transformers
风格：
    forward(self, input_ids, attention_mask, position_ids, **kwargs)

这与 verl 的 keyword-only 调用约定
(``transformer_impl.py:1526: raw_output = self.module(**model_inputs, use_cache=False)``)
直接兼容，**无需重写 forward**。

但仍需安装 NPU 必需 patches：
  1. ``patch_visual_fsdp_unshard``：``model.visual`` 的 ``nn.Embedding`` 子模块
     被 verl FSDP2 单独 wrap，forward 直接读 ``.weight`` 时 DTensor 不解。
     与旧模型 ``patch_audio_encoder_fsdp_unshard`` 同症同方，仅目标改名
     （``apm`` → ``model.visual``）。
  2. ``patch_audio_encoder_fsdp_unshard``：``model.audio_encoder`` 同症。
  3. ``patch_precomputed_embeddings``（可选）：``MINICPMO_EMBED_CACHE_DIR``
     启用时冻结 encoder 并 patch ``get_audio_embedding``/``get_vision_embedding``。
     新增 video encoder 的预计算缓存。

See memory:
  - feedback_fsdp2_npu_model_cpu_nondeterministic.md (unshard hook 不能跳过)
  - feedback_precompute_cache_miss_fatal.md (cache miss FATAL)
  - feedback_precompute_all_modalities.md (所有 frozen encoder 都要 precompute)
  - feedback_npu_rope_backward_broadcast.md (RoPE 修复)
"""

import logging
import os

import torch

from verl_omni_ext.probes import diagnostic_io as _diag

logger = logging.getLogger(__name__)

_FORWARD_FLAG = "_verl_omni_qwen36_whisper_forward_adapter"

# Marks the module as already having the visual encoder FSDP unshard hook installed
_VISUAL_UNSHARD_FLAG = "_verl_omni_qwen36_visual_fsdp_unshard_hook"
# Marks the module as already having the audio encoder FSDP unshard hook installed
_AUDIO_UNSHARD_FLAG = "_verl_omni_qwen36_audio_fsdp_unshard_hook"

_MM_TRACE = os.environ.get("MINICPMO_MM_TRACE", "0") == "1"
_MM_TRACE_FIRST_N = int(os.environ.get("MINICPMO_MM_TRACE_FIRST_N", "8"))

_MM_FORWARD_CALL_COUNT = 0

# 新模型视觉塔嵌套在 model.visual（与旧模型 vpm 不同）
_VISUAL_ATTR_CANDIDATES = (
    ("model", "visual"),  # 新模型首选：module.model.visual
    ("visual",),          # fallback: module.visual
    ("vpm",),              # 兼容旧模型
    ("vision_tower",),
    ("vision_model",),
)

# 新模型音频塔嵌套在 model.audio_encoder（与旧模型 apm 不同）
_AUDIO_ATTR_CANDIDATES = (
    ("model", "audio_encoder"),  # 新模型首选
    ("apm",),                     # 兼容旧模型
    ("audio_encoder",),
    ("audio_model",),
)

# 新模型音频投影嵌套在 model.audio_projection（与旧模型 audio_projection_layer 不同）
_AUDIO_PROJ_ATTR_CANDIDATES = (
    ("model", "audio_projection"),
    ("audio_projection_layer",),
    ("audio_projection",),
)


def _dist_rank():
    """Global rank, or -1 when torch.distributed is not up yet."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _resolve_submodule(module, candidate_paths):
    """Resolve a nested submodule by trying each candidate path tuple.

    新模型模块路径嵌套：``module.model.visual`` / ``module.model.audio_encoder`` /
    ``module.model.audio_projection`` / ``module.model.language_model``。
    旧模型直接挂在 ``module.vpm`` / ``module.apm``。
    返回 ``(attr_path_tuple, submodule)`` 或 ``(None, None)``。
    """
    for path in candidate_paths:
        sub = module
        ok = True
        for attr in path:
            sub = getattr(sub, attr, None)
            if sub is None or not hasattr(sub, "forward"):
                ok = False
                break
        if ok:
            return path, sub
    return None, None


def _unshard_pre_hook_factory(label, call_counter_holder):
    """Factory for the FSDP unshard pre-hook, parameterized by label.

    逻辑与 MiniCPM-o 5.0 的 ``_apm_unshard_pre_hook`` 完全一致，仅日志 label 与
    flag 名不同。同一个 hook 用于 visual 与 audio encoder：sorted modules →
    callable(unshard) → unshard() each.
    """
    def _hook(module, args):
        call_counter_holder[0] += 1
        call_no = call_counter_holder[0]
        if _diag.limited(f"qwen36_{label}_encoder_forward"):
            _diag.emit(
                f"qwen36_{label}_encoder_forward", call=call_no, rank=_dist_rank(),
                module=type(module).__name__, positional_args=args,
                note="Existing unshard pre-hook observes positional inputs only; see training forward for keyword metadata",
            )

        targets = sorted(
            (
                (_name, child)
                for _name, child in module.named_modules()
                if callable(getattr(child, "unshard", None))
            ),
            key=lambda name_child: name_child[0],
        )

        unsharded, failed = [], []
        for _name, child in targets:
            try:
                child.unshard()
                unsharded.append(_name)
            except Exception as exc:
                failed.append(f"{_name}: {type(exc).__name__}: {exc}")

        if failed or call_no <= _MM_TRACE_FIRST_N or _MM_TRACE:
            _diag.console_or_file("qwen36_forward_message", 
                f"[verl-omni] {label} unshard rank={_dist_rank()} call={call_no} "
                f"n={len(unsharded)}/{len(targets)} failed={failed or '(none)'}",
                flush=True,
            )

    return _hook


_VISUAL_UNSHARD_COUNTER = [0]
_AUDIO_UNSHARD_COUNTER = [0]


def patch_visual_fsdp_unshard(module):
    """Install FSDP unshard hook on the visual encoder (``model.visual``).

    新模型视觉塔嵌套在 ``module.model.visual``，与旧模型 ``module.vpm`` 不同。
    与旧 ``patch_audio_encoder_fsdp_unshard`` 同症同方，仅目标改名。
    """
    if getattr(module, _VISUAL_UNSHARD_FLAG, False):
        return

    path, visual_sub = _resolve_submodule(module, _VISUAL_ATTR_CANDIDATES)
    if visual_sub is None:
        _diag.console_or_file("qwen36_forward_message", 
            "[verl-omni] Qwen3.6-whisper visual encoder not found under "
            f"{[p for p in _VISUAL_ATTR_CANDIDATES]}; skipping FSDP unshard patch. "
            "If training fails with 'mixed torch.Tensor and DTensor' in the visual encoder, "
            "this is why.",
            flush=True,
        )
        return

    hook = _unshard_pre_hook_factory("visual", _VISUAL_UNSHARD_COUNTER)
    visual_sub.register_forward_pre_hook(hook)
    setattr(module, _VISUAL_UNSHARD_FLAG, True)

    fsdp_children = [
        n for n, m in visual_sub.named_modules()
        if callable(getattr(m, "unshard", None))
    ][:20]
    _diag.console_or_file("qwen36_forward_message", 
        f"[verl-omni] Qwen3.6-whisper visual encoder FSDP unshard hook installed on "
        f"{'.'.join(path)} -> {type(visual_sub).__name__} (FSDP children: {fsdp_children}).",
        flush=True,
    )


def patch_audio_encoder_fsdp_unshard(module):
    """Install FSDP unshard hook on the audio encoder (``model.audio_encoder``).

    新模型音频塔嵌套在 ``module.model.audio_encoder``，与旧模型 ``module.apm`` 不同。
    """
    if getattr(module, _AUDIO_UNSHARD_FLAG, False):
        return

    path, audio_sub = _resolve_submodule(module, _AUDIO_ATTR_CANDIDATES)
    if audio_sub is None:
        _diag.console_or_file("qwen36_forward_message", 
            "[verl-omni] Qwen3.6-whisper audio encoder not found under "
            f"{[p for p in _AUDIO_ATTR_CANDIDATES]}; skipping FSDP unshard patch. "
            "If training fails with 'mixed torch.Tensor and DTensor' in the audio encoder, "
            "this is why.",
            flush=True,
        )
        return

    hook = _unshard_pre_hook_factory("audio", _AUDIO_UNSHARD_COUNTER)
    audio_sub.register_forward_pre_hook(hook)
    setattr(module, _AUDIO_UNSHARD_FLAG, True)

    fsdp_children = [
        n for n, m in audio_sub.named_modules()
        if callable(getattr(m, "unshard", None))
    ][:20]
    _diag.console_or_file("qwen36_forward_message", 
        f"[verl-omni] Qwen3.6-whisper audio encoder FSDP unshard hook installed on "
        f"{'.'.join(path)} -> {type(audio_sub).__name__} (FSDP children: {fsdp_children}).",
        flush=True,
    )


def build_qwen36_forward_adapter(module):
    """Install NPU-required patches on the Qwen3.6-whisper module. Returns the module.

    新模型 forward 签名已与 verl 兼容，无需重写 forward。仅安装：
      1. visual encoder FSDP unshard hook
      2. audio encoder FSDP unshard hook
      3. 预计算 embedding 注入（如 MINICPMO_EMBED_CACHE_DIR 启用）

    Idempotent via ``_FORWARD_FLAG``.
    """
    if getattr(module, _FORWARD_FLAG, False):
        return module

    # 1. FSDP unshard hooks — always install, even with precompute cache
    # (compute_log_prob / ref_log_prob paths re-forward without precomputed embeds;
    #  cache miss in those paths needs the unshard hook to avoid DTensor mismatch).
    # See memory: feedback_fsdp2_npu_model_cpu_nondeterministic.md
    patch_visual_fsdp_unshard(module)
    patch_audio_encoder_fsdp_unshard(module)

    # 1c. Patch verl_F.entropy_from_logits to always use chunking
    # verl's prepare_model_outputs calls entropy_from_logits even when
    # entropy_coeff=0 (because _compute_old_log_prob hardcodes calculate_entropy=True).
    # The engine_config.entropy_from_logits_with_chunking flag is not reliably
    # propagated from hydra overrides (hydra rebuilds fsdp_config object).
    # Fix: directly patch verl_F.entropy_from_logits to use chunked version.
    try:
        from verl.utils import torch_functional as verl_F
        if not getattr(verl_F.entropy_from_logits, "_q36_chunked", False):
            _chunk_size = int(os.environ.get("Q36_ENTROPY_CHUNK_SIZE", "512"))
            def _chunked_entropy_from_logits(logits):
                # verl's entropy_from_logits_with_chunking assumes 2D (batch, vocab)
                # but use_remove_padding=False gives 3D (batch, seq_len, vocab).
                # Flatten to 2D, compute chunked, reshape back.
                orig_shape = logits.shape
                if logits.dim() == 3:
                    logits = logits.view(-1, orig_shape[-1])  # (batch*seq, vocab)
                entropy = torch.zeros(logits.shape[0], device=logits.device)
                for i in range(0, logits.shape[0], _chunk_size):
                    logits_chunk = logits[i : i + _chunk_size].float()
                    pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
                    entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
                    entropy[i : i + _chunk_size] = entropy_chunk
                if len(orig_shape) == 3:
                    entropy = entropy.view(orig_shape[0], orig_shape[1])
                return entropy
            _chunked_entropy_from_logits._q36_chunked = True
            verl_F.entropy_from_logits = _chunked_entropy_from_logits
            _diag.console_or_file("qwen36_forward_message", f"[verl-omni] Qwen3.6-whisper: patched entropy_from_logits to use chunking (chunk_size={_chunk_size}, 3D-aware)", flush=True)
    except Exception as e:
        _diag.console_or_file("qwen36_forward_message", f"[verl-omni] Qwen3.6-whisper: failed to patch entropy_from_logits: {e}", flush=True)

    # 1b. Forward pre-hook: normalize pixel_values from list to tensor
    # verl's batch collation may wrap pixel_values back into a list after
    # mm_inputs.py flattened it. modeling_qwen3_5_moe.py's get_image_features
    # does pixel_values.type() which fails on list.
    def _normalize_pixel_values_hook(mod, args, kwargs):
        # verl 的 forward_step 调用 self.module(**model_inputs, use_cache=False)
        # pixel_values / tgt_sizes 直接作为 kwargs 传入，不是包在 data dict 里。
        # 两种调用方式都要支持：
        #   1. module(data=data_dict)  — rollout 路径
        #   2. module(pixel_values=..., tgt_sizes=..., input_ids=...)  — compute_log_prob 路径
        data = kwargs.get("data") or (args[0] if args else None)
        target = data if isinstance(data, dict) else kwargs

        if _diag.limited("qwen36_training_forward"):
            config = getattr(mod, "config", None)
            config_ids = {}
            for label, candidate in (("config", config), ("text_config", getattr(config, "text_config", None))):
                for name in ("image_token_id", "video_token_id", "audio_token_id"):
                    value = getattr(candidate, name, None)
                    if isinstance(value, int):
                        config_ids[f"{label}.{name}"] = value
            _diag.emit(
                "qwen36_training_forward", rank=_dist_rank(), module=type(mod).__name__,
                training=bool(getattr(mod, "training", False)), grad_enabled=torch.is_grad_enabled(),
                input_ids=_diag.token_info(target.get("input_ids")),
                pixel_values=target.get("pixel_values"), image_grid_thw=target.get("image_grid_thw"),
                tgt_sizes=target.get("tgt_sizes"), position_ids=target.get("position_ids"),
                input_features=target.get("input_features"), audio_features=target.get("audio_features"),
                config_token_ids=config_ids, existing_image_token_check=248056,
            )

        pv = target.get("pixel_values")
        if isinstance(pv, list):
            _diag.console_or_file("qwen36_forward_message", f"[verl-omni] _normalize_pixel_values_hook: pixel_values is list, flattening...", flush=True)
            flattened = []
            def _collect(x):
                if isinstance(x, torch.Tensor):
                    flattened.append(x)
                elif isinstance(x, (list, tuple)):
                    for sub in x:
                        _collect(sub)
            _collect(pv)
            if flattened:
                if len(flattened) == 1:
                    target["pixel_values"] = flattened[0]
                else:
                    try:
                        target["pixel_values"] = torch.cat(flattened, dim=0)
                    except Exception:
                        target["pixel_values"] = torch.stack(flattened, dim=0)
                _diag.console_or_file("qwen36_forward_message", f"[verl-omni] _normalize_pixel_values_hook: flattened to {target['pixel_values'].shape}", flush=True)

        ts = target.get("tgt_sizes")
        if isinstance(ts, list):
            flattened_ts = []
            def _collect_ts(x):
                if isinstance(x, torch.Tensor):
                    flattened_ts.append(x)
                elif isinstance(x, (list, tuple)):
                    for sub in x:
                        _collect_ts(sub)
            _collect_ts(ts)
            if flattened_ts:
                if len(flattened_ts) == 1:
                    target["tgt_sizes"] = flattened_ts[0]
                else:
                    try:
                        target["tgt_sizes"] = torch.cat(flattened_ts, dim=0)
                    except Exception:
                        target["tgt_sizes"] = torch.stack(flattened_ts, dim=0)

        # Qwen3-VL 用 image_grid_thw 而不是 tgt_sizes
        gthw = target.get("image_grid_thw")
        if isinstance(gthw, list):
            flattened_gthw = []
            def _collect_gthw(x):
                if isinstance(x, torch.Tensor):
                    flattened_gthw.append(x)
                elif isinstance(x, (list, tuple)):
                    for sub in x:
                        _collect_gthw(sub)
            _collect_gthw(gthw)
            if flattened_gthw:
                if len(flattened_gthw) == 1:
                    target["image_grid_thw"] = flattened_gthw[0]
                else:
                    try:
                        target["image_grid_thw"] = torch.cat(flattened_gthw, dim=0)
                    except Exception:
                        target["image_grid_thw"] = torch.stack(flattened_gthw, dim=0)

        # compute_log_prob 路径：input_ids 没有被 processor 展开（没有 image token），
        # 但 pixel_values 仍然存在。modeling_qwen3_5_moe.py 的 get_placeholder_mask
        # 会检查 input_ids 里有没有 image token——如果没有但 pixel_values 存在，
        # 会报 "Image features and image tokens do not match"。
        # 修复：如果 input_ids 里没有 image token，清除 pixel_values 和 image_grid_thw。
        input_ids = target.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and pv is not None:
            image_token_id = 248056  # <|image_pad|>
            has_image_token = (input_ids == image_token_id).any().item()
            if not has_image_token:
                if _diag.enabled():
                    strict = os.environ.get("MINICPMO_DIAG_STRICT_MM") == "1"
                    if _diag.limited("qwen36_training_image_drop"):
                        _diag.emit(
                            "qwen36_training_image_drop", strict=strict, rank=_dist_rank(),
                            reason="No token 248056; existing hook would remove pixel_values and image_grid_thw",
                            input_ids=_diag.token_info(input_ids), pixel_values=target.get("pixel_values"),
                            image_grid_thw=target.get("image_grid_thw"), checked_image_token_id=image_token_id,
                        )
                    if strict:
                        raise RuntimeError(
                            "[Qwen36 diagnostic strict] Training forward has image pixels but no token 248056; "
                            "refusing to drop pixel_values/image_grid_thw. Check processor/tokenizer expansion and "
                            "the model's configured image_token_id; event=qwen36_training_image_drop"
                        )
                # input_ids 里没有 image token — 这是 compute_log_prob 路径
                # 清除 pixel_values 和 image_grid_thw，跳过 image features 计算
                if "pixel_values" in target:
                    del target["pixel_values"]
                if "image_grid_thw" in target:
                    del target["image_grid_thw"]
                if "pixel_values" in kwargs:
                    del kwargs["pixel_values"]
                if "image_grid_thw" in kwargs:
                    del kwargs["image_grid_thw"]
                _diag.console_or_file("qwen36_forward_message", f"[verl-omni] _normalize_pixel_values_hook: no image token in input_ids, cleared pixel_values+grid_thw (compute_log_prob path)", flush=True)
    module.register_forward_pre_hook(_normalize_pixel_values_hook, with_kwargs=True)

    # 2. Precomputed embedding injection (optional)
    _precompute_cache_dir = os.environ.get("MINICPMO_EMBED_CACHE_DIR", "")
    if _precompute_cache_dir:
        from .precompute import patch_precomputed_embeddings
        patch_precomputed_embeddings(module, _precompute_cache_dir)

    setattr(module, _FORWARD_FLAG, True)
    _diag.console_or_file("qwen36_forward_message", 
        "[verl-omni] Qwen3.6-whisper forward adapter installed: visual+audio FSDP unshard hooks; "
        f"precompute cache={'ON ' + _precompute_cache_dir if _precompute_cache_dir else 'off'}. "
        "forward 签名已兼容 verl keyword-only 调用，无需重写 forward。",
        flush=True,
    )
    return module
