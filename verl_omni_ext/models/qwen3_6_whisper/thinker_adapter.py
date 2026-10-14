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
"""OmniModelBase adapter for Qwen3.6-35B-A3B-whisper thinker training.

Registration key
----------------
``("Qwen3_5MoeForConditionalGeneration", "thinker")``.  The architecture string is
what the checkpoint's ``config.json`` actually reports -- ``architectures:
["Qwen3_5MoeForConditionalGeneration"]``, confirmed against the released
Qwen3.6-35B-A3B-whisper-0814-rl checkpoint -- *not* the vLLM-side class name.
Three distinct names are in play and mixing them up is the easiest way to get
this wrong:

===========================  ==========================================
HF ``architectures[0]``      ``Qwen3_5MoeForConditionalGeneration``
vLLM ``model_arch``          ``MiniCPMO50OmniForConditionalGeneration`` (复用)
vLLM ``model_stage``         ``llm``
verl-omni stage key          ``thinker``                 (this file)
===========================  ==========================================

新模型与 MiniCPM-o 5.0 的关键差异（影响 adapter 实现）：

1. **配置嵌套化**：LM 参数从顶层移入 ``text_config`` 子对象，需要在
   ``configure_model`` 中显式打平一份给 verl 用。
2. **模块路径改名**：
   - 旧 ``vpm.*`` / ``vit_merger.*`` / ``resampler.mlp.*`` → 新 ``model.visual.*``
     + ``model.visual.merger.*``（无独立 ``resampler``，merger 内嵌 visual）
   - 旧 ``apm.*`` + ``audio_projection_layer.*`` → 新 ``model.audio_encoder.*``
     + ``model.audio_projection.*``
   - 旧 ``llm.*`` → 新 ``model.language_model.*`` + 顶层 ``lm_head``
   - 旧 ``tts.*`` → **无**（新模型移除 TTS）
3. **forward 签名标准 transformers 风格**：
   ``forward(self, input_ids, attention_mask, position_ids, **kwargs)`` 而非
   MiniCPM-o 的 ``forward(self, data, **kwargs)``，无需旧模型的 data dict 解包，
   但仍需 NPU patches（vision/audio encoder FSDP unshard hook）。
4. **显式模态 token id**：
   ``image_token_id=248056`` / ``video_token_id=248057`` / ``audio_token_id=248076``
   / ``vision_start_token_id=248053`` / ``vision_end_token_id=248054``。
   旧模型用文本标记（``<image>`` / ``<audio_start|>``）。
5. **新增 video 模态**：``vision_config.temporal_patch_size=2``，
   ``video_preprocessor_config.json`` 用 ``Qwen3VLVideoProcessor``。

详见 ``mdfiles/qwen3_6_whisper_RL训练适配_开发文档.md`` §3。
"""

import logging
import os
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


def _flatten_text_config(model):
    """把 ``config.text_config.*`` 打平到 ``config.*``，给 verl 的 flat-config 读法用。

    新模型 config.json 是嵌套形式（LM 参数在 ``text_config`` 子对象里），
    verl 部分代码用 ``config.hidden_size`` / ``config.num_hidden_layers``
    等 flat 读法。打平一份幂等属性，避免漏改。

    See memory: feedback_qwen35_moe_config_version_field.md
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    if getattr(cfg, "_verl_omni_flattened", False):
        return model

    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        for k in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "vocab_size",
            "num_experts", "num_experts_per_tok", "router_aux_loss_coef",
            "shared_expert_intermediate_size", "moe_intermediate_size",
            "head_dim", "max_position_embeddings", "rms_norm_eps",
            "rope_theta", "partial_rotary_factor", "mrope_interleaved",
            "mrope_section", "layer_types", "linear_conv_kernel_dim",
            "linear_key_head_dim", "linear_num_key_heads",
            "linear_num_value_heads", "linear_value_head_dim",
            "mamba_ssm_dtype", "mtp_num_hidden_layers",
        ):
            if hasattr(tc, k) and not hasattr(cfg, k):
                try:
                    setattr(cfg, k, getattr(tc, k))
                except Exception:
                    pass  # frozen config 不允许 setattr 时跳过

    # 顶层 version 字段缺失会导致 verl-omni get_version_by_config 走 fallback 路径失败。
    # 推荐 A：在 config.json 加 "version":"5.0"；这里做防御性 fallback。
    if not getattr(cfg, "version", None):
        try:
            setattr(cfg, "version", "5.0")
            print(
                "[verl-omni] Qwen3.6-whisper: config.version missing, set to '5.0' "
                "(推荐在 config.json 顶层加 \"version\":\"5.0\" 替代此 fallback).",
                flush=True,
            )
        except Exception:
            pass

    try:
        setattr(cfg, "_verl_omni_flattened", True)
    except Exception:
        pass
    return model


def _report_mtp(module):
    """Log whether the backbone carries an unused MTP head, and optionally freeze it.

    新模型 LM 骨架与 MiniCPM-o 5.0 相同（Qwen3.5-MoE 40 层 / MTP-1），但子模块路径
    改为 ``model.language_model.mtp.*``（推测）。

    与 MiniCPM-o 5.0 版本的差异：MTP 子模块路径不同，需在 ``model.language_model``
    下查找而非 ``module.llm``。
    """
    llm = (
        getattr(module, "language_model", None)
        or getattr(getattr(module, "model", None), "language_model", None)
        or getattr(module, "llm", None)
    )
    mtp = getattr(llm, "mtp", None) if llm is not None else None
    if mtp is None:
        print(
            "[verl-omni] Qwen3.6-whisper MTP check: no MTP head found on this checkpoint, "
            "nothing to do.",
            flush=True,
        )
        return module

    try:
        params = list(mtp.parameters())
        n_elem = sum(p.numel() for p in params)
    except Exception:
        params, n_elem = [], 0

    if os.environ.get("MINICPMO_FREEZE_MTP", "0") == "1":
        for p in params:
            p.requires_grad_(False)
        print(
            f"[verl-omni] Qwen3.6-whisper MTP check: froze MTP ({len(params)} tensors, "
            f"{n_elem / 1e6:.1f}M elements). State dict is unchanged, so rollout weight sync "
            f"still sees them; only optimizer state is saved (~{n_elem * 8 / 2**30:.2f} GiB "
            "fp32 Adam moments, before sharding).",
            flush=True,
        )
    else:
        print(
            f"[verl-omni] Qwen3.6-whisper MTP check: MTP present ({len(params)} tensors, "
            f"{n_elem / 1e6:.1f}M elements) and NOT used by the training forward -- verl reads "
            f"output.logits, the main head. Harmless but not free: ~{n_elem * 8 / 2**30:.2f} GiB "
            "of fp32 Adam moments and one extra slice of every weight sync. "
            "Set MINICPMO_FREEZE_MTP=1 to freeze it (state dict stays identical).",
            flush=True,
        )
    return module


def _report_param_census(module):
    """Print a per-prefix parameter census, plus the node's host-memory counters.

    新模型顶层模块名：
      - ``lm_head`` / ``model.language_model.*`` / ``model.visual.*``
        / ``model.audio_encoder.*`` / ``model.audio_projection.*``

    与旧模型（``vpm.`` / ``apm.`` / ``llm.`` / ``tts.`` / ``resampler.`` /
    ``vit_merger.`` / ``audio_projection_layer.``）的命名差异会影响 census 输出
    的前缀分组，但不影响逻辑本身。
    """
    from verl_omni_ext.models.qwen3_6_whisper.fsdp_cpu_offload_pin import report_host_memory

    buckets = {}
    for name, p in module.named_parameters():
        prefix = name.split(".", 1)[0]
        total, trainable = buckets.get(prefix, (0, 0))
        buckets[prefix] = (total + p.numel(), trainable + (p.numel() if p.requires_grad else 0))

    rendered = " ".join(
        f"{prefix}={total / 1e9:.2f}B{'' if total == trainable else f'(trainable {trainable / 1e9:.2f}B)'}"
        for prefix, (total, trainable) in sorted(buckets.items(), key=lambda kv: -kv[1][0])
    )
    grand = sum(t for t, _ in buckets.values())
    print(f"[verl-omni] Qwen3.6-whisper param census: total={grand / 1e9:.2f}B  {rendered}", flush=True)
    report_host_memory("after configure_model")
    return module


def _load_audio_encoder(module, model_config):
    """Manually load audio encoder + projection from checkpoint.

    The model class ``Qwen3_5MoeForConditionalGeneration`` (transformers built-in)
    includes ``model.visual`` and ``model.language_model`` but NOT
    ``model.audio_encoder`` or ``model.audio_projection``. The checkpoint has these
    weights but they are silently dropped as "UNEXPECTED" during ``from_pretrained``.

    This function:
      1. Creates a ``WhisperEncoder`` from ``config.audio_config``
      2. Creates an ``AudioProjection`` (2-layer MLP) from ``audio_config.d_model``
         to ``text_config.hidden_size``
      3. Loads weights from the checkpoint safetensors files
      4. Attaches to ``module.model.audio_encoder`` and ``module.model.audio_projection``
      5. Freezes all parameters

    Without this, the audio modality is silently lost — the model trains without
    audio, and the precompute cache cannot generate audio embeddings.
    """
    import json
    import os

    import torch
    import torch.nn as nn

    model_sub = getattr(module, "model", None)
    if model_sub is None:
        return

    # Already has audio_encoder — nothing to do
    if getattr(model_sub, "audio_encoder", None) is not None:
        print("[verl-omni] Qwen3.6-whisper: audio_encoder already present on model, "
              "skipping manual load.", flush=True)
        return

    model_path = getattr(model_config, "path", None)
    if not model_path:
        print("[verl-omni] Qwen3.6-whisper: no model_path in config, cannot load "
              "audio encoder.", flush=True)
        return

    cfg = getattr(module, "config", None)
    if cfg is None:
        return

    audio_cfg = getattr(cfg, "audio_config", None)
    if audio_cfg is None:
        print("[verl-omni] Qwen3.6-whisper: no audio_config in model config, "
              "cannot load audio encoder.", flush=True)
        return

    # --- 1. Create WhisperEncoder from audio_config ---
    try:
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
    except ImportError as e:
        print(f"[verl-omni] Qwen3.6-whisper: cannot import WhisperEncoder: {e}. "
              "Audio encoder will not be loaded.", flush=True)
        return

    # Convert to plain dict for WhisperConfig construction
    if hasattr(audio_cfg, "to_dict"):
        audio_cfg_dict = audio_cfg.to_dict()
    elif isinstance(audio_cfg, dict):
        audio_cfg_dict = dict(audio_cfg)
    else:
        audio_cfg_dict = dict(vars(audio_cfg))

    # WhisperConfig accepts most keys via **kwargs (PretrainedConfig), but
    # some keys like "architectures", "_name_or_path" are harmless attributes.
    try:
        whisper_cfg = WhisperConfig(**audio_cfg_dict)
    except TypeError:
        # Filter to known WhisperConfig fields
        import inspect
        valid = set(inspect.signature(WhisperConfig.__init__).parameters.keys())
        filtered = {k: v for k, v in audio_cfg_dict.items() if k in valid}
        whisper_cfg = WhisperConfig(**filtered)

    audio_encoder = WhisperEncoder(whisper_cfg)

    # --- 2. Create AudioProjection (2-layer MLP) ---
    d_model = audio_cfg_dict.get("d_model", 1024)
    text_hidden = 2048
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        text_hidden = getattr(text_cfg, "hidden_size", None) or (
            text_cfg.get("hidden_size", 2048) if isinstance(text_cfg, dict) else 2048
        )

    class AudioProjection(nn.Module):
        """2-layer MLP: d_model → hidden_size → hidden_size."""
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)
        def forward(self, x):
            return self.linear2(torch.nn.functional.gelu(self.linear1(x)))

    audio_projection = AudioProjection(d_model, text_hidden)

    # --- 3. Load weights from checkpoint safetensors ---
    from safetensors.torch import load_file as load_safetensors

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        weight_map = idx.get("weight_map", {})
        # Find safetensors files containing audio keys
        audio_files = set()
        for key in weight_map:
            if "audio_encoder" in key or "audio_projection" in key:
                audio_files.add(weight_map[key])
        all_state = {}
        for f in sorted(audio_files):
            all_state.update(load_safetensors(os.path.join(model_path, f)))
    else:
        st_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st_path):
            all_state = load_safetensors(st_path)
        else:
            print(f"[verl-omni] Qwen3.6-whisper: no safetensors found at {model_path}, "
                  "cannot load audio encoder weights.", flush=True)
            return

    # Load audio encoder weights (strip 'model.audio_encoder.' prefix)
    encoder_state = {}
    for key, val in all_state.items():
        if key.startswith("model.audio_encoder."):
            new_key = key.replace("model.audio_encoder.", "", 1)
            encoder_state[new_key] = val
    if encoder_state:
        missing, unexpected = audio_encoder.load_state_dict(encoder_state, strict=False)
        print(f"[verl-omni] Qwen3.6-whisper audio encoder loaded: {len(encoder_state)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"  missing keys (first 5): {missing[:5]}", flush=True)
        if unexpected:
            print(f"  unexpected keys (first 5): {unexpected[:5]}", flush=True)

    # Load audio projection weights (strip 'model.audio_projection.' prefix)
    proj_state = {}
    for key, val in all_state.items():
        if key.startswith("model.audio_projection."):
            new_key = key.replace("model.audio_projection.", "", 1)
            proj_state[new_key] = val
    if proj_state:
        missing, unexpected = audio_projection.load_state_dict(proj_state, strict=False)
        print(f"[verl-omni] Qwen3.6-whisper audio projection loaded: {len(proj_state)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    # --- 4. Attach to module.model ---
    model_sub.audio_encoder = audio_encoder
    model_sub.audio_projection = audio_projection

    # --- 5. Freeze ---
    for p in audio_encoder.parameters():
        p.requires_grad_(False)
    for p in audio_projection.parameters():
        p.requires_grad_(False)
    audio_encoder.eval()
    audio_projection.eval()

    n_enc = sum(p.numel() for p in audio_encoder.parameters())
    n_proj = sum(p.numel() for p in audio_projection.parameters())
    print(f"[verl-omni] Qwen3.6-whisper audio encoder attached: "
          f"audio_encoder ({n_enc / 1e6:.1f}M, frozen) + "
          f"audio_projection ({n_proj / 1e6:.1f}M, frozen)", flush=True)


@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen36WhisperThinkerAdapter(OmniModelBase):
    """verl-omni adapter for Qwen3.6-35B-A3B-whisper thinker-only GSPO training.

    The thinker's LLM backbone *is* Qwen3.5-MoE (与 MiniCPM-o 5.0 完全相同)，所以
    memory, precision and micro-batch 结论直接继承（fp32 master weights, one
    micro-batch per device, ``weight_nz_mode=0``）。差异在视觉塔：新模型使用
    Qwen3-VL deepstack 视觉编码器（``model.visual.*``），patch=16，支持 video；
    旧 MiniCPM-o 使用 SigLIP UHD NaViT（``vpm.*``），patch=14，无 video。
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Strip nothing -- in particular, do not strip ``tts``.

        新模型本身无 TTS，但保持返回空列表以与接口一致：RL rollout 用 thinker-only
        pipeline，TTS 即便存在也不会被加载。新模型已移除 TTS 组件，此返回值实际
        无影响。
        """
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        """Install the forward-signature adapter, then defer to the base.

        新模型 forward 签名已是标准 transformers 风格
        ``forward(self, input_ids, attention_mask, position_ids, **kwargs)``，无需
        旧 MiniCPM-o 的 data-dict 解包层。但仍需安装 NPU 必需 patches：

        1. **FSDP unshard hooks**：``model.visual`` 与 ``model.audio_encoder`` 的
           ``nn.Embedding`` 子模块被 verl FSDP2 单独 wrap，forward 直接读
           ``.weight`` 时 DTensor 不解。与旧模型同症同方，仅目标改名。
        2. **预计算缓存**（可选）：``MINICPMO_EMBED_CACHE_DIR`` 启用时冻结 encoder
           并 patch ``get_audio_embedding``/``get_vision_embedding``。
        3. **RoPE 修复**：LM 骨架相同，NPU ``aclnnRotaryPositionEmbeddingGrad``
           broadcast bug 仍存在，需替换为标准 PyTorch RoPE。

        See memory: feedback_npu_rope_backward_broadcast.md,
                    feedback_fsdp2_npu_model_cpu_nondeterministic.md
        """
        from verl_omni_ext.models.qwen3_6_whisper.forward import build_qwen36_forward_adapter

        module = super().configure_model(module, model_config)
        _flatten_text_config(module)
        _load_audio_encoder(module, model_config)
        cls._configure_multimodal_encoders(module)
        module = build_qwen36_forward_adapter(module)
        _report_mtp(module)
        _report_param_census(module)
        return module

    @classmethod
    def _configure_multimodal_encoders(cls, module):
        """Configure freeze/tune switches for visual and audio encoders.

        新模型模块路径：
          - 视觉：``module.model.visual.*`` + ``module.model.visual.merger.*``
          - 音频：``module.model.audio_encoder.*`` + ``module.model.audio_projection.*``

        与旧 MiniCPM-o 差异：
          - 旧 ``module.vpm`` / ``module.vit_merger`` / ``module.resampler`` →
            新 ``module.model.visual``（merger 内嵌，无独立 resampler）
          - 旧 ``module.apm`` / ``module.audio_projection_layer`` →
            新 ``module.model.audio_encoder`` / ``module.model.audio_projection``
        """
        tune_vision = os.environ.get("MINICPMO_TUNE_VISION", "0") == "1"
        tune_audio = os.environ.get("MINICPMO_TUNE_AUDIO", "0") == "1"

        # 视觉塔在新模型下嵌套在 model.visual
        visual_sub = getattr(module, "model", None)
        if visual_sub is not None:
            visual_sub = getattr(visual_sub, "visual", None)
        if visual_sub is None:
            # 兼容性 fallback
            for name in ("vpm", "visual", "vision_model"):
                visual_sub = getattr(module, name, None)
                if visual_sub is not None:
                    break
        if visual_sub is not None:
            visual_sub.train() if tune_vision else visual_sub.eval()
            for p in visual_sub.parameters():
                p.requires_grad_(tune_vision)

        # 音频塔在新模型下嵌套在 model.audio_encoder / model.audio_projection
        model_sub = getattr(module, "model", None)
        audio_sub = getattr(model_sub, "audio_encoder", None) if model_sub else None
        audio_proj = getattr(model_sub, "audio_projection", None) if model_sub else None
        if audio_sub is None:
            for name in ("apm", "audio_encoder", "audio_model"):
                audio_sub = getattr(module, name, None)
                if audio_sub is not None:
                    break
        if audio_proj is None:
            for name in ("audio_projection_layer", "audio_projection"):
                audio_proj = getattr(module, name, None)
                if audio_proj is not None:
                    break

        for sub in (audio_sub, audio_proj):
            if sub is None:
                continue
            sub.train() if tune_audio else sub.eval()
            for p in sub.parameters():
                p.requires_grad_(tune_audio)

        print(
            f"[verl-omni] Qwen3.6-whisper multimodal encoder status: "
            f"Visual={'TUNE' if tune_vision else 'FREEZE'}, "
            f"Audio={'TUNE' if tune_audio else 'FREEZE'}",
            flush=True,
        )

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Standard HF tokenizer, loaded through remote code.

        新模型 ``tokenizer_class: "Qwen2Tokenizer"``，不再使用 MiniCPM-o 的
        ``MiniCPMOTokenizerFast``。但 HF ``AutoTokenizer.from_pretrained`` 仍能
        正确解析（trust_remote_code=True）。vocab 248320（+176 新 token）。
        """
        from verl.utils import hf_tokenizer

        return hf_tokenizer(
            model_path,
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
        )

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Return the real HF processor -- ``hf_processor`` alone cannot.

        ``processing_minicpmo.py`` 与 MiniCPM-o 5.0 byte-identical，processor
        类名仍为 ``MiniCPMOProcessor``，不在 verl 的 ``hf_processor`` 白名单里，
        因此 ``hf_processor`` 仍返回 None，需走 ``build_minicpmo_processor``
        fallback（直接复用 MiniCPM-o 5.0 的实现）。

        新增 video 模态由 ``MiniCPMOProcessor`` 内置支持（与 vLLM-Omni 一致），
        无需在此处理。
        """
        from verl.utils import hf_processor

        from verl_omni_ext.models.qwen3_6_whisper.mm_inputs import apply_minicpmo_mm_inputs_patch
        from verl_omni_ext.models.qwen3_6_whisper.processor import build_minicpmo_processor

        trust_remote_code = getattr(model_config, "trust_remote_code", False)
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code)
        if processor is not None:
            apply_minicpmo_mm_inputs_patch(processor)
            return processor

        if os.environ.get("MINICPMO_DISABLE_PROCESSOR_FALLBACK", "0") == "1":
            logger.warning(
                "MINICPMO_DISABLE_PROCESSOR_FALLBACK=1: leaving the processor as None. "
                "RLHFDataset will measure prompt length with the tokenizer instead of the "
                "processor. Text-only training is unaffected; any row with an "
                "images/videos/audios column will trip `assert self.processor is not None`."
            )
            return None

        processor = build_minicpmo_processor(model_path, trust_remote_code=trust_remote_code)
        if processor is None:
            logger.warning(
                "No processor could be built for %s. Text-only training still works, but any "
                "dataset row with an image/video/audio column will trip RLHFDataset's "
                "`assert self.processor is not None`.",
                model_path,
            )
        else:
            apply_minicpmo_mm_inputs_patch(processor)
        return processor