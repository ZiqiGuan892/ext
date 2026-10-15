"""Qwen3.6-35B-A3B-whisper 插件包（槽位① thinker | 槽位② rollout | L2 补丁 | 槽位④ 数据集）。

新模型 Qwen3.6-35B-A3B-whisper-0814-rl 与 MiniCPM-o 5.0 共享同一 Qwen3.5-MoE
LM 骨架与同一份 ``processing_minicpmo.py`` processor；差异在视觉塔（Qwen3-VL
deepstack + video 模态，模块路径 ``model.visual.*``）、音频编码器模块路径
（``model.audio_encoder.*`` 而非 ``apm.*``），且无 TTS。

模块级补丁必须在任何 remote code 被 import 之前执行——``from_pretrained`` 在
``configure_model`` 之前（fsdp/omni_impl.py），所以让 transformers 能 import 得动
remote code 的补丁放在这里，不是放在 ``configure_model`` 里。

顺序有讲究：
  1. ``apply_minicpmo_auto_register_guard`` 必须先落地——checkpoint 的
     processing_minicpmo.py 用 str 调 register，新版 transformers 拒收，连带
     modeling 的 import 都进不来。Processor 文件与 MiniCPM-o 5.0 byte-identical，
     风险相同，直接复用同一份 patch。
  2. ``apply_minicpmo_automodel_fallback`` 在 config.json auto_map 缺
     "AutoModelForMultimodalLM" 时兜底——新模型 auto_map 仅映射 AutoProcessor，
     模型类由环境内置 ``Qwen3_5MoeForConditionalGeneration`` 提供；fallback 路径
     不会触发，但保留以防环境变动。
  3. ``apply_fsdp2_cpu_offload_pin_memory_patch`` 与 transformers 无关：verl
     硬编码 ``CPUOffloadPolicy(pin_memory=True)``，Ascend 上会导致 EH0002 崩溃。
     LM 骨架与 MiniCPM-o 相同，问题相同，复用同一补丁。
"""
import logging
import os
import torch

from verl_omni_ext.probes import diagnostic_io as _diag
from verl_omni_ext.probes import gdn_trace as _gdn_trace

from verl_omni_ext.models.qwen3_6_whisper.fsdp_cpu_offload_pin import apply_fsdp2_cpu_offload_pin_memory_patch
from verl_omni_ext.models.qwen3_6_whisper.patches import (
    apply_minicpmo_auto_register_guard,
    apply_minicpmo_automodel_fallback,
)

logger = logging.getLogger(__name__)


def _diag_mm_fallback(event, message, **fields):
    """Observe existing corrective behavior, or fail before it in explicit strict runs."""
    if not _diag.enabled():
        return
    strict = os.environ.get("MINICPMO_DIAG_STRICT_MM") == "1"
    if _diag.limited(event):
        for name in ("prompt_ids", "input_tokens"):
            if name in fields:
                fields[name] = _diag.token_info(fields[name])
        _diag.emit(event, strict=strict, message=message, **fields)
    if strict:
        raise RuntimeError(f"[Qwen36 diagnostic strict] {message}; event={event}")


def _diag_gdn_schema(raw_op):
    """Read the actual torch overload schema without invoking backend methods."""
    try:
        schema = getattr(raw_op, "_schema", None)
        return _diag.text_info(str(schema)) if schema is not None else None
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


def apply_batch_invariant_sum_patch():
    try:
        import vllm_ascend.batch_invariant as bi_mod

        if hasattr(bi_mod, "reduce_sum") and not getattr(bi_mod, "_bool_sum_patched", False):
            orig_reduce_sum = bi_mod.reduce_sum

            def safe_reduce_sum(x, *args, **kwargs):
                if isinstance(x, torch.Tensor) and x.dtype == torch.bool:
                    out = orig_reduce_sum(x.to(torch.float32), *args, **kwargs)
                    return out.to(torch.long)
                return orig_reduce_sum(x, *args, **kwargs)

            bi_mod.reduce_sum = safe_reduce_sum
            bi_mod._bool_sum_patched = True
            logger.info(f"[PID {os.getpid()}] Successfully patched vllm_ascend.batch_invariant.reduce_sum for DT_BOOL!")
    except Exception as e:
        logger.warning(f"Failed to patch batch_invariant reduce_sum: {e}")

def ensure_ascend_gdn_bf16_patched():
    try:
        import vllm_ascend.ops.gdn as gdn_mod

        if not hasattr(torch.ops, "_C_ascend"):
            if _diag.limited("qwen36_gdn_patch_unavailable", 1):
                _diag.emit("qwen36_gdn_patch_unavailable", reason="torch.ops._C_ascend missing")
            return

        c_ascend = torch.ops._C_ascend
        if not hasattr(c_ascend, "npu_recurrent_gated_delta_rule"):
            if _diag.limited("qwen36_gdn_patch_unavailable", 1):
                _diag.emit("qwen36_gdn_patch_unavailable", reason="npu_recurrent_gated_delta_rule missing")
            return

        target_op = c_ascend.npu_recurrent_gated_delta_rule
        if getattr(target_op, "_is_bf16_patched", False):
            return

        orig_call = target_op.default if hasattr(target_op, "default") else target_op

        class SafeGDNOpWrapper:
            def __init__(self, raw_op):
                self._raw_op = raw_op
                self._is_bf16_patched = True
                # A loaded-code marker, unlike a hash of files replaced on disk.
                self._gdn_probe_revision = _gdn_trace.PROBE_REVISION

            def __call__(self, *args, **kwargs):
                native = _diag.enabled() and os.environ.get("MINICPMO_DIAG_GDN_NATIVE") == "1"
                observe = _diag.limited("qwen36_gdn_call")
                if native:
                    if observe:
                        _diag.emit(
                            "qwen36_gdn_call", native=True, positional_args=args,
                            keyword_args=kwargs, keyword_args_after=kwargs,
                            state_cast_back=False, state_fp32_to_bf16=False,
                            backend=_diag.module_info(gdn_mod),
                            raw_op_schema=_diag_gdn_schema(self._raw_op),
                            probe_revision=self._gdn_probe_revision,
                            numeric_error_measured=False,
                        )
                    # AB experiment only: the installed backend may reject these dtypes.
                    # Positional arguments retain their original order and meaning.
                    trace = _gdn_trace.begin(self._raw_op, args, kwargs, kwargs, native=True)
                    try:
                        out = self._raw_op(*args, **kwargs)
                    except Exception as exc:
                        _gdn_trace.failed(trace, "kernel", exc)
                        raise
                    _gdn_trace.after_kernel(trace, self._raw_op, args, kwargs, out, [])
                    _gdn_trace.finish(trace, [])
                    return out
                cast_back_list = []

                new_args = list(args)

                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        if k == "g":
                            if v.dtype != torch.float32:
                                new_kwargs[k] = v.to(torch.float32)
                            else:
                                new_kwargs[k] = v
                        elif k in ("state", "beta", "query", "key", "value"):
                            if v.dtype != torch.bfloat16:
                                v_bf16 = v.to(torch.bfloat16)
                                if k == "state":
                                    cast_back_list.append((v_bf16, v))
                                new_kwargs[k] = v_bf16
                            else:
                                new_kwargs[k] = v
                        else:
                            new_kwargs[k] = v
                    else:
                        new_kwargs[k] = v

                if observe:
                    _diag.emit(
                        "qwen36_gdn_call", native=False, positional_args=args,
                        keyword_args=kwargs, keyword_args_after=new_kwargs,
                        state_cast_back=bool(cast_back_list),
                        state_fp32_to_bf16=any(original.dtype == torch.float32 for _, original in cast_back_list),
                        backend=_diag.module_info(gdn_mod),
                        raw_op_schema=_diag_gdn_schema(self._raw_op),
                        probe_revision=self._gdn_probe_revision,
                        numeric_error_measured=False,
                    )
                trace = _gdn_trace.begin(self._raw_op, args, kwargs, new_kwargs)
                try:
                    out = self._raw_op(*new_args, **new_kwargs)
                except Exception as exc:
                    _gdn_trace.failed(trace, "kernel", exc)
                    raise
                _gdn_trace.after_kernel(trace, self._raw_op, new_args, new_kwargs, out, cast_back_list)

                try:
                    for bf16_t, orig_fp32_t in cast_back_list:
                        orig_fp32_t.copy_(bf16_t)
                except Exception as exc:
                    _gdn_trace.failed(trace, "copyback", exc)
                    raise
                _gdn_trace.finish(trace, cast_back_list)
                return out

            def __getattr__(self, name):
                return getattr(self._raw_op, name)

        wrapped_op = SafeGDNOpWrapper(orig_call)

        try:
            c_ascend.npu_recurrent_gated_delta_rule = wrapped_op
        except Exception:
            pass

        if hasattr(target_op, "default"):
            try:
                target_op.default = wrapped_op
            except Exception:
                pass

        if not _diag.enabled():
            logger.info(f"[PID {os.getpid()}] Hooked torch.ops._C_ascend.npu_recurrent_gated_delta_rule directly!")
        if _diag.limited("qwen36_gdn_patch_install", 1):
            _diag.emit(
                "qwen36_gdn_patch_install", backend=_diag.module_info(gdn_mod),
                probe_revision=wrapped_op._gdn_probe_revision,
                packet_hooked=getattr(c_ascend.npu_recurrent_gated_delta_rule, "_is_bf16_patched", False),
                default_hooked=getattr(getattr(target_op, "default", None), "_is_bf16_patched", False),
            )

    except Exception as e:
        if not _diag.enabled():
            logger.warning(f"[PID {os.getpid()}] Failed to hook torch.ops._C_ascend: {e}")
        if _diag.limited("qwen36_gdn_patch_unavailable", 1):
            _diag.emit("qwen36_gdn_patch_unavailable", error=str(e))

ensure_ascend_gdn_bf16_patched()
apply_batch_invariant_sum_patch()

def apply_model_stage_mapping_patch():
    try:
        import sys
        from vllm_omni.model_executor.models.minicpmo_5_0 import minicpmo_5_0_omni
        target_cls = minicpmo_5_0_omni.MiniCPMO50OmniForConditionalGeneration
        if getattr(target_cls, "_stage_mapped", False):
            return

        orig_init = target_cls.__init__

        def safe_init(self, *args, **kwargs):
            ensure_ascend_gdn_bf16_patched()
            apply_batch_invariant_sum_patch()
            vllm_cfg = kwargs.get("vllm_config", None)
            new_args = list(args)

            if vllm_cfg is None and new_args:
                for idx, arg in enumerate(new_args):
                    if hasattr(arg, "model_config"):
                        vllm_cfg = new_args.pop(idx)
                        break

            if vllm_cfg is None:
                cfg_candidate = kwargs.get("config", None)
                if hasattr(cfg_candidate, "model_config"):
                    vllm_cfg = cfg_candidate

            if vllm_cfg is None:
                frame = sys._getframe(1)
                while frame:
                    if "vllm_config" in frame.f_locals:
                        vllm_cfg = frame.f_locals["vllm_config"]
                        break
                    frame = frame.f_back

            if vllm_cfg is not None:
                kwargs["vllm_config"] = vllm_cfg
                if hasattr(vllm_cfg, "model_config"):
                    stage = getattr(vllm_cfg.model_config, "model_stage", None)
                    if stage in ("thinker", "text"):
                        vllm_cfg.model_config.model_stage = "llm"

            orig_init(self, *new_args, **kwargs)

        target_cls.__init__ = safe_init
        target_cls._stage_mapped = True
        logger.info("[verl-omni-ext] Successfully injected vllm_config keyword-only resolver & stage mapping.")
    except Exception as e:
        logger.warning(f"Failed to patch stage mapping: {e}")

def apply_qwen36_vllm_prompt_updates_patch():
    try:
        import vllm.multimodal.processing.processor as vllm_proc_mod
        NativePlaceholderRange = getattr(vllm_proc_mod, "PlaceholderRange", None)

        class SafePlaceholderItem:
            __slots__ = ("offset", "length")
            def __init__(self, offset: int, length: int):
                self.offset = offset
                self.length = length

            def to_range(self):
                if NativePlaceholderRange is not None:
                    try:
                        return NativePlaceholderRange(offset=self.offset, length=self.length)
                    except TypeError:
                        return NativePlaceholderRange(self.offset, self.length)
                return self

            def __repr__(self):
                return f"SafePlaceholderItem(offset={self.offset}, length={self.length})"

        # 结合 Tokenizer 真实输出
        MODALITY_TOKEN_IDS = {
            "image": [248078, 248056],
            "video": [248057],
            "audio": [248070, 248071, 248076, 248077, 248096],
        }

        def _build_placeholders_from_ids(prompt_ids, modality, expected_count=None):
            target_ids = MODALITY_TOKEN_IDS.get(modality, [])
            target_set = set(target_ids)

            placeholders = []
            in_span = False
            start_idx = 0
            for idx, tid in enumerate(prompt_ids):
                if tid in target_set:
                    if not in_span:
                        in_span = True
                        start_idx = idx
                else:
                    if in_span:
                        in_span = False
                        placeholders.append(SafePlaceholderItem(offset=start_idx, length=idx - start_idx))
            if in_span:
                placeholders.append(SafePlaceholderItem(offset=start_idx, length=len(prompt_ids) - start_idx))

            if not placeholders:
                return placeholders

            # 核心对齐逻辑：
            # 若实际媒体对象为 1 个（expected_count == 1），但被断成了多个片段：
            # 1. 数量必须收敛为 1，杜绝 input_processor 发生 IndexError: list index out of range
            # 2. 长度必须是所有片段的真实长度之和（sum(len)），杜绝首尾相减算出的 1151 != 604 错误！
            if expected_count == 1 and len(placeholders) > 1:
                _diag_mm_fallback(
                    "qwen36_placeholder_span_merge", "Merging disjoint modality token spans into one range",
                    modality=modality, expected_count=expected_count,
                    spans=[{"offset": p.offset, "length": p.length} for p in placeholders],
                )
                total_valid_len = sum(p.length for p in placeholders)
                return [SafePlaceholderItem(offset=placeholders[0].offset, length=total_valid_len)]
            elif expected_count is not None and expected_count > 0 and len(placeholders) > expected_count:
                _diag_mm_fallback(
                    "qwen36_placeholder_span_truncate", "Dropping excess modality token spans",
                    modality=modality, expected_count=expected_count, actual_count=len(placeholders),
                )
                return placeholders[:expected_count]

            return placeholders

        target_cls = getattr(vllm_proc_mod, "BaseMultiModalProcessor", None)
        if target_cls and not getattr(target_cls, "_patched_qwen36_final_align_v3", False):
            orig_apply = target_cls._apply_prompt_updates
            orig_validate = getattr(target_cls, "_validate_mm_placeholders", None)

            def safe_apply(self, prompt_ids, *args, **kwargs):
                try:
                    return orig_apply(self, prompt_ids, *args, **kwargs)
                except (AssertionError, TypeError, Exception) as e:
                    _diag_mm_fallback(
                        "qwen36_prompt_update_fallback", "Native prompt update failed; recovering placeholders from token IDs",
                        error=f"{type(e).__name__}: {e}", prompt_ids=prompt_ids,
                        processor=type(self).__name__, arguments=args, keyword_arguments=kwargs,
                    )
                    mm_items = None
                    if args:
                        mm_items = args[0]
                    elif "mm_items" in kwargs:
                        mm_items = kwargs["mm_items"]

                    recovered = {}
                    for mod in ("image", "audio", "video"):
                        items_list = []
                        if mm_items is not None:
                            if hasattr(mm_items, "get"):
                                items_list = mm_items.get(mod) or []
                            elif hasattr(mm_items, mod):
                                items_list = getattr(mm_items, mod) or []

                        # 只有真正传入该媒体对象时才处理，并传递其实际对象数量进行对齐
                        if items_list:
                            expected_count = len(items_list) if isinstance(items_list, (list, tuple)) else 1
                            spans = _build_placeholders_from_ids(prompt_ids, mod, expected_count=expected_count)
                            if spans:
                                recovered[mod] = spans

                    return prompt_ids, recovered

            def safe_validate(self, mm_placeholders, mm_item_counts):
                if orig_validate:
                    try:
                        return orig_validate(self, mm_placeholders, mm_item_counts)
                    except RuntimeError as e:
                        if "Expected there to be" in str(e):
                            _diag_mm_fallback(
                                "qwen36_placeholder_validation_ignored", "Ignoring native placeholder count validation error",
                                error=str(e), placeholders=mm_placeholders, item_counts=mm_item_counts,
                            )
                            return
                        raise e

            target_cls._apply_prompt_updates = safe_apply
            if orig_validate:
                target_cls._validate_mm_placeholders = safe_validate
            target_cls._patched_qwen36_final_align_v3 = True
            _diag.console_or_file("qwen36_adapter_message", f"[verl-omni][PID {os.getpid()}] Patch updated: Modality strictly conditioned & slice spans merged to match count.", flush=True)

    except Exception as e:
        _diag.console_or_file("qwen36_adapter_message", f"[verl-omni] Patch failed: {e}", flush=True)


def apply_mrope_text_len_fix_patch():
    """Retain the existing flat-position fallback and optionally diagnose it.

    The crash: ``np.broadcast_to(np.arange(text_len), (3, text_len))`` when
    ``text_len = offset - st`` is negative (because safe_apply recovered
    placeholders have offsets that don't perfectly align with what
    _q36_iter_mm_grid_hw expects). The existing fallback replaces all three
    position axes with the same linear sequence; it does not preserve spatial
    positions. Diagnostic strict mode raises before that replacement.
    """
    try:
        import numpy as np
        import torch
        from vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni_llm import (
            MiniCPMO50OmniLLMForConditionalGeneration,
        )
        if getattr(MiniCPMO50OmniLLMForConditionalGeneration, "_mrope_text_len_fixed", False):
            return
        orig_get_mrope = MiniCPMO50OmniLLMForConditionalGeneration.get_mrope_input_positions

        def safe_get_mrope(self, input_tokens, mm_features, **kwargs):
            try:
                return orig_get_mrope(self, input_tokens, mm_features, **kwargs)
            except ValueError as e:
                if "broadcast shape must be non-negative" in str(e):
                    _diag_mm_fallback(
                        "qwen36_mrope_flat_fallback", "Replacing failed multimodal position IDs with repeated flat 1D positions",
                        error=str(e), input_tokens=input_tokens,
                        mm_features=mm_features, keyword_arguments=kwargs,
                    )
                    # text_len went negative; fall back to flat 1D positions
                    text_len = len(input_tokens)
                    positions = np.broadcast_to(
                        np.arange(text_len), (3, text_len)
                    )
                    delta = int(positions.max()) + 1 - text_len
                    return torch.tensor(positions, dtype=torch.long), delta
                raise

        MiniCPMO50OmniLLMForConditionalGeneration.get_mrope_input_positions = safe_get_mrope
        MiniCPMO50OmniLLMForConditionalGeneration._mrope_text_len_fixed = True
        _diag.console_or_file("qwen36_adapter_message", "[verl-omni] Patched get_mrope_input_positions: fallback to flat 1D on negative text_len", flush=True)
    except Exception as e:
        _diag.console_or_file("qwen36_adapter_message", f"[verl-omni] Failed to patch get_mrope_input_positions: {e}", flush=True)

import torch
def patch_qwen_model_pixel_values_forward():
    """在不修改 verl 和 transformers 源码的前提下，自动将 micro_batch 中被 verl 打包为 list 的 pixel_values

    在模型前向时安全转换为 torch.Tensor。
    """
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qm_mod

        # 获取可能涉及视觉特征提取的模型类
        target_classes = []
        for name in ("Qwen3_5_MoeModel", "Qwen3_5_MoeForCausalLM"):
            cls = getattr(qm_mod, name, None)
            if cls is not None:
                target_classes.append(cls)

        for target_cls in target_classes:
            if getattr(target_cls, "_patched_pixel_values_list_safe", False):
                continue

            orig_get_image_features = getattr(
                target_cls, "get_image_features", None
            )
            if orig_get_image_features is not None:

                def safe_get_image_features(
                    self, pixel_values, *args, **kwargs
                ):
                    if isinstance(pixel_values, (list, tuple)):
                        flattened = []

                        def _collect(x):
                            if isinstance(x, torch.Tensor):
                                flattened.append(x)
                            elif isinstance(x, (list, tuple)):
                                for sub in x:
                                    _collect(sub)

                        _collect(pixel_values)

                        if flattened:
                            try:
                                if flattened[0].dim() == 4:
                                    pixel_values = torch.cat(flattened, dim=0)
                                else:
                                    pixel_values = torch.stack(flattened, dim=0)
                            except Exception:
                                pixel_values = torch.cat(
                                    [
                                        f.view(-1, *f.shape[-3:])
                                        for f in flattened
                                    ],
                                    dim=0,
                                )

                    return orig_get_image_features(
                        self, pixel_values, *args, **kwargs
                    )

                target_cls.get_image_features = safe_get_image_features
                target_cls._patched_pixel_values_list_safe = True
                _diag.console_or_file("qwen36_adapter_message", 
                    f"[verl-omni] Successfully patched {target_cls.__name__}.get_image_features (handles list input).",
                    flush=True,
                )

    except Exception as e:
        _diag.console_or_file("qwen36_adapter_message", f"[verl-omni] Failed to patch get_image_features: {e}", flush=True)

def patch_vllm_merge_multimodal_embeddings():
    """彻底解决图音混合或多模态切片时，特征数与占位符掩码数量不匹配 (如 747 vs 1268) 导致的

    RuntimeError: shape mismatch。
    """
    try:
        from vllm.model_executor.models import utils as model_utils
        if getattr(model_utils, "_patched_merge_mm_embeds_safe", False):
            return

        orig_merge = model_utils._merge_multimodal_embeddings

        def safe_merge_multimodal_embeddings(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: list[torch.Tensor],
            placeholder_token_id: int | list[int] | None = None,
            *,
            is_multimodal: torch.Tensor | None = None,
            **kwargs
        ):
            if not multimodal_embeddings:
                return inputs_embeds

            # 展平待插入的多模态特征
            valid_mm_embeds = [e for e in multimodal_embeddings if e is not None and e.numel() > 0]
            if not valid_mm_embeds:
                return inputs_embeds

            mm_embeds_flat = torch.cat(valid_mm_embeds, dim=0)
            target_count = mm_embeds_flat.shape[0]

            # 获取当前的布尔掩码
            if is_multimodal is not None:
                mask = is_multimodal
            else:
                input_ids = kwargs.get("input_ids")
                if input_ids is not None and placeholder_token_id is not None:
                    if isinstance(placeholder_token_id, int):
                        mask = (input_ids == placeholder_token_id)
                    else:
                        mask = torch.isin(input_ids, torch.tensor(placeholder_token_id, device=input_ids.device))
                else:
                    merge_kwargs = dict(kwargs)
                    if placeholder_token_id is not None:
                        merge_kwargs["placeholder_token_id"] = placeholder_token_id
                    if is_multimodal is not None:
                        merge_kwargs["is_multimodal"] = is_multimodal
                    return orig_merge(inputs_embeds, multimodal_embeddings, **merge_kwargs)

            num_placeholders = mask.sum().item()

            # 关键防御：如果出现特征数与掩码数不匹配（例如 747 特征 对应 1268 掩码）
            if num_placeholders != target_count:
                _diag_mm_fallback(
                    "qwen36_embedding_merge_mismatch", "Multimodal features and placeholder count differ; existing fallback would reorder or truncate",
                    features=target_count, placeholders=num_placeholders,
                    action="keep_last_placeholder_positions" if num_placeholders > target_count else "truncate_features",
                    inputs_embeds=inputs_embeds, multimodal_embeddings=multimodal_embeddings,
                    mask=mask, placeholder_token_id=placeholder_token_id,
                )
                _diag.console_or_file("qwen36_adapter_message", 
                    f"[verl-omni] Aligning mismatched multimodal embeddings: "
                    f"features={target_count} vs placeholders={num_placeholders}. Adjusting mask dynamically...",
                    flush=True
                )
                
                # 构造一个尺寸完全匹配 target_count 的精确掩码
                true_indices = torch.where(mask)[0]
                new_mask = torch.zeros_like(mask, dtype=torch.bool)
                
                if num_placeholders > target_count:
                    # 掩码包含了其他模态的占位符（如 1268 - 747 = 521）
                    # 取最后 target_count 个位置（通常音频后置）或最长匹配区间
                    matched_indices = true_indices[-target_count:]
                    new_mask[matched_indices] = True
                else:
                    # 特征比槽位多，截断特征
                    mm_embeds_flat = mm_embeds_flat[:num_placeholders]
                    new_mask = mask

                # 采用对齐后的精确掩码与特征执行插入
                inputs_embeds[new_mask] = mm_embeds_flat.to(dtype=inputs_embeds.dtype)
                return inputs_embeds

            # 长度一致时走原生逻辑
            # 注意：全部用 keyword arg 传递，避免 positional arg 与 **kwargs
            # 里的同名 key 冲突（"got multiple values for argument"）
            merge_kwargs = dict(kwargs)
            if placeholder_token_id is not None:
                merge_kwargs["placeholder_token_id"] = placeholder_token_id
            if is_multimodal is not None:
                merge_kwargs["is_multimodal"] = is_multimodal
            return orig_merge(
                inputs_embeds,
                multimodal_embeddings,
                **merge_kwargs
            )

        model_utils._merge_multimodal_embeddings = safe_merge_multimodal_embeddings
        model_utils._patched_merge_mm_embeds_safe = True
        _diag.console_or_file("qwen36_adapter_message", "[verl-omni] Successfully patched vLLM _merge_multimodal_embeddings for robust shape alignment.", flush=True)

    except Exception as e:
        _diag.console_or_file("qwen36_adapter_message", f"[verl-omni] Failed to patch _merge_multimodal_embeddings: {e}", flush=True)

patch_vllm_merge_multimodal_embeddings()

patch_qwen_model_pixel_values_forward()
apply_mrope_text_len_fix_patch()
apply_qwen36_vllm_prompt_updates_patch()
apply_model_stage_mapping_patch()
apply_minicpmo_auto_register_guard()
apply_minicpmo_automodel_fallback()
apply_fsdp2_cpu_offload_pin_memory_patch()

from . import dataset  # noqa: F401
from . import thinker_adapter  # noqa: F401
from .thinker_adapter import Qwen36WhisperThinkerAdapter

__all__ = ["Qwen36WhisperThinkerAdapter"]

try:
    from verl_omni_ext.models.qwen3_6_whisper.rollout_adapter import Qwen36WhisperRolloutAdapter
    from verl_omni_ext.models.qwen3_6_whisper import vllm_omni  # noqa: F401
    __all__.append("Qwen36WhisperRolloutAdapter")
except Exception as e:
    logger.warning(
        "Qwen3.6-whisper rollout adapter unavailable: %s. Training-side loading still works; "
        "rollout does not.",
        e,
    )
