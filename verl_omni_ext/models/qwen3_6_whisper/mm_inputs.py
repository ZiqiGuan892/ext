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
"""Build Qwen3.6-whisper's ``multi_modal_inputs`` without re-decoding the prompt.

The defect this replaces
------------------------
``AgentLoopWorker._compute_multi_modal_inputs`` rebuilds the processor inputs for
training like this::

    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
    multi_modal_inputs = build_multimodal_processor_inputs(self.processor, text=[current_text], ...)

For a Qwen-style processor that round-trips.  For MiniCPM-o / Qwen3.6-whisper it
silently destroys exactly the thing training needs.

``skip_special_tokens=True`` strips every sentinel the model's
``MiniCPMOProcessor._convert`` scans for. The re-processed text contains no
media markers at all, so ``image_bound`` / ``audio_bounds`` come back **empty**.
The tensors still have the right dtype and a plausible shape, nothing raises,
and ``get_vllm_embedding`` then scatters zero vision embeddings into the
sequence -- a wrong-answer machine that reports no error.

The fix
-------
Do not re-derive the text. The expanded ``input_ids`` are already the ground
truth, so scan them directly for the sentinels and emit bounds in the padded
training coordinate system by construction. Pixels and audio features are
obtained from the two processor entry points that are **text-independent** --
``process_image`` and ``audio_feature_extract`` -- so nothing depends on a
decode round-trip.

与新模型的差异（vs MiniCPM-o 5.0）
---------------------------------
MiniCPM-o 5.0 用文本哨兵 token（``im_start_id`` / ``im_end_id`` /
``slice_start_id`` / ``slice_end_id`` / ``audio_start_id`` / ``audio_end_id``）。

Qwen3.6-35B-A3B-whisper-0814-rl 改用 config.json 顶层显式 token id：
  - ``image_token_id = 248056``
  - ``video_token_id = 248057``
  - ``audio_token_id = 248076``
  - ``vision_start_token_id = 248053``
  - ``vision_end_token_id   = 248054``

显式 token id 让 bounds 扫描更直接：``image_bound`` 由连续 ``image_token_id``
区间的 ``[start, end]`` 对组成，``audio_bounds`` 同理对 ``audio_token_id``。
``video_bound`` 由连续 ``video_token_id`` 区间组成（新模态，需新增）。

**关键不变量**（See memory: feedback_precompute_cache_miss_fatal.md）：
  ``audio_bounds[i]`` 必须是 ``[[s, e]]``（pair 列表），**不是** ``[s, e]``
  （单 pair）——否则 ``get_omni_embedding`` 的 ``for embs, bound in
  zip(audio_embs, bounds)`` 会把 ``bound[0]`` 当 int 解包失败。

Scope
-----
Installed from ``Qwen36WhisperThinkerAdapter.configure_processor`` and gated on
the processor being MiniCPM-o's (same processor class as 5.0), so other models
in the same process keep upstream behaviour.
"""

import logging
import os
import torch

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_verl_minicpmo_mm_inputs_patch"  # 复用同一 flag，避免与 5.0 双开时冲突
_DISABLE_ENV = "MINICPMO_DISABLE_MM_INPUTS_PATCH"
_METHOD_NAME = "_compute_multi_modal_inputs"
_LOG_PREFIX = "[verl-omni] Qwen3.6-whisper multi_modal_inputs"

_first_call_reported = False
_mismatch_reported = False

__all__ = ["apply_minicpmo_mm_inputs_patch", "build_minicpmo_multi_modal_inputs"]


# 模态 token IDs — 需要同时支持 Qwen3-VL 风格和 MiniCPM-o 风格的 prompt。
#
# 数据集 prompt 用 MiniCPM-o 风格的 `<image>` (248078) / `<audio>` / `<video>`，
# 而不是 Qwen3-VL 风格的 `<|image_pad|>` (248056) / `<|audio_pad|>` (248076) /
# `<|video_pad|>` (248057)。
#
# processing_minicpmo.py 的 get_audio_placeholder 会把 audio placeholder 展开成
# `<|audio_start|>` (248070) + `<unk>*N` + `<|audio_end|>` (248071)。
#
# 所以 bounds 扫描需要支持所有可能的 token IDs：
#
# Image: <image> (248078), <|image_pad|> (248056)
# Video: <video> (不在 tokenizer), <|video_pad|> (248057)
# Audio: <audio> (不在 tokenizer), <|audio|> (248096), <|audio_pad|> (248076),
#        <|audio_start|> (248070) ... <|audio_end|> (248071)
#
# 对于 audio，processing_minicpmo.py 展开后的 token 是 <|audio_start|> (248070)，
# 所以 bounds 扫描应该找 248070 而不是 248076。

# MiniCPM-o 风格 token IDs (来自 tokenizer.json)
_MINICPMO_IMAGE_TOKEN_ID = 248078   # <image>
_MINICPMO_AUDIO_START_ID = 248070    # <|audio_start|>
_MINICPMO_AUDIO_END_ID = 248071      # <|audio_end|>

# Qwen3-VL 风格 token IDs (来自 config.json)
_Q36_IMAGE_TOKEN_ID = 248056         # <|image_pad|>
_Q36_VIDEO_TOKEN_ID = 248057         # <|video_pad|>
_Q36_AUDIO_TOKEN_ID = 248076         # <|audio_pad|>
_Q36_VISION_START_ID = 248053        # <|vision_start|>
_Q36_VISION_END_ID = 248054          # <|vision_end|>

# 图像 bounds：扫描 MiniCPM-o 的 <image> (248078) 和 Qwen3-VL 的 <|image_pad|> (248056)
_IMAGE_TOKEN_IDS = [_MINICPMO_IMAGE_TOKEN_ID, _Q36_IMAGE_TOKEN_ID]

# 视频 bounds：扫描 Qwen3-VL 的 <|video_pad|> (248057)
# MiniCPM-o 的 <video> 不在 tokenizer 中，video 数据会被 dataset 拆成 frames-as-images
_VIDEO_TOKEN_IDS = [_Q36_VIDEO_TOKEN_ID]

# 音频 bounds：扫描 <|audio_start|> (248070) 和 <|audio_pad|> (248076)
# processing_minicpmo.py 展开的 placeholder 用 <|audio_start|> (248070)
_AUDIO_TOKEN_IDS = [_MINICPMO_AUDIO_START_ID, _Q36_AUDIO_TOKEN_ID]


def _is_minicpmo_processor(processor) -> bool:
    """True when this processor is MiniCPM-o's, shim subclass included.

    ``processing_minicpmo.py`` 与 MiniCPM-o 5.0 byte-identical，processor 类名仍为
    ``MiniCPMOProcessor``，shim 子类名 ``MiniCPMOProcessorVerlShim``。
    """
    return any(base.__name__.startswith("MiniCPMOProcessor") for base in type(processor).__mro__)


def _describe(cls) -> str:
    name = (
        getattr(cls, "__name__", None)
        or getattr(cls, "_actor_name", None)
        or getattr(getattr(cls, "_modified_class", None), "__name__", None)
        or type(cls).__name__
    )
    return f"{name}#{id(cls):x}"


def _worker_class_candidates():
    """Every class an agent-loop actor might resolve the method on."""
    found, problems = [], []
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        found.append(AgentLoopWorker)
    except Exception as e:
        problems.append(f"AgentLoopWorker import failed: {e!r}")
    try:
        from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

        found.append(AgentLoopWorkerTQ)
        modified = getattr(getattr(AgentLoopWorkerTQ, "__ray_metadata__", None), "modified_class", None)
        if modified is not None:
            found.append(modified)
        else:
            problems.append("AgentLoopWorkerTQ exposes no __ray_metadata__.modified_class")
    except Exception as e:
        problems.append(f"AgentLoopWorkerTQ import failed: {e!r}")

    unique, seen = [], set()
    for cls in found:
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if hasattr(cls, _METHOD_NAME):
            unique.append(cls)
        else:
            problems.append(f"{_describe(cls)} has no {_METHOD_NAME}")
    return unique, problems


def _pristine_method(candidates):
    for lookup in (lambda cls: cls.__dict__.get(_METHOD_NAME), lambda cls: getattr(cls, _METHOD_NAME, None)):
        for cls in candidates:
            fn = lookup(cls)
            if fn is not None and not getattr(fn, _PATCH_FLAG, False):
                return fn
    return None


def apply_minicpmo_mm_inputs_patch(processor) -> bool:
    """Replace ``_compute_multi_modal_inputs`` on every candidate worker class."""
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        print(
            f"{_LOG_PREFIX}: declined, disable switch {_DISABLE_ENV} is on. Upstream's "
            "_compute_multi_modal_inputs stays, which rebuilds inputs from a "
            "skip_special_tokens decode; image_bound/audio_bounds will be empty and no media "
            "embedding reaches the sequence. Text-only runs are unaffected.",
            flush=True,
        )
        return False

    if not _is_minicpmo_processor(processor):
        print(
            f"{_LOG_PREFIX}: declined, {type(processor).__name__} is not MiniCPM-o's processor; "
            "upstream's builder stays in place for it.",
            flush=True,
        )
        return False

    candidates, problems = _worker_class_candidates()
    if not candidates:
        print(
            f"{_LOG_PREFIX}: declined, no agent-loop worker class could be reached "
            f"({'; '.join(problems)}). Multimodal training would be handed empty bounds.",
            flush=True,
        )
        return False

    original = _pristine_method(candidates)
    if original is None:
        print(f"{_LOG_PREFIX}: already installed on {[_describe(c) for c in candidates]}.", flush=True)
        return True

    def _compute_multi_modal_inputs(self, output, input_ids):
        global _first_call_reported
        if self.processor is None or not _is_minicpmo_processor(self.processor):
            return original(self, output, input_ids)
        result = build_minicpmo_multi_modal_inputs(
            self.processor,
            input_ids,
            output.multi_modal_data or {},
            output.mm_processor_kwargs,
        )
        if not _first_call_reported:
            _first_call_reported = True
            shapes = {k: tuple(getattr(v, "shape", ())) or type(v).__name__ for k, v in result.items()}
            print(
                f"{_LOG_PREFIX} builder, first call: dispatched on {_describe(type(self))} "
                f"seq_len={int(input_ids.shape[-1])} built={shapes or '(no media in this sample)'}",
                flush=True,
            )
        return result

    setattr(_compute_multi_modal_inputs, _PATCH_FLAG, True)
    installed = []
    for cls in candidates:
        try:
            setattr(cls, _METHOD_NAME, _compute_multi_modal_inputs)
            installed.append(_describe(cls))
        except Exception as e:
            problems.append(f"{_describe(cls)} rejected the assignment: {e!r}")

    if not installed:
        print(
            f"{_LOG_PREFIX}: declined, no candidate accepted the patch ({'; '.join(problems)}). "
            "Multimodal training would be handed empty bounds.",
            flush=True,
        )
        return False

    print(
        f"{_LOG_PREFIX}: installed on {installed}; bounds are scanned from the training input_ids "
        "instead of a skip_special_tokens decode."
        + (f" Notes: {'; '.join(problems)}" if problems else ""),
        flush=True,
    )
    return True


_IMAGE_TOKEN_IDS = [248078, 248056]
_VIDEO_TOKEN_IDS = [248057]
# 音频 bounds 只扫描 <|audio_start|> (248070) 和 <|audio_pad|> (248076)
# 不扫描 248071 (<|audio_end|>)——单个 end token 会被扫成长度1的碎片 span
# 也不扫描 248077/248096——这些不是 audio placeholder
_AUDIO_TOKEN_IDS = [248070, 248076]

def _flatten_and_concat_tensors(item):
    """递归展平嵌套的 list/tuple 并安全拼接为单个 Tensor"""
    if item is None:
        return None
    if isinstance(item, torch.Tensor):
        return item
    if isinstance(item, (list, tuple)):
        flattened = []
        def _collect(x):
            if isinstance(x, torch.Tensor):
                flattened.append(x)
            elif isinstance(x, (list, tuple)):
                for sub in x:
                    _collect(sub)
        _collect(item)
        if not flattened:
            return None
        if len(flattened) == 1:
            return flattened[0]
        # 优先使用 cat 拼在 batch 维，若形状一致再尝试 stack
        try:
            return torch.cat(flattened, dim=0)
        except Exception:
            return torch.stack(flattened, dim=0)
    return item


def build_minicpmo_multi_modal_inputs(processor, input_ids, multi_modal_data, mm_processor_kwargs=None):
    images = multi_modal_data.get("images") or None
    audios = multi_modal_data.get("audios") or None
    videos = multi_modal_data.get("videos") or None
    if images is None and audios is None and videos is None:
        return {}

    _require_media_entrypoints(processor, images, audios)

    kwargs = dict(mm_processor_kwargs or {})
    inputs = {}
    ids = input_ids.squeeze(0)

    if images is not None:
        image_kwargs = {}
        if "max_slice_nums" in kwargs:
            image_kwargs["max_slice_nums"] = kwargs["max_slice_nums"]

        # Qwen3.6-whisper 用 Qwen3-VL 的 ViT 视觉塔（patch_size=16, temporal_patch_size=2）。
        # MiniCPMOProcessor.process_image 用 SigLIP 格式（patch_size=14, reshape_by_patch）——不兼容。
        # 修复：直接用 Qwen2VLImageProcessor（transformers 内置）处理图像，
        # 与 Qwen3.5-VL 推理侧完全一致。
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

            # 归一化 images 为 list[PIL.Image]
            from PIL import Image as _PILImage
            if isinstance(images, _PILImage.Image):
                images_list = [images]
            elif isinstance(images, list) and images and isinstance(images[0], _PILImage.Image):
                images_list = images
            elif isinstance(images, list) and images and isinstance(images[0], list):
                images_list = [img for sublist in images for img in sublist if img is not None]
            else:
                images_list = images if isinstance(images, list) else [images]

            # 用 Qwen2VLImageProcessor 处理
            vl_processor = Qwen2VLImageProcessor(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
            )
            result = vl_processor.preprocess(images=images_list, return_tensors="pt")

            pv = result.get("pixel_values")
            if pv is not None:
                inputs["pixel_values"] = pv  # (num_patches, 1536)

            gthw = result.get("image_grid_thw")
            if gthw is not None:
                inputs["image_grid_thw"] = gthw  # (N, 3)

            print(f"[verl-omni] Qwen3-VL image processing: {len(images_list)} images, "
                  f"pixel_values shape={inputs.get('pixel_values', torch.empty(0)).shape}, "
                  f"grid_thw={inputs.get('image_grid_thw', None)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[verl-omni] Qwen3-VL image processing failed: {e}, falling back to processor.process_image", flush=True)
            traceback.print_exc()
            # Fallback: 用 MiniCPMOProcessor 的 SigLIP 格式
            image_inputs = processor.process_image(images=images, return_tensors="pt", **image_kwargs)
            pv = image_inputs.get("pixel_values")
            if pv is not None:
                inputs["pixel_values"] = _flatten_and_concat_tensors(pv)
            ts = image_inputs.get("tgt_sizes")
            if ts is not None:
                ts_tensor = _flatten_and_concat_tensors(ts)
                if isinstance(ts_tensor, torch.Tensor) and not ts_tensor.is_floating_point():
                    inputs["tgt_sizes"] = ts_tensor
                elif isinstance(ts_tensor, torch.Tensor):
                    inputs["tgt_sizes"] = ts_tensor.long()
                else:
                    inputs["tgt_sizes"] = torch.tensor(ts, dtype=torch.long)

    if audios is not None:
        sampling_rate = int(kwargs.get("sampling_rate", 16000))
        audio_list = audios if isinstance(audios, list) else [audios]
        audio_features, audio_feature_lens, _ = processor.audio_feature_extract(
            audios=audio_list,
            sampling_rate=sampling_rate,
        )
        if len(audio_features):
            # 解包到 per-sample 形状，避免多余的 Batch 维
            inputs["audio_features"] = audio_features[0] if len(audio_features) == 1 else audio_features
            inputs["audio_feature_lens"] = audio_feature_lens[0] if len(audio_feature_lens) == 1 else audio_feature_lens

    bounds = _scan_bounds_by_token_id(ids)
    # 始终生成 image_bound — verl 的 extract_multi_modal_inputs 用 has_image_bound
    # 决定是否对 audio_bounds 等 list 类型走 torch.cat。如果 batch 内混合了
    # 有图片和无图片的样本，micro-batch 可能只含 audio 样本（无 image_bound），
    # 导致 has_image_bound=False，audio_bounds（list）走 torch.cat 崩溃。
    # 生成空 image_bound 确保 has_image_bound=True。
    if images is not None:
        inputs["image_bound"] = bounds["image_bound"]
    else:
        inputs["image_bound"] = torch.zeros((0, 2), dtype=torch.long)
    if audios is not None:
        inputs["audio_bounds"] = bounds["audio_bounds"]
    if videos is not None:
        inputs["video_bound"] = bounds.get("video_bound")

    _warn_on_bound_mismatch(inputs, images, audios, videos)
    return inputs

def _require_media_entrypoints(processor, images, audios):
    """Fail with an explanation if the processor lacks the text-independent entry points."""
    missing = [
        name
        for name, needed in (("process_image", images is not None), ("audio_feature_extract", audios is not None))
        if needed and not hasattr(processor, name)
    ]
    if missing:
        raise AttributeError(
            f"{type(processor).__name__} has no {missing}. This builder needs the processor's "
            "text-independent media entry points, because the whole point is to avoid re-deriving "
            "text: upstream's _compute_multi_modal_inputs rebuilds inputs from a "
            "skip_special_tokens decode, which strips the sentinels image_bound/audio_bounds are "
            "scanned from and yields empty bounds with no error. Set {_DISABLE_ENV}=1 to take the "
            "(silently wrong) upstream path anyway."
        )


def _scan_bounds_by_token_id(ids):
    """Scan input_ids for the new model's explicit modality token ids.

    新模型使用 config.json 顶层 token id：
      - image_token_id=248056 标记 image 媒体区段
      - video_token_id=248057 标记 video 媒体区段
      - audio_token_id=248076 标记 audio 媒体区段

    每个 modality 的连续 token id 区间形成一对 [start, end] bound。返回每个
    modality 的 ``[[s1, e1], [s2, e2], ...]`` pair 列表（即使只有一对也必须是
    pair 列表，不能是单 pair）。

    数据集 prompt 可能用两种风格的 token：
      - MiniCPM-o 风格：``<image>`` (248078) / ``<|audio_start|>`` (248070)
      - Qwen3-VL 风格：``<|image_pad|>`` (248056) / ``<|audio_pad|>`` (248076)

    需要扫描所有可能的 token IDs。

    See memory: feedback_precompute_cache_miss_fatal.md
        audio_bounds[i] 必须是 [[s, e]] 不是 [s, e]
    """
    import torch

    def _scan(token_ids):
        """Scan for runs of any of the given token IDs.

        Returns tensor of shape (N, 2) with [start, end] pairs.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for tid in token_ids:
            mask |= (ids == tid)
        if not mask.any():
            return torch.zeros((0, 2), dtype=torch.long)
        # Find runs of consecutive True
        diff = torch.diff(mask.int())
        starts = (torch.where(diff == 1)[0] + 1).tolist()
        ends = (torch.where(diff == -1)[0] + 1).tolist()
        if mask[0]:
            starts.insert(0, 0)
        if mask[-1]:
            ends.append(len(mask))
        pairs = []
        for s, e in zip(starts, ends):
            pairs.append([int(s), int(e)])
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long)

    image_bound = _scan(_IMAGE_TOKEN_IDS)
    video_bound = _scan(_VIDEO_TOKEN_IDS)

    # audio_bounds 必须是 list of list of [s, e]（嵌套结构），不是 tensor (N, 2)。
    # modeling_minicpmo.py 的 get_omni_embedding 做：
    #   bounds = audio_bounds[i]          # 期望是 list of [s, e]
    #   for embs, bound in zip(audio_embs, bounds):
    #       torch.arange(bound[0], bound[1])
    # 如果 audio_bounds 是 tensor (N, 2)，audio_bounds[i] 是 1D tensor [s, e]，
    # zip 遍历后 bound 是 0-dim tensor → bound[0] 报 TypeError。
    # 修复：转成 list of [[s, e]]（每个 span 一个 pair list）
    audio_bounds_tensor = _scan(_AUDIO_TOKEN_IDS)
    if isinstance(audio_bounds_tensor, torch.Tensor) and audio_bounds_tensor.dim() == 2:
        audio_bounds = [[[int(pair[0]), int(pair[1])] for pair in audio_bounds_tensor]]
    elif isinstance(audio_bounds_tensor, torch.Tensor) and audio_bounds_tensor.numel() == 0:
        audio_bounds = []
    else:
        audio_bounds = [[[int(s), int(e)]] for s, e in audio_bounds_tensor.tolist()] if hasattr(audio_bounds_tensor, 'tolist') else []

    return {
        "image_bound": image_bound,
        "audio_bounds": audio_bounds,
        "video_bound": video_bound,
    }


def _warn_on_bound_mismatch(inputs, images, audios, videos):
    """Say so when the span count does not match the media count."""
    global _mismatch_reported

    messages = []
    image_bound = inputs.get("image_bound")
    if images is not None and image_bound is not None and len(image_bound) == 0:
        messages.append(
            f"{len(images)} image(s) in this sample but 0 image spans in the training input_ids. "
            "No vision embedding will be scattered into the sequence and nothing downstream will "
            "raise. Most likely the prompt ids were not built by the HF processor (so the "
            "image sentinels are absent), or truncation removed the span."
        )

    audio_bounds = inputs.get("audio_bounds")
    if audios is not None and audio_bounds is not None and len(audio_bounds) == 0:
        messages.append(
            f"{len(audios)} audio clip(s) in this sample but 0 audio spans in the training "
            "input_ids. The audio encoder's output has nowhere to be placed."
        )

    video_bound = inputs.get("video_bound")
    if videos is not None and video_bound is not None and len(video_bound) == 0:
        messages.append(
            f"{len(videos)} video(s) in this sample but 0 video spans in the training input_ids. "
            "The video encoder's output has nowhere to be placed."
        )

    for message in messages:
        logger.warning("Qwen3.6-whisper: %s", message)
    if messages and not _mismatch_reported:
        _mismatch_reported = True
        print(f"{_LOG_PREFIX} bound mismatch, first occurrence: {' | '.join(messages)}", flush=True)
