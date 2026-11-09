"""Use checkpoint video pixels and matching temporal spans in Huanyu rollout.

Video profiling uses the checkpoint's maximum image size. Native total-token
and frame-count budgeting still charge one full temporal-group token budget
per source frame, conservatively overestimating paired video requirements.
"""

from collections.abc import Mapping
from functools import wraps
from importlib import import_module
from numbers import Integral

from .video_frontend import preprocess_video_frames, video_grid_for_size, video_placeholder


_NATIVE = "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni_llm"
_CLASS = "MiniCPMO50OmniLLMMultiModalProcessor"
_INFO_CLASS = "MiniCPMO50OmniLLMProcessingInfo"
_MODEL_CLASS = "MiniCPMO50OmniLLMForConditionalGeneration"
_MARKER = "_huanyu_paired_video_frontend"
_CACHE = "_huanyu_video_image_processor"


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _is_huanyu(info):
    config = info.get_hf_config()
    return _field(_field(config, "audio_config"), "model_type") == "huanyu_conformer_0_6b"


def _model_huanyu_config(model):
    for candidate in (_field(model, "config"), _field(model, "vllm_config")):
        config = _field(candidate, "model_config", candidate)
        config = _field(config, "hf_config", config)
        if _field(_field(config, "audio_config"), "model_type") == "huanyu_conformer_0_6b":
            return config
    return None


def _video_grid_spans(model, config, input_tokens, feature, limit):
    grid = feature.data["video_grid_thw"].data
    if hasattr(grid, "reshape"):
        grid = grid.reshape(-1).tolist()
    merge = _field(_field(config, "vision_config"), "spatial_merge_size")
    if isinstance(merge, bool) or not isinstance(merge, Integral) or merge <= 0:
        raise ValueError("Huanyu video M-RoPE requires a positive integer spatial_merge_size")
    if (
        not isinstance(grid, (list, tuple))
        or len(grid) != 3
        or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in grid)
    ):
        raise ValueError("Huanyu video M-RoPE requires a positive integer grid [T, H, W]")
    temporal, height, width = map(int, grid)
    if height % merge or width % merge:
        raise ValueError("Huanyu video M-RoPE grid H/W must be divisible by spatial_merge_size")
    height, width = height // merge, width // merge
    count = height * width
    offset = feature.mm_position.offset
    length = _field(feature.mm_position, "length")
    if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0 or offset >= limit:
        raise ValueError("Huanyu video multimodal offset is outside its input token region")
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0 or offset + length > limit:
            raise ValueError("Huanyu video multimodal length exceeds its input token region")
        limit = offset + length
    cursor = offset
    for group in range(temporal):
        try:
            begin = input_tokens.index(model.vision_start_token_id, cursor, limit)
        except ValueError as exc:
            raise ValueError(f"Huanyu video is missing temporal span {group + 1}/{temporal}") from exc
        start, end = begin + 1, begin + 1 + count
        if (
            model.video_token_id in input_tokens[cursor:begin]
            or end >= limit
            or input_tokens[end] != model.vision_end_token_id
            or any(token != model.video_token_id for token in input_tokens[start:end])
        ):
            raise ValueError(
                f"Huanyu video temporal span {group + 1}/{temporal} must contain exactly {count} video pads"
            )
        yield start, height, width, count
        cursor = end + 1
    if model.video_token_id in input_tokens[cursor:limit]:
        raise ValueError("Huanyu video contains more temporal spans than video_grid_thw declares")


def _image_processor(info):
    config = info.get_hf_config()
    model_path = _field(config, "_name_or_path") or _field(_field(_field(info, "ctx"), "model_config"), "model")
    if not model_path:
        raise ValueError("Huanyu video processing requires the checkpoint path in HF config or model config")
    cached = getattr(info, _CACHE, None)
    if cached is not None and cached[0] == model_path:
        processor = cached[1]
    else:
        # Match build_huanyu_processor; do not substitute the native module's
        # global resize/normalization constants for the checkpoint's settings.
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

        processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    vision_config = _field(config, "vision_config")
    for field, processor_field in (
        ("patch_size", "patch_size"),
        ("temporal_patch_size", "temporal_patch_size"),
        ("spatial_merge_size", "merge_size"),
    ):
        expected, actual = _field(vision_config, field), getattr(processor, processor_field, None)
        if expected is not None and actual != expected:
            raise ValueError(f"Huanyu video {field}: processor {actual} != encoder {expected}")
    setattr(info, _CACHE, (model_path, processor))
    return processor


def _process_videos(self, native, mm_data):
    videos = mm_data.get("videos")
    if videos is None:
        return {}
    parsed = self.data_parser.parse_mm_data({"video": videos}).get_items(
        "video", (native.DictEmbeddingItems, native.VideoProcessorItems),
    )
    if isinstance(parsed, native.DictEmbeddingItems):
        return {}
    clips = []
    for index in range(len(parsed)):
        video = parsed.get(index)
        # vLLM VideoProcessorItems may attach sampling metadata. Keep its
        # already-sampled frames; the current native prompt has no timestamps.
        if isinstance(video, tuple) and len(video) == 2:
            metadata = video[1]
            if metadata is None or isinstance(metadata, Mapping) or hasattr(metadata, "fps"):
                video, _metadata = video
        clips.append(video)
    if not clips:
        return {}
    return preprocess_video_frames(_image_processor(self.info), clips)


def _video_frame_size(info, native):
    processor = _image_processor(info)
    native_geometry = native.Q36_VISION_CONFIG_DICT
    for field, processor_field in (("patch_size", "patch_size"), ("spatial_merge_size", "merge_size")):
        expected, actual = _field(native_geometry, field), getattr(processor, processor_field, None)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, Integral)
            or actual <= 0
            or expected != actual
        ):
            raise ValueError(
                f"Huanyu video sizing helper {field}: native {expected!r} != checkpoint {actual!r}; "
                "cannot use the native helper with different spatial geometry"
            )
    max_pixels = getattr(processor, "max_pixels", None)
    if max_pixels is None:
        size = getattr(processor, "size", None)
        max_pixels = _field(size, "max_pixels")
        if max_pixels is None:
            max_pixels = _field(size, "longest_edge")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, Integral) or max_pixels <= 0:
        raise ValueError("Huanyu video profiling requires a positive checkpoint max_pixels or size.longest_edge")
    return native._q36_image_size_with_most_features(int(max_pixels))


def install_huanyu_video_rollout():
    """Install once before native processor construction; preserve other models."""
    native = import_module(_NATIVE)
    cls = getattr(native, _CLASS)
    info_cls = getattr(native, _INFO_CLASS)
    model_cls = getattr(native, _MODEL_CLASS)
    original_process = cls.process_videos
    original_updates = cls._get_prompt_updates
    original_grids = model_cls._q36_iter_mm_grid_hw
    original_prompt = cls.get_video_prompt_texts
    original_frame_size = info_cls.get_video_frame_size_with_most_features
    original_frame_tokens = info_cls.get_max_video_frame_tokens
    installed = [
        getattr(method, _MARKER, False)
        for method in (
            original_process, original_updates, original_grids,
            original_prompt, original_frame_size, original_frame_tokens,
        )
    ]
    if all(installed):
        return True
    if any(installed):
        raise RuntimeError("Huanyu video rollout hooks are only partially installed")

    @wraps(original_process)
    def process_videos(self, mm_data, mm_kwargs, tok_kwargs):
        if not _is_huanyu(self.info):
            return original_process(self, mm_data, mm_kwargs, tok_kwargs)
        return _process_videos(self, native, mm_data)

    @wraps(original_prompt)
    def get_video_prompt_texts(self, image_size, num_frames):
        if not _is_huanyu(self.info):
            return original_prompt(self, image_size, num_frames)
        processor = _image_processor(self.info)
        grid = video_grid_for_size(processor, image_size.height, image_size.width, num_frames)
        return video_placeholder(grid, processor.merge_size)

    @wraps(original_frame_size)
    def get_video_frame_size_with_most_features(self):
        if not _is_huanyu(self):
            return original_frame_size(self)
        return _video_frame_size(self, native)

    @wraps(original_frame_tokens)
    def get_max_video_frame_tokens(self):
        if not _is_huanyu(self):
            return original_frame_tokens(self)
        size = self.get_video_frame_size_with_most_features()
        processor = _image_processor(self)
        _, height, width = video_grid_for_size(processor, size.height, size.width, 1)
        # Keep the native source-frame multiplier and source-frame selection.
        # For N frames, N * tokens_per_group >= ceil(N / temporal) * tokens_per_group.
        return height * width // processor.merge_size**2

    @wraps(original_updates)
    def get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = original_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        if not _is_huanyu(self.info):
            return updates

        def replace_video(item_index):
            grid = out_mm_kwargs["video"][item_index]["video_grid_thw"].data
            merge = _image_processor(self.info).merge_size
            text = video_placeholder(grid, merge)
            return native.PromptUpdateDetails.select_text(text, "<|video_pad|>")

        # Retain native image/audio update objects, including their closures
        # and embedding masks. Both supported compact video spellings expand
        # into exactly the same per-temporal-group spans as training.
        kept = [update for update in updates if update.modality != "video"]
        for target in (native.Q36_VIDEO_PLACEHOLDER, "(<video>./</video>)"):
            kept.append(native.PromptReplacement(modality="video", target=target, replacement=replace_video))
        return kept

    @wraps(original_grids)
    def iter_mm_grid_hw(self, input_tokens, mm_features):
        config = _model_huanyu_config(self)
        if config is None:
            yield from original_grids(self, input_tokens, mm_features)
            return
        tokens = input_tokens.tolist() if hasattr(input_tokens, "tolist") else list(input_tokens)
        features = sorted(mm_features, key=lambda feature: feature.mm_position.offset)
        for index, feature in enumerate(features):
            if feature.modality != "video":
                yield from original_grids(self, input_tokens, [feature])
                continue
            limit = features[index + 1].mm_position.offset if index + 1 < len(features) else len(tokens)
            if limit > len(tokens):
                raise ValueError("Huanyu multimodal feature offset exceeds the input token length")
            yield from _video_grid_spans(self, config, tokens, feature, limit)

    setattr(process_videos, _MARKER, True)
    setattr(get_prompt_updates, _MARKER, True)
    setattr(iter_mm_grid_hw, _MARKER, True)
    setattr(get_video_prompt_texts, _MARKER, True)
    setattr(get_video_frame_size_with_most_features, _MARKER, True)
    setattr(get_max_video_frame_tokens, _MARKER, True)
    cls.process_videos = process_videos
    cls._get_prompt_updates = get_prompt_updates
    model_cls._q36_iter_mm_grid_hw = iter_mm_grid_hw
    cls.get_video_prompt_texts = get_video_prompt_texts
    info_cls.get_video_frame_size_with_most_features = get_video_frame_size_with_most_features
    info_cls.get_max_video_frame_tokens = get_max_video_frame_tokens
    print(
        "[huanyu-video] layout=temporal_pairs_v1 frontend=checkpoint "
        "spans=per_temporal_group rope=per_temporal_group budget=source_frames_conservative",
        flush=True,
    )
    return True
