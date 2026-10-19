"""Huanyu training processor; never use the checkpoint's old 128-bin auto_map.

The historical ``load_huanyu_processor`` helper remains available for inspecting
legacy exports. Training must use ``build_huanyu_processor``: it combines the
Qwen vision frontend and tokenizer with the active vLLM-Omni Huanyu frontend.
"""

import importlib
import inspect
import json
from pathlib import Path
import types

import numpy as np
import torch


IMAGE_PATTERN = "(<image>./</image>)"
AUDIO_PATTERN = "(<audio>./</audio>)"


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _token_id(tokenizer, token, expected=None):
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1 or ids[0] < 0:
        raise ValueError(f"Huanyu marker {token!r} must encode to exactly one token, got {ids}")
    if expected is not None and int(ids[0]) != int(expected):
        raise ValueError(f"Huanyu marker {token!r}: tokenizer id {ids[0]} != config id {expected}")
    return int(ids[0])


def _items(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (tuple, list)) else [value]


class HuanyuProcessor:
    """One trajectory at a time, matching AgentLoop's processor contract.

    Images (including video frames) keep Qwen spatial grids. Audio placeholders
    are expanded per 30-second chunk, using real frame lengths, then rescanned
    from the actual token IDs when the training batch is assembled.
    """

    _huanyu_training_processor = True
    audio_chunk_seconds = 30
    audio_encoder_downsample = 4
    audio_projector_downsample = 2

    def __init__(self, tokenizer, image_processor, audio_frontend, config, audio_preprocessor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_frontend = audio_frontend
        self.config = config
        self.audio_preprocessor = dict(audio_preprocessor)
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.sampling_rate = int(audio_preprocessor["sampling_rate"])
        self.feature_size = int(audio_preprocessor["feature_size"])
        self.n_fft = int(audio_preprocessor.get("n_fft", 400))
        self.hop_length = int(audio_preprocessor.get("hop_length", 160))
        self.audio_n_fft = self.n_fft
        self.audio_hop_length = self.hop_length
        self.spatial_merge_size = int(_get(_get(config, "vision_config"), "spatial_merge_size", 2))
        self.audio_start_token_id = _token_id(tokenizer, "<|audio_start|>", _get(config, "audio_start_token_id"))
        self.audio_end_token_id = _token_id(tokenizer, "<|audio_end|>", _get(config, "audio_end_token_id"))
        self.audio_placeholder_token_id = _token_id(tokenizer, "<unk>")
        self.image_token_id = _token_id(tokenizer, "<|image_pad|>", _get(config, "image_token_id"))
        self.vision_start_token_id = _token_id(tokenizer, "<|vision_start|>", _get(config, "vision_start_token_id"))
        self.vision_end_token_id = _token_id(tokenizer, "<|vision_end|>", _get(config, "vision_end_token_id"))

    def apply_chat_template(self, messages, **kwargs):
        flattened = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                flattened.append(message)
                continue
            parts = []
            for item in content:
                kind = item.get("type")
                if kind == "text":
                    parts.append(item.get("text", ""))
                elif kind == "image":
                    parts.append(IMAGE_PATTERN)
                elif kind == "audio":
                    parts.append(AUDIO_PATTERN)
                elif kind == "video":
                    from .dataset import video_prompt_parts
                    parts.extend(video_prompt_parts(item, self.sampling_rate))
                else:
                    raise ValueError(f"Unsupported Huanyu message item {kind!r}")
            flattened.append({**message, "content": "\n".join(parts)})
        return self.tokenizer.apply_chat_template(flattened, **kwargs)

    def _audio(self, audios, sampling_rate):
        if int(sampling_rate) != self.sampling_rate:
            raise ValueError(f"Huanyu expects {self.sampling_rate} Hz waveforms, got {sampling_rate}")
        features, lengths, groups = [], [], []
        for source in _items(audios):
            waveform = np.asarray(source, dtype=np.float32)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("Huanyu audio must be a finite mono waveform [samples]")
            tokens = []
            for offset in range(0, len(waveform), 30 * self.sampling_rate):
                chunk = waveform[offset:offset + 30 * self.sampling_rate]
                frames = (len(chunk) - self.n_fft) // self.hop_length + 1
                count = ((frames + 3) // 4) // 2
                if frames < 1 or count < 1:
                    raise ValueError("Huanyu audio chunk is too short to produce a token; no tail was dropped")
                feat, actual_frames = self.audio_frontend.extract_fbank(chunk)
                if tuple(feat.shape) != (self.feature_size, frames) or int(actual_frames) != frames:
                    raise ValueError(f"Huanyu frontend returned {tuple(feat.shape)}, expected {(self.feature_size, frames)}")
                if not torch.isfinite(feat).all():
                    raise ValueError("Huanyu fbank contains NaN/Inf")
                features.append(feat.float().contiguous())
                lengths.append(frames)
                tokens.append(count)
            if not tokens:
                raise ValueError("Huanyu audio waveform is empty")
            groups.append(tokens)
        return features, torch.tensor(lengths, dtype=torch.long), groups

    def _vision(self, images):
        images = _items(images)
        if not images:
            return {}
        result = dict(self.image_processor(images=images, return_tensors="pt"))
        grids = result.get("image_grid_thw")
        pixels = result.get("pixel_values")
        if grids is None or pixels is None or tuple(grids.shape) != (len(images), 3):
            raise ValueError("Qwen vision processor must return pixel_values and one image_grid_thw per image")
        return {"pixel_values": pixels, "image_grid_thw": grids.long()}

    @staticmethod
    def _replace(text, pattern, replacements):
        if text.count(pattern) != len(replacements):
            raise ValueError(f"Huanyu {pattern}: prompt has {text.count(pattern)} placeholders but {len(replacements)} media items")
        for replacement in replacements:
            text = text.replace(pattern, replacement, 1)
        return text

    def __call__(self, text, images=None, videos=None, audios=None, audio=None,
                 sampling_rate=16000, return_tensors="pt", **kwargs):
        if _items(videos):
            raise ValueError("Huanyu dataset must decode videos to ordered image frames before processor invocation")
        texts = [text] if isinstance(text, str) else list(text)
        if len(texts) != 1:
            raise ValueError("HuanyuProcessor supports one trajectory per call; AgentLoop handles batching")
        if audios is not None and audio is not None:
            raise ValueError("Supply either audio or audios, not both")
        audio_features, audio_lengths, groups = self._audio(audios if audios is not None else audio, sampling_rate)
        vision = self._vision(images)
        image_spans = []
        for grid in vision.get("image_grid_thw", []):
            patches = int(grid.prod())
            merge = self.spatial_merge_size ** 2
            if patches % merge:
                raise ValueError("Vision grid is not divisible by the configured spatial merge")
            image_spans.append("<|vision_start|>" + "<|image_pad|>" * (patches // merge) + "<|vision_end|>")
        rendered = self._replace(texts[0], IMAGE_PATTERN, image_spans)
        audio_spans = ["".join("<|audio_start|>" + "<unk>" * n + "<|audio_end|>" for n in group) for group in groups]
        rendered = self._replace(rendered, AUDIO_PATTERN, audio_spans)
        encoded = dict(self.tokenizer(rendered, add_special_tokens=False, return_tensors=return_tensors))
        encoded.update(vision)
        encoded.update(self._bound_inputs(encoded["input_ids"], audio_features, audio_lengths, groups, vision))
        return encoded

    def _bound_inputs(self, input_ids, features, lengths, groups, vision):
        from .audio_alignment import find_audio_bounds
        ids = torch.as_tensor(input_ids).reshape(-1)
        audio_bounds = find_audio_bounds(
            ids, [n for group in groups for n in group],
            start_token_id=self.audio_start_token_id, end_token_id=self.audio_end_token_id,
            placeholder_token_id=self.audio_placeholder_token_id,
        )
        image_bounds, start = [], None
        for index, token in enumerate(ids.tolist()):
            if token == self.image_token_id:
                start = index if start is None else start
            elif start is not None:
                image_bounds.append((start, index))
                start = None
        if start is not None:
            image_bounds.append((start, len(ids)))
        expected = [int(grid.prod()) // self.spatial_merge_size ** 2 for grid in vision.get("image_grid_thw", [])]
        actual = [end - begin for begin, end in image_bounds]
        if actual != expected:
            raise ValueError(f"Huanyu image token lengths {actual} != image grid lengths {expected}")
        return {
            "huanyu_audio_features": features,
            "huanyu_audio_lengths": lengths,
            "huanyu_audio_chunk_counts": torch.tensor([len(features)], dtype=torch.long),
            "huanyu_audio_bounds": torch.tensor(audio_bounds, dtype=torch.long).reshape(-1, 2),
            "image_bound": torch.tensor(image_bounds, dtype=torch.long).reshape(-1, 2),
        }

    def build_multi_modal_inputs(self, input_ids, multi_modal_data, mm_processor_kwargs=None):
        data = multi_modal_data or {}
        kwargs = dict(mm_processor_kwargs or {})
        if _items(data.get("videos")):
            raise ValueError("Huanyu training expects dataset videos converted to image frames")
        vision = self._vision(data.get("images"))
        features, lengths, groups = self._audio(data.get("audios"), kwargs.get("sampling_rate", self.sampling_rate))
        return {**vision, **self._bound_inputs(input_ids, features, lengths, groups, vision)}

    def dedup_pad_tokens(self, prompt_ids):
        """The Huanyu rollout hook contracts spans using actual media metadata.

        A token-only contraction cannot distinguish two consecutive input audios
        from two chunks belonging to one audio. Refuse that ambiguity instead of
        guessing from adjacency or changing the number of media items.
        """
        ids = list(prompt_ids)
        if self.audio_start_token_id in ids or self.vision_start_token_id in ids:
            raise RuntimeError(
                "Huanyu rollout received expanded media tokens without metadata-aware "
                "contraction; install the Huanyu rollout_inputs hook before starting servers"
            )
        return ids


def _rope_token_types(processor, input_ids, attention_mask=None):
    """Describe the actual padded sequence for the newer Qwen RoPE API.

    Audio slots use ordinary sequential positions (type 0); their embeddings
    are replaced separately by the Huanyu forward bridge. Video frames decoded
    by our dataset use image_pad and therefore image grids/type 1.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Huanyu RoPE requires input_ids with shape [batch, sequence]")
    token_types = torch.zeros_like(input_ids, dtype=torch.long)
    token_types.masked_fill_(input_ids == processor.image_token_id, 1)
    video_token_id = _get(processor.config, "video_token_id")
    if video_token_id is not None:
        token_types.masked_fill_(input_ids == int(video_token_id), 2)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Huanyu RoPE attention_mask must match input_ids [batch, sequence]")
        token_types.masked_fill_(~attention_mask.to(device=input_ids.device, dtype=torch.bool), 0)
    return token_types


def _bind_qwen_rope(processor):
    """Use the installed HF implementation rather than fabricate text positions."""
    module = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    owner = next((getattr(module, name, None) for name in
                  ("Qwen3_5MoeModel", "Qwen3_5MoeForConditionalGeneration")
                  if callable(getattr(getattr(module, name, None), "get_rope_index", None))), None)
    if owner is None:
        raise RuntimeError("Installed Qwen3.5-MoE model exposes no get_rope_index; cannot construct Huanyu multimodal positions")
    # Current HF versions factor vision-position construction into one helper.
    for name in ("get_vision_position_ids", "get_llm_pos_ids_for_vision"):
        method = getattr(owner, name, None)
        if method is not None:
            descriptor = inspect.getattr_static(owner, name)
            bound = method if isinstance(descriptor, (staticmethod, classmethod)) else types.MethodType(method, processor)
            setattr(processor, name, bound)
    native = owner.get_rope_index
    signature = inspect.signature(native)
    parameters = signature.parameters

    def get_rope_index(self, *args, **kwargs):
        supported = {key: value for key, value in kwargs.items() if key in parameters}
        bound = signature.bind_partial(self, *args, **supported)
        if "mm_token_type_ids" in parameters and bound.arguments.get("mm_token_type_ids") is None:
            bound.arguments["mm_token_type_ids"] = _rope_token_types(
                self, bound.arguments.get("input_ids"), bound.arguments.get("attention_mask")
            )
        positions, deltas = native(*bound.args, **bound.kwargs)
        return positions.long(), deltas

    processor.get_rope_index = types.MethodType(get_rope_index, processor)


def build_huanyu_processor(model_path, model_config):
    from .runtime_imports import load_huanyu_native

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    trust = bool(_get(model_config, "trust_remote_code", False))
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust)
    audio_config = _get(config, "audio_config", {})
    if _get(audio_config, "model_type") != "huanyu_conformer_0_6b":
        raise ValueError("Huanyu processor requires audio_config.model_type=huanyu_conformer_0_6b")
    path = Path(model_path) / "audio_preprocessor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Huanyu training requires validated declarative audio frontend metadata: {path}")
    pre = json.loads(path.read_text(encoding="utf-8"))
    if pre.get("feature_type") != "fbank" or int(pre.get("feature_size", 0)) != int(_get(audio_config, "input_feat_per_channel", 0)):
        raise ValueError("Huanyu audio_preprocessor_config feature width must match audio_config.input_feat_per_channel")
    if int(pre.get("sampling_rate", 0)) != 16000:
        raise ValueError("This Huanyu integration requires the verified 16000 Hz frontend")
    if int(pre["feature_size"]) != 40:
        raise ValueError("This training bridge supports the verified 40-bin declarative Huanyu checkpoint only")
    if int(_get(audio_config, "projector_downsample_rate", 0)) != 2:
        raise ValueError("Huanyu processor requires verified VGG /4 and projector fold /2")
    runtime = load_huanyu_native()
    frontend = runtime.HuanyuFbankFeatureExtractor(
        feature_size=int(pre["feature_size"]), sampling_rate=int(pre["sampling_rate"]),
        n_fft=int(pre.get("n_fft", 400)), hop_length=int(pre.get("hop_length", 160)),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust)
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
    processor = HuanyuProcessor(tokenizer, image_processor, frontend, config, pre)
    _bind_qwen_rope(processor)
    install_agent_loop_bridge()
    return processor


def install_agent_loop_bridge():
    """Patch only Huanyu instances; all other processors call their old method."""
    candidates = []
    for module_name, class_name in (
        ("verl.experimental.agent_loop.agent_loop", "AgentLoopWorker"),
        ("verl.trainer.ppo.v1.agent_loop_tq", "AgentLoopWorkerTQ"),
    ):
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        candidates.append(cls)
        modified = getattr(getattr(cls, "__ray_metadata__", None), "modified_class", None)
        if modified is not None:
            candidates.append(modified)
    installed = 0
    for cls in dict.fromkeys(candidates):
        original = getattr(cls, "_compute_multi_modal_inputs", None)
        if original is None:
            continue
        if getattr(original, "_huanyu_bridge", False):
            installed += 1
            continue

        def build(self, output, input_ids, _original=original):
            if not isinstance(getattr(self, "processor", None), HuanyuProcessor):
                return _original(self, output, input_ids)
            return self.processor.build_multi_modal_inputs(
                input_ids, output.multi_modal_data, getattr(output, "mm_processor_kwargs", None)
            )

        build._huanyu_bridge = True
        setattr(cls, "_compute_multi_modal_inputs", build)
        installed += 1
    if not installed:
        raise RuntimeError("No AgentLoopWorker multimodal-input method could be wired for Huanyu")


def load_huanyu_processor(model_path: str, *, trust_remote_code: bool):
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load Huanyu processor from {model_path}: "
            f"{type(exc).__name__}: {exc}. Check processing_huanyu.py and its "
            "dependencies; if it reads mel_filter_path, that file must exist "
            "on each worker. No Whisper/tokenizer-only fallback was applied."
        ) from exc
    if not any(base.__name__ == "HuanyuMiniCPMOProcessor" for base in type(processor).__mro__):
        raise TypeError(
            f"Expected checkpoint HuanyuMiniCPMOProcessor, got "
            f"{type(processor).__module__}.{type(processor).__name__}"
        )
    if not callable(processor) or getattr(processor, "tokenizer", None) is None:
        raise TypeError("Huanyu processor must be callable and expose its tokenizer")
    return processor
