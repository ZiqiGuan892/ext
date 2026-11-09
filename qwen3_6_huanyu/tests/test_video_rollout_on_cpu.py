"""Exercise the native Huanyu video hooks without vLLM or model weights."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_video_rollout_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.video_rollout", ROOT / "video_rollout.py")
rollout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rollout
spec.loader.exec_module(rollout)


@dataclass
class PromptReplacement:
    modality: str
    target: str
    replacement: object


class VideoProcessorItems:
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def get(self, index):
        return self.videos[index]


class DictEmbeddingItems:
    pass


class ImageProcessor:
    patch_size = 1
    temporal_patch_size = 2
    merge_size = 2
    max_pixels = 64

    def __call__(self, images, **kwargs):
        pixels, grids = [], []
        for frame in images:
            frame = torch.as_tensor(np.array(frame)).float()
            height, width, _ = frame.shape
            pixels.append(frame.reshape(-1, 3, 1).repeat(1, 1, 2).reshape(-1, 6))
            grids.append([1, height, width])
        return {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.tensor(grids)}


def make_native():
    class Info:
        def __init__(self, huanyu=True):
            self.config = SimpleNamespace(
                _name_or_path="checkpoint-2748",
                audio_config={"model_type": "huanyu_conformer_0_6b" if huanyu else "whisper"},
                vision_config=SimpleNamespace(patch_size=1, temporal_patch_size=2, spatial_merge_size=2),
            )
            self.ctx = SimpleNamespace(model_config=SimpleNamespace())

        def get_hf_config(self):
            return self.config

        def get_video_frame_size_with_most_features(self):
            return SimpleNamespace(height=2, width=2)

        def get_max_video_frame_tokens(self):
            return 1

        def get_max_video_frames(self, max_tokens):
            return max_tokens // self.get_max_video_frame_tokens()

        def get_num_frames_with_most_features(self, seq_len, mm_counts):
            remaining = seq_len - 2 * mm_counts.get("image", 0) - 3 * mm_counts.get("audio", 0)
            max_frames = self.get_max_video_frames(remaining)
            per_video = min(max_frames // max(mm_counts.get("video", 0), 1), 16)
            return max(per_video, 1)

        def get_max_video_tokens(self, seq_len, mm_counts):
            return self.get_max_video_frame_tokens() * self.get_num_frames_with_most_features(seq_len, mm_counts)

    class Processor:
        def __init__(self, huanyu=True, videos=None, embedded=False):
            self.info = Info(huanyu)
            items = DictEmbeddingItems() if embedded else VideoProcessorItems(videos or [])
            parsed = SimpleNamespace(get_items=Mock(return_value=items))
            self.data_parser = SimpleNamespace(parse_mm_data=Mock(return_value=parsed))
            self.original_calls = []
            self.updates = [
                PromptReplacement("image", "image", object()),
                PromptReplacement("video", "old-video", object()),
                PromptReplacement("audio", "audio", object()),
            ]

        def process_videos(self, mm_data, mm_kwargs, tok_kwargs):
            self.original_calls.append((mm_data, mm_kwargs, tok_kwargs))
            return {"native": True}

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            self.original_calls.append((mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
            return self.updates

        def _apply_prompt_updates(self, token_ids, mm_prompt_updates):
            self.original_calls.append((token_ids, mm_prompt_updates))
            return token_ids, {"legacy": True}

        def get_video_prompt_texts(self, image_size, num_frames):
            return f"native:{image_size.width}x{image_size.height}:{num_frames}"

    class Model:
        vision_start_token_id = 14
        vision_end_token_id = 15
        video_token_id = 16

        def __init__(self, huanyu=True):
            self.config = SimpleNamespace(
                audio_config={"model_type": "huanyu_conformer_0_6b" if huanyu else "whisper"},
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )
            self.original_calls = []

        def _q36_iter_mm_grid_hw(self, input_tokens, mm_features):
            self.original_calls.append((input_tokens, mm_features))
            for feature in mm_features:
                yield (feature.mm_position.offset, 99, 99, 99)

    return SimpleNamespace(
        MiniCPMO50OmniLLMMultiModalProcessor=Processor,
        MiniCPMO50OmniLLMProcessingInfo=Info,
        MiniCPMO50OmniLLMForConditionalGeneration=Model,
        DictEmbeddingItems=DictEmbeddingItems,
        VideoProcessorItems=VideoProcessorItems,
        PromptReplacement=PromptReplacement,
        PromptUpdateDetails=SimpleNamespace(select_text=lambda text, marker: (text, marker)),
        Q36_VIDEO_PLACEHOLDER="<|vision_start|><|video_pad|><|vision_end|>",
        Q36_VISION_CONFIG_DICT={"patch_size": 1, "spatial_merge_size": 2, "temporal_patch_size": 2},
        _q36_image_size_with_most_features=Mock(
            side_effect=lambda pixels: SimpleNamespace(height=2, width=pixels // 2),
        ),
    )


def feature(grid, offset, length=None, modality="video"):
    position = SimpleNamespace(offset=offset)
    if length is not None:
        position.length = length
    return SimpleNamespace(
        modality=modality, mm_position=position,
        data={modality + "_grid_thw": SimpleNamespace(data=torch.tensor(grid))},
    )


class VideoRolloutTests(unittest.TestCase):
    def setUp(self):
        self.native = make_native()
        self.import_patch = patch.object(rollout, "import_module", return_value=self.native)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)
        self.assertTrue(rollout.install_huanyu_video_rollout())
        self.cls = self.native.MiniCPMO50OmniLLMMultiModalProcessor

    def test_huanyu_raw_and_metadata_wrapped_clips_share_training_frontend(self):
        videos = [np.stack([np.full((2, 2, 3), value) for value in (11, 37, 59)])]
        proc = self.cls(videos=[(videos[0], {"fps": 1})])
        image_processor = ImageProcessor()
        with patch.object(rollout, "_image_processor", return_value=image_processor):
            result = proc.process_videos({"videos": videos}, {}, {})
        expected = rollout.preprocess_video_frames(image_processor, videos)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key])
        self.assertEqual(result["video_grid_thw"].tolist(), [[2, 2, 2]])
        torch.testing.assert_close(result["pixel_values_videos"][0], torch.tensor([11., 37., 11., 37., 11., 37.]))
        torch.testing.assert_close(result["pixel_values_videos"][4], torch.tensor([59., 59., 59., 59., 59., 59.]))
        self.assertEqual(proc.original_calls, [])
        raw = self.cls(videos=videos)
        with patch.object(rollout, "_image_processor", return_value=image_processor):
            actual = raw.process_videos({"videos": videos}, {}, {})
        torch.testing.assert_close(actual["pixel_values_videos"], result["pixel_values_videos"])
        proc.data_parser.parse_mm_data.assert_called_once_with({"video": videos})

    def test_video_metadata_objects_are_unwrapped_without_treating_two_frames_as_metadata(self):
        frames = np.stack([np.full((2, 2, 3), value) for value in (7, 13)])
        for video, groups in (((frames, SimpleNamespace(fps=1)), 1), (tuple(frames), 1), (tuple(frames) * 2, 2)):
            with self.subTest(video_type=type(video[1]).__name__):
                proc = self.cls(videos=[video])
                with patch.object(rollout, "_image_processor", return_value=ImageProcessor()):
                    result = proc.process_videos({"videos": [video]}, {}, {})
                self.assertEqual(result["video_grid_thw"].tolist(), [[groups, 2, 2]])

    def test_precomputed_embeddings_and_absent_video_are_left_for_native_fields(self):
        proc = self.cls(embedded=True)
        with patch.object(rollout, "_image_processor") as loader:
            self.assertEqual(proc.process_videos({"videos": {"video_embeds": object()}}, {}, {}), {})
            self.assertEqual(proc.process_videos({}, {}, {}), {})
            loader.assert_not_called()
        self.assertEqual(proc.original_calls, [])

    def test_non_huanyu_keeps_original_inputs_and_update_objects(self):
        proc = self.cls(huanyu=False)
        data, options, tokens = {"videos": [object()]}, {"resize": 9}, {"padding": True}
        with patch.object(rollout, "_image_processor") as loader:
            self.assertEqual(proc.process_videos(data, options, tokens), {"native": True})
            self.assertIs(proc._get_prompt_updates(data, options, tokens), proc.updates)
            loader.assert_not_called()
        self.assertEqual(proc.original_calls, [(data, options, tokens), (data, options, tokens)])

    def test_video_updates_keep_image_audio_and_select_only_video_pads(self):
        proc = self.cls()
        out = {"video": [
            {"video_grid_thw": SimpleNamespace(data=torch.tensor([2, 4, 4]))},
            {"video_grid_thw": SimpleNamespace(data=torch.tensor([[1, 2, 4]]))},
        ]}
        updates = proc._get_prompt_updates({}, {}, out)
        self.assertIs(updates[0], proc.updates[0])
        self.assertIs(updates[1], proc.updates[2])
        video_updates = updates[2:]
        self.assertEqual([u.target for u in video_updates], [self.native.Q36_VIDEO_PLACEHOLDER, "(<video>./</video>)"])
        with patch.object(rollout, "_image_processor", return_value=ImageProcessor()):
            for update in video_updates:
                text, selected = update.replacement(0)
                self.assertEqual(text, rollout.video_placeholder([2, 4, 4], 2))
                self.assertEqual(selected, "<|video_pad|>")
                self.assertEqual(text.count("<|vision_start|>"), 2)
                self.assertEqual(text.count(selected), 8)
                self.assertEqual(update.replacement(1)[0].count(selected), 2)
        self.assertEqual(len(proc.updates), 3)

    def test_installation_is_idempotent(self):
        process, updates = self.cls.process_videos, self.cls._get_prompt_updates
        apply = self.cls._apply_prompt_updates
        iterator = self.native.MiniCPMO50OmniLLMForConditionalGeneration._q36_iter_mm_grid_hw
        prompt = self.cls.get_video_prompt_texts
        info = self.native.MiniCPMO50OmniLLMProcessingInfo
        size, budget = info.get_video_frame_size_with_most_features, info.get_max_video_frame_tokens
        self.assertTrue(rollout.install_huanyu_video_rollout())
        self.assertIs(self.cls.process_videos, process)
        self.assertIs(self.cls._get_prompt_updates, updates)
        self.assertIs(self.cls._apply_prompt_updates, apply)
        self.assertIs(self.native.MiniCPMO50OmniLLMForConditionalGeneration._q36_iter_mm_grid_hw, iterator)
        self.assertIs(self.cls.get_video_prompt_texts, prompt)
        self.assertIs(info.get_video_frame_size_with_most_features, size)
        self.assertIs(info.get_max_video_frame_tokens, budget)

    def test_full_range_prompt_update_is_huanyu_only(self):
        ids, updates, expected = [3, 4], {"video": [[]]}, object()
        huanyu, other = self.cls(), self.cls(huanyu=False)
        with patch.object(rollout, "apply_huanyu_prompt_updates", return_value=expected) as apply:
            self.assertIs(huanyu._apply_prompt_updates(ids, updates), expected)
            self.assertEqual(other._apply_prompt_updates(ids, updates), (ids, {"legacy": True}))
        apply.assert_called_once()
        self.assertIs(apply.call_args.args[0], huanyu)
        self.assertEqual(huanyu.original_calls, [])
        self.assertEqual(other.original_calls, [(ids, updates)])

    def test_checkpoint_maximum_grid_budget_covers_even_odd_and_smaller_requests(self):
        proc = self.cls()
        processor = ImageProcessor()
        with patch.object(rollout, "_image_processor", return_value=processor):
            size = proc.info.get_video_frame_size_with_most_features()
            self.assertEqual(size.height * size.width, processor.max_pixels)
            self.native._q36_image_size_with_most_features.assert_called_with(64)
            group_budget = proc.info.get_max_video_frame_tokens()
            self.assertEqual(group_budget, 16)
            for count in (1, 2, 3, 5):
                for height, width in ((2, 32), (4, 8)):
                    with self.subTest(count=count, height=height, width=width):
                        frames = np.zeros((count, height, width, 3), dtype=np.uint8)
                        parsed = proc.data_parser.parse_mm_data.return_value
                        parsed.get_items.return_value = VideoProcessorItems([frames])
                        actual = proc.process_videos({"videos": [frames]}, {}, {})
                        tokens = int(actual["video_grid_thw"].prod(dim=1).sum()) // processor.merge_size**2
                        seq_len = group_budget * count
                        self.assertEqual(proc.info.get_num_frames_with_most_features(seq_len, {"video": 1}), count)
                        budget = proc.info.get_max_video_tokens(seq_len, {"video": 1})
                        self.assertEqual(budget, group_budget * count)
                        self.assertGreaterEqual(budget, tokens)

    def test_budget_max_pixels_fallbacks_and_missing_or_invalid_limits(self):
        proc = self.cls()
        processor = ImageProcessor()
        processor.max_pixels = None
        with patch.object(rollout, "_image_processor", return_value=processor):
            for key in ("max_pixels", "longest_edge"):
                processor.size = {key: 32}
                size = proc.info.get_video_frame_size_with_most_features()
                self.assertEqual(size.height * size.width, 32)
            for value in (None, 0, -64, True, 64.0):
                processor.size = {} if value is None else {"longest_edge": value}
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive checkpoint max_pixels"):
                    proc.info.get_video_frame_size_with_most_features()

    def test_budget_rejects_native_spatial_geometry_mismatch_and_preserves_other_models(self):
        proc = self.cls()
        with patch.object(rollout, "_image_processor", return_value=ImageProcessor()):
            for key in ("patch_size", "spatial_merge_size"):
                original = self.native.Q36_VISION_CONFIG_DICT[key]
                self.native.Q36_VISION_CONFIG_DICT[key] = 9
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "different spatial geometry"):
                    proc.info.get_video_frame_size_with_most_features()
                self.native.Q36_VISION_CONFIG_DICT[key] = original
        other = self.cls(huanyu=False).info
        with patch.object(rollout, "_image_processor") as loader:
            size = other.get_video_frame_size_with_most_features()
            self.assertEqual((size.height, size.width), (2, 2))
            self.assertEqual(other.get_max_video_frame_tokens(), 1)
            self.assertEqual(other.get_max_video_tokens(5, {"video": 1}), 5)
            loader.assert_not_called()

    def test_dummy_video_prompt_uses_measured_checkpoint_geometry_and_grouped_spans(self):
        proc = self.cls()
        processor = ImageProcessor()
        size = SimpleNamespace(height=4, width=2)
        with patch.object(rollout, "_image_processor", return_value=processor):
            text = proc.get_video_prompt_texts(size, 5)
        self.assertEqual(text, rollout.video_placeholder([3, 4, 2], 2))
        other = self.cls(huanyu=False)
        self.assertEqual(other.get_video_prompt_texts(size, 5), "native:2x4:5")

    def test_checkpoint_processor_is_cached_per_info_and_uses_model_config_fallback(self):
        proc = self.cls()
        loader = Mock(side_effect=[ImageProcessor(), ImageProcessor()])
        module = SimpleNamespace(Qwen2VLImageProcessor=SimpleNamespace(from_pretrained=loader))
        name = "transformers.models.qwen2_vl.image_processing_qwen2_vl"
        with patch.dict(sys.modules, {name: module}):
            first = rollout._image_processor(proc.info)
            self.assertIs(rollout._image_processor(proc.info), first)
            loader.assert_called_once_with("checkpoint-2748")
            proc.info.get_hf_config()._name_or_path = ""
            proc.info.ctx.model_config.model = "fallback-checkpoint"
            self.assertIsNot(rollout._image_processor(proc.info), first)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(loader.call_args.args, ("fallback-checkpoint",))

    def test_missing_checkpoint_path_is_rejected(self):
        proc = self.cls()
        proc.info.get_hf_config()._name_or_path = ""
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            rollout._image_processor(proc.info)

    def test_checkpoint_geometry_must_match_vision_encoder_including_cached_processor(self):
        for name in ("patch_size", "temporal_patch_size", "spatial_merge_size"):
            with self.subTest(name=name):
                proc = self.cls()
                loader = Mock(return_value=ImageProcessor())
                module = SimpleNamespace(Qwen2VLImageProcessor=SimpleNamespace(from_pretrained=loader))
                with patch.dict(sys.modules, {"transformers.models.qwen2_vl.image_processing_qwen2_vl": module}):
                    rollout._image_processor(proc.info)
                    setattr(proc.info.get_hf_config().vision_config, name, 9)
                    with self.assertRaisesRegex(ValueError, name + ": processor"):
                        rollout._image_processor(proc.info)
                    loader.assert_called_once()

    def test_video_iterator_yields_every_temporal_group_and_delegates_images(self):
        model = self.native.MiniCPMO50OmniLLMForConditionalGeneration()
        span = [14, 16, 16, 16, 16, 15]
        tokens = [100] + span * 3 + [101, 14, 13, 15, 102] + [14, 16, 16, 15]
        video1 = feature([3, 4, 4], offset=1, length=18)
        image = feature([1, 2, 2], offset=20, length=3, modality="image")
        video2 = feature([1, 2, 4], offset=24, length=4)
        result = list(model._q36_iter_mm_grid_hw(tokens, [video2, image, video1]))
        self.assertEqual(result, [(2, 2, 2, 4), (8, 2, 2, 4), (14, 2, 2, 4), (20, 99, 99, 99), (25, 1, 2, 2)])
        self.assertEqual(model.original_calls, [(tokens, [image])])

    def test_video_iterator_honors_vllm_config_and_non_huanyu_delegates(self):
        model = self.native.MiniCPMO50OmniLLMForConditionalGeneration()
        model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=model.config))
        model.config = None
        tokens, video = [14, 16, 15], feature([1, 2, 2], 0, 3)
        self.assertEqual(list(model._q36_iter_mm_grid_hw(tokens, [video])), [(1, 1, 1, 1)])
        other = self.native.MiniCPMO50OmniLLMForConditionalGeneration(huanyu=False)
        self.assertEqual(list(other._q36_iter_mm_grid_hw(tokens, [video])), [(0, 99, 99, 99)])
        self.assertEqual(other.original_calls, [(tokens, [video])])

    def test_video_iterator_rejects_missing_extra_short_and_cross_feature_spans(self):
        cases = (
            ([14, 16, 15], feature([2, 2, 2], 0, 3), [], "missing temporal span"),
            ([14, 16, 15] * 2, feature([1, 2, 2], 0, 6), [], "more temporal spans"),
            ([14, 16, 15], feature([1, 4, 4], 0, 3), [], "exactly 4 video pads"),
            ([14, 16, 16, 15], feature([1, 2, 2], 0, 4), [], "exactly 1 video pads"),
            ([14, 16, 15], feature([1, 2, 2], 0, 2), [], "exactly 1 video pads"),
            ([14, 16, 15], feature([1, 2, 2], 0, 4), [], "length exceeds"),
            ([14, 16, 15] * 2, feature([2, 2, 2], 0), [feature([1, 2, 2], 3, 3)], "missing temporal span"),
            ([14, 16, 15], feature([1, 3, 2], 0, 3), [], "divisible"),
        )
        for tokens, video, others, message in cases:
            with self.subTest(message=message):
                model = self.native.MiniCPMO50OmniLLMForConditionalGeneration()
                with self.assertRaisesRegex(ValueError, message):
                    list(model._q36_iter_mm_grid_hw(tokens, [video] + others))


if __name__ == "__main__":
    unittest.main()
