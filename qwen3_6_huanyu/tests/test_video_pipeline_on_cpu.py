"""Exercise paired videos across dataset, processor, rollout and frozen cache."""

import importlib
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_video_pipeline_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
processor_module = importlib.import_module(f"{PACKAGE}.processor")
cache_module = importlib.import_module(f"{PACKAGE}.cache")
forward_module = importlib.import_module(f"{PACKAGE}.forward")
rollout_module = importlib.import_module(f"{PACKAGE}.rollout_inputs")
precompute_module = importlib.import_module(f"{PACKAGE}.precompute")
dataset_stub = ModuleType("verl.utils.dataset.rl_dataset")
dataset_stub.RLHFDataset = type("RLHFDataset", (), {})
with patch.dict(sys.modules, {dataset_stub.__name__: dataset_stub}):
    dataset_module = importlib.import_module(f"{PACKAGE}.dataset")
sys.modules[f"{PACKAGE}.dataset"] = dataset_module

# Reuse the independent coordinate-based patch fixture without importing the
# test module under its pytest name or modifying any other test's globals.
fixture_spec = importlib.util.spec_from_file_location(
    "_huanyu_video_pipeline_patch_fixture", ROOT / "tests" / "test_video_frontend_on_cpu.py",
)
patch_fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(patch_fixture)


class Tokenizer:
    markers = {
        "<|audio_start|>": 10, "<|audio_end|>": 11, "<unk>": 12,
        "<|image_pad|>": 13, "<|vision_start|>": 14, "<|vision_end|>": 15,
        "<|video_pad|>": 16,
    }
    chat_template = "fixture"

    def encode(self, text, **kwargs):
        parts = re.split("(" + "|".join(map(re.escape, self.markers)) + ")", text)
        return [token for part in parts for token in (
            [self.markers[part]] if part in self.markers else [1000 + ord(char) for char in part]
        )]

    def decode(self, ids):
        markers = {value: key for key, value in self.markers.items()}
        return "".join(markers[token] if token in markers else chr(token - 1000) for token in ids)

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class ImageFrontend(patch_fixture.ImageProcessor):
    def __call__(self, images, return_tensors):
        result = super().__call__([np.array(image) for image in images], return_tensors)
        return {**result, "pixel_values": result["pixel_values"].float()}


class AudioFrontend:
    def extract_fbank(self, waveform):
        frames = (len(waveform) - 400) // 160 + 1
        return torch.full((40, frames), float(waveform[0]), dtype=torch.float32), frames


def make_processor():
    config = SimpleNamespace(
        audio_start_token_id=10, audio_end_token_id=11, image_token_id=13,
        vision_start_token_id=14, vision_end_token_id=15, video_token_id=16,
        vision_config=SimpleNamespace(patch_size=2, temporal_patch_size=2, spatial_merge_size=2),
        text_config=SimpleNamespace(hidden_size=4),
    )
    return processor_module.HuanyuProcessor(
        Tokenizer(), ImageFrontend(), AudioFrontend(), config,
        {"sampling_rate": 16000, "feature_size": 40},
    )


class FrozenFeatureEncoders:
    """Distinct deterministic token values, without a checkpoint or accelerator."""

    def __init__(self):
        self.visual_grids = []
        self.audio_calls = 0

    def vision(self, pixels, grid):
        self.visual_grids.append(grid.tolist())
        tokens = int(grid.prod()) // 4
        values = pixels.reshape(tokens, 4, -1).mean(dim=(1, 2))
        return values[:, None].expand(-1, 4).contiguous()

    def audio(self, features, length):
        self.audio_calls += 1
        tokens = ((length + 3) // 4) // 2
        return torch.arange(tokens, dtype=torch.float32)[:, None].expand(-1, 4) + 10


class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(2048, 4)
        self.seen = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        self.seen = kwargs
        if kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None:
            raise AssertionError("Frozen cached forward must not run the visual encoder")
        return self.embedding(input_ids)


class VideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.processor = make_processor()
        self.frames = {
            "clip-one": [Image.fromarray(patch_fixture.rgb_frame(seed)) for seed in (7, 53, 97)],
            "clip-two": [Image.fromarray(patch_fixture.rgb_frame(seed, 4, 8)) for seed in (151, 191)],
        }
        helper = SimpleNamespace(
            video_spec_from_item=lambda item: item["video"],
            sample_frames=lambda spec: self.frames[spec],
            use_audio_in_video=lambda: False,
        )
        helper_patch = patch.object(dataset_module, "_video_helpers", return_value=helper)
        helper_patch.start()
        self.addCleanup(helper_patch.stop)
        self.image = Image.fromarray(patch_fixture.rgb_frame(17, 4, 8))
        self.audio = np.full(16000, 0.25, dtype=np.float32)
        self.messages = [{"role": "user", "content": [
            {"type": "text", "text": "Compare the clips"},
            {"type": "image", "image": self.image},
            {"type": "video", "video": "clip-one"},
            {"type": "audio", "audio": self.audio},
            {"type": "video", "video": "clip-two"},
        ]}]
        images, videos, audios = dataset_module.Qwen36HuanyuOmniRLHFDataset._process_multi_modal_info(
            self.messages, 2, {"mm_processor_kwargs": {"sampling_rate": 16000}},
        )
        self.media = {"images": images, "videos": videos, "audios": audios}
        self.row = {"images": [{"image": self.image}], "videos": [
            {"video": "clip-one"}, {"video": "clip-two"},
        ], "audios": [{"audio": self.audio}]}
        self.text = self.processor.apply_chat_template(self.messages)

    def online(self):
        return self.processor(self.text, **self.media)

    def cache(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        return cache_module.HuanyuEmbeddingCache(
            Path(temporary.name) / "cache", {"model": "fixture", "video_execution": "temporal_pairs_v1"}, create=True,
        )

    def test_dataset_preserves_two_clips_and_offline_decoder_uses_identical_frames(self):
        self.assertEqual([len(self.media[key]) for key in ("images", "videos", "audios")], [1, 2, 1])
        self.assertEqual([video.shape for video in self.media["videos"]], [(3, 8, 8, 3), (2, 4, 8, 3)])
        decoded = dataset_module.decode_huanyu_media(self.row)
        for items, expected in zip(decoded, self.media.values(), strict=True):
            for actual_item, expected_item in zip(items, expected, strict=True):
                np.testing.assert_array_equal(np.asarray(actual_item), np.asarray(expected_item))
        self.assertEqual(self.text.count(processor_module.IMAGE_PATTERN), 1)
        self.assertEqual(self.text.count(processor_module.VIDEO_PATTERN), 2)
        self.assertEqual(self.text.count(processor_module.AUDIO_PATTERN), 1)

    def test_processor_expands_temporal_groups_and_rebuilds_padded_training_inputs(self):
        online = self.online()
        self.assertEqual(online["image_grid_thw"].tolist(), [[1, 2, 4]])
        self.assertEqual(online["video_grid_thw"].tolist(), [[2, 4, 4], [1, 2, 4]])
        video_spans = forward_module._image_spans(online["input_ids"][0], 16)
        self.assertEqual([end - start for start, end in video_spans], [4, 4, 2])
        self.assertEqual(int((online["input_ids"] == 13).sum()), 2)
        before = online["input_ids"].clone()
        padded_ids = torch.cat((torch.zeros(1, 5, dtype=torch.long), before, torch.tensor([[1100, 1101]])), dim=1)
        snapshot = padded_ids.clone()
        rebuilt = self.processor.build_multi_modal_inputs(padded_ids, self.media)
        torch.testing.assert_close(padded_ids, snapshot)
        torch.testing.assert_close(online["input_ids"], before)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "huanyu_audio_lengths"):
            torch.testing.assert_close(rebuilt[key], online[key])
        for key in ("image_bound", "huanyu_audio_bounds"):
            torch.testing.assert_close(rebuilt[key], online[key] + 5)
        self.assertEqual(forward_module._image_spans(padded_ids[0], 16), [(start + 5, end + 5) for start, end in video_spans])

    def test_rollout_contraction_and_reexpansion_preserve_the_training_prompt(self):
        online = self.online()
        original = online["input_ids"].clone()
        videos_before = [video.copy() for video in self.media["videos"]]
        raw = {"image": self.media["images"], "video": self.media["videos"], "audio": self.media["audios"]}
        contracted = rollout_module.contract_rollout_prompt(self.processor, original[0].tolist(), raw)
        self.assertEqual(contracted, self.processor.tokenizer.encode(self.text))
        reexpanded = self.processor(self.processor.tokenizer.decode(contracted), **self.media)
        torch.testing.assert_close(reexpanded["input_ids"], original)
        torch.testing.assert_close(online["input_ids"], original)
        torch.testing.assert_close(reexpanded["video_grid_thw"], online["video_grid_thw"])
        for video, before in zip(self.media["videos"], videos_before, strict=True):
            np.testing.assert_array_equal(video, before)

    def test_offline_cache_processor_and_padded_forward_agree_on_video_order(self):
        online = self.online()
        cache = self.cache()
        encoders = FrozenFeatureEncoders()
        counts = precompute_module.process_row(self.row, self.processor, cache, encoders)
        self.assertEqual(counts, {"audio": 1, "vision": 3, "created": 4, "reused": 0})
        self.assertEqual(encoders.visual_grids, [[1, 2, 4], [2, 4, 4], [1, 2, 4]])
        self.processor.embedding_cache = cache
        cached = self.online()
        for key in ("input_ids", "attention_mask", "image_grid_thw", "video_grid_thw", "image_bound",
                    "huanyu_audio_bounds", "huanyu_audio_lengths"):
            torch.testing.assert_close(cached[key], online[key])
        for key in ("pixel_values", "pixel_values_videos", "huanyu_audio_features"):
            self.assertNotIn(key, cached)
        self.assertEqual(cached["huanyu_cached_video_counts"].tolist(), [2])
        self.assertEqual([len(value) for value in cached["huanyu_cached_video_embeds"]], [8, 2])
        padded_ids = torch.cat((torch.zeros(1, 3, dtype=torch.long), cached["input_ids"]), dim=1)
        rebuilt = self.processor.build_multi_modal_inputs(padded_ids, self.media)
        positions = torch.arange(padded_ids.numel()).reshape(1, 1, -1).expand(3, 1, -1)
        model = EmbeddingModel()

        def forbidden(*args):
            raise AssertionError("Cached forward must not run the audio encoder")

        forward_module.install_huanyu_forward(
            model, start_token_id=10, end_token_id=11, placeholder_token_id=12,
            image_token_id=13, video_token_id=16, spatial_merge_size=2,
            use_precompute=True, encode_audio=forbidden,
        )
        expected = torch.nn.functional.embedding(padded_ids, model.embedding.weight).detach().clone()
        for token_id, field in ((12, "huanyu_cached_audio_embeds"), (13, "huanyu_cached_vision_embeds"),
                                (16, "huanyu_cached_video_embeds")):
            expected[0, padded_ids[0] == token_id] = torch.cat(cached[field]).float()
        actual = model(padded_ids, position_ids=positions, **rebuilt)
        torch.testing.assert_close(actual, expected)
        self.assertIs(model.seen["position_ids"], positions)
        torch.testing.assert_close(model.seen["video_grid_thw"], online["video_grid_thw"])
        verified = precompute_module.process_row(self.row, self.processor, cache, verify_only=True)
        self.assertEqual(verified["reused"], 4)
        self.assertEqual(len(encoders.visual_grids), 3)

    def test_old_independent_frame_entries_cannot_satisfy_paired_video_keys(self):
        cache = self.cache()
        video = self.media["videos"][0]
        old = self.processor._vision(list(video))
        old_keys = set()
        encoders = FrozenFeatureEncoders()
        for pixels, grid in cache_module.split_vision_inputs(old["pixel_values"], old["image_grid_thw"]):
            key = cache_module.vision_key(pixels, grid)
            old_keys.add(key)
            cache.put("vision", key, encoders.vision(pixels, grid), int(grid.prod()) // 4, 4)
        paired = self.processor._video([video])
        key = cache_module.vision_key(paired["pixel_values_videos"], paired["video_grid_thw"][0])
        self.assertNotIn(key, old_keys)
        self.processor.embedding_cache = cache
        with self.assertRaisesRegex(FileNotFoundError, "cache miss"):
            self.processor(processor_module.VIDEO_PATTERN, videos=[video])


if __name__ == "__main__":
    unittest.main()
