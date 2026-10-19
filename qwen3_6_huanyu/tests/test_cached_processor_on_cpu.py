"""Cache use must preserve rollout media and the processor's sequence contract."""

import importlib
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_cached_processor_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
processor_module = importlib.import_module(f"{PACKAGE}.processor")
cache_module = importlib.import_module(f"{PACKAGE}.cache")


class Tokenizer:
    markers = {"<|audio_start|>": 10, "<|audio_end|>": 11, "<unk>": 12,
               "<|image_pad|>": 13, "<|vision_start|>": 14, "<|vision_end|>": 15}
    chat_template = "fixture"

    def encode(self, text, **kwargs):
        parts = re.split("(" + "|".join(map(re.escape, self.markers)) + ")", text)
        result = []
        for part in parts:
            result.extend([self.markers[part]] if part in self.markers else [1000 + ord(c) for c in part])
        return result

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Frontend:
    def extract_fbank(self, waveform):
        frames = (len(waveform) - 400) // 160 + 1
        return torch.full((40, frames), float(waveform[0]), dtype=torch.float32), frames


def vision(images, **kwargs):
    pixels = [torch.full((8, 24), float(image[0, 0, 0]), dtype=torch.float32) for image in images]
    return {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.tensor([[1, 2, 4]] * len(images))}


class StrictFixtureCache:
    def __init__(self, values):
        self.values = dict(values)
        self.calls = []

    def get(self, kind, key, *, expected_tokens, hidden_size):
        self.calls.append((kind, key, expected_tokens, hidden_size))
        if (kind, key) not in self.values:
            raise FileNotFoundError(f"Missing {kind} cache entry")
        value = self.values[(kind, key)]
        if value.shape != (expected_tokens, hidden_size):
            raise ValueError("Fixture cache shape differs from the requested media contract")
        return value


def make_processor():
    config = SimpleNamespace(
        audio_start_token_id=10, audio_end_token_id=11, image_token_id=13,
        vision_start_token_id=14, vision_end_token_id=15,
        vision_config=SimpleNamespace(spatial_merge_size=2), text_config=SimpleNamespace(hidden_size=4),
    )
    return processor_module.HuanyuProcessor(
        Tokenizer(), vision, Frontend(), config, {"sampling_rate": 16000, "feature_size": 40},
    )


def audio_entry(value, frames, embed_value):
    feature = torch.full((40, frames), value, dtype=torch.float32)
    key = cache_module.audio_key(feature, frames)
    tokens = ((frames + 3) // 4) // 2
    return ("audio", key), torch.full((tokens, 4), embed_value, dtype=torch.bfloat16)


def vision_entry(value, embed_value):
    pixels = torch.full((8, 24), value, dtype=torch.float32)
    grid = torch.tensor([1, 2, 4])
    key = cache_module.vision_key(pixels, grid)
    return ("vision", key), torch.full((2, 4), embed_value, dtype=torch.bfloat16)


class CachedProcessorTests(unittest.TestCase):
    def test_cached_call_keeps_ids_masks_bounds_and_grids_identical_to_online(self):
        proc = make_processor()
        image = np.full((2, 2, 3), 3, dtype=np.float32)
        waveform = np.ones(16000, dtype=np.float32)
        text = "question " + processor_module.IMAGE_PATTERN + " " + processor_module.AUDIO_PATTERN
        online = proc(text, images=[image], audios=[waveform])
        proc.embedding_cache = StrictFixtureCache(dict([audio_entry(1, 98, 7), vision_entry(3, 9)]))
        cached = proc(text, images=[image], audios=[waveform])
        for field in ("input_ids", "attention_mask", "image_grid_thw", "image_bound",
                      "huanyu_audio_bounds", "huanyu_audio_lengths", "huanyu_audio_chunk_counts"):
            torch.testing.assert_close(cached[field], online[field])
        self.assertNotIn("pixel_values", cached)
        self.assertNotIn("huanyu_audio_features", cached)
        self.assertEqual(cached["huanyu_cached_image_counts"].tolist(), [1])
        self.assertEqual(cached["huanyu_cache_enabled"].tolist(), [1])
        torch.testing.assert_close(cached["huanyu_cached_audio_embeds"][0], torch.full((12, 4), 7, dtype=torch.bfloat16))
        torch.testing.assert_close(cached["huanyu_cached_vision_embeds"][0], torch.full((2, 4), 9, dtype=torch.bfloat16))
        self.assertEqual([(kind, tokens, hidden) for kind, _, tokens, hidden in proc.embedding_cache.calls],
                         [("audio", 12, 4), ("vision", 2, 4)])

    def test_agent_loop_rebuild_retains_raw_rollout_media_and_padded_alignment(self):
        proc = make_processor()
        images = [np.full((2, 2, 3), 3, np.float32), np.full((2, 2, 3), 5, np.float32)]
        waveform = np.ones(16000, dtype=np.float32)
        text = processor_module.IMAGE_PATTERN + processor_module.AUDIO_PATTERN + processor_module.IMAGE_PATTERN
        original = proc(text, images=images, audios=[waveform])
        padded_ids = torch.cat([torch.zeros((1, 5), dtype=torch.long), original["input_ids"], torch.tensor([[1100, 1101]])], dim=1)
        media = {"images": images, "audios": [waveform]}
        proc.embedding_cache = StrictFixtureCache(dict([audio_entry(1, 98, 7), vision_entry(3, 9), vision_entry(5, 11)]))
        rebuilt = proc.build_multi_modal_inputs(padded_ids, media)
        self.assertIs(media["images"], images)
        self.assertIs(media["audios"][0], waveform)
        np.testing.assert_array_equal(images[0], np.full((2, 2, 3), 3, np.float32))
        np.testing.assert_array_equal(waveform, np.ones(16000, np.float32))
        torch.testing.assert_close(rebuilt["image_bound"], original["image_bound"] + 5)
        torch.testing.assert_close(rebuilt["huanyu_audio_bounds"], original["huanyu_audio_bounds"] + 5)
        self.assertEqual([item[0, 0].item() for item in rebuilt["huanyu_cached_vision_embeds"]], [9, 11])
        self.assertEqual(rebuilt["huanyu_cached_image_counts"].tolist(), [2])
        self.assertNotIn("pixel_values", rebuilt)
        self.assertNotIn("huanyu_audio_features", rebuilt)

    def test_long_audio_cache_preserves_two_chunks_and_true_lengths(self):
        proc = make_processor()
        proc.embedding_cache = StrictFixtureCache(dict([audio_entry(1, 2998, 6), audio_entry(1, 98, 7)]))
        result = proc(processor_module.AUDIO_PATTERN, audios=[np.ones(31 * 16000, np.float32)])
        self.assertEqual(result["huanyu_audio_lengths"].tolist(), [2998, 98])
        self.assertEqual(result["huanyu_audio_chunk_counts"].tolist(), [2])
        self.assertEqual([len(item) for item in result["huanyu_cached_audio_embeds"]], [375, 12])
        self.assertEqual(result["huanyu_audio_bounds"].diff(dim=-1).flatten().tolist(), [375, 12])
        self.assertEqual([item[0, 0].item() for item in result["huanyu_cached_audio_embeds"]], [6, 7])

    def test_cache_miss_does_not_return_raw_encoder_inputs(self):
        proc = make_processor()
        proc.embedding_cache = StrictFixtureCache({})
        for text, kwargs in (
            (processor_module.AUDIO_PATTERN, {"audios": [np.ones(16000, np.float32)]}),
            (processor_module.IMAGE_PATTERN, {"images": [np.ones((2, 2, 3), np.float32)]}),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(FileNotFoundError, "Missing"):
                    proc(text, **kwargs)

    def test_text_only_cached_samples_emit_uniform_batch_metadata_without_lookups(self):
        proc = make_processor()
        proc.embedding_cache = StrictFixtureCache({})
        result = proc("plain text")
        self.assertEqual(result["huanyu_cached_audio_embeds"], [])
        self.assertEqual(result["huanyu_cached_vision_embeds"], [])
        self.assertEqual(result["huanyu_audio_chunk_counts"].tolist(), [0])
        self.assertEqual(result["huanyu_cached_image_counts"].tolist(), [0])
        self.assertEqual(result["huanyu_cache_enabled"].tolist(), [1])
        self.assertFalse(proc.embedding_cache.calls)
        self.assertNotIn("huanyu_audio_features", result)

    def test_online_mode_still_supplies_features_without_cache_metadata(self):
        result = make_processor()(processor_module.AUDIO_PATTERN, audios=[np.ones(16000, np.float32)])
        self.assertEqual(tuple(result["huanyu_audio_features"][0].shape), (40, 98))
        self.assertNotIn("huanyu_cache_enabled", result)


if __name__ == "__main__":
    unittest.main()
