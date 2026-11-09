"""Huanyu must keep complete video items when the legacy fallback is installed."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_prompt_updates_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
prompt = importlib.import_module(f"{PACKAGE}.prompt_updates")
video = importlib.import_module(f"{PACKAGE}.video_rollout")


def legacy_wrapper(orig_apply):
    def safe_apply(self, token_ids, updates):
        try:
            return orig_apply(self, token_ids, updates)
        except Exception:
            # The deployed fallback can keep just one of the video's 38 runs.
            return token_ids, {"video": [SimpleNamespace(offset=1, length=252)]}

    safe_apply.__module__ = "verl_omni_ext.models.qwen3_6_whisper"
    return safe_apply


def item_range(ids, offset, pad):
    mask = torch.tensor(ids) == pad
    position = SimpleNamespace(offset=offset, length=len(ids), is_embed=mask)
    return SimpleNamespace(start_idx=offset, length=len(ids), is_embed=mask, to_range=lambda: position)


class PromptUpdateTests(unittest.TestCase):
    def setUp(self):
        self.video_ids = ([14] + [16] * 252 + [15]) * 38
        self.updates = {"video": [[object()]]}
        self.range = item_range(self.video_ids, 3, 16)
        self.processor = SimpleNamespace(_find_mm_placeholders=Mock(return_value={"video": [self.range]}))

    def test_preexpanded_video_keeps_all_groups_and_mask_through_mrope(self):
        # This is an exact-full-content match supplied by vLLM's finder; no
        # updater (including the unsafe fallback) should run for this input.
        original = Mock(side_effect=AssertionError("already expanded"))
        wrapped = legacy_wrapper(original)
        ids = [1, 2, 3] + self.video_ids + [4]
        actual_ids, placeholders = prompt.apply_huanyu_prompt_updates(
            self.processor, prompt.native_prompt_updater(wrapped), ids, self.updates,
        )
        original.assert_not_called()
        self.assertIs(actual_ids, ids)
        self.assertIs(placeholders["video"][0], self.range)
        position = self.range.to_range()
        self.assertEqual(position.length, 9652)
        self.assertEqual(int(position.is_embed.sum()), 9576)
        feature = SimpleNamespace(
            mm_position=position, data={"video_grid_thw": SimpleNamespace(data=torch.tensor([38, 28, 36]))},
        )
        model = SimpleNamespace(vision_start_token_id=14, vision_end_token_id=15, video_token_id=16)
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        spans = list(video._video_grid_spans(model, config, ids, feature, len(ids)))
        self.assertEqual(spans, [(4 + group * 254, 14, 18, 252) for group in range(38)])

    def test_compact_input_calls_exact_original_updater_and_keeps_its_masks(self):
        self.processor._find_mm_placeholders.return_value = {}
        expected = (self.video_ids, {"video": [self.range]})
        original = Mock(return_value=expected)
        wrapped = legacy_wrapper(legacy_wrapper(original))
        ids = [100, 101]
        actual = prompt.apply_huanyu_prompt_updates(
            self.processor, prompt.native_prompt_updater(wrapped), ids, self.updates,
        )
        original.assert_called_once_with(self.processor, ids, self.updates)
        self.assertIs(actual[0], expected[0])
        self.assertIs(actual[1], expected[1])

    def test_original_failure_is_not_replaced_with_one_successful_pad_run(self):
        self.processor._find_mm_placeholders.return_value = {}
        error = ValueError("missing native replacement")
        original = Mock(side_effect=error)
        with self.assertRaises(ValueError) as caught:
            prompt.apply_huanyu_prompt_updates(
                self.processor, prompt.native_prompt_updater(legacy_wrapper(original)), [14, 16, 15], self.updates,
            )
        self.assertIs(caught.exception, error)

    def test_multiple_videos_with_image_audio_keep_item_order_and_masks(self):
        image_ids, audio_ids, short_video = [14, 13, 15], [20, 22, 22, 21], [14, 16, 15]
        ids = image_ids + self.video_ids + audio_ids + short_video
        placeholders = {
            "image": [item_range(image_ids, 0, 13)],
            "video": [self.range, item_range(short_video, len(ids) - 3, 16)],
            "audio": [item_range(audio_ids, len(image_ids) + len(self.video_ids), 22)],
        }
        updates = {"image": [[object()]], "video": [[object()], [object()]], "audio": [[object()]]}
        self.processor._find_mm_placeholders.return_value = placeholders
        original = Mock(side_effect=AssertionError("already expanded"))
        actual = prompt.apply_huanyu_prompt_updates(self.processor, original, ids, updates)
        original.assert_not_called()
        self.assertIs(actual[0], ids)
        self.assertIs(actual[1], placeholders)

    def test_partial_match_does_not_suppress_native_update(self):
        updates = {"video": [[object()], [object()]]}
        original = Mock(side_effect=ValueError("second video missing"))
        with self.assertRaisesRegex(ValueError, "second video missing"):
            prompt.apply_huanyu_prompt_updates(self.processor, original, self.video_ids, updates)
        original.assert_called_once()

    def test_incomplete_native_result_fails_before_engine_dispatch(self):
        self.processor._find_mm_placeholders.return_value = {}
        original = Mock(return_value=([1], {}))
        with self.assertRaisesRegex(ValueError, "item counts differ"):
            prompt.apply_huanyu_prompt_updates(self.processor, original, [1], self.updates)

    def test_unrelated_wrappers_are_preserved_and_broken_legacy_wrapper_fails(self):
        def safe_apply(self, ids, updates):
            return ids, updates

        self.assertIs(prompt.native_prompt_updater(safe_apply), safe_apply)
        safe_apply.__module__ = "verl_omni_ext.models.qwen3_6_whisper"
        with self.assertRaisesRegex(RuntimeError, "cannot resolve legacy"):
            prompt.native_prompt_updater(safe_apply)


if __name__ == "__main__":
    unittest.main()
