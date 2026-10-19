"""CPU checks for the Huanyu multimodal data contract, without model weights."""

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_processor_bridge_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.processor", ROOT / "processor.py")
processor_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = processor_module
spec.loader.exec_module(processor_module)


class Tokenizer:
    markers = {
        "<|audio_start|>": 10, "<|audio_end|>": 11, "<unk>": 12,
        "<|image_pad|>": 13, "<|vision_start|>": 14, "<|vision_end|>": 15,
    }
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

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)


class Frontend:
    def extract_fbank(self, waveform):
        frames = (len(waveform) - 400) // 160 + 1
        return torch.ones(40, frames), frames


def vision(images, **kwargs):
    return {
        "pixel_values": torch.ones(len(images) * 16, 24),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
    }


def processor():
    config = SimpleNamespace(
        audio_start_token_id=10, audio_end_token_id=11, image_token_id=13,
        vision_start_token_id=14, vision_end_token_id=15,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    return processor_module.HuanyuProcessor(
        Tokenizer(), vision, Frontend(), config,
        {"sampling_rate": 16000, "feature_size": 40},
    )


class HuanyuProcessorTests(unittest.TestCase):
    def test_audio_and_image_bounds_match_token_slots_and_true_lengths(self):
        proc = processor()
        audio = np.zeros(16000, dtype=np.float32)
        result = proc(
            ["question " + processor_module.IMAGE_PATTERN + " " + processor_module.AUDIO_PATTERN],
            images=[object()], audios=[audio],
        )
        self.assertEqual(result["huanyu_audio_lengths"].tolist(), [98])
        self.assertEqual(result["huanyu_audio_chunk_counts"].tolist(), [1])
        self.assertEqual(result["huanyu_audio_features"][0].shape, (40, 98))
        bounds = result["huanyu_audio_bounds"].tolist()
        self.assertEqual(bounds[0][1] - bounds[0][0], 12)
        begin, end = bounds[0]
        self.assertEqual(result["input_ids"][0, begin:end].tolist(), [12] * 12)
        self.assertEqual(result["image_bound"].diff(dim=-1).item(), 4)

    def test_rebuild_uses_actual_padded_ids_instead_of_decoded_text(self):
        proc = processor()
        waveform = np.zeros(16000, dtype=np.float32)
        result = proc(processor_module.AUDIO_PATTERN, audios=[waveform])
        original = result["huanyu_audio_bounds"].clone()
        ids = torch.cat([torch.zeros((1, 7), dtype=torch.long), result["input_ids"], torch.tensor([[1001, 1002]])], dim=1)
        rebuilt = proc.build_multi_modal_inputs(ids, {"audios": [waveform]})
        torch.testing.assert_close(rebuilt["huanyu_audio_bounds"], original + 7)
        self.assertEqual(rebuilt["image_bound"].shape, (0, 2))

    def test_long_audio_chunk_lengths_are_not_padded_lengths(self):
        proc = processor()
        data = proc(processor_module.AUDIO_PATTERN, audios=[np.zeros(31 * 16000, np.float32)])
        self.assertEqual(data["huanyu_audio_lengths"].tolist(), [2998, 98])
        self.assertEqual(data["huanyu_audio_chunk_counts"].tolist(), [2])
        self.assertEqual(data["huanyu_audio_bounds"].diff(dim=-1).flatten().tolist(), [375, 12])

    def test_two_distinct_audios_keep_item_boundaries_in_chat_template(self):
        proc = processor()
        messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "audio"}]}]
        rendered = proc.apply_chat_template(messages)
        self.assertEqual(rendered, processor_module.AUDIO_PATTERN + "\n" + processor_module.AUDIO_PATTERN)

    def test_short_audio_tail_and_wrong_rate_are_rejected(self):
        proc = processor()
        with self.assertRaisesRegex(ValueError, "too short"):
            proc(processor_module.AUDIO_PATTERN, audios=[np.zeros(30 * 16000 + 200, np.float32)])
        with self.assertRaisesRegex(ValueError, "16000"):
            proc(processor_module.AUDIO_PATTERN, audios=[np.zeros(16000, np.float32)], sampling_rate=48000)

    def test_feature_width_mismatch_never_falls_back_to_whisper(self):
        proc = processor()
        proc.audio_frontend = SimpleNamespace(extract_fbank=lambda w: (torch.zeros(128, 98), 98))
        with self.assertRaisesRegex(ValueError, "expected.*40"):
            proc(processor_module.AUDIO_PATTERN, audios=[np.zeros(16000, np.float32)])

    def test_audio_and_image_alignment_errors_fail_instead_of_dropping_media(self):
        proc = processor()
        with self.assertRaisesRegex(ValueError, "span count"):
            proc.build_multi_modal_inputs(torch.tensor([[1000]]), {"audios": [np.zeros(16000, np.float32)]})
        with self.assertRaisesRegex(ValueError, "image token lengths"):
            proc.build_multi_modal_inputs(torch.tensor([[14, 13, 15]]), {"images": [object()]})

    def test_empty_keys_keep_per_sample_collation_contract(self):
        result = processor().build_multi_modal_inputs(torch.tensor([[1000, 1001]]), {})
        self.assertEqual(result["huanyu_audio_features"], [])
        self.assertEqual(result["huanyu_audio_lengths"].shape, (0,))
        self.assertEqual(result["huanyu_audio_chunk_counts"].tolist(), [0])
        self.assertEqual(result["huanyu_audio_bounds"].shape, (0, 2))
        self.assertEqual(result["image_bound"].shape, (0, 2))

    def test_dedup_requires_media_aware_rollout_hook(self):
        proc = processor()
        self.assertEqual(proc.dedup_pad_tokens([1000, 1001]), [1000, 1001])
        with self.assertRaisesRegex(RuntimeError, "metadata-aware"):
            proc.dedup_pad_tokens([10, 12, 11])

    def test_native_rope_is_used_and_integer_positions_preserved(self):
        class Model:
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                assert image_grid_thw is grids
                return torch.ones((3, 1, 2), dtype=torch.float32) * 2053, torch.tensor([[3]])
        grids = torch.tensor([[1, 4, 4]])
        proc = processor()
        with patch.object(processor_module.importlib, "import_module", return_value=SimpleNamespace(Qwen3_5MoeModel=Model)):
            processor_module._bind_qwen_rope(proc)
        ids, deltas = proc.get_rope_index(torch.tensor([[1, 2]]), image_grid_thw=grids, unrelated_kwarg=True)
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(ids[0, 0, 0].item(), 2053)
        self.assertEqual(deltas.item(), 3)

    def test_new_builder_checks_actual_frontend_width_before_loading_runtime(self):
        config = SimpleNamespace(audio_config={
            "model_type": "huanyu_conformer_0_6b", "input_feat_per_channel": 40,
            "projector_downsample_rate": 2,
        })
        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config), AutoTokenizer=object(),
        )
        image_module = SimpleNamespace(Qwen2VLImageProcessor=object())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            (Path(directory) / "audio_preprocessor_config.json").write_text(
                '{"feature_type":"fbank","feature_size":128,"sampling_rate":16000}'
            )
            with patch.dict(sys.modules, {
                "transformers": fake_transformers,
                "transformers.models.qwen2_vl.image_processing_qwen2_vl": image_module,
            }):
                with self.assertRaisesRegex(ValueError, "feature width"):
                    processor_module.build_huanyu_processor(directory, {})

    def bind_native_rope(self, proc, method):
        owner = type("Model", (), {"get_rope_index": method})
        with patch.object(processor_module.importlib, "import_module", return_value=SimpleNamespace(Qwen3_5MoeModel=owner)):
            processor_module._bind_qwen_rope(proc)

    def test_required_token_types_follow_padded_image_audio_sequence(self):
        proc = processor()
        # Two images separated by audio; padding intentionally uses image id.
        ids = torch.tensor([[13, 1001, 14, 13, 13, 13, 13, 15,
                             10, 12, 12, 11, 14, 13, 13, 13, 13, 15, 1002, 13]])
        mask = torch.tensor([[0] + [1] * 18 + [0]])
        grids = torch.tensor([[1, 4, 4], [1, 4, 4]])
        before = ids.clone()
        native_positions = torch.arange(60).reshape(3, 1, 20) + 2053
        native_deltas = torch.tensor([[-4]])

        def native(owner, input_ids, mm_token_type_ids, image_grid_thw=None, attention_mask=None):
            self.assertIs(input_ids, ids)
            self.assertIs(image_grid_thw, grids)
            self.assertIs(attention_mask, mask)
            self.assertEqual(mm_token_type_ids.tolist(), [[0, 0, 0, 1, 1, 1, 1, 0,
                                                         0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]])
            self.assertEqual(mm_token_type_ids.dtype, torch.long)
            self.assertEqual(mm_token_type_ids.device, ids.device)
            return native_positions, native_deltas

        self.bind_native_rope(proc, native)
        positions, deltas = proc.get_rope_index(input_ids=ids, image_grid_thw=grids, attention_mask=mask)
        torch.testing.assert_close(ids, before)
        torch.testing.assert_close(positions, native_positions)
        self.assertIs(deltas, native_deltas)

    def test_required_token_types_for_text_and_audio_are_zero(self):
        proc = processor()
        for values in ([1001, 1002, 1003, 1004], [10, 12, 12, 11]):
            with self.subTest(values=values):
                ids = torch.tensor([values])

                def native(owner, input_ids, mm_token_type_ids, image_grid_thw=None):
                    self.assertEqual(mm_token_type_ids.tolist(), [[0, 0, 0, 0]])
                    self.assertIsNone(image_grid_thw)
                    return torch.arange(4).expand(3, 1, 4), torch.zeros(1, 1, dtype=torch.long)

                self.bind_native_rope(proc, native)
                positions, _ = proc.get_rope_index(ids)
                self.assertEqual(positions.shape, (3, 1, 4))

    def test_supplied_token_types_are_preserved_for_positional_and_keyword_calls(self):
        proc = processor()
        ids = torch.tensor([[1001, 1002]])
        supplied = torch.zeros_like(ids, dtype=torch.int32)

        def native(owner, input_ids, mm_token_type_ids, image_grid_thw=None):
            self.assertIs(mm_token_type_ids, supplied)
            return torch.zeros(3, 1, 2, dtype=torch.long), torch.zeros(1, 1)

        self.bind_native_rope(proc, native)
        proc.get_rope_index(ids, supplied)
        proc.get_rope_index(input_ids=ids, mm_token_type_ids=supplied)

    def test_explicit_none_token_types_are_completed(self):
        proc = processor()
        ids = torch.tensor([[14, 13, 15]])

        def native(owner, input_ids, *, mm_token_type_ids, image_grid_thw=None):
            self.assertEqual(mm_token_type_ids.tolist(), [[0, 1, 0]])
            return torch.zeros(3, 1, 3, dtype=torch.long), torch.zeros(1, 1)

        self.bind_native_rope(proc, native)
        proc.get_rope_index(ids, mm_token_type_ids=None, image_grid_thw=torch.tensor([[1, 2, 2]]))

    def test_video_token_types_differ_from_decoded_image_frames(self):
        proc = processor()
        proc.config.video_token_id = 16
        ids = torch.tensor([[14, 13, 15, 14, 16, 15], [13, 14, 13, 15, 1000, 13]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]])
        types = processor_module._rope_token_types(proc, ids, mask)
        self.assertEqual(types.tolist(), [[0, 1, 0, 0, 2, 0], [0, 0, 1, 0, 0, 0]])

    def test_token_type_completion_rejects_malformed_input_or_padding_mask(self):
        proc = processor()
        with self.assertRaisesRegex(ValueError, "input_ids.*shape"):
            processor_module._rope_token_types(proc, torch.tensor([1, 2]))
        with self.assertRaisesRegex(ValueError, "attention_mask must match"):
            processor_module._rope_token_types(proc, torch.tensor([[1, 2]]), torch.tensor([1, 1]))

    def test_agent_loop_hook_leaves_other_models_unchanged(self):
        class Worker:
            def _compute_multi_modal_inputs(self, output, input_ids):
                return "original"

        class TQ(Worker):
            pass

        modules = {
            "verl.experimental.agent_loop.agent_loop": SimpleNamespace(AgentLoopWorker=Worker),
            "verl.trainer.ppo.v1.agent_loop_tq": SimpleNamespace(AgentLoopWorkerTQ=TQ),
        }
        with patch.object(processor_module.importlib, "import_module", side_effect=modules.__getitem__):
            processor_module.install_agent_loop_bridge()
            processor_module.install_agent_loop_bridge()
        worker = TQ()
        worker.processor = object()
        output = SimpleNamespace(multi_modal_data={}, mm_processor_kwargs={})
        self.assertEqual(worker._compute_multi_modal_inputs(output, torch.tensor([[1000]])), "original")
        worker.processor = processor()
        self.assertIn("huanyu_audio_features", worker._compute_multi_modal_inputs(output, torch.tensor([[1000]])))


if __name__ == "__main__":
    unittest.main()
