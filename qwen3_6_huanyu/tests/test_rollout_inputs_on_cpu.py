"""Huanyu rollout item/chunk contracts, without Ray, vLLM or model weights."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


_PATH = Path(__file__).resolve().parents[1] / "rollout_inputs.py"
_SPEC = importlib.util.spec_from_file_location("huanyu_rollout_inputs_tested", _PATH)
rollout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollout)


class Waveform:
    ndim = 1

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"(<image>./</image>)": [100, 101], "(<audio>./</audio>)": [200, 201]}[text]


class Processor:
    _huanyu_training_processor = True
    vision_start_token_id, vision_end_token_id, image_token_id = 10, 11, 12
    audio_start_token_id, audio_end_token_id, audio_placeholder_token_id = 20, 21, 22
    sampling_rate, n_fft, hop_length = 16000, 400, 160
    audio_chunk_seconds, audio_encoder_downsample, audio_projector_downsample = 30, 4, 2
    tokenizer = Tokenizer()

    def dedup_pad_tokens(self, ids):
        if self.audio_start_token_id in ids or self.vision_start_token_id in ids:
            raise RuntimeError("metadata contraction was not performed")
        return ids


def audio_span(length):
    return [20] + [22] * length + [21]


class RolloutInputsTest(unittest.TestCase):
    def test_long_audio_is_one_item_and_adjacent_second_audio_stays_separate(self):
        # A 31s waveform has 375+12 tokens; an adjacent 1s waveform has 12.
        # Their source item boundary cannot be inferred from token adjacency.
        ids = [7] + audio_span(375) + audio_span(12) + audio_span(12) + [8]
        original = list(ids)
        actual = rollout.contract_rollout_prompt(
            Processor(), ids, {"audio": [Waveform(31 * 16000), Waveform(16000)]}
        )
        self.assertEqual(actual, [7, 200, 201, 200, 201, 8])
        self.assertEqual(ids, original)

    def test_interleaved_images_text_and_audio_preserve_order(self):
        ids = [10, 12, 12, 11, 7] + audio_span(12) + [8, 10, 12, 11]
        actual = rollout.contract_rollout_prompt(
            Processor(), ids, {"image": [object(), object()], "audio": [(Waveform(16000), 16000)]}
        )
        self.assertEqual(actual, [100, 101, 7, 200, 201, 8, 100, 101])

    def test_audio_counts_lengths_and_short_tail_fail(self):
        for ids, media in (
            (audio_span(11), {"audio": [Waveform(16000)]}),
            (audio_span(12), {"audio": [Waveform(31 * 16000)]}),
            (audio_span(12) * 2, {"audio": [Waveform(16000)]}),
            ([7], {"audio": [Waveform(16000)]}),
            (audio_span(375), {"audio": [Waveform(30 * 16000 + 100)]}),
            (audio_span(12), {"audio": [(Waveform(16000), 8000)]}),
        ):
            with self.subTest(ids_count=len(ids), media=media), self.assertRaises(ValueError):
                rollout.contract_rollout_prompt(Processor(), ids, media)

    def test_malformed_or_missing_image_markers_fail(self):
        for ids, media in (
            ([10, 11], {"image": [object()]}),
            ([10, 12, 99, 11], {"image": [object()]}),
            ([10, 12, 11], {}),
            ([11], {}),
            ([12], {}),
            ([7], {"image": [object()]}),
            ([100, 101, 10, 12, 11], {"image": [object()]}),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rollout.contract_rollout_prompt(Processor(), ids, media)

    def test_text_only_unchanged_and_video_requires_frame_conversion(self):
        self.assertEqual(rollout.contract_rollout_prompt(Processor(), [7, 22, 8], {}), [7, 22, 8])
        with self.assertRaises(ValueError):
            rollout.contract_rollout_prompt(Processor(), [7], {"video": [object()]})
        with self.assertRaises(TypeError):
            rollout.contract_rollout_prompt(Processor(), [True], {})

    def test_gated_server_wrapper_preserves_other_models_and_is_idempotent(self):
        class Server:
            def _preprocess_input(self, prompt_ids, sampling_params, multi_modal_data, *, sentinel=None):
                processor = self.model_config.processor
                if getattr(processor, "_huanyu_training_processor", False):
                    prompt_ids = processor.dedup_pad_tokens(prompt_ids)
                return prompt_ids, sampling_params, multi_modal_data, sentinel

        rollout._install_on_server_class(Server)
        installed = Server._preprocess_input
        rollout._install_on_server_class(Server)
        self.assertIs(Server._preprocess_input, installed)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        params = {"max_tokens": 17}
        media = {"audio": [Waveform(16000)]}
        result = server._preprocess_input(
            prompt_ids=audio_span(12), sampling_params=params, multi_modal_data=media, sentinel="same"
        )
        self.assertEqual(result[0], [200, 201])
        self.assertIs(result[1], params)
        self.assertIs(result[2], media)
        self.assertEqual(result[3], "same")
        server.model_config = SimpleNamespace(processor=object())
        result = server._preprocess_input("opaque_other_model_ids", params, None)
        self.assertEqual(result[0], "opaque_other_model_ids")
        self.assertIsNone(result[2])

    def test_incompatible_server_signature_rejected(self):
        class Server:
            def _preprocess_input(self, unrelated):
                return unrelated

        with self.assertRaises(RuntimeError):
            rollout._install_on_server_class(Server)


if __name__ == "__main__":
    unittest.main()
