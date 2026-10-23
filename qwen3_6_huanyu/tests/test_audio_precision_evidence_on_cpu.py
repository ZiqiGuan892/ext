"""Real tensors verify isolated audio shapes and successful injection evidence."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_cached_forward_on_cpu as fixture


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_audio_isolation_cpu", ROOT / "audio_rollout.py")
audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio)
forward = fixture.forward


class NativeAudio:
    def __init__(self):
        self.calls = []
        self.invalid_groups = False
        self.fail_on_call = None

    def get_audio_hidden_states(self, data):
        chunk, length = data["audio_features"][0], data["audio_feature_lens"][0]
        self.calls.append((chunk.clone(), length.clone()))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("native encoder failure")
        if self.invalid_groups:
            return []
        return [chunk.transpose(0, 1) * 2]


class AudioIsolationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MINICPMO_ISOLATE_MEDIA_ENCODERS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_real_chunks_are_cropped_and_regrouped_without_cross_item_padding(self):
        thinker = NativeAudio()
        native = SimpleNamespace(thinker=thinker)
        with redirect_stdout(io.StringIO()) as installed:
            self.assertTrue(audio.install_audio_isolation(native))
            hook = thinker.get_audio_hidden_states
            self.assertTrue(audio.install_audio_isolation(native))
        self.assertIs(thinker.get_audio_hidden_states, hook)
        self.assertEqual(installed.getvalue().count("phase=installed"), 1)
        self.assertNotIn("phase=executed", installed.getvalue())
        raw = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
        lengths = [torch.tensor([5, 3]), torch.tensor([7])]
        with redirect_stdout(io.StringIO()) as output:
            actual = thinker.get_audio_hidden_states({"audio_features": raw, "audio_feature_lens": lengths})
            thinker.get_audio_hidden_states({"audio_features": raw, "audio_feature_lens": lengths})
        self.assertEqual([tuple(chunk.shape) for chunk, _ in thinker.calls[:3]], [(2, 5), (2, 3), (2, 7)])
        torch.testing.assert_close(actual[0], torch.cat([raw[0, :, :5].t(), raw[1, :, :3].t()]) * 2)
        torch.testing.assert_close(actual[1], raw[2, :, :7].t() * 2)
        self.assertEqual(output.getvalue().count("phase=executed"), 1)
        self.assertIn("chunks=3 groups=2", output.getvalue())

    def test_tensor_lengths_and_list_features_preserve_values_and_gradients(self):
        native = NativeAudio()
        audio.install_audio_isolation(native)
        chunks = [torch.randn(2, 8, requires_grad=True), torch.randn(2, 6, requires_grad=True)]
        results = native.get_audio_hidden_states({"audio_features": chunks,
                                                 "audio_feature_lens": torch.tensor([[4], [3]])})
        self.assertEqual([tuple(value.shape) for value in results], [(4, 2), (3, 2)])
        sum(value.sum() for value in results).backward()
        for chunk, length in zip(chunks, (4, 3)):
            expected = torch.zeros_like(chunk)
            expected[:, :length] = 2
            torch.testing.assert_close(chunk.grad, expected)

    def test_invalid_or_empty_input_and_native_failure_do_not_claim_execution(self):
        cases = [
            ({"audio_features": [torch.ones(2, 5)], "audio_feature_lens": []}, ValueError, "count mismatch"),
            ({"audio_features": [torch.ones(2, 5)], "audio_feature_lens": [torch.tensor([0])]}, ValueError, "frame count"),
            ({"audio_features": [torch.ones(2, 5)], "audio_feature_lens": [torch.tensor([6])]}, ValueError, "frame count"),
            ({"audio_features": [], "audio_feature_lens": [torch.tensor([], dtype=torch.long)]}, ValueError, "at least one"),
        ]
        for data, error, text in cases:
            with self.subTest(text=text):
                native = NativeAudio()
                audio.install_audio_isolation(native)
                with redirect_stdout(io.StringIO()) as output:
                    with self.assertRaisesRegex(error, text):
                        native.get_audio_hidden_states(data)
                self.assertNotIn("phase=executed", output.getvalue())
        for mode in ("bad_return", "second_chunk_failure", "empty"):
            with self.subTest(mode=mode):
                native = NativeAudio()
                audio.install_audio_isolation(native)
                data = {"audio_features": [torch.ones(2, 5)] * 2,
                        "audio_feature_lens": [torch.tensor([5, 5])]}
                with redirect_stdout(io.StringIO()) as output:
                    if mode == "empty":
                        self.assertEqual(native.get_audio_hidden_states({"audio_features": [], "audio_feature_lens": []}), [])
                    else:
                        native.invalid_groups = mode == "bad_return"
                        native.fail_on_call = 2 if mode == "second_chunk_failure" else None
                        with self.assertRaises(RuntimeError):
                            native.get_audio_hidden_states(data)
                self.assertNotIn("phase=executed", output.getvalue())

    def test_disabled_keeps_native_method_and_missing_method_fails(self):
        native = NativeAudio()
        original = native.get_audio_hidden_states
        with patch.dict(os.environ, {"MINICPMO_ISOLATE_MEDIA_ENCODERS": "0"}), redirect_stdout(io.StringIO()) as output:
            self.assertFalse(audio.install_audio_isolation(native))
        self.assertEqual(native.get_audio_hidden_states, original)
        self.assertIn("phase=disabled", output.getvalue())
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "requires native get_audio_hidden_states"):
                audio.install_audio_isolation(SimpleNamespace())
        self.assertNotIn("phase=installed", output.getvalue())


class InjectionEvidenceTests(unittest.TestCase):
    def make_model(self, *, cached=False, trainable=False, behavior="normal"):
        class HFModel(fixture.RecordingHFModel):
            def forward(self, input_ids, **kwargs):
                if behavior == "skip_embedding":
                    return torch.zeros((*input_ids.shape, 4))
                result = super().forward(input_ids, **kwargs)
                if behavior == "fail_after_embedding":
                    raise RuntimeError("decoder failure")
                return result

        model = HFModel()
        model.projector = torch.nn.Linear(2, 4, bias=False)

        def encode_audio(module, batch, lens):
            values = [module.projector(chunk[:, :((int(length) + 3) // 4) // 2].t())
                      for chunk, length in zip(batch, lens)]
            return values if trainable else [value.detach() for value in values]

        forward.install_huanyu_forward(model, start_token_id=2, end_token_id=3, placeholder_token_id=4,
                                       image_token_id=6, use_precompute=cached,
                                       require_audio_only=trainable, encode_audio=encode_audio)
        return model

    def online_inputs(self):
        return torch.tensor([[9, 2, 4, 4, 3, 10]]), {
            "huanyu_audio_features": [torch.arange(32, dtype=torch.float32).reshape(2, 16)],
            "huanyu_audio_lengths": [16], "huanyu_audio_chunk_counts": [1],
        }

    def test_online_success_reports_once_with_actual_embedding_values_and_gradient(self):
        for trainable in (False, True):
            with self.subTest(trainable=trainable):
                model = self.make_model(trainable=trainable)
                ids, kwargs = self.online_inputs()
                expected = model.embedding(ids).detach().clone()
                expected[0, 2:4] = model.projector(kwargs["huanyu_audio_features"][0][:, :2].t())
                with redirect_stdout(io.StringIO()) as output:
                    actual = model(ids, **kwargs)
                    model(ids, **kwargs)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                if trainable:
                    self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
                else:
                    self.assertIsNone(model.projector.weight.grad)
                self.assertEqual(output.getvalue().count("component=embedding_injection phase=executed"), 1)
                self.assertIn("mode=online_trainable" if trainable else "mode=online_frozen", output.getvalue())
                self.assertIn("audio_tokens=2", output.getvalue())
                self.assertIn(f"audio_requires_grad={trainable}", output.getvalue())

    def test_cached_success_reports_once_and_uses_verified_tensor_values(self):
        model = self.make_model(cached=True)
        ids, kwargs = fixture.mixed_inputs()
        expected = model.embedding(ids).detach().clone()
        expected[1, 3:5], expected[1, 7:9], expected[2, 8:9] = 9, 7, 8
        with redirect_stdout(io.StringIO()) as output:
            actual = model(ids, **kwargs)
            model(ids, **kwargs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(output.getvalue().count("component=embedding_injection phase=executed"), 1)
        self.assertIn("mode=cached_frozen chunks=2 audio_tokens=3", output.getvalue())

    def test_failed_alignment_or_decoder_consumption_never_reports_success(self):
        for cached in (False, True):
            for behavior in ("normal", "skip_embedding", "fail_after_embedding"):
                with self.subTest(cached=cached, behavior=behavior):
                    model = self.make_model(cached=cached, behavior=behavior)
                    ids, kwargs = fixture.mixed_inputs() if cached else self.online_inputs()
                    if behavior == "normal":
                        kwargs["huanyu_audio_chunk_counts"] = [99]
                    with redirect_stdout(io.StringIO()) as output:
                        with self.assertRaises((ValueError, RuntimeError)):
                            model(ids, **kwargs)
                    self.assertNotIn("phase=executed", output.getvalue())
                    self.assertIsNone(forward._ACTIVE_AUDIO.get())

    def test_text_only_does_not_claim_audio_execution_for_online_or_cached_path(self):
        ids = torch.tensor([[9, 10]])
        for cached in (False, True):
            with self.subTest(cached=cached):
                model = self.make_model(cached=cached)
                kwargs = ({"huanyu_cache_enabled": [1], "huanyu_cached_audio_embeds": [],
                           "huanyu_cached_vision_embeds": [], "huanyu_audio_lengths": [],
                           "huanyu_audio_chunk_counts": [0], "huanyu_cached_image_counts": [0]}
                          if cached else {})
                with redirect_stdout(io.StringIO()) as output:
                    actual = model(ids, **kwargs)
                torch.testing.assert_close(actual, model.embedding(ids))
                self.assertNotIn("phase=executed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
