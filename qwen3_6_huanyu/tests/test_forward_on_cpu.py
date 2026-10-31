"""Exercise the Huanyu/HF forward boundary without Transformers or vLLM."""

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_forward_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
forward = importlib.import_module(f"{PACKAGE}.forward")


class ToyHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.image_seen = None
        self.fail = False
        self.skip_embedding = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None, **kwargs):
        self.image_seen = (pixel_values, image_grid_thw)
        self.kwargs_seen = kwargs
        if self.fail:
            raise RuntimeError("original forward failed")
        if self.skip_embedding:
            return torch.zeros((*input_ids.shape, 4))
        return self.embedding(input_ids)


class ForwardTests(unittest.TestCase):
    def make_model(self, chunks):
        model = ToyHFModel()
        calls = []

        def encode(owner, features, lengths):
            self.assertIs(owner, model)
            calls.append((features, lengths))
            return chunks

        forward.install_huanyu_forward(
            model, start_token_id=2, end_token_id=3, placeholder_token_id=4, encode_audio=encode
        )
        return model, calls

    def test_mixed_batch_padding_and_two_chunks_keep_image_inputs(self):
        chunks = [torch.full((2, 4), 7.0), torch.full((1, 4), 8.0)]
        model, calls = self.make_model(chunks)
        ids = torch.tensor([[0, 0, 9, 10, 11, 12, 13, 14, 15],
                            [0, 2, 4, 4, 3, 2, 4, 3, 10]])
        before = model.embedding(ids).detach().clone()
        pixels = torch.ones(4, 8)
        grid = torch.tensor([[1, 2, 2]])
        result = model(
            ids, huanyu_audio_features=[[], [torch.ones(40, 16), torch.ones(40, 8)]],
            huanyu_audio_lengths=[[], torch.tensor([16, 8])],
            huanyu_audio_chunk_counts=[torch.tensor([0]), torch.tensor([2])],
            # Offsets from an unpadded prompt must not cause misplaced injection.
            huanyu_audio_bounds=[[], torch.tensor([[1, 3], [5, 6]])],
            image_bound=[torch.tensor([[2, 6]]), torch.empty(0, 2)],
            pixel_values=[pixels], image_grid_thw=[grid], use_cache=False,
        )
        torch.testing.assert_close(result[0], before[0])
        torch.testing.assert_close(result[1, 2:4], chunks[0])
        torch.testing.assert_close(result[1, 6:7], chunks[1])
        torch.testing.assert_close(result[1, [0, 1, 4, 5, 7, 8]], before[1, [0, 1, 4, 5, 7, 8]])
        torch.testing.assert_close(model.image_seen[0], pixels)
        self.assertEqual(model.kwargs_seen, {"use_cache": False})
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][0].shape), (2, 40, 16))

    def test_backward_preserves_text_embedding_gradients(self):
        model, _ = self.make_model([torch.ones(2, 4)])
        output = model(torch.tensor([[9, 2, 4, 4, 3]]),
                       huanyu_audio_features=[torch.ones(40, 16)], huanyu_audio_lengths=[16])
        output.sum().backward()
        torch.testing.assert_close(model.embedding.weight.grad[9], torch.ones(4))
        torch.testing.assert_close(model.embedding.weight.grad[4], torch.zeros(4))

    def test_aqa_transport_positions_are_restored_before_hf_forward(self):
        audio = torch.ones(2, 4, requires_grad=True)
        model, _ = self.make_model([audio])
        ids = torch.tensor([[9, 2, 4, 4, 3]])
        positions = torch.arange(5).unsqueeze(0)
        with patch.dict(os.environ, {"HUANYU_AQA_AUDIO_TRAIN": "1"}):
            output = model(
                ids, position_ids=positions,
                huanyu_audio_features=[torch.ones(40, 16)], huanyu_audio_lengths=[16],
            )
        restored = model.kwargs_seen["position_ids"]
        self.assertEqual(tuple(restored.shape), (4, 1, 5))
        torch.testing.assert_close(restored, positions.unsqueeze(0).expand(4, -1, -1))
        output.sum().backward()
        torch.testing.assert_close(audio.grad, torch.ones_like(audio))
        torch.testing.assert_close(model.embedding.weight.grad[9], torch.ones(4))

    def test_text_only_forward_never_encodes_audio(self):
        model, calls = self.make_model([])
        ids = torch.tensor([[9, 10]])
        expected = model.embedding(ids)
        actual = model(ids, huanyu_audio_features=[], huanyu_audio_lengths=[])
        torch.testing.assert_close(actual, expected)
        self.assertFalse(calls)

    def test_stem_transport_positions_are_restored_before_text_only_hf_forward(self):
        model, calls = self.make_model([])
        ids = torch.tensor([[9, 10, 11], [12, 13, 14]])
        positions = torch.arange(3).expand(2, -1) + 4096
        with patch.dict(os.environ, {"HUANYU_AQA_AUDIO_TRAIN": "0", "HUANYU_STEM_TEXT_TRAIN": "1"}):
            output = model(ids, position_ids=positions)
        torch.testing.assert_close(
            model.kwargs_seen["position_ids"], positions.unsqueeze(0).expand(4, -1, -1)
        )
        self.assertEqual(model.kwargs_seen["position_ids"].dtype, torch.long)
        self.assertFalse(calls)
        output.sum().backward()
        for token in ids.flatten():
            torch.testing.assert_close(model.embedding.weight.grad[token], torch.ones(4))

    def test_alignment_and_missing_features_fail_instead_of_zero_filling(self):
        model, _ = self.make_model([torch.ones(1, 4)])
        ids = torch.tensor([[2, 4, 4, 3]])
        with self.assertRaisesRegex(ValueError, "features are missing"):
            model(ids)
        with self.assertRaisesRegex(ValueError, "placeholder length"):
            model(ids, huanyu_audio_features=[torch.ones(40, 8)], huanyu_audio_lengths=[8])
        with self.assertRaisesRegex(ValueError, "tokens are missing"):
            model(torch.tensor([[9, 10]]), huanyu_audio_features=[torch.ones(40, 8)], huanyu_audio_lengths=[8])

    def test_exception_clears_embedding_context(self):
        model, _ = self.make_model([torch.ones(1, 4)])
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "original forward failed"):
            model(torch.tensor([[2, 4, 3]]), huanyu_audio_features=[torch.ones(40, 8)], huanyu_audio_lengths=[8])
        model.fail = False
        # A stale context would reject these ids or inject the previous audio.
        torch.testing.assert_close(model(torch.tensor([[10, 11]])), model.embedding(torch.tensor([[10, 11]])))

    def test_missing_embedding_call_is_not_silent_success(self):
        model, _ = self.make_model([torch.ones(1, 4)])
        model.skip_embedding = True
        with self.assertRaisesRegex(RuntimeError, "did not consume"):
            model(torch.tensor([[2, 4, 3]]), huanyu_audio_features=[torch.ones(40, 8)], huanyu_audio_lengths=[8])

    def test_legacy_cache_and_packed_inputs_are_rejected(self):
        model, _ = self.make_model([])
        with self.assertRaisesRegex(ValueError, "unverified embedding caches"):
            model(torch.tensor([[9]]), _precomputed_audio_embeds=[torch.ones(1, 4)])
        with self.assertRaisesRegex(ValueError, "use_remove_padding=false"):
            model(torch.tensor([9, 10]))

    def test_cross_sample_audio_counts_are_validated_before_encoding(self):
        model, calls = self.make_model([torch.ones(1, 4)])
        ids = torch.tensor([[2, 4, 3], [9, 10, 11]])
        kwargs = dict(huanyu_audio_features=[[], [torch.ones(40, 8)]],
                      huanyu_audio_lengths=[[], [8]])
        with self.assertRaisesRegex(ValueError, "per-sample audio counts"):
            model(ids, **kwargs, huanyu_audio_chunk_counts=[0, 1])
        with self.assertRaisesRegex(ValueError, "requires per-sample"):
            model(ids, **kwargs)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
