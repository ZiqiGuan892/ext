"""Exercise frozen media injection, batch alignment and decoder recomputation."""

import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint_module


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_cached_forward_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
forward = importlib.import_module(f"{PACKAGE}.forward")
# Exercise real eager checkpoint/autograd without importing the optional Dynamo
# compiler stack (the lightweight local Torch installation lacks SymPy's mpmath).
# Only compiler-disable decorators are unwrapped; saved-tensor hooks, backward
# and the decoder recomputation still execute the original PyTorch code.
checkpoint = inspect.unwrap(checkpoint_module.checkpoint)


class RecordingHFModel(nn.Module):
    def __init__(self, recompute=False):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.decoder = nn.Linear(4, 4, bias=False)
        self.recompute = recompute
        self.forward_calls = 0
        self.decoder_calls = 0
        self.embedding_calls = 0
        self.seen = None
        self.embedding.register_forward_hook(self.count_embeddings)

    def count_embeddings(self, module, args, output):
        self.embedding_calls += 1

    def get_input_embeddings(self):
        return self.embedding

    def decode(self, value):
        self.decoder_calls += 1
        return self.decoder(value).sin()

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        self.seen = (input_ids, kwargs)
        if kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None:
            raise AssertionError("The cached path attempted native visual encoding")
        embedded = self.embedding(input_ids)
        return checkpoint(self.decode, embedded, use_reentrant=False) if self.recompute else embedded


def mixed_inputs():
    ids = torch.tensor([
        [0, 0, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9],
        [0, 9, 5, 6, 6, 7, 2, 4, 4, 3, 10, 11],
        [0, 0, 0, 0, 0, 0, 9, 2, 4, 3, 10, 11],
    ])
    kwargs = {
        "huanyu_cache_enabled": [torch.tensor([1])] * 3,
        "huanyu_cached_audio_embeds": [[], [torch.full((2, 4), 7.0, dtype=torch.bfloat16)],
                                       [torch.full((1, 4), 8.0, dtype=torch.bfloat16)]],
        "huanyu_cached_vision_embeds": [[], [torch.full((2, 4), 9.0, dtype=torch.bfloat16)], []],
        "huanyu_audio_lengths": [[], torch.tensor([16]), torch.tensor([8])],
        "huanyu_audio_chunk_counts": [0, 1, 1],
        "huanyu_cached_image_counts": [0, 1, 0],
        "image_grid_thw": [[], torch.tensor([[1, 2, 4]]), []],
        "position_ids": torch.arange(108).reshape(3, 3, 12) + 2038,
        "attention_mask": ids.ne(0).long(),
        "use_cache": False,
    }
    return ids, kwargs


class CachedForwardTests(unittest.TestCase):
    def make_model(self, *, recompute=False, use_precompute=True):
        torch.manual_seed(12)
        model = RecordingHFModel(recompute=recompute)

        def forbidden_encoder(*args):
            raise AssertionError("Cached forward must never call the online audio encoder")

        forward.install_huanyu_forward(
            model, start_token_id=2, end_token_id=3, placeholder_token_id=4,
            image_token_id=6, spatial_merge_size=2, use_precompute=use_precompute,
            encode_audio=forbidden_encoder,
        )
        return model

    def test_mixed_padded_batch_matches_manual_scatter_and_keeps_positions(self):
        model = self.make_model()
        ids, kwargs = mixed_inputs()
        expected = torch.nn.functional.embedding(ids, model.embedding.weight).detach().clone()
        expected[1, 3:5] = 9
        expected[1, 7:9] = 7
        expected[2, 8:9] = 8
        result = model(ids, **kwargs)
        torch.testing.assert_close(result, expected)
        self.assertIs(model.seen[0], ids)
        self.assertIs(model.seen[1]["position_ids"], kwargs["position_ids"])
        self.assertIs(model.seen[1]["attention_mask"], kwargs["attention_mask"])
        torch.testing.assert_close(model.seen[1]["image_grid_thw"], torch.tensor([[1, 2, 4]]))
        self.assertEqual(set(model.seen[1]), {"position_ids", "image_grid_thw", "attention_mask", "use_cache"})
        self.assertEqual(model.embedding_calls, 1)

    def test_checkpoint_backward_recomputes_decoder_without_losing_injected_media(self):
        model = self.make_model(recompute=True)
        ids, kwargs = mixed_inputs()
        recompute_guard = getattr(checkpoint_module, "_run_fn_with_dynamo_disabled", None)
        guard_context = (patch.object(checkpoint_module, "_run_fn_with_dynamo_disabled",
                                      inspect.unwrap(recompute_guard))
                         if recompute_guard is not None else nullcontext())
        with guard_context:
            output = model(ids, **kwargs)
            output.square().sum().backward()
        self.assertGreaterEqual(model.decoder_calls, 2)
        self.assertEqual(model.embedding_calls, 1)
        self.assertGreater(model.embedding.weight.grad[9].abs().sum().item(), 0)
        torch.testing.assert_close(model.embedding.weight.grad[4], torch.zeros(4))
        torch.testing.assert_close(model.embedding.weight.grad[6], torch.zeros(4))
        self.assertTrue(torch.isfinite(model.decoder.weight.grad).all())
        for sample in kwargs["huanyu_cached_audio_embeds"] + kwargs["huanyu_cached_vision_embeds"]:
            for tensor in sample:
                self.assertIsNone(tensor.grad)
        # Backward must not leave an injection context on the embedding module.
        pure_ids = torch.tensor([[9, 10]])
        torch.testing.assert_close(model.embedding(pure_ids),
                                   torch.nn.functional.embedding(pure_ids, model.embedding.weight))

    def test_cross_sample_count_mismatch_fails_even_when_global_total_matches(self):
        for key, counts in (("huanyu_cached_image_counts", [1, 0, 0]),
                            ("huanyu_audio_chunk_counts", [1, 0, 1])):
            with self.subTest(key=key):
                model = self.make_model()
                ids, kwargs = mixed_inputs()
                kwargs[key] = counts
                with self.assertRaises(ValueError):
                    model(ids, **kwargs)
                self.assertEqual(model.forward_calls, 0)

    def test_invalid_cached_tensor_is_rejected_before_hf_forward(self):
        invalid = {
            "dtype": torch.ones(2, 4, dtype=torch.float32),
            "width": torch.ones(2, 3, dtype=torch.bfloat16),
            "nonfinite": torch.full((2, 4), float("nan"), dtype=torch.bfloat16),
            "trainable": torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True),
            "tokens": torch.ones(1, 4, dtype=torch.bfloat16),
        }
        for name, tensor in invalid.items():
            with self.subTest(name=name):
                model = self.make_model()
                ids, kwargs = mixed_inputs()
                kwargs["huanyu_cached_audio_embeds"][1] = [tensor]
                with self.assertRaises(ValueError):
                    model(ids, **kwargs)
                self.assertEqual(model.forward_calls, 0)

    def test_image_span_grid_and_cache_lengths_must_all_match(self):
        model = self.make_model()
        ids, kwargs = mixed_inputs()
        kwargs["image_grid_thw"] = torch.tensor([[1, 4, 4]])
        with self.assertRaisesRegex(ValueError, "span/grid/embedding"):
            model(ids, **kwargs)
        self.assertEqual(model.forward_calls, 0)

    def test_cache_markers_are_required_for_every_sample(self):
        for marker in (None, [1, 0, 1], [1]):
            with self.subTest(marker=marker):
                model = self.make_model()
                ids, kwargs = mixed_inputs()
                kwargs["huanyu_cache_enabled"] = marker
                with self.assertRaisesRegex(ValueError, "every sample"):
                    model(ids, **kwargs)
                self.assertEqual(model.forward_calls, 0)

    def test_raw_media_in_cached_mode_is_never_an_online_fallback(self):
        for key, value in (("pixel_values", torch.ones(8, 24)),
                           ("pixel_values_videos", torch.ones(8, 24)),
                           ("huanyu_audio_features", [torch.ones(40, 16)])):
            with self.subTest(key=key):
                model = self.make_model()
                ids, kwargs = mixed_inputs()
                kwargs[key] = value
                with self.assertRaisesRegex(ValueError, "raw encoder inputs"):
                    model(ids, **kwargs)
                self.assertEqual(model.forward_calls, 0)

    def test_image_cache_requires_multimodal_positions(self):
        model = self.make_model()
        ids, kwargs = mixed_inputs()
        kwargs.pop("position_ids")
        with self.assertRaisesRegex(ValueError, "multimodal position_ids"):
            model(ids, **kwargs)
        self.assertEqual(model.forward_calls, 0)

    def test_cached_inputs_in_online_mode_fail_explicitly(self):
        model = self.make_model(use_precompute=False)
        ids, kwargs = mixed_inputs()
        with self.assertRaisesRegex(ValueError, "HUANYU_USE_PRECOMPUTE=1"):
            model(ids, **kwargs)

    def test_text_only_sample_still_requires_and_accepts_cache_contract(self):
        model = self.make_model()
        ids = torch.tensor([[9, 10]])
        expected = torch.nn.functional.embedding(ids, model.embedding.weight)
        actual = model(ids, huanyu_cache_enabled=[1], huanyu_cached_audio_embeds=[],
                       huanyu_cached_vision_embeds=[], huanyu_audio_lengths=[],
                       huanyu_audio_chunk_counts=[0], huanyu_cached_image_counts=[0])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
