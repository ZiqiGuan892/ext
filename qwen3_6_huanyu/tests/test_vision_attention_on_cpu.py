"""Query-chunked visual attention preserves all keys and native eager semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_vision_attention_cpu_tests", ROOT / "vision_attention.py")
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


def reference_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Small HF-compatible eager fixture, including GQA and FP32 softmax."""
    groups = getattr(module, "num_key_value_groups", 1)
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


class VisionAttentionTests(unittest.TestCase):
    def setUp(self):
        self.vision = torch.nn.Module()
        self.vision.attn = torch.nn.Module()
        self.vision.attn.num_key_value_groups = 1
        self.vision.attn.is_causal = False
        self.vision.eval()
        self.calls = []

        def recorded_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            self.calls.append({"module": module, "query": query, "key": key, "value": value,
                               "mask": attention_mask, "scaling": scaling, "dropout": dropout,
                               "kwargs": kwargs})
            return reference_eager(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        self.original = recorded_eager
        self.hf = SimpleNamespace(eager_attention_forward=self.original)

    @staticmethod
    def tensors(dtype=torch.float32, query_heads=4, key_heads=4):
        generator = torch.Generator().manual_seed(17)
        return tuple(torch.randn(shape, generator=generator).to(dtype) for shape in
                     ((2, query_heads, 7, 4), (2, key_heads, 9, 4), (2, key_heads, 9, 4)))

    def assert_parity(self, query, key, value, mask=None):
        expected, _ = reference_eager(self.vision.attn, query, key, value, mask, 0.37)
        with torch.no_grad(), attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
            actual, weights = self.hf.eager_attention_forward(
                self.vision.attn, query, key, value, mask, scaling=0.37, dropout=0.0,
            )
        tolerance = 0.01 if query.dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        self.assertEqual(actual.dtype, query.dtype)
        self.assertEqual(tuple(actual.shape), (2, 7, 4, 4))
        self.assertIsNone(weights)
        self.assertIs(self.hf.eager_attention_forward, self.original)

    def test_float32_and_bfloat16_match_full_attention(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_parity(*self.tensors(dtype))

    def test_query_dependent_mask_is_sliced_consistently(self):
        query, key, value = self.tensors()
        mask = torch.zeros(2, 1, 7, 9)
        mask[:, :, :, -1] = float("-inf")
        mask[:, :, 3:6, 0] = -10.0
        self.assert_parity(query, key, value, mask)
        self.assertEqual([tuple(call["mask"].shape) for call in self.calls],
                         [(2, 1, 3, 9), (2, 1, 3, 9), (2, 1, 1, 9)])
        torch.testing.assert_close(self.calls[1]["mask"], mask[:, :, 3:6, :])

    def test_broadcast_query_mask_is_not_reduced_to_empty_slices(self):
        query, key, value = self.tensors()
        mask = torch.zeros(2, 1, 1, 9)
        mask[:, :, :, -2:] = float("-inf")
        self.assert_parity(query, key, value, mask)
        self.assertEqual(len(self.calls), 3)
        for call in self.calls:
            torch.testing.assert_close(call["mask"], mask)

    def test_grouped_query_attention_keeps_native_key_value_repetition(self):
        self.vision.attn.num_key_value_groups = 2
        self.assert_parity(*self.tensors(torch.bfloat16, query_heads=4, key_heads=2))
        self.assertTrue(all(call["key"].shape[1] == 2 for call in self.calls))

    def test_every_query_chunk_attends_to_all_original_keys_and_values(self):
        query, key, value = self.tensors()
        with torch.no_grad(), attention.chunked_vision_attention(
            self.hf, self.vision, query_chunk_size=3,
        ) as stats:
            self.hf.eager_attention_forward(
                self.vision.attn, query, key, value, None, scaling=0.37, dropout=0.0, audit_tag="unchanged",
            )
        self.assertEqual([call["query"].shape[2] for call in self.calls], [3, 3, 1])
        for call in self.calls:
            self.assertIs(call["key"], key)
            self.assertIs(call["value"], value)
            self.assertEqual(call["scaling"], 0.37)
            self.assertEqual(call["dropout"], 0.0)
            self.assertEqual(call["kwargs"], {"audit_tag": "unchanged"})
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["chunks"], 3)
        self.assertEqual(stats["max_query_tokens"], 7)
        self.assertEqual(stats["max_key_tokens"], 9)
        self.assertEqual(stats["max_chunk_tokens"], 3)

    def test_separate_frame_calls_never_share_keys_or_values(self):
        first = (torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 8, 4), torch.ones(1, 2, 8, 4))
        second = (torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 3, 4), torch.full((1, 2, 3, 4), 7.0))
        with torch.no_grad(), attention.chunked_vision_attention(
            self.hf, self.vision, query_chunk_size=3,
        ) as stats:
            first_output, _ = self.hf.eager_attention_forward(self.vision.attn, *first, None, scaling=0.5)
            second_output, _ = self.hf.eager_attention_forward(self.vision.attn, *second, None, scaling=0.5)
        torch.testing.assert_close(first_output, torch.ones(1, 5, 2, 4))
        torch.testing.assert_close(second_output, torch.full((1, 2, 2, 4), 7.0))
        self.assertEqual(len(self.calls), 3)
        for call in self.calls[:2]:
            self.assertIs(call["key"], first[1])
            self.assertIs(call["value"], first[2])
        self.assertIs(self.calls[2]["key"], second[1])
        self.assertIs(self.calls[2]["value"], second[2])
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["chunks"], 3)

    def test_foreign_module_is_untouched_even_with_grad_enabled(self):
        foreign = torch.nn.Module().eval()
        query, key, value = self.tensors()
        query.requires_grad_(True)
        expected, expected_weights = reference_eager(foreign, query, key, value, None, 0.5)
        with attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3) as stats:
            actual, weights = self.hf.eager_attention_forward(foreign, query, key, value, None, scaling=0.5)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(weights, expected_weights)
        self.assertTrue(actual.requires_grad)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0]["query"], query)
        self.assertEqual(stats["calls"], 0)

    def test_selected_module_rejects_grad_enabled_execution(self):
        with self.assertRaises((RuntimeError, ValueError)):
            with attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
                self.hf.eager_attention_forward(self.vision.attn, *self.tensors(), None, scaling=0.5)
        self.assertIs(self.hf.eager_attention_forward, self.original)

    def test_selected_module_rejects_training_mode(self):
        self.vision.train()
        with self.assertRaises((RuntimeError, ValueError)):
            with torch.no_grad(), attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
                self.hf.eager_attention_forward(self.vision.attn, *self.tensors(), None, scaling=0.5)
        self.assertIs(self.hf.eager_attention_forward, self.original)

    def test_selected_module_rejects_nonzero_dropout(self):
        with self.assertRaises((RuntimeError, ValueError)):
            with torch.no_grad(), attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
                self.hf.eager_attention_forward(
                    self.vision.attn, *self.tensors(), None, scaling=0.5, dropout=0.1,
                )
        self.assertIs(self.hf.eager_attention_forward, self.original)

    def test_selected_module_rejects_causal_attention(self):
        for module_causal, kwargs in ((True, {}), (False, {"is_causal": True})):
            with self.subTest(module_causal=module_causal, kwargs=kwargs):
                self.vision.attn.is_causal = module_causal
                with self.assertRaises((RuntimeError, ValueError)):
                    with torch.no_grad(), attention.chunked_vision_attention(
                        self.hf, self.vision, query_chunk_size=3,
                    ):
                        self.hf.eager_attention_forward(
                            self.vision.attn, *self.tensors(), None, scaling=0.5, **kwargs,
                        )
                self.assertIs(self.hf.eager_attention_forward, self.original)

    def test_original_attention_failure_restores_the_hf_function(self):
        def failing(*args, **kwargs):
            raise RuntimeError("fixture kernel failure")
        self.hf.eager_attention_forward = failing
        with self.assertRaisesRegex(RuntimeError, "fixture kernel failure"):
            with torch.no_grad(), attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
                self.hf.eager_attention_forward(self.vision.attn, *self.tensors(), None, scaling=0.5)
        self.assertIs(self.hf.eager_attention_forward, failing)

    def test_wrong_output_layout_is_rejected_and_restored(self):
        def wrong_layout(module, query, key, value, attention_mask, scaling, **kwargs):
            # Native contract is [B,Q,H,D], not [B,H,Q,D].
            return torch.zeros_like(query), None
        self.hf.eager_attention_forward = wrong_layout
        with self.assertRaises((RuntimeError, ValueError)):
            with torch.no_grad(), attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=3):
                self.hf.eager_attention_forward(self.vision.attn, *self.tensors(), None, scaling=0.5)
        self.assertIs(self.hf.eager_attention_forward, wrong_layout)

    def test_invalid_query_chunk_size_is_rejected_without_leaking_patch(self):
        for chunk_size in (0, -1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaises((RuntimeError, ValueError)):
                    with attention.chunked_vision_attention(self.hf, self.vision, query_chunk_size=chunk_size):
                        pass
                self.assertIs(self.hf.eager_attention_forward, self.original)


if __name__ == "__main__":
    unittest.main()
