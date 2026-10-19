"""CPU contracts for Huanyu prompt/chunk alignment, without importing ext."""

import importlib.util
from pathlib import Path
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / "audio_alignment.py"
_SPEC = importlib.util.spec_from_file_location("huanyu_audio_alignment_tested", _PATH)
alignment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(alignment)


def find(ids, lengths):
    return alignment.find_audio_bounds(
        ids, lengths, start_token_id=10, end_token_id=11, placeholder_token_id=12
    )


class AudioBoundsTest(unittest.TestCase):
    def test_two_chunks_with_text_and_boundary_tokens(self):
        ids = torch.tensor([7, 10, 12, 12, 11, 8, 10, 12, 11, 12])
        self.assertEqual(find(ids, [2, 1]), [(2, 4), (7, 8)])
        self.assertEqual(find([7, 12, 8], []), [])

    def test_rejects_missing_extra_and_wrong_length_chunks(self):
        for ids, lengths in (
            ([10, 12, 11], []),
            ([7, 8], [1]),
            ([10, 12, 11], [2]),
            ([10, 12, 12, 11], [1]),
            ([10, 12, 11], [0]),
        ):
            with self.subTest(ids=ids, lengths=lengths), self.assertRaises(ValueError):
                find(ids, lengths)

    def test_rejects_bad_boundaries_and_mixed_content(self):
        for ids in ([10, 12], [11], [10, 10, 12, 11, 11], [10, 11], [10, 99, 11]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                find(ids, [1])

    def test_rejects_ambiguous_ids_and_noninteger_prompt(self):
        with self.assertRaises(ValueError):
            alignment.find_audio_bounds([], [], start_token_id=10, end_token_id=10, placeholder_token_id=12)
        for ids in ([10.0, 12.0, 11.0], [True], torch.tensor([[10, 12, 11]])):
            with self.subTest(ids=ids), self.assertRaises(TypeError):
                find(ids, [1])


class AudioEmbeddingTest(unittest.TestCase):
    def test_values_and_autograd_leave_inputs_and_boundaries_unchanged(self):
        text = torch.arange(20, dtype=torch.float32).reshape(10, 2).clone().requires_grad_()
        chunk_a = torch.full((2, 2), 100.0, requires_grad=True)
        chunk_b = torch.full((1, 2), 200.0, requires_grad=True)
        original = text.detach().clone()
        bounds = find([7, 10, 12, 12, 11, 8, 10, 12, 11, 9], [2, 1])
        output = alignment.insert_audio_embeddings(text, [chunk_a, chunk_b], bounds)
        expected = original.clone()
        expected[2:4] = 100.0
        expected[7:8] = 200.0
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(text.detach(), original)
        output.sum().backward()
        expected_text_grad = torch.ones_like(text)
        expected_text_grad[2:4] = 0
        expected_text_grad[7:8] = 0
        torch.testing.assert_close(text.grad, expected_text_grad)
        torch.testing.assert_close(chunk_a.grad, torch.ones_like(chunk_a))
        torch.testing.assert_close(chunk_b.grad, torch.ones_like(chunk_b))

    def test_rejects_invalid_overlapping_and_unsorted_bounds(self):
        text = torch.zeros(6, 2)
        cases = [([-1, 1],), ([2, 2],), ([1, 7],), ([0, 2], [1, 3]), ([3, 5], [0, 2])]
        for bounds in cases:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                alignment.insert_audio_embeddings(text, [torch.zeros(2, 2)] * len(bounds), bounds)

    def test_rejects_feature_length_hidden_dimension_dtype_and_count(self):
        text = torch.zeros(6, 2)
        for chunks in ([torch.zeros(1, 2)], [torch.zeros(2, 3)], [torch.zeros(2, 2, dtype=torch.float64)], []):
            with self.subTest(shapes=[x.shape for x in chunks]), self.assertRaises(ValueError):
                alignment.insert_audio_embeddings(text, chunks, [(1, 3)])
        torch.testing.assert_close(text, torch.zeros_like(text))

    def test_empty_audio_keeps_text_gradient(self):
        text = torch.ones(3, 2, requires_grad=True)
        output = alignment.insert_audio_embeddings(text, [], [])
        torch.testing.assert_close(output, text)
        self.assertIsNot(output, text)
        output.sum().backward()
        torch.testing.assert_close(text.grad, torch.ones_like(text))


if __name__ == "__main__":
    unittest.main()
