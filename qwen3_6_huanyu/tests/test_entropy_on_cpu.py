"""Numerical and integration checks for bounded Huanyu entropy computation."""

from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_entropy_cpu_tests", ROOT / "entropy.py")
entropy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entropy
spec.loader.exec_module(entropy)


def reference_entropy(logits):
    work = logits if logits.dtype == torch.float64 else logits.float()
    log_probability = torch.log_softmax(work, dim=-1)
    return -(log_probability.exp() * log_probability).sum(dim=-1)


@contextmanager
def fake_verl(original=reference_entropy):
    verl = ModuleType("verl")
    verl.__path__ = []
    utils = ModuleType("verl.utils")
    utils.__path__ = []
    functional = ModuleType("verl.utils.torch_functional")
    functional.entropy_from_logits = original
    verl.utils = utils
    utils.torch_functional = functional
    with patch.dict(sys.modules, {
        "verl": verl,
        "verl.utils": utils,
        "verl.utils.torch_functional": functional,
    }):
        yield functional


class EntropyNumericsTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(2026)

    def random(self, shape, dtype=torch.float64):
        return torch.randn(shape, generator=self.generator, dtype=dtype)

    def test_values_and_weighted_gradients_match_native_for_2d_and_3d(self):
        for shape in ((7, 11), (2, 5, 11)):
            for chunk_size in (1, 3, 100):
                with self.subTest(shape=shape, chunk_size=chunk_size):
                    logits = self.random(shape).requires_grad_()
                    reference = logits.detach().clone().requires_grad_()
                    result = entropy.chunked_entropy_from_logits(logits, chunk_size)
                    expected = reference_entropy(reference)
                    cotangent = self.random(shape[:-1])
                    (result * cotangent).sum().backward()
                    (expected * cotangent).sum().backward()
                    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(logits, reference)
                    self.assertEqual(result.shape, shape[:-1])

    def test_half_and_bfloat_inputs_promote_entropy_to_float32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                logits = self.random((2, 5, 13), dtype).requires_grad_()
                reference = logits.detach().clone().requires_grad_()
                result = entropy.chunked_entropy_from_logits(logits, 3)
                expected = reference_entropy(reference)
                cotangent = self.random((2, 5), torch.float32)
                (result * cotangent).sum().backward()
                (expected * cotangent).sum().backward()
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(logits.grad.dtype, dtype)
                torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)
                tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
                torch.testing.assert_close(logits.grad, reference.grad, rtol=tolerance, atol=2e-5)

    def test_noncontiguous_token_and_vocabulary_layouts(self):
        examples = (
            self.random((13, 7)).transpose(0, 1),
            self.random((3, 2, 13)).transpose(0, 1),
            self.random((2, 5, 26))[..., ::2],
        )
        for logits in examples:
            with self.subTest(shape=tuple(logits.shape), stride=logits.stride()):
                self.assertFalse(logits.is_contiguous())
                logits.requires_grad_()
                reference = logits.detach().clone().requires_grad_()
                result = entropy.chunked_entropy_from_logits(logits, 3)
                expected = reference_entropy(reference)
                cotangent = self.random(logits.shape[:-1])
                (result * cotangent).sum().backward()
                (expected * cotangent).sum().backward()
                torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-11, atol=1e-12)

    def test_double_precision_gradcheck(self):
        logits = self.random((2, 3, 5)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: entropy.chunked_entropy_from_logits(value, 2),
            (logits,), eps=1e-6, atol=1e-5, rtol=1e-4,
        ))

    def test_uniform_and_large_shifted_logits_are_stable(self):
        uniform = torch.full((2, 3, 17), 1e8, dtype=torch.float32)
        actual = entropy.chunked_entropy_from_logits(uniform, 2)
        torch.testing.assert_close(actual, torch.full((2, 3), math.log(17)), rtol=1e-6, atol=1e-6)
        logits = self.random((5, 17))
        expected = entropy.chunked_entropy_from_logits(logits, 2)
        shifted = entropy.chunked_entropy_from_logits(logits + 1e8, 2)
        torch.testing.assert_close(shifted, expected, rtol=1e-8, atol=1e-8)

    def test_single_class_has_zero_entropy_and_gradient(self):
        logits = self.random((2, 4, 1)).requires_grad_()
        output = entropy.chunked_entropy_from_logits(logits, 3)
        output.sum().backward()
        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_no_grad_and_inference_mode(self):
        logits = self.random((2, 5, 11)).requires_grad_()
        for context in (torch.no_grad, torch.inference_mode):
            with self.subTest(context=context.__name__), context():
                output = entropy.chunked_entropy_from_logits(logits, 3)
                self.assertFalse(output.requires_grad)
                torch.testing.assert_close(output, reference_entropy(logits))

    def test_zero_tokens_preserve_shape_dtype_and_gradient(self):
        for shape in ((0, 7), (2, 0, 7), (0, 3, 7)):
            with self.subTest(shape=shape):
                logits = torch.empty(shape, dtype=torch.float32, requires_grad=True)
                output = entropy.chunked_entropy_from_logits(logits, 3)
                self.assertEqual(output.shape, shape[:-1])
                self.assertEqual(output.dtype, torch.float32)
                output.sum().backward()
                self.assertEqual(logits.grad.shape, shape)
                self.assertEqual(logits.grad.numel(), 0)

    def test_every_forward_probability_block_is_bounded(self):
        logits = self.random((2, 7, 13)).transpose(0, 1)
        block_shapes = []
        original = entropy._entropy_block

        def observe(block):
            block_shapes.append(tuple(block.shape))
            self.assertLessEqual(block.numel() // block.shape[-1], 3)
            self.assertEqual(block.shape[-1], 13)
            return original(block)

        with torch.no_grad(), patch.object(entropy, "_entropy_block", side_effect=observe):
            actual = entropy.chunked_entropy_from_logits(logits, 3)
        self.assertGreater(len(block_shapes), 1)
        self.assertEqual(sum(math.prod(shape[:-1]) for shape in block_shapes), 14)
        torch.testing.assert_close(actual, reference_entropy(logits))

    def test_default_chunk_bounds_forward_and_backward_without_copying_response_slice(self):
        logits = self.random((2, 525, 19), torch.float32)[:, 5:522, :].requires_grad_()
        self.assertFalse(logits.is_contiguous())
        reference = logits.detach().clone().requires_grad_()
        shapes = []
        original = entropy._probability_terms

        def observe(block):
            shapes.append(tuple(block.shape))
            self.assertEqual(block.untyped_storage().data_ptr(), logits.untyped_storage().data_ptr())
            return original(block)

        with patch.object(entropy, "_probability_terms", side_effect=observe):
            actual = entropy.chunked_entropy_from_logits(logits)
            forward_shapes = list(shapes)
            actual.sum().backward()
        expected = reference_entropy(reference)
        expected.sum().backward()
        self.assertEqual(forward_shapes, [(512, 19), (5, 19)] * 2)
        self.assertEqual(shapes, forward_shapes * 2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(logits, reference)

    def test_higher_order_gradient_is_explicitly_unsupported(self):
        logits = self.random((3, 7)).requires_grad_()
        output = entropy.chunked_entropy_from_logits(logits, 2)
        first, = torch.autograd.grad(output.sum(), logits, create_graph=True)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first.sum(), logits)

    def test_autograd_does_not_retain_full_probability_tensor(self):
        logits = self.random((2, 7, 13)).requires_grad_()
        retained = []

        def record(tensor):
            retained.append((tensor.numel(), tensor.untyped_storage().data_ptr()))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
            output = entropy.chunked_entropy_from_logits(logits, 3)
        self.assertTrue(retained)
        for count, storage in retained:
            if count > output.numel():
                self.assertEqual(storage, logits.untyped_storage().data_ptr())
        output.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_input_shapes_and_dtypes_are_rejected(self):
        for logits in (torch.tensor(1.), torch.ones(5), torch.ones(1, 2, 3, 4),
                       torch.ones(2, 0), torch.ones(2, 3, 0),
                       torch.ones(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.complex64)):
            with self.subTest(shape=tuple(logits.shape), dtype=logits.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    entropy.chunked_entropy_from_logits(logits, 3)

    def test_invalid_chunk_sizes_are_rejected(self):
        for chunk_size in (0, -1, 1.5, True, False, "3", None):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaises((TypeError, ValueError)):
                    entropy.chunked_entropy_from_logits(torch.ones(2, 3), chunk_size)


class EntropyInstallationTests(unittest.TestCase):
    def test_installation_replaces_target_and_reports_first_real_call_once(self):
        output = io.StringIO()
        with fake_verl() as functional, redirect_stdout(output):
            installed = entropy.install_huanyu_entropy_chunking(3)
            self.assertIs(functional.entropy_from_logits, installed)
            self.assertTrue(installed._huanyu_entropy_chunked)
            self.assertEqual(installed._huanyu_entropy_chunk_size, 3)
            install_text = output.getvalue()
            self.assertTrue(install_text.strip())
            logits = torch.zeros(2, 5, 7)
            actual = functional.entropy_from_logits(logits)
            first_call_text = output.getvalue()
            self.assertNotEqual(first_call_text, install_text)
            functional.entropy_from_logits(logits)
            self.assertEqual(output.getvalue(), first_call_text)
        torch.testing.assert_close(actual, reference_entropy(logits))

    def test_repeated_installation_is_idempotent_but_reconfiguration_fails(self):
        with fake_verl() as functional, redirect_stdout(io.StringIO()) as output:
            first = entropy.install_huanyu_entropy_chunking(3)
            install_text = output.getvalue()
            self.assertIs(entropy.install_huanyu_entropy_chunking(3), first)
            self.assertEqual(output.getvalue(), install_text)
            with self.assertRaises(RuntimeError):
                entropy.install_huanyu_entropy_chunking(4)
            self.assertIs(functional.entropy_from_logits, first)

    def test_invalid_installation_chunk_size_does_not_mutate_target(self):
        for chunk_size in (0, -1, 1.5, True, "3", None):
            with self.subTest(chunk_size=chunk_size), fake_verl() as functional:
                with self.assertRaises((TypeError, ValueError)):
                    entropy.install_huanyu_entropy_chunking(chunk_size)
                self.assertIs(functional.entropy_from_logits, reference_entropy)

    def test_missing_or_noncallable_target_fails_closed(self):
        for missing in (False, True):
            with self.subTest(missing=missing), fake_verl(None) as functional:
                if missing:
                    del functional.entropy_from_logits
                with self.assertRaises(RuntimeError):
                    entropy.install_huanyu_entropy_chunking(3)


if __name__ == "__main__":
    unittest.main()
