"""Numerical, memory-boundary and dispatcher checks without an NPU backend."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_log_probs_tests", ROOT / "log_probs.py")
log_probs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_probs)


def reference(logits, labels):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    return -loss.reshape(logits.shape[:-1])


def random_logits(shape, dtype=torch.float64):
    return torch.randn(shape, generator=torch.Generator().manual_seed(42), dtype=dtype)


@pytest.mark.parametrize("shape", [(7, 11), (2, 5, 11), (2, 2, 3, 11)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_values_and_weighted_first_gradients(shape, dtype):
    logits = random_logits(shape, dtype).requires_grad_()
    baseline = logits.detach().clone().requires_grad_()
    labels = torch.arange(logits.numel() // shape[-1]).reshape(shape[:-1]) % shape[-1]
    actual = log_probs.chunked_logprobs_from_logits(logits, labels, reference, chunk_size=3)
    expected = reference(baseline, labels)
    weights = random_logits(shape[:-1], dtype)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, baseline.grad)
    torch.testing.assert_close(logits, baseline)


def test_autocast_is_preserved_when_backward_runs_outside_context():
    logits = random_logits((2, 7, 13), torch.bfloat16).requires_grad_()
    baseline = logits.detach().clone().requires_grad_()
    labels = torch.zeros(2, 7, dtype=torch.long)
    observed = []

    def backend(block, targets):
        observed.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        return reference(block, targets)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = log_probs.chunked_logprobs_from_logits(logits, labels, backend, chunk_size=3)
        expected = reference(baseline, labels)
    assert actual.dtype == expected.dtype == torch.float32
    actual.sum().backward()
    expected.sum().backward()
    assert observed == [(True, torch.bfloat16)] * 12
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, baseline.grad)


@pytest.mark.parametrize("layout", ["response_slice", "vocab_stride", "transposed"])
def test_noncontiguous_logits_do_not_get_flattened_into_a_full_copy(layout):
    if layout == "response_slice":
        logits = random_logits((2, 7, 11))[:, 1:6]
    elif layout == "vocab_stride":
        logits = random_logits((2, 5, 22))[..., ::2]
    else:
        logits = random_logits((5, 2, 11)).transpose(0, 1)
    assert not logits.is_contiguous()
    logits.requires_grad_()
    baseline = logits.detach().clone().requires_grad_()
    labels = torch.zeros(logits.shape[:-1], dtype=torch.long)
    records = []

    def backend(block, targets):
        assert block.ndim == 2 and block.shape[0] <= 3 and block.shape[1] == 11
        assert block.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()
        records.append(block.shape[0])
        return reference(block, targets)

    actual = log_probs.chunked_logprobs_from_logits(logits, labels.flatten(), backend, chunk_size=3)
    expected = reference(baseline, labels)
    actual.sum().backward()
    expected.sum().backward()
    assert sum(records) == labels.numel() * 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, baseline.grad)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_old_log_prob_modes(mode):
    logits = random_logits((2, 7, 11)).requires_grad_()
    labels = torch.zeros(2, 7, dtype=torch.long)
    with mode():
        actual = log_probs.chunked_logprobs_from_logits(logits, labels, reference, chunk_size=3)
        assert not actual.requires_grad
        torch.testing.assert_close(actual, reference(logits, labels))


def test_default_chunk_bounds_forward_backward_and_saved_storage():
    logits = random_logits((2, 527, 19))[:, 2:525].requires_grad_()
    labels = torch.zeros(2, 523, dtype=torch.long)
    calls, saved = [], []

    def backend(block, targets):
        calls.append(tuple(block.shape))
        assert block.shape[0] <= 512
        return reference(block, targets)

    def pack(tensor):
        saved.append(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        actual = log_probs.chunked_logprobs_from_logits(logits, labels, backend)
    # Forward retains original logits and small labels, never a complete CE
    # probability/gradient workspace nor all of the per-chunk autograd graphs.
    assert set(saved) <= {logits.untyped_storage().data_ptr(), labels.untyped_storage().data_ptr()}
    assert calls == [(512, 19), (11, 19)] * 2
    actual.sum().backward()
    assert calls == [(512, 19), (11, 19)] * 4


def test_gradcheck_and_shared_logits_with_entropy():
    logits = random_logits((2, 3, 5)).requires_grad_()
    labels = torch.tensor([[0, 1, 2], [3, 4, 0]])
    assert torch.autograd.gradcheck(
        lambda value: log_probs.chunked_logprobs_from_logits(value, labels, reference, 2), (logits,)
    )
    baseline = logits.detach().clone().requires_grad_()

    def objective(value, use_chunked):
        selected = log_probs.chunked_logprobs_from_logits(value, labels, reference, 2) if use_chunked else reference(value, labels)
        logp = value.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return selected.sum() + entropy.sum() * 0.03

    objective(logits, True).backward()
    objective(baseline, False).backward()
    torch.testing.assert_close(logits.grad, baseline.grad)


def test_native_ignore_label_and_single_class_semantics():
    logits = random_logits((5, 1)).requires_grad_()
    labels = torch.tensor([0, -100, 0, -100, 0])
    output = log_probs.chunked_logprobs_from_logits(logits, labels, reference, 2)
    output.sum().backward()
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_higher_order_gradients_are_explicitly_unsupported():
    logits = random_logits((7, 11)).requires_grad_()
    labels = torch.zeros(7, dtype=torch.long)
    output = log_probs.chunked_logprobs_from_logits(logits, labels, reference, 3)
    first, = torch.autograd.grad(output.sum(), logits, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(first.sum(), logits)


@pytest.mark.parametrize("shape", [(0, 5), (2, 0, 5), (0, 3, 5)])
def test_empty_delegates_to_backend(shape):
    logits = torch.empty(shape, dtype=torch.float32, requires_grad=True)
    labels = torch.empty(shape[:-1], dtype=torch.long)
    result = log_probs.chunked_logprobs_from_logits(logits, labels, reference)
    assert result.shape == labels.shape
    result.sum().backward()
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("fail_in_backward", [False, True])
def test_backend_errors_remain_visible(fail_in_backward):
    error = torch.OutOfMemoryError("native allocation failed")
    logits = random_logits((7, 11)).requires_grad_()
    labels = torch.zeros(7, dtype=torch.long)

    def backend(block, targets):
        if torch.is_grad_enabled() == fail_in_backward:
            raise error
        return reference(block, targets)

    with pytest.raises(torch.OutOfMemoryError) as caught:
        log_probs.chunked_logprobs_from_logits(logits, labels, backend, 3).sum().backward()
    assert caught.value is error


def test_invalid_inputs_are_rejected():
    for chunk_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="chunk_size"):
            log_probs.chunked_logprobs_from_logits(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), reference, chunk_size)
    for logits in (torch.zeros(3), torch.zeros(2, 0), torch.zeros(2, 3, dtype=torch.long)):
        with pytest.raises((ValueError, TypeError)):
            log_probs.chunked_logprobs_from_logits(logits, torch.zeros(2, dtype=torch.long), reference)
    with pytest.raises(ValueError, match="one label"):
        log_probs.chunked_logprobs_from_logits(torch.zeros(2, 3), torch.zeros(3, dtype=torch.long), reference)


@pytest.fixture
def native_dispatcher(monkeypatch):
    functional = ModuleType("verl.utils.torch_functional")
    functional.__dict__.update(torch=torch, reference=reference)
    # The native helper and global dispatch contract from the server traceback
    # (torch_functional.py:104,151). The device kernel alone is replaced on CPU.
    exec('''
NPU_CROSS_ENTROPY_LOSS_AVAILABLE = True
def logprobs_from_logits_torch_npu(logits, labels):
    batch_dim = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits, labels.reshape(-1), reduction="none")
    return -loss.view(*batch_dim)
def logprobs_from_logits(logits, labels, inplace_backward=True):
    if NPU_CROSS_ENTROPY_LOSS_AVAILABLE:
        return logprobs_from_logits_torch_npu(logits, labels)
    return reference(logits, labels)
''', functional.__dict__)
    shapes = []

    def native_ce(logits, labels, reduction):
        assert reduction == "none"
        shapes.append(tuple(logits.shape))
        return F.cross_entropy(logits, labels, reduction=reduction), None, None, None

    functional.torch_npu = SimpleNamespace(npu_cross_entropy_loss=native_ce)
    verl, utils = ModuleType("verl"), ModuleType("verl.utils")
    verl.__path__, utils.__path__ = [], []
    verl.utils, utils.torch_functional = utils, functional
    for module in (verl, utils, functional):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return functional, shapes


def test_engine_import_alias_reaches_chunked_native_helper(native_dispatcher, capsys):
    functional, shapes = native_dispatcher
    engine_alias = functional.logprobs_from_logits  # Engine imported this BEFORE installation.
    installed = log_probs.install_huanyu_logprob_chunking(3)
    assert log_probs.install_huanyu_logprob_chunking(3) is installed
    logits = random_logits((7, 11)).requires_grad_()
    labels = torch.zeros(7, dtype=torch.long)
    with torch.no_grad():
        output = engine_alias(logits=logits, labels=labels)
        torch.testing.assert_close(output, reference(logits, labels))
    engine_alias(logits, labels, inplace_backward=False).sum().backward()
    assert shapes == [(3, 11), (3, 11), (1, 11)] * 3
    assert capsys.readouterr().out.count("[huanyu-logprobs] active") == 2
    with pytest.raises(RuntimeError, match="already fixed"):
        log_probs.install_huanyu_logprob_chunking(4)


def test_non_npu_dispatch_is_unchanged(native_dispatcher):
    functional, shapes = native_dispatcher
    functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE = False
    original = functional.logprobs_from_logits
    log_probs.install_huanyu_logprob_chunking(3)
    assert original is functional.logprobs_from_logits
    logits = random_logits((7, 11))
    labels = torch.zeros(7, dtype=torch.long)
    torch.testing.assert_close(original(logits, labels), reference(logits, labels))
    assert shapes == []


def test_unknown_dispatcher_cannot_silently_claim_installation(native_dispatcher):
    functional, _ = native_dispatcher
    original = functional.logprobs_from_logits_torch_npu
    functional.logprobs_from_logits = lambda logits, labels: original(logits, labels)
    with pytest.raises(RuntimeError, match="resolves the NPU helper"):
        log_probs.install_huanyu_logprob_chunking()
    assert functional.logprobs_from_logits_torch_npu is original


def test_missing_or_changed_backend_signature_is_rejected(native_dispatcher):
    functional, _ = native_dispatcher
    functional.logprobs_from_logits_torch_npu = None
    with pytest.raises(RuntimeError, match="not callable"):
        log_probs.install_huanyu_logprob_chunking()
    functional.logprobs_from_logits_torch_npu = lambda logits, labels, weight=None: None
    with pytest.raises(RuntimeError, match="signature"):
        log_probs.install_huanyu_logprob_chunking()
