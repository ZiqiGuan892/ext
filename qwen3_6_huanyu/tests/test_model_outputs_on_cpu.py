"""Numerics and allocation boundaries for padded Huanyu FSDP output statistics."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_model_outputs_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


log_probs = load("log_probs")
entropy_impl = load("entropy")
outputs_impl = load("model_outputs")


def native_reference(logits, labels):
    return -F.cross_entropy(logits, labels, reduction="none")


def reference_entropy(logits):
    work = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
    logp = work.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


def _scale_logits_by_temperature(logits, temperature, *, is_unit_temperature):
    return logits if is_unit_temperature else logits / temperature.clamp(min=1e-8).to(logits.dtype)


def dense(seed=42, dtype=torch.float64):
    return torch.randn(3, 9, 11, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def packed(logits, lengths):
    return torch.cat([logits[b, :length] for b, length in enumerate(lengths)])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("lengths", [(2, 7, 4), (9, 1, 8), (0, 7, 0)])
def test_ragged_values_weighted_gradients_and_padding(dtype, lengths):
    value = dense(dtype=dtype).requires_grad_()
    baseline = value.detach().clone().requires_grad_()
    # Deliberately arbitrary packed labels, including cross-sample boundaries.
    labels = (torch.arange(sum(lengths)) * 7 + 3) % value.shape[-1]
    calls = []

    def backend(block, targets):
        assert block.shape[0] <= 3
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        calls.append(tuple(block.shape))
        return native_reference(block, targets)

    actual = log_probs.chunked_ragged_logprobs_from_logits(value, labels, lengths, backend, 3)
    expected = native_reference(packed(baseline, lengths), labels)
    weights = torch.linspace(-0.7, 1.2, len(labels), dtype=dtype)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value.grad, baseline.grad)
    assert sum(shape[0] for shape in calls) == 2 * sum(lengths)
    for batch, length in enumerate(lengths):
        assert torch.count_nonzero(value.grad[batch, length:]) == 0


def test_ragged_noncontiguous_input_and_saved_storage():
    value = dense()[:, 1:8, ::2].requires_grad_()
    baseline = value.detach().clone().requires_grad_()
    lengths = (2, 7, 3)
    labels = torch.arange(sum(lengths)) % value.shape[-1]
    saved = []

    def save(tensor):
        saved.append(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
        actual = log_probs.chunked_ragged_logprobs_from_logits(value, labels, lengths, native_reference, 3)
    assert set(saved) <= {value.untyped_storage().data_ptr(), labels.untyped_storage().data_ptr()}
    expected = native_reference(packed(baseline, lengths), labels)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value.grad, baseline.grad)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_ragged_inference_modes(mode):
    value = dense().requires_grad_()
    lengths = (2, 7, 4)
    labels = torch.arange(sum(lengths)) % value.shape[-1]
    with mode():
        actual = log_probs.chunked_ragged_logprobs_from_logits(value, labels, lengths, native_reference, 3)
        assert not actual.requires_grad
        torch.testing.assert_close(actual, native_reference(packed(value, lengths), labels))


@pytest.mark.parametrize("shape,lengths", [((3, 9, 11), (0, 0, 0)), ((0, 9, 11), ())])
def test_ragged_empty_preserves_backend_and_autograd(shape, lengths):
    value = torch.zeros(shape, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)
    actual = log_probs.chunked_ragged_logprobs_from_logits(value, labels, lengths, native_reference)
    assert actual.shape == (0,)
    actual.sum().backward()
    assert value.grad.shape == value.shape
    assert torch.count_nonzero(value.grad) == 0


def test_ragged_autocast_and_gradcheck():
    lengths = (2, 7, 4)
    labels = torch.arange(sum(lengths)) % 11
    value = dense(dtype=torch.bfloat16).requires_grad_()
    baseline = value.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = log_probs.chunked_ragged_logprobs_from_logits(value, labels, lengths, native_reference, 3)
        expected = native_reference(packed(baseline, lengths), labels)
    actual.sum().backward()
    expected.sum().backward()
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value.grad, baseline.grad)
    small = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 2, 3])
    assert torch.autograd.gradcheck(
        lambda x: log_probs.chunked_ragged_logprobs_from_logits(x, targets, (1, 2), native_reference, 2), (small,)
    )


@pytest.mark.parametrize("lengths", [(2, 10, 4), (2, -1, 4), (2, 7), (True, 7, 4)])
def test_invalid_ragged_lengths(lengths):
    with pytest.raises(ValueError, match="lengths"):
        log_probs.chunked_ragged_logprobs_from_logits(dense(), torch.zeros(13, dtype=torch.long), lengths, native_reference)


def test_invalid_packed_labels_and_logits():
    with pytest.raises(ValueError, match="packed label"):
        log_probs.chunked_ragged_logprobs_from_logits(dense(), torch.zeros(3, dtype=torch.long), (2, 7, 4), native_reference)
    with pytest.raises(ValueError, match="floating-point"):
        log_probs.chunked_ragged_logprobs_from_logits(torch.zeros(2, 3), torch.zeros(2), (1, 1), native_reference)


def engine_class():
    class Engine:
        def __init__(self, module, checkpointing=False):
            self.module = module
            self.engine_config = SimpleNamespace(entropy_checkpointing=checkpointing)
            self.original_calls = 0

        def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
            # The relevant pre-fix FSDP contract; unsupported modes are sentinels.
            self.original_calls += 1
            if micro_batch.get("fallback", False):
                return {"fallback": logits_processor_func}
            logits = _scale_logits_by_temperature(
                output.logits, output_args["temperature"].unsqueeze(-1).unsqueeze(-1),
                is_unit_temperature=output_args.get("temperature_is_one", False),
            )
            offsets = micro_batch["input_ids"].offsets()
            lengths = offsets.diff()
            starts = torch.zeros_like(lengths)
            entropy = reference_entropy(logits) if micro_batch.get("calculate_entropy") else None
            logits = torch.nested.narrow(logits, 1, starts, lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            labels = output_args["input_ids_rmpad_rolled"]
            logp = native_reference(logits_rmpad, labels)
            result = {"log_probs": torch.nested.nested_tensor_from_jagged(logp, offsets)}
            if entropy is not None:
                entropy = torch.cat([entropy[b, :length] for b, length in enumerate(lengths.tolist())])
                result["entropy"] = torch.nested.nested_tensor_from_jagged(entropy, offsets)
            return result
    return Engine


@pytest.fixture
def engine_module(monkeypatch):
    impl = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.FSDPEngine = type("UnusedBaseEngine", (), {})
    impl.FSDPEngineWithLMHead = engine_class()
    impl.tu = SimpleNamespace(get_non_tensor_data=lambda data, key, default: data.get(key, default))
    impl.DatasetPadMode = SimpleNamespace(NO_PADDING="no_padding")
    impl.DTensor = type("FakeDTensor", (), {})
    impl._scale_logits_by_temperature = _scale_logits_by_temperature
    impl.verl_F = SimpleNamespace(entropy_from_logits=entropy_impl.chunked_entropy_from_logits)
    impl.logprobs_from_logits = native_reference
    monkeypatch.setitem(sys.modules, impl.__name__, impl)
    return impl


def batch():
    offsets = torch.tensor([0, 2, 9, 13])
    labels = (torch.arange(13) * 7 + 3) % 11
    input_ids = torch.nested.nested_tensor_from_jagged(labels, offsets)
    return {"input_ids": input_ids, "use_remove_padding": False}, {
        "input_ids_rmpad_rolled": torch.roll(labels, -1), "temperature": torch.ones(3), "temperature_is_one": True,
    }


@pytest.mark.parametrize("calculate_entropy", [False, True])
@pytest.mark.parametrize("temperature", [None, [1.0, 0.7, 1.4]])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_engine_outputs_gradients_and_no_vocabulary_cat(engine_module, monkeypatch, calculate_entropy, temperature, checkpointing):
    impl = engine_module
    model = SimpleNamespace()
    engine = impl.FSDPEngineWithLMHead(model, checkpointing)
    original = engine.prepare_model_outputs
    micro_batch, args = batch()
    micro_batch["calculate_entropy"] = calculate_entropy
    if temperature is not None:
        args["temperature"] = torch.tensor(temperature)
        args["temperature_is_one"] = False
    value = dense().requires_grad_()
    baseline = value.detach().clone().requires_grad_()
    expected = original(SimpleNamespace(logits=baseline), args, micro_batch, None)
    outputs_impl.install_huanyu_scalar_output_packing(model)
    native_cat = torch.cat

    def scalar_cat(tensors, *extra, **kwargs):
        tensors = tuple(tensors)
        assert all(tensor.ndim == 1 for tensor in tensors), "Unexpected full-vocabulary packing"
        return native_cat(tensors, *extra, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch, "cat", scalar_cat)
        actual = engine.prepare_model_outputs(SimpleNamespace(logits=value), args, micro_batch, None)
        loss = sum(result.values().sum() * (0.3 if key == "entropy" else 1) for key, result in actual.items())
        loss.backward()
    sum(result.values().sum() * (0.3 if key == "entropy" else 1) for key, result in expected.items()).backward()
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].offsets() is micro_batch["input_ids"].offsets()
        torch.testing.assert_close(actual[key].values(), expected[key].values())
    torch.testing.assert_close(value.grad, baseline.grad)
    assert engine.original_calls == 1


@pytest.mark.parametrize("flag,value", [
    ("use_remove_padding", True), ("use_fused_kernels", True), ("pad_mode", "other"),
    ("distillation_use_topk", True), ("distillation_only", True), ("calculate_sum_pi_squared", True),
])
def test_unsupported_modes_delegate(engine_module, flag, value):
    model = SimpleNamespace()
    outputs_impl.install_huanyu_scalar_output_packing(model)
    engine = engine_module.FSDPEngineWithLMHead(model)
    micro_batch, args = batch()
    micro_batch.update({flag: value, "fallback": True})
    sentinel = object()
    assert engine.prepare_model_outputs(None, args, micro_batch, sentinel) == {"fallback": sentinel}
    assert engine.original_calls == 1


def test_model_scope_peft_wrapping_and_idempotence(engine_module):
    model = SimpleNamespace()
    outputs_impl.install_huanyu_scalar_output_packing(model)
    method = engine_module.FSDPEngineWithLMHead.prepare_model_outputs
    assert outputs_impl.install_huanyu_scalar_output_packing(model) is model
    assert engine_module.FSDPEngineWithLMHead.prepare_model_outputs is method
    assert not hasattr(engine_module.FSDPEngine, "prepare_model_outputs")
    peft = SimpleNamespace(get_base_model=lambda: model)
    engine = engine_module.FSDPEngineWithLMHead(SimpleNamespace(_orig_mod=peft))
    micro_batch, args = batch()
    with torch.no_grad():
        actual = engine.prepare_model_outputs(SimpleNamespace(logits=dense()), args, micro_batch, None)
    assert actual["log_probs"].values().shape == (13,)
    assert engine.original_calls == 0
    other = engine_module.FSDPEngineWithLMHead(SimpleNamespace())
    micro_batch["fallback"] = True
    assert other.prepare_model_outputs(None, args, micro_batch, "untouched") == {"fallback": "untouched"}


def test_engine_oom_is_not_swallowed(engine_module):
    model = SimpleNamespace()
    outputs_impl.install_huanyu_scalar_output_packing(model)
    engine = engine_module.FSDPEngineWithLMHead(model)
    error = torch.OutOfMemoryError("native allocation failed")

    def fail(*args, **kwargs):
        raise error

    engine_module.logprobs_from_logits = fail
    micro_batch, args = batch()
    with pytest.raises(torch.OutOfMemoryError) as caught:
        engine.prepare_model_outputs(SimpleNamespace(logits=dense()), args, micro_batch, None)
    assert caught.value is error


def test_unknown_engine_signature_and_source_are_rejected(engine_module):
    def changed(self, *args):
        return {}
    engine_module.FSDPEngineWithLMHead.prepare_model_outputs = changed
    with pytest.raises(RuntimeError, match="signature"):
        outputs_impl.install_huanyu_scalar_output_packing(SimpleNamespace())

    def unknown(self, output, output_args, micro_batch, logits_processor_func):
        return {}
    engine_module.FSDPEngineWithLMHead.prepare_model_outputs = unknown
    with pytest.raises(RuntimeError, match="implementation"):
        outputs_impl.install_huanyu_scalar_output_packing(SimpleNamespace())
