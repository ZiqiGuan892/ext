"""Replay must find and consume installed MoE targets through LoRA wrappers."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_replay_wrapper_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
hook = importlib.import_module(f"{PACKAGE}.expert_replay.engine_hook")
moe = importlib.import_module(f"{PACKAGE}.expert_replay.moe_kernel")
controller = hook.controller


class SparseMoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = SimpleNamespace(num_experts=4)


class LoraWrapper(torch.nn.Module):
    """PEFT LoRA's public base-model and attribute-forwarding contract."""

    def __init__(self, model):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = model

    def get_base_model(self):
        return self.base_model.model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_model":
                raise
            # PeftModel -> LoraModel.model -> underlying causal LM. In
            # particular, wrapper.model is the causal LM, not its .model.
            if name == "model":
                return self.base_model.model
            return getattr(self.base_model.model, name)


def shell(module, attr):
    wrapper = torch.nn.Module()
    setattr(wrapper, attr, module)
    return wrapper


def model(multimodal=True):
    result = torch.nn.Module()
    result.model = torch.nn.Module()
    language = torch.nn.Module()
    language.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in language.layers:
        layer.mlp = SparseMoeBlock()
    if multimodal:
        result.model.language_model = language
    else:
        result.model = language
    setattr(result, hook._MARK, True)
    return result


@pytest.fixture(autouse=True)
def clean_controller():
    controller.end_batch()
    yield
    controller.end_batch()


@pytest.fixture
def installed(monkeypatch):
    # Exercise the real installer without importing Transformers or NPU ops.
    hf = ModuleType("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    hf.Qwen3_5MoeSparseMoeBlock = SparseMoeBlock
    monkeypatch.setitem(sys.modules, hf.__name__, hf)
    monkeypatch.setenv("MINICPMO_MOE_MATCH_ROLLOUT", "0")
    monkeypatch.setenv("HUANYU_EXPERT_REPLAY", "1")
    monkeypatch.setenv("HUANYU_REPLAY_DEBUG", "0")

    def build(multimodal=True):
        base = model(multimodal)
        assert moe.install_fused_moe(base) == 2
        return base

    return build


@pytest.mark.parametrize("multimodal", [True, False])
@pytest.mark.parametrize("wrapping", ["bare", "lora", "compile_lora", "fsdp_compile_lora", "lora_fsdp"])
def test_geometry_after_installer_then_wrappers(installed, multimodal, wrapping):
    base = installed(multimodal)
    wrapped = base
    if wrapping == "lora_fsdp":
        wrapped = LoraWrapper(shell(base, "_fsdp_wrapped_module"))
    elif wrapping != "bare":
        wrapped = LoraWrapper(base)
        if "compile" in wrapping:
            wrapped = shell(wrapped, "_orig_mod")
        if "fsdp" in wrapping:
            wrapped = shell(wrapped, "_fsdp_wrapped_module")
    engine = SimpleNamespace(module=wrapped, engine_config=SimpleNamespace(forward_only=False))
    assert hook._replay_engine(engine)
    assert hook._geometry(engine) == (2, 4)
    # Geometry discovery must leave the installed forward and absolute layer IDs intact.
    for index, layer in enumerate(moe._language_layers(base)):
        assert layer.mlp.forward.__func__ is moe._fused_moe_forward
        assert getattr(layer.mlp, moe._LAYER_ATTR) == index


def test_peft_model_attribute_is_not_the_decoder_container(installed):
    base = installed()
    wrapped = LoraWrapper(base)
    assert wrapped.model is base
    with pytest.raises(RuntimeError, match="language model layers"):
        moe._language_layers(wrapped)
    assert hook._moe_geometry(wrapped) == (2, 4)


def test_wrapper_cycle_and_both_wrapper_links(installed):
    base = installed()
    outer = SimpleNamespace()
    outer._fsdp_wrapped_module = outer
    outer._orig_mod = LoraWrapper(base)
    outer.get_base_model = lambda: outer
    assert hook._geometry(SimpleNamespace(module=outer)) == (2, 4)
    assert len(list(hook._model_chain(outer))) == 3


def test_ref_and_unmarked_models_do_not_replay(installed):
    base = installed()
    wrapped = shell(LoraWrapper(base), "_orig_mod")
    ref = SimpleNamespace(module=wrapped, engine_config=SimpleNamespace(forward_only=True))
    assert not hook._replay_engine(ref)
    delattr(base, hook._MARK)
    ref.engine_config.forward_only = False
    assert not hook._replay_engine(ref)


def test_genuinely_missing_kernel_still_fails():
    engine = SimpleNamespace(module=LoraWrapper(model()))
    with pytest.raises(RuntimeError, match="no fused MoE blocks installed"):
        hook._geometry(engine)
    assert not hasattr(engine, "_huanyu_replay_geometry")


def routing_batch():
    # Two samples of lengths 4 and 2; each last token has no routing record.
    values = torch.tensor([[[0, 1], [1, 2]]] * 6, dtype=torch.int64)
    values[[3, 5]] = 0
    routed = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 4, 6]))
    return {"routed_experts": routed}, {"input_ids": torch.ones(2, 4, dtype=torch.long)}


def test_wrapped_actor_arms_targets_and_replays_with_router_gradients(installed):
    base = installed()
    engine = SimpleNamespace(module=shell(LoraWrapper(base), "_fsdp_wrapped_module"))
    batch, inputs = routing_batch()
    controller.begin_batch()
    hook._arm_micro_batch(engine, batch, inputs)
    expected = torch.zeros(8, 2, 2, dtype=torch.long)
    expected[:4] = batch["routed_experts"].values()[:4]
    expected[4:6] = batch["routed_experts"].values()[4:]
    recorded = torch.tensor([True, True, True, False, True, False, False, False])
    for index, layer in enumerate(moe._language_layers(base)):
        logits = torch.tensor([[0.1, 0.3, 0.8, 1.2]] * 8, requires_grad=True)
        weights, indices = logits.softmax(-1).topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        actual_indices, actual_weights = moe._substitute_replay(layer.mlp, logits, weights, indices)
        torch.testing.assert_close(actual_indices[recorded], expected[recorded, index])
        torch.testing.assert_close(actual_indices[~recorded], indices[~recorded])
        torch.testing.assert_close(actual_weights[~recorded], weights[~recorded])
        reference = logits.softmax(-1).gather(1, expected[:, index])
        reference = reference / reference.sum(-1, keepdim=True)
        torch.testing.assert_close(actual_weights[recorded], reference[recorded])
        (actual_weights * torch.tensor([0.2, 0.7])).sum().backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[recorded].abs().sum() > 0
        # Checkpoint recomputation must still see exactly this micro-batch.
        repeated, _ = moe._substitute_replay(layer.mlp, logits, weights, indices)
        torch.testing.assert_close(repeated, actual_indices)
    controller.assert_all_layers_fired()
    controller.end_batch()
    assert not controller.is_armed()


@pytest.mark.parametrize("invalid", ["missing", "layers", "expert_ids"])
def test_wrapped_actor_keeps_routing_validation(installed, invalid):
    engine = SimpleNamespace(module=LoraWrapper(installed()))
    batch, inputs = routing_batch()
    if invalid == "missing":
        batch = {}
        message = "has no 'routed_experts'"
    else:
        routed = batch["routed_experts"]
        values = routed.values()[:, :1] if invalid == "layers" else routed.values() + 8
        batch = {"routed_experts": torch.nested.nested_tensor_from_jagged(values, routed.offsets())}
        message = "capture layer count" if invalid == "layers" else "outside"
    controller.begin_batch()
    with pytest.raises(RuntimeError, match=message):
        hook._arm_micro_batch(engine, batch, inputs)
    assert not controller.is_armed()
