"""Image boundaries must survive contraction, expansion and R3 validation."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch


def load_fixture(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


visual = load_fixture("_image_visual_fixture", "test_video_rollout_on_cpu.py")
training = load_fixture("_image_training_fixture", "test_processor_bridge_on_cpu.py")
rollout = visual.rollout
inputs = importlib.import_module(f"{visual.PACKAGE}.rollout_inputs")
replay = importlib.import_module(f"{visual.PACKAGE}.expert_replay.server_patch")
checker = importlib.import_module(f"{visual.PACKAGE}.check_image_contract")


@pytest.fixture
def native():
    module = visual.make_native()
    with patch.object(rollout, "import_module", return_value=module), patch.object(
        rollout, "_image_processor", return_value=visual.ImageProcessor()
    ):
        rollout.install_huanyu_video_rollout()
        yield module


def image_updates(native, grids):
    proc = native.MiniCPMO50OmniLLMMultiModalProcessor()
    out = {"image": [{"image_grid_thw": SimpleNamespace(data=torch.tensor(g))} for g in grids]}
    updates = proc._get_prompt_updates({}, {}, out)
    assert [u for u in updates if u.modality == "audio"] == [proc.updates[2]]
    return [u for u in updates if u.modality == "image"]


@pytest.mark.parametrize("grid,count", [([1, 22, 30], 165), ([[1, 40, 88]], 880), ([1, 2, 2], 1)])
def test_both_image_spellings_keep_boundaries_and_only_select_embedding_slots(native, grid, count):
    tokenizer = training.Tokenizer()
    updates = image_updates(native, [grid])
    assert [u.target for u in updates] == ["<|vision_start|><|image_pad|><|vision_end|>", "(<image>./</image>)"]
    for update in updates:
        text, selected = update.replacement(0)
        ids = tokenizer.encode(text)
        assert ids == [14] + [13] * count + [15]
        assert selected == "<|image_pad|>"
        mask = np.array(ids) == tokenizer.encode(selected)[0]
        assert not mask[0] and not mask[-1] and mask.sum() == count


def replace_once(ids, target, replacement):
    start = next(i for i in range(len(ids)) if ids[i:i + len(target)] == target)
    return ids[:start] + replacement + ids[start + len(target):]


def test_aqav_396_token_prompt_roundtrips_without_shifting_replay_rows(native, monkeypatch):
    monkeypatch.setenv("HUANYU_REPLAY_DEBUG", "0")
    proc = training.processor()
    proc.image_processor = lambda images, **kw: {
        "image_grid_thw": torch.tensor([[1, 22, 30]]), "pixel_values": torch.ones(660, 4)
    }
    waveform = np.zeros(16000, dtype=np.float32)
    prompt = proc("x" * 215 + "(<image>./</image>)(<audio>./</audio>)",
                  images=[object()], audios=[waveform])["input_ids"][0].tolist()
    assert len(prompt) == 396
    contracted = inputs.contract_rollout_prompt(proc, prompt, {"image": [object()], "audio": [waveform]})
    update = image_updates(native, [[1, 22, 30]])[1]
    text, _ = update.replacement(0)
    expanded = replace_once(contracted, proc.tokenizer.encode(update.target), proc.tokenizer.encode(text))
    # Audio expansion is unchanged by this patch; keep its original 12 slots.
    expanded = replace_once(expanded, proc.tokenizer.encode("(<audio>./</audio>)"), [10] + [12] * 12 + [11])
    assert expanded == prompt
    routes = np.tile(np.array([0, 1]), (len(prompt) + 3 - 1, 2, 1))
    assert replay.validate_routed_experts(routes, vllm_prompt_len=len(expanded), num_output_tokens=3,
                                         training_prompt_len=len(prompt)) is routes
    # The deployed pads-only replacement reproduced 394 vs 396. The strict
    # replay check must still reject it; no padding/cropping of route rows.
    old = [token for token in expanded if token not in (14, 15)]
    assert len(old) == 394
    with pytest.raises(RuntimeError, match="shifted by 2"):
        replay.validate_routed_experts(routes[:-2], vllm_prompt_len=len(old), num_output_tokens=3,
                                       training_prompt_len=len(prompt))


def test_multiple_images_use_each_grid_and_keep_interleaved_text(native):
    proc = training.processor()
    grids = [[1, 2, 2], [1, 4, 6]]
    proc.image_processor = lambda images, **kw: {
        "image_grid_thw": torch.tensor(grids), "pixel_values": torch.ones(28, 4)
    }
    media = [object(), object()]
    prompt = proc("a(<image>./</image>)b(<image>./</image>)c", images=media)["input_ids"][0].tolist()
    actual = inputs.contract_rollout_prompt(proc, prompt, {"image": media})
    update = image_updates(native, grids)[1]
    for index in range(2):
        text, _ = update.replacement(index)
        actual = replace_once(actual, proc.tokenizer.encode(update.target), proc.tokenizer.encode(text))
    assert actual == prompt


@pytest.mark.parametrize("grid", [[1, 3, 3], [0, 4, 4]])
def test_invalid_grid_does_not_silently_truncate_embedding_slots(native, grid):
    with pytest.raises(ValueError, match="spatial merge"):
        image_updates(native, [grid])[0].replacement(0)


@pytest.mark.parametrize("fault", [None, "ids", "range", "mask", "missing"])
def test_native_checker_rejects_incomplete_or_misaligned_expansion(fault):
    proc = training.processor()
    inputs = proc("check (<image>./</image>) end", images=[object()])
    ids = inputs["input_ids"][0].tolist()
    begin = ids.index(14)
    calls = []

    class Boundary:
        def __init__(self):
            raise AssertionError("must not initialize an engine")

        def _get_prompt_updates(self, *args):
            return "updates"

        def _bind_and_group_updates(self, updates, counts):
            assert counts == {"image": 1}
            return "resolved"

        def _apply_prompt_updates(self, source, resolved):
            calls.append(source)
            position = SimpleNamespace(offset=begin + (fault == "range"), length=6,
                                       is_embed=None if fault == "mask" else [False, True, True, True, True, False])
            images = [] if fault == "missing" else [SimpleNamespace(to_range=lambda: position)]
            return (ids[:-2] if fault == "ids" else ids), {"image": images}

    native = SimpleNamespace(MiniCPMO50OmniLLMMultiModalProcessor=Boundary)
    if fault is None:
        checker.verify_updates(proc, native, SimpleNamespace(_name_or_path="2748"), inputs)
        assert len(calls) == 3  # both compact spellings plus already-expanded IDs
    else:
        with pytest.raises(AssertionError):
            checker.verify_updates(proc, native, SimpleNamespace(_name_or_path="2748"), inputs)
