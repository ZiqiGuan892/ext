"""Exercise media-dependent gather ordering without loading an NPU model.

The fake FSDP units model the public sharded/unsharded state transitions and log
only actual gathers. They do not validate HCCL, stream ordering, or device memory.
The real per-item vision wrapper and real torch forward/backward run on CPU.
"""

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("huanyu_vision_fsdp_test", ROOT / "vision_fsdp.py")
media = load("huanyu_media_precision_test", ROOT / "qwen_precision/media_precision.py")


def fake_shard(module, name, trace):
    state = SimpleNamespace(sharded=True, reshard_after_forward=True, gathers=0, releases=0)

    def unshard(self):
        if state.sharded:
            trace.append(name)
            state.gathers += 1
            state.sharded = False

    def reshard(self):
        state.releases += 1
        state.sharded = True

    def set_reshard_after_forward(self, enabled, recurse=True):
        assert recurse is False
        state.reshard_after_forward = enabled

    module.unshard = MethodType(unshard, module)
    module.reshard = MethodType(reshard, module)
    module.set_reshard_after_forward = MethodType(set_reshard_after_forward, module)
    module.shard_state = state
    module.register_forward_pre_hook(lambda child, args: child.unshard())

    def post(child, args, output):
        if state.reshard_after_forward:
            child.reshard()

    module.register_forward_hook(post)


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
        self.pos_embed = torch.nn.Embedding(1, 2)
        with torch.no_grad():
            for block in self.blocks:
                block.weight.copy_(torch.tensor([[0.3, 0.1], [0.2, 0.4]]))
            self.pos_embed.weight.fill_(0.05)
        self.requires_grad_(False)

    def forward(self, hidden_states, grid_thw):
        # Like the native tower, read an embedding weight without invoking its
        # module hook; pre-unsharding must cover this unit too.
        hidden_states = hidden_states + self.pos_embed.weight
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states, pooler_output=hidden_states)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = Vision()
        self.model.language = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.model.language.weight.fill_(0.2)
        self.trace = []

    def forward(self, images=0, videos=0, error=None):
        features = torch.tensor([[0.5, 1.0]])
        for count in (images, videos):
            if count:
                pixels = torch.arange(1, count * 2 + 1, dtype=torch.float32).reshape(count, 2)
                grids = torch.ones(count, 3, dtype=torch.long)
                output = self.model.visual(pixels, grid_thw=grids)
                features = features + output.pooler_output.sum(0)
        if error:
            raise error
        return self.model.language(features)


def wrap_fsdp(model):
    for name, child in model.model.visual.named_modules():
        if isinstance(child, (torch.nn.Linear, torch.nn.Embedding)):
            fake_shard(child, f"visual.{name}", model.trace)
    fake_shard(model.model.language, "language", model.trace)
    # Mirror the existing visual-only unshard hook. Keeping it installed must
    # not cause a second gather after the base-forward guard gathered the units.
    def visual_pre(tower, args):
        for _, child in sorted(tower.named_modules()):
            if callable(getattr(child, "unshard", None)):
                child.unshard()
    model.model.visual.register_forward_pre_hook(visual_pre)


def make_model(monkeypatch, *, managed=True, sharded=True):
    monkeypatch.setenv("MINICPMO_ISOLATE_MEDIA_ENCODERS", "1")
    model = Model()
    assert media.install_actor(model)
    if managed:
        assert guard.install_frozen_vision_fsdp(model)
    # Match the real lifecycle: FSDP is installed after the forward wrappers.
    if sharded:
        wrap_fsdp(model)
    return model


ORDER = ["visual.blocks.0", "visual.blocks.1", "visual.pos_embed", "language"]


def assert_released(model):
    for _, child in guard._vision_units(model.model.visual):
        assert child.shard_state.sharded


def test_reproduces_old_collective_mismatch_between_one_and_two_images(monkeypatch):
    one = make_model(monkeypatch, managed=False)
    two = make_model(monkeypatch, managed=False)
    one(images=1)
    two(images=2)
    assert one.trace == ORDER
    assert two.trace == ORDER[:-1] + ["visual.blocks.0", "visual.blocks.1", "language"]
    # This second image gather is precisely where one rank may gather language
    # parameters while another is gathering vision parameters.
    assert one.trace != two.trace


@pytest.mark.parametrize("direct", [False, True], ids=["module-call", "peft-direct-forward"])
@pytest.mark.parametrize("images,videos", [(0, 0), (1, 0), (3, 0), (0, 2), (2, 3)])
def test_uniform_gathers_with_variable_media_and_repeated_calls(monkeypatch, direct, images, videos):
    model = make_model(monkeypatch)
    call = model.forward if direct else model
    # Even residual unsharded state from earlier inference is normalized.
    model.model.visual.blocks[0].shard_state.sharded = False
    for image_count, video_count in [(images, videos), (0, 0), (2, 1)]:
        model.trace.clear()
        call(images=image_count, videos=video_count)
        assert model.trace == ORDER
        assert_released(model)
        assert model.model.language.shard_state.reshard_after_forward is True


@pytest.mark.parametrize("direct", [False, True])
def test_preserves_outputs_language_gradients_and_frozen_vision(monkeypatch, direct):
    model = make_model(monkeypatch)
    reference = make_model(monkeypatch, managed=False)
    actual = (model.forward if direct else model)(images=2, videos=3)
    expected = reference(images=2, videos=3)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.model.language.weight.grad, reference.model.language.weight.grad)
    assert all(p.grad is None for p in model.model.visual.parameters())
    assert_released(model)


@pytest.mark.parametrize("error", [ValueError("bad image"), torch.OutOfMemoryError("NPU allocation failed")])
def test_original_forward_error_is_preserved_and_units_released(monkeypatch, error):
    model = make_model(monkeypatch)
    with pytest.raises(type(error)) as caught:
        model.forward(images=2, error=error)
    assert caught.value is error
    assert_released(model)
    # Exception unwinding must also clear the guard's reentrancy flag.
    model.trace.clear()
    model(images=1)
    assert model.trace == ORDER


def test_gather_failure_releases_other_units_without_entering_model(monkeypatch):
    model = make_model(monkeypatch)
    unit = model.model.visual.blocks[1]
    original = unit.unshard
    error = RuntimeError("all-gather failed")
    def fail():
        raise error
    unit.unshard = fail
    with pytest.raises(RuntimeError) as caught:
        model(images=1)
    assert caught.value is error
    assert "language" not in model.trace
    assert_released(model)
    unit.unshard = original
    model.trace.clear()
    model(images=1)
    assert model.trace == ORDER


@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_failure_is_visible_without_masking_original_error(monkeypatch, body_fails):
    model = make_model(monkeypatch)
    unit = model.model.visual.blocks[0]
    original_release = unit.reshard
    original_forward = model.model.language.forward
    error = ValueError("original forward failure")

    def fail_release():
        raise RuntimeError("release failed")

    def body(*args, **kwargs):
        unit.reshard = fail_release
        if body_fails:
            raise error
        return original_forward(*args, **kwargs)

    model.model.language.forward = body
    with pytest.raises(ValueError if body_fails else RuntimeError) as caught:
        model(images=2)
    if body_fails:
        assert caught.value is error
        assert "release failed" in caught.value.__notes__[0]
    else:
        assert "release failed" in str(caught.value)
    # Failure in one release must not prevent releasing the remaining units.
    assert model.model.visual.blocks[1].shard_state.sharded
    assert model.model.visual.pos_embed.shard_state.sharded
    unit.reshard = original_release


def test_unsupported_fsdp_api_fails_before_any_gather_or_policy_change(monkeypatch):
    model = make_model(monkeypatch)
    model.model.visual.pos_embed.set_reshard_after_forward = None
    with pytest.raises(RuntimeError, match="lacks set_reshard_after_forward"):
        model.forward(images=1)
    assert model.trace == []
    assert model.model.visual.blocks[0].shard_state.reshard_after_forward is True


def test_discovery_after_install_and_idempotence(monkeypatch):
    model = make_model(monkeypatch, sharded=False)
    method = model.forward
    assert not guard.install_frozen_vision_fsdp(model)
    assert model.forward == method
    model(images=2)  # No FSDP yet: ordinary CPU model call is unaffected.
    wrap_fsdp(model)
    model(images=2)
    assert model.trace == ORDER


@pytest.mark.parametrize("before_install", [False, True])
def test_trainable_vision_is_not_managed(monkeypatch, before_install):
    model = make_model(monkeypatch, managed=False)
    if before_install:
        model.model.visual.requires_grad_(True)
        assert not guard.install_frozen_vision_fsdp(model)
    else:
        assert guard.install_frozen_vision_fsdp(model)
        # Simulates trainable adapters added by PEFT after model configuration.
        model.model.visual.requires_grad_(True)
    model(images=2).sum().backward()
    assert model.trace != ORDER
    assert model.model.visual.blocks[0].shard_state.reshard_after_forward is True
    assert model.model.visual.blocks[0].weight.grad is not None


def test_no_vision_is_a_noop():
    model = Model()
    del model.model.visual
    assert not guard.install_frozen_vision_fsdp(model)


def test_probe_wraps_guard_and_direct_forward(monkeypatch, tmp_path):
    probe = load("huanyu_vision_probe_test", ROOT / "train_hang_probe.py")
    monkeypatch.setenv("HUANYU_TRAIN_HANG_DIR", str(tmp_path))
    monkeypatch.setenv("HUANYU_TRAIN_HANG_SECONDS", "60")
    model = make_model(monkeypatch)
    installed = probe.install_train_hang_probe(model)
    try:
        model.forward(images=2)
        assert model.trace == ORDER
        text = next(tmp_path.glob("*.jsonl")).read_text()
        assert '"entrypoint": "direct_forward"' in text
        assert '"event": "forward_exit"' in text
    finally:
        installed.close()
