"""Training diagnostics preserve results/errors and dump live stacks without an NPU."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "train_hang_probe.py"
spec = importlib.util.spec_from_file_location("huanyu_train_hang_probe", SOURCE)
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, error=None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = Visual()
        self.error = error

    def forward(self, x):
        result = self.model.visual(x)
        if self.error is not None:
            raise self.error
        return result


def records(probe):
    return [json.loads(line) for line in probe.path.with_suffix(".jsonl").read_text().splitlines()]


@pytest.fixture
def enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HUANYU_TRAIN_HANG_DIR", str(tmp_path))
    monkeypatch.setenv("HUANYU_TRAIN_HANG_CALLS", "2")
    timers = []
    monkeypatch.setattr(probe_module.faulthandler, "dump_traceback_later", lambda *a, **k: timers.append("start"))
    monkeypatch.setattr(probe_module.faulthandler, "cancel_dump_traceback_later", lambda: timers.append("cancel"))
    return timers


def test_disabled_does_not_change_forward(monkeypatch):
    monkeypatch.delenv("HUANYU_TRAIN_HANG_DIR", raising=False)
    model = Model()
    original = model.forward
    assert probe_module.install_train_hang_probe(model) is None
    assert model.forward == original
    assert not model._forward_pre_hooks


def test_limits_calls_preserves_result_and_observes_existing_pre_hook(enabled):
    model = Model()
    probe = probe_module.install_train_hang_probe(model)
    try:
        assert probe_module.install_train_hang_probe(model) is None
        x = torch.ones(2, 3, requires_grad=True)
        for _ in range(3):
            assert model(x) is x
        entries = records(probe)
        assert len([r for r in entries if r["event"] == "forward_enter"]) == 2
        exits = [r for r in entries if r["event"] == "forward_exit"]
        assert all(r["python_forward_returned"] and not r["exception_logged"] for r in exits)
        assert all(r["scope_calls"]["model.visual.blocks.1"] == 1 for r in exits)
        assert enabled == ["start", "cancel", "start", "cancel"]
        assert probe.thread.is_alive()
    finally:
        probe.close()
    assert not probe.thread.is_alive()


@pytest.mark.parametrize("error", [torch.OutOfMemoryError("simulated OOM"), ValueError("Vision patch/grid mismatch")])
def test_original_error_and_traceback_survive(enabled, error):
    model = Model(error)
    probe = probe_module.install_train_hang_probe(model)
    try:
        with pytest.raises(type(error)) as caught:
            model(torch.zeros(1))
        assert caught.value is error
        entries = records(probe)
        failure = next(r for r in entries if r["event"] == "exception")
        assert failure["error"] == str(error)
        assert "raise self.error" in failure["traceback"]
        assert entries[-1]["event"] == "forward_exit"
        assert entries[-1]["exception_logged"]
        assert not entries[-1]["python_forward_returned"]
        assert enabled == ["start", "cancel"]
    finally:
        probe.close()


def test_visual_entry_precedes_failing_unshard_hook(enabled):
    model = Model()
    def fail_unshard(*args):
        raise RuntimeError("simulated unshard failure")
    model.model.visual.register_forward_pre_hook(fail_unshard)
    probe = probe_module.install_train_hang_probe(model)
    try:
        with pytest.raises(RuntimeError, match="unshard"):
            model(torch.zeros(1))
        entries = records(probe)
        failure = next(r for r in entries if r["event"] == "exception")
        assert failure["last_phase"] == "model.visual.enter"
        assert not any(r["event"] == "scope_return" for r in entries)
    finally:
        probe.close()


def test_metadata_never_reads_tensor_values():
    class TensorMetadataOnly:
        shape, dtype, device = (2, 3), "bfloat16", "npu:4"
        def __repr__(self):
            raise AssertionError("must not read tensor values")
        def cpu(self):
            raise AssertionError("must not transfer tensors")
    result = probe_module._metadata({"pixels": TensorMetadataOnly()})
    assert result["pixels"] == dict(shape=[2, 3], dtype="bfloat16", device="npu:4")


def test_memory_does_not_initialize_device_or_synchronize(enabled, monkeypatch):
    model = Model()
    probe = probe_module.install_train_hang_probe(model)
    try:
        backend = SimpleNamespace(is_initialized=lambda: False)
        probe.torch = SimpleNamespace(npu=backend)
        assert "not initialized" in probe.memory()["npu_allocator"]
        calls = []
        backend.is_initialized = lambda: True
        backend.current_device = lambda: 4
        def memory_stats(device):
            calls.append(device)
            return {"allocated_bytes.all.current": 123, "num_ooms": 1}
        backend.memory_stats = memory_stats
        result = probe.memory()
        assert result["npu_device"] == 4
        assert result["npu_allocator"]["allocated_bytes.all.current"] == 123
        assert result["npu_allocator"]["num_ooms"] == 1
        assert calls == [4]
    finally:
        probe.close()


def test_real_timer_writes_live_stack_and_does_not_kill_process(tmp_path):
    # Exercise CPython's actual timer in a separate process so no pytest or Ray
    # watchdog is replaced. No model/weights/torch import in this subprocess.
    script = '''
import importlib.util, pathlib, sys, time
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = m.TrainHangProbe(sys.argv[2], SimpleNamespace(), seconds=0.05, calls=1)
p.root_enter(None, (), {})
time.sleep(0.2)
p.returned = True
p.root_exit(None, (), {}, None)
p.close()
assert "Timeout" in p.path.with_suffix(".stacks.txt").read_text()
print("survived")
'''
    result = subprocess.run([sys.executable, "-c", script, str(SOURCE), str(tmp_path)],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "survived" in result.stdout
