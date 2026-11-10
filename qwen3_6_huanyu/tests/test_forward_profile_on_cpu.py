"""CPU contract checks for the opt-in Huanyu worker profiler.

These test wrapping/lifecycle without importing torch, vLLM, or model weights;
they do not validate device tracing or numerical/kernel performance.
"""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "forward_profile.py"
    if not path.is_file():  # Full development tree; publishing uses qwen3_6_huanyu/tests/.
        path = Path(__file__).resolve().parents[3] / "verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py"
    spec = importlib.util.spec_from_file_location("huanyu_forward_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(helper, monkeypatch, tmp_path):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "3")
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_ACTIVE", "2")
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_LEVEL", "1")
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_MODEL", "2748")
    events = []

    class Profiler:
        def __init__(self):
            self.prof_if = SimpleNamespace(prof_path="")
            self.callbacks = {}
            self.emit_metadata = True

        def start(self):
            events.append("start")
            if "start" in self.callbacks:
                self.callbacks["start"]()
            trace = self.directory / (self.tag + "_ascend_pt")
            self.prof_if.prof_path = str(trace)
            (trace / "FRAMEWORK").mkdir(parents=True)
            (trace / "FRAMEWORK/torch.op_range").write_bytes(b"mock-framework")
            (trace / "PROF_0001/device_0/data").mkdir(parents=True)
            (trace / "PROF_0001/device_0/data/aicore.data").write_bytes(b"mock-device")

        def step(self):
            events.append("step")
            if "step" in self.callbacks:
                self.callbacks["step"]()

        def stop(self):
            events.append("stop")
            if "stop" in self.callbacks:
                self.callbacks["stop"]()
            if self.emit_metadata and self.prof_if.prof_path:
                (Path(self.prof_if.prof_path) / "profiler_info_0.json").write_text(json.dumps({
                    "config": {"mock": True}, "start_info": {"mock": True}, "end_info": {"mock": True},
                }), encoding="utf-8")

    class NPU:
        def synchronize(self, device=None):
            events.append(("original_sync", device))

    npu = NPU()
    profiler = Profiler()

    @contextmanager
    def scope(label):
        events.append(("enter", label))
        try:
            yield
        finally:
            events.append(("exit", label))

    created = []

    def factory(directory, tag, active, level):
        created.append((directory, tag, active, level))
        profiler.directory, profiler.tag = directory, tag
        return profiler, scope, npu, "test-only"

    monkeypatch.setattr(helper, "_make_profiler", factory)
    monkeypatch.setattr(helper, "_identity", lambda runner: {
        "host": "cpu-test", "pid": 42, "parent_pid": 24, "model": helper._model_path(runner),
        "tp_rank": 0, "tp_ranks": [0],
    })
    monkeypatch.setattr(helper._Capture, "attention_metadata", lambda self: None)

    class GatedDelta:
        def forward(self, value, *, video_grid):
            return self._forward_core(value, video_grid=video_grid)

        def _forward_core(self, value, *, video_grid):
            npu.synchronize("npu:0")
            events.append(("gdn", value, video_grid))
            return value

    class Attention:
        def forward(self, value, *, video_grid):
            events.append(("attention", value, video_grid))
            return value

    class SparseMoe:
        def forward(self, value, *, video_grid):
            events.append(("moe", value, video_grid))
            return value

    class Model:
        def __init__(self):
            self.gdn = GatedDelta()
            self.attention = Attention()
            self.moe = SparseMoe()

        def named_modules(self):
            return iter([
                ("", self), ("thinker.llm.model.layers.0.linear_attn", self.gdn),
                ("thinker.llm.model.layers.1.self_attn", self.attention),
                ("thinker.llm.model.layers.1.mlp", self.moe),
            ])

    class Runner:
        def __init__(self, model_path="/models/2748"):
            self.model_config = SimpleNamespace(model=model_path)
            self.model = Model()
            self.expected_result = object()
            self.failure = None
            self.in_forward = None

        def _model_forward(self, value, *, video_grid):
            # This is deliberately replaced before the profiler sees it.
            raise AssertionError("stale forward used")

        def execute_model(self, scheduler_output, value, *, video_grid):
            return self._model_forward(value, video_grid=video_grid)

    # Simulate a video-forward repair installed after class definition.
    def current_forward(self, value, *, video_grid):
        events.append(("current_forward", value, video_grid))
        if self.in_forward:
            self.in_forward()
        self.model.gdn.forward(value, video_grid=video_grid)
        self.model.attention.forward(value, video_grid=video_grid)
        self.model.moe.forward(value, video_grid=video_grid)
        if self.failure is not None:
            raise self.failure
        return self.expected_result

    def new_runner(model_path="/models/2748"):
        runner = Runner(model_path)
        runner._model_forward = MethodType(current_forward, runner)
        return runner

    return SimpleNamespace(
        helper=helper, Runner=Runner, new_runner=new_runner, npu=npu,
        events=events, created=created, out=tmp_path, profiler=profiler,
    )


def scheduler(tokens=4):
    return SimpleNamespace(
        total_num_scheduled_tokens=tokens,
        num_scheduled_tokens={"request": tokens} if tokens else {},
        scheduled_encoder_inputs={},
    )


def test_capture_preserves_current_video_forward_and_restores_after_window(runtime):
    h = runtime.helper
    runner = runtime.new_runner()
    live_forward = runner._model_forward
    h.bind_runner(runtime.Runner)
    # Installation is idempotent: no nested profiler/lifecycle on the same runner.
    bound_execute = runtime.Runner.execute_model
    h.bind_runner(runtime.Runner)
    assert runtime.Runner.execute_model is bound_execute
    grid = object()
    empty = scheduler(0)
    assert runner.execute_model(empty, "dummy", video_grid=grid) is runner.expected_result
    for i in range(6):
        assert runner.execute_model(scheduler(), i, video_grid=grid) is runner.expected_result

    assert [e for e in runtime.events if isinstance(e, str)] == ["start", "step", "step", "step", "stop"]
    assert len(runtime.created) == 1
    assert runtime.created[0][2:] == (2, 1)
    # Only original code invokes synchronization, once per execute call.
    assert runtime.events.count(("original_sync", "npu:0")) == 7
    assert runner._model_forward is live_forward
    assert "synchronize" not in vars(runtime.npu)
    for _, module in runner.model.named_modules():
        assert "forward" not in vars(module)
        assert "_forward_core" not in vars(module)
    assert all(e[2] is grid for e in runtime.events if isinstance(e, tuple) and e[0] == "current_forward")

    manifest = json.loads(next(runtime.out.rglob("*.json")).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert [r["engine_iteration"] for r in manifest["iterations"]] == [3, 4, 5]
    assert [r["warmup"] for r in manifest["iterations"]] == [True, False, False]
    for category in ("GDN", "Attention", "MoE"):
        assert any(key.startswith("huanyu/" + category + "/") for key in manifest["hits"])
    assert manifest["hits"]["huanyu/explicit_npu_synchronize"] == 3
    forward_target = next(t for t in manifest["targets"] if t["label"] == "huanyu/forward")
    assert "current_forward" in forward_target["source_chain"][0]["qualname"]
    assert forward_target["source_chain"][0]["sha256"]


@pytest.mark.parametrize("mode", ["disabled", "different_model", "similar_model_basename"])
def test_disabled_or_other_model_does_not_profile(runtime, monkeypatch, mode):
    if mode == "disabled":
        monkeypatch.delenv("HUANYU_FORWARD_PROFILE_DIR")
    path = "/models/2748" if mode == "disabled" else "/models/judge" if mode == "different_model" else "/models/27480"
    runner = runtime.new_runner(path)
    runtime.helper.bind_runner(runtime.Runner)
    for _ in range(5):
        assert runner.execute_model(scheduler(), 1, video_grid=None) is runner.expected_result
    assert not runtime.created
    assert not hasattr(runner, runtime.helper._MARKER)


def test_model_exception_is_not_replaced_and_patches_are_restored(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    runner = runtime.new_runner()
    live_forward = runner._model_forward
    error = ValueError("video grid regression")
    runner.failure = error
    runtime.helper.bind_runner(runtime.Runner)
    with pytest.raises(ValueError) as caught:
        runner.execute_model(scheduler(), 2, video_grid="new-grid")
    assert caught.value is error
    assert runner._model_forward is live_forward
    assert "synchronize" not in vars(runtime.npu)
    assert [e for e in runtime.events if isinstance(e, str)] == ["start", "stop"]
    manifest = json.loads(next(runtime.out.rglob("*.json")).read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert manifest["iterations"][0]["error"] is True
    assert manifest["errors"][0]["phase"] == "forward"


def test_inflight_progress_distinguishes_forward_step_and_stop(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    h = runtime.helper
    runner = runtime.new_runner()
    h.bind_runner(runtime.Runner)
    observations = []

    def observe(expected):
        state = getattr(runner, h._MARKER)
        data = json.loads(state.manifest.read_text(encoding="utf-8"))
        assert data["phase"] == expected
        assert data["updated_at"] >= data["phase_started_at"]
        observations.append(data)

    runtime.profiler.callbacks = {
        "start": lambda: observe("profiler.start"),
        "step": lambda: observe("profiler.step"),
        "stop": lambda: observe("profiler.stop"),
    }
    runner.in_forward = lambda: observe("forward")
    for _ in range(3):
        runner.execute_model(scheduler(), 1, video_grid=None)
    forwards = [d for d in observations if d["phase"] == "forward"]
    steps = [d for d in observations if d["phase"] == "profiler.step"]
    assert [d["forwards_completed"] for d in forwards] == [0, 1, 2]
    assert all(d["status"] == "capturing" for d in forwards)
    assert [len(d["iterations"]) for d in steps] == [1, 2, 3]
    assert [d["steps_completed"] for d in steps] == [0, 1, 2]
    assert observations[-1]["status"] == "stopping"
    assert observations[-1]["steps_completed"] == 3
    final = h.capture_status(runtime.out, expected_workers=1, require_complete=True)[0][1]
    assert final["phase"] == "finished"
    assert final["status"] == "complete"


def test_profiler_step_barrier_reports_nested_phase_without_extra_sync(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    h = runtime.helper
    runner = runtime.new_runner()
    snapshots = []

    def existing_sync(device=None):
        state = getattr(runner, h._MARKER)
        snapshots.append(json.loads(state.manifest.read_text(encoding="utf-8")))

    runtime.npu.synchronize = existing_sync
    runtime.profiler.callbacks["step"] = lambda: runtime.npu.synchronize()
    h.bind_runner(runtime.Runner)
    runner.execute_model(scheduler(), 1, video_grid=None)
    state = getattr(runner, h._MARKER)
    assert len(snapshots) == 2  # original forward barrier + original profiler barrier
    assert snapshots[-1]["phase_stack"] == ["profiler.step", "profiler.synchronize"]
    state.at_exit()
    assert json.loads(state.manifest.read_text())["status"] == "incomplete"


@pytest.mark.parametrize("stage", ["start", "step", "stop"])
def test_profiler_failure_cannot_be_reported_complete(runtime, monkeypatch, stage):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    runner = runtime.new_runner()
    original = runner._model_forward
    failure = RuntimeError(stage + " failed")

    def fail():
        raise failure

    runtime.profiler.callbacks[stage] = fail
    runtime.helper.bind_runner(runtime.Runner)
    with pytest.raises(RuntimeError) as caught:
        for _ in range(3):
            runner.execute_model(scheduler(), 1, video_grid=None)
    assert caught.value is failure
    state = getattr(runner, runtime.helper._MARKER)
    assert runner._model_forward is original
    data = json.loads(state.manifest.read_text())
    assert data["status"] == "incomplete"
    assert data["errors"][0]["phase"] == "profiler." + stage
    assert runtime.events.count("stop") == 1
    assert runtime.helper.main(["check", "--profile-dir", str(runtime.out)]) == 2


def test_suppressed_native_finalization_error_does_not_pass(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    runtime.profiler.emit_metadata = False
    runner = runtime.new_runner()
    runtime.helper.bind_runner(runtime.Runner)
    for _ in range(3):
        runner.execute_model(scheduler(), 1, video_grid=None)
    data = json.loads(getattr(runner, runtime.helper._MARKER).manifest.read_text())
    assert data["steps_completed"] == 3
    assert data["status"] == "incomplete"
    assert "profiler_info" in data["raw_errors"][0]


def test_forward_failure_survives_cleanup_failure(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    runner = runtime.new_runner()
    original = ValueError("original forward failure")
    runner.failure = original

    def stop_failure():
        raise RuntimeError("cleanup failed")

    runtime.profiler.callbacks["stop"] = stop_failure
    runtime.helper.bind_runner(runtime.Runner)
    with pytest.raises(ValueError) as caught:
        runner.execute_model(scheduler(), 1, video_grid=None)
    assert caught.value is original
    data = json.loads(getattr(runner, runtime.helper._MARKER).manifest.read_text())
    assert [e["phase"] for e in data["errors"]] == ["forward", "profiler.stop"]


def test_install_remove_keeps_forward_and_edits_added_after_install(helper, tmp_path):
    source = "def forward(x, *, video_grid):\n    return (x, video_grid)\n"
    extension = tmp_path / "worker_extension.py"
    extension.write_text(source, encoding="utf-8")
    helper.edit_hook(extension)
    installed = extension.read_text(encoding="utf-8")
    helper.edit_hook(extension)
    assert extension.read_text(encoding="utf-8") == installed
    assert len(list(tmp_path.glob("*.profile-backup-*"))) == 1
    with extension.open("a", encoding="utf-8") as stream:
        stream.write("\nNEW_VIDEO_FIX = True\n")
    helper.edit_hook(extension, remove=True)
    assert extension.read_text(encoding="utf-8") == source + "\nNEW_VIDEO_FIX = True\n"
    assert len(list(tmp_path.glob("*.profile-backup-*"))) == 2


def test_unrecognized_hook_is_not_overwritten(helper, tmp_path):
    extension = tmp_path / "worker_extension.py"
    original = "# huanyu-forward-profile:begin\nNEWER_VERSION = True\n"
    extension.write_text(original, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected existing profiler hook"):
        helper.edit_hook(extension)
    assert extension.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.profile-backup-*"))


@pytest.mark.parametrize("prefix", ["visual", "vpm", "apm", "audio_encoder", "tts", "talker", "code2wav"])
def test_media_module_is_not_misclassified_as_language_attention(helper, prefix):
    assert helper._category(f"model.{prefix}.layers.0.self_attn", SimpleNamespace()) is None
    assert helper._category(f"model.{prefix}.layers.0.linear_attn", SimpleNamespace()) is None


def test_scheduler_never_reads_tensor_values(helper):
    class DeviceScalar:
        def __int__(self):
            raise AssertionError("device readback")

        def __bool__(self):
            raise AssertionError("device readback")

    data = SimpleNamespace(
        total_num_scheduled_tokens=DeviceScalar(), num_scheduled_tokens={"r": DeviceScalar()},
        scheduled_encoder_inputs={},
    )
    assert helper._scheduler(data) == {"scheduled_tokens": None, "scheduled_requests": 1, "encoder_requests": 0}


def test_disabled_install_does_not_import_native_stack(helper, monkeypatch):
    monkeypatch.delenv("HUANYU_FORWARD_PROFILE_DIR", raising=False)

    def forbidden_import(*args, **kwargs):
        raise AssertionError("native stack imported while disabled")

    monkeypatch.setattr(helper.importlib, "import_module", forbidden_import)
    helper.install()


@pytest.fixture
def completed_capture(runtime, monkeypatch):
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_START", "1")
    runner = runtime.new_runner()
    runtime.helper.bind_runner(runtime.Runner)
    for _ in range(3):
        runner.execute_model(scheduler(), 1, video_grid=None)
    runtime.state = getattr(runner, runtime.helper._MARKER)
    runtime.trace = Path(runtime.state.trace_dir)
    runtime.parse_calls = []
    api = ModuleType("torch_npu.profiler.profiler")

    def analyse(path, **kwargs):
        runtime.parse_calls.append((path, kwargs))
        _parsed_files(runtime.helper, Path(path))

    api.analyse = analyse
    # No real torch or NPU dependency even if this test runs on an NPU host.
    for name in ("torch_npu", "torch_npu.profiler"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "torch_npu.profiler.profiler", api)
    runtime.api = api
    return runtime


def _parsed_files(helper, trace, *, forward=True, kernels=True):
    timeline, kernel_path = helper._output_paths(trace)
    timeline.parent.mkdir(exist_ok=True)
    timeline.write_text(json.dumps({"traceEvents": [
        {"name": "huanyu/forward" if forward else "other", "ph": "X", "dur": 2},
    ]}), encoding="utf-8")
    kernel_path.write_text("Name,Duration(us)\n" + ("MatMul,1.2\n" if kernels else ""), encoding="utf-8")


def test_analysis_checks_artifacts_and_runs_one_leaf_at_a_time(completed_capture):
    r = completed_capture
    result = r.helper.analyse_captures(r.out, expected_workers=1)
    assert result["status"] == "PASS"
    assert result["results"][0]["forward_ranges"] == 1
    assert result["results"][0]["npu_kernels"] == 1
    assert r.parse_calls == [(str(r.trace), {"max_process_number": 1, "export_type": "text"})]
    assert r.helper.main(["verify", "--profile-dir", str(r.out), "--expected-workers", "1"]) == 0
    assert len(r.parse_calls) == 1  # verify does not rerun the parser


@pytest.mark.parametrize("mode", ["silent_failure", "stale_outputs", "no_kernels", "no_forward", "bad_json"])
def test_parser_return_is_not_sufficient_for_success(completed_capture, mode):
    r = completed_capture
    if mode == "stale_outputs":
        _parsed_files(r.helper, r.trace)

    def failing_parser(path, **kwargs):
        if mode in ("silent_failure", "stale_outputs"):
            return  # torch_npu may log a failure without raising an exception
        _parsed_files(r.helper, r.trace, forward=mode != "no_forward", kernels=mode != "no_kernels")
        if mode == "bad_json":
            r.helper._output_paths(r.trace)[0].write_text('{"truncated":')

    r.api.analyse = failing_parser
    assert r.helper.main(["analyse", "--profile-dir", str(r.out)]) == 2
    report = json.loads((r.out / "forward_profile_analysis.json").read_text())
    assert report["status"] == "FAIL"
    assert report["errors"]


@pytest.mark.parametrize("missing", ["metadata", "device_data", "framework", "worker"])
def test_analysis_rejects_incomplete_inputs_before_import_or_parse(completed_capture, missing):
    r = completed_capture
    if missing == "metadata":
        (r.trace / "profiler_info_0.json").unlink()
    elif missing == "device_data":
        (r.trace / "PROF_0001/device_0/data/aicore.data").unlink()
    elif missing == "framework":
        (r.trace / "FRAMEWORK/torch.op_range").write_bytes(b"")
    args = ["analyse", "--profile-dir", str(r.out), "--expected-workers", "2" if missing == "worker" else "1"]
    assert r.helper.main(args) == 2
    assert not r.parse_calls


def test_atomic_progress_retains_last_snapshot_if_replace_fails(helper, tmp_path, monkeypatch):
    target = tmp_path / "capture.json"
    helper._atomic_json(target, {"phase": "forward"})

    def fail_replace(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(helper.os, "replace", fail_replace)
    with pytest.raises(OSError):
        helper._atomic_json(target, {"phase": "profiler.step"})
    assert json.loads(target.read_text()) == {"phase": "forward"}
