"""No NPU/model/Ray imports: preserve live execution and audit profile coverage."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "serve_forward_profile.py").is_file():
    ROOT = Path(__file__).resolve().parents[3] / "verl_omni_ext/models/qwen3_6_huanyu"


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_test", ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    helper = load("serve_forward_profile")
    events, scopes = [], []
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_ACTIVE", "2")
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_START", "1")
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_ARM_FILE", str(tmp_path / "arm"))
    for key in ("HUANYU_FORWARD_PROFILE_DIR", "VLLM_TORCH_PROFILER_DIR"):
        monkeypatch.delenv(key, raising=False)

    class Module:
        def __init__(self, name):
            self.name = name

        def forward(self, value, *, video_grid):
            events.append((self.name, value, video_grid))
            return value

    class Model(Module):
        def __init__(self):
            super().__init__("root")
            self.children = {p: Module(p) for p in (
                "llm.layers.0.linear_attn", "llm.layers.1.self_attn", "llm.layers.1.mlp",
                "llm.norm", "llm.embed_tokens", "llm.rotary_emb", "audio_encoder.conv",
                "visual.block", "lm_head", "new_unclassified_fusion")}

        def named_modules(self):
            return [("", self), *self.children.items()]

        def forward(self, value, *, video_grid):
            for child in self.children.values():
                child.forward(value, video_grid=video_grid)
            return value

    class Runner:
        model_config = SimpleNamespace(model="/models/2748", enforce_eager=True)

        def __init__(self):
            self.model = Model()
            self.result = object()
            self.failure = False

        def _model_forward(self, value, *, video_grid):
            self.model.forward(value, video_grid=video_grid)
            if self.failure:
                raise LookupError("live forward failed")
            return self.result

        def execute_model(self, scheduler_output, value, *, video_grid):
            return self._model_forward(value, video_grid=video_grid)

        def sample_tokens(self):
            events.append("sample")
            return self.result

    class Profiler:
        prof_if = SimpleNamespace(prof_path="")
        failure = False

        def start(self):
            events.append("start")
            trace = tmp_path / "trace"
            (trace / "FRAMEWORK").mkdir(parents=True)
            (trace / "FRAMEWORK/torch.op_range").write_bytes(b"data")
            (trace / "PROF_1/device_0/data").mkdir(parents=True)
            (trace / "PROF_1/device_0/data/kernels").write_bytes(b"data")
            self.prof_if = SimpleNamespace(prof_path=str(trace))

        def stop(self):
            events.append("stop")
            if self.failure:
                raise RuntimeError("native finalization failed")
            (Path(self.prof_if.prof_path) / "profiler_info_0.json").write_text(json.dumps({
                "config": {"mock": 1}, "start_info": {"mock": 1}, "end_info": {"mock": 1}}))

    @contextmanager
    def scope(name):
        scopes.append(name)
        yield

    profiler = Profiler()
    npu = SimpleNamespace(synchronize=lambda: events.append("sync"))
    monkeypatch.setattr(helper, "make_profiler", lambda *a: (profiler, scope, npu))
    monkeypatch.setattr(helper, "identity", lambda runner: dict(
        pid=123, tp_rank=0, tp_ranks=[0], parent_pid=12, host="test", model="/models/2748"))
    helper.bind_runner(Runner)
    return SimpleNamespace(h=helper, Runner=Runner, profiler=profiler, scopes=scopes, events=events,
                           root=tmp_path, arm=tmp_path / "arm", npu=npu)


def run(r, runner, value=1):
    return runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=16), value, video_grid="new-video-grid")


def test_all_modules_and_sample_are_retained_without_added_sync(runtime):
    r = runtime
    runner = r.Runner()
    # Simulate a repair installed after class construction: retain the live method.
    original = runner._model_forward
    runner._model_forward = lambda value, **kw: original(value + 7, **kw)
    live = runner._model_forward
    assert run(r, runner) is runner.result  # Not armed: warmup untouched.
    assert "start" not in r.events
    r.arm.touch()
    for _ in range(2):
        assert run(r, runner, 5) is runner.result
        assert runner.sample_tokens() is runner.result
    assert "stop" not in r.events  # Preserve sample after final forward.
    assert run(r, runner, 6) is runner.result
    assert r.events.count("stop") == 1
    assert "sync" not in r.events
    assert runner._model_forward is live
    state = getattr(runner, r.h.MARKER)
    assert state.status == "complete"
    assert state.completed == 2
    assert any("new_unclassified_fusion" in n for n in r.scopes)
    for path in runner.model.children:
        assert any(path in n for n in r.scopes)
    assert "serve/runner/sample_tokens" in r.scopes
    assert ("new_unclassified_fusion", 12, "new-video-grid") in r.events
    r.h.captures(r.root, expected=1, complete=True)


def test_non_eager_and_nested_profiler_rejected(runtime, monkeypatch):
    r = runtime
    runner = r.Runner()
    runner.model_config = SimpleNamespace(model="x", enforce_eager=False)
    with pytest.raises(RuntimeError, match="enforce_eager"):
        run(r, runner)
    runner.model_config.enforce_eager = True
    monkeypatch.setenv("HUANYU_FORWARD_PROFILE_DIR", "/other")
    with pytest.raises(RuntimeError, match="other profilers"):
        run(r, runner)


def test_audio_capture_requires_executed_encoder_and_projector(runtime):
    r = runtime
    runner = r.Runner()
    r.arm.touch()
    for _ in range(3):
        run(r, runner)
    # Runtime fixture has an audio encoder, but no projector.
    with pytest.raises(RuntimeError, match="No audio projector calls captured"):
        r.h.captures(r.root, expected=1, complete=True, require_audio=True)
    state = getattr(runner, r.h.MARKER)
    data = json.loads(state.manifest.read_text())
    data["hits"]["serve/module/Linear/llm.audio_projection_layer::Projector/forward"] = 1
    state.manifest.write_text(json.dumps(data))
    r.h.captures(r.root, expected=1, complete=True, require_audio=True)
    data["hits"] = {k: v for k, v in data["hits"].items() if "audio_encoder" not in k}
    state.manifest.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="No audio encoder calls captured"):
        r.h.captures(r.root, expected=1, complete=True, require_audio=True)


def test_original_exception_survives_stop_failure_and_restores_all_wrappers(runtime):
    r = runtime
    runner = r.Runner()
    original = runner.model.forward
    r.arm.touch()
    runner.failure, r.profiler.failure = True, True
    with pytest.raises(LookupError, match="live forward failed"):
        run(r, runner)
    state = getattr(runner, r.h.MARKER)
    assert state.done and state.status == "incomplete"
    assert runner.model.forward == original
    with pytest.raises(RuntimeError, match="incomplete"):
        r.h.captures(r.root, 1, complete=True)


def test_disabled_wrapper_is_inert_and_install_is_idempotent(runtime, monkeypatch):
    r = runtime
    bound = r.Runner.execute_model
    r.h.bind_runner(r.Runner)
    assert r.Runner.execute_model is bound
    monkeypatch.delenv("HUANYU_SERVE_PROFILE_DIR")
    runner = r.Runner()
    assert run(r, runner) is runner.result
    assert not hasattr(runner, r.h.MARKER)
    assert not list(r.root.glob("serve*.json"))


def test_entry_preserves_current_source_and_later_edits(tmp_path, monkeypatch):
    h = load("serve_forward_profile")
    path = tmp_path / "runner.py"
    before = "class NPUARModelRunner:\n    def execute_model(self, scheduler_output):\n        return scheduler_output\n"
    path.write_text(before)
    h.edit_entry(path)
    once = path.read_bytes()
    h.edit_entry(path)
    assert path.read_bytes() == once
    monkeypatch.delenv("HUANYU_SERVE_PROFILE_DIR", raising=False)
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert namespace["NPUARModelRunner"]().execute_model(42) == 42
    path.write_text(path.read_text() + "# later video repair\n")
    h.edit_entry(path, remove=True)
    assert path.read_text() == before + "# later video repair\n"
    assert list(tmp_path.glob("*.serve-profile-backup-*"))


def test_unknown_kernel_and_zero_duration_rows_are_included(tmp_path):
    report = load("serve_profile_report")
    p = tmp_path / "kernels.csv"
    p.write_text("Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                 "new_fused_op,unknown,MIX_AIC,10,5\n"
                 "hcom_allReduce_,HCCL,COMMUNICATION,11,8\n"
                 "new_fused_op,unknown,MIX_AIC,20,0\n")
    rows, summary = report.kernel_report(p)
    assert summary["rows"] == 3
    assert summary["zero_duration_rows"] == 1
    assert summary["total_task_duration_us"] == 13
    assert summary["device_task_interval_union_us"] == 9  # Overlap not double counted.
    assert sum(r["count"] for r in rows) == 3
    assert {r["name"] for r in rows} == {"new_fused_op", "hcom_allReduce_"}
    p.write_text(p.read_text() + "bad,unknown,NPU,21,garbage\n")
    with pytest.raises(ValueError, match="no rows silently dropped"):
        report.kernel_report(p)


def test_all_trace_events_retained_but_parent_child_cpu_not_double_counted(tmp_path):
    report = load("serve_profile_report")
    p = tmp_path / "trace.json"
    event = lambda name, ts, dur: dict(ph="X", pid=1, tid=2, name=name, ts=ts, dur=dur, cat="test")
    p.write_text(json.dumps({"traceEvents": [
        event("serve/execute/1", 0, 100), event("serve/module/Other/root/forward", 10, 80),
        event("serve/module/Norm/norm/forward", 20, 30), event("aten::new_op", 21, 10),
        dict(ph="s", id=3, pid=1, tid=2, ts=22),
    ]}))
    host, all_ranges, summary = report.trace_report(p)
    values = {r["name"]: r for r in host}
    assert values["serve/execute/1"]["self_cpu_us"] == 20
    assert values["serve/module/Other/root/forward"]["self_cpu_us"] == 50
    assert values["serve/module/Norm/norm/forward"]["self_cpu_us"] == 30
    assert len(all_ranges) == 4 and summary["metadata_flow_or_other_events"] == 1


def test_profiler_settings_capture_both_devices_and_runtime_without_online_parser(monkeypatch, tmp_path):
    h = load("serve_forward_profile")
    received = {}
    api = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
        ProfilerLevel=SimpleNamespace(Level2=2), AiCMetrics=SimpleNamespace(AiCoreNone="none"),
        _ExperimentalConfig=lambda **kw: kw,
        tensorboard_trace_handler=lambda *a, **kw: (a, kw),
        profile=lambda **kw: received.update(kw) or object())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=api))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(record_function=object()), npu=object()))
    h.make_profiler(tmp_path, "tp0", 2, False, True)
    assert received["activities"] == ["CPU", "NPU"]
    assert received["with_stack"] is True
    assert received["experimental_config"]["data_simplification"] is False
    assert received["on_trace_ready"][1]["analyse_flag"] is False
    assert "schedule" not in received


def test_prepare_creates_only_eager_preserves_capacity_and_original(tmp_path):
    h = load("serve_forward_profile")
    import yaml
    source, out = tmp_path / "source.yaml", tmp_path / "eager.yaml"
    config = dict(pipeline="minicpmo_5_0_thinker_only", enable_prefix_caching=True,
                  stages=[dict(stage_id=0, enforce_eager=False, max_num_seqs=16)],
                  platforms=dict(npu=dict(stages=[dict(stage_id=0, tensor_parallel_size=2,
                                                     devices="0,1", gpu_memory_utilization=0.9)])))
    source.write_text(yaml.safe_dump(config))
    before = source.read_bytes()
    h.prepare_eager(source, out)
    actual = yaml.safe_load(out.read_text())
    for stage in (actual["stages"][0], actual["platforms"]["npu"]["stages"][0]):
        assert stage["enforce_eager"] is True
        assert stage["compilation_config"]["cudagraph_mode"] == "NONE"
        assert stage["enable_prefix_caching"] is False
    assert actual["platforms"]["npu"]["stages"][0]["tensor_parallel_size"] == 2
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        h.prepare_eager(source, out)


def test_native_stop_return_without_metadata_is_not_success(runtime, monkeypatch):
    r = runtime
    runner = r.Runner()
    r.arm.touch()
    monkeypatch.setattr(r.profiler, "stop", lambda: r.events.append("suppressed_native_error"))
    for _ in range(3):
        run(r, runner)
    assert getattr(runner, r.h.MARKER).status == "incomplete"
    with pytest.raises(RuntimeError, match="profiler_info"):
        r.h.captures(r.root, 1, complete=True)


@pytest.mark.parametrize("missing_call", [False, True])
def test_full_report_keeps_unknown_ops_and_audits_exact_module_call_counts(tmp_path, missing_call):
    report = load("serve_profile_report")
    trace = tmp_path / "trace"
    output = trace / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    label = "serve/module/Other/new_layer/forward"
    events = [dict(name=label, cat="cpu_op", pid=5, tid=6, ph="X", ts=1, dur=2)]
    (output / "trace_view.json").write_text(json.dumps(events))
    (output / "kernel_details.csv").write_text("Name,Type,Duration(us)\nunknown_custom_kernel,UNKNOWN,3\n")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(dict(targets=[dict(label=label, wrapped=True)])))
    manifest = dict(status="complete", errors=[], active=1, completed=1, trace_dir=str(trace),
                    inventory=str(inventory), identity=dict(tp_rank=0), hits={label: 2 if missing_call else 1})
    if missing_call:
        with pytest.raises(ValueError, match="missing in parsed trace"):
            report.report_capture(manifest)
    else:
        result = report.report_capture(manifest)
        assert result["all_exported_kernel_rows_accounted_for"] is True
    summary = json.loads((trace / "serve_summary/summary.json").read_text())
    assert summary["device"]["rows"] == 1
    assert bool(summary["missing_executed_module_ranges"]) is missing_call
    assert "unknown_custom_kernel" in (trace / "serve_summary/all_device_ops.csv").read_text(encoding="utf-8-sig")


def test_active_conditional_entry_loads_standalone_module_without_training_import(tmp_path, monkeypatch):
    h = load("serve_forward_profile")
    path = tmp_path / "runner.py"
    path.write_text("class NPUARModelRunner:\n    def execute_model(self, scheduler_output):\n        return scheduler_output\n")
    h.edit_entry(path)
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_DIR", str(tmp_path))
    existing = {n for n in sys.modules if n.startswith(("verl", "vllm", "torch"))}
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert getattr(namespace["NPUARModelRunner"].execute_model, h.MARKER) is True
    assert {n for n in sys.modules if n.startswith(("verl", "vllm", "torch"))} == existing


def test_start_counts_only_real_work_after_arming(runtime, monkeypatch):
    r = runtime
    monkeypatch.setenv("HUANYU_SERVE_PROFILE_START", "2")
    runner = r.Runner()
    run(r, runner)
    r.arm.touch()
    runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0), 2, video_grid="empty")
    run(r, runner)
    assert "start" not in r.events
    for _ in range(3):
        run(r, runner)
    state = getattr(runner, r.h.MARKER)
    assert [row["iteration"] for row in state.rows] == [2, 3]
    assert state.status == "complete"
