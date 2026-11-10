"""Opt-in, bounded NPU traces of the *installed* Huanyu rollout implementation.

This file can be copied on its own. ``python forward_profile.py install`` adds
an opt-in import to the sibling worker_extension.py; it never replaces forward.
No torch/NPU imports or runtime patches occur when profiling is disabled.
"""

import argparse
import atexit
from collections import Counter
from contextlib import contextmanager, nullcontext
import csv
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import uuid


_MARKER = "_huanyu_forward_profiler"
_HOOK = '''\n# huanyu-forward-profile:begin
import os as _huanyu_profile_os
if _huanyu_profile_os.environ.get("HUANYU_FORWARD_PROFILE_DIR"):
    from .forward_profile import install as _install_huanyu_forward_profile
    _install_huanyu_forward_profile()
# huanyu-forward-profile:end
'''


def _atomic_json(path, data):
    # Readers must never see a half-written snapshot. No background NPU thread
    # or extra device synchronization is needed for progress reporting.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _raw_errors(trace):
    if trace is None or not trace.is_dir():
        return ["No trace directory for this capture"]
    errors = []
    info_files = [p for p in trace.glob("profiler_info*.json")
                  if re.fullmatch(r"profiler_info(?:_\d+)?\.json", p.name)]
    if not info_files:
        errors.append("Missing profiler_info*.json (finalization did not finish)")
    for info in info_files:
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(not data.get(k) for k in ("config", "start_info", "end_info")):
                errors.append(f"Incomplete profiler metadata: {info.name}")
        except (OSError, ValueError) as error:
            errors.append(f"Unreadable {info.name}: {error}")
    if not any(p.is_file() and p.stat().st_size for p in (trace / "FRAMEWORK").rglob("*")):
        errors.append("Missing nonempty FRAMEWORK data")
    if not any(p.is_file() and p.stat().st_size for d in trace.glob("PROF_*")
               for p in d.glob("device_*/data/*")):
        errors.append("Missing nonempty CANN device raw data (PROF_*/device_*/data/*)")
    return errors


def _positive(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _source_chain(method):
    records, seen = [], set()
    while callable(method) and id(method) not in seen:
        seen.add(id(method))
        code = getattr(method, "__code__", None)
        if code is not None:
            path = Path(code.co_filename)
            records.append({
                "file": str(path), "line": code.co_firstlineno,
                "qualname": getattr(method, "__qualname__", ""),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            })
        method = getattr(method, "__wrapped__", None)
    return records


def _model_path(runner):
    cfg = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    if cfg is None:
        cfg = getattr(runner, "model_config", None)
    return str(getattr(cfg, "model", ""))


def _category(path, module):
    parts = set(path.lower().split("."))
    if parts & {"visual", "vpm", "apm", "audio_encoder", "tts", "talker", "code2wav"}:
        return None
    leaf = path.rsplit(".", 1)[-1].lower()
    cls = type(module).__name__.lower()
    if leaf == "linear_attn" or any(x in cls for x in ("gateddelta", "gated_delta", "deltanet")):
        return "GDN"
    if leaf in {"self_attn", "self_attention"}:
        return "Attention"
    if leaf == "mlp":
        return "MoE" if "moe" in cls or hasattr(module, "experts") else "MLP"
    return None


def _scheduler(scheduler):
    scheduled = getattr(scheduler, "num_scheduled_tokens", {})
    total = getattr(scheduler, "total_num_scheduled_tokens", None)
    if type(total) is not int and isinstance(scheduled, dict) and all(type(v) is int for v in scheduled.values()):
        total = sum(scheduled.values())
    return {
        "scheduled_tokens": total if type(total) is int else None,
        "scheduled_requests": len(scheduled) if isinstance(scheduled, dict) else None,
        "encoder_requests": len(getattr(scheduler, "scheduled_encoder_inputs", {}) or {}),
    }


def _identity(runner):
    import torch

    identity = {
        "host": socket.gethostname(), "pid": os.getpid(), "parent_pid": os.getppid(),
        "model": _model_path(runner), "device": str(getattr(runner, "device", "")),
        "torch_version": str(torch.__version__),
        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else None,
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("MINICPMO_", "HUANYU_", "HCCL_"))
                        or k in {"ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "MASTER_ADDR", "MASTER_PORT"}},
    }
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        identity.update(tp_rank=group.rank_in_group, tp_ranks=list(group.ranks))
    except (ImportError, AttributeError, AssertionError):
        identity["tp_rank"] = None
    return identity


def _make_profiler(directory, tag, active, level):
    import torch
    import torch_npu

    api = torch_npu.profiler
    profile = api.profile(
        activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
        schedule=api.schedule(wait=0, warmup=1, active=active, repeat=1),
        on_trace_ready=api.tensorboard_trace_handler(str(directory), worker_name=tag, analyse_flag=False),
        record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
        experimental_config=api._ExperimentalConfig(profiler_level=getattr(api.ProfilerLevel, f"Level{level}")),
    )
    return profile, torch.profiler.record_function, torch.npu, str(torch_npu.__version__)


class _Capture:
    def __init__(self, runner):
        self.runner = runner
        self.start_at = _positive("HUANYU_FORWARD_PROFILE_START", "101")
        self.active_steps = _positive("HUANYU_FORWARD_PROFILE_ACTIVE", "30")
        self.level = _positive("HUANYU_FORWARD_PROFILE_LEVEL", "1")
        if self.level not in (1, 2):
            raise ValueError("HUANYU_FORWARD_PROFILE_LEVEL must be 1 or 2")
        self.iteration = 0
        self.steps = 0
        self.done = False
        self.profiler = None
        self.patches = []
        self.targets = []
        self.hits = Counter()
        self.rows = []
        self.row = None
        self.status = "starting"
        self.phases = []
        self.phase_started_at = time.time()
        self.last_completed_phase = None
        self.forwards_completed = 0
        self.errors = []
        self.trace_dir = None
        self.raw_errors = []

    @contextmanager
    def phase(self, name):
        self.phases.append((name, time.time()))
        self.write()
        started = time.perf_counter()
        try:
            yield
        except BaseException as error:
            # Keep the innermost failing boundary even after cleanup enters stop.
            if not self.errors or self.errors[-1].get("message") != str(error):
                self.errors.append({"phase": name, "type": type(error).__name__, "message": str(error)})
            raise
        else:
            self.last_completed_phase = {"phase": name, "elapsed_s": time.perf_counter() - started,
                                         "engine_iteration": self.iteration, "at": time.time()}
        finally:
            self.phases.pop()
            self.phase_started_at = time.time()
            self.write()

    def wrap(self, obj, method, label, metadata=False):
        original = getattr(obj, method, None)
        if not callable(original):
            return
        if inspect.iscoroutinefunction(original) or inspect.isgeneratorfunction(original):
            return
        self.targets.append({"label": label, "source_chain": _source_chain(original)})
        previous = (method in vars(obj), vars(obj).get(method))

        @wraps(original)
        def measured(*args, **kwargs):
            if self.profiler is None or self.done:
                return original(*args, **kwargs)
            self.hits[label] += 1
            if metadata:
                self.attention_metadata()
            is_forward = label == "huanyu/forward"
            # A final profiler.step() can itself stop/finalize the trace. Show
            # a barrier inside step/stop separately, without adding a barrier.
            profiler_sync = label == "huanyu/explicit_npu_synchronize" and any(
                name.startswith("profiler.") for name, _ in self.phases)
            boundary = "profiler.synchronize" if profiler_sync else "forward"
            with self.phase(boundary) if is_forward or profiler_sync else nullcontext():
                with nullcontext() if profiler_sync else self.scope(label):
                    result = original(*args, **kwargs)
                if is_forward:
                    self.forwards_completed += 1
                return result

        setattr(obj, method, measured)
        self.patches.append((obj, method, previous))

    def attention_metadata(self):
        # Only Python integers; never read an NPU tensor to classify a batch.
        if self.row is None or self.row.get("attention_metadata"):
            return
        try:
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
        except (ImportError, AssertionError, AttributeError, RuntimeError):
            return
        values = metadata.values() if isinstance(metadata, dict) else [metadata]
        rows = []
        for value in values:
            fields = {key: getattr(value, key, None) for key in ("num_prefills", "num_decodes", "num_prefill_tokens", "num_decode_tokens")}
            fields = {k: v for k, v in fields.items() if type(v) is int}
            if fields and fields not in rows:
                rows.append(fields)
        self.row["attention_metadata"] = rows

    def begin(self):
        identity = _identity(self.runner)
        group = re.sub(r"[^A-Za-z0-9_-]", "_", f"{identity['host']}_parent{identity['parent_pid']}")
        self.directory = Path(os.environ["HUANYU_FORWARD_PROFILE_DIR"]) / group
        self.directory.mkdir(parents=True, exist_ok=True)
        tag = f"pid{identity['pid']}_tp{identity.get('tp_rank')}_t{time.time_ns()}"
        self.manifest = self.directory / f"{tag}.json"
        self.identity = identity
        self.execute_sources = _source_chain(type(self.runner).execute_model)
        with self.phase("profiler.create"):
            self.profiler, self.scope, npu, version = _make_profiler(self.directory, tag, self.active_steps, self.level)
        self.identity["torch_npu_version"] = version
        self.wrap(self.runner, "_model_forward", "huanyu/forward", metadata=True)
        if not callable(getattr(self.runner, "_model_forward", None)):
            self.wrap(self.runner.model, "forward", "huanyu/forward", metadata=True)
        for path, module in self.runner.model.named_modules():
            kind = _category(path, module)
            if kind:
                prefix = f"huanyu/{kind}/{path}:{type(module).__name__}"
                self.wrap(module, "forward", prefix, metadata=(kind == "GDN"))
                if kind == "GDN":
                    self.wrap(module, "_forward_core", prefix + "/core", metadata=True)
        # Labels existing explicit barriers; NEVER adds a device synchronize.
        self.wrap(npu, "synchronize", "huanyu/explicit_npu_synchronize")
        print(f"[huanyu-forward-profile] START manifest={self.manifest}", flush=True)
        atexit.register(self.at_exit)
        with self.phase("profiler.start"):
            self.profiler.start()
        self.status = "capturing"
        self.write()

    def at_exit(self):
        if self.profiler is not None and not self.done:
            try:
                self.finish("incomplete")
            except Exception as error:
                print(f"[huanyu-forward-profile] exit cleanup failed: {error}", flush=True)

    def write(self):
        if not hasattr(self, "manifest"):
            return
        native_path = getattr(getattr(self.profiler, "prof_if", None), "prof_path", None)
        if native_path:
            self.trace_dir = str(native_path)
        now = time.time()
        phase, entered = self.phases[-1] if self.phases else (
            "waiting_for_execute" if not self.done else "finished", self.phase_started_at)
        _atomic_json(self.manifest, {
            "schema_version": 2, "identity": self.identity, "status": self.status, "warmup": 1,
            "active": self.active_steps, "level": self.level,
            "phase": phase, "phase_stack": [name for name, _ in self.phases],
            "phase_started_at": entered, "updated_at": now,
            "engine_iteration": self.iteration, "forwards_completed": self.forwards_completed,
            "steps_completed": self.steps, "expected_steps": self.active_steps + 1,
            "current_iteration": self.row, "last_completed_phase": self.last_completed_phase,
            "trace_dir": self.trace_dir, "errors": self.errors, "raw_errors": self.raw_errors,
            "execute_source_chain": self.execute_sources,
            "targets": self.targets, "hits": dict(self.hits), "iterations": self.rows,
            "note": "CPU ranges are inclusive; correlate device events. parent_pid grouping needs TP-rank verification.",
        })

    def finish(self, status):
        if self.done:
            return
        self.status = "stopping"
        atexit.unregister(self.at_exit)
        try:
            if self.profiler is not None:
                with self.phase("profiler.stop"):
                    self.profiler.stop()
            with self.phase("capture.verify"):
                self.raw_errors = _raw_errors(Path(self.trace_dir) if self.trace_dir else None)
            self.status = status if not self.raw_errors and not self.errors else "incomplete"
        except BaseException:
            self.status = "incomplete"
            raise
        finally:
            for obj, name, (existed, value) in reversed(self.patches):
                if existed:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
            self.patches.clear()
            self.profiler = None
            self.done = True
            if hasattr(self, "manifest"):
                self.write()
                print(f"[huanyu-forward-profile] {self.status.upper()} steps={self.steps}/{self.active_steps + 1} "
                      f"manifest={self.manifest} raw_errors={self.raw_errors} errors={self.errors}", flush=True)

    def call(self, original, args, kwargs, scheduler):
        if self.done:
            return original(self.runner, *args, **kwargs)
        row = _scheduler(scheduler)
        # Do not count dummy/profile_run invocations without scheduled tokens.
        if not row["scheduled_tokens"]:
            return original(self.runner, *args, **kwargs)
        self.iteration += 1
        if self.iteration < self.start_at:
            return original(self.runner, *args, **kwargs)
        self.row = {"engine_iteration": self.iteration, "capture_index": self.steps,
                    "warmup": self.steps == 0, **row}
        try:
            if self.profiler is None:
                self.begin()
            with self.phase("execute_model"):
                started = time.perf_counter()
                with self.scope(f"huanyu/execute/iteration_{self.iteration}"):
                    result = original(self.runner, *args, **kwargs)
                self.rows.append({**self.row, "host_seconds": time.perf_counter() - started})
                self.row = None
            with self.phase("profiler.step"):
                self.profiler.step()
                self.steps += 1
            print(f"[huanyu-forward-profile] PROGRESS pid={self.identity['pid']} iteration={self.iteration} "
                  f"forward={self.forwards_completed} step={self.steps}/{self.active_steps + 1}", flush=True)
            if self.steps == self.active_steps + 1:
                self.finish("complete")
            return result
        except BaseException:
            if self.row is not None:
                self.rows.append({**self.row, "error": True})
                self.row = None
            try:
                self.finish("incomplete")
            except Exception as error:
                print(f"[huanyu-forward-profile] cleanup failed: {error}", flush=True)
            raise


def bind_runner(cls):
    original = cls.execute_model
    if getattr(original, _MARKER, False):
        return
    signature = inspect.signature(original)
    if "scheduler_output" not in signature.parameters:
        raise RuntimeError("Current execute_model has no scheduler_output; inspect current runner before profiling")

    @wraps(original)
    def execute(self, *args, **kwargs):
        expected = os.environ.get("HUANYU_FORWARD_PROFILE_MODEL", "2748")
        actual = _model_path(self).rstrip("/\\").replace("\\", "/").split("/")[-1]
        if not os.environ.get("HUANYU_FORWARD_PROFILE_DIR") or actual != expected:
            return original(self, *args, **kwargs)
        state = getattr(self, _MARKER, None)
        if state is None:
            state = _Capture(self)
            setattr(self, _MARKER, state)
        bound = signature.bind(self, *args, **kwargs)
        return state.call(original, args, kwargs, bound.arguments["scheduler_output"])

    setattr(execute, _MARKER, True)
    cls.execute_model = execute


def install():
    if os.environ.get("HUANYU_FORWARD_PROFILE_DIR"):
        module = importlib.import_module("vllm_omni.platforms.npu.worker.npu_ar_model_runner")
        bind_runner(module.NPUARModelRunner)


def edit_hook(path, remove=False):
    text = path.read_text(encoding="utf-8")
    if "# huanyu-forward-profile:begin" in text and _HOOK not in text:
        raise RuntimeError(f"Unexpected existing profiler hook; inspect {path} manually")
    updated = text.replace(_HOOK, "") if remove else text if _HOOK in text else text + _HOOK
    if updated == text:
        print("Hook already in requested state:", path)
        return
    compile(updated, str(path), "exec")
    backup = path.with_name(path.name + ".profile-backup-" + uuid.uuid4().hex[:10])
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8", newline="\n")
    print("Updated:", path, "\nBackup:", backup)


def _manifests(root):
    paths = [root] if root.is_file() else sorted(root.rglob("pid*_tp*_t*.json"))
    found = []
    for path in paths:
        if not re.fullmatch(r"pid\d+_tp(?:\d+|None)_t\d+\.json", path.name):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "identity" in data and "iterations" in data:
            found.append((path, data))
    if not found:
        raise RuntimeError(f"No capture manifests in {root}; check hook installation and capture START threshold")
    return found


def _trace_for(path, data):
    if data.get("trace_dir"):
        return Path(data["trace_dir"])
    # Support inspecting old captures; never borrow metadata from another rank.
    leaves = list(path.parent.glob(path.stem + "*_ascend_pt"))
    return leaves[0] if len(leaves) == 1 else None


def capture_status(root, expected_workers=None, require_complete=False):
    records = _manifests(root)
    problems = []
    if expected_workers is not None and len(records) != expected_workers:
        problems.append(f"Expected {expected_workers} worker manifests, found {len(records)}")
    for path, data in records:
        who = data["identity"]
        phase_age = max(0, time.time() - data.get("phase_started_at", path.stat().st_mtime))
        last = data.get("last_completed_phase") or {}
        print(f"{who.get('host')} pid={who.get('pid')} tp={who.get('tp_rank')} "
              f"status={data.get('status')} phase={data.get('phase', 'unknown_old_format')} "
              f"phase_age_s={phase_age:.1f} iteration={data.get('engine_iteration', '?')} "
              f"forward={data.get('forwards_completed', '?')} "
              f"step={data.get('steps_completed', '?')}/{data.get('expected_steps', '?')} "
              f"last={last.get('phase', '?')} stack={data.get('phase_stack', [])}", flush=True)
        if data.get("errors") or data.get("raw_errors"):
            print("  errors:", data.get("errors"), data.get("raw_errors"), flush=True)
        if require_complete:
            issues = _raw_errors(_trace_for(path, data))
            if data.get("status") != "complete":
                issues.insert(0, f"Capture status={data.get('status')}, phase={data.get('phase')}")
            if data.get("schema_version", 0) < 2:
                issues.append("Old capture format; recapture with the progress-enabled helper")
            if data.get("steps_completed", 0) != data.get("active", 0) + 1:
                issues.append("Warmup + active window not completed")
            if not data.get("hits", {}).get("huanyu/forward", 0):
                issues.append("No forward hook hits")
            if data.get("errors") or data.get("raw_errors"):
                issues.append("Capture recorded an error")
            problems.extend(f"{path.name}: {issue}" for issue in issues)
    if problems:
        raise RuntimeError("\n".join(problems))
    return records


def _output_paths(trace):
    out = trace / "ASCEND_PROFILER_OUTPUT"
    return out / "trace_view.json", out / "kernel_details.csv"


def _output_signature(trace):
    return [(p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None for p in _output_paths(trace)]


def _verify_outputs(trace):
    timeline, kernels = _output_paths(trace)
    with timeline.open(encoding="utf-8") as stream:
        data = json.load(stream)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    forward_count = sum(e.get("name") == "huanyu/forward" and e.get("ph") == "X" and e.get("dur", 0) > 0
                        for e in events if isinstance(e, dict))
    if not forward_count:
        raise RuntimeError(f"No timed huanyu/forward ranges in {timeline}")
    kernel_count = 0
    with kernels.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Name") and float(row.get("Duration(us)") or 0) > 0:
                kernel_count += 1
    if not kernel_count:
        raise RuntimeError(f"No timed NPU kernels in {kernels}")
    return {"trace_dir": str(trace), "forward_ranges": forward_count, "npu_kernels": kernel_count}


def analyse_captures(root, expected_workers=None, verify_only=False):
    records = capture_status(root, expected_workers, require_complete=True)
    if not verify_only:
        # Import only after raw-data validation. Status/check stay CPU-only.
        from torch_npu.profiler.profiler import analyse
    report = {"status": "running", "results": [], "errors": [],
              "note": "Checks forward ranges and NPU kernel output, not numerical correctness or all communication views."}
    report_path = (root.parent if root.is_file() else root) / "forward_profile_analysis.json"
    for manifest, data in records:
        trace = _trace_for(manifest, data)
        report.update(current_trace=str(trace), updated_at=time.time())
        _atomic_json(report_path, report)
        try:
            if not verify_only:
                before = _output_signature(trace)
                print(f"[huanyu-forward-profile] ANALYSE {trace}", flush=True)
                analyse(str(trace), max_process_number=1, export_type="text")
                after = _output_signature(trace)
                if any(old == new for old, new in zip(before, after)):
                    raise RuntimeError("Parser did not refresh trace_view.json and kernel_details.csv; inspect trace logs/")
            result = _verify_outputs(trace)
            report["results"].append(result)
            print(f"[huanyu-forward-profile] PARSE_PASS {json.dumps(result)}", flush=True)
        except Exception as error:
            report["errors"].append({"manifest": str(manifest), "error": f"{type(error).__name__}: {error}"})
            # Stop at the first bad rank instead of repeatedly parsing all ranks.
            report["status"] = "FAIL"
            _atomic_json(report_path, report)
            raise RuntimeError(f"Analysis failed: {trace}; see {report_path} and {trace / 'logs'}") from error
    report.update(status="PASS", current_trace=None, updated_at=time.time())
    _atomic_json(report_path, report)
    print(f"[huanyu-forward-profile] ANALYSIS_PASS workers={len(records)} report={report_path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall", "status", "check", "analyse", "verify"))
    parser.add_argument("--worker-extension", type=Path, default=Path(__file__).with_name("worker_extension.py"))
    parser.add_argument("--profile-dir", type=Path, help="Capture root, one worker group, or one worker manifest")
    parser.add_argument("--expected-workers", type=int, help="Fail if the number of captures differs")
    args = parser.parse_args(argv)
    if args.expected_workers is not None and args.expected_workers < 1:
        parser.error("--expected-workers must be positive")
    if args.action in ("install", "uninstall"):
        edit_hook(args.worker_extension, remove=args.action == "uninstall")
        return 0
    if args.profile_dir is None:
        parser.error("--profile-dir is required")
    try:
        if args.action in ("status", "check"):
            records = capture_status(args.profile_dir, args.expected_workers, require_complete=args.action == "check")
            if args.action == "check":
                print(f"[huanyu-forward-profile] CAPTURE_PASS workers={len(records)}", flush=True)
        else:
            analyse_captures(args.profile_dir, args.expected_workers, verify_only=args.action == "verify")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[huanyu-forward-profile] FAIL: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
