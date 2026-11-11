"""Standalone, opt-in eager NPU profiling. No training package is imported.

Install a conditional entry at the end of the *installed* NPU runner module.
Wrap live callables, never substitute a saved model/forward implementation.
Capture every TP worker, all named modules, and CPU+NPU profiler events.
"""

import argparse
import ast
import atexit
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid


BEGIN = "# huanyu-serve-profile:begin"
END = "# huanyu-serve-profile:end"
MARKER = "_huanyu_serve_capture"
PREFIX = "HUANYU_SERVE_PROFILE_"


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def positive(name, default):
    value = int(os.environ.get(PREFIX + name, default))
    if value < 1:
        raise ValueError(PREFIX + name + " must be positive")
    return value


def flag(name):
    value = os.environ.get(PREFIX + name, "0")
    if value not in {"0", "1"}:
        raise ValueError(PREFIX + name + " must be 0 or 1")
    return value == "1"


def model_config(runner):
    return (getattr(getattr(runner, "vllm_config", None), "model_config", None)
            or getattr(runner, "model_config", None))


def category(path, module):
    """Descriptive module labels only; never filter the actual profiler data."""
    text = (path + "." + type(module).__name__).lower()
    if any(x in path.lower().split(".") for x in ("visual", "vpm", "apm", "audio_encoder", "talker", "tts")):
        return "Media"
    if any(x in text for x in ("linear_attn", "gateddelta", "gated_delta", "deltanet")):
        return "GDN"
    if any(x in text for x in ("self_attn", "attention")):
        return "Attention"
    if "moe" in text or "expert" in text:
        return "MoE"
    if "norm" in text:
        return "Norm"
    if "embed" in text:
        return "Embedding"
    if "rotary" in text or "rope" in text:
        return "RoPE"
    if "lm_head" in text or "logits" in text:
        return "Logits"
    if "mlp" in text:
        return "MLP"
    if "linear" in text:
        return "Linear"
    return "Other"  # Unknown/new modules are instrumented too.


def identity(runner):
    import socket
    import torch
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    versions = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "vllm-omni"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    config = model_config(runner)
    return dict(host=socket.gethostname(), pid=os.getpid(), parent_pid=os.getppid(),
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else None,
                tp_rank=group.rank_in_group, tp_ranks=list(group.ranks),
                model=str(config.model), enforce_eager=config.enforce_eager,
                versions=versions, device=str(getattr(runner, "device", "")),
                environment={k: v for k, v in os.environ.items()
                             if k.startswith(("HUANYU_", "MINICPMO_", "HCCL_"))
                             or k in {"PYTHONPATH", "VLLM_OMNI_EXTERNAL_MODULES", "ASCEND_RT_VISIBLE_DEVICES",
                                      "ASCEND_LAUNCH_BLOCKING", "TORCH_COMPILE_DISABLE"}},
                training_packages_loaded=[k for k in ("verl", "verl_omni", "verl_omni_ext") if k in sys.modules])


def make_profiler(directory, tag, level, shapes, stack):
    import torch
    import torch_npu

    api = torch_npu.profiler
    # No scheduled final step, online parser, per-layer barriers or HW-counter
    # collection. stop still performs native finalization; it can fail/hang.
    profile = api.profile(
        activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
        on_trace_ready=api.tensorboard_trace_handler(str(directory), worker_name=tag, analyse_flag=False),
        record_shapes=shapes, with_stack=stack, profile_memory=False, with_modules=False,
        experimental_config=api._ExperimentalConfig(
            profiler_level=getattr(api.ProfilerLevel, f"Level{level}"),
            aic_metrics=api.AiCMetrics.AiCoreNone, data_simplification=False),
    )
    return profile, torch.profiler.record_function, torch.npu


def raw_errors(trace):
    if trace is None or not trace.is_dir():
        return ["No raw trace directory"]
    errors = []
    info = [p for p in trace.glob("profiler_info*.json")
            if re.fullmatch(r"profiler_info(?:_\d+)?\.json", p.name)]
    if not info:
        errors.append("Missing finalized profiler_info*.json")
    for path in info:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not all(data.get(k) for k in ("config", "start_info", "end_info")):
                errors.append(f"Incomplete metadata: {path}")
        except (OSError, ValueError, AttributeError) as exc:
            errors.append(f"Invalid metadata: {exc}")
    if not any(p.is_file() and p.stat().st_size for p in (trace / "FRAMEWORK").rglob("*")):
        errors.append("Missing FRAMEWORK data")
    if not any(p.stat().st_size for d in trace.glob("PROF_*") for p in d.glob("device_*/data/*") if p.is_file()):
        errors.append("Missing CANN device raw data")
    return errors


class Capture:
    def __init__(self, runner):
        if getattr(model_config(runner), "enforce_eager", None) is not True:
            raise RuntimeError("Serve profiling requires the effective stage enforce_eager=True")
        if os.environ.get("HUANYU_FORWARD_PROFILE_DIR") or os.environ.get("VLLM_TORCH_PROFILER_DIR"):
            raise RuntimeError("Disable other profilers before enabling HUANYU_SERVE_PROFILE_DIR")
        self.runner = runner
        self.start = positive("START", "1")
        self.active = positive("ACTIVE", "10")
        self.level = positive("LEVEL", "2")
        if self.level not in (1, 2):
            raise ValueError("PROFILE_LEVEL must be 1 or 2")
        self.shapes, self.stack = flag("SHAPES"), flag("STACK")
        self.arm = os.environ.get(PREFIX + "ARM_FILE")
        self.iteration = self.completed = 0
        self.current_row = None
        self.profiler = None
        self.done = False
        self.patches, self.targets, self.rows, self.errors = [], [], [], []
        self.hits = Counter()
        self.source_hashes = {}
        self.wrapped = set()
        self.phase, self.status = "waiting_for_arm", "waiting"
        self.trace_dir = None
        self.who = identity(runner)
        self.directory = Path(os.environ[PREFIX + "DIR"]).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tag = f"serve_pid{self.who['pid']}_tp{self.who['tp_rank']}_{time.time_ns()}"
        self.manifest = self.directory / (self.tag + ".json")
        self.inventory = self.directory / (self.tag + ".inventory.json")
        self.execute_source = self.sources(type(runner).execute_model)
        self.write()
        print(f"[huanyu-serve-profile] READY tp={self.who['tp_rank']} manifest={self.manifest}", flush=True)

    def sources(self, method):
        rows, seen = [], set()
        while callable(method) and id(method) not in seen:
            seen.add(id(method))
            code = getattr(method, "__code__", None)
            if code:
                path = Path(code.co_filename)
                key = str(path)
                if key not in self.source_hashes:
                    self.source_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                rows.append(dict(file=key, line=code.co_firstlineno, sha256=self.source_hashes[key]))
            method = getattr(method, "__wrapped__", None)
        return rows

    def write(self):
        native = getattr(getattr(self.profiler, "prof_if", None), "prof_path", None)
        if native:
            self.trace_dir = str(native)
        atomic_json(self.manifest, dict(
            schema_version=1, identity=self.who, status=self.status, phase=self.phase,
            updated_at=time.time(), start=self.start, active=self.active, level=self.level,
            record_shapes=self.shapes, with_stack=self.stack, arm_file=self.arm,
            engine_iteration=self.iteration, completed=self.completed, trace_dir=self.trace_dir,
            inventory=str(self.inventory), execute_source=self.execute_source, iterations=self.rows,
            errors=self.errors, hits=dict(self.hits) if self.done else None,
            note="Module ranges are CPU inclusive. Device tasks may overlap. No automatic module/kernel attribution."))

    @contextmanager
    def boundary(self, name):
        previous, self.phase = self.phase, name
        self.write()
        try:
            yield
        except BaseException as exc:
            self.errors.append(dict(phase=name, error=f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            self.phase = previous
            self.write()

    def wrap(self, obj, name, label):
        key = (id(obj), name)
        if key in self.wrapped:
            return
        original = getattr(obj, name, None)
        if not callable(original):
            return
        target = dict(label=label, source=self.sources(original))
        self.targets.append(target)
        if inspect.iscoroutinefunction(original) or inspect.isgeneratorfunction(original):
            target["wrapped"] = False
            target["reason"] = "async/generator: rely on native operator trace"
            return
        previous = (name in vars(obj), vars(obj).get(name))

        @wraps(original)
        def measured(*args, **kwargs):
            if self.profiler is None or self.done or self.status != "capturing":
                return original(*args, **kwargs)
            self.hits[label] += 1
            if name == "_model_forward" or label.endswith("/<root>::" + type(obj).__name__ + "/forward"):
                self.attention_metadata()
            with self.scope(label):
                return original(*args, **kwargs)

        setattr(obj, name, measured)
        target["wrapped"] = True
        self.wrapped.add(key)
        self.patches.append((obj, name, previous))

    def attention_metadata(self):
        if self.current_row is None or self.current_row.get("attention_metadata"):
            return
        try:
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata
        except (ImportError, AssertionError, RuntimeError, AttributeError):
            return
        entries = metadata.values() if isinstance(metadata, dict) else [metadata]
        output = []
        for item in entries:
            values = {key: getattr(item, key, None) for key in
                      ("num_prefills", "num_decodes", "num_prefill_tokens", "num_decode_tokens")}
            values = {k: v for k, v in values.items() if type(v) is int}
            if values and values not in output:
                output.append(values)
        self.current_row["attention_metadata"] = output

    def begin(self):
        self.status = "starting"
        with self.boundary("profiler.create"):
            self.profiler, self.scope, npu = make_profiler(
                self.directory, self.tag, self.level, self.shapes, self.stack)
        # Discover the current model tree. New/unknown modules and media are
        # retained; functional operations are captured by torch_npu itself.
        for path, module in self.runner.model.named_modules():
            prefix = f"serve/module/{category(path, module)}/{path or '<root>'}::{type(module).__name__}"
            for method in ("forward", "_forward_core", "compute_logits"):
                self.wrap(module, method, prefix + "/" + method)
        for method in ("_model_forward", "_preprocess", "_prepare_inputs", "_execute_mm_encoder",
                       "compute_logits", "_sample", "sample_tokens", "_postprocess",
                       "_bookkeeping_sync", "_bookkeeping_async"):
            self.wrap(self.runner, method, "serve/runner/" + method)
        self.wrap(npu, "synchronize", "serve/existing_npu_synchronize")
        atomic_json(self.inventory, dict(targets=self.targets, source_hashes=self.source_hashes))
        atexit.register(self.at_exit)
        with self.boundary("profiler.start"):
            self.profiler.start()
        self.status, self.phase = "capturing", "between_execute_calls"
        self.write()

    def finish(self, status):
        if self.done:
            return
        self.status = "stopping"
        atexit.unregister(self.at_exit)
        try:
            if self.profiler is not None:
                with self.boundary("profiler.stop"):
                    self.profiler.stop()
            issues = raw_errors(Path(self.trace_dir) if self.trace_dir else None)
            self.errors.extend(dict(phase="verify_raw", error=x) for x in issues)
            self.status = status if not self.errors else "incomplete"
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
            self.profiler, self.done, self.phase = None, True, "finished"
            self.write()
            print(f"[huanyu-serve-profile] {self.status.upper()} tp={self.who['tp_rank']} "
                  f"completed={self.completed}/{self.active} manifest={self.manifest}", flush=True)

    def at_exit(self):
        if not self.done:
            try:
                self.finish("incomplete")
            except Exception as exc:
                print(f"[huanyu-serve-profile] exit cleanup failed: {exc}", flush=True)

    def call(self, original, args, kwargs, scheduler):
        if self.done:
            return original(self.runner, *args, **kwargs)
        # Stop at the NEXT execute boundary: this includes sample_tokens and
        # asynchronous device work following the final captured forward.
        if self.profiler is not None and self.completed == self.active:
            self.finish("complete")
            return original(self.runner, *args, **kwargs)
        if self.arm and not Path(self.arm).is_file():
            return original(self.runner, *args, **kwargs)
        tokens = getattr(scheduler, "total_num_scheduled_tokens", None)
        if type(tokens) is not int:
            values = getattr(scheduler, "num_scheduled_tokens", {})
            tokens = sum(values.values()) if isinstance(values, dict) and all(type(v) is int for v in values.values()) else None
        if not tokens:
            return original(self.runner, *args, **kwargs)
        self.iteration += 1
        if self.iteration < self.start:
            return original(self.runner, *args, **kwargs)
        try:
            if self.profiler is None:
                self.begin()
            requests = getattr(scheduler, "num_scheduled_tokens", {})
            self.current_row = dict(iteration=self.iteration, scheduled_tokens=tokens,
                                    scheduled_requests=len(requests) if isinstance(requests, dict) else None,
                                    encoder_requests=len(getattr(scheduler, "scheduled_encoder_inputs", {}) or {}))
            with self.boundary("execute_model"):
                started = time.perf_counter()
                with self.scope(f"serve/execute/{self.iteration}"):
                    result = original(self.runner, *args, **kwargs)
                self.completed += 1
                self.rows.append({**self.current_row, "host_elapsed_s": time.perf_counter() - started})
                self.current_row = None
            return result
        except BaseException:
            try:
                self.finish("incomplete")
            except Exception as exc:
                print(f"[huanyu-serve-profile] cleanup failed: {exc}", flush=True)
            raise


def bind_runner(cls):
    original = cls.execute_model
    if getattr(original, MARKER, False):
        return
    signature = inspect.signature(original)
    if "scheduler_output" not in signature.parameters:
        raise RuntimeError("Installed execute_model has no scheduler_output parameter; inspect this runner")

    @wraps(original)
    def execute(self, *args, **kwargs):
        if not os.environ.get(PREFIX + "DIR"):
            return original(self, *args, **kwargs)
        state = getattr(self, MARKER, None)
        if state is None:
            state = Capture(self)
            setattr(self, MARKER, state)
        bound = signature.bind(self, *args, **kwargs)
        return state.call(original, args, kwargs, bound.arguments["scheduler_output"])

    setattr(execute, MARKER, True)
    cls.execute_model = execute


def edit_entry(path, remove=False):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    hook = (f"\n{BEGIN}\n"
            "import os as _hsp_os\n"
            f"if _hsp_os.environ.get({PREFIX + 'DIR'!r}):\n"
            "    import importlib.util as _hsp_util\n"
            "    import sys as _hsp_sys\n"
            f"    _hsp_spec = _hsp_util.spec_from_file_location('_huanyu_serve_profile', {str(Path(__file__).resolve())!r})\n"
            "    _hsp_module = _hsp_util.module_from_spec(_hsp_spec)\n"
            "    _hsp_sys.modules['_huanyu_serve_profile'] = _hsp_module\n"
            "    _hsp_spec.loader.exec_module(_hsp_module)\n"
            "    _hsp_module.bind_runner(NPUARModelRunner)\n"
            f"{END}\n")
    if BEGIN in text or END in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
            raise ValueError("Unrecognized existing hook; refusing to alter this file")
        start = text.index(BEGIN)
        if text[max(0, start - 2):start] == "\r\n":
            start -= 2
        elif start and text[start - 1] == "\n":
            start -= 1
        stop = text.index(END) + len(END)
        if text[stop:stop + 2] == "\r\n":
            stop += 2
        elif text[stop:stop + 1] == "\n":
            stop += 1
        text = text[:start] + text[stop:]
    elif remove:
        print("No serve hook installed")
        return
    tree = ast.parse(text)
    if not any(isinstance(n, ast.ClassDef) and n.name == "NPUARModelRunner" for n in tree.body):
        raise ValueError("Expected the actual module defining NPUARModelRunner")
    updated = text if remove else text + hook
    compile(updated, str(path), "exec")
    if updated.encode("utf-8") == raw:
        print("Hook already up to date")
        return
    backup = path.with_name(path.name + ".serve-profile-backup-" + uuid.uuid4().hex[:10])
    backup.write_bytes(raw)
    path.write_bytes(updated.encode("utf-8"))
    print(f"Updated: {path}\nBackup: {backup}")


def captures(root, expected=None, complete=False):
    records = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(root.glob("serve_pid*_tp*.json"))
               if not p.name.endswith(".inventory.json")]
    if not records:
        raise RuntimeError("No worker manifests: check the installed hook, environment and actual runner import path")
    if expected is not None and len(records) != expected:
        raise RuntimeError(f"Expected {expected} worker manifests, got {len(records)}")
    issues = []
    ranks = []
    for path, data in records:
        who = data["identity"]
        ranks.append(who["tp_rank"])
        if expected is not None and len(who.get("tp_ranks", [])) != expected:
            issues.append(f"{path.name}: TP group size does not match --expected-workers={expected}")
        print(f"tp={who['tp_rank']} pid={who['pid']} {data['status']} phase={data['phase']} "
              f"completed={data['completed']}/{data['active']} manifest={path}", flush=True)
        if complete:
            if data["status"] != "complete" or data["completed"] != data["active"] or data["errors"]:
                issues.append(f"{path.name}: incomplete window/finalization: {data['errors']}")
            issues.extend(raw_errors(Path(data["trace_dir"]) if data.get("trace_dir") else None))
            if not any(k.startswith("serve/module/") and v for k, v in (data.get("hits") or {}).items()):
                issues.append(f"{path.name}: no live module hooks executed")
            if not Path(data["inventory"]).is_file():
                issues.append(f"{path.name}: missing module/source inventory")
    if expected is not None and sorted(ranks) != list(range(expected)):
        issues.append(f"Expected one TP group with ranks 0..{expected - 1}; got {ranks}")
    if issues:
        raise RuntimeError("\n".join(issues))
    return records


def prepare_eager(source, output):
    import yaml

    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data.get("pipeline") != "minicpmo_5_0_thinker_only":
        raise ValueError("Expected the installed minicpmo_5_0_thinker_only deploy config")
    groups = [data, data.get("platforms", {}).get("npu", {})]
    for group in groups:
        stages = group.get("stages", [])
        if len(stages) != 1 or stages[0].get("stage_id") != 0 or "engine_args" in stages[0]:
            raise ValueError("Expected flat single stage_id=0 in both base and NPU overlay")
        stage = stages[0]
        if any(k in (stage.get("engine_extras") or {}) for k in
               ("enforce_eager", "compilation_config", "enable_prefix_caching")):
            raise ValueError("Conflicting engine_extras overrides")
        if group is data or "enable_prefix_caching" in group:
            group["enable_prefix_caching"] = False
        stage["enforce_eager"] = True
        stage["enable_prefix_caching"] = False
        stage.setdefault("compilation_config", {})["cudagraph_mode"] = "NONE"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    print(f"Eager-only config: {output.resolve()}; original TP/devices/capacity retained")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "uninstall", "status", "check", "analyse"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner-file", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--expected-workers", type=int)
    args = parser.parse_args(argv)
    try:
        if args.expected_workers is not None and args.expected_workers < 1:
            parser.error("--expected-workers must be positive")
        if args.action == "prepare":
            if args.source is None or args.output is None:
                parser.error("--source and --output are required")
            prepare_eager(args.source, args.output)
        elif args.action in {"install", "uninstall"}:
            if args.runner_file is None:
                parser.error("--runner-file is required")
            edit_entry(args.runner_file, args.action == "uninstall")
        else:
            if args.profile_dir is None:
                parser.error("--profile-dir is required")
            records = captures(args.profile_dir, args.expected_workers, args.action != "status")
            if args.action == "analyse":
                from torch_npu.profiler.profiler import analyse
                from serve_profile_report import report_capture

                for _, data in records:
                    trace = Path(data["trace_dir"])
                    print(f"Parsing one worker: {trace}", flush=True)
                    files = [trace / "ASCEND_PROFILER_OUTPUT" / f for f in ("trace_view.json", "kernel_details.csv")]
                    before = [(p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files]
                    analyse(str(trace), max_process_number=1, export_type="text")
                    after = [(p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files]
                    if any(a is None or a == b for a, b in zip(after, before)):
                        raise RuntimeError(f"Parser did not create/refresh trace and kernels: {trace}")
                    report_capture(data)
            if args.action != "status":
                print(f"PASS: {len(records)} workers")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
