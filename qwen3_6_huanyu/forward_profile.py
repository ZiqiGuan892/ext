"""Opt-in, bounded NPU traces of the *installed* Huanyu rollout implementation.

This file can be copied on its own. ``python forward_profile.py install`` adds
an opt-in import to the sibling worker_extension.py; it never replaces forward.
No torch/NPU imports or runtime patches occur when profiling is disabled.
"""

import argparse
import atexit
from collections import Counter
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
            with self.scope(label):
                return original(*args, **kwargs)

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
        self.write("starting")
        print(f"[huanyu-forward-profile] START manifest={self.manifest}", flush=True)
        self.profiler.start()
        atexit.register(self.at_exit)

    def at_exit(self):
        if self.profiler is not None and not self.done:
            try:
                self.finish("incomplete")
            except Exception as error:
                print(f"[huanyu-forward-profile] exit cleanup failed: {error}", flush=True)

    def write(self, status):
        self.manifest.write_text(json.dumps({
            "identity": self.identity, "status": status, "warmup": 1,
            "active": self.active_steps, "level": self.level,
            "execute_source_chain": _source_chain(type(self.runner).execute_model),
            "targets": self.targets, "hits": dict(self.hits), "iterations": self.rows,
            "note": "CPU ranges are inclusive; correlate device events. parent_pid grouping needs TP-rank verification.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def finish(self, status):
        self.done = True
        atexit.unregister(self.at_exit)
        try:
            if self.profiler is not None:
                self.profiler.stop()
        finally:
            for obj, name, (existed, value) in reversed(self.patches):
                if existed:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
            self.patches.clear()
            self.profiler = None
            if hasattr(self, "manifest"):
                self.write(status)
                print(f"[huanyu-forward-profile] {status.upper()} manifest={self.manifest}", flush=True)

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
            started = time.perf_counter()
            with self.scope(f"huanyu/execute/iteration_{self.iteration}"):
                result = original(self.runner, *args, **kwargs)
            self.rows.append({**self.row, "host_seconds": time.perf_counter() - started})
            self.row = None
            self.profiler.step()
            self.steps += 1
            if self.steps == self.active_steps + 1:
                self.finish("complete")
            return result
        except BaseException:
            if self.row is not None:
                self.rows.append({**self.row, "error": True})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("--worker-extension", type=Path, default=Path(__file__).with_name("worker_extension.py"))
    args = parser.parse_args()
    edit_hook(args.worker_extension, remove=args.action == "uninstall")