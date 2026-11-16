"""Opt-in host-side training diagnostics; no tensor copies or device barriers.

HUANYU_TRAIN_HANG_DIR enables per-process JSONL and timed Python stack dumps.
The first HUANYU_TRAIN_HANG_CALLS (default 2) model calls are observed. A return
event means Python returned, not that asynchronous NPU work completed. FSDP
hooks installed later may execute outside these module scopes; use the stack
dump to identify the actual wait. This probe owns faulthandler's process-wide
dump_traceback_later timer while a selected forward is active.
"""

import atexit
from collections import Counter
import faulthandler
from functools import wraps
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import traceback
from types import MethodType


def _host_memory():
    result = {}
    try:
        result["proc_status"] = [line for line in Path("/proc/self/status").read_text().splitlines()
                                 if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:"))]
    except OSError as exc:
        result["proc_status_error"] = str(exc)
    # cgroup v2: report the observed path and counters, never infer an OOM from
    # a historical nonzero counter. Compare baseline/end within the same group.
    try:
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line[3:] for line in groups if line.startswith("0::"))
        base = Path("/sys/fs/cgroup")
        candidate = base / relative.lstrip("/")
        if not (candidate / "memory.events").is_file():
            candidate = base
        result["cgroup_path"] = str(candidate)
        for name in ("memory.current", "memory.max", "memory.events"):
            result[name] = (candidate / name).read_text().strip()
    except (OSError, StopIteration) as exc:
        result["cgroup_error"] = str(exc) or "cgroup v2 not available"
    return result


def _metadata(value, depth=0):
    # Only tensor metadata: no repr(tensor), .cpu(), .item(), or value reductions.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(getattr(value, "device", "unknown"))}
    if depth < 2 and isinstance(value, dict):
        return {str(k): _metadata(v, depth + 1) for k, v in list(value.items())[:16]}
    if depth < 2 and isinstance(value, (list, tuple)):
        return {"length": len(value), "items": [_metadata(v, depth + 1) for v in value[:4]]}
    return {"type": type(value).__name__}


def _selected(name):
    return (name in ("model.visual", "audio_encoder", "audio_projection", "model.language_model", "lm_head")
            or re.fullmatch(r"model\.visual\.(patch_embed|merger|blocks\.\d+)", name) is not None
            or re.fullmatch(r"model\.language_model\.(embed_tokens|norm|layers\.\d+(\.(self_attn|linear_attn|mlp))?)",
                            name) is not None)


class TrainHangProbe:
    def __init__(self, directory, torch_module, *, seconds=120, calls=2):
        self.torch = torch_module
        self.seconds, self.limit = seconds, calls
        self.call = 0
        self.active = False
        self.phase = "installed"
        self.counts = Counter()
        self.handles = []
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.rank = os.environ.get("RANK", "unknown")
        self.pid = os.getpid()
        self.host = socket.gethostname()
        path = Path(directory) / f"{self.host}_pid{self.pid}_{time.time_ns()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.events = path.with_suffix(".jsonl").open("a", encoding="utf-8", buffering=1)
        self.stacks = path.with_suffix(".stacks.txt").open("a", encoding="utf-8", buffering=1)
        self.emit("installed", memory=_host_memory(), timeout_seconds=seconds, max_calls=calls)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="huanyu-train-hang")
        self.thread.start()
        atexit.register(self.close)

    def emit(self, event, **fields):
        try:
            with self.lock:
                self.events.write(json.dumps(dict(time=time.time(), monotonic=time.monotonic(),
                                                 pid=self.pid, host=self.host, rank=self.rank,
                                                 call=self.call, event=event, **fields), ensure_ascii=True) + "\n")
        except Exception as exc:
            # Diagnostic I/O must not replace the training exception.
            print(f"[huanyu-train-hang] write failed: {type(exc).__name__}: {exc}", flush=True)

    def memory(self):
        result = _host_memory()
        backend = getattr(self.torch, "npu", None)
        try:
            if backend is not None and backend.is_initialized():
                device = backend.current_device()
                stats = backend.memory_stats(device)
                result["npu_device"] = device
                result["npu_allocator"] = {key: stats.get(key) for key in (
                    "allocated_bytes.all.current", "allocated_bytes.all.peak",
                    "reserved_bytes.all.current", "reserved_bytes.all.peak",
                    "num_ooms", "num_alloc_retries")}
            else:
                result["npu_allocator"] = "not initialized/available"
        except Exception as exc:
            result["npu_memory_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _heartbeat(self):
        while not self.closed.wait(15):
            if self.active:
                # No NPU API from a background thread. The independent C-level
                # faulthandler timer still works if Python cannot obtain the GIL.
                self.emit("heartbeat", last_phase=self.phase, memory=_host_memory())

    def root_enter(self, module, args, kwargs):
        if self.call >= self.limit:
            return
        self.call += 1
        self.active, self.returned, self.failed = True, False, False
        self.counts.clear()
        self.phase = "forward.enter"
        dist = getattr(self.torch, "distributed", None)
        if dist is not None and dist.is_initialized():
            self.rank = dist.get_rank()
        self.stacks.write(f"\ncall={self.call} rank={self.rank} started={time.time()}\n")
        self.stacks.flush()
        faulthandler.dump_traceback_later(self.seconds, repeat=True, file=self.stacks, exit=False)
        self.emit("forward_enter", inputs=_metadata({"args": args, **kwargs}), memory=self.memory())

    def root_exit(self, module, args, kwargs, output):
        if not self.active:
            return
        faulthandler.cancel_dump_traceback_later()
        self.active = False
        self.emit("forward_exit", python_forward_returned=self.returned, exception_logged=self.failed,
                  last_phase=self.phase, scope_calls=dict(self.counts), memory=self.memory())

    def enter(self, name, args, kwargs):
        if self.active:
            self.phase = name + ".enter"
            self.counts[name] += 1
            self.emit("scope_enter", scope=name, occurrence=self.counts[name],
                      inputs=_metadata({"args": args, **kwargs}), memory=self.memory())

    def leave(self, name):
        if self.active:
            self.phase = name + ".return"
            self.emit("scope_return", scope=name, occurrence=self.counts[name])

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        if self.active:
            faulthandler.cancel_dump_traceback_later()
            self.active = False
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)
        for handle in self.handles:
            handle.remove()
        self.events.close()
        self.stacks.close()


def install_train_hang_probe(module):
    directory = os.environ.get("HUANYU_TRAIN_HANG_DIR", "").strip()
    if not directory or getattr(module, "_huanyu_train_hang_probe", None) is not None:
        return None
    seconds = float(os.environ.get("HUANYU_TRAIN_HANG_SECONDS", "120"))
    calls = int(os.environ.get("HUANYU_TRAIN_HANG_CALLS", "2"))
    if not 0 < seconds < float("inf") or calls < 1:
        raise ValueError("HUANYU_TRAIN_HANG_SECONDS and HUANYU_TRAIN_HANG_CALLS must be positive")
    import torch

    probe = TrainHangProbe(directory, torch, seconds=seconds, calls=calls)
    original = module.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        try:
            output = original(*args, **kwargs)
        except BaseException as exc:
            if probe.active:
                probe.failed = True
                try:
                    probe.emit("exception", error_type=type(exc).__name__, error=str(exc),
                               last_phase=probe.phase, traceback=traceback.format_exc(), memory=probe.memory())
                except Exception:
                    # Even a host MemoryError while formatting diagnostics must
                    # preserve the original exception and its traceback.
                    pass
            raise
        if probe.active:
            probe.returned = True
        return output

    module.forward = MethodType(forward, module)
    probe.handles.append(module.register_forward_pre_hook(probe.root_enter, prepend=True, with_kwargs=True))
    probe.handles.append(module.register_forward_hook(probe.root_exit, always_call=True, with_kwargs=True))
    for name, child in module.named_modules():
        if not _selected(name):
            continue
        def before(mod, args, kwargs, scope=name):
            probe.enter(scope, args, kwargs)
        def after(mod, args, output, scope=name):
            probe.leave(scope)
        # Log entry before the existing visual-unshard prehook.
        probe.handles.append(child.register_forward_pre_hook(before, prepend=True, with_kwargs=True))
        probe.handles.append(child.register_forward_hook(after))
    module._huanyu_train_hang_probe = probe
    print(f"[huanyu-train-hang] installed pid={probe.pid} files={probe.path}.* "
          f"calls={calls} stack_after={seconds}s", flush=True)
    return probe


def report(directory):
    """Read live or finished per-process logs, without importing torch."""
    paths = sorted(Path(directory).glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No training hang logs in {directory}")
    for path in paths:
        rows, incomplete = [], 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    incomplete += 1
        if not rows:
            print(f"{path}: no readable records, incomplete={incomplete}")
            continue
        last = rows[-1]
        print(f"\nrank={last['rank']} pid={last['pid']} call={last['call']} "
              f"last_event={last['event']} incomplete={incomplete}\nfile={path}")
        current = [r for r in rows if r["call"] == last["call"]]
        phases = [r for r in current if r["event"] in ("scope_enter", "scope_return", "forward_enter", "forward_exit")]
        for row in phases[-6:]:
            print(f"  {row['time']:.3f} {row['event']} {row.get('scope', '')} {row.get('occurrence', '')}")
        visual = next((r for r in reversed(current) if r.get("scope") == "model.visual"
                       and r["event"] == "scope_enter"), None)
        if visual:
            print("visual_inputs:", json.dumps(visual["inputs"]))
        latest_memory = next((r["memory"] for r in reversed(rows) if "memory" in r), {})
        allocator = next((r["memory"] for r in reversed(rows)
                          if isinstance(r.get("memory", {}).get("npu_allocator"), dict)), {})
        print("latest_host_memory:", json.dumps(latest_memory))
        print("last_npu_allocator:", json.dumps(allocator.get("npu_allocator", "unavailable")))
        for row in current:
            if row["event"] == "exception":
                print(row["traceback"])
        print("python_stacks:", path.with_suffix(".stacks.txt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize training hang diagnostics; no model/NPU imports")
    parser.add_argument("directory")
    report(parser.parse_args().directory)
