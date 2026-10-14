"""Bounded, opt-in diagnostic records; never copy device tensor contents.

This module deliberately has no ML/runtime imports. All workers append to one
JSONL file on a shared filesystem, under an OS file lock. Budgets are global to
that file, not multiplied by the number of workers. Host/PID remain in records.
"""

import hashlib
import json
import os
import socket
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

_lock = threading.RLock()
_counts = Counter()
_warned = False
_full_paths = set()
LOG_NAME = "rollout_diagnostics.jsonl"
_MAX_RECORD_BYTES = 16384
_LIMIT_RESERVE = 1024


def _write_failure(exc):
    """Keep failure state in memory; never create per-worker fallback logs.

    The launcher checks shared-directory access before training and checks the
    output afterwards. An unwritable destination cannot reliably log its own
    failure; diagnostics must still never interrupt model execution.
    """
    global _warned
    _warned = True


def enabled():
    return os.environ.get("MINICPMO_ROLLOUT_DIAG") == "1"


def _limit(name, default, maximum):
    try:
        return max(0, min(int(os.environ.get(name, default)), maximum))
    except (ValueError, TypeError):
        return default


def limited(event, limit=None):
    """Reserve one bounded observation, without changing the observed operation."""
    if not enabled():
        return False
    if limit is None:
        limit = _limit("MINICPMO_ROLLOUT_DIAG_MAX_EVENTS", 4, 10000)
    with _lock:
        key = (os.getpid(), event)
        _counts[key] += 1
        return _counts[key] <= limit


def text_info(value):
    """Retain a bounded excerpt and a hash of the original string."""
    if not isinstance(value, str):
        return {"type": type(value).__name__}
    limit = _limit("MINICPMO_ROLLOUT_DIAG_TEXT_CHARS", 256, 32768)
    info = {
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest(),
        "truncated": len(value) > limit,
    }
    if len(value) <= limit:
        info["text"] = value
    else:
        half = limit // 2
        info.update(head=value[:half], tail=value[-half:] if half else "")
    return info


def _tensor_info(value):
    # Access metadata only: no .cpu(), .item(), .numpy(), min(), norm() or repr().
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": [str(dim) if not isinstance(dim, int) else dim for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "cpu/unknown")),
            "values_read": False,
        }
    return None


def token_info(value):
    """CPU list token IDs are hashed; tensors are described without reading them."""
    try:
        tensor = _tensor_info(value)
        if tensor is not None:
            return tensor
        if isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in value):
            digest = hashlib.sha256()
            for token in value:
                digest.update(f"{token},".encode("ascii"))
            return {"count": len(value), "sha256": digest.hexdigest(),
                    "head": list(value[:32]), "tail": list(value[-32:]),
                    "truncated": len(value) > 64}
        return summarize(value)
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


def summarize(value, _depth=0):
    """Bound recursion, containers and strings, including unrecognized objects."""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= _limit("MINICPMO_ROLLOUT_DIAG_TEXT_CHARS", 256, 32768) else text_info(value)
        tensor = _tensor_info(value)
        if tensor is not None:
            return tensor
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"type": type(value).__name__, "bytes": len(value)}
        if _depth >= 7:
            return {"type": type(value).__name__, "depth_limited": True}
        if isinstance(value, Mapping):
            result = {}
            for index, (key, item) in enumerate(value.items()):
                if index == 24:
                    result["__remaining_items__"] = max(0, len(value) - index)
                    break
                key = str(key)[:128]
                if any(word in key.lower() for word in ("authorization", "api_key", "password", "secret")):
                    result[key] = "<redacted>"
                else:
                    result[key] = summarize(item, _depth + 1)
            return result
        if isinstance(value, (tuple, list)):
            items = [summarize(item, _depth + 1) for item in value[:64]]
            if len(value) <= 64:
                return items
            return {"type": type(value).__name__, "count": len(value), "items": items, "truncated": True}
        return {"type": type(value).__name__, "module": type(value).__module__}
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


def module_info(module):
    try:
        info = {"name": getattr(module, "__name__", type(module).__name__),
                "file": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None)}
        path = info["file"]
        if path and str(path).endswith(".py") and Path(path).stat().st_size <= 2_000_000:
            info["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return info
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


@contextmanager
def _file_lock(stream):
    """Serialize append + budget check across processes and shared-FS nodes.

    Bounded lock wait prevents a diagnostic lock from stalling training. POSIX
    locks are released by the OS when a worker exits, including after a crash.
    """
    if os.name == "nt":
        import msvcrt

        def acquire():
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

        def release():
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def acquire():
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release():
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    deadline = time.monotonic() + 0.25
    while True:
        try:
            acquire()
            break
        except (BlockingIOError, PermissionError):
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic file lock timed out") from None
            time.sleep(0.005)
    try:
        yield
    finally:
        release()


def _encode_record(record):
    data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > _MAX_RECORD_BYTES:
        # Keep a valid JSON line and the event identity even for huge metadata.
        original = json.dumps(record["fields"], ensure_ascii=False, allow_nan=False)
        record["fields"] = {
            "record_truncated": True, "chars": len(original),
            "sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "head": original[:512], "tail": original[-512:],
        }
        data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    return data


def _last_record(stream):
    """Read only the bounded tail; repair an interrupted final append."""
    size = stream.seek(0, os.SEEK_END)
    if not size:
        return 0, None
    start = max(0, size - _MAX_RECORD_BYTES * 2)
    stream.seek(start)
    tail = stream.read()
    if not tail.endswith(b"\n"):
        end = tail.rfind(b"\n") + 1
        if not end and start:
            raise ValueError("diagnostic tail is not recoverable")
        stream.truncate(start + end)
        tail = tail[:end]
        size = start + end
    if not tail:
        return size, None
    return size, json.loads(tail.splitlines()[-1])


def _append_record(path, record):
    if str(path) in _full_paths:
        return
    max_records = _limit("MINICPMO_ROLLOUT_DIAG_MAX_RECORDS", 2000, 100000)
    max_bytes = max(_LIMIT_RESERVE, _limit("MINICPMO_ROLLOUT_DIAG_MAX_BYTES", 4 * 1024 * 1024,
                                         64 * 1024 * 1024))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as stream, _file_lock(stream):
        size, last = _last_record(stream)
        if last and last.get("event") == "diagnostic_limit_reached":
            _full_paths.add(str(path))
            return
        count = last["record_index"] if last else 0
        record["record_index"] = count + 1
        data = _encode_record(record)
        reason = "record_limit" if count >= max_records else "file_size_limit"
        if count >= max_records or size + len(data) + _LIMIT_RESERVE > max_bytes:
            record = {"event": "diagnostic_limit_reached", "record_index": count + 1,
                      "time_ns": time.time_ns(), "host": socket.gethostname(), "pid": os.getpid(),
                      "fields": {"reason": reason, "max_records": max_records, "max_bytes": max_bytes,
                                 "message": "Further diagnostic records are suppressed; training continues."}}
            data = _encode_record(record)
            _full_paths.add(str(path))
            if size + len(data) > max_bytes:
                return
        # All cooperating writers hold the lock for the complete line.
        stream.seek(0, os.SEEK_END)
        remaining = memoryview(data)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise OSError("diagnostic append made no progress")
            remaining = remaining[written:]


def emit(event, **fields):
    """Best-effort append to one bounded file; never affect model execution."""
    if not enabled():
        return
    try:
        directory = os.environ.get("MINICPMO_ROLLOUT_DIAG_DIR", "")
        if not directory:
            raise ValueError("MINICPMO_ROLLOUT_DIAG_DIR is required when diagnostics are enabled")
        with _lock:
            host = socket.gethostname()
            record = {"event": event, "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "host": host, "pid": os.getpid(), "thread": threading.get_ident(),
                      "fields": summarize(fields)}
            _append_record(Path(directory) / LOG_NAME, record)
    except Exception as exc:
        _write_failure(exc)


def console_or_file(event, message, *, flush=True):
    """Move existing diagnostic prints to JSONL while preserving disabled behavior."""
    if enabled():
        if limited(event):
            emit(event, message=text_info(message))
    else:
        print(message, flush=flush)


def startup():
    """Record loaded modules, not whichever installation happens to be on PATH."""
    if limited("process_start", 1):
        names = ("verl_omni_ext", "verl_omni", "verl", "vllm", "vllm_omni", "vllm_ascend", "torch",
                 "verl_omni_ext.models.qwen3_6_whisper",
                 "verl_omni_ext.models.qwen3_6_whisper.forward",
                 "verl_omni_ext.models.qwen3_6_whisper.mm_inputs",
                 "verl_omni_ext.reward.genrm_verifier")
        emit("process_start", executable=sys.executable, cwd=os.getcwd(),
             modules={name: module_info(sys.modules[name]) for name in names if name in sys.modules},
             switches={key: value for key, value in os.environ.items()
                       if key.startswith(("MINICPMO_DIAG_", "MINICPMO_ROLLOUT_DIAG"))})
