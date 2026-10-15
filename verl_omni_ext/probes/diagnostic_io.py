"""Bounded, opt-in diagnostic records; never copy device tensor contents.

This module deliberately has no ML/runtime imports. All workers append to one
plain-text log on a shared filesystem, under an OS file lock. Budgets are global to
that file, not multiplied by the number of workers. Host/PID remain in records.
"""

import hashlib
import os
import socket
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.RLock()
_counts = Counter()
_warned = False
# Retained for older in-process test/reset callers; no path is permanently
# suppressed here, so replacing a capped file can start a fresh diagnostic run.
_full_paths = set()
_scan_cache = {}
_write_failures = {"count": 0, "last_type": None, "first_time_ns": None}
LOG_NAME = "rollout_diagnostics.log"
DIAGNOSTICS_REVISION = "2026-10-15-log-v2"
_MAX_RECORD_BYTES = 16384
_LIMIT_RESERVE = 1024
_LIMIT_PREFIX = b"DIAG event='diagnostic_limit_reached' "


def _write_failure(exc):
    """Keep failure state in memory; never create per-worker fallback logs.

    Retry on the next event. Once writing succeeds, a recovery notice in the same
    file reports the suppressed attempts. No device objects or exception traces
    are retained, and failures never spill into the training console.
    """
    global _warned
    _warned = True
    with _lock:
        if not _write_failures["count"]:
            _write_failures["first_time_ns"] = time.time_ns()
        _write_failures["count"] = min(_write_failures["count"] + 1, 2**31 - 1)
        _write_failures["last_type"] = type(exc).__name__[:128]


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
    """One physical text line; repr escapes newlines without a JSON dependency.

    Fields have already passed through summarize(), so no device tensor is
    formatted. The fixed prefix is used only to recognize our terminal budget
    notice; ordinary old lines are never parsed or required to have a schema.
    """
    record = dict(record)
    event = str(record.pop("event", "unknown"))[:128]
    time_ns = record.get("time_ns")
    if isinstance(time_ns, int):
        record["time_utc"] = datetime.fromtimestamp(time_ns / 1e9, timezone.utc).isoformat(timespec="milliseconds")

    def encode():
        parts = [f"DIAG event={event!r}"]
        parts.extend(f"{key}={value!r}" for key, value in record.items() if key != "fields")
        parts.append(f"fields={record.get('fields', {})!r}")
        return (" ".join(parts) + "\n").encode("utf-8", errors="backslashreplace")

    data = encode()
    if len(data) > _MAX_RECORD_BYTES:
        original = repr(record.get("fields", {}))
        record["fields"] = {
            "record_truncated": True, "chars": len(original),
            "sha256": hashlib.sha256(original.encode("utf-8", errors="backslashreplace")).hexdigest(),
            "head": original[:512], "tail": original[-512:],
        }
        data = encode()
    if len(data) > _MAX_RECORD_BYTES:
        # Top-level metadata is internal, but keep the bound even for a malformed
        # caller. A shortened line remains useful without needing to parse it.
        data = data[:_MAX_RECORD_BYTES - 32].decode("utf-8", errors="ignore").encode("utf-8") + b" ... record_truncated=True\n"
    return data


def _scan_log(stream, path):
    """Count physical lines incrementally; preserve every existing byte.

    Under the file lock, cached inode/offset state is safe for cooperating append
    writers. A rotated or truncated file starts a fresh scan. Only a bounded
    prefix of the unfinished last line is retained, never its whole contents.
    """
    stat = os.fstat(stream.fileno())
    identity = (stat.st_dev, stat.st_ino)
    key = str(path)
    state = _scan_cache.get(key)
    if state is None or state["identity"] != identity or stat.st_size < state["offset"]:
        state = {"identity": identity, "offset": 0, "newlines": 0, "prefix": b"", "limited": False}
    else:
        state = dict(state)
    stream.seek(state["offset"])
    while True:
        block = stream.read(65536)
        if not block:
            break
        parts = block.split(b"\n")
        for part in parts[:-1]:
            prefix = (state["prefix"] + part)[:len(_LIMIT_PREFIX)]
            state["limited"] |= prefix == _LIMIT_PREFIX
            state["newlines"] += 1
            state["prefix"] = b""
        state["prefix"] = (state["prefix"] + parts[-1])[:len(_LIMIT_PREFIX)]
        state["offset"] += len(block)
    state["limited"] |= state["prefix"] == _LIMIT_PREFIX
    if key not in _scan_cache and len(_scan_cache) >= 16:
        _scan_cache.pop(next(iter(_scan_cache)))
    _scan_cache[key] = state
    return state["offset"], state["newlines"] + bool(state["prefix"]), bool(state["prefix"]), state["limited"]


def _append_record(path, record):
    max_records = _limit("MINICPMO_ROLLOUT_DIAG_MAX_RECORDS", 2000, 100000)
    max_bytes = max(_LIMIT_RESERVE, _limit("MINICPMO_ROLLOUT_DIAG_MAX_BYTES", 4 * 1024 * 1024,
                                         64 * 1024 * 1024))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as stream, _file_lock(stream):
        if os.fstat(stream.fileno()).st_size >= max_bytes:
            return False
        size, count, incomplete_tail, capped = _scan_log(stream, path)
        if capped:
            return False
        # An interrupted append is evidence. Separate it from this record instead
        # of deleting it or attempting to decode its contents.
        separator = b"\n" if incomplete_tail else b""
        record["record_index"] = count + 1
        data = _encode_record(record)
        reason = "record_limit" if count >= max_records else "file_size_limit"
        normal_record = True
        if count >= max_records or size + len(separator) + len(data) + _LIMIT_RESERVE > max_bytes:
            record = {"event": "diagnostic_limit_reached", "record_index": count + 1,
                      "time_ns": time.time_ns(), "host": socket.gethostname(), "pid": os.getpid(),
                      "fields": {"reason": reason, "max_records": max_records, "max_bytes": max_bytes,
                                 "message": "Further diagnostic records are suppressed; training continues."}}
            data = _encode_record(record)
            normal_record = False
            if size + len(separator) + len(data) > max_bytes:
                return False
        # All cooperating writers hold the lock for the complete line.
        stream.seek(0, os.SEEK_END)
        remaining = memoryview(separator + data)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise OSError("diagnostic append made no progress")
            remaining = remaining[written:]
        return normal_record


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
            path = Path(directory) / LOG_NAME
            if _write_failures["count"]:
                recovery = {"event": "diagnostic_write_recovered", "time_ns": time.time_ns(),
                            "host": host, "pid": os.getpid(), "fields": dict(_write_failures)}
                if _append_record(path, recovery):
                    _write_failures.update(count=0, last_type=None, first_time_ns=None)
            record = {"event": event, "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "host": host, "pid": os.getpid(), "thread": threading.get_ident(),
                      "fields": summarize(fields)}
            _append_record(path, record)
    except Exception as exc:
        _write_failure(exc)


def console_or_file(event, message, *, flush=True):
    """Move existing diagnostic prints to the log while preserving disabled behavior."""
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
                 "verl_omni_ext.reward.genrm_verifier",
                 "verl_omni_ext.probes.diagnostic_io",
                 "verl_omni_ext.probes.rollout_trace", "verl_omni_ext.probes.gdn_trace")
        probe_sources = {}
        for name in ("diagnostic_io", "rollout_trace", "gdn_trace"):
            module_name = "verl_omni_ext.probes." + name
            module = sys.modules.get(module_name)
            if module is not None:
                probe_sources[name] = {"loaded": True, **module_info(module)}
                continue
            # Some hooks are imported later. Hash the adjacent source without
            # importing it, and explicitly distinguish disk presence from use.
            source = Path(__file__).with_name(name + ".py")
            info = {"loaded": False, "file": str(source)}
            try:
                info["source_exists"] = source.is_file()
                if source.is_file() and source.stat().st_size <= 2_000_000:
                    info["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            except Exception as exc:
                info["observation_error"] = type(exc).__name__
            probe_sources[name] = info
        emit("process_start", diagnostics_revision=DIAGNOSTICS_REVISION, executable=sys.executable, cwd=os.getcwd(),
             probe_sources=probe_sources,
             modules={name: module_info(sys.modules[name]) for name in names if name in sys.modules},
             switches={key: value for key, value in os.environ.items()
                       if key.startswith(("MINICPMO_DIAG_", "MINICPMO_ROLLOUT_DIAG"))})
