"""Bounded, opt-in diagnostic records; never copy device tensor contents.

This module deliberately has no ML/runtime imports. JSONL files are separated by
host and PID so Ray workers cannot overwrite one another's evidence.
"""

import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

_lock = threading.RLock()
_counts = Counter()
_warned = False


def _write_failure(exc):
    """Report sink failures to a separate file, never to Ray's console stream."""
    global _warned
    if _warned:
        return
    _warned = True
    try:
        directory = Path(tempfile.gettempdir()) / "verl_omni_ext_rollout_diag_errors"
        directory.mkdir(parents=True, exist_ok=True)
        host = "".join(c if c.isalnum() or c in "-_." else "_" for c in socket.gethostname())
        with (directory / f"error-{host}-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "event": "diagnostic_sink_error", "pid": os.getpid(), "host": host,
                "time_ns": time.time_ns(), "error_type": type(exc).__name__,
                "error": str(exc)[:2048],
                "requested_directory": os.environ.get("MINICPMO_ROLLOUT_DIAG_DIR", ""),
            }, ensure_ascii=True) + "\n")
    except Exception:
        # Never redirect diagnostic text into the main training log, even if
        # both destinations are unavailable. The absence of traces is not success.
        pass


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
        limit = _limit("MINICPMO_ROLLOUT_DIAG_MAX_EVENTS", 16, 10000)
    with _lock:
        key = (os.getpid(), event)
        _counts[key] += 1
        return _counts[key] <= limit


def text_info(value):
    """Retain a bounded excerpt and a hash of the original string."""
    if not isinstance(value, str):
        return {"type": type(value).__name__}
    limit = _limit("MINICPMO_ROLLOUT_DIAG_TEXT_CHARS", 1024, 32768)
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
            return value if len(value) <= _limit("MINICPMO_ROLLOUT_DIAG_TEXT_CHARS", 1024, 32768) else text_info(value)
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


def emit(event, **fields):
    """Best-effort write; observation failures must not stop model execution.

    The caller uses limited() for hot paths. A separate global record budget
    bounds disk usage even when hooks report recurring errors.
    """
    if not enabled():
        return
    try:
        directory = os.environ.get("MINICPMO_ROLLOUT_DIAG_DIR", "")
        if not directory:
            raise ValueError("MINICPMO_ROLLOUT_DIAG_DIR is required when diagnostics are enabled")
        with _lock:
            key = (os.getpid(), "__total_records__")
            maximum = _limit("MINICPMO_ROLLOUT_DIAG_MAX_RECORDS", 2000, 100000)
            if _counts[key] >= maximum:
                return
            _counts[key] += 1
            host = socket.gethostname()
            record = {"event": event, "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "host": host, "pid": os.getpid(), "thread": threading.get_ident(),
                      "fields": summarize(fields)}
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            safe_host = "".join(c if c.isalnum() or c in "-_." else "_" for c in host)
            with (path / f"trace-{safe_host}-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
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
