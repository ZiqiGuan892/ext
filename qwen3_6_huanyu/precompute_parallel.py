"""Run independent, row-sharded Huanyu precomputations on explicitly chosen devices."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


TAIL_BYTES = 64 * 1024
WORKER_SCRIPT = Path(__file__).with_name("precompute.py")


class TerminationRequested(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _device(value):
    if value == "cpu":
        return value
    match = re.fullmatch(r"(npu|cuda):([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("Device must be npu:N, cuda:N, or cpu")
    return f"{match[1]}:{int(match[2])}"


def _positive_interval(value):
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("Status interval must be a finite positive number")
    return interval


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="All remaining arguments are forwarded unchanged to precompute.py.",
    )
    parser.add_argument("--devices", nargs="+", type=_device, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--status-interval", type=_positive_interval, default=10.0)
    args, forwarded = parser.parse_known_args(argv)
    if len(set(args.devices)) != len(args.devices):
        parser.error("Each device may be assigned to only one worker")
    reserved = {"--device", "--num-shards", "--shard-index"}
    if any(arg.split("=", 1)[0] in reserved for arg in forwarded):
        parser.error("--device, --num-shards and --shard-index are assigned by this launcher")
    return args, forwarded


def worker_command(device, rank, world_size, forwarded, worker_script=WORKER_SCRIPT):
    return [sys.executable, "-u", str(worker_script), *forwarded,
            "--device", device, "--num-shards", str(world_size), "--shard-index", str(rank)]


def read_log_tail(path, max_bytes=TAIL_BYTES):
    """Read at most a fixed-size tail, including while a worker appends to it."""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - max_bytes))
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def log_progress(path):
    tail = read_log_tail(path)
    matches = list(re.finditer(r"^\[huanyu-precompute\]\s+(\d+)/(\d+)\s+rows\b", tail, re.MULTILINE))
    if matches:
        done, total = map(int, matches[-1].groups())
    else:
        startup = list(re.finditer(
            r"^\[huanyu-precompute\]\s+mode=\S+\s+rows=(\d+)\s+input_rows=", tail, re.MULTILINE,
        ))
        done, total = (0, int(startup[-1][1])) if startup else (None, None)
    markers = list(re.finditer(r"^\[huanyu-precompute\] COMPLETE\s+([^\r\n]+)", tail, re.MULTILINE))
    complete_rows = None
    summary = None
    if markers:
        try:
            payload, _ = json.JSONDecoder().raw_decode(markers[-1][1])
            if isinstance(payload, dict):
                summary = payload
                if type(payload.get("rows")) is int and payload["rows"] >= 0:
                    complete_rows = payload["rows"]
        except ValueError:
            pass
    if summary is not None and type(summary.get("assigned_rows")) is int and summary["assigned_rows"] >= 0:
        total = summary["assigned_rows"]
    complete_valid = bool(markers) and total is not None and complete_rows == total
    if complete_valid:
        done = complete_rows
    return {"done": done, "total": total, "complete": bool(markers),
            "complete_rows": complete_rows, "complete_valid": complete_valid, "summary": summary}


@dataclass
class Worker:
    rank: int
    device: str
    log_path: Path
    process: object


def completion_errors(workers):
    errors = []
    summaries = []
    for worker in workers:
        status = log_progress(worker.log_path)
        summary = status["summary"]
        fields = ("rows", "assigned_rows", "input_rows", "shard_index", "num_shards")
        if (not status["complete_valid"] or not isinstance(summary, dict)
                or any(type(summary.get(field)) is not int for field in fields)):
            errors.append(f"rank={worker.rank}: COMPLETE JSON rows/shard metadata missing or inconsistent; log={worker.log_path}")
            continue
        expected = max(0, (summary["input_rows"] + len(workers) - 1 - worker.rank) // len(workers))
        if (summary["input_rows"] < 0 or summary["num_shards"] != len(workers)
                or summary["shard_index"] != worker.rank or summary["assigned_rows"] != expected):
            errors.append(f"rank={worker.rank}: COMPLETE shard assignment differs from launcher; log={worker.log_path}")
            continue
        summaries.append(summary)
    if len(summaries) == len(workers):
        input_counts = {summary["input_rows"] for summary in summaries}
        if len(input_counts) != 1 or sum(summary["rows"] for summary in summaries) != summaries[0]["input_rows"]:
            errors.append("COMPLETE summaries disagree on input rows or do not cover the full input")
    return errors


def stop_children(processes, timeout=15.0):
    """Stop only this launcher's children; bound both graceful and forced waits."""
    remaining = [process for process in processes if process.poll() is None]
    for process in remaining:
        try:
            process.terminate()
        except OSError as error:
            if not isinstance(error, ProcessLookupError):
                print(f"[huanyu-parallel] WARNING terminate pid={process.pid}: {error}", file=sys.stderr)
    deadline = time.monotonic() + timeout
    for process in remaining:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    remaining = [process for process in remaining if process.poll() is None]
    for process in remaining:
        try:
            process.kill()
        except OSError as error:
            if not isinstance(error, ProcessLookupError):
                print(f"[huanyu-parallel] WARNING kill pid={process.pid}: {error}", file=sys.stderr)
    deadline = time.monotonic() + timeout
    for process in remaining:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"[huanyu-parallel] WARNING child pid={process.pid} did not exit after kill", file=sys.stderr)


def _status(workers, codes, elapsed, stream):
    statuses = [log_progress(worker.log_path) for worker in workers]
    known = [status for status in statuses if status["total"] is not None and status["done"] is not None]
    done = sum(status["done"] for status in known)
    total = sum(status["total"] for status in known)
    rows = f"{done}/{total}" if known else "pending"
    detail = " ".join(
        f"{worker.rank}:{status['done']}/{status['total']}"
        if status["total"] is not None and status["done"] is not None else f"{worker.rank}:pending"
        for worker, status in zip(workers, statuses, strict=True)
    )
    print(f"[huanyu-parallel] active={sum(code is None for code in codes)} "
          f"finished={sum(code == 0 for code in codes)} "
          f"failed={sum(code is not None and code != 0 for code in codes)} "
          f"rows={rows} progress_workers={len(known)}/{len(workers)} "
          f"elapsed={elapsed:.0f}s workers=[{detail}]", file=stream, flush=True)


def run_workers(devices, forwarded, log_root, status_interval=10.0, *, popen=None,
                sleep=None, stream=None, worker_script=WORKER_SCRIPT):
    popen = subprocess.Popen if popen is None else popen
    sleep = time.sleep if sleep is None else sleep
    stream = sys.stdout if stream is None else stream
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    prefix = f"huanyu_precompute_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=log_root))
    print(f"[huanyu-parallel] logs={run_dir}", file=stream, flush=True)
    workers = []
    started = time.monotonic()
    worker_env = dict(os.environ)
    worker_env.setdefault("OMP_NUM_THREADS", "2")
    worker_env.setdefault("MKL_NUM_THREADS", "2")
    try:
        for rank, device in enumerate(devices):
            log_path = run_dir / f"worker_{rank}.log"
            command = worker_command(device, rank, len(devices), forwarded, worker_script)
            with log_path.open("xb") as handle:
                process = popen(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=worker_env)
            workers.append(Worker(rank, device, log_path, process))
            print(f"[huanyu-parallel] rank={rank} device={device} pid={process.pid} log={log_path}",
                  file=stream, flush=True)
        next_status = started
        while True:
            codes = [worker.process.poll() for worker in workers]
            failures = [(worker, code) for worker, code in zip(workers, codes, strict=True)
                        if code is not None and code != 0]
            if failures:
                for worker, code in failures:
                    print(f"[huanyu-parallel] FAILED rank={worker.rank} device={worker.device} "
                          f"exit={code} log={worker.log_path}", file=stream, flush=True)
                return 1
            if all(code == 0 for code in codes):
                errors = completion_errors(workers)
                if errors:
                    for error in errors:
                        print(f"[huanyu-parallel] FAILED exit=0 but {error}", file=stream, flush=True)
                    return 1
                _status(workers, codes, time.monotonic() - started, stream)
                print(f"[huanyu-parallel] COMPLETE workers={len(workers)} "
                      f"elapsed={time.monotonic() - started:.1f}s logs={run_dir}", file=stream, flush=True)
                return 0
            now = time.monotonic()
            if now >= next_status:
                _status(workers, codes, now - started, stream)
                next_status = now + status_interval
            sleep(min(0.2, max(0.0, next_status - time.monotonic())))
    finally:
        stop_children([worker.process for worker in workers])


@contextmanager
def termination_handlers():
    previous = {}
    requested = False

    def handle(signum, frame):
        nonlocal requested
        if requested:
            return  # Let the first signal's bounded child cleanup finish.
        requested = True
        raise TerminationRequested(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    args, forwarded = parse_cli(argv)
    try:
        with termination_handlers():
            return run_workers(args.devices, forwarded, args.log_dir, args.status_interval)
    except TerminationRequested as error:
        print(f"[huanyu-parallel] interrupted by signal {error.signum}; children stopped", file=sys.stderr)
        return 128 + error.signum
    except KeyboardInterrupt:
        print("[huanyu-parallel] interrupted; children stopped", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[huanyu-parallel] FAILED {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
