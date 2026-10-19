"""Multi-device launcher orchestration, bounded log reads and child cleanup."""

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_parallel_cpu_tests", ROOT / "precompute_parallel.py")
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def completed_log(rows=3, rank=0, world_size=1, input_rows=None):
    input_rows = rows * world_size if input_rows is None else input_rows
    summary = {"rows": rows, "created": 1, "assigned_rows": rows, "shard_index": rank,
               "num_shards": world_size, "input_rows": input_rows}
    return (f"[huanyu-precompute] mode=populate rows={rows} input_rows={input_rows} shard={rank}/{world_size}\n"
            f"[huanyu-precompute] {rows}/{rows} rows created=1\n"
            f'[huanyu-precompute] COMPLETE {json.dumps(summary)} elapsed=0.1s\n').encode()


def command_log(command):
    return completed_log(rank=int(command[-1]), world_size=int(command[-3]))


class FakeChild:
    def __init__(self, code=None, ignore_terminate=False):
        self.returncode = code
        self.pid = 1234
        self.terminated = False
        self.killed = False
        self.ignore_terminate = ignore_terminate
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake child", timeout)
        return self.returncode


class ParallelPrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()

    def parse(self, extra):
        return launcher.parse_cli(["--log-dir", str(self.root), *extra])

    def assert_invalid(self, extra):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parse(extra)
        self.assertEqual(error.exception.code, 2)

    def test_cli_forwards_worker_arguments_unchanged(self):
        forwarded = ["--parquet", "train file.parquet", "val.parquet", "--model_path", "/model",
                     "--cache_dir", "/cache", "--verify-only"]
        args, actual = self.parse(["--devices", "npu:0", "npu:3", "--status-interval", "0.25", *forwarded])
        self.assertEqual(args.devices, ["npu:0", "npu:3"])
        self.assertEqual(args.status_interval, 0.25)
        self.assertEqual(actual, forwarded)

    def test_duplicate_devices_and_invalid_intervals_are_rejected(self):
        for devices in (["cpu", "cpu"], ["npu:01", "npu:1"], ["cuda:0", "cuda:0"]):
            with self.subTest(devices=devices):
                self.assert_invalid(["--devices", *devices])
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                self.assert_invalid(["--devices", "cpu", "--status-interval", value])

    def test_launcher_owned_worker_options_are_rejected(self):
        for flag in ("--device", "--num-shards", "--shard-index"):
            for option in ([flag, "1"], [f"{flag}=1"]):
                with self.subTest(option=option):
                    self.assert_invalid(["--devices", "cpu", *option])

    def test_every_worker_has_a_unique_device_and_row_shard(self):
        devices = ["npu:0", "npu:2", "npu:5"]
        calls = []

        def popen(command, **kwargs):
            calls.append(command)
            kwargs["stdout"].write(command_log(command))
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            return FakeChild(0)

        result = launcher.run_workers(devices, ["--parquet", "train.parquet"], self.root,
                                      popen=popen, stream=self.output)
        self.assertEqual(result, 0)
        self.assertIn("[huanyu-parallel] COMPLETE", self.output.getvalue())
        self.assertEqual(len(calls), 3)
        for rank, command in enumerate(calls):
            self.assertEqual(command[:3], [sys.executable, "-u", str(launcher.WORKER_SCRIPT)])
            self.assertEqual(command[3:5], ["--parquet", "train.parquet"])
            self.assertEqual(command[-6:], ["--device", devices[rank], "--num-shards", "3", "--shard-index", str(rank)])

    def test_repeated_launches_create_distinct_log_directories(self):
        def popen(command, **kwargs):
            kwargs["stdout"].write(command_log(command))
            return FakeChild(0)
        for _ in range(2):
            self.assertEqual(launcher.run_workers(["cpu"], [], self.root, popen=popen, stream=self.output), 0)
        directories = list(self.root.iterdir())
        self.assertEqual(len(directories), 2)
        for directory in directories:
            self.assertTrue((directory / "worker_0.log").is_file())

    def test_actual_subprocess_workers_cover_uneven_input_and_write_logs(self):
        script = self.root / "cpu_worker.py"
        script.write_text(
            "import argparse, json\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--device')\n"
            "p.add_argument('--num-shards', type=int)\n"
            "p.add_argument('--shard-index', type=int)\n"
            "p.add_argument('--input-rows', type=int)\n"
            "a = p.parse_args()\n"
            "rows = len(range(a.shard_index, a.input_rows, a.num_shards))\n"
            "print('device=' + a.device, flush=True)\n"
            "print('[huanyu-precompute] COMPLETE ' + json.dumps(dict(\n"
            "    rows=rows, assigned_rows=rows, input_rows=a.input_rows,\n"
            "    shard_index=a.shard_index, num_shards=a.num_shards)), flush=True)\n",
            encoding="utf-8",
        )
        result = launcher.run_workers(
            ["npu:0", "npu:1", "npu:2"], ["--input-rows", "7"], self.root / "logs",
            worker_script=script, stream=self.output, status_interval=0.1,
        )
        self.assertEqual(result, 0)
        self.assertIn("rows=7/7", self.output.getvalue())
        logs = list((self.root / "logs").glob("*/worker_*.log"))
        self.assertEqual(len(logs), 3)
        for rank in range(3):
            path = next(path for path in logs if path.name == f"worker_{rank}.log")
            self.assertIn(f"device=npu:{rank}", path.read_text())
            self.assertEqual(launcher.log_progress(path)["complete_rows"], (7 + 2 - rank) // 3)

    def test_malformed_completion_does_not_break_progress_reporting(self):
        path = self.root / "worker.log"
        path.write_text('[huanyu-precompute] COMPLETE {"assigned_rows": 3}\n', encoding="utf-8")
        worker = launcher.Worker(0, "cpu", path, FakeChild())
        launcher._status([worker], [None], 1.0, self.output)
        self.assertIn("rows=pending", self.output.getvalue())
        self.assertTrue(launcher.completion_errors([worker]))

    def test_nonzero_exit_stops_other_children_and_never_reports_complete(self):
        children = [FakeChild(7), FakeChild()]
        pending = iter(children)

        def popen(command, **kwargs):
            kwargs["stdout"].write(command_log(command))
            return next(pending)

        result = launcher.run_workers(["npu:0", "npu:1"], [], self.root, popen=popen, stream=self.output)
        self.assertEqual(result, 1)
        self.assertTrue(children[1].terminated)
        self.assertFalse(children[0].terminated)
        self.assertNotIn("[huanyu-parallel] COMPLETE", self.output.getvalue())

    def test_zero_exit_requires_valid_complete_rows(self):
        logs = (b"ordinary output\n", completed_log().replace(b'"rows": 3', b'"rows": 2'),
                b"[huanyu-precompute] COMPLETE not-json\n")
        for log in logs:
            with self.subTest(log=log):
                def popen(command, **kwargs):
                    kwargs["stdout"].write(log)
                    return FakeChild(0)
                result = launcher.run_workers(["cpu"], [], self.root, popen=popen, stream=self.output)
                self.assertEqual(result, 1)
        self.assertNotIn("[huanyu-parallel] COMPLETE", self.output.getvalue())

    def test_spawn_failure_stops_previously_started_children(self):
        child = FakeChild()
        calls = 0

        def popen(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("spawn failed")
            return child

        with self.assertRaisesRegex(OSError, "spawn failed"):
            launcher.run_workers(["npu:0", "npu:1"], [], self.root, popen=popen, stream=self.output)
        self.assertTrue(child.terminated)

    def test_keyboard_interrupt_and_sigterm_stop_owned_children(self):
        for interruption in (KeyboardInterrupt(), launcher.TerminationRequested(signal.SIGTERM)):
            with self.subTest(interruption=type(interruption).__name__):
                child = FakeChild()
                def sleep(seconds):
                    raise interruption
                with self.assertRaises(type(interruption)):
                    launcher.run_workers(["cpu"], [], self.root, popen=lambda *a, **k: child,
                                         sleep=sleep, stream=self.output)
                self.assertTrue(child.terminated)

    def test_unresponsive_child_is_killed_after_bounded_grace_period(self):
        child = FakeChild(ignore_terminate=True)
        stopped = FakeChild(0)
        launcher.stop_children([child, stopped], timeout=0)
        self.assertTrue(child.terminated)
        self.assertTrue(child.killed)
        self.assertEqual(child.wait_timeouts, [0.0, 0.0])
        self.assertFalse(stopped.terminated)
        self.assertFalse(stopped.killed)

    def test_log_tail_is_bounded_and_uses_last_progress(self):
        path = self.root / "worker.log"
        path.write_bytes(b"x" * (launcher.TAIL_BYTES * 2) + b"\n"
                         b"[huanyu-precompute] 1/10 rows first\n"
                         b"[huanyu-precompute] 8/10 rows second\n")
        tail = launcher.read_log_tail(path)
        self.assertLessEqual(len(tail.encode()), launcher.TAIL_BYTES)
        progress = launcher.log_progress(path)
        self.assertEqual((progress["done"], progress["total"]), (8, 10))
        self.assertFalse(progress["complete"])

    def test_empty_shard_uses_startup_assignment_not_input_file_rows(self):
        path = self.root / "worker.log"
        path.write_bytes(b"[huanyu-precompute] mode=populate rows=0 input_rows=1 shard=1/2\n"
                         b"[huanyu-precompute] start /dataset/train.parquet rows=1\n"
                         b'[huanyu-precompute] COMPLETE {"rows": 0} elapsed=0.1s\n')
        progress = launcher.log_progress(path)
        self.assertEqual((progress["done"], progress["total"]), (0, 0))
        self.assertTrue(progress["complete_valid"])

    def test_completion_metadata_survives_startup_scrolling_out_of_tail(self):
        path = self.root / "worker.log"
        complete_line = completed_log(rows=3).splitlines()[-1] + b"\n"
        path.write_bytes(b"x" * (launcher.TAIL_BYTES * 2) + b"\n" + complete_line)
        worker = launcher.Worker(0, "cpu", path, FakeChild(0))
        self.assertEqual(launcher.completion_errors([worker]), [])
        status = launcher.log_progress(path)
        self.assertEqual((status["done"], status["total"]), (3, 3))

    def test_completion_rejects_wrong_rank_world_and_global_input_coverage(self):
        for rank, world, total in ((1, 1, 3), (0, 2, 6), (0, 1, 4)):
            with self.subTest(rank=rank, world=world, total=total):
                path = self.root / "worker.log"
                path.write_bytes(completed_log(rank=rank, world_size=world, input_rows=total))
                worker = launcher.Worker(0, "cpu", path, FakeChild(0))
                self.assertTrue(launcher.completion_errors([worker]))

        paths = [self.root / "worker_0.log", self.root / "worker_1.log"]
        paths[0].write_bytes(completed_log(rows=3, rank=0, world_size=2, input_rows=6))
        paths[1].write_bytes(completed_log(rows=4, rank=1, world_size=2, input_rows=8))
        workers = [launcher.Worker(rank, f"npu:{rank}", path, FakeChild(0)) for rank, path in enumerate(paths)]
        self.assertIn("disagree on input rows", launcher.completion_errors(workers)[0])

    def test_cpu_thread_defaults_preserve_explicit_user_values(self):
        environments = []

        def popen(command, **kwargs):
            environments.append(kwargs["env"])
            kwargs["stdout"].write(command_log(command))
            return FakeChild(0)

        with patch.dict(launcher.os.environ, {"OMP_NUM_THREADS": "7", "OTHER_SETTING": "keep"}, clear=True):
            self.assertEqual(launcher.run_workers(["cpu"], [], self.root, popen=popen, stream=self.output), 0)
            self.assertNotIn("MKL_NUM_THREADS", launcher.os.environ)
        self.assertEqual(environments[0]["OMP_NUM_THREADS"], "7")
        self.assertEqual(environments[0]["MKL_NUM_THREADS"], "2")
        self.assertEqual(environments[0]["OTHER_SETTING"], "keep")

    def test_signal_handlers_are_restored(self):
        original = {signal.SIGINT: object(), signal.SIGTERM: object()}
        installed = {}

        def replace(signum, handler):
            previous = installed.get(signum, original[signum])
            installed[signum] = handler
            return previous

        with patch.object(launcher.signal, "signal", side_effect=replace):
            with launcher.termination_handlers():
                with self.assertRaises(launcher.TerminationRequested):
                    installed[signal.SIGTERM](signal.SIGTERM, None)
                installed[signal.SIGINT](signal.SIGINT, None)  # Repeated signals cannot interrupt cleanup.
        self.assertEqual(installed, original)


if __name__ == "__main__":
    unittest.main()
