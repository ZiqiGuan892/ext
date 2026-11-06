"""Timing probes preserve execution, request isolation, and honest aggregation."""

import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_huanyu_gen_timing_cpu", ROOT / "gen_timing.py")
timing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timing
SPEC.loader.exec_module(timing)

ENGINE_PACKAGE = "_huanyu_engine_timing_cpu"
package = ModuleType(ENGINE_PACKAGE)
package.__path__ = []
sys.modules[ENGINE_PACKAGE] = package
sys.modules[ENGINE_PACKAGE + ".gen_timing"] = timing
ENGINE_SPEC = importlib.util.spec_from_file_location(
    ENGINE_PACKAGE + ".gen_timing_engine", ROOT / "gen_timing_engine.py")
engine = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = engine
ENGINE_SPEC.loader.exec_module(engine)


def records(output):
    return [json.loads(line.split(timing.PREFIX, 1)[1])
            for line in output.getvalue().splitlines() if timing.PREFIX in line]


class TimingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"HUANYU_GEN_TIMING": "1"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        timing._reported.clear()
        context = timing._context.set(None)
        self.addCleanup(timing._context.reset, context)

    def test_disabled_installer_does_not_import_frameworks_and_wrapper_returns_original(self):
        result = object()

        class Worker:
            def call(self, payload):
                return payload

        timing.wrap_method(Worker, "call", "call")
        with patch.dict(os.environ, {"HUANYU_GEN_TIMING": "0"}):
            with patch.object(timing.importlib, "import_module", side_effect=AssertionError("imported")):
                with redirect_stdout(io.StringIO()) as output:
                    timing.install()
                    self.assertIs(Worker().call(result), result)
                    with timing.span("disabled"):
                        pass
        self.assertEqual(output.getvalue(), "")

    def test_scoped_wrapper_only_records_when_parent_context_exists(self):
        class Worker:
            def call(self, data):
                return data

        timing.wrap_method(Worker, "call", "child", require_context=True)
        value = object()
        with redirect_stdout(io.StringIO()) as output:
            self.assertIs(Worker().call(value), value)
        self.assertEqual(output.getvalue(), "")
        with redirect_stdout(io.StringIO()) as output:
            with timing.span("parent", step=7):
                self.assertIs(Worker().call(value), value)
        child, parent = records(output)
        self.assertEqual(child["parent_span_id"], parent["span_id"])
        self.assertEqual(child["step"], 7)
        self.assertTrue(child["inclusive"])
        self.assertEqual(child["clock"], "host_wall")
        self.assertEqual(timing.get_context(), {})

    def test_nested_wall_intervals_use_monotonic_durations(self):
        with patch.object(timing.time, "perf_counter", side_effect=[100.0, 101.0, 103.0, 105.0]):
            with patch.object(timing.time, "time_ns", side_effect=[1000, 2000, 500, 400]):
                with redirect_stdout(io.StringIO()) as output:
                    with timing.span("parent", uid="a"):
                        with timing.span("child"):
                            pass
        child, parent = records(output)
        self.assertEqual(child["elapsed_s"], 2.0)
        self.assertEqual(parent["elapsed_s"], 5.0)
        self.assertEqual(child["uid"], "a")
        self.assertEqual(child["parent_span_id"], parent["span_id"])
        self.assertIsNone(parent["parent_span_id"])

    def test_async_concurrent_trajectories_keep_separate_identifiers(self):
        async def exercise():
            all_started = asyncio.Event()
            arrived = 0
            observations = []

            class Worker:
                async def trajectory(self, sampling_params, trajectory, **kwargs):
                    nonlocal arrived
                    arrived += 1
                    if arrived == 2:
                        all_started.set()
                    await asyncio.wait_for(all_started.wait(), timeout=2)
                    with timing.span("child"):
                        await asyncio.sleep(0)
                        observations.append(timing.get_context())
                    return kwargs["payload"]

            timing.wrap_method(Worker, "trajectory", "agent.trajectory")
            self.assertTrue(asyncio.iscoroutinefunction(Worker.trajectory))
            values = [object(), object()]
            worker = Worker()
            returned = await asyncio.gather(*[
                worker.trajectory({}, {"step": 7}, uid=uid, session_id=i,
                                  global_steps=7, payload=values[i])
                for i, uid in enumerate(("first", "second"))
            ])
            self.assertTrue(all(left is right for left, right in zip(returned, values)))
            self.assertEqual({(o["uid"], o["session_id"]) for o in observations},
                             {("first", 0), ("second", 1)})
            self.assertEqual(timing.get_context(), {})

        with redirect_stdout(io.StringIO()) as output:
            asyncio.run(exercise())
        spans = records(output)
        roots = {row["span_id"]: row for row in spans if row["phase"] == "agent.trajectory"}
        self.assertEqual(len(roots), 2)
        for row in spans:
            self.assertEqual(row["step"], 7)
            if row["phase"] == "child":
                self.assertEqual(row["uid"], roots[row["parent_span_id"]]["uid"])
        self.assertNotIn("payload", output.getvalue())

    def test_sync_exception_object_and_context_survive_logging_failure(self):
        error = RuntimeError("business exception")

        class Worker:
            def call(self):
                raise error

        timing.wrap_method(Worker, "call", "call")
        with patch("builtins.print", side_effect=OSError("closed output")):
            with self.assertRaises(RuntimeError) as raised:
                Worker().call()
        self.assertIs(raised.exception, error)
        self.assertEqual(timing.get_context(), {})

    def test_explicit_async_wrapper_waits_for_sync_returned_awaitable(self):
        async def exercise(output):
            started = asyncio.Event()
            release = asyncio.Event()
            observed = []
            result = object()

            class Worker:
                def decorated(self, *, uid, session_id):
                    async def original_coroutine():
                        observed.append(timing.get_context())
                        started.set()
                        await release.wait()
                        observed.append(timing.get_context())
                        return result

                    return original_coroutine()

            self.assertFalse(asyncio.iscoroutinefunction(Worker.decorated))
            timing.wrap_method(Worker, "decorated", "agent.queue_put", asynchronous=True)
            with timing.span("parent", uid="outer", step=7):
                parent = timing.get_context()
                task = asyncio.create_task(Worker().decorated(uid="inner", session_id="session"))
                try:
                    await asyncio.wait_for(started.wait(), timeout=2)
                    self.assertFalse(task.done())
                    self.assertEqual(records(output), [], "span ended before the awaitable completed")
                    self.assertEqual(timing.get_context(), parent)
                finally:
                    release.set()
                    returned = await task
                self.assertIs(returned, result)
                child, = records(output)
                self.assertEqual(child["parent_span_id"], parent["span_id"])
                self.assertEqual(child["elapsed_s"], 7.0)
                self.assertEqual(timing.get_context(), parent)
            self.assertEqual(observed[0], observed[1])
            self.assertEqual(observed[0]["uid"], "inner")
            self.assertEqual(observed[0]["session_id"], "session")
            self.assertEqual(observed[0]["step"], 7)
            self.assertEqual(timing.get_context(), {})

        with patch.object(timing.time, "perf_counter", side_effect=[100.0, 101.0, 108.0, 109.0]):
            with redirect_stdout(io.StringIO()) as output:
                asyncio.run(exercise(output))
        child, parent = records(output)
        self.assertEqual(child["phase"], "agent.queue_put")
        self.assertEqual(child["status"], "ok")
        self.assertEqual(parent["phase"], "parent")

    def test_reserved_metadata_does_not_override_measured_status_or_hide_exception(self):
        error = ValueError("original failure")
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(ValueError) as raised:
                with timing.span("reserved", clock="wrong", elapsed_s=-1, status="wrong", event="wrong"):
                    raise error
        self.assertIs(raised.exception, error)
        event, = records(output)
        self.assertEqual(event["event"], "span")
        self.assertEqual(event["clock"], "host_wall")
        self.assertEqual(event["status"], "error")
        self.assertGreaterEqual(event["elapsed_s"], 0)

    def test_async_cancellation_is_rethrown_and_recorded(self):
        async def exercise():
            started = asyncio.Event()

            class Worker:
                async def call(self, *, uid):
                    started.set()
                    await asyncio.Event().wait()

            timing.wrap_method(Worker, "call", "cancelled")
            task = asyncio.create_task(Worker().call(uid="cancel-me"))
            done, _ = await asyncio.wait((task, asyncio.create_task(started.wait())),
                                         timeout=2, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                await task
            self.assertTrue(started.is_set(), "wrapped call never reached its original coroutine")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(timing.get_context(), {})

        with redirect_stdout(io.StringIO()) as output:
            asyncio.run(exercise())
        event, = records(output)
        self.assertEqual(event["error_type"], "CancelledError")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["uid"], "cancel-me")

    def test_ray_modified_class_is_patched_once_and_missing_target_is_retryable(self):
        name = "_fake_huanyu_timing_actor"
        fake = ModuleType(name)
        fake.Actor = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=None))
        methods = (("call", "ray.call", False),)
        with patch.dict(sys.modules, {name: fake}):
            with redirect_stdout(io.StringIO()) as output:
                timing._bind(name, "Actor", methods)

                class Worker:
                    def call(self, *, uid):
                        return uid

                fake.Actor.__ray_metadata__.modified_class = Worker
                timing._bind(name, "Actor", methods)
                first = Worker.call
                timing._bind(name, "Actor", methods)
                self.assertIs(Worker.call, first)
                self.assertEqual(Worker().call(uid="one"), "one")
        observed = records(output)
        self.assertEqual(len([row for row in observed if row["event"] == "span"]), 1)
        self.assertEqual([row["status"] for row in observed if row["event"] == "hook"],
                         ["unavailable", "installed"])

    def test_generator_methods_are_not_mistaken_for_completed_execution(self):
        class Worker:
            def stream(self):
                yield 1

            async def async_stream(self):
                yield 1

        original = Worker.stream
        original_async = Worker.async_stream
        self.assertFalse(timing.wrap_method(Worker, "stream", "stream"))
        self.assertFalse(timing.wrap_method(Worker, "async_stream", "stream"))
        self.assertIs(Worker.stream, original)
        self.assertIs(Worker.async_stream, original_async)

    def test_wrapped_method_cloudpickle_does_not_capture_contextvar(self):
        try:
            import cloudpickle
        except ImportError:
            self.skipTest("cloudpickle not installed")

        class Worker:
            def call(self, value):
                return value + 1

        timing.wrap_method(Worker, "call", "serialized")
        restored = cloudpickle.loads(cloudpickle.dumps(Worker.call))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(restored(Worker(), 41), 42)
        event, = records(output)
        self.assertEqual(event["phase"], "serialized")

    def test_summary_retains_failures_and_does_not_assign_engine_rows_a_step(self):
        def line(**fields):
            return "(Worker pid=1) " + timing.PREFIX + json.dumps(dict(event="span", **fields)) + "\n"

        lines = [
            "unrelated log\n", timing.PREFIX + "not-json\n",
            line(phase="request", step=7, elapsed_s=2.0, status="ok"),
            line(phase="request", step=7, elapsed_s=4.0, status="ok"),
            line(phase="request", step=8, elapsed_s=100.0, status="ok"),
            line(phase="request", step=7, elapsed_s=1.5, status="error"),
            line(phase="engine", elapsed_s=0.25, status="ok"),
        ]
        report = timing.summarize(lines, step=7)
        self.assertEqual(report[("request", "host_wall", "ok")], [2, 6.0, 4.0])
        self.assertEqual(report[("request", "host_wall", "error")], [1, 1.5, 1.5])
        self.assertEqual(len(report), 2)
        self.assertIn(("engine", "host_wall", "ok"), timing.summarize(lines))


def make_runner():
    result = object()

    class Model:
        def forward(self, payload):
            return payload

        def compute_logits(self, output):
            return output

    class Runner:
        def __init__(self):
            self.model = Model()
            self.local_rank = 0

        def _execute_mm_encoder(self, scheduler_output):
            return scheduler_output

        def _preprocess(self, scheduler_output):
            return self._execute_mm_encoder(scheduler_output)

        def execute_model(self, scheduler_output):
            self._preprocess(scheduler_output)
            return self.model.compute_logits(self.model.forward(result))

        def sample_tokens(self):
            return result

    return Runner, result


class EngineTimingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"HUANYU_GEN_TIMING": "1", "HUANYU_GEN_TIMING_ENGINE_EVERY": "3"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        context = engine._ACTIVE.set(None)
        self.addCleanup(engine._ACTIVE.reset, context)
        engine._NOTICES.clear()
        core_context = timing._context.set(None)
        self.addCleanup(timing._context.reset, core_context)
        self.scheduler = SimpleNamespace(num_scheduled_tokens={"opaque-request-id": 1},
                                         total_num_scheduled_tokens=1, scheduled_encoder_inputs={})

    def test_first_and_every_n_iterations_measure_real_calls_and_keep_returns(self):
        cls, result = make_runner()
        with redirect_stdout(io.StringIO()) as output:
            self.assertTrue(engine.bind_runner_class(cls))
            first_wrapper = cls.execute_model
            self.assertTrue(engine.bind_runner_class(cls))
            self.assertIs(cls.execute_model, first_wrapper)
            runner = cls()
            for _ in range(4):
                self.assertIs(runner.execute_model(self.scheduler), result)
                self.assertIs(runner.sample_tokens(), result)
        spans = [item for item in records(output) if item["event"] == "span"]
        for phase in ("engine.execute_model", "engine.preprocess", "engine.mm_encoder",
                      "engine.forward", "engine.compute_logits", "engine.sample_tokens"):
            selected = [item for item in spans if item["phase"] == phase]
            self.assertEqual([item["engine_iteration"] for item in selected], [1, 3], phase)
        roots = {item["span_id"]: item for item in spans if item["phase"] == "engine.execute_model"}
        for item in spans:
            self.assertEqual(item["clock"], "host_wall")
            self.assertNotIn("step", item)
            if item["phase"] == "engine.forward":
                self.assertIn(item["parent_span_id"], roots)
        self.assertIsNone(engine._ACTIVE.get())

    def test_disabled_engine_does_not_import_backend_or_add_iteration_state(self):
        cls, result = make_runner()
        with redirect_stdout(io.StringIO()):
            engine.bind_runner_class(cls)
        original_import = importlib.import_module

        def no_backend(name, *args):
            if name != ENGINE_PACKAGE + ".gen_timing":
                raise AssertionError("unexpected backend import: " + name)
            return original_import(name, *args)

        with patch.dict(os.environ, {"HUANYU_GEN_TIMING": "0"}):
            with patch.object(engine.importlib, "import_module", side_effect=no_backend):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertFalse(engine.install_engine())
                    runner = cls()
                    self.assertIs(runner.execute_model(self.scheduler), result)
                    self.assertIs(runner.sample_tokens(), result)
        self.assertEqual(output.getvalue(), "")
        self.assertFalse(hasattr(runner, engine._ITERATION_ATTR))

    def test_failure_resets_sampling_context_and_preserves_exception(self):
        cls, result = make_runner()
        error = RuntimeError("device failure")
        with redirect_stdout(io.StringIO()) as output:
            engine.bind_runner_class(cls)
            runner = cls()

            def broken_forward(payload):
                raise error

            runner.model.forward = broken_forward
            with self.assertRaises(RuntimeError) as raised:
                runner.execute_model(self.scheduler)
            self.assertIs(raised.exception, error)
            self.assertIsNone(engine._ACTIVE.get())
            self.assertIsNone(getattr(runner, engine._LAST_ATTR))
            self.assertIs(runner.sample_tokens(), result)
        spans = [item for item in records(output) if item["event"] == "span"]
        root, = [item for item in spans if item["phase"] == "engine.execute_model"]
        self.assertEqual(root["status"], "error")
        self.assertFalse(any(item["phase"] == "engine.sample_tokens" for item in spans))

    def test_scheduler_metadata_never_reads_tensor_values(self):
        class DeviceValue:
            def __int__(self):
                raise AssertionError("synchronized device value")

            def __str__(self):
                raise AssertionError("stringified device value")

        cls, result = make_runner()
        scheduler = SimpleNamespace(num_scheduled_tokens={"request": DeviceValue()},
                                    total_num_scheduled_tokens=DeviceValue(), scheduled_encoder_inputs={})
        with redirect_stdout(io.StringIO()) as output:
            engine.bind_runner_class(cls)
            self.assertIs(cls().execute_model(scheduler), result)
        root, = [item for item in records(output)
                 if item["event"] == "span" and item["phase"] == "engine.execute_model"]
        self.assertEqual(root["scheduled_requests"], 1)
        self.assertNotIn("scheduled_tokens", root)

    def test_unsupported_runner_is_reported_without_wrapping(self):
        class Runner:
            def execute_model(self, other_argument):
                return other_argument

        original = Runner.execute_model
        with redirect_stdout(io.StringIO()) as output:
            self.assertFalse(engine.bind_runner_class(Runner))
        self.assertIs(Runner.execute_model, original)
        self.assertTrue(any(item["event"] == "engine_hook_unavailable" for item in records(output)))


if __name__ == "__main__":
    unittest.main()
