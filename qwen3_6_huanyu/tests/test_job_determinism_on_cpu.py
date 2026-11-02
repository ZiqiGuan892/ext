"""Verify opt-in Actor and rollout initialization without an NPU runtime."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("job_determinism_tested", ROOT / "job_determinism.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
ENV = dict(check._EXPECTED, HUANYU_JOB_DETERMINISM="1")


class JobDeterminismTests(unittest.TestCase):
    def exercise(self, *, actor=False, changes=None, reset=None):
        calls = []
        state = NS(enabled=False, warn=True)
        native = NS(_batch_invariant_LIB=None)

        def enable(value, *, warn_only):
            calls.append("enable")
            state.enabled, state.warn = value, warn_only

        torch = NS(use_deterministic_algorithms=enable,
                   are_deterministic_algorithms_enabled=lambda: state.enabled,
                   is_deterministic_algorithms_warn_only_enabled=lambda: state.warn)
        test = self

        class Worker:
            rank = 3

            def init_device(self, value):
                calls.append("native")
                test.assertTrue(state.enabled)
                test.assertFalse(state.warn)
                if reset == "torch":
                    state.enabled = False
                elif reset == "env":
                    os.environ["HCCL_DETERMINISTIC"] = "false"
                elif reset == "batch":
                    native._batch_invariant_LIB = object()
                elif reset == "exception":
                    raise RuntimeError("native failed")
                return value

        output = io.StringIO()
        with patch.dict(os.environ, dict(ENV, **(changes or {}))), \
                patch.dict(sys.modules, {"torch": torch, "vllm_ascend.batch_invariant": native}), \
                patch.object(check.importlib, "import_module", return_value=NS(NPUARWorker=Worker)) as imported, \
                redirect_stdout(output):
            error = None
            try:
                if actor:
                    enabled = check.configure_job_determinism()
                else:
                    enabled = check.install_job_determinism_check()
                    if enabled:
                        wrapped = Worker.init_device
                        check.install_job_determinism_check()
                        self.assertIs(wrapped, Worker.init_device)
                        self.assertEqual(Worker().init_device("result"), "result")
            except RuntimeError as exc:
                error = str(exc)
            if os.environ["HUANYU_JOB_DETERMINISM"] == "0":
                self.assertFalse(enabled)
                imported.assert_not_called()
        return calls, output.getvalue(), error

    def test_actor_ref_and_rollout_set_and_report_actual_torch_state(self):
        for actor in (False, True):
            calls, output, error = self.exercise(actor=actor)
            self.assertIsNone(error)
            self.assertEqual(calls, ["enable"] if actor else ["enable", "native"])
            self.assertIn("SETTINGS_VERIFIED", output)
            record = json.loads(output[output.index("{"):])
            self.assertTrue(record["torch_deterministic_algorithms"])
            self.assertEqual(record["environment"], check._EXPECTED)
            self.assertIn("actor_ref_model_config" if actor else "rollout_worker_init_rank_3", output)

    def test_missing_environment_fails_before_native_or_torch_state_changes(self):
        for key in check._EXPECTED:
            for actor in (False, True):
                calls, output, error = self.exercise(actor=actor, changes={key: ""})
                self.assertEqual(calls, [])
                self.assertIn(key, error)
                self.assertNotIn("SETTINGS_VERIFIED", output)

    def test_native_overrides_or_exceptions_do_not_report_success(self):
        for reset in ("torch", "env", "batch", "exception"):
            _, output, error = self.exercise(reset=reset)
            self.assertIsNotNone(error)
            self.assertNotIn("SETTINGS_VERIFIED", output)

    def test_disabled_policy_does_not_import_native_worker_or_change_torch(self):
        for actor in (False, True):
            calls, output, error = self.exercise(actor=actor, changes={"HUANYU_JOB_DETERMINISM": "0"})
            self.assertEqual((calls, output, error), ([], "", None))


if __name__ == "__main__":
    unittest.main()
