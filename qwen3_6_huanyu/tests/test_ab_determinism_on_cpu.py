"""Exercise native worker initialization order and failed determinism settings."""

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
spec = importlib.util.spec_from_file_location("ab_determinism_tested", ROOT / "ab_determinism.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
ENV = {"HUANYU_TEXT_AB_DETERMINISM": "1", "HUANYU_TEXT_AB_SIDE": "serve",
       "HUANYU_TEXT_AB_BATCH_INVARIANT": "0", "VLLM_BATCH_INVARIANT": "0",
       "HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1",
       "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1"}


class ABDeterminismTests(unittest.TestCase):
    def exercise(self, *, env=None, reset=None, original_error=False, batch=False):
        state = NS(enabled=False, warn=True, calls=[])
        native = NS(_batch_invariant_LIB=None)

        def enable(value, *, warn_only):
            state.calls.append("enable")
            state.enabled, state.warn = value, warn_only

        torch = NS(use_deterministic_algorithms=enable,
                   are_deterministic_algorithms_enabled=lambda: state.enabled,
                   is_deterministic_algorithms_warn_only_enabled=lambda: state.warn)
        test = self

        class Worker:
            rank = 2

            def init_device(self, marker):
                test.assertTrue(state.enabled)
                test.assertFalse(state.warn)
                state.calls.append("native")
                if original_error:
                    raise RuntimeError("native initialization failed")
                if batch:
                    os.environ.update(VLLM_BATCH_INVARIANT="1", HCCL_DETERMINISTIC="strict")
                    native._batch_invariant_LIB = object()
                if reset == "torch":
                    state.enabled = False
                elif reset == "warn":
                    state.warn = True
                elif reset == "env":
                    os.environ["LCCL_DETERMINISTIC"] = "0"
                elif reset == "registration":
                    native._batch_invariant_LIB = object()
                return marker

        output = io.StringIO()
        environment = dict(ENV)
        if batch:
            environment.update(HUANYU_TEXT_AB_BATCH_INVARIANT="1", HCCL_DETERMINISTIC="true")
        environment.update(env or {})
        with patch.dict(os.environ, environment), patch.dict(sys.modules, {
                "torch": torch, "vllm_ascend.batch_invariant": native}), \
                patch.object(check.importlib, "import_module", return_value=NS(NPUARWorker=Worker)), \
                redirect_stdout(output):
            self.assertTrue(check.install_ab_determinism_check())
            wrapped = Worker.init_device
            self.assertTrue(check.install_ab_determinism_check())
            self.assertIs(wrapped, Worker.init_device)
            error = None
            try:
                self.assertEqual(Worker().init_device("result"), "result")
            except RuntimeError as exc:
                error = str(exc)
        return state.calls, output.getvalue(), error

    def test_serve_and_rollout_enable_before_native_init_and_verify_after(self):
        for side in ("serve", "rollout"):
            calls, output, error = self.exercise(env={"HUANYU_TEXT_AB_SIDE": side})
            self.assertIsNone(error)
            self.assertEqual(calls, ["enable", "native"])
            self.assertIn("scope=ab_worker_init status=SETTINGS_VERIFIED", output)
            record = json.loads(output[output.index("{"):])
            self.assertEqual(record["side"], side)
            self.assertTrue(record["torch_deterministic_algorithms"])
            self.assertFalse(record["torch_deterministic_warn_only"])
            self.assertFalse(record["batch_invariant_registered"])

    def test_missing_flags_stop_before_backend_initialization(self):
        for key in ("HCCL_DETERMINISTIC", "NCCL_DETERMINISTIC", "LCCL_DETERMINISTIC",
                    "ASCEND_LAUNCH_BLOCKING", "VLLM_BATCH_INVARIANT", "HUANYU_TEXT_AB_SIDE"):
            calls, output, error = self.exercise(env={key: ""})
            self.assertEqual(calls, [])
            self.assertIn(key, error)
            self.assertIn("status=FAILED", output)

    def test_native_resets_are_not_silently_repaired(self):
        for reset in ("torch", "warn", "env", "registration"):
            calls, output, error = self.exercise(reset=reset)
            self.assertIsNotNone(error)
            self.assertEqual(calls, ["enable", "native"])
            self.assertNotIn("SETTINGS_VERIFIED", output)

    def test_existing_batch_invariant_native_transition_remains_supported(self):
        calls, output, error = self.exercise(batch=True)
        self.assertIsNone(error)
        self.assertIn('"batch_invariant_registered": true', output)

    def test_original_failure_propagates_without_success_log(self):
        _, output, error = self.exercise(original_error=True)
        self.assertEqual(error, "native initialization failed")
        self.assertNotIn("SETTINGS_VERIFIED", output)

    def test_normal_training_and_repeat_do_not_install_ab_hook(self):
        with patch.dict(os.environ, {"HUANYU_TEXT_AB_DETERMINISM": "0"}), \
                patch.object(check.importlib, "import_module") as load:
            self.assertFalse(check.install_ab_determinism_check())
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
