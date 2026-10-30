"""Observe native initialization and refuse skipped setup, without tensor IO."""

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
spec = importlib.util.spec_from_file_location("repeat_determinism_tested", ROOT / "repeat_determinism.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
ENV = {"HUANYU_SERVE_REPEAT_DETERMINISM": "1", "VLLM_BATCH_INVARIANT": "1",
       "HCCL_DETERMINISTIC": "strict", "LCCL_DETERMINISTIC": "1",
       "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1"}


class RepeatDeterminismTests(unittest.TestCase):
    def exercise(self, *, missing=False, skip=False, torch_enabled=True, env_override=None, original_error=False):
        native = NS(_batch_invariant_LIB=None)
        state = NS(enabled=False, calls=0)
        torch = NS(are_deterministic_algorithms_enabled=lambda: state.enabled,
                   is_deterministic_algorithms_warn_only_enabled=lambda: True)

        class Worker:
            rank = 3

            def init_device(self, marker):
                state.calls += 1
                if original_error:
                    raise RuntimeError("native device failed")
                if not skip:
                    native._batch_invariant_LIB = object()
                state.enabled = torch_enabled
                return marker

        output = io.StringIO()
        modules = {"torch": torch, "vllm_ascend.batch_invariant": None if missing else native}
        with patch.dict(os.environ, {**ENV, **(env_override or {})}), patch.dict(sys.modules, modules), \
                patch.object(check.importlib, "import_module", return_value=NS(NPUARWorker=Worker)) as load, \
                redirect_stdout(output):
            self.assertTrue(check.install_repeat_determinism_check())
            wrapped = Worker.init_device
            self.assertTrue(check.install_repeat_determinism_check())
            self.assertIs(Worker.init_device, wrapped)
            error = None
            try:
                self.assertEqual(Worker().init_device("native-result"), "native-result")
            except RuntimeError as exc:
                error = str(exc)
            self.assertEqual(state.calls, 1)
            self.assertEqual([x.args[0] for x in load.call_args_list], [check._WORKER, check._WORKER])
        return output.getvalue(), error

    def test_post_init_verified_once_without_tensor_probe_import(self):
        output, error = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(output.count("[huanyu-determinism]"), 1)
        self.assertIn("actual scope=worker_init status=SETTINGS_VERIFIED", output)
        record = json.loads(output[output.index("{"):])
        self.assertEqual(record["rank"], 3)
        self.assertTrue(record["batch_invariant_registered"])
        self.assertTrue(record["torch_deterministic_algorithms"])
        self.assertTrue(record["torch_deterministic_warn_only"])
        self.assertEqual(record["errors"], [])

    def test_disabled_or_unknown_native_initialization_never_certified(self):
        for opts in ({"skip": True}, {"missing": True}, {"torch_enabled": False}):
            with self.subTest(opts=opts):
                output, error = self.exercise(**opts)
                self.assertIn("check failed", error)
                self.assertIn("status=FAILED", output)
                self.assertNotIn("SETTINGS_VERIFIED", output)

    def test_each_overridden_environment_flag_is_rejected(self):
        for name in ENV.keys() - {"HUANYU_SERVE_REPEAT_DETERMINISM"}:
            with self.subTest(name=name):
                _, error = self.exercise(env_override={name: "0"})
                self.assertIn(name + " must be", error)

    def test_original_startup_error_is_not_hidden_or_claimed_verified(self):
        output, error = self.exercise(original_error=True)
        self.assertEqual(error, "native device failed")
        self.assertEqual(output, "")

    def test_non_repeat_does_not_import_or_patch_native_worker(self):
        with patch.dict(os.environ, {"HUANYU_SERVE_REPEAT_DETERMINISM": ""}), \
                patch.object(check.importlib, "import_module") as load:
            self.assertFalse(check.install_repeat_determinism_check())
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
