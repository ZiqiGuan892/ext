"""Validate full-policy propagation and actual CPUOffloadPolicy construction."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_policy_audit_cpu", ROOT / "precision_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
TARGET = "verl.workers.engine.fsdp.transformer_impl"


class PrecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        policy._REPORTED.clear()

    def full_environment(self):
        return {**dict.fromkeys(policy.FULL_ON, "1"), "MINICPMO_FSDP2_PIN_MEMORY": "0",
                "HUANYU_PRECISION_PROFILE": "full", "HUANYU_GRAPH_MODE": "0"}

    def test_full_requires_every_flag_and_does_not_claim_success_on_failure(self):
        expected = self.full_environment()
        for missing in (*policy.FULL_ON, "MINICPMO_FSDP2_PIN_MEMORY"):
            with self.subTest(missing=missing), patch.dict(os.environ, expected, clear=True):
                del os.environ[missing]
                with redirect_stdout(io.StringIO()) as output:
                    with self.assertRaisesRegex(RuntimeError, "not propagated to rollout"):
                        policy.report_policy("rollout")
                self.assertNotIn("phase=configured", output.getvalue())
        for key, value in (("MINICPMO_PRIVATE_HYBRID_CACHE", "0"),
                           ("MINICPMO_FSDP2_PIN_MEMORY", "1")):
            with self.subTest(key=key), patch.dict(os.environ, {**expected, key: value}, clear=True):
                with self.assertRaisesRegex(RuntimeError, key):
                    policy.report_policy("training")

    def test_full_eager_and_once_per_role_configuration(self):
        with patch.dict(os.environ, self.full_environment(), clear=True):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(policy.report_policy("training"), "full")
                policy.report_policy("training")
                policy.report_policy("rollout")
            self.assertEqual(output.getvalue().count("phase=configured role=training"), 1)
            self.assertEqual(output.getvalue().count("phase=configured role=rollout"), 1)
            for field in ("host=", "pid=", "source=", "MINICPMO_PRIVATE_HYBRID_CACHE=1"):
                self.assertIn(field, output.getvalue())
            os.environ["HUANYU_GRAPH_MODE"] = "1"
            with redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(RuntimeError, "require eager mode"):
                    policy.report_policy("rollout")
            self.assertNotIn("phase=configured", output.getvalue())

    def test_custom_allows_explicit_optouts_and_invalid_profile_fails(self):
        with patch.dict(os.environ, {"HUANYU_PRECISION_PROFILE": "custom", "HUANYU_GRAPH_MODE": "1",
                                     "MINICPMO_PRIVATE_HYBRID_CACHE": "0"}, clear=True):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(policy.report_policy("rollout"), "custom")
            self.assertIn("MINICPMO_PRIVATE_HYBRID_CACHE=0", output.getvalue())
        self.assertEqual(policy.report_policy("direct_python"), "custom")
        with patch.dict(os.environ, {"HUANYU_PRECISION_PROFILE": "ful"}):
            with self.assertRaisesRegex(ValueError, "Unknown HUANYU_PRECISION_PROFILE"):
                policy.report_policy("rollout")

    def test_reports_actual_worker_communication_environment_once(self):
        environment = {**self.full_environment(), "HCCL_OP_EXPANSION_MODE": "AI_CPU",
                       "HUANYU_HCCL_OP_EXPANSION_MODE": "auto", "RANK": "14",
                       "WORLD_SIZE": "24", "LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "12"}
        with patch.dict(os.environ, environment, clear=True), redirect_stdout(io.StringIO()) as output:
            policy.report_policy("actor/ref")
            policy.report_policy("actor/ref")
            self.assertEqual(output.getvalue().count("[huanyu-communication]"), 1)
            for field in ("HCCL_OP_EXPANSION_MODE=AI_CPU", "RANK=14", "WORLD_SIZE=24",
                          "LOCAL_WORLD_SIZE=12", "ASCEND_RT_VISIBLE_DEVICES=<unset>"):
                self.assertIn(field, output.getvalue())

    def test_actual_policy_forces_positional_and_keyword_pin_false_once(self):
        target = SimpleNamespace(CPUOffloadPolicy=CPUOffloadPolicy)
        with patch.dict(sys.modules, {TARGET: target}):
            with redirect_stdout(io.StringIO()) as installed:
                self.assertTrue(policy.install_fsdp_pin_audit())
                wrapper = target.CPUOffloadPolicy
                self.assertTrue(policy.install_fsdp_pin_audit())
            self.assertIs(wrapper, target.CPUOffloadPolicy)
            self.assertEqual(installed.getvalue().count("phase=installed"), 1)
            self.assertNotIn("phase=executed", installed.getvalue())
            with redirect_stdout(io.StringIO()) as executed:
                for result in (target.CPUOffloadPolicy(True), target.CPUOffloadPolicy(pin_memory=True),
                               target.CPUOffloadPolicy()):
                    self.assertIsInstance(result, CPUOffloadPolicy)
                    self.assertIs(result.pin_memory, False)
            self.assertEqual(executed.getvalue().count("phase=executed"), 1)

    def test_existing_shared_factory_cls_uses_real_class_signature(self):
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return CPUOffloadPolicy(*args, **kwargs)

        factory.cls = CPUOffloadPolicy
        target = SimpleNamespace(CPUOffloadPolicy=factory)
        with patch.dict(sys.modules, {TARGET: target}):
            policy.install_fsdp_pin_audit()
            self.assertIs(target.CPUOffloadPolicy.cls, CPUOffloadPolicy)
            self.assertIs(target.CPUOffloadPolicy(True).pin_memory, False)
            self.assertEqual(calls, [((False,), {})])

    def test_pin_policy_result_is_checked_and_failure_never_reports_executed(self):
        def faulty_factory(pin_memory=True):
            return CPUOffloadPolicy(pin_memory=True)

        target = SimpleNamespace(CPUOffloadPolicy=faulty_factory)
        with patch.dict(sys.modules, {TARGET: target}):
            policy.install_fsdp_pin_audit()
            with redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(RuntimeError, "did not preserve pin_memory=False"):
                    target.CPUOffloadPolicy()
            self.assertNotIn("phase=executed", output.getvalue())

    def test_without_offload_selection_no_cpu_policy_is_created_or_execution_claimed(self):
        offload_disabled = OffloadPolicy()
        target = SimpleNamespace(CPUOffloadPolicy=CPUOffloadPolicy, OffloadPolicy=OffloadPolicy)
        with patch.dict(sys.modules, {TARGET: target}), redirect_stdout(io.StringIO()) as output:
            policy.install_fsdp_pin_audit()
            selected = target.OffloadPolicy()
        self.assertIsInstance(selected, type(offload_disabled))
        self.assertNotIsInstance(selected, CPUOffloadPolicy)
        self.assertIs(target.OffloadPolicy, OffloadPolicy)
        self.assertIn("offload_selection=unchanged", output.getvalue())
        self.assertNotIn("phase=executed", output.getvalue())

    def test_pin_optout_imports_nothing_and_unsupported_signature_fails(self):
        with patch.dict(os.environ, {"MINICPMO_FSDP2_PIN_MEMORY": "1"}), \
                patch.object(policy.importlib, "import_module", side_effect=AssertionError("unexpected import")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertFalse(policy.install_fsdp_pin_audit())
        self.assertIn("phase=disabled", output.getvalue())
        target = SimpleNamespace(CPUOffloadPolicy=lambda: object())
        with patch.dict(sys.modules, {TARGET: target}), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "Unsupported FSDP"):
                policy.install_fsdp_pin_audit()
        self.assertNotIn("phase=installed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
