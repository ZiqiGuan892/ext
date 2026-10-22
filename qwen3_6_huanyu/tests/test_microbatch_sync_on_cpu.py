"""Huanyu microbatch memory policy, tested without importing Torch or verl."""

from contextlib import contextmanager, nullcontext, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_microbatch_sync_cpu_tests", ROOT / "microbatch_sync.py")
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


class Model:
    def __init__(self):
        self.sync = True
        self.set_calls = []

    def set_requires_gradient_sync(self, value):
        self.set_calls.append(value)
        self.sync = value


def modern_engine_class():
    class Engine:
        def __init__(self, module):
            self.module = module
            self.original_calls = []

        @contextmanager
        def _gradient_sync_context(self, *, is_last_micro_batch):
            self.original_calls.append(is_last_micro_batch)
            self.module.set_requires_gradient_sync(is_last_micro_batch)
            try:
                yield
            finally:
                self.module.set_requires_gradient_sync(True)

    return Engine


def legacy_engine_class():
    class Engine:
        def __init__(self, module):
            self.module = module

        def forward_backward_batch(self, losses):
            ctx = nullcontext()
            for loss in losses:
                with ctx:
                    loss.backward()
            return "completed"

    return Engine


@contextmanager
def fake_engine(engine_class):
    impl = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.FSDPEngine = engine_class
    with patch.dict(sys.modules, {impl.__name__: impl}):
        yield impl


class MicrobatchSyncTests(unittest.TestCase):
    def setUp(self):
        policy._REPORTED.clear()

    def test_opt_in_keeps_sync_for_non_final_and_final_microbatches(self):
        engine_class = modern_engine_class()
        model = Model()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()) as output:
            self.assertIs(policy.install_huanyu_microbatch_sync(model), model)
            engine = engine_class(model)
            for last in (False, False, True):
                with engine._gradient_sync_context(is_last_micro_batch=last):
                    self.assertTrue(model.sync)
            self.assertEqual(engine.original_calls, [])
            self.assertEqual(model.set_calls, [True] * 6)
            self.assertEqual(output.getvalue().count("sync_each_microbatch=enabled"), 1)
            self.assertIn("mode=fsdp_context", output.getvalue())

    def test_unmarked_other_model_keeps_original_no_sync(self):
        engine_class = modern_engine_class()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()) as output:
            policy.install_huanyu_microbatch_sync(Model())
            engine = engine_class(Model())
            with engine._gradient_sync_context(is_last_micro_batch=False):
                self.assertFalse(engine.module.sync)
            self.assertTrue(engine.module.sync)
            self.assertEqual(engine.original_calls, [False])
            self.assertEqual(output.getvalue(), "")

    def test_disabled_policy_does_not_import_engine(self):
        model = Model()
        with patch.object(policy.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            policy.install_huanyu_microbatch_sync(model, enabled=False)
        self.assertIs(getattr(model, policy._MARKER), False)

    def test_policy_can_be_disabled_after_install_without_affecting_other_models(self):
        engine_class = modern_engine_class()
        model = Model()
        with fake_engine(engine_class):
            policy.install_huanyu_microbatch_sync(model)
            policy.install_huanyu_microbatch_sync(model, enabled=False)
            engine = engine_class(model)
            with engine._gradient_sync_context(is_last_micro_batch=False):
                self.assertFalse(model.sync)
            self.assertEqual(engine.original_calls, [False])

    def test_exception_restores_enabled_sync(self):
        engine_class = modern_engine_class()
        model = Model()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()):
            policy.install_huanyu_microbatch_sync(model)
            with self.assertRaisesRegex(RuntimeError, "backward failed"):
                with engine_class(model)._gradient_sync_context(is_last_micro_batch=False):
                    model.sync = False
                    raise RuntimeError("backward failed")
            self.assertTrue(model.sync)

    def test_idempotent_install_and_worker_wide_single_evidence_line(self):
        engine_class = modern_engine_class()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()) as output:
            first = Model()
            policy.install_huanyu_microbatch_sync(first)
            hooked = engine_class._gradient_sync_context
            second = Model()
            policy.install_huanyu_microbatch_sync(second)
            self.assertIs(engine_class._gradient_sync_context, hooked)
            for model in (first, second):
                with engine_class(model)._gradient_sync_context(is_last_micro_batch=False):
                    self.assertTrue(model.sync)
            self.assertEqual(output.getvalue().count("sync_each_microbatch=enabled"), 1)

    def test_mark_survives_fsdp_or_compile_wrapper(self):
        engine_class = modern_engine_class()
        model = Model()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()):
            policy.install_huanyu_microbatch_sync(model)
            wrapped = SimpleNamespace(_orig_mod=SimpleNamespace(_fsdp_wrapped_module=model))
            with engine_class(wrapped)._gradient_sync_context(is_last_micro_batch=False):
                self.assertTrue(model.sync)
            self.assertEqual(model.set_calls, [True, True])

    def test_config_fallback_requires_huanyu_audio_architecture_and_real_boolean(self):
        engine_class = modern_engine_class()
        with fake_engine(engine_class), redirect_stdout(io.StringIO()):
            policy.install_huanyu_microbatch_sync(Model())
            for audio_type, flag, should_enable in (
                ("huanyu_conformer_0_6b", True, True),
                ("whisper", True, False),
                ("huanyu_conformer_0_6b", "true", False),
            ):
                with self.subTest(audio_type=audio_type, flag=flag):
                    model = Model()
                    model.config = SimpleNamespace(
                        huanyu_sync_each_microbatch=flag, model_type="qwen3_5_moe",
                        audio_config={"model_type": audio_type},
                    )
                    engine = engine_class(model)
                    with engine._gradient_sync_context(is_last_micro_batch=False):
                        self.assertEqual(model.sync, should_enable)

    def test_unknown_wrapped_model_fails_before_entering_backward(self):
        engine_class = modern_engine_class()
        with fake_engine(engine_class):
            model = SimpleNamespace()
            policy.install_huanyu_microbatch_sync(model)
            with self.assertRaisesRegex(RuntimeError, "wrapped FSDP"):
                with engine_class(model)._gradient_sync_context(is_last_micro_batch=False):
                    self.fail("unverified backward must not run")

    def test_disabled_inner_marker_overrides_outer_copied_hf_config(self):
        model = Model()
        policy.install_huanyu_microbatch_sync(model, enabled=False)
        wrapped = SimpleNamespace(
            _orig_mod=model,
            config=SimpleNamespace(
                huanyu_sync_each_microbatch=True, model_type="qwen3_5_moe",
                audio_config={"model_type": "huanyu_conformer_0_6b"},
            ),
        )
        self.assertFalse(policy._enabled_for_engine(SimpleNamespace(module=wrapped)))

    def test_fsdp1_keeps_sync_by_not_entering_original_no_sync(self):
        engine_class = modern_engine_class()
        model = SimpleNamespace(no_sync=lambda: self.fail("no_sync must not be entered"))
        with fake_engine(engine_class), redirect_stdout(io.StringIO()):
            policy.install_huanyu_microbatch_sync(model)
            engine = engine_class(model)
            with engine._gradient_sync_context(is_last_micro_batch=False):
                pass
            self.assertEqual(engine.original_calls, [])

    def test_legacy_known_loop_is_retained_and_sync_state_restored(self):
        engine_class = legacy_engine_class()
        model = Model()
        seen = []
        loss = SimpleNamespace(backward=lambda: seen.append(model.sync))
        with fake_engine(engine_class), redirect_stdout(io.StringIO()) as output:
            policy.install_huanyu_microbatch_sync(model)
            hooked = engine_class.forward_backward_batch
            policy.install_huanyu_microbatch_sync(model)
            self.assertIs(hooked, engine_class.forward_backward_batch)
            model.sync = False
            self.assertEqual(engine_class(model).forward_backward_batch([loss, loss]), "completed")
            self.assertEqual(seen, [True, True])
            self.assertTrue(model.sync)
            self.assertIn("mode=verified_legacy_loop", output.getvalue())

    def test_legacy_unmarked_model_remains_unchanged(self):
        engine_class = legacy_engine_class()
        model = Model()
        model.sync = False
        with fake_engine(engine_class), redirect_stdout(io.StringIO()) as output:
            policy.install_huanyu_microbatch_sync(Model())
            seen = []
            loss = SimpleNamespace(backward=lambda: seen.append(model.sync))
            engine_class(model).forward_backward_batch([loss])
            self.assertEqual(seen, [False])
            self.assertEqual(output.getvalue(), "")

    def test_legacy_with_hidden_no_sync_is_rejected(self):
        class UnsafeEngine:
            def forward_backward_batch(self, losses):
                for loss in losses:
                    with self.module.no_sync():
                        loss.backward()

        with fake_engine(UnsafeEngine), self.assertRaisesRegex(RuntimeError, "unverified|gradient-sync"):
            policy.install_huanyu_microbatch_sync(Model())

    def test_legacy_direct_sync_flag_change_is_rejected_without_marking_model(self):
        class UnsafeEngine:
            def forward_backward_batch(self, losses):
                self.module.require_backward_grad_sync = False
                for loss in losses:
                    loss.backward()

        model = Model()
        with fake_engine(UnsafeEngine), self.assertRaisesRegex(RuntimeError, "gradient-sync"):
            policy.install_huanyu_microbatch_sync(model)
        self.assertFalse(hasattr(model, policy._MARKER))

    def test_legacy_uninspectable_or_wrapped_loop_is_rejected(self):
        def delegated(self, data):
            return self.other_forward_backward(data)

        for method, message in ((len, "Cannot inspect"), (delegated, "no inspectable backward")):
            with self.subTest(method=method):
                engine_class = type("UnknownEngine", (), {"forward_backward_batch": method})
                with fake_engine(engine_class), self.assertRaisesRegex(RuntimeError, message):
                    policy.install_huanyu_microbatch_sync(Model())

    def test_legacy_unknown_context_disguised_as_ctx_is_rejected(self):
        class UnknownContextEngine:
            def forward_backward_batch(self, losses):
                ctx = self.make_unknown_context()
                for loss in losses:
                    with ctx:
                        loss.backward()

        with fake_engine(UnknownContextEngine), self.assertRaisesRegex(RuntimeError, "unverified context"):
            policy.install_huanyu_microbatch_sync(Model())

    def test_legacy_exception_restores_sync(self):
        engine_class = legacy_engine_class()
        model = Model()

        def failed_backward():
            model.sync = False
            raise RuntimeError("backward failed")

        with fake_engine(engine_class), redirect_stdout(io.StringIO()):
            policy.install_huanyu_microbatch_sync(model)
            with self.assertRaisesRegex(RuntimeError, "backward failed"):
                engine_class(model).forward_backward_batch([SimpleNamespace(backward=failed_backward)])
            self.assertTrue(model.sync)

    def test_invalid_enabled_value_is_rejected(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                policy.install_huanyu_microbatch_sync(Model(), enabled=value)


if __name__ == "__main__":
    unittest.main()
