"""Disk validation keeps native weights, while the control synchronizes once."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "checkpoint_validation.py"
SPEC = importlib.util.spec_from_file_location("huanyu_checkpoint_validation_tests", PATH)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def make_trainer():
    class Trainer:
        def __init__(self):
            self.events = []
            self.disk_weights = [1, 2, 3]
            self.config = NS(
                trainer=NS(val_only=True, val_before_train=True, resume_mode="disable",
                           v1=NS(trainer_mode="omni_sync")),
                actor_rollout_ref=NS(
                    model=NS(architecture="Qwen36HuanyuForTraining", path="/checkpoint/2748", lora_rank=0),
                    rollout=NS(name="vllm_omni", load_format="safetensors", free_cache_engine=False),
                ),
            )
            self.global_steps = 0
            self.reward_loop_manager = NS(reward_loop_worker_handles=[object()])
            self.checkpoint_manager = NS(update_weights=self.sync)

        def sync(self, step):
            self.events.append(("sync", step))
            self.disk_weights[:] = [4, 5, 6]

        def on_init_end(self):
            self.checkpoint_manager.update_weights(self.global_steps)

        def on_train_begin(self):
            self.events.append("train")

    validation._bind(Trainer)
    return Trainer()


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {validation.ENV: "disk", "HUANYU_AUDIO_PROBE_DIR": "/probe",
                                          validation.FULL_AUDIT_ENV: ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        versions = patch.object(validation, "_set_rollout_version", side_effect=lambda t: t.events.append("version=0"))
        self.versions = versions.start()
        self.addCleanup(versions.stop)

    def test_disk_keeps_checkpoint_weights_and_forbids_later_training_or_sync(self):
        trainer = make_trainer()
        weights = trainer.disk_weights
        with redirect_stdout(io.StringIO()) as output:
            trainer.on_init_end()
        self.assertIs(trainer.disk_weights, weights)
        self.assertEqual(weights, [1, 2, 3])
        self.assertEqual(trainer.events, ["version=0"])
        self.assertIn("initial_actor_sync=skipped", output.getvalue())
        with self.assertRaisesRegex(RuntimeError, "subsequent"):
            trainer.checkpoint_manager.update_weights()
        with self.assertRaisesRegex(RuntimeError, "training is forbidden"):
            trainer.on_train_begin()

    def test_sync_control_updates_exactly_once_then_validates_only(self):
        trainer = make_trainer()
        with patch.dict(os.environ, {validation.ENV: "sync"}), redirect_stdout(io.StringIO()):
            trainer.on_init_end()
        self.assertEqual(trainer.events, [("sync", 0), "version=0"])
        self.assertEqual(trainer.disk_weights, [4, 5, 6])
        with self.assertRaises(RuntimeError):
            trainer.checkpoint_manager.update_weights(1)

    def test_full_audit_stops_after_sync_rpcs_before_any_validation_or_training(self):
        trainer = make_trainer()
        with patch.dict(os.environ, {validation.ENV: "sync", validation.FULL_AUDIT_ENV: "/full-audit",
                                     "HUANYU_AUDIO_PROBE_DIR": ""}):
            with self.assertRaisesRegex(RuntimeError, "AUDIT_ONLY_STOP"):
                trainer.on_init_end()
            self.assertEqual(trainer.events, [("sync", 0)])
            with self.assertRaisesRegex(RuntimeError, "training is forbidden"):
                trainer.on_train_begin()
        self.versions.assert_not_called()

    def test_full_audit_requires_guarded_sync_even_when_mode_unset(self):
        for mode in ("", "disk"):
            with patch.dict(os.environ, {validation.ENV: mode, validation.FULL_AUDIT_ENV: "/audit"}):
                with self.assertRaisesRegex(ValueError, "requires"):
                    validation.install_checkpoint_validation()
        trainer = make_trainer()
        trainer.config.actor_rollout_ref.model.architecture = "OtherModel"
        with patch.dict(os.environ, {validation.ENV: "sync", validation.FULL_AUDIT_ENV: "/audit"}):
            with self.assertRaisesRegex(ValueError, "Huanyu training adapter"):
                trainer.on_init_end()
        self.assertEqual(trainer.events, [])

    def test_disabled_and_other_architecture_preserve_regular_training(self):
        for disabled in (True, False):
            trainer = make_trainer()
            if not disabled:
                trainer.config.actor_rollout_ref.model.architecture = "OtherModel"
            with patch.dict(os.environ, {validation.ENV: "" if disabled else "disk"}):
                trainer.on_init_end()
                trainer.on_train_begin()
                trainer.checkpoint_manager.update_weights(1)
            self.assertEqual(trainer.events, [("sync", 0), "train", ("sync", 1)])
        self.versions.assert_not_called()

    def test_unsafe_settings_fail_before_skipping_or_synchronizing(self):
        for owner, field, value in (
            ("trainer", "val_only", False), ("trainer", "val_before_train", False),
            ("trainer", "resume_mode", "auto"), ("rollout", "free_cache_engine", True),
            ("rollout", "load_format", "dummy"), ("rollout", "name", "vllm"),
            ("model", "lora_rank", 8),
        ):
            with self.subTest(field=field):
                trainer = make_trainer()
                obj = trainer.config.trainer if owner == "trainer" else getattr(trainer.config.actor_rollout_ref, owner)
                setattr(obj, field, value)
                with self.assertRaises(ValueError):
                    trainer.on_init_end()
                self.assertEqual(trainer.events, [])

    def test_colocated_reward_resumed_run_and_missing_probe_are_rejected(self):
        trainer = make_trainer()
        trainer.reward_loop_manager.reward_loop_worker_handles = None
        with self.assertRaisesRegex(ValueError, "separate reward"):
            trainer.on_init_end()
        trainer = make_trainer()
        trainer.global_steps = 5
        with self.assertRaisesRegex(ValueError, "global_steps"):
            trainer.on_init_end()
        with patch.dict(os.environ, {"HUANYU_AUDIO_PROBE_DIR": ""}):
            with self.assertRaisesRegex(ValueError, "PROBE_DIR"):
                make_trainer().on_init_end()

    def test_sync_failure_propagates_without_success_or_version_update(self):
        trainer = make_trainer()
        def fail(*args):
            raise RuntimeError("broken sync")
        trainer.checkpoint_manager.update_weights = fail
        with patch.dict(os.environ, {validation.ENV: "sync"}), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "broken sync"):
                trainer.on_init_end()
        self.assertEqual(output.getvalue(), "")
        self.versions.assert_not_called()

    def test_binding_is_idempotent_and_invalid_mode_is_rejected(self):
        trainer = make_trainer()
        original = type(trainer).on_init_end
        validation._bind(type(trainer))
        self.assertIs(type(trainer).on_init_end, original)
        with patch.dict(os.environ, {validation.ENV: "typo"}):
            with self.assertRaisesRegex(ValueError, "must be"):
                trainer.on_init_end()

    def test_version_rpc_sets_only_metadata_on_every_replica(self):
        events = []
        replicas = [NS(server_handle=NS(set_global_steps=NS(remote=lambda x, i=i: events.append((i, x)) or i)))
                    for i in range(3)]
        trainer = NS(llm_server_manager=NS(get_replicas=lambda: replicas))
        ray = ModuleType("ray")
        ray.get = lambda refs: events.append(("wait", refs))
        # Reload in isolation to exercise the real function patched in setUp.
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        with patch.dict("sys.modules", {"ray": ray}):
            module._set_rollout_version(trainer)
            self.assertEqual(events, [(0, 0), (1, 0), (2, 0), ("wait", [0, 1, 2])])
            trainer.llm_server_manager.get_replicas = lambda: []
            with self.assertRaisesRegex(RuntimeError, "handles"):
                module._set_rollout_version(trainer)

    def test_installer_is_opt_in_and_patches_the_runtime_trainer_class(self):
        trainer_module = ModuleType("verl.trainer.ppo.v1.trainer_sync")
        trainer_module.PPOTrainerSync = type(make_trainer())
        with patch.dict("sys.modules", {trainer_module.__name__: trainer_module}):
            with patch.object(validation, "_bind") as bind:
                with patch.dict(os.environ, {validation.ENV: ""}):
                    self.assertFalse(validation.install_checkpoint_validation())
                bind.assert_not_called()
                self.assertTrue(validation.install_checkpoint_validation())
                bind.assert_called_once_with(trainer_module.PPOTrainerSync)


if __name__ == "__main__":
    unittest.main()
