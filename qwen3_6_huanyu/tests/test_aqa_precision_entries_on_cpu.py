"""Check precision-entry routing and final Hydra argv using CPU command stubs."""

import importlib.util
import json
from pathlib import Path
import shutil
import unittest


_spec = importlib.util.spec_from_file_location(
    "aqa_precision_fixture", Path(__file__).with_name("test_aqa_job_launcher_on_cpu.py")
)
_aqa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aqa)
EXAMPLES = _aqa.EXAMPLES
JOB = "run_huanyu_aqa_precision_32card_job.sh"
DIRECT = "run_huanyu_aqa_precision.sh"


@unittest.skipUnless(_aqa._job.find_bash(), "Bash is required for launcher argv integration")
class AqaPrecisionEntryTests(unittest.TestCase):
    def setUp(self):
        self.aqa = _aqa.AqaJobLauncherTests("runTest")
        self.addCleanup(self.aqa.doCleanups)
        self.aqa.setUp()
        self.fixture = self.aqa.fixture
        for name in (JOB, DIRECT):
            shutil.copyfile(EXAMPLES / name, self.fixture.examples / name)
        shutil.copyfile(EXAMPLES.parent / "rule_based_reward.py", self.fixture.package / "rule_based_reward.py")
        self.fixture.entry = self.fixture.examples / JOB
        # Exercise the actual setup/install guard, not only the final argv.
        # Hardware preflights remain excluded by the underlying CPU fixture.
        backend = EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh"
        prefix = backend.read_text(encoding="utf-8").split("# =============================================================================", 1)[0]
        deployed = self.fixture.examples / backend.name
        deployed.write_text(prefix + deployed.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        self.fixture.command_stub.write_text(
            self.fixture.command_stub.read_text(encoding="utf-8").replace(
                'if command == "pgrep":',
                'if command == "ray" and os.environ.get("HUANYU_TEST_FAIL_RAY") == "1":\n'
                '    raise SystemExit(17)\n'
                'if command == "pgrep":',
            ), encoding="utf-8",
        )

    def assert_precision(self, capture):
        self.aqa.assert_precision_environment(capture["env"])
        actual = _aqa.effective_arguments(capture)
        for key, value in _aqa.PRECISION_ENV.items():
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), value, key)
        self.assertEqual(actual["actor_rollout_ref.rollout.enforce_eager"], "true")
        self.assertEqual(actual["reward.reward_model.rollout.enforce_eager"], "true")
        self.assertEqual(actual["data.max_prompt_length"], "3072")
        self.assertEqual(actual["data.max_response_length"], "2048")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_model_len"], "5248")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "WANDB_MODE"]), "offline")
        self.assertEqual(capture["env"]["HUANYU_SKIP_EDITABLE_INSTALL"], "1")
        self.assertEqual(capture["env"]["HUANYU_USE_PRECOMPUTE"], "0")
        self.assertTrue(actual["trainer.rollout_data_dir"].endswith("/rollout_dump"))
        self.assertEqual(actual["trainer.logger"], "[console]")
        return actual

    def test_head_uses_new_direct_entry_and_forwards_full_policy_to_training(self):
        # Stale individual flags and CLI opt-outs must not weaken this entry.
        capture = self.aqa.launch(
            {"HUANYU_PRECISION_PROFILE": "custom", "HUANYU_GRAPH_MODE": "1",
             "MINICPMO_PRIVATE_HYBRID_CACHE": "0", "MINICPMO_STABLE_LINEAR": "0"},
            ("++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_STABLE_LINEAR=0",
             "actor_rollout_ref.rollout.enforce_eager=false"),
        )
        actual = self.assert_precision(capture)
        self.assertEqual(capture["env"]["HUANYU_JOB_MODE"], "aqa_precision")
        self.assertEqual(actual["data.train_files"], _aqa._job.shell_path(self.aqa.train))
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "false")
        self.assertEqual(actual["reward.reward_model.n_gpus_per_node"], "16")
        self.assertEqual(len(self.fixture.events("train")), 1)
        # Bootstrap installs once, but the direct entry never repeats it.
        self.assertEqual(len(self.fixture.events("pip")), 1)
        self.assertEqual(len(self.fixture.events("env-source")), 1)
        ray = self.fixture.events("ray")
        self.assertTrue(any("--head" in event["args"] for event in ray))
        self.assertTrue(any("status" in event["args"] for event in ray))
        for event in ray:
            self.aqa.assert_precision_environment(event["env"])

    def test_worker_enables_full_policy_after_site_env_and_only_joins_ray(self):
        env_file = self.fixture.examples / "env_export_huanyu.sh"
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write(
                "export HUANYU_PRECISION_PROFILE=custom HUANYU_GRAPH_MODE=1\n"
                "export MINICPMO_STABLE_LINEAR=0 MINICPMO_PRIVATE_HYBRID_CACHE=0\n"
            )
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--block", ray[0]["args"])
        self.assertNotIn("--head", ray[0]["args"])
        self.aqa.assert_precision_environment(ray[0]["env"])
        self.assertEqual(ray[0]["env"]["HUANYU_GRAPH_MODE"], "0")
        self.assertEqual(ray[0]["env"]["TORCH_COMPILE_DISABLE"], "1")

    def test_direct_reuses_prepared_environment_without_install_or_ray_start(self):
        self.fixture.entry = self.fixture.examples / DIRECT
        (self.fixture.examples / "env_export_huanyu.sh").write_text("exit 91\n", encoding="utf-8")
        capture = self.aqa.launch({
            "HUANYU_PRECISION_PROFILE": "custom", "HUANYU_GRAPH_MODE": "1",
            "HUANYU_FREEZE_AUDIO": "true", "RM_ENABLE": "0", "RAY_ADDRESS": "192.0.2.10:6925",
            "DEFAULT_LOCAL_DIR": self.fixture.env["PROJECT_ROOT"] + "/checkpoints",
            "TENSORBOARD_DIR": self.fixture.env["PROJECT_ROOT"] + "/tensorboard_log",
        })
        actual = self.assert_precision(capture)
        self.assertEqual(capture["env"]["DEFAULT_LOCAL_DIR"], capture["env"]["HUANYU_RUN_DIR"] + "/checkpoints")
        self.assertEqual(capture["env"]["TENSORBOARD_DIR"], capture["env"]["HUANYU_RUN_DIR"] + "/tensorboard")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assertEqual(actual["reward.reward_model.enable"], "false")
        self.assertIn("rule_based_reward", actual["reward.custom_reward_function.path"])
        for kind in ("pip", "git", "env-source", "helper"):
            self.assertFalse(self.fixture.events(kind), kind)
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertEqual(ray[0]["args"], ["status", "--address=192.0.2.10:6925"])

    def test_direct_unreachable_ray_exits_before_native_imports_or_training(self):
        self.fixture.entry = self.fixture.examples / DIRECT
        result = self.fixture.run_launcher({"HUANYU_TEST_FAIL_RAY": "1", "RAY_ADDRESS": None})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("existing Ray is not reachable", result.stdout + result.stderr)
        self.assertEqual(self.fixture.events("ray")[0]["args"], ["status", "--address=auto"])
        for kind in ("train", "pip", "git", "runtime", "env-source"):
            self.assertFalse(self.fixture.events(kind), kind)


if __name__ == "__main__":
    unittest.main()
