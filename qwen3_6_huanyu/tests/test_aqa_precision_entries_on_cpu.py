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
JOB = "run_huanyu_32card_job.sh"
DIRECT = "run_huanyu_aqa_audio_train.sh"


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
        self.fixture.entry = self.fixture.examples / DIRECT
        # Keep Linux dump paths as Hydra values when running under Git Bash.
        self.fixture.env["MSYS2_ARG_CONV_EXCL"] = self.fixture.env.get("MSYS2_ARG_CONV_EXCL", "") + (
            ";trainer.rollout_data_dir=;trainer.validation_data_dir="
        )
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

    def settings(self, prompt=3072, response=2048):
        total = prompt + response + 128
        return {
            "NUM_NPUS": "16", "NNODES": "1", "ROLLOUT_TP": "4",
            "RM_ENABLE": "1", "RM_N_GPUS_PER_NODE": "16", "RM_NNODES": "1", "RM_TP": "4",
            "TRAIN_BATCH_SIZE": "4", "PPO_MINI_BATCH_SIZE": "4", "ROLLOUT_N": "4",
            "HUANYU_PRECISION_PROFILE": "full", "HUANYU_GRAPH_MODE": "0",
            "MAX_PROMPT_LENGTH": str(prompt), "MAX_RESPONSE_LENGTH": str(response),
            "ROLLOUT_PROMPT_LENGTH": str(prompt), "ROLLOUT_MAX_MODEL_LEN": str(total),
            "MAX_TOKEN_LEN_PER_GPU": str(total), "MINICPMO_MM_PROMPT_BUDGET": str(prompt),
        }

    def assert_configuration(self, capture, prompt, response, sync=False):
        actual = _aqa.effective_arguments(capture)
        self.aqa.assert_precision_environment(capture["env"])
        for key, value in _aqa.PRECISION_ENV.items():
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), value, key)
        expected = {
            "data.max_prompt_length": str(prompt), "data.max_response_length": str(response),
            "actor_rollout_ref.rollout.prompt_length": str(prompt),
            "actor_rollout_ref.rollout.response_length": str(response),
            "actor_rollout_ref.rollout.max_model_len": str(prompt + response + 128),
            "actor_rollout_ref.rollout.max_num_batched_tokens": str(prompt + response + 128),
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": str(prompt + response + 128),
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": str(prompt + response + 128),
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": str(prompt + response + 128),
            "actor_rollout_ref.actor.use_dynamic_bsz": "false",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
            "trainer.n_gpus_per_node": "16", "trainer.nnodes": "1",
            "reward.reward_model.n_gpus_per_node": "16", "reward.reward_model.nnodes": "1",
            "actor_rollout_ref.rollout.enforce_eager": "true",
            "reward.reward_model.rollout.enforce_eager": "true",
        }
        if sync:
            expected["actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch"] = "true"
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"]), str(prompt))
        self.assertEqual(capture["env"]["HUANYU_SKIP_EDITABLE_INSTALL"], "1")
        self.assertTrue(actual["trainer.rollout_data_dir"].endswith("/rollout_dump"))
        self.assertTrue(actual["trainer.validation_data_dir"].endswith("/validation_dump"))
        return actual

    def assert_no_setup(self):
        for kind in ("pip", "git", "env-source", "helper"):
            self.assertFalse(self.fixture.events(kind), kind)
        self.assertEqual([event["args"] for event in self.fixture.events("ray")],
                         [["status", "--address=auto"]])

    def test_direct_5k_preserves_full_precision_and_independent_outputs(self):
        config = self.settings()
        config.update(DEFAULT_LOCAL_DIR=self.fixture.env["PROJECT_ROOT"] + "/checkpoints",
                      TENSORBOARD_DIR=self.fixture.env["PROJECT_ROOT"] + "/tensorboard_log",
                      MINICPMO_STABLE_LINEAR="0", TOTAL_TRAINING_STEPS="2")
        capture = self.aqa.launch(config, ("actor_rollout_ref.actor.optim.lr=2e-6",))
        actual = self.assert_configuration(capture, 3072, 2048)
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        self.assertEqual(actual["actor_rollout_ref.actor.optim.lr"], "2e-6")
        self.assertEqual(capture["env"]["DEFAULT_LOCAL_DIR"], capture["env"]["HUANYU_RUN_DIR"] + "/checkpoints")
        self.assert_no_setup()

    def test_direct_12k_and_eos_are_forwarded(self):
        capture = self.aqa.launch(self.settings(2048, 10240), (
            "actor_rollout_ref.rollout.ignore_eos=true",
            "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true",
        ))
        actual = self.assert_configuration(capture, 2048, 10240, sync=True)
        self.assertEqual(actual["actor_rollout_ref.rollout.ignore_eos"], "true")
        self.assert_no_setup()

    def test_direct_16k_sync_and_cli_defaults(self):
        capture = self.aqa.launch(self.settings(14336, 2048), (
            "actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
            "actor_rollout_ref.ref.fsdp_config.fsdp_size=-1",
            "actor_rollout_ref.rollout.max_num_seqs=32", "trainer.test_freq=25",
        ))
        actual = self.assert_configuration(capture, 14336, 2048, sync=True)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "sdpa")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_num_seqs"], "32")
        self.assertEqual(actual["trainer.test_freq"], "25")
        self.assert_no_setup()

    def test_direct_rules_and_explicit_dump_paths_survive(self):
        capture = self.aqa.launch(dict(self.settings(), RM_ENABLE="0", HUANYU_FREEZE_AUDIO="true"), (
            "trainer.rollout_data_dir=/custom/rollout_dump", "trainer.validation_data_dir=/custom/validation_dump",
        ))
        actual = _aqa.effective_arguments(capture)
        self.assertEqual(actual["reward.reward_model.enable"], "false")
        self.assertIn("rule_based_reward", actual["reward.custom_reward_function.path"])
        self.assertEqual(actual["trainer.rollout_data_dir"], "/custom/rollout_dump")
        self.assertEqual(actual["trainer.validation_data_dir"], "/custom/validation_dump")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assert_no_setup()

    def test_direct_unreachable_ray_never_starts_training_or_native_imports(self):
        result = self.fixture.run_launcher({"HUANYU_TEST_FAIL_RAY": "1", "RAY_ADDRESS": None})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("existing Ray is not reachable", result.stdout + result.stderr)
        for kind in ("train", "runtime", "pip", "helper"):
            self.assertFalse(self.fixture.events(kind), kind)

    def use_job(self):
        self.fixture.entry = self.fixture.examples / JOB
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                "export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 ROLLOUT_MAX_MODEL_LEN=5248\n"
                "export TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=8\n"
                "export HUANYU_PRECISION_PROFILE=custom MINICPMO_STABLE_LINEAR=0 HUANYU_GRAPH_MODE=1\n"
                "export TOTAL_TRAINING_STEPS=99\n"
            )
        return dict(self.settings(14336, 2048), HUANYU_JOB_MODE="aqa_unfreeze", TOTAL_TRAINING_STEPS="2")

    def test_common_job_preserves_16k_arguments_across_site_setup(self):
        config = self.use_job()
        capture = self.aqa.launch(config, (
            "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true",
            "actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        ))
        actual = self.assert_configuration(capture, 14336, 2048, sync=True)
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        self.assertEqual(actual["data.train_batch_size"], "4")
        self.assertEqual(len(self.fixture.events("pip")), 1)
        self.assertEqual(len(self.fixture.events("env-source")), 1)
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])
        self.assertEqual(self.fixture.events("ray")[1]["args"], ["status", "--address=192.0.2.10:6925"])
        self.assertEqual(len(self.fixture.events("train")), 1)

    def test_common_job_worker_only_joins_ray_with_requested_precision(self):
        config = self.use_job()
        result = self.fixture.run_launcher(dict(config, VC_TASK_INDEX="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--block", ray[0]["args"])
        self.aqa.assert_precision_environment(ray[0]["env"])
        self.assertEqual(ray[0]["env"]["MAX_PROMPT_LENGTH"], "14336")
        self.assertEqual(ray[0]["env"]["HUANYU_GRAPH_MODE"], "0")

    def test_common_job_reuses_existing_cluster(self):
        config = self.use_job()
        capture = self.aqa.launch(dict(config, HUANYU_TEST_RAY_LIVE="1"))
        self.assert_configuration(capture, 14336, 2048)
        self.assertFalse(self.fixture.events("pip"))
        self.assertEqual([event["args"] for event in self.fixture.events("ray")],
                         [["status", "--address=192.0.2.10:6925"]])

    def test_job_readiness_failure_never_starts_training(self):
        config = self.use_job()
        result = self.fixture.run_launcher(dict(config, HUANYU_TEST_FAIL_WAIT="wait-cluster"))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(any("status" in event["args"] for event in self.fixture.events("ray")))


if __name__ == "__main__":
    unittest.main()
