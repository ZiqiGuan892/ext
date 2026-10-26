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
DIRECT_16K = "run_huanyu_aqa_16k_24_8.sh"
JOB_16K = "run_huanyu_aqa_16k_24_8_job.sh"


@unittest.skipUnless(_aqa._job.find_bash(), "Bash is required for launcher argv integration")
class AqaPrecisionEntryTests(unittest.TestCase):
    def setUp(self):
        self.aqa = _aqa.AqaJobLauncherTests("runTest")
        self.addCleanup(self.aqa.doCleanups)
        self.aqa.setUp()
        self.fixture = self.aqa.fixture
        for name in (JOB, DIRECT, DIRECT_16K, JOB_16K):
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

    def use_16k_entry(self):
        self.fixture.entry = self.fixture.examples / DIRECT_16K
        _aqa.pq.write_table(_aqa.pa.Table.from_pylist([self.aqa.row] * 12), self.aqa.train)
        (self.fixture.examples / "env_export_huanyu.sh").write_text("exit 91\n", encoding="utf-8")

    def test_direct_16k_keeps_24_plus_8_and_full_precision_through_all_wrappers(self):
        self.use_16k_entry()
        capture = self.aqa.launch(
            {"NUM_NPUS": "16", "NNODES": "1", "RM_N_GPUS_PER_NODE": "16", "RM_NNODES": "1",
             "RM_ENABLE": "0", "ROLLOUT_TP": "8", "RM_TP": "8", "TRAIN_BATCH_SIZE": "4",
             "HUANYU_PRECISION_PROFILE": "custom", "MINICPMO_STABLE_LINEAR": "0",
             "MAX_PROMPT_LENGTH": "3072", "RAY_ADDRESS": "192.0.2.10:6925"},
            ("trainer.n_gpus_per_node=16", "trainer.nnodes=1", "reward.reward_model.n_gpus_per_node=16",
             "reward.reward_model.nnodes=1", "reward.reward_model.enable=false",
             "data.max_prompt_length=3072", "actor_rollout_ref.rollout.max_model_len=5248",
             "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
             "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=3072",
             "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_STABLE_LINEAR=0",
             "actor_rollout_ref.actor.optim.lr=2e-6"),
        )
        env, actual = capture["env"], _aqa.effective_arguments(capture)
        self.aqa.assert_precision_environment(env)
        for key, value in _aqa.PRECISION_ENV.items():
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), value, key)
        expected = {
            "trainer.n_gpus_per_node": "12", "trainer.nnodes": "2",
            "reward.reward_model.n_gpus_per_node": "4", "reward.reward_model.nnodes": "2",
            "reward.reward_model.enable": "true", "reward.reward_model.enable_resource_pool": "true",
            "actor_rollout_ref.rollout.tensor_model_parallel_size": "4",
            "reward.reward_model.rollout.tensor_model_parallel_size": "4",
            "actor_rollout_ref.rollout.agent.num_workers": "6",
            "data.train_batch_size": "6", "actor_rollout_ref.actor.ppo_mini_batch_size": "6",
            "actor_rollout_ref.rollout.n": "4", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
            "data.max_prompt_length": "14336", "data.max_response_length": "2048",
            "actor_rollout_ref.rollout.prompt_length": "14336", "actor_rollout_ref.rollout.response_length": "2048",
            "actor_rollout_ref.rollout.max_model_len": "16512",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "16512",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "16512",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "16512",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "16512",
            "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
            "actor_rollout_ref.model.override_config.huanyu_freeze_audio": "false",
            "actor_rollout_ref.actor.use_dynamic_bsz": "false",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": "-1",
            "actor_rollout_ref.ref.fsdp_config.fsdp_size": "-1",
            "actor_rollout_ref.rollout.max_num_seqs": "32",
            "actor_rollout_ref.rollout.enforce_eager": "true",
            "reward.reward_model.rollout.enforce_eager": "true",
            "actor_rollout_ref.actor.optim.lr": "2e-6",
        }
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)
        self.assertEqual(env["REWARD_POOL_NPUS"], "8")
        self.assertEqual(env["RM_SPLIT_NODES"], "0")
        self.assertEqual(env["HUANYU_USE_PRECOMPUTE"], "0")
        self.assertEqual(env["HUANYU_SKIP_EDITABLE_INSTALL"], "1")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"]), "14336")
        self.assertEqual(actual["data.train_files"], _aqa._job.shell_path(self.aqa.train))
        self.assertIn("genrm_verifier", actual["reward.custom_reward_function.path"])
        for kind in ("pip", "git", "env-source", "helper"):
            self.assertFalse(self.fixture.events(kind), kind)
        self.assertEqual([event["args"] for event in self.fixture.events("ray")],
                         [["status", "--address=192.0.2.10:6925"]])

    def test_direct_16k_response_split_and_frozen_audio_comparison(self):
        self.use_16k_entry()
        capture = self.aqa.launch({"HUANYU_16K_RESPONSE_LENGTH": "4096",
                                   "HUANYU_16K_ATTN_IMPLEMENTATION": "eager", "HUANYU_FREEZE_AUDIO": "true"})
        actual = _aqa.effective_arguments(capture)
        self.assertEqual(actual["data.max_prompt_length"], "12288")
        self.assertEqual(actual["data.max_response_length"], "4096")
        self.assertEqual(actual["actor_rollout_ref.rollout.prompt_length"], "12288")
        self.assertEqual(actual["actor_rollout_ref.rollout.response_length"], "4096")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "eager")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"]), "12288")

    def test_direct_16k_rejects_invalid_length_or_unreachable_cluster(self):
        self.use_16k_entry()
        for updates in ({"HUANYU_16K_RESPONSE_LENGTH": "16384"}, {"HUANYU_16K_RESPONSE_LENGTH": "02048"},
                        {"HUANYU_16K_ATTN_IMPLEMENTATION": "invalid"}, {"RAY_ADDRESS": "local"},
                        {"HUANYU_TEST_FAIL_RAY": "1"}):
            with self.subTest(updates=updates):
                result = self.fixture.run_launcher(updates)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("train", "pip", "git", "runtime", "env-source"):
                    self.assertFalse(self.fixture.events(kind), kind)

    def use_16k_job(self):
        self.fixture.entry = self.fixture.examples / JOB_16K
        _aqa.pq.write_table(_aqa.pa.Table.from_pylist([self.aqa.row] * 12), self.aqa.train)
        # Simulate an older site environment with 5K, 16+16 and frozen defaults.
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                "export HUANYU_JOB_MODE=frozen_cache HUANYU_PRECISION_PROFILE=custom HUANYU_GRAPH_MODE=1\n"
                "export MINICPMO_STABLE_LINEAR=0 MINICPMO_PRIVATE_HYBRID_CACHE=0\n"
                "export HUANYU_FREEZE_AUDIO=true HUANYU_USE_PRECOMPUTE=1 MINICPMO_TUNE_AUDIO=0\n"
                "export NUM_NPUS=16 NNODES=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1 ROLLOUT_TP=8 RM_TP=8\n"
                "export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 MAX_PROMPT_LENGTH=3072\n"
                "export MAX_RESPONSE_LENGTH=2048 ROLLOUT_MAX_MODEL_LEN=5248\n"
                "export HUANYU_16K_RESPONSE_LENGTH=123 HUANYU_16K_ATTN_IMPLEMENTATION=invalid\n"
                "export HUANYU_AQA_TRAIN_FILE=/old/train.parquet HUANYU_AQA_VAL_FILE=/old/validation.parquet\n"
                "export HCCL_OP_EXPANSION_MODE=AIV HUANYU_HCCL_OP_EXPANSION_MODE=AIV\n"
            )

    def assert_16k_job_environment(self, env, response="2048"):
        self.aqa.assert_precision_environment(env)
        for key, value in dict(_aqa.MODE_ENV, **{
            "HUANYU_JOB_MODE": "aqa_16k_24_8", "HUANYU_GRAPH_MODE": "0",
            "NUM_NPUS": "12", "NNODES": "2", "RM_N_GPUS_PER_NODE": "4", "RM_NNODES": "2",
            "ROLLOUT_TP": "4", "RM_TP": "4", "REWARD_POOL_NPUS": "8", "RM_SPLIT_NODES": "0",
            "TRAIN_BATCH_SIZE": "6", "PPO_MINI_BATCH_SIZE": "6", "ROLLOUT_N": "4",
            "MAX_PROMPT_LENGTH": str(16384 - int(response)), "MAX_RESPONSE_LENGTH": response,
            "ROLLOUT_MAX_MODEL_LEN": "16512", "HUANYU_16K_RESPONSE_LENGTH": response,
            "HUANYU_HCCL_OP_EXPANSION_MODE": "auto", "HCCL_OP_EXPANSION_MODE": "AI_CPU",
        }).items():
            self.assertEqual(env[key], value, key)

    def test_16k_job_bootstraps_then_head_runs_24_plus_8_training_once(self):
        self.use_16k_job()
        capture = self.aqa.launch(arguments=(
            "data.max_prompt_length=3072", "trainer.n_gpus_per_node=16",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
            "reward.reward_model.n_gpus_per_node=16", "reward.reward_model.enable=false",
            "actor_rollout_ref.rollout.enforce_eager=false", "actor_rollout_ref.actor.optim.lr=2e-6",
        ))
        self.assert_16k_job_environment(capture["env"])
        actual = _aqa.effective_arguments(capture)
        for key, value in {
            "trainer.n_gpus_per_node": "12", "trainer.nnodes": "2",
            "reward.reward_model.n_gpus_per_node": "4", "reward.reward_model.nnodes": "2",
            "reward.reward_model.enable": "true", "actor_rollout_ref.rollout.tensor_model_parallel_size": "4",
            "reward.reward_model.rollout.tensor_model_parallel_size": "4",
            "actor_rollout_ref.rollout.agent.num_workers": "6", "data.train_batch_size": "6",
            "actor_rollout_ref.actor.ppo_mini_batch_size": "6", "actor_rollout_ref.rollout.n": "4",
            "data.max_prompt_length": "14336", "data.max_response_length": "2048",
            "actor_rollout_ref.rollout.prompt_length": "14336", "actor_rollout_ref.rollout.max_model_len": "16512",
            "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
            "actor_rollout_ref.rollout.enforce_eager": "true", "actor_rollout_ref.actor.optim.lr": "2e-6",
            "data.train_files": _aqa._job.shell_path(self.aqa.train),
            "data.val_files": _aqa._job.shell_path(self.aqa.validation),
        }.items():
            self.assertEqual(actual[key], value, key)
        for key, value in dict(_aqa.PRECISION_ENV, **_aqa.MODE_ENV,
                               MINICPMO_MM_PROMPT_BUDGET="14336",
                               HUANYU_HCCL_OP_EXPANSION_MODE="auto", HCCL_OP_EXPANSION_MODE="AI_CPU").items():
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), value, key)
        self.assertEqual(len(self.fixture.events("train")), 1)
        self.assertEqual(len(self.fixture.events("pip")), 1)
        self.assertEqual(len(self.fixture.events("env-source")), 1)
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 2)
        self.assertIn("--head", ray[0]["args"])
        self.assertIn('--resources={"NPU": 16}', ray[0]["args"])
        self.assertEqual(ray[1]["args"], ["status", "--address=192.0.2.10:6925"])
        for event in ray + self.fixture.events("runtime"):
            self.assert_16k_job_environment(event["env"])
        events = [json.loads(line) for line in self.fixture.events_file.read_text().splitlines()]
        wait = next(event for event in events if event["kind"] == "helper" and event["args"][0] == "wait-cluster")
        self.assertLess(events.index(ray[0]), events.index(wait))
        self.assertLess(events.index(wait), events.index(self.fixture.events("train")[0]))

    def test_16k_job_worker_only_joins_with_full_precision_and_32_physical_cards(self):
        self.use_16k_job()
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--address=192.0.2.10:6925", ray[0]["args"])
        self.assertIn("--block", ray[0]["args"])
        self.assertIn('--resources={"NPU": 16}', ray[0]["args"])
        self.assertNotIn("--head", ray[0]["args"])
        self.assert_16k_job_environment(ray[0]["env"])

    def test_16k_job_reuses_cluster_and_preserves_response_split_across_site_env(self):
        self.use_16k_job()
        capture = self.aqa.launch({"HUANYU_TEST_RAY_LIVE": "1", "HUANYU_16K_RESPONSE_LENGTH": "4096",
                                   "HUANYU_16K_ATTN_IMPLEMENTATION": "eager"})
        self.assert_16k_job_environment(capture["env"], response="4096")
        actual = _aqa.effective_arguments(capture)
        self.assertEqual(actual["data.max_prompt_length"], "12288")
        self.assertEqual(actual["data.max_response_length"], "4096")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "eager")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"]), "12288")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "HUANYU_PRECISION_PROFILE"]), "full")
        self.assertFalse(self.fixture.events("pip"))
        self.assertEqual([event["args"] for event in self.fixture.events("ray")],
                         [["status", "--address=192.0.2.10:6925"]])
        sync = next(event for event in self.fixture.events("helper") if event["args"][0] == "sync")
        self.assertIn("--ray-running", sync["args"])

    def test_16k_job_invalid_response_stops_before_bootstrap(self):
        self.use_16k_job()
        result = self.fixture.run_launcher({"HUANYU_16K_RESPONSE_LENGTH": "16384"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HUANYU_16K_RESPONSE_LENGTH", result.stdout + result.stderr)
        for kind in ("helper", "pip", "env-source", "ray", "train"):
            self.assertFalse(self.fixture.events(kind), kind)

    def test_16k_explicit_host_survives_stale_site_env_and_reused_ray(self):
        self.use_16k_job()
        capture = self.aqa.launch({"HUANYU_HCCL_OP_EXPANSION_MODE": "HOST",
                                   "HCCL_OP_EXPANSION_MODE": "AIV", "HUANYU_TEST_RAY_LIVE": "1"}, (
            '++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE="AIV"',
            '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_HCCL_OP_EXPANSION_MODE="AIV"',
        ))
        actual = _aqa.effective_arguments(capture)
        for key in ("HUANYU_HCCL_OP_EXPANSION_MODE", "HCCL_OP_EXPANSION_MODE"):
            self.assertEqual(capture["env"][key], "HOST")
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), "HOST")
            for event in self.fixture.events("runtime") + self.fixture.events("ray"):
                self.assertEqual(event["env"][key], "HOST")
        self.assertFalse(any("start" in event["args"] for event in self.fixture.events("ray")))

    def test_16k_job_cluster_readiness_failure_never_starts_training(self):
        self.use_16k_job()
        result = self.fixture.run_launcher({"HUANYU_TEST_FAIL_WAIT": "wait-cluster"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("training not started", result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(any("status" in event["args"] for event in self.fixture.events("ray")))


if __name__ == "__main__":
    unittest.main()
