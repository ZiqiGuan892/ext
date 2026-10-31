"""Run scaled STEM job composition with command stubs and real Parquet input."""

import importlib.util
import json
from pathlib import Path
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
_spec = importlib.util.spec_from_file_location(
    "_stem_multinode_fixture", Path(__file__).with_name("test_aqa_job_launcher_on_cpu.py")
)
_aqa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aqa)
_job = _aqa._job


@unittest.skipUnless(_job.find_bash(), "Bash is required for launcher argv integration")
class StemMultinodeJobTests(unittest.TestCase):
    def setUp(self):
        self.case = _aqa.AqaJobLauncherTests("runTest")
        self.addCleanup(self.case.doCleanups)
        self.case.setUp()
        self.fixture = self.case.fixture
        for name in ("run_huanyu_stem_job.sh", "run_huanyu_stem.sh", "run_huanyu.sh"):
            shutil.copyfile(EXAMPLES / name, self.fixture.examples / name)
        self.fixture.entry = self.fixture.examples / "run_huanyu_stem_job.sh"
        # The site dependency is absent on the CPU host. Keep the real rule
        # preflight and exercise its dispatch contract with a tiny stand-in.
        reward = self.fixture.project / "verl/utils/reward_score"
        reward.mkdir(parents=True)
        for package in (self.fixture.project / "verl", self.fixture.project / "verl/utils"):
            (package / "__init__.py").write_text("", encoding="utf-8")
        (reward / "math_dapo.py").write_text(
            "def compute_score(solution_str, ground_truth):\n"
            "    return float(solution_str == r'\\boxed{' + ground_truth + '}')\n",
            encoding="utf-8",
        )
        (reward / "__init__.py").write_text(
            "from . import math_dapo\n"
            "def default_compute_score(data_source, solution_str, ground_truth):\n"
            "    assert data_source == 'math_dapo'\n"
            "    return math_dapo.compute_score(solution_str, ground_truth)\n",
            encoding="utf-8",
        )
        # Retain the actual all-node orchestrator; replace only DNS/hardware and
        # cluster readiness operations. The generated host count follows the
        # script's real --expected-nodes argument, including ranks above one.
        helper = self.fixture.examples / "job_bootstrap.py"
        helper.write_text(
            helper.read_text(encoding="utf-8").replace(
                'print("192.0.2.10,192.0.2.11")',
                'count = int(sys.argv[sys.argv.index("--expected-nodes") + 1]) '
                'if "--expected-nodes" in sys.argv else 2\n'
                '    print(",".join("192.0.2.%d" % (10 + index) for index in range(count)))',
            ), encoding="utf-8",
        )
        data = self.fixture.project / "stem data"
        data.mkdir()
        row = {
            "data_source": "math_dapo", "ability": "math",
            "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "train", "index": 0},
        }
        for split in ("train", "val"):
            _aqa.pq.write_table(_aqa.pa.Table.from_pylist([row] * 128), data / (split + ".parquet"))
        for key in list(self.fixture.env):
            if key.startswith("STEM_") or key in ("TOTAL_NPUS", "VAL_BEFORE_TRAIN", "TEST_FREQ"):
                self.fixture.env.pop(key)
        self.fixture.env.update(STEM_DATA_DIR=_job.shell_path(data), PROJECT_NAME="huanyu_stem")
        self.configure_cluster(128)

    def configure_cluster(self, total):
        nodes = total // 16
        self.fixture.env.update(
            TOTAL_NPUS=str(total), VC_WORKER_NUM=str(nodes),
            VC_WORKER_HOSTS=",".join("node-%d" % index for index in range(nodes)),
        )

    def launch(self, updates=None, arguments=()):
        capture = self.case.launch(updates, arguments)
        return capture, _aqa.effective_arguments(capture)

    def assert_scaled(self, capture, actual, total, tp=8):
        expected = {
            "trainer.n_gpus_per_node": "16", "trainer.nnodes": str(total // 16),
            "data.train_batch_size": "128", "actor_rollout_ref.actor.ppo_mini_batch_size": "128",
            "actor_rollout_ref.rollout.n": "16",
            "data.max_prompt_length": "2048", "data.max_response_length": "10240",
            "actor_rollout_ref.rollout.max_model_len": "12288",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "12288",
            "actor_rollout_ref.rollout.max_num_seqs": "8",
            "actor_rollout_ref.rollout.tensor_model_parallel_size": str(tp),
            "actor_rollout_ref.rollout.agent.num_workers": "128",
            "actor_rollout_ref.rollout.data_parallel_size": "1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size": "1",
            "actor_rollout_ref.actor.fsdp_config.param_offload": "true",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "true",
            "actor_rollout_ref.actor.fsdp_config.offload_policy": "true",
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": "512",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": "-1",
            "actor_rollout_ref.ref.fsdp_config.fsdp_size": "-1",
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.actor.use_dynamic_bsz": "false",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch": "true",
            "actor_rollout_ref.model.override_config.huanyu_freeze_audio": "true",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_encoder": "false",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_projector": "false",
            "reward.reward_model.enable": "false",
            "reward.reward_model.enable_resource_pool": "false",
        }
        for key, value in expected.items():
            observed = actual[key].lower() if value in ("true", "false") else actual[key]
            self.assertEqual(observed, value, key)
        self.assertNotIn("genrm", actual.get("reward.custom_reward_function.path", "").lower())
        self.assertEqual(actual["reward.custom_reward_function.path"], "null")
        self.assertEqual(actual["reward.reward_manager.name"], "naive")
        self.assertEqual(capture["env"]["RM_ENABLE"], "0")
        self.assertEqual(capture["env"]["REWARD_POOL_NPUS"], "0")
        self.assertEqual(capture["env"]["HUANYU_STEM_TEXT_TRAIN"], "1")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "HUANYU_STEM_TEXT_TRAIN"]), "1")
        self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + "MINICPMO_HUANYU_ROLLOUT_TP"]), str(tp))
        self.assertEqual(actual["actor_rollout_ref.rollout.ignore_eos"].lower(), "false")
        self.assertEqual(capture["env"]["HUANYU_USE_PRECOMPUTE"], "0")

    def test_128_cards_rule_reward_and_final_batch_lengths(self):
        capture, actual = self.launch()
        self.assert_scaled(capture, actual, 128)
        self.assertEqual(len(self.fixture.events("train")), 1)
        starts = [event for event in self.fixture.events("ray") if "start" in event["args"]]
        self.assertEqual(len(starts), 1)
        self.assertIn("--head", starts[0]["args"])
        wait = next(event for event in self.fixture.events("helper") if event["args"][0] == "wait-cluster")
        ips = wait["args"][wait["args"].index("--expected-ips") + 1]
        self.assertEqual(len(ips.split(",")), 8)

    def test_256_cards_doubles_actor_nodes_without_increasing_per_gpu_batch(self):
        self.configure_cluster(256)
        capture, actual = self.launch()
        self.assert_scaled(capture, actual, 256)

    def test_tp16_override_is_preserved_through_precision_policy(self):
        capture, actual = self.launch({"ROLLOUT_TP": "16"})
        self.assert_scaled(capture, actual, 128, tp=16)

    def test_site_environment_cannot_restore_aqa_pools_batches_or_genrm(self):
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                "export NUM_NPUS=16 NNODES=1 RM_ENABLE=1 RM_NNODES=1 REWARD_POOL_NPUS=16\n"
                "export ROLLOUT_TP=4 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4\n"
                "export STEM_CONTEXT_LENGTH=5k STEM_CONTEXT_SLACK=128 STEM_REWARD_MODE=genrm\n"
                "export STEM_DYNAMIC_BSZ=true STEM_SYNC_EACH_MICROBATCH=false ROLLOUT_MAX_NUM_SEQS=32\n"
                "export HUANYU_USE_PRECOMPUTE=1 HUANYU_STEM_TEXT_TRAIN=0\n"
            )
        capture, actual = self.launch()
        self.assert_scaled(capture, actual, 128)

    def test_last_rank_in_each_cluster_only_joins_ray(self):
        for total in (128, 256):
            with self.subTest(total=total):
                self.configure_cluster(total)
                rank = total // 16 - 1
                result = self.fixture.run_launcher({"VC_TASK_INDEX": str(rank)})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.fixture.events("train"))
                rays = self.fixture.events("ray")
                self.assertEqual(len(rays), 1)
                self.assertIn("--block", rays[0]["args"])
                self.assertIn("--node-ip-address=192.0.2.%d" % (10 + rank), rays[0]["args"])
                self.assertNotIn("--head", rays[0]["args"])
                self.assertEqual(rays[0]["env"]["HUANYU_STEM_TEXT_TRAIN"], "1")
                self.fixture.events_file.write_text("", encoding="utf-8")

    def test_invalid_layout_stops_before_install_or_ray(self):
        for updates in (
            {"TOTAL_NPUS": "96"}, {"VC_TASK_INDEX": "8"},
            {"ROLLOUT_TP": "3"}, {"VC_WORKER_NUM": "16"},
        ):
            with self.subTest(updates=updates):
                result = self.fixture.run_launcher(updates)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("pip", "ray", "train"):
                    self.assertFalse(self.fixture.events(kind), kind)
                if self.fixture.events_file.exists():
                    self.fixture.events_file.write_text("", encoding="utf-8")

    def test_broken_rule_scorer_does_not_start_training(self):
        scorer = self.fixture.project / "verl/utils/reward_score/math_dapo.py"
        scorer.write_text("def compute_score(solution_str, ground_truth):\n    return 0.0\n", encoding="utf-8")
        result = self.fixture.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("rule scorer smoke check failed", result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))

    def test_stale_audit_flags_are_cleared_in_driver_and_ray_runtime(self):
        flags = {
            "HUANYU_CHECKPOINT_VALIDATION_MODE": "disk",
            "HUANYU_FULL_WEIGHT_AUDIT_DIR": "/old/audit",
            "HUANYU_AUDIO_PROBE_DIR": "/old/probe",
            "HUANYU_EMBEDDING_AB_DIR": "/old/embeddings",
        }
        capture, actual = self.launch(flags)
        for key in flags:
            self.assertEqual(capture["env"][key], "", key)
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), "", key)


if __name__ == "__main__":
    unittest.main()
