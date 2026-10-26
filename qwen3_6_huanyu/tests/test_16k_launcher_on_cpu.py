"""Check the frozen 16K profile through real shell argv composition, without NPUs."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ENTRY = "run_huanyu_32card_16k_job.sh"
ENV_PREFIX = "ray_kwargs.ray_init.runtime_env.env_vars."

# Reuse only the existing fixture setup/helpers, not its test cases. Importing
# by path avoids importing the plugin and its optional model dependencies.
_spec = importlib.util.spec_from_file_location(
    "huanyu_job_test_fixture", Path(__file__).with_name("test_job_launcher_on_cpu.py")
)
_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_job)


def effective_arguments(capture):
    return dict(
        arg.lstrip("+").split("=", 1)
        for arg in capture["argv"]
        if "=" in arg and not arg.startswith("--")
    )


@unittest.skipUnless(_job.find_bash(), "Bash is required for launcher argv integration")
class Huanyu16KLauncherTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _job.HuanyuJobLauncherTests("runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.entry = self.fixture.examples / ENTRY
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        for key in list(self.fixture.env):
            if key.startswith(("MAX_", "ROLLOUT_", "TRAIN_", "PPO_", "ACTOR_", "REF_")):
                self.fixture.env.pop(key)
        capture = self.fixture.project / "capture_train.py"
        extra_keys = (
            "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "ROLLOUT_PROMPT_LENGTH",
            "ROLLOUT_MAX_MODEL_LEN", "MAX_TOKEN_LEN_PER_GPU", "MINICPMO_MM_PROMPT_BUDGET",
            "EXPERIMENT_NAME", "HUANYU_GRAPH_MODE",
            "HUANYU_16K_POOL_LAYOUT", "RM_N_GPUS_PER_NODE", "RM_NNODES", "RM_SPLIT_NODES",
            "ROLLOUT_TP", "RM_TP", "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "ROLLOUT_N",
            "NPUS_PER_NODE",
        )
        capture.write_text(
            capture.read_text(encoding="utf-8").replace(
                "keys = (\n", f"keys = {extra_keys!r} + (\n", 1
            ),
            encoding="utf-8",
        )

    def launch(self, updates=None, arguments=()):
        result = self.fixture.run_launcher(updates, arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.fixture.capture_file.is_file(), result.stdout + result.stderr)
        return json.loads(self.fixture.capture_file.read_text(encoding="utf-8"))

    def assert_lengths(self, capture, response=2048):
        prompt = 16384 - response
        expected_env = {
            "MAX_PROMPT_LENGTH": str(prompt), "MAX_RESPONSE_LENGTH": str(response),
            "ROLLOUT_PROMPT_LENGTH": str(prompt + 32),
            "ROLLOUT_MAX_MODEL_LEN": "16512", "MAX_TOKEN_LEN_PER_GPU": "16512",
            "MINICPMO_MM_PROMPT_BUDGET": str(prompt + 32),
        }
        for key, value in expected_env.items():
            self.assertEqual(capture["env"][key], value, key)
        actual = effective_arguments(capture)
        expected_args = {
            "data.max_prompt_length": str(prompt), "data.max_response_length": str(response),
            "actor_rollout_ref.rollout.prompt_length": str(prompt + 32),
            "actor_rollout_ref.rollout.response_length": str(response),
            "actor_rollout_ref.rollout.max_model_len": "16512",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "16512",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "16512",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "16512",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "16512",
        }
        for key, value in expected_args.items():
            self.assertEqual(actual[key], value, key)
        # Runtime environments must receive a string, not Hydra's integer.
        media = json.loads(actual[ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"])
        self.assertEqual(media, str(prompt + 32))
        return actual

    def test_default_head_replaces_stale_5k_budgets_and_keeps_frozen_cache(self):
        capture = self.launch({
            "HUANYU_JOB_MODE": "aqa_unfreeze",
            "MAX_PROMPT_LENGTH": "3072", "MAX_RESPONSE_LENGTH": "512",
            "ROLLOUT_PROMPT_LENGTH": "3104", "ROLLOUT_MAX_MODEL_LEN": "5248",
            "MAX_TOKEN_LEN_PER_GPU": "5248", "MINICPMO_MM_PROMPT_BUDGET": "3104",
            "NUM_NPUS": "12", "NNODES": "2", "RM_N_GPUS_PER_NODE": "4",
            "RM_NNODES": "2", "REWARD_POOL_NPUS": "8", "RM_SPLIT_NODES": "0",
        })
        actual = self.assert_lengths(capture)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "sdpa")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_num_seqs"], "32")
        self.assertEqual(actual["actor_rollout_ref.rollout.disable_log_stats"], "false")
        self.assertEqual(capture["env"]["EXPERIMENT_NAME"], "huanyu_16k_sampleall_cache_32card")
        for key, expected in {
            "HUANYU_FREEZE_AUDIO": "true", "HUANYU_USE_PRECOMPUTE": "1",
            "MINICPMO_TUNE_AUDIO": "0", "NNODES": "1", "NUM_NPUS": "16",
            "REWARD_POOL_NPUS": "16", "RM_ENABLE": "1",
            "HUANYU_16K_POOL_LAYOUT": "16+16", "RM_N_GPUS_PER_NODE": "16",
            "RM_NNODES": "1", "RM_SPLIT_NODES": "1", "ROLLOUT_TP": "4", "RM_TP": "4",
            "TRAIN_BATCH_SIZE": "32", "PPO_MINI_BATCH_SIZE": "32", "ROLLOUT_N": "4",
        }.items():
            self.assertEqual(capture["env"][key], expected, key)
        self.assertEqual(len(self.fixture.events("train")), 1)
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])

    def test_profile_options_preserve_total_length_and_allow_eager_fallback(self):
        capture = self.launch({
            "HUANYU_16K_RESPONSE_LENGTH": "4096",
            "HUANYU_16K_ATTN_IMPLEMENTATION": "eager",
            "HUANYU_16K_MAX_NUM_SEQS": "16",
            "EXPERIMENT_NAME": "my_16k_comparison",
        })
        actual = self.assert_lengths(capture, response=4096)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "eager")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_num_seqs"], "16")
        self.assertEqual(capture["env"]["EXPERIMENT_NAME"], "my_16k_comparison")

    def test_invalid_options_stop_before_installing_or_starting_ray(self):
        for key, values in {
            "HUANYU_16K_RESPONSE_LENGTH": ("0", "-1", "16384", "16385", "1.5", "abc"),
            "HUANYU_16K_MAX_NUM_SEQS": ("0", "-1", "abc"),
            "HUANYU_16K_ATTN_IMPLEMENTATION": ("auto", "flash_attention_2"),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    result = self.fixture.run_launcher({key: value})
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    for kind in ("helper", "pip", "ray", "train"):
                        self.assertFalse(self.fixture.events(kind), kind)

    def test_invalid_pool_tp_or_batch_stops_before_setup(self):
        invalid = (
            {"HUANYU_16K_POOL_LAYOUT": "24/8"},
            {"HUANYU_16K_POOL_LAYOUT": "32+0"},
            {"ROLLOUT_TP": "0"},
            {"ROLLOUT_TP": "04"},
            {"ROLLOUT_TP": "abc"},
            {"ROLLOUT_TP": "8"},
            {"RM_TP": "0"},
            {"RM_TP": "8"},
            {"TRAIN_BATCH_SIZE": "0"},
            {"TRAIN_BATCH_SIZE": "32"},
            {"PPO_MINI_BATCH_SIZE": "32"},
            {"PPO_MINI_BATCH_SIZE": "18"},
            {"ROLLOUT_N": "0"},
        )
        for updates in invalid:
            with self.subTest(updates=updates):
                result = self.fixture.run_launcher({"HUANYU_16K_POOL_LAYOUT": "24+8", **updates})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("helper", "pip", "env-source", "ray", "train"):
                    self.assertFalse(self.fixture.events(kind), kind)

    def test_worker_joins_without_running_training(self):
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(self.fixture.capture_file.exists())
        self.assertEqual(len(self.fixture.events("ray")), 1)
        args = self.fixture.events("ray")[0]["args"]
        self.assertIn("--address=192.0.2.10:6925", args)
        self.assertIn("--block", args)
        self.assertNotIn("--head", args)

    def test_24_plus_8_worker_keeps_physical_16_cards_and_only_joins_ray(self):
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1", "HUANYU_16K_POOL_LAYOUT": "24+8"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(self.fixture.capture_file.exists())
        self.assertEqual(len(self.fixture.events("ray")), 1)
        args = self.fixture.events("ray")[0]["args"]
        self.assertIn('--resources={"NPU": 16}', args)
        self.assertIn("--address=192.0.2.10:6925", args)
        self.assertIn("--block", args)
        self.assertNotIn("--head", args)

    def test_external_entry_uses_project_backend_not_adjacent_script(self):
        # Both script location and cwd differ from PROJECT_ROOT. A sibling
        # decoy must never be executed, even if it has the expected filename.
        temporary = tempfile.TemporaryDirectory(
            prefix=".external_16k_", dir=ROOT / "tests",
        )
        self.addCleanup(temporary.cleanup)
        external = Path(temporary.name)
        self.fixture.entry = external / "launch 16k.sh"
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        (external / "run_huanyu_32card_job.sh").write_text(
            '#!/usr/bin/env bash\necho "WRONG adjacent backend" >&2\nexit 91\n',
            encoding="utf-8", newline="\n",
        )
        self.use_real_training_composition()
        actual = self.assert_lengths(self.launch(arguments=("trainer.total_training_steps=2",)))
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        self.assertEqual(len(self.fixture.events("train")), 1)
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])

    def test_default_root_is_selected_and_invalid_explicit_roots_stop_before_setup(self):
        # Xtrace shows the selected real default without creating /opt paths.
        # Invalid length guarantees this cannot launch the real server backend
        # even when these CPU tests run on a host where that checkout exists.
        result = self.fixture.run_launcher({"PROJECT_ROOT": None, "HUANYU_16K_RESPONSE_LENGTH": "invalid"}, trace=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"PROJECT_ROOT={_job.DEFAULT_PROJECT_ROOT}", result.stderr)
        self.assertIn(f"{_job.DEFAULT_PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh", result.stderr)
        for root in ("relative/project", _job.shell_path(self.fixture.project / "missing")):
            with self.subTest(root=root):
                result = self.fixture.run_launcher({"PROJECT_ROOT": root})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("helper", "pip", "ray", "train"):
                    self.assertFalse(self.fixture.events(kind), kind)

    def use_real_training_composition(self):
        """Run actual mix defaults and backend argv, omit hardware preflight."""
        examples = self.fixture.examples
        mix = "run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh"
        shutil.copyfile(EXAMPLES / mix, examples / mix)
        shutil.copyfile(EXAMPLES / "graph_mode.sh", examples / "graph_mode.sh")
        shutil.copyfile(EXAMPLES / "precision_mode.sh", examples / "precision_mode.sh")
        source = (EXAMPLES / "run_huanyu.sh").read_text(encoding="utf-8")
        start = source.index('exec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/')
        # Retain the real Huanyu entry's final protected overrides. The middle
        # 32-card wrapper only adds path preflights, env defaults and filter=false.
        terminal_exec = source[start:].replace("run_qwen36_huanyu_32card.sh", mix, 1)
        (examples / "run_huanyu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            '_huanyu_ray_pythonpath=\'"fixture-pythonpath"\'\n' + terminal_exec,
            encoding="utf-8", newline="\n",
        )
        backend = (EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
        command = re.search(r"^python3 -m verl_omni\.trainer\.main_omni\b", backend, re.MULTILINE)
        self.assertIsNotNone(command, "Backend training command not found")
        workers = re.search(r"^ROLLOUT_NUM_WORKERS=.*$", backend, re.MULTILINE)
        self.assertIsNotNone(workers, "Backend rollout replica calculation not found")
        (examples / "run_qwen36_huanyu_gspo_npu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh" "$@"\n'
            '_CHAT_TEMPLATE_ARGS=()\n_DATASET_ARGS=()\n'
            'DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints"\n'
            + workers.group(0) + "\n"
            + backend[command.start():],
            encoding="utf-8", newline="\n",
        )
        python_stub = self.fixture.bin / "python3"
        source = python_stub.read_text(encoding="utf-8")
        source = source.replace(
            '#!/usr/bin/env bash\n',
            '#!/usr/bin/env bash\n'
            'if [[ "${1:-}" == -m && "${2:-}" == verl_omni.trainer.main_omni ]]; then\n'
            '  exec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_TRAIN_CAPTURE" "$@"\nfi\n',
            1,
        )
        python_stub.write_text(source, encoding="utf-8", newline="\n")

    def test_final_backend_preserves_16k_against_mix_defaults(self):
        self.use_real_training_composition()
        actual = self.assert_lengths(self.launch())
        for key, expected in {
            "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
            "actor_rollout_ref.rollout.max_num_seqs": "32",
            "actor_rollout_ref.rollout.disable_log_stats": "false",
            "actor_rollout_ref.model.enable_gradient_checkpointing": "true",
            "actor_rollout_ref.actor.use_dynamic_bsz": "true",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "true",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "true",
            "actor_rollout_ref.rollout.enable_chunked_prefill": "false",
            "actor_rollout_ref.model.use_remove_padding": "false",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
            "actor_rollout_ref.rollout.tensor_model_parallel_size": "4",
            "actor_rollout_ref.rollout.gpu_memory_utilization": "0.5",
            "reward.reward_model.rollout.max_model_len": "32768",
            "actor_rollout_ref.rollout.enforce_eager": "true",
            "reward.reward_model.rollout.enforce_eager": "true",
            "actor_rollout_ref.rollout.n": "4",
            "actor_rollout_ref.rollout.agent.num_workers": "4",
            "trainer.n_gpus_per_node": "16",
            "trainer.nnodes": "1",
            "reward.reward_model.n_gpus_per_node": "16",
            "reward.reward_model.nnodes": "1",
            "reward.reward_model.rollout.tensor_model_parallel_size": "4",
            "data.train_batch_size": "32",
            "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
            "actor_rollout_ref.model.path": _job.DEFAULT_MODEL_PATH,
            "reward.reward_model.model_path": _job.DEFAULT_RM_MODEL_PATH,
            "data.train_files": _job.DEFAULT_TRAIN_FILE,
            "data.val_files": _job.DEFAULT_VAL_FILE,
        }.items():
            observed = actual[key].lower() if expected in ("true", "false") else actual[key]
            self.assertEqual(observed, expected, key)

    def assert_24_plus_8_backend(self, capture, rollout_tp=4, rm_tp=4, train_batch=24, mini_batch=24):
        actual = self.assert_lengths(capture)
        for key, expected in {
            "HUANYU_16K_POOL_LAYOUT": "24+8", "NUM_NPUS": "12", "NNODES": "2",
            "RM_N_GPUS_PER_NODE": "4", "RM_NNODES": "2", "REWARD_POOL_NPUS": "8",
            "RM_SPLIT_NODES": "0", "NPUS_PER_NODE": "16",
            "ROLLOUT_TP": str(rollout_tp), "RM_TP": str(rm_tp),
            "TRAIN_BATCH_SIZE": str(train_batch), "PPO_MINI_BATCH_SIZE": str(mini_batch),
            "ROLLOUT_N": "4",
        }.items():
            self.assertEqual(capture["env"][key], expected, key)
        for key, expected in {
            "trainer.n_gpus_per_node": "12", "trainer.nnodes": "2",
            "reward.reward_model.n_gpus_per_node": "4", "reward.reward_model.nnodes": "2",
            "actor_rollout_ref.rollout.tensor_model_parallel_size": str(rollout_tp),
            "reward.reward_model.rollout.tensor_model_parallel_size": str(rm_tp),
            "actor_rollout_ref.rollout.agent.num_workers": str(24 // rollout_tp),
            "actor_rollout_ref.rollout.n": "4", "data.train_batch_size": str(train_batch),
            "actor_rollout_ref.actor.ppo_mini_batch_size": str(mini_batch),
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": "-1",
            "actor_rollout_ref.ref.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.ref.fsdp_config.fsdp_size": "-1",
        }.items():
            self.assertEqual(actual[key], expected, key)
        reward_world = int(actual["reward.reward_model.n_gpus_per_node"]) * int(actual["reward.reward_model.nnodes"])
        reward_tp = int(actual["reward.reward_model.rollout.tensor_model_parallel_size"])
        self.assertEqual(reward_world // reward_tp, 8 // rm_tp)
        self.assertIn('--resources={"NPU": 16}', self.fixture.events("ray")[0]["args"])
        self.assertEqual(len(self.fixture.events("train")), 1)

    def test_opt_in_24_plus_8_reaches_final_backend(self):
        self.use_real_training_composition()
        capture = self.launch({"HUANYU_16K_POOL_LAYOUT": "24+8"})
        self.assert_24_plus_8_backend(capture)

    def test_24_plus_8_preserves_upper_pool_and_tp_through_site_environment(self):
        self.use_real_training_composition()
        env_file = self.fixture.examples / "env_export_huanyu.sh"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "export NUM_NPUS=16 NNODES=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1\n"
                "export REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1 ROLLOUT_TP=8 RM_TP=8\n"
                "export MODEL_PATH=/stale/whisper RM_MODEL_PATH=/stale/reward\n"
                "export TRAIN_FILE=/stale/train.parquet VAL_FILE=/stale/validation.parquet\n"
            )
        paths = {"MODEL_PATH": "/explicit/actor", "RM_MODEL_PATH": "/explicit/reward",
                 "TRAIN_FILE": "/explicit/train.parquet", "VAL_FILE": "/explicit/validation.parquet"}
        capture = self.launch({
            "HUANYU_16K_POOL_LAYOUT": "24+8", "ROLLOUT_TP": "2", "RM_TP": "2",
            "TRAIN_BATCH_SIZE": "48", "PPO_MINI_BATCH_SIZE": "24",
            **paths,
        })
        self.assert_24_plus_8_backend(capture, rollout_tp=2, rm_tp=2, train_batch=48)
        actual = effective_arguments(capture)
        for env, config in (("MODEL_PATH", "actor_rollout_ref.model.path"),
                            ("RM_MODEL_PATH", "reward.reward_model.model_path"),
                            ("TRAIN_FILE", "data.train_files"), ("VAL_FILE", "data.val_files")):
            self.assertEqual(actual[config], paths[env], config)

    def test_user_hydra_overrides_survive_but_graph_selector_keeps_authority(self):
        self.use_real_training_composition()
        capture = self.launch(
            {"HUANYU_PRECISION_PROFILE": "custom", "HUANYU_GRAPH_MODE": "1"},
            arguments=(
                "--scheduler_mount=/not/a/hydra/key",
                "actor_rollout_ref.model.override_config.attn_implementation=eager",
                "actor_rollout_ref.rollout.max_num_seqs=16",
                "actor_rollout_ref.rollout.enforce_eager=true",
                "trainer.total_training_steps=2",
                "trainer.experiment_name=long context comparison",
            ),
        )
        actual = self.assert_lengths(capture)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.attn_implementation"], "eager")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_num_seqs"], "16")
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        self.assertEqual(actual["trainer.experiment_name"], "long context comparison")
        self.assertEqual(actual["actor_rollout_ref.rollout.enforce_eager"], "false")
        self.assertEqual(actual["reward.reward_model.rollout.enforce_eager"], "false")
        self.assertFalse(any(arg.startswith("--scheduler_mount=") for arg in capture["argv"]))


if __name__ == "__main__":
    unittest.main()
