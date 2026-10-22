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
            "MAX_PROMPT_LENGTH": "3072", "MAX_RESPONSE_LENGTH": "512",
            "ROLLOUT_PROMPT_LENGTH": "3104", "ROLLOUT_MAX_MODEL_LEN": "5248",
            "MAX_TOKEN_LEN_PER_GPU": "5248", "MINICPMO_MM_PROMPT_BUDGET": "3104",
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

    def test_missing_relative_or_wrong_project_root_stops_before_setup(self):
        for root in (None, "relative/project", _job.shell_path(self.fixture.project / "missing")):
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
        (examples / "run_qwen36_huanyu_gspo_npu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            '_CHAT_TEMPLATE_ARGS=()\n_DATASET_ARGS=()\n'
            'ROLLOUT_NUM_WORKERS=4\nDEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints"\n'
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
            "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
        }.items():
            self.assertEqual(actual[key].lower(), expected, key)

    def test_user_hydra_overrides_survive_but_graph_selector_keeps_authority(self):
        self.use_real_training_composition()
        capture = self.launch(
            {"HUANYU_GRAPH_MODE": "1"},
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
