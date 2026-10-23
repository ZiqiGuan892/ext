"""Exercise the AQA two-node job profile with real training argv and stubbed Ray."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ENTRY = "run_huanyu_aqa_unfreeze_32card_job.sh"
ENV_PREFIX = "ray_kwargs.ray_init.runtime_env.env_vars."
MODE_ENV = {
    "HUANYU_FREEZE_AUDIO": "false", "HUANYU_USE_PRECOMPUTE": "0",
    "HUANYU_AQA_AUDIO_TRAIN": "1", "MINICPMO_USE_PRECOMPUTE": "0",
    "MINICPMO_TUNE_AUDIO": "1", "MINICPMO_TUNE_VISION": "0",
    "HUANYU_EMBED_CACHE_DIR": "", "Q36_BACKWARD_PREFETCH": "off",
}
PRECISION_ENV = {
    "MINICPMO_HUANYU_GDN_FP32_TRAINING": "1", "MINICPMO_GDN_FP32_REFERENCE": "1",
    "MINICPMO_HUANYU_LAYERWISE_RELOAD": "1", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE": "1",
    "MINICPMO_ISOLATE_MEDIA_ENCODERS": "1", "MINICPMO_STABLE_LINEAR": "0",
    "MINICPMO_GDN_PREFILL_REFERENCE": "0", "MINICPMO_MOE_TP_REFERENCE": "0",
    "MINICPMO_ROW_TP_REFERENCE": "0", "MINICPMO_HUANYU_ROLLOUT_TP": "4",
    "VLLM_OMNI_EXTERNAL_MODULES": "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks",
}

# Import fixture helpers by path, without importing the optional model package.
_spec = importlib.util.spec_from_file_location(
    "huanyu_aqa_job_fixture", Path(__file__).with_name("test_job_launcher_on_cpu.py")
)
_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_job)


def effective_arguments(capture):
    return dict(
        arg.lstrip("+").split("=", 1)
        for arg in capture["argv"] if "=" in arg and not arg.startswith("--")
    )


@unittest.skipUnless(_job.find_bash(), "Bash is required for launcher argv integration")
class AqaJobLauncherTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _job.HuanyuJobLauncherTests("runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.entry = self.fixture.examples / ENTRY
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        for key in list(self.fixture.env):
            if key.startswith(("MAX_", "ROLLOUT_", "TRAIN_", "PPO_", "ACTOR_", "REF_")):
                self.fixture.env.pop(key)
        # This profile must not require a cache that belongs to another run.
        (self.fixture.cache / "manifest.json").unlink()
        self.train = self.fixture.project / "aqa train.parquet"
        self.validation = self.fixture.project / "aqa validation.parquet"
        audio = self.fixture.project / "audio.wav"
        audio.write_bytes(b"fixture audio; only references are checked")
        self.row = {
            "extra_info": {"raw_source": "aqa_real"},
            "prompt": [{"role": "user", "content": "<audio> What was said?"}],
            "images": [], "videos": [], "audios": [str(audio)], "ability": "audio",
            "reward_model": {"style": "model", "ground_truth": "Spoken answer."},
        }
        pq.write_table(pa.Table.from_pylist([self.row] * 4), self.train)
        pq.write_table(pa.Table.from_pylist([self.row]), self.validation)
        self.fixture.env.update(
            HUANYU_AQA_TRAIN_FILE=_job.shell_path(self.train),
            HUANYU_AQA_VAL_FILE=_job.shell_path(self.validation),
        )
        capture = self.fixture.project / "capture_train.py"
        capture.write_text(
            capture.read_text(encoding="utf-8").replace(
                "env={key: os.environ.get(key) for key in keys}", "env=dict(os.environ)"
            ), encoding="utf-8",
        )
        command = self.fixture.command_stub
        command.write_text(
            command.read_text(encoding="utf-8").replace(
                'record(command, sys.argv[2:])',
                'record(command, sys.argv[2:], env=dict(os.environ))',
            ), encoding="utf-8",
        )
        native = self.fixture.package / "runtime_imports.py"
        native.write_text(
            native.read_text(encoding="utf-8").replace(
                'record("runtime", sys.argv[1:])',
                'record("runtime", sys.argv[1:], env=dict(os.environ))',
            ), encoding="utf-8",
        )
        self.use_real_training_composition()

    def use_real_training_composition(self):
        """Keep actual AQA checks and final overrides; omit hardware preflights."""
        examples = self.fixture.examples
        for name in (
            "run_huanyu_aqa_audio_train.sh", "check_aqa_audio_train.py", "run_output.sh",
            "graph_mode.sh", "precision_mode.sh", "run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh",
        ):
            shutil.copyfile(EXAMPLES / name, examples / name)
        # The intermediate 32-card wrapper checks the real server's model path.
        # Its training target is retained, so actual GenRM and backend defaults
        # still participate in final argument precedence.
        (examples / "run_qwen36_huanyu_32card.sh").write_text(
            '#!/usr/bin/env bash\nexec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/'
            'run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh" "$@"\n',
            encoding="utf-8", newline="\n",
        )
        backend = (EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
        command = re.search(r"^python3 -m verl_omni\.trainer\.main_omni\b", backend, re.MULTILINE)
        self.assertIsNotNone(command, "Backend training command not found")
        (examples / "run_qwen36_huanyu_gspo_npu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh" "$@"\n'
            '_CHAT_TEMPLATE_ARGS=()\n_DATASET_ARGS=()\nROLLOUT_NUM_WORKERS=4\n'
            + backend[command.start():], encoding="utf-8", newline="\n",
        )
        python_stub = self.fixture.bin / "python3"
        python_stub.write_text(
            python_stub.read_text(encoding="utf-8").replace(
                '#!/usr/bin/env bash\n',
                '#!/usr/bin/env bash\n'
                'if [[ "${1:-}" == -m && "${2:-}" == verl_omni.trainer.main_omni ]]; then\n'
                '  exec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_TRAIN_CAPTURE" "$@"\nfi\n', 1,
            ), encoding="utf-8", newline="\n",
        )

    def launch(self, updates=None, arguments=()):
        result = self.fixture.run_launcher(updates, arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.fixture.capture_file.is_file(), result.stdout + result.stderr)
        return json.loads(self.fixture.capture_file.read_text(encoding="utf-8"))

    def assert_online_environment(self, env):
        for key, value in MODE_ENV.items():
            self.assertEqual(env[key], value, key)

        for key, value in {
            "HUANYU_JOB_MODE": "aqa_unfreeze", "NNODES": "1", "NUM_NPUS": "16",
            "REWARD_POOL_NPUS": "16", "RM_ENABLE": "1",
        }.items():
            self.assertEqual(env[key], value, key)

    def assert_precision_environment(self, env, overrides=None):
        for key, value in dict(PRECISION_ENV, **(overrides or {})).items():
            self.assertEqual(env[key], value, key)

    def test_head_uses_aqa_online_static_training_and_genrm_with_5k_budget(self):
        capture = self.launch({"TOTAL_TRAINING_STEPS": "2"})
        self.assert_online_environment(capture["env"])
        self.assert_precision_environment(capture["env"])
        actual = effective_arguments(capture)
        expected = {
            "data.train_files": self.train.as_posix(),
            "data.val_files": self.validation.as_posix(),
            "data.custom_cls.path": "pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset",
            "actor_rollout_ref.model.architecture": "Qwen36HuanyuForTraining",
            "actor_rollout_ref.model.override_config.huanyu_freeze_audio": "false",
            "actor_rollout_ref.model.use_remove_padding": "false",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name": "qwen3_6_huanyu",
            "actor_rollout_ref.rollout.enable_prefix_caching": "false",
            "data.max_prompt_length": "3072", "data.max_response_length": "2048",
            "actor_rollout_ref.rollout.prompt_length": "3072",
            "actor_rollout_ref.rollout.response_length": "2048",
            "actor_rollout_ref.rollout.max_model_len": "5248",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "5248",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.actor.use_dynamic_bsz": "false",
            "actor_rollout_ref.actor.ppo_mini_batch_size": "4",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
            "actor_rollout_ref.actor.fsdp_config.forward_prefetch": "false",
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": "1",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb": "0",
            "actor_rollout_ref.rollout.n": "4", "data.train_batch_size": "4",
            "reward.reward_model.enable": "true", "reward.reward_model.n_gpus_per_node": "16",
            "reward.reward_model.nnodes": "1", "trainer.total_training_steps": "2",
        }
        for key, value in expected.items():
            observed = actual[key].lower() if value in ("true", "false") else actual[key]
            self.assertEqual(observed, value, key)
        for key, value in dict(MODE_ENV, **PRECISION_ENV).items():
            self.assertEqual(json.loads(actual[ENV_PREFIX + key]), value, key)
        self.assertEqual(json.loads(actual[ENV_PREFIX + "MINICPMO_MM_PROMPT_BUDGET"]), "3072")
        self.assertNotIn("audio_encoder", actual["actor_rollout_ref.model.exclude_modules"])
        self.assertIn("genrm_verifier", actual["reward.custom_reward_function.path"])
        self.assertEqual(capture["env"]["EXPERIMENT_NAME"], "huanyu_aqa_unfreeze_5k_32card")
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])
        self.assertEqual(len(self.fixture.events("train")), 1)
        for event in self.fixture.events("runtime") + self.fixture.events("ray"):
            self.assert_online_environment(event["env"])
            self.assert_precision_environment(event["env"])

    def test_external_entry_overrides_inherited_frozen_sampleall_and_16k_settings(self):
        temporary = tempfile.TemporaryDirectory(prefix=".external_aqa_", dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        external = Path(temporary.name)
        self.fixture.entry = external / "launch aqa.sh"
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        (external / "run_huanyu_32card_job.sh").write_text("exit 91\n", encoding="utf-8")
        capture = self.launch({
            "HUANYU_JOB_MODE": "frozen_cache", "HUANYU_FREEZE_AUDIO": "true",
            "HUANYU_USE_PRECOMPUTE": "1", "MINICPMO_USE_PRECOMPUTE": "1",
            "MINICPMO_TUNE_AUDIO": "0", "MINICPMO_TUNE_VISION": "1", "RM_ENABLE": "0",
            "TRAIN_FILE": "/old/sampleall/train.parquet", "VAL_FILE": "/old/sampleall/validation.parquet",
            "MAX_PROMPT_LENGTH": "14336", "MAX_RESPONSE_LENGTH": "4096",
            "ROLLOUT_PROMPT_LENGTH": "14368", "ROLLOUT_MAX_MODEL_LEN": "16512",
            "MAX_TOKEN_LEN_PER_GPU": "16512", "MINICPMO_MM_PROMPT_BUDGET": "14368",
            "TRAIN_BATCH_SIZE": "32", "PPO_MINI_BATCH_SIZE": "32", "ROLLOUT_N": "8",
        })
        self.assert_online_environment(capture["env"])
        actual = effective_arguments(capture)
        self.assertEqual(actual["data.train_files"], self.train.as_posix())
        self.assertEqual(actual["data.val_files"], self.validation.as_posix())
        self.assertEqual(actual["data.max_prompt_length"], "3072")
        self.assertEqual(actual["data.max_response_length"], "2048")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_model_len"], "5248")
        self.assertEqual(actual["data.train_batch_size"], "4")
        self.assertEqual(actual["actor_rollout_ref.actor.ppo_mini_batch_size"], "4")
        self.assertEqual(actual["actor_rollout_ref.rollout.n"], "4")

    def test_worker_joins_with_online_environment_and_never_trains(self):
        # The server env file is allowed to contain defaults from a frozen run.
        env_file = self.fixture.examples / "env_export_huanyu.sh"
        env_file.write_text(
            env_file.read_text(encoding="utf-8")
            + "export HUANYU_FREEZE_AUDIO=true HUANYU_USE_PRECOMPUTE=1 MINICPMO_TUNE_AUDIO=0\n"
            + "export Q36_BACKWARD_PREFETCH=on\n",
            encoding="utf-8",
        )
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--address=192.0.2.10:6925", ray[0]["args"])
        self.assertIn("--block", ray[0]["args"])
        self.assertNotIn("--head", ray[0]["args"])
        self.assert_online_environment(ray[0]["env"])
        self.assert_precision_environment(ray[0]["env"])
        for event in self.fixture.events("runtime"):
            self.assert_online_environment(event["env"])
            self.assert_precision_environment(event["env"])

    def test_precision_policy_survives_site_env_and_reaches_existing_ray(self):
        env_file = self.fixture.examples / "env_export_huanyu.sh"
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write(
                "export MINICPMO_HUANYU_GDN_FP32_TRAINING=0 MINICPMO_HUANYU_LAYERWISE_RELOAD=0\n"
                "export MINICPMO_GDN_FP32_REFERENCE=1 MINICPMO_STABLE_LINEAR=1\n"
                "export VLLM_OMNI_EXTERNAL_MODULES=old.whisper.entry ROLLOUT_TP=8\n"
            )
        explicit = {"MINICPMO_GDN_FP32_REFERENCE": "0", "MINICPMO_MOE_TP_REFERENCE": "4"}
        capture = self.launch(dict(explicit, HUANYU_TEST_RAY_LIVE="1", ROLLOUT_TP="8"), (
            "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
            '++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_HUANYU_LAYERWISE_RELOAD="0"',
            "actor_rollout_ref.rollout.enable_prefix_caching=true",
        ))
        self.assert_online_environment(capture["env"])
        self.assert_precision_environment(capture["env"], explicit)
        actual = effective_arguments(capture)
        self.assertEqual(actual["actor_rollout_ref.rollout.tensor_model_parallel_size"], "4")
        self.assertEqual(actual["actor_rollout_ref.rollout.enable_prefix_caching"], "false")
        for key, value in dict(PRECISION_ENV, **explicit).items():
            self.assertEqual(json.loads(actual[ENV_PREFIX + key]), value, key)
        self.assertTrue(self.fixture.events("runtime"))
        for event in self.fixture.events("runtime") + self.fixture.events("ray"):
            self.assert_online_environment(event["env"])
            self.assert_precision_environment(event["env"], explicit)
        self.assertFalse(any("start" in event["args"] for event in self.fixture.events("ray")))

    def test_invalid_aqa_data_stops_before_training(self):
        mixed = dict(self.row, extra_info={"raw_source": "vaqa"})
        pq.write_table(pa.Table.from_pylist([mixed]), self.validation)
        result = self.fixture.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mixed sources", result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))

    def test_invalid_job_mode_fails_before_setup(self):
        self.fixture.entry = self.fixture.examples / "run_huanyu_32card_job.sh"
        result = self.fixture.run_launcher({"HUANYU_JOB_MODE": "typo"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HUANYU_JOB_MODE", result.stdout + result.stderr)
        for kind in ("helper", "pip", "ray", "train"):
            self.assertFalse(self.fixture.events(kind), kind)

    def test_explicit_absolute_project_root_is_required(self):
        for root in (None, "relative/project", _job.shell_path(self.fixture.project / "missing")):
            with self.subTest(root=root):
                result = self.fixture.run_launcher({"PROJECT_ROOT": root})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("helper", "pip", "ray", "train"):
                    self.assertFalse(self.fixture.events(kind), kind)


if __name__ == "__main__":
    unittest.main()
