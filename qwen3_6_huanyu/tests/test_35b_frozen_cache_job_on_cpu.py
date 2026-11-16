"""Exercise the 35B cached profile through the shared job with command stubs."""


import importlib.util
import json
from pathlib import Path
import shutil
import unittest

# Retained as historical coverage for the removed example CLI.
import pytest
pytest.skip("Example CLI removed by the Huanyu examples consolidation", allow_module_level=True)



ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ENTRY = "run_huanyu_35b_frozen_cache_32card_job.sh"
ENV_PREFIX = "ray_kwargs.ray_init.runtime_env.env_vars."
MODEL = "/opt/huawei/dataset/model_gy/downloaded_models/2748"
DATA = "/opt/huawei/dataset/infra_data/data"

_spec = importlib.util.spec_from_file_location(
    "huanyu_35b_job_fixture", Path(__file__).with_name("test_job_launcher_on_cpu.py"))
_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_job)


def effective_arguments(capture):
    return dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"]
                if "=" in arg and not arg.startswith("--"))


@unittest.skipUnless(_job.find_bash(), "Bash is required for launcher argv integration")
class Frozen35BJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _job.HuanyuJobLauncherTests("runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.entry = self.fixture.examples / ENTRY
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        self.fixture.env["HUANYU_35B_CACHE_DIR"] = _job.shell_path(self.fixture.cache)
        for key in list(self.fixture.env):
            if key.startswith(("MAX_", "ROLLOUT_", "TRAIN_", "PPO_", "ACTOR_", "REF_")) or key == "AQA_DYNAMIC_BSZ":
                self.fixture.env.pop(key)
        capture = self.fixture.project / "capture_train.py"
        capture.write_text(capture.read_text(encoding="utf-8").replace(
            "env={key: os.environ.get(key) for key in keys}", "env=dict(os.environ)"), encoding="utf-8")
        stub = self.fixture.command_stub
        stub.write_text(stub.read_text(encoding="utf-8").replace(
            'record(command, sys.argv[2:])', 'record(command, sys.argv[2:], env=dict(os.environ))'), encoding="utf-8")
        self.fixture.env["MSYS2_ENV_CONV_EXCL"] += (
            ";MODEL_PATH;TRAIN_FILE;VAL_FILE;HUANYU_35B_MODEL_PATH;HUANYU_35B_TRAIN_FILE;HUANYU_35B_VAL_FILE")

    def launch(self, updates=None, arguments=()):
        result = self.fixture.run_launcher(updates, arguments)
        self.fixture.assert_success(result)
        return result, json.loads(self.fixture.capture_file.read_text(encoding="utf-8"))

    def assert_frozen_cache(self, env):
        for key, expected in {
            "HUANYU_USE_PRECOMPUTE": "1", "MINICPMO_USE_PRECOMPUTE": "0",
            "HUANYU_FREEZE_AUDIO": "true", "HUANYU_TRAIN_AUDIO_ENCODER": "false",
            "HUANYU_TRAIN_AUDIO_PROJECTOR": "false", "MINICPMO_TUNE_AUDIO": "0",
            "MINICPMO_TUNE_VISION": "0", "HUANYU_AQA_AUDIO_TRAIN": "0",
            "HUANYU_STEM_TEXT_TRAIN": "0",
        }.items():
            self.assertEqual(env.get(key), expected, key)

    def test_head_prepares_ray_then_trains_with_2748_and_aqa_sized_defaults(self):
        _, capture = self.launch()
        self.assert_frozen_cache(capture["env"])
        self.assertEqual(capture["env"]["MODEL_PATH"], MODEL)
        self.assertEqual(capture["env"]["TRAIN_FILE"], DATA + "/train.parquet")
        self.assertEqual(capture["env"]["VAL_FILE"], DATA + "/validation.parquet")
        actual = effective_arguments(capture)
        for key, expected in {
            "data.train_batch_size": "16", "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
            "actor_rollout_ref.rollout.n": "8", "data.max_prompt_length": "2048",
            "data.max_response_length": "3072", "actor_rollout_ref.rollout.max_model_len": "5248",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "5248",
            "actor_rollout_ref.actor.use_dynamic_bsz": "true",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "true",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "true",
            "actor_rollout_ref.model.override_config.huanyu_freeze_audio": "true",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_encoder": "false",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_projector": "false",
        }.items():
            self.assertEqual(actual[key], expected, key)
        for key in ("HUANYU_USE_PRECOMPUTE", "HUANYU_FREEZE_AUDIO", "HUANYU_TRAIN_AUDIO_ENCODER",
                    "HUANYU_TRAIN_AUDIO_PROJECTOR", "HUANYU_AQA_AUDIO_TRAIN", "HUANYU_STEM_TEXT_TRAIN"):
            self.assertEqual(json.loads(actual[ENV_PREFIX + key]), capture["env"][key], key)
        events = [json.loads(line) for line in self.fixture.events_file.read_text().splitlines()]
        positions = {kind: next(i for i, item in enumerate(events) if item["kind"] == kind)
                     for kind in ("env-source", "ray", "train")}
        self.assertLess(positions["env-source"], positions["ray"])
        self.assertLess(positions["ray"], positions["train"])
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])

    def test_dedicated_data_and_cache_survive_stale_aqa_and_site_environment(self):
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                'export MODEL_PATH=/old/model TRAIN_FILE=/old/train VAL_FILE=/old/val\n'
                'export HUANYU_AQA_TRAIN_FILE=/old/aqa_train HUANYU_AQA_VAL_FILE=/old/aqa_val\n'
                'export HUANYU_EMBED_CACHE_DIR=/old/cache HUANYU_USE_PRECOMPUTE=0\n'
                'export HUANYU_TRAIN_AUDIO_ENCODER=true HUANYU_TRAIN_AUDIO_PROJECTOR=true\n'
                'export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4\n'
                'export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 AQA_DYNAMIC_BSZ=false\n')
        _, capture = self.launch({
            "HUANYU_35B_TRAIN_FILE": "/data/35b train.parquet",
            "HUANYU_35B_VAL_FILE": "/data/35b validation.parquet",
            "MODEL_PATH": "/stale/aqa_model", "TRAIN_FILE": "/stale/aqa_train", "VAL_FILE": "/stale/aqa_val",
            "HUANYU_AQA_TRAIN_FILE": "/stale/aqa train.parquet",
            "HUANYU_AQA_VAL_FILE": "/stale/aqa validation.parquet",
            "HUANYU_USE_PRECOMPUTE": "0", "HUANYU_FREEZE_AUDIO": "false",
            "HUANYU_TRAIN_AUDIO_ENCODER": "true", "HUANYU_TRAIN_AUDIO_PROJECTOR": "true",
            "HUANYU_AQA_AUDIO_TRAIN": "1", "HUANYU_STEM_TEXT_TRAIN": "1",
        }, arguments=(
            "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=false",
            "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=true",
        ))
        self.assert_frozen_cache(capture["env"])
        self.assertEqual(capture["env"]["MODEL_PATH"], MODEL)
        self.assertEqual(capture["env"]["TRAIN_FILE"], "/data/35b train.parquet")
        self.assertEqual(capture["env"]["VAL_FILE"], "/data/35b validation.parquet")
        self.assertEqual(capture["env"]["HUANYU_EMBED_CACHE_DIR"], self.fixture.cache.as_posix())
        for key, expected in {"TRAIN_BATCH_SIZE": "16", "PPO_MINI_BATCH_SIZE": "16", "ROLLOUT_N": "8",
                              "MAX_PROMPT_LENGTH": "2048", "MAX_RESPONSE_LENGTH": "3072",
                              "AQA_DYNAMIC_BSZ": "true"}.items():
            self.assertEqual(capture["env"][key], expected, key)
        actual = effective_arguments(capture)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_train_audio_encoder"], "false")

    def test_environment_overrides_and_trailing_hydra_arguments_keep_precedence(self):
        _, capture = self.launch({
            "HUANYU_35B_MODEL_PATH": "/models/2748-copy", "HUANYU_35B_TRAIN_FILE": "/data/custom train.parquet",
            "HUANYU_35B_VAL_FILE": "/data/custom val.parquet", "TRAIN_BATCH_SIZE": "32",
            "PPO_MINI_BATCH_SIZE": "16", "ROLLOUT_N": "4", "MAX_PROMPT_LENGTH": "1024",
            "MAX_RESPONSE_LENGTH": "2048", "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "8192",
            "AQA_DYNAMIC_BSZ": "false",
        }, arguments=("actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true", "trainer.test_freq=3"))
        actual = effective_arguments(capture)
        self.assertEqual(capture["env"]["MODEL_PATH"], "/models/2748-copy")
        self.assertEqual(capture["env"]["TRAIN_FILE"], "/data/custom train.parquet")
        for key, expected in {
            "data.train_batch_size": "32", "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
            "actor_rollout_ref.rollout.n": "4", "data.max_prompt_length": "1024",
            "data.max_response_length": "2048", "actor_rollout_ref.rollout.max_model_len": "3200",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "8192",
            "actor_rollout_ref.actor.use_dynamic_bsz": "false",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "false",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "true", "trainer.test_freq": "3",
        }.items():
            self.assertEqual(actual[key], expected, key)

    def test_worker_joins_with_frozen_cache_and_never_submits_training(self):
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.fixture.assert_worker_supervised_until_exit(result)
        self.assertFalse(self.fixture.events("train"))
        ray, = self.fixture.events("ray")
        self.assertNotIn("--head", ray["args"])
        self.assertIn('--resources={"NPU": 16}', ray["args"])
        self.assert_frozen_cache(ray["env"])

    def test_missing_manifest_prevents_ray_and_training(self):
        (self.fixture.cache / "manifest.json").unlink()
        result = self.fixture.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cache manifest not readable", result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("ray"))
        self.assertFalse(self.fixture.events("train"))


if __name__ == "__main__":
    unittest.main()
