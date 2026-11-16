"""Exercise the prepared-Ray video entry without installing or starting services."""


import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# Retained as historical coverage for the removed example CLI.
import pytest
pytest.skip("Example CLI removed by the Huanyu examples consolidation", allow_module_level=True)



EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ENTRY = "run_huanyu_video_train.sh"
ENV_PREFIX = "ray_kwargs.ray_init.runtime_env.env_vars."
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required")
class VideoTrainingLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="huanyu video train ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        examples = self.root / "examples" / "qwen3_6_huanyu"
        examples.mkdir(parents=True)
        self.entry = examples / ENTRY
        shutil.copyfile(EXAMPLES / ENTRY, self.entry)
        self.capture = self.root / "capture.json"
        capture_script = self.root / "capture.py"
        capture_script.write_text(
            "import json, os, pathlib, sys\n"
            "prefixes = ('HUANYU_', 'MINICPMO_', 'TRAIN_', 'PPO_', 'ROLLOUT_', 'RM_', 'MAX_')\n"
            "keys = ('MODEL_PATH', 'TRAIN_FILE', 'VAL_FILE', 'Q36_CKPT_DIR', 'NUM_NPUS', 'NNODES', "
            "'REWARD_POOL_NPUS', 'TOTAL_TRAINING_STEPS', 'RAY_ADDRESS', 'HCCL_LOGIC_SUPERPOD_ID')\n"
            "env = {k: v for k, v in os.environ.items() if k in keys or k.startswith(prefixes)}\n"
            "pathlib.Path(os.environ['HUANYU_TEST_CAPTURE']).write_text("
            "json.dumps({'argv': sys.argv[1:], 'env': env}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        (examples / "run_huanyu.sh").write_text(
            '#!/usr/bin/env bash\nexec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_CAPTURE_SCRIPT" "$@"\n',
            encoding="utf-8",
        )
        # A new prepared entry must not accidentally delegate to a setup job.
        for name in ("prepare_huanyu_env.sh", "start_huanyu_ray.sh", "run_huanyu_32card_job.sh"):
            (examples / name).write_text('echo "unexpected setup/Ray entry" >&2\nexit 91\n', encoding="utf-8")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("HUANYU_", "MINICPMO_"))}
        self.env.update(
            PROJECT_ROOT=shell_path(self.root), HUANYU_TEST_PYTHON=sys.executable,
            HUANYU_TEST_CAPTURE_SCRIPT=str(capture_script), HUANYU_TEST_CAPTURE=str(self.capture),
            MSYS2_ARG_CONV_EXCL="*", MSYS2_ENV_CONV_EXCL="*",
            TRAIN_FILE="/old/aqa/train.parquet", VAL_FILE="/old/aqa/validation.parquet",
            MODEL_PATH="/old/model", TRAIN_BATCH_SIZE="32", MAX_PROMPT_LENGTH="2048",
            TOTAL_TRAINING_STEPS="-1", NUM_NPUS="16", NNODES="2", ROLLOUT_TP="8",
            HUANYU_USE_PRECOMPUTE="1", MINICPMO_USE_PRECOMPUTE="1", HUANYU_EMBED_CACHE_DIR="/old/cache",
            HUANYU_AQA_AUDIO_TRAIN="1", HUANYU_STEM_TEXT_TRAIN="1", MINICPMO_TUNE_VISION="1",
            MINICPMO_VIDEO_MAX_FRAMES="8", HCCL_LOGIC_SUPERPOD_ID="old", RAY_ADDRESS="10.0.0.1:6925",
        )

    def launch(self, updates=None, args=()):
        env = dict(self.env, **(updates or {}))
        result = subprocess.run([BASH, "--noprofile", "--norc", shell_path(self.entry), *args],
                                env=env, capture_output=True, text=True, timeout=30)
        return result

    def read_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        capture = json.loads(self.capture.read_text(encoding="utf-8"))
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"] if "=" in arg)
        return capture["env"], actual

    def test_prepared_environment_cannot_replace_video_data_or_enable_caches(self):
        env, actual = self.read_success(self.launch())
        self.assertEqual(env["TRAIN_FILE"], "/opt/huawei/dataset/infra_data/data/video_data/train.parquet")
        self.assertEqual(env["VAL_FILE"], "/opt/huawei/dataset/infra_data/data/video_data/validation.parquet")
        self.assertTrue(env["MODEL_PATH"].endswith("/2748"))
        self.assertEqual(env["Q36_CKPT_DIR"], env["MODEL_PATH"])
        self.assertEqual(env["RAY_ADDRESS"], "10.0.0.1:6925")
        self.assertEqual(env["HUANYU_SKIP_EDITABLE_INSTALL"], "1")
        self.assertNotIn("HCCL_LOGIC_SUPERPOD_ID", env)
        self.assertEqual(int(env["NUM_NPUS"]) * int(env["NNODES"]) + int(env["REWARD_POOL_NPUS"]), 32)
        self.assertEqual(int(env["PPO_MINI_BATCH_SIZE"]) * int(env["ROLLOUT_N"]) // 16, 1)
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        self.assertEqual(actual["data.train_max_samples"], "8")
        self.assertEqual(env["ROLLOUT_MAX_MODEL_LEN"], "32768")
        for key, value in {
            "HUANYU_USE_PRECOMPUTE": "0", "MINICPMO_USE_PRECOMPUTE": "0",
            "HUANYU_EMBED_CACHE_DIR": "", "MINICPMO_EMBED_CACHE_DIR": "",
            "HUANYU_AQA_AUDIO_TRAIN": "0", "HUANYU_STEM_TEXT_TRAIN": "0",
            "MINICPMO_TUNE_VISION": "0", "MINICPMO_VIDEO_FPS": "1.0", "MINICPMO_VIDEO_MAX_FRAMES": "192",
        }.items():
            self.assertEqual(env[key], value, key)
            self.assertEqual(json.loads(actual[ENV_PREFIX + key]), value, key)

    def test_dedicated_overrides_keep_sample_count_and_length_budgets_consistent(self):
        env, actual = self.read_success(self.launch({
            "HUANYU_VIDEO_MODEL_PATH": "/models/another 2748",
            "HUANYU_VIDEO_TRAIN_FILE": "/data/video train.parquet",
            "HUANYU_VIDEO_VAL_FILE": "/data/video val.parquet",
            "HUANYU_VIDEO_STEPS": "3", "HUANYU_VIDEO_BATCH_SIZE": "8",
            "HUANYU_VIDEO_PROMPT_LENGTH": "16384", "HUANYU_VIDEO_RESPONSE_LENGTH": "1024",
        }, ("trainer.save_freq=1",)))
        self.assertEqual(actual["trainer.total_training_steps"], "3")
        self.assertEqual(actual["data.train_max_samples"], "24")
        self.assertEqual(actual["trainer.save_freq"], "1")
        self.assertEqual(env["PPO_MINI_BATCH_SIZE"], "8")
        self.assertEqual(env["MAX_TOKEN_LEN_PER_GPU"], "17408")
        self.assertEqual(actual["actor_rollout_ref.rollout.max_num_batched_tokens"], "17408")
        self.assertEqual(actual["data.train_files"], env["TRAIN_FILE"])
        self.assertEqual(json.loads(actual[ENV_PREFIX + "Q36_CKPT_DIR"]), env["MODEL_PATH"])

    def test_cache_and_freeze_contract_survives_conflicting_forwarded_arguments(self):
        _, actual = self.read_success(self.launch(args=(
            f'++{ENV_PREFIX}HUANYU_USE_PRECOMPUTE="1"',
            "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=false",
        )))
        self.assertEqual(json.loads(actual[ENV_PREFIX + "HUANYU_USE_PRECOMPUTE"]), "0")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")

    def test_invalid_work_sizes_stop_before_delegating(self):
        for updates in ({"HUANYU_VIDEO_BATCH_SIZE": "2"}, {"HUANYU_VIDEO_STEPS": "0"},
                        {"HUANYU_VIDEO_PROMPT_LENGTH": "32k"}):
            with self.subTest(updates=updates):
                result = self.launch(updates)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
