"""AQA preflight and deployment-wrapper contracts without Ray or model weights."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SPEC = importlib.util.spec_from_file_location("huanyu_aqa_preflight", EXAMPLES / "check_aqa_audio_train.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def aqa_row(audio_path):
    return {
        "extra_info": {"raw_source": "aqa_real"},
        "prompt": [{"role": "user", "content": "<audio> What was said?"}],
        "images": [],
        "videos": [],
        "audios": [str(audio_path)],
        "reward_model": {"style": "model", "ground_truth": "A spoken answer."},
        "ability": "audio",
    }


class AqaPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="huanyu_aqa_check_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.audio = self.directory / "audio.wav"
        # File-reference validation intentionally does not decode waveform data.
        self.audio.write_bytes(b"opaque-audio")

    def test_streaming_check_preserves_input_and_accepts_referenced_audio(self):
        parquet = self.directory / "train.parquet"
        pq.write_table(pa.Table.from_pylist([aqa_row(self.audio)] * 5), parquet)
        before = parquet.read_bytes()
        self.assertEqual(checker.check_split(parquet, minimum_rows=4, batch_size=2), 5)
        self.assertEqual(parquet.read_bytes(), before)

    def test_rejects_mixed_source_with_file_and_row_location(self):
        parquet = self.directory / "mixed.parquet"
        wrong = aqa_row(self.audio)
        wrong["extra_info"]["raw_source"] = "vaqa"
        pq.write_table(pa.Table.from_pylist([aqa_row(self.audio), wrong]), parquet)
        with self.assertRaisesRegex(ValueError, "row=1.*mixed sources"):
            checker.check_split(parquet, batch_size=1)

    def test_rejects_invalid_media_prompt_or_ground_truth(self):
        changes = [
            ("images", ["image.png"], "images/videos"),
            ("videos", ["video.mp4"], "images/videos"),
            ("audios", [], "non-empty list"),
            ("audios", [str(self.directory / "missing.wav")], "does not exist"),
            ("audios", [f"{self.audio}::"], "empty zarr"),
            ("prompt", [], "prompt"),
            ("prompt", [{"role": "system", "content": "Only a system"}], "user message"),
            ("prompt", [{"role": "user", "content": ""}], "non-empty string"),
            ("reward_model", {"ground_truth": None}, "ground_truth"),
            ("reward_model", {"ground_truth": "  "}, "ground_truth"),
            ("ability", "text", "ability"),
        ]
        for field, value, error in changes:
            with self.subTest(field=field, value=value):
                row = aqa_row(self.audio)
                row[field] = value
                with self.assertRaisesRegex(ValueError, error):
                    checker.validate_aqa_row(row, set())

    def test_accepts_audio_dict_bytes_and_zarr_without_opening_member(self):
        for item in (
            {"audio": str(self.audio)},
            {"audio_url": str(self.audio)},
            {"bytes": b"audio"},
            f"{self.audio}::member-id",
        ):
            row = aqa_row(self.audio)
            row["audios"] = [item]
            checker.validate_aqa_row(row, set())

    def test_fails_on_too_few_train_rows_or_missing_schema(self):
        parquet = self.directory / "short.parquet"
        pq.write_table(pa.Table.from_pylist([aqa_row(self.audio)]), parquet)
        with self.assertRaisesRegex(ValueError, "smaller than required"):
            checker.check_split(parquet, minimum_rows=4)
        pq.write_table(pa.Table.from_pylist([{"prompt": "x"}]), parquet)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            checker.check_split(parquet)


def find_bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name == "nt":
        return str(git_bash) if git_bash.is_file() else None
    return shutil.which("bash")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    if os.name == "nt":
        return f"/{value[0].lower()}{value[2:]}"
    return value


@unittest.skipUnless(find_bash(), "Bash is required for launcher argv integration")
class AudioTrainingLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="huanyu_aqa_launcher_")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.examples = self.project / "examples" / "qwen3_6_huanyu"
        self.examples.mkdir(parents=True)
        for name in ("run_huanyu_aqa_audio_train.sh", "check_aqa_audio_train.py"):
            shutil.copyfile(EXAMPLES / name, self.examples / name)
        self.model = self.project / "verl_omni_ext" / "models" / "qwen3_6_huanyu"
        self.model.mkdir(parents=True)
        (self.model / "runtime_imports.py").write_text("pass\n")
        self.bin = self.project / "bin"
        self.bin.mkdir()
        python_stub = self.bin / "python3"
        python_stub.write_text('#!/usr/bin/env bash\nexec "$HUANYU_TEST_PYTHON" "$@"\n', encoding="utf-8")
        python_stub.chmod(0o755)
        self.capture = self.project / "capture.json"
        capture = (
            "import json, os, pathlib, sys; "
            "pathlib.Path(os.environ['HUANYU_TEST_CAPTURE']).write_text("
            "json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ)}))"
        )
        (self.examples / "run_qwen36_huanyu_32card.sh").write_text(
            "#!/usr/bin/env bash\npython3 - \"$@\" <<'PY'\n" + capture + "\nPY\n", encoding="utf-8"
        )
        self.audio = self.project / "audio.wav"
        self.audio.write_bytes(b"audio")
        self.train = self.project / "train.parquet"
        self.validation = self.project / "validation.parquet"
        pq.write_table(pa.Table.from_pylist([aqa_row(self.audio)] * 8), self.train)
        pq.write_table(pa.Table.from_pylist([aqa_row(self.audio)]), self.validation)
        self.env = dict(os.environ)
        # Avoid caller training settings changing the tests' baseline.
        for key in list(self.env):
            if key.startswith(("HUANYU_", "MINICPMO_", "ROLLOUT_", "PPO_", "DEBUG_")):
                self.env.pop(key)
        self.env.update(
            PROJECT_ROOT=shell_path(self.project),
            TRAIN_FILE=shell_path(self.train),
            VAL_FILE=shell_path(self.validation),
            HUANYU_TEST_PYTHON=sys.executable,
            HUANYU_TEST_CAPTURE=str(self.capture),
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            HUANYU_USE_PRECOMPUTE="1",
            MINICPMO_USE_PRECOMPUTE="1",
            MINICPMO_TUNE_VISION="1",
            TRAIN_BATCH_SIZE="4",
            PPO_MINI_BATCH_SIZE="4",
            ROLLOUT_N="4",
            TOTAL_TRAINING_STEPS="2",
        )

    def run_script(self, updates=None, arguments=()):
        env = dict(self.env, **(updates or {}))
        return subprocess.run(
            [find_bash(), shell_path(self.examples / "run_huanyu_aqa_audio_train.sh"), *arguments],
            env=env, text=True, capture_output=True, timeout=60,
        )

    def test_unfrozen_mode_overrides_stale_cache_and_bypassing_cli_settings(self):
        result = self.run_script(arguments=(
            "actor_rollout_ref.actor.use_dynamic_bsz=true",
            "actor_rollout_ref.model.lora_rank=16",
            "actor_rollout_ref.actor.strategy=megatron",
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4",
            "trainer.n_gpus_per_node=8",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        ))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        capture = json.loads(self.capture.read_text())
        actual = dict(item.lstrip("+").split("=", 1) for item in capture["argv"])
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "false")
        self.assertEqual(actual["actor_rollout_ref.actor.use_dynamic_bsz"], "false")
        self.assertEqual(actual["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], "1")
        self.assertEqual(actual["actor_rollout_ref.actor.fsdp_config.forward_prefetch"], "false")
        self.assertEqual(actual["actor_rollout_ref.model.lora_rank"], "0")
        for role in ("actor", "ref"):
            self.assertEqual(actual[f"actor_rollout_ref.{role}.strategy"], "fsdp2")
            self.assertEqual(actual[f"actor_rollout_ref.{role}.ulysses_sequence_parallel_size"], "1")
            self.assertEqual(actual[f"actor_rollout_ref.{role}.fsdp_config.ulysses_sequence_parallel_size"], "1")
        self.assertEqual(actual["actor_rollout_ref.ref.log_prob_use_dynamic_bsz"], "false")
        self.assertEqual(actual["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb"], "0")
        self.assertEqual(actual["actor_rollout_ref.rollout.tensor_model_parallel_size"], "4")
        self.assertEqual(actual["trainer.n_gpus_per_node"], "16")
        self.assertEqual(actual["trainer.nnodes"], "1")
        self.assertEqual(actual["reward.reward_model.n_gpus_per_node"], "16")
        self.assertEqual(actual["reward.reward_model.nnodes"], "1")
        self.assertEqual(actual["trainer.total_training_steps"], "2")
        for key, value in (
            ("HUANYU_USE_PRECOMPUTE", "0"),
            ("MINICPMO_USE_PRECOMPUTE", "0"),
            ("MINICPMO_TUNE_AUDIO", "1"),
            ("MINICPMO_TUNE_VISION", "0"),
            ("HUANYU_AQA_AUDIO_TRAIN", "1"),
            ("Q36_BACKWARD_PREFETCH", "off"),
        ):
            self.assertEqual(capture["env"][key], value)
            self.assertEqual(json.loads(actual["ray_kwargs.ray_init.runtime_env.env_vars." + key]), value)

    def test_frozen_online_control_and_unlimited_steps(self):
        result = self.run_script({"HUANYU_FREEZE_AUDIO": "true", "TOTAL_TRAINING_STEPS": "-1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        capture = json.loads(self.capture.read_text())
        actual = dict(item.lstrip("+").split("=", 1) for item in capture["argv"])
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assertEqual(capture["env"]["MINICPMO_TUNE_AUDIO"], "0")
        self.assertEqual(capture["env"]["HUANYU_USE_PRECOMPUTE"], "0")
        self.assertEqual(actual["trainer.total_training_steps"], "null")

    def test_rejects_unequal_rank_batching_debug_mode_and_bad_hyperparameter(self):
        for updates, error in (
            ({"PPO_MINI_BATCH_SIZE": "8"}, "ppo_mini_batch"),
            ({"TRAIN_BATCH_SIZE": "6"}, "train_batch"),
            ({"DEBUG_MODE": "1"}, "TOTAL_TRAINING_STEPS=2"),
            ({"HUANYU_FREEZE_AUDIO": "maybe"}, "must be true or false"),
        ):
            with self.subTest(updates=updates):
                result = self.run_script(updates)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(error, result.stderr)
                self.assertFalse(self.capture.exists())

    def test_invalid_validation_source_stops_before_backend(self):
        bad = aqa_row(self.audio)
        bad["extra_info"]["raw_source"] = "text"
        pq.write_table(pa.Table.from_pylist([bad]), self.validation)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixed sources", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
