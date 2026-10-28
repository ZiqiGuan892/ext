"""Exercise WAV subset preparation and launcher handoff without Ray or models."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def find_bash():
    if os.name == "nt":
        path = Path("C:/Program Files/Git/bin/bash.exe")
        return str(path) if path.is_file() else None
    return shutil.which("bash")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@unittest.skipUnless(find_bash(), "Bash is required")
class Wav100LauncherTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("HUANYU_TEST_TMPDIR", tempfile.gettempdir())).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="wav100 launcher ", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.examples = self.project / "examples" / "qwen3_6_huanyu"
        self.examples.mkdir(parents=True)
        for name in ("run_huanyu_aqa_wav100.sh", "run_output.sh"):
            shutil.copyfile(EXAMPLES / name, self.examples / name)
        self.prep_capture = self.project / "prep.json"
        self.train_capture = self.project / "train.json"
        (self.examples / "prepare_aqa_wav_parquet.py").write_text(
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['TEST_PREP_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "if os.environ.get('TEST_PREP_FAIL'): sys.exit(17)\n"
            "p = pathlib.Path(sys.argv[sys.argv.index('--output-parquet') + 1])\n"
            "p.parent.mkdir(parents=True, exist_ok=True)\n"
            "p.write_bytes(b'prepared subset')\n",
            encoding="utf-8",
        )
        (self.examples / "run_huanyu_aqa_audio_train.sh").write_text(
            '#!/usr/bin/env bash\n'
            'source "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_output.sh"\n'
            'python3 - "$@" <<\'PY\'\n'
            'import json, os, pathlib, sys\n'
            "pathlib.Path(os.environ['TEST_TRAIN_CAPTURE']).write_text("
            "json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
            'PY\n', encoding="utf-8",
        )
        bin_dir = self.project / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "python3"
        stub.write_text('#!/usr/bin/env bash\nexec "$TEST_PYTHON" "$@"\n', encoding="utf-8")
        stub.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("HUANYU_", "_HUANYU_", "AQA_WAV_")):
                self.env.pop(key)
        self.env.pop("TOTAL_TRAINING_STEPS", None)
        self.env.update(PROJECT_ROOT=shell_path(self.project), TEST_PYTHON=sys.executable,
                        TEST_PREP_CAPTURE=str(self.prep_capture), TEST_TRAIN_CAPTURE=str(self.train_capture),
                        PATH=str(bin_dir) + os.pathsep + self.env.get("PATH", ""))

    def run_script(self, updates=None, arguments=()):
        return subprocess.run([find_bash(), shell_path(self.examples / "run_huanyu_aqa_wav100.sh"), *arguments],
                              env=dict(self.env, **(updates or {})), text=True, capture_output=True, timeout=30)

    def capture(self):
        value = json.loads(self.train_capture.read_text())
        value["options"] = dict(arg.lstrip("+").split("=", 1) for arg in value["argv"] if "=" in arg)
        return value

    def test_defaults_prepare_100_then_validate_and_train_with_frozen_audio(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        prep = dict(zip(*[iter(json.loads(self.prep_capture.read_text()))] * 2))
        self.assertEqual(prep["--sample-size"], "100")
        self.assertEqual(prep["--seed"], "42")
        value = self.capture()
        env, opts = value["env"], value["options"]
        self.assertEqual(env["TRAIN_FILE"], env["VAL_FILE"])
        self.assertEqual(env["TRAIN_FILE"], env["HUANYU_AQA_TRAIN_FILE"])
        self.assertEqual(env["VAL_FILE"], env["HUANYU_AQA_VAL_FILE"])
        self.assertTrue(env["TRAIN_FILE"].endswith("/wav100/train.parquet"))
        self.assertEqual(env["HUANYU_TRAIN_AUDIO_ENCODER"], "false")
        self.assertEqual(env["HUANYU_TRAIN_AUDIO_PROJECTOR"], "false")
        self.assertEqual(env["HUANYU_FREEZE_AUDIO"], "true")
        self.assertEqual(env["TOTAL_TRAINING_STEPS"], "25")
        self.assertEqual(env["NUM_NPUS"], "16")
        self.assertEqual(env["RM_N_GPUS_PER_NODE"], "16")
        self.assertEqual(env["ROLLOUT_TP"], "4")
        self.assertEqual(env["TRAIN_BATCH_SIZE"], "4")
        self.assertEqual(env["PPO_MINI_BATCH_SIZE"], "4")
        self.assertEqual(env["ROLLOUT_N"], "4")
        self.assertEqual(env["HUANYU_PRECISION_PROFILE"], "full")
        self.assertEqual(env["HUANYU_GRAPH_MODE"], "0")
        self.assertEqual(env["HUANYU_USE_PRECOMPUTE"], "0")
        self.assertEqual(opts["trainer.val_before_train"], "true")
        self.assertEqual(opts["trainer.total_epochs"], "1")
        self.assertEqual(opts["data.val_max_samples"], "100")
        self.assertEqual(opts["data.train_max_samples"], "100")
        self.assertEqual(len(list((self.project / "runs").glob("huanyu_*"))), 1)
        self.assertIn("same_subset_diagnostic=true", result.stdout)

    def test_explicit_paths_checkpoint_and_short_run_survive_but_stale_modes_do_not(self):
        result = self.run_script({"AQA_WAV_SOURCE_PARQUET": "/source/validation.parquet",
                                  "AQA_WAV_EXPORT_DIR": "/source/serving waves", "AQA_WAV_SAMPLE_SEED": "7",
                                  "MODEL_PATH": "/models/explicit", "RM_MODEL_PATH": "/models/reward",
                                  "TOTAL_TRAINING_STEPS": "2", "HUANYU_TRAIN_AUDIO_ENCODER": "true",
                                  "HUANYU_TRAIN_AUDIO_PROJECTOR": "true", "NUM_NPUS": "12",
                                  "RM_N_GPUS_PER_NODE": "4", "HUANYU_USE_PRECOMPUTE": "1",
                                  "TRAIN_FILE": "/old/train.parquet", "VAL_FILE": "/old/val.parquet"},
                                 ("actor_rollout_ref.rollout.temperature=1", "actor_rollout_ref.rollout.ignore_eos=true",
                                  "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens=10240",
                                  "data.val_max_samples=5", "trainer.test_freq=5", "trainer.save_freq=-1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        value = self.capture()
        env, opts = value["env"], value["options"]
        # Git Bash translates absolute POSIX paths when launching native Python.
        self.assertTrue(env["MODEL_PATH"].replace("\\", "/").endswith("/models/explicit"))
        self.assertTrue(env["RM_MODEL_PATH"].replace("\\", "/").endswith("/models/reward"))
        self.assertEqual(env["TOTAL_TRAINING_STEPS"], "2")
        self.assertEqual(env["HUANYU_TRAIN_AUDIO_ENCODER"], "false")
        self.assertEqual(env["NUM_NPUS"], "16")
        self.assertEqual(env["RM_N_GPUS_PER_NODE"], "16")
        self.assertNotEqual(env["TRAIN_FILE"], "/old/train.parquet")
        for key, expected in {"temperature": "0", "top_p": "1", "top_k": "-1",
                              "repetition_penalty": "1", "ignore_eos": "false"}.items():
            self.assertEqual(opts["actor_rollout_ref.rollout." + key], expected)
        self.assertEqual(opts["data.val_max_samples"], "100")
        self.assertEqual(opts["actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens"], "0")
        self.assertEqual(opts["trainer.test_freq"], "-1")
        self.assertEqual(opts["trainer.save_freq"], "-1")
        prep = json.loads(self.prep_capture.read_text())
        self.assertEqual(prep[prep.index("--seed") + 1], "7")

    def test_preparation_failure_does_not_start_training(self):
        result = self.run_script({"TEST_PREP_FAIL": "1"})
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertTrue(self.prep_capture.is_file())
        self.assertFalse(self.train_capture.exists())


if __name__ == "__main__":
    unittest.main()
