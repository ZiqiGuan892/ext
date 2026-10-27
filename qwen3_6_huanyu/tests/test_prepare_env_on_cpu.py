"""Exercise standalone environment preparation without package installs or Ray."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
_spec = importlib.util.spec_from_file_location(
    "huanyu_job_fixture_for_prepare", Path(__file__).with_name("test_job_launcher_on_cpu.py")
)
job_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(job_fixture)


@unittest.skipUnless(job_fixture.find_bash(), "Bash is required for environment preparation")
class HuanyuPrepareEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = job_fixture.HuanyuJobLauncherTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        for name in ("prepare_huanyu_env.sh", "prepared_env.py", "graph_mode.sh"):
            shutil.copyfile(EXAMPLES / name, fixture.examples / name)
        self.entry = fixture.examples / "prepare_huanyu_env.sh"
        self.capture = fixture.project / "after_prepare.json"
        self.capture_script = fixture.project / "capture_prepared_environment.py"
        self.capture_script.write_text(
            "import json, os\nfrom pathlib import Path\n"
            "Path(os.environ['HUANYU_TEST_PREPARE_CAPTURE']).write_text(\n"
            "    json.dumps(dict(env=dict(os.environ), cwd=os.getcwd())), encoding='utf-8')\n",
            encoding="utf-8",
        )
        fixture.env.update(
            HUANYU_TEST_PREPARE_ENTRY=job_fixture.shell_path(self.entry),
            HUANYU_TEST_PREPARE_CAPTURE=str(self.capture),
            HUANYU_TEST_PREPARE_CAPTURE_SCRIPT=str(self.capture_script),
            HUANYU_PREPARE_LOG_ROOT=job_fixture.shell_path(fixture.project / "prepare logs"),
            GIT_CONFIG_GLOBAL=str(fixture.project / "test.gitconfig"),
        )
        for key in ("VC_TASK_INDEX", "VC_WORKER_HOSTS", "VC_WORKER_NUM"):
            fixture.env.pop(key, None)

    def run_prepare(self, updates=None, *, sourced=True, prefix=""):
        env = dict(self.fixture.env)
        for key, value in (updates or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        body = prefix + "\n"
        if sourced:
            body += """
set +e +u +x
set -o pipefail
export HUANYU_TEST_OPTIONS_BEFORE="$-"
if source "$HUANYU_TEST_PREPARE_ENTRY"; then
    prepare_status=0
else
    prepare_status=$?
fi
export HUANYU_TEST_OPTIONS_AFTER="$-"
export HUANYU_TEST_PREPARE_STATUS="$prepare_status"
"$HUANYU_TEST_PYTHON" "$HUANYU_TEST_PREPARE_CAPTURE_SCRIPT"
exit "$prepare_status"
"""
        else:
            body += 'exec bash "$HUANYU_TEST_PREPARE_ENTRY"\n'
        return subprocess.run(
            [job_fixture.find_bash(), "--noprofile", "--norc", "-c", body],
            cwd=self.fixture.project.parent,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    def append_site_env(self, text):
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(text)

    def prepared(self):
        return json.loads(self.capture.read_text(encoding="utf-8"))

    def activation_files(self):
        return list((self.fixture.project / "prepare logs").glob("*/activate.sh"))

    def assert_no_ray_or_training(self):
        self.assertFalse(self.fixture.events("ray"))
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(self.fixture.capture_file.exists())
        self.assertFalse(any(
            event["args"][0] in ("wait-head", "wait-cluster", "resolve")
            for event in self.fixture.events("helper")
        ))

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_ray_or_training()

    def test_source_prepares_and_exports_without_node_metadata_or_ray_cli(self):
        self.fixture.bin.joinpath("ray").unlink()
        self.append_site_env("set -ex\nexport HUANYU_TEST_SITE_VALUE='source worked'\n")
        result = self.run_prepare()
        self.assert_success(result)
        actual = self.prepared()
        env = actual["env"]
        self.assertEqual(env["HUANYU_TEST_SITE_VALUE"], "source worked")
        self.assertEqual(env["HUANYU_TEST_OPTIONS_BEFORE"], env["HUANYU_TEST_OPTIONS_AFTER"])
        self.assertEqual(Path(actual["cwd"]).resolve(), self.fixture.project.resolve())
        self.assertEqual(env["HUANYU_SKIP_EDITABLE_INSTALL"], "1")
        self.assertEqual(env["HUANYU_USE_PRECOMPUTE"], "1")
        self.assertEqual(env["HUANYU_TRAIN_AUDIO_ENCODER"], "false")
        self.assertEqual(env["HUANYU_TRAIN_AUDIO_PROJECTOR"], "false")
        self.assertEqual(env["HUANYU_PRECISION_PROFILE"], "full")
        self.assertEqual(env["MINICPMO_STABLE_LINEAR"], "1")
        self.assertEqual(len(self.fixture.events("pip")), 1)
        self.assertEqual(len(self.fixture.events("runtime")), 1)
        operations = [event["args"][0] for event in self.fixture.events("helper")]
        self.assertEqual(operations, ["sync", "check-devices"])
        self.assertEqual(len(self.activation_files()), 1)
        self.assertTrue(self.activation_files()[0].with_name("prepare.log").is_file())

    def test_bash_reports_activation_command_without_starting_ray(self):
        result = self.run_prepare(sourced=False)
        self.assert_success(result)
        self.assertIn("source ", result.stdout)
        self.assertIn("activate.sh", result.stdout)
        self.assertFalse(self.capture.exists())
        self.assertEqual(len(self.activation_files()), 1)

    def test_bash_activation_replays_temporary_managed_environment(self):
        expected = {
            "MODEL_PATH": "temporary-actor", "RM_MODEL_PATH": "temporary-reward",
            "TRAIN_FILE": "temporary-train.parquet", "VAL_FILE": "temporary-validation.parquet",
        }
        exclusions = self.fixture.env.get("MSYS2_ENV_CONV_EXCL", "") + ";PATH"
        result = self.run_prepare({**expected, "MSYS2_ENV_CONV_EXCL": exclusions}, sourced=False)
        self.assert_success(result)
        replay = dict(self.fixture.env, MSYS2_ENV_CONV_EXCL="*",
                      TEST_ACTIVATION=job_fixture.shell_path(self.activation_files()[0]))
        # These assignments belonged only to the `bash prepare...` process;
        # the shell sourcing its activation has no original runtime settings.
        for key in (*expected, "PROJECT_ROOT", "HUANYU_PRECISION_PROFILE", "LD_PRELOAD"):
            replay.pop(key, None)
        result = subprocess.run(
            [job_fixture.find_bash(), "--noprofile", "--norc", "-c",
             'source "$TEST_ACTIVATION" && '
             '"$HUANYU_TEST_PYTHON" "$HUANYU_TEST_PREPARE_CAPTURE_SCRIPT"'],
            cwd=self.fixture.project.parent, env=replay, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        self.assert_success(result)
        actual = self.prepared()
        for key, value in expected.items():
            self.assertEqual(actual["env"][key], value, key)
        self.assertEqual(Path(actual["env"]["PROJECT_ROOT"]).resolve(), self.fixture.project.resolve())
        self.assertEqual(actual["env"]["HUANYU_PRECISION_PROFILE"], "full")
        self.assertEqual(actual["env"]["LD_PRELOAD"], job_fixture.shell_path(self.fixture.jemalloc))
        self.assertEqual(Path(actual["cwd"]).resolve(), self.fixture.project.resolve())

    def test_readonly_parent_variable_reports_activation_failure(self):
        self.append_site_env("export HUANYU_TEST_READONLY=prepared\n")
        result = self.run_prepare(prefix="readonly HUANYU_TEST_READONLY=original")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_ray_or_training()
        self.assertIn("readonly variable", result.stderr)
        self.assertNotIn("environment activated", result.stdout)
        self.assertEqual(len(self.activation_files()), 1)

    def test_source_failure_in_middle_of_site_env_does_not_publish_or_leak(self):
        self.append_site_env(
            "set -ex\nexport HUANYU_TEST_LEAKED=partial\n"
            "bash -c 'exit 37'\nexport HUANYU_TEST_AFTER_FAILURE=unexpected\n"
        )
        result = self.run_prepare()
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assert_no_ray_or_training()
        actual = self.prepared()
        self.assertNotIn("HUANYU_TEST_LEAKED", actual["env"])
        self.assertNotIn("HUANYU_TEST_AFTER_FAILURE", actual["env"])
        self.assertNotIn("HUANYU_TEST_ENV_SOURCED", actual["env"])
        self.assertEqual(actual["env"]["HUANYU_TEST_OPTIONS_BEFORE"], actual["env"]["HUANYU_TEST_OPTIONS_AFTER"])
        self.assertEqual(Path(actual["cwd"]).resolve(), self.fixture.project.parent.resolve())
        self.assertFalse(self.fixture.events("runtime"))
        self.assertFalse(self.activation_files())

    def test_native_check_failure_does_not_publish_activation(self):
        result = self.run_prepare({"HUANYU_TEST_RUNTIME_EXIT": "134"})
        self.assertEqual(result.returncode, 134, result.stdout + result.stderr)
        self.assert_no_ray_or_training()
        self.assertFalse(self.activation_files())
        self.assertFalse(any(event["args"][0] == "check-devices" for event in self.fixture.events("helper")))

    def test_existing_ray_is_not_started_reinstalled_or_waited_on(self):
        result = self.run_prepare({"HUANYU_TEST_RAY_LIVE": "1"})
        self.assert_success(result)
        sync = [event for event in self.fixture.events("helper") if event["args"][0] == "sync"]
        self.assertEqual(len(sync), 1)
        self.assertIn("--ray-running", sync[0]["args"])
        self.assertFalse(self.fixture.events("pip"))
        self.assertFalse(self.fixture.events("git"))

    def test_custom_env_file_preserves_caller_output_paths_and_ray_address(self):
        custom = self.fixture.project / "source env export.sh"
        custom.write_text(
            "export HUANYU_TEST_CUSTOM_ENV=1\n"
            "export DEFAULT_LOCAL_DIR=/obsolete/checkpoints TENSORBOARD_DIR=/obsolete/tensorboard\n"
            "export GENRM_DUMP_DIR=/obsolete/dump RAY_ADDRESS=192.0.2.1:9999\n"
            "export HUANYU_EMBED_CACHE_DIR=/obsolete/cache\n",
            encoding="utf-8",
        )
        expected = {
            "DEFAULT_LOCAL_DIR": "caller-checkpoints",
            "TENSORBOARD_DIR": "caller-tensorboard",
            "GENRM_DUMP_DIR": "caller-dump",
            "RAY_ADDRESS": "192.0.2.99:6925",
        }
        result = self.run_prepare({"HUANYU_ENV_FILE": job_fixture.shell_path(custom), **expected})
        self.assert_success(result)
        env = self.prepared()["env"]
        self.assertEqual(env["HUANYU_TEST_CUSTOM_ENV"], "1")
        self.assertNotIn("HUANYU_TEST_ENV_SOURCED", env)
        for key, value in expected.items():
            self.assertEqual(env[key], value, key)
        self.assertEqual(Path(env["HUANYU_EMBED_CACHE_DIR"]).resolve(), self.fixture.cache.resolve())

    def test_failed_rerun_does_not_reuse_previous_activation(self):
        first = self.run_prepare()
        self.assert_success(first)
        activation = self.activation_files()[0]
        original = activation.read_bytes()
        second = self.run_prepare({"HUANYU_TEST_RUNTIME_EXIT": "134"})
        self.assertEqual(second.returncode, 134, second.stdout + second.stderr)
        self.assert_no_ray_or_training()
        self.assertEqual(self.activation_files(), [activation])
        self.assertEqual(activation.read_bytes(), original)
        self.assertNotIn("HUANYU_TEST_ENV_SOURCED", self.prepared()["env"])
        directories = list((self.fixture.project / "prepare logs").glob("prepare_*"))
        self.assertEqual(len(directories), 2)

    def test_aqa_component_choices_and_paths_survive_stale_site_environment(self):
        self.append_site_env(
            "export MODEL_PATH=/obsolete/actor RM_MODEL_PATH=/obsolete/reward\n"
            "export TRAIN_FILE=/obsolete/train.parquet VAL_FILE=/obsolete/validation.parquet\n"
            "export HUANYU_TRAIN_AUDIO_ENCODER=true HUANYU_TRAIN_AUDIO_PROJECTOR=false\n"
            "export HUANYU_USE_PRECOMPUTE=1 HUANYU_EMBED_CACHE_DIR=/obsolete/cache\n"
            "export HUANYU_PRECISION_PROFILE=custom MINICPMO_STABLE_LINEAR=0\n"
        )
        result = self.run_prepare({
            "HUANYU_JOB_MODE": "aqa_unfreeze",
            "HUANYU_TRAIN_AUDIO_ENCODER": "false",
            "HUANYU_TRAIN_AUDIO_PROJECTOR": "true",
            "MODEL_PATH": "actor-custom",
            "RM_MODEL_PATH": "reward-custom",
            "HUANYU_AQA_TRAIN_FILE": "aqa-train-custom.parquet",
            "HUANYU_AQA_VAL_FILE": "aqa-validation-custom.parquet",
        })
        self.assert_success(result)
        env = self.prepared()["env"]
        for key, expected in {
            "MODEL_PATH": "actor-custom", "RM_MODEL_PATH": "reward-custom",
            "TRAIN_FILE": "aqa-train-custom.parquet", "VAL_FILE": "aqa-validation-custom.parquet",
            "HUANYU_TRAIN_AUDIO_ENCODER": "false", "HUANYU_TRAIN_AUDIO_PROJECTOR": "true",
            "HUANYU_FREEZE_AUDIO": "false", "HUANYU_USE_PRECOMPUTE": "0",
            "HUANYU_EMBED_CACHE_DIR": "", "HUANYU_AQA_AUDIO_TRAIN": "1",
            "HUANYU_PRECISION_PROFILE": "full", "MINICPMO_STABLE_LINEAR": "1",
        }.items():
            self.assertEqual(env[key], expected, key)

    def test_aqa_precision_selection_survives_setup_before_native_import(self):
        for profile, stable in (("full", "1"), ("custom", "0")):
            with self.subTest(profile=profile):
                result = self.run_prepare({
                    "HUANYU_JOB_MODE": "aqa_unfreeze", "HUANYU_PRECISION_PROFILE": profile,
                    "HUANYU_GRAPH_MODE": "0", "MINICPMO_STABLE_LINEAR": "0",
                })
                self.assert_success(result)
                env = self.prepared()["env"]
                self.assertEqual(env["HUANYU_PRECISION_PROFILE"], profile)
                self.assertEqual(env["MINICPMO_STABLE_LINEAR"], stable)
                self.assertEqual(env["HUANYU_USE_PRECOMPUTE"], "0")
                self.assertEqual(env["HUANYU_GRAPH_MODE"], "0")
                self.assertEqual(self.fixture.events("runtime")[-1]["precision"]["HUANYU_PRECISION_PROFILE"], profile)

    def test_cache_failure_does_not_publish_a_ready_environment(self):
        self.fixture.cache.joinpath("manifest.json").unlink()
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cache manifest", result.stdout + result.stderr)
        self.assert_no_ray_or_training()
        self.assertFalse(self.activation_files())


@unittest.skipUnless(job_fixture.find_bash(), "Bash is required to replay generated environment")
class PreparedEnvironmentTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".prepared_env_test_", dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.helper = EXAMPLES / "prepared_env.py"
        self.snapshot = self.root / "before.json"
        self.activation = self.root / "activate.sh"

    def command(self, arguments, env):
        return subprocess.run(
            [sys.executable, "-S", "-B", str(self.helper), *arguments],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_hash_snapshot_and_shell_round_trip_preserve_values_and_unsets(self):
        before = dict(os.environ, PROJECT_ROOT=self.root.as_posix(),
                      UNRELATED_TEST_SECRET="private sentinel must not persist",
                      HUANYU_TEST_REMOVED="remove me")
        result = self.command(["snapshot", str(self.snapshot)], before)
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot_text = self.snapshot.read_text(encoding="utf-8")
        self.assertNotIn(before["UNRELATED_TEST_SECRET"], snapshot_text)
        hashes = json.loads(snapshot_text)
        self.assertEqual(len(hashes["HUANYU_TEST_REMOVED"]), 64)

        value = "space 'quote' \"double\" $(touch must-not-exist) `false` $HOME\nsecond line"
        after = dict(before, HUANYU_TEST_CHANGED=value, HUANYU_TEST_EMPTY="",
                     HUANYU_PREPARE_ONLY="1", HUANYU_PREPARE_SNAPSHOT=str(self.snapshot))
        after.pop("HUANYU_TEST_REMOVED")
        result = self.command([
            "write", "--snapshot", str(self.snapshot), "--output", str(self.activation),
            "--stack-limit", "8192",
        ], after)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.activation.read_text(encoding="utf-8")
        self.assertNotIn(before["UNRELATED_TEST_SECRET"], text)
        self.assertNotIn("export HUANYU_PREPARE_ONLY", text)
        self.assertNotIn("export HUANYU_PREPARE_SNAPSHOT", text)
        self.assertIn("unset HUANYU_TEST_REMOVED", text)
        if os.name != "nt":
            self.assertEqual(self.activation.stat().st_mode & 0o777, 0o600)

        captured = self.root / "captured.json"
        capture_script = self.root / "capture.py"
        capture_script.write_text(
            "import json, os\nfrom pathlib import Path\n"
            "Path(os.environ['HUANYU_TEST_RESULT']).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n",
            encoding="utf-8",
        )
        replay = dict(before, HUANYU_TEST_ACTIVATION=job_fixture.shell_path(self.activation),
                      HUANYU_TEST_PYTHON=sys.executable, HUANYU_TEST_CAPTURE_SCRIPT=str(capture_script),
                      HUANYU_TEST_RESULT=str(captured))
        result = subprocess.run(
            [job_fixture.find_bash(), "--noprofile", "--norc", "-c",
             'source "$HUANYU_TEST_ACTIVATION" && "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_CAPTURE_SCRIPT"'],
            env=replay, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = json.loads(captured.read_text(encoding="utf-8"))
        self.assertEqual(actual["HUANYU_TEST_CHANGED"], value)
        self.assertEqual(actual["HUANYU_TEST_EMPTY"], "")
        self.assertNotIn("HUANYU_TEST_REMOVED", actual)
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_invalid_stack_limit_leaves_no_activation_file(self):
        env = dict(os.environ, PROJECT_ROOT=self.root.as_posix())
        result = self.command(["snapshot", str(self.snapshot)], env)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command([
            "write", "--snapshot", str(self.snapshot), "--output", str(self.activation),
            "--stack-limit", "8192; echo injected",
        ], env)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.activation.exists())


if __name__ == "__main__":
    unittest.main()
