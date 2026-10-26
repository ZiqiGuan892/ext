"""Run the two-node launcher against command stubs, without Ray or package installs."""

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
LAUNCHER = EXAMPLES / "run_huanyu_32card_job.sh"
DEFAULT_PROJECT_ROOT = "/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new"
DEFAULT_MODEL_PATH = (
    "/opt/huawei/dataset/model_gy/downloaded_models/"
    "Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens"
)
DEFAULT_RM_MODEL_PATH = "/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B"
DEFAULT_TRAIN_FILE = "/opt/huawei/dataset/infra_data/data/train.parquet"
DEFAULT_VAL_FILE = "/opt/huawei/dataset/infra_data/data/validation.parquet"


def find_bash():
    if os.name == "nt":
        path = Path("C:/Program Files/Git/bin/bash.exe")
        return str(path) if path.is_file() else None
    return shutil.which("bash")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


RECORDER = """
import json
import os
from pathlib import Path
import sys

def record(kind, args, **extra):
    precision_names = (
        "HUANYU_PRECISION_PROFILE", "MINICPMO_FSDP2_PIN_MEMORY",
        "MINICPMO_HUANYU_GDN_FP32_TRAINING", "MINICPMO_GDN_FP32_REFERENCE",
        "MINICPMO_HUANYU_LAYERWISE_RELOAD", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE",
        "MINICPMO_ISOLATE_MEDIA_ENCODERS", "MINICPMO_STABLE_LINEAR",
        "MINICPMO_GDN_PREFILL_REFERENCE", "MINICPMO_MOE_TP_REFERENCE",
        "MINICPMO_ROW_TP_REFERENCE", "MINICPMO_HUANYU_ROLLOUT_TP",
        "VLLM_OMNI_EXTERNAL_MODULES", "HUANYU_USE_PRECOMPUTE",
        "HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC",
        "HUANYU_HCCL_OP_EXPANSION_MODE",
        "MINICPMO_GDN_CACHE_BARRIER", "MINICPMO_GDN_L2NORM_REFERENCE",
        "MINICPMO_GDN_CONV_REFERENCE", "MINICPMO_GDN_GATING_REFERENCE", "MINICPMO_PRIVATE_HYBRID_CACHE",
    )
    item = dict(kind=kind, args=args, preload=os.environ.get("LD_PRELOAD"),
                precision={key: os.environ.get(key) for key in precision_names}, **extra)
    with Path(os.environ["HUANYU_TEST_EVENTS"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item) + "\\n")
"""


@unittest.skipUnless(find_bash(), "Bash is required for launcher argv integration")
class HuanyuJobLauncherTests(unittest.TestCase):
    def setUp(self):
        # Keep all fixture files and the launcher's lock/logs inside Huanyu.
        temporary = tempfile.TemporaryDirectory(
            prefix=".job_launcher_test_", dir=ROOT / "tests",
        )
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.examples = self.project / "examples/qwen3_6_huanyu"
        self.examples.mkdir(parents=True)
        self.entry = self.examples / LAUNCHER.name
        shutil.copyfile(LAUNCHER, self.entry)
        shutil.copyfile(EXAMPLES / "precision_mode.sh", self.examples / "precision_mode.sh")
        shutil.copyfile(EXAMPLES / "audio_train_mode.sh", self.examples / "audio_train_mode.sh")
        shutil.copyfile(EXAMPLES / "check_native_runtime.py", self.examples / "check_native_runtime.py")
        self.package = self.project / "verl_omni_ext/models/qwen3_6_huanyu"
        self.package.mkdir(parents=True)
        (self.project / "pyproject.toml").write_text("# fixture only\n", encoding="utf-8")
        (self.examples / "env_export_huanyu.sh").write_text(
            '#!/usr/bin/env bash\n'
            'python3 "$HUANYU_TEST_COMMAND_STUB" env-source\n'
            'export HUANYU_TEST_ENV_SOURCED=1\n', encoding="utf-8",
        )
        (self.examples / "job_bootstrap.py").write_text(
            RECORDER + """
operation = sys.argv[1]
record("helper", sys.argv[1:])
if operation == "resolve":
    print("192.0.2.10,192.0.2.11")
elif operation in ("wait-cluster", "wait-head"):
    if os.environ.get("HUANYU_TEST_FAIL_WAIT") == operation:
        print("fixture readiness failure", file=sys.stderr)
        raise SystemExit(19)
elif operation not in ("sync", "check-devices"):
    raise RuntimeError("Unexpected bootstrap operation: " + operation)
""",
            encoding="utf-8",
        )
        (self.package / "runtime_imports.py").write_text(
            RECORDER + 'record("runtime", sys.argv[1:])\n'
            'raise SystemExit(int(os.environ.get("HUANYU_TEST_RUNTIME_EXIT", "0")))\n',
            encoding="utf-8",
        )
        self.events_file = self.project / "events.jsonl"
        self.capture_file = self.project / "train.json"
        train_capture = self.project / "capture_train.py"
        train_capture.write_text(
            RECORDER + """
record("train", sys.argv[1:])
keys = (
    "HUANYU_USE_PRECOMPUTE", "HUANYU_EMBED_CACHE_DIR", "HUANYU_FREEZE_AUDIO",
    "HUANYU_TRAIN_AUDIO_ENCODER", "HUANYU_TRAIN_AUDIO_PROJECTOR",
    "HUANYU_AQA_AUDIO_TRAIN", "MINICPMO_USE_PRECOMPUTE", "MINICPMO_TUNE_AUDIO",
    "MINICPMO_TUNE_VISION", "WANDB_MODE", "HUANYU_TEST_ENV_SOURCED",
    "NNODES", "NUM_NPUS", "REWARD_POOL_NPUS", "RM_ENABLE", "RAY_ADDRESS", "LD_PRELOAD",
    "PROJECT_ROOT", "MODEL_PATH", "RM_MODEL_PATH", "TRAIN_FILE", "VAL_FILE",
    "HUANYU_HCCL_OP_EXPANSION_MODE", "HCCL_OP_EXPANSION_MODE",
)
Path(os.environ["HUANYU_TEST_CAPTURE"]).write_text(
    json.dumps(dict(argv=sys.argv[1:], env={key: os.environ.get(key) for key in keys})),
    encoding="utf-8",
)
""",
            encoding="utf-8",
        )
        (self.examples / "run_huanyu.sh").write_text(
            '#!/usr/bin/env bash\nexec "$HUANYU_TEST_PYTHON" '
            '"$HUANYU_TEST_TRAIN_CAPTURE" "$@"\n',
            encoding="utf-8",
        )
        self.command_stub = self.project / "command_stub.py"
        self.command_stub.write_text(
            RECORDER + """
command = sys.argv[1]
record(command, sys.argv[2:])
if command == "pgrep":
    if os.environ.get("HUANYU_TEST_RAY_LIVE") == "once":
        events = Path(os.environ["HUANYU_TEST_EVENTS"]).read_text().splitlines()
        calls = sum(json.loads(line)["kind"] == "pgrep" for line in events)
        raise SystemExit(0 if calls == 1 else 1)
    raise SystemExit(0 if os.environ.get("HUANYU_TEST_RAY_LIVE") == "1" else 1)
if command not in ("ray", "git", "pip", "flock", "env-source"):
    raise RuntimeError("Unexpected external command: " + command)
""",
            encoding="utf-8",
        )
        self.bin = self.project / "bin"
        self.bin.mkdir()
        for name in ("ray", "git", "pgrep", "flock"):
            self.write_executable(
                name,
                '#!/usr/bin/env bash\nexec "$HUANYU_TEST_PYTHON" '
                f'"$HUANYU_TEST_COMMAND_STUB" {name} "$@"\n',
            )
        self.write_executable(
            "python3",
            '#!/usr/bin/env bash\n'
            'if [[ "${1:-}" == -m && "${2:-}" == pip ]]; then\n'
            '  shift 2\n'
            '  exec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_COMMAND_STUB" pip "$@"\n'
            'fi\n'
            'exec "$HUANYU_TEST_PYTHON" "$@"\n',
        )
        self.write_executable(
            "timeout",
            '#!/usr/bin/env bash\n'
            '[[ "$1" == --foreground ]] || exit 91\n'
            'shift\n'
            '[[ "$1" == *s ]] || exit 92\n'
            'shift\n'
            'exec "$@"\n',
        )
        self.cache = self.project / "cache with spaces"
        self.cache.mkdir()
        (self.cache / "manifest.json").write_text("{}\n", encoding="utf-8")
        # These tests check environment/argv transport, not jemalloc behavior.
        # MSYS requires a loadable DLL for LD_PRELOAD or fork fails before any
        # command stub runs; use its existing trusted zlib, never an allocator.
        self.jemalloc = self.project / "libjemalloc.so.2"
        self.write_preload_fixture(self.jemalloc)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith((
                "HUANYU_", "MINICPMO_", "VC_", "RAY_", "RM_", "Q36_", "DEBUG_",
            )) or key in {
                "PYTHON", "PROJECT_ROOT", "EXPERIMENT_NAME", "WANDB_MODE",
                "NNODES", "NUM_NPUS", "REWARD_POOL_NPUS", "TMPDIR", "LD_PRELOAD",
                "MODEL_PATH", "TRAIN_FILE", "VAL_FILE",
            }:
                self.env.pop(key)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            PROJECT_ROOT=shell_path(self.project),
            VC_TASK_INDEX="0",
            VC_WORKER_NUM="2",
            VC_WORKER_HOSTS="node-zero,node-one",
            HUANYU_TEST_PYTHON=sys.executable,
            HUANYU_TEST_COMMAND_STUB=str(self.command_stub),
            HUANYU_TEST_TRAIN_CAPTURE=str(train_capture),
            HUANYU_TEST_CAPTURE=str(self.capture_file),
            HUANYU_TEST_EVENTS=str(self.events_file),
            HUANYU_EMBED_CACHE_DIR=shell_path(self.cache),
            HUANYU_JEMALLOC_LIBRARY=shell_path(self.jemalloc),
            HUANYU_VLLM_SOURCE=shell_path(self.project / "fake vllm source"),
            TMPDIR=shell_path(self.project),
            RAY_CLUSTER_WAIT_TIMEOUT="2",
            RAY_CLUSTER_POLL_INTERVAL="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUTF8="1",
        )
        # Preserve the Linux LD_PRELOAD list for assertions and Hydra values;
        # MSYS otherwise translates it for each native Python fixture call.
        exclusions = self.env.get("MSYS2_ENV_CONV_EXCL", "")
        self.env["MSYS2_ENV_CONV_EXCL"] = exclusions + (";" if exclusions else "") + "LD_PRELOAD"
        argument_exclusions = self.env.get("MSYS2_ARG_CONV_EXCL", "")
        self.env["MSYS2_ARG_CONV_EXCL"] = (
            argument_exclusions + (";" if argument_exclusions else "")
            + "++ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD="
            # These are Linux server paths carried inside Hydra overrides,
            # not native Windows Python filenames to translate into Git/opt.
            + ";actor_rollout_ref.model.path=;reward.reward_model.model_path="
            + ";data.train_files=;data.val_files="
        )

    def write_executable(self, name, source):
        path = self.bin / name
        path.write_text(source, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def write_preload_fixture(self, path):
        if os.name == "nt":
            # Distinct Git DLLs avoid MSYS fork address conflicts from loading
            # multiple renamed copies of the same library in one process.
            library = "msys-z.dll"
            if path.name in ("libother.so", "libafterenv.so"):
                library = "msys-bz2-1.dll"
            elif path.name == "libjemalloc-legacy.so.2":
                library = "msys-lzma-5.dll"
            installed = Path(find_bash()).parents[1] / "usr/bin" / library
            if not installed.is_file():
                self.skipTest(f"Git Bash's bundled {library} is required for LD_PRELOAD fixtures")
            shutil.copyfile(installed, path)
        else:
            # glibc ignores non-library preloads with a warning, allowing Linux
            # argv tests without loading a real allocator into the test process.
            path.write_bytes(b"not an ELF shared library\n")

    def run_launcher(self, updates=None, arguments=(), *, trace=False):
        env = dict(self.env)
        for key, value in (updates or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        command = [find_bash(), *(["-x"] if trace else []), shell_path(self.entry), *arguments]
        if os.name == "nt" and "LD_PRELOAD" in env:
            # MSYS cannot preload a Linux-style mixed/duplicate list while
            # forking the path-discovery subshells. Keep it as a shell variable
            # until the real launcher normalizes and exports it. Linux tests
            # retain the inherited environment; all recorded child processes
            # below still receive the launcher's actual exported LD_PRELOAD.
            env["HUANYU_TEST_INITIAL_PRELOAD"] = env.pop("LD_PRELOAD")
            command = [find_bash(), *(["-x"] if trace else []), "-c", 'LD_PRELOAD="$HUANYU_TEST_INITIAL_PRELOAD"; export -n LD_PRELOAD; source "$@"',
                       "huanyu-fixture", shell_path(self.entry), *arguments]
        return subprocess.run(
            command,
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )

    def events(self, kind):
        if not self.events_file.exists():
            return []
        return [
            event for line in self.events_file.read_text(encoding="utf-8").splitlines()
            if (event := json.loads(line))["kind"] == kind
        ]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_precision_defaults(self, event):
        for key in (
            "MINICPMO_HUANYU_GDN_FP32_TRAINING", "MINICPMO_GDN_FP32_REFERENCE",
            "MINICPMO_HUANYU_LAYERWISE_RELOAD", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE",
            "MINICPMO_ISOLATE_MEDIA_ENCODERS",
            "MINICPMO_STABLE_LINEAR", "MINICPMO_GDN_PREFILL_REFERENCE",
            "MINICPMO_GDN_CACHE_BARRIER", "MINICPMO_GDN_L2NORM_REFERENCE",
            "MINICPMO_GDN_CONV_REFERENCE", "MINICPMO_GDN_GATING_REFERENCE", "MINICPMO_PRIVATE_HYBRID_CACHE",
        ):
            self.assertEqual(event["precision"][key], "1", event)
        for key in (
            "MINICPMO_MOE_TP_REFERENCE", "MINICPMO_ROW_TP_REFERENCE", "MINICPMO_FSDP2_PIN_MEMORY",
        ):
            self.assertEqual(event["precision"][key], "0", event)
        self.assertEqual(event["precision"]["HUANYU_PRECISION_PROFILE"], "full", event)
        self.assertEqual(event["precision"]["VLLM_OMNI_EXTERNAL_MODULES"],
                         "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks", event)
        self.assertEqual(event["precision"]["MINICPMO_HUANYU_ROLLOUT_TP"], "4", event)
        self.assertEqual(event["precision"]["HUANYU_USE_PRECOMPUTE"], "1", event)

    def test_head_starts_ray_then_waits_before_starting_training_once(self):
        result = self.run_launcher()
        self.assert_success(result)
        self.assertIn("native import check exited successfully", result.stdout)
        ray = self.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--head", ray[0]["args"])
        self.assertIn("--node-ip-address=192.0.2.10", ray[0]["args"])
        self.assertNotIn("--block", ray[0]["args"])
        waits = [event for event in self.events("helper") if event["args"][0] == "wait-cluster"]
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]["args"][-2:], ["--mode", "all"])
        self.assertEqual(len(self.events("train")), 1)
        events = [json.loads(line) for line in self.events_file.read_text().splitlines()]
        self.assertLess(events.index(ray[0]), events.index(waits[0]))
        self.assertLess(events.index(waits[0]), events.index(self.events("train")[0]))
        self.assertEqual(len(self.events("pip")), 1)
        self.assertEqual(self.events("pip")[0]["args"][:4], [
            "install", "--no-build-isolation", "--no-deps", "-e",
        ])
        self.assertEqual(len(self.events("runtime")), 1)
        for kind in ("runtime", "ray", "train"):
            for event in self.events(kind):
                self.assert_precision_defaults(event)

    def test_worker_joins_head_with_block_and_never_runs_training(self):
        result = self.run_launcher({"VC_TASK_INDEX": "1"})
        self.assert_success(result)
        ray = self.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--address=192.0.2.10:6925", ray[0]["args"])
        self.assertIn("--node-ip-address=192.0.2.11", ray[0]["args"])
        self.assertIn("--block", ray[0]["args"])
        self.assertNotIn("--head", ray[0]["args"])
        self.assertTrue(any(event["args"][0] == "wait-head" for event in self.events("helper")))
        self.assertFalse(self.events("train"))
        self.assertFalse(self.capture_file.exists())
        for kind in ("runtime", "ray"):
            for event in self.events(kind):
                self.assert_precision_defaults(event)

    def test_site_env_cannot_override_precision_defaults_or_explicit_opt_out(self):
        env_file = self.examples / "env_export_huanyu.sh"
        env_file.write_text(
            env_file.read_text(encoding="utf-8")
            + "export MINICPMO_GDN_FP32_REFERENCE=1\n"
            + "export MINICPMO_HUANYU_GDN_FP32_TRAINING=0\n"
            + "export MINICPMO_HUANYU_LAYERWISE_RELOAD=0\n"
            + "export MINICPMO_STABLE_LINEAR=0\n"
            + "export MINICPMO_GDN_PREFILL_REFERENCE=1\n"
            + "export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_whisper.vllm_omni\n",
            encoding="utf-8",
        )
        result = self.run_launcher({
            "HUANYU_PRECISION_PROFILE": "custom",
            "MINICPMO_GDN_FP32_REFERENCE": "0", "MINICPMO_STABLE_LINEAR": "1",
        })
        self.assert_success(result)
        expected = {
            "HUANYU_PRECISION_PROFILE": "custom",
            "MINICPMO_GDN_FP32_REFERENCE": "0",
            "MINICPMO_HUANYU_GDN_FP32_TRAINING": "1",
            "MINICPMO_HUANYU_LAYERWISE_RELOAD": "1",
            "MINICPMO_STABLE_LINEAR": "1",
            "MINICPMO_GDN_PREFILL_REFERENCE": "0",
            "VLLM_OMNI_EXTERNAL_MODULES": "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks",
        }
        actual = dict(arg.lstrip("+").split("=", 1)
                      for arg in json.loads(self.capture_file.read_text())["argv"])
        for key, value in expected.items():
            for kind in ("runtime", "ray", "train"):
                for event in self.events(kind):
                    self.assertEqual(event["precision"][key], value, event)
            self.assertEqual(json.loads(actual[
                "ray_kwargs.ray_init.runtime_env.env_vars." + key
            ]), value)

    def test_reused_ray_gets_actual_tp_cache_and_late_hccl_as_quoted_runtime_strings(self):
        env_file = self.examples / "env_export_huanyu.sh"
        env_file.write_text(
            env_file.read_text(encoding="utf-8")
            + "export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true\n"
            + "export ROLLOUT_TP=4 HUANYU_USE_PRECOMPUTE=0\n",
            encoding="utf-8",
        )
        result = self.run_launcher(
            {"HUANYU_TEST_RAY_LIVE": "1"},
            arguments=("actor_rollout_ref.rollout.tensor_model_parallel_size=8",),
        )
        self.assert_success(result)
        self.assertFalse(self.events("ray"))
        capture = json.loads(self.capture_file.read_text())
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"])
        expected = {
            "MINICPMO_HUANYU_ROLLOUT_TP": "8", "HUANYU_USE_PRECOMPUTE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV", "HCCL_DETERMINISTIC": "true",
            "HUANYU_HCCL_OP_EXPANSION_MODE": "auto",
            "MINICPMO_HUANYU_GDN_FP32_TRAINING": "1",
            "MINICPMO_GDN_FP32_REFERENCE": "1", "MINICPMO_HUANYU_LAYERWISE_RELOAD": "1",
            "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE": "1", "MINICPMO_ISOLATE_MEDIA_ENCODERS": "1",
            "MINICPMO_STABLE_LINEAR": "1", "HUANYU_PRECISION_PROFILE": "full", "MINICPMO_FSDP2_PIN_MEMORY": "0",
            "VLLM_OMNI_EXTERNAL_MODULES": "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks",
        }
        for key, value in expected.items():
            self.assertEqual(json.loads(actual[
                "ray_kwargs.ray_init.runtime_env.env_vars." + key
            ]), value)
            for kind in ("runtime", "train"):
                for event in self.events(kind):
                    self.assertEqual(event["precision"][key], value, event)

    def test_full_profile_restores_complete_policy_after_stale_site_environment(self):
        env_file = self.examples / "env_export_huanyu.sh"
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write(
                "export HUANYU_PRECISION_PROFILE=custom MINICPMO_FSDP2_PIN_MEMORY=1\n"
                "export MINICPMO_GDN_PREFILL_REFERENCE=0 MINICPMO_GDN_CONV_REFERENCE=0\n"
                "export MINICPMO_STABLE_LINEAR=0 MINICPMO_PRIVATE_HYBRID_CACHE=0\n"
            )
        result = self.run_launcher({
            "MINICPMO_GDN_FP32_REFERENCE": "0", "MINICPMO_GDN_GATING_REFERENCE": "0",
            "MINICPMO_FSDP2_PIN_MEMORY": "1",
        })
        self.assert_success(result)
        self.assertIn("overrides inherited values", result.stdout)
        for kind in ("runtime", "ray", "train"):
            for event in self.events(kind):
                self.assert_precision_defaults(event)
        actual = dict(arg.lstrip("+").split("=", 1)
                      for arg in json.loads(self.capture_file.read_text())["argv"])
        for key, value in {"HUANYU_PRECISION_PROFILE": "full", "MINICPMO_FSDP2_PIN_MEMORY": "0",
                           "MINICPMO_STABLE_LINEAR": "1", "MINICPMO_PRIVATE_HYBRID_CACHE": "1"}.items():
            self.assertEqual(json.loads(actual["ray_kwargs.ray_init.runtime_env.env_vars." + key]), value)

    def test_allocator_precedes_first_python_probe_pip_env_source_and_ray(self):
        result = self.run_launcher()
        self.assert_success(result)
        expected = shell_path(self.jemalloc)
        for kind in ("helper", "pip", "env-source", "runtime", "ray", "train"):
            with self.subTest(kind=kind):
                self.assertTrue(self.events(kind), kind)
                for event in self.events(kind):
                    self.assertEqual(event["preload"], expected, event)
        events = [json.loads(line) for line in self.events_file.read_text().splitlines()]
        source = self.events("env-source")[0]
        self.assertLess(events.index(source), events.index(self.events("runtime")[0]))
        self.assertLess(events.index(self.events("runtime")[0]), events.index(self.events("ray")[0]))
        capture = json.loads(self.capture_file.read_text())
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"])
        self.assertEqual(capture["env"]["LD_PRELOAD"], expected)
        self.assertEqual(json.loads(actual[
            "ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD"
        ]), expected)

    def test_allocator_is_inherited_by_worker_probe_and_ray_start(self):
        result = self.run_launcher({"VC_TASK_INDEX": "1"})
        self.assert_success(result)
        for kind in ("helper", "pip", "env-source", "runtime", "ray"):
            with self.subTest(kind=kind):
                self.assertTrue(self.events(kind), kind)
                for event in self.events(kind):
                    self.assertEqual(event["preload"], shell_path(self.jemalloc), event)
        self.assertFalse(self.events("train"))

    def test_explicit_allocator_wins_and_preserves_other_preloads_without_duplicates(self):
        other = self.project / "libother.so"
        self.write_preload_fixture(other)
        legacy = self.project / "libjemalloc-legacy.so.2"
        self.write_preload_fixture(legacy)
        chosen = shell_path(self.jemalloc)
        other_path = shell_path(other)
        result = self.run_launcher({
            "Q36_JEMALLOC_LIBRARY": shell_path(legacy),
            "LD_PRELOAD": f"{shell_path(legacy)}:{other_path} {chosen}:{chosen}",
        })
        self.assert_success(result)
        for kind in ("helper", "env-source", "runtime", "ray", "train"):
            for event in self.events(kind):
                pieces = event["preload"].replace(":", " ").split()
                self.assertEqual(pieces, [chosen, other_path], event)

    def test_q36_allocator_override_remains_supported(self):
        result = self.run_launcher({
            "HUANYU_JEMALLOC_LIBRARY": None,
            "Q36_JEMALLOC_LIBRARY": shell_path(self.jemalloc),
        })
        self.assert_success(result)
        self.assertEqual(self.events("runtime")[0]["preload"], shell_path(self.jemalloc))

    def test_environment_source_cannot_remove_selected_allocator(self):
        other = self.project / "libafterenv.so"
        self.write_preload_fixture(other)
        env_file = self.examples / "env_export_huanyu.sh"
        env_file.write_text(
            env_file.read_text(encoding="utf-8")
            + f"export LD_PRELOAD='{shell_path(other)}'\n",
            encoding="utf-8",
        )
        result = self.run_launcher()
        self.assert_success(result)
        self.assertEqual(self.events("env-source")[0]["preload"], shell_path(self.jemalloc))
        for kind in ("runtime", "ray", "train"):
            for event in self.events(kind):
                pieces = event["preload"].replace(":", " ").split()
                self.assertEqual(pieces, [shell_path(self.jemalloc), shell_path(other)], event)
        capture = json.loads(self.capture_file.read_text())
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"])
        self.assertEqual(json.loads(actual[
            "ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD"
        ]), capture["env"]["LD_PRELOAD"])

    def test_invalid_allocator_override_fails_before_any_python_probe_or_ray(self):
        spaced = self.project / "library with space.so"
        spaced.write_bytes(b"not an ELF shared library\n")
        for candidate in (
            "relative/libjemalloc.so.2",
            shell_path(self.project / "missing-libjemalloc.so.2"),
            shell_path(self.project),
            shell_path(spaced),
            shell_path(self.jemalloc) + ":/second/libjemalloc.so.2",
        ):
            with self.subTest(candidate=candidate):
                result = self.run_launcher({"HUANYU_JEMALLOC_LIBRARY": candidate})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("jemalloc", (result.stdout + result.stderr).lower())
                for kind in ("helper", "pip", "env-source", "runtime", "ray", "train"):
                    self.assertFalse(self.events(kind), kind)

    def test_native_probe_abort_exit_is_not_ignored(self):
        result = self.run_launcher({"HUANYU_TEST_RUNTIME_EXIT": "134"})
        self.assertEqual(result.returncode, 134, result.stdout + result.stderr)
        self.assertEqual(len(self.events("runtime")), 1)
        self.assertFalse(self.events("ray"))
        self.assertFalse(self.events("train"))
        self.assertFalse(any(event["args"][0] == "check-devices" for event in self.events("helper")))
        self.assertNotIn("native import check exited successfully", result.stdout + result.stderr)

    def test_missing_node_index_does_not_silently_start_another_head(self):
        result = self.run_launcher({"VC_TASK_INDEX": None})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("VC_TASK_INDEX", result.stdout + result.stderr)
        for kind in ("ray", "pip", "git", "train"):
            self.assertFalse(self.events(kind), kind)

    def test_failed_cluster_readiness_never_starts_training(self):
        result = self.run_launcher({"HUANYU_TEST_FAIL_WAIT": "wait-cluster"})
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertIn("fixture readiness failure", result.stdout + result.stderr)
        self.assertFalse(self.events("train"))
        self.assertFalse(self.capture_file.exists())
        self.assertFalse(any("stop" in event["args"] for event in self.events("ray")))

    def test_invalid_ports_fail_before_preparing_code_or_ray(self):
        for updates in (
            {"RAY_PORT": "abc"},
            {"RAY_PORT": "70000"},
            {"RAY_PORT": "8960", "RAY_DASHBOARD_PORT": "8960"},
        ):
            with self.subTest(updates=updates):
                result = self.run_launcher(updates)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                for kind in ("helper", "git", "pip", "ray", "train"):
                    self.assertFalse(self.events(kind), kind)

    def test_missing_cache_manifest_prevents_ray_and_training_start(self):
        (self.cache / "manifest.json").unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cache manifest not readable", result.stdout + result.stderr)
        self.assertFalse(self.events("ray"))
        self.assertFalse(self.events("train"))

    def test_filters_scheduler_metadata_and_preserves_hydra_argument_boundaries(self):
        kept = (
            "--cfg=job",
            "--config-name=fixture config",
            "trainer.experiment_name=two words",
            "+test.note=left right",
            "~test.removed",
        )
        result = self.run_launcher(arguments=(
            "--dataset_mount=/not/a/hydra/key",
            *kept,
            "--scheduler_name=with spaces",
        ))
        self.assert_success(result)
        actual = json.loads(self.capture_file.read_text())["argv"]
        self.assertEqual(actual[:len(kept)], list(kept))
        self.assertFalse(any(arg.startswith(("--dataset_mount=", "--scheduler_name=")) for arg in actual))

    def test_cache_and_freeze_settings_reach_training_and_ray_runtime_environment(self):
        result = self.run_launcher(
            {"HUANYU_FREEZE_AUDIO": "false", "MINICPMO_TUNE_AUDIO": "1",
             "HUANYU_TRAIN_AUDIO_ENCODER": "true", "HUANYU_TRAIN_AUDIO_PROJECTOR": "true"},
            arguments=("++actor_rollout_ref.model.override_config.huanyu_freeze_audio=false",
                       "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=true",
                       "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=true"),
        )
        self.assert_success(result)
        capture = json.loads(self.capture_file.read_text())
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"])
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], "true")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_train_audio_encoder"], "false")
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_train_audio_projector"], "false")
        for key, expected in {
            "HUANYU_USE_PRECOMPUTE": "1",
            "HUANYU_FREEZE_AUDIO": "true",
            "HUANYU_TRAIN_AUDIO_ENCODER": "false",
            "HUANYU_TRAIN_AUDIO_PROJECTOR": "false",
            # Git Bash translates this environment path for the native Python stub.
            "HUANYU_EMBED_CACHE_DIR": self.cache.as_posix(),
            "HUANYU_AQA_AUDIO_TRAIN": "0",
            "MINICPMO_USE_PRECOMPUTE": "0",
            "MINICPMO_TUNE_AUDIO": "0",
            "MINICPMO_TUNE_VISION": "0",
            "WANDB_MODE": "offline",
        }.items():
            with self.subTest(variable=key):
                self.assertEqual(capture["env"][key], expected)
                self.assertEqual(json.loads(actual[
                    "ray_kwargs.ray_init.runtime_env.env_vars." + key
                ]), expected)
        for key, expected in {
            "NNODES": "1", "NUM_NPUS": "16", "REWARD_POOL_NPUS": "16",
            "RM_ENABLE": "1", "HUANYU_TEST_ENV_SOURCED": "1",
        }.items():
            self.assertEqual(capture["env"][key], expected)

    def test_existing_ray_verifies_code_without_reinstalling_or_restarting(self):
        result = self.run_launcher({"HUANYU_TEST_RAY_LIVE": "1"})
        self.assert_success(result)
        sync = [event for event in self.events("helper") if event["args"][0] == "sync"]
        self.assertEqual(len(sync), 1)
        self.assertIn("--ray-running", sync[0]["args"])
        self.assertFalse(self.events("pip"))
        self.assertFalse(self.events("git"))
        self.assertFalse(self.events("ray"))
        self.assertEqual(len(self.events("train")), 1)
        self.assertTrue(any(
            event["args"][0] == "wait-cluster" for event in self.events("helper")
        ))

    def test_existing_worker_checks_membership_and_never_runs_training(self):
        result = self.run_launcher({
            "VC_TASK_INDEX": "1", "HUANYU_TEST_RAY_LIVE": "once",
        })
        # The stub reports the existing raylet gone on the subsequent liveness
        # check, so the fixture exits promptly instead of sleeping indefinitely.
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("worker raylet exited", result.stdout + result.stderr)
        waits = [event for event in self.events("helper") if event["args"][0] == "wait-cluster"]
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]["args"][-2:], ["--mode", "local"])
        for kind in ("ray", "pip", "git", "train"):
            self.assertFalse(self.events(kind), kind)


if __name__ == "__main__":
    unittest.main()
