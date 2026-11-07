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
        "MINICPMO_GDN_CACHE_BARRIER", "MINICPMO_GDN_L2NORM_REFERENCE",
        "MINICPMO_GDN_CONV_REFERENCE", "MINICPMO_GDN_GATING_REFERENCE", "MINICPMO_PRIVATE_HYBRID_CACHE",
    )
    item = dict(kind=kind, args=args, preload=os.environ.get("LD_PRELOAD"),
                precision={key: os.environ.get(key) for key in precision_names}, **extra)
    item["hccl_logic_superpod_id"] = os.environ.get("HCCL_LOGIC_SUPERPOD_ID")
    bashrc = Path(os.environ["HUANYU_TEST_BASHRC"])
    item["bashrc"] = bashrc.read_text(encoding="utf-8") if bashrc.exists() else None
    item["launcher_env"] = {key: os.environ[key] for key in (
        "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
    ) if key in os.environ}
    item["native_env"] = {key: os.environ.get(key) for key in (
        "HUANYU_VLLM_SOURCE", "HUANYU_VLLM_OMNI_ROOT", "PYTHONPATH",
    )}
    item["ray_startup_env"] = {key: os.environ.get(key) for key in (
        "RAY_worker_maximum_startup_concurrency", "RAY_prestart_worker_first_driver",
        "RAY_enable_worker_prestart", "HUANYU_JOB_PREFLIGHT", "HUANYU_RAY_READINESS_CHECK",
    )}
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
        self.home = self.project / "test user"
        self.home.mkdir()
        self.bashrc = self.home / ".bashrc"
        self.examples = self.project / "examples/qwen3_6_huanyu"
        self.examples.mkdir(parents=True)
        self.entry = self.examples / LAUNCHER.name
        shutil.copyfile(LAUNCHER, self.entry)
        shutil.copyfile(EXAMPLES / "precision_mode.sh", self.examples / "precision_mode.sh")
        shutil.copyfile(EXAMPLES / "audio_train_mode.sh", self.examples / "audio_train_mode.sh")
        shutil.copyfile(EXAMPLES / "native_source.sh", self.examples / "native_source.sh")
        self.native_source = self.project / "fake vllm source"
        for relative in (
            "vllm_omni/__init__.py",
            "vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py",
            "vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py",
        ):
            target = self.native_source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# native fixture\n", encoding="utf-8")
        shutil.copyfile(EXAMPLES / "check_native_runtime.py", self.examples / "check_native_runtime.py")
        self.package = self.project / "verl_omni_ext/models/qwen3_6_huanyu"
        self.package.mkdir(parents=True)
        (self.project / "pyproject.toml").write_text("# fixture only\n", encoding="utf-8")
        (self.examples / "env_export_huanyu.sh").write_text(
            '#!/usr/bin/env bash\n'
            'python3 "$HUANYU_TEST_COMMAND_STUB" env-source\n'
            'export HUANYU_TEST_ENV_SOURCED=1\n', encoding="utf-8",
        )
        real_bootstrap = self.project / "real_job_bootstrap.py"
        shutil.copyfile(EXAMPLES / "job_bootstrap.py", real_bootstrap)
        (self.examples / "job_bootstrap.py").write_text(
            RECORDER + """
operation = sys.argv[1]
record("helper", sys.argv[1:])
if operation == "clear-superpod-bashrc":
    import runpy
    runpy.run_path(os.environ["HUANYU_TEST_REAL_BOOTSTRAP"], run_name="__main__")
elif operation == "resolve":
    count = int(sys.argv[sys.argv.index("--expected-nodes") + 1]) if "--expected-nodes" in sys.argv else 2
    print(",".join("192.0.2.%d" % (10 + i) for i in range(count)))
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
    "HUANYU_STEM_TEXT_TRAIN", "HUANYU_CLUSTER_NODES", "RM_NNODES", "ROLLOUT_TP",
    "HUANYU_RAY_READINESS_CHECK", "HUANYU_JOB_PREFLIGHT",
    "STEM_CONTEXT_LENGTH", "STEM_PROMPT_LENGTH", "STEM_DATA_DIR",
    "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "ROLLOUT_N", "ROLLOUT_MAX_NUM_SEQS",
    "ACTOR_OFFLOAD_POLICY", "ROLLOUT_NUM_WORKERS", "HUANYU_CHECKPOINT_VALIDATION_MODE", "HUANYU_FULL_WEIGHT_AUDIT_DIR",
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
        shutil.copyfile(self.examples / "run_huanyu.sh", self.examples / "run_huanyu_stem.sh")
        self.command_stub = self.project / "command_stub.py"
        self.command_stub.write_text(
            RECORDER + """
command = sys.argv[1]
record(command, sys.argv[2:])
if command == "flock" and os.environ.get("HUANYU_TEST_LOCK_BUSY") == "1":
    raise SystemExit(1)
if command == "ray" and "start" in sys.argv[2:]:
    raise SystemExit(int(os.environ.get("HUANYU_TEST_RAY_START_EXIT", "0")))
if command == "pgrep":
    events = [json.loads(line) for line in Path(os.environ["HUANYU_TEST_EVENTS"]).read_text().splitlines()]
    if os.environ.get("HUANYU_TEST_RAY_LIVE") == "once":
        calls = sum(event["kind"] == "pgrep" for event in events)
        raise SystemExit(0 if calls == 1 else 1)
    joined = any(e["kind"] == "ray" and "start" in e["args"] and "--head" not in e["args"] for e in events)
    if joined:
        # Simulate a live detached worker for one supervision cycle, then loss.
        raise SystemExit(1 if any(e["kind"] == "sleep" for e in events) else 0)
    raise SystemExit(0 if os.environ.get("HUANYU_TEST_RAY_LIVE") == "1" else 1)
if command not in ("ray", "git", "pip", "flock", "env-source", "sleep"):
    raise RuntimeError("Unexpected external command: " + command)
""",
            encoding="utf-8",
        )
        self.bin = self.project / "bin"
        self.bin.mkdir()
        for name in ("ray", "git", "pgrep", "flock", "sleep"):
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
                "HUANYU_", "MINICPMO_", "VC_", "RAY_", "RM_", "Q36_", "DEBUG_", "STEM_",
            )) or key in {
                "PYTHON", "PROJECT_ROOT", "EXPERIMENT_NAME", "WANDB_MODE",
                "NNODES", "NUM_NPUS", "REWARD_POOL_NPUS", "TMPDIR", "LD_PRELOAD",
                "MODEL_PATH", "TRAIN_FILE", "VAL_FILE",
                "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
            }:
                self.env.pop(key)
            elif key.startswith("TORCHELASTIC_"):
                self.env.pop(key)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            HOME=shell_path(self.home),
            PROJECT_ROOT=shell_path(self.project),
            VC_TASK_INDEX="0",
            VC_WORKER_NUM="2",
            VC_WORKER_HOSTS="node-zero,node-one",
            HUANYU_TEST_PYTHON=sys.executable,
            HUANYU_TEST_COMMAND_STUB=str(self.command_stub),
            HUANYU_TEST_TRAIN_CAPTURE=str(train_capture),
            HUANYU_TEST_CAPTURE=str(self.capture_file),
            HUANYU_TEST_EVENTS=str(self.events_file),
            HUANYU_TEST_BASHRC=str(self.bashrc),
            HUANYU_TEST_REAL_BOOTSTRAP=str(real_bootstrap),
            HUANYU_EMBED_CACHE_DIR=shell_path(self.cache),
            HUANYU_JEMALLOC_LIBRARY=shell_path(self.jemalloc),
            HUANYU_VLLM_SOURCE=shell_path(self.project / "fake vllm source"),
            TMPDIR=shell_path(self.project),
            RAY_CLUSTER_WAIT_TIMEOUT="2",
            RAY_CLUSTER_POLL_INTERVAL="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUTF8="1",
        )
        if os.name == "nt":
            # Git Bash prepends /mingw64/bin on startup, ahead of the injected
            # Windows PATH. Restore stub priority in every child Bash so git
            # config --global never reaches the user's real Git configuration.
            bash_env = self.project / "fixture_bash_env.sh"
            bash_env.write_text('export PATH="$HUANYU_TEST_BIN:$PATH"\n', encoding="utf-8")
            self.env.update(BASH_ENV=shell_path(bash_env), HUANYU_TEST_BIN=shell_path(self.bin))
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

    def seed_superpod_bashrc(self):
        before = b'export USER_SETTING="keep me"\n'
        after = (
            b'# HCCL_LOGIC_SUPERPOD_ID is documented here\n'
            b'alias huanyu_status="echo ready"\n'
        )
        self.original_bashrc = before + b'export HCCL_LOGIC_SUPERPOD_ID=0\n' + after
        self.cleaned_bashrc = before + b':\n' + after
        self.bashrc.write_bytes(self.original_bashrc)

    def assert_superpod_bashrc_cleaned(self):
        self.assertEqual(self.bashrc.read_bytes(), self.cleaned_bashrc)
        backups = list(self.home.glob(".bashrc.huanyu-superpod-backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), self.original_bashrc)
        # Capture the persistent file at command execution, proving cleanup
        # precedes both new Ray startup and readiness checks for reused Ray.
        commands = self.events("ray") + [
            event for event in self.events("helper") if event["args"][0] == "wait-cluster"
        ]
        for event in commands:
            self.assertEqual(event["bashrc"], self.cleaned_bashrc.decode("utf-8"), event)

    def assert_worker_supervised_until_exit(self, result):
        # The fixture's raylet disappears after one sleep; a node job must
        # stay alive after ray start returns, then surface that loss as failure.
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("worker launcher staying alive", result.stdout)
        self.assertIn("worker raylet exited", result.stdout + result.stderr)
        self.assertEqual([e["args"] for e in self.events("sleep")], [["30"]])

    def test_nonzero_local_ranks_skip_before_lock_and_all_setup(self):
        for rank in (1, 15):
            with self.subTest(rank=rank):
                result = self.run_launcher({
                    "LOCAL_RANK": str(rank), "RANK": str(16 + rank),
                    "LOCAL_WORLD_SIZE": "16", "WORLD_SIZE": "128",
                    "HUANYU_TEST_LOCK_BUSY": "1",
                })
                self.assert_success(result)
                self.assertIn("skipping duplicate node launcher", result.stdout)
                self.assertFalse(self.events_file.exists())

    def test_legacy_launch_local_rank_argument_forms_skip_before_setup(self):
        for args in (("--local-rank=3",), ("--local_rank=3",),
                     ("--local-rank", "3"), ("--local_rank", "3")):
            with self.subTest(args=args):
                result = self.run_launcher(arguments=args)
                self.assert_success(result)
                self.assertFalse(self.events_file.exists())

    def test_local_rank_zero_preserves_hydra_args_and_clears_outer_rank_env(self):
        result = self.run_launcher({
            "LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "128",
            "LOCAL_WORLD_SIZE": "16", "GROUP_RANK": "0", "GROUP_WORLD_SIZE": "8",
            "ROLE_RANK": "0", "ROLE_WORLD_SIZE": "128", "TORCHELASTIC_RUN_ID": "outer-launch",
        }, arguments=("--local-rank", "0", "trainer.total_training_steps=2",
                      "--local_rank=0", "++test.text=value with spaces"))
        self.assert_success(result)
        self.assertEqual(len(self.events("train")), 1)
        argv = json.loads(self.capture_file.read_text())["argv"]
        self.assertIn("trainer.total_training_steps=2", argv)
        self.assertIn("++test.text=value with spaces", argv)
        self.assertFalse(any(x.startswith("--local") or x == "0" for x in argv))
        for kind in ("helper", "pip", "runtime", "ray", "train"):
            for event in self.events(kind):
                self.assertEqual(event["launcher_env"], {}, (kind, event))

    def test_invalid_or_conflicting_local_rank_stops_before_setup(self):
        for updates, args in (
            ({"LOCAL_RANK": "-1"}, ()), ({"LOCAL_RANK": "bad"}, ()),
            ({"LOCAL_RANK": ""}, ()), ({}, ("--local-rank=",)),
            ({}, ("--local_rank",)), ({}, ("--local-rank", "trainer.x=2")),
            ({"LOCAL_RANK": "0"}, ("--local-rank=1",)),
            ({}, ("--local-rank=0", "--local_rank=1")),
        ):
            with self.subTest(updates=updates, args=args):
                result = self.run_launcher(updates, arguments=args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("local rank", result.stderr)
                self.assertFalse(self.events_file.exists())

    def test_independent_job_still_fails_when_node_lock_is_busy(self):
        result = self.run_launcher({"HUANYU_TEST_LOCK_BUSY": "1"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("another Huanyu job launcher", result.stdout + result.stderr)
        for kind in ("pip", "ray", "train"):
            self.assertFalse(self.events(kind))

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

    def test_default_job_skips_extra_probes_and_starts_training(self):
        result = self.run_launcher({
            "HUANYU_TEST_RUNTIME_EXIT": "134", "HUANYU_TEST_FAIL_WAIT": "wait-cluster",
        })
        self.assert_success(result)
        self.assertFalse(self.events("runtime"))
        self.assertFalse(any(e["args"][0] in ("check-devices", "wait-cluster")
                             for e in self.events("helper")))
        self.assertEqual(len(self.events("ray")), 1)
        self.assertEqual(len(self.events("train")), 1)
        self.assertIn("preflight=0", result.stdout)
        self.assertIn("not prechecked", result.stdout)
        self.assertNotIn("nodes registered", result.stdout)

    def test_preflight_and_startup_policy_survive_site_env_on_both_nodes(self):
        with (self.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                "export HUANYU_JOB_PREFLIGHT=1 HUANYU_RAY_READINESS_CHECK=1\n"
                "export RAY_worker_maximum_startup_concurrency=99\n"
                "export RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=1\n"
            )
        for rank in ("0", "1"):
            with self.subTest(rank=rank):
                self.events_file.write_text("", encoding="utf-8")
                result = self.run_launcher({
                    "VC_TASK_INDEX": rank, "HUANYU_JOB_PREFLIGHT": "0",
                    "HUANYU_RAY_READINESS_CHECK": "1", "HUANYU_TEST_RUNTIME_EXIT": "134",
                    "HUANYU_TEST_FAIL_WAIT": "wait-cluster",
                })
                if rank == "0":
                    self.assert_success(result)
                else:
                    self.assert_worker_supervised_until_exit(result)
                    self.assertTrue(any(e["args"][0] == "wait-head" for e in self.events("helper")))
                    self.assertFalse(self.events("train"))
                self.assertFalse(self.events("runtime"))
                self.assertEqual(len(self.events("ray")), 1)
                self.assertEqual(self.events("ray")[0]["ray_startup_env"], {
                    "RAY_worker_maximum_startup_concurrency": "32",
                    "RAY_prestart_worker_first_driver": "1", "RAY_enable_worker_prestart": "0",
                    "HUANYU_JOB_PREFLIGHT": "0", "HUANYU_RAY_READINESS_CHECK": "0",
                })

    def test_default_worker_stops_if_head_never_becomes_available(self):
        result = self.run_launcher({"VC_TASK_INDEX": "1", "HUANYU_TEST_FAIL_WAIT": "wait-head"})
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertFalse(self.events("ray"))
        self.assertFalse(self.events("train"))
        self.assertFalse(self.events("sleep"))

    def test_invalid_preflight_fails_before_setup(self):
        result = self.run_launcher({"HUANYU_JOB_PREFLIGHT": "yes"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("HUANYU_JOB_PREFLIGHT", result.stderr)
        self.assertFalse(self.events_file.exists())

    def test_head_starts_ray_then_waits_before_starting_training_once(self):
        self.seed_superpod_bashrc()
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher()
        self.assert_success(result)
        self.assert_superpod_bashrc_cleaned()
        self.assertIn("native import check exited successfully", result.stdout)
        ray = self.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--head", ray[0]["args"])
        self.assertEqual(ray[0]["args"], ["start", "--head", "--port=6925",
                         "--dashboard-host=0.0.0.0", "--dashboard-port=8960", '--resources={"NPU": 16}'])
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

    def test_worker_uses_manual_start_command_and_keeps_job_alive(self):
        self.seed_superpod_bashrc()
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher({"VC_TASK_INDEX": "1"})
        self.assert_worker_supervised_until_exit(result)
        self.assert_superpod_bashrc_cleaned()
        ray = self.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIn("--address=192.0.2.10:6925", ray[0]["args"])
        self.assertEqual(ray[0]["args"], ["start", "--address=192.0.2.10:6925", '--resources={"NPU": 16}'])
        self.assertNotIn("--head", ray[0]["args"])
        self.assertTrue(any(event["args"][0] == "wait-head" for event in self.events("helper")))
        self.assertFalse(self.events("train"))
        self.assertFalse(self.capture_file.exists())
        for kind in ("runtime", "ray"):
            for event in self.events(kind):
                self.assert_precision_defaults(event)

    def test_ray_start_failure_stops_head_and_worker_without_training_or_keepalive(self):
        for rank in ("0", "1"):
            with self.subTest(rank=rank):
                result = self.run_launcher({"VC_TASK_INDEX": rank, "HUANYU_TEST_RAY_START_EXIT": "23"})
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertFalse(self.events("train"))
                self.assertFalse(self.events("sleep"))
                self.assertNotIn("worker launcher staying alive", result.stdout)
                self.events_file.write_text("", encoding="utf-8")

    def test_stem_head_uses_all_128_cards_and_restores_profile_after_site_env(self):
        env_file = self.examples / "env_export_huanyu.sh"
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write(
                "export NNODES=1 NUM_NPUS=16 RM_ENABLE=1 RM_NNODES=1 ROLLOUT_TP=4\n"
                "export STEM_CONTEXT_LENGTH=5k STEM_PROMPT_LENGTH=1024 STEM_DATA_DIR=/old/data\n"
                "export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 ROLLOUT_MAX_NUM_SEQS=128\n"
                "export ACTOR_OFFLOAD_POLICY=false HUANYU_STEM_TEXT_TRAIN=0\n"
                "export ROLLOUT_NUM_WORKERS=4 HUANYU_CHECKPOINT_VALIDATION_MODE=disk HUANYU_FULL_WEIGHT_AUDIT_DIR=/stale/audit\n"
            )
        result = self.run_launcher({
            "HUANYU_JOB_MODE": "stem", "HUANYU_CLUSTER_NODES": "8",
            "VC_WORKER_NUM": "8", "VC_WORKER_HOSTS": ",".join(f"node-{i}" for i in range(8)),
            "NNODES": "8", "RM_ENABLE": "0", "RM_NNODES": "0", "ROLLOUT_TP": "16",
            "STEM_CONTEXT_LENGTH": "12k", "STEM_PROMPT_LENGTH": "2048",
            "STEM_DATA_DIR": "/stem/data", "TRAIN_FILE": "/stale/aqa.parquet",
            "TRAIN_BATCH_SIZE": "128", "PPO_MINI_BATCH_SIZE": "128", "ROLLOUT_N": "16",
            "ROLLOUT_MAX_NUM_SEQS": "4", "ACTOR_OFFLOAD_POLICY": "true",
            "ROLLOUT_NUM_WORKERS": "128", "HUANYU_CHECKPOINT_VALIDATION_MODE": "", "HUANYU_FULL_WEIGHT_AUDIT_DIR": "",
        })
        self.assert_success(result)
        capture = json.loads(self.capture_file.read_text())
        for name, value in {
            "NNODES": "8", "NUM_NPUS": "16", "RM_ENABLE": "0", "RM_NNODES": "0",
            "REWARD_POOL_NPUS": "0", "HUANYU_STEM_TEXT_TRAIN": "1", "HUANYU_CLUSTER_NODES": "8",
            "HUANYU_USE_PRECOMPUTE": "0", "HUANYU_EMBED_CACHE_DIR": "", "HUANYU_AQA_AUDIO_TRAIN": "0",
            "HUANYU_TRAIN_AUDIO_ENCODER": "false", "HUANYU_TRAIN_AUDIO_PROJECTOR": "false",
            "STEM_CONTEXT_LENGTH": "12k", "STEM_PROMPT_LENGTH": "2048",
            "TRAIN_BATCH_SIZE": "128", "PPO_MINI_BATCH_SIZE": "128", "ROLLOUT_N": "16",
            "ROLLOUT_MAX_NUM_SEQS": "4", "ROLLOUT_TP": "16", "ACTOR_OFFLOAD_POLICY": "true",
            "ROLLOUT_NUM_WORKERS": "128", "HUANYU_CHECKPOINT_VALIDATION_MODE": "", "HUANYU_FULL_WEIGHT_AUDIT_DIR": "",
        }.items():
            self.assertEqual(capture["env"][name], value, name)
        self.assertTrue(capture["env"]["TRAIN_FILE"].endswith("/stem/data/train.parquet"))
        actual = dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"])
        self.assertEqual(json.loads(actual["ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_STEM_TEXT_TRAIN"]), "1")
        self.assertFalse(any(event["args"][0] == "wait-cluster" for event in self.events("helper")))
        self.assertEqual(capture["env"]["HUANYU_RAY_READINESS_CHECK"], "0")
        self.assertIn("expected capacity=128 NPU, not prechecked", result.stdout)
        self.assertEqual(len(self.events("train")), 1)

    def test_stem_last_256_card_worker_joins_without_submitting_training(self):
        result = self.run_launcher({
            "HUANYU_JOB_MODE": "stem", "HUANYU_CLUSTER_NODES": "16", "NNODES": "16",
            "VC_WORKER_NUM": "16", "VC_TASK_INDEX": "15", "RM_ENABLE": "0",
            "VC_WORKER_HOSTS": ",".join(f"node-{i}" for i in range(16)),
        })
        self.assert_worker_supervised_until_exit(result)
        self.assertFalse(self.events("train"))
        ray = self.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertEqual(ray[0]["args"], ["start", "--address=192.0.2.10:6925", '--resources={"NPU": 16}'])

    def test_stem_invalid_topology_rejected_before_setup(self):
        base = {"HUANYU_JOB_MODE": "stem", "HUANYU_CLUSTER_NODES": "8", "NNODES": "8", "RM_ENABLE": "0",
                "VC_WORKER_NUM": "8", "VC_WORKER_HOSTS": ",".join(f"node-{i}" for i in range(8))}
        for extra in ({"NNODES": "7"}, {"RM_ENABLE": "1"}, {"VC_TASK_INDEX": "8"},
                      {"VC_WORKER_NUM": "16"}, {"HUANYU_CLUSTER_NODES": "4"}, {"ROLLOUT_TP": "3"}):
            with self.subTest(extra=extra):
                result = self.run_launcher(dict(base, **extra))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                for kind in ("helper", "pip", "env-source", "ray", "train"):
                    self.assertFalse(self.events(kind), kind)

    def test_site_env_cannot_override_precision_defaults_or_explicit_opt_out(self):
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher({"VC_TASK_INDEX": "1"})
        self.assert_worker_supervised_until_exit(result)
        for kind in ("helper", "pip", "env-source", "runtime", "ray"):
            with self.subTest(kind=kind):
                self.assertTrue(self.events(kind), kind)
                for event in self.events(kind):
                    self.assertEqual(event["preload"], shell_path(self.jemalloc), event)
        self.assertFalse(self.events("train"))

    def test_explicit_allocator_wins_and_preserves_other_preloads_without_duplicates(self):
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher({
            "HUANYU_JEMALLOC_LIBRARY": None,
            "Q36_JEMALLOC_LIBRARY": shell_path(self.jemalloc),
        })
        self.assert_success(result)
        self.assertEqual(self.events("runtime")[0]["preload"], shell_path(self.jemalloc))

    def test_environment_source_cannot_remove_selected_allocator(self):
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher({"HUANYU_TEST_FAIL_WAIT": "wait-cluster"})
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertIn("fixture readiness failure", result.stdout + result.stderr)
        self.assertFalse(self.events("train"))
        self.assertFalse(self.capture_file.exists())
        self.assertFalse(any("stop" in event["args"] for event in self.events("ray")))

    def test_readiness_opt_out_survives_site_env_and_skips_probe(self):
        with (self.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write("export HUANYU_RAY_READINESS_CHECK=1\n")
        result = self.run_launcher({"HUANYU_RAY_READINESS_CHECK": "0",
                                    "HUANYU_TEST_FAIL_WAIT": "wait-cluster"})
        self.assert_success(result)
        self.assertFalse(any(event["args"][0] == "wait-cluster" for event in self.events("helper")))
        self.assertIn("skipping native/device/cluster preflight", result.stdout)
        self.assertNotIn("nodes registered", result.stdout)
        self.assertEqual(len(self.events("train")), 1)

    def test_invalid_readiness_flag_fails_before_setup(self):
        result = self.run_launcher({"HUANYU_RAY_READINESS_CHECK": "yes"})
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("HUANYU_RAY_READINESS_CHECK", result.stderr)
        self.assertFalse(self.events_file.exists())

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

    def test_existing_ray_uses_source_without_copy_reinstall_or_restart(self):
        self.seed_superpod_bashrc()
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
        result = self.run_launcher({"HUANYU_TEST_RAY_LIVE": "1"})
        self.assert_success(result)
        self.assert_superpod_bashrc_cleaned()
        self.assertFalse(any(event["args"][0] == "sync" for event in self.events("helper")))
        self.assertIn("existing workers keep their loaded code", result.stdout)
        self.assertFalse(self.events("pip"))
        self.assertFalse(self.events("git"))
        self.assertFalse(self.events("ray"))
        self.assertEqual(len(self.events("train")), 1)
        self.assertTrue(any(
            event["args"][0] == "wait-cluster" for event in self.events("helper")
        ))
        self.assert_success(self.run_launcher({"HUANYU_TEST_RAY_LIVE": "1"}))
        self.assert_superpod_bashrc_cleaned()

    def test_selected_source_survives_old_site_environment(self):
        with (self.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                'export HUANYU_VLLM_SOURCE=/obsolete/source\n'
                'export HUANYU_VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni\n'
                'export PYTHONPATH=/vllm-workspace/vllm-omni:/preserved/dependency\n'
            )
        before = {str(p.relative_to(self.native_source)): p.read_bytes()
                  for p in self.native_source.rglob("*.py")}
        result = self.run_launcher()
        self.assert_success(result)
        self.assertFalse(any(e["args"][0] == "sync" for e in self.events("helper")))
        for event in self.events("runtime") + self.events("ray") + self.events("train"):
            for key in ("HUANYU_VLLM_SOURCE", "HUANYU_VLLM_OMNI_ROOT"):
                self.assertEqual(Path(event["native_env"][key]).resolve(), self.native_source.resolve())
            self.assertNotIn("/vllm-workspace/vllm-omni", event["native_env"]["PYTHONPATH"])
            self.assertNotIn("fake vllm source", event["native_env"]["PYTHONPATH"])
            self.assertIn("preserved/dependency", event["native_env"]["PYTHONPATH"].replace("\\", "/"))
        self.assertEqual(before, {str(p.relative_to(self.native_source)): p.read_bytes()
                                  for p in self.native_source.rglob("*.py")})

    def test_incomplete_selected_source_fails_before_install_or_ray(self):
        (self.native_source / "vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py").unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source is incomplete", result.stderr + result.stdout)
        for kind in ("pip", "env-source", "runtime", "ray", "train"):
            self.assertFalse(self.events(kind), kind)

    def test_existing_worker_checks_membership_and_never_runs_training(self):
        self.env["HUANYU_JOB_PREFLIGHT"] = "1"
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

    def test_existing_worker_opt_out_does_not_claim_verified_membership(self):
        result = self.run_launcher({
            "VC_TASK_INDEX": "1", "HUANYU_TEST_RAY_LIVE": "once",
            "HUANYU_RAY_READINESS_CHECK": "0", "HUANYU_TEST_FAIL_WAIT": "wait-cluster",
        })
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("cluster membership unchecked", result.stdout)
        self.assertIn("worker raylet exited", result.stdout)
        self.assertFalse(any(event["args"][0] == "wait-cluster" for event in self.events("helper")))
        for kind in ("ray", "pip", "git", "train"):
            self.assertFalse(self.events(kind), kind)


if __name__ == "__main__":
    unittest.main()
