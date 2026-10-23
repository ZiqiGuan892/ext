"""Exercise Huanyu graph-mode propagation without importing model or Ray code."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BACKEND = EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh"
ENV_PREFIX = "ray_kwargs.ray_init.runtime_env.env_vars."
GRAPH_ENV = (
    "HUANYU_GRAPH_MODE",
    "VLLM_TORCH_COMPILE",
    "TORCH_COMPILE_DISABLE",
    "NPU_GRAPH_CAPTURE_DISABLE",
)


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
class HuanyuGraphModeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="huanyu graph test ")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.examples = self.project / "examples" / "qwen3_6_huanyu"
        self.examples.mkdir(parents=True)
        self.helper = self.examples / "graph_mode.sh"
        shutil.copyfile(EXAMPLES / "graph_mode.sh", self.helper)
        shutil.copyfile(EXAMPLES / "precision_mode.sh", self.examples / "precision_mode.sh")
        shutil.copyfile(EXAMPLES / "run_output.sh", self.examples / "run_output.sh")
        self.capture = self.project / "capture.json"
        self.capture_script = self.project / "capture.py"
        self.capture_script.write_text(
            "import json, os, pathlib, sys\n"
            "keys = " + repr(GRAPH_ENV + ("HUANYU_ENFORCE_EAGER",)) + "\n"
            "pathlib.Path(os.environ['HUANYU_TEST_CAPTURE']).write_text(json.dumps({\n"
            "    'argv': sys.argv[1:], 'env': {key: os.environ.get(key) for key in keys}\n"
            "}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        self.bin = self.project / "bin"
        self.bin.mkdir()
        python_stub = self.bin / "python3"
        python_stub.write_text(
            '#!/usr/bin/env bash\n'
            'exec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_CAPTURE_SCRIPT" "$@"\n',
            encoding="utf-8",
        )
        python_stub.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("HUANYU_", "MINICPMO_", "ROLLOUT_", "PPO_", "DEBUG_")):
                self.env.pop(key)
        self.env.update(
            PROJECT_ROOT=shell_path(self.project),
            HUANYU_TEST_PYTHON=sys.executable,
            HUANYU_TEST_CAPTURE_SCRIPT=str(self.capture_script),
            HUANYU_TEST_CAPTURE=str(self.capture),
            PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            # Start with the opposite of the default, as can happen after
            # sourcing an older environment script or reusing a shell.
            VLLM_TORCH_COMPILE="1",
            TORCH_COMPILE_DISABLE="0",
            NPU_GRAPH_CAPTURE_DISABLE="0",
        )

    def run_shell(self, body, updates=None, arguments=()):
        self.capture.unlink(missing_ok=True)
        script = self.examples / "run test.sh"
        script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
        return subprocess.run(
            [find_bash(), shell_path(script), *arguments],
            env=dict(self.env, **(updates or {})),
            cwd=self.project,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def read_capture(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.capture.is_file(), result.stdout + result.stderr)
        return json.loads(self.capture.read_text(encoding="utf-8"))

    def assert_graph_settings(self, capture, mode, check_argv=True):
        mode = int(mode)
        expected_env = {
            "HUANYU_GRAPH_MODE": str(mode),
            "HUANYU_ENFORCE_EAGER": "false" if mode else "true",
            "VLLM_TORCH_COMPILE": str(mode),
            "TORCH_COMPILE_DISABLE": str(1 - mode),
            "NPU_GRAPH_CAPTURE_DISABLE": str(1 - mode),
        }
        self.assertEqual(capture["env"], expected_env)
        if check_argv:
            actual = dict(
                argument.lstrip("+").split("=", 1)
                for argument in capture["argv"]
                if "=" in argument
            )
            for prefix in ("actor_rollout_ref.rollout", "reward.reward_model.rollout"):
                self.assertEqual(actual[prefix + ".enforce_eager"], expected_env["HUANYU_ENFORCE_EAGER"])
            for key in GRAPH_ENV:
                # Ray runtime_env values must be strings, not Hydra bool/int.
                value = json.loads(actual[ENV_PREFIX + key])
                self.assertIsInstance(value, str, key)
                self.assertEqual(value, expected_env[key], key)
            return actual
        return None

    def test_unset_mode_defaults_to_eager_and_overrides_inherited_graph_flags(self):
        result = self.run_shell(
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            'python3 "${HUANYU_GRAPH_ARGS[@]}"\n'
        )
        capture = self.read_capture(result)
        self.assert_graph_settings(capture, 0)
        self.assertEqual(len(capture["argv"]), 6)

    def test_boolean_and_numeric_modes_override_stale_environment(self):
        for value, mode in (("true", 1), ("1", 1), ("false", 0), ("0", 0)):
            with self.subTest(value=value):
                result = self.run_shell(
                    'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
                    'python3 "${HUANYU_GRAPH_ARGS[@]}"\n',
                    updates={
                        "HUANYU_GRAPH_MODE": value,
                        "VLLM_TORCH_COMPILE": str(1 - mode),
                        "TORCH_COMPILE_DISABLE": str(mode),
                        "NPU_GRAPH_CAPTURE_DISABLE": str(mode),
                    },
                )
                self.assert_graph_settings(self.read_capture(result), mode)

    def test_repeated_sourcing_and_switching_do_not_accumulate_arguments(self):
        for first, last in (("1", "0"), ("0", "1")):
            with self.subTest(first=first, last=last):
                result = self.run_shell(
                    'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
                    f"export HUANYU_GRAPH_MODE={last}\n"
                    "huanyu_configure_graph_mode\n"
                    'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
                    'python3 "${HUANYU_GRAPH_ARGS[@]}"\n',
                    updates={"HUANYU_GRAPH_MODE": first},
                )
                capture = self.read_capture(result)
                self.assert_graph_settings(capture, last)
                self.assertEqual(len(capture["argv"]), 6)

    def test_invalid_mode_fails_before_backend_is_called(self):
        for value in ("auto", "2", "True ", "invalid"):
            with self.subTest(value=value):
                result = self.run_shell(
                    'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh" || exit $?\n'
                    'python3 "${HUANYU_GRAPH_ARGS[@]}"\n',
                    updates={"HUANYU_GRAPH_MODE": value},
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("HUANYU_GRAPH_MODE", result.stderr)
                self.assertFalse(self.capture.exists())

    def test_real_backend_configures_flags_before_its_first_python_import(self):
        text = BACKEND.read_text(encoding="utf-8")
        # Execute the real prefix only as far as the initial pip subprocess.
        # git and python are stubs: no installation, global config, or NPU work.
        match = re.search(r"^python3 -m pip install[^\n]*\n", text, flags=re.MULTILINE)
        self.assertIsNotNone(match, "Cannot locate the backend's first Python subprocess")
        prefix = text[:match.end()]
        for value in ("0", "1", "invalid"):
            with self.subTest(mode=value):
                result = self.run_shell(
                    "git() { return 0; }\n" + prefix,
                    updates={"HUANYU_GRAPH_MODE": value},
                )
                if value == "invalid":
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertFalse(self.capture.exists())
                else:
                    capture = self.read_capture(result)
                    self.assert_graph_settings(capture, value, check_argv=False)
                    self.assertEqual(capture["argv"][:3], ["-m", "pip", "install"])

    def test_real_training_command_overrides_stale_settings_for_all_huanyu_modes(self):
        text = BACKEND.read_text(encoding="utf-8")
        start = re.search(r"^python3 -m verl_omni\.trainer\.main_omni\b", text, flags=re.MULTILINE)
        self.assertIsNotNone(start, "Cannot locate the backend's real training command")
        # The actual multi-line command is retained, including its final argv
        # ordering. Only expensive preflight/setup above it is omitted.
        training_command = text[start.start():]
        setup = (
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh" "$@"\n'
            "_CHAT_TEMPLATE_ARGS=()\n"
            "_DATASET_ARGS=()\n"
            "TRAIN_FILE=train.parquet VAL_FILE=validation.parquet MODEL_PATH=model\n"
            "MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 MAX_TOKEN_LEN_PER_GPU=6272\n"
            "ROLLOUT_NUM_WORKERS=4 PROJECT_NAME=graph_test EXPERIMENT_NAME=graph_test\n"
            "DEFAULT_LOCAL_DIR=checkpoints NUM_NPUS=16 NNODES=1\n"
        )
        for mode in (0, 1):
            for rm_enable in (0, 1):
                for frozen in ("true", "false"):
                    with self.subTest(graph_mode=mode, rm_enable=rm_enable, frozen=frozen):
                        opposite_eager = "true" if mode else "false"
                        arguments = [
                            f"actor_rollout_ref.rollout.enforce_eager={opposite_eager}",
                            f"reward.reward_model.rollout.enforce_eager={opposite_eager}",
                            f"actor_rollout_ref.model.override_config.huanyu_freeze_audio={frozen}",
                            f"reward.reward_model.enable={'true' if rm_enable else 'false'}",
                        ]
                        for key in GRAPH_ENV:
                            arguments.append(f'++{ENV_PREFIX}{key}="stale"')
                        result = self.run_shell(
                            setup + training_command,
                            updates={
                                "HUANYU_GRAPH_MODE": str(mode),
                                "RM_ENABLE": str(rm_enable),
                                "HUANYU_FREEZE_AUDIO": frozen,
                            },
                            arguments=arguments,
                        )
                        capture = self.read_capture(result)
                        actual = self.assert_graph_settings(capture, mode)
                        self.assertEqual(capture["argv"][:2], ["-m", "verl_omni.trainer.main_omni"])
                        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], frozen)
                        self.assertEqual(actual["reward.reward_model.enable"], "true" if rm_enable else "false")
                        # Inference graphs must not turn on training compilation.
                        self.assertEqual(actual["actor_rollout_ref.actor.use_torch_compile"], "False")
                        self.assertEqual(actual["actor_rollout_ref.actor.fsdp_config.use_torch_compile"], "False")


if __name__ == "__main__":
    unittest.main()
