"""Check the actual Bash launcher without importing Ray, models or NPU code."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


RECIPE = Path(__file__).resolve().parents[1] / "aqav"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash is required")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


def environment(replay, plugins):
    env = dict(os.environ, PROJECT_ROOT=shell_path(RECIPE.parent), AVQA_OPTIMIZED="1",
               HUANYU_GRAPH_MODE="1", AVQA_REPLAY=replay, HUANYU_EXPERT_REPLAY="1",
               HUANYU_REPLAY_DEBUG="1", HUANYU_TRAIN_HANG_DIR="",
               MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*")
    if plugins is None:
        env.pop("VLLM_PLUGINS", None)
    else:
        env["VLLM_PLUGINS"] = plugins
    return env


@pytest.mark.parametrize("replay", ["0", "1"])
@pytest.mark.parametrize("plugins", [None, "", "ascend", "ascend,huanyu_r3", "other,ascend"])
def test_recipe_worker_flags_plugin_preservation_and_full_run(replay, plugins):
    result = subprocess.run([
        BASH, shell_path(RECIPE / "run.sh"), "--dry-run", "data.filter_overlong_prompts=false",
        "trainer.total_training_steps=null", "trainer.total_epochs=10",
    ], env=environment(replay, plugins), capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    config = dict(arg.lstrip("+").split("=", 1) for arg in shlex.split(result.stdout)[3:])
    for key in ("actor_rollout_ref.rollout.enable_rollout_routing_replay",
                "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_return_routed_experts"):
        assert config[key] == ("true" if replay == "1" else "false")
    prefix = "ray_kwargs.ray_init.runtime_env.env_vars."
    assert json.loads(config[prefix + "HUANYU_EXPERT_REPLAY"]) == replay
    for key in ("HUANYU_REPLAY_DEBUG", "HUANYU_FINITE_PROBE", "MINICPMO_HANG_PROBE"):
        assert json.loads(config[prefix + key]) == "0"
    assert json.loads(config[prefix + "HUANYU_TRAIN_HANG_DIR"]) == ""
    expected = plugins
    if plugins is not None and replay == "1" and "huanyu_r3" not in plugins.split(","):
        expected = plugins + ",huanyu_r3" if plugins else "huanyu_r3"
    if expected is None:
        assert prefix + "VLLM_PLUGINS" not in config
    else:
        assert json.loads(config[prefix + "VLLM_PLUGINS"]) == expected
    for key, expected in {
        "data.train_batch_size": "128", "actor_rollout_ref.rollout.n": "16",
        "actor_rollout_ref.model.lora_rank": "32", "actor_rollout_ref.rollout.tensor_model_parallel_size": "2",
        "trainer.n_gpus_per_node": "16", "trainer.total_training_steps": "null", "trainer.total_epochs": "10",
        "data.filter_overlong_prompts": "false", "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "20480",
    }.items():
        assert config[key] == expected


@pytest.mark.parametrize("replay,optimized,message", [
    ("1", "0", "AVQA_REPLAY=1 要求 AVQA_OPTIMIZED=1"), ("invalid", "1", "AVQA_REPLAY 必须为 0 或 1"),
])
def test_invalid_modes_fail_before_python(replay, optimized, message):
    env = environment(replay, "ascend")
    env["AVQA_OPTIMIZED"] = optimized
    env["PYTHON_BIN"] = "this-python-must-not-run"
    result = subprocess.run([BASH, shell_path(RECIPE / "run.sh"), "--dry-run"], env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 2 and message in result.stderr


def test_driver_environment_matches_worker_configuration(tmp_path):
    shim = tmp_path / "python-shim"
    shim.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$HUANYU_EXPERT_REPLAY" "$VLLM_PLUGINS"\n',
                    encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    env = environment("1", "ascend")
    env["PYTHON_BIN"] = shell_path(shim)
    result = subprocess.run([BASH, shell_path(RECIPE / "run.sh"), "--config"], env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1", "ascend,huanyu_r3"]
