"""Run the standalone launcher with an argv recorder, without Ray or model imports."""

from dataclasses import dataclass
from enum import Enum
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "examples/run_huanyu_train.sh"
DEFAULT_Q36_CKPT_DIR = "/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl"
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash is required")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


def effective(argv):
    return dict(arg.lstrip("+").split("=", 1) for arg in argv if "=" in arg)


@pytest.fixture
def launch():
    with tempfile.TemporaryDirectory(prefix=".config_launcher_", dir=ROOT / "tests") as directory:
        work = Path(directory)
        recorder = work / "record python"
        recorder.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\0" "$@" > "$TEST_ARGV"\n'
            'env -0 > "$TEST_ENV"\n'
            'exit "${TEST_EXIT:-0}"\n',
            encoding="utf-8", newline="\n",
        )
        recorder.chmod(0o755)
        argv_file, env_file = work / "argv", work / "env"
        # Do not let the developer's actual training settings affect this fixture.
        env = {key: value for key, value in os.environ.items() if key.upper() in {
            "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT",
        }}
        env.update({
            "PROJECT_ROOT": shell_path(work), "PYTHON_BIN": shell_path(recorder),
            "HUANYU_RUN_DIR": shell_path(work / "run"),
            "TEST_ARGV": shell_path(argv_file), "TEST_ENV": shell_path(env_file),
            "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*",
        })

        def run(overrides=None, arguments=()):
            result = subprocess.run(
                [BASH, shell_path(ENTRY), *arguments], env=env | (overrides or {}),
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            argv = argv_file.read_bytes().decode().rstrip("\0").split("\0") if argv_file.exists() else []
            child_env = dict(
                item.split("=", 1) for item in env_file.read_bytes().decode().rstrip("\0").split("\0")
            ) if env_file.exists() else {}
            return result, argv, child_env, work

        yield run


def test_default_direct_launch_has_huanyu_routing_and_separate_reward_pool(launch):
    result, argv, child_env, work = launch()
    assert result.returncode == 0, result.stdout + result.stderr
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    config = effective(argv[2:])
    assert len(config) == len(argv[2:]), "Defaults must not hide later duplicate overrides"
    assert config["data.train_files"] == '"/opt/huawei/dataset/infra_data/data/train.parquet"'
    assert config["data.val_files"] == '"/opt/huawei/dataset/infra_data/data/validation.parquet"'
    assert config["actor_rollout_ref.model.path"].endswith('/2748"')
    assert child_env["Q36_CKPT_DIR"] == DEFAULT_Q36_CKPT_DIR
    assert json.loads(config["ray_kwargs.ray_init.runtime_env.env_vars.Q36_CKPT_DIR"]) == child_env["Q36_CKPT_DIR"]
    assert config["actor_rollout_ref.model.architecture"] == "Qwen36HuanyuForTraining"
    assert config["actor_rollout_ref.model.external_lib"].endswith("qwen3_6_huanyu")
    assert config["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name"] == "qwen3_6_huanyu"
    assert config["trainer.nnodes"] == "1"
    assert config["trainer.n_gpus_per_node"] == "16"
    assert config["reward.reward_model.enable_resource_pool"] == "true"
    assert config["reward.reward_model.n_gpus_per_node"] == "16"
    assert config["reward.reward_model.nnodes"] == "1"
    assert config["actor_rollout_ref.actor.fsdp_config.model_dtype"] == "fp32"
    assert config["actor_rollout_ref.rollout.dtype"] == "bfloat16"
    assert config["ray_kwargs.ray_init.address"] == '"auto"'
    assert child_env["HUANYU_USE_PRECOMPUTE"] == "0"
    assert child_env["MINICPMO_USE_PRECOMPUTE"] == "0"
    assert child_env["MINICPMO_TUNE_AUDIO"] == "0"
    assert child_env["HUANYU_FREEZE_AUDIO"] == "true"
    assert (work / "run/console.log").exists()


@pytest.mark.parametrize("checkpoint_dir", [None, '/tmp/vision checkpoint "selected"'])
def test_checkpoint_directory_reaches_driver_and_ray_workers(launch, checkpoint_dir):
    overrides = {"MODEL_PATH": "/tmp/custom actor"}
    if checkpoint_dir is not None:
        overrides["Q36_CKPT_DIR"] = checkpoint_dir
    result, argv, child_env, _ = launch(overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = checkpoint_dir or DEFAULT_Q36_CKPT_DIR
    config = effective(argv)
    assert child_env["Q36_CKPT_DIR"] == expected
    assert json.loads(config["ray_kwargs.ray_init.runtime_env.env_vars.Q36_CKPT_DIR"]) == expected
    assert f"Q36_CKPT_DIR={expected}" in result.stdout


def test_genrm_explicit_registration_replaces_stale_driver_and_ray_environment(launch):
    # A long-lived raylet and an older activation file can carry Actor-only
    # imports. Both registry layers need explicit modules in this new job.
    result, argv, child_env, _ = launch({
        "VERL_USE_EXTERNAL_MODULES": "verl_omni,verl_omni_ext",
        "VLLM_OMNI_EXTERNAL_MODULES": "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    config = effective(argv)
    expected = {
        "VERL_USE_EXTERNAL_MODULES": [
            "verl_omni", "verl_omni_ext", "verl_omni_ext.models.qwen3_6_huanyu",
            "verl_omni_ext.models.qwen3_5_moe",
        ],
        "VLLM_OMNI_EXTERNAL_MODULES": [
            "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks",
            "verl_omni_ext.models.qwen3_5_moe.vllm_omni",
        ],
    }
    for key, modules in expected.items():
        assert child_env[key].split(",") == modules
        assert json.loads(config[f"ray_kwargs.ray_init.runtime_env.env_vars.{key}"]).split(",") == modules
    assert config["reward.reward_model.rollout.engine_kwargs.vllm_omni.output_mode"] == "ar"
    assert config["reward.reward_model.rollout.engine_kwargs.vllm_omni.pipeline_name"] == "qwen3_5_moe"
    assert config["reward.reward_model.rollout.max_num_seqs"] == "64"


def test_environment_paths_profile_and_cli_overrides_reach_final_command(launch):
    profile = '/tmp/forward profile "sample"'
    result, argv, child_env, _ = launch({
        "TRAIN_FILE": "/tmp/train data.parquet", "VAL_FILE": "/tmp/val data.parquet",
        "MAX_PROMPT_LENGTH": "4096", "MAX_RESPONSE_LENGTH": "3072",
        "ACTOR_MODEL_DTYPE": "bfloat16", "TRAIN_BATCH_SIZE": "16",
        "PPO_MINI_BATCH_SIZE": "16", "ROLLOUT_N": "8",
        "TOTAL_TRAINING_STEPS": "2", "HUANYU_FORWARD_PROFILE_DIR": profile,
        "RAY_ADDRESS": "192.0.2.10:6925", "HUANYU_USE_PRECOMPUTE": "1",
        "HUANYU_EMBED_CACHE_DIR": "/tmp/embedding cache",
    }, (
        "actor_rollout_ref.rollout.max_num_seqs=12",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "data.train_max_samples=8",
    ))
    assert result.returncode == 0, result.stdout + result.stderr
    config = effective(argv)
    assert config["data.train_files"] == '"/tmp/train data.parquet"'
    assert config["data.val_files"] == '"/tmp/val data.parquet"'
    assert config["actor_rollout_ref.actor.fsdp_config.model_dtype"] == "bfloat16"
    assert config["trainer.total_training_steps"] == "2"
    assert config["actor_rollout_ref.rollout.max_model_len"] == str(4096 + 3072 + 128)
    assert config["actor_rollout_ref.rollout.max_num_seqs"] == "12"
    assert config["data.train_max_samples"] == "8"
    assert config["ray_kwargs.ray_init.address"] == '"192.0.2.10:6925"'
    for key, value in {
        "HUANYU_FORWARD_PROFILE_DIR": profile, "HUANYU_USE_PRECOMPUTE": "1",
        "HUANYU_EMBED_CACHE_DIR": "/tmp/embedding cache", "MINICPMO_HUANYU_ROLLOUT_TP": "8",
    }.items():
        assert child_env[key] == value
        assert json.loads(config[f"ray_kwargs.ray_init.runtime_env.env_vars.{key}"]) == value


def test_short_profile_recapture_command_reaches_ray_workers(launch):
    result, argv, child_env, _ = launch({
        "NUM_NPUS": "8", "RM_N_GPUS_PER_NODE": "8",
        "TRAIN_FILE": "/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet",
        "VAL_FILE": "/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet",
        "TRAIN_BATCH_SIZE": "4", "PPO_MINI_BATCH_SIZE": "4",
        "MAX_RESPONSE_LENGTH": "128", "ROLLOUT_IGNORE_EOS": "true",
        "TOTAL_TRAINING_STEPS": "2", "SAVE_FREQ": "-1", "TEST_FREQ": "-1",
        "HUANYU_GEN_TIMING": "0", "HUANYU_FORWARD_PROFILE_DIR": "/tmp/profile-recapture",
        "HUANYU_FORWARD_PROFILE_START": "5", "HUANYU_FORWARD_PROFILE_ACTIVE": "5",
    }, ("data.train_max_samples=8", "data.val_max_samples=4"))
    assert result.returncode == 0, result.stdout + result.stderr
    config = effective(argv)
    assert config["data.train_files"].endswith('/aqa_dataset/train.parquet"')
    assert config["data.val_files"].endswith('/aqa_dataset/validation.parquet"')
    # The recapture must fit the reported 16 available devices, including GenRM.
    assert config["trainer.n_gpus_per_node"] == config["reward.reward_model.n_gpus_per_node"] == "8"
    assert config["trainer.nnodes"] == config["reward.reward_model.nnodes"] == "1"
    assert config["reward.reward_model.enable"] == "true"
    assert config["reward.reward_model.enable_resource_pool"] == "true"
    assert config["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "4"
    assert config["reward.reward_model.rollout.tensor_model_parallel_size"] == "4"
    assert config["actor_rollout_ref.rollout.agent.num_workers"] == "2"
    assert config["data.train_batch_size"] == "4"
    assert config["actor_rollout_ref.actor.ppo_mini_batch_size"] == "4"
    assert config["actor_rollout_ref.rollout.n"] == "4"
    assert config["data.max_response_length"] == "128"
    assert config["actor_rollout_ref.rollout.response_length"] == "128"
    assert config["actor_rollout_ref.rollout.ignore_eos"] == "true"
    assert config["trainer.total_training_steps"] == "2"
    assert config["trainer.save_freq"] == config["trainer.test_freq"] == "-1"
    for key, value in {
        "HUANYU_FORWARD_PROFILE_DIR": "/tmp/profile-recapture", "HUANYU_GEN_TIMING": "0",
        "HUANYU_FORWARD_PROFILE_START": "5", "HUANYU_FORWARD_PROFILE_ACTIVE": "5",
    }.items():
        assert child_env[key] == value
        assert json.loads(config[f"ray_kwargs.ray_init.runtime_env.env_vars.{key}"]) == value


def test_rule_mode_can_use_both_nodes_for_actor_and_preserves_prepared_environment(launch):
    result, argv, child_env, _ = launch({
        "HUANYU_REWARD_MODE": "rule_based", "NNODES": "2",
        "TOTAL_TRAINING_STEPS": "-1", "HTTP_PROXY": "http://site-proxy:8080",
        "HTTPS_PROXY": "http://site-proxy:8080", "PYTHONPATH": "/prepared/python/modules",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    config = effective(argv)
    assert config["trainer.nnodes"] == "2"
    assert config["reward.reward_model.enable"] == "false"
    assert config["reward.reward_model.enable_resource_pool"] == "false"
    assert "reward.reward_model.model_path" not in config
    assert config["trainer.total_training_steps"] == "null"
    assert config["actor_rollout_ref.rollout.agent.num_workers"] == "8"
    assert child_env["HTTP_PROXY"] == child_env["HTTPS_PROXY"] == "http://site-proxy:8080"
    assert child_env["PYTHONPATH"].endswith(":/prepared/python/modules")
    assert "/vllm-workspace/vllm-omni" not in child_env["PYTHONPATH"]
    assert child_env["VLLM_OMNI_EXTERNAL_MODULES"] == "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks"
    assert "verl_omni_ext.models.qwen3_5_moe" not in child_env["VERL_USE_EXTERNAL_MODULES"].split(",")


def test_dry_run_does_not_execute_python_or_create_training_output(launch):
    result, argv, _, work = launch(arguments=("--dry-run", "trainer.total_training_steps=2"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not argv
    assert not (work / "run").exists()
    rendered = shlex.split(result.stdout)
    assert rendered[1:3] == ["-m", "verl_omni.trainer.main_omni"]
    assert effective(rendered)["trainer.total_training_steps"] == "2"


def test_full_precision_rejects_graph_before_launch(launch):
    result, argv, _, work = launch({"HUANYU_GRAPH_MODE": "1"})
    assert result.returncode == 2
    assert "full" in result.stderr
    assert not argv
    assert not (work / "run").exists()


def test_custom_graph_mode_propagates_to_both_engines(launch):
    result, argv, child_env, _ = launch({"HUANYU_PRECISION_PROFILE": "custom", "HUANYU_GRAPH_MODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    config = effective(argv)
    assert config["actor_rollout_ref.rollout.enforce_eager"] == "false"
    assert config["reward.reward_model.rollout.enforce_eager"] == "false"
    assert child_env["TORCH_COMPILE_DISABLE"] == "0"
    assert json.loads(config["ray_kwargs.ray_init.runtime_env.env_vars.NPU_GRAPH_CAPTURE_DISABLE"]) == "0"


def test_training_failure_is_not_hidden_by_log_tee(launch):
    result, argv, _, _ = launch({"TEST_EXIT": "7"})
    assert argv[:2] == ["-m", "verl_omni.trainer.main_omni"]
    assert result.returncode == 7


def test_explicit_genrm_imports_register_adapter_and_llm_ar_pipeline(monkeypatch):
    """Exercise the real two registry entry points without importing model weights."""
    import sys

    adapters, pipelines = {}, {}

    class AdapterBase:
        @classmethod
        def register(cls, name):
            def decorate(adapter):
                adapters[name] = adapter
                return adapter
            return decorate

    class StageExecutionType(Enum):
        LLM_AR = "llm_ar"

    @dataclass
    class PipelineConfig:
        model_type: str
        model_arch: str
        stages: tuple

    class StagePipelineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def module(name, path=None, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        if path is not None:
            value.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, value)

    ext_root = ROOT.parents[1]
    # Model loading and NPU patches are outside this registry-only test.
    module("verl", "")
    module("verl.utils", "")
    module("verl.utils.device", is_npu_available=False)
    module("verl_omni", "")
    module("verl_omni.pipelines", "")
    module("verl_omni.pipelines.model_base", OmniRolloutPipelineBase=AdapterBase)
    module("vllm_omni", "")
    module("vllm_omni.config", "")
    module("vllm_omni.config.stage_config", PipelineConfig=PipelineConfig,
           StagePipelineConfig=StagePipelineConfig, StageExecutionType=StageExecutionType)
    module("vllm_omni.config.pipeline_registry",
           register_pipeline=lambda pipeline: pipelines.update({pipeline.model_type: pipeline}))
    module("verl_omni_ext", ext_root)
    module("verl_omni_ext.models", ext_root / "models")
    for name in tuple(sys.modules):
        if name == "verl_omni_ext.models.common" or name.startswith("verl_omni_ext.models.qwen3_5_moe"):
            monkeypatch.delitem(sys.modules, name)
    module("verl_omni_ext.models.qwen3_5_moe.thinker_adapter")
    module("verl_omni_ext.models.qwen3_5_moe.patches")
    before = set(sys.modules)
    try:
        importlib.import_module("verl_omni_ext.models.qwen3_5_moe")
        assert "qwen3_5_moe" in adapters
        # The verl-side adapter import alone does not register the native engine pipeline.
        assert not pipelines
        importlib.import_module("verl_omni_ext.models.qwen3_5_moe.vllm_omni")
        pipeline = pipelines["qwen3_5_moe_thinker_only"]
        assert pipeline.model_arch == "Qwen3_5MoeForConditionalGeneration"
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].execution_type == StageExecutionType.LLM_AR
        assert pipeline.stages[0].model_stage == "thinker"
        assert adapters["qwen3_5_moe"].get_pipeline_id() == pipeline.model_type
    finally:
        for name in set(sys.modules) - before:
            if name == "verl_omni_ext.models.common" or name.startswith("verl_omni_ext.models.qwen3_5_moe"):
                sys.modules.pop(name, None)
