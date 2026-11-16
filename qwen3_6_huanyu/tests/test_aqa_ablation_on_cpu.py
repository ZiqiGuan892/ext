"""Validate launch policies and report denominators without Ray/NPU/model imports."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
spec = importlib.util.spec_from_file_location("aqa_ablation_under_test", ROOT / "aqa_ablation.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def shell_path(path):
    path = Path(path).resolve().as_posix()
    return f"/{path[0].lower()}{path[2:]}" if os.name == "nt" else path


@pytest.mark.parametrize("group,profile,graph,moe,replay,bisect", [
    ("full", "full", "0", "0", "0", "none"),
    ("kernel", "kernel", "0", "0", "0", "none"),
    ("graph", "kernel_graph", "1", "0", "0", "none"),
    ("moe", "kernel_graph", "1", "1", "0", "none"),
    ("replay", "kernel_graph", "1", "1", "1", "none"),
    ("gdn_off", "kernel", "0", "0", "0", "gdn_off"),
    ("rms_off", "kernel", "0", "0", "0", "rms_off"),
])
def test_launch_and_real_backend_policy(tmp_path, group, profile, graph, moe, replay, bisect):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("Bash required")
    root = tmp_path / "project with spaces"
    env = dict(os.environ, PROJECT_ROOT=shell_path(root), MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*",
               MINICPMO_GDN_NPU_KERNEL="1", MINICPMO_MOE_NPU_KERNEL="1", HUANYU_EXPERT_REPLAY="1",
               HUANYU_KERNEL_BISECT="rms_off", HUANYU_FORWARD_PROFILE_DIR="old-profile")
    result = subprocess.run([BASH, shell_path(ROOT / "examples/run_huanyu_aqa_ablation.sh"), group, "--dry-run"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert not root.exists(), "dry-run must not mutate the project"
    declarations = [line for line in result.stdout.splitlines() if line.startswith("declare -x ")]
    exports = dict(shlex.split(line)[2].split("=", 1) for line in declarations)
    assert exports["HUANYU_PRECISION_PROFILE"] == profile
    assert exports["HUANYU_GRAPH_MODE"] == graph
    assert exports["MINICPMO_MOE_NPU_KERNEL"] == moe
    assert exports["HUANYU_EXPERT_REPLAY"] == replay
    assert exports["HUANYU_KERNEL_BISECT"] == bisect
    assert exports["HUANYU_FORWARD_PROFILE_DIR"] == ""
    assert exports["HUANYU_FREEZE_AUDIO"] == "true"
    assert exports["HUANYU_USE_PRECOMPUTE"] == "0"
    assert exports["NUM_NPUS"] == "16"
    assert exports["TRAIN_BATCH_SIZE"] == "16" and exports["ROLLOUT_N"] == "8"
    command = shlex.split(result.stdout.splitlines()[-1])
    overrides = dict(arg.lstrip("+").split("=", 1) for arg in command[2:])
    assert overrides["reward.reward_model.enable"] == "false"
    assert json.loads(overrides["ray_kwargs.ray_init.runtime_env.env_vars.Q36_CKPT_DIR"]) == exports["Q36_CKPT_DIR"]
    assert json.loads(overrides["ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FREEZE_AUDIO"]) == "true"
    # Execute the SAME inlined policies the actual backend uses, catching policy drift.
    source = (ROOT / "examples/run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
    chunks = [source.split(f"# BEGIN HUANYU {name} policy\n")[1].split(f"# END HUANYU {name} policy")[0]
              for name in ("graph", "precision")]
    script = tmp_path / "policies.sh"
    script.write_text("set -euo pipefail\n" + "\n".join(declarations + chunks) +
                      '\nprintf "ARG:%s\\n" "${HUANYU_GRAPH_ARGS[@]}" "${HUANYU_PRECISION_ARGS[@]}"\n',
                      encoding="utf-8", newline="\n")
    resolved = subprocess.run([BASH, shell_path(script)], env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=30)
    assert resolved.returncode == 0, resolved.stdout + resolved.stderr
    final = dict(line[4:].lstrip("+").split("=", 1) for line in resolved.stdout.splitlines() if line.startswith("ARG:"))
    prefix = "ray_kwargs.ray_init.runtime_env.env_vars."
    for key in ("HUANYU_PRECISION_PROFILE", "HUANYU_GRAPH_MODE", "HUANYU_EXPERT_REPLAY"):
        assert json.loads(final[prefix + key]) == exports[key]
    assert final["actor_rollout_ref.rollout.enforce_eager"] == ("false" if graph == "1" else "true")
    assert json.loads(final[prefix + "MINICPMO_GDN_NPU_KERNEL"]) == ("0" if group in ("full", "gdn_off") else "1")
    assert json.loads(final[prefix + "MINICPMO_NPU_RMS_NORM"]) == ("0" if group in ("full", "rms_off") else "1")
    if group != "full":
        assert final["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mamba_ssm_cache_dtype"] == "bfloat16"


def test_plugin_requires_unique_entry_and_respects_allowlist():
    ep = SimpleNamespace(name="huanyu_r3", value=reporter.PLUGIN)
    assert reporter.plugin_problems([ep], "ascend,huanyu_r3", True) == []
    assert reporter.plugin_problems([], None, True)
    assert reporter.plugin_problems([ep, ep], None, True)
    assert reporter.plugin_problems([ep], "ascend", True)
    assert reporter.plugin_problems([], None, False) == []


def write_run(tmp_path):
    (tmp_path / "ablation.json").write_text(json.dumps({"group": "kernel", "env": {
        "TOTAL_TRAINING_STEPS": "10", "NUM_NPUS": "16", "NNODES": "1"}}))
    # Unequal step durations make arithmetic-mean throughput an incorrect answer.
    lines = [f"(TaskRunner pid=1) step:{i} - perf/total_num_tokens:16000 - timing_s/step:{i} - timing_s/gen:1"
             for i in range(1, 11)]
    (tmp_path / "console.log").write_text("\n".join(lines) + "\n" + lines[-1], encoding="utf-8")
    (tmp_path / "exit_code.txt").write_text("0")


def test_report_deduplicates_and_weights_by_time(tmp_path):
    write_run(tmp_path)
    result = reporter.summarize(tmp_path)
    assert result["complete"]
    assert result["measured_steps"] == 8
    assert result["tokens_per_s_per_device"] == pytest.approx(8 * 16000 / sum(range(3, 11)) / 16)
    assert result["graph_replay_observed"].startswith("unknown")
    assert not any(result["evidence"].values())


@pytest.mark.parametrize("failure", ["failed_exit", "no_exit", "missing_step", "nan", "zero_time"])
def test_partial_or_invalid_runs_are_not_ranked(tmp_path, failure):
    write_run(tmp_path)
    log = tmp_path / "console.log"
    if failure == "failed_exit":
        (tmp_path / "exit_code.txt").write_text("1")
    elif failure == "no_exit":
        (tmp_path / "exit_code.txt").unlink()
    elif failure == "missing_step":
        log.write_text(re.sub(r".*step:4 -[^\n]*\n", "", log.read_text()))
    elif failure == "nan":
        log.write_text(log.read_text().replace("timing_s/gen:1", "timing_s/gen:nan"))
    else:
        log.write_text(re.sub(r"timing_s/step:\d+", "timing_s/step:0", log.read_text()))
    result = reporter.summarize(tmp_path)
    assert not result["complete"]
    assert result["tokens_per_s_per_device"] is None
    json.dumps(result, allow_nan=False)
