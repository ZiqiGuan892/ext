"""Preflight and honest throughput summaries for the 16-card AQA ablation."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from importlib import metadata

PLUGIN = "verl_omni_ext.models.qwen3_6_huanyu.expert_replay.plugin:register"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC = re.compile(r"([\w/]+):\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)(?=\s|$)")


def plugin_problems(entries, allowlist, replay):
    if not replay:
        return []
    matches = [ep for ep in entries if ep.name == "huanyu_r3"]
    problems = []
    if len(matches) != 1 or matches[0].value != PLUGIN:
        problems.append("需要唯一 huanyu_r3 -> " + PLUGIN + "；更新声明后用当前 Python 重新 pip install -e 项目根目录")
    if allowlist is not None and "huanyu_r3" not in allowlist.split(","):
        problems.append("VLLM_PLUGINS 白名单缺少 huanyu_r3；添加它并保留已有插件（如 ascend）")
    return problems


def preflight(root, replay):
    problems = []
    for name in ("MODEL_PATH", "Q36_CKPT_DIR", "TRAIN_FILE", "VAL_FILE"):
        value = os.environ.get(name)
        if not value or not Path(value).exists():
            problems.append(f"{name} 不存在: {value!r}")
    probe = root / "verl_omni_ext/probes/hccl_policy.py"
    if not probe.is_file():
        problems.append(f"缺少 HCCL 策略探针: {probe}")
    else:
        spec = importlib.util.spec_from_file_location("_hccl_policy_preflight", probe)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        problems.extend(module.check_hccl_policy()["problems"])
    entries = metadata.entry_points(group="vllm.general_plugins")
    problems.extend(plugin_problems(entries, os.getenv("VLLM_PLUGINS"), replay))
    print("python:", sys.executable)
    print("general_plugins:", [(ep.name, ep.value) for ep in entries])
    if problems:
        raise SystemExit("FATAL:\n" + "\n".join(problems))
    print("[huanyu-ablation] preflight PASS（路径/环境/元数据；尚未验证设备执行）")


def manifest(directory, group):
    keys = ("PROJECT_ROOT", "MODEL_PATH", "Q36_CKPT_DIR", "TRAIN_FILE", "VAL_FILE", "NUM_NPUS", "NNODES",
            "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "ROLLOUT_N", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
            "TOTAL_TRAINING_STEPS", "HUANYU_PRECISION_PROFILE", "HUANYU_GRAPH_MODE", "HUANYU_KERNEL_BISECT",
            "MINICPMO_MOE_NPU_KERNEL", "HUANYU_EXPERT_REPLAY", "HUANYU_WSYNC_PROFILE", "HCCL_OP_EXPANSION_MODE",
            "HUANYU_FREEZE_AUDIO", "HUANYU_USE_PRECOMPUTE", "ACTOR_DYNAMIC_BSZ", "ACTOR_MP_REDUCE_DTYPE")
    data = {"group": group, "python": sys.executable, "env": {key: os.getenv(key) for key in keys}}
    (directory / "ablation.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_steps(text):
    steps = {}
    for line in ANSI.sub("", text).splitlines():
        match = re.search(r"\bstep:\s*(\d+)\s+-", line)
        if match:
            metrics = {key: float(value) for key, value in METRIC.findall(line)}
            if "perf/total_num_tokens" in metrics and "timing_s/step" in metrics:
                steps[int(match[1])] = metrics
    return steps


def summarize(directory, warmup=2):
    config = json.loads((directory / "ablation.json").read_text(encoding="utf-8"))
    env = config["env"]
    target = int(env["TOTAL_TRAINING_STEPS"])
    text = (directory / "console.log").read_text(encoding="utf-8", errors="replace")
    steps = read_steps(text)
    missing = sorted(set(range(1, target + 1)) - steps.keys())
    rows = [steps[i] for i in range(warmup + 1, target + 1) if i in steps]
    exit_file = directory / "exit_code.txt"
    exited_ok = exit_file.is_file() and exit_file.read_text().strip() == "0"
    finite = all(math.isfinite(value) for row in rows for value in row.values())
    valid = exited_ok and not missing and bool(rows) and finite
    totals = [row["perf/total_num_tokens"] for row in rows]
    durations = [row["timing_s/step"] for row in rows]
    valid = valid and all(value > 0 for value in durations) and all(value >= 0 for value in totals)
    averages = {key: statistics.mean(row[key] for row in rows if key in row)
                for key in sorted({key for row in rows for key in row})} if finite else {}
    devices = int(env["NUM_NPUS"]) * int(env["NNODES"])
    # Only a fully finished run is ranked. Partial logs retain their diagnostic averages.
    throughput = sum(totals) / sum(durations) / devices if valid else None
    clean_lines = [line for line in ANSI.sub("", text).splitlines() if not re.match(r"^\++ ", line)]
    clean = "\n".join(clean_lines)
    markers = {
        "gdn_installed": r"\[huanyu-kernel\] actor GDN.*installed.*new=[1-9]",
        "rms_installed": r"\[huanyu-kernel\].*rms_norm.*installed.*new=[1-9]",
        "moe_installed": r"\[huanyu-kernel\] actor fused MoE installed.*new=[1-9]",
        "rollout_native_gdn_verified": r"\[huanyu-kernel\] rollout native GDN verified",
        "graph_mode_resolved": r"\[huanyu-kernel\] graph mode requested; resolved cudagraph_mode=",
        "replay_worker_hook": r"\[huanyu-replay\] plugin applied runner_hook.install_runner_hooks",
        "replay_scheduler_hook": r"\[huanyu-replay\] plugin applied scheduler_hook.install_scheduler_hooks",
        "replay_capture_bound": r"\[huanyu-replay\] capture bound:",
        "replay_routes_stored": r"\[huanyu-replay\] scheduler storing routed experts:",
        "replay_actor_hook": r"\[huanyu-replay\] engine hook installed:",
        "hccl_env_consistent": r"\[hccl-policy\] OK .*profile=aiv HCCL_OP_EXPANSION_MODE=AIV",
    }
    evidence = {key: bool(re.search(pattern, clean)) for key, pattern in markers.items()}
    return {"group": config["group"], "directory": str(directory), "complete": valid,
            "missing_steps": missing, "measured_steps": len(rows), "discarded_warmup_steps": warmup,
            "tokens_per_s_per_device": throughput, "means": averages, "evidence": evidence,
            "graph_replay_observed": "unknown: resolved mode does not prove graph replay",
            "all_ranks_verified": False, "env": env}


def report(root, warmup):
    runs = sorted(root.glob("*/ablation.json"))
    if not runs:
        raise SystemExit("没有 ablation.json：请指定本次 BENCH_ROOT")
    rows = [summarize(path.parent, warmup) for path in runs]
    (root / "report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("group\tcomplete\tsteps\ttok/s/card\tstep_s\tgen_s\told_log_prob_s\tupdate_actor_s\tupdate_weights_s\tpearson")
    for row in rows:
        means = row["means"]
        values = [row["tokens_per_s_per_device"], *(means.get(key) for key in (
            "timing_s/step", "timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor",
            "timing_s/update_weights", "training/rollout_actor_probs_pearson_corr"))]
        print(f"{row['group']}\t{row['complete']}\t{row['measured_steps']}\t" +
              "\t".join("NA" if value is None else f"{value:.6f}" for value in values))
        print("  runtime evidence:", json.dumps(row["evidence"], ensure_ascii=False))
    print("吞吐=sum(total_num_tokens)/sum(step_seconds)/16；默认丢弃前两步。完整性与优化生效证据分别报告。")
    print("图模式 resolved/捕获日志不能单独证明图实际回放；安装日志也不代表所有 rank 的每次调用均已验证。")
    print("Report:", root / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plugins", help="检查整套测试所需的 replay 安装入口")
    check = sub.add_parser("check")
    check.add_argument("--project-root", type=Path, required=True)
    check.add_argument("--replay", type=int, choices=(0, 1), required=True)
    record = sub.add_parser("manifest")
    record.add_argument("directory", type=Path)
    record.add_argument("--group", required=True)
    summary = sub.add_parser("report")
    summary.add_argument("directory", type=Path)
    summary.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if args.command == "plugins":
        problems = plugin_problems(metadata.entry_points(group="vllm.general_plugins"), os.getenv("VLLM_PLUGINS"), True)
        if problems:
            raise SystemExit("FATAL:\n" + "\n".join(problems))
        print("[huanyu-ablation] huanyu_r3 metadata PASS")
    elif args.command == "check":
        preflight(args.project_root, args.replay)
    elif args.command == "manifest":
        manifest(args.directory, args.group)
    else:
        if args.warmup_steps < 0:
            parser.error("--warmup-steps must be non-negative")
        report(args.directory, args.warmup_steps)


if __name__ == "__main__":
    main()
