"""
HCCL 通信策略的声明与校验（trainer / rollout server 共用）。

要解决的问题（2026-11-10 实测链条）：
  1. 所有训练路径都 export HCCL_OP_EXPANSION_MODE=AIV，但 profile db 证明大
     allGather/reduceScatter 静默回退 RING-RING-NHR（AICPU RPC 路径）——
     「设了」和「生效」之间没有任何一层校验；
  2. 独立 rollout 脚本（run_rollout_sample_16card.sh 等）根本不带 HCCL env，
     server 侧 TP allReduce 留在 ~1.9ms/次的 AICPU RPC 路径（AIV 模式 0.5ms）；
  3. vllm-ascend 的 batch-invariant 模式会在进程内强写 HCCL_DETERMINISTIC=strict，
     而 row_tp/moe_tp 数值参考正好咬合这一对 env。

本模块是策略的唯一权威定义：
  - PROFILE_TO_EXPANSION：HUANYU_HCCL_PROFILE -> 期望的 HCCL_OP_EXPANSION_MODE；
  - check_hccl_policy()：声明的 profile 与进程实际 env 的一致性检查；
  - assert_hccl_policy()：不一致时 raise（fail-loud；CLAUDE.md「静默失败比崩溃贵」）。

调用点：
  - driver 预检：examples/qwen3_6_huanyu/check_native_runtime.py；
  - trainer worker：thinker_adapter.configure_model（每个 rank 都过）；
  - rollout server：engine_hooks（VLLM_OMNI_EXTERNAL_MODULES 入口）。

CPU 契约：只有进程 env 里**声明了** HUANYU_HCCL_PROFILE 才校验；
未声明 = 纯读取报告，永不 raise（CPU 开发机/单测不受影响）。
"""
import argparse
import json
import os

# HUANYU_HCCL_PROFILE -> 期望的 HCCL_OP_EXPANSION_MODE（None = 必须不存在）。
# 合法取值表来自 CANN 9.1.0 EI0001 报错原文：
#   Atlas A2: AIV | HOST | HOST_TS；Atlas A3: AI_CPU | AIV；Atlas 300I: AI_CPU | HOST。
PROFILE_TO_EXPANSION = {
    # 生产默认：小 allReduce 绕开 AICPU RPC（1.9ms->0.5ms，3-4x）；aclgraph 前置条件。
    "aiv": "AIV",
    # 对照/排障：强制经典 HOST 路径。
    "host": "HOST",
    # 对照：AICPU unfold 但无 AIV（≈ 不设时的默认行为）。
    "host_ts": "HOST_TS",
    # 对照：完全不设展开模式（复现旧独立脚本的默认形态）。
    "unset": None,
    # 逃生舱：完全尊重进程已有 env，只回声不校验（HCCL_ALGO 等实验组合用）。
    "custom": "custom",
}

KNOWN_EXPANSION_VALUES = ("AIV", "HOST", "HOST_TS", "AI_CPU")

# 策略相关的全部 env；自检报告原样带出，供「谁设置的」排查。
HCCL_POLICY_KEYS = (
    "HUANYU_HCCL_PROFILE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_DETERMINISTIC",
    "HCCL_ALGO",
    "HCCL_BUFFSIZE",
    "HCCL_ENABLE_OPEN_AIV",
)


def check_hccl_policy(env=None, profile=None):
    """一致性检查：声明的 profile  vs 进程实际 env。

    Args:
        env: 待检查的 env 映射（默认 os.environ）。
        profile: 显式指定 profile；默认取 env 里的 HUANYU_HCCL_PROFILE。

    Returns:
        dict: {ok, profile, expected, actual, problems[], env{key: value}}
        未声明 profile 时 ok=True 且 problems 为空（纯读取，不评判）。
    """
    env = dict(os.environ if env is None else env)
    profile = profile if profile is not None else env.get("HUANYU_HCCL_PROFILE")
    snapshot = {k: env.get(k) for k in HCCL_POLICY_KEYS}
    report = {
        "ok": True,
        "profile": profile,
        "expected": None,
        "actual": env.get("HCCL_OP_EXPANSION_MODE"),
        "problems": [],
        "env": snapshot,
    }
    if not profile:
        return report  # 未声明：纯读取（CPU 契约）

    if profile not in PROFILE_TO_EXPANSION:
        report["ok"] = False
        report["problems"].append(
            f"unknown HUANYU_HCCL_PROFILE={profile!r}; known: {sorted(PROFILE_TO_EXPANSION)}"
        )
        return report

    actual = report["actual"]
    # 合法性检查与 profile 无关：非法值要到 HCCL init 期才被拒绝（EI0001），提前拦下。
    if actual is not None and actual not in KNOWN_EXPANSION_VALUES:
        report["ok"] = False
        report["problems"].append(
            f"HCCL_OP_EXPANSION_MODE={actual!r} 不在已知合法值 {KNOWN_EXPANSION_VALUES} "
            "（EI0001：A2=AIV|HOST|HOST_TS，A3=AI_CPU|AIV）"
        )

    expected = PROFILE_TO_EXPANSION[profile]
    report["expected"] = expected
    if expected == "custom":
        return report  # 逃生舱：只回声（合法性已在上面查过）

    if expected is None and actual is not None:
        report["ok"] = False
        report["problems"].append(
            f"profile=unset 要求 HCCL_OP_EXPANSION_MODE 不存在，实际为 {actual!r}"
        )
    elif expected is not None and actual != expected:
        report["ok"] = False
        report["problems"].append(
            f"profile={profile} 要求 HCCL_OP_EXPANSION_MODE={expected}，实际为 {actual!r}。"
            "worker 进程的 env 来自 raylet（ray start 时刻的快照），driver 后改不可见——"
            "检查 hccl_mode.sh 是否在 ray start 之前 source、precision_mode.sh 转发清单、"
            "以及是否有库在进程内改写（vllm-ascend batch-invariant 会强写 HCCL_DETERMINISTIC）"
        )
    return report


def assert_hccl_policy(env=None, profile=None, quiet_ok=False):
    """fail-loud 入口：不一致时 raise RuntimeError；一致时打一行生效证据。"""
    report = check_hccl_policy(env=env, profile=profile)
    tag = "[hccl-policy]"
    if not report["ok"]:
        raise RuntimeError(
            f"{tag} HCCL 策略校验失败 pid={os.getpid()}: "
            + "; ".join(report["problems"])
            + f"\n{tag} env={json.dumps(report['env'], ensure_ascii=False)}"
        )
    if not quiet_ok and report["profile"]:
        print(
            f"{tag} OK pid={os.getpid()} profile={report['profile']} "
            f"HCCL_OP_EXPANSION_MODE={report['actual'] or '<unset>'} "
            f"HCCL_DETERMINISTIC={report['env'].get('HCCL_DETERMINISTIC') or '<unset>'}",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="显式指定 profile（默认取 env）")
    parser.add_argument("--strict", action="store_true", help="不一致时退出码 1")
    args = parser.parse_args()
    report = check_hccl_policy(profile=args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()