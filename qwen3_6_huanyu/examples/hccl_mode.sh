#!/usr/bin/env bash
# HCCL 通信策略单一事实源（trainer 与 rollout server 共用）。
# 必须在 ray start / 任何 server 进程启动之前 source——worker 进程的 env 来自
# raylet 快照，之后再 export 只活在 driver 里（env_export_huanyu.sh 顶部注释的教训）。
#
# profile 语义（HUANYU_HCCL_PROFILE，默认 aiv）：
#   aiv      生产默认。小 allReduce 绕开 AICPU RPC：1.9ms -> 0.5ms（3-4x，
#            perf_report/analysis_11_10 微基准实测，卡 12-15 / rankSize=4）；
#            大 allGather/reduceScatter 对展开模式不敏感（带宽/偏斜主导），
#            但 AIV 无害，且是 aclgraph 捕获的前置条件（vllm-ascend patch 证实）。
#   host     强制经典 HOST 路径（对照/排障）。
#   host_ts  AICPU unfold、无 AIV（≈ 不设时的默认行为；对照）。
#   unset    完全不设 HCCL_OP_EXPANSION_MODE（复现旧独立脚本形态；对照）。
#   custom   完全尊重调用方已设的 HCCL_* env，本文件只回声不修改（实验逃生舱）。
#
# 取值合法性（CANN 9.1.0 EI0001 报错原文）：Atlas A2 = AIV|HOST|HOST_TS，
# Atlas A3 = AI_CPU|AIV。写 AICPU 会在 HCCL init 期才被拒绝——hccl_policy 探针
# 提前在启动期拦下。
#
# HCCL_DETERMINISTIC 不由本文件合成：它改变归约顺序（row_tp/moe_tp 数值参考咬合
# 它的取值），调用方要设自己设；本文件只负责回声生效值。

huanyu_configure_hccl_mode() {
    local profile="${HUANYU_HCCL_PROFILE:-aiv}"
    case "${profile}" in
        aiv)
            export HUANYU_HCCL_PROFILE=aiv
            export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
            ;;
        host)
            export HUANYU_HCCL_PROFILE=host
            export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-HOST}"
            ;;
        host_ts)
            export HUANYU_HCCL_PROFILE=host_ts
            export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-HOST_TS}"
            ;;
        unset)
            export HUANYU_HCCL_PROFILE=unset
            unset HCCL_OP_EXPANSION_MODE
            ;;
        custom)
            export HUANYU_HCCL_PROFILE=custom
            ;;
        *)
            echo "[huanyu-hccl] FATAL: HUANYU_HCCL_PROFILE must be aiv|host|host_ts|unset|custom, got '${profile}'" >&2
            return 2
            ;;
    esac
    # 生效值回声：「env 设了但 worker 看不见」这类 bug 每次都靠事后翻 runtime_env
    # 才发现；这一行让它变成启动 3 秒就能看到的东西。
    echo "[huanyu-hccl] profile=${HUANYU_HCCL_PROFILE} HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-<unset>} HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-<unset>} HCCL_ALGO=${HCCL_ALGO:-<unset>} HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-<unset>} HCCL_ENABLE_OPEN_AIV=${HCCL_ENABLE_OPEN_AIV:-<unset>}"
}

huanyu_configure_hccl_mode