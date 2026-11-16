#!/usr/bin/env bash
# 权重同步（update_weights）优化的单一事实源，trainer 与 rollout server 共用。
# 必须在 ray start / 任何 server 进程启动之前 source——worker 进程 env 来自 raylet
# 快照，之后再 export 只活在 driver（env_export_huanyu.sh 顶部注释的教训）。
#
# 新 knob：HUANYU_WSYNC_PROFILE（默认 off，与已有 graph/kernel/hccl/prefetch 并列）：
#   off     （默认）全部优化关闭，权重同步数据路径与上游逐字节一致。
#   on      打开全部已验证优化（skip-staging / bf16-gather / lookahead / early-ack /
#           moe-fast / direct-load）；仅 HUANYU_WSYNC_LOOKAHEAD_MB 可再调（默认 1024）。
#   custom  逐项读取子开关（HUANYU_WSYNC_SKIP_STAGING / _BF16_GATHER / _EARLY_ACK /
#           _MOE_FAST / _DIRECT_LOAD 布尔 + _LOOKAHEAD_MB 整数），用于 A/B。
#
# 观测开关（与 profile 正交，任何 profile 下都读）：
#   HUANYU_WSYNC_TRACE   off|on|sync  每 rank 每次同步输出 [huanyu-wsync] JSON 阶段计时。
#   HUANYU_WSYNC_VERIFY  bool         首次同步前后对 rollout 全参数/buffer 做指纹比对。
#   HUANYU_WSYNC_BUCKET_MB  整数      覆盖 rollout.checkpoint_engine 的 IPC bucket 大小；同一值
#                                     也透传给 actor 侧桶规划器（单一事实源）。
#
# 桶布局（与 profile 正交，默认 passthrough = 上游 state_dict 顺序，逐字节不变）：
#   HUANYU_WSYNC_LAYOUT        passthrough|layer_grouped|bestfit  发送顺序策略；非 passthrough
#                              必须同时给 HUANYU_WSYNC_BUCKET_MB。
#   HUANYU_WSYNC_LAYOUT_GROUP  parent_module|decoder_layer        「须整组到齐」的分组粒度。
#   HUANYU_WSYNC_LAYOUT_DUMP   目录                               rank0 首次同步落盘完整清单。
#   HUANYU_WSYNC_STRICT        bool                               moe_fast 静默回落改为 raise。
#
# actor 侧子类经 verl EngineRegistry 的 vendor 键选中：当优化或 trace 开启时，
# 本文件导出 VERL_ENGINE_VENDOR=huanyu_wsync，get_engine_cls 即优先命中
# (device, huanyu_wsync)；off 且无观测时不设，注册表回落上游默认键（零影响）。

huanyu_configure_weight_sync() {
    local profile="${HUANYU_WSYNC_PROFILE:-off}"
    case "${profile}" in
        off|on|custom) ;;
        *)
            echo "[huanyu-wsync] FATAL: HUANYU_WSYNC_PROFILE must be off|on|custom, got '${profile}'" >&2
            return 2
            ;;
    esac
    local trace="${HUANYU_WSYNC_TRACE:-off}"
    case "${trace}" in
        off|on|sync|0|false|no) ;;
        1|true|yes) trace=on ;;
        *)
            echo "[huanyu-wsync] FATAL: HUANYU_WSYNC_TRACE must be off|on|sync, got '${trace}'" >&2
            return 2
            ;;
    esac
    local verify="${HUANYU_WSYNC_VERIFY:-0}"

    export HUANYU_WSYNC_PROFILE="${profile}"
    export HUANYU_WSYNC_TRACE="${trace}"
    export HUANYU_WSYNC_VERIFY="${verify}"

    # 需要 actor 引擎子类：优化开启（profile != off）或观测开启（trace != off）。
    local needs_engine=0
    if [[ "${profile}" != "off" || "${trace}" != "off" ]]; then
        needs_engine=1
        export VERL_ENGINE_VENDOR="${VERL_ENGINE_VENDOR:-huanyu_wsync}"
    fi

    # runtime_env 透传：worker 进程只认 raylet 快照，必须显式带上。
    HUANYU_WSYNC_ARGS=(
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_WSYNC_PROFILE=\"${HUANYU_WSYNC_PROFILE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_WSYNC_TRACE=\"${HUANYU_WSYNC_TRACE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_WSYNC_VERIFY=\"${HUANYU_WSYNC_VERIFY}\""
    )
    local _key
    for _key in HUANYU_WSYNC_LOOKAHEAD_MB HUANYU_WSYNC_SKIP_STAGING HUANYU_WSYNC_BF16_GATHER \
                HUANYU_WSYNC_EARLY_ACK HUANYU_WSYNC_MOE_FAST HUANYU_WSYNC_DIRECT_LOAD HUANYU_WSYNC_VERIFY_DIR \
                HUANYU_WSYNC_BUCKET_MB HUANYU_WSYNC_LAYOUT HUANYU_WSYNC_LAYOUT_GROUP \
                HUANYU_WSYNC_LAYOUT_DUMP HUANYU_WSYNC_STRICT; do
        if [[ -n "${!_key:-}" ]]; then
            export "${_key}"
            HUANYU_WSYNC_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=\"${!_key}\"")
        fi
    done
    if [[ "${needs_engine}" == "1" ]]; then
        HUANYU_WSYNC_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.VERL_ENGINE_VENDOR=\"${VERL_ENGINE_VENDOR}\"")
    fi
    if [[ -n "${HUANYU_WSYNC_BUCKET_MB:-}" ]]; then
        HUANYU_WSYNC_ARGS+=("actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${HUANYU_WSYNC_BUCKET_MB}")
    fi

    local layout="${HUANYU_WSYNC_LAYOUT:-passthrough}"
    if [[ "${layout}" != "passthrough" && -z "${HUANYU_WSYNC_BUCKET_MB:-}" ]]; then
        echo "[huanyu-wsync] FATAL: HUANYU_WSYNC_LAYOUT=${layout} requires HUANYU_WSYNC_BUCKET_MB (planner and sender must share the bucket size)" >&2
        return 2
    fi

    echo "[huanyu-wsync] profile=${HUANYU_WSYNC_PROFILE} trace=${HUANYU_WSYNC_TRACE} verify=${HUANYU_WSYNC_VERIFY} vendor=${VERL_ENGINE_VENDOR:-<unset>} lookahead_mb=${HUANYU_WSYNC_LOOKAHEAD_MB:-<default>} bucket_mb=${HUANYU_WSYNC_BUCKET_MB:-<default>} layout=${layout} strict=${HUANYU_WSYNC_STRICT:-0}"
}

huanyu_configure_weight_sync