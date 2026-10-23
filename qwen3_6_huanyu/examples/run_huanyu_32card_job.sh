#!/usr/bin/env bash
# Run on BOTH nodes. VC_TASK_INDEX=0 starts training; 1 only joins Ray.
# Deploy to ${PROJECT_ROOT}/examples/qwen3_6_huanyu/ with the Huanyu package.
# Defaults: frozen Huanyu, 35B sampleall, precomputed q128 cache, actor16 + RM16.
# HUANYU_JOB_MODE=aqa_unfreeze selects online AQA training via its own entry.
# Never stops an existing Ray cluster or deletes an old checkout.
set -euo pipefail

_HUANYU_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_HUANYU_EXAMPLE_DIR}/../.." && pwd)}"
cd "$PROJECT_ROOT"
# The downstream training entry also invokes python3; keep installation and
# probes in that same interpreter environment.
_PYTHON=python3
_HELPER="${_HUANYU_EXAMPLE_DIR}/job_bootstrap.py"
_ENV_FILE="${HUANYU_ENV_FILE:-${_HUANYU_EXAMPLE_DIR}/env_export_huanyu.sh}"
_VLLM_SOURCE="${HUANYU_VLLM_SOURCE:-/opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit}"
_VLLM_TARGET=/vllm-workspace/vllm-omni
_RANK="${VC_TASK_INDEX:-}"
_WAIT="${RAY_CLUSTER_WAIT_TIMEOUT:-900}"
_INTERVAL="${RAY_CLUSTER_POLL_INTERVAL:-5}"
_PORT="${RAY_PORT:-6925}"
_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8960}"
_JOB_MODE="${HUANYU_JOB_MODE:-frozen_cache}"
# Defaults for direct invocation; a profile such as the 16K wrapper supplies
# these first. Snapshot before sourcing the site env so it cannot replace the
# caller's pool sizes. Physical Ray capacity remains 16 NPUs on each node.
declare -A _HUANYU_POOL_CONFIG=(
    [NUM_NPUS]="${NUM_NPUS:-16}" [NNODES]="${NNODES:-1}"
    [ROLLOUT_TP]="${ROLLOUT_TP:-4}" [RM_TP]="${RM_TP:-4}"
    [RM_N_GPUS_PER_NODE]="${RM_N_GPUS_PER_NODE:-16}" [RM_NNODES]="${RM_NNODES:-1}"
    [RM_SPLIT_NODES]="${RM_SPLIT_NODES:-1}"
)
for _key in NUM_NPUS NNODES ROLLOUT_TP RM_TP RM_N_GPUS_PER_NODE RM_NNODES; do
    if [[ ! "${_HUANYU_POOL_CONFIG[$_key]}" =~ ^[1-9][0-9]?$ ]]; then
        echo "FATAL: $_key must be a positive integer (without leading zeroes)" >&2
        exit 2
    fi
done
_ACTOR_PER_NODE=${_HUANYU_POOL_CONFIG[NUM_NPUS]}
_ACTOR_NODES=${_HUANYU_POOL_CONFIG[NNODES]}
_RM_PER_NODE=${_HUANYU_POOL_CONFIG[RM_N_GPUS_PER_NODE]}
_RM_NODES=${_HUANYU_POOL_CONFIG[RM_NNODES]}
if (( _ACTOR_PER_NODE > 16 || _RM_PER_NODE > 16 || _ACTOR_NODES > 2 || _RM_NODES > 2 ||
      _ACTOR_PER_NODE * _ACTOR_NODES + _RM_PER_NODE * _RM_NODES != 32 ||
      _ACTOR_PER_NODE % ${_HUANYU_POOL_CONFIG[ROLLOUT_TP]} != 0 ||
      _RM_PER_NODE % ${_HUANYU_POOL_CONFIG[RM_TP]} != 0 )); then
    echo 'FATAL: pool sizes must total 32 on two 16-card nodes; each per-node pool must be divisible by its TP' >&2
    exit 2
fi
# Each per-node pool is STRICT_PACK. Check it can be packed into two physical
# nodes; checking only the 32-card total would accept impossible [12,12]+[8].
_POOLS_FIT=0
for (( _a=0; _a<=_ACTOR_NODES; _a++ )); do
    for (( _r=0; _r<=_RM_NODES; _r++ )); do
        if (( _a * _ACTOR_PER_NODE + _r * _RM_PER_NODE == 16 )); then
            _POOLS_FIT=1
        fi
    done
done
if (( ! _POOLS_FIT )); then
    echo 'FATAL: per-node pools cannot fit on two 16-card nodes; for actor24 + RM8 use actor12x2 and RM4x2' >&2
    exit 2
fi
if [[ "$_JOB_MODE" == aqa_unfreeze ]] &&
    (( _ACTOR_PER_NODE != 16 || _ACTOR_NODES != 1 || _RM_PER_NODE != 16 || _RM_NODES != 1 )); then
    echo 'FATAL: the online AQA entry still requires actor16 + RM16; configurable pools apply to the frozen entry' >&2
    exit 2
fi
case "$_JOB_MODE" in
    frozen_cache) _TRAIN_ENTRY=run_huanyu.sh ;;
    aqa_unfreeze) _TRAIN_ENTRY=run_huanyu_aqa_audio_train.sh ;;
    *) echo 'FATAL: HUANYU_JOB_MODE must be frozen_cache or aqa_unfreeze' >&2; exit 2 ;;
esac
[[ -f "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" ]] || {
    echo "FATAL: missing training entry $PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" >&2
    exit 1
}
for _command in "$_PYTHON" ray git pgrep flock timeout; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing command $_command" >&2; exit 1; }
done
if [[ "$_RANK" != 0 && "$_RANK" != 1 ]]; then
    echo 'FATAL: set VC_TASK_INDEX=0 on head and VC_TASK_INDEX=1 on worker; no implicit rank 0' >&2
    exit 2
fi
if [[ -n "${VC_WORKER_NUM:-}" && "$VC_WORKER_NUM" != 2 ]]; then
    echo 'FATAL: this launcher expects two nodes, each with 16 NPUs' >&2
    exit 2
fi
for _key in _WAIT _INTERVAL _PORT _DASHBOARD_PORT; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: $_key must be a positive integer" >&2
        exit 2
    fi
done
if (( _PORT > 65535 || _DASHBOARD_PORT > 65535 || _PORT == _DASHBOARD_PORT )); then
    echo 'FATAL: Ray and dashboard ports must be distinct values in 1..65535' >&2
    exit 2
fi

# Platform jobs are non-interactive and may never read the image's .bashrc
# jemalloc export. Set it before Python/pip and the Ascend setup scripts (which
# also import torch), not just before Ray. This restores the allocator setup in
# the known working job entry; a native import must still exit successfully.
_HUANYU_JOB_JEMALLOC="${HUANYU_JEMALLOC_LIBRARY:-${Q36_JEMALLOC_LIBRARY:-}}"
_HUANYU_PRELOAD_LIBS=()
_HUANYU_ORIGINAL_PRELOAD="${LD_PRELOAD:-}"
read -r -a _HUANYU_PRELOAD_LIBS <<< "${_HUANYU_ORIGINAL_PRELOAD//:/ }"
if [[ -z "$_HUANYU_JOB_JEMALLOC" ]]; then
    for _library in "${_HUANYU_PRELOAD_LIBS[@]}"; do
        if [[ "$_library" == /* && "${_library##*/}" == libjemalloc*.so* && -f "$_library" && -r "$_library" ]]; then
            _HUANYU_JOB_JEMALLOC="$_library"
            break
        fi
    done
fi
if [[ -z "$_HUANYU_JOB_JEMALLOC" ]]; then
    for _library in /usr/lib/"$(uname -m)"-linux-gnu/libjemalloc.so.2 \
        /usr/lib64/libjemalloc.so.2 /usr/lib/libjemalloc.so.2 \
        /lib/"$(uname -m)"-linux-gnu/libjemalloc.so.2; do
        if [[ -f "$_library" && -r "$_library" ]]; then
            _HUANYU_JOB_JEMALLOC="$_library"
            break
        fi
    done
fi
if [[ "$_HUANYU_JOB_JEMALLOC" != /* || ! -f "$_HUANYU_JOB_JEMALLOC" || ! -r "$_HUANYU_JOB_JEMALLOC" \
    || "$_HUANYU_JOB_JEMALLOC" == *[[:space:]:]* ]]; then
    echo 'FATAL: set HUANYU_JEMALLOC_LIBRARY to a readable absolute libjemalloc.so path on BOTH nodes (no spaces/colons)' >&2
    exit 1
fi
huanyu_preload_allocator() {
    local library
    local -a libraries=()
    local current_preload="${LD_PRELOAD:-}"
    read -r -a libraries <<< "${current_preload//:/ }"
    # Keep unrelated preloads, but do not stack multiple jemalloc versions.
    export LD_PRELOAD="$_HUANYU_JOB_JEMALLOC"
    for library in "${libraries[@]}"; do
        if [[ "${library##*/}" != libjemalloc*.so* && "$library" != "$_HUANYU_JOB_JEMALLOC" ]]; then
            export LD_PRELOAD="$LD_PRELOAD:$library"
        fi
    done
}
huanyu_preload_allocator
[[ -f "$_HELPER" && -f "$_ENV_FILE" ]] || { echo "FATAL: missing Huanyu package or env file" >&2; exit 1; }
echo "[huanyu-job] allocator before Python/env setup: LD_PRELOAD=$LD_PRELOAD"
_IPS="$("$_PYTHON" -B "$_HELPER" resolve "${VC_WORKER_HOSTS:-}")"
IFS=, read -r -a _NODE_IPS <<< "$_IPS"
_MASTER_IP="${_NODE_IPS[0]}"
_NODE_IP="${_NODE_IPS[$_RANK]}"

# Filter only scheduler --NAME=value metadata; preserve Hydra argv boundaries.
_TRAIN_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --cfg=*|--package=*|--config-path=*|--config-name=*|--config-dir=*|--experimental-rerun=*|--info=*)
            _TRAIN_ARGS+=("$_arg") ;;
        --*=*) echo "[huanyu-job] ignoring scheduler argument ${_arg%%=*}" ;;
        *) _TRAIN_ARGS+=("$_arg") ;;
    esac
done

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_sampleall_cache_32card}"
_LOG_DIR="${HUANYU_JOB_LOG_DIR:-$PROJECT_ROOT/runs/$EXPERIMENT_NAME/cluster}"
mkdir -p "$_LOG_DIR"
exec > >(tee -a "$_LOG_DIR/rank_$_RANK.log") 2>&1
# Local lock stays held by this launcher, never inherited by a detached raylet.
exec 9>"${TMPDIR:-/tmp}/huanyu-32card-${UID}.lock"
flock -n 9 || { echo 'FATAL: another Huanyu job launcher is already running on this node' >&2; exit 1; }
echo "[huanyu-job] rank=$_RANK node=$_NODE_IP head=$_MASTER_IP:$_PORT log=$_LOG_DIR/rank_$_RANK.log"

_LIVE_RAY=0
if pgrep -x raylet >/dev/null; then _LIVE_RAY=1; fi
_SYNC_ARGS=()
if (( _LIVE_RAY )); then _SYNC_ARGS+=(--ray-running); fi
"$_PYTHON" -B "$_HELPER" sync "$_VLLM_SOURCE" "$_VLLM_TARGET" "${_SYNC_ARGS[@]}"
if (( _LIVE_RAY )); then
    echo '[huanyu-job] reusing local Ray; code matches, editable installation skipped'
else
    # Install this project's plugin, not the shared verl / verl-omni repositories.
    [[ -f "$PROJECT_ROOT/pyproject.toml" || -f "$PROJECT_ROOT/setup.py" ]] || {
        echo "FATAL: no packaging file under PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1;
    }
    git config --global --add safe.directory "$PROJECT_ROOT"
    "$_PYTHON" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
fi

# Source the established Huanyu settings. It reads optional/unset variables and
# enables xtrace itself; restore our shell options afterwards.
export HUANYU_GRAPH_MODE="${HUANYU_GRAPH_MODE:-0}"
set +u
source "$_ENV_FILE"
set +x
set -euo pipefail
cd "$PROJECT_ROOT"
huanyu_preload_allocator
echo "[huanyu-job] allocator for native check/Ray: LD_PRELOAD=$LD_PRELOAD"
export RAY_PORT="$_PORT" RAY_DASHBOARD_PORT="$_DASHBOARD_PORT"
export RAY_ADDRESS="$_MASTER_IP:$_PORT"
export RAY_USAGE_STATS_ENABLED=0 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HUANYU_VLLM_OMNI_ROOT="$_VLLM_TARGET"
export MINICPMO_USE_PRECOMPUTE=0 MINICPMO_TUNE_VISION=0
case "$_JOB_MODE" in
    frozen_cache)
        export HUANYU_USE_PRECOMPUTE=1 HUANYU_FREEZE_AUDIO=true
        export HUANYU_EMBED_CACHE_DIR="${HUANYU_EMBED_CACHE_DIR:-/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128}"
        export MINICPMO_TUNE_AUDIO=0 HUANYU_AQA_AUDIO_TRAIN=0
        [[ -r "$HUANYU_EMBED_CACHE_DIR/manifest.json" ]] || {
            echo "FATAL: cache manifest not readable on rank $_RANK: $HUANYU_EMBED_CACHE_DIR/manifest.json" >&2
            exit 1
        }
        ;;
    aqa_unfreeze)
        # Apply before Ray starts on BOTH nodes, after the site environment.
        # Empty the cache path as well as disabling it, including runtime_env
        # below for workers launched by an already-running frozen-cache raylet.
        export HUANYU_USE_PRECOMPUTE=0 HUANYU_FREEZE_AUDIO=false
        export HUANYU_EMBED_CACHE_DIR=""
        export MINICPMO_TUNE_AUDIO=1 HUANYU_AQA_AUDIO_TRAIN=1
        export Q36_BACKWARD_PREFETCH=off
        ;;
esac
export DEBUG_MODE=0 DEBUG_SAMPLES=0
for _key in "${!_HUANYU_POOL_CONFIG[@]}"; do
    export "$_key=${_HUANYU_POOL_CONFIG[$_key]}"
done
export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
export NPUS_PER_NODE=16 RM_ENABLE=1
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
export WANDB_MODE="${WANDB_MODE:-offline}"
echo "[huanyu-job] mode=$_JOB_MODE freeze_audio=$HUANYU_FREEZE_AUDIO precompute=$HUANYU_USE_PRECOMPUTE train_entry=$_TRAIN_ENTRY"
echo "[huanyu-job] physical=16x2 actor=${NUM_NPUS}x${NNODES} rollout_TP=$ROLLOUT_TP RM=${RM_N_GPUS_PER_NODE}x${RM_NNODES} RM_TP=$RM_TP requested=$((NUM_NPUS * NNODES + REWARD_POOL_NPUS))"
"$_PYTHON" -B "${_HUANYU_EXAMPLE_DIR}/check_native_runtime.py" \
    --check --expected-root "$_VLLM_TARGET"
echo '[huanyu-job] native import check exited successfully'
"$_PYTHON" -B "$_HELPER" check-devices

wait_for_cluster() {
    local status=0
    timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-cluster \
        --address "$RAY_ADDRESS" --expected-ips "$_IPS" --local-ip "$_NODE_IP" \
        --timeout "$_WAIT" --interval "$_INTERVAL" --mode "$1" || status=$?
    if (( status )); then
        echo "FATAL: Ray readiness check failed/timed out (exit $status); training not started" >&2
        return "$status"
    fi
}

if (( _RANK == 0 )); then
    if (( ! _LIVE_RAY )); then
        ray start --head --node-ip-address="$_NODE_IP" --port="$RAY_PORT" \
            --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" \
            --resources='{"NPU": 16}' 9>&-
    fi
    wait_for_cluster all

    # Explicitly propagate cache settings and the allocator even when the Raylet
    # predates this job. Neither Huanyu nor LD_PRELOAD is in the shared ext
    # environment whitelist; Ray can also strip its own allocator preload.
    _RUNTIME_ARGS=()
    for _key in HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR HUANYU_FREEZE_AUDIO \
        HUANYU_AQA_AUDIO_TRAIN MINICPMO_USE_PRECOMPUTE MINICPMO_TUNE_AUDIO \
        MINICPMO_TUNE_VISION WANDB_MODE LD_PRELOAD; do
        _value="$("$_PYTHON" -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
        _RUNTIME_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=${_value}")
    done
    echo "[huanyu-job] both nodes ready (16 + 16 NPU); head starting $_JOB_MODE Huanyu training"
    bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" \
        "${_TRAIN_ARGS[@]}" \
        "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=${HUANYU_FREEZE_AUDIO}" \
        "${_RUNTIME_ARGS[@]}"
else
    timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-head \
        --address "$RAY_ADDRESS" --timeout "$_WAIT" --interval "$_INTERVAL"
    if (( _LIVE_RAY )); then
        wait_for_cluster local
        echo '[huanyu-job] worker already belongs to the selected cluster; waiting, no training on this node'
        while pgrep -x raylet >/dev/null; do sleep 30; done
        echo 'FATAL: worker raylet exited' >&2
        exit 1
    else
        echo '[huanyu-job] joining as worker; no training on this node'
        # Keep this shell alive so FD 9 holds the local launcher lock.
        ray start --address="$RAY_ADDRESS" --node-ip-address="$_NODE_IP" \
            --resources='{"NPU": 16}' --block 9>&-
    fi
fi
