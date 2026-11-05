#!/usr/bin/env bash
# Start Ray only, once per node, in an already prepared environment.
# Keep job_bootstrap.py beside this file. No model/checkpoint path is needed.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo '[huanyu-ray] use bash start_huanyu_ray.sh, not source' >&2
    return 2
fi
set -euo pipefail

_CHECK="${HUANYU_JOB_PREFLIGHT:-${HUANYU_RAY_READINESS_CHECK:-0}}"
for _arg in "$@"; do
    case "$_arg" in
        --check) _CHECK=1 ;;
        -h|--help)
            cat <<'HELP'
Usage: bash start_huanyu_ray.sh [--check]
Run once on every node after activating its prepared environment.

Required:
  VC_TASK_INDEX       Node rank: 0 starts head, 1..N-1 join head.
  VC_WORKER_HOSTS     Comma-separated IPs/hosts, head first, identical on all nodes.
Optional:
  VC_WORKER_NUM       Must match host count; otherwise inferred from the list.
  RAY_PORT            GCS port (6925).
  RAY_DASHBOARD_PORT  Head Dashboard port (8960).
  RAY_CLUSTER_WAIT_TIMEOUT / RAY_CLUSTER_POLL_INTERVAL  Seconds (900 / 5).
  HUANYU_JOB_PREFLIGHT=1  Same GCS check as --check (default 0).
  HUANYU_RAY_READINESS_CHECK  Fallback alias when HUANYU_JOB_PREFLIGHT is unset.

Each node advertises 16 NPUs. --check waits for all nodes on head and local
registration on workers, directly through GCS. Start nodes concurrently.
Existing Ray is reused; new startup settings cannot alter a running raylet.
The script returns after startup/checks, leaving Ray running in the background.
HELP
            exit 0 ;;
        *) echo "FATAL: unknown option: $_arg (use --help)" >&2; exit 2 ;;
    esac
done
[[ "$_CHECK" == 0 || "$_CHECK" == 1 ]] || {
    echo 'FATAL: HUANYU_JOB_PREFLIGHT (or HUANYU_RAY_READINESS_CHECK) must be 0 or 1' >&2; exit 2;
}

# Match the node jobs if a platform invokes this entry once per device.
if [[ -v LOCAL_RANK ]]; then
    [[ "$LOCAL_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'FATAL: invalid LOCAL_RANK' >&2; exit 2; }
    if [[ "$LOCAL_RANK" != 0 ]]; then
        echo "[huanyu-ray] local_rank=$LOCAL_RANK skipping duplicate node launcher"
        exit 0
    fi
fi
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
for _key in ${!TORCHELASTIC_@}; do unset "$_key"; done

_HOSTS="${VC_WORKER_HOSTS:-}"
[[ -n "$_HOSTS" ]] || { echo 'FATAL: set VC_WORKER_HOSTS, head first' >&2; exit 2; }
IFS=, read -r -a _HOST_LIST <<< "$_HOSTS"
_NODES="${VC_WORKER_NUM:-${#_HOST_LIST[@]}}"
_RANK="${VC_TASK_INDEX:-}"
[[ "$_NODES" =~ ^[1-9][0-9]{0,2}$ ]] && (( _NODES <= 256 )) || {
    echo 'FATAL: VC_WORKER_NUM must be an integer in 1..256' >&2; exit 2;
}
(( ${#_HOST_LIST[@]} == _NODES )) || { echo 'FATAL: VC_WORKER_NUM disagrees with VC_WORKER_HOSTS' >&2; exit 2; }
[[ "$_RANK" =~ ^(0|[1-9][0-9]{0,2})$ ]] && (( _RANK < _NODES )) || {
    echo "FATAL: VC_TASK_INDEX must be 0..$((_NODES - 1))" >&2; exit 2;
}
_PORT="${RAY_PORT:-6925}"
_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8960}"
_WAIT="${RAY_CLUSTER_WAIT_TIMEOUT:-900}"
_INTERVAL="${RAY_CLUSTER_POLL_INTERVAL:-5}"
for _key in _PORT _DASHBOARD_PORT; do
    [[ "${!_key}" =~ ^[1-9][0-9]{0,4}$ ]] && (( ${!_key} <= 65535 )) || {
        echo 'FATAL: Ray ports must be integers in 1..65535' >&2; exit 2;
    }
done
[[ "$_PORT" != "$_DASHBOARD_PORT" ]] || { echo 'FATAL: GCS and Dashboard ports must differ' >&2; exit 2; }
for _key in _WAIT _INTERVAL; do
    [[ "${!_key}" =~ ^[1-9][0-9]{0,5}$ ]] || {
        echo 'FATAL: Ray wait timeout/interval must be positive integer seconds' >&2; exit 2;
    }
done

_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_HELPER="$_DIR/job_bootstrap.py"
[[ -r "$_HELPER" ]] || { echo "FATAL: missing $_HELPER" >&2; exit 2; }
for _command in python3 ray pgrep flock timeout; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing command $_command" >&2; exit 1; }
done
# Check the deployed helper before starting Ray. Old --state-api implementations
# used the Dashboard; do not silently fall back to that interface.
if (( _CHECK )); then
    if ! _CHECK_HELP="$(python3 -B "$_HELPER" wait-cluster --help)" || [[ "$_CHECK_HELP" != *"--gcs"* ]]; then
        echo "FATAL: $_HELPER does not support the required wait-cluster --gcs interface." >&2
        echo 'Deploy job_bootstrap.py from the same release as start_huanyu_ray.sh, then rerun --check.' >&2
        echo '[huanyu-ray] Ray has not been started or stopped by this invocation.' >&2
        exit 2
    fi
fi
# Share the existing node job lock; do not pass it to detached Ray processes.
exec 9>"${TMPDIR:-/tmp}/huanyu-32card-${UID}.lock"
flock -n 9 || { echo 'FATAL: another Huanyu launcher is running on this node' >&2; exit 1; }
_IPS="$(python3 -B "$_HELPER" resolve "$_HOSTS" --expected-nodes "$_NODES")"
IFS=, read -r -a _NODE_IPS <<< "$_IPS"
export RAY_ADDRESS="${_NODE_IPS[0]}:$_PORT"
export HUANYU_JOB_PREFLIGHT="$_CHECK" HUANYU_RAY_READINESS_CHECK="$_CHECK"
export RAY_worker_maximum_startup_concurrency=32
export RAY_prestart_worker_first_driver=1
export RAY_enable_worker_prestart=0
export RAY_USAGE_STATS_ENABLED=0 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

echo "[huanyu-ray] rank=$_RANK nodes=$_NODES NPU_per_node=16 address=$RAY_ADDRESS check=$_CHECK startup=32/1/0"
_LIVE_RAY=0
if pgrep -x raylet >/dev/null; then _LIVE_RAY=1; fi
if (( _RANK != 0 )); then
    timeout --foreground "${_WAIT}s" python3 -B "$_HELPER" wait-head \
        --address "$RAY_ADDRESS" --timeout "$_WAIT" --interval "$_INTERVAL" || exit $?
fi
if (( _LIVE_RAY )); then
    echo '[huanyu-ray] reusing existing raylet; its inherited startup settings are unchanged'
elif (( _RANK == 0 )); then
    ray start --head --port="$_PORT" --dashboard-host=0.0.0.0 \
        --dashboard-port="$_DASHBOARD_PORT" --resources='{"NPU": 16}' 9>&- || exit $?
else
    ray start --address="$RAY_ADDRESS" --resources='{"NPU": 16}' 9>&- || exit $?
fi
if (( _CHECK )); then
    _MODE=local
    if (( _RANK == 0 )); then _MODE=all; fi
    timeout --foreground "${_WAIT}s" python3 -B "$_HELPER" wait-cluster --gcs \
        --address "$RAY_ADDRESS" --expected-ips "$_IPS" --local-ip "${_NODE_IPS[$_RANK]}" \
        --mode "$_MODE" --timeout "$_WAIT" --interval "$_INTERVAL" || exit $?
    echo '[huanyu-ray] READY: GCS registration and NPU resources verified'
else
    echo '[huanyu-ray] startup/reuse complete; cluster membership/resources not checked (use --check)'
fi
printf 'For training in the calling terminal: export RAY_ADDRESS=%q\n' "$RAY_ADDRESS"
