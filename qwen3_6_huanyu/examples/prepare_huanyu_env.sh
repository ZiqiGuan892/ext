#!/usr/bin/env bash
# Run/source on EACH node. Pass --start-ray to start and verify the cluster.
# With Ray enabled, set VC_TASK_INDEX and the same ordered VC_WORKER_HOSTS
# on every node. Rank 0 waits for all nodes; start the scripts concurrently.
# Usage: source prepare_huanyu_env.sh [--stem] [--start-ray]
# Preparation always runs native/device checks, even with HUANYU_JOB_PREFLIGHT=0.
# --start-ray also checks GCS registration; no Dashboard or probe driver needed.
# Activation exports the shared Ray worker startup policy (32/1/0) for manual starts.
# Run the job's setup in an isolated shell so env_export's set -ex cannot alter the caller's shell.
# source applies the resulting environment; bash prints an activation command.
# Neither injects a vLLM-Omni checkout into PYTHONPATH or creates a .bashrc hook.
# Preparation retires the old generated source hook and checks the installed package.
huanyu_prepare_environment() {
    local mode="${HUANYU_JOB_MODE:-frozen_cache}" setup_entry=run_huanyu_32card_job.sh
    local start_ray=0 arg
    local -a setup_args=()
    for arg in "$@"; do
        case "$arg" in
            --stem) mode=stem ;;
            --start-ray) start_ray=1 ;;
            *) setup_args+=("$arg") ;;
        esac
    done
    local project="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
    if [[ "$mode" == stem ]]; then
        setup_entry=run_huanyu_stem_job.sh
        project="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}"
    fi
    local examples log_root run_dir snapshot activation status
    [[ "$project" == /* && -d "$project" ]] || {
        echo '[huanyu-env] FATAL: PROJECT_ROOT must be an existing absolute directory' >&2
        return 2
    }
    examples="$project/examples/qwen3_6_huanyu"
    for activation in run_huanyu_32card_job.sh "$setup_entry" prepared_env.py; do
        [[ -r "$examples/$activation" ]] || {
            echo "[huanyu-env] FATAL: missing $examples/$activation" >&2
            return 2
        }
    done
    log_root="${HUANYU_PREPARE_LOG_ROOT:-$project/runs/environment}"
    [[ "$log_root" == /* ]] || {
        echo '[huanyu-env] FATAL: HUANYU_PREPARE_LOG_ROOT must be absolute' >&2
        return 2
    }
    run_dir="$(umask 077; mkdir -p -- "$log_root" &&
        mktemp -d "$log_root/prepare_$(date +%Y%m%d_%H%M%S)_XXXXXX")" || return $?
    snapshot="$run_dir/before.json"
    activation="$run_dir/activate.sh"
    # Standard library only; no Torch/plugin imports before allocator setup.
    python3 -S "$examples/prepared_env.py" snapshot "$snapshot" || return $?
    echo "[huanyu-env] preparing mode=$mode log=$run_dir/prepare.log"
    # A fresh bash retains strict error handling even when this sourced function
    # is called in an if/|| expression.
    if PROJECT_ROOT="$project" HUANYU_JOB_MODE="$mode" HUANYU_PREPARE_ONLY=1 \
        HUANYU_PREPARE_SNAPSHOT="$snapshot" HUANYU_PREPARE_LOG_DIR="$run_dir" \
        HUANYU_PREPARED_ENV_FILE="$activation" HUANYU_PREPARE_START_RAY="$start_ray" \
        bash "$examples/$setup_entry" "${setup_args[@]}" \
        > >(tee -a "$run_dir/prepare.log") 2>&1; then
        [[ -s "$activation" ]] || {
            echo '[huanyu-env] FATAL: setup returned without an activation file' >&2
            return 1
        }
    else
        status=$?
        echo "[huanyu-env] FAILED exit=$status; see $run_dir/prepare.log" >&2
        return "$status"
    fi
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        source "$activation" || return $?
    else
        echo '[huanyu-env] preparation complete. Apply the environment on this node:'
        printf 'source %q\n' "$activation"
    fi
}

huanyu_prepare_environment "$@"
