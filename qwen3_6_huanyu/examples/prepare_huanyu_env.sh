#!/usr/bin/env bash
# Run/source on EACH node before manually starting Ray. Reuse the job's setup
# in an isolated shell so env_export's set -ex cannot alter the caller's shell.
# source applies the resulting environment; bash prints an activation command.
huanyu_prepare_environment() {
    local project="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
    local examples log_root run_dir snapshot activation status
    [[ "$project" == /* && -d "$project" ]] || {
        echo '[huanyu-env] FATAL: PROJECT_ROOT must be an existing absolute directory' >&2
        return 2
    }
    examples="$project/examples/qwen3_6_huanyu"
    for activation in run_huanyu_32card_job.sh prepared_env.py; do
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
    echo "[huanyu-env] preparing mode=${HUANYU_JOB_MODE:-frozen_cache} log=$run_dir/prepare.log"
    # A fresh bash retains strict error handling even when this sourced function
    # is called in an if/|| expression. No Ray/engine processes are launched.
    if PROJECT_ROOT="$project" HUANYU_PREPARE_ONLY=1 \
        HUANYU_PREPARE_SNAPSHOT="$snapshot" HUANYU_PREPARE_LOG_DIR="$run_dir" \
        HUANYU_PREPARED_ENV_FILE="$activation" \
        bash "$examples/run_huanyu_32card_job.sh" "$@" \
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
        echo '[huanyu-env] preparation complete. Apply the environment in this node'
        echo '[huanyu-env] before manually starting Ray and an ordinary training entry:'
        printf 'source %q\n' "$activation"
    fi
}

huanyu_prepare_environment "$@"
