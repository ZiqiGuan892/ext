#!/usr/bin/env bash
# Compatibility entry; the dedicated launch chain lives in examples/qwen3_6_huanyu.
# Source env_export_huanyu.sh and start Ray beforehand.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${QWEN36_HUANYU_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
exec bash "${_HUANYU_EXAMPLE_DIR}/run_huanyu.sh" "$@"
