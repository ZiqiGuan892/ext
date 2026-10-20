#!/usr/bin/env bash
# Compatibility entry; the dedicated launch chain lives in examples/qwen3_6_huanyu.
# Source env_export_huanyu.sh and start Ray beforehand.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${QWEN36_HUANYU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
exec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu.sh" "$@"
