#!/usr/bin/env bash
# Compatibility entry; the dedicated launch chain lives in examples/qwen3_6_huanyu.
# Source env_export_huanyu.sh and start Ray beforehand.
set -euo pipefail
_HUANYU_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${QWEN36_HUANYU_ROOT:-$(cd "${_HUANYU_EXAMPLE_DIR}/../.." && pwd)}}"
exec bash "${_HUANYU_EXAMPLE_DIR}/run_huanyu.sh" "$@"
