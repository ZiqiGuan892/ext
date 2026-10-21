#!/usr/bin/env bash
# Source from the outer launcher and again from the backend. Descendants keep
# the same console stream; a new launch always gets an atomically unique folder.
if [[ -n "${_HUANYU_OUTPUTS_ACTIVE:-}" &&
      "${_HUANYU_OUTPUTS_ACTIVE}" == "${HUANYU_CONSOLE_LOG:-}" &&
      -f "${HUANYU_CONSOLE_LOG}" ]]; then
    return 0
fi

_huanyu_runs_root="$(cd "${PROJECT_ROOT:?PROJECT_ROOT is required}" && pwd)/runs" || return $?
mkdir -p "${_huanyu_runs_root}" || return $?
HUANYU_RUN_DIR="$(mktemp -d "${_huanyu_runs_root}/huanyu_$(date +%Y%m%d_%H%M%S)_XXXXXX")" || return $?
export HUANYU_RUN_DIR
export HUANYU_CONSOLE_LOG="${HUANYU_RUN_DIR}/console.log"
# Deliberately replace an inherited fixed path from env_export_huanyu.sh.
export GENRM_DUMP_DIR="${HUANYU_RUN_DIR}/genrm_dump"
mkdir -p "${GENRM_DUMP_DIR}" || return $?
: > "${HUANYU_CONSOLE_LOG}" || return $?
export _HUANYU_OUTPUTS_ACTIVE="${HUANYU_CONSOLE_LOG}"
exec > >(tee -a "${HUANYU_CONSOLE_LOG}") 2>&1
echo "[huanyu-output] run_dir=${HUANYU_RUN_DIR}"
echo "[huanyu-output] console=${HUANYU_CONSOLE_LOG}"
echo "[huanyu-output] genrm_dump=${GENRM_DUMP_DIR}"
unset _huanyu_runs_root
