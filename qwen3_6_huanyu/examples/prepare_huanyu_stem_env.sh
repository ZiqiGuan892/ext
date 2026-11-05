#!/usr/bin/env bash
# Run once on EACH node. Pass --start-ray to start and verify Ray. Source to activate the
# prepared environment here; bash prints the activation command for this node.
# Same 16/32/128/256-card STEM profile as run_huanyu_stem_job.sh.
# Uses prepare_huanyu_env.sh's mandatory environment checks, optional GCS check
# with --start-ray, and Ray worker startup policy for all supported card counts.
# HUANYU_ENV_FILE may select another site env_export script.
huanyu_prepare_stem_environment() {
    local examples
    examples="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return $?
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        source "$examples/prepare_huanyu_env.sh" --stem "$@"
    else
        bash "$examples/prepare_huanyu_env.sh" --stem "$@"
    fi
}

huanyu_prepare_stem_environment "$@"
