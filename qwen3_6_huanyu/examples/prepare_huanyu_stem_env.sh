#!/usr/bin/env bash
# Run once on EACH node, before manually starting Ray. Source to activate the
# prepared environment here; bash prints the activation command for this node.
# Same 128/256-card STEM profile as run_huanyu_stem_job.sh, with NO Ray calls.
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
