#!/usr/bin/env bash
# Source in each launcher before Python imports. No copying or installation.
huanyu_select_native_source() {
    local root="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit}}"
    local entry retained="" required
    local -a entries=()
    if [[ "$root" != /* || "$root" == *:* || "$root" == *$'\n'* ]]; then
        echo 'FATAL: HUANYU_VLLM_SOURCE must be an absolute directory without colons/newlines' >&2
        return 2
    fi
    for required in vllm_omni/__init__.py \
        vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py \
        vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py; do
        [[ -r "$root/$required" ]] || {
            echo "FATAL: Huanyu-enabled source is incomplete: $root/$required" >&2
            return 1
        }
    done
    root="$(cd -- "$root" && pwd -P)" || return $?
    IFS=: read -r -a entries <<< "${PYTHONPATH:-}"
    for entry in "${entries[@]}"; do
        case "${entry%/}" in
            ""|"$root"|"${HUANYU_VLLM_SOURCE:-$root}"|"${HUANYU_VLLM_OMNI_ROOT:-$root}"|\
            /vllm-workspace/vllm-omni|/vllm-workspace/vllm-omni/*) continue ;;
        esac
        retained+="${retained:+:}$entry"
    done
    export HUANYU_VLLM_SOURCE="$root" HUANYU_VLLM_OMNI_ROOT="$root"
    export PYTHONPATH="$root${retained:+:$retained}"
}

huanyu_select_native_source
