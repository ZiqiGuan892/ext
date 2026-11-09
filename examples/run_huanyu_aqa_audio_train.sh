#!/usr/bin/env bash
# Copy the contents of this examples directory to:
#   ${PROJECT_ROOT}/examples/qwen3_6_huanyu/
# Source the environment and start Ray on all nodes before running this entry.
# Default pools: actor16 + RM16; upper entries may supply another topology.
# RM_ENABLE=0 uses the actor pool and rules, without allocating RM devices.
# HUANYU_AQA_USE_PRECOMPUTE=1 freezes encoder/projector and reads AQA embeddings.
set -euo pipefail

# BEGIN HUANYU output policy
huanyu_prepare_output() {
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
# Deliberately replace an inherited fixed path from environment preparation.
export GENRM_DUMP_DIR="${HUANYU_RUN_DIR}/genrm_dump"
mkdir -p "${GENRM_DUMP_DIR}" || return $?
: > "${HUANYU_CONSOLE_LOG}" || return $?
export _HUANYU_OUTPUTS_ACTIVE="${HUANYU_CONSOLE_LOG}"
exec > >(tee -a "${HUANYU_CONSOLE_LOG}") 2>&1
echo "[huanyu-output] run_dir=${HUANYU_RUN_DIR}"
echo "[huanyu-output] console=${HUANYU_CONSOLE_LOG}"
echo "[huanyu-output] genrm_dump=${GENRM_DUMP_DIR}"
unset _huanyu_runs_root

}
# END HUANYU output policy
# BEGIN HUANYU audio policy
huanyu_resolve_audio_train_mode() {
    local legacy="${HUANYU_FREEZE_AUDIO-false}" default_train key value
    case "$legacy" in
        true) default_train=false ;;
        false) default_train=true ;;
        *) echo '[huanyu-audio-mode] FATAL: HUANYU_FREEZE_AUDIO must be true or false' >&2; return 2 ;;
    esac
    for key in HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR; do
        value="${!key-$default_train}"
        case "$value" in
            true|false) export "$key=$value" ;;
            *) echo "[huanyu-audio-mode] FATAL: $key must be true or false" >&2; return 2 ;;
        esac
    done
    if [[ "$HUANYU_TRAIN_AUDIO_ENCODER" == true || "$HUANYU_TRAIN_AUDIO_PROJECTOR" == true ]]; then
        export HUANYU_FREEZE_AUDIO=false MINICPMO_TUNE_AUDIO=1
    else
        export HUANYU_FREEZE_AUDIO=true MINICPMO_TUNE_AUDIO=0
    fi
}
# END HUANYU audio policy
# BEGIN HUANYU graph policy
huanyu_configure_graph_mode() {
    if [[ "${HUANYU_PRECISION_PROFILE:-full}" == full && ( "${HUANYU_GRAPH_MODE:-0}" == 1 || "${HUANYU_GRAPH_MODE:-0}" == true ) ]]; then
        echo "[huanyu-graph] FATAL: HUANYU_PRECISION_PROFILE=full cannot capture Python precision reference kernels; use HUANYU_GRAPH_MODE=0, or explicitly select profile=custom for a separate comparison" >&2
        return 2
    fi
    case "${HUANYU_GRAPH_MODE:-0}" in
        1|true)
            export HUANYU_GRAPH_MODE=1 HUANYU_ENFORCE_EAGER=false
            export VLLM_TORCH_COMPILE=1
            export TORCH_COMPILE_DISABLE=0 NPU_GRAPH_CAPTURE_DISABLE=0
            ;;
        0|false)
            export HUANYU_GRAPH_MODE=0 HUANYU_ENFORCE_EAGER=true
            export VLLM_TORCH_COMPILE=0
            export TORCH_COMPILE_DISABLE=1 NPU_GRAPH_CAPTURE_DISABLE=1
            ;;
        *)
            echo "[huanyu-graph] FATAL: HUANYU_GRAPH_MODE must be 0/1 or false/true" >&2
            return 2
            ;;
    esac

    # Explicit enable/disable values also override flags inherited from an
    # already-running Raylet. Merely unsetting the driver's environment cannot
    # remove those worker-side flags. Keep values as Hydra-quoted strings.
    HUANYU_GRAPH_ARGS=(
        "actor_rollout_ref.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "reward.reward_model.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_GRAPH_MODE=\"${HUANYU_GRAPH_MODE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_TORCH_COMPILE=\"${VLLM_TORCH_COMPILE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_COMPILE_DISABLE=\"${TORCH_COMPILE_DISABLE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.NPU_GRAPH_CAPTURE_DISABLE=\"${NPU_GRAPH_CAPTURE_DISABLE}\""
    )
}
# END HUANYU graph policy
# BEGIN HUANYU precision policy
huanyu_configure_precision_mode() {
    local _key _value _arg _tp="${ROLLOUT_TP:-4}"
    local _profile="${HUANYU_PRECISION_PROFILE:-full}"
    case "$_profile" in
        full|custom) ;;
        *) echo "[huanyu-precision] FATAL: HUANYU_PRECISION_PROFILE must be full or custom" >&2; return 2 ;;
    esac
    if [[ "$_profile" == full && ( "${HUANYU_GRAPH_MODE:-0}" == 1 || "${HUANYU_GRAPH_MODE:-0}" == true ) ]]; then
        echo "[huanyu-precision] FATAL: profile=full uses Python precision reference kernels and cannot capture inference graphs; set HUANYU_GRAPH_MODE=0, or explicitly select HUANYU_PRECISION_PROFILE=custom for a separate comparison" >&2
        return 2
    fi
    export HUANYU_PRECISION_PROFILE="$_profile"
    local -a _changed=()
    local -a _on=(
        MINICPMO_HUANYU_GDN_FP32_TRAINING MINICPMO_GDN_FP32_REFERENCE
        MINICPMO_HUANYU_LAYERWISE_RELOAD MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE
        MINICPMO_ISOLATE_MEDIA_ENCODERS
    )
    local -a _full=(
        MINICPMO_GDN_CACHE_BARRIER MINICPMO_GDN_PREFILL_REFERENCE
        MINICPMO_GDN_L2NORM_REFERENCE MINICPMO_GDN_CONV_REFERENCE
        MINICPMO_GDN_GATING_REFERENCE MINICPMO_PRIVATE_HYBRID_CACHE
        MINICPMO_STABLE_LINEAR
    )
    local -a _off=(
        MINICPMO_HF_NORM_REFERENCE
        MINICPMO_HF_ROUTER_REFERENCE MINICPMO_HF_TEXT_ROPE_REFERENCE
        MINICPMO_MOE_MATCH_ROLLOUT MINICPMO_ISOLATE_FULL_ATTENTION
        MINICPMO_VISION_POSITION_REFERENCE MINICPMO_NATIVE_VISION_REFERENCE
    )
    for _key in "${_on[@]}" "${_full[@]}" "${_off[@]}"; do
        if [[ "$_profile" == full && " ${_on[*]} ${_full[*]} " == *" $_key "* ]]; then
            if [[ -v "$_key" && "${!_key}" != 1 ]]; then
                _changed+=("$_key=${!_key}->1")
            fi
            printf -v "$_key" '%s' 1
        elif [[ ! -v "$_key" ]]; then
            case " ${_on[*]} " in
                *" $_key "*) printf -v "$_key" '%s' 1 ;;
                *) printf -v "$_key" '%s' 0 ;;
            esac
        fi
        case "${!_key}" in
            0|1) export "$_key" ;;
            *) echo "[huanyu-precision] FATAL: $_key must be 0 or 1" >&2; return 2 ;;
        esac
    done
    if [[ "$_profile" == full ]]; then
        if [[ -v MINICPMO_FSDP2_PIN_MEMORY && "$MINICPMO_FSDP2_PIN_MEMORY" != 0 ]]; then
            _changed+=("MINICPMO_FSDP2_PIN_MEMORY=$MINICPMO_FSDP2_PIN_MEMORY->0")
        fi
        export MINICPMO_FSDP2_PIN_MEMORY=0
    else
        export MINICPMO_FSDP2_PIN_MEMORY="${MINICPMO_FSDP2_PIN_MEMORY:-0}"
    fi
    case "$MINICPMO_FSDP2_PIN_MEMORY" in
        0|1) ;;
        *) echo "[huanyu-precision] FATAL: MINICPMO_FSDP2_PIN_MEMORY must be 0 or 1" >&2; return 2 ;;
    esac
    # Hydra applies the last CLI override. Use that exact TP in both preflight
    # arithmetic validation and the worker environment, not a stale shell TP.
    for _arg in "$@"; do
        _key="${_arg%%=*}"
        while [[ "$_key" == +* ]]; do _key="${_key#+}"; done
        if [[ "$_key" == actor_rollout_ref.rollout.tensor_model_parallel_size ]]; then
            _tp="${_arg#*=}"
            _tp="${_tp#\"}"; _tp="${_tp%\"}"
            _tp="${_tp#\'}"; _tp="${_tp%\'}"
        fi
    done
    if [[ ! "$_tp" =~ ^[1-9][0-9]*$ ]]; then
        echo "[huanyu-precision] FATAL: rollout tensor_model_parallel_size must be a positive integer" >&2
        return 2
    fi
    export ROLLOUT_TP="$_tp" MINICPMO_HUANYU_ROLLOUT_TP="$_tp"
    for _key in MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE; do
        _value="${!_key:-0}"
        case "$_value" in
            0) ;;
            4)
                if [[ "$_tp" != 4 ]]; then
                    echo "[huanyu-precision] FATAL: $_key=4 requires actual rollout TP=4, got $_tp" >&2
                    return 2
                fi ;;
            *) echo "[huanyu-precision] FATAL: $_key supports only 0 or 4" >&2; return 2 ;;
        esac
        export "$_key=$_value"
    done
    export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_huanyu.engine_hooks
    export HUANYU_USE_PRECOMPUTE="${HUANYU_USE_PRECOMPUTE:-0}"
    case "$HUANYU_USE_PRECOMPUTE" in
        0|1) ;;
        *) echo "[huanyu-precision] FATAL: HUANYU_USE_PRECOMPUTE must be 0 or 1" >&2; return 2 ;;
    esac
    # The two-node job snapshots these defaults/caller overrides across the
    # site env file. Cache mode, TP and HCCL are resolved later by the launcher.
    HUANYU_PRECISION_KEYS=(HUANYU_PRECISION_PROFILE "${_on[@]}" "${_full[@]}" "${_off[@]}" \
        MINICPMO_FSDP2_PIN_MEMORY MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE)
    # Opt-in job policy is snapshotted across site setup and explicitly sent to
    # fresh Ray workers even when their raylet was started with other defaults.
    export HUANYU_JOB_DETERMINISM="${HUANYU_JOB_DETERMINISM:-0}"
    case "$HUANYU_JOB_DETERMINISM" in
        0|1) ;;
        *) echo '[huanyu-precision] FATAL: HUANYU_JOB_DETERMINISM must be 0 or 1' >&2; return 2 ;;
    esac
    HUANYU_PRECISION_KEYS+=(HUANYU_JOB_DETERMINISM)
    # Preserve optional timing through site setup and existing Ray clusters.
    export HUANYU_GEN_TIMING="${HUANYU_GEN_TIMING:-0}"
    export HUANYU_GEN_TIMING_ENGINE_EVERY="${HUANYU_GEN_TIMING_ENGINE_EVERY:-100}"
    case "$HUANYU_GEN_TIMING" in
        0|1) ;;
        *) echo '[huanyu-timing] FATAL: HUANYU_GEN_TIMING must be 0 or 1' >&2; return 2 ;;
    esac
    if [[ ! "$HUANYU_GEN_TIMING_ENGINE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
        echo '[huanyu-timing] FATAL: HUANYU_GEN_TIMING_ENGINE_EVERY must be a positive integer' >&2
        return 2
    fi
    HUANYU_PRECISION_KEYS+=(HUANYU_GEN_TIMING HUANYU_GEN_TIMING_ENGINE_EVERY)
    if [[ "$HUANYU_JOB_DETERMINISM" == 1 ]]; then
        export HCCL_DETERMINISTIC=strict ASCEND_LAUNCH_BLOCKING=1 NCCL_DETERMINISTIC=1
        export LCCL_DETERMINISTIC=1 VLLM_BATCH_INVARIANT=0
        HUANYU_PRECISION_KEYS+=(ASCEND_LAUNCH_BLOCKING NCCL_DETERMINISTIC LCCL_DETERMINISTIC VLLM_BATCH_INVARIANT)
    fi
    HUANYU_PRECISION_ARGS=()
    if [[ "$HUANYU_JOB_DETERMINISM" == 1 ]]; then
        # verl's full_determinism option also enables batch-invariant kernels.
        # This job intentionally selects the independent worker policy instead.
        HUANYU_PRECISION_ARGS+=(
            'actor_rollout_ref.rollout.full_determinism=false'
            '++reward.reward_model.rollout.full_determinism=false'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enable_batch_invariant=false'
        )
    fi
    # Keep the outer actor engine kwargs consistent with stage-0 engine_extras:
    # native deploy merging must not restore the generic colocated extension.
    # This applies to the actor only; the reward model keeps its own extension.
    HUANYU_PRECISION_ARGS+=(
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls=verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
    )
    for _key in "${HUANYU_PRECISION_KEYS[@]}" \
        MINICPMO_HUANYU_ROLLOUT_TP VLLM_OMNI_EXTERNAL_MODULES HUANYU_USE_PRECOMPUTE \
        HCCL_OP_EXPANSION_MODE HCCL_DETERMINISTIC; do
        _value="${!_key:-}"
        # Do not introduce an empty HCCL setting: native CANN may distinguish
        # an empty value from an unset variable. Forward only a defined policy.
        if [[ "$_key" == HCCL_* && -z "$_value" ]]; then continue; fi
        # Hydra quoted strings preserve the driver's actual numerical policy
        # when reusing an already-running Raylet.
        _value="${_value//\\/\\\\}"
        _value="${_value//\"/\\\"}"
        HUANYU_PRECISION_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=\"${_value}\"")
    done
    if (( ${#_changed[@]} )); then
        echo "[huanyu-precision] profile=full overrides inherited values: ${_changed[*]}"
    fi
    # Repeated source calls in one shell revalidate/rebuild argv but only repeat
    # the detailed report if its actual values changed. Child launchers report
    # independently, and the worker Python installers report active execution.
    local _report="profile=$_profile actual_TP=$_tp cache=$HUANYU_USE_PRECOMPUTE graph=${HUANYU_GRAPH_MODE:-0}"
    for _key in "${HUANYU_PRECISION_KEYS[@]}"; do _report+=" $_key=${!_key}"; done
    if [[ "${_HUANYU_PRECISION_LAST_REPORT:-}" != "$_report" ]]; then
        echo "[huanyu-precision] profile=$_profile actual_TP=$_tp cache=$HUANYU_USE_PRECOMPUTE graph=${HUANYU_GRAPH_MODE:-0} hooks=$VLLM_OMNI_EXTERNAL_MODULES"
        for _key in "${HUANYU_PRECISION_KEYS[@]}"; do
            echo "[huanyu-precision] effective $_key=${!_key}"
        done
        _HUANYU_PRECISION_LAST_REPORT="$_report"
    fi
}
# END HUANYU precision policy
# BEGIN HUANYU native policy
huanyu_select_native_source() {
    local root="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}}"
    local explicit="${1:-}" old_source="${HUANYU_VLLM_SOURCE:-}" old_root="${HUANYU_VLLM_OMNI_ROOT:-}"
    local entry retained="" required
    local -a entries=()
    # Retire source choices inherited from old prepare/activate.sh files.
    if [[ "$explicit" != --explicit-source ]]; then
        case "${root%/}" in
            */vllm-omni-xy|*/vllm-omni-xy-*) root=/vllm-workspace/vllm-omni ;;
        esac
    fi
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
            ""|"$root"|"$old_source"|"$old_root"|\
            /vllm-workspace/vllm-omni|/vllm-workspace/vllm-omni/*|\
            */vllm-omni-xy|*/vllm-omni-xy/*|*/vllm-omni-xy-*|*/vllm-omni-xy-*/*) continue ;;
        esac
        # A differently named checkout must not shadow the installed package.
        [[ -f "$entry/vllm_omni/__init__.py" ]] && continue
        retained+="${retained:+:}$entry"
    done
    # These are expected-origin checks, not import-path overrides.
    export HUANYU_VLLM_SOURCE="$root" HUANYU_VLLM_OMNI_ROOT="$root"
    if [[ "$explicit" == --explicit-source ]]; then
        export PYTHONPATH="$root${retained:+:$retained}"
    elif [[ -n "$retained" ]]; then
        export PYTHONPATH="$retained"
    else
        unset PYTHONPATH
    fi
}
# END HUANYU native policy


export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
cd "${PROJECT_ROOT}"
_HUANYU_MODEL_DIR="${PROJECT_ROOT}/verl_omni_ext/models/qwen3_6_huanyu"
huanyu_prepare_output || exit $?
# Prepared-environment entry: do not repeat installation or create a local Ray
# cluster when the requested cluster is unavailable. Job setup happens upstream.
export HUANYU_SKIP_EDITABLE_INSTALL=1
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
[[ "$RAY_ADDRESS" != local ]] || { echo 'FATAL: use RAY_ADDRESS=auto or head_ip:port for the prepared cluster' >&2; exit 2; }
_HUANYU_RAY_TIMEOUT="${HUANYU_RAY_STATUS_TIMEOUT:-30}"
[[ "$_HUANYU_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: HUANYU_RAY_STATUS_TIMEOUT must be positive seconds' >&2; exit 2; }
if ! timeout --foreground "${_HUANYU_RAY_TIMEOUT}s" ray status --address="$RAY_ADDRESS"; then
    echo 'FATAL: existing Ray is not reachable; prepare the cluster before training' >&2
    exit 1
fi
# Canonical reward selection wins over the legacy RM_ENABLE switch.
if [[ -n "${HUANYU_REWARD_MODE:-}" ]]; then
    case "${HUANYU_REWARD_MODE,,}" in
        rm|genrm) export HUANYU_REWARD_MODE=RM RM_ENABLE=1 ;;
        rule_based|ruled_based|rule) export HUANYU_REWARD_MODE=rule_based RM_ENABLE=0 ;;
        *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based (ruled_based is accepted)' >&2; exit 2 ;;
    esac
else
    export RM_ENABLE="${RM_ENABLE:-1}"
    case "$RM_ENABLE" in
        1) export HUANYU_REWARD_MODE=RM ;;
        0) export HUANYU_REWARD_MODE=rule_based ;;
        *) echo '[huanyu-aqa] FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
    esac
fi
export NUM_NPUS="${NUM_NPUS:-16}" NNODES="${NNODES:-1}" ROLLOUT_TP="${ROLLOUT_TP:-4}"
for _huanyu_key in NUM_NPUS NNODES ROLLOUT_TP; do
    [[ "${!_huanyu_key}" =~ ^[1-9][0-9]{0,4}$ ]] || {
        echo "[huanyu-aqa] FATAL: ${_huanyu_key} must be an integer in 1..99999" >&2; exit 2;
    }
done
if (( NUM_NPUS % ROLLOUT_TP != 0 )); then
    echo '[huanyu-aqa] FATAL: ROLLOUT_TP must divide the per-node actor pool' >&2
    exit 2
fi
_huanyu_actor_world=$((NUM_NPUS * NNODES))
_huanyu_rollout_replicas=$((_huanyu_actor_world / ROLLOUT_TP))
case "${RM_ENABLE}" in
    1)
        _HUANYU_BACKEND="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh"
        export RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}" RM_NNODES="${RM_NNODES:-1}"
        export RM_TP="${RM_TP:-4}" RM_SPLIT_NODES="${RM_SPLIT_NODES:-1}"
        for _huanyu_key in RM_N_GPUS_PER_NODE RM_NNODES RM_TP; do
            [[ "${!_huanyu_key}" =~ ^[1-9][0-9]{0,4}$ ]] || {
                echo "[huanyu-aqa] FATAL: ${_huanyu_key} must be an integer in 1..99999" >&2; exit 2;
            }
        done
        if (( RM_N_GPUS_PER_NODE % RM_TP != 0 )); then
            echo '[huanyu-aqa] FATAL: RM_TP must divide the per-node reward pool' >&2
            exit 2
        fi
        export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
        _huanyu_reward_args=(
            reward.reward_model.enable=true
            reward.reward_model.enable_resource_pool=true
            "reward.reward_model.n_gpus_per_node=${RM_N_GPUS_PER_NODE}"
            "reward.reward_model.nnodes=${RM_NNODES}"
            "reward.reward_model.rollout.tensor_model_parallel_size=${RM_TP}"
        )
        ;;
    0)
        # AQA has its own character-bigram rule; bypass the GenRM scorer.
        _HUANYU_BACKEND="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_gspo_npu.sh"
        export REWARD_POOL_NPUS=0 RM_NNODES=0 RM_SPLIT_NODES=0
        # The prompt-side <think> opener is absent from decoded responses. Keep
        # the rule's final-answer requirement consistent with the chat template.
        _huanyu_thinking=false
        if [[ "${ENABLE_THINKING:-1}" == 1 ]]; then
            _huanyu_thinking=true
        fi
        _huanyu_reward_args=(
            reward.reward_model.enable=false
            reward.reward_model.enable_resource_pool=false
            reward.reward_model.model_path=null
            reward.reward_model.n_gpus_per_node=0
            reward.reward_model.nnodes=0
            reward.reward_manager.source=register
            reward.reward_manager.name=naive
            reward.custom_reward_function.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.rule_based_reward
            reward.custom_reward_function.name=compute_score
            "++reward.custom_reward_function.reward_kwargs.require_think_end=${_huanyu_thinking}"
            "++data.apply_chat_template_kwargs.enable_thinking=${_huanyu_thinking}"
        )
        ;;
    *) echo "[huanyu-aqa] FATAL: RM_ENABLE must be 0 or 1" >&2; exit 2 ;;
esac
for _huanyu_file in "${_HUANYU_MODEL_DIR}/runtime_imports.py" "${_HUANYU_BACKEND}" \
    "${_HUANYU_EXAMPLE_DIR}/job_bootstrap.py"; do
    if [[ ! -f "${_huanyu_file}" ]]; then
        echo "[huanyu-aqa] FATAL: missing ${_huanyu_file}; deploy the complete Huanyu package and examples" >&2
        exit 2
    fi
done

# Explicit AQA opt-in avoids inheriting a 35B/STEM cache from environment setup.
# Cached embeddings require BOTH audio components frozen, including when old
# unfreeze flags remain exported. Model overrides below enforce the same mode.
export HUANYU_AQA_USE_PRECOMPUTE="${HUANYU_AQA_USE_PRECOMPUTE:-0}"
case "$HUANYU_AQA_USE_PRECOMPUTE" in
    1)
        export HUANYU_AQA_CACHE_DIR="${HUANYU_AQA_CACHE_DIR:-/opt/huawei/dataset/infra_data/data/embed_cache_2748_huanyu_aqa}"
        export HUANYU_EMBED_CACHE_DIR="$HUANYU_AQA_CACHE_DIR"
        if [[ ! -f "$HUANYU_EMBED_CACHE_DIR/manifest.json" || ! -r "$HUANYU_EMBED_CACHE_DIR/manifest.json" ]]; then
            echo "[huanyu-aqa] FATAL: readable cache manifest required: $HUANYU_EMBED_CACHE_DIR/manifest.json; precompute first" >&2
            exit 2
        fi
        export HUANYU_FREEZE_AUDIO=true
        export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
        export HUANYU_USE_PRECOMPUTE=1
        _huanyu_audio_mode=cached
        ;;
    0)
        export HUANYU_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
        _huanyu_audio_mode=online
        ;;
    *) echo '[huanyu-aqa] FATAL: HUANYU_AQA_USE_PRECOMPUTE must be 0 or 1' >&2; exit 2 ;;
esac
huanyu_resolve_audio_train_mode || exit $?
export HUANYU_AQA_AUDIO_TRAIN=1
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_VISION=0
export Q36_BACKWARD_PREFETCH=off
export ACTOR_FORWARD_PREFETCH="${ACTOR_FORWARD_PREFETCH:-false}"
export ACTOR_OFFLOAD_POLICY="${ACTOR_OFFLOAD_POLICY:-false}"
export ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-false}"
export FILTER_OVERLONG=false

# Explicit generic paths win; AQA aliases only supply mode-specific defaults.
export TRAIN_FILE="${TRAIN_FILE:-${HUANYU_AQA_TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
export VAL_FILE="${VAL_FILE:-${HUANYU_AQA_VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
echo "[huanyu-aqa] train=$TRAIN_FILE validation=$VAL_FILE"
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}"

# Default to one sequence per actor rank. Larger mini-batches are allowed when
# trajectories divide evenly across ranks; micro-batching then accumulates
# gradients and can retain unsharded gradients when no_sync is enabled.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
for _huanyu_integer in "${TRAIN_BATCH_SIZE}" "${PPO_MINI_BATCH_SIZE}" "${ROLLOUT_N}" "${PPO_MICRO_BATCH_SIZE_PER_GPU}"; do
    if [[ ! "${_huanyu_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[huanyu-aqa] FATAL: batch sizes and ROLLOUT_N must be positive integers" >&2
        exit 2
    fi
done
_huanyu_mini_sequences=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))
if (( _huanyu_mini_sequences % _huanyu_actor_world != 0 || ROLLOUT_N < 2 ||
      PPO_MINI_BATCH_SIZE % _huanyu_rollout_replicas != 0 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "[huanyu-aqa] FATAL: require (ppo_mini_batch*rollout.n)%${_huanyu_actor_world}=0, rollout.n>=2, ppo_mini_batch%${_huanyu_rollout_replicas}=0 and train_batch%ppo_mini_batch=0" >&2
    exit 2
fi

if [[ "${ACTOR_DYNAMIC_BSZ:-${AQA_DYNAMIC_BSZ:-false}}" == false ]] &&
   (( _huanyu_mini_sequences % (_huanyu_actor_world * PPO_MICRO_BATCH_SIZE_PER_GPU) != 0 )); then
    echo '[huanyu-aqa] FATAL: static PPO trajectories must divide into per-rank microbatches' >&2
    exit 2
fi

# DEBUG_MODE rewrites batching and may create temporary debug parquet files.
# For a short run use TOTAL_TRAINING_STEPS=2; do not use that old debug path.
if [[ "${DEBUG_MODE:-0}" != 0 || "${DEBUG_SAMPLES:-0}" != 0 ]]; then
    echo "[huanyu-aqa] FATAL: use TOTAL_TRAINING_STEPS=2 instead of DEBUG_MODE/DEBUG_SAMPLES" >&2
    exit 2
fi
export DEBUG_MODE=0 DEBUG_SAMPLES=0
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH}}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((ROLLOUT_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128))}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN}}"
export MINICPMO_MM_PROMPT_BUDGET="${MINICPMO_MM_PROMPT_BUDGET:-${ROLLOUT_PROMPT_LENGTH}}"
_huanyu_steps="${TOTAL_TRAINING_STEPS:--1}"
if [[ "${_huanyu_steps}" == -1 || "${_huanyu_steps}" == null ]]; then
    _huanyu_steps=null
elif [[ ! "${_huanyu_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[huanyu-aqa] FATAL: TOTAL_TRAINING_STEPS must be a positive integer, -1 or null" >&2
    exit 2
fi
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
_huanyu_experiment="audio_freeze_${HUANYU_FREEZE_AUDIO}_${_huanyu_audio_mode}"
if [[ "$HUANYU_TRAIN_AUDIO_ENCODER" != "$HUANYU_TRAIN_AUDIO_PROJECTOR" ]]; then
    _huanyu_experiment="audio_encoder_${HUANYU_TRAIN_AUDIO_ENCODER}_projector_${HUANYU_TRAIN_AUDIO_PROJECTOR}_${_huanyu_audio_mode}"
fi
if [[ "${RM_ENABLE}" == 0 ]]; then
    _huanyu_experiment+="_rule_based_${_huanyu_actor_world}card"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${_huanyu_experiment}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

# Preserve the non-reward OmniMix settings when calling the base backend
# directly. The common AQA settings below still override batching and freezing.
_huanyu_backend_defaults=()
if [[ "${RM_ENABLE}" == 0 ]]; then
    if [[ ! -f "${_HUANYU_MODEL_DIR}/rule_based_reward.py" ]]; then
        echo "[huanyu-aqa] FATAL: deploy ${_HUANYU_MODEL_DIR}/rule_based_reward.py" >&2
        exit 2
    fi
    export MINICPMO_ENABLE_MM_FORWARD=1
    export MINICPMO_ALLOW_PACKED_MM_FORWARD=1
    export MINICPMO_FSDP2_PIN_MEMORY=0
    export MINICPMO_DISABLE_MM_DEDUP=1
    export MINICPMO_USE_CUSTOM_DATASET=1
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}" REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bfloat16}"
    export VLLM_ASCEND_WEIGHT_NZ_MODE=0 ASCEND_WEIGHT_NZ_MODE=0 ASCEND_ENABLE_TRANS_DATA_CACHE=0
    _huanyu_backend_defaults=(
        ++data.video_key=videos
        ++data.audio_key=audios
        ++data.mm_processor_kwargs.sampling_rate=16000
        ++data.mm_processor_kwargs.max_slice_nums=1
        ++data.mm_processor_kwargs.use_image_id=false
        +actor_rollout_ref.model.override_config.stream_input=false
        actor_rollout_ref.model.override_config.attn_implementation=eager
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
        actor_rollout_ref.actor.entropy_from_logits_chunk_size=512
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512
        actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']"
        actor_rollout_ref.actor.entropy_checkpointing=False
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
        ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
        ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true
        actor_rollout_ref.rollout.enforce_eager=true
        "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}"
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB:-1024}"
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
        actor_rollout_ref.ref.entropy_checkpointing=False
        actor_rollout_ref.ref.entropy_from_logits_with_chunking=False
    )
fi

# The direct AQA entry (including RM_ENABLE=0) must establish the same policy
# before its first native import. Do not rely only on the final backend or a
# previously started Raylet. The chosen TP is enforced in the final overrides.
huanyu_configure_graph_mode || exit $?
huanyu_configure_precision_mode "$@" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" || exit $?
echo "[huanyu-aqa] precision train_GDN=$MINICPMO_HUANYU_GDN_FP32_TRAINING rollout_GDN=$MINICPMO_GDN_FP32_REFERENCE reload=$MINICPMO_HUANYU_LAYERWISE_RELOAD KV_zero=$MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE hooks=$VLLM_OMNI_EXTERNAL_MODULES"

# Check all rows without decoding audio; a validation split containing mixed
# OmniMix rows must not enter this mode even when val_before_train=false.
python3 -B "${_HUANYU_EXAMPLE_DIR}/job_bootstrap.py" check-aqa \
    --train "${TRAIN_FILE}" --validation "${VAL_FILE}" --train-batch-size "${TRAIN_BATCH_SIZE}"

huanyu_select_native_source || exit $?
_huanyu_retained_path=""
IFS=: read -r -a _huanyu_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|"${HUANYU_VLLM_OMNI_ROOT}"|"${PROJECT_ROOT}"|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit/*|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit/*) continue ;;
    esac
    _huanyu_retained_path+="${_huanyu_retained_path:+:}${_huanyu_path}"
done
export PYTHONPATH="${PROJECT_ROOT}${_huanyu_retained_path:+:${_huanyu_retained_path}}"
python3 -B "${_HUANYU_MODEL_DIR}/runtime_imports.py" --check --expected-root "${HUANYU_VLLM_OMNI_ROOT}"

_huanyu_ray_args=()
for _huanyu_env_key in PYTHONPATH HUANYU_VLLM_OMNI_ROOT HUANYU_VLLM_SOURCE HUANYU_FREEZE_AUDIO \
    HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR MINICPMO_USE_PRECOMPUTE \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH GENRM_DUMP_DIR HUANYU_REWARD_MODE; do
    _huanyu_env_json="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "${_huanyu_env_key}")"
    _huanyu_ray_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_huanyu_env_key}=${_huanyu_env_json}")
done

# Defaults precede caller arguments, so explicit trainer paths and media budget
# remain overridable without separate length/precision entry scripts.
_huanyu_budget_json="$(python3 -c 'import json, os; print(json.dumps(os.environ["MINICPMO_MM_PROMPT_BUDGET"]))')"
echo "[huanyu-aqa] freeze_audio=${HUANYU_FREEZE_AUDIO} mode=${_huanyu_audio_mode} cache=${HUANYU_USE_PRECOMPUTE} visual=frozen; actor=${_huanyu_actor_world}NPU RM=${REWARD_POOL_NPUS}NPU RM_ENABLE=${RM_ENABLE} actor_pool=${NUM_NPUS}x${NNODES}"
if [[ "$HUANYU_USE_PRECOMPUTE" == 1 ]]; then
    echo "[huanyu-aqa] embedding_cache=$HUANYU_EMBED_CACHE_DIR (actor/ref; rollout uses native audio encoding)"
fi
echo "[huanyu-aqa] train_audio_encoder=$HUANYU_TRAIN_AUDIO_ENCODER train_audio_projector=$HUANYU_TRAIN_AUDIO_PROJECTOR"
if [[ "${RM_ENABLE}" == 0 ]]; then
    echo "[huanyu-aqa] reward=rule_based (character bigram Dice against ground_truth; no model/API)"
fi
echo "[huanyu-aqa] train_batch=${TRAIN_BATCH_SIZE} ppo_mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} microbatch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU (used when dynamic_bsz=false)"
echo "[huanyu-aqa] mini_batch_trajectories=${_huanyu_mini_sequences} sequences_per_actor_rank=$((_huanyu_mini_sequences / _huanyu_actor_world))"
echo "[huanyu-aqa] prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} steps=${_huanyu_steps}"
echo "[huanyu-aqa] dump defaults: rollout=${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump} validation=${VALIDATION_DATA_DIR:-$HUANYU_RUN_DIR/validation_dump} genrm=${GENRM_DUMP_DIR} (trailing trainer overrides take precedence)"
echo "[huanyu-aqa] actor_forward_prefetch=${ACTOR_FORWARD_PREFETCH} ref_forward_prefetch=${REF_FORWARD_PREFETCH:-false} Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH}; HCCL timeout is unchanged"

# Dynamic batching defaults precede user arguments so each phase can opt in.
# Required settings follow user arguments so the checked datasets and execution
# contract cannot be bypassed by stale command-line options. Model selection is
# the verl alias; the checkpoint's HF architecture and RM alias remain intact.
exec bash "${_HUANYU_BACKEND}" "${_huanyu_backend_defaults[@]}" \
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_DYNAMIC_BSZ:-${AQA_DYNAMIC_BSZ:-false}}" \
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_DYNAMIC_BSZ:-${AQA_DYNAMIC_BSZ:-false}}" \
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-${AQA_DYNAMIC_BSZ:-false}}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=${_huanyu_budget_json}" \
    "trainer.logger=${TRAINER_LOGGER:-[console]}" \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}" \
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR:-$HUANYU_RUN_DIR/validation_dump}" \
    "trainer.test_freq=${TEST_FREQ:-10}" "$@" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset \
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset \
    actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu \
    actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.lora_rank=0 \
    "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=${HUANYU_FREEZE_AUDIO}" \
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=${HUANYU_TRAIN_AUDIO_ENCODER}" \
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=${HUANYU_TRAIN_AUDIO_PROJECTOR}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    "actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD}" \
    'actor_rollout_ref.model.exclude_modules=.*model\.visual.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*' \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=${ACTOR_FORWARD_PREFETCH}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-true}" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false}" \
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY}" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    "actor_rollout_ref.ref.fsdp_config.forward_prefetch=${REF_FORWARD_PREFETCH:-false}" \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" \
    "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS:-$_huanyu_rollout_replicas}" \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    "${_huanyu_reward_args[@]}" \
    "trainer.n_gpus_per_node=${NUM_NPUS}" \
    "trainer.nnodes=${NNODES}" \
    trainer.total_training_steps="${_huanyu_steps}" \
    "${_huanyu_ray_args[@]}" "${HUANYU_PRECISION_ARGS[@]}"
