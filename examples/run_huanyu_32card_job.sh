#!/usr/bin/env bash
# Run on EVERY node. VC_TASK_INDEX=0 starts training; others only join Ray.
# Legacy modes use two nodes; STEM supports 8/16 nodes through its job wrapper.
# Deploy to ${PROJECT_ROOT}/examples/qwen3_6_huanyu/ with the Huanyu package.
# Defaults: frozen Huanyu, 35B sampleall, precomputed q128 cache, actor16 + RM16.
# HUANYU_JOB_MODE=aqa_unfreeze selects online AQA training via its own entry.
# AQA length and gradient-sync settings are supplied by environment/Hydra args.
# Precision defaults: the complete full policy, including FP32 training/GDN,
# prefill/gating/conv/L2/barrier, private hybrid cache, stable linear, layerwise
# IPC reload, logical KV zeroing, isolated media and FSDP pin_memory=0.
# Apply the same policy before Ray on BOTH nodes and explicitly in runtime_env
# for training. full requires graph mode=0; explicit profile=custom permits A/B.
# Uses the selected vLLM-Omni source directly; never copies/replaces checkouts.
# Never stops an existing Ray cluster.
set -euo pipefail

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



# Some schedulers wrap this node-level entry in torchrun/torch.distributed.launch
# with one Bash process per NPU. Elect local rank 0 BEFORE Python, setup, logs or
# flock; peers exit successfully so the outer launcher does not kill the owner.
# VC_TASK_INDEX remains the node rank: global RANK=0 would exclude other nodes.
_HUANYU_LOCAL_RANK="${LOCAL_RANK-}"
_HUANYU_HAS_LOCAL_RANK=0
if [[ -v LOCAL_RANK ]]; then _HUANYU_HAS_LOCAL_RANK=1; fi
_HUANYU_NODE_ARGS=()
while (( $# )); do
    case "$1" in
        --local-rank=*|--local_rank=*) _HUANYU_ARG_RANK="${1#*=}"; shift ;;
        --local-rank|--local_rank)
            (( $# >= 2 )) || { echo 'FATAL: missing local rank argument value' >&2; exit 2; }
            _HUANYU_ARG_RANK="$2"; shift 2 ;;
        *) _HUANYU_NODE_ARGS+=("$1"); shift; continue ;;
    esac
    if (( _HUANYU_HAS_LOCAL_RANK )) && [[ "$_HUANYU_LOCAL_RANK" != "$_HUANYU_ARG_RANK" ]]; then
        echo 'FATAL: conflicting local rank values in LOCAL_RANK / launcher arguments' >&2
        exit 2
    fi
    _HUANYU_LOCAL_RANK="$_HUANYU_ARG_RANK"
    _HUANYU_HAS_LOCAL_RANK=1
done
set -- "${_HUANYU_NODE_ARGS[@]}"
if (( _HUANYU_HAS_LOCAL_RANK )); then
    [[ "$_HUANYU_LOCAL_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo 'FATAL: local rank must be a non-negative integer without leading zeroes' >&2; exit 2;
    }
    if [[ "$_HUANYU_LOCAL_RANK" != 0 ]]; then
        echo "[huanyu-job] local_rank=$_HUANYU_LOCAL_RANK node_rank=${VC_TASK_INDEX:-unknown} skipping duplicate node launcher; local_rank=0 owns setup/Ray"
        exit 0
    fi
    echo "[huanyu-job] local_rank=0 node_rank=${VC_TASK_INDEX:-unknown} selected node launcher pid=$$"
    # Ray/verl creates its own distributed workers. The outer launcher's ranks
    # must not make setup probes or the Ray driver look like a DDP worker.
    unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
    for _key in ${!TORCHELASTIC_@}; do unset "$_key"; done
fi

# Normal jobs opt in to extra native/device/cluster probes. Preparation always
# validates the environment; --start-ray also verifies registration through GCS.
# Keep the previous readiness switch as an alias; the new switch takes priority.
_PREFLIGHT="${HUANYU_JOB_PREFLIGHT:-${HUANYU_RAY_READINESS_CHECK:-0}}"
case "$_PREFLIGHT" in
    0|1) ;;
    *) echo 'FATAL: HUANYU_JOB_PREFLIGHT (or HUANYU_RAY_READINESS_CHECK) must be 0 or 1' >&2; exit 2 ;;
esac
_PREPARE_ONLY="${HUANYU_PREPARE_ONLY:-0}"
case "$_PREPARE_ONLY" in
    0) ;;
    1)
        [[ -r "${HUANYU_PREPARE_SNAPSHOT:-}" && -n "${HUANYU_PREPARED_ENV_FILE:-}" &&
           -d "${HUANYU_PREPARE_LOG_DIR:-}" ]] || {
            echo 'FATAL: use prepare_huanyu_env.sh for environment-only preparation' >&2
            exit 2
        }
        ;;
    *) echo 'FATAL: HUANYU_PREPARE_ONLY must be 0 or 1' >&2; exit 2 ;;
esac

_PREPARE_RAY=0
if (( _PREPARE_ONLY )); then
    _PREPARE_RAY="${HUANYU_PREPARE_START_RAY:-0}"
    [[ "$_PREPARE_RAY" == 0 || "$_PREPARE_RAY" == 1 ]] || { echo 'FATAL: invalid prepare Ray selection' >&2; exit 2; }
fi
_MANAGE_RAY=$(( ! _PREPARE_ONLY || _PREPARE_RAY ))

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
cd "$PROJECT_ROOT"
if [[ "${HUANYU_JOB_MODE:-frozen_cache}" == aqa_unfreeze ]]; then
    export TRAIN_FILE="${TRAIN_FILE:-${HUANYU_AQA_TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
    export VAL_FILE="${VAL_FILE:-${HUANYU_AQA_VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
elif [[ "${HUANYU_JOB_MODE:-}" == stem ]]; then
    # Explicit common paths take priority over STEM aliases and directory defaults.
    export STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
    export TRAIN_FILE="${TRAIN_FILE:-${STEM_TRAIN_FILE:-$STEM_DATA_DIR/train.parquet}}"
    if [[ -z "${VAL_FILE:-}" && -z "${STEM_VAL_FILE:-}" &&
          ! -f "$STEM_DATA_DIR/val.parquet" && -f "$STEM_DATA_DIR/validation.parquet" ]]; then
        export STEM_VAL_FILE="$STEM_DATA_DIR/validation.parquet"
    fi
    export VAL_FILE="${VAL_FILE:-${STEM_VAL_FILE:-$STEM_DATA_DIR/val.parquet}}"
fi
# Resolve deployment paths once. Site environment files may still contain old
# defaults; neither they nor downstream wrappers should replace caller paths.
declare -A _HUANYU_PATH_CONFIG=(
    [PROJECT_ROOT]="$PROJECT_ROOT"
    [MODEL_PATH]="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
    [RM_MODEL_PATH]="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}"
    [TRAIN_FILE]="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}"
    [VAL_FILE]="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}"
)
if (( _PREPARE_ONLY )); then
    for _key in DEFAULT_LOCAL_DIR TENSORBOARD_DIR GENRM_DUMP_DIR HUANYU_EMBED_CACHE_DIR RAY_ADDRESS; do
        if [[ -v "$_key" ]]; then _HUANYU_PATH_CONFIG["$_key"]="${!_key}"; fi
    done
fi
for _key in "${!_HUANYU_PATH_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PATH_CONFIG[$_key]}"
done
# The downstream training entry also invokes python3; keep installation and
# probes in that same interpreter environment.
_PYTHON=python3
_HELPER="${_HUANYU_EXAMPLE_DIR}/job_bootstrap.py"
_ENV_FILE="${HUANYU_ENV_FILE:-}"
_VLLM_SOURCE="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}}"
_RANK="${VC_TASK_INDEX:-}"
_WAIT="${RAY_CLUSTER_WAIT_TIMEOUT:-900}"
_INTERVAL="${RAY_CLUSTER_POLL_INTERVAL:-5}"
_PORT="${RAY_PORT:-6925}"
_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8960}"
_JOB_MODE="${HUANYU_JOB_MODE:-frozen_cache}"
# Resolve reward selection before pool validation and preserve it across site
# setup. An explicit public mode wins over the old per-entry switches.
_REWARD_MODE="${HUANYU_REWARD_MODE:-}"
if [[ -z "$_REWARD_MODE" ]]; then
    if [[ "$_JOB_MODE" == stem && -n "${STEM_REWARD_MODE:-}" ]]; then
        _REWARD_MODE="$STEM_REWARD_MODE"
    elif [[ -n "${RM_ENABLE:-}" ]]; then
        case "$RM_ENABLE" in
            0) _REWARD_MODE=rule_based ;;
            1) _REWARD_MODE=RM ;;
            *) echo 'FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
        esac
    elif [[ "$_JOB_MODE" == stem ]]; then
        _REWARD_MODE=rule_based
    else
        _REWARD_MODE=RM
    fi
fi
case "${_REWARD_MODE,,}" in
    rm|genrm) _REWARD_MODE=RM; _REWARD_ENABLED=1 ;;
    rule|rule_based|ruled_based) _REWARD_MODE=rule_based; _REWARD_ENABLED=0 ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based' >&2; exit 2 ;;
esac
export HUANYU_REWARD_MODE="$_REWARD_MODE" RM_ENABLE="$_REWARD_ENABLED"
_RAY_READINESS_CHECK="$_PREFLIGHT"
_CLUSTER_NODES="${HUANYU_CLUSTER_NODES:-2}"
if [[ "$_JOB_MODE" == stem ]]; then
    [[ "$_CLUSTER_NODES" == 1 || "$_CLUSTER_NODES" == 2 || "$_CLUSTER_NODES" == 8 || "$_CLUSTER_NODES" == 16 ]] || {
        echo 'FATAL: STEM jobs require HUANYU_CLUSTER_NODES=1, 2, 8 or 16 (16 NPUs per node)' >&2; exit 2;
    }
else
    [[ "$_CLUSTER_NODES" == 2 ]] || {
        echo 'FATAL: frozen_cache and aqa_unfreeze jobs require HUANYU_CLUSTER_NODES=2' >&2; exit 2;
    }
fi
# Defaults for direct invocation; a profile such as the 16K wrapper supplies
# these first. Snapshot before sourcing the site env so it cannot replace the
# caller's pool sizes. Physical Ray capacity remains 16 NPUs on each node.
declare -A _HUANYU_POOL_CONFIG=(
    [NUM_NPUS]="${NUM_NPUS:-16}" [NNODES]="${NNODES:-1}"
    [ROLLOUT_TP]="${ROLLOUT_TP:-4}" [RM_TP]="${RM_TP:-4}"
    [RM_N_GPUS_PER_NODE]="${RM_N_GPUS_PER_NODE:-16}" [RM_NNODES]="${RM_NNODES:-1}"
    [RM_SPLIT_NODES]="${RM_SPLIT_NODES:-1}"
)
if (( ! _REWARD_ENABLED )); then _HUANYU_POOL_CONFIG[RM_NNODES]=0; fi
for _key in NUM_NPUS NNODES ROLLOUT_TP RM_TP RM_N_GPUS_PER_NODE RM_NNODES; do
    if [[ "$_key" == RM_NNODES && "${_HUANYU_POOL_CONFIG[$_key]}" == 0 && "$_REWARD_ENABLED" == 0 ]]; then continue; fi
    if [[ ! "${_HUANYU_POOL_CONFIG[$_key]}" =~ ^[1-9][0-9]?$ ]]; then
        echo "FATAL: $_key must be a positive integer (without leading zeroes)" >&2
        exit 2
    fi
done
_ACTOR_PER_NODE=${_HUANYU_POOL_CONFIG[NUM_NPUS]}
_ACTOR_NODES=${_HUANYU_POOL_CONFIG[NNODES]}
_RM_PER_NODE=${_HUANYU_POOL_CONFIG[RM_N_GPUS_PER_NODE]}
_RM_NODES=${_HUANYU_POOL_CONFIG[RM_NNODES]}
if (( ! _PREPARE_ONLY )); then
if [[ "$_JOB_MODE" == stem ]]; then
    if (( _ACTOR_PER_NODE != 16 || _ACTOR_NODES + _RM_NODES > _CLUSTER_NODES ||
          _ACTOR_PER_NODE % ${_HUANYU_POOL_CONFIG[ROLLOUT_TP]} != 0 ||
          (_REWARD_ENABLED && (_RM_PER_NODE != 16 || _RM_NODES < 1 ||
           _RM_PER_NODE % ${_HUANYU_POOL_CONFIG[RM_TP]} != 0)) )); then
        echo 'FATAL: STEM requires 16-card node-local Actor/RM pools within the physical nodes, with each TP dividing 16' >&2
        exit 2
    fi
else
if (( _ACTOR_PER_NODE > 16 || _RM_PER_NODE > 16 || _ACTOR_NODES > 2 || _RM_NODES > 2 ||
      _ACTOR_PER_NODE * _ACTOR_NODES + _RM_PER_NODE * _RM_NODES > 32 ||
      _ACTOR_PER_NODE % ${_HUANYU_POOL_CONFIG[ROLLOUT_TP]} != 0 ||
      (_REWARD_ENABLED && (_RM_NODES < 1 || _RM_PER_NODE % ${_HUANYU_POOL_CONFIG[RM_TP]} != 0)) )); then
    echo 'FATAL: pool sizes must fit within 32 on two 16-card nodes; each enabled per-node pool must be divisible by its TP' >&2
    exit 2
fi
# Each per-node pool is STRICT_PACK. Check it can be packed into two physical
# nodes; checking only the total would accept impossible [12,12]+[8].
_POOLS_FIT=0
for (( _a=0; _a<=_ACTOR_NODES; _a++ )); do
    for (( _r=0; _r<=_RM_NODES; _r++ )); do
        if (( _a * _ACTOR_PER_NODE + _r * _RM_PER_NODE <= 16 &&
              (_ACTOR_NODES - _a) * _ACTOR_PER_NODE + (_RM_NODES - _r) * _RM_PER_NODE <= 16 )); then
            _POOLS_FIT=1
        fi
    done
done
if (( ! _POOLS_FIT )); then
    echo 'FATAL: per-node pools cannot fit on two 16-card nodes; for actor24 + RM8 use actor12x2 and RM4x2' >&2
    exit 2
fi
if [[ "$_JOB_MODE" == aqa_unfreeze ]] && (( _REWARD_ENABLED )) &&
    (( _ACTOR_PER_NODE != 16 || _ACTOR_NODES != 1 || _RM_PER_NODE != 16 || _RM_NODES != 1 )); then
    echo 'FATAL: AQA jobs require actor16 + RM16' >&2
    exit 2
fi
fi  # STEM assigns whole nodes; the two-node multimodal mode also permits smaller pools.
fi  # Pool packing is checked when launching the job, not during standalone setup.
case "$_JOB_MODE" in
    frozen_cache) _TRAIN_ENTRY=run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh ;;
    aqa_unfreeze) _TRAIN_ENTRY=run_huanyu_aqa_audio_train.sh ;;
    stem) _TRAIN_ENTRY=run_huanyu_stem.sh ;;
    *) echo 'FATAL: HUANYU_JOB_MODE must be frozen_cache, aqa_unfreeze or stem' >&2; exit 2 ;;
esac
# Keep the component choices through site environment setup and into existing
# Ray workers. The frozen-cache entry always freezes both audio components.
if [[ "$_JOB_MODE" == frozen_cache || "$_JOB_MODE" == stem ]]; then
    export HUANYU_FREEZE_AUDIO=true
    export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
fi
huanyu_resolve_audio_train_mode || exit $?
declare -A _HUANYU_AUDIO_CONFIG=()
for _key in HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR MINICPMO_TUNE_AUDIO; do
    _HUANYU_AUDIO_CONFIG["$_key"]="${!_key}"
done
[[ "$_PREPARE_ONLY" == 1 || -f "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" ]] || {
    echo "FATAL: missing training entry $PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" >&2
    exit 1
}
_REQUIRED_COMMANDS=("$_PYTHON" git pgrep flock)
if (( _MANAGE_RAY )); then _REQUIRED_COMMANDS+=(ray timeout); fi
for _command in "${_REQUIRED_COMMANDS[@]}"; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing command $_command" >&2; exit 1; }
done
if (( _MANAGE_RAY )); then
    if [[ ! "$_RANK" =~ ^(0|[1-9][0-9]?)$ ]] || (( _RANK >= _CLUSTER_NODES )); then
        echo "FATAL: set VC_TASK_INDEX=0..$((_CLUSTER_NODES - 1)); rank 0 is head; no implicit rank 0" >&2
        exit 2
    fi
    if [[ -n "${VC_WORKER_NUM:-}" && "$VC_WORKER_NUM" != "$_CLUSTER_NODES" ]]; then
        echo "FATAL: this launcher expects $_CLUSTER_NODES nodes, each with 16 NPUs" >&2
        exit 2
    fi
fi
for _key in _WAIT _INTERVAL _PORT _DASHBOARD_PORT; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: $_key must be a positive integer" >&2
        exit 2
    fi
done
if (( _PORT > 65535 || _DASHBOARD_PORT > 65535 || _PORT == _DASHBOARD_PORT )); then
    echo 'FATAL: Ray and dashboard ports must be distinct values in 1..65535' >&2
    exit 2
fi

# Only load preparation functions here; the standalone prepare entry is not run.
# Keep allocator setup before the first Python/node probe.
source "${_HUANYU_EXAMPLE_DIR}/prepare_huanyu_env.sh" --functions-only
huanyu_prepare_allocator

[[ -f "$_HELPER" && ( -z "$_ENV_FILE" || -f "$_ENV_FILE" ) ]] || {
    echo "FATAL: missing job_bootstrap.py or explicit HUANYU_ENV_FILE" >&2; exit 1;
}
echo "[huanyu-job] allocator before Python/env setup: LD_PRELOAD=$LD_PRELOAD"
if (( _MANAGE_RAY )); then
    _IPS="$("$_PYTHON" -B "$_HELPER" resolve "${VC_WORKER_HOSTS:-}" --expected-nodes "$_CLUSTER_NODES")"
    IFS=, read -r -a _NODE_IPS <<< "$_IPS"
    _MASTER_IP="${_NODE_IPS[0]}"
    _NODE_IP="${_NODE_IPS[$_RANK]}"
fi

# Filter only scheduler --NAME=value metadata; preserve Hydra argv boundaries.
_TRAIN_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --cfg=*|--package=*|--config-path=*|--config-name=*|--config-dir=*|--experimental-rerun=*|--info=*)
            _TRAIN_ARGS+=("$_arg") ;;
        --*=*) echo "[huanyu-job] ignoring scheduler argument ${_arg%%=*}" ;;
        *) _TRAIN_ARGS+=("$_arg") ;;
    esac
done

# Preserve exported training settings in every mode, including frozen_cache.
# Site setup provides defaults; it must not replace the caller's experiment.
declare -A _HUANYU_PROFILE_CONFIG=()
for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
        ROLLOUT_N ACTOR_MODEL_DTYPE REF_MODEL_DTYPE ACTOR_OFFLOAD_POLICY \
        ACTOR_FORWARD_PREFETCH ACTIVATION_OFFLOAD FILTER_OVERLONG \
        MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH ROLLOUT_PROMPT_LENGTH \
        ROLLOUT_MAX_MODEL_LEN ROLLOUT_MAX_NUM_BATCHED_TOKENS MAX_TOKEN_LEN_PER_GPU MINICPMO_MM_PROMPT_BUDGET AQA_DYNAMIC_BSZ \
        ROLLOUT_TEMPERATURE ROLLOUT_TOP_P ROLLOUT_TOP_K ROLLOUT_REPETITION_PENALTY \
        ROLLOUT_FREQUENCY_PENALTY ROLLOUT_PRESENCE_PENALTY ROLLOUT_LOGPROBS_MODE \
        ROLLOUT_GPU_MEM_UTIL HUANYU_GRAPH_MODE PROJECT_NAME EXPERIMENT_NAME TOTAL_TRAINING_STEPS \
        UPDATE_WEIGHTS_BUCKET_MB TOTAL_EPOCHS TRAINER_LOGGER SAVE_FREQ RESUME_MODE ENABLE_THINKING \
        DEFAULT_LOCAL_DIR TENSORBOARD_DIR GENRM_DUMP_DIR ROLLOUT_DATA_DIR VALIDATION_DATA_DIR WANDB_MODE \
        TEST_FREQ VAL_BEFORE_TRAIN ROLLOUT_MAX_NUM_SEQS ROLLOUT_NUM_WORKERS HUANYU_RUN_DIR HUANYU_RUN_BASE \
        HUANYU_EMBED_CACHE_DIR HUANYU_JOB_PROFILE HUANYU_REWARD_MODE Q36_BACKWARD_PREFETCH \
        HUANYU_STEM_TEXT_TRAIN HUANYU_CHECKPOINT_VALIDATION_MODE \
        HUANYU_FULL_WEIGHT_AUDIT_DIR HUANYU_FULL_WEIGHT_AUDIT_RUN_ID HUANYU_WEIGHT_SYNC_AUDIT_DIR \
        HUANYU_AUDIO_PROBE_DIR HUANYU_TEXT_AB_DIR HUANYU_EMBEDDING_AB_DIR HUANYU_EMBEDDING_AB_SIDE \
        ${!ACTOR_@} ${!REF_@} ${!ROLLOUT_@} ${!PPO_@} ${!RM_@} ${!STEM_@} ${!MAX_@} \
        ${!GENRM_@} ${!HUANYU_GEN_@} ${!HUANYU_FORWARD_@} ${!HUANYU_NPU_@}; do
    if [[ -v "$_key" ]]; then _HUANYU_PROFILE_CONFIG["$_key"]="${!_key}"; fi
done
if [[ "$_JOB_MODE" == aqa_unfreeze ]]; then
    _HUANYU_PROFILE_CONFIG[HUANYU_GRAPH_MODE]="${HUANYU_GRAPH_MODE:-0}"
    for _key in HUANYU_AQA_TRAIN_FILE HUANYU_AQA_VAL_FILE; do
        if [[ -v "$_key" ]]; then _HUANYU_PROFILE_CONFIG["$_key"]="${!_key}"; fi
    done
fi

# Own the numerical policy at this entry, even with an older/custom site env.
# The default full profile overrides stale per-kernel opt-outs. custom keeps
# explicit caller values for comparisons; site env must not replace the policy.
huanyu_configure_precision_mode "${_TRAIN_ARGS[@]}" || exit $?
declare -A _HUANYU_PRECISION_CONFIG=()
for _key in "${HUANYU_PRECISION_KEYS[@]}"; do
    _HUANYU_PRECISION_CONFIG["$_key"]="${!_key}"
done

if (( _PREPARE_ONLY )); then
    # The outer environment entry captures early errors as well as these steps.
    _LOG_DIR="$HUANYU_PREPARE_LOG_DIR"
else
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_sampleall_cache_32card}"
    _LOG_DIR="${HUANYU_JOB_LOG_DIR:-$PROJECT_ROOT/runs/$EXPERIMENT_NAME/cluster}"
    mkdir -p "$_LOG_DIR"
    exec > >(tee -a "$_LOG_DIR/rank_$_RANK.log") 2>&1
fi
# Local lock stays held by this launcher, never inherited by a detached raylet.
exec 9>"${TMPDIR:-/tmp}/huanyu-32card-${UID}.lock"
flock -n 9 || { echo 'FATAL: another Huanyu job launcher is already running on this node' >&2; exit 1; }
if (( _PREPARE_ONLY )); then
    echo "[huanyu-env] prepare mode=$_JOB_MODE start_ray=$_PREPARE_RAY log=$_LOG_DIR/prepare.log"
else
    echo "[huanyu-job] rank=$_RANK platform_ip=$_NODE_IP ray_node_ip=auto head=$_MASTER_IP:$_PORT log=$_LOG_DIR/rank_$_RANK.log"
fi

_LIVE_RAY=0
if pgrep -x raylet >/dev/null; then _LIVE_RAY=1; fi
export HUANYU_VLLM_SOURCE="$_VLLM_SOURCE"
huanyu_select_native_source || exit $?
_VLLM_SOURCE="$HUANYU_VLLM_SOURCE"
echo "[huanyu-job] expected installed vLLM-Omni=$_VLLM_SOURCE (no source PYTHONPATH injection)"
# All dependency/site setup lives in prepare_huanyu_env.sh. The job retains
# node/resource orchestration and restores the caller's settings afterwards.
export HUANYU_GRAPH_MODE="${HUANYU_GRAPH_MODE:-0}"
set +u
huanyu_prepare_node "$_LIVE_RAY" "$_ENV_FILE" "$@"
set +x
set -euo pipefail
# A custom/older site env may reset these to /vllm-workspace/vllm-omni.
# Restore the expected installed origin and remove old source-path overrides.
export HUANYU_VLLM_SOURCE="$_VLLM_SOURCE"
huanyu_select_native_source || exit $?
for _key in "${!_HUANYU_PATH_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PATH_CONFIG[$_key]}"
done
for _key in "${!_HUANYU_PROFILE_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PROFILE_CONFIG[$_key]}"
done
if [[ "$_JOB_MODE" == aqa_unfreeze || "$_JOB_MODE" == stem ]]; then
    # Only discard legacy site defaults. An identical path explicitly exported
    # by the caller is intentional and must reach the training backend.
    if [[ -z "${_HUANYU_PROFILE_CONFIG[DEFAULT_LOCAL_DIR]+set}" &&
          "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then
        unset DEFAULT_LOCAL_DIR
    fi
    if [[ -z "${_HUANYU_PROFILE_CONFIG[TENSORBOARD_DIR]+set}" &&
          "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then
        unset TENSORBOARD_DIR
    fi
fi
export Q36_CKPT_DIR="$MODEL_PATH"
cd "$PROJECT_ROOT"
huanyu_preload_allocator
echo "[huanyu-job] allocator for native check/Ray: LD_PRELOAD=$LD_PRELOAD"
export RAY_PORT="$_PORT" RAY_DASHBOARD_PORT="$_DASHBOARD_PORT"
export HUANYU_JOB_PREFLIGHT="$_PREFLIGHT"
export HUANYU_RAY_READINESS_CHECK="$_RAY_READINESS_CHECK"
if (( _MANAGE_RAY )); then export RAY_ADDRESS="$_MASTER_IP:$_PORT"; fi
export HUANYU_JOB_MODE="$_JOB_MODE" HUANYU_CLUSTER_NODES="$_CLUSTER_NODES"
export RAY_USAGE_STATS_ENABLED=0 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
# Apply after site settings, before ray start on every node. These control
# process startup, not the total AgentLoopWorker count. Existing raylets retain
# their environment; prepare snapshots also carry these values for manual starts.
export RAY_worker_maximum_startup_concurrency=32
export RAY_prestart_worker_first_driver=1
export RAY_enable_worker_prestart=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export MINICPMO_USE_PRECOMPUTE=0 MINICPMO_TUNE_VISION=0
export HUANYU_STEM_TEXT_TRAIN=0
for _key in "${!_HUANYU_AUDIO_CONFIG[@]}"; do
    export "$_key=${_HUANYU_AUDIO_CONFIG[$_key]}"
done
case "$_JOB_MODE" in
    frozen_cache)
        export HUANYU_USE_PRECOMPUTE=1 HUANYU_FREEZE_AUDIO=true
        export HUANYU_EMBED_CACHE_DIR="${HUANYU_EMBED_CACHE_DIR:-/opt/huawei/dataset/infra_data/data/embed_cache_huanyu_sampleall_q128}"
        export MINICPMO_TUNE_AUDIO=0 HUANYU_AQA_AUDIO_TRAIN=0
        [[ -r "$HUANYU_EMBED_CACHE_DIR/manifest.json" ]] || {
            echo "FATAL: cache manifest not readable on rank $_RANK: $HUANYU_EMBED_CACHE_DIR/manifest.json" >&2
            exit 1
        }
        ;;
    aqa_unfreeze)
        # Apply before Ray starts on BOTH nodes, after the site environment.
        # Empty the cache path as well as disabling it, including runtime_env
        # below for workers launched by an already-running frozen-cache raylet.
        export HUANYU_USE_PRECOMPUTE=0
        export HUANYU_EMBED_CACHE_DIR=""
        export HUANYU_AQA_AUDIO_TRAIN=1
        export Q36_BACKWARD_PREFETCH=off
        ;;
    stem)
        export HUANYU_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
        export HUANYU_AQA_AUDIO_TRAIN=0 HUANYU_STEM_TEXT_TRAIN=1
        export Q36_BACKWARD_PREFETCH=off
        ;;
esac
export DEBUG_MODE=0 DEBUG_SAMPLES=0
for _key in "${!_HUANYU_POOL_CONFIG[@]}"; do
    export "$_key=${_HUANYU_POOL_CONFIG[$_key]}"
done
export NPUS_PER_NODE=16
export HUANYU_REWARD_MODE="$_REWARD_MODE" RM_ENABLE="$_REWARD_ENABLED"
if (( ! _REWARD_ENABLED )); then
    export RM_ENABLE=0 RM_NNODES=0 REWARD_POOL_NPUS=0
else
    export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES)) RM_ENABLE=1
fi
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
export WANDB_MODE="${WANDB_MODE:-offline}"
for _key in "${!_HUANYU_PRECISION_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PRECISION_CONFIG[$_key]}"
done
# Source our helper again after site settings and the final pool/cache mode.
# This also prevents an older env file from selecting the Whisper engine hook.
huanyu_configure_precision_mode "${_TRAIN_ARGS[@]}" || exit $?
echo "[huanyu-job] precision before Ray: train_GDN=${MINICPMO_HUANYU_GDN_FP32_TRAINING} rollout_GDN=${MINICPMO_GDN_FP32_REFERENCE} reload=${MINICPMO_HUANYU_LAYERWISE_RELOAD} KV_zero=${MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE} isolate_media=${MINICPMO_ISOLATE_MEDIA_ENCODERS} actual_TP=${MINICPMO_HUANYU_ROLLOUT_TP}"
echo "[huanyu-job] inference hooks: ${VLLM_OMNI_EXTERNAL_MODULES}"
echo "[huanyu-job] mode=$_JOB_MODE reward=$HUANYU_REWARD_MODE freeze_audio=$HUANYU_FREEZE_AUDIO precompute=$HUANYU_USE_PRECOMPUTE train_entry=$_TRAIN_ENTRY"
echo "[huanyu-job] train_audio_encoder=$HUANYU_TRAIN_AUDIO_ENCODER train_audio_projector=$HUANYU_TRAIN_AUDIO_PROJECTOR"
echo "[huanyu-job] physical=16x${_CLUSTER_NODES} actor=${NUM_NPUS}x${NNODES} rollout_TP=$ROLLOUT_TP RM=${RM_N_GPUS_PER_NODE}x${RM_NNODES} RM_TP=$RM_TP requested=$((NUM_NPUS * NNODES + REWARD_POOL_NPUS))"
if (( _PREFLIGHT || _PREPARE_ONLY )); then
    "$_PYTHON" -B "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py" \
        --check --expected-root "$_VLLM_SOURCE" || exit $?
    echo '[huanyu-job] native import check exited successfully'
    "$_PYTHON" -B "$_HELPER" check-devices || exit $?
else
    echo '[huanyu-job] skipping native/device/cluster preflight (HUANYU_JOB_PREFLIGHT=0)'
fi

wait_for_cluster() {
    local -a options=()
    if (( _PREPARE_RAY )); then
        options+=(--gcs)
    fi
    local status=0
    timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-cluster \
        --address "$RAY_ADDRESS" --expected-ips "$_IPS" --local-ip "$_NODE_IP" \
        --timeout "$_WAIT" --interval "$_INTERVAL" --mode "$1" "${options[@]}" || status=$?
    if (( status )); then
        echo "FATAL: Ray readiness check failed/timed out (exit $status); training not started" >&2
        return "$status"
    fi
}

if (( _MANAGE_RAY )); then
    # Apply after site setup, matching the standalone Ray launcher on every node.
    unset HCCL_LOGIC_SUPERPOD_ID
    "$_PYTHON" -B "$_HELPER" clear-superpod-bashrc "${HOME:?HOME must be set}/.bashrc"
    echo "[huanyu-job] Ray rank=$_RANK address=$RAY_ADDRESS local_ip=$_NODE_IP"
    if (( _RANK == 0 )); then
        if (( ! _LIVE_RAY )); then
            ray start --head --port="$RAY_PORT" \
                --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" \
                --resources='{"NPU": 16}' 9>&-
        fi
        if (( _PREFLIGHT || _PREPARE_RAY )); then wait_for_cluster all || exit $?; fi
    else
        timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-head \
            --address "$RAY_ADDRESS" --timeout "$_WAIT" --interval "$_INTERVAL" || exit $?
        if (( ! _LIVE_RAY )); then
            echo '[huanyu-job] joining as worker; no training on this node'
            ray start --address="$RAY_ADDRESS" --resources='{"NPU": 16}' 9>&-
        fi
        if (( _PREPARE_RAY || (_PREFLIGHT && _LIVE_RAY) )); then wait_for_cluster local || exit $?; fi
    fi
fi

if (( _PREPARE_ONLY )); then
    if (( _LIVE_RAY )); then
        echo '[huanyu-env] existing Ray was left running; activation does not update its inherited environment'
    fi
    huanyu_publish_prepared_environment "${_HUANYU_EXAMPLE_DIR}"
    exit 0
fi

if (( _RANK == 0 )); then
    # Explicitly propagate cache settings and the allocator even when the Raylet
    # predates this job. Neither Huanyu nor LD_PRELOAD is in the shared ext
    # environment whitelist; Ray can also strip its own allocator preload.
    _RUNTIME_ARGS=()
    for _key in HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR HUANYU_FREEZE_AUDIO \
        HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
        HUANYU_AQA_AUDIO_TRAIN HUANYU_STEM_TEXT_TRAIN MINICPMO_USE_PRECOMPUTE MINICPMO_TUNE_AUDIO \
        MINICPMO_TUNE_VISION WANDB_MODE LD_PRELOAD HUANYU_REWARD_MODE; do
        _value="$("$_PYTHON" -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
        _RUNTIME_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=${_value}")
    done
    if [[ "$_RAY_READINESS_CHECK" == 1 ]]; then
        echo "[huanyu-job] all $_CLUSTER_NODES nodes registered ($((_CLUSTER_NODES * 16)) NPU); head starting $_JOB_MODE Huanyu training (preflight=$_PREFLIGHT)"
    else
        echo "[huanyu-job] head submitting $_JOB_MODE training to $RAY_ADDRESS (preflight=$_PREFLIGHT); expected capacity=$((_CLUSTER_NODES * 16)) NPU, not prechecked"
    fi
    bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" \
        "${_TRAIN_ARGS[@]}" \
        "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=${HUANYU_FREEZE_AUDIO}" \
        "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=${HUANYU_TRAIN_AUDIO_ENCODER}" \
        "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=${HUANYU_TRAIN_AUDIO_PROJECTOR}" \
        "${_RUNTIME_ARGS[@]}" "${HUANYU_PRECISION_ARGS[@]}"
else
    if (( _LIVE_RAY )); then
        if [[ "$_RAY_READINESS_CHECK" == 1 ]]; then
            echo '[huanyu-job] worker registered in the selected cluster; waiting, no training on this node'
        else
            echo '[huanyu-job] keeping existing worker raylet; cluster membership unchecked, no training on this node'
        fi
    fi
    # ray start returns after startup, just like the working manual command.
    # Keep the node job and its lock alive without creating a Ray probe driver.
    echo '[huanyu-job] worker launcher staying alive; monitoring local raylet only'
    while pgrep -x raylet >/dev/null; do sleep 30; done
    echo 'FATAL: worker raylet exited' >&2
    exit 1
fi
