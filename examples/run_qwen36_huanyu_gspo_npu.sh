#!/usr/bin/env bash
# Final Huanyu training backend. Reuse an already prepared Ray cluster.
# Owns model routing, native-source checks, logging, graph/precision policy,
# preflight checks and the final Hydra command. See ../README.md for parameters.
set -eo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$PROJECT_ROOT" || exit $?
EXT_ROOT="${PROJECT_ROOT}"

# BEGIN HUANYU output policy
huanyu_init_output() {
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
huanyu_init_output || exit $?
# END HUANYU output policy
# BEGIN HUANYU native policy
huanyu_select_native_source() {
    local root="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}}"
    local explicit="${1:-}" old_source="${HUANYU_VLLM_SOURCE:-}" old_root="${HUANYU_VLLM_OMNI_ROOT:-}"
    local entry retained="" required
    local -a entries=()
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
        [[ -f "$entry/vllm_omni/__init__.py" ]] && continue
        retained+="${retained:+:}$entry"
    done
    export HUANYU_VLLM_SOURCE="$root" HUANYU_VLLM_OMNI_ROOT="$root"
    if [[ "$explicit" == --explicit-source ]]; then
        export PYTHONPATH="$root${retained:+:$retained}"
    elif [[ -n "$retained" ]]; then
        export PYTHONPATH="$retained"
    else
        unset PYTHONPATH
    fi
}
huanyu_select_native_source || exit $?
# END HUANYU native policy

set -x
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

    HUANYU_GRAPH_ARGS=(
        "actor_rollout_ref.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "reward.reward_model.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_GRAPH_MODE=\"${HUANYU_GRAPH_MODE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_TORCH_COMPILE=\"${VLLM_TORCH_COMPILE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_COMPILE_DISABLE=\"${TORCH_COMPILE_DISABLE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.NPU_GRAPH_CAPTURE_DISABLE=\"${NPU_GRAPH_CAPTURE_DISABLE}\""
    )
}

huanyu_configure_graph_mode
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
    HUANYU_PRECISION_KEYS=(HUANYU_PRECISION_PROFILE "${_on[@]}" "${_full[@]}" "${_off[@]}" \
        MINICPMO_FSDP2_PIN_MEMORY MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE)
    export HUANYU_JOB_DETERMINISM="${HUANYU_JOB_DETERMINISM:-0}"
    case "$HUANYU_JOB_DETERMINISM" in
        0|1) ;;
        *) echo '[huanyu-precision] FATAL: HUANYU_JOB_DETERMINISM must be 0 or 1' >&2; return 2 ;;
    esac
    HUANYU_PRECISION_KEYS+=(HUANYU_JOB_DETERMINISM)
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
        HUANYU_PRECISION_ARGS+=(
            'actor_rollout_ref.rollout.full_determinism=false'
            '++reward.reward_model.rollout.full_determinism=false'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enable_batch_invariant=false'
        )
    fi
    HUANYU_PRECISION_ARGS+=(
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls=verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
    )
    for _key in "${HUANYU_PRECISION_KEYS[@]}" \
        MINICPMO_HUANYU_ROLLOUT_TP VLLM_OMNI_EXTERNAL_MODULES HUANYU_USE_PRECOMPUTE \
        HCCL_OP_EXPANSION_MODE HCCL_DETERMINISTIC; do
        _value="${!_key:-}"
        if [[ "$_key" == HCCL_* && -z "$_value" ]]; then continue; fi
        _value="${_value//\\/\\\\}"
        _value="${_value//\"/\\\"}"
        HUANYU_PRECISION_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=\"${_value}\"")
    done
    if (( ${#_changed[@]} )); then
        echo "[huanyu-precision] profile=full overrides inherited values: ${_changed[*]}"
    fi
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

huanyu_configure_precision_mode "$@"
# END HUANYU precision policy

echo "[huanyu-graph] HUANYU_GRAPH_MODE=${HUANYU_GRAPH_MODE} actor/RM.enforce_eager=${HUANYU_ENFORCE_EAGER} VLLM_TORCH_COMPILE=${VLLM_TORCH_COMPILE} TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE} NPU_GRAPH_CAPTURE_DISABLE=${NPU_GRAPH_CAPTURE_DISABLE}"
export PYTHONPATH="${EXT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export FILTER_OVERLONG="${FILTER_OVERLONG:-false}"
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_AUDIO="${MINICPMO_TUNE_AUDIO:-0}"
_huanyu_ray_pythonpath="$(python3 -c 'import json, os; print(json.dumps(os.environ["PYTHONPATH"]))')"
_huanyu_ray_native_root="$(python3 -c 'import json, os; print(json.dumps(os.environ["HUANYU_VLLM_OMNI_ROOT"]))')"
_HUANYU_MODEL_ARGS=(
    actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu
    actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining
    actor_rollout_ref.model.model_stage=thinker
    actor_rollout_ref.model.use_remove_padding=false
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${_huanyu_ray_pythonpath}"
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_VLLM_OMNI_ROOT=${_huanyu_ray_native_root}"
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_VLLM_SOURCE=${_huanyu_ray_native_root}"
)
echo "[huanyu] adapter=Qwen36HuanyuForTraining dataset=Qwen36HuanyuOmniRLHFDataset pipeline=qwen3_6_huanyu; freeze_audio=${HUANYU_FREEZE_AUDIO:-true} precompute=${HUANYU_USE_PRECOMPUTE:-0} length_filter=${FILTER_OVERLONG}"
case "${HUANYU_SKIP_EDITABLE_INSTALL:-0}" in
    0)
        git config --global --add safe.directory "${EXT_ROOT}" || exit $?
        python3 -m pip install -e "${EXT_ROOT}" --no-deps --no-build-isolation || exit $?
        ;;
    1) echo '[huanyu-setup] using installed plugin; editable installation skipped' ;;
    *) echo 'FATAL: HUANYU_SKIP_EDITABLE_INSTALL must be 0 or 1' >&2; exit 2 ;;
esac
python3 -B "${EXT_ROOT}/verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py" --check

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
export VERL_OMNI_EXT_PATCH_STRICT=1       # L2 补丁未打上直接 raise（改动 C）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export USE_OPTIMIZED_MODEL=0
export ASCEND_SECURITY_CHECK_DISABLED=1
export WANDB_MODE=${WANDB_MODE:-offline}

export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export VERL_USE_EXTERNAL_MODULES="verl_omni,verl_omni_ext"

export MINICPMO_FSDP2_PIN_MEMORY=${MINICPMO_FSDP2_PIN_MEMORY:-0}  # 0=fsdp pin 补丁生效，1=恢复 verl 原样
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-false}

export Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH:-off}  # off=不预取（默认，防死锁），on=torch 默认

export MINICPMO_GDN_FP32_REFERENCE=${MINICPMO_GDN_FP32_REFERENCE:-1}
export MINICPMO_GDN_STATE_BF16=${MINICPMO_GDN_STATE_BF16:-1}  # legacy A/B path when FP32_REFERENCE=0

export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1

unset PYTORCH_NPU_ALLOC_CONF

export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-3600}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

echo "[env] HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT} ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT} TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT} HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT}"
echo "[env] HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE} HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-<unset>} MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-0} MINICPMO_HANG_DUMP_DIR=${MINICPMO_HANG_DUMP_DIR:-<unset>}"
echo "[env] MINICPMO_DISABLE_IMAGE_EXPAND=${MINICPMO_DISABLE_IMAGE_EXPAND:-<unset>} MINICPMO_DISABLE_TRAIN_PROMPT_EXPAND=${MINICPMO_DISABLE_TRAIN_PROMPT_EXPAND:-<unset>} MINICPMO_IMAGE_SPAN_DROP_SENTINELS=${MINICPMO_IMAGE_SPAN_DROP_SENTINELS:-<unset>} MINICPMO_MM_ALIGN_STRICT=${MINICPMO_MM_ALIGN_STRICT:-<unset>} MINICPMO_ROLLOUT_DIAG=${MINICPMO_ROLLOUT_DIAG:-1} MINICPMO_GDN_STATE_BF16=${MINICPMO_GDN_STATE_BF16:-<unset>}"

export GENRM_FALLBACK_ON_ERROR=${GENRM_FALLBACK_ON_ERROR:-similarity}
echo "[env] GENRM_FALLBACK_ON_ERROR=${GENRM_FALLBACK_ON_ERROR}"

SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

set -eo pipefail

MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/model_gy/downloaded_models/2748"}

TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/infra_data/data/train.parquet"}
VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/infra_data/data/validation.parquet"}

for _f in "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [ ! -f "${_f}" ]; then
        echo "FATAL: 数据文件不存在: ${_f}"
        echo "       跑 ${EXT_ROOT}/examples/minicpmo_5_0/prepare_gsm8k_text_only.py 生成，或用 TRAIN_FILE=/VAL_FILE= 指定。"
        exit 1
    fi
    if ! python3 -c "
import sys
import pyarrow.parquet as pq
cols = set(pq.read_schema(sys.argv[1]).names)
need = {'prompt', 'data_source', 'reward_model'}
if not need <= cols:
    print('FATAL: %s 不是 verl RL 格式。' % sys.argv[1])
    print('       缺列: %s' % sorted(need - cols))
    print('       实际列: %s' % sorted(cols))
    if {'question', 'answer'} <= cols:
        print('       -> 这是**原始** GSM8K。跑 prepare_gsm8k_text_only.py 转换后再训。')
    print('       硬要跳过本检查: SKIP_DATASET_SCHEMA_CHECK=1')
    sys.exit(1)
" "${_f}"; then
        [ "${SKIP_DATASET_SCHEMA_CHECK:-0}" = "1" ] || exit 1
        echo "WARN: SKIP_DATASET_SCHEMA_CHECK=1，跳过 ${_f} 的 schema 校验。"
    fi
done

ROLLOUT_TP=${ROLLOUT_TP:-4}

_DETECTED_NPUS=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
NUM_NPUS=${NUM_NPUS:-${_DETECTED_NPUS:-8}}
if [ -n "${NNODES:-}" ]; then
    _NNODES_SRC="显式 NNODES"
elif [ -n "${VC_WORKER_NUM:-}" ]; then
    NNODES=${VC_WORKER_NUM}
    _NNODES_SRC="VC_WORKER_NUM"
elif [ -n "${MA_NUM_HOSTS:-}" ]; then
    NNODES=${MA_NUM_HOSTS}
    _NNODES_SRC="MA_NUM_HOSTS(平台注入，通常与『本机实际跑几个节点』无关)"
else
    NNODES=1
    _NNODES_SRC="默认 1"
fi
if [ "${NNODES}" -gt 1 ] && [ "${_NNODES_SRC}" != "显式 NNODES" ]; then
    echo "WARNING: NNODES=${NNODES} 是从 ${_NNODES_SRC} 推出来的，不是显式设置。" >&2
    echo "         单机多卡请显式 export NNODES=1；否则 global pool = NUM_NPUS x NNODES 会翻倍，" >&2
    echo "         叠加 reward pool 后超出集群卡数（真机形态：desired 48 > available 32）。" >&2
fi

if [ "${NUM_NPUS}" -le 0 ] 2>/dev/null; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not a positive integer."
    echo "       /dev/davinci* probe found '${_DETECTED_NPUS}'. Set NUM_NPUS explicitly."
    exit 1
fi
if [ "${_DETECTED_NPUS}" -gt 0 ] && [ "${NUM_NPUS}" -gt "${_DETECTED_NPUS}" ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} exceeds ${_DETECTED_NPUS} devices visible on this node."
    echo "       This is the aclInit 107001 failure (ranks ${_DETECTED_NPUS}..$((NUM_NPUS-1)))."
    echo "       Use NUM_NPUS=${_DETECTED_NPUS} on a single node, or raise NNODES for multi-node."
    exit 1
fi
if [ $((NUM_NPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not divisible by ROLLOUT_TP=${ROLLOUT_TP};"
    echo "       rollout.agent.num_workers (replica count) would be wrong."
    exit 1
fi
ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS:-$((NUM_NPUS * NNODES / ROLLOUT_TP))}
echo "Cluster: NUM_NPUS=${NUM_NPUS} (detected ${_DETECTED_NPUS}) NNODES=${NNODES}" \
     "ROLLOUT_TP=${ROLLOUT_TP} replicas=$((NUM_NPUS * NNODES / ROLLOUT_TP)) agent_workers=${ROLLOUT_NUM_WORKERS} IF=${SOCKET_IFNAME}"

if [ "${HUANYU_RAY_READINESS_CHECK:-1}" = 0 ]; then
    echo '[huanyu-train] Ray capacity precheck skipped (HUANYU_RAY_READINESS_CHECK=0)'
elif command -v ray >/dev/null 2>&1; then
    _ray_npu=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ NPU' | head -n1 | sed 's#.*/##; s# NPU##' || true)
    if [ -n "${_ray_npu:-}" ]; then
        _ray_npu_int=$(printf '%.0f' "${_ray_npu}")
        REWARD_POOL_NPUS=${REWARD_POOL_NPUS:-0}
        _want=$(( NUM_NPUS * NNODES + REWARD_POOL_NPUS ))
        echo "Ray cluster reports ${_ray_npu_int} NPU; this job wants ${_want}" \
             "= global(${NUM_NPUS}x${NNODES}) + reward(${REWARD_POOL_NPUS})."
        if [ "${_ray_npu_int}" -lt "${_want}" ]; then
            echo "FATAL: Ray cluster only has ${_ray_npu_int} NPU but the job needs ${_want}," >&2
            echo "       of which global pool takes ${NUM_NPUS}x${NNODES} and the reward pool" >&2
            echo "       (enable_resource_pool=true) takes ${REWARD_POOL_NPUS}." >&2
            echo "       先查 NNODES 是否被平台注入的 MA_NUM_HOSTS/VC_WORKER_NUM 顶大了" >&2
            echo "       (单机请 export NNODES=1)，再考虑重启 Ray 或缩小 reward pool。" >&2
            exit 1
        fi
    fi
fi

for _v in MODEL_PATH TRAIN_FILE VAL_FILE; do
    _p="${!_v}"
    if [ -z "${_p}" ]; then
        echo "FATAL: ${_v} is empty. Set it explicitly."
        exit 1
    fi
    if [ ! -e "${_p}" ]; then
        echo "FATAL: ${_v} does not exist: ${_p}"
        exit 1
    fi
done

ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-fp32}

REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bfloat16}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}

ROLLOUT_N=${ROLLOUT_N:-8}

_WORLD_SIZE=$(( NUM_NPUS * NNODES ))
_SEQS_PER_RANK=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N / _WORLD_SIZE ))
if [ "${_SEQS_PER_RANK}" -lt 1 ]; then
    echo "FATAL: PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} × rollout.n=${ROLLOUT_N}" \
         "= $(( PPO_MINI_BATCH_SIZE * ROLLOUT_N )) < world_size=${_WORLD_SIZE};" >&2
    echo "       每卡分不到一条序列。请调大 PPO_MINI_BATCH_SIZE。" >&2
    exit 1
fi
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-${_SEQS_PER_RANK}}
_NUM_MICRO_BATCHES=$(( _SEQS_PER_RANK / PPO_MICRO_BATCH_SIZE_PER_GPU ))
echo "Actor batching: ${_SEQS_PER_RANK} seq/rank / micro_bsz=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
     "=> ${_NUM_MICRO_BATCHES} micro-batch(es)"
_HUANYU_SYNC_EACH=false
for _arg in "$@"; do
    _sync_arg="${_arg#++}"; _sync_arg="${_sync_arg#+}"
    case "$_sync_arg" in
        actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=*)
            _HUANYU_SYNC_EACH="${_sync_arg#*=}" ;;
    esac
done
if [[ "$_HUANYU_SYNC_EACH" == true ]]; then
    echo '[huanyu-memory] requested sync_each_microbatch=true; check runtime enabled log for sharded gradient synchronization'
fi
if [ "${_NUM_MICRO_BATCHES}" -gt 1 ] && [[ "$_HUANYU_SYNC_EACH" != true ]]; then
    echo "WARNING: micro-batch 数 > 1，会触发未分片 fp32 梯度囤积（约 3.0 GiB/层 × 40 层）。" >&2
    echo "         除非你清楚自己在做什么，否则 OOM 几乎是必然的。" >&2
fi
if [ "${_SEQS_PER_RANK}" -gt 4 ] && [[ "$_HUANYU_SYNC_EACH" != true ]]; then
    echo "WARNING: 每卡 ${_SEQS_PER_RANK} 条序列。这一版 verl 在 ${_WORLD_SIZE} 卡下**没有安全解**：" >&2
    echo "         micro_bsz 设小 -> 多 micro-batch -> 未分片梯度囤积 OOM；" >&2
    echo "         micro_bsz 设大 -> 单次前反传激活/logits 撑爆 -> 同样 OOM。" >&2
    echo "         唯一出路是加卡（抬 NNODES）或减小 PPO_MINI_BATCH_SIZE / ROLLOUT_N。" >&2
fi

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8448}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
_SEQ_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))

ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$(( _SEQ_LEN + 128 ))}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-$(( MAX_PROMPT_LENGTH + 32 ))}

export MINICPMO_MM_PROMPT_BUDGET="${MINICPMO_MM_PROMPT_BUDGET:-$ROLLOUT_PROMPT_LENGTH}"

MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$(( _SEQ_LEN * _SEQS_PER_RANK ))}

echo "Seq len: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} L=${_SEQ_LEN}" \
     "| rollout max_model_len=${ROLLOUT_MAX_MODEL_LEN} (batched_tokens 同值) | token budget/gpu=${MAX_TOKEN_LEN_PER_GPU}" \
     "| 媒体预算=${MINICPMO_MM_PROMPT_BUDGET}"

RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}

ENABLE_THINKING=${ENABLE_THINKING:-1}
if [ "${ENABLE_THINKING}" = "1" ]; then
    _ENABLE_THINKING_PY=True
    echo "enable_thinking = True   -> prompt 以未闭合的 <think>\\n 结尾，模型先推理再作答"
    echo "                            （与不传该 kwarg 渲染相同；这一行是防御，防的是别处传 False）"
else
    _ENABLE_THINKING_PY=False
    echo "enable_thinking = False  -> ⚠️ prompt 以**已闭合的空 <think></think>** 结尾，"
    echo "                            模型将跳过推理直接作答。"
    echo "                            这条路只适合 V3 的『能不能跑起来』验证（response 更短、更快），"
    echo "                            **不要用于正式的 Thinker-Only 训练**。"
fi
_CHAT_TEMPLATE_ARGS=( "++data.apply_chat_template_kwargs.enable_thinking=${_ENABLE_THINKING_PY}" )

if [ "${_ENABLE_THINKING_PY}" = "True" ] && [ "${MAX_RESPONSE_LENGTH}" -lt 1024 ]; then
    echo "⚠️  [warn] enable_thinking=True 但 MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} < 1024：" >&2
    echo "          CoT 极可能在 \boxed{} 之前被硬截断（实测 512 时整段复读、无 </think>）。" >&2
    echo "          除非你在做刻意的截断实验，否则请把预算抬到 2048 或关闭 thinking。" >&2
fi

# Huanyu always uses its multimodal dataset; fixed routing is appended last.

PROJECT_NAME=${PROJECT_NAME:-qwen3_6_whisper_gspo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_qwen3_6_whisper_thinker_npu}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${EXT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${EXT_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${TENSORBOARD_DIR}" "${DEFAULT_LOCAL_DIR}"

set +x
echo "=== 前置自检 ==="
python3 - <<'PY'
import os
import sys
from importlib.metadata import entry_points

assert any(
    ep.name == "qwen3_6_whisper"
    and ep.value == "verl_omni_ext.models.qwen3_6_whisper"
    for ep in entry_points(group="verl_omni.models")
), "Qwen3.6 plugin entry point missing; install this checkout with python3 -m pip install -e . --no-deps --no-build-isolation"

import verl_omni_ext  # 本脚本已 export VERL_USE_EXTERNAL_MODULES 含 verl_omni_ext → 自动 activate()
from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase

key = ("Qwen3_5MoeForConditionalGeneration", "thinker")
assert key in OmniModelBase._registry, (
    f"Adapter {key} not registered! 检查 pyproject.toml entry-points 与 verl_omni_ext/__init__.py _load_all()"
)
print("✓ 训练侧适配器注册:", key)

assert "qwen3_6_whisper" in OmniRolloutPipelineBase._registry, "Rollout adapter qwen3_6_whisper not registered!"
print("✓ rollout 适配器注册: qwen3_6_whisper")

from verl_omni_ext._patchkit import self_check, patch_state_line, verify_patches_alive

res = self_check()
for name, ok in res.items():
    assert ok, f"L2 patch {name!r} not applied!"
print("✓ L2 patches applied:", list(res) or "(无 L2 补丁)")
print(patch_state_line())
assert all(verify_patches_alive().values()), "Some L2 patch lost at runtime!"
print("✓ L2 patches alive")
os._exit(0)
PY
set -x
echo "=== 自检通过 ==="

huanyu_configure_precision_mode "$@" || exit $?
echo "[huanyu-precision] training_GDN=${MINICPMO_HUANYU_GDN_FP32_TRAINING} rollout_GDN=${MINICPMO_GDN_FP32_REFERENCE} layerwise_reload=${MINICPMO_HUANYU_LAYERWISE_RELOAD} actual_TP=${MINICPMO_HUANYU_ROLLOUT_TP} cache=${HUANYU_USE_PRECOMPUTE}"
# -1/null uses total_epochs; a positive exported limit applies to every entry.
_HUANYU_TOTAL_STEPS="${TOTAL_TRAINING_STEPS:-null}"
case "$_HUANYU_TOTAL_STEPS" in
    -1|null) _HUANYU_TOTAL_STEPS=null ;;
    *) [[ "$_HUANYU_TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]] || {
        echo 'FATAL: TOTAL_TRAINING_STEPS must be positive, -1 or null' >&2; exit 2;
    } ;;
esac
echo "[huanyu-tracking] WANDB_MODE=${WANDB_MODE:-offline} forwarded to Ray (explicit runtime-env overrides take precedence)"
python3 -m verl_omni.trainer.main_omni \
    trainer.device=npu \
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE=\"${WANDB_MODE:-offline}\"" \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.image_key=images \
    data.shuffle=true \
    data.seed=42 \
    data.filter_overlong_prompts=${FILTER_OVERLONG} \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    "${_CHAT_TEMPLATE_ARGS[@]}" \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni_ext \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD:-false} \
    actor_rollout_ref.model.exclude_modules=".*model\.visual.*|.*model\.audio_encoder.*|.*model\.audio_projection.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.trainer_type=policy_gradient \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
    actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY:-0.1} \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_DYNAMIC_BSZ:-true} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-true} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${ACTOR_FORWARD_PREFETCH:-false} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE:-bfloat16} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY:-true} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-true} \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=${REF_FORWARD_PREFETCH:-true} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE:-bfloat16} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_DYNAMIC_BSZ:-true} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:-20} \
    ++actor_rollout_ref.rollout.repetition_penalty=${ROLLOUT_REPETITION_PENALTY:-1.05} \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=${ROLLOUT_FREQUENCY_PENALTY:-0.0} \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=${ROLLOUT_PRESENCE_PENALTY:-0.0} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-4} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN:-4096}} \
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH:-4160} \
    actor_rollout_ref.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-true} \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-true} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB:-4096} \
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_whisper \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    algorithm.adv_estimator=grpo \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    trainer.v1.trainer_mode=omni_sync \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-false} \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.n_gpus_per_node=${NUM_NPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=$([[ "${DEBUG_MODE:-0}" == "1" ]] && echo -1 || echo "${SAVE_FREQ:-50}") \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.total_epochs=${TOTAL_EPOCHS:-10} \
    "trainer.total_training_steps=${_HUANYU_TOTAL_STEPS}" \
    "trainer.logger=${TRAINER_LOGGER:-[console]}" \
    "$@" \
    "${_HUANYU_MODEL_ARGS[@]}" \
    "${HUANYU_GRAPH_ARGS[@]}" \
    "${HUANYU_PRECISION_ARGS[@]}"
