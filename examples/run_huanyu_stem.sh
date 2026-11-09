#!/usr/bin/env bash
# STEM text-only training on an already prepared Ray cluster.
# Run ONCE on the head node. No environment setup, installs or Ray start/stop.
# Direct defaults: Actor16 + GenRM16, TP4; frozen media, dynamic token batches.
# HUANYU_REWARD_MODE=rule_based selects math_dapo; RM selects the GenRM model.
# run_huanyu_stem_job.sh manages 16/32/128/256 physical cards (rule default).
# Defaults: no initial validation; engine-default gradient synchronization.
# Lengths are token LIMITS (1K=1024), including the rendered chat prompt:
#   bash run_huanyu_stem.sh                           # 2048 + 3072 = 5K
#   STEM_CONTEXT_LENGTH=12k bash run_huanyu_stem.sh   # 2048 + 10240
#   STEM_CONTEXT_LENGTH=16k STEM_PROMPT_LENGTH=4096 bash run_huanyu_stem.sh
#   TOTAL_TRAINING_STEPS=3 STEM_IGNORE_EOS=true bash run_huanyu_stem.sh
# STEM_DATA_DIR contains train.parquet and val.parquet (converter defaults).
# Override individual files with STEM_TRAIN_FILE / STEM_VAL_FILE.
# MODEL_PATH / RM_MODEL_PATH override checkpoints; other Hydra args are accepted.
# This entry owns data, length, pools, media freeze and microbatch settings;
# change those through the variables above, not conflicting Hydra overrides.
# Rollout max_num_batched_tokens, dynamic batching and sync_each_microbatch
# defaults accept Hydra overrides. The rollout token budget defaults to the
# engine context length; e.g. actor_rollout_ref.rollout.max_num_batched_tokens=16384.
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

}
# END HUANYU output policy
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


export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_STEM_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_qwen36_huanyu_gspo_npu.sh run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh; do
    [[ -f "$_STEM_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_STEM_EXAMPLES/$_file" >&2; exit 2; }
done

_stem_tokens() {
    local value="${1,,}" factor=1
    if [[ "$value" == *k ]]; then value="${value%k}"; factor=1024; fi
    [[ "$value" =~ ^[1-9][0-9]{0,6}$ ]] || {
        echo "FATAL: invalid token budget '$1'; use a positive integer or e.g. 12k" >&2; return 2;
    }
    printf '%s' "$((value * factor))"
}
_STEM_CONTEXT="$(_stem_tokens "${STEM_CONTEXT_LENGTH:-5k}")"
_STEM_PROMPT="$(_stem_tokens "${MAX_PROMPT_LENGTH:-${STEM_PROMPT_LENGTH:-2048}}")"
export MAX_PROMPT_LENGTH="$_STEM_PROMPT"
if [[ -n "${MAX_RESPONSE_LENGTH:-}" ]]; then
    MAX_RESPONSE_LENGTH="$(_stem_tokens "$MAX_RESPONSE_LENGTH")"
    _STEM_CONTEXT=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
(( MAX_PROMPT_LENGTH < _STEM_CONTEXT )) || {
    echo 'FATAL: STEM_PROMPT_LENGTH must be less than STEM_CONTEXT_LENGTH' >&2; exit 2;
}
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((_STEM_CONTEXT - MAX_PROMPT_LENGTH))}"
_STEM_CONTEXT=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ROLLOUT_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MINICPMO_MM_PROMPT_BUDGET="$MAX_PROMPT_LENGTH"
_STEM_SLACK="${STEM_CONTEXT_SLACK:-128}"
[[ "$_STEM_SLACK" =~ ^(0|[1-9][0-9]{0,4})$ ]] || { echo 'FATAL: invalid STEM_CONTEXT_SLACK' >&2; exit 2; }
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((_STEM_CONTEXT + _STEM_SLACK))}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}"
(( ROLLOUT_MAX_MODEL_LEN >= _STEM_CONTEXT )) || {
    echo "FATAL: ROLLOUT_MAX_MODEL_LEN must cover prompt + response" >&2; exit 2;
}
_STEM_REWARD_SELECTION="${HUANYU_REWARD_MODE:-${STEM_REWARD_MODE:-}}"
if [[ -z "$_STEM_REWARD_SELECTION" ]]; then
    case "${RM_ENABLE:-1}" in
        0) _STEM_REWARD_SELECTION=rule_based ;;
        1) _STEM_REWARD_SELECTION=RM ;;
        *) echo 'FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
    esac
fi
case "${_STEM_REWARD_SELECTION,,}" in
    rm|genrm) export HUANYU_REWARD_MODE=RM STEM_REWARD_MODE=genrm RM_ENABLE=1 ;;
    rule_based|ruled_based|rule) export HUANYU_REWARD_MODE=rule_based STEM_REWARD_MODE=rule RM_ENABLE=0 ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based; STEM_REWARD_MODE accepts genrm or rule' >&2; exit 2 ;;
esac
_STEM_IGNORE_EOS="${STEM_IGNORE_EOS:-false}"
case "$_STEM_IGNORE_EOS" in true|false) ;; *) echo 'FATAL: STEM_IGNORE_EOS must be true or false' >&2; exit 2 ;; esac
_STEM_STEPS="${TOTAL_TRAINING_STEPS:-null}"
case "$_STEM_STEPS" in -1|null) _STEM_STEPS=null ;; *)
    [[ "$_STEM_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: TOTAL_TRAINING_STEPS must be positive, -1 or null' >&2; exit 2; } ;;
esac

# Explicit generic paths win; STEM aliases supply mode-specific defaults.
_STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
_STEM_TRAIN="${TRAIN_FILE:-${STEM_TRAIN_FILE:-$_STEM_DATA_DIR/train.parquet}}"
_STEM_VAL="${VAL_FILE:-${STEM_VAL_FILE:-$_STEM_DATA_DIR/val.parquet}}"
if [[ -z "${VAL_FILE:-${STEM_VAL_FILE:-}}" && ! -f "$_STEM_VAL" && -f "$_STEM_DATA_DIR/validation.parquet" ]]; then
    _STEM_VAL="$_STEM_DATA_DIR/validation.parquet"
fi
for _file in "$_STEM_TRAIN" "$_STEM_VAL"; do
    [[ -r "$_file" ]] || { echo "FATAL: unreadable STEM parquet: $_file" >&2; exit 2; }
done
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
[[ "$RAY_ADDRESS" != local ]] || { echo 'FATAL: use the existing Ray cluster (auto or head_ip:port)' >&2; exit 2; }
_STEM_RAY_CHECK="${HUANYU_RAY_READINESS_CHECK:-1}"
case "$_STEM_RAY_CHECK" in
    0|1) ;;
    *) echo 'FATAL: HUANYU_RAY_READINESS_CHECK must be 0 or 1' >&2; exit 2 ;;
esac
huanyu_prepare_output
if [[ "$_STEM_RAY_CHECK" == 1 ]]; then
    _STEM_RAY_TIMEOUT="${HUANYU_RAY_STATUS_TIMEOUT:-30}"
    [[ "$_STEM_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: HUANYU_RAY_STATUS_TIMEOUT must be positive' >&2; exit 2; }
    if ! timeout --foreground "${_STEM_RAY_TIMEOUT}s" ray status --address="$RAY_ADDRESS"; then
        echo 'FATAL: existing Ray is not reachable; prepare the requested Ray nodes first' >&2; exit 1
    fi
else
    echo '[huanyu-stem] Ray status precheck skipped (HUANYU_RAY_READINESS_CHECK=0); trainer will connect to Ray'
fi

# GenRM reads the question from extra_info, not prompt. Keep original columns
# and metadata, adding question to per-run copies on the shared project mount.
# The source parquets are never overwritten; all workers need this run directory.
if [[ "$STEM_REWARD_MODE" == rule ]]; then
    # Import/test the actual installed scorer before allocating model memory.
    python3 - <<'PY'
import inspect
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo
def score(text):
    result = default_compute_score(data_source="math_dapo", solution_str=text, ground_truth="4")
    return float(result["score"] if isinstance(result, dict) else result)
good, bad = score(r"\boxed{4}"), score(r"\boxed{5}")
if not good > bad:
    raise RuntimeError("math_dapo rule scorer smoke check failed: correct=%s wrong=%s" % (good, bad))
print("[huanyu-stem] rule=math_dapo source=%s correct_score=%s wrong_score=%s"
      % (inspect.getsourcefile(math_dapo), good, bad), flush=True)
PY
fi
python3 - "$_STEM_TRAIN" "$_STEM_VAL" "$HUANYU_RUN_DIR/stem_data" "$STEM_REWARD_MODE" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

destination = Path(sys.argv[3])
destination.mkdir(parents=True, exist_ok=True)
for split, source in zip(("train", "validation"), sys.argv[1:3]):
    table = pq.read_table(source)
    missing = {"data_source", "prompt", "reward_model", "extra_info"} - set(table.column_names)
    media = {"images", "videos", "audios"} & set(table.column_names)
    if missing or media or not table.num_rows:
        raise ValueError("%s: missing=%s media=%s rows=%s; expected nonempty text-only STEM data"
                         % (source, sorted(missing), sorted(media), table.num_rows))
    if sys.argv[4] == "rule" and set(table.column("data_source").to_pylist()) != {"math_dapo"}:
        raise ValueError("%s: rule mode requires data_source=math_dapo for every row" % source)
    metadata = []
    for index, row in enumerate(table.select(["prompt", "reward_model", "extra_info"]).to_pylist()):
        prompt, reward, extra = row["prompt"], row["reward_model"], row["extra_info"]
        if (not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict)
                or prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str)
                or not prompt[0]["content"].strip() or not isinstance(reward, dict)
                or not isinstance(reward.get("ground_truth"), str) or not reward["ground_truth"].strip()
                or not isinstance(extra, dict)):
            raise ValueError("%s row %d: expected one user text prompt, string ground_truth and extra_info" % (source, index))
        metadata.append(dict(extra, question=prompt[0]["content"]))
    old_field = table.schema.field("extra_info")
    fields = [field for field in old_field.type if field.name != "question"]
    extra_type = pa.struct(fields + [pa.field("question", pa.string())])
    table = table.set_column(table.schema.get_field_index("extra_info"),
                             pa.field("extra_info", extra_type, metadata=old_field.metadata),
                             pa.array(metadata, type=extra_type))
    output = destination / (split + ".parquet")
    pq.write_table(table, output)
    print("[huanyu-stem] split=%s rows=%d source=%s prepared=%s question=prompt[0].content"
          % (split, table.num_rows, source, output), flush=True)
PY
export TRAIN_FILE="$HUANYU_RUN_DIR/stem_data/train.parquet" VAL_FILE="$HUANYU_RUN_DIR/stem_data/validation.parquet"

# Pool dimensions describe Actor and RM independently, not the physical job.
export NUM_NPUS="${NUM_NPUS:-16}" NNODES="${NNODES:-1}" ROLLOUT_TP="${ROLLOUT_TP:-4}"
for _key in NUM_NPUS NNODES ROLLOUT_TP; do
    [[ "${!_key}" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
done
if [[ "$RM_ENABLE" == 1 ]]; then
    export RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}" RM_NNODES="${RM_NNODES:-1}" RM_TP="${RM_TP:-4}" RM_SPLIT_NODES="${RM_SPLIT_NODES:-1}"
    for _key in RM_N_GPUS_PER_NODE RM_NNODES RM_TP; do
        [[ "${!_key}" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
    done
    (( RM_N_GPUS_PER_NODE % RM_TP == 0 )) || { echo 'FATAL: RM_TP must divide the per-node reward pool' >&2; exit 2; }
    export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
else
    export REWARD_POOL_NPUS=0 RM_NNODES=0 RM_N_GPUS_PER_NODE=16 RM_TP=4 RM_SPLIT_NODES=0
fi
if [[ "${STEM_SCALED_JOB:-0}" == 1 ]]; then
    [[ "$NUM_NPUS" == 16 && "${HUANYU_CLUSTER_NODES:-}" =~ ^(1|2|8|16)$ ]] || {
        echo 'FATAL: scaled STEM requires 1/2/8/16 physical nodes of 16 NPUs; use run_huanyu_stem_job.sh' >&2; exit 2;
    }
    (( NUM_NPUS * NNODES + REWARD_POOL_NPUS <= HUANYU_CLUSTER_NODES * 16 )) || {
        echo 'FATAL: Actor + RM pools exceed the physical STEM cluster' >&2; exit 2;
    }
    for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N ROLLOUT_TP ROLLOUT_MAX_NUM_SEQS; do
        [[ "${!_key:-}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
    done
    (( 16 % ROLLOUT_TP == 0 && TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 &&
       PPO_MINI_BATCH_SIZE * ROLLOUT_N % (NUM_NPUS * NNODES) == 0 )) || {
        echo 'FATAL: TP must divide 16; batch must divide into minibatches; trajectories must divide Actor DP' >&2; exit 2;
    }
else
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}" ROLLOUT_N="${ROLLOUT_N:-4}" ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
fi
_STEM_ACTOR_WORLD=$((NUM_NPUS * NNODES))
_STEM_FSDP_SIZE=-1
if (( _STEM_ACTOR_WORLD == 32 )); then
    # Hybrid sharding: two replicas with 16 ranks per shard group.
    _STEM_FSDP_SIZE=16
fi
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
[[ "$PPO_MICRO_BATCH_SIZE_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo 'FATAL: PPO_MICRO_BATCH_SIZE_PER_GPU must be positive' >&2; exit 2;
}
_STEM_DYNAMIC="${STEM_DYNAMIC_BSZ:-true}"
_STEM_SYNC="${STEM_SYNC_EACH_MICROBATCH:-false}"
for _flag in "$_STEM_DYNAMIC" "$_STEM_SYNC"; do
    case "$_flag" in true|false) ;; *) echo 'FATAL: STEM batching/sync flags must be true or false' >&2; exit 2 ;; esac
done
if [[ "${ACTOR_DYNAMIC_BSZ:-$_STEM_DYNAMIC}" == false ]] &&
   (( PPO_MINI_BATCH_SIZE * ROLLOUT_N % (_STEM_ACTOR_WORLD * PPO_MICRO_BATCH_SIZE_PER_GPU) != 0 )); then
    echo 'FATAL: static PPO trajectories must divide into per-rank microbatches' >&2; exit 2
fi
export DEBUG_MODE=0 HUANYU_SKIP_EDITABLE_INSTALL=1
export HUANYU_PRECISION_PROFILE="${HUANYU_PRECISION_PROFILE:-full}" HUANYU_GRAPH_MODE="${HUANYU_GRAPH_MODE:-0}"
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_AQA_AUDIO_TRAIN=0 HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
# Pure text has identical RoPE axes. Transport their common 1D positions so
# TQ/verl dynamic batching does not confuse the four axes with token lengths.
export HUANYU_STEM_TEXT_TRAIN=1
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0 Q36_BACKWARD_PREFETCH=off
export FILTER_OVERLONG=true WANDB_MODE="${WANDB_MODE:-offline}"
# Legacy GenRM defaults. Rule mode explicitly disables its resource pool and
# custom scorer below; the parquet style alone does not select the reward path.
export GENRM_SCORING_MODE=verdict GENRM_FALLBACK_ON_ERROR=zero
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_stem_${_STEM_CONTEXT}_16_16}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
huanyu_configure_graph_mode
huanyu_configure_precision_mode "$@" "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"

# Explicit runtime_env also reaches workers spawned by an older Ray daemon.
_STEM_ENV_ARGS=()
for _key in HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_STEM_TEXT_TRAIN HUANYU_USE_PRECOMPUTE MINICPMO_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH MINICPMO_MM_PROMPT_BUDGET \
    GENRM_SCORING_MODE GENRM_FALLBACK_ON_ERROR GENRM_DUMP_DIR WANDB_MODE HUANYU_REWARD_MODE; do
    _value="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
    _STEM_ENV_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=$_value")
done
echo "[huanyu-stem] actor=${NUM_NPUS}x${NNODES} actor_DP=$_STEM_ACTOR_WORLD fsdp_size=$_STEM_FSDP_SIZE train_TP=1 SP=1 rollout_TP=$ROLLOUT_TP replicas=$((_STEM_ACTOR_WORLD / ROLLOUT_TP)) reward=$STEM_REWARD_MODE RM_cards=$REWARD_POOL_NPUS"
echo "[huanyu-stem] BS=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE N=$ROLLOUT_N trajectories=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) microbatch=$PPO_MICRO_BATCH_SIZE_PER_GPU microbatches_per_rank=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N / (NUM_NPUS * NNODES * PPO_MICRO_BATCH_SIZE_PER_GPU)))"
echo "[huanyu-stem] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$_STEM_CONTEXT engine=$ROLLOUT_MAX_MODEL_LEN ignore_eos=$_STEM_IGNORE_EOS steps=$_STEM_STEPS"
echo "[huanyu-stem] text_only=true encoder_train=false projector_train=false cache=off precision=$HUANYU_PRECISION_PROFILE graph=$HUANYU_GRAPH_MODE reward=$STEM_REWARD_MODE"
echo "[huanyu-stem] HUANYU_STEM_TEXT_TRAIN=$HUANYU_STEM_TEXT_TRAIN position_transport=1d model_positions=4_axes"
echo "[huanyu-stem] defaults: val_before_train=${VAL_BEFORE_TRAIN:-false} test_freq=${TEST_FREQ:-10} actor/rollout/ref.dynamic_bsz=$_STEM_DYNAMIC token_budget=$MAX_TOKEN_LEN_PER_GPU sync_each_microbatch=$_STEM_SYNC (trailing Hydra overrides take precedence)"
_STEM_REWARD_ARGS=(reward.reward_model.enable=true reward.reward_model.enable_resource_pool=true
    "reward.reward_model.n_gpus_per_node=$RM_N_GPUS_PER_NODE" "reward.reward_model.nnodes=$RM_NNODES"
    "reward.reward_model.rollout.tensor_model_parallel_size=$RM_TP")
if [[ "$STEM_REWARD_MODE" == rule ]]; then
    _STEM_REWARD_ARGS=(reward.reward_model.enable=false reward.reward_model.enable_resource_pool=false
        reward.custom_reward_function.path=null reward.reward_manager.source=register reward.reward_manager.name=naive)
fi
_STEM_ROLLOUT_ARGS=()
if [[ "${STEM_SCALED_JOB:-0}" == 1 ]]; then
    # The backend derives a small worker count from replicas. Preserve the
    # explicit producer concurrency so larger jobs can feed all engines.
    _STEM_ROLLOUT_ARGS=("actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS:-128}"
        actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.pipeline_model_parallel_size=1)
fi

# These defaults precede user arguments; length and resource constraints follow.
_STEM_BACKEND=run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
if [[ "$STEM_REWARD_MODE" == rule ]]; then _STEM_BACKEND=run_qwen36_huanyu_gspo_npu.sh; fi
exec bash "$_STEM_EXAMPLES/$_STEM_BACKEND" \
    "trainer.logger=${TRAINER_LOGGER:-[console]}" "trainer.project_name=${PROJECT_NAME:-huanyu_stem}" \
    "trainer.total_training_steps=$_STEM_STEPS" \
    "trainer.test_freq=${TEST_FREQ:-10}" "trainer.val_before_train=${VAL_BEFORE_TRAIN:-false}" \
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_DYNAMIC_BSZ:-$_STEM_DYNAMIC}" \
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-$_STEM_DYNAMIC}" \
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_DYNAMIC_BSZ:-$_STEM_DYNAMIC}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}" \
    "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=$_STEM_SYNC" \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}" \
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR:-$HUANYU_RUN_DIR/validation_dump}" "$@" \
    "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE" "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=8 \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH" \
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.ignore_eos=$_STEM_IGNORE_EOS" \
    ++actor_rollout_ref.model.override_config.huanyu_freeze_audio=true \
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=false \
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=false \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=$_STEM_FSDP_SIZE" \
    "actor_rollout_ref.ref.fsdp_config.fsdp_size=$_STEM_FSDP_SIZE" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS" actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    "${_STEM_REWARD_ARGS[@]}" "${_STEM_ROLLOUT_ARGS[@]}" \
    "trainer.n_gpus_per_node=$NUM_NPUS" "trainer.nnodes=$NNODES" "${_STEM_ENV_ARGS[@]}"
