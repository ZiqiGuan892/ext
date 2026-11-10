#!/usr/bin/env bash
# OmniMix + independent GenRM pool. Defaults: Actor16 + GenRM16, TP4.
# Adds dataset/reward settings, then invokes the single Huanyu training backend.
# Requires a prepared environment and running Ray; never installs packages or
# sources CANN/ATB setup. Run/source prepare_huanyu_env.sh first, or use a job entry.
# Caller Hydra overrides follow these defaults; Huanyu model/precision policy
# is enforced by the final backend. See ../README.md.
set -eo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$PROJECT_ROOT" || exit $?

export FILTER_OVERLONG="${FILTER_OVERLONG:-false}"
export ROLLOUT_FREQUENCY_PENALTY="${ROLLOUT_FREQUENCY_PENALTY:-0.0}"
_REWARD_MODE="${HUANYU_REWARD_MODE:-}"
if [[ -z "$_REWARD_MODE" ]]; then
    case "${RM_ENABLE:-1}" in
        0) _REWARD_MODE=rule_based ;;
        1) _REWARD_MODE=RM ;;
        *) echo 'FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
    esac
fi
case "${_REWARD_MODE,,}" in
    rm|genrm) export HUANYU_REWARD_MODE=RM RM_ENABLE=1 ;;
    rule|rule_based|ruled_based) export HUANYU_REWARD_MODE=rule_based RM_ENABLE=0 ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based' >&2; exit 2 ;;
esac
export RM_MODEL_PATH=${RM_MODEL_PATH:-"/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B"}

export NNODES="${NNODES:-1}"
export NUM_NPUS="${NUM_NPUS:-16}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export RM_TP="${RM_TP:-4}"
export RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}"
export RM_NNODES="${RM_NNODES:-1}"
if [[ "$RM_ENABLE" == 0 ]]; then export RM_NNODES=0; fi
export RM_SPLIT_NODES="${RM_SPLIT_NODES:-1}"
export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
export RM_GPU_MEM_UTIL=${RM_GPU_MEM_UTIL:-0.7}
export RM_RESPONSE_LENGTH=${RM_RESPONSE_LENGTH:-1024}
export RM_MAX_MODEL_LEN=${RM_MAX_MODEL_LEN:-32768}
export RM_PROMPT_LENGTH=${RM_PROMPT_LENGTH:-$((RM_MAX_MODEL_LEN - RM_RESPONSE_LENGTH))}
export RM_NUM_REWARD_WORKERS=${RM_NUM_REWARD_WORKERS:-8}

export TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/infra_data/data/train.parquet"}
export VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/infra_data/data/validation.parquet"}

export MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/model_gy/downloaded_models/2748"}

export MINICPMO_USE_PRECOMPUTE=${MINICPMO_USE_PRECOMPUTE:-0}
export MINICPMO_EMBED_CACHE_DIR=${MINICPMO_EMBED_CACHE_DIR:-/opt/huawei/dataset/ccb_data/data/omni/embed_cache_qwen36_sampleall}
export MINICPMO_VIDEO_MAX_FRAMES=${MINICPMO_VIDEO_MAX_FRAMES:-8}
export MINICPMO_VIDEO_FPS=1.0

export MINICPMO_ENABLE_MM_FORWARD=1
export MINICPMO_ALLOW_PACKED_MM_FORWARD=1
export MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-0}
export MINICPMO_FSDP2_PIN_MEMORY=0
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-false}

export MINICPMO_DISABLE_MM_DEDUP=1

export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
export REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bfloat16}"

export MINICPMO_USE_CUSTOM_DATASET=1
export VLLM_ASCEND_WEIGHT_NZ_MODE=0
export ASCEND_WEIGHT_NZ_MODE=0
export ASCEND_ENABLE_TRANS_DATA_CACHE=0

export DEBUG_MODE=${DEBUG_MODE:-0}
if [ "${DEBUG_MODE}" = "1" ]; then
    export TRAIN_BATCH_SIZE=32
    export ROLLOUT_N=2
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
    export MAX_PROMPT_LENGTH=3072
    export TOTAL_TRAINING_STEPS=2
    export PPO_MINI_BATCH_SIZE=32
    export DEBUG_SAMPLES=${DEBUG_SAMPLES:-0}
    if [ "${DEBUG_SAMPLES}" -gt 0 ] 2>/dev/null; then
        mkdir -p "${PROJECT_ROOT}/runs"
        _MINI_DIR="$(mktemp -d "${PROJECT_ROOT}/runs/huanyu_debug_data.XXXXXX")"
        _MINI_TRAIN="${_MINI_DIR}/mini_train_${DEBUG_SAMPLES}.parquet"
        _MINI_VAL="${_MINI_DIR}/mini_val_${DEBUG_SAMPLES}.parquet"
        python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${TRAIN_FILE}')
mini = t.slice(0, ${DEBUG_SAMPLES})
pq.write_table(mini, '${_MINI_TRAIN}')
v = pq.read_table('${VAL_FILE}')
mini_v = v.slice(0, max(1, ${DEBUG_SAMPLES}//4))
pq.write_table(mini_v, '${_MINI_VAL}')
print(f'[debug] mini parquet: {${DEBUG_SAMPLES}} train + {max(1, ${DEBUG_SAMPLES}//4)} val')
"
        export TRAIN_FILE=${_MINI_TRAIN}
        export VAL_FILE=${_MINI_VAL}
        export FILTER_OVERLONG=${FILTER_OVERLONG:-false}
        _WS=$((NUM_NPUS * NNODES))
        _TP=${ROLLOUT_TP}
        _bs=${DEBUG_SAMPLES}
        _bs=$((_bs / _TP * _TP))           # 向下取整到 TP 的倍数
        _bs=$((_bs < _WS ? _bs < _TP ? _TP : _TP * (_bs / _TP) : _bs))
        if [ "$_bs" -lt "$_TP" ]; then _bs=$_TP; fi
        export TRAIN_BATCH_SIZE=$_bs
        export PPO_MINI_BATCH_SIZE=$_bs
        export ROLLOUT_N=$((_WS / _bs * (_bs / _TP)))
        if [ "${ROLLOUT_N}" -lt 1 ]; then export ROLLOUT_N=1; fi
    fi
    echo "[debug] DEBUG_MODE=1: batch=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}, resp_len=${MAX_RESPONSE_LENGTH}, prompt_len=${MAX_PROMPT_LENGTH}, steps=${TOTAL_TRAINING_STEPS}, samples=${DEBUG_SAMPLES}"
else
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
    export ROLLOUT_N=${ROLLOUT_N:-4}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:--1}
    echo "[train] Full dataset: batch=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}, mini_batch=${PPO_MINI_BATCH_SIZE}, prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, steps=${TOTAL_TRAINING_STEPS}"
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_omni_mix_npu_genrm_qwen36}
export PROJECT_NAME=${PROJECT_NAME:-omni_mix_qwen36}

_TRAIN_TOKEN_BUDGET=${MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_MAX_MODEL_LEN:-$(( ${MAX_PROMPT_LENGTH:-3072} + ${MAX_RESPONSE_LENGTH:-2048} + 128 ))}}
_OMNI_MIX_ARGS=(
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset
    ++data.video_key=videos
    ++data.audio_key=audios
    data.filter_overlong_prompts=${FILTER_OVERLONG:-true}
    ++data.mm_processor_kwargs.sampling_rate=16000
    ++data.mm_processor_kwargs.max_slice_nums=1
    ++data.mm_processor_kwargs.use_image_id=false
    +actor_rollout_ref.model.override_config.stream_input=false
    actor_rollout_ref.model.override_config.attn_implementation=eager
    actor_rollout_ref.model.use_remove_padding=false
    actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD:-false}
    actor_rollout_ref.model.exclude_modules=".*model\.visual.*|.*model\.audio_encoder.*|.*model\.audio_projection.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*"
    actor_rollout_ref.actor.freeze_vision_tower=true
    actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_DYNAMIC_BSZ:-true}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${_TRAIN_TOKEN_BUDGET}
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false}
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY}
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']"
    actor_rollout_ref.actor.entropy_checkpointing=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-true}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${_TRAIN_TOKEN_BUDGET}
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB:-1024}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_DYNAMIC_BSZ:-true}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${_TRAIN_TOKEN_BUDGET}
    actor_rollout_ref.ref.entropy_checkpointing=False
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=False
)

if [[ "$RM_ENABLE" == 1 ]]; then
_REWARD_ARGS=(
    reward.custom_reward_function.path=pkg://verl_omni_ext.reward.genrm_verifier
    reward.custom_reward_function.name=compute_score
    reward.reward_model.enable=True
    reward.reward_model.enable_resource_pool=True
    "reward.reward_model.model_path=${RM_MODEL_PATH}"
    reward.reward_model.n_gpus_per_node=${RM_N_GPUS_PER_NODE}
    reward.reward_model.nnodes=${RM_NNODES}
    reward.reward_model.rollout.name=vllm_omni
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.output_mode=ar
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_5_moe
    reward.reward_model.rollout.tensor_model_parallel_size=${RM_TP}
    reward.reward_model.rollout.gpu_memory_utilization=${RM_GPU_MEM_UTIL}
    reward.reward_model.rollout.free_cache_engine=False
    reward.reward_model.rollout.prompt_length=${RM_PROMPT_LENGTH}
    reward.reward_model.rollout.response_length=${RM_RESPONSE_LENGTH}
    reward.reward_model.rollout.max_model_len=${RM_MAX_MODEL_LEN}
    reward.reward_model.rollout.max_num_batched_tokens=${RM_MAX_NUM_BATCHED_TOKENS:-$RM_MAX_MODEL_LEN}
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true
    reward.reward_model.rollout.max_num_seqs=${RM_MAX_NUM_SEQS:-64}
    reward.reward_model.rollout.dtype=bfloat16
    reward.reward_model.rollout.enforce_eager=true
    reward.reward_model.rollout.enable_prefix_caching=false
    reward.reward_model.rollout.enable_chunked_prefill=False
    reward.reward_model.rollout.skip_tokenizer_init=False
    ++reward.reward_model.rollout.full_determinism=false
    ++reward.reward_model.rollout.seed=42
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
)
else
# Native rule dispatch uses each parquet row's data_source and ground_truth.
# AQA and STEM entries supply their task-specific rules after these defaults.
_REWARD_ARGS=(
    reward.reward_model.enable=False
    reward.reward_model.enable_resource_pool=False
    reward.custom_reward_function.path=null
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
)
fi

if [[ "${RM_ENABLE}" == "1" ]]; then
    _RM_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
    if (( RM_N_GPUS_PER_NODE % RM_TP != 0 )); then
        echo "[FATAL] RM_N_GPUS_PER_NODE=${RM_N_GPUS_PER_NODE} must be divisible by RM_TP=${RM_TP}" >&2
        exit 2
    fi
    if (( _RM_NPUS < RM_TP )); then
        echo "[FATAL] judge total NPUs=${_RM_NPUS} < RM_TP=${RM_TP}" >&2
        exit 2
    fi
    echo "[genrm] RM_ENABLE=1, judge=${_RM_NPUS}NPU (TP=${RM_TP}, replicas=$((_RM_NPUS / RM_TP)))"
    echo "[genrm] actor_pool=${NUM_NPUS}x${NNODES} reward_pool=${RM_N_GPUS_PER_NODE}x${RM_NNODES}; RM_SPLIT_NODES=${RM_SPLIT_NODES} (layout label only; actual placement comes from the pool sizes)"
fi

echo "=== Qwen3.6-Huanyu OmniMix reward=${HUANYU_REWARD_MODE} ==="
echo "train      = ${TRAIN_FILE}"
echo "val        = ${VAL_FILE}"
echo "model      = ${MODEL_PATH}"
echo "judge      = ${RM_MODEL_PATH}"
echo "cache_dir  = ${MINICPMO_EMBED_CACHE_DIR}"
echo "debug      = DEBUG_MODE=${DEBUG_MODE}"
echo

exec bash "${_HERE}/run_qwen36_huanyu_gspo_npu.sh" "${_OMNI_MIX_ARGS[@]}" "${_REWARD_ARGS[@]}" "$@"
