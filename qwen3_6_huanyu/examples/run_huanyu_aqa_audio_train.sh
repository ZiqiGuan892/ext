#!/usr/bin/env bash
# Deploy this file AND check_aqa_audio_train.py to:
#   ${PROJECT_ROOT}/examples/qwen3_6_huanyu/
# Source the environment and start Ray on all nodes before running this entry.
# This wrapper reuses the existing Huanyu 32-card / OmniMix + GenRM backend.
set -euo pipefail

_HUANYU_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_HUANYU_EXAMPLE_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"
_HUANYU_MODEL_DIR="${PROJECT_ROOT}/verl_omni_ext/models/qwen3_6_huanyu"
_HUANYU_BACKEND="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_32card.sh"
for _huanyu_file in "${_HUANYU_MODEL_DIR}/runtime_imports.py" "${_HUANYU_BACKEND}" \
    "${_HUANYU_EXAMPLE_DIR}/check_aqa_audio_train.py"; do
    if [[ ! -f "${_huanyu_file}" ]]; then
        echo "[huanyu-aqa] FATAL: missing ${_huanyu_file}; deploy to PROJECT_ROOT/examples/qwen3_6_huanyu" >&2
        exit 2
    fi
done

# This is an online AQA-only mode. Cache flags cannot be inherited from a
# previous frozen-encoder run. The HF override is the actual model hyperparameter.
export HUANYU_FREEZE_AUDIO="${HUANYU_FREEZE_AUDIO:-false}"
case "${HUANYU_FREEZE_AUDIO}" in
    false) export MINICPMO_TUNE_AUDIO=1 ;;
    true) export MINICPMO_TUNE_AUDIO=0 ;;
    *) echo "[huanyu-aqa] FATAL: HUANYU_FREEZE_AUDIO must be true or false" >&2; exit 2 ;;
esac
export HUANYU_AQA_AUDIO_TRAIN=1
export HUANYU_USE_PRECOMPUTE=0
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_VISION=0
export Q36_BACKWARD_PREFETCH=off
export ACTOR_FORWARD_PREFETCH=false
export ACTOR_OFFLOAD_POLICY=false
export ACTIVATION_OFFLOAD=false
export FILTER_OVERLONG=false

export TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/train.parquet}"
export VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/validation.parquet}"
export MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B}"

# The existing OmniMix backend fixes actor/rollout=16 and reward=16 NPUs.
# One sequence per actor rank avoids retaining full unsharded gradients during
# FSDP2 no_sync accumulation. More train samples may form multiple mini-batches.
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 REWARD_POOL_NPUS=16
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
for _huanyu_integer in "${TRAIN_BATCH_SIZE}" "${PPO_MINI_BATCH_SIZE}" "${ROLLOUT_N}"; do
    if [[ ! "${_huanyu_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[huanyu-aqa] FATAL: batch sizes and ROLLOUT_N must be positive integers" >&2
        exit 2
    fi
done
if (( PPO_MINI_BATCH_SIZE * ROLLOUT_N != 16 || ROLLOUT_N < 2 ||
      PPO_MINI_BATCH_SIZE % 4 != 0 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "[huanyu-aqa] FATAL: require ppo_mini_batch*rollout.n=16, rollout.n>=2, ppo_mini_batch%4=0 and train_batch%ppo_mini_batch=0" >&2
    exit 2
fi

# DEBUG_MODE rewrites batching and may write shared mini_debug parquet files.
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
_huanyu_steps="${TOTAL_TRAINING_STEPS:--1}"
if [[ "${_huanyu_steps}" == -1 || "${_huanyu_steps}" == null ]]; then
    _huanyu_steps=null
elif [[ ! "${_huanyu_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[huanyu-aqa] FATAL: TOTAL_TRAINING_STEPS must be a positive integer, -1 or null" >&2
    exit 2
fi
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-audio_freeze_${HUANYU_FREEZE_AUDIO}_online}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/tensorboard}"
export GENRM_DUMP_DIR="${GENRM_DUMP_DIR:-${PROJECT_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/genrm_dump}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

# Check all rows without decoding audio; a validation split containing mixed
# OmniMix rows must not enter this mode even when val_before_train=false.
python3 -B "${_HUANYU_EXAMPLE_DIR}/check_aqa_audio_train.py" \
    --train "${TRAIN_FILE}" --validation "${VAL_FILE}" --train-batch-size "${TRAIN_BATCH_SIZE}"

export HUANYU_VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
_huanyu_retained_path=""
IFS=: read -r -a _huanyu_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|"${HUANYU_VLLM_OMNI_ROOT}"|"${PROJECT_ROOT}"|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit) continue ;;
    esac
    _huanyu_retained_path+="${_huanyu_retained_path:+:}${_huanyu_path}"
done
export PYTHONPATH="${HUANYU_VLLM_OMNI_ROOT}:${PROJECT_ROOT}${_huanyu_retained_path:+:${_huanyu_retained_path}}"
python3 -B "${_HUANYU_MODEL_DIR}/runtime_imports.py" --check --expected-root "${HUANYU_VLLM_OMNI_ROOT}"

_huanyu_ray_args=()
for _huanyu_env_key in PYTHONPATH HUANYU_VLLM_OMNI_ROOT HUANYU_FREEZE_AUDIO \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_USE_PRECOMPUTE MINICPMO_USE_PRECOMPUTE \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH; do
    _huanyu_env_json="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "${_huanyu_env_key}")"
    _huanyu_ray_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_huanyu_env_key}=${_huanyu_env_json}")
done

echo "[huanyu-aqa] freeze_audio=${HUANYU_FREEZE_AUDIO} cache=off visual=frozen; actor=16NPU RM=16NPU"
echo "[huanyu-aqa] train_batch=${TRAIN_BATCH_SIZE} ppo_mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} static_microbatch=1"
echo "[huanyu-aqa] prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} steps=${_huanyu_steps}"
echo "[huanyu-aqa] forward_prefetch=off Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH}; HCCL timeout is unchanged"

# Required settings follow user arguments so the checked datasets and execution
# contract cannot be bypassed by stale command-line options. Model selection is
# the verl alias; the checkpoint's HF architecture and RM alias remain intact.
exec bash "${_HUANYU_BACKEND}" "$@" \
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
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=false \
    'actor_rollout_ref.model.exclude_modules=.*model\.visual.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*' \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.offload_policy=false \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    reward.reward_model.n_gpus_per_node=16 \
    reward.reward_model.nnodes=1 \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.total_training_steps="${_huanyu_steps}" \
    "${_huanyu_ray_args[@]}"
