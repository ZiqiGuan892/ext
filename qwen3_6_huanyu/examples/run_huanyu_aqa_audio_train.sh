#!/usr/bin/env bash
# Copy the contents of this examples directory to:
#   ${PROJECT_ROOT}/examples/qwen3_6_huanyu/
# Source the environment and start Ray on all nodes before running this entry.
# RM_ENABLE=1 retains the 32-card GenRM path; RM_ENABLE=0 uses 16 cards and rules.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/hfy-model-gy/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
cd "${PROJECT_ROOT}"
_HUANYU_MODEL_DIR="${PROJECT_ROOT}/verl_omni_ext/models/qwen3_6_huanyu"
source "${_HUANYU_EXAMPLE_DIR}/run_output.sh" || exit $?
export RM_ENABLE="${RM_ENABLE:-1}"
case "${RM_ENABLE}" in
    1)
        _HUANYU_BACKEND="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_32card.sh"
        export REWARD_POOL_NPUS=16
        _huanyu_reward_args=(
            reward.reward_model.n_gpus_per_node=16
            reward.reward_model.nnodes=1
        )
        ;;
    0)
        # The GenRM wrapper unconditionally re-enables RM. Bypass it entirely.
        _HUANYU_BACKEND="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_gspo_npu.sh"
        export REWARD_POOL_NPUS=0
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
for _huanyu_file in "${_HUANYU_EXAMPLE_DIR}/check_native_runtime.py" "${_HUANYU_BACKEND}" \
    "${_HUANYU_EXAMPLE_DIR}/check_aqa_audio_train.py" "${_HUANYU_EXAMPLE_DIR}/precision_mode.sh"; do
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
export HUANYU_EMBED_CACHE_DIR=""
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_VISION=0
export Q36_BACKWARD_PREFETCH=off
export ACTOR_FORWARD_PREFETCH=false
export ACTOR_OFFLOAD_POLICY=false
export ACTIVATION_OFFLOAD=false
export FILTER_OVERLONG=false

# Upgrade the generic defaults inherited from env_export_huanyu.sh in AQA
# mode. Dedicated AQA overrides take precedence; other explicit paths survive.
if [[ "${TRAIN_FILE:-}" == /opt/huawei/dataset/infra_data/data/train.parquet ]]; then
    unset TRAIN_FILE
fi
if [[ "${VAL_FILE:-}" == /opt/huawei/dataset/infra_data/data/validation.parquet ]]; then
    unset VAL_FILE
fi
export TRAIN_FILE="${HUANYU_AQA_TRAIN_FILE:-${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
export VAL_FILE="${HUANYU_AQA_VAL_FILE:-${VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
echo "[huanyu-aqa] train=$TRAIN_FILE validation=$VAL_FILE"
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.5-35B-A3B}"

# Both paths use 16 actor/rollout NPUs; only GenRM needs another 16 NPUs.
# One sequence per actor rank avoids retaining full unsharded gradients during
# FSDP2 no_sync accumulation. More train samples may form multiple mini-batches.
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
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
_huanyu_steps="${TOTAL_TRAINING_STEPS:--1}"
if [[ "${_huanyu_steps}" == -1 || "${_huanyu_steps}" == null ]]; then
    _huanyu_steps=null
elif [[ ! "${_huanyu_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[huanyu-aqa] FATAL: TOTAL_TRAINING_STEPS must be a positive integer, -1 or null" >&2
    exit 2
fi
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
_huanyu_experiment="audio_freeze_${HUANYU_FREEZE_AUDIO}_online"
if [[ "${RM_ENABLE}" == 0 ]]; then
    _huanyu_experiment+="_rule_based_16card"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${_huanyu_experiment}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/runs/${PROJECT_NAME}/${EXPERIMENT_NAME}/tensorboard}"
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
    export ACTOR_MODEL_DTYPE=fp32 REF_MODEL_DTYPE=bfloat16
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
        actor_rollout_ref.rollout.max_num_seqs=64
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
        actor_rollout_ref.ref.entropy_checkpointing=False
        actor_rollout_ref.ref.entropy_from_logits_with_chunking=False
    )
fi

# The direct AQA entry (including RM_ENABLE=0) must establish the same policy
# before its first native import. Do not rely only on the final backend or a
# previously started Raylet. TP4 below is also enforced in the final overrides.
source "${_HUANYU_EXAMPLE_DIR}/precision_mode.sh" "$@" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 || exit $?
echo "[huanyu-aqa] precision train_GDN=$MINICPMO_HUANYU_GDN_FP32_TRAINING rollout_GDN=$MINICPMO_GDN_FP32_REFERENCE reload=$MINICPMO_HUANYU_LAYERWISE_RELOAD KV_zero=$MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE hooks=$VLLM_OMNI_EXTERNAL_MODULES"

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
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit/*|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit/*) continue ;;
    esac
    _huanyu_retained_path+="${_huanyu_retained_path:+:}${_huanyu_path}"
done
export PYTHONPATH="${HUANYU_VLLM_OMNI_ROOT}:${PROJECT_ROOT}${_huanyu_retained_path:+:${_huanyu_retained_path}}"
python3 -B "${_HUANYU_EXAMPLE_DIR}/check_native_runtime.py" --check --expected-root "${HUANYU_VLLM_OMNI_ROOT}"

_huanyu_ray_args=()
for _huanyu_env_key in PYTHONPATH HUANYU_VLLM_OMNI_ROOT HUANYU_FREEZE_AUDIO \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR MINICPMO_USE_PRECOMPUTE \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH GENRM_DUMP_DIR; do
    _huanyu_env_json="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "${_huanyu_env_key}")"
    _huanyu_ray_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_huanyu_env_key}=${_huanyu_env_json}")
done

echo "[huanyu-aqa] freeze_audio=${HUANYU_FREEZE_AUDIO} cache=off visual=frozen; actor=16NPU RM=${REWARD_POOL_NPUS}NPU RM_ENABLE=${RM_ENABLE}"
if [[ "${RM_ENABLE}" == 0 ]]; then
    echo "[huanyu-aqa] reward=rule_based (character bigram Dice against ground_truth; no model/API)"
fi
echo "[huanyu-aqa] train_batch=${TRAIN_BATCH_SIZE} ppo_mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N} static_microbatch=1"
echo "[huanyu-aqa] prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} steps=${_huanyu_steps}"
echo "[huanyu-aqa] forward_prefetch=off Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH}; HCCL timeout is unchanged"

# Required settings follow user arguments so the checked datasets and execution
# contract cannot be bypassed by stale command-line options. Model selection is
# the verl alias; the checkpoint's HF architecture and RM alias remain intact.
exec bash "${_HUANYU_BACKEND}" "${_huanyu_backend_defaults[@]}" "$@" \
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
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    "${_huanyu_reward_args[@]}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.total_training_steps="${_huanyu_steps}" \
    "${_huanyu_ray_args[@]}" "${HUANYU_PRECISION_ARGS[@]}"
