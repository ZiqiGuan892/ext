#!/usr/bin/env bash
# Online AQA + trainable Huanyu encoder/projector + GenRM, 32 NPUs, 5K budget.
# Changes from the frozen sampleall job:
#   encoder/projector training: independent flags, both true by default
#   HUANYU_USE_PRECOMPUTE: 1 -> 0 (raw audio, no cache manifest required)
#   dataset: migrated AQA under /opt/huawei/dataset/infra_data/data/aqa_dataset
#   train/mini batch: 32/32 -> 4/4; rollout.n stays 4 (16 actor sequences)
#   dynamic batches -> static microbatch=1; SP=1; prefetch=off in AQA entry
#   length: prompt=3072 + response=2048 = 5120; engine/token budget=5248
# Precision policy (shared with the frozen job, applied before native imports):
#   training GDN: FP32 recurrence with autocast disabled (padded HF API)
#   rollout GDN: slot-local FP32 state; IPC: layerwise reload including audio
#   hybrid KV blocks: logical-to-physical zeroing; vision item isolation on
#   prefix cache and MM processor cache: off while audio weights change
# Explicit numerical environment overrides remain available for A/B tests;
# individual kernel overrides require profile=custom.
# Train attention remains eager. The default full precision profile also
# requires inference eager mode; graph comparisons require profile=custom.
#
# Run this SAME command on both nodes. VC_TASK_INDEX=0 starts Ray head and
# training; VC_TASK_INDEX=1 only joins Ray. Actor/rollout uses 16 NPUs and
# GenRM uses another 16 NPUs. The original job owns setup and readiness checks.
# PROJECT_ROOT defaults to the shared Huanyu deployment; an override must be
# absolute. This wrapper may be stored outside it.
# Deploy the supporting scripts to PROJECT_ROOT/examples/qwen3_6_huanyu/.
#
# Optional AQA paths (checked by check_aqa_audio_train.py before training):
#   HUANYU_AQA_TRAIN_FILE=/path/to/aqa/train.parquet
#   HUANYU_AQA_VAL_FILE=/path/to/aqa/validation.parquet
# Smoke run: TOTAL_TRAINING_STEPS=2 bash /path/to/this/script.sh
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/hfy-model-gy/verl_omni_qwen36_gzq_new}"
if [[ "$PROJECT_ROOT" != /* ]]; then
    echo 'FATAL: set PROJECT_ROOT to the absolute training-project path' >&2
    exit 2
fi
_HUANYU_AQA_JOB="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh"
_HUANYU_AQA_PRECISION="${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh"
_HUANYU_AQA_AUDIO_MODE="${PROJECT_ROOT}/examples/qwen3_6_huanyu/audio_train_mode.sh"
[[ -f "$_HUANYU_AQA_JOB" && -f "$_HUANYU_AQA_PRECISION" && -f "$_HUANYU_AQA_AUDIO_MODE" ]] || {
    echo "FATAL: deploy run_huanyu_32card_job.sh, precision_mode.sh and audio_train_mode.sh under PROJECT_ROOT/examples/qwen3_6_huanyu" >&2
    exit 1
}

# Set each option independently (true=train, false=freeze). Edit these defaults
# or export the same values on BOTH nodes before launching. For example,
# encoder=false/projector=true trains only the projector. Both false keeps
# online audio encoding and trains only the remaining trainable model parts.
export HUANYU_TRAIN_AUDIO_ENCODER="${HUANYU_TRAIN_AUDIO_ENCODER-true}"
export HUANYU_TRAIN_AUDIO_PROJECTOR="${HUANYU_TRAIN_AUDIO_PROJECTOR-true}"
# The job snapshots these choices and restores them after sourcing the site env.
export HUANYU_JOB_MODE=aqa_unfreeze
export HUANYU_FREEZE_AUDIO=false HUANYU_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export HUANYU_AQA_AUDIO_TRAIN=1 MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_VISION=0
source "$_HUANYU_AQA_AUDIO_MODE" || exit $?
# Do not let the shared environment's generic dataset defaults mask AQA paths.
if [[ "${TRAIN_FILE:-}" == /opt/huawei/dataset/infra_data/data/train.parquet ]]; then
    unset TRAIN_FILE
fi
if [[ "${VAL_FILE:-}" == /opt/huawei/dataset/infra_data/data/validation.parquet ]]; then
    unset VAL_FILE
fi
export TRAIN_FILE="${HUANYU_AQA_TRAIN_FILE:-${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
export VAL_FILE="${HUANYU_AQA_VAL_FILE:-${VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048
export ROLLOUT_PROMPT_LENGTH=3072 ROLLOUT_MAX_MODEL_LEN=5248
export MAX_TOKEN_LEN_PER_GPU=5248 MINICPMO_MM_PROMPT_BUDGET=3072
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_unfreeze_5k_32card}"

# Use the same policy on both nodes, even when this wrapper lives elsewhere.
# The AQA training entry requires TP4; resolve that effective TP before Ray as
# well, rather than validating an inherited frozen-profile TP which is ignored.
source "$_HUANYU_AQA_PRECISION" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4 || exit $?
echo "[huanyu-aqa-job] train=$TRAIN_FILE validation=$VAL_FILE"
echo "[huanyu-aqa-job] freeze_audio=$HUANYU_FREEZE_AUDIO train_audio_encoder=$HUANYU_TRAIN_AUDIO_ENCODER train_audio_projector=$HUANYU_TRAIN_AUDIO_PROJECTOR cache=off actor=16NPU GenRM=16NPU prompt=3072 response=2048"
echo "[huanyu-aqa-job] precision train_GDN=$MINICPMO_HUANYU_GDN_FP32_TRAINING rollout_GDN=$MINICPMO_GDN_FP32_REFERENCE reload=$MINICPMO_HUANYU_LAYERWISE_RELOAD KV_zero=$MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE hooks=$VLLM_OMNI_EXTERNAL_MODULES"

# Length/attention defaults precede caller Hydra overrides. AQA's checked
# dataset, audio component modes and synchronized microbatch settings are enforced
# by run_huanyu_aqa_audio_train.sh after caller arguments. TOTAL_TRAINING_STEPS
# is its supported run-length control. Logs/GenRM retain unique runs folders.
exec bash "$_HUANYU_AQA_JOB" \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.prompt_length=3072 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=5248 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5248 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5248 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=5248 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=5248 \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    '++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET="3072"' \
    "$@" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
