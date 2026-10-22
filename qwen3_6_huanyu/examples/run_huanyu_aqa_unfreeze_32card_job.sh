#!/usr/bin/env bash
# Online AQA + trainable Huanyu encoder/projector + GenRM, 32 NPUs, 5K budget.
# Changes from the frozen sampleall job:
#   HUANYU_FREEZE_AUDIO: true -> false; MINICPMO_TUNE_AUDIO: 0 -> 1
#   HUANYU_USE_PRECOMPUTE: 1 -> 0 (raw audio, no cache manifest required)
#   dataset: omni_mix_f8_genrm_sampleall -> omni_mix_aqa_only
#   train/mini batch: 32/32 -> 4/4; rollout.n stays 4 (16 actor sequences)
#   dynamic batches -> static microbatch=1; SP=1; prefetch=off in AQA entry
#   length: prompt=3072 + response=2048 = 5120; engine/token budget=5248
# Train attention remains eager; HUANYU_GRAPH_MODE still controls inference.
#
# Run this SAME command on both nodes. VC_TASK_INDEX=0 starts Ray head and
# training; VC_TASK_INDEX=1 only joins Ray. Actor/rollout uses 16 NPUs and
# GenRM uses another 16 NPUs. The original job owns setup and readiness checks.
# PROJECT_ROOT must be absolute; this wrapper may be stored outside it.
# Deploy the supporting scripts to PROJECT_ROOT/examples/qwen3_6_huanyu/.
#
# Optional AQA paths (checked by check_aqa_audio_train.py before training):
#   HUANYU_AQA_TRAIN_FILE=/path/to/aqa/train.parquet
#   HUANYU_AQA_VAL_FILE=/path/to/aqa/validation.parquet
# Smoke run: TOTAL_TRAINING_STEPS=2 bash /path/to/this/script.sh
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || "$PROJECT_ROOT" != /* ]]; then
    echo 'FATAL: set PROJECT_ROOT to the absolute training-project path' >&2
    exit 2
fi
export PROJECT_ROOT
_HUANYU_AQA_JOB="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh"
[[ -f "$_HUANYU_AQA_JOB" ]] || {
    echo "FATAL: missing existing job launcher: $_HUANYU_AQA_JOB" >&2
    exit 1
}

# Dedicated profile variables prevent old sampleall/16K settings from leaking
# into online audio training. The job reapplies the mode after sourcing env.
export HUANYU_JOB_MODE=aqa_unfreeze
export TRAIN_FILE="${HUANYU_AQA_TRAIN_FILE:-/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/train.parquet}"
export VAL_FILE="${HUANYU_AQA_VAL_FILE:-/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/validation.parquet}"
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048
export ROLLOUT_PROMPT_LENGTH=3072 ROLLOUT_MAX_MODEL_LEN=5248
export MAX_TOKEN_LEN_PER_GPU=5248 MINICPMO_MM_PROMPT_BUDGET=3072
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_unfreeze_5k_32card}"

echo "[huanyu-aqa-job] train=$TRAIN_FILE validation=$VAL_FILE"
echo '[huanyu-aqa-job] freeze_audio=false cache=off actor=16NPU GenRM=16NPU prompt=3072 response=2048'

# Length/attention defaults precede caller Hydra overrides. AQA's checked
# dataset, trainable-audio and synchronized microbatch settings are enforced
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
    "$@"
