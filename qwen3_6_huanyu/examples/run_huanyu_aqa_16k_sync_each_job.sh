#!/usr/bin/env bash
# One-click 16K AQA training: run this SAME entry on BOTH 16-NPU nodes.
# VC_WORKER_HOSTS=head_ip,worker_ip must match; VC_TASK_INDEX=0 on head, =1 on worker.
# The common job syncs vLLM-Omni, installs the plugin, sources the environment,
# starts/reuses Ray and waits for both nodes. Only the head runs the direct
# run_huanyu_aqa_16k_sync_each.sh entry; the worker joins Ray and stays alive.
# Actor16 + RM16 on separate nodes, TP4, full precision.
# Online unfrozen audio; static micro batch 1 with per-microbatch gradient sync.
# 16K = prompt14336 + response2048; engine16512 includes 128 spare tokens.
# Optional HUANYU_16K_RESPONSE_LENGTH=4096 gives prompt12288; attention defaults
# to sdpa (HUANYU_16K_ATTN_IMPLEMENTATION=eager is available for comparison).
# Deploy this file AND the updated run_huanyu_32card_job.sh, plus the existing
# direct entry and its dependencies, to PROJECT_ROOT/examples/qwen3_6_huanyu/.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_AQA_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu_32card_job.sh run_huanyu_aqa_16k_sync_each.sh precision_mode.sh graph_mode.sh; do
    [[ -f "$_AQA_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_AQA_EXAMPLES/$_file" >&2; exit 2; }
done
export HUANYU_16K_RESPONSE_LENGTH="${HUANYU_16K_RESPONSE_LENGTH:-2048}"
export HUANYU_16K_ATTN_IMPLEMENTATION="${HUANYU_16K_ATTN_IMPLEMENTATION:-sdpa}"
[[ "$HUANYU_16K_RESPONSE_LENGTH" =~ ^[1-9][0-9]{0,4}$ ]] && (( HUANYU_16K_RESPONSE_LENGTH < 16384 )) || {
    echo 'FATAL: HUANYU_16K_RESPONSE_LENGTH must be an integer in 1..16383' >&2; exit 2;
}
case "$HUANYU_16K_ATTN_IMPLEMENTATION" in
    sdpa|eager) ;;
    *) echo 'FATAL: HUANYU_16K_ATTN_IMPLEMENTATION must be sdpa or eager' >&2; exit 2 ;;
esac

export HUANYU_JOB_MODE=aqa_16k_sync_each
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_FREEZE_AUDIO=false HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0
export HUANYU_EMBED_CACHE_DIR="" HUANYU_AQA_AUDIO_TRAIN=1
export MINICPMO_TUNE_AUDIO=1 MINICPMO_TUNE_VISION=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export RM_ENABLE=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1 REWARD_POOL_NPUS=16
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 PPO_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_RESPONSE_LENGTH="$HUANYU_16K_RESPONSE_LENGTH"
export MAX_PROMPT_LENGTH="$((16384 - MAX_RESPONSE_LENGTH))"
export ROLLOUT_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MINICPMO_MM_PROMPT_BUDGET="$MAX_PROMPT_LENGTH"
export ROLLOUT_MAX_MODEL_LEN=16512 MAX_TOKEN_LEN_PER_GPU=16512
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_16k_16_16_precision}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ "${TRAIN_FILE:-}" == /opt/huawei/dataset/infra_data/data/train.parquet ]]; then unset TRAIN_FILE; fi
if [[ "${VAL_FILE:-}" == /opt/huawei/dataset/infra_data/data/validation.parquet ]]; then unset VAL_FILE; fi
export TRAIN_FILE="${HUANYU_AQA_TRAIN_FILE:-${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
export VAL_FILE="${HUANYU_AQA_VAL_FILE:-${VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
export HUANYU_AQA_TRAIN_FILE="$TRAIN_FILE" HUANYU_AQA_VAL_FILE="$VAL_FILE"
source "$_AQA_EXAMPLES/precision_mode.sh" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4
echo "[huanyu-aqa-16k-job] train=$TRAIN_FILE validation=$VAL_FILE"
echo "[huanyu-aqa-16k-job] actor=16x1=16NPU RM=16x1=16NPU TP=4 prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH engine=16512 dynamic_bsz=false micro_batch_size_per_gpu=1 sync_each_microbatch=true"
echo "[huanyu-aqa-16k-job] profile=full graph=0 freeze_audio=false cache=off attention=$HUANYU_16K_ATTN_IMPLEMENTATION"
exec bash "$_AQA_EXAMPLES/run_huanyu_32card_job.sh" "$@" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true
