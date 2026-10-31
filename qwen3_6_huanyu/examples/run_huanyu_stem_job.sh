#!/usr/bin/env bash
# Run this SAME entry on every 16-NPU node. Only VC_TASK_INDEX=0 submits training.
# TOTAL_NPUS=128 (8 nodes) or 256 (16 nodes); all devices serve Actor/rollout.
# CPU math_dapo rule reward, no GenRM. Defaults: BS128, mini128, N16, 2K+10K.
# Reuses the common job's vLLM-Omni backup/replacement, editable install, site
# environment, native checks, Ray head/worker startup and readiness checks.
# Required: VC_TASK_INDEX=0..nodes-1, VC_WORKER_HOSTS=comma-separated host list.
# Identical PROJECT_ROOT/checkpoint/data paths must be mounted on every node;
# PROJECT_ROOT/runs must be shared (prepared Parquet lives under the run dir).
# Optional: ROLLOUT_TP=16; default TP8. TP is for inference, training uses FSDP.
# 10K is an output LIMIT, not a forced length. For a full-length stress test:
# TOTAL_TRAINING_STEPS=2 STEM_IGNORE_EOS=true bash .../run_huanyu_stem_job.sh
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_STEM_JOB_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu_32card_job.sh run_huanyu_stem.sh run_huanyu.sh; do
    [[ -r "$_STEM_JOB_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_STEM_JOB_EXAMPLES/$_file" >&2; exit 2; }
done
if [[ -z "${TOTAL_NPUS:-}" ]]; then
    case "${VC_WORKER_NUM:-}" in 8) TOTAL_NPUS=128 ;; 16) TOTAL_NPUS=256 ;; *) TOTAL_NPUS=128 ;; esac
fi
case "$TOTAL_NPUS" in 128|256) ;; *) echo 'FATAL: TOTAL_NPUS must be 128 or 256' >&2; exit 2 ;; esac
export TOTAL_NPUS HUANYU_CLUSTER_NODES=$((TOTAL_NPUS / 16))
[[ "${VC_TASK_INDEX:-}" =~ ^(0|[1-9][0-9]?)$ ]] && (( VC_TASK_INDEX < HUANYU_CLUSTER_NODES )) || {
    echo "FATAL: VC_TASK_INDEX must be 0..$((HUANYU_CLUSTER_NODES - 1))" >&2; exit 2;
}
if [[ -n "${VC_WORKER_NUM:-}" && "$VC_WORKER_NUM" != "$HUANYU_CLUSTER_NODES" ]]; then
    echo 'FATAL: VC_WORKER_NUM disagrees with TOTAL_NPUS/16' >&2; exit 2
fi
export HUANYU_JOB_MODE=stem STEM_SCALED_JOB=1 STEM_REWARD_MODE=rule
export NUM_NPUS=16 NNODES="$HUANYU_CLUSTER_NODES" NPUS_PER_NODE=16
export RM_ENABLE=0 RM_NNODES=0 RM_N_GPUS_PER_NODE=16 RM_TP=4 RM_SPLIT_NODES=0 REWARD_POOL_NPUS=0
export ROLLOUT_TP="${ROLLOUT_TP:-8}" ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
case "$ROLLOUT_TP" in 4|8|16) ;; *) echo 'FATAL: ROLLOUT_TP must be 4, 8 or 16' >&2; exit 2 ;; esac
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}" ROLLOUT_N="${ROLLOUT_N:-16}"
for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N ROLLOUT_MAX_NUM_SEQS; do
    [[ "${!_key}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
done
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 &&
   PPO_MINI_BATCH_SIZE * ROLLOUT_N % TOTAL_NPUS == 0 )) || {
    echo 'FATAL: BS must divide into PPO minibatches, and mini_batch*N must divide Actor DP' >&2; exit 2;
}
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
export TRAIN_FILE="${STEM_TRAIN_FILE:-$STEM_DATA_DIR/train.parquet}"
export VAL_FILE="${STEM_VAL_FILE:-$STEM_DATA_DIR/val.parquet}"
if [[ -z "${STEM_VAL_FILE:-}" && ! -f "$VAL_FILE" && -f "$STEM_DATA_DIR/validation.parquet" ]]; then
    export VAL_FILE="$STEM_DATA_DIR/validation.parquet"
fi
export STEM_CONTEXT_LENGTH="${STEM_CONTEXT_LENGTH:-12k}" STEM_PROMPT_LENGTH="${STEM_PROMPT_LENGTH:-2048}"
export STEM_CONTEXT_SLACK=0 STEM_DYNAMIC_BSZ=false STEM_SYNC_EACH_MICROBATCH=true
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 ACTOR_MODEL_DTYPE=fp32 REF_MODEL_DTYPE=bfloat16
export ACTOR_FORWARD_PREFETCH=false ACTIVATION_OFFLOAD=false
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
export ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS:-128}"
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export HUANYU_STEM_TEXT_TRAIN=1 HUANYU_AQA_AUDIO_TRAIN=0
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_stem_${TOTAL_NPUS}card_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}" TEST_FREQ="${TEST_FREQ:-10}"

# An old diagnostic shell must not turn this training run into disk-only mode
# or stop after a destructive weight audit. Also clear reused Ray worker envs.
_STEM_JOB_ARGS=()
for _key in HUANYU_CHECKPOINT_VALIDATION_MODE HUANYU_FULL_WEIGHT_AUDIT_DIR HUANYU_FULL_WEIGHT_AUDIT_RUN_ID \
    HUANYU_WEIGHT_SYNC_AUDIT_DIR HUANYU_AUDIO_PROBE_DIR HUANYU_TEXT_AB_DIR \
    HUANYU_EMBEDDING_AB_DIR HUANYU_EMBEDDING_AB_SIDE; do
    export "$_key="
    _STEM_JOB_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=\"\"")
done
echo "[huanyu-stem-job] physical=${NNODES}x16=$TOTAL_NPUS actor=$TOTAL_NPUS GenRM=0 reward=math_dapo"
echo "[huanyu-stem-job] training FSDP=$TOTAL_NPUS SP=1 TP=1; rollout TP=$ROLLOUT_TP replicas=$((TOTAL_NPUS / ROLLOUT_TP)) max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
echo "[huanyu-stem-job] BS=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE N=$ROLLOUT_N trajectories=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) microbatch=1 sync_each=true"
echo "[huanyu-stem-job] context=$STEM_CONTEXT_LENGTH prompt=$STEM_PROMPT_LENGTH EOS_allowed=$([[ "${STEM_IGNORE_EOS:-false}" == true ]] && echo false || echo true)"

exec bash "$_STEM_JOB_EXAMPLES/run_huanyu_32card_job.sh" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    "$@" "${_STEM_JOB_ARGS[@]}"
