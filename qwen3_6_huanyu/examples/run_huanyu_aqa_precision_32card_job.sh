#!/usr/bin/env bash
# One-click AQA precision training: run on BOTH 16-NPU nodes.
# VC_TASK_INDEX=0 starts Ray head and training; =1 joins Ray and stays alive.
# VC_WORKER_HOSTS=head_ip,worker_ip must be identical on both nodes.
# The common bootstrap syncs vLLM-Omni, installs the plugin when needed,
# sources env_export_huanyu.sh and waits for both nodes. Only the head then
# runs run_huanyu_aqa_precision.sh (online/unfrozen, 5K, actor16 + GenRM16).
# Full precision repairs and eager mode are enforced before Ray startup and
# forwarded through runtime_env, including when an existing cluster is reused.
# PROJECT_ROOT locates dependencies even when this script is stored elsewhere.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_AQA_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu_32card_job.sh run_huanyu_aqa_precision.sh precision_mode.sh; do
    [[ -f "$_AQA_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_AQA_EXAMPLES/$_file" >&2; exit 2; }
done
export HUANYU_JOB_MODE=aqa_precision
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_FREEZE_AUDIO=false HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0
export HUANYU_EMBED_CACHE_DIR="" HUANYU_AQA_AUDIO_TRAIN=1
export MINICPMO_TUNE_AUDIO=1 MINICPMO_TUNE_VISION=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export RM_ENABLE=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_precision_5k_32card}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ "${TRAIN_FILE:-}" == /opt/huawei/dataset/infra_data/data/train.parquet ]]; then unset TRAIN_FILE; fi
if [[ "${VAL_FILE:-}" == /opt/huawei/dataset/infra_data/data/validation.parquet ]]; then unset VAL_FILE; fi
export TRAIN_FILE="${HUANYU_AQA_TRAIN_FILE:-${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
export VAL_FILE="${HUANYU_AQA_VAL_FILE:-${VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
source "$_AQA_EXAMPLES/precision_mode.sh" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4
echo "[huanyu-aqa-precision-job] train=$TRAIN_FILE validation=$VAL_FILE profile=full graph=0 actor=16 RM=16"
exec bash "$_AQA_EXAMPLES/run_huanyu_32card_job.sh" "$@" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
