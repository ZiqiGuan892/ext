#!/usr/bin/env bash
# Online AQA: frozen encoder/projector, deterministic Actor/ref and rollout.
# Run on both 16-NPU nodes (VC_TASK_INDEX=0 head, =1 worker).
# Reuse the unfreeze job's setup, Ray roles and dataset; keep BS4/mini4/N4 and 5K.
# The LLM still trains. Batch-invariant kernels remain disabled.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || {
    echo 'FATAL: set PROJECT_ROOT to the absolute training-project path' >&2; exit 2;
}
_AQA_FROZEN_ENTRY="$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_unfreeze_32card_job.sh"
[[ -f "$_AQA_FROZEN_ENTRY" ]] || {
    echo "FATAL: missing $_AQA_FROZEN_ENTRY" >&2; exit 1;
}

# Explicit component flags win over inherited unfreeze settings. The shared job
# snapshots them before sourcing the site's environment and forwards them to Ray.
export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_FREEZE_AUDIO=true MINICPMO_TUNE_AUDIO=0
export HUANYU_JOB_DETERMINISM=1 HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 RM_ENABLE=1
export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1
# Keep this comparison profile stable when the unfreeze job's defaults change.
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 AQA_DYNAMIC_BSZ=false
export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 ROLLOUT_PROMPT_LENGTH=3072
export ROLLOUT_MAX_MODEL_LEN=5248 ROLLOUT_MAX_NUM_BATCHED_TOKENS=5248
export MAX_TOKEN_LEN_PER_GPU=5248 MINICPMO_MM_PROMPT_BUDGET=3072
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_frozen_deterministic_5k_32card}"

echo '[huanyu-aqa-frozen-job] encoder=frozen projector=frozen cache=off LLM=train actor=16NPU GenRM=16NPU'
echo '[huanyu-aqa-frozen-job] determinism=on batch_invariant=off graph=eager; actual worker/audio state follows in training logs'
exec bash "$_AQA_FROZEN_ENTRY" "$@"
