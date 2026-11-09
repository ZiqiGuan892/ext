#!/usr/bin/env bash
# Run this SAME entry on every 16-NPU node. Only VC_TASK_INDEX=0 submits training.
# TOTAL_NPUS=16/32/128/256 (1/2/8/16 physical nodes).
# HUANYU_REWARD_MODE=rule_based (default) uses CPU math_dapo; RM reserves one
# 16-NPU node by default (32 cards: Actor16 + RM16). BS128, mini128, N16, 2K+10K.
# Reuses the common job's vLLM-Omni source selection, editable install, site
# environment, native checks and Ray head/worker startup.
# Extra job probes default off; HUANYU_JOB_PREFLIGHT=1 opts in.
# HUANYU_RAY_READINESS_CHECK remains a fallback alias.
# Required: VC_TASK_INDEX=0..nodes-1, VC_WORKER_HOSTS=comma-separated host list.
# If the platform launches one Bash per NPU, the common job elects LOCAL_RANK=0
# on EACH node; other local ranks exit before setup/Ray and keep all NPUs free.
# Identical PROJECT_ROOT/checkpoint/data paths must be mounted on every node;
# PROJECT_ROOT/runs must be shared (prepared Parquet lives under the run dir).
# Optional: ROLLOUT_TP=16; default TP8. TP is for inference, training uses FSDP.
# STEM_IGNORE_EOS=false (default) allows EOS; true ignores EOS during rollout.
# 10K is an output LIMIT; ignoring EOS does not disable other stop conditions.
# Example for a length stress test:
# TOTAL_TRAINING_STEPS=2 STEM_IGNORE_EOS=true bash .../run_huanyu_stem_job.sh
# --train-only: on VC_TASK_INDEX=0 after prepare_huanyu_env.sh and
# Ray startup. Keeps this profile but skips ALL common job setup/Ray startup.
#
# Launch commands (choose ONE size; run the same command on every job node):
# export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
# export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
# export STEM_DATA_DIR=/opt/huawei/dataset/infra_data/data/rl_stem_data
# vLLM-Omni uses the existing /vllm-workspace/vllm-omni installation.
#
# 16 cards: one node; platform VC_TASK_INDEX=0, VC_WORKER_NUM=1.
# TOTAL_NPUS=16 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=4 \
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem_job.sh"
# 32 cards: two nodes, Actor/Ref FSDP groups of 16, rollout TP8 by default.
# The platform supplies VC_TASK_INDEX=0/1 and VC_WORKER_NUM=2.
# TOTAL_NPUS=32 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=4 \
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem_job.sh"
# 128 cards: eight nodes; platform VC_TASK_INDEX=0..7, VC_WORKER_NUM=8.
# TOTAL_NPUS=128 TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=128 ROLLOUT_N=16 \
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem_job.sh"
# 256 cards: sixteen nodes; platform VC_TASK_INDEX=0..15, VC_WORKER_NUM=16.
# TOTAL_NPUS=256 TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=128 ROLLOUT_N=16 \
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem_job.sh"
#
# VC_WORKER_HOSTS must list all nodes, head first (one host for 16 cards).
# Set these VC_* variables yourself only if the platform does not supply them.
# TOTAL_NPUS is physical capacity; NUM_NPUS stays 16 per Actor node.
# NNODES is Actor nodes; RM_NNODES is reward nodes (default 1 in RM mode).
# RM_MODEL_PATH selects the reward checkpoint; rule_based needs no RM model.
# TRAIN_BATCH_SIZE -> data.train_batch_size (global prompt count)
# PPO_MINI_BATCH_SIZE -> actor.ppo_mini_batch_size (prompt count before N repeats)
# ROLLOUT_N -> rollout.n (responses per prompt); trajectories per step = BS*N.
# Required: TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 and
# (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % (NUM_NPUS * NNODES) == 0.
# ROLLOUT_TP=4/8/16 (default 8); replicas = Actor cards/ROLLOUT_TP.
# Batch/card settings use these environment variables, not Hydra overrides.
# If environment and Ray are ALREADY prepared, run only on the head:
# TOTAL_NPUS=32 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=4 RAY_ADDRESS=auto \
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem_job.sh" --train-only
set -euo pipefail

_STEM_TRAIN_ONLY=0
if [[ "${1:-}" == --train-only ]]; then _STEM_TRAIN_ONLY=1; shift; fi
if (( _STEM_TRAIN_ONLY )) && [[ "${HUANYU_PREPARE_ONLY:-0}" == 1 ]]; then
    echo 'FATAL: --train-only cannot be used during environment preparation' >&2; exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_STEM_JOB_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
export HUANYU_RAY_READINESS_CHECK="${HUANYU_JOB_PREFLIGHT:-${HUANYU_RAY_READINESS_CHECK:-0}}"
case "$HUANYU_RAY_READINESS_CHECK" in
    0|1) ;;
    *) echo 'FATAL: HUANYU_JOB_PREFLIGHT (or HUANYU_RAY_READINESS_CHECK) must be 0 or 1' >&2; exit 2 ;;
esac
for _file in run_huanyu_32card_job.sh run_huanyu_stem.sh run_qwen36_huanyu_gspo_npu.sh; do
    [[ -r "$_STEM_JOB_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_STEM_JOB_EXAMPLES/$_file" >&2; exit 2; }
done
if [[ -z "${TOTAL_NPUS:-}" ]]; then
    case "${VC_WORKER_NUM:-}" in 1) TOTAL_NPUS=16 ;; 2) TOTAL_NPUS=32 ;; 8) TOTAL_NPUS=128 ;; 16) TOTAL_NPUS=256 ;; *) TOTAL_NPUS=128 ;; esac
fi
case "$TOTAL_NPUS" in 16|32|128|256) ;; *) echo 'FATAL: TOTAL_NPUS must be 16, 32, 128 or 256' >&2; exit 2 ;; esac
export TOTAL_NPUS HUANYU_CLUSTER_NODES=$((TOTAL_NPUS / 16))
if [[ "${HUANYU_PREPARE_ONLY:-0}" != 1 ]]; then
    [[ "${VC_TASK_INDEX:-}" =~ ^(0|[1-9][0-9]?)$ ]] && (( VC_TASK_INDEX < HUANYU_CLUSTER_NODES )) || {
        echo "FATAL: VC_TASK_INDEX must be 0..$((HUANYU_CLUSTER_NODES - 1))" >&2; exit 2;
    }
    if [[ -n "${VC_WORKER_NUM:-}" && "$VC_WORKER_NUM" != "$HUANYU_CLUSTER_NODES" ]]; then
        echo 'FATAL: VC_WORKER_NUM disagrees with TOTAL_NPUS/16' >&2; exit 2
    fi
fi
if (( _STEM_TRAIN_ONLY )); then
    [[ "$VC_TASK_INDEX" == 0 && "${LOCAL_RANK:-0}" == 0 ]] || {
        echo 'FATAL: --train-only must run once on VC_TASK_INDEX=0, without per-device launchers' >&2; exit 2;
    }
fi
export HUANYU_JOB_MODE=stem STEM_SCALED_JOB=1
_STEM_JOB_REWARD="${HUANYU_REWARD_MODE:-${STEM_REWARD_MODE:-}}"
if [[ -z "$_STEM_JOB_REWARD" ]]; then
    case "${RM_ENABLE:-0}" in
        0) _STEM_JOB_REWARD=rule_based ;;
        1) _STEM_JOB_REWARD=RM ;;
        *) echo 'FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
    esac
fi
case "${_STEM_JOB_REWARD,,}" in
    rm|genrm) export HUANYU_REWARD_MODE=RM STEM_REWARD_MODE=genrm RM_ENABLE=1 ;;
    rule_based|ruled_based|rule) export HUANYU_REWARD_MODE=rule_based STEM_REWARD_MODE=rule RM_ENABLE=0 ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based; STEM_REWARD_MODE accepts genrm or rule' >&2; exit 2 ;;
esac
export NUM_NPUS=16 NPUS_PER_NODE=16
if [[ "$RM_ENABLE" == 1 ]]; then
    export RM_NNODES="${RM_NNODES:-1}" RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}" RM_TP="${RM_TP:-4}" RM_SPLIT_NODES=1
    for _key in RM_NNODES RM_N_GPUS_PER_NODE RM_TP; do
        [[ "${!_key}" =~ ^[1-9][0-9]{0,4}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
    done
    [[ "$RM_N_GPUS_PER_NODE" == 16 ]] && (( RM_N_GPUS_PER_NODE % RM_TP == 0 )) || {
        echo 'FATAL: scaled STEM RM uses complete 16-NPU nodes and RM_TP must divide 16' >&2; exit 2;
    }
    (( HUANYU_CLUSTER_NODES > RM_NNODES )) || {
        echo 'FATAL: STEM RM mode needs at least one Actor node plus one RM node (at least 32 NPUs)' >&2; exit 2;
    }
    export REWARD_POOL_NPUS=$((RM_NNODES * RM_N_GPUS_PER_NODE))
    export NNODES="${NNODES:-$((HUANYU_CLUSTER_NODES - RM_NNODES))}"
else
    export NNODES="${NNODES:-$HUANYU_CLUSTER_NODES}"
    export RM_NNODES=0 RM_N_GPUS_PER_NODE=16 RM_TP=4 RM_SPLIT_NODES=0 REWARD_POOL_NPUS=0
fi
[[ "$NNODES" =~ ^[1-9][0-9]{0,4}$ ]] || { echo 'FATAL: NNODES must be positive Actor node count' >&2; exit 2; }
_STEM_JOB_ACTOR_NPUS=$((NUM_NPUS * NNODES))
(( _STEM_JOB_ACTOR_NPUS + REWARD_POOL_NPUS <= TOTAL_NPUS )) || {
    echo 'FATAL: Actor + RM pools exceed TOTAL_NPUS; NNODES counts Actor nodes only' >&2; exit 2;
}
export ROLLOUT_TP="${ROLLOUT_TP:-8}" ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
case "$ROLLOUT_TP" in 4|8|16) ;; *) echo 'FATAL: ROLLOUT_TP must be 4, 8 or 16' >&2; exit 2 ;; esac
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}" ROLLOUT_N="${ROLLOUT_N:-16}"
for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N ROLLOUT_MAX_NUM_SEQS; do
    [[ "${!_key}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
done
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 &&
   PPO_MINI_BATCH_SIZE * ROLLOUT_N % _STEM_JOB_ACTOR_NPUS == 0 )) || {
    echo 'FATAL: BS must divide into PPO minibatches, and mini_batch*N must divide Actor DP' >&2; exit 2;
}
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
export TRAIN_FILE="${TRAIN_FILE:-${STEM_TRAIN_FILE:-$STEM_DATA_DIR/train.parquet}}"
_STEM_EXPLICIT_VAL="${VAL_FILE:-${STEM_VAL_FILE:-}}"
export VAL_FILE="${VAL_FILE:-${STEM_VAL_FILE:-$STEM_DATA_DIR/val.parquet}}"
if [[ -z "$_STEM_EXPLICIT_VAL" && ! -f "$VAL_FILE" && -f "$STEM_DATA_DIR/validation.parquet" ]]; then
    export VAL_FILE="$STEM_DATA_DIR/validation.parquet"
fi
export STEM_PROMPT_LENGTH="${STEM_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH:-2048}}"
export STEM_CONTEXT_LENGTH="${STEM_CONTEXT_LENGTH:-12k}"
# Maps to actor_rollout_ref.rollout.ignore_eos in run_huanyu_stem.sh.
# Set the default before job setup so site env files cannot change it.
export STEM_IGNORE_EOS="${STEM_IGNORE_EOS:-false}"
case "$STEM_IGNORE_EOS" in
    true|false) ;;
    *) echo 'FATAL: STEM_IGNORE_EOS must be true or false' >&2; exit 2 ;;
esac
export STEM_CONTEXT_SLACK="${STEM_CONTEXT_SLACK:-0}" STEM_DYNAMIC_BSZ="${STEM_DYNAMIC_BSZ:-false}" STEM_SYNC_EACH_MICROBATCH="${STEM_SYNC_EACH_MICROBATCH:-true}"
# Resolve the length profile before site setup so its defaults cannot replace
# either exported generic lengths or the selected STEM context.
huanyu_stem_job_tokens() {
    local value="${1,,}" factor=1
    if [[ "$value" == *k ]]; then value="${value%k}"; factor=1024; fi
    [[ "$value" =~ ^[1-9][0-9]{0,6}$ ]] || { echo "FATAL: invalid token budget '$1'" >&2; return 2; }
    printf '%s' "$((value * factor))"
}
MAX_PROMPT_LENGTH="$(huanyu_stem_job_tokens "${MAX_PROMPT_LENGTH:-$STEM_PROMPT_LENGTH}")"
_STEM_JOB_CONTEXT="$(huanyu_stem_job_tokens "$STEM_CONTEXT_LENGTH")"
MAX_RESPONSE_LENGTH="$(huanyu_stem_job_tokens "${MAX_RESPONSE_LENGTH:-$((_STEM_JOB_CONTEXT - MAX_PROMPT_LENGTH))}")"
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export STEM_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" STEM_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
[[ "$STEM_CONTEXT_SLACK" =~ ^(0|[1-9][0-9]{0,4})$ ]] || { echo 'FATAL: invalid STEM_CONTEXT_SLACK' >&2; exit 2; }
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((STEM_CONTEXT_LENGTH + STEM_CONTEXT_SLACK))}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}" REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bfloat16}"
export ACTOR_FORWARD_PREFETCH="${ACTOR_FORWARD_PREFETCH:-false}" ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-false}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
export ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS:-128}"
export HUANYU_PRECISION_PROFILE="${HUANYU_PRECISION_PROFILE:-full}" HUANYU_GRAPH_MODE="${HUANYU_GRAPH_MODE:-0}"
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export HUANYU_STEM_TEXT_TRAIN=1 HUANYU_AQA_AUDIO_TRAIN=0
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_stem_${TOTAL_NPUS}card_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}" TEST_FREQ="${TEST_FREQ:-10}"

_STEM_JOB_ARGS=()
if (( TOTAL_NPUS == 32 )); then
    # The common launcher resolves precision and exports TP before starting Ray.
    _STEM_JOB_ARGS+=("actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP")
fi
# An old diagnostic shell must not turn this training run into disk-only mode
# or stop after a destructive weight audit. Also clear reused Ray worker envs.
for _key in HUANYU_CHECKPOINT_VALIDATION_MODE HUANYU_FULL_WEIGHT_AUDIT_DIR HUANYU_FULL_WEIGHT_AUDIT_RUN_ID \
    HUANYU_WEIGHT_SYNC_AUDIT_DIR HUANYU_AUDIO_PROBE_DIR HUANYU_TEXT_AB_DIR \
    HUANYU_EMBEDDING_AB_DIR HUANYU_EMBEDDING_AB_SIDE; do
    export "$_key="
    _STEM_JOB_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=\"\"")
done
echo "[huanyu-stem-job] physical=${HUANYU_CLUSTER_NODES}x16=$TOTAL_NPUS actor=$_STEM_JOB_ACTOR_NPUS GenRM=$REWARD_POOL_NPUS reward=$HUANYU_REWARD_MODE"
echo "[huanyu-stem-job] training DP=$_STEM_JOB_ACTOR_NPUS SP=1 TP=1; rollout TP=$ROLLOUT_TP replicas=$((_STEM_JOB_ACTOR_NPUS / ROLLOUT_TP)) max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
echo "[huanyu-stem-job] BS=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE N=$ROLLOUT_N trajectories=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) microbatch=$PPO_MICRO_BATCH_SIZE_PER_GPU sync_each=$STEM_SYNC_EACH_MICROBATCH"
echo "[huanyu-stem-job] context=$STEM_CONTEXT_LENGTH prompt=$STEM_PROMPT_LENGTH ignore_eos=$STEM_IGNORE_EOS EOS_allowed=$([[ "$STEM_IGNORE_EOS" == true ]] && echo false || echo true)"

_STEM_ENTRY=run_huanyu_32card_job.sh
if (( _STEM_TRAIN_ONLY )); then
    _STEM_ENTRY=run_huanyu_stem.sh
    cd "$PROJECT_ROOT"
    export HUANYU_SKIP_EDITABLE_INSTALL=1
    # The common job normally supplies this; keep it for manually started Ray
    # too, since Ray may strip the preload from newly created worker processes.
    _value="$(python3 -c 'import json, os; print(json.dumps(os.environ.get("LD_PRELOAD", "")))')"
    _STEM_JOB_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD=$_value")
    echo "[huanyu-stem-job] train-only head=$VC_TASK_INDEX Ray=${RAY_ADDRESS:-auto}; setup and Ray startup skipped"
fi
exec bash "$_STEM_JOB_EXAMPLES/$_STEM_ENTRY" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-true}" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-true}" \
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY:-true}" \
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=${ACTOR_FORWARD_PREFETCH}" \
    "actor_rollout_ref.ref.fsdp_config.forward_prefetch=${REF_FORWARD_PREFETCH:-false}" \
    "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-true}" \
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB:-2048}" \
    "$@" "${_STEM_JOB_ARGS[@]}"
