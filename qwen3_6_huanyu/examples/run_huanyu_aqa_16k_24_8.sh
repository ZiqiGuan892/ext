#!/usr/bin/env bash
# Direct AQA entry for an already prepared two-node, 32-NPU Ray cluster.
# Run ONCE on the head node after env_export / package installation / Ray setup.
# Each physical 16-NPU node supplies actor12 + RM4; both rollouts use TP4.
# Online audio, Huanyu unfrozen by default, full precision policy, graph off.
# 16K = prompt14336 + response2048; engine budget16512 includes 128 spare tokens.
# Optional: HUANYU_16K_RESPONSE_LENGTH=4096 (prompt becomes12288),
# HUANYU_16K_ATTN_IMPLEMENTATION=eager (default sdpa), HUANYU_FREEZE_AUDIO=true.
# No env sourcing, repository replacement, package installs or Ray start/stop.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_AQA_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu_aqa_audio_train.sh run_output.sh graph_mode.sh precision_mode.sh; do
    [[ -f "$_AQA_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_AQA_EXAMPLES/$_file" >&2; exit 2; }
done
_AQA_RESPONSE="${HUANYU_16K_RESPONSE_LENGTH:-2048}"
_AQA_ATTN="${HUANYU_16K_ATTN_IMPLEMENTATION:-sdpa}"
[[ "$_AQA_RESPONSE" =~ ^[1-9][0-9]{0,4}$ ]] && (( _AQA_RESPONSE < 16384 )) || {
    echo 'FATAL: HUANYU_16K_RESPONSE_LENGTH must be an integer in 1..16383' >&2; exit 2;
}
case "$_AQA_ATTN" in
    sdpa|eager) ;;
    *) echo 'FATAL: HUANYU_16K_ATTN_IMPLEMENTATION must be sdpa or eager' >&2; exit 2 ;;
esac
source "$_AQA_EXAMPLES/run_output.sh"

# This entry owns pool sizes, TP, batching and the combined 16K length budget.
# Override stale settings inherited from the site's shared environment file.
export NUM_NPUS=12 NNODES=2 ROLLOUT_TP=4
export RM_ENABLE=1 RM_N_GPUS_PER_NODE=4 RM_NNODES=2 RM_TP=4
export REWARD_POOL_NPUS=8 RM_SPLIT_NODES=0
export TRAIN_BATCH_SIZE=6 PPO_MINI_BATCH_SIZE=6 ROLLOUT_N=4
# 6 prompts x 4 responses = 24 sequences: exactly one per actor rank/update.
export MAX_RESPONSE_LENGTH="$_AQA_RESPONSE"
export MAX_PROMPT_LENGTH="$((16384 - MAX_RESPONSE_LENGTH))"
export ROLLOUT_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MINICPMO_MM_PROMPT_BUDGET="$MAX_PROMPT_LENGTH"
export ROLLOUT_MAX_MODEL_LEN=16512 MAX_TOKEN_LEN_PER_GPU=16512
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0 HUANYU_SKIP_EDITABLE_INSTALL=1
export HUANYU_FREEZE_AUDIO="${HUANYU_FREEZE_AUDIO:-false}"
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export WANDB_MODE="${WANDB_MODE:-offline}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_16k_24_8_precision}"
if [[ "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then unset DEFAULT_LOCAL_DIR; fi
if [[ "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then unset TENSORBOARD_DIR; fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
_AQA_ROLLOUT_DIR="${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}"
source "$_AQA_EXAMPLES/graph_mode.sh"
source "$_AQA_EXAMPLES/precision_mode.sh" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
[[ "$RAY_ADDRESS" != local ]] || { echo 'FATAL: use RAY_ADDRESS=auto or head_ip:port for the prepared cluster' >&2; exit 2; }
_AQA_RAY_TIMEOUT="${HUANYU_RAY_STATUS_TIMEOUT:-30}"
[[ "$_AQA_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: HUANYU_RAY_STATUS_TIMEOUT must be positive seconds' >&2; exit 2; }
for _command in ray timeout; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing $_command in the prepared environment" >&2; exit 2; }
done
if ! timeout --foreground "${_AQA_RAY_TIMEOUT}s" ray status --address="$RAY_ADDRESS"; then
    echo 'FATAL: existing Ray is not reachable; prepare both 16-NPU nodes first' >&2
    exit 1
fi
echo "[huanyu-aqa-16k] actor=12x2=24NPU RM=4x2=8NPU rollout_TP=4 replicas=6 RM_TP=4 replicas=2"
echo "[huanyu-aqa-16k] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=16384 engine=16512 train_batch=6 mini_batch=6 rollout_n=4"
echo "[huanyu-aqa-16k] precision=full graph=0 attention=$_AQA_ATTN freeze_audio=$HUANYU_FREEZE_AUDIO cache=off console=$HUANYU_CONSOLE_LOG"

exec bash "$_AQA_EXAMPLES/run_huanyu_aqa_audio_train.sh" \
    'trainer.logger=[console]' "trainer.rollout_data_dir=$_AQA_ROLLOUT_DIR" "$@" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH" \
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.max_model_len=16512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16512 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16512 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16512 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16512 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    "actor_rollout_ref.model.override_config.attn_implementation=$_AQA_ATTN" \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.disable_log_stats=false \
    reward.reward_model.enable=true reward.reward_model.enable_resource_pool=true \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=\"$MINICPMO_MM_PROMPT_BUDGET\""
