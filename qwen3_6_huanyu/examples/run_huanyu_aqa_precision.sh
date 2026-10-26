#!/usr/bin/env bash
# Direct AQA training; source your environment, install the plugin and start
# Ray on all nodes beforehand. Run this entry ONCE on the head node.
# Defaults: trainable Huanyu, online audio, actor16 + GenRM16, TP4,
# prompt3072 + response2048 (5120 tokens; engine budget5248).
# RM_ENABLE=0 uses actor16 + the existing rule-based reward instead.
# HUANYU_FREEZE_AUDIO=true permits frozen online-audio comparisons (no cache).
# Precision is enforced: full profile, eager inference, FP32 GDN, reference
# prefill/gating/conv/L2/barrier, private hybrid cache, stable linear,
# layerwise reload, KV zeroing, isolated media and FSDP pin_memory=0.
# This entry never sources env_export, installs packages or starts/stops Ray.
# Deploy all companion scripts to PROJECT_ROOT/examples/qwen3_6_huanyu/.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_AQA_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu_aqa_audio_train.sh run_output.sh graph_mode.sh precision_mode.sh; do
    [[ -f "$_AQA_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_AQA_EXAMPLES/$_file" >&2; exit 2; }
done
source "$_AQA_EXAMPLES/run_output.sh"

export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_SKIP_EDITABLE_INSTALL=1
export HUANYU_FREEZE_AUDIO="${HUANYU_FREEZE_AUDIO:-false}"
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048
export ROLLOUT_PROMPT_LENGTH=3072 ROLLOUT_MAX_MODEL_LEN=5248
export MAX_TOKEN_LEN_PER_GPU=5248 MINICPMO_MM_PROMPT_BUDGET=3072
export RM_ENABLE="${RM_ENABLE:-1}" WANDB_MODE="${WANDB_MODE:-offline}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_precision_5k}"
# Replace the shared env file's fixed output defaults with this run's folder;
# preserve separately chosen paths and trailing trainer overrides.
if [[ "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then unset DEFAULT_LOCAL_DIR; fi
if [[ "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then unset TENSORBOARD_DIR; fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
_AQA_ROLLOUT_DIR="${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}"
source "$_AQA_EXAMPLES/graph_mode.sh"
source "$_AQA_EXAMPLES/precision_mode.sh" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4

# Explicit auto connects to an existing local cluster; unlike an unset Ray
# address, it cannot silently create a new local cluster if none is found.
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
[[ "$RAY_ADDRESS" != local ]] || { echo 'FATAL: RAY_ADDRESS=local would create a new cluster; use auto or head_ip:port' >&2; exit 2; }
_AQA_RAY_TIMEOUT="${HUANYU_RAY_STATUS_TIMEOUT:-30}"
[[ "$_AQA_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: HUANYU_RAY_STATUS_TIMEOUT must be positive seconds' >&2; exit 2; }
for _command in ray timeout; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing $_command in the prepared environment" >&2; exit 2; }
done
echo "[huanyu-aqa-precision] checking existing Ray at $RAY_ADDRESS; package installation skipped"
if ! timeout --foreground "${_AQA_RAY_TIMEOUT}s" ray status --address="$RAY_ADDRESS"; then
    echo 'FATAL: existing Ray is not reachable; prepare the cluster or use run_huanyu_aqa_precision_32card_job.sh on both nodes' >&2
    exit 1
fi
echo "[huanyu-aqa-precision] profile=full graph=0 freeze_audio=$HUANYU_FREEZE_AUDIO RM_ENABLE=$RM_ENABLE prompt=3072 response=2048"
echo "[huanyu-aqa-precision] console=$HUANYU_CONSOLE_LOG rollout=$_AQA_ROLLOUT_DIR GenRM=$GENRM_DUMP_DIR"

exec bash "$_AQA_EXAMPLES/run_huanyu_aqa_audio_train.sh" \
    data.max_prompt_length=3072 data.max_response_length=2048 \
    actor_rollout_ref.rollout.prompt_length=3072 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=5248 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5248 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5248 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=5248 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=5248 \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    '++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET="3072"' \
    'trainer.logger=[console]' \
    "trainer.rollout_data_dir=$_AQA_ROLLOUT_DIR" "$@"
