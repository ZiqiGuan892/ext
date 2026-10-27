#!/usr/bin/env bash
# STEM text-only training on an already prepared two-node, 32-NPU Ray cluster.
# Run ONCE on the head node. No environment setup, installs or Ray start/stop.
# Actor16 + GenRM16, TP4; frozen media, full precision, static microbatch1.
# Lengths are token LIMITS (1K=1024), including the rendered chat prompt:
#   bash run_huanyu_stem.sh                           # 2048 + 3072 = 5K
#   STEM_CONTEXT_LENGTH=12k bash run_huanyu_stem.sh   # 2048 + 10240
#   STEM_CONTEXT_LENGTH=16k STEM_PROMPT_LENGTH=4096 bash run_huanyu_stem.sh
#   TOTAL_TRAINING_STEPS=3 STEM_IGNORE_EOS=true bash run_huanyu_stem.sh
# STEM_DATA_DIR contains train.parquet and val.parquet (converter defaults).
# Override individual files with STEM_TRAIN_FILE / STEM_VAL_FILE.
# MODEL_PATH / RM_MODEL_PATH override checkpoints; other Hydra args are accepted.
# This entry owns data, length, pools, media freeze and microbatch settings;
# change those through the variables above, not conflicting Hydra overrides.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
[[ "$PROJECT_ROOT" == /* ]] || { echo 'FATAL: PROJECT_ROOT must be absolute' >&2; exit 2; }
_STEM_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _file in run_huanyu.sh run_output.sh graph_mode.sh precision_mode.sh; do
    [[ -f "$_STEM_EXAMPLES/$_file" ]] || { echo "FATAL: missing $_STEM_EXAMPLES/$_file" >&2; exit 2; }
done

_stem_tokens() {
    local value="${1,,}" factor=1
    if [[ "$value" == *k ]]; then value="${value%k}"; factor=1024; fi
    [[ "$value" =~ ^[1-9][0-9]{0,6}$ ]] || {
        echo "FATAL: invalid token budget '$1'; use a positive integer or e.g. 12k" >&2; return 2;
    }
    printf '%s' "$((value * factor))"
}
_STEM_CONTEXT="$(_stem_tokens "${STEM_CONTEXT_LENGTH:-5k}")"
_STEM_PROMPT="$(_stem_tokens "${STEM_PROMPT_LENGTH:-2048}")"
export MAX_PROMPT_LENGTH="$_STEM_PROMPT"
(( MAX_PROMPT_LENGTH < _STEM_CONTEXT )) || {
    echo 'FATAL: STEM_PROMPT_LENGTH must be less than STEM_CONTEXT_LENGTH' >&2; exit 2;
}
export MAX_RESPONSE_LENGTH="$((_STEM_CONTEXT - MAX_PROMPT_LENGTH))"
export ROLLOUT_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MINICPMO_MM_PROMPT_BUDGET="$MAX_PROMPT_LENGTH"
export ROLLOUT_MAX_MODEL_LEN="$((_STEM_CONTEXT + 128))" MAX_TOKEN_LEN_PER_GPU="$((_STEM_CONTEXT + 128))"
_STEM_IGNORE_EOS="${STEM_IGNORE_EOS:-false}"
case "$_STEM_IGNORE_EOS" in true|false) ;; *) echo 'FATAL: STEM_IGNORE_EOS must be true or false' >&2; exit 2 ;; esac
_STEM_STEPS="${TOTAL_TRAINING_STEPS:-null}"
case "$_STEM_STEPS" in -1|null) _STEM_STEPS=null ;; *)
    [[ "$_STEM_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: TOTAL_TRAINING_STEPS must be positive, -1 or null' >&2; exit 2; } ;;
esac

# Dedicated STEM variables prevent an inherited AQA TRAIN_FILE from taking over.
_STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
_STEM_TRAIN="${STEM_TRAIN_FILE:-$_STEM_DATA_DIR/train.parquet}"
_STEM_VAL="${STEM_VAL_FILE:-$_STEM_DATA_DIR/val.parquet}"
if [[ -z "${STEM_VAL_FILE:-}" && ! -f "$_STEM_VAL" && -f "$_STEM_DATA_DIR/validation.parquet" ]]; then
    _STEM_VAL="$_STEM_DATA_DIR/validation.parquet"
fi
for _file in "$_STEM_TRAIN" "$_STEM_VAL"; do
    [[ -r "$_file" ]] || { echo "FATAL: unreadable STEM parquet: $_file" >&2; exit 2; }
done
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
[[ "$RAY_ADDRESS" != local ]] || { echo 'FATAL: use the existing Ray cluster (auto or head_ip:port)' >&2; exit 2; }
_STEM_RAY_TIMEOUT="${HUANYU_RAY_STATUS_TIMEOUT:-30}"
[[ "$_STEM_RAY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: HUANYU_RAY_STATUS_TIMEOUT must be positive' >&2; exit 2; }
source "$_STEM_EXAMPLES/run_output.sh"
if ! timeout --foreground "${_STEM_RAY_TIMEOUT}s" ray status --address="$RAY_ADDRESS"; then
    echo 'FATAL: existing Ray is not reachable; prepare both 16-NPU nodes first' >&2; exit 1
fi

# GenRM reads the question from extra_info, not prompt. Keep original columns
# and metadata, adding question to per-run copies on the shared project mount.
# The source parquets are never overwritten; all workers need this run directory.
python3 - "$_STEM_TRAIN" "$_STEM_VAL" "$HUANYU_RUN_DIR/stem_data" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

destination = Path(sys.argv[3])
destination.mkdir(parents=True, exist_ok=True)
for split, source in zip(("train", "validation"), sys.argv[1:3]):
    table = pq.read_table(source)
    missing = {"data_source", "prompt", "reward_model", "extra_info"} - set(table.column_names)
    media = {"images", "videos", "audios"} & set(table.column_names)
    if missing or media or not table.num_rows:
        raise ValueError("%s: missing=%s media=%s rows=%s; expected nonempty text-only STEM data"
                         % (source, sorted(missing), sorted(media), table.num_rows))
    metadata = []
    for index, row in enumerate(table.select(["prompt", "reward_model", "extra_info"]).to_pylist()):
        prompt, reward, extra = row["prompt"], row["reward_model"], row["extra_info"]
        if (not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict)
                or prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str)
                or not prompt[0]["content"].strip() or not isinstance(reward, dict)
                or not isinstance(reward.get("ground_truth"), str) or not reward["ground_truth"].strip()
                or not isinstance(extra, dict)):
            raise ValueError("%s row %d: expected one user text prompt, string ground_truth and extra_info" % (source, index))
        metadata.append(dict(extra, question=prompt[0]["content"]))
    old_field = table.schema.field("extra_info")
    fields = [field for field in old_field.type if field.name != "question"]
    extra_type = pa.struct(fields + [pa.field("question", pa.string())])
    table = table.set_column(table.schema.get_field_index("extra_info"),
                             pa.field("extra_info", extra_type, metadata=old_field.metadata),
                             pa.array(metadata, type=extra_type))
    output = destination / (split + ".parquet")
    pq.write_table(table, output)
    print("[huanyu-stem] split=%s rows=%d source=%s prepared=%s question=prompt[0].content"
          % (split, table.num_rows, source, output), flush=True)
PY
export TRAIN_FILE="$HUANYU_RUN_DIR/stem_data/train.parquet" VAL_FILE="$HUANYU_RUN_DIR/stem_data/validation.parquet"

# Override stale 24+8, AQA and debug settings inherited from shared env files.
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 RM_ENABLE=1
export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 PPO_MICRO_BATCH_SIZE_PER_GPU=1
export DEBUG_MODE=0 HUANYU_SKIP_EDITABLE_INSTALL=1 HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_AQA_AUDIO_TRAIN=0 HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0 Q36_BACKWARD_PREFETCH=off
export FILTER_OVERLONG=true WANDB_MODE="${WANDB_MODE:-offline}"
# Math uses binary answer-equivalence judging. The parquet's style=rule does
# not select a rule scorer when this custom GenRM function is configured.
export GENRM_SCORING_MODE=verdict GENRM_FALLBACK_ON_ERROR=zero
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_stem_${_STEM_CONTEXT}_16_16}"
if [[ "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then unset DEFAULT_LOCAL_DIR; fi
if [[ "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then unset TENSORBOARD_DIR; fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
source "$_STEM_EXAMPLES/graph_mode.sh"
source "$_STEM_EXAMPLES/precision_mode.sh" "$@" actor_rollout_ref.rollout.tensor_model_parallel_size=4

# Explicit runtime_env also reaches workers spawned by an older Ray daemon.
_STEM_ENV_ARGS=()
for _key in HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_USE_PRECOMPUTE MINICPMO_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH MINICPMO_MM_PROMPT_BUDGET \
    GENRM_SCORING_MODE GENRM_FALLBACK_ON_ERROR GENRM_DUMP_DIR WANDB_MODE; do
    _value="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
    _STEM_ENV_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=$_value")
done
echo "[huanyu-stem] actor=16x1 RM=16x1 actor_FSDP=16 rollout_TP=4 replicas=4 RM_TP=4 replicas=4"
echo "[huanyu-stem] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$_STEM_CONTEXT engine=$ROLLOUT_MAX_MODEL_LEN ignore_eos=$_STEM_IGNORE_EOS steps=$_STEM_STEPS"
echo "[huanyu-stem] text_only=true encoder_train=false projector_train=false cache=off precision=full graph=0 dynamic_bsz=false micro_batch=1 sync_each_microbatch=true reward=GenRM/verdict"

exec bash "$_STEM_EXAMPLES/run_huanyu.sh" \
    'trainer.logger=[console]' trainer.project_name=huanyu_stem \
    "trainer.total_training_steps=$_STEM_STEPS" \
    "trainer.test_freq=${TEST_FREQ:-10}" "trainer.val_before_train=${VAL_BEFORE_TRAIN:-true}" \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}" \
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR:-$HUANYU_RUN_DIR/validation_dump}" "$@" \
    "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE" data.train_batch_size=4 \
    data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=8 \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH" \
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.ignore_eos=$_STEM_IGNORE_EOS" \
    ++actor_rollout_ref.model.override_config.huanyu_freeze_audio=true \
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=false \
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=false \
    ++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.max_num_seqs=4 actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    reward.reward_model.enable=true reward.reward_model.enable_resource_pool=true \
    reward.reward_model.n_gpus_per_node=16 reward.reward_model.nnodes=1 \
    reward.reward_model.rollout.tensor_model_parallel_size=4 \
    trainer.n_gpus_per_node=16 trainer.nnodes=1 "${_STEM_ENV_ARGS[@]}"
