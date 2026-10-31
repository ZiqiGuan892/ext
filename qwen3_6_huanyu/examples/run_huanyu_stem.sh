#!/usr/bin/env bash
# STEM text-only training on an already prepared Ray cluster.
# Run ONCE on the head node. No environment setup, installs or Ray start/stop.
# Direct defaults: Actor16 + GenRM16, TP4; frozen media, dynamic token batches.
# run_huanyu_stem_job.sh selects 128/256 Actor cards and CPU rule reward instead.
# Defaults: no initial validation; engine-default gradient synchronization.
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
# Dynamic batching and sync_each_microbatch defaults accept Hydra overrides.
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
_STEM_SLACK="${STEM_CONTEXT_SLACK:-128}"
[[ "$_STEM_SLACK" =~ ^(0|[1-9][0-9]{0,4})$ ]] || { echo 'FATAL: invalid STEM_CONTEXT_SLACK' >&2; exit 2; }
export ROLLOUT_MAX_MODEL_LEN="$((_STEM_CONTEXT + _STEM_SLACK))" MAX_TOKEN_LEN_PER_GPU="$((_STEM_CONTEXT + _STEM_SLACK))"
export STEM_REWARD_MODE="${STEM_REWARD_MODE:-genrm}"
case "$STEM_REWARD_MODE" in rule|genrm) ;; *) echo 'FATAL: STEM_REWARD_MODE must be rule or genrm' >&2; exit 2 ;; esac
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
if [[ "$STEM_REWARD_MODE" == rule ]]; then
    # Import/test the actual installed scorer before allocating model memory.
    python3 - <<'PY'
import inspect
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_dapo
def score(text):
    result = default_compute_score(data_source="math_dapo", solution_str=text, ground_truth="4")
    return float(result["score"] if isinstance(result, dict) else result)
good, bad = score(r"\boxed{4}"), score(r"\boxed{5}")
if not good > bad:
    raise RuntimeError("math_dapo rule scorer smoke check failed: correct=%s wrong=%s" % (good, bad))
print("[huanyu-stem] rule=math_dapo source=%s correct_score=%s wrong_score=%s"
      % (inspect.getsourcefile(math_dapo), good, bad), flush=True)
PY
fi
python3 - "$_STEM_TRAIN" "$_STEM_VAL" "$HUANYU_RUN_DIR/stem_data" "$STEM_REWARD_MODE" <<'PY'
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
    if sys.argv[4] == "rule" and set(table.column("data_source").to_pylist()) != {"math_dapo"}:
        raise ValueError("%s: rule mode requires data_source=math_dapo for every row" % source)
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
if [[ "${STEM_SCALED_JOB:-0}" == 1 ]]; then
    [[ "$STEM_REWARD_MODE" == rule && "${NUM_NPUS:-}" == 16 && "${NNODES:-}" =~ ^(8|16)$ ]] || {
        echo 'FATAL: scaled STEM requires rule reward and 8/16 nodes of 16 NPUs; use run_huanyu_stem_job.sh' >&2; exit 2;
    }
    for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N ROLLOUT_TP ROLLOUT_MAX_NUM_SEQS; do
        [[ "${!_key:-}" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "FATAL: invalid $_key" >&2; exit 2; }
    done
    (( 16 % ROLLOUT_TP == 0 && TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 &&
       PPO_MINI_BATCH_SIZE * ROLLOUT_N % (NUM_NPUS * NNODES) == 0 )) || {
        echo 'FATAL: TP must divide 16; batch must divide into minibatches; trajectories must divide Actor DP' >&2; exit 2;
    }
    export RM_ENABLE=0 REWARD_POOL_NPUS=0 RM_NNODES=0 RM_N_GPUS_PER_NODE=16 RM_TP=4 RM_SPLIT_NODES=0
else
    export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 RM_ENABLE=1
    export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1
    export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 ROLLOUT_MAX_NUM_SEQS=4
    if [[ "$STEM_REWARD_MODE" == rule ]]; then export RM_ENABLE=0 REWARD_POOL_NPUS=0 RM_NNODES=0; fi
fi
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
_STEM_DYNAMIC="${STEM_DYNAMIC_BSZ:-true}"
_STEM_SYNC="${STEM_SYNC_EACH_MICROBATCH:-false}"
for _flag in "$_STEM_DYNAMIC" "$_STEM_SYNC"; do
    case "$_flag" in true|false) ;; *) echo 'FATAL: STEM batching/sync flags must be true or false' >&2; exit 2 ;; esac
done
export DEBUG_MODE=0 HUANYU_SKIP_EDITABLE_INSTALL=1 HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_AQA_AUDIO_TRAIN=0 HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
# Pure text has identical RoPE axes. Transport their common 1D positions so
# TQ/verl dynamic batching does not confuse the four axes with token lengths.
export HUANYU_STEM_TEXT_TRAIN=1
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0 Q36_BACKWARD_PREFETCH=off
export FILTER_OVERLONG=true WANDB_MODE="${WANDB_MODE:-offline}"
# Legacy GenRM defaults. Rule mode explicitly disables its resource pool and
# custom scorer below; the parquet style alone does not select the reward path.
export GENRM_SCORING_MODE=verdict GENRM_FALLBACK_ON_ERROR=zero
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_stem_${_STEM_CONTEXT}_16_16}"
if [[ "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then unset DEFAULT_LOCAL_DIR; fi
if [[ "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then unset TENSORBOARD_DIR; fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}"
source "$_STEM_EXAMPLES/graph_mode.sh"
source "$_STEM_EXAMPLES/precision_mode.sh" "$@" "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"

# Explicit runtime_env also reaches workers spawned by an older Ray daemon.
_STEM_ENV_ARGS=()
for _key in HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_STEM_TEXT_TRAIN HUANYU_USE_PRECOMPUTE MINICPMO_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH MINICPMO_MM_PROMPT_BUDGET \
    GENRM_SCORING_MODE GENRM_FALLBACK_ON_ERROR GENRM_DUMP_DIR WANDB_MODE; do
    _value="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
    _STEM_ENV_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=$_value")
done
echo "[huanyu-stem] actor=${NUM_NPUS}x${NNODES} actor_DP/FSDP=$((NUM_NPUS * NNODES)) train_TP=1 SP=1 rollout_TP=$ROLLOUT_TP replicas=$((NUM_NPUS * NNODES / ROLLOUT_TP)) reward=$STEM_REWARD_MODE RM_cards=$REWARD_POOL_NPUS"
echo "[huanyu-stem] BS=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE N=$ROLLOUT_N trajectories=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) microbatch=1 microbatches_per_rank=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N / (NUM_NPUS * NNODES)))"
echo "[huanyu-stem] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$_STEM_CONTEXT engine=$ROLLOUT_MAX_MODEL_LEN ignore_eos=$_STEM_IGNORE_EOS steps=$_STEM_STEPS"
echo "[huanyu-stem] text_only=true encoder_train=false projector_train=false cache=off precision=full graph=0 reward=$STEM_REWARD_MODE"
echo "[huanyu-stem] HUANYU_STEM_TEXT_TRAIN=$HUANYU_STEM_TEXT_TRAIN position_transport=1d model_positions=4_axes"
echo "[huanyu-stem] defaults: val_before_train=${VAL_BEFORE_TRAIN:-false} test_freq=${TEST_FREQ:-10} actor/rollout/ref.dynamic_bsz=$_STEM_DYNAMIC token_budget=$MAX_TOKEN_LEN_PER_GPU sync_each_microbatch=$_STEM_SYNC (trailing Hydra overrides take precedence)"
_STEM_REWARD_ARGS=(reward.reward_model.enable=true reward.reward_model.enable_resource_pool=true
    "reward.reward_model.n_gpus_per_node=$RM_N_GPUS_PER_NODE" "reward.reward_model.nnodes=$RM_NNODES"
    "reward.reward_model.rollout.tensor_model_parallel_size=$RM_TP")
if [[ "$STEM_REWARD_MODE" == rule ]]; then
    _STEM_REWARD_ARGS=(reward.reward_model.enable=false reward.reward_model.enable_resource_pool=false
        reward.custom_reward_function.path=null reward.reward_manager.source=register reward.reward_manager.name=naive)
fi
_STEM_ROLLOUT_ARGS=()
if [[ "${STEM_SCALED_JOB:-0}" == 1 ]]; then
    # The backend derives a small worker count from replicas. Preserve the
    # explicit producer concurrency so larger jobs can feed all engines.
    _STEM_ROLLOUT_ARGS=("actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS:-128}"
        actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.pipeline_model_parallel_size=1)
fi

# These defaults precede user arguments; length and resource constraints follow.
exec bash "$_STEM_EXAMPLES/run_huanyu.sh" \
    'trainer.logger=[console]' trainer.project_name=huanyu_stem \
    "trainer.total_training_steps=$_STEM_STEPS" \
    "trainer.test_freq=${TEST_FREQ:-10}" "trainer.val_before_train=${VAL_BEFORE_TRAIN:-false}" \
    "actor_rollout_ref.actor.use_dynamic_bsz=$_STEM_DYNAMIC" \
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$_STEM_DYNAMIC" \
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$_STEM_DYNAMIC" \
    "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=$_STEM_SYNC" \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR:-$HUANYU_RUN_DIR/rollout_dump}" \
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR:-$HUANYU_RUN_DIR/validation_dump}" "$@" \
    "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE" "data.train_batch_size=$TRAIN_BATCH_SIZE" \
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
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS" actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    "${_STEM_REWARD_ARGS[@]}" "${_STEM_ROLLOUT_ARGS[@]}" \
    "trainer.n_gpus_per_node=$NUM_NPUS" "trainer.nnodes=$NNODES" "${_STEM_ENV_ARGS[@]}"
