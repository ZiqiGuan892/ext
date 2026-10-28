#!/usr/bin/env bash
# Diagnostic training on 100 validation rows using the SAME exported WAV files
# as serving. Run once on the head AFTER environment preparation and Ray setup.
# This entry does not install packages, replace code, or start/stop Ray.
# Train and validation deliberately use the same subset for before/after audio
# comparison; their rewards are not an independent held-out evaluation.
# Override source/export paths with AQA_WAV_SOURCE_PARQUET / AQA_WAV_EXPORT_DIR,
# selection with AQA_WAV_SAMPLE_SEED, checkpoints with MODEL_PATH / RM_MODEL_PATH,
# and shorten the run with TOTAL_TRAINING_STEPS. Fixed diagnostic options below
# follow caller Hydra options; use the ordinary AQA entry for other settings.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_wav_examples="$PROJECT_ROOT/examples/qwen3_6_huanyu"
for _wav_file in run_output.sh prepare_aqa_wav_parquet.py run_huanyu_aqa_audio_train.sh; do
    [[ -r "$_wav_examples/$_wav_file" ]] || { echo "FATAL: missing $_wav_examples/$_wav_file" >&2; exit 2; }
done
source "$_wav_examples/run_output.sh"

_wav_source="${AQA_WAV_SOURCE_PARQUET:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}"
_wav_export="${AQA_WAV_EXPORT_DIR:-/opt/huawei/dataset/infra_data/data/validation_vllm}"
_wav_subset="$HUANYU_RUN_DIR/wav100/train.parquet"
python3 "$_wav_examples/prepare_aqa_wav_parquet.py" \
    --parquet "$_wav_source" --export-dir "$_wav_export" \
    --output-parquet "$_wav_subset" --sample-size 100 --seed "${AQA_WAV_SAMPLE_SEED:-42}"

# Dedicated overrides defeat stale datasets inherited from env_export.sh.
export TRAIN_FILE="$_wav_subset" VAL_FILE="$_wav_subset"
export HUANYU_AQA_TRAIN_FILE="$_wav_subset" HUANYU_AQA_VAL_FILE="$_wav_subset"
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}"
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 RM_ENABLE=1
export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0 DEBUG_MODE=0 DEBUG_SAMPLES=0
export ROLLOUT_TEMPERATURE=0 ROLLOUT_TOP_P=1 ROLLOUT_TOP_K=-1
export ROLLOUT_REPETITION_PENALTY=1 ROLLOUT_FREQUENCY_PENALTY=0 ROLLOUT_PRESENCE_PENALTY=0
export MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 ROLLOUT_PROMPT_LENGTH=3072
export MINICPMO_MM_PROMPT_BUDGET=3072 ROLLOUT_MAX_MODEL_LEN=5248 MAX_TOKEN_LEN_PER_GPU=5248
export VAL_BEFORE_TRAIN=true TEST_FREQ=-1 TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_aqa_wav100_frozen}"

echo "[huanyu-wav100] selected=100 seed=${AQA_WAV_SAMPLE_SEED:-42} train=$TRAIN_FILE validation=$VAL_FILE"
echo "[huanyu-wav100] input=exported_WAV cache=off train_encoder=false train_projector=false actor=16NPU RM=16NPU TP=4"
echo "[huanyu-wav100] temperature=0 top_p=1 top_k=-1 repetition_penalty=1 ignore_eos=false prompt=3072 response=2048"
echo "[huanyu-wav100] validate_before_update=true same_subset_diagnostic=true steps=$TOTAL_TRAINING_STEPS; selection=$HUANYU_RUN_DIR/wav100/train.selection.json"

# The ordinary entry enforces static microbatch=1 and the common Huanyu audio
# execution settings. These fixed Hydra values also override stale caller args.
exec bash "$_wav_examples/run_huanyu_aqa_audio_train.sh" "$@" \
    data.shuffle=false data.seed=42 data.train_max_samples=100 data.val_max_samples=100 \
    data.max_prompt_length=3072 data.max_response_length=2048 \
    actor_rollout_ref.rollout.prompt_length=3072 actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=5248 actor_rollout_ref.rollout.max_num_batched_tokens=5248 \
    actor_rollout_ref.rollout.temperature=0 actor_rollout_ref.rollout.top_p=1 actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.repetition_penalty=1 \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens=0 \
    actor_rollout_ref.rollout.ignore_eos=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    trainer.val_before_train=true trainer.test_freq=-1 trainer.total_epochs=1
