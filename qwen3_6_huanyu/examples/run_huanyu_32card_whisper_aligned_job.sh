#!/usr/bin/env bash
# Frozen Huanyu comparison against the supplied corrected Whisper verl command.
# Changes from the existing 32-card Huanyu defaults (not a 16K profile):
#   attention: eager -> sdpa; Actor dtype: fp32 -> bfloat16
#   train batch: 32 (unchanged); PPO mini batch: 32 -> 16; microbatch: 1
#   rollout.n: 4 -> 8; rollout TP: 4 -> 2; actor replicas/workers: 4 -> 8
#   repetition_penalty: 1.05 -> 1.0; logprobs_mode: explicitly raw_logprobs
#   rollout memory utilization: 0.5 -> 0.65; max_num_seqs: 64 -> 4
#   weight update bucket: 1024 -> 4096 MiB; CPU offload_policy: false -> true
#   filter_overlong_prompts: false -> true; truncation stays error
#   dynamic batching, gradient checkpointing, GSPO and eager rollout stay on.
#
# Required Huanyu differences: padded [B,S] forward (remove_padding=false),
# Huanyu model/dataset/pipeline/worker hooks, frozen audio + Huanyu embedding
# cache, and existing GenRM configuration. The reference did not provide RM
# settings, actual length variables, or the expanded _DATASET/_CHAT arrays.
# Keep the existing coherent 5K budget: prompt3072 + response2048, rollout
# prompt3104, engine/token budget5248. Do not copy its inconsistent literal
# fallbacks max_model_len4096 / prompt_length4160 into an actual run.
# BF16 storage follows the supplied comparison, not a claim of better training
# accuracy; ACTOR_MODEL_DTYPE=fp32 selects the existing master-weight dtype.
#
# Run this SAME command on both nodes; VC_TASK_INDEX=0 starts training and 1
# only joins Ray. Reuse installation/native checks/locks/logging of the original
# job. This wrapper may live anywhere; helpers come from PROJECT_ROOT.
#   bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_32card_whisper_aligned_job.sh"
# Optional env overrides use the same names as the supplied Whisper command.
# Keep related length values consistent when overriding individual Hydra keys.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
if [[ "$PROJECT_ROOT" != /* ]]; then
    echo 'FATAL: PROJECT_ROOT must be an absolute training-project path' >&2
    exit 2
fi
_HUANYU_WHISPER_JOB="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh"
export HUANYU_JOB_PROFILE=whisper_aligned HUANYU_JOB_MODE=frozen_cache
export NUM_NPUS=16 NNODES=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1
export REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1
export ROLLOUT_TP="${ROLLOUT_TP:-2}" RM_TP="${RM_TP:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

for _key in ROLLOUT_TP RM_TP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE_PER_GPU ROLLOUT_N MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]{0,6}$ ]]; then
        echo "FATAL: $_key must be a positive integer without leading zeroes" >&2
        exit 2
    fi
done
export ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + 32))}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128))}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}"
for _key in ROLLOUT_PROMPT_LENGTH ROLLOUT_MAX_MODEL_LEN MAX_TOKEN_LEN_PER_GPU; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]{0,6}$ ]]; then
        echo "FATAL: $_key must be a positive integer without leading zeroes" >&2
        exit 2
    fi
done
if (( ROLLOUT_PROMPT_LENGTH < MAX_PROMPT_LENGTH ||
      ROLLOUT_MAX_MODEL_LEN < ROLLOUT_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ||
      MAX_TOKEN_LEN_PER_GPU < ROLLOUT_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
    echo 'FATAL: rollout prompt must cover data prompt; engine and token budgets must cover rollout prompt + response' >&2
    exit 2
fi
if (( NUM_NPUS % ROLLOUT_TP || RM_N_GPUS_PER_NODE % RM_TP ||
      TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE ||
      (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % NUM_NPUS ||
      (TRAIN_BATCH_SIZE * ROLLOUT_N) % NUM_NPUS ||
      (PPO_MINI_BATCH_SIZE * ROLLOUT_N / NUM_NPUS) % PPO_MICRO_BATCH_SIZE_PER_GPU )); then
    echo 'FATAL: TP must divide its 16-card pool; train/mini/micro batches must divide evenly across actor ranks' >&2
    exit 2
fi
case "${HUANYU_GRAPH_MODE:-0}" in
    0|false) export HUANYU_GRAPH_MODE=0 ;;
    *) echo 'FATAL: the Whisper-aligned comparison requires eager rollout; set HUANYU_GRAPH_MODE=0' >&2; exit 2 ;;
esac
export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bfloat16}"
export REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bfloat16}"
export ACTOR_FORWARD_PREFETCH="${ACTOR_FORWARD_PREFETCH:-false}"
export ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-false}"
export ACTOR_OFFLOAD_POLICY=true FILTER_OVERLONG=true
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}" ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
export ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.0}"
export ROLLOUT_FREQUENCY_PENALTY="${ROLLOUT_FREQUENCY_PENALTY:-0.0}"
export ROLLOUT_PRESENCE_PENALTY="${ROLLOUT_PRESENCE_PENALTY:-0.0}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
export MINICPMO_MM_PROMPT_BUDGET="$ROLLOUT_PROMPT_LENGTH"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_whisper_aligned_32card}"
[[ -f "$_HUANYU_WHISPER_JOB" ]] || {
    echo "FATAL: missing existing job launcher: $_HUANYU_WHISPER_JOB" >&2
    exit 1
}

echo "[huanyu-whisper-aligned] actor16 + RM16; rollout_TP=$ROLLOUT_TP replicas=$((16 / ROLLOUT_TP)); RM_TP=$RM_TP"
echo "[huanyu-whisper-aligned] batch=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BATCH_SIZE micro=$PPO_MICRO_BATCH_SIZE_PER_GPU rollout_n=$ROLLOUT_N actor_dtype=$ACTOR_MODEL_DTYPE ref_dtype=$REF_MODEL_DTYPE attention=sdpa"
echo "[huanyu-whisper-aligned] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH rollout_prompt=$ROLLOUT_PROMPT_LENGTH engine=$ROLLOUT_MAX_MODEL_LEN token_budget=$MAX_TOKEN_LEN_PER_GPU"
echo "[huanyu-whisper-aligned] logprobs=$ROLLOUT_LOGPROBS_MODE repetition=$ROLLOUT_REPETITION_PENALTY memory=$ROLLOUT_GPU_MEM_UTIL max_seqs=4 bucket_MiB=4096 filter=true offload_policy=true"
echo '[huanyu-whisper-aligned] retained Huanyu differences: adapter/dataset/pipeline, remove_padding=false, frozen cache, GenRM and precision hooks'

# Explicit final Hydra arguments override the inherited OmniMix defaults. The
# original job/entry still owns mandatory Huanyu hooks and output directories.
# Caller Hydra overrides follow this profile; graph and frozen-audio guards
# remain owned by the base job. ++ supports both schemas with/without the new
# logprobs_mode key; the server must implement raw_logprobs to honor its meaning.
exec bash "$_HUANYU_WHISPER_JOB" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    "actor_rollout_ref.model.enable_activation_offload=$ACTIVATION_OFFLOAD" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=$ACTOR_FORWARD_PREFETCH" \
    "actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_MODEL_DTYPE" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=true \
    "actor_rollout_ref.ref.fsdp_config.model_dtype=$REF_MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
    "actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE" \
    "++actor_rollout_ref.rollout.logprobs_mode=$ROLLOUT_LOGPROBS_MODE" \
    "actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P" \
    "actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K" \
    "++actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY" \
    "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=$ROLLOUT_FREQUENCY_PENALTY" \
    "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=$ROLLOUT_PRESENCE_PENALTY" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH" \
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    "actor_rollout_ref.rollout.agent.num_workers=$((16 / ROLLOUT_TP))" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=\"${MINICPMO_MM_PROMPT_BUDGET}\"" \
    "$@"
