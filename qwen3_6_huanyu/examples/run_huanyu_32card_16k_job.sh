#!/usr/bin/env bash
# Frozen Huanyu + cached sampleall + GenRM, 16K prompt/response budget.
# Configuration changes from the original frozen 32-card defaults:
#   train attention: eager -> sdpa
#   data prompt: 3072 -> 14336; response stays 2048; total: 5120 -> 16384
#   rollout prompt: 3104 -> 14368
#   rollout max_model_len / max_num_batched_tokens: 5248 -> 16512
#   actor PPO / rollout-logprob / ref-logprob token budgets: 5248 -> 16512
#   actor/ref/rollout-logprob: dynamic batching -> static microbatch=1
#   actor FSDP2 CPU offload_policy: false -> true; reshard_after_forward=true
#   Huanyu backward: synchronize/shard gradients after EVERY microbatch
#   actor/ref forward_prefetch=false; backward_prefetch=off; CPU pin_memory=0
#   MINICPMO_MM_PROMPT_BUDGET: 3104 -> 14368 (also passed to Ray workers)
#   rollout max_num_seqs: 64 -> 32
#   rollout disable_log_stats: true -> false (enable throughput/cache stats)
#   default experiment: huanyu_sampleall_cache_32card -> huanyu_16k_sampleall_cache_32card
# Gradient checkpointing stays enabled. Global/mini batch, TP, sampling, GSPO,
# FP32 master weights, GDN precision, frozen cache and GenRM are retained.
# CPU offload trades throughput for headroom; activation offload stays disabled.
#
# Set PROJECT_ROOT to the absolute training-project path. This 16K script may
# live anywhere; its backend always comes from PROJECT_ROOT/examples/qwen3_6_huanyu.
# Example: PROJECT_ROOT=/path/to/project bash /path/to/this/script.sh
# Run this SAME script on both nodes; the existing job launcher owns setup,
# Ray head/worker selection (VC_TASK_INDEX=0/1), and head-only training.
#
# Defaults: prompt=14336, response=2048 (sum=16384); engine=16512 includes
# the existing 128-token safety margin. Media tokens count toward the prompt.
# Optional environment knobs:
#   HUANYU_16K_RESPONSE_LENGTH=4096        # prompt becomes 12288; total stays 16K
#   HUANYU_16K_ATTN_IMPLEMENTATION=eager   # compare against the SDPA default
#   HUANYU_16K_MAX_NUM_SEQS=16             # rollout concurrency (default 32)
# HUANYU_GRAPH_MODE retains its existing meaning/default (inference eager).
# Additional Hydra arguments are forwarded last and can override this profile;
# use HUANYU_16K_RESPONSE_LENGTH to change the prompt/response split together.
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" || "$PROJECT_ROOT" != /* ]]; then
    echo 'FATAL: set PROJECT_ROOT to the absolute training-project path' >&2
    exit 2
fi
export PROJECT_ROOT
export HUANYU_JOB_MODE=frozen_cache
_HUANYU_16K_JOB="${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh"
_HUANYU_16K_RESPONSE="${HUANYU_16K_RESPONSE_LENGTH:-2048}"
_HUANYU_16K_ATTN="${HUANYU_16K_ATTN_IMPLEMENTATION:-sdpa}"
_HUANYU_16K_SEQS="${HUANYU_16K_MAX_NUM_SEQS:-32}"

# Validate before the delegated job performs installation or starts Ray. Reject
# leading zeroes as well, so Bash cannot interpret lengths as octal numbers.
if [[ ! "$_HUANYU_16K_RESPONSE" =~ ^[1-9][0-9]{0,4}$ ]] ||
    (( _HUANYU_16K_RESPONSE >= 16384 )); then
    echo 'FATAL: HUANYU_16K_RESPONSE_LENGTH must be an integer in 1..16383' >&2
    exit 2
fi
if [[ ! "$_HUANYU_16K_SEQS" =~ ^[1-9][0-9]{0,4}$ ]]; then
    echo 'FATAL: HUANYU_16K_MAX_NUM_SEQS must be an integer in 1..99999' >&2
    exit 2
fi
case "$_HUANYU_16K_ATTN" in
    sdpa|eager) ;;
    *) echo 'FATAL: HUANYU_16K_ATTN_IMPLEMENTATION must be sdpa or eager' >&2; exit 2 ;;
esac
[[ -f "$_HUANYU_16K_JOB" ]] || {
    echo "FATAL: missing existing job launcher: $_HUANYU_16K_JOB" >&2
    exit 1
}

# Deliberately replace inherited 5K length settings. Exports keep shell-side
# checks consistent; the explicit Hydra arguments below defeat mix-wrapper
# defaults such as attention=eager and max_num_seqs=64.
export MAX_RESPONSE_LENGTH="$_HUANYU_16K_RESPONSE"
export MAX_PROMPT_LENGTH="$((16384 - MAX_RESPONSE_LENGTH))"
export ROLLOUT_PROMPT_LENGTH="$((MAX_PROMPT_LENGTH + 32))"
export ROLLOUT_MAX_MODEL_LEN=16512 MAX_TOKEN_LEN_PER_GPU=16512
export MINICPMO_MM_PROMPT_BUDGET="$ROLLOUT_PROMPT_LENGTH"
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 ACTOR_OFFLOAD_POLICY=true
export MINICPMO_FSDP2_PIN_MEMORY=0 Q36_BACKWARD_PREFETCH=off
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_16k_sampleall_cache_32card}"

# Huanyu uses padded [B,S] training inputs. Dynamic batching budgets effective
# tokens, not B*max(S), and ignores ppo_micro_batch_size_per_gpu. Static size 1
# prevents padding inflation without shortening a single trajectory. The
# opt-in Huanyu sync guard prevents unsharded gradients from accumulating over
# those microbatches, including verl versions without the newer config field.
# CPUOffloadPolicy removes resident shard/gradient/Adam-state pressure during
# backward; optimizer_offload alone reloads Adam states at train-mode entry.
# Keep activation offload disabled: this server previously failed in its
# activation tensor_pop path, and it changes the checkpoint implementation.

# Preserve the existing batch size, rollout.n, TP, master dtype, GDN, GenRM,
# cache validation and graph-mode policy. Chunked prefill remains off until
# its interaction with this server's Huanyu/GDN implementation is validated.
# SDPA selects the HF backend; actual NPU fusion and numerical equivalence need
# a server-side smoke test. This launcher does not change GDN precision.
echo "[huanyu-16k] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=16384 engine=$ROLLOUT_MAX_MODEL_LEN media_budget=$MINICPMO_MM_PROMPT_BUDGET"
echo "[huanyu-16k] train_attention=$_HUANYU_16K_ATTN token_budget/gpu=$MAX_TOKEN_LEN_PER_GPU rollout_max_num_seqs=$_HUANYU_16K_SEQS graph_mode=${HUANYU_GRAPH_MODE:-0}"
echo '[huanyu-16k] memory: static_microbatch=1 sync_each_microbatch=true fsdp2_cpu_offload=true activation_offload=false; global/mini batch and rollout.n unchanged'

exec bash "$_HUANYU_16K_JOB" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH" \
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" \
    "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_MODEL_LEN" \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.model.override_config.attn_implementation=$_HUANYU_16K_ATTN" \
    ++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=false \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=false \
    "actor_rollout_ref.rollout.max_num_seqs=$_HUANYU_16K_SEQS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.disable_log_stats=false \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=\"${MINICPMO_MM_PROMPT_BUDGET}\"" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_FSDP2_PIN_MEMORY="0"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.Q36_BACKWARD_PREFETCH="off"' \
    "$@"
