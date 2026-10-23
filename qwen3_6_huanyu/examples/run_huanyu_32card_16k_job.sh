#!/usr/bin/env bash
# Frozen Huanyu + cached sampleall + GenRM, 16K prompt/response budget.
# Configuration changes from the original frozen 32-card defaults:
#   train attention: eager -> sdpa
#   data prompt: 3072 -> 14336; response stays 2048; total: 5120 -> 16384
#   rollout prompt: 3104 -> 14368
#   rollout max_model_len / max_num_batched_tokens: 5248 -> 16512
#   actor PPO / rollout-logprob / ref-logprob token budgets: 5248 -> 16512
#   MINICPMO_MM_PROMPT_BUDGET: 3104 -> 14368 (also passed to Ray workers)
#   rollout max_num_seqs: 64 -> 32
#   rollout disable_log_stats: true -> false (enable throughput/cache stats)
#   resource pools: default stays actor16 + RM16; HUANYU_16K_POOL_LAYOUT=24+8
#     selects actor12/node x 2 + RM4/node x 2, with disjoint device pools
#   rollout/RM TP: default stays 4; replicas 4/4 -> 6/2 only in the 24+8 layout
#   train/mini batch: default stays 32/32; 24/24 in 24+8 (rollout.n stays 4)
#   FSDP uses the whole actor pool; actor/ref sequence parallel size stays 1
#   default experiment: huanyu_sampleall_cache_32card -> huanyu_16k_sampleall_cache_32card
# Gradient checkpointing and dynamic batching stay enabled. Sampling,
# GSPO, FP32 master weights, GDN precision, frozen cache and GenRM are retained.
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
#   HUANYU_16K_POOL_LAYOUT=24+8           # opt in; unset keeps 16+16
#   ROLLOUT_TP=4 RM_TP=4                # each TP group must fit one pool/node
#   TRAIN_BATCH_SIZE=48 PPO_MINI_BATCH_SIZE=48  # optional compatible batch
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

# This is the resource configuration authority for the 16K entry. The job and
# mix wrappers consume these exports, instead of restoring their old 16+16
# constants. Counts below are pool resources, not physical Ray node capacity.
export HUANYU_16K_POOL_LAYOUT="${HUANYU_16K_POOL_LAYOUT:-16+16}"
case "$HUANYU_16K_POOL_LAYOUT" in
    16+16)
        export NUM_NPUS=16 NNODES=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1
        export REWARD_POOL_NPUS=16 RM_SPLIT_NODES=1
        _HUANYU_16K_BATCH=32
        ;;
    24+8)
        # Two physical 16-card nodes: three actor TP4 replicas + one RM TP4
        # replica per node. An RM pool of [8] cannot fit beside actor [12,12].
        export NUM_NPUS=12 NNODES=2 RM_N_GPUS_PER_NODE=4 RM_NNODES=2
        export REWARD_POOL_NPUS=8 RM_SPLIT_NODES=0
        _HUANYU_16K_BATCH=24
        ;;
    *) echo 'FATAL: HUANYU_16K_POOL_LAYOUT must be 16+16 or 24+8' >&2; exit 2 ;;
esac
export ROLLOUT_TP="${ROLLOUT_TP:-4}" RM_TP="${RM_TP:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$_HUANYU_16K_BATCH}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$_HUANYU_16K_BATCH}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
for _key in ROLLOUT_TP RM_TP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]{0,4}$ ]]; then
        echo "FATAL: $_key must be an integer in 1..99999" >&2
        exit 2
    fi
done
_HUANYU_16K_WORLD=$((NUM_NPUS * NNODES))
if (( NUM_NPUS % ROLLOUT_TP != 0 || RM_N_GPUS_PER_NODE % RM_TP != 0 )); then
    echo 'FATAL: rollout/RM TP must divide its per-node pool size; TP groups must stay within one node' >&2
    exit 2
fi
if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 ||
      (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % _HUANYU_16K_WORLD != 0 ||
      (TRAIN_BATCH_SIZE * ROLLOUT_N) % _HUANYU_16K_WORLD != 0 )); then
    echo "FATAL: train batch must divide into PPO minibatches; batch * rollout.n must divide evenly across $_HUANYU_16K_WORLD actor ranks. Use TRAIN_BATCH_SIZE=$_HUANYU_16K_BATCH PPO_MINI_BATCH_SIZE=$_HUANYU_16K_BATCH ROLLOUT_N=4 for this layout." >&2
    exit 2
fi

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
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_16k_sampleall_cache_32card}"

# Preserve master dtype, GDN, GenRM,
# cache validation and graph-mode policy. Chunked prefill remains off until
# its interaction with this server's Huanyu/GDN implementation is validated.
# SDPA selects the HF backend; actual NPU fusion and numerical equivalence need
# a server-side smoke test. This launcher does not change GDN precision.
echo "[huanyu-16k] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=16384 engine=$ROLLOUT_MAX_MODEL_LEN media_budget=$MINICPMO_MM_PROMPT_BUDGET"
echo "[huanyu-16k] train_attention=$_HUANYU_16K_ATTN token_budget/gpu=$MAX_TOKEN_LEN_PER_GPU rollout_max_num_seqs=$_HUANYU_16K_SEQS graph_mode=${HUANYU_GRAPH_MODE:-0}"
echo "[huanyu-16k] pools=$HUANYU_16K_POOL_LAYOUT actor=${NUM_NPUS}x${NNODES} RM=${RM_N_GPUS_PER_NODE}x${RM_NNODES} rollout_TP=$ROLLOUT_TP replicas=$((_HUANYU_16K_WORLD / ROLLOUT_TP)) RM_TP=$RM_TP replicas=$((REWARD_POOL_NPUS / RM_TP))"
echo "[huanyu-16k] train_batch=$TRAIN_BATCH_SIZE ppo_mini_batch=$PPO_MINI_BATCH_SIZE rollout_n=$ROLLOUT_N sequences/actor_rank=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N / _HUANYU_16K_WORLD))"

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
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    "actor_rollout_ref.rollout.max_num_seqs=$_HUANYU_16K_SEQS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.disable_log_stats=false \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_MM_PROMPT_BUDGET=\"${MINICPMO_MM_PROMPT_BUDGET}\"" \
    "$@"
