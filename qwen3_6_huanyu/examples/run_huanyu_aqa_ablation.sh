#!/usr/bin/env bash
# 已准备好环境和 Ray 时运行：五组 16 卡 AQA 吞吐对照；每组默认 10 个训练 step。
# 用法：bash run_huanyu_aqa_ablation.sh all [--dry-run]
# 单组：full / kernel / graph / moe / replay；补充二分：gdn_off / rms_off。
set -euo pipefail
_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
_helper="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqa_ablation.py"
_group="${1:-all}"; (( $# == 0 )) || shift
_dry=0
if [[ "${1:-}" == --dry-run ]]; then _dry=1; shift; fi
(( $# == 0 )) || { echo 'FATAL: 参数为 group [--dry-run]，公共配置请通过环境变量指定' >&2; exit 2; }
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
[[ "$TOTAL_TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'FATAL: TOTAL_TRAINING_STEPS 必须为正整数' >&2; exit 2; }
export BENCH_ROOT="${BENCH_ROOT:-$PROJECT_ROOT/runs/aqa_ablation_$(date +%Y%m%d_%H%M%S)}"
if [[ "$_group" == all ]]; then
    if (( ! _dry )); then "$PYTHON_BIN" "$_helper" plugins; fi
    for _variant in full kernel graph moe replay; do
        if (( _dry )); then bash "${BASH_SOURCE[0]}" "$_variant" --dry-run
        else bash "${BASH_SOURCE[0]}" "$_variant"; fi
    done
    if (( ! _dry )); then "$PYTHON_BIN" "$_helper" report "$BENCH_ROOT"; fi
    exit 0
fi

# 显式覆盖上一组残留的优化开关。所有组固定 AIV、无预取、FP32 梯度归约、原生权重同步。
export HUANYU_PRECISION_PROFILE=kernel HUANYU_GRAPH_MODE=0 HUANYU_KERNEL_BISECT=none
export MINICPMO_GDN_NPU_KERNEL=0 MINICPMO_NPU_RMS_NORM=0
export MINICPMO_MOE_NPU_KERNEL=0 HUANYU_EXPERT_REPLAY=0
case "$_group" in
    full) export HUANYU_PRECISION_PROFILE=full ;;
    kernel) ;;
    graph|moe|replay) export HUANYU_PRECISION_PROFILE=kernel_graph HUANYU_GRAPH_MODE=1
        [[ "$_group" == graph ]] || export MINICPMO_MOE_NPU_KERNEL=1
        [[ "$_group" != replay ]] || export HUANYU_EXPERT_REPLAY=1 ;;
    gdn_off|rms_off) export HUANYU_KERNEL_BISECT="$_group" ;;
    *) echo "FATAL: 未知组 $_group" >&2; exit 2 ;;
esac
export HUANYU_HCCL_PROFILE=aiv HCCL_OP_EXPANSION_MODE=AIV
export HUANYU_WSYNC_PROFILE=off HUANYU_WSYNC_TRACE=off HUANYU_WSYNC_VERIFY=0 HUANYU_WSYNC_LAYOUT=passthrough
export HUANYU_WSYNC_BUCKET_MB=1024
unset VERL_ENGINE_VENDOR
export Q36_BACKWARD_PREFETCH=off ACTOR_FORWARD_PREFETCH=false REF_FORWARD_PREFETCH=false
export ACTOR_MP_PARAM_DTYPE=bfloat16 ACTOR_MP_REDUCE_DTYPE=fp32 ACTOR_MP_BUFFER_DTYPE=fp32
export MINICPMO_MOE_MATCH_ROLLOUT=0 MINICPMO_MOE_TP_REFERENCE=0 MINICPMO_ROW_TP_REFERENCE=0
export MINICPMO_HF_NORM_REFERENCE=0 MINICPMO_HF_ROUTER_REFERENCE=0 MINICPMO_HF_TEXT_ROPE_REFERENCE=0
export MINICPMO_ISOLATE_FULL_ATTENTION=0 MINICPMO_VISION_POSITION_REFERENCE=0 MINICPMO_NATIVE_VISION_REFERENCE=0
export VERL_OMNI_EXT_PATCH_STRICT=1 HUANYU_REPLAY_DEBUG=0 HUANYU_MOE_FINITE_DEBUG=0 HUANYU_FINITE_PROBE=0
export HUANYU_GEN_TIMING=0 HUANYU_JOB_DETERMINISM=0 ASCEND_LAUNCH_BLOCKING=0
export HUANYU_FORWARD_PROFILE_DIR='' HUANYU_WEIGHT_SYNC_AUDIT_DIR='' HUANYU_FULL_WEIGHT_AUDIT_DIR=''
export HUANYU_EMBEDDING_AB_DIR='' MINICPMO_ROLLOUT_DIAG=0

# 公共负载：2748，16 卡 Actor + colocate rollout，TP4，规则奖励，BS16/n8。
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export Q36_CKPT_DIR="${Q36_CKPT_DIR:-/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl}"
export TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}"
export VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}"
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 REWARD_POOL_NPUS=0 ROLLOUT_NUM_WORKERS=4
export MINICPMO_FSDP2_PIN_MEMORY=0
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" ROLLOUT_N="${ROLLOUT_N:-8}"
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ROLLOUT_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
export ROLLOUT_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128))
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_MODEL_LEN" MAX_TOKEN_LEN_PER_GPU="$ROLLOUT_MAX_MODEL_LEN"
export ROLLOUT_MAX_NUM_SEQS=64 ROLLOUT_GPU_MEM_UTIL=0.6 UPDATE_WEIGHTS_BUCKET_MB=1024
export ACTOR_DYNAMIC_BSZ=true REF_LOG_PROB_DYNAMIC_BSZ=true ROLLOUT_LOG_PROB_DYNAMIC_BSZ=true
export ACTOR_MODEL_DTYPE=fp32 REF_MODEL_DTYPE=bfloat16 ACTOR_OFFLOAD_POLICY=false ACTIVATION_OFFLOAD=false
export HUANYU_FREEZE_AUDIO=true HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export MINICPMO_TUNE_AUDIO=0 MINICPMO_TUNE_VISION=0 HUANYU_AQA_AUDIO_TRAIN=0
export HUANYU_USE_PRECOMPUTE=0 MINICPMO_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=''
export FILTER_OVERLONG=false ENABLE_THINKING=1 DEBUG_MODE=0
export SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=false TRAINER_LOGGER='[console]'
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export EXPERIMENT_NAME="aqa_16card_${_group}_step${TOTAL_TRAINING_STEPS}" PROJECT_NAME=huanyu_aqa_ablation
export HUANYU_RUN_DIR="$BENCH_ROOT/$EXPERIMENT_NAME" HUANYU_CONSOLE_LOG="$BENCH_ROOT/$EXPERIMENT_NAME/console.log"
export DEFAULT_LOCAL_DIR="$HUANYU_RUN_DIR/checkpoints" TENSORBOARD_DIR="$HUANYU_RUN_DIR/tensorboard"
export GENRM_DUMP_DIR="$HUANYU_RUN_DIR/genrm_dump" _HUANYU_OUTPUTS_ACTIVE="$HUANYU_CONSOLE_LOG"

_args=(
    "++ray_kwargs.ray_init.address=\"$RAY_ADDRESS\""
    ++data.audio_key=audios ++data.video_key=videos ++data.mm_processor_kwargs.sampling_rate=16000
    +actor_rollout_ref.model.override_config.stream_input=false actor_rollout_ref.model.override_config.attn_implementation=eager
    ++actor_rollout_ref.model.override_config.huanyu_freeze_audio=true
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=false
    ++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=false
    reward.reward_model.enable=false reward.reward_model.enable_resource_pool=false
    reward.custom_reward_function.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.rule_based_reward
    reward.custom_reward_function.name=compute_score ++reward.custom_reward_function.reward_kwargs.require_think_end=true
    "trainer.rollout_data_dir=\"$HUANYU_RUN_DIR/rollout_dump\""
    "trainer.validation_data_dir=\"$HUANYU_RUN_DIR/validation_dump\""
)
# 核心精度/图开关由后端统一透传；以下补齐已有 Ray 集群不能从 driver 自动继承的变量。
for _key in Q36_CKPT_DIR HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION HUANYU_AQA_AUDIO_TRAIN HUANYU_EMBED_CACHE_DIR \
    Q36_BACKWARD_PREFETCH HUANYU_FORWARD_PROFILE_DIR HUANYU_WEIGHT_SYNC_AUDIT_DIR \
    HUANYU_FULL_WEIGHT_AUDIT_DIR HUANYU_EMBEDDING_AB_DIR HUANYU_MOE_FINITE_DEBUG HUANYU_FINITE_PROBE \
    MINICPMO_ROLLOUT_DIAG ASCEND_LAUNCH_BLOCKING; do
    _val="${!_key}"; _val="${_val//\\/\\\\}"; _val="${_val//\"/\\\"}"
    _args+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=\"$_val\"")
done
if [[ -v VLLM_PLUGINS ]]; then
    _args+=("++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PLUGINS=\"$VLLM_PLUGINS\"")
fi
printf '[huanyu-ablation] group=%s profile=%s graph=%s moe=%s replay=%s steps=%s run=%s\n' \
    "$_group" "$HUANYU_PRECISION_PROFILE" "$HUANYU_GRAPH_MODE" "$MINICPMO_MOE_NPU_KERNEL" "$HUANYU_EXPERT_REPLAY" "$TOTAL_TRAINING_STEPS" "$HUANYU_RUN_DIR"
if (( _dry )); then
    # 输出完整子进程环境与追加参数；不创建目录、不检查模型、不连接 Ray。
    export -p | grep -E ' (HUANYU_|MINICPMO_|ACTOR_|REF_|ROLLOUT_|Q36_|NUM_NPUS|NNODES|TRAIN_|PPO_|MAX_|MODEL_PATH|VAL_FILE|HCCL_|VERL_ENGINE_VENDOR)'
    printf '%q ' bash "$_dir/run_qwen36_huanyu_gspo_npu.sh" "${_args[@]}"; printf '\n'
    exit 0
fi
[[ ! -e "$HUANYU_RUN_DIR" ]] || { echo "FATAL: 已有目录 $HUANYU_RUN_DIR，请更换 BENCH_ROOT，避免混合日志" >&2; exit 2; }
# 后端使用 python3；把调用者选定的 Python 所在目录放到 PATH 前，并校验一致性。
_python="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
export PATH="$(dirname "$_python"):$PATH"; hash -r
[[ "$(python3 -c 'import sys; print(sys.executable)')" -ef "$_python" ]] || { echo 'FATAL: PYTHON_BIN 与 python3 不属于同一解释器' >&2; exit 2; }
"$_python" "$_helper" check --project-root "$PROJECT_ROOT" --replay "$HUANYU_EXPERT_REPLAY"
mkdir -p "$HUANYU_RUN_DIR"
: > "$HUANYU_CONSOLE_LOG"
"$_python" "$_helper" manifest "$HUANYU_RUN_DIR" --group "$_group"
set +e
bash "$_dir/run_qwen36_huanyu_gspo_npu.sh" "${_args[@]}" 2>&1 | tee -a "$HUANYU_CONSOLE_LOG"
_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$_rc" > "$HUANYU_RUN_DIR/exit_code.txt"
(( _rc == 0 )) || exit "$_rc"
