#!/usr/bin/env bash
# Huanyu AQA 在线音频训练（生产跑法，16 卡 colocate + 规则奖励，无 profiler）。
#
# 与 perf_report/analysis_11_11b 的 A/B 脚本同一条链路，区别在于：A/B 脚本是
# 4 步 mock reward + profiler 的测量跑，本脚本是可落 checkpoint 的生产跑，且
# **所有优化项合并打开**（A/B 的 variant 互斥，装不下全部）。
#
# 每一项优化都是下面「优化开关台账」里的一个 0/1，默认全开。关掉任意一项即
# 回到该项的 A/B 对照组，用于线上复现归因。开关之间的依赖（replay 依赖融合
# MoE kernel、aclgraph 依赖 kernel 档案等）在台账之后统一校验，不满足即退出，
# 不静默降级。
#
# 用法（容器内，PROJECT_ROOT 已部署）：
#   bash examples/qwen3_6_huanyu/run_huanyu_aqa_optimized.sh
#   HUANYU_OPT_ROUTER_REPLAY=0 bash examples/qwen3_6_huanyu/run_huanyu_aqa_optimized.sh
#   TOTAL_TRAINING_STEPS=4 SAVE_FREQ=-1 bash examples/qwen3_6_huanyu/run_huanyu_aqa_optimized.sh
set -euo pipefail

# ============================ 优化开关台账 ============================
# 这一段是本脚本唯一需要改的地方：一项优化一行，1=开 0=关。
# 「收益」列为 analysis_11_11b 顺序 A/B 实测（16 卡 × 4 步稳态均值）。
#
# 项目                      | 落点                               | 实测收益
# --------------------------|------------------------------------|------------------
# FSDP2 预取                | forward_prefetch + backward prefetch| gen/update 重叠
# bf16 梯度归约             | mixed_precision.reduce_dtype        | 梯度 RS 流量减半
# 融合 GDN kernel（训练侧） | chunk_gated_delta_rule 注入点       | GDN 算子 ~2.1x
# npu_rms_norm（训练侧）    | Qwen3_5MoeRMSNorm 实例 forward      | 随档案
# 融合 MoE kernel           | HF 逐专家循环 → GroupedMatmulV5    | update_actor −6%
# R3 路由回放               | 训练侧复用 rollout 实际路由         | probs_diff −30%
# HCCL AIV 展开             | HCCL_OP_EXPANSION_MODE=AIV          | 集合通信下沉
# rollout aclgraph          | decode 图捕获                       | gen 段
# 权重同步栈                | weight_sync_mode.sh 的 wsync profile| update_weights 34.2→9.0s
export HUANYU_OPT_FSDP_PREFETCH="${HUANYU_OPT_FSDP_PREFETCH:-1}"
export HUANYU_OPT_BF16_REDUCE="${HUANYU_OPT_BF16_REDUCE:-1}"
export HUANYU_OPT_GDN_KERNEL="${HUANYU_OPT_GDN_KERNEL:-1}"
export HUANYU_OPT_RMS_KERNEL="${HUANYU_OPT_RMS_KERNEL:-1}"
export HUANYU_OPT_MOE_KERNEL="${HUANYU_OPT_MOE_KERNEL:-1}"
export HUANYU_OPT_ROUTER_REPLAY="${HUANYU_OPT_ROUTER_REPLAY:-1}"
export HUANYU_OPT_AIV_HCCL="${HUANYU_OPT_AIV_HCCL:-1}"
export HUANYU_OPT_ROLLOUT_GRAPH="${HUANYU_OPT_ROLLOUT_GRAPH:-1}"
export HUANYU_OPT_WEIGHT_SYNC="${HUANYU_OPT_WEIGHT_SYNC:-1}"
_HUANYU_OPT_KEYS=(
    HUANYU_OPT_FSDP_PREFETCH HUANYU_OPT_BF16_REDUCE
    HUANYU_OPT_GDN_KERNEL HUANYU_OPT_RMS_KERNEL
    HUANYU_OPT_MOE_KERNEL HUANYU_OPT_ROUTER_REPLAY
    HUANYU_OPT_AIV_HCCL HUANYU_OPT_ROLLOUT_GRAPH
    HUANYU_OPT_WEIGHT_SYNC
)
# ======================== 优化开关台账结束 ============================

for _key in "${_HUANYU_OPT_KEYS[@]}"; do
    case "${!_key}" in
        0|1) ;;
        *) echo "[huanyu-opt] FATAL: ${_key} must be 0 or 1, got '${!_key}'" >&2; exit 2 ;;
    esac
done

# ---- 开关 → 底层 env 的翻译（依赖不满足即 fail-loud）----
# 1) FSDP2 预取：两个方向分别由 backend 的 fsdp_config 与 Q36 hook 读取。
if [[ "$HUANYU_OPT_FSDP_PREFETCH" == 1 ]]; then
    export Q36_BACKWARD_PREFETCH=on ACTOR_FORWARD_PREFETCH=true
else
    export Q36_BACKWARD_PREFETCH=off ACTOR_FORWARD_PREFETCH=false
fi

# 2) 混合精度：param bf16 本就是引擎隐式默认，真正的变量是 reduce_dtype。
export ACTOR_MP_PARAM_DTYPE=bfloat16
export ACTOR_MP_BUFFER_DTYPE=fp32
if [[ "$HUANYU_OPT_BF16_REDUCE" == 1 ]]; then
    export ACTOR_MP_REDUCE_DTYPE=bf16
else
    export ACTOR_MP_REDUCE_DTYPE=fp32
fi

# 3) 训练侧融合 kernel：GDN 与 rms_norm 同属 kernel 档案，后端精度策略的
#    HUANYU_KERNEL_BISECT 一次只能单独回退其中一个；两个都关则退回 full 档
#    （Python FP32 参考），该档案与 aclgraph 互斥。
if [[ "$HUANYU_OPT_GDN_KERNEL" == 1 || "$HUANYU_OPT_RMS_KERNEL" == 1 ]]; then
    if [[ "$HUANYU_OPT_GDN_KERNEL" == 1 && "$HUANYU_OPT_RMS_KERNEL" == 1 ]]; then
        export HUANYU_KERNEL_BISECT=none
    elif [[ "$HUANYU_OPT_GDN_KERNEL" == 0 ]]; then
        export HUANYU_KERNEL_BISECT=gdn_off
    else
        export HUANYU_KERNEL_BISECT=rms_off
    fi
    if [[ "$HUANYU_OPT_ROLLOUT_GRAPH" == 1 ]]; then
        export HUANYU_PRECISION_PROFILE=kernel_graph HUANYU_GRAPH_MODE=1
    else
        export HUANYU_PRECISION_PROFILE=kernel HUANYU_GRAPH_MODE=0
    fi
elif [[ "$HUANYU_OPT_ROLLOUT_GRAPH" == 1 ]]; then
    echo "[huanyu-opt] FATAL: HUANYU_OPT_ROLLOUT_GRAPH=1 需要 kernel 档案，但 GDN 与 rms kernel 都已关闭（full 档用 Python 参考实现，抓不了推理图）" >&2
    exit 2
else
    # 显式写 none：调用方环境里残留的 gdn_off/rms_off 在 full 档会被
    # 后端精度策略判为非法，报的还是另一回事的错。
    export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0 HUANYU_KERNEL_BISECT=none
fi

# 4) 融合 MoE kernel 与 R3：replay 只存在于融合 kernel 内，且要求补丁 strict。
export MINICPMO_MOE_NPU_KERNEL="$HUANYU_OPT_MOE_KERNEL"
export HUANYU_EXPERT_REPLAY="$HUANYU_OPT_ROUTER_REPLAY"
if [[ "$HUANYU_OPT_ROUTER_REPLAY" == 1 && "$HUANYU_OPT_MOE_KERNEL" != 1 ]]; then
    echo "[huanyu-opt] FATAL: HUANYU_OPT_ROUTER_REPLAY=1 需要 HUANYU_OPT_MOE_KERNEL=1（回放只在融合 MoE kernel 内实现）" >&2
    exit 2
fi
if [[ "$HUANYU_OPT_ROUTER_REPLAY" == 1 ]]; then
    export VERL_OMNI_EXT_PATCH_STRICT=1
fi
# 路由一致率日志默认关（它逐层统计，开了有可观 host 开销）。
export HUANYU_REPLAY_DEBUG="${HUANYU_REPLAY_DEBUG:-0}"
export HUANYU_MOE_FINITE_DEBUG="${HUANYU_MOE_FINITE_DEBUG:-0}"
export HUANYU_FINITE_PROBE="${HUANYU_FINITE_PROBE:-0}"

# 5) HCCL 展开模式。hccl_mode.sh 取 ${HCCL_OP_EXPANSION_MODE:-...}，即调用方
#    环境里的残留值优先于 profile；本开关要说了算，就得把两者一起写死
#    （check_native_runtime.py 的预检查的正是这两者是否一致）。
if [[ "$HUANYU_OPT_AIV_HCCL" == 1 ]]; then
    export HUANYU_HCCL_PROFILE=aiv HCCL_OP_EXPANSION_MODE=AIV
else
    export HUANYU_HCCL_PROFILE=host HCCL_OP_EXPANSION_MODE=HOST
fi

# 6) update_weights 权重同步栈：后端 source 的 weight_sync_mode.sh 默认 profile=off
#    （关闭时连引擎子类都不注册，与上游逐字节一致），所以生产跑必须显式打开。
#    profile=on 已含 skip-staging / bf16-gather / lookahead / early-ack / moe-fast /
#    direct-load；桶尺寸与 strict 与 profile 正交，要单独钉死。trace 是测量用的
#    逐阶段计时，生产默认关。
if [[ "$HUANYU_OPT_WEIGHT_SYNC" == 1 ]]; then
    export HUANYU_WSYNC_PROFILE=on
    export HUANYU_WSYNC_BUCKET_MB="${HUANYU_WSYNC_BUCKET_MB:-2048}"
    export HUANYU_WSYNC_STRICT=1
else
    export HUANYU_WSYNC_PROFILE=off
fi
export HUANYU_WSYNC_TRACE="${HUANYU_WSYNC_TRACE:-off}"

# ---- 部署路径与数据（可用 env 覆盖）----
export PROJECT_ROOT="${PROJECT_ROOT:-/disk/lianyingzhao/xy_rl_omni}"
_EXAMPLES="$PROJECT_ROOT/examples/qwen3_6_huanyu"
cd "$PROJECT_ROOT"
export MODEL_PATH="${MODEL_PATH:-/disk/lianyingzhao/models/2748}"
export TRAIN_FILE="${TRAIN_FILE:-/disk/lianyingzhao/dataset/aqa_dataset/train.parquet}"
export VAL_FILE="${VAL_FILE:-/disk/lianyingzhao/dataset/aqa_dataset/validation.parquet}"
for _file in "$_EXAMPLES/hccl_mode.sh" "$_EXAMPLES/prepare_huanyu_env.sh" "$_EXAMPLES/job_bootstrap.py" \
    "$_EXAMPLES/run_qwen36_huanyu_gspo_npu.sh" "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/rule_based_reward.py"; do
    [[ -f "$_file" ]] || { echo "[huanyu-opt] FATAL: missing $_file" >&2; exit 2; }
done

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext

# ---- AQA 在线音频训练环境（与 analysis_11_11b 基线一致）----
export HUANYU_FREEZE_AUDIO=false
export MINICPMO_TUNE_AUDIO=1
export HUANYU_AQA_AUDIO_TRAIN=1
export HUANYU_USE_PRECOMPUTE=0
export HUANYU_EMBED_CACHE_DIR=""
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_VISION=0
export ACTOR_OFFLOAD_POLICY=false
export ACTIVATION_OFFLOAD=false
export FILTER_OVERLONG=false
export DEBUG_MODE=0 DEBUG_SAMPLES=0
export MINICPMO_ENABLE_MM_FORWARD=1
export MINICPMO_ALLOW_PACKED_MM_FORWARD=1
export MINICPMO_FSDP2_PIN_MEMORY=0
export MINICPMO_DISABLE_MM_DEDUP=1
export MINICPMO_USE_CUSTOM_DATASET=1
export ACTOR_MODEL_DTYPE=fp32 REF_MODEL_DTYPE=bfloat16
export VLLM_ASCEND_WEIGHT_NZ_MODE=0 ASCEND_WEIGHT_NZ_MODE=0 ASCEND_ENABLE_TRANS_DATA_CACHE=0

# ---- 资源池：16 卡全给 actor（colocate trainer+rollout），规则奖励无 RM 池 ----
export NUM_NPUS="${NUM_NPUS:-16}" NNODES="${NNODES:-1}" ROLLOUT_TP="${ROLLOUT_TP:-4}"
export NPUS_PER_NODE="$NUM_NPUS"
export REWARD_POOL_NPUS=0
export NUM_GPUS_ACTOR_ROLLOUT_REWARD="$NUM_NPUS"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}" PPO_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-$MAX_PROMPT_LENGTH}"
export ROLLOUT_MAX_MODEL_LEN=$((ROLLOUT_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128))
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}"

# 生产跑默认不限步数；35B fp32 actor 每次保存 ~117 GB，SAVE_FREQ 由调用方按盘量定。
_STEPS="${TOTAL_TRAINING_STEPS:-null}"
[[ "$_STEPS" == -1 ]] && _STEPS=null
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
export PROJECT_NAME="${PROJECT_NAME:-huanyu_aqa}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aqa_optimized_${NUM_NPUS}card}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$PROJECT_ROOT/runs/$PROJECT_NAME/$EXPERIMENT_NAME/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$PROJECT_ROOT/runs/$PROJECT_NAME/$EXPERIMENT_NAME/tensorboard}"
export WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VC_WORKER_HOSTS="${VC_WORKER_HOSTS:-127.0.0.1}"
export VC_TASK_INDEX=0
export SOCKET_IFNAME="${SOCKET_IFNAME:-enp23s0f3}"

_TP_ARG="actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
set +u
# 站点环境（原 env_export_huanyu.sh）在 prepare_huanyu_env.sh 的函数里，它要求
# 调用方先定义 precision/graph/native 三段策略；直接从后端取，保证与后端同一份。
# 不走 huanyu_prepare_node：那条路会 pip install，这里假定 prepare 已跑过。
# 运行日志目录由后端自己初始化。
eval "$(awk '/^huanyu_(configure_precision_mode|configure_graph_mode|select_native_source)\(\) \{/{f=1} f{print} f&&/^\}$/{f=0}' \
    "$_EXAMPLES/run_qwen36_huanyu_gspo_npu.sh")"
declare -F huanyu_configure_precision_mode huanyu_configure_graph_mode huanyu_select_native_source >/dev/null || {
    echo "[huanyu-opt] FATAL: run_qwen36_huanyu_gspo_npu.sh no longer defines the policy functions" >&2; exit 2;
}
source "$_EXAMPLES/prepare_huanyu_env.sh" --functions-only || exit $?
huanyu_prepare_site_environment "$_TP_ARG" || exit $?
set +x
set -euo pipefail
export HUANYU_VLLM_OMNI_ROOT="${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
export PYTHONPATH="${HUANYU_VLLM_OMNI_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -B "$_EXAMPLES/job_bootstrap.py" check-aqa \
    --train "$TRAIN_FILE" --validation "$VAL_FILE" --train-batch-size "$TRAIN_BATCH_SIZE"
python3 -B "$_EXAMPLES/check_native_runtime.py" --check --expected-root "$HUANYU_VLLM_OMNI_ROOT"

_NODE_IP="${NODE_IP:-127.0.0.1}"
export RAY_PORT="${RAY_PORT:-6925}"
export RAY_ADDRESS="${_NODE_IP}:${RAY_PORT}"
export RAY_memory_monitor_refresh_ms=0
if ! pgrep -x raylet >/dev/null; then
    ray start --head --node-ip-address="$_NODE_IP" --port="$RAY_PORT" \
        --object-store-memory=16000000000 \
        --resources="{\"NPU\": ${NUM_NPUS}}"
fi

# Ray worker 的 runtime env：融合 kernel / replay 的开关必须随 worker 走，
# 否则 driver 开了而 worker 侧 precision_policy 断言的是另一组值。
_RAY_ENV_ARGS=()
for _key in PYTHONPATH HUANYU_VLLM_OMNI_ROOT HUANYU_FREEZE_AUDIO \
    HUANYU_AQA_AUDIO_TRAIN HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR MINICPMO_USE_PRECOMPUTE \
    MINICPMO_TUNE_AUDIO MINICPMO_TUNE_VISION Q36_BACKWARD_PREFETCH LD_PRELOAD \
    MINICPMO_MOE_NPU_KERNEL HUANYU_EXPERT_REPLAY HUANYU_REPLAY_DEBUG HUANYU_MOE_FINITE_DEBUG \
    HUANYU_FINITE_PROBE; do
    _val="$(python3 -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
    _RAY_ENV_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=${_val}")
done

echo "[huanyu-opt] 台账: prefetch=${HUANYU_OPT_FSDP_PREFETCH} bf16_reduce=${HUANYU_OPT_BF16_REDUCE} gdn_kernel=${HUANYU_OPT_GDN_KERNEL} rms_kernel=${HUANYU_OPT_RMS_KERNEL} moe_kernel=${HUANYU_OPT_MOE_KERNEL} router_replay=${HUANYU_OPT_ROUTER_REPLAY} aiv_hccl=${HUANYU_OPT_AIV_HCCL} rollout_graph=${HUANYU_OPT_ROLLOUT_GRAPH} weight_sync=${HUANYU_OPT_WEIGHT_SYNC}"
echo "[huanyu-opt] 落值: profile=${HUANYU_PRECISION_PROFILE} graph=${HUANYU_GRAPH_MODE} bisect=${HUANYU_KERNEL_BISECT} gdn=${MINICPMO_GDN_NPU_KERNEL} rms=${MINICPMO_NPU_RMS_NORM} mp=${ACTOR_MP_PARAM_DTYPE}/${ACTOR_MP_REDUCE_DTYPE} hccl=${HUANYU_HCCL_PROFILE}/${HCCL_OP_EXPANSION_MODE:-<unset>} fwd_prefetch=${ACTOR_FORWARD_PREFETCH} wsync=${HUANYU_WSYNC_PROFILE}/${HUANYU_WSYNC_BUCKET_MB:-<default>}MB/strict=${HUANYU_WSYNC_STRICT:-0}/trace=${HUANYU_WSYNC_TRACE}"
echo "[huanyu-opt] 作业: actor=${NUM_NPUS}x${NNODES}(colocate) TP=${ROLLOUT_TP} reward=rule_based batch=${TRAIN_BATCH_SIZE}/${PPO_MINI_BATCH_SIZE}/n${ROLLOUT_N} steps=${_STEPS} save_freq=${SAVE_FREQ} -> ${DEFAULT_LOCAL_DIR}"

exec bash "$_EXAMPLES/run_qwen36_huanyu_gspo_npu.sh" \
    ++data.video_key=videos \
    ++data.audio_key=audios \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    ++data.mm_processor_kwargs.max_slice_nums=1 \
    ++data.mm_processor_kwargs.use_image_id=false \
    +actor_rollout_ref.model.override_config.stream_input=false \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512 \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False \
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']" \
    actor_rollout_ref.actor.entropy_checkpointing=False \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.entropy_checkpointing=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=False \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset \
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset \
    actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu \
    actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.lora_rank=0 \
    "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=${HUANYU_FREEZE_AUDIO}" \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=false \
    'actor_rollout_ref.model.exclude_modules=.*model\.visual.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*' \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch="${ACTOR_FORWARD_PREFETCH}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.offload_policy=false \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" \
    "actor_rollout_ref.rollout.agent.num_workers=$((NUM_NPUS * NNODES / ROLLOUT_TP))" \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    reward.reward_model.enable=false \
    reward.reward_model.enable_resource_pool=false \
    reward.reward_model.model_path=null \
    reward.reward_model.n_gpus_per_node=0 \
    reward.reward_model.nnodes=0 \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.rule_based_reward \
    reward.custom_reward_function.name=compute_score \
    "++reward.custom_reward_function.reward_kwargs.require_think_end=true" \
    "++data.apply_chat_template_kwargs.enable_thinking=true" \
    "trainer.n_gpus_per_node=${NUM_NPUS}" \
    "trainer.nnodes=${NNODES}" \
    trainer.total_training_steps="${_STEPS}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}" \
    "${_RAY_ENV_ARGS[@]}" "$@"