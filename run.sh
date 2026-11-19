#!/usr/bin/env bash
# AVQA-R1-6K：将官方 GPU LoRA 配方用于单节点 16 NPU。
# 环境与 Ray 须已准备；不安装依赖、不启动 Ray。额外 Hydra 参数最后生效。
# --dry-run 仅打印命令；--config 输出完整配置；--filter 离线过滤，不启动训练或 Ray。
# 来源：https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh
# 对齐的是公开脚本显式参数；未取得 W&B 原作业的完整配置/依赖版本。
set -eo pipefail

# launch.sh 已将平台的 --名称=值 导出到环境。Hydra 接收配方覆盖项和自己的 CLI 参数。
if [[ "${1:-}" == --platform-launch ]]; then
    shift
    trainer_args=()
    for arg in "$@"; do
        case "$arg" in
            --cfg=*|--info=*|--package=*) ;;
            *) [[ "$arg" =~ ^--[a-zA-Z_][a-zA-Z0-9_]*= ]] && continue ;;
        esac
        trainer_args+=("$arg")
    done
    set -- "${trainer_args[@]}"
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$HERE/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${AVQA_DATA_DIR:-/opt/huawei/dataset/infra_data/data/avqa_r1_6k}"
family=${AVQA_MODEL_FAMILY:-qwen3_omni}
case "$family" in
    qwen3_omni)
        checkpoint=Qwen3-Omni-30B-A3B-Instruct
        experiment=qwen3_omni_lora_avqa_16npu
        optimizer_offload=false ;;
    huanyu)
        checkpoint=2748
        experiment=huanyu_2748_lora_avqa_16npu
        optimizer_offload=true ;;
    *) echo "Unknown AVQA_MODEL_FAMILY: $family" >&2; exit 2 ;;
esac
MODEL_PATH="${AVQA_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/$checkpoint}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/avqa_${checkpoint}_$(date +%Y%m%d_%H%M%S)_$$}"
REWARD_FILE="$HERE/choice_reward.py"

# 原实验显式配置。保持 BS、N、优化器和 LoRA 范围；卡数从 4 改为 16。
ARGS=(
    "data.train_files=\"${AVQA_TRAIN_FILE:-$DATA_DIR/train.parquet}\"" # 原始训练划分，不继承旧任务 TRAIN_FILE。
    "data.val_files=\"${AVQA_VAL_FILE:-$DATA_DIR/validation.parquet}\"" # 原始验证划分。
    "data.train_batch_size=128" # 一次采样 128 个问题。
    "data.max_prompt_length=4096" # 含图片和音频 token 的输入上限。
    "data.max_response_length=12288" # 输出上限 12K，允许 EOS 提前结束。
    "data.truncation=error" # 不静默截断媒体。
    "data.filter_overlong_prompts=true" # 用当前模型的 tokenizer 过滤超长问题，实际保留数会变化。
    "++data.mm_processor_kwargs.sampling_rate=16000" # 音频采样率。
    "actor_rollout_ref.model.path=\"$MODEL_PATH\"" # 使用所选模型的初始权重。
    "actor_rollout_ref.model.lora_rank=32" # LoRA 秩，不能沿用全参数训练的 0。
    "actor_rollout_ref.model.lora_alpha=64" # LoRA 缩放系数。
    "actor_rollout_ref.model.lora_dtype=float32" # 可训练 adapter 使用 FP32。
    "actor_rollout_ref.model.lora.merge=true" # 合并 LoRA 后向 rollout 同步权重。
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" # 激活重计算。
    "actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj','o_proj']" # 保留官方目标名称，不扩展到 GDN/专家。
    "actor_rollout_ref.actor.freeze_vision_tower=true" # 视觉编码器冻结。
    "actor_rollout_ref.actor.strategy=fsdp2" # FSDP2 分片。
    "actor_rollout_ref.actor.optim.lr=3e-6" # 官方 LoRA 学习率。
    "actor_rollout_ref.actor.optim.weight_decay=0.01" # 官方权重衰减。
    "actor_rollout_ref.actor.optim.clip_grad=1.0" # 梯度范数上限。
    "actor_rollout_ref.actor.ppo_mini_batch_size=16" # 每次优化对应 16 个原始问题。
    "actor_rollout_ref.actor.use_dynamic_bsz=true" # 按 token 预算分 microbatch。
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480" # 单卡训练 token 预算。
    "actor_rollout_ref.actor.use_kl_loss=false" # 不额外增加 KL loss。
    "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" # GSPO 策略损失。
    "actor_rollout_ref.actor.clip_ratio_low=3e-4" # 下侧裁剪阈值。
    "actor_rollout_ref.actor.clip_ratio_high=4e-4" # 上侧裁剪阈值。
    "actor_rollout_ref.actor.clip_ratio_c=10.0" # dual-clip 阈值。
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" # 按序列平均后再求均值。
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" # 基座参数精度。
    "actor_rollout_ref.actor.fsdp_config.param_offload=true" # Actor 参数可卸载。
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload"
    "actor_rollout_ref.rollout.n=16" # 每题生成 16 条，共 2048 条/批。
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2" # 保留 TP2，16 卡对应 8 个副本。
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" # 保留本实验的 0.7，不使用其他任务的 0.6。
    "actor_rollout_ref.rollout.load_format=safetensors" # 加载真实权重。
    "actor_rollout_ref.rollout.prompt_length=4160" # 官方 rollout 输入预算。
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true" # old_log_prob 动态 batch。
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480" # old_log_prob token 预算。
    "actor_rollout_ref.rollout.enable_prefix_caching=false" # 关闭前缀缓存。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar" # 文本自回归输出。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256" # 与官方 stage 引擎并发一致。
    "actor_rollout_ref.rollout.max_num_seqs=256" # 同时保证新版外层配置一致。
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256]" # 保留官方捕图尺寸。
    "actor_rollout_ref.rollout.val_kwargs.n=1" # 每个验证问题一次生成。
    "actor_rollout_ref.rollout.val_kwargs.temperature=1.0" # 原脚本值；是否采样仍由框架 do_sample 决定。
    "actor_rollout_ref.rollout.val_kwargs.top_p=0.7" # 原脚本值，不自行改为其他验证策略。
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true" # Ref 动态 batch。
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480" # Ref token 预算。
    "actor_rollout_ref.ref.fsdp_config.param_offload=true" # Ref 参数可卸载。
    "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16" # Ref 参数精度。
    "algorithm.adv_estimator=grpo" # 组内奖励优势。
    "algorithm.use_kl_in_reward=false" # 奖励中不加 KL。
    "reward.reward_manager.source=register" # 使用已注册管理器。
    "reward.reward_manager.name=naive" # 官方 naive 管理器。
    "reward.custom_reward_function.path=\"$REWARD_FILE\"" # 原版 choice_reward，不使用 GenRM。
    "reward.custom_reward_function.name=compute_score" # 严格比较 answer 标签中的内容。
    "trainer.val_before_train=false" # 正式训练保持原脚本设置，基线另跑。
    "trainer.balance_batch=true" # 按长度均衡训练 rank。
    "trainer.critic_warmup=0" # GRPO 不预热 critic。
    "trainer.logger=${TRAINER_LOGGER:-[console,wandb]}" # 记录训练曲线，默认 wandb 离线。
    "trainer.project_name=gspo" # 日志项目名。
    "trainer.experiment_name=$experiment" # 与官方模型实验分开记录。
    "trainer.n_gpus_per_node=16" # 一节点 16 NPU。
    "trainer.nnodes=1" # 不额外占用 GenRM 节点。
    "trainer.save_freq=50" # 每 50 step 保存。
    "trainer.test_freq=10" # 每 10 step 验证。
    "trainer.total_epochs=10" # 按过滤后的实际数据长度计算总 step，不硬编码 348。
)

# Ascend/FSDP2 与多模态训练的公共设置。
ARGS+=(
    "trainer.device=npu"
    "++ray_kwargs.ray_init.address=\"${RAY_ADDRESS:-auto}\""
    "data.image_key=images"
    "++data.audio_key=audios"
    "actor_rollout_ref.model.model_stage=thinker"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.model.use_remove_padding=false" # 与原 AQAV 对照使用相同的 padded batch。
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" # Ascend 训练 attention 后端。
    "actor_rollout_ref.actor.use_torch_compile=false" # 训练侧关闭编译，与 rollout 图模式独立。
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false"
    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=false"
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true" # NPU 长输出的熵计算分块。
    "actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048"
    "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true"
    "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=2048"
    'actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*|.*audio_encoder.*|.*audio_projection.*"'
    "actor_rollout_ref.rollout.name=vllm_omni"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0"
    "reward.reward_model.enable=false"
    "reward.reward_model.enable_resource_pool=false"
    "trainer.resume_mode=disable" # 每次明确从指定初始模型出发，不误接旧任务。
    "trainer.default_local_dir=\"$RUN_DIR/checkpoints\""
    "trainer.rollout_data_dir=\"$RUN_DIR/rollout_dump\""
    "trainer.validation_data_dir=\"$RUN_DIR/validation_dump\""
)

RUNTIME_ENV=(
    "PYTHONPATH=$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "VLLM_WORKER_MULTIPROC_METHOD=spawn"
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1"
    "RAY_DEDUP_LOGS=0"
    "PYTHONUNBUFFERED=1"
    "HYDRA_FULL_ERROR=1"
    "WANDB_MODE=${WANDB_MODE:-offline}"
    "HUANYU_HCCL_PROFILE=aiv" "HCCL_OP_EXPANSION_MODE=AIV" # 新版公共通信策略；同时传给 driver/worker。
    "VLLM_ASCEND_ENABLE_NZ=0" "VLLM_ASCEND_WEIGHT_NZ_MODE=0" "ASCEND_WEIGHT_NZ_MODE=0"
)
if [[ "$family" == huanyu ]]; then
    source "$HERE/huanyu_2748.sh"
else
    replay=false
    ARGS+=(
        "data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets"
        "data.custom_cls.name=QwenOmniRLHFDataset"
        "actor_rollout_ref.model.architecture=Qwen3OmniMoeForConditionalGeneration"
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe"
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=thinker_only"
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true"
    )
    RUNTIME_ENV+=("VERL_USE_EXTERNAL_MODULES=verl_omni" "VLLM_OMNI_EXTERNAL_MODULES="
                  "HUANYU_EXPERT_REPLAY=0")
fi
ARGS+=(
    "actor_rollout_ref.rollout.enable_rollout_routing_replay=$replay"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_return_routed_experts=$replay"
)
# 若用户设了插件白名单，driver 和 Ray worker 必须一致；不覆盖已有 Ascend 插件。
if [[ -v VLLM_PLUGINS ]]; then
    RUNTIME_ENV+=("VLLM_PLUGINS=$VLLM_PLUGINS")
fi
case "${HUANYU_GRAPH_MODE:-1}" in
    1) eager=false; compile=1; disable=0 ;;
    0) eager=true; compile=0; disable=1 ;;
    *) echo 'HUANYU_GRAPH_MODE 必须为 0 或 1' >&2; exit 2 ;;
esac
RUNTIME_ENV+=("HUANYU_GRAPH_MODE=${HUANYU_GRAPH_MODE:-1}" "VLLM_TORCH_COMPILE=$compile"
              "TORCH_COMPILE_DISABLE=$disable" "NPU_GRAPH_CAPTURE_DISABLE=$disable")
ARGS+=("actor_rollout_ref.rollout.enforce_eager=$eager")
for entry in "${RUNTIME_ENV[@]}"; do
    export "$entry"
    key="${entry%%=*}"; value="${entry#*=}"
    value="${value//\\/\\\\}"; value="${value//\"/\\\"}"
    ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$key=\"$value\"")
done

mode=run
case "${1:-}" in
    --dry-run) mode=dry; shift ;;
    --config) mode=config; shift ;;
    --filter) mode=filter; shift ;;
esac
if [[ "$mode" == dry ]]; then
    printf '%q ' "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi
cd -- "$PROJECT_ROOT"
[[ -f "$REWARD_FILE" ]] || { echo "缺少官方奖励实现：$REWARD_FILE" >&2; exit 2; }
if [[ "$mode" == filter ]]; then
    # 使用同一套 Hydra 参数和模型适配器，避免离线/训练的长度计算分叉。
    export AVQA_FILTER_OUTPUT_DIR="${AVQA_FILTER_OUTPUT_DIR:-${DATA_DIR}_filtered}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
    exec "$PYTHON_BIN" "$HERE/filter_data.py" "${ARGS[@]}" "$@" \
        "data.filter_overlong_prompts_workers=${AVQA_FILTER_WORKERS:-8}"
fi
if [[ "$mode" == config ]]; then
    exec "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "$@" --cfg job --resolve
fi
mkdir -p -- "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
printf '[avqa] recipe=published-main model=%s actor=16 TP=2 LoRA=32 reward=choice run=%s\n' "$MODEL_PATH" "$RUN_DIR"
printf '[avqa] adapter=%s graph=%s route_replay=%s hccl=aiv\n' "$family" "$HUANYU_GRAPH_MODE" "$replay"
_hccl_probe="$PROJECT_ROOT/verl_omni_ext/probes/hccl_policy.py"
[[ -f "$_hccl_probe" ]] || { echo "缺少通信策略探针：$_hccl_probe" >&2; exit 2; }
"$PYTHON_BIN" "$_hccl_probe" --strict || exit $?
printf '%q ' "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "$@" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
# 在加载权重前记录框架默认值；未知字段导致失败时，不继续浪费 NPU 启动时间。
"$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "$@" --cfg job --resolve > "$RUN_DIR/resolved-config.txt"
exec "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "$@"
