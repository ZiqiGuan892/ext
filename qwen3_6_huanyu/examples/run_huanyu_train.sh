#!/usr/bin/env bash
# Huanyu 单文件训练入口：仅在 head 执行，Python 环境、插件和 Ray 集群须已准备好。
# 用法：bash run_huanyu_train.sh [Hydra 参数...]；--dry-run 只打印命令，不连接 Ray。
# ${变量:-默认值} 可在文件内修改或通过环境变量覆盖；末尾 Hydra 参数优先级最高。
# 本文件列出本入口使用的配置；未列出的框架字段仍使用 verl/verl-omni 自身默认值。
set -eo pipefail

# ===== 1. 路径、Python 与已有 Ray 集群 =====
PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}" # 当前插件仓库根目录。
PYTHON_BIN="${PYTHON_BIN:-python3}" # 沿用已准备的 Python，可指定解释器绝对路径。
RAY_ADDRESS="${RAY_ADDRESS:-auto}" # 连接已有集群，auto 找不到集群时失败而不新建 Ray。
TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}" # 通用多模态训练集。
VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}" # 通用多模态验证集。
MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}" # Huanyu Actor/Ref 及 rollout 的初始权重。
RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}" # GenRM 评分模型权重。

# ===== 2. 资源池、batch 与长度：默认 Actor16 + 独立 GenRM16 =====
NUM_NPUS="${NUM_NPUS:-16}" # Actor 资源池每节点卡数，包含其共置 rollout。
NNODES="${NNODES:-1}" # Actor 资源池节点数，不是整个 Ray 集群节点数。
ROLLOUT_TP="${ROLLOUT_TP:-4}" # 每个 Actor rollout 副本的张量并行卡数。
RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}" # 独立 GenRM 资源池每节点卡数。
RM_NNODES="${RM_NNODES:-1}" # 独立 GenRM 资源池节点数。
RM_TP="${RM_TP:-4}" # 每个 GenRM 副本的张量并行卡数。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" # 每次采样的原始 prompt 数，沿用 OmniMix 默认值。
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}" # 每次 PPO 更新的原始 prompt 数。
ROLLOUT_N="${ROLLOUT_N:-4}" # 每个 prompt 生成的轨迹数，默认每批共 128 条。
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" # 关闭动态 batch 时每卡每次前反向的序列数。
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}" # 数据侧 prompt token 上限，包含展开后的媒体 token。
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}" # 每条 rollout 最多生成的 token 数。
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + 32))}" # 引擎 prompt 预算，保留模板余量。
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128))}" # 引擎总上下文上限。
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}" # 每次引擎调度的 token 总预算。
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$ROLLOUT_MAX_MODEL_LEN}" # 动态 batch 下每卡训练及 log_prob 的 token 预算。
RM_RESPONSE_LENGTH="${RM_RESPONSE_LENGTH:-1024}" # GenRM 最多生成的评分 token 数。
RM_MAX_MODEL_LEN="${RM_MAX_MODEL_LEN:-32768}" # GenRM 总上下文上限。
RM_PROMPT_LENGTH="${RM_PROMPT_LENGTH:-$((RM_MAX_MODEL_LEN - RM_RESPONSE_LENGTH))}" # GenRM 输入 prompt 上限。
HUANYU_REWARD_MODE="${HUANYU_REWARD_MODE:-RM}" # RM 使用 GenRM；rule_based 按 data_source 调用已注册规则。

# ===== 3. 训练周期、输出目录 =====
PROJECT_NAME="${PROJECT_NAME:-omni_mix_qwen36}" # 日志平台的项目名。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_2748_omni_mix}" # 当前训练实验名称。
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}" # 正整数限制训练 step，null 或 -1 按 epoch 结束。
HUANYU_RUN_DIR="${HUANYU_RUN_DIR:-$PROJECT_ROOT/runs/huanyu_$(date +%Y%m%d_%H%M%S)_$$}" # 本次日志、checkpoint 和生成结果目录。
if [[ "$TOTAL_TRAINING_STEPS" == -1 ]]; then TOTAL_TRAINING_STEPS=null; fi

# ===== 4. 进程环境：同时显式传给 Ray worker，避免沿用旧 raylet 的变量 =====
# 不安装依赖、不 source 环境脚本、不修改代理/PATH/.bashrc、不启动或停止 Ray。
# 只添加当前 ext 仓库；vLLM-Omni 沿用已安装版本，不把其源码加入 PYTHONPATH。
RUNTIME_ENV=(
    "PYTHONPATH=$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" # 让各节点的 worker 导入当前 ext 仓库。
    "VERL_OMNI_EXT_PATCH_STRICT=1" # 必要的兼容补丁失效时立即报错。
    "HUANYU_VLLM_OMNI_ROOT=${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}" # 校验已安装原生 Huanyu 实现的来源。
    "HUANYU_VLLM_SOURCE=${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}" # 推理 worker 使用相同的原生源码校验路径。
    "VLLM_WORKER_MULTIPROC_METHOD=spawn" # 推理子进程使用 spawn 启动方式。
    "VLLM_USE_V1=0" # 沿用当前 OmniMix 后端的版本兼容开关。
    "USE_OPTIMIZED_MODEL=0" # 关闭镜像中的替代模型实现。
    "ASCEND_SECURITY_CHECK_DISABLED=1" # 沿用 Ascend 镜像的安全检查兼容设置。
    "VERL_DATAPROTO_SERIALIZATION_METHOD=numpy" # DataProto 跨进程序列化方式。
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1" # 由训练框架管理 NPU 可见设备。
    "RAY_DEDUP_LOGS=0" # 保留各 worker 的诊断日志。
    "HYDRA_FULL_ERROR=1" # Hydra 异常打印完整调用栈。
    "PYTHONUNBUFFERED=1" # Python 日志实时输出。
    "WANDB_MODE=${WANDB_MODE:-offline}" # 使用 wandb logger 时默认离线记录。
    "TENSORBOARD_DIR=${TENSORBOARD_DIR:-$HUANYU_RUN_DIR/tensorboard}" # 使用 TensorBoard logger 时的输出目录。
)

# ===== 5. 媒体输入：默认冻结 encoder/projector/ViT，在线编码，不使用缓存 =====
RUNTIME_ENV+=(
    "HUANYU_FREEZE_AUDIO=true" # Huanyu 音频编码器和投影层冻结状态。
    "HUANYU_TRAIN_AUDIO_ENCODER=false" # 不更新音频编码器权重。
    "HUANYU_TRAIN_AUDIO_PROJECTOR=false" # 不更新音频投影层权重。
    "HUANYU_AQA_AUDIO_TRAIN=0" # 使用通用多模态路径，不启用纯音频 AQA 限制。
    "HUANYU_STEM_TEXT_TRAIN=0" # 不强制纯文本 STEM 路径。
    "HUANYU_USE_PRECOMPUTE=${HUANYU_USE_PRECOMPUTE:-0}" # 设为 1 时使用 Huanyu 预计算 embedding。
    "HUANYU_EMBED_CACHE_DIR=${HUANYU_EMBED_CACHE_DIR:-}" # 启用缓存时填写与模型、数据匹配的缓存目录。
    "MINICPMO_USE_PRECOMPUTE=0" # 禁用不兼容的 Whisper 旧缓存路径。
    "MINICPMO_TUNE_AUDIO=0" # 关闭旧路径的音频解冻开关。
    "MINICPMO_TUNE_VISION=0" # 冻结视觉编码器。
    "MINICPMO_USE_CUSTOM_DATASET=1" # 启用自定义多模态数据集兼容逻辑。
    "MINICPMO_ENABLE_MM_FORWARD=1" # 启用多模态训练前向。
    "MINICPMO_ALLOW_PACKED_MM_FORWARD=1" # 允许已有多模态前向兼容逻辑。
    "MINICPMO_DISABLE_MM_DEDUP=1" # 避免跨样本媒体去重影响训练对齐。
    "MINICPMO_VIDEO_FPS=${MINICPMO_VIDEO_FPS:-1.0}" # 数据行未指定时的视频抽帧频率。
    "MINICPMO_VIDEO_MAX_FRAMES=${MINICPMO_VIDEO_MAX_FRAMES:-8}" # 数据行未指定时的帧数上限，沿用 OmniMix 默认值。
    "MINICPMO_MM_PROMPT_BUDGET=${MINICPMO_MM_PROMPT_BUDGET:-$ROLLOUT_PROMPT_LENGTH}" # 多模态展开的 prompt token 预算。
)

# ===== 6. Huanyu 精度与 graph：full 使用参考实现并关闭 graph =====
# 切换单个参考算子进行实验时，同时把 HUANYU_PRECISION_PROFILE 改为 custom。
RUNTIME_ENV+=(
    "HUANYU_PRECISION_PROFILE=${HUANYU_PRECISION_PROFILE:-full}" # full 校验完整精度策略，custom 允许独立实验。
    "HUANYU_GRAPH_MODE=${HUANYU_GRAPH_MODE:-0}" # 0 使用 eager，1 开启推理 graph，full 仅允许 0。
    "MINICPMO_HUANYU_GDN_FP32_TRAINING=${MINICPMO_HUANYU_GDN_FP32_TRAINING:-1}" # 训练 GDN 使用 FP32 参考计算。
    "MINICPMO_GDN_FP32_REFERENCE=${MINICPMO_GDN_FP32_REFERENCE:-1}" # rollout GDN 使用 FP32 参考计算。
    "MINICPMO_HUANYU_LAYERWISE_RELOAD=${MINICPMO_HUANYU_LAYERWISE_RELOAD:-1}" # 权重同步采用 Huanyu 逐层加载。
    "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE=${MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE:-1}" # 启用混合块零初始化参考行为。
    "MINICPMO_ISOLATE_MEDIA_ENCODERS=${MINICPMO_ISOLATE_MEDIA_ENCODERS:-1}" # 媒体编码器使用独立精度处理。
    "MINICPMO_GDN_CACHE_BARRIER=${MINICPMO_GDN_CACHE_BARRIER:-1}" # GDN 状态缓存读写增加同步保护。
    "MINICPMO_GDN_PREFILL_REFERENCE=${MINICPMO_GDN_PREFILL_REFERENCE:-1}" # GDN prefill 使用参考实现。
    "MINICPMO_GDN_L2NORM_REFERENCE=${MINICPMO_GDN_L2NORM_REFERENCE:-1}" # GDN L2 归一化使用参考实现。
    "MINICPMO_GDN_CONV_REFERENCE=${MINICPMO_GDN_CONV_REFERENCE:-1}" # GDN 卷积使用参考实现。
    "MINICPMO_GDN_GATING_REFERENCE=${MINICPMO_GDN_GATING_REFERENCE:-1}" # GDN 门控使用参考实现。
    "MINICPMO_PRIVATE_HYBRID_CACHE=${MINICPMO_PRIVATE_HYBRID_CACHE:-1}" # 使用独立混合模型状态缓存。
    "MINICPMO_STABLE_LINEAR=${MINICPMO_STABLE_LINEAR:-1}" # 启用稳定线性层计算路径。
    "MINICPMO_HF_NORM_REFERENCE=${MINICPMO_HF_NORM_REFERENCE:-0}" # 是否替换为 HF 归一化参考路径。
    "MINICPMO_HF_ROUTER_REFERENCE=${MINICPMO_HF_ROUTER_REFERENCE:-0}" # 是否替换为 HF MoE 路由参考路径。
    "MINICPMO_HF_TEXT_ROPE_REFERENCE=${MINICPMO_HF_TEXT_ROPE_REFERENCE:-0}" # 是否启用文本 RoPE 参考路径。
    "MINICPMO_MOE_MATCH_ROLLOUT=${MINICPMO_MOE_MATCH_ROLLOUT:-0}" # 是否启用 MoE rollout 对齐实验。
    "MINICPMO_ISOLATE_FULL_ATTENTION=${MINICPMO_ISOLATE_FULL_ATTENTION:-0}" # 是否隔离 full-attention 计算。
    "MINICPMO_VISION_POSITION_REFERENCE=${MINICPMO_VISION_POSITION_REFERENCE:-0}" # 是否启用视觉位置参考路径。
    "MINICPMO_NATIVE_VISION_REFERENCE=${MINICPMO_NATIVE_VISION_REFERENCE:-0}" # 是否启用原生视觉参考路径。
    "MINICPMO_MOE_TP_REFERENCE=${MINICPMO_MOE_TP_REFERENCE:-0}" # MoE TP4 参考模式，0 关闭、4 开启。
    "MINICPMO_ROW_TP_REFERENCE=${MINICPMO_ROW_TP_REFERENCE:-0}" # 行并行 TP4 参考模式，0 关闭、4 开启。
    "MINICPMO_FSDP2_PIN_MEMORY=0" # 关闭 FSDP2 CPU offload 锁页内存。
    "Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH:-off}" # 默认关闭反向预取，沿用已验证行为。
    "MINICPMO_GDN_STATE_BF16=${MINICPMO_GDN_STATE_BF16:-1}" # 关闭 FP32 参考时的旧 GDN 状态精度开关。
    "HUANYU_JOB_DETERMINISM=0" # 默认不启用整作业强制确定性模式。
)

# ===== 7. Ascend 执行与通信 =====
RUNTIME_ENV+=(
    "VLLM_ASCEND_ENABLE_NZ=0" # 禁止推理权重转换为 NZ 格式。
    "VLLM_ASCEND_WEIGHT_NZ_MODE=0" # 关闭 vLLM Ascend 权重 NZ 模式。
    "ASCEND_WEIGHT_NZ_MODE=0" # 关闭 Ascend 侧权重 NZ 模式。
    "ASCEND_ENABLE_TRANS_DATA_CACHE=0" # 关闭权重布局转换缓存。
    "MULTI_STREAM_MEMORY_REUSE=2" # 沿用多流内存复用策略。
    "TASK_QUEUE_ENABLE=0" # 沿用已验证的任务队列设置。
    "ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}" # 设为 1 可同步定位算子异常，但会降低吞吐。
    "ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}" # ACLNN 算子缓存容量。
    "CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}" # Ascend CPU 亲和性配置。
    "ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}" # Ascend 运行时日志级别。
    "HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}" # HCCL 集合通信执行超时秒数。
    "HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}" # HCCL 建链超时秒数。
    "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}" # HCCL 通信端口起点。
    "ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}" # NPU 同步超时秒数。
    "TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-3600}" # PyTorch 分布式超时秒数。
    "HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}" # HCCL 算子展开模式。
    "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${SOCKET_IFNAME:-eth0}}" # HCCL 使用的网卡。
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME:-eth0}}" # Gloo 使用的网卡。
    "TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-${SOCKET_IFNAME:-eth0}}" # 推理 TP 通信使用的网卡。
)

# ===== 8. 诊断与 GenRM 日志 =====
RUNTIME_ENV+=(
    "HUANYU_GEN_TIMING=${HUANYU_GEN_TIMING:-0}" # 开启 gen 各阶段的 host_wall 计时日志。
    "HUANYU_GEN_TIMING_ENGINE_EVERY=${HUANYU_GEN_TIMING_ENGINE_EVERY:-100}" # 引擎每隔多少轮采集一次计时。
    "HUANYU_FORWARD_PROFILE_DIR=${HUANYU_FORWARD_PROFILE_DIR:-}" # 非空时启用已安装的 forward profiler 钩子。
    "HUANYU_FORWARD_PROFILE_START=${HUANYU_FORWARD_PROFILE_START:-101}" # forward profiler 开始采集的迭代号。
    "HUANYU_FORWARD_PROFILE_ACTIVE=${HUANYU_FORWARD_PROFILE_ACTIVE:-30}" # forward profiler 采集的迭代数。
    "HUANYU_FORWARD_PROFILE_LEVEL=${HUANYU_FORWARD_PROFILE_LEVEL:-1}" # Ascend profiler 详细程度，支持 1 或 2。
    "HUANYU_FORWARD_PROFILE_MODEL=${HUANYU_FORWARD_PROFILE_MODEL:-2748}" # profiler 只匹配该模型目录名。
    "MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-0}" # 开启训练挂起诊断。
    "GENRM_FALLBACK_ON_ERROR=${GENRM_FALLBACK_ON_ERROR:-similarity}" # GenRM 调用失败时的评分回退方式。
    "GENRM_DUMP_DIR=${GENRM_DUMP_DIR:-$HUANYU_RUN_DIR/genrm_dump}" # 保存 GenRM 请求和评分结果。
)

# ===== 9. 数据读取与 chat template =====
ARGS=(
    "trainer.device=npu" # 使用 Ascend NPU 训练。
    "++ray_kwargs.ray_init.address=\"$RAY_ADDRESS\"" # 强制连接已经启动的 Ray 集群。
    "data.train_files=\"$TRAIN_FILE\"" # 训练 parquet 文件。
    "data.val_files=\"$VAL_FILE\"" # 验证 parquet 文件。
    "data.train_batch_size=$TRAIN_BATCH_SIZE" # 每个采样批次的 prompt 数。
    "data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}" # -1 使用全部训练样本。
    "data.val_max_samples=${VAL_MAX_SAMPLES:--1}" # -1 使用全部验证样本。
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" # 展开后的 prompt 长度上限。
    "data.max_response_length=$MAX_RESPONSE_LENGTH" # 训练侧 response 长度上限。
    "data.image_key=images" # parquet 中的图片列名。
    "++data.video_key=videos" # parquet 中的视频列名。
    "++data.audio_key=audios" # parquet 中的音频列名。
    "data.shuffle=true" # 每轮打乱训练样本。
    "data.seed=${DATA_SEED:-42}" # 数据随机种子。
    "data.filter_overlong_prompts=${FILTER_OVERLONG:-false}" # 是否预先过滤超长 prompt，沿用 OmniMix 默认关闭。
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS:-64}" # 超长样本预过滤并行进程数。
    "data.truncation=error" # 遇到仍超限的样本时报告错误，避免静默截断媒体。
    "++data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-true}" # 模板是否开启模型思考模式。
    "data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset" # 使用 Huanyu 多模态数据加载模块。
    "data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset" # 明确选择 Huanyu 数据集类。
    "++data.mm_processor_kwargs.sampling_rate=16000" # 音频目标采样率。
    "++data.mm_processor_kwargs.max_slice_nums=1" # 多模态兼容处理的图片切片数。
    "++data.mm_processor_kwargs.use_image_id=false" # 不添加图片编号标记。
)

# ===== 10. 模型适配、冻结范围与训练精度 =====
ARGS+=(
    "actor_rollout_ref.model.path=\"$MODEL_PATH\"" # Actor/Ref 的初始 checkpoint。
    "actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu" # 显式导入 Huanyu 训练适配器。
    "actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining" # 防止按 HF 原始架构误选 Whisper 适配器。
    "actor_rollout_ref.model.model_stage=thinker" # 只训练 thinker 阶段。
    "actor_rollout_ref.model.trust_remote_code=true" # 允许 checkpoint 的自定义模型代码。
    "++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION:-eager}" # 训练 attention 实现。
    "++actor_rollout_ref.model.override_config.stream_input=false" # 使用非流式输入。
    "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=true" # 冻结音频 encoder 和 projector。
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=false" # 音频编码器不参与优化。
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=false" # 音频投影层不参与优化。
    "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true" # 每个 microbatch 同步分片梯度，避免累计未分片梯度。
    "actor_rollout_ref.model.lora_rank=0" # 不启用 LoRA，训练未冻结的完整参数。
    "actor_rollout_ref.model.use_remove_padding=false" # Huanyu 多模态前向保留 padding。
    "actor_rollout_ref.model.enable_gradient_checkpointing=true" # 用重计算降低激活显存占用。
    "actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD:-false}" # 是否将激活卸载到 CPU。
    'actor_rollout_ref.model.exclude_modules=".*model\.visual.*|.*model\.audio_encoder.*|.*model\.audio_projection.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*"' # 排除冻结媒体模块及音频生成模块。
    "actor_rollout_ref.actor.freeze_vision_tower=true" # 冻结视觉编码器。
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE:-fp32}" # Actor 参数加载精度，沿用 OmniMix 的 FP32 默认值。
    "actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE:-bfloat16}" # Ref 参数加载精度。
    "actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE:-bfloat16}" # vLLM rollout 模型计算精度。
)

# ===== 11. Actor 优化器、GSPO 损失与 batch =====
ARGS+=(
    "actor_rollout_ref.actor.trainer_type=policy_gradient" # 使用策略梯度训练器。
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}" # 优化器学习率。
    "actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY:-0.1}" # 权重衰减系数。
    "actor_rollout_ref.actor.optim.clip_grad=${ACTOR_CLIP_GRAD:-1.0}" # 梯度范数裁剪阈值。
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" # 每次优化的 prompt 数。
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU" # 静态 batch 时的每卡序列数。
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_DYNAMIC_BSZ:-true}" # 根据 token 数拆分训练 microbatch。
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" # 每卡动态训练 token 上限。
    "actor_rollout_ref.actor.use_torch_compile=false" # 训练端关闭 torch.compile。
    "actor_rollout_ref.actor.use_kl_loss=false" # 不加入对 Ref 的 KL loss。
    "actor_rollout_ref.actor.entropy_coeff=0" # 不在 loss 中加入熵奖励。
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true" # 分块计算词表熵，降低显存峰值。
    "actor_rollout_ref.actor.entropy_from_logits_chunk_size=512" # 每次熵计算处理的 token 数。
    "actor_rollout_ref.actor.entropy_checkpointing=false" # 关闭额外的熵计算重算。
    "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" # 使用 GSPO 策略损失。
    "actor_rollout_ref.actor.clip_ratio_low=3e-4" # GSPO 概率比下侧裁剪范围。
    "actor_rollout_ref.actor.clip_ratio_high=4e-4" # GSPO 概率比上侧裁剪范围。
    "actor_rollout_ref.actor.clip_ratio_c=10.0" # 双重裁剪的额外边界。
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" # 先按序列内 token 再按序列平均 loss。
)

# ===== 12. FSDP 分片、offload 与 Ref log_prob =====
ARGS+=(
    "actor_rollout_ref.actor.strategy=fsdp2" # Actor 使用 FSDP2 分片。
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" # Actor 不启用 Ulysses 序列并行。
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1" # 使用整个 Actor world 作为分片范围。
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-true}" # Actor 参数可卸载到 CPU。
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-false}" # 是否卸载优化器状态。
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=${ACTOR_FORWARD_PREFETCH:-false}" # 是否预取下一层参数。
    "actor_rollout_ref.actor.fsdp_config.use_orig_params=true" # 保留原始参数视图。
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false" # FSDP 层关闭编译。
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY:-false}" # 是否启用 FSDP CPUOffloadPolicy。
    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true" # 前向后重新分片以释放完整参数。
    "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true" # 引擎层同步启用分块熵计算。
    "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512" # 引擎层熵计算 token 分块大小。
    "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=false" # 引擎层关闭额外熵重算。
    "actor_rollout_ref.ref.strategy=fsdp2" # Ref 使用 FSDP2。
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1" # Ref 不启用序列并行。
    "actor_rollout_ref.ref.fsdp_config.fsdp_size=-1" # Ref 使用全局分片组。
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-true}" # Ref 参数卸载到 CPU。
    "actor_rollout_ref.ref.fsdp_config.forward_prefetch=${REF_FORWARD_PREFETCH:-true}" # Ref 是否预取参数。
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" # Ref 静态 log_prob 每卡序列数。
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_DYNAMIC_BSZ:-true}" # Ref log_prob 使用动态 batch。
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" # Ref log_prob 每卡 token 预算。
    "actor_rollout_ref.ref.entropy_checkpointing=false" # Ref 关闭熵重算。
    "actor_rollout_ref.ref.entropy_from_logits_with_chunking=false" # Ref 不另开熵分块路径。
)

# ===== 13. rollout 引擎、采样与权重同步 =====
ARGS+=(
    "actor_rollout_ref.rollout.name=vllm_omni" # 使用 vLLM-Omni 生成引擎。
    "actor_rollout_ref.rollout.mode=async" # 使用异步请求接口，训练循环仍由 omni_sync 调度。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar" # 使用自回归文本输出。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu" # 显式选择 Huanyu rollout pipeline。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls=verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension" # 安装共置权重同步与诊断扩展。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0" # 引擎内禁用权重 NZ 转换。
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true" # 开启引擎 RL 配置。
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" # 每个 prompt 的采样次数。
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0}" # 训练采样温度。
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95}" # 核采样概率阈值。
    "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:-20}" # 采样保留的候选 token 数。
    "++actor_rollout_ref.rollout.repetition_penalty=${ROLLOUT_REPETITION_PENALTY:-1.05}" # 重复 token 惩罚系数。
    "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=${ROLLOUT_FREQUENCY_PENALTY:-0.0}" # 按出现次数惩罚 token。
    "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=${ROLLOUT_PRESENCE_PENALTY:-0.0}" # 对已出现 token 施加惩罚。
    "actor_rollout_ref.rollout.ignore_eos=${ROLLOUT_IGNORE_EOS:-false}" # false 时生成 EOS 后可提前停止。
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" # 每个副本的 TP 卡数。
    "actor_rollout_ref.rollout.data_parallel_size=1" # 每个引擎内部的 DP 大小，副本数由资源池另算。
    "actor_rollout_ref.rollout.pipeline_model_parallel_size=1" # 不启用流水线并行。
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.6}" # rollout 的 NPU 内存预算比例。
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}" # 每个引擎同时调度的序列上限。
    "actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN" # 推理总上下文上限。
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS" # 单轮推理调度 token 预算。
    "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH" # 推理 prompt 长度预算。
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH" # 推理输出长度预算。
    "actor_rollout_ref.rollout.load_format=safetensors" # 使用 safetensors 权重格式。
    "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-true}" # 训练阶段释放推理缓存占用。
    "actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_PREFIX_CACHING:-false}" # 是否复用 prompt 前缀 KV cache。
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_CHUNKED_PREFILL:-false}" # 是否将长 prompt 分块 prefill。
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-true}" # old_log_prob 使用动态 batch。
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" # old_log_prob 每卡 token 预算。
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" # old_log_prob 静态每卡序列数。
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB:-1024}" # 权重同步分桶大小，单位 MiB。
    "actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS:-$((NUM_NPUS * NNODES / ROLLOUT_TP))}" # 发起生成请求的 AgentLoopWorker 数。
    "actor_rollout_ref.rollout.val_kwargs.n=1" # 每个验证 prompt 只生成一条结果。
    "actor_rollout_ref.rollout.val_kwargs.temperature=0" # 验证使用贪心采样。
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0" # 验证不做 top-p 截断。
    "actor_rollout_ref.rollout.val_kwargs.top_k=-1" # 验证不做 top-k 截断。
)

# ===== 14. 奖励：默认独立 GenRM 池，可切换为数据集规则奖励 =====
ARGS+=(
    "reward.reward_manager.source=register" # 从已注册的奖励管理器中选择实现。
    "reward.reward_manager.name=naive" # 使用 naive 奖励管理器。
)
case "$HUANYU_REWARD_MODE" in
    RM|rm|genrm)
        RUNTIME_ENV+=(
            "VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext,verl_omni_ext.models.qwen3_6_huanyu,verl_omni_ext.models.qwen3_5_moe" # 显式加载 Actor 和 GenRM 的训练侧适配器，不只依赖安装元数据发现。
            "VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_huanyu.engine_hooks,verl_omni_ext.models.qwen3_5_moe.vllm_omni" # 每个推理进程同时注册 Huanyu 与 GenRM 的 AR pipeline。
        )
        ARGS+=(
            "reward.custom_reward_function.path=pkg://verl_omni_ext.reward.genrm_verifier" # 使用 GenRM 评分函数。
            "reward.custom_reward_function.name=compute_score" # GenRM 评分函数入口。
            "reward.reward_model.enable=true" # 启用奖励模型。
            "reward.reward_model.enable_resource_pool=true" # 奖励模型使用独立卡池。
            "reward.reward_model.model_path=\"$RM_MODEL_PATH\"" # 奖励模型 checkpoint。
            "reward.reward_model.n_gpus_per_node=$RM_N_GPUS_PER_NODE" # 奖励池每节点卡数。
            "reward.reward_model.nnodes=$RM_NNODES" # 奖励池节点数。
            "reward.reward_model.rollout.name=vllm_omni" # GenRM 使用 vLLM-Omni。
            "++reward.reward_model.rollout.engine_kwargs.vllm_omni.output_mode=ar" # GenRM 使用文本自回归输出。
            "++reward.reward_model.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_5_moe" # 选择 Qwen3.5 MoE 的 AR pipeline。
            "reward.reward_model.rollout.tensor_model_parallel_size=$RM_TP" # GenRM 每副本 TP 卡数。
            "reward.reward_model.rollout.gpu_memory_utilization=${RM_GPU_MEM_UTIL:-0.7}" # 沿用 OmniMix 的独立奖励池内存比例。
            "reward.reward_model.rollout.free_cache_engine=false" # 独立奖励池保留引擎缓存。
            "reward.reward_model.rollout.prompt_length=$RM_PROMPT_LENGTH" # GenRM prompt 上限。
            "reward.reward_model.rollout.response_length=$RM_RESPONSE_LENGTH" # GenRM 输出上限。
            "reward.reward_model.rollout.max_model_len=$RM_MAX_MODEL_LEN" # GenRM 总上下文上限。
            "reward.reward_model.rollout.max_num_batched_tokens=${RM_MAX_NUM_BATCHED_TOKENS:-$RM_MAX_MODEL_LEN}" # GenRM 单轮调度 token 预算。
            "++reward.reward_model.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0" # GenRM 关闭权重 NZ 转换。
            "++reward.reward_model.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true" # GenRM 启用引擎 RL 兼容配置。
            "reward.reward_model.rollout.max_num_seqs=${RM_MAX_NUM_SEQS:-64}" # GenRM 每副本并发序列上限。
            "reward.reward_model.rollout.dtype=${RM_DTYPE:-bfloat16}" # 奖励模型推理精度。
            "reward.reward_model.rollout.enable_prefix_caching=false" # GenRM 关闭前缀缓存。
            "reward.reward_model.rollout.enable_chunked_prefill=false" # GenRM 不使用分块 prefill。
            "reward.reward_model.rollout.skip_tokenizer_init=false" # GenRM 初始化 tokenizer。
            "++reward.reward_model.rollout.full_determinism=false" # GenRM 不启用全确定性模式。
            "++reward.reward_model.rollout.seed=42" # GenRM 随机种子。
        )
        ;;
    rule_based|rule)
        RUNTIME_ENV+=(
            "VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext,verl_omni_ext.models.qwen3_6_huanyu" # 规则模式显式加载 Huanyu 训练适配器。
            "VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_huanyu.engine_hooks" # 规则模式只需要 Huanyu 推理扩展。
        )
        ARGS+=(
            "reward.reward_model.enable=false" # 规则模式不加载奖励模型。
            "reward.reward_model.enable_resource_pool=false" # 规则模式不申请额外奖励卡池。
            "reward.custom_reward_function.path=null" # 根据 parquet 的 data_source 分派已有规则，需数据源有对应 scorer。
        )
        ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE 必须为 RM 或 rule_based' >&2; exit 2 ;;
esac

# ===== 15. 算法、训练调度、验证与 checkpoint =====
ARGS+=(
    "algorithm.adv_estimator=grpo" # 根据同一 prompt 的多条奖励计算组内优势。
    "algorithm.trainer_type=policy_gradient" # 使用策略梯度算法路径。
    "algorithm.sample_source=online" # 每个 step 在线采样轨迹。
    "algorithm.use_kl_in_reward=false" # 不在奖励中附加 KL 惩罚。
    "algorithm.norm_adv_by_std_in_grpo=true" # 按组内标准差归一化优势。
    "trainer.v1.trainer_mode=omni_sync" # 采样和更新按同步训练周期执行。
    "trainer.n_gpus_per_node=$NUM_NPUS" # Actor 池每节点设备数，NPU 仍沿用此配置键名。
    "trainer.nnodes=$NNODES" # Actor 池节点数。
    "trainer.resume_mode=${RESUME_MODE:-disable}" # 默认不自动恢复旧 checkpoint。
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-false}" # 是否在首步训练前执行验证。
    "trainer.balance_batch=true" # 按序列长度均衡各训练 rank 的工作量。
    "trainer.critic_warmup=0" # GRPO 不设置 critic 预热步骤。
    "trainer.project_name=$PROJECT_NAME" # 日志项目名称。
    "trainer.experiment_name=$EXPERIMENT_NAME" # 日志实验名称。
    "trainer.default_local_dir=\"${DEFAULT_LOCAL_DIR:-$HUANYU_RUN_DIR/checkpoints}\"" # checkpoint 输出目录。
    "trainer.rollout_data_dir=\"$HUANYU_RUN_DIR/rollout_dump\"" # 保存训练生成结果。
    "trainer.validation_data_dir=\"$HUANYU_RUN_DIR/validation_dump\"" # 保存验证生成结果。
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','extra']" # 保存模型和额外状态，不保存优化器状态。
    "trainer.save_freq=${SAVE_FREQ:-50}" # 每多少训练 step 保存一次 checkpoint。
    "trainer.test_freq=${TEST_FREQ:-10}" # 每多少训练 step 验证一次。
    "trainer.total_epochs=${TOTAL_EPOCHS:-10}" # 遍历训练集的最大轮数。
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" # 限制总更新步数，null 由 epoch 决定。
    "trainer.logger=${TRAINER_LOGGER:-[console]}" # 日志输出后端列表。
)

# ===== 执行部分：将同一配置同时应用于 driver 与 Ray worker =====
for _entry in "${RUNTIME_ENV[@]}"; do export "$_entry"; done
case "$HUANYU_GRAPH_MODE" in
    0|false) _eager=true; _compile=0; _disable=1 ;;
    1|true) _eager=false; _compile=1; _disable=0 ;;
    *) echo 'FATAL: HUANYU_GRAPH_MODE 必须为 0/1 或 false/true' >&2; exit 2 ;;
esac
if [[ "$HUANYU_PRECISION_PROFILE" == full && "$_eager" != true ]]; then
    echo 'FATAL: full 精度模式不能开启 graph；graph 实验需显式选择 custom' >&2
    exit 2
fi
RUNTIME_ENV+=(
    "VLLM_TORCH_COMPILE=$_compile" # 与 HUANYU_GRAPH_MODE 联动的推理编译开关。
    "TORCH_COMPILE_DISABLE=$_disable" # 与 graph 模式联动的编译禁用开关。
    "NPU_GRAPH_CAPTURE_DISABLE=$_disable" # 与 graph 模式联动的 NPU 捕图禁用开关。
)
ARGS+=(
    "actor_rollout_ref.rollout.enforce_eager=$_eager" # Actor rollout 的执行模式由 HUANYU_GRAPH_MODE 决定。
    "reward.reward_model.rollout.enforce_eager=$_eager" # GenRM 的执行模式由 HUANYU_GRAPH_MODE 决定。
)

_dry_run=0
if [[ "${1:-}" == --dry-run ]]; then _dry_run=1; shift; fi
# 同步末尾 CLI 覆盖后的真实 rollout TP，供精度与权重对齐补丁使用。
_effective_tp="$ROLLOUT_TP"
for _arg in "$@"; do
    _key="${_arg%%=*}"
    while [[ "$_key" == +* ]]; do _key="${_key#+}"; done
    if [[ "$_key" == actor_rollout_ref.rollout.tensor_model_parallel_size ]]; then
        _effective_tp="${_arg#*=}"
    fi
done
RUNTIME_ENV+=("MINICPMO_HUANYU_ROLLOUT_TP=$_effective_tp") # 向 worker 明确传递实际 rollout TP。
_ENV_ARGS=()
for _entry in "${RUNTIME_ENV[@]}"; do
    export "$_entry"
    _key="${_entry%%=*}"
    _value="${_entry#*=}"
    _value="${_value//\\/\\\\}"
    _value="${_value//\"/\\\"}"
    _ENV_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.$_key=\"$_value\"")
done
if (( _dry_run )); then
    printf '%q ' "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "${_ENV_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

[[ -d "$PROJECT_ROOT" ]] || { echo "FATAL: 仓库不存在：$PROJECT_ROOT" >&2; exit 2; }
# 数据、权重和缓存由框架按最终 Hydra 配置检查，避免误检查被 CLI 覆盖的默认路径。
cd -- "$PROJECT_ROOT"
mkdir -p -- "$HUANYU_RUN_DIR"
exec > >(tee -a "$HUANYU_RUN_DIR/console.log") 2>&1
printf '[huanyu-train] model=%s train=%s val=%s\n' "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"
printf '[huanyu-train] actor=%sx%s reward=%s ray=%s run=%s\n' "$NUM_NPUS" "$NNODES" "$HUANYU_REWARD_MODE" "$RAY_ADDRESS" "$HUANYU_RUN_DIR"
exec "$PYTHON_BIN" -m verl_omni.trainer.main_omni "${ARGS[@]}" "${_ENV_ARGS[@]}" "$@"
