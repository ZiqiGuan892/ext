#!/usr/bin/env bash
# =============================================================================
# Qwen3.6-35B-A3B-whisper-0814-rl thinker · GSPO · Ascend NPU · OmniMix · GenRM
# =============================================================================
# 入口脚本：设置 OmniMix + GenRM 环境变量，然后 exec 主脚本。
# 主脚本 run_qwen36_whisper_gspo_npu.sh 负责所有训练参数。
#
# 移植自 verl_omni_minicpmo_debug/examples/minicpmo_5_0/
# run_minicpmo_thinker_gspo_npu_omni_mix_genrm.sh，调整为新模型 Qwen3.6-whisper。
#
# 与 5.0 基线入口脚本的差异：
#   · MODEL_PATH → Qwen3.6-35B-A3B-whisper-0814-rl
#   · data.custom_cls.path → pkg://verl_omni_ext.models.qwen3_6_whisper.dataset
#   · data.custom_cls.name → Qwen36WhisperOmniRLHFDataset
#   · exclude_modules 正则更新为新模型模块路径（model.visual.* /
#     model.audio_encoder.* / model.audio_projection.*）
#   · rollout pipeline_name → qwen3_6_whisper（verl 侧注册键；主脚本里设）
#
# 其他训练配置与 5.0 基线入口脚本保持一致（DEBUG_SAMPLES / RM_MAX_MODEL_LEN /
# optimizer_offload=false / sampleall 路径 / filter_overlong_prompts 等）。
# =============================================================================

set -eo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- GenRM 配置 ----
export RM_ENABLE=1
export RM_MODEL_PATH=${RM_MODEL_PATH:-"/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B"}

# ---- 集群拓扑 ----
export NNODES=1
export NUM_NPUS=16               # 训练用 16 卡
export ROLLOUT_TP=4              # 4 个训练 replica
export RM_TP=4
export RM_N_GPUS_PER_NODE=16     # judge 16 卡
export RM_NNODES=1               # judge 在 1 个节点（worker）
export RM_SPLIT_NODES=1          # judge 在独立节点，跳过单机校验
export RM_GPU_MEM_UTIL=${RM_GPU_MEM_UTIL:-0.7}
export RM_PROMPT_LENGTH=${RM_PROMPT_LENGTH:-4096}
export RM_RESPONSE_LENGTH=${RM_RESPONSE_LENGTH:-1024}
export RM_MAX_MODEL_LEN=${RM_MAX_MODEL_LEN:-8192}
export RM_NUM_REWARD_WORKERS=${RM_NUM_REWARD_WORKERS:-8}

# ---- OmniMix 数据集路径 ----
export TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/train.parquet"}
export VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/validation.parquet"}

# ---- 模型路径：Qwen3.6-35B-A3B-whisper-0814-rl ----
export MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl"}

# ---- 预计算缓存 ----
export MINICPMO_EMBED_CACHE_DIR=${MINICPMO_EMBED_CACHE_DIR:-/opt/huawei/dataset/ccb_data/data/omni/embed_cache_qwen36_sampleall}
export MINICPMO_VIDEO_MAX_FRAMES=${MINICPMO_VIDEO_MAX_FRAMES:-8}
export MINICPMO_VIDEO_FPS=1.0

# ---- forward adapter 相关 ----
export MINICPMO_ENABLE_MM_FORWARD=1
export MINICPMO_ALLOW_PACKED_MM_FORWARD=1
export MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-0}
export MINICPMO_FSDP2_PIN_MEMORY=0
export ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-false}

# ---- 禁用 MiniCPM-o 风格的 prompt token 收缩 ----
# Qwen3.6 用显式 image_token_id/audio_token_id，不是 MiniCPM-o 的 <image>./</image> 文本模式
# _contract_media_spans 会用 pad_token_id(0) 冒充 im_start_id，导致 prompt 全面错乱
export MINICPMO_DISABLE_MM_DEDUP=1

# ---- 模型 dtype（代码1 经验：actor fp32，ref bfloat16）----
export ACTOR_MODEL_DTYPE=fp32
export REF_MODEL_DTYPE=bfloat16

# ---- OmniMix 数据集使用自定义 dataset class ----
export MINICPMO_USE_CUSTOM_DATASET=1
export VLLM_ASCEND_WEIGHT_NZ_MODE=0
export ASCEND_WEIGHT_NZ_MODE=0
export ASCEND_ENABLE_TRANS_DATA_CACHE=0

# ---- Debug 模式 ----
export DEBUG_MODE=${DEBUG_MODE:-0}
if [ "${DEBUG_MODE}" = "1" ]; then
    export TRAIN_BATCH_SIZE=32
    export ROLLOUT_N=2
    export MAX_RESPONSE_LENGTH=512
    export MAX_PROMPT_LENGTH=3072
    export TOTAL_TRAINING_STEPS=2
    export PPO_MINI_BATCH_SIZE=32
    # DEBUG_SAMPLES: 生成一个只有 N 条的 mini parquet，跳过 122K 样本的 prompt 过滤
    # 用法: DEBUG_SAMPLES=4 DEBUG_MODE=1 bash ..._omni_mix_genrm.sh
    export DEBUG_SAMPLES=${DEBUG_SAMPLES:-0}
    if [ "${DEBUG_SAMPLES}" -gt 0 ] 2>/dev/null; then
        _MINI_DIR="/opt/huawei/dataset/ccb_data/data/omni/mini_debug"
        mkdir -p "${_MINI_DIR}"
        _MINI_TRAIN="${_MINI_DIR}/mini_train_${DEBUG_SAMPLES}.parquet"
        _MINI_VAL="${_MINI_DIR}/mini_val_${DEBUG_SAMPLES}.parquet"
        python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${TRAIN_FILE}')
mini = t.slice(0, ${DEBUG_SAMPLES})
pq.write_table(mini, '${_MINI_TRAIN}')
v = pq.read_table('${VAL_FILE}')
mini_v = v.slice(0, max(1, ${DEBUG_SAMPLES}//4))
pq.write_table(mini_v, '${_MINI_VAL}')
print(f'[debug] mini parquet: {${DEBUG_SAMPLES}} train + {max(1, ${DEBUG_SAMPLES}//4)} val')
"
        export TRAIN_FILE=${_MINI_TRAIN}
        export VAL_FILE=${_MINI_VAL}
        export FILTER_OVERLONG=false
        # DEBUG_SAMPLES 必须满足整除约束:
        #   TRAIN_BATCH_SIZE <= DEBUG_SAMPLES
        #   PPO_MINI_BATCH_SIZE * ROLLOUT_N 被 world_size(16) 整除
        #   PPO_MINI_BATCH_SIZE 被 ROLLOUT_TP(4) 整除
        # 推荐值: 16, 32, 64（= 16 的倍数）
        # 4 条数据太小: TRAIN_BATCH_SIZE=32 > 4 会报错
        _WS=${NUM_NPUS}
        _TP=${ROLLOUT_TP}
        # 确保 batch <= samples, batch 被 TP 整除, batch*N 被 WS 整除
        _bs=${DEBUG_SAMPLES}
        _bs=$((_bs / _TP * _TP))           # 向下取整到 TP 的倍数
        _bs=$((_bs < _WS ? _bs < _TP ? _TP : _TP * (_bs / _TP) : _bs))
        if [ "$_bs" -lt "$_TP" ]; then _bs=$_TP; fi
        export TRAIN_BATCH_SIZE=$_bs
        export PPO_MINI_BATCH_SIZE=$_bs
        # ROLLOUT_N: batch*N 被 WS 整除 → N = WS / gcd(batch, WS)
        export ROLLOUT_N=$((_WS / _bs * (_bs / _TP)))
        if [ "${ROLLOUT_N}" -lt 1 ]; then export ROLLOUT_N=1; fi
    fi
    echo "[debug] DEBUG_MODE=1: batch=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}, resp_len=${MAX_RESPONSE_LENGTH}, prompt_len=${MAX_PROMPT_LENGTH}, steps=${TOTAL_TRAINING_STEPS}, samples=${DEBUG_SAMPLES}"
else
    # ---- 正式训练参数（全量数据集）----
    # 不限制步数，合理 batch + 序列长度，避免 update_actor 过长导致心跳超时
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
    export ROLLOUT_N=${ROLLOUT_N:-4}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:--1}
    echo "[train] Full dataset: batch=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}, mini_batch=${PPO_MINI_BATCH_SIZE}, prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, steps=${TOTAL_TRAINING_STEPS}"
fi

# ---- 实验名 ----
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_omni_mix_npu_genrm_qwen36}
export PROJECT_NAME=${PROJECT_NAME:-omni_mix_qwen36}

# ---- OmniMix 特有的 hydra 参数 ----
# 覆盖主脚本的默认值，对齐 NPU 经验值
_OMNI_MIX_ARGS=(
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_whisper.dataset
    data.custom_cls.name=Qwen36WhisperOmniRLHFDataset
    ++data.video_key=videos
    ++data.audio_key=audios
    data.filter_overlong_prompts=${FILTER_OVERLONG:-true}
    ++data.mm_processor_kwargs.sampling_rate=16000
    ++data.mm_processor_kwargs.max_slice_nums=1
    ++data.mm_processor_kwargs.use_image_id=false
    +actor_rollout_ref.model.override_config.stream_input=false
    actor_rollout_ref.model.override_config.attn_implementation=eager
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_activation_offload=false
    # 新模型模块路径：model.visual.* / model.audio_encoder.* / model.audio_projection.*
    # （旧模型是 vpm.* / apm.* / audio_projection_layer.*）
    actor_rollout_ref.model.exclude_modules=".*model\.visual.*|.*model\.audio_encoder.*|.*model\.audio_projection.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*"
    actor_rollout_ref.actor.freeze_vision_tower=true
    actor_rollout_ref.actor.use_dynamic_bsz=true
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY}
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']"
    actor_rollout_ref.actor.entropy_checkpointing=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=5120
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true 
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.max_num_seqs=64
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=5120
    actor_rollout_ref.ref.entropy_checkpointing=False
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=False
)

# ---- GenRM 奖励参数（Hydra 覆盖）----
_REWARD_ARGS=(
    reward.custom_reward_function.path=pkg://verl_omni_ext.reward.genrm_verifier
    reward.custom_reward_function.name=compute_score
    reward.reward_model.enable=True
    reward.reward_model.enable_resource_pool=True
    reward.reward_model.model_path=${RM_MODEL_PATH}
    reward.reward_model.n_gpus_per_node=${RM_N_GPUS_PER_NODE}
    reward.reward_model.nnodes=${RM_NNODES}
    reward.reward_model.rollout.name=vllm_omni
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.output_mode=ar
    ++reward.reward_model.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_5_moe
    reward.reward_model.rollout.tensor_model_parallel_size=${RM_TP}
    reward.reward_model.rollout.gpu_memory_utilization=${RM_GPU_MEM_UTIL}
    reward.reward_model.rollout.free_cache_engine=False
    reward.reward_model.rollout.prompt_length=${RM_PROMPT_LENGTH}
    reward.reward_model.rollout.response_length=${RM_RESPONSE_LENGTH}
    reward.reward_model.rollout.max_model_len=${RM_MAX_MODEL_LEN}
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true 
    reward.reward_model.rollout.max_num_seqs=64
    reward.reward_model.rollout.dtype=bfloat16
    reward.reward_model.rollout.enforce_eager=true
    reward.reward_model.rollout.enable_prefix_caching=false
    reward.reward_model.rollout.enable_chunked_prefill=False
    reward.reward_model.rollout.skip_tokenizer_init=False
    ++reward.reward_model.rollout.full_determinism=false
    ++reward.reward_model.rollout.seed=42
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
)

# ---- 卡数校验 ----
if [[ "${RM_ENABLE}" == "1" ]]; then
    _RM_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))
    if (( RM_N_GPUS_PER_NODE % RM_TP != 0 )); then
        echo "[FATAL] RM_N_GPUS_PER_NODE=${RM_N_GPUS_PER_NODE} must be divisible by RM_TP=${RM_TP}" >&2
        exit 2
    fi
    if (( _RM_NPUS < RM_TP )); then
        echo "[FATAL] judge total NPUs=${_RM_NPUS} < RM_TP=${RM_TP}" >&2
        exit 2
    fi
    echo "[genrm] RM_ENABLE=1, judge=${_RM_NPUS}NPU (TP=${RM_TP}, replicas=$((_RM_NPUS / RM_TP)))"
    echo "[genrm] RM_SPLIT_NODES=${RM_SPLIT_NODES} (1=judge 在独立节点)"
fi

echo "=== Qwen3.6-35B-A3B-whisper OmniMix GenRM ==="
echo "train      = ${TRAIN_FILE}"
echo "val        = ${VAL_FILE}"
echo "model      = ${MODEL_PATH}"
echo "judge      = ${RM_MODEL_PATH}"
echo "cache_dir  = ${MINICPMO_EMBED_CACHE_DIR}"
echo "debug      = DEBUG_MODE=${DEBUG_MODE}"
echo

# exec 主脚本，传递 OmniMix + GenRM 参数
exec bash "${_HERE}/run_qwen36_whisper_gspo_npu.sh" "${_OMNI_MIX_ARGS[@]}" "${_REWARD_ARGS[@]}" "$@"