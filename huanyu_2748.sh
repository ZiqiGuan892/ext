# Sourced by run.sh for the original Huanyu 2748 recipe.
Q36_CKPT_DIR="${Q36_CKPT_DIR:-/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl}"
AVQA_OPTIMIZED="${AVQA_OPTIMIZED:-0}" # 1：融合 GDN/RMS/MoE + 图模式，保持 LoRA 配方。
AVQA_REPLAY="${AVQA_REPLAY:-0}" # 专用开关，避免继承其他任务的 HUANYU_EXPERT_REPLAY。
case "$AVQA_OPTIMIZED" in
    0) precision=custom; gdn_fp32=1 ;;
    1)
        precision=kernel_graph; gdn_fp32=0
        [[ "${HUANYU_GRAPH_MODE:-1}" == 1 ]] || {
            echo 'AVQA_OPTIMIZED=1 要求 HUANYU_GRAPH_MODE=1；原配方 eager 对照请用 AVQA_OPTIMIZED=0' >&2
            exit 2
        }
        ;;
    *) echo 'AVQA_OPTIMIZED 必须为 0 或 1' >&2; exit 2 ;;
esac
case "$AVQA_REPLAY" in
    0) replay=false ;;
    1)
        [[ "$AVQA_OPTIMIZED" == 1 ]] || {
            echo 'AVQA_REPLAY=1 要求 AVQA_OPTIMIZED=1：回放依赖融合 MoE kernel' >&2
            exit 2
        }
        replay=true
        # 未设置白名单时保持 vLLM 自动发现；已有白名单则追加回放插件。
        if [[ -v VLLM_PLUGINS ]]; then
            case ",$VLLM_PLUGINS," in
                *,huanyu_r3,*) ;;
                *) VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}huanyu_r3" ;;
            esac
        fi
        ;;
    *) echo 'AVQA_REPLAY 必须为 0 或 1' >&2; exit 2 ;;
esac
ARGS+=(
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu" # Huanyu 专用注册入口。
    "data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset"
    "data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset"
    "actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu"
    "actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining"
    "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=true"
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=false"
    "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=false"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls=verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
)
if [[ "$AVQA_OPTIMIZED" == 1 ]]; then
    ARGS+=(
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mamba_ssm_cache_dtype=bfloat16" # 原生 GDN kernel 使用的状态精度。
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true" # 引擎 RL 兼容配置。
    )
fi
RUNTIME_ENV+=(
    "Q36_CKPT_DIR=$Q36_CKPT_DIR"
    "VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext,verl_omni_ext.models.qwen3_6_huanyu"
    "VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_huanyu.engine_hooks"
    "VERL_OMNI_EXT_PATCH_STRICT=1"
    "HUANYU_VLLM_OMNI_ROOT=${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
    "HUANYU_VLLM_SOURCE=${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
    "HUANYU_FREEZE_AUDIO=true" "HUANYU_TRAIN_AUDIO_ENCODER=false" "HUANYU_TRAIN_AUDIO_PROJECTOR=false"
    "HUANYU_AQA_AUDIO_TRAIN=0" "HUANYU_STEM_TEXT_TRAIN=0"
    "HUANYU_USE_PRECOMPUTE=0" "HUANYU_EMBED_CACHE_DIR=" "MINICPMO_USE_PRECOMPUTE=0"
    "MINICPMO_TUNE_AUDIO=0" "MINICPMO_TUNE_VISION=0"
    "MINICPMO_USE_CUSTOM_DATASET=1" "MINICPMO_ENABLE_MM_FORWARD=1"
    "MINICPMO_ALLOW_PACKED_MM_FORWARD=1" "MINICPMO_DISABLE_MM_DEDUP=1"
    "HUANYU_PRECISION_PROFILE=$precision" # 不继承 full 参考模式的强制 eager。
    # 显式指定整组开关，避免继承此前 AQA 对照实验的零散配置。
    "MINICPMO_GDN_NPU_KERNEL=$AVQA_OPTIMIZED" "MINICPMO_NPU_RMS_NORM=$AVQA_OPTIMIZED" "MINICPMO_MOE_NPU_KERNEL=$AVQA_OPTIMIZED"
    "HUANYU_EXPERT_REPLAY=$AVQA_REPLAY" "HUANYU_KERNEL_BISECT=none" "HUANYU_REPLAY_DEBUG=0"
    "HUANYU_MOE_FINITE_DEBUG=0" "HUANYU_FINITE_PROBE=0"
    "HUANYU_WSYNC_PROFILE=off" "HUANYU_WSYNC_TRACE=off" "HUANYU_WSYNC_VERIFY=0"
    "HUANYU_WSYNC_LAYOUT=passthrough" "Q36_BACKWARD_PREFETCH=off"
    "MINICPMO_HUANYU_GDN_FP32_TRAINING=$gdn_fp32" # 融合 GDN 与 FP32 参考实现互斥。
    "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE=$AVQA_OPTIMIZED" # kernel_graph 档案要求的 hybrid 状态初始化。
    "MINICPMO_ISOLATE_MEDIA_ENCODERS=1" "MINICPMO_HUANYU_LAYERWISE_RELOAD=1"
    "MINICPMO_FSDP2_PIN_MEMORY=0"
    "HUANYU_GEN_TIMING=0" "HUANYU_FORWARD_PROFILE_DIR="
    "HUANYU_FULL_WEIGHT_AUDIT_DIR=" "HUANYU_WEIGHT_SYNC_AUDIT_DIR=" "HUANYU_EMBEDDING_AB_DIR="
    "HUANYU_JOB_DETERMINISM=0" "MINICPMO_HANG_PROBE=0"
    "HUANYU_TRAIN_HANG_DIR=${HUANYU_TRAIN_HANG_DIR:-}" # 非空时记录训练前向阶段、内存与线程栈。
    "HUANYU_TRAIN_HANG_SECONDS=${HUANYU_TRAIN_HANG_SECONDS:-120}" # 前向未返回时每隔该秒数保存线程栈，不终止进程。
    "HUANYU_TRAIN_HANG_CALLS=${HUANYU_TRAIN_HANG_CALLS:-2}" # 每个 Actor 仅记录最初几次模型前向。
)
for key in MINICPMO_GDN_FP32_REFERENCE \
    MINICPMO_GDN_CACHE_BARRIER MINICPMO_GDN_PREFILL_REFERENCE MINICPMO_GDN_L2NORM_REFERENCE \
    MINICPMO_GDN_CONV_REFERENCE MINICPMO_GDN_GATING_REFERENCE MINICPMO_PRIVATE_HYBRID_CACHE \
    MINICPMO_STABLE_LINEAR MINICPMO_HF_NORM_REFERENCE MINICPMO_HF_ROUTER_REFERENCE \
    MINICPMO_HF_TEXT_ROPE_REFERENCE MINICPMO_MOE_MATCH_ROLLOUT MINICPMO_ISOLATE_FULL_ATTENTION \
    MINICPMO_VISION_POSITION_REFERENCE MINICPMO_NATIVE_VISION_REFERENCE \
    MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE; do
    RUNTIME_ENV+=("$key=0")
done
# CLI 若显式覆盖 TP/输入预算，将诊断和预处理环境与最终 Hydra 参数保持一致。
effective_tp=2; effective_prompt=4160
for arg in "$@"; do
    key="${arg%%=*}"
    while [[ "$key" == +* ]]; do key="${key#+}"; done
    case "$key" in
        actor_rollout_ref.rollout.tensor_model_parallel_size) effective_tp="${arg#*=}" ;;
        actor_rollout_ref.rollout.prompt_length) effective_prompt="${arg#*=}" ;;
    esac
done
RUNTIME_ENV+=("MINICPMO_HUANYU_ROLLOUT_TP=$effective_tp" "MINICPMO_MM_PROMPT_BUDGET=$effective_prompt")
