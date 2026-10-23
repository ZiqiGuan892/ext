#!/usr/bin/env bash
# 32 卡全量跑 Qwen3.6-Huanyu 的 **thin wrapper**。
#
# 它不重写训练命令——真正的 verl 训练命令、前置自检、数据集 schema 检查、尺寸
# 硬检查全在共享后端 run_qwen36_huanyu_gspo_npu.sh 里；OmniMix 数据集 + GenRM 判官
# 的参数由 run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh 追加（它再 exec 后端）。
# 本脚本合并原 no_length_filter.sh 的配置：
#   1. 选择 /vllm-workspace/vllm-omni，关闭 train/validation 的长度预筛；
#   2. 指定完整 Huanyu checkpoint；
#   3. 保留 32 卡默认配置（global 16 + reward 16 = 32，NNODES=1）；
#   4. 默认关闭累计 frequency_penalty，保留环境变量显式覆盖。
#
# 口径来源：run_qwen36_huanyu_gspo_npu.sh 的「Ray 侧资源账本校验」段
#     （grep -n "Ray cluster only has" 定位；我加过注释后行号会漂，别写死行号）
#     _want = NUM_NPUS * NNODES + REWARD_POOL_NPUS   # 必须 ≤ ray 集群 NPU
# 32 卡满配 = global 16（actor/rollout）×1 + reward 16（genrm 判官）。
# ⚠️ 绝不能设 NUM_NPUS=32 + 默认 reward 16 —— 那会要 48 > 32，脚本 FATAL。
#
# 前置：各节点由你 source 更新后的 env_export.sh，并按新 vLLM-Omni 路径启动
# 32 NPU Ray 集群。本脚本不 source 环境、不安装依赖、不 start/stop ray。
# 此处导出路径仅影响新进程，不能切换已启动的旧 Ray Worker。
#
# 用法（由 run_huanyu.sh 传入 Huanyu 适配参数后调用）：
#   bash examples/qwen3_6_huanyu/run_qwen36_huanyu_32card.sh
#
# 起跑后等自检通过、看到一条训练 loss 后，核对 fp 生效证据（每进程一行）：
#   grep "rollout-sampling.*frequency_penalty" <本次运行目录>/console.log
#   diag 应看到 frequency_penalty←0.0；非默认惩罚仍由 sampling_params 日志列出。

set -euo pipefail

# 支持从任意目录启动；依赖脚本统一从 PROJECT_ROOT 读取。
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/hfy-model-gy/verl_omni_qwen36_gzq_new}"
_REPO_ROOT="${PROJECT_ROOT}"
cd "$_REPO_ROOT"

# ---------------------------------------------------------------- 0) vLLM-Omni 版本及长度预筛
_huanyu_vllm_omni_root=/vllm-workspace/vllm-omni
if [ ! -f "${_huanyu_vllm_omni_root}/vllm_omni/__init__.py" ]; then
    echo "ERROR: ${_huanyu_vllm_omni_root}/vllm_omni/__init__.py not found; refusing to use the old vllm-omni-xy checkout." >&2
    exit 1
fi

# 保留 ext/verl 等路径，清除已知旧版路径和新版本的重复项。
_huanyu_retained_pythonpath=""
IFS=: read -r -a _huanyu_python_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_python_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|/opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|/vllm-workspace/vllm-omni|/opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit)
            continue ;;
    esac
    _huanyu_retained_pythonpath="${_huanyu_retained_pythonpath:+${_huanyu_retained_pythonpath}:}${_huanyu_path}"
done
export PYTHONPATH="${_huanyu_vllm_omni_root}${_huanyu_retained_pythonpath:+:${_huanyu_retained_pythonpath}}"
export FILTER_OVERLONG=false
echo "[huanyu config] vLLM-Omni Python root: ${_huanyu_vllm_omni_root}; FILTER_OVERLONG=${FILTER_OVERLONG}"
unset _huanyu_vllm_omni_root _huanyu_retained_pythonpath _huanyu_python_entries _huanyu_path

# ---------------------------------------------------------------- 1) 模型
# 默认选择 Huanyu，保留调用方显式传入的 checkpoint。
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens}"

# ---------------------------------------------------------------- 2) 尺寸口径
# Direct-entry defaults only. An upper job/profile can supply every value;
# never reset an explicit 12x2 actor / 4x2 RM allocation to 16+16.
export NUM_NPUS="${NUM_NPUS:-16}"        # actor/rollout NPUs per pool node
export NNODES="${NNODES:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export RM_N_GPUS_PER_NODE="${RM_N_GPUS_PER_NODE:-16}" RM_NNODES="${RM_NNODES:-1}"
export RM_TP="${RM_TP:-4}" RM_SPLIT_NODES="${RM_SPLIT_NODES:-1}"
export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES))

# ---------------------------------------------------------------- 3) 采样
# fp=0.5 会把反复使用的日期、汉字和公式 token 持续扣分。在修复 FP32 GDN
# 后的四样本对照中，关闭 fp 恢复了图表题的中文推导和答案结束符。
# 先前禁止 fp=0 的守卫基于旧推理路径，不能继续作为当前默认值的依据。
export ROLLOUT_FREQUENCY_PENALTY="${ROLLOUT_FREQUENCY_PENALTY:-0.0}"

# ---------------------------------------------------------------- 4) 起跑
echo "=== Huanyu MODEL_PATH=${MODEL_PATH} ==="
echo "=== 32 卡全量包装：NUM_NPUS=${NUM_NPUS} NNODES=${NNODES} ROLLOUT_TP=${ROLLOUT_TP} REWARD_POOL_NPUS=${REWARD_POOL_NPUS} ==="
echo "===   ROLLOUT_FREQUENCY_PENALTY=${ROLLOUT_FREQUENCY_PENALTY}  ROLLOUT_REPETITION_PENALTY=${ROLLOUT_REPETITION_PENALTY:-1.05} ==="

# 真训练/自检/尺寸检查/日志捕获全在下游脚本（pipeline_name=qwen3_6_whisper
# 已由后端脚本设好，这里不重复传，避免 hydra 键冲突）。
#
# 入口是 OmniMix+GenRM 那层（目标跑法：GenRM 判官打分），不是裸后端 —— 它拿到
# 本脚本 export 的 NUM_NPUS/NNODES/ROLLOUT_TP/REWARD_POOL_NPUS/fp 后，追加
# _OMNI_MIX_ARGS + _REWARD_ARGS 再 exec 后端。日志由 run_output.sh 保存到
# ${PROJECT_ROOT}/runs/ 下独立运行目录中的 console.log，启动时打印完整路径。
# 最后覆盖，确保下游默认值或传入参数不会重新打开长度预筛。
exec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh" \
    "$@" data.filter_overlong_prompts=false
