#!/usr/bin/env bash
# 单节点 16 NPU AVQA-R1-6K 全量训练。run.sh 负责训练配方和 Ray worker 的环境。
set -euo pipefail

die() { echo "[aqav] $*" >&2; exit 1; }

# 挂载目录、编译与日志设置在环境准备和 Ray 启动时就需要生效。
for arg in "$@"; do
    if [[ "$arg" =~ ^--[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
        export "${arg#--}"
    fi
done

aqav_rel=verl_omni_ext/models/qwen3_6_huanyu/aqav_open
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    launch_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    if [[ "$launch_dir" == */"$aqav_rel" ]]; then
        PROJECT_ROOT="${launch_dir%/"$aqav_rel"}"
    else
        PROJECT_ROOT=/opt/huawei/dataset/ccb_data/code/xy_rl_omni-develop_qwen36_whisper
    fi
fi
[[ -d "$PROJECT_ROOT" ]] || die "找不到项目目录：$PROJECT_ROOT，请设置 PROJECT_ROOT"
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
export PROJECT_ROOT
aqav_dir="$PROJECT_ROOT/$aqav_rel"
[[ -s "$aqav_dir/run.sh" ]] || die "找不到 AQAV 训练脚本或文件为空：$aqav_dir/run.sh"
export SHELL_ROOT="${SHELL_ROOT:-$PROJECT_ROOT/examples/qwen3_6_huanyu}"

model_root="${model_gy:-/opt/huawei/dataset/model_gy}"
data_root="${infra_data:-/opt/huawei/dataset/infra_data}"
export AVQA_MODEL_PATH="${AVQA_MODEL_PATH:-${model_root%/}/downloaded_models/Qwen3-Omni-30B-A3B-Instruct}"
filtered="${data_root%/}/data/avqa_r1_6k_2748_filtered"
export AVQA_TRAIN_FILE="${AVQA_TRAIN_FILE:-$filtered/train.parquet}"
export AVQA_VAL_FILE="${AVQA_VAL_FILE:-$filtered/validation.parquet}"
[[ -d "$AVQA_MODEL_PATH" ]] || die "找不到模型目录：$AVQA_MODEL_PATH"
for file in "$AVQA_TRAIN_FILE" "$AVQA_VAL_FILE" "$SHELL_ROOT/prepare_huanyu_env.sh" \
            "$SHELL_ROOT/start_huanyu_ray.sh"; do
    [[ -s "$file" ]] || die "文件缺失或为空：$file"
done

export MODEL_PATH="$AVQA_MODEL_PATH"
export HUANYU_TRAIN_HANG_DIR="" HUANYU_FINITE_PROBE=0 MINICPMO_HANG_PROBE=0 HUANYU_REPLAY_DEBUG=0
unset HUANYU_TRAIN_HANG_SECONDS HUANYU_TRAIN_HANG_CALLS
export AVQA_MODEL_FAMILY=qwen3_omni AVQA_REPLAY=0 HUANYU_EXPERT_REPLAY=0
export VERL_USE_EXTERNAL_MODULES=verl_omni VLLM_OMNI_EXTERNAL_MODULES=""
export HUANYU_GRAPH_MODE=1 VLLM_TORCH_COMPILE=1 NPU_GRAPH_CAPTURE_DISABLE=0 TORCH_COMPILE_DISABLE=0
export VLLM_ASCEND_ENABLE_NZ=0 VLLM_ASCEND_WEIGHT_NZ_MODE=0 ASCEND_WEIGHT_NZ_MODE=0
export HUANYU_HCCL_PROFILE=aiv HCCL_OP_EXPANSION_MODE=AIV
# 复用节点的 jemalloc 和 CANN 设置。Huanyu 的模型检查会要求 2748 的音频缓存。
source "$SHELL_ROOT/prepare_huanyu_env.sh" --functions-only "$@"
huanyu_prepare_allocator
set +u
huanyu_prepare_cann_environment
set -euo pipefail
export CPATH="/usr/include${CPATH:+:$CPATH}" VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_EXTENSIONS_DIR=/cache/.torch_extension NPUS_PER_NODE=16 NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
export VLLM_USE_V1=1 VLLM_NO_SUBCLASS_INSPECT=1
export RAY_health_check_failure_threshold=60 RAY_health_check_period_ms=30000
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-900} TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-600}
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0} TP_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2} TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000} CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
unset PYTORCH_NPU_ALLOC_CONF

export PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
"$PYTHON_BIN" -c 'import json, sys
from pathlib import Path
from qwen_omni_utils import process_mm_info
config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
if config.get("model_type") != "qwen3_omni_moe":
    raise SystemExit(f"AVQA_MODEL_PATH 需要 Qwen3-Omni checkpoint: {sys.argv[1]}")' "$AVQA_MODEL_PATH"

export VC_TASK_INDEX=0 VC_WORKER_NUM=1
node_ip=$(hostname -I | awk '{print $1}')
[[ -n "$node_ip" ]] || die 'hostname -I 未返回本节点 IP'
export VC_WORKER_HOSTS="$node_ip" RAY_ADDRESS="$node_ip:${RAY_PORT:-6925}"
bash "$SHELL_ROOT/start_huanyu_ray.sh" --check

export WANDB_MODE=offline TRAINER_LOGGER='[console,wandb]'
mkdir -p -- "$PROJECT_ROOT/runs"
RUN_DIR=$(mktemp -d -- "$PROJECT_ROOT/runs/avqa_qwen3_omni_replay0_full_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export RUN_DIR LOG="$RUN_DIR/console.log"
printf 'RUN_DIR=%s\nLOG=%s\n' "$RUN_DIR" "$LOG"

exec bash "$aqav_dir/run.sh" --platform-launch \
    data.filter_overlong_prompts=false data.train_max_samples=-1 data.val_max_samples=-1 \
    trainer.total_training_steps=null trainer.total_epochs=10 trainer.val_before_train=false \
    trainer.test_freq=10 trainer.save_freq=50 trainer.resume_mode=disable trainer.resume_from_path=null \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=true "$@"
