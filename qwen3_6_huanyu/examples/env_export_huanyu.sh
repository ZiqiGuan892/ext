#!/bin/bash
set -ex
_huanyu_env_example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_huanyu_env_example_dir}/precision_mode.sh" || {
    return 2 2>/dev/null || exit 2
}

# Huanyu 环境；在每个节点 ray start 前 source 本文件。
# source 前 export PROJECT_ROOT 可覆盖部署目录；默认使用 hfy-model-gy 下的新项目。
# ext 从 PROJECT_ROOT 加载，verl / verl-omni 继续使用公共安装。
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/hfy-model-gy/verl_omni_qwen36_gzq_new}"
# 兼容旧 Huanyu 入口；唯一需要配置的代码根目录是 PROJECT_ROOT。
export QWEN36_HUANYU_ROOT="${PROJECT_ROOT}"
export HUANYU_VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
unset QWEN36_GZQ_ROOT

# PYTHONPATH 加入 Python 包的父目录，兼容直接布局及外层插件仓库目录。
_huanyu_ext_python_root=""
for _huanyu_candidate in \
    "${PROJECT_ROOT}" \
    "${PROJECT_ROOT}/verl-omni_ext" \
    "${PROJECT_ROOT}/verl-omni-ext" \
    "${PROJECT_ROOT}/verl_omni_ext"; do
    if [ -f "${_huanyu_candidate}/verl_omni_ext/__init__.py" ]; then
        _huanyu_ext_python_root="${_huanyu_candidate}"
        break
    fi
done
if [ -z "${_huanyu_ext_python_root}" ]; then
    echo "ERROR: ${PROJECT_ROOT} 下未找到可导入的 verl_omni_ext/__init__.py。" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "${HUANYU_VLLM_OMNI_ROOT}/vllm_omni/__init__.py" ]; then
    echo "ERROR: ${HUANYU_VLLM_OMNI_ROOT}/vllm_omni/__init__.py 不存在；不能回退到其他 vLLM-Omni 版本。" >&2
    return 1 2>/dev/null || exit 1
fi

# source 同时切换工作目录：Python 会优先搜索 cwd，不能停留在另一个副本。
# 之后可直接使用相同的相对路径训练命令，无需额外手动 cd。
if ! cd -- "${PROJECT_ROOT}"; then
    echo "ERROR: 无法进入 PROJECT_ROOT=${PROJECT_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

# 清除旧 ext 根、嵌套目录、路径别名，以及旧 vLLM-Omni 路径。
# 随后仅把本次选定的包父目录放回首部；保留 CANN 和公共依赖路径。
_huanyu_retained_pythonpath=""
IFS=: read -r -a _huanyu_python_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_python_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|"${PROJECT_ROOT%/}"|"${PROJECT_ROOT%/}"/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy/*|\
        /vllm-workspace/vllm-omni|\
        /vllm-workspace/vllm-omni/*|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit/*)
            continue ;;
    esac
    _huanyu_retained_pythonpath="${_huanyu_retained_pythonpath:+${_huanyu_retained_pythonpath}:}${_huanyu_path}"
done
export PYTHONPATH="${HUANYU_VLLM_OMNI_ROOT}:${_huanyu_ext_python_root}:/opt/huawei/algorithm/:/opt/huawei/dataset/ccb_data/code/package_install/verl${_huanyu_retained_pythonpath:+:${_huanyu_retained_pythonpath}}"
echo "[huanyu env] ext Python root: ${_huanyu_ext_python_root}"
echo "[huanyu env] vLLM-Omni Python root: ${HUANYU_VLLM_OMNI_ROOT}"
_huanyu_graph_mode_script="${_huanyu_env_example_dir}/graph_mode.sh"
unset _huanyu_env_example_dir
unset _huanyu_candidate _huanyu_ext_python_root
unset _huanyu_retained_pythonpath _huanyu_python_entries _huanyu_path

# 覆盖旧环境残留的输出路径；如需自定义，在 source 本文件之后再覆盖。
# 基础启动脚本会读取这两个变量，确保输出写入本次主目录。
export DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints"
export TENSORBOARD_DIR="${PROJECT_ROOT}/tensorboard_log"
# 两个代码目录之间切换时，迁移继承的诊断目录并保留其相对路径。
# 其他用户指定的独立诊断目录保持不变。
case "${MINICPMO_ROLLOUT_DIAG_DIR:-}" in
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36_gzq_new/}" ;;
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/*|\
    /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36_gzq/}" ;;
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36/*|\
    /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36/}" ;;
esac

pip install mathruler
pip install zarr

export TORCH_EXTENSIONS_DIR=/cache/.torch_extension
export NPUS_PER_NODE=16
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.5-35B-A3B}"
export TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}"
export VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}"
export Q36_CKPT_DIR="${MODEL_PATH}"
echo "[huanyu-paths] project=$PROJECT_ROOT actor=$MODEL_PATH reward=$RM_MODEL_PATH"
echo "[huanyu-paths] train=$TRAIN_FILE validation=$VAL_FILE"

# 这里**曾经**有一行 `export MINICPMO_DISABLE_IMAGE_EXPAND=1`（2026-10-10 调试残留）。
# 它的位置决定了它的危险：本文件在 `ray start` **之前**被 source，于是每个 Ray worker
# 都把它当默认值，而 driver / run 脚本里看到的是 `<unset>` —— 谁跑训练都看不见它，
# 跑出来的却全是「修复前形态」。那一轮（10-10 21:19）的代价是 12/12 条图像 rollout
# 全在 _assert_bounds_agree 抛 bound mismatch，图像样本直接对不上位。
#
# 现在这层保护在包内：worker_env.reconcile_worker_env() 在插件 import 时把 worker 侧的
# `=1` 撤销（并打印一行说明）。真要切回旧形态，必须再显式加 ACK，两个开关同时存在：
#
# export MINICPMO_LEGACY_IMAGE_SPAN_ACK=1     # 只有它也在 worker env 里，上面那个才被尊重
export MINICPMO_DISABLE_IMAGE_EXPAND=0

MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_ADDR=$(getent hosts "$MASTER_ADDR" | awk '{print $1; exit}')
RAY_PORT=6925
RAY_DASHBOARD_PORT=8960

############################## COMMON FUN ##########################

rm_exist_dir(){
    if [ -d "$1" ] ; then
      rm -rf "$1"
    fi
}

print_stage(){
  echo "###################  $1  #############################"
}

print_stage "ENV PRINT"
echo "$@"

#############################  PRE CHECK ##########################
print_stage "PRE CHECK"
WORKER_RANK=${VC_TASK_INDEX:-0}
DEPLOY_LOG_PATH="${PROJECT_ROOT}/logs/$WORKER_RANK"
MASTER_INFO_FILE="${PROJECT_ROOT}/logs/0/master_info.txt"
WORKER_LOG_PATH="$DEPLOY_LOG_PATH/worker_log_$WORKER_RANK"

mkdir -p "$DEPLOY_LOG_PATH"

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export USE_OPTIMIZED_MODEL=0
# One switch for eager/graph inference; the training backend reapplies it and
# explicitly forwards it to Ray, so later per-run changes also reach workers.
source "${_huanyu_graph_mode_script}" || {
    return 2 2>/dev/null || exit 2
}
unset _huanyu_graph_mode_script
export WANDB_MODE=${WANDB_MODE:-offline}
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NPUS_PER_NODE}
export VLLM_NO_SUBCLASS_INSPECT=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [ -f "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
fi

export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

# =============================================================================
# 必须在 ray start 之前设的 env（见下面为什么）
# =============================================================================
# WorkerDict / vLLMOmniHttpServer 都是 Ray actor，env 继承自 raylet，而 raylet
# 是下面那两条 `ray start` fork 出来的。所以 `ray start` 之后再 export 的东西
# 只活在 driver 进程里，训练 rank 上根本看不到。
#
# verl 确实有 runtime_env 透传，但那是 verl/trainer/constants_ppo.py 里一份
# 写死的白名单（TOKENIZERS_PARALLELISM / NCCL_DEBUG / VLLM_* / VERL_* ...），
# 一个 HCCL_* 都不在里面。log 25 打印的 runtime_env 可以逐字核对。
#
# 这直接坑掉了基础脚本 §2 的整段 HCCL 配置 —— 它在 line 105-116 export，
# 而基础脚本是 line 101 之后才被调用的。后果（log 25 实测）：
#   * HCCL_CONNECT_TIMEOUT 一直是默认 120s，不是 7200。log 20 那个"静默 174 秒
#     后 EI0006"就是 120s 默认超时，不是什么 adapter_hccp.cc 硬编码。
#   * HCCL_SOCKET_IFNAME 从未生效，HCCL 一直在自己猜网卡。
#   * MINICPMO_MM_TRACE / TORCH_DISTRIBUTED_DEBUG 同理，两轮诊断的观测都是瞎的。
#
# 反例：MINICPMO_ENABLE_MM_FORWARD=1 一直好使，因为它本来就在 ray start 之前。
#
# 新增变量往这里加，不要往下面的实验脚本里加。
# -----------------------------------------------------------------------------

# 通信超时。EI0006 的直接原因：多模态视觉负载跨卡差 7 倍（slices=7 vs 1），
# 快的卡先到 all_reduce，在 120s 内等不到慢的卡就判定建链失败。
#
# CONNECT 保持 7200：上面那条 EI0006 就是它太小造成的，别动。
# 其余三个从 7200/3600 降到分钟级 —— 它们不是"容错余量"，是"挂了多久才报错"。
# log 29 在 14:28 卡死、16:04 被人工 kill，95 分钟里没有任何一层抛异常，就是因为
# 7200s 一直没到点；「没报错」当时被我们误读成「没有未完成的通信」。
# 取值依据：plog 实测算子之间最长的合法间隔是 10.1s（反传计算），300s 有 30 倍余量。
#
# EXEC 从 600 降到 300 是 log 31 的教训：那一轮 18:16:53 十五张卡进了 AICPU 的
# AllGather kernel，EXEC_TIMEOUT 本该在 18:26:53 点名是哪次集合通信卡住，但 rank 0
# 的 hang probe 在 18:19:05 就把进程杀了，Ascend 侧那份报告从来没机会打出来。
# 现在的排期是：120s/240s 出 Python 栈（全卡，见下面的 cross-rank trigger），
# 300s 出 HCCL 的算子级报错，480+45s 才自杀 —— 三份证据都拿到再退出。
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-900}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-600}

# 网卡名。不设的话 HCCL/GLOO 自己挑，多网卡机器上可能挑错。
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

# 注意：**故意不设** HCCL_IF_BASE_PORT=64000（基础脚本 line 107 有）。
# 它和上面 HCCL_*_SOCKET_PORT_RANGE="auto" 是两套互斥的端口分配机制，而 auto
# 这条路是目前唯一在 32 卡上真正建起链的（同节点 46/46 成功）。要换先单独验。

# NPU 运行时调优，取自基础脚本 §1，同样因为时序问题从未到达训练 rank
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export ASCEND_SECURITY_CHECK_DISABLED=${ASCEND_SECURITY_CHECK_DISABLED:-1}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

# 基础脚本 line 101 的 `unset` 同样只作用于 driver，这里补一次。
# max_split_size_mb:128 会把分配器切碎，而 MoE 专家权重是大块张量。
unset PYTORCH_NPU_ALLOC_CONF

# 诊断开关。排障期间默认打开，跑通之后再关回去：
#   MINICPMO_MM_TRACE=1        每次多模态前向 / 每次 apm unshard 都打一行
#   MINICPMO_MM_TRACE_FIRST_N  即使 MM_TRACE=0 也无条件打头 N 次（默认 8）。
#                              旧代码是 `call_no == 1`，而 compute_log_prob 每卡
#                              恰好 2 个 micro-batch —— 每次都挂在第 2 轮，而第 2
#                              轮从来没有埋点。log 24 那个"32 卡全是 n=1，所以
#                              通信次数对齐"的结论就是这么来的，是门控的假象。
#   TORCH_DISTRIBUTED_DEBUG    DETAIL 用 gloo 校验每次集合通信的跨 rank 一致性。
#                              它本身会拖慢，只在怀疑通信次数不一致时开。
export MINICPMO_MM_TRACE=${MINICPMO_MM_TRACE:-1}
export MINICPMO_MM_TRACE_FIRST_N=${MINICPMO_MM_TRACE_FIRST_N:-8}
[ -n "${TORCH_DISTRIBUTED_DEBUG}" ] && export TORCH_DISTRIBUTED_DEBUG

# =============================================================================
# 挂起取证（verl_omni/utils/hang_probe.py）
# =============================================================================
# py-spy 在这个容器里是 EPERM（没有 CAP_SYS_PTRACE），gdb / ray stack 同理，
# 所以只能让进程自己 dump 自己的栈。hang_probe 在 verl_omni/__init__.py 里安装，
# 每个 Ray worker 都会经过那里。两种取栈方式：
#
#   1) 手动：在任意节点上 `kill -USR1 <pid>`，立刻输出全部线程的 Python 栈。
#      整节点一把梭：
#        for p in $(pgrep -f "ray::WorkerDict"); do kill -USR1 $p; done
#      对 futex 里睡着的进程照样有效（信号会打断等待）。
#
#   2) 自动：MINICPMO_HANG_TIMEOUT_SEC 秒内埋点没有推进就自动 dump。
#      120s：合法的埋点间隔实测最大 ~10s（一个 micro-batch 的前反传），120s 已有
#      10 倍余量，而且远早于上面 HCCL_EXEC_TIMEOUT=300，HCCL 报超时之前 Python
#      侧栈就已经落盘了。原来的 300s 只是为了躲 torch_npu 那个 600s 看门狗。
#
#   3) MINICPMO_HANG_ABORT_AFTER=4：**训练开始之后**连续四次 dump 才自杀（exit 99）。
#      「训练开始之后」是 log 31 补的：原来 dump 计数器是全局的，rank 0 在 18:14:04
#      因为启动宽限期（120×4=480s）dump 过一次，等 18:19:05 真卡住时那已经算第 2 次
#      → 当场自杀。结果是唯一一次跑进 compute_log_prob 的实验只留下一张栈，而且是
#      「在等另外十五张卡」的那张。现在启动期的 dump 不再花训练期的预算。
#      4 × 120s = 480s，越过上面 HCCL_EXEC_TIMEOUT=300 的报错点。设 0 恢复只 dump 不退。
#
#   4) MINICPMO_HANG_ABORT_GRACE_SEC=45：决定自杀之后先等 45 秒再 os._exit。
#      自杀本身就是 Ray 拆任务的扳机，而真正想看的东西（同伴卡的栈、HCCL 正要抛的
#      算子级报错）都排在自己那份 dump 后面。log 31 就是没等这 45 秒。
#
#   5) 跨卡联动 dump：谁先到阈值，谁就写一个 $MINICPMO_HANG_DUMP_DIR/HANG_TRIGGER，
#      同节点其余各卡在一个轮询周期内一起出栈（没有开关，watchdog 开着就有）。
#      集合通信挂起只看一张卡的栈是没用的 —— log 31 死在「15 张卡在 AICPU kernel 里
#      等，rank 0 进不去」，而我们手里只有 rank 0 那一张。
#      => 最坏情况 480 + 45 ≈ 8.7 分钟，带 16 张栈 + HCCL 算子级报错退出，不用盯屏。
#
# stderr 只打主线程（538 个线程 × 32 卡的全量会淹掉 driver 日志），
# 全量写到 MINICPMO_HANG_DUMP_DIR 下的每卡一个文件。
export MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-1}
export MINICPMO_HANG_TIMEOUT_SEC=${MINICPMO_HANG_TIMEOUT_SEC:-120}
export MINICPMO_HANG_ABORT_AFTER=${MINICPMO_HANG_ABORT_AFTER:-4}
export MINICPMO_HANG_ABORT_GRACE_SEC=${MINICPMO_HANG_ABORT_GRACE_SEC:-45}
export MINICPMO_HANG_DUMP_DIR="${DEPLOY_LOG_PATH}/hang_dump"
mkdir -p "${MINICPMO_HANG_DUMP_DIR}"
# 上一轮留下的 trigger 会让本轮每张卡一启动就白 dump 一次。hang_probe 内部已用进程
# 启动时间挡了一道，这里再清一次，重跑同一个目录时更稳。
rm -f "${MINICPMO_HANG_DUMP_DIR}/HANG_TRIGGER"

# === 多模态对齐兜底的响亮化（verl_omni_ext/models/qwen3_6_whisper/align_guard.py）===
# 两条兜底会在不抛异常的前提下改变模型看到的序列，是「复读退化」的头号嫌疑：
#   1) mrope 位置编码：text_len 为负时把整段序列的 (t,h,w) 换成 flat 1D；
#   2) merge：视觉/音频特征数与占位符数不等时，取最后 N 个占位符位置塞特征。
# 两者的失效形态都不是崩，而是「数字慢慢不对」，所以默认 1：一经触发立刻 raise，
# 把问题炸在启动后几分钟内、还带着 tag/pid/长度信息。
# 置 0 = 只响亮打印 + 计数（[verl-omni] ALIGN-FALLBACK <tag> #N pid=...），
# 并沿用旧的兜底行为，用来跑完一整轮取证。
export MINICPMO_MM_ALIGN_STRICT=${MINICPMO_MM_ALIGN_STRICT:-1}

# PYTHONPATH 已在文件开头设置，必须在各节点 ray start 之前 source。
ulimit -s unlimited

# === Ray 心跳超时（避免 update_actor 长时间运行导致节点被标记 dead）===
export RAY_health_check_failure_threshold=60
export RAY_health_check_period_ms=30000

# === GenRM dump（Judge 判分结果离线检查）===
export GENRM_DUMP_DIR="${PROJECT_ROOT}/genrm_dump"
export GENRM_DUMP_MAX=50
