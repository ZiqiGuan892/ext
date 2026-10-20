#!/usr/bin/env bash
# MiniCPM-o 5.0 Thinker-only GSPO full-parameter training, omni V1 trainer.
# Modalities: text (-> text). Image/video/audio input is supported by the model
# but is NOT enabled here — see §4.6 below and the plan's §4.6 / V5.
# Hardware: Ascend NPU (910B/910C, 64GB)。卡数由 NUM_NPUS 决定，默认自动探测。
#
# 本脚本是 verl-omni 一度放在源码树内（examples/gspo_trainer/minicpmo_5_0/，
# 已清出）的 run_minicpmo_5_0_thinker_gspo_npu.sh 的 out-of-tree 移植版
# （原脚本本身又是 run_qwen35_moe_thinker_gspo_npu.sh 的移植版，
# **只改了 6 处**，每处标了 [MiniCPM]）。
#
# out-of-tree 改动（相对 verl-omni 上游脚本）：
#   A) 槽位③: VERL_USE_EXTERNAL_MODULES 追加 verl_omni_ext + 训练命令里补
#      actor_rollout_ref.model.external_lib=verl_omni_ext
#   B) GP-004: VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.minicpmo_5_0.vllm_omni
#   C) VERL_OMNI_EXT_PATCH_STRICT=1: L2 补丁未打上直接 raise
#   D) 前置自检（§5 前一段）：MiniCPMO 适配器/rollout 注册 + L2 补丁 applied+alive
#   E) 两个 transformers 补丁的模块路径挪到 verl_omni_ext/models/minicpmo_5_0/
#   F) 自定义 Dataset 的 custom_cls.path 改为 pkg://verl_omni_ext.models.minicpmo_5_0.dataset
#   G) VERL_OMNI_ROOT → EXT_ROOT：checkpoint / tensorboard 默认落到本包根目录
# -----------------------------------------------------------------------------
# Prerequisite 1（推理侧）：vllm-omni 的 pipeline 定义来自 ext 包（GP-004），
#   安装态 vLLM-Omni 需先打过 gate patch：
#     bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni
#   "minicpmo_5_0_thinker_only" 由 verl_omni_ext.models.minicpmo_5_0.vllm_omni 注册。
#
# Prerequisite 2（训练侧）：checkpoint 走 transformers remote code，与新版
# transformers 有两处不兼容，两个补丁（auto_register_guard / automodel_fallback）
# 都在 verl_omni_ext/models/minicpmo_5_0/patches.py 的模块级自动生效
# （本脚本 set 了 VERL_USE_EXTERNAL_MODULES，每个 worker 都会 import 到）。
# **不需要改 checkpoint。**
# 自检：
#   python3 -c "import verl_omni_ext; \
#     from verl_omni.pipelines.model_base import OmniModelBase; \
#     print(OmniModelBase.get_class_by_name('MiniCPMO', 'thinker'))"
# -----------------------------------------------------------------------------
set -x

# Direct launches bypass run_q36_job.sh, which installs the local entry points.
# A source checkout on sys.path is insufficient: plugin discovery uses installed
# distribution metadata. Install before the first Python/plugin import.
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXT_ROOT="${PROJECT_ROOT}"
# All Huanyu entry points converge here. Apply before Python/plugin imports,
# and append the same settings last in the training argv below.
source "${PROJECT_ROOT}/verl_omni_ext/models/qwen3_6_huanyu/graph_mode.sh" || exit $?
echo "[huanyu-graph] HUANYU_GRAPH_MODE=${HUANYU_GRAPH_MODE} actor/RM.enforce_eager=${HUANYU_ENFORCE_EAGER} VLLM_TORCH_COMPILE=${VLLM_TORCH_COMPILE} TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE} NPU_GRAPH_CAPTURE_DISABLE=${NPU_GRAPH_CAPTURE_DISABLE}"
export PYTHONPATH="${EXT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Match the job launcher for shared mounts. setuptools-scm strips Git environment
# overrides, so the trust entry must be in Git's configuration before installation.
git config --global --add safe.directory "${EXT_ROOT}" || exit $?
python3 -m pip install -e "${EXT_ROOT}" --no-deps --no-build-isolation || exit $?

# =============================================================================
# §1. Ascend / vLLM 运行时
# =============================================================================
export CPATH=/usr/include${CPATH:+:$CPATH}
# 注意：VLLM_ASCEND_ENABLE_NZ **没有任何代码读它**（vllm_ascend 全树 grep 无命中），
# 留着只为不惊动历史，别指望它生效。真正关 NZ 的是下面命令里的
# actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
# —— wake_up() 读的是各引擎自己的 additional_config（vllm_ascend/utils.py:281
# 明确 "Get config value instead of env"），free_cache_engine=True 时必过这一关。
export VLLM_ASCEND_ENABLE_NZ=0
# 改动 A/B/C：加载链三件套（verl_omni 上游注册 + out-of-tree 包 + GP-004 vllm-omni 侧）
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_whisper.vllm_omni
export VERL_OMNI_EXT_PATCH_STRICT=1       # L2 补丁未打上直接 raise（改动 C）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export USE_OPTIMIZED_MODEL=0
# Compile/capture flags are selected by graph_mode.sh, not hard-coded here.
export ASCEND_SECURITY_CHECK_DISABLED=1
export WANDB_MODE=${WANDB_MODE:-offline}

# verl DataProto 序列化：参考脚本已验证值，numpy 路径在 NPU 上更稳
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export VERL_USE_EXTERNAL_MODULES="verl_omni,verl_omni_ext"

# ---------------------------------------------------------------------------
# MINICPMO_FSDP2_PIN_MEMORY —— 默认 0 = 补丁生效（已移植到 out-of-tree）
# ---------------------------------------------------------------------------
# 默认 0 = verl_omni_ext/models/minicpmo_5_0/fsdp_cpu_offload_pin.py 在包
# import 期由 __init__.py 调用 apply_fsdp2_cpu_offload_pin_memory_patch()，
# 把 verl/workers/engine/fsdp/transformer_impl.py:443 硬编码的
# CPUOffloadPolicy(pin_memory=True) 打成 pin_memory=False（actor/ref 同支）。
# 设 1 = 恢复 verl 原样（即恢复崩溃）。
# 补丁 scoped 到 MiniCPM-o：同一 os 布局里 qwen3_5_moe/qwen3_omni 走别的
# pipeline 包，不受影响。真机上补丁是否打上可由每个 Ray worker 的
# "[verl-omni] Patched FSDP2 CPUOffloadPolicy" 行确认。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 锁页（pinned）宿主内存预算 —— logs/13.txt 的死因（以下为上游注释原文）
# ---------------------------------------------------------------------------
# 13.txt 里 16 张卡中 13 张在**第一次 loss.backward()** 同时死掉：
#   _fsdp_collectives.py:625  new_sharded_grad.to(cpu, non_blocking=True)
#   RuntimeError: ... srclen=134217728, kind=2
#   Invalid_Argument_Null_Pointer(EH0002): Argument [dst] must not be null
# kind=2 是 Device→Host，dst 是**宿主端**目标指针，null 意味着宿主分配失败。
#
# 判定它是锁页池而不是普通内存，靠的是**这个失败挑尺寸**：13 条栈里 srclen 只有
# 134217728 和 67108864 两个值（2^29 / 2^28 个 fp32 元素，正好是 MoE 融合专家的
# [256,2048,1024] 和 [256,2048,512]）。同一个 foreach_reduce 循环里更小的 norm、
# bias、注意力投影全都拷成功了。如果是异步 D2H 这条路本身不通，第一个最小的张量
# 就该失败；只有最大的两种 buffer 失败，说明是分配器要不到连续区域。
# 而全日志没有任何 OOM killer、没有 DefaultCPUAllocator 报错、没有 MemoryError ——
# 这正是锁页内存耗尽的表现：它是一块独立的、不可换出的、小得多的预算，
# 耗尽时分配器就是返回空指针。
#
# 两个锁页消耗方，分别由下面两个开关控制：
#
# 1) MINICPMO_FSDP2_PIN_MEMORY —— 主因。
#    verl 在 transformer_impl.py:443 写死 CPUOffloadPolicy(pin_memory=True)，
#    没有任何 config 开关，且 actor 和 ref 都走这一支（ref 靠 forward_only 命中）。
#    它做两件事：init 时把每个参数分片 pin 住（36B 模型单卡 ~9 GiB，
#    整节点 16 卡 fp32 actor + bf16 ref 约 200 GiB 锁页），以及让上面那次 D2H
#    变成 non_blocking。默认 0 = 由 verl_omni_ext/models/minicpmo_5_0/
#    fsdp_cpu_offload_pin.py 打补丁（包 import 期执行）改成 pin_memory=False，
#    两个效果一起消失；设 1 恢复 verl 原样（即恢复崩溃）。
#    代价：梯度 D2H 不再与 reduce-scatter 重叠，反传变慢，数值不变。
#
# 2) ACTIVATION_OFFLOAD —— 次要，但在同一个反传窗口里争同一块锁页内存。
#    verl/utils/activation_offload.py:173-182 的 offload() 对每个存下来的激活
#    做 torch.empty(..., pin_memory=True) + copy_(non_blocking=True)。
#    梯度检查点已经开着（enable_gradient_checkpointing=true），
#    而 13.txt 里实际序列只有 180~270 token，激活本来就不大，
#    这个 offload 买到的显存很有限，却在最要命的时刻占着锁页。默认关。
#    ⚠️ 2026-10-14 实测：这里原本写成 `:-true`，与本节注释自相矛盾，且**必崩**。
#    开着 activation offload + enable_gradient_checkpointing=true 时，反传里
#    tensor_pop 拿到的不是张量而是占位元组 (key, shape)（activation_offload.py:311
#    bulk_offload_group 写回），说明该 group 还没被 bulk_reload_group 换回真张量
#    —— 前/反向的 group 计数错位了。该方案假定前后向按同一串 FSDP layer 镜像执行
#    （num_offload_group = len(fsdp layers)-1，逐层 commit），本模型多了
#    visual/audio 塔、forward 又被 verl_omni_ext 打过补丁，这个假定不成立。
#    表现：16 卡全中，`assert not isinstance(tensor, tuple)`（:295）—— 不是 OOM，
#    显存指标全干净。默认关；genrm 入口脚本第 169 行也显式设了 false。
export MINICPMO_FSDP2_PIN_MEMORY=${MINICPMO_FSDP2_PIN_MEMORY:-0}  # 0=fsdp pin 补丁生效，1=恢复 verl 原样
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-false}

# 3) 反向预取：FSDP2 的**默认**反向预取按「静态前向顺序」猜下一个要 unshard 的单元
#    （torch `_fsdp_param_group._backward_prefetch` 自己的注释：may have mistargeted
#    prefetches if not all modules used in forward are used in this backward）。
#    本模型图像/音频 rollout 落在不同 DP rank（实测 12 图像 / 4 音频），两组实际执行的
#    子图不同 ⇒ 猜偏 ⇒ 其中一组多发一条 all-gather ⇒ 集合通信序列错位 ⇒ HCCL 按算子
#    序号配对 ⇒ 第 1 步 backward 死锁（run#78 实测：逐条对齐到 i=1643，图像组在 1644
#    多发一块视觉块的 all_gather）。off = 补丁把预取目标指成自身（等价于不预取，
#    `unshard()` 已 unsharded 时早退、不发集合通信），代价是失去反向的重量叠；
#    on 恢复 torch 默认（会死锁）。补丁在 fsdp_backward_prefetch.py，见其 docstring。
export Q36_BACKWARD_PREFETCH=${Q36_BACKWARD_PREFETCH:-off}  # off=不预取（默认，防死锁），on=torch 默认

# 4) GDN 循环状态精度。当前默认走本仓库 FP32 recurrence，保留已验证的 prefill。
#    以下是 legacy BF16 路径的历史诊断；FP32_REFERENCE=1 时不执行降精度补丁。
#    Qwen3.6-whisper 的 LM 40 层里 30 层是 linear_attention（GDN），ckpt config.json 的
#    text_config.mamba_ssm_dtype='float32' ⇒ vLLM 把 GDN 的 ssm_state 推成 fp32，而
#    vllm-ascend 的自定义算子 npu_recurrent_gated_delta_rule 原生支持 DT_FLOAT
#    （vllm_ascend/ops/gdn.py:469/494/578 就是不加转换直接传，csrc 的 op_host 声明
#    Input("state").DataType({DT_BF16, DT_FLOAT})）。
#    但本包 ensure_ascend_gdn_bf16_patched()（models/qwen3_6_whisper/__init__.py:67）
#    却把它拦下来：每步 decode 前 .to(bfloat16)、调用后 copy_ 回 fp32 buffer ⇒
#    整条前缀的循环状态每步被舍入一次（bf16 ≈ 3 位十进制），30 层 × 上百步 decode 累积。
#    形态吻合：先几百字正常 → 忽然跑偏 → 字级复读 → 自己停在换行；越长越容易崩，
#    且与模态无关（10-13 纯文本 GSM8K 37/44 复读、10-14 图像 17/21，短音频 0 复读），
#    采样惩罚不能修复循环状态的数值失配；过强的累计惩罚还会损伤长回答。
#    0 = 原 dtype 直传；1 = 保留历史行为（每步降 bf16）。
#    对照基线（genrm_dump_qwen36 里 repeat_hint 非空比例）：10-13 文本 37/44=84%，
#    10-14 16 时那轮 18/26=69%。设 0 后这个比例应显著下降。
#
#    ★ 2026-10-14 19:38 实测：设 0 **直接崩**，默认只能留 1。原因不在本包：
#      CANN 9.1.0 自己也装了同名算子（built-in），而 vllm-ascend 的扩展
#        vllm_ascend_C.cpython-312-aarch64-linux-gnu.so 的 NEEDED 里是 **CANN 的
#        /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libopapi.so**
#        （它定义了 aclnnRecurrentGatedDeltaRule），
#      厂商那份支持 fp32 的 vendors/custom_transformer/op_api/lib/libcust_opapi.so
#        **全仓没有任何 .so 把它列进 NEEDED**（只有自引用），所以它根本不会被加载。
#      结果运行时走 CANN 内置实现 → 报
#        RuntimeError: call aclnnRecurrentGatedDeltaRule failed,
#        AclNN_Parameter_Error(EZ1001): Tensor params.state not implemented for
#        DT_FLOAT, should be in dtype support list [DT_BFLOAT16,]
#      （厂商 csrc 源码里 STATE_TYPE_SUPPORT_LIST 是 {DT_BF16, DT_FLOAT}，但那是
#        源码，编进 libopapi.so 的是 CANN 自己那份 bf16-only 的。）
#      所以「GDN 降 bf16」不是随便加的，是这颗 CANN 上的硬约束；要试 fp32 需要
#      让厂商库赢过 CANN 的，唯一不改 /vllm-workspace 的办法是预加载：
#        LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so \
#        MINICPMO_GDN_STATE_BF16=0 bash examples/qwen3_6_huanyu/run_qwen36_huanyu_32card.sh
#      风险：厂商库里同名 l0op::/aclnn* 符号会全局抢占（含其它自定义算子），
#      可能引发「数字慢慢不对」这类静默问题 —— 只作为一次性的对照组实验跑。
# Use the local FP32 decode recurrence instead of rounding the whole recurrent cache
# on every decode step. This avoids the CANN DT_FLOAT limitation without
# modifying vllm-workspace or relying on LD_PRELOAD symbol resolution.
export MINICPMO_GDN_FP32_REFERENCE=${MINICPMO_GDN_FP32_REFERENCE:-1}
export MINICPMO_GDN_STATE_BF16=${MINICPMO_GDN_STATE_BF16:-1}  # legacy A/B path when FP32_REFERENCE=0

# NPU 运行时调优（取自参考脚本 §3，均为已验证值）
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1

# PYTORCH_NPU_ALLOC_CONF：**故意 unset**，与参考脚本一致。
# 原值 max_split_size_mb:128 会把分配器切得很碎，而 MoE 的专家权重是大块张量；
# 在 gpu_memory_utilization=0.85 这种高占用下，小 split 反而更容易触发碎片型 OOM。
# 若确认需要限制 split，再显式 export，不要保留旧的 128。
unset PYTORCH_NPU_ALLOC_CONF

# =============================================================================
# §2. HCCL / 通信超时
# =============================================================================
# 一律写成 ${VAR:-默认}：**已经设过的值优先**。
#
# 2026-10-10 的挂起事故（log 见 qwen36_whisper_full.log）暴露出这六行是一把
# "静默覆盖"的刀：本脚本原来无条件 `export HCCL_EXEC_TIMEOUT=7200`，而
# env_export.sh 同一个变量写的是 `export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}`
# ——后者只在未设置时兜底，于是 env_export.sh 的排障值一律被这里盖回 7200。
# 实测证据（本文件 worker 侧 runtime_env 注入清单，log 755 行）：
#
#   变量                       env_export.sh 想要   实际生效
#   HCCL_EXEC_TIMEOUT          300                  7200
#   ACL_DEVICE_SYNC_TIMEOUT    900                  7200
#   TORCH_DIST_TIMEOUT         600                  3600
#   HCCL_IF_BASE_PORT          故意不设              64000
#
# 后果就是 env_export.sh 自己注释里点名批评过的那件事："它们不是容错余量，
# 是挂了多久才报错"。那一轮 16 卡卡死 18 分钟，HCCL 的算子级报错（本该 300s
# 点名是哪次集合通信、哪个 peer）一次都没打出来，手里只剩设备侧的 kernel 超时。
#
# 改成 ${VAR:-...} 之后：source 过 env_export.sh 的跑法用 env_export 的值，
# 没 source 的（多机参考脚本跑法）仍是原来这组默认值，行为不变。
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
# HCCL_IF_BASE_PORT 与 HCCL_*_SOCKET_PORT_RANGE="auto" 是两套互斥的端口分配
# 机制。env_export.sh 里"故意不设"这条，指的是它自己的文件不提供默认值；
# 本文件的 64000 只在外部完全没设时才生效（要彻底走 auto，用 `unset
# HCCL_IF_BASE_PORT` 或下面那条 echo 确认清楚再决定，别两边都开着）。
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-3600}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

# 生效值回声。上面那类"两个文件都 export、谁覆盖谁"的 bug 每次都是靠
# 事后翻 runtime_env 才发现的；这四行让它变成启动 3 秒就能看到的东西。
echo "[env] HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT} ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT} TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT} HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT}"
echo "[env] HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE} HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-<unset>} MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-0} MINICPMO_HANG_DUMP_DIR=${MINICPMO_HANG_DUMP_DIR:-<unset>}"
# 对照实验的开关回声。2026-10-10 那轮挂起，事后是靠翻 runtime_env 注入清单才确认
# 「MINICPMO_DISABLE_IMAGE_EXPAND 当时根本没进到 worker」——注入清单是 1000 行
# 之后的数字，而这一行在启动 3 秒处。想让这轮日志有意义，先看这一行。
echo "[env] MINICPMO_DISABLE_IMAGE_EXPAND=${MINICPMO_DISABLE_IMAGE_EXPAND:-<unset>} MINICPMO_DISABLE_TRAIN_PROMPT_EXPAND=${MINICPMO_DISABLE_TRAIN_PROMPT_EXPAND:-<unset>} MINICPMO_IMAGE_SPAN_DROP_SENTINELS=${MINICPMO_IMAGE_SPAN_DROP_SENTINELS:-<unset>} MINICPMO_MM_ALIGN_STRICT=${MINICPMO_MM_ALIGN_STRICT:-<unset>} MINICPMO_ROLLOUT_DIAG=${MINICPMO_ROLLOUT_DIAG:-1} MINICPMO_GDN_STATE_BF16=${MINICPMO_GDN_STATE_BF16:-<unset>}"

# [2026-10-13] 判官降级分：omni_mix **没有规则评分器**，判官一旦塌缩/请求失败，
# 旧的 rule 兜底只会落到 ``rule_failed:NotImplementedError`` ⇒ 该样本 0 分
# （而且和「判官判它错」长得一模一样）。``similarity`` 改成量参考答案与学生答案的
# 字符 2-gram Dice 相似度，给一个粗粒度但**非零**的止损分；``genrm_fallback=1.0``
# 依旧标记着「这一分不是判出来的」，所以在 dump/指标里仍然分得开。
# 取值：rule（旧默认）/ zero / similarity（omni_mix 推荐）。
export GENRM_FALLBACK_ON_ERROR=${GENRM_FALLBACK_ON_ERROR:-similarity}
echo "[env] GENRM_FALLBACK_ON_ERROR=${GENRM_FALLBACK_ON_ERROR}"

# 网卡名：多机 HCCL/GLOO 通信用，不可为空否则可能选错网卡
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

# =============================================================================
# §3. Ray 运行时（取自参考脚本 §7）
# =============================================================================
# RAY_DEDUP_LOGS=0：Ray 默认会把多 rank 的相同日志折叠成
# "[repeated 15x across cluster]"，导致无法区分是哪几个 rank 出错。排障期间必须关掉。
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# 70+ GB 权重加载慢，默认超时不够，会表现为 worker 注册失败/GCS 超时
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0

# 让 verl/vLLM 自己管设备可见性，不要由 Ray 改写 ASCEND_RT_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

# Ascend 的 env 脚本不是 -e clean 的，放在它们之后
set -eo pipefail

# 改动 G：out-of-tree 包根（本仓根）= ext 包目录的父两级；改名 EXT_ROOT 避免
# checkpoint / tensorboard 默认写到上游 verl-omni 仓里
# EXT_ROOT was resolved before installing the local plugin above.

# =============================================================================
# §4. 路径与规模参数
# =============================================================================
# [MiniCPM 改动 1/6] 模型路径
MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl"}

# [MiniCPM 改动 2/6] 数据集：**先用纯文本**，不要一上来就用带图的 geo3k。
#
# 原因见方案 §4.6：verl 的多模态数据管线是照 Qwen-VL 的形状写死的
# （qwen_vl_utils.process_vision_info / image_grid_thw / processor(videos=,audio=)），
# 而 MiniCPM-o 是 UHD NaViT + tgt_sizes/image_bound，对不上。带图的 batch 要么
# TypeError，要么**不报错但算错**。先用纯文本把「能训」这件事立住，多模态单独立项。
#
# 纯文本可用 gsm8k / math 这类；换成带图数据集之前，先跑通方案的 V5 前置检查。
TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/gsm8k/test.parquet"}

# ---- 数据集格式前置校验（坑 #32）----
# verl 要的是**预处理过**的 RL parquet：列 prompt / data_source / reward_model。
# Hub 上 openai/gsm8k 原始文件只有 question / answer 两列，长这样：
#     .../gsm8k/main/train-00000-of-00001.parquet
# 拿原始文件直接训**不会报错**：RLHFDataset.maybe_filter_out_long_prompts 里
# doc2len 抛 KeyError('prompt') 被 `except Exception` 吞掉，返回
# max_prompt_length+1，于是 7473 行全被当成"太长"过滤光，只在日志里刷
# "Error processing one of the samples, skipping..."。V3 (2026-08-29) 就这么
# 烧掉一次。这里提前拦掉。
#
# 转换：python3 "${EXT_ROOT}/examples/minicpmo_5_0/prepare_gsm8k_text_only.py" \
#           --raw-dir <原始目录> --out-dir <输出目录>
for _f in "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [ ! -f "${_f}" ]; then
        echo "FATAL: 数据文件不存在: ${_f}"
        echo "       跑 ${EXT_ROOT}/examples/minicpmo_5_0/prepare_gsm8k_text_only.py 生成，或用 TRAIN_FILE=/VAL_FILE= 指定。"
        exit 1
    fi
    if ! python3 -c "
import sys
import pyarrow.parquet as pq
cols = set(pq.read_schema(sys.argv[1]).names)
need = {'prompt', 'data_source', 'reward_model'}
if not need <= cols:
    print('FATAL: %s 不是 verl RL 格式。' % sys.argv[1])
    print('       缺列: %s' % sorted(need - cols))
    print('       实际列: %s' % sorted(cols))
    if {'question', 'answer'} <= cols:
        print('       -> 这是**原始** GSM8K。跑 prepare_gsm8k_text_only.py 转换后再训。')
    print('       硬要跳过本检查: SKIP_DATASET_SCHEMA_CHECK=1')
    sys.exit(1)
" "${_f}"; then
        [ "${SKIP_DATASET_SCHEMA_CHECK:-0}" = "1" ] || exit 1
        echo "WARN: SKIP_DATASET_SCHEMA_CHECK=1，跳过 ${_f} 的 schema 校验。"
    fi
done

ROLLOUT_TP=${ROLLOUT_TP:-4}

# ---- 卡数探测与前置校验 ----
# 默认按 /dev/davinci* 探测。Ray 的 --resources 里的 NPU 只是个计数器，
# **不校验物理卡是否存在**：NUM_NPUS 报大了，多出来的 rank 会拿到不存在的
# device id，报 aclInit error 107001 "Invalid device ID"。这里提前拦掉。
_DETECTED_NPUS=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
NUM_NPUS=${NUM_NPUS:-${_DETECTED_NPUS:-8}}
# [2026-10-13] 节点数要连**来源**一起记：本机环境里有 MA_NUM_HOSTS=2（平台注入的），
# 旧写法默默把它当 NNODES=2 ⇒ global pool 16x2=32，再加 reward pool 的 16 卡 ⇒
# trainer 里报 "Total available GPUs 32.0 is less than total desired GPUs 48"。
# 那 16 张卡是**凭空多出来的**，因为本机就一个节点。来源写清楚，warning 才说得清。
if [ -n "${NNODES:-}" ]; then
    _NNODES_SRC="显式 NNODES"
elif [ -n "${VC_WORKER_NUM:-}" ]; then
    NNODES=${VC_WORKER_NUM}
    _NNODES_SRC="VC_WORKER_NUM"
elif [ -n "${MA_NUM_HOSTS:-}" ]; then
    NNODES=${MA_NUM_HOSTS}
    _NNODES_SRC="MA_NUM_HOSTS(平台注入，通常与『本机实际跑几个节点』无关)"
else
    NNODES=1
    _NNODES_SRC="默认 1"
fi
if [ "${NNODES}" -gt 1 ] && [ "${_NNODES_SRC}" != "显式 NNODES" ]; then
    echo "WARNING: NNODES=${NNODES} 是从 ${_NNODES_SRC} 推出来的，不是显式设置。" >&2
    echo "         单机多卡请显式 export NNODES=1；否则 global pool = NUM_NPUS x NNODES 会翻倍，" >&2
    echo "         叠加 reward pool 后超出集群卡数（真机形态：desired 48 > available 32）。" >&2
fi

if [ "${NUM_NPUS}" -le 0 ] 2>/dev/null; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not a positive integer."
    echo "       /dev/davinci* probe found '${_DETECTED_NPUS}'. Set NUM_NPUS explicitly."
    exit 1
fi
if [ "${_DETECTED_NPUS}" -gt 0 ] && [ "${NUM_NPUS}" -gt "${_DETECTED_NPUS}" ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} exceeds ${_DETECTED_NPUS} devices visible on this node."
    echo "       This is the aclInit 107001 failure (ranks ${_DETECTED_NPUS}..$((NUM_NPUS-1)))."
    echo "       Use NUM_NPUS=${_DETECTED_NPUS} on a single node, or raise NNODES for multi-node."
    exit 1
fi
if [ $((NUM_NPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not divisible by ROLLOUT_TP=${ROLLOUT_TP};"
    echo "       rollout.agent.num_workers (replica count) would be wrong."
    exit 1
fi
ROLLOUT_NUM_WORKERS=$((NUM_NPUS / ROLLOUT_TP))
echo "Cluster: NUM_NPUS=${NUM_NPUS} (detected ${_DETECTED_NPUS}) NNODES=${NNODES}" \
     "ROLLOUT_TP=${ROLLOUT_TP} replicas=${ROLLOUT_NUM_WORKERS} IF=${SOCKET_IFNAME}"

# Ray 侧资源账本校验：脚本可能连的是已存在的集群，
# 若集群是按 16 卡起的，光改 NUM_NPUS 没用。
if command -v ray >/dev/null 2>&1; then
    _ray_npu=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ NPU' | head -n1 | sed 's#.*/##; s# NPU##' || true)
    if [ -n "${_ray_npu:-}" ]; then
        _ray_npu_int=$(printf '%.0f' "${_ray_npu}")
        # [2026-10-13] 加上 reward pool —— trainer 的资源账本是**所有 pool 相加**
        # （trainer_base._create_resource_pools：global_pool + reward_pool），
        # 只算 global 会让 32>=32 通过、然后在 trainer 里以一条看不懂的消息炸掉。
        # 该值与启动命令里的 reward.reward_model.n_gpus_per_node 必须一致。
        REWARD_POOL_NPUS=${REWARD_POOL_NPUS:-16}
        _want=$(( NUM_NPUS * NNODES + REWARD_POOL_NPUS ))
        echo "Ray cluster reports ${_ray_npu_int} NPU; this job wants ${_want}" \
             "= global(${NUM_NPUS}x${NNODES}) + reward(${REWARD_POOL_NPUS})."
        if [ "${_ray_npu_int}" -lt "${_want}" ]; then
            echo "FATAL: Ray cluster only has ${_ray_npu_int} NPU but the job needs ${_want}," >&2
            echo "       of which global pool takes ${NUM_NPUS}x${NNODES} and the reward pool" >&2
            echo "       (enable_resource_pool=true) takes ${REWARD_POOL_NPUS}." >&2
            echo "       先查 NNODES 是否被平台注入的 MA_NUM_HOSTS/VC_WORKER_NUM 顶大了" >&2
            echo "       (单机请 export NNODES=1)，再考虑重启 Ray 或缩小 reward pool。" >&2
            exit 1
        fi
    fi
fi

# ---- 数据文件防护 ----
# verl 的 copy_to_local 对空字符串会抛 IndexError: string index out of range
# （fs.py:238 的 src[-1] 越界），堆栈很深不好读，这里提前拦掉。
for _v in MODEL_PATH TRAIN_FILE VAL_FILE; do
    _p="${!_v}"
    if [ -z "${_p}" ]; then
        echo "FATAL: ${_v} is empty. Set it explicitly."
        exit 1
    fi
    if [ ! -e "${_p}" ]; then
        echo "FATAL: ${_v} does not exist: ${_p}"
        exit 1
    fi
done

# =============================================================================
# §4.1 显存预算
# =============================================================================
# gpu_memory_utilization 是 **vLLM 引擎可用显存的总占比，模型权重算在里面**，
# 不是「扣掉权重之后留给 KV cache 的那一份」。设小了会在 KV cache 分配处报
# ValueError（剩余空间装不下一条请求），而不是报「显存不够」。
#
# MiniCPM-o 5.0 的 checkpoint 共 71.9 GB（thinker 71.07 + tts 0.84）。thinker-only
# rollout 只加载 thinker，权重地板约：
#   71.07 GB / TP=4 ≈ 17.8 GB/卡     （与 Qwen3.5-35B-A3B 的 17.56 GB 同量级）
# 单卡 total 61.27 GB，故地板占比 ≈ 0.29，低于约 0.35 连权重都放不下。
#
# 0.85 是同 TP=4 / 同 910B 64GB 上 Qwen3.5-MoE 跑通的值，权重量级一致，直接沿用。
# 若 OOM，往【下】退：0.85 -> 0.80 -> 0.70 -> 0.60。
# 真要省显存，正确的杠杆是 ROLLOUT_TP 4->8（权重降到约 8.9 GB/卡）。
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}

# =============================================================================
# §4.2 精度开关 —— 改错会「能跑但不学」，是最隐蔽的一个
# =============================================================================
# verl/verl/workers/engine/fsdp/transformer_impl.py:239-245
#     torch_dtype = self.engine_config.model_dtype
#     if torch_dtype is None:
#         # if it is training, we force torch_dtype to fp32
#         torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
#
# 显式传 bfloat16 会**顶掉**这段 fp32 强制，等于拿掉 optimizer 的 fp32 master weight。
# bf16 只有 8 bit 尾数（相对精度 ~2^-8 ≈ 0.4%），而 lr=1e-6 的单步更新量远小于该量级，
# 绝大部分更新会被直接舍入掉 —— 训练照跑、loss 正常、reward 就是不涨。
#
# 前向/反向仍是 bf16：由 MixedPrecisionPolicy(param_dtype=bfloat16) 单独负责
# （同文件 :378 与 :436），与本开关无关，改 fp32 **不会**把计算退成 fp32。
#
# Qwen3.5-MoE 实测：bfloat16 的 reward 在 0.25~0.45 震荡不涨；
# fp32 从 0.28 涨到 0.83（step 101）。同主干，结论直接适用。
#
# fp32 master weight 会增加 host 内存（optimizer state 在 CPU 上），
# 若 host 侧吃紧，退让顺序见 §4.4，**不要**拿这个开关换内存 ——
# 换来的是「能跑但不学」。
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-fp32}

# ref 是 forward_only、不持 optimizer state，且本配置 use_kl_loss=false
# （ref logprob 不进 loss），bf16 在数值上无害且省显存，故保持 bfloat16。
REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bfloat16}

# =============================================================================
# §4.4 训练超参
# =============================================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}

# PPO_MINI_BATCH_SIZE：4 表示每批做 16 次更新。
# GSPO 的 clip_ratio 只有 3e-4/4e-4，off-policy 漂移会被 clip 大量截断，
# 若观察到 clip fraction 接近 1，理论上该调大这个值 ——
# **但在 16 卡上调不动**：它会同比放大每卡序列数，而下面
# PPO_MICRO_BATCH_SIZE_PER_GPU 的注释解释了为什么每卡 >4 条时两头都是 OOM。
# 真要调，先加卡。
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}

ROLLOUT_N=${ROLLOUT_N:-8}

# -----------------------------------------------------------------------------
# PPO_MICRO_BATCH_SIZE_PER_GPU —— 必须让「每卡 micro-batch 数 == 1」，否则必 OOM
# -----------------------------------------------------------------------------
# 这不是调优参数，是这一版 verl 的硬约束。原因在
# verl/workers/engine/fsdp/transformer_impl.py:672-698 `_gradient_sync_context`：
# 除最后一个 micro-batch 外，它对 FSDP2 调 set_requires_gradient_sync(False)，
# 其 docstring 自己承认代价是 "temporarily retaining unsharded gradients until
# the final backward"。落到 torch 里是 _fsdp_param_group.py:389-394 —— reshard 之后
# 直接 return，**不做 reduce-scatter**；再经 _fsdp_param.py:639-645
# to_accumulated_grad_if_needed()，因为 reduce_dtype 被硬编码成 fp32
# （transformer_impl.py:379）而 grad 是 bf16，dtype 不等 => 升位到 fp32 并保留。
#
# 于是每多一个 micro-batch，就要在卡上囤住**全部已反传层的未分片 fp32 梯度**。
# MiniCPM-o 的 thinker 主干就是 Qwen3.5-MoE，专家层规模同量级：
#     单层 expert 参数 805,306,368 × 4 B = 3.0 GiB
#     × 40 层                            ≈ 120 GiB      <- 单卡只有 61.27 GiB
# 注意 CPUOffloadPolicy 救不了：它只搬 reduce-scatter **之后**的分片梯度，
# 这条路径压根没走到 reduce-scatter。
#
# micro-batch 数 = 每卡 mini-batch 序列数 / PPO_MICRO_BATCH_SIZE_PER_GPU，其中
#     每卡 mini-batch 序列数 = PPO_MINI_BATCH_SIZE × rollout.n / world_size
# rollout.n 那一乘是 trainer 内部做的（verl/trainer/ppo/v1/trainer_base.py:1674-1675），
# 脚本里看不到。
_WORLD_SIZE=$(( NUM_NPUS * NNODES ))
_SEQS_PER_RANK=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N / _WORLD_SIZE ))
if [ "${_SEQS_PER_RANK}" -lt 1 ]; then
    echo "FATAL: PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} × rollout.n=${ROLLOUT_N}" \
         "= $(( PPO_MINI_BATCH_SIZE * ROLLOUT_N )) < world_size=${_WORLD_SIZE};" >&2
    echo "       每卡分不到一条序列。请调大 PPO_MINI_BATCH_SIZE。" >&2
    exit 1
fi
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-${_SEQS_PER_RANK}}
_NUM_MICRO_BATCHES=$(( _SEQS_PER_RANK / PPO_MICRO_BATCH_SIZE_PER_GPU ))
echo "Actor batching: ${_SEQS_PER_RANK} seq/rank / micro_bsz=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
     "=> ${_NUM_MICRO_BATCHES} micro-batch(es)"
if [ "${_NUM_MICRO_BATCHES}" -gt 1 ]; then
    echo "WARNING: micro-batch 数 > 1，会触发未分片 fp32 梯度囤积（约 3.0 GiB/层 × 40 层）。" >&2
    echo "         除非你清楚自己在做什么，否则 OOM 几乎是必然的。" >&2
fi
if [ "${_SEQS_PER_RANK}" -gt 4 ]; then
    echo "WARNING: 每卡 ${_SEQS_PER_RANK} 条序列。这一版 verl 在 ${_WORLD_SIZE} 卡下**没有安全解**：" >&2
    echo "         micro_bsz 设小 -> 多 micro-batch -> 未分片梯度囤积 OOM；" >&2
    echo "         micro_bsz 设大 -> 单次前反传激活/logits 撑爆 -> 同样 OOM。" >&2
    echo "         唯一出路是加卡（抬 NNODES）或减小 PPO_MINI_BATCH_SIZE / ROLLOUT_N。" >&2
fi
# 如果 ${_SEQS_PER_RANK} 条序列的激活放不下，正确的退让顺序是：
#   1) 缩序列长度（见下面 §4.5）—— 首选，因为 logits 栈跟长度是线性的；
#   2) PPO_MINI_BATCH_SIZE 减半（4->2），每卡降到 1 条 —— 代价是 optimizer step 数翻倍，
#      GSPO 的 clip 3e-4/4e-4 对 off-policy 漂移很敏感，改完盯 clip fraction；
#   3) 最后才是动 micro_bsz —— 因为动它就会重新引入上面那条囤积路径。

# =============================================================================
# §4.5 序列长度 —— 单次前反传显存的主导项
# =============================================================================
# 本配置 use_remove_padding=False（原因见启动命令上方的说明），所以一条序列在
# 前反传里**按 padding 后的满长度算**：
#
#     L = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
#
# 吃显存最狠的是全词表 logits：
#     单份 logits = seq/卡 × L × vocab_size × 2 B (bf16)，实际约 3 份
#     （logits + log_softmax + 反传梯度；entropy_coeff=0 且 calculate_entropy
#       解析为 False，所以 entropy 那份不产生）
# Qwen3.5-MoE 在 vocab=248144、seq/卡=2、L=4096 下约 11.4 GiB，可跑。
# MiniCPM-o 的 vocab 需从 checkpoint config 确认（thinker 主干同为 Qwen3.5-MoE，
# 但 MiniCPM 系列常有自己的扩充词表）：
#     python3 -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); \
#                 print(c.get('vocab_size') or c.get('llm_config',{}).get('vocab_size'))" ${MODEL_PATH}
# 若显著大于 248144，按比例把 L 压一档再起跑。
# [2026-10-13] 默认 2048 → 8448：这个值同时是三样东西的上限 ——（a）文本过滤
# data.max_prompt_length，（b）rollout.prompt_length = 本值 + 32，（c）训练侧
# L = 本值 + MAX_RESPONSE_LENGTH 的显存口径。**多模态展开后的长度只看 (b)**：
# 一张 2544x3231 的图在 rollout 侧就是 8080 个 <|image_pad|>，2048/3072 的预算
# 会让 7.7% 的行（实测 2 万行抽样）在 rollout 侧撞 "Multimodal prompt produced
# N tokens" 而整条 rollout 报废。8448 把这个比例压到 0.7%（p99=8242）。
# 代价是显存：L 从 5120 涨到 10496，前反传的 logits（约 3 份，见 §4.5 算式）
# 大致同比例上涨。**OOM 时优先降回本值**（再配 MAX_TOKEN_LEN_PER_GPU），
# 不要单独去降 ROLLOUT_PROMPT_LENGTH —— 那样过滤与 rollout 的口径会脱节。
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8448}
# 2048 而非 1024：thinking 模型的 CoT 在 1024 下常在写出 \boxed{} 前就被截断，
# 被截样本判 0 分，reward 被摁住爬不起来（Qwen3.5 实测 clip_ratio 0.81~0.91）。
# 1024→2048（2026-10-10）：genrm_dump_qwen36 里被截样本的 token 数正好压在
# 511/512，且整段没有 </think> —— 说明它连思维链的中段都没写完。512 及以下
# 对 thinking 模型不是"省时间"，是"这一步直接跑废"（见下方长度预算注释）。
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
_SEQ_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))

# rollout 侧跟着一起缩，否则 KV cache 仍按大长度预留、白占显存
#
# [2026-10-13] 同时驱动 `rollout.max_num_batched_tokens`（见下方 hydra 参数区）：
# vLLM 的 SchedulerConfig 要求 max_num_batched_tokens >= max_model_len，否则直接
# pydantic 报 "max_num_batched_tokens (8192) is smaller than max_model_len (10624)"
# 起不来。verl 的 RolloutConfig 默认 8192（rollout.py:196），本脚本原先不设它 ——
# 一旦把 MAX_PROMPT_LENGTH 抬到 8448（⇒ max_model_len=10624）两值就脱节。
# 与 max_model_len 同源即可满足约束，又不为 chunked prefill 多要显存
# （enable_chunked_prefill=False，一条序列的 prefill 必须一次装下）。
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$(( _SEQ_LEN + 128 ))}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-$(( MAX_PROMPT_LENGTH + 32 ))}

# 媒体预算：把 rollout.prompt_length 交给数据集，让它在**加载期**就丢掉
# 「文本短、媒体大」的行。2026-10-13 事故：上游 filter_overlong_prompts 只量
# 文本（53 tokens），图片在 rollout 侧展开成 8080 个 <|image_pad|> ⇒ 超过
# prompt_length=3104 ⇒ 上游 fail-loud 报 "Multimodal prompt produced N tokens"。
# 实测（2 万行抽样）：预算 3104 丢 7.7% 行、4096 丢 4.1%、5120 丢 3.2%。
# 想少丢就调大本值 —— 代价在训练侧：L=prompt+response 直接决定前反传显存。
export MINICPMO_MM_PROMPT_BUDGET="${MINICPMO_MM_PROMPT_BUDGET:-$ROLLOUT_PROMPT_LENGTH}"

# token 预算 = 每卡序列数 × L。当前 use_dynamic_bsz=False，这个值不生效（见启动命令
# 上方说明），但保持它跟长度一致：万一哪天把 dynamic_bsz 打开，这个值恰好等于
# 「每卡全部序列装进 1 个 micro-batch」，不会退回多 micro-batch 的囤积路径。
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$(( _SEQ_LEN * _SEQS_PER_RANK ))}

echo "Seq len: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} L=${_SEQ_LEN}" \
     "| rollout max_model_len=${ROLLOUT_MAX_MODEL_LEN} (batched_tokens 同值) | token budget/gpu=${MAX_TOKEN_LEN_PER_GPU}" \
     "| 媒体预算=${MINICPMO_MM_PROMPT_BUDGET}"

# ⚠️ 调小 MAX_PROMPT_LENGTH 有一个静默副作用：data.filter_overlong_prompts=true
#    会把 prompt 超长的样本**直接丢掉**（不是截断，truncation=error）。
#    跑起来后在日志里搜 "filter" / 数据集条数，确认丢掉的比例可接受；
#    若丢太多，优先把 MAX_RESPONSE_LENGTH 再压一档，而不是把 prompt 加回去。

RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}

# -----------------------------------------------------------------------------
# enable_thinking —— 便宜的防御，不是必需的修复
# -----------------------------------------------------------------------------
# checkpoint 自带的 chat_template.jinja 里是这么写的（V0-11 §(1) 原样打印，
# 2026-08-29 实测。此前这段注释把条件记成了 `is true`，**方向是反的**，已订正）：
#
#     {%- if add_generation_prompt %}
#         {{- '<|im_start|>assistant\n' }}
#         {%- if enable_thinking is defined and enable_thinking is false %}
#             {{- '<think>\n\n</think>\n\n' }}         ← **已闭合的空思考块**
#         {%- else %}
#             {{- '<think>\n' }}                       ← 未闭合，模型接着推理
#         {%- endif %}
#     {%- endif %}
#
# 条件是 `is false`，所以：
#   不传  -> else -> '<think>\n'          思考**默认就是开的**
#   传 True -> else -> '<think>\n'        与不传逐字相同（V0-11 §(1b) 实测 True）
#   传 False -> if  -> 已闭合的空块        **这才是关掉思考的那条路**
#
# 那下面这一行还留着干什么？它买的不是"打开思考"，是三件事：
#   1. 挡住别处塞进来的 False（配置继承、别人改脚本、上游默认值变动）；
#   2. 把意图钉在 config dump 里，模板将来改默认值时不会跟着漂；
#   3. 给 ENABLE_THINKING=0 一条**显式**的关闭路径，用于 V3 那种"先跑通"的场合。
# 代价是 0。对 Thinker-Only 项目来说，唯一一条**不报错也不告警**的失败路径
# 就是思考被悄悄关掉，值得用一行配置堵死。
#
# 一处设置覆盖三个调用点（都读 data.apply_chat_template_kwargs）：
#   rl_dataset.py:210   doc2len 的 processor 分支     ← 长度过滤
#   rl_dataset.py:250   doc2len 的 tokenizer 分支     ← 长度过滤（当前走这条）
#   agent_loop.py:412/436  rollout 真正拼 prompt      ← 训练时的实际输入
# 这一点没有因为方向订正而改变，而且现在更要紧：将来谁真要关思考，必须三处同关。
# 只关一处，长度过滤用的 prompt 会和训练用的 prompt 不同，过滤结果失去意义
# （而且照例不会有任何提示）。
#
# 用 `++` 而不是 `+`：apply_chat_template_kwargs 在 legacy_data.yaml:119 是 `{}`，
# 按理 `+` 正确；但 `++` 在"键已存在"时也不会失败，代价是拼错键名不会被发现——
# 键名已核对过源码，这里更怕的是为一个前缀白起一次 16 卡。
ENABLE_THINKING=${ENABLE_THINKING:-1}
if [ "${ENABLE_THINKING}" = "1" ]; then
    _ENABLE_THINKING_PY=True
    echo "enable_thinking = True   -> prompt 以未闭合的 <think>\\n 结尾，模型先推理再作答"
    echo "                            （与不传该 kwarg 渲染相同；这一行是防御，防的是别处传 False）"
else
    _ENABLE_THINKING_PY=False
    echo "enable_thinking = False  -> ⚠️ prompt 以**已闭合的空 <think></think>** 结尾，"
    echo "                            模型将跳过推理直接作答。"
    echo "                            这条路只适合 V3 的『能不能跑起来』验证（response 更短、更快），"
    echo "                            **不要用于正式的 Thinker-Only 训练**。"
fi
_CHAT_TEMPLATE_ARGS=( "++data.apply_chat_template_kwargs.enable_thinking=${_ENABLE_THINKING_PY}" )

# 与长度预算的关系：开了 thinking，response 会显著变长。上面 MAX_RESPONSE_LENGTH
# 取 2048 就是按"要写 CoT"定的（见该处注释：1024 时常在 \boxed{} 前被截断，
# 截断样本判 0 分，reward 被摁住）。
#
# ⚠️ 反面经验（2026-10-10，别再照抄旧结论）：以前这里写着"把 thinking 关掉就能
# 顺手把 MAX_RESPONSE_LENGTH 压到 512"。实测 512 对 thinking 模型是灾难而不是
# 提速：genrm_dump_qwen36 里被截样本正好卡在 511/512 token、连 </think> 都没有，
# 尾部退化成复读（「平安、平安、…」），judge 吐不出 Verdict → 整组 reward=0。
# 想压缩预算请在 ENABLE_THINKING=False（模板会注入**已闭合**的空 </think>，
# 模型直接作答）的前提下压，且至少要留够最终答案的长度。
if [ "${_ENABLE_THINKING_PY}" = "True" ] && [ "${MAX_RESPONSE_LENGTH}" -lt 1024 ]; then
    echo "⚠️  [warn] enable_thinking=True 但 MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} < 1024：" >&2
    echo "          CoT 极可能在 \boxed{} 之前被硬截断（实测 512 时整段复读、无 </think>）。" >&2
    echo "          除非你在做刻意的截断实验，否则请把预算抬到 2048 或关闭 thinking。" >&2
fi


# -----------------------------------------------------------------------------
# 自定义 Dataset 类 —— 第 5 号坑（processor 没有对话模板）的**结构性**解法
# -----------------------------------------------------------------------------
# 现有两条兜底都是"把 processor 修好"或"把 processor 关掉"：
#   · minicpmo_5_0_processor.py 给它补模板，补完真渲染一遍，渲染不了就交回 None；
#   · MINICPMO_DISABLE_PROCESSOR_FALLBACK=1 直接不构造 processor。
# 两条都能让纯文本跑起来，但都绕不过一个事实：**只要 processor 存在，
# RLHFDataset 就会拿它去渲染对话模板**（rl_dataset.py 只看 `processor is not None`）。
# 到了多模态阶段 processor 必须存在（要拿它处理像素），那时这两条兜底都失效。
#
# GPU 上那套真的在训 MiniCPM-o 4.5 的 verl（MB_NPU/verl_mm-...-half_duplex_omni）
# 走的是第三条路，也是唯一一条到多模态还成立的路：**换掉 Dataset 类**。
#   examples/minicpmo4.5/minicpmo45-one-step-off-new-videor1.sh:
#       data.custom_cls.path=pkg://verl.utils.dataset.minicpmo_dataset
#       data.custom_cls.name=MiniCPMORLHFDataset
# 它那个类里，文本和模板一律走 tokenizer，processor 只负责像素。分工是：
#       tokenizer -> 文本、对话模板、长度        processor -> 像素和音频
#
# MiniCPMOThinkerRLHFDataset 就是这条路的最小实现：只改"用谁量长度"，
# 其余（下载、读 parquet、__getitem__、多模态、断点续训）全部继承不动。
# 附带解决坑 31 / 坑 32 共有的那个最贵的问题——**过滤到 0 行不再是静默状态**，会直接报错并点名根因。
#
# 默认关闭：V3 还卡着，一次只动一个变量。V0-10 第 (0) 节若证明训练机代码是新的
# 而 V3 仍死在同一行，这就是下一个要打开的开关（比 DISABLE_PROCESSOR_FALLBACK 更该先试）。
MINICPMO_USE_CUSTOM_DATASET=${MINICPMO_USE_CUSTOM_DATASET:-0}
_DATASET_ARGS=()
if [ "${MINICPMO_USE_CUSTOM_DATASET}" = "1" ]; then
    _DATASET_ARGS=(
        # 改动 F：class 名不变（verl_omni_ext/models/minicpmo_5_0/dataset.py 同款导出），
        # path 指向 out-of-tree 包
        "data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_whisper.dataset"
        "data.custom_cls.name=Qwen36WhisperThinkerRLHFDataset"
    )
    echo "dataset  = Qwen36WhisperThinkerRLHFDataset（长度过滤走 tokenizer；过滤到 0 行会报错而不是静默）"
else
    echo "dataset  = 默认 RLHFDataset（长度过滤看 processor 在不在）"
    echo "           若 V3 仍死在 apply_chat_template，设 MINICPMO_USE_CUSTOM_DATASET=1 重跑"
fi


# [MiniCPM 改动 3/6] 实验命名
PROJECT_NAME=${PROJECT_NAME:-qwen3_6_whisper_gspo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_qwen3_6_whisper_thinker_npu}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${EXT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# _TensorboardAdapter 默认写相对路径 tensorboard_log/<proj>/<exp>（tracking.py:299），
# cwd 一变或容器一销毁就找不到，显式指向持久化目录。
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${EXT_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${TENSORBOARD_DIR}" "${DEFAULT_LOCAL_DIR}"

# =============================================================================
# 前置自检（out-of-tree 改动 D）
# 把「跑到第 40 分钟才炸」变成「启动 3 秒内退出并告诉你缺什么」。
#   1) 训练侧适配器已注册（槽位①，pyproject entry_points → _load_all，key=("MiniCPMO","thinker")）
#   2) rollout 适配器已注册（槽位②，key=minicpmo_5_0）
#   3) L2 补丁全部 applied + alive（_patchkit + watchdog）
# set -x 只在出错时刷屏，自检段临时关掉免得输出版面爆炸。
# =============================================================================
set +x
echo "=== 前置自检 ==="
python3 - <<'PY'
import os
import sys
from importlib.metadata import entry_points

assert any(
    ep.name == "qwen3_6_whisper"
    and ep.value == "verl_omni_ext.models.qwen3_6_whisper"
    for ep in entry_points(group="verl_omni.models")
), "Qwen3.6 plugin entry point missing; install this checkout with python3 -m pip install -e . --no-deps --no-build-isolation"

import verl_omni_ext  # 本脚本已 export VERL_USE_EXTERNAL_MODULES 含 verl_omni_ext → 自动 activate()
from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase

key = ("Qwen3_5MoeForConditionalGeneration", "thinker")
assert key in OmniModelBase._registry, (
    f"Adapter {key} not registered! 检查 pyproject.toml entry-points 与 verl_omni_ext/__init__.py _load_all()"
)
print("✓ 训练侧适配器注册:", key)

assert "qwen3_6_whisper" in OmniRolloutPipelineBase._registry, "Rollout adapter qwen3_6_whisper not registered!"
print("✓ rollout 适配器注册: qwen3_6_whisper")

from verl_omni_ext._patchkit import self_check, patch_state_line, verify_patches_alive

res = self_check()
for name, ok in res.items():
    assert ok, f"L2 patch {name!r} not applied!"
print("✓ L2 patches applied:", list(res) or "(无 L2 补丁)")
print(patch_state_line())
assert all(verify_patches_alive().values()), "Some L2 patch lost at runtime!"
print("✓ L2 patches alive")
os._exit(0)
PY
set -x
echo "=== 自检通过 ==="

# =============================================================================
# §5. 启动
# =============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 严禁在下面这段反斜杠续行里用 `#` 注释掉某一行参数。
#
# bash 在词法阶段就把 `#` 当注释开始，一直吃到行尾 —— 包括那一行末尾的续行符 `\`。
# 续行链一断，整条命令就在那里结束，**该行之后的所有参数以及末尾的管道全部丢失**，
# 剩下的行还会被当成独立命令执行，报出一个跟你改的东西毫无关系的错误。
#
# 要临时去掉某个参数：直接删行。
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# 下面这条命令里有两个**看着生效、实际被吞掉**的参数，别按字面理解：
#
# 1) actor.fsdp_config.param_offload / optimizer_offload
#    只要 offload_policy=true（或 forward_only，ref worker 就是），
#    transformer_impl.py:440-444 会无条件把 _is_offload_param、_is_offload_optimizer
#    强制改回 False，改用 FSDP2 原生的 CPUOffloadPolicy(pin_memory=True)。
#    重要推论：CPUOffloadPolicy 只搬 reduce-scatter **之后**的分片梯度，
#    因此它对 §4.4 说的那条未分片梯度囤积路径完全无效 —— 别指望它救 OOM。
#    那个 pin_memory=True 是写死的，且正是 logs/13.txt 的死因；
#    现在由 MINICPMO_FSDP2_PIN_MEMORY 控制，见本文件上方那段长注释。
#
# 2) actor.ppo_max_token_len_per_gpu
#    仅在 use_dynamic_bsz=True 时用于切分；这里是 False，该值不起作用。
#    真正决定单次前反传规模的是 ppo_micro_batch_size_per_gpu。
#
# 还有一个**故意不设**的参数：actor_rollout_ref.rollout.limit_images。
#    verl 见到它就把 engine_kwargs["limit_mm_per_prompt"] 整个换成 {"image": N}
#    （vllm_async_server.py:253），会把 audio/video 的额度一并抹掉。
#    MiniCPM-o 的三模态额度由 MiniCPMO50RolloutAdapter.get_stage_engine_extras
#    通过 engine_extras 提供，见 verl_omni_ext/models/minicpmo_5_0/rollout_adapter.py。
#
# [MiniCPM 改动 4/6、5/6、6/6] 都在下面这条命令**里面**，因此不能就地标注
# （见上面的大写警告：续行段里加 `#` 会截断整条命令）。三处分别是：
#   4/6  actor_rollout_ref.model.trust_remote_code=True      （Qwen3.5 版是 False）
#   5/6  ...rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_5_0
#                                                            （Qwen3.5 版是 qwen3_5_moe）
#   6/6  末尾 tee 的日志文件名
# 除这三行外，整条命令与 run_qwen35_moe_thinker_gspo_npu.sh 逐字相同 ——
# 唯一的例外是 data 段末尾展开的两个数组：
#   "${_CHAT_TEMPLATE_ARGS[@]}"  enable_thinking，见上方 §4 末尾那一大段
#   "${_DATASET_ARGS[@]}"        自定义 Dataset 类，见紧随其后那一段（默认展开为空）
# 它们用数组展开而不是直接写死一行，正是因为这段续行里不能加 `#`：
# 要让这些参数可开可关，只能把分支放到命令外面。
# ─────────────────────────────────────────────────────────────────────────────
# [2026-10-10 新增] 采样参数必须与 checkpoint 的 generation_config.json 对齐
# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.6-35B-A3B-whisper-0814-rl/generation_config.json 给的是
#     do_sample=true, temperature=1.0, top_k=20, top_p=0.95
# 而 verl 的 RolloutConfig 默认 top_k=-1 / top_p=1.0（rollout.py:40-42），并且
#     vllm_async_server.py:1196 `_get_override_generation_config()` 会把
#     temperature/top_k/top_p 显式塞进 engine 的 override_generation_config，
#     **盖掉 checkpoint 自带的 generation_config.json**。
# 于是此前实际生效的是 top_k=-1 / top_p=1（全词表无截断采样）。这正是
# genrm_dump_qwen36 里 thinking 模式跑到中途退化成「平安、平安、…」复读的直接
# 诱因：temperature=1.0 配无截断的 tail 会持续放大尾部低概率 token（若只是预算
# 不够，复读应当出现在末尾截断处，而不是生成到 ~200 token 就开始）。
# 显式写回 checkpoint 的推荐值（temperature 本来就是 1.0，一并写死，避免上游
# 再改默认值）。多模态模型尤其要紧：视觉/音频 token 混在序列里时，无截断采样
# 更容易塌进重复。
# 可用环境变量覆盖：ROLLOUT_TOP_P / ROLLOUT_TOP_K / ROLLOUT_TEMPERATURE。
# 注意 argv 里已按「续行段不能加 `#`」的规矩写死数值，注释只能放在这里。
# ─────────────────────────────────────────────────────────────────────────────
# [2026-10-10 续] repetition_penalty：上一段修好了 top_k/top_p，复读仍在
# ─────────────────────────────────────────────────────────────────────────────
# 证据：genrm_dump_qwen36/sample_0001.json 的 actor 输出
#     repeat='关于'x446；qwen36_whisper_full.log 的 rollout-diag
#     repeat='102307,5205'x188@135。
# 同一条音频 prompt 的 n=4 里，有的干净（sample_0002 被 judge 判 1.0），有的从第
# 135 个 token 起塌进同一个 2-gram —— 采样塌陷，不是模型不会答。T=1.0/top_p=0.95/
# top_k=20 配 repetition_penalty=1.0 是最容易塌的组合；同一家族的 judge
# （Qwen3.5-35B-A3B）自己的 generation_config.json 就写着 1.1。
#
# 注意这里**不能**直接抄 generation_config.json 的结论：actor 的
# generation_config.json 里根本没有 repetition_penalty 字段，默认就是 1.0。所以
# 这是「主动加一道闸」，默认 1.05（比 judge 的 1.1 保守：actor 采样要保留多样性，
# judge 只要不塌就行）。1.0 恢复上游行为（对照组用）。
#
# 关键：这个字段**必须**配合包内 rollout_sampling.py 的 L2 补丁才生效 —— 上游
# agent loop 把它写死成 1.0（agent_loop.py:594 / agent_loop_tq.py:68），
# RolloutConfig.repetition_penalty 本来是死配置。
# 补丁打上后每个 agent-loop 进程打**两行**，缺一不可：
#   [rollout-sampling] 采样闸补丁已安装到: [...]      ← import 期，点名打在哪些类上
#   [rollout-sampling][diag] pid=… cls=… 观测: …      ← 第一次调用时，无条件打
#                                                      （「补丁调到没有」的唯一答案）
#   [rollout-sampling] repetition_penalty=1.05 …      ← 只有真注入了才打（生效行）
# run#10 的教训：self-check 报 ✓ applied、真机却一行都没有 —— 原因是补丁只打在
# 基类上，而 Ray 会把继承的方法复制到 backing class，actor 解析不到。现在三个候选
# 类都打（基类 / TQ 装饰类 / Ray backing class），共用同一个 replacement。
# 可用环境变量覆盖：ROLLOUT_REPETITION_PENALTY。
#
# ⚠️ 前缀必须是 ``++``（2026-10-10 22:0x 修）：`repetition_penalty` 在
# RolloutConfig 这个 dataclass 里（verl/workers/config/rollout.py:168），但
# **不在** hydra 的 rollout.yaml 里，于是 struct 里没有这个键，写成
# ``actor_rollout_ref.rollout.repetition_penalty=…`` 会在启动第 30 秒直接抛
#   omegaconf.errors.ConfigAttributeError: Key 'repetition_penalty' is not in struct
#   hydra.errors.ConfigCompositionException: Could not override ...
# 整轮跑废（run_006 就是这么死的，N=1 的 run_001 死在这次改动之前、侥幸没踩到）。
# ``++`` = 有则覆盖、无则新增，两种情况都对（``+`` 只在「本来没有」时对）。
# 同文件里 ``++data.apply_chat_template_kwargs.enable_thinking=…`` 是同一个理由。
#
# ─────────────────────────────────────────────────────────────────────────────
# [2026-10-10 23:3x 续] frequency_penalty / presence_penalty：rp 上线后复读仍在
# ─────────────────────────────────────────────────────────────────────────────
# rp **压不住环**，这是 vLLM 的语义而不是配置没生效：它只问「这个 token 出现过
# 没有」，出现 1 次和出现 477 次统一除以 1.1。于是 2-gram 环（judge `' an'x477`、
# actor `'102307,5205'x188@135`）在 rp=1.1 下照样转 —— 环里每个 token 都「出现过」。
#
# 压环要的是**累计**惩罚：frequency_penalty 按 token 在本序列里出现的**次数**
# 线性扣 logits，被生成 477 次的那个 token 会被越扣越狠。judge 侧一直有这个开关
# （GENRM_FREQUENCY_PENALTY，默认 0.0），actor 侧此前**没有这条路** ——
# 上游 agent loop 的 sampling_params 只有 temperature/top_p/top_k/
# repetition_penalty/logprobs（agent_loop.py:590-596）。本包的 L2 补丁
# （rollout_sampling.py）刚把这几条补上，生效证据是每进程一行的
# [rollout-sampling] frequency_penalty=...
# 可用环境变量覆盖：ROLLOUT_FREQUENCY_PENALTY / ROLLOUT_PRESENCE_PENALTY。
#
# FP32 GDN 修复后默认 fp=0：fp=0.5 会累计抑制正常重复使用的数字/日期/公式。
# 四样本对照中，图表题恢复了中文推导、</think> 和最终答案；保留显式覆盖，
# 但不再用强频率惩罚掩盖推理侧的数值问题。
# ⚠️ 注释永远不要插进下方 `\` 续行块中间：会把命令截断、丢掉后面全部 hydra
# 覆盖（见 memory: hydra-continuation-comment-trap，判据 = xtrace 行长度骤缩）。
#
# ⚠️ 这两个闸**不能**写成 ``++actor_rollout_ref.rollout.frequency_penalty=…``
# （2026-10-10 23:40 修）：它们**不是** RolloutConfig 的字段（整个
# verl/workers/config/rollout.py 里只有 repetition_penalty 一处），而
# ``omega_conf_to_dataclass`` 是 ``OmegaConf.merge(OmegaConf.structured(RolloutConfig), cfg)``
# → 多出来的键会抛 ``ConfigKeyError: Key 'frequency_penalty' not in 'RolloutConfig'``
# （本机跑 ``OmegaConf.merge`` 那一行实测过：侧信道键 OK、直写键抛错），
# 也就是「整轮起不来」，不是「读不到」。
# 所以走**侧信道**：``rollout.engine_kwargs.q36_sampling.<闸>``。
#   · ``engine_kwargs`` 是自由字典（rollout.py:137），塞什么键都不触发结构校验；
#   · 服务端只读 ``engine_kwargs[<引擎名>]``（vllm_async_server.py:287，引擎名 =
#     vllm_omni），所以 ``q36_sampling`` 这一层**永远传不到引擎里**，是纯配置位；
#   · 消费方是本包的 rollout_sampling.py 补丁，不是引擎。
#
# 默认 0.0 = 与上游逐字节相同的分布（0.0 是 vLLM 的 no-op）。
#
# ⚠️ ref 侧 `fsdp_config.forward_prefetch` 仍是 true（本配置下 ref worker 从不执行：
# `compute_ref_log_prob` 在日志里出现 0 次）。前向预取与反向预取是**同一个缺陷的两份**
# （verl 的静态表 modules[i+1:i+2] vs torch 的默认 post_forward_order 反向序）——
# 真要启用 ref worker，这里必须一并改成 false，否则图像/音频 rank 子图不同时会同样死锁。
#
# ⚠️ 下面这条命令是 `\` 续行的**一整条**：中间**不能插注释行**（注释会把续行吃掉，
# 命令在注释前就结束，后面所有覆盖被静默丢弃 —— 2026-10-12 02:0x 实测：`output_mode=ar`
# 被丢 ⇒ ARStrategy 退化成 DiffusionStrategy ⇒ `ConfigKeyError: Key 'hf_config_path'
# not in 'DiffusionModelConfig'`，每轮 4 分钟就死）。同类说明只能写在命令**之前或之后**。
python3 -m verl_omni.trainer.main_omni \
    trainer.device=npu \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.image_key=images \
    data.shuffle=true \
    data.seed=42 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    "${_CHAT_TEMPLATE_ARGS[@]}" \
    "${_DATASET_ARGS[@]}" \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni_ext \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD:-false} \
    actor_rollout_ref.model.exclude_modules=".*model\.visual.*|.*model\.audio_encoder.*|.*model\.audio_projection.*|.*tts.*|.*talker.*|.*code2wav.*|.*code_predictor.*" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.trainer_type=policy_gradient \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${ACTOR_FORWARD_PREFETCH:-false} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE:-bfloat16} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE:-bfloat16} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:-20} \
    ++actor_rollout_ref.rollout.repetition_penalty=${ROLLOUT_REPETITION_PENALTY:-1.05} \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=${ROLLOUT_FREQUENCY_PENALTY:-0.0} \
    ++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=${ROLLOUT_PRESENCE_PENALTY:-0.0} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.65} \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_MODEL_LEN:-4096} \
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH:-4160} \
    actor_rollout_ref.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_whisper \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    algorithm.adv_estimator=grpo \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    trainer.v1.trainer_mode=omni_sync \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-false} \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.n_gpus_per_node=${NUM_NPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=$([[ "${DEBUG_MODE:-0}" == "1" ]] && echo -1 || echo "${SAVE_FREQ:-50}") \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@" \
    "${HUANYU_GRAPH_ARGS[@]}" \
    2>&1 | tee run_qwen36_huanyu_gspo_npu.log
