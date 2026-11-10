# Huanyu 启动脚本与参数 Wiki

更新：2026-11-10。以下说明对应精简后的本地脚本；历史实验结果见[适配与排障记录](HUANYU_ADAPTATION_AND_TROUBLESHOOTING_ZH.md)，不要把历史配置当作当前默认值。

## 1. 目录、保留文件与职责

本分支的 `examples/` 是独立脚本与 Wiki 发布包，含以下 **11 个启动/辅助文件**。服务器运行目录为 `$PROJECT_ROOT/examples/qwen3_6_huanyu/`；模型适配、预计算和诊断代码使用完整工程中的 `verl_omni_ext/models/qwen3_6_huanyu/`。

| 文件 | 职责与执行位置 |
|---|---|
| `prepare_huanyu_env.sh` | 集中管理 editable/依赖安装、jemalloc、站点环境及 CANN/ATB；`source` 后应用环境，`--start-ray` 同时拉起并检查 Ray |
| `start_huanyu_ray.sh` | 已准备环境时，每节点只启动/复用 Ray；`--check` 做 GCS 资源检查 |
| `prepared_env.py` | 环境快照、差异与 activation 文件生成；由 prepare 调用 |
| `job_bootstrap.py` | 部署/节点解析、NPU 检查、GCS 等待、AQA 数据预检等辅助命令 |
| `run_huanyu_32card_job.sh` | 每节点 job 入口：环境、Ray、资源布局；仅 head 提交训练 |
| `run_huanyu_stem_job.sh` | STEM 16/32 等规模配置；`--train-only` 跳过准备和 Ray 启动 |
| `run_huanyu_stem.sh` | 已有环境/Ray 的 STEM 数据准备、文本模式和奖励配置 |
| `run_huanyu_aqa_audio_train.sh` | 已有环境/Ray 的 AQA 入口，音频冻结/解冻和数据预检 |
| `run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh` | OmniMix 参数；按奖励模式配置 GenRM 独立池或规则评分 |
| `run_qwen36_huanyu_gspo_npu.sh` | 最终训练后端：Huanyu adapter、预检、参数汇总和 `python -m verl_omni.trainer.main_omni` |
| `run_huanyu_train.sh` | 已有环境/Ray 的单文件训练入口；按功能集中列出配置及中文行内注释，直接调用 Python 训练模块 |

原 `run_huanyu.sh`、`run_qwen36_huanyu_32card.sh` 的职责并入最后两个后端；graph、precision、原生源码选择和运行日志初始化不再依赖独立 `.sh` 文件。其他专用 wrapper、AB/serve CLI、转换工具不再放在 examples。下列通用入口加参数替代专用训练 wrapper。

## 2. 调用链

```text
prepare_huanyu_env.sh
  ├─ prepared_env.py：生成环境快照与 activate.sh
  └─ run_huanyu_32card_job.sh（仅准备模式）
       └─ job_bootstrap.py：节点、设备、可选 Ray/GCS 检查
  --stem 时先经 run_huanyu_stem_job.sh 设定 STEM 资源/环境

start_huanyu_ray.sh
  └─ job_bootstrap.py：解析节点、等待 head/集群

run_huanyu_32card_job.sh（两节点各执行一次，head 分流训练）
  ├─ prepare_huanyu_env.sh 的准备函数：安装、allocator、站点/CANN/ATB 环境
  ├─ frozen_cache → run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
  ├─ aqa_unfreeze → run_huanyu_aqa_audio_train.sh
  └─ stem → run_huanyu_stem.sh

run_huanyu_stem_job.sh
  ├─ 默认 → run_huanyu_32card_job.sh → run_huanyu_stem.sh
  └─ --train-only → head 直接 run_huanyu_stem.sh；其他节点退出

run_huanyu_aqa_audio_train.sh / run_huanyu_stem.sh
  ├─ GenRM → run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
  └─ rule  → run_qwen36_huanyu_gspo_npu.sh

run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
  └─ run_qwen36_huanyu_gspo_npu.sh
       └─ python3 -m verl_omni.trainer.main_omni

run_huanyu_train.sh（已有环境/Ray，仅 head 执行）
  └─ python3 -m verl_omni.trainer.main_omni
```

`prepare --start-ray` 和独立 Ray 脚本共用启动策略，但 prepare 还会准备/验证 Python 环境、生成 activation，因此不是完全相同的操作。以上入口不依赖已经删除的六个 helper 文件。

环境安装只有 `prepare_huanyu_env.sh` 一处实现。job 以内部 `--functions-only` 接口加载准备函数，不执行 prepare 的独立入口，所以不会递归启动。单独调用 prepare 仍复用 job 的仅准备模式做资源配置和检查，成功后生成 activation；两条路径都只执行一轮准备。

训练入口（包括直接 `bash run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh`）只设置本次参数、自检并启动训练，不运行 pip/git 安装，也不再 source CANN/ATB 环境脚本。`HUANYU_SKIP_EDITABLE_INSTALL` 仅保留在 activation 中兼容旧部署；当前训练脚本不读取它，设为 `0` 也不会安装。缺少插件时自检失败，应先调用 prepare。

## 3. 部署与环境准备

显式设置项目路径，避免普通入口和 STEM job 的历史默认根目录差异。各节点必须使用相同的插件和脚本版本。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export SHELL_ROOT="$PROJECT_ROOT/examples/qwen3_6_huanyu"
git clone --branch examples --single-branch https://github.com/ZiqiGuan892/ext.git /tmp/huanyu-examples
mkdir -p "$SHELL_ROOT"
cp /tmp/huanyu-examples/examples/*.sh "$SHELL_ROOT/"
cp /tmp/huanyu-examples/examples/*.py "$SHELL_ROOT/"
```

复制不会清理部署目录中历史遗留文件；部署时应以这 11 个文件的清单核对，整套更新。Python 包也要同步，不能只复制 shell 文件。

AQA 两节点准备并启动 Ray，在两个终端并行执行以下公共设置；主节点设 `VC_TASK_INDEX=0`，从节点设 `1`。将 IP 替换为当前作业的真实地址，列表顺序必须一致。

```bash
export VC_WORKER_HOSTS=172.16.7.181,172.16.2.73
export VC_WORKER_NUM=2
export VC_TASK_INDEX=0                    # 从节点改为 1
export HUANYU_JOB_MODE=aqa_unfreeze
source "$SHELL_ROOT/prepare_huanyu_env.sh" --start-ray
```

仅准备环境：去掉 `--start-ray`。之后每节点执行 `bash "$SHELL_ROOT/start_huanyu_ray.sh" --check`。用 `bash prepare_huanyu_env.sh` 而非 `source` 时，它打印 activation 路径，需在调用终端再 `source` 该文件。

STEM 使用 `TOTAL_NPUS=16` 或 `32`，并执行 `source "$SHELL_ROOT/prepare_huanyu_env.sh" --stem --start-ray`；16 卡时 host 列表只有本机、节点数为 1。冻结缓存准备模式为 `HUANYU_JOB_MODE=frozen_cache`，需要可读且匹配的 `HUANYU_EMBED_CACHE_DIR/manifest.json`；prepare 不生成 embedding cache。

站点环境如需额外配置，可设置 `HUANYU_ENV_FILE=/绝对路径/site_env.sh`；默认使用 prepare 内置的站点配置，不再需要 examples 下的 env_export 文件。安装和 CANN/ATB 初始化由 prepare 统一执行，自定义 site 文件无需再重复这些操作。选定可运行训练依赖的 `python3` 后再准备，prepare 沿用当前终端的 Python/代理设置。

准备时使用同一个 `python3` 安装 ext（`pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"`）以及 `mathruler zarr`。本机已有 Ray 时跳过包安装，避免运行中的 worker 与磁盘环境不一致；仍应用站点配置并执行相应检查。普通 job 调用准备后启动训练；如果已经准备并激活环境、启动 Ray，可直接运行 mix/AQA 或 STEM 的 `--train-only`，无需再次安装依赖。

Ray 配置：

| 参数 | 默认/含义 |
|---|---|
| `VC_TASK_INDEX` | 物理节点序号；head 为 0 |
| `VC_WORKER_HOSTS` | 按 head 在前排列的逗号分隔 IP/host |
| `RAY_PORT` / `RAY_DASHBOARD_PORT` | `6925` / `8960` |
| `RAY_CLUSTER_WAIT_TIMEOUT` / `RAY_CLUSTER_POLL_INTERVAL` | `900` / `5` 秒 |
| `HUANYU_JOB_PREFLIGHT` | 默认 `0`；设 `1` 启用 job 的额外检查。独立 Ray 的 `--check` 和 prepare 的 `--start-ray` 明确检查 |
| worker startup | `RAY_worker_maximum_startup_concurrency=32`、`RAY_prestart_worker_first_driver=1`、`RAY_enable_worker_prestart=0` |

每节点注册 `NPU:16`。head 检查所有节点，worker 检查本机注册；检查直接走 GCS，不要求 Dashboard API，也不创建训练 driver。复用运行中的 Ray 不会修改旧 raylet 的启动参数。独立 `bash start_huanyu_ray.sh` 不会向父 shell 导出变量；按输出设置 `RAY_ADDRESS=head_ip:6925`，避免 `auto` 指向另一个 session。

默认使用现有 `/vllm-workspace/vllm-omni` 安装，不把 checkout 插入 `PYTHONPATH`。源码来源预检仍保留，HF checkpoint、ext 包和 native backend 的版本必须匹配。

## 4. 训练命令：先 export，再启动 job

模型、数据、batch、吞吐和精度参数统一在 job 启动前 `export`。多节点时每个节点使用同一份参数，只有 `VC_TASK_INDEX` 不同。以下示例需要先按第3节设置 `PROJECT_ROOT`、`SHELL_ROOT` 和实际节点地址；不需要手改 shell 文件里的默认值。

### 已有环境与 Ray：集中配置的单文件入口

需要直接在脚本内查看或编辑参数时使用 `run_huanyu_train.sh`。它按路径、资源、数据、媒体、精度、通信、诊断、Actor、FSDP、rollout、奖励和训练调度分组，每项有中文行内注释；不再经过其他训练 wrapper，也不会准备环境、安装依赖或启停 Ray。默认通用 `data/train.parquet` / `validation.parquet`、2748、Actor16 + 独立 GenRM16、TP4、BS32/mini32/N4、prompt3072/response2048、full 精度、eager、冻结音频/视觉、在线编码且不使用 embedding 缓存。视频参数只作为数据行未指定时的回退值，沿用 OmniMix 的 fps=1、max_frames=8。

新入口显式设置 microbatch=1 和 `huanyu_sync_each_microbatch=true`，以在多 microbatch 时同步分片梯度；Actor 参数精度仍默认 FP32，Ref/rollout/GenRM 默认 BF16。全部配置以本文件为准；它列出了该入口使用的覆盖项，没有展开框架的每一个内部默认字段。

仅在 head 执行：

```bash
bash "$SHELL_ROOT/run_huanyu_train.sh"
```

可修改文件中的默认值、导出同名环境变量，或在命令末尾追加 Hydra 覆盖。关联长度、资源及精度开关建议通过脚本或环境变量设置，让派生值同步计算；`RAY_ADDRESS` 未设置时使用 `auto`，找不到已有集群会报错。默认日志位于 `$PROJECT_ROOT/runs/huanyu_<时间>_<PID>/console.log`。

```bash
# 只查看最终启动参数，不加载模型、不连接 Ray、不创建输出目录。
bash "$SHELL_ROOT/run_huanyu_train.sh" --dry-run

# 两个训练 step，演示环境变量和末尾 Hydra 参数覆盖。
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 TOTAL_TRAINING_STEPS=2 \
  bash "$SHELL_ROOT/run_huanyu_train.sh" actor_rollout_ref.rollout.max_num_seqs=16
```

`HUANYU_REWARD_MODE=rule_based` 会关闭独立 GenRM 池，按数据行 `data_source` 分派已有规则；数据集必须有对应 scorer。使用 forward profiler 仍需先按 [forward profiling 说明](docs/forward_profile.md) 安装钩子，再设置 `HUANYU_FORWARD_PROFILE_DIR`。

GenRM 模式同时显式加载 `verl_omni_ext.models.qwen3_5_moe`（verl 适配器）和 `verl_omni_ext.models.qwen3_5_moe.vllm_omni`（原生 AR pipeline），并将模块列表写入本次 Ray runtime_env。原生 vLLM-Omni 必须支持当前安装约定的 `VLLM_OMNI_EXTERNAL_MODULES` 加载入口。若仍报 `Qwen3_5MoeForConditionalGeneration does not support request-level batching`，应检查报错进程的部署配置是否使用 `qwen3_5_moe_thinker_only`、stage 是否为 `llm_ar`，以及是否存在插件导入失败；降低 `max_num_seqs` 不能验证 AR pipeline 已正确选择。

### AQA：两节点各执行，Actor16 + GenRM16

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export HUANYU_REWARD_MODE=RM
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export TRAIN_FILE=/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet
export VAL_FILE=/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048
export AQA_DYNAMIC_BSZ=true MAX_TOKEN_LEN_PER_GPU=8192
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 ROLLOUT_MAX_NUM_SEQS=16
export ROLLOUT_GPU_MEM_UTIL=0.6 RM_GPU_MEM_UTIL=0.7
export TOTAL_TRAINING_STEPS=2 TRAINER_LOGGER='[console]' WANDB_MODE=disabled
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

只由 head 提交训练。AQA 默认 Actor `16×1`、rollout TP4（4 个副本）、GenRM `16×1`，合计需要32卡；默认 encoder/projector 都训练，视觉冻结。已有环境/Ray 时，可在 head 沿用上述 export 后直接执行 `bash "$SHELL_ROOT/run_huanyu_aqa_audio_train.sh"`，避免重复准备。

16卡、无GenRM实验使用已有Ray的 AQA直接入口，并 `export HUANYU_REWARD_MODE=rule_based`；奖励改为字符二元组 Dice 规则，训练目标随之改变。完整音频模式见[AQA 说明](AQA_AUDIO_TRAINING_ZH.md)。

### STEM：16 卡、短配置

单节点设置 `VC_WORKER_NUM=1`、`VC_TASK_INDEX=0`，`VC_WORKER_HOSTS` 只包含本机地址，再执行：

```bash
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export TRAIN_FILE=/opt/huawei/dataset/infra_data/data/rl_stem_data/train.parquet
export VAL_FILE=/opt/huawei/dataset/infra_data/data/rl_stem_data/val.parquet
export TOTAL_NPUS=16 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4
export HUANYU_REWARD_MODE=rule_based
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=2048 STEM_IGNORE_EOS=false
export STEM_DYNAMIC_BSZ=true MAX_TOKEN_LEN_PER_GPU=8192
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 ROLLOUT_MAX_NUM_SEQS=8
export TOTAL_TRAINING_STEPS=2 TRAINER_LOGGER='[console]' WANDB_MODE=disabled
bash "$SHELL_ROOT/run_huanyu_stem_job.sh"
```

32卡使用两节点地址和节点数，把 `TOTAL_NPUS` 改为32，例如 BS8/mini8/N4，保证每次 mini 的轨迹数能被 Actor world size 整除。每节点运行job；已有集群时只需在head执行同样export后使用 `bash "$SHELL_ROOT/run_huanyu_stem_job.sh" --train-only`，它跳过环境准备和Ray启动。

STEM job 默认全部卡做 Actor、规则奖励、TP8。**当前代码训练 TP=1、SP=1，所以规则模式下 32 张 Actor 卡的全局 DP 仍是 32；`fsdp_size=16` 仅限制分片组大小，不能把全局 DP 变成 16。** 直接 `run_huanyu_stem.sh` 默认是 Actor16 + GenRM16、TP4、BS4/mini4/N4、5K context；它与 scaled job 默认值不同。选择规模配置时优先使用 `run_huanyu_stem_job.sh`。

STEM job 也接受 `export HUANYU_REWARD_MODE=RM`：32 张物理卡默认划分为 Actor16 + RM16；默认从物理节点中预留一台16卡节点给RM，其余节点做Actor。16张物理卡的scaled job不能按这种布局同时运行Actor和RM，会在启动前报错。切换奖励模式时重新设置或清理旧的 `NUM_NPUS`、`NNODES`、`RM_N_GPUS_PER_NODE`、`RM_NNODES`，避免沿用上次显式导出的资源池。

### 通用多模态数据、冻结音频与视觉、GenRM

使用 common job 时，在两节点各执行以下设置后启动；缓存目录必须在每个节点可读：

```bash
export HUANYU_JOB_MODE=frozen_cache
export HUANYU_REWARD_MODE=RM
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export TRAIN_FILE=/opt/huawei/dataset/infra_data/data/train.parquet
export VAL_FILE=/opt/huawei/dataset/infra_data/data/validation.parquet
export HUANYU_FREEZE_AUDIO=true HUANYU_USE_PRECOMPUTE=1
export HUANYU_EMBED_CACHE_DIR=/opt/huawei/dataset/infra_data/data/embed_cache_2748_current
export TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

缓存目录需提前用模型包中的 `precompute.py` / `precompute_parallel.py` 生成。common job 会显式向 Ray worker 转发缓存目录。已有环境/Ray时，head也可以沿用上述export直接运行后端；必须同时转发缓存目录，不能假定旧raylet继承了当前终端的新变量：

```bash
bash "$SHELL_ROOT/run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_EMBED_CACHE_DIR=\"${HUANYU_EMBED_CACHE_DIR}\""
```

无需缓存的冻结在线模式使用后端直接入口并 `export HUANYU_USE_PRECOMPUTE=0`；视觉/音频仍会在线编码。common job的 `frozen_cache` 模式本身始终启用缓存。

OC-RL 4B 数据是纯图文数据，不能给要求每条样本都有音频的 AQA 入口；使用通用 GenRM 入口并指定实际 parquet，见[数据 Wiki](OC_RL_4B_DATASET_WIKI_ZH.md)。

### 替代已删除的 35B frozen-cache job

原 `run_huanyu_35b_frozen_cache_32card_job.sh` 的入口已删除。下面用保留的common job显式恢复其模型、数据、缓存、资源池、batch、长度和动态批处理配置。两个节点都执行，沿用当前作业的 `VC_WORKER_HOSTS` / `VC_WORKER_NUM=2`，并分别设置 `VC_TASK_INDEX=0` / `1`：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export SHELL_ROOT="$PROJECT_ROOT/examples/qwen3_6_huanyu"
export HUANYU_JOB_MODE=frozen_cache HUANYU_REWARD_MODE=RM
export HUANYU_CLUSTER_NODES=2 VC_WORKER_NUM=2
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export TRAIN_FILE=/opt/huawei/dataset/infra_data/data/train.parquet
export VAL_FILE=/opt/huawei/dataset/infra_data/data/validation.parquet
export RM_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B
export HUANYU_EMBED_CACHE_DIR=/opt/huawei/dataset/infra_data/data/embed_cache_2748_huanyu_35b
export HUANYU_USE_PRECOMPUTE=1 MINICPMO_USE_PRECOMPUTE=0
export HUANYU_FREEZE_AUDIO=true
export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1
export TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=3072
export ROLLOUT_PROMPT_LENGTH=2048 MINICPMO_MM_PROMPT_BUDGET=2048
export ROLLOUT_MAX_MODEL_LEN=5248 ROLLOUT_MAX_NUM_BATCHED_TOKENS=5248
export MAX_TOKEN_LEN_PER_GPU=5248
export ROLLOUT_MAX_NUM_SEQS=64 ROLLOUT_NUM_WORKERS=4 UPDATE_WEIGHTS_BUCKET_MB=1024
export ACTOR_DYNAMIC_BSZ=true ROLLOUT_LOG_PROB_DYNAMIC_BSZ=true REF_LOG_PROB_DYNAMIC_BSZ=true
export ACTOR_MODEL_DTYPE=fp32 REF_MODEL_DTYPE=bfloat16
export ACTOR_PARAM_OFFLOAD=true ACTOR_OPTIMIZER_OFFLOAD=false REF_PARAM_OFFLOAD=true
export ACTOR_FORWARD_PREFETCH=false REF_FORWARD_PREFETCH=true
export ACTOR_OFFLOAD_POLICY=false ACTIVATION_OFFLOAD=false
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export FILTER_OVERLONG=false
export ROLLOUT_GPU_MEM_UTIL=0.6 RM_GPU_MEM_UTIL=0.7
export ROLLOUT_FREE_CACHE_ENGINE=true HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export TOTAL_TRAINING_STEPS=-1
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

这里的 `35b` 指既有实验配置与数据命名；实际Actor checkpoint由 `MODEL_PATH` 决定。旧 `HUANYU_35B_MODEL_PATH` / `HUANYU_35B_TRAIN_FILE` / `HUANYU_35B_VAL_FILE` / `HUANYU_35B_CACHE_DIR` 专用wrapper变量不再作为入口，分别改为通用的 `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE` / `HUANYU_EMBED_CACHE_DIR`。缓存目录必须包含与当前模型、数据和预处理版本匹配的manifest；该命令不生成或改写缓存。需要两步验证时另加 `export TOTAL_TRAINING_STEPS=2`，正式运行可设 `-1`。

## 5. 重要参数与覆盖方式

### 奖励模式与资源池

128/256 卡 STEM 切到 RM 时，默认分别为 Actor112/240 + RM16。原默认 batch 不一定整除新的 Actor DP；可对应设置 `TRAIN_BATCH_SIZE=PPO_MINI_BATCH_SIZE=112` 或 `240`，并保留 `ROLLOUT_N=16`。入口会检查 `mini_batch * N` 是否整除实际 Actor DP。

统一通过 job 前的环境变量选择奖励，不需要修改脚本：

```bash
export HUANYU_REWARD_MODE=RM          # 独立 GenRM 奖励模型
# 或
export HUANYU_REWARD_MODE=rule_based  # 规则评分，不启动 RM 资源池
```

`ruled_based`、`rule` 是 `rule_based` 的兼容写法，`genrm` 是 `RM` 的兼容写法。显式 `HUANYU_REWARD_MODE` 优先于旧的 `RM_ENABLE` 和 `STEM_REWARD_MODE`；未指定统一变量时保留旧变量的兼容行为。最终会规范化 `RM_ENABLE`，并与实际奖励配置和资源池保持一致。环境准备和 job 的站点环境加载会保留调用方的奖励选择。

| 入口/任务 | 未显式选模式时的默认 | `rule_based` 的评分方式 | 资源说明 |
|---|---|---|---|
| common job `frozen_cache` / OmniMix 后端 | RM | `naive` manager 按 parquet 的 `data_source` 调用已有规则 scorer | 默认Actor16；RM模式另用16卡，规则模式不创建RM池 |
| common job `aqa_unfreeze` / AQA直接入口 | RM | `verl_omni_ext.models.qwen3_6_huanyu.rule_based_reward`，字符bigram Dice | 默认Actor16；RM模式另用16卡 |
| `run_huanyu_stem_job.sh` | rule_based | STEM预处理的数据路由到 `math_dapo` | 规则模式默认全部物理卡做Actor；RM模式默认预留16卡给RM |
| `run_huanyu_stem.sh`直接入口 | RM | `math_dapo` | 默认Actor16；RM模式另用16卡 |

规则模式并不意味着任意数据集都能用同一种评分器。通用OmniMix模式要求实际 `data_source` 被已有规则dispatcher支持；自定义VL/开放问答数据通常仍需明确的评分实现或使用RM。AQA的Dice分数和GenRM语义判分不是同一个训练目标。

common job 的物理集群仍为两台16卡机器。切到规则模式后，默认Actor16保持不变，释放的另16卡不会自动变成Actor；要让32卡都做Actor，可同时 `export NUM_NPUS=16 NNODES=2`，并按32卡重新校验BS/mini/N，例如AQA使用BS8/mini8/N4。STEM scaled job通过 `TOTAL_NPUS` 管理物理规模，不能把物理总卡数误当成Actor池大小。

RM模式的重要参数也通过export透传：`RM_MODEL_PATH`、`RM_N_GPUS_PER_NODE`、`RM_NNODES`、`RM_TP`、`RM_GPU_MEM_UTIL`、`RM_PROMPT_LENGTH`、`RM_RESPONSE_LENGTH`、`RM_MAX_MODEL_LEN`、`RM_MAX_NUM_BATCHED_TOKENS`、`RM_MAX_NUM_SEQS`。改变奖励模式或池大小后，以启动摘要中的reward、Actor/RM卡数和最终Hydra的 `reward.reward_model.enable` / `enable_resource_pool` 为准。

### 模型、数据、长度和 batch

| 用途 | 推荐设置 | 说明 |
|---|---|---|
| Actor checkpoint | `MODEL_PATH` | 默认 `/opt/huawei/dataset/model_gy/downloaded_models/2748` |
| GenRM checkpoint | `RM_MODEL_PATH` | 默认 `/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B` |
| 通用训练/验证数据 | `TRAIN_FILE` / `VAL_FILE` | 默认 `infra_data/data/train.parquet` / `validation.parquet` |
| AQA 数据 | `TRAIN_FILE` / `VAL_FILE` | 显式通用路径优先；未设置时才使用 `HUANYU_AQA_TRAIN_FILE` / `HUANYU_AQA_VAL_FILE`，再回落到 `infra_data/data/aqa_dataset/` |
| STEM 数据 | `TRAIN_FILE` / `VAL_FILE` | 显式通用路径优先；未设置时使用 `STEM_TRAIN_FILE` / `STEM_VAL_FILE` 或 `STEM_DATA_DIR`；默认 `rl_stem_data/train.parquet` / `val.parquet`，无默认val时回落到 `validation.parquet` |
| 全局 prompt batch | `TRAIN_BATCH_SIZE` | 每 step 轨迹数约为 `BS × ROLLOUT_N` |
| PPO mini batch | `PPO_MINI_BATCH_SIZE` | 按重复采样前 prompt 数计数；mini 轨迹数为 `mini × N` |
| 每 prompt 回答数 | `ROLLOUT_N` | GRPO 需有效组内比较；AQA 要求 `N >= 2` |
| prompt/response上限 | `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | AQA默认4096/2048；STEM也支持，显式通用值优先于STEM别名，并按两者之和重算总长度 |
| STEM prompt/总长度回落值 | `STEM_PROMPT_LENGTH` / `STEM_CONTEXT_LENGTH` | job默认2048/12288；未显式指定response时，response=总长度−prompt；job默认slack=0 |
| 推理引擎context | `ROLLOUT_MAX_MODEL_LEN` | 可独立export，必须覆盖prompt+response；默认AQA为两者之和+128，STEM job为两者之和 |
| 长度预筛 | `FILTER_OVERLONG`（通用） | GenRM通用默认false；AQA强制false，STEM强制true |
| EOS | STEM 用 `STEM_IGNORE_EOS=false` | false 允许 EOS 提前结束；true 可能显著增加生成 token 数 |
| 总 step | `TOTAL_TRAINING_STEPS` | `2`表示两步短跑；`-1`或`null`表示按epoch运行 |

AQA 默认 BS4/mini4/N4；STEM scaled job 默认 BS128/mini128/N16，直接使用默认配置可能很重。通用 GenRM 入口默认 BS32/mini32/N4、prompt3072/response2048、动态批处理开启。基本约束：`BS % mini == 0`，`mini × N` 能被 Actor 卡数整除。AQA 还要求 mini 能被 rollout 副本数整除。静态Actor批处理时，每rank轨迹数`mini × N / Actor卡数`还需能被`PPO_MICRO_BATCH_SIZE_PER_GPU`整除。输入路径、batch、TP 等会参与 shell 的前置校验，应在job前export，不要只修改下游Hydra值。

### 并行、吞吐、显存

| 用途 | 配置 | 区别/默认 |
|---|---|---|
| Actor 卡数 | `NUM_NPUS × NNODES` | 分别为每池节点卡数与池节点数，不是机器上的总 NPU 资源 |
| STEM 总卡数 | `TOTAL_NPUS` | 物理总卡数；规则模式默认全部Actor，RM模式默认预留一台16卡节点给奖励池；Actor分片组上限16 |
| rollout TP | `ROLLOUT_TP` | AQA/GenRM 通用入口默认4；STEM job默认8；副本数=Actor卡数/TP |
| Actor 显存比例 | `ROLLOUT_GPU_MEM_UTIL` | 默认 `0.6`，映射 `actor_rollout_ref.rollout.gpu_memory_utilization` |
| GenRM 显存比例 | `RM_GPU_MEM_UTIL` | 独立预算，默认 `0.7` |
| 生成并发 | `ROLLOUT_MAX_NUM_SEQS` | 映射 `actor_rollout_ref.rollout.max_num_seqs`；AQA/GenRM默认64，STEM job默认8 |
| 调度 token budget | `ROLLOUT_MAX_NUM_BATCHED_TOKENS` | 映射 `actor_rollout_ref.rollout.max_num_batched_tokens`；默认随引擎长度，不等于context或BS |
| AQA三阶段动态批处理默认 | `AQA_DYNAMIC_BSZ` | 默认false，统一给下列三个阶段提供默认值 |
| STEM三阶段动态批处理默认 | `STEM_DYNAMIC_BSZ` | job默认false、直接STEM默认true；可在job前export |
| Actor动态批处理 | `ACTOR_DYNAMIC_BSZ` | 映射 `actor_rollout_ref.actor.use_dynamic_bsz`；显式设置优先于上述模式默认 |
| Actor old log-prob动态批处理 | `ROLLOUT_LOG_PROB_DYNAMIC_BSZ` | 映射 `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz`；不是生成引擎动态调度 |
| Ref log-prob动态批处理 | `REF_LOG_PROB_DYNAMIC_BSZ` | 映射 `actor_rollout_ref.ref.log_prob_use_dynamic_bsz`；可单独设置 |
| 静态Actor microbatch | `PPO_MICRO_BATCH_SIZE_PER_GPU` | 映射 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`；AQA/STEM默认1，静态模式用于切分每rank的mini轨迹 |
| 静态log-prob microbatch | `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` | 同时映射rollout/ref的`log_prob_micro_batch_size_per_gpu`；默认1 |
| 动态批token上限 | `MAX_TOKEN_LEN_PER_GPU` | 映射actor PPO、rollout/ref log-prob三个 `*_max_token_len_per_gpu` |
| 权重同步bucket | `UPDATE_WEIGHTS_BUCKET_MB` | 映射 `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`；单位MiB；base默认4096、GenRM/AQA默认1024、STEM job默认2048 |
| AgentLoopWorker数 | `ROLLOUT_NUM_WORKERS` | 映射 `actor_rollout_ref.rollout.agent.num_workers`；AQA默认等于rollout副本数，STEM job默认128 |

AQA在job前统一开启动态批处理，并设置生成调度预算：

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export AQA_DYNAMIC_BSZ=true MAX_TOKEN_LEN_PER_GPU=8192
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 ROLLOUT_MAX_NUM_SEQS=16
export UPDATE_WEIGHTS_BUCKET_MB=2048
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

如仅一个阶段需要不同设置，再export `ACTOR_DYNAMIC_BSZ`、`ROLLOUT_LOG_PROB_DYNAMIC_BSZ` 或 `REF_LOG_PROB_DYNAMIC_BSZ`。动态模式主要按`MAX_TOKEN_LEN_PER_GPU`划分microbatch；静态模式使用相应microbatch条数。`ROLLOUT_MAX_MODEL_LEN`、`ROLLOUT_MAX_NUM_BATCHED_TOKENS`和`MAX_TOKEN_LEN_PER_GPU`分别控制context、推理调度和训练/log-prob预算，可独立export。STEM job默认 `STEM_SYNC_EACH_MICROBATCH=true`，Actor参数/优化器offload和CPU offload policy均开启；直接STEM默认按引擎梯度同步。

AgentLoopWorker 与 TP worker 不是一回事：AQA 默认 AgentLoop 数等于 rollout 副本数；STEM job 通过 `ROLLOUT_NUM_WORKERS` 设置，默认128。增加 AgentLoop 数不增加 NPU 容量，还会增加初始化和 CPU 开销。

### 精度、图、音频和 offload

| 配置 | 当前约定 |
|---|---|
| `HUANYU_PRECISION_PROFILE=full` | 默认；完整 reference 策略会将必需开关重置为1，单独 export 0 不能覆盖 full |
| `HUANYU_PRECISION_PROFILE=custom` | AQA/STEM/通用入口均可用于显式数值/性能对照，保留单独开关；不等同于所有精度开关关闭 |
| `HUANYU_GRAPH_MODE=0` | 各入口默认eager；可export为1，但必须同时选择custom，full仍拒绝graph=1 |
| `HUANYU_FREEZE_AUDIO` | 旧整体冻结开关；显式 encoder/projector 开关优先，详见 AQA 文档 |
| `HUANYU_TRAIN_AUDIO_ENCODER` / `HUANYU_TRAIN_AUDIO_PROJECTOR` | 可分别控制；只要一个训练，就不能使用冻结 embedding cache |
| `HUANYU_USE_PRECOMPUTE` | Huanyu 独立缓存；不要用旧 `MINICPMO_USE_PRECOMPUTE=1` 代替 |
| AQA offload默认 | `param_offload=true`、`optimizer_offload=false`、`offload_policy=false`，关闭前向/反向prefetch和activation offload；可用下表支持的环境变量调整 |

### 优化器、offload与训练周期：环境变量映射

| job前export的变量 | 底层配置 | 默认/说明 |
|---|---|---|
| `ACTOR_LR` | `actor_rollout_ref.actor.optim.lr` | `1e-6` |
| `ACTOR_WEIGHT_DECAY` | `actor_rollout_ref.actor.optim.weight_decay` | `0.1` |
| `ACTOR_PARAM_OFFLOAD` | `actor_rollout_ref.actor.fsdp_config.param_offload` | 默认true |
| `ACTOR_OPTIMIZER_OFFLOAD` | `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | AQA默认false，STEM job默认true |
| `ACTOR_OFFLOAD_POLICY` | `actor_rollout_ref.actor.fsdp_config.offload_policy` | AQA默认false，STEM job默认true；与param_offload不同 |
| `REF_PARAM_OFFLOAD` | `actor_rollout_ref.ref.fsdp_config.param_offload` | 默认true |
| `ACTOR_FORWARD_PREFETCH` / `REF_FORWARD_PREFETCH` | actor/ref `fsdp_config.forward_prefetch` | AQA/STEM job默认false |
| `ACTIVATION_OFFLOAD` | `actor_rollout_ref.model.enable_activation_offload` | 默认false |
| `TOTAL_EPOCHS` | `trainer.total_epochs` | 默认10；显式总step还会限制训练步数 |
| `TOTAL_TRAINING_STEPS` | `trainer.total_training_steps` | 短跑设2；完整epoch用-1或null |
| `SAVE_FREQ` / `TEST_FREQ` | `trainer.save_freq` / `trainer.test_freq` | 默认50/10 |
| `TRAINER_LOGGER` | `trainer.logger` | 默认`[console]`；shell中用单引号包住列表 |

AQA 关闭两个 GDN reference 的显式实验命令：

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export HUANYU_PRECISION_PROFILE=custom HUANYU_GRAPH_MODE=0
export MINICPMO_GDN_PREFILL_REFERENCE=0 MINICPMO_GDN_FP32_REFERENCE=0
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

这会改变数值路径，应重新看 rollout/actor 对齐；不能把 full 的一致性结论直接沿用。设置图模式也只作用于推理端，不等于开启 FSDP 训练 `torch.compile`。

### 参数优先级与生效核对

1. **先export，再启动job。** 表中环境变量会传给后续脚本；job准备站点环境时应保留调用方显式值。未设置才采用入口默认值，旧终端遗留的export也算显式值。
2. 模型用`MODEL_PATH`；数据统一用`TRAIN_FILE`/`VAL_FILE`。专用AQA/STEM路径别名仅作为未设置通用路径时的回落。STEM的`MAX_PROMPT_LENGTH`/`MAX_RESPONSE_LENGTH`同样优先于专用长度别名；切换任务时应重新export实际文件和长度，防止沿用上次任务设置。
3. 奖励模式显式 `HUANYU_REWARD_MODE` 优先于旧 `RM_ENABLE` / `STEM_REWARD_MODE`。三阶段动态开关优先取各自显式变量，再取AQA/STEM模式默认。full precision、Huanyu模型适配、SP/资源合法性和音频缓存契约仍是强制约束；不能用矛盾的环境值绕过。
4. 末尾Hydra参数保留给临时高级调试，部分键会被下游约束再次覆盖；日常重要参数以本页export接口为准。实际生效值以最终Hydra配置为准。
5. 最终后端统一锁定Huanyu adapter/dataset/pipeline，并将有效precision/graph/runtime环境传给新Ray worker。运行中的训练进程不会自动应用后来在另一个终端export的值。

启动后核对 `[huanyu-aqa]` / `[huanyu-stem]` 的模型资源、BS/N、长度，`[huanyu-precision]` 的profile/实际TP，以及 Hydra 最终配置；不要只看中间 `set -x` 的某一次赋值。

## 6. 日志、检查点和诊断

每次训练创建唯一 `$PROJECT_ROOT/runs/huanyu_<时间>_<随机后缀>/`。`[huanyu-output] console=...` 给出本次 `console.log`；嵌套入口沿用同一运行目录，不重复tee。`HUANYU_RUN_DIR`、`HUANYU_CONSOLE_LOG`、`GENRM_DUMP_DIR`由脚本生成，不作为用户自定义输出入口；可通过下表的checkpoint和回答目录变量指定产物位置。prepare单独写`runs/environment/prepare_*/prepare.log`和`activate.sh`。

| 用途 | 变量/参数 |
|---|---|
| 检查点目录 | `DEFAULT_LOCAL_DIR` 或 `trainer.default_local_dir` |
| rollout/验证回答 | `ROLLOUT_DATA_DIR` / `VALIDATION_DATA_DIR` 或 `trainer.rollout_data_dir` / `trainer.validation_data_dir` |
| 保存/验证频率 | `SAVE_FREQ`、`TEST_FREQ` 或 `trainer.save_freq`、`trainer.test_freq`，以具体入口最终约束为准 |
| 关闭 wandb | `export WANDB_MODE=disabled TRAINER_LOGGER='[console]'`；offline仍是wandb离线记录 |
| gen 分段计时 | `HUANYU_GEN_TIMING=1`，见[计时说明](docs/gen_timing.md) |
| forward 内部 NPU profiler | 见[forward profile](docs/forward_profile.md)，包含GDN/Attention/MoE/HCCL分析限制 |

音频短尾的最新处理策略、旧缓存失效与原生验证命令见[短尾修复说明](docs/audio_short_tails.md)。短尾、视频配对等语义变化后要使用新 embedding 缓存目录；不能手工改 manifest 复用旧缓存。外部审计失败缓存也应更新命名空间并重筛。

## 发布包校验

部署前可在克隆目录执行 `sha256sum -c examples/SHA256SUMS`。本包依赖服务器已安装的 verl、verl-omni、vllm-omni 和 Huanyu ext 适配代码。音频短尾和视频帧修复对应 [profile 适配代码](https://github.com/ZiqiGuan892/ext/tree/4b003b26321b7d5e35d14167666515a1d760a5b7/qwen3_6_huanyu)。
