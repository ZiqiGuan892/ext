# Huanyu examples 启动与参数说明

按 2026-10-27 当前脚本整理，覆盖本目录所有 `.sh`、`.py`。下文默认值以未继承其他环境配置为前提；专用入口固定的参数另行说明。

**冻结 Huanyu、读取 embedding cache：使用 `run_huanyu_32card_job.sh`。AQA 在线训练：使用对应的 `aqa` 入口。** 两类入口都继续训练语言模型；冻结音频不等于冻结整个 Actor。

## 1. 先选最外层入口

一键入口负责环境、vLLM-Omni 同步和 Ray；普通入口要求这些已经准备好，仅在主节点启动一次。

| 场景 | 最外层脚本 | 运行位置 | 主要默认配置 |
|---|---|---|---|
| 仅准备环境，之后手动启动Ray与训练 | `source prepare_huanyu_env.sh` | 每个节点 | 复用一键准备流程；默认frozen_cache，不连接/启动Ray，不提交训练 |
| 冻结 Huanyu、OmniMix/通用多模态数据、缓存训练 | `run_huanyu_32card_job.sh` | 两节点各执行一次 | `frozen_cache`，encoder/projector 都冻结，Actor16 + RM16，prompt3072 + response2048 |
| 冻结缓存训练，16K | `run_huanyu_32card_16k_job.sh` | 两节点 | prompt14336 + response2048，SDPA，默认16+16 |
| 冻结缓存训练，对照 Whisper 启动参数 | `run_huanyu_32card_whisper_aligned_job.sh` | 两节点 | 保留 Huanyu 接口；Actor BF16、rollout TP2、RM TP4，5K |
| AQA 在线训练，encoder/projector 可分别开关 | `run_huanyu_aqa_unfreeze_32card_job.sh` | 两节点 | 默认二者都训练，关闭缓存，16+16，5K |
| AQA 在线训练，强制完整精度修复 | `run_huanyu_aqa_precision_32card_job.sh` | 两节点 | `full`、graph=0、16+16、5K |
| AQA 在线训练，16K、逐 micro-batch 同步 | `run_huanyu_aqa_16k_sync_each_job.sh` | 两节点 | 固定16+16、TP4、静态 micro=1、逐 micro-batch 梯度同步 |
| 已准备环境和 Ray，冻结/缓存训练 | `run_huanyu.sh` | 仅主节点 | 需提前配置冻结与缓存环境；复用通用训练链 |
| 已准备环境和 Ray，AQA 自定义长度/参数 | `run_huanyu_aqa_audio_train.sh` | 仅主节点 | 在线音频，默认4096+2048，16+16，静态 micro=1 |
| 已准备环境和 Ray，AQA 完整精度修复 | `run_huanyu_aqa_precision.sh` | 仅主节点 | 强制 `full`、graph=0，默认3072+2048；跳过安装 |
| 已准备环境和 Ray，AQA 16K 同步模式 | `run_huanyu_aqa_16k_sync_each.sh` | 仅主节点 | 与对应一键入口相同的训练配置；跳过安装 |
| 单机16卡 AQA，关闭模型奖励 | `run_huanyu_aqa_audio_train_16card.sh` | 仅主节点 | Actor16、RM0，规则奖励；默认4096+2048 |

当前所有训练链通过 `precision_mode.sh` 默认选择 `full`。名称不带 `precision` 并不表示关闭精度修复；AQA precision 和 sync_each 专用入口还会强制 `full` 和 graph=0。

旧的 AQA `run_huanyu_aqa_16k_24_8*.sh` 已改名为 `run_huanyu_aqa_16k_sync_each*.sh`，当前配置是 **16+16**。冻结版 `run_huanyu_32card_16k_job.sh` 仍保留 `HUANYU_16K_POOL_LAYOUT=24+8` 可选配置；选项存在不代表此前跨节点 HCCL 问题已经解决。

## 2. 部署目录与默认路径

源码中的本目录位于 `verl_omni_ext/models/qwen3_6_huanyu/examples/`；服务器执行时，脚本从 `$PROJECT_ROOT/examples/qwen3_6_huanyu/` 查找同伴文件。更新时应同时部署 Huanyu Python 包和全部脚本，避免只更新最外层文件。`env_export_huanyu.sh` 也要与 `precision_mode.sh`、`graph_mode.sh` 放在同一目录。

```bash
# 若实际项目在 infra_data/code 下，使用实际路径覆盖这里。
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new
mkdir -p "$PROJECT_ROOT/examples/qwen3_6_huanyu"
cp "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/examples/"*.sh \
   "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/examples/"*.py \
   "$PROJECT_ROOT/examples/qwen3_6_huanyu/"
```

| 变量/用途 | 当前默认值 |
|---|---|
| `PROJECT_ROOT` | `/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new` |
| `MODEL_PATH` | `/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens` |
| `RM_MODEL_PATH` | `/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B` |
| 通用 `TRAIN_FILE` / `VAL_FILE` | `/opt/huawei/dataset/infra_data/data/train.parquet` / 同目录 `validation.parquet` |
| AQA 数据 | `/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet` / 同目录 `validation.parquet` |
| `HUANYU_EMBED_CACHE_DIR`，冻结一键模式 | `/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128` |
| `HUANYU_VLLM_SOURCE`，一键同步源 | `/opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit` |
| vLLM-Omni 同步目标及运行目录 | `/vllm-workspace/vllm-omni` |

“35B”“4B”“sampleall”不会自动选择另一份数据，实际内容由 parquet 路径决定。冻结缓存模式还要求缓存与所选数据、模型和预处理匹配；两节点均需能读取 `manifest.json` 和缓存内容。

## 3. 完整启动链路

### 3.1 一键入口的共同准备过程

所有 `*_job.sh` 最终进入 `run_huanyu_32card_job.sh`，依次：

1. 解析路径、卡池和音频开关，检查节点编号、地址及 jemalloc。
2. 配置精度策略，创建节点日志并持有本地 launcher 锁。
3. 调用 `job_bootstrap.py sync` 同步 vLLM-Omni。源码摘要相同则跳过；不同时先复制校验、备份旧目录再替换。Ray 已运行且源码不同时退出，避免覆盖正在使用的代码。
4. 无本地 Ray 时执行项目 editable 安装；随后 source `env_export_huanyu.sh`，恢复入口保存的路径、卡池、音频及精度配置。
5. 检查冻结缓存（仅 `frozen_cache`）、原生导入及设备；启动或复用 Ray。
6. `VC_TASK_INDEX=0` 等待两个16卡节点就绪后提交训练；`=1` 只加入 Ray 并保持运行。音频、缓存和精度配置传入 Ray `runtime_env`。

当前同步与导入目标仍固定为 `/vllm-workspace/vllm-omni`。仅在外部 `export PYTHONPATH=新源目录` 不会跳过同步，下游还会重新排列导入路径；直接从源目录运行的方案尚未接入这些脚本。

### 3.2 冻结缓存训练

```text
run_huanyu_32card_16k_job.sh 或 run_huanyu_32card_whisper_aligned_job.sh（可选配置层）
  → run_huanyu_32card_job.sh（mode=frozen_cache；也可直接调用）
  → run_huanyu.sh
  → run_qwen36_huanyu_32card.sh
  → run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
  → run_qwen36_huanyu_gspo_npu.sh
  → python3 -m verl_omni.trainer.main_omni
```

普通入口从 `run_huanyu.sh` 开始；兼容入口 `run_32card.sh` 也转入它。`run_huanyu.sh` 选择 Huanyu adapter、dataset、pipeline，并关闭 remove-padding；GenRM 包装层配置奖励模型；最底层脚本检查并拼装最终命令。中间层保留的 Whisper 默认参数由外层 Huanyu 参数覆盖，不宜绕过外层直接启动中间脚本。

### 3.3 AQA 在线训练

```text
run_huanyu_aqa_unfreeze_32card_job.sh
  → run_huanyu_32card_job.sh（mode=aqa_unfreeze）
  → run_huanyu_aqa_audio_train.sh

run_huanyu_aqa_precision_32card_job.sh
  → run_huanyu_32card_job.sh（mode=aqa_precision）
  → run_huanyu_aqa_precision.sh
  → run_huanyu_aqa_audio_train.sh

run_huanyu_aqa_16k_sync_each_job.sh
  → run_huanyu_32card_job.sh（mode=aqa_16k_sync_each）
  → run_huanyu_aqa_16k_sync_each.sh
  → run_huanyu_aqa_audio_train.sh
```

到达共用 AQA 入口后继续：

```text
run_huanyu_aqa_audio_train.sh
  ├─ audio_train_mode.sh / precision_mode.sh / run_output.sh
  ├─ check_aqa_audio_train.py：检查训练和验证集
  ├─ check_native_runtime.py：检查实际导入的原生运行时
  └─ RM_ENABLE=1
       → run_qwen36_huanyu_32card.sh
       → run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
       → run_qwen36_huanyu_gspo_npu.sh
       → python3 -m verl_omni.trainer.main_omni
```

AQA 入口限定音频数据，在线编码，关闭 embedding cache；两个音频组件都冻结也不会自动切换为缓存读取。它强制静态 batching、micro-batch=1、FSDP2、sequence parallel=1、视觉塔冻结及 Huanyu 适配参数。

### 3.4 单机16卡规则奖励

```text
run_huanyu_aqa_audio_train_16card.sh
  → run_huanyu_aqa_audio_train.sh（RM_ENABLE=0）
  → run_qwen36_huanyu_gspo_npu.sh（绕过 GenRM 包装层）
  → python3 -m verl_omni.trainer.main_omni
```

奖励为 Huanyu `rule_based_reward.compute_score`，计算最终答案与 ground truth 的字符二元组集合 Dice，不加载 RM、不调用外部 API。它是词面相似度奖励，与 GenRM 语义判分不同。默认 thinking 模式下，未输出结束思考标记的答案记0；`ENABLE_THINKING=0` 会同步调整模板与规则要求。

`RM_ENABLE=0` 适用于普通 AQA、普通 precision 或16卡规则入口。32卡一键入口及 GenRM 包装层会重新启用 RM，不能只在它们前面 export 0来关闭奖励池。

## 4. 启动命令

### 只准备环境，手动启动 Ray 和训练

在每个节点的 Bash 终端执行一次；不要求先设置 `VC_WORKER_HOSTS` 或 `VC_TASK_INDEX`。默认准备冻结缓存训练：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new
export HUANYU_JOB_MODE=frozen_cache
export HUANYU_VLLM_SOURCE=/opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit
export HUANYU_EMBED_CACHE_DIR=/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128
source "$PROJECT_ROOT/examples/qwen3_6_huanyu/prepare_huanyu_env.sh"
```

它会设置jemalloc，按摘要比较并同步vLLM-Omni到 `/vllm-workspace/vllm-omni`（保留旧目录备份），安装当前项目插件，加载 `env_export_huanyu.sh`，恢复路径/音频/精度配置，检查原生导入和至少16张可见NPU。env文件还包含mathruler/zarr安装、CANN/ATB环境加载、日志目录与诊断文件准备。源目录已匹配时跳过复制。调用链为：

```text
source prepare_huanyu_env.sh
  → 独立bash运行 run_huanyu_32card_job.sh（HUANYU_PREPARE_ONLY=1）
  → job_bootstrap.py sync → editable安装 → source环境文件
  → 音频/精度配置 → 原生导入检查 → check-devices
  → prepared_env.py生成activate.sh → 结束准备子进程
  → 在当前终端source activate.sh，保留环境变量和工作目录
```

准备过程不执行Ray启动/停止、集群等待、训练入口或worker保活。完成后再分别手动启动Ray，仅在主节点执行普通训练脚本，例如冻结用 `run_huanyu.sh`。准备成功会设 `HUANYU_SKIP_EDITABLE_INSTALL=1`，普通入口可复用已安装插件。

AQA在线训练将上面模式与组件配置改为下例；准备完成后使用对应普通训练入口：

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export HUANYU_TRAIN_AUDIO_ENCODER=true HUANYU_TRAIN_AUDIO_PROJECTOR=true
source "$PROJECT_ROOT/examples/qwen3_6_huanyu/prepare_huanyu_env.sh"
```

支持 `frozen_cache`、`aqa_unfreeze`、`aqa_precision`、`aqa_16k_sync_each`；后两者强制full精度和graph=0。AQA准备关闭embedding cache，默认使用AQA的train/validation路径。模式只选择环境、音频和精度策略；长度、batch与梯度同步仍由后续普通启动脚本配置，选择 `aqa_16k_sync_each` 后应运行 `run_huanyu_aqa_16k_sync_each.sh`。

默认环境文件是本目录的 `env_export_huanyu.sh`。如站点文件叫 `source_env_export.sh`，在准备前 `export HUANYU_ENV_FILE=/绝对路径/source_env_export.sh`。项目/模型/数据路径、显式的输出目录与缓存路径会在source后恢复。不要用一个会主动启动Ray或训练的文件作为环境文件。

`source` 会将成功准备的环境应用到当前终端，保留原有shell选项。若用 `bash prepare_huanyu_env.sh`，准备完成后须执行它打印的 `source .../activate.sh`；子进程不能直接修改父终端环境。准备子进程失败不会生成可用的新激活文件，也不会应用部分环境到父终端；已完成的复制/安装不会因此撤销。激活文件保留受管理的运行配置及其他环境变化，包含unset，并用私有权限保存。

准备日志和激活文件在 `$PROJECT_ROOT/runs/environment/prepare_<时间>_<随机串>/`，可通过 `HUANYU_PREPARE_LOG_ROOT` 改根目录；分别为 `prepare.log`、`activate.sh`。每个节点使用自己刚生成的激活文件。准备时若本地已有Ray，源码不同则拒绝替换；源码一致则跳过复制和项目editable安装，但仍加载环境并检查。新环境不会自动进入已有raylet，应在手动启动Ray之前完成准备。

### 两节点一键启动

两节点使用相同项目路径和 IP 列表；主节点 `VC_TASK_INDEX=0`，从节点改为1。下例选择冻结缓存训练：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new
export VC_WORKER_HOSTS="主节点IP,从节点IP"
export VC_TASK_INDEX=0                       # 从节点设置为1
export HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1

HUANYU_JOB_MODE=frozen_cache \
TRAIN_FILE=/opt/huawei/dataset/infra_data/data/train.parquet \
VAL_FILE=/opt/huawei/dataset/infra_data/data/validation.parquet \
HUANYU_EMBED_CACHE_DIR=/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128 \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_32card_job.sh"
```

切换场景时，两节点使用同一个最外层脚本和相同配置。例如冻结16K改用 `run_huanyu_32card_16k_job.sh`；AQA 16K改用 `run_huanyu_aqa_16k_sync_each_job.sh`。AQA 数据优先用下节的专用变量指定。

jemalloc 先查 `HUANYU_JEMALLOC_LIBRARY` / `Q36_JEMALLOC_LIBRARY`，再查 `LD_PRELOAD` 和系统常见目录。找不到时设置两节点各自可读的绝对库路径，例如 `export HUANYU_JEMALLOC_LIBRARY=/实际路径/libjemalloc.so.2`。

### 已准备环境和 Ray 的普通启动

各节点先按上一节 `source prepare_huanyu_env.sh`，随后手动准备Ray。仅在主节点执行：

```bash
RAY_ADDRESS=auto TOTAL_TRAINING_STEPS=2 \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_precision.sh"
```

正式训练将 `TOTAL_TRAINING_STEPS` 改为 `-1`（不限制总步数，仍受 epoch 限制）。AQA precision / sync_each 普通入口不 source 环境、不安装包、不启停 Ray，并检查现有集群；通用普通入口下游可能执行 editable 安装，已安装好时可设 `HUANYU_SKIP_EDITABLE_INSTALL=1`。

普通冻结入口 `run_huanyu.sh` 要求在两节点启动 Ray 前配置 `HUANYU_FREEZE_AUDIO=true`、两个组件训练开关为 `false`、`HUANYU_USE_PRECOMPUTE=1` 和缓存目录。一键冻结入口自动完成这些设置。

## 5. 如何改重要参数

### 5.1 参数传递和优先级

- Shell 环境变量写在 `bash` 前，或先 `export`，如 `MAX_RESPONSE_LENGTH=4096 bash ...`。参数名区分大小写，长度用整数，不能写 `10k`。
- Hydra 参数写在脚本后，如 `actor_rollout_ref.actor.optim.lr=5e-7`。它只改最终配置，不一定修改前置检查使用的 Shell 变量。
- 同一 Hydra 键通常由最终命令中最后一项决定；脚本收到用户参数后还可能追加固定值，不能笼统认为“用户写在命令末尾就一定生效”。

| 入口 | 覆盖规则 |
|---|---|
| 冻结一键入口 | 固定冻结两个音频组件、启用缓存和 RM；默认full，可显式选custom。路径、卡池、音频和精度值在加载环境文件后恢复 |
| 冻结16K / Whisper 对照 | 默认长度等 Hydra 参数在用户参数前；改长度仍需对齐 Shell 预算、rollout 预算和媒体预算 |
| AQA 共用入口 | 数据、组件、batch、TP、micro=1、dynamic=false等在用户参数后追加；通过对应环境变量调整可配置项 |
| AQA unfreeze / precision 5K | Shell 长度设置为3072/2048；只export新长度不会改变默认。可同步覆盖完整 Hydra 长度组，任意长度实验更适合共用 AQA 入口 |
| AQA sync_each | 固定16+16、TP4、batch4/mini4/n4、micro1、静态 batching、逐micro同步、full、graph0。总长度16K，通过 `HUANYU_16K_RESPONSE_LENGTH` 改输入/输出分配 |
| 最底层后端 | 用户参数后再次追加graph和precision参数；应使用 `HUANYU_GRAPH_MODE`、`HUANYU_PRECISION_PROFILE` 控制 |

### 5.2 数据、模型和音频冻结

通用数据使用 `TRAIN_FILE` / `VAL_FILE`；AQA优先使用 `HUANYU_AQA_TRAIN_FILE` / `HUANYU_AQA_VAL_FILE`，其次才是非通用默认值的 `TRAIN_FILE` / `VAL_FILE`。AQA会将通用默认路径自动换成 AQA路径。模型分别通过 `MODEL_PATH`、`RM_MODEL_PATH` 指定。

```bash
HUANYU_AQA_TRAIN_FILE=/你的AQA目录/train.parquet \
HUANYU_AQA_VAL_FILE=/你的AQA目录/validation.parquet \
HUANYU_TRAIN_AUDIO_ENCODER=false \
HUANYU_TRAIN_AUDIO_PROJECTOR=true \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_unfreeze_32card_job.sh"
```

上例只训练projector；一键入口需要两节点都这样设置。

| `HUANYU_TRAIN_AUDIO_ENCODER` | `HUANYU_TRAIN_AUDIO_PROJECTOR` | 行为 |
|---|---|---|
| `true` | `true` | 两者都训练 |
| `false` | `true` | 只训练projector |
| `true` | `false` | 只训练encoder |
| `false` | `false` | 两者冻结；AQA仍在线编码，语言模型继续训练 |

组件开关优先于旧的 `HUANYU_FREEZE_AUDIO`。只设旧开关不能覆盖已有组件值；unfreeze一键入口默认显式将二者设为true，冻结对照应显式设二者为false。`frozen_cache` 始终强制二者为false。环境值会转换为模型配置 `huanyu_train_audio_encoder` / `huanyu_train_audio_projector`。

### 5.3 长度、batch与显存参数

| 参数 | 含义/使用方式 |
|---|---|
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 通用入口的数据输入/输出上限；音频、图片展开token占用输入预算 |
| `ROLLOUT_PROMPT_LENGTH` / `ROLLOUT_MAX_MODEL_LEN` | rollout输入预算/引擎总长度；总长度应覆盖输入和输出 |
| `MAX_TOKEN_LEN_PER_GPU` | Actor、rollout/ref log-prob的token预算；静态batching时不能代替micro-batch控制 |
| `MINICPMO_MM_PROMPT_BUDGET` | 多模态prompt预算，需与rollout输入上限一致 |
| `HUANYU_16K_RESPONSE_LENGTH` | 两种16K入口的输出长度，默认2048；输入自动为 `16384 - 输出`，引擎16512 |
| `HUANYU_16K_ATTN_IMPLEMENTATION` | 两种16K入口的训练attention，默认sdpa，可选eager |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` / `ROLLOUT_N` | 每批prompt、PPO mini-batch prompt、每prompt生成数 |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | 静态训练的每卡micro-batch；AQA强制1 |
| `actor_rollout_ref.rollout.max_num_seqs` | 推理并发，与训练micro-batch不同；冻结16K也支持 `HUANYU_16K_MAX_NUM_SEQS`，AQA sync_each固定32 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | Actor推理内存比例；GenRM包装层默认0.5会覆盖基础环境默认，最终修改用Hydra；RM可用 `RM_GPU_MEM_UTIL` |

| 入口 | prompt / response | rollout prompt / engine |
|---|---|---|
| 冻结通用、Whisper对照 | 3072 / 2048 | 3104 / 5248 |
| AQA共用、16卡规则 | 4096 / 2048 | 4096 / 6272 |
| AQA unfreeze一键、precision | 3072 / 2048 | 3072 / 5248 |
| 冻结16K | 14336 / 2048 | 14368 / 16512 |
| AQA 16K sync_each | 14336 / 2048 | 14336 / 16512 |

AQA共用入口要求 `PPO_MINI_BATCH_SIZE × ROLLOUT_N = Actor总卡数`、`ROLLOUT_N >= 2`、mini-batch能被rollout副本数整除、train batch能被mini-batch整除。16+16默认 `4×4=16`，每个Actor rank每次mini-batch一条序列。冻结通用默认train32/mini32/n4，Whisper对照为32/16/8，不要直接套到AQA。

例如，已准备环境与Ray时，使用 **2K输入 + 10K输出**：

```bash
MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=10240 \
ROLLOUT_PROMPT_LENGTH=2048 ROLLOUT_MAX_MODEL_LEN=12288 \
MAX_TOKEN_LEN_PER_GPU=12288 MINICPMO_MM_PROMPT_BUDGET=2048 \
NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4 \
RM_ENABLE=1 RM_N_GPUS_PER_NODE=16 RM_NNODES=1 RM_TP=4 RM_SPLIT_NODES=1 \
TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 \
HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0 \
HUANYU_SKIP_EDITABLE_INSTALL=1 RAY_ADDRESS=auto \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_audio_train.sh" \
  actor_rollout_ref.rollout.response_length=10240 \
  actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  ++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true \
  actor_rollout_ref.rollout.max_num_seqs=32
```

共用AQA入口已经关闭Actor、rollout log-prob、ref log-prob的dynamic batching，并固定micro=1。同步开关要求部署模型目录中的 `microbatch_sync.py` 等Python代码；让每次micro-batch反向传播同步梯度，不改变optimizer更新频率。它不保证任意长序列不OOM，也不控制vLLM初始化内存。

忽略EOS可追加 `actor_rollout_ref.rollout.ignore_eos=true`；需要最小生成10240再加 `++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens=10240`。这依赖当前 `rollout_inputs.py` 的透传，需连同Python包部署；字段名不是 `ignore_eos_token`。显式停止条件、请求中断或长度预算仍可能终止生成。

### 5.4 精度、卡数、学习率与步数

- `HUANYU_PRECISION_PROFILE=full` 启用训练/推理GDN FP32、prefill/gating/conv/L2/barrier、私有hybrid cache、stable linear、逐层权重同步、KV清零和媒体隔离，并设FSDP pin_memory=0；不表示所有模型运算都变成FP32。
- full会把上述精度开关重新设为1，单独export某项为0不生效。单项对照需在允许的入口选择 `HUANYU_PRECISION_PROFILE=custom`；custom保留显式开关和各项默认，并非全部关闭。AQA precision / sync_each会强制full。
- `HUANYU_GRAPH_MODE=0` 为eager推理；full不能搭配graph=1。graph只控制Actor/RM推理，不开启Actor FSDP编译。
- `NUM_NPUS × NNODES` 是Actor池，`RM_N_GPUS_PER_NODE × RM_NNODES` 是RM池。16+16的两池各用一个节点，故双方NNODES=1，整个Ray集群仍有两节点。默认rollout/RM TP4，各4个推理副本；Actor/ref训练使用FSDP2，TP值不表示训练也按TP4切分。
- GenRM长度独立：默认总长32768、输出1024、输入31744，可用 `RM_MAX_MODEL_LEN`、`RM_RESPONSE_LENGTH`、`RM_PROMPT_LENGTH` 调整。
- 学习率、epoch、保存/验证间隔用Hydra，例如末尾加 `actor_rollout_ref.actor.optim.lr=5e-7 trainer.total_epochs=3 trainer.save_freq=20 trainer.test_freq=10`。
- AQA总步数用 `TOTAL_TRAINING_STEPS=2`，正式训练用 `-1` 或 `null`；AQA会在用户参数后追加总步数配置。冻结链短跑直接加 `trainer.total_training_steps=2`。AQA不使用 `DEBUG_MODE` / `DEBUG_SAMPLES`。

## 6. 全部辅助脚本说明

前文列出最外层入口；下表补齐中间层和辅助工具，通常不作为训练第一入口。

| 文件 | 作用 |
|---|---|
| `run_32card.sh` | 兼容入口，转入run_huanyu.sh，不建立Ray |
| `run_qwen36_huanyu_32card.sh` | 资源/路径/采样默认层，转入OmniMix + GenRM包装层 |
| `run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh` | 配置多模态、GenRM和奖励池，强制RM_ENABLE=1 |
| `run_qwen36_huanyu_gspo_npu.sh` | 最终检查和训练命令拼装 |
| `env_export_huanyu.sh` | 在Ray启动前source；设置路径/Ascend/通信，切换工作目录、创建日志，并包含mathruler/zarr安装操作；不启停Ray |
| `prepared_env.py` | 仅用标准库保存环境摘要，在准备检查全部通过后生成可source的运行配置及环境差异；处理export/unset，不连接Ray |
| `audio_train_mode.sh` | 解析encoder/projector开关，兼容旧冻结开关 |
| `precision_mode.sh` | 精度开关、实际TP、Huanyu worker extension和Ray环境参数 |
| `graph_mode.sh` | 统一Actor/RM推理graph/eager，校验与full的互斥 |
| `run_output.sh` | 创建唯一运行目录及console.log、genrm_dump；同一调用链重复source复用目录 |
| `job_bootstrap.py` | `resolve`、`sync`、`check-devices`、`wait-head`、`wait-cluster`，供一键启动调用 |
| `check_native_runtime.py` | 原生导入与路径检查，常用 `--check --expected-root /vllm-workspace/vllm-omni` |
| `check_runtime.py` | 必填 `--model-path`，检查导入/processor/合成音频图片/mRoPE；不加载权重、不启Ray |
| `check_aqa_audio_train.py` | `--train`、`--validation`、`--train-batch-size`，检查AQA来源/字段/媒体路径；不解码音频、不检查归档内部成员 |
| `migrate_aqa_audio.py` | 复制音频归档并改写parquet，支持预检查、批量读取和进度日志，不执行训练 |

AQA检查会读取train和validation全部行，即使关闭训练前验证也要提供AQA音频验证集。它要求 `raw_source=aqa_real`、非空prompt/ground_truth/audio且无image/video。归档引用使用 `archive.zarr.zip::member_id`。

数据迁移例子：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/migrate_aqa_audio.py" \
  --parquet /原数据/train.parquet /原数据/validation.parquet \
  --source-root /原音频根目录 \
  --destination-root /opt/huawei/dataset/infra_data/data/aqa_dataset \
  --dry-run
```

正式写入去掉 `--dry-run`。目标根目录下写同名parquet，音频复制到 `audios/`，默认parquet写实际目标音频路径，不做 `/home` → `/opt` 转换。脚本默认目标仍是 `/opt/huawei/dataset/hfy-infra-gy/data/aqa_dataset`，与训练默认路径不同，因此示例显式指定目标。归档成员引用保留，已有输出parquet拒绝覆盖；等待 `COMPLETE` 后再使用产物。可用 `--batch-size`、`--log-every`、`--progress-interval` 调整读取与日志频率。

## 7. 日志与生效确认

- 独立环境准备：`$PROJECT_ROOT/runs/environment/prepare_<时间>_<随机串>/prepare.log`，可用 `HUANYU_PREPARE_LOG_ROOT` 覆盖根目录；`[huanyu-env] READY` 表示检查通过。
- 一键节点日志：`$PROJECT_ROOT/runs/$EXPERIMENT_NAME/cluster/rank_0.log` / `rank_1.log`，可用 `HUANYU_JOB_LOG_DIR` 覆盖。
- 训练控制台：`[huanyu-output] console=...` 指出的 `$PROJECT_ROOT/runs/huanyu_<时间>_<随机串>/console.log`；GenRM dump在同一目录。
- checkpoint/TensorBoard由 `DEFAULT_LOCAL_DIR` / `TENSORBOARD_DIR` 或最终trainer配置决定。env文件会设为项目固定目录；AQA precision和sync_each普通入口默认改为独立运行目录。自定义路径需注意source顺序，不要假定所有入口保存位置一致。

```bash
LOG=/本次运行目录/console.log
grep -E '\[huanyu-(job|aqa|precision|cache|audio-train|memory|sampling)' "$LOG"
```

| 检查项 | 日志证据 |
|---|---|
| 数据集、入口 | `[huanyu-paths]`、`[huanyu-aqa] train=...`、节点日志中 `[huanyu-job] mode=... train_entry=...` |
| 两个音频组件是否允许训练 | `train_audio_encoder=... train_audio_projector=...`；模型端 `[huanyu-audio-train] configured component=... trainable=...` |
| 实际经过反向传播 | `[huanyu-audio-train] backward component=audio_encoder ...` / `component=audio_projection ...`；配置为true不等于已反传，需等待更新阶段 |
| 精度传到哪些进程 | `[huanyu-precision] phase=configured role=... profile=full`；installed表示安装，executed才表示相应运行路径执行 |
| 缓存启用 | `[huanyu-job] ... precompute=1`、`[huanyu-cache] strict audio+vision cache reader: ...` |
| 逐micro-batch同步 | `[huanyu-memory] ... sync_each_microbatch=enabled ...` |
| EOS/最小生成参数透传 | `[huanyu-sampling] ignore_eos=True min_tokens=...` |

只有 `Engine core initialization failed` 时，向前找 `WorkerProc failed to start` 对应子进程的首个异常；父进程汇总错误和清理时的SIGKILL不能单独确定OOM。本文核对的是脚本配置与调用链，实际NPU结果以本次服务器日志为准。
