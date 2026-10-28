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
| AQA 一键启动，自定义长度/同步 | `HUANYU_JOB_MODE=aqa_unfreeze bash run_huanyu_32card_job.sh` | 两节点 | 直接转入共用AQA入口，长度、精度和同步由参数配置 |
| 已准备环境和 Ray，冻结/缓存训练 | `run_huanyu.sh` | 仅主节点 | 需提前配置冻结与缓存环境；复用通用训练链 |
| 已准备环境和 Ray，AQA 自定义长度/参数 | `run_huanyu_aqa_audio_train.sh` | 仅主节点 | 在线音频，默认4096+2048，16+16，静态 micro=1 |
| 从 validation 抽100条，用已有WAV诊断训练 | `run_huanyu_aqa_wav100.sh` | 仅主节点 | 环境/Ray已准备；两音频组件冻结，16+16，先验证100条再训练 |
| 已准备环境和 Ray，STEM 纯文本 | `run_huanyu_stem.sh` | 仅主节点 | 默认5K、16+16、full、冻结音频、micro1、逐micro同步；支持12K及更长 |
| 单机16卡 AQA，关闭模型奖励 | `run_huanyu_aqa_audio_train_16card.sh` | 仅主节点 | Actor16、RM0，规则奖励；默认4096+2048 |

当前所有训练链通过 `precision_mode.sh` 默认选择 `full`。名称不带 `precision` 并不表示关闭精度修复。AQA统一用环境变量 `HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0` 明确选择完整修复。

AQA的precision、16K sync_each四个专用入口已移除，改用本文共用入口命令；旧job模式也已移除。一键AQA统一设置 `HUANYU_JOB_MODE=aqa_unfreeze`，默认 **16+16**。冻结版 `run_huanyu_32card_16k_job.sh` 仍保留 `HUANYU_16K_POOL_LAYOUT=24+8` 可选配置；选项存在不代表此前跨节点HCCL问题已经解决。

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

自定义5K/12K/16K、精度和梯度同步参数
  → run_huanyu_32card_job.sh（mode=aqa_unfreeze）
  → run_huanyu_aqa_audio_train.sh
```

到达共用 AQA 入口后继续：

```text
run_huanyu_aqa_audio_train.sh
  ├─ audio_train_mode.sh / graph_mode.sh / precision_mode.sh / run_output.sh
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

`RM_ENABLE=0` 适用于共用 AQA 或16卡规则入口。32卡一键入口及 GenRM 包装层会重新启用 RM，不能只在它们前面 export 0来关闭奖励池。

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

支持 `frozen_cache`、`aqa_unfreeze` 两种模式。AQA准备关闭embedding cache，默认使用AQA的train/validation路径；通过 `HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0` 选择完整修复。准备完成后统一运行 `run_huanyu_aqa_audio_train.sh`，长度、batch与梯度同步由启动参数配置。

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

切换场景时，两节点使用同一个最外层脚本和相同配置。例如冻结16K改用 `run_huanyu_32card_16k_job.sh`；自定义长度的AQA用 `HUANYU_JOB_MODE=aqa_unfreeze bash run_huanyu_32card_job.sh`，传入5.3节的长度和同步参数。AQA数据优先用下节的专用变量指定。

jemalloc 先查 `HUANYU_JEMALLOC_LIBRARY` / `Q36_JEMALLOC_LIBRARY`，再查 `LD_PRELOAD` 和系统常见目录。找不到时设置两节点各自可读的绝对库路径，例如 `export HUANYU_JEMALLOC_LIBRARY=/实际路径/libjemalloc.so.2`。

### 已准备环境和 Ray 的普通启动

各节点先按上一节 `source prepare_huanyu_env.sh`，随后手动准备Ray。仅在主节点执行：

```bash
RAY_ADDRESS=auto TOTAL_TRAINING_STEPS=2 \
HUANYU_PRECISION_PROFILE=full HUANYU_GRAPH_MODE=0 \
MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=2048 \
ROLLOUT_PROMPT_LENGTH=3072 ROLLOUT_MAX_MODEL_LEN=5248 \
MAX_TOKEN_LEN_PER_GPU=5248 MINICPMO_MM_PROMPT_BUDGET=3072 \
TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_audio_train.sh"
```

正式训练将 `TOTAL_TRAINING_STEPS` 改为 `-1`（不限制总步数，仍受epoch限制）。共用AQA入口跳过editable安装、不加载环境、不启停Ray，并先检查现有集群。冻结普通入口下游可能执行editable安装，已安装好时可设 `HUANYU_SKIP_EDITABLE_INSTALL=1`。

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
| AQA unfreeze 5K包装 | Shell长度设置为3072/2048；自定义长度直接调用共用AQA入口，或设置mode=aqa_unfreeze调用通用job |
| 通用job的AQA模式 | 保留显式长度、batch、精度、输出目录和总步数，加载站点环境后恢复，再转入共用AQA入口 |
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
| `HUANYU_16K_RESPONSE_LENGTH` | 冻结16K入口的输出长度，默认2048；输入自动为 `16384 - 输出`，引擎16512；AQA使用MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH |
| `HUANYU_16K_ATTN_IMPLEMENTATION` | 冻结16K入口的训练attention，默认sdpa，可选eager；AQA使用对应Hydra参数 |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` / `ROLLOUT_N` | 每批prompt、PPO mini-batch prompt、每prompt生成数 |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | 静态训练的每卡micro-batch；AQA强制1 |
| `actor_rollout_ref.rollout.max_num_seqs` | 推理并发，与训练micro-batch不同；冻结16K也支持 `HUANYU_16K_MAX_NUM_SEQS`，AQA可通过Hydra配置 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | Actor推理内存比例；GenRM包装层默认0.5会覆盖基础环境默认，最终修改用Hydra；RM可用 `RM_GPU_MEM_UTIL` |

| 入口 | prompt / response | rollout prompt / engine |
|---|---|---|
| 冻结通用、Whisper对照 | 3072 / 2048 | 3104 / 5248 |
| AQA共用、16卡规则 | 4096 / 2048 | 4096 / 6272 |
| AQA unfreeze一键 | 3072 / 2048 | 3072 / 5248 |
| 冻结16K | 14336 / 2048 | 14368 / 16512 |
| AQA 16K命令行示例 | 14336 / 2048 | 14336 / 16512 |

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

替代原16K入口：将上例长度改成 `MAX_PROMPT_LENGTH=14336 MAX_RESPONSE_LENGTH=2048 ROLLOUT_PROMPT_LENGTH=14336 ROLLOUT_MAX_MODEL_LEN=16512 MAX_TOKEN_LEN_PER_GPU=16512 MINICPMO_MM_PROMPT_BUDGET=14336`，response_length参数改为2048，保留SDPA和逐micro同步，并添加 `actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.ref.fsdp_config.fsdp_size=-1`。需要一键准备和Ray时，在两节点设置同样参数，将执行文件换为 `run_huanyu_32card_job.sh` 并在bash前添加 `HUANYU_JOB_MODE=aqa_unfreeze`。

忽略EOS可追加 `actor_rollout_ref.rollout.ignore_eos=true`；需要最小生成10240再加 `++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens=10240`。这依赖当前 `rollout_inputs.py` 的透传，需连同Python包部署；字段名不是 `ignore_eos_token`。显式停止条件、请求中断或长度预算仍可能终止生成。

### 5.4 精度、卡数、学习率与步数

- `HUANYU_PRECISION_PROFILE=full` 启用训练/推理GDN FP32、prefill/gating/conv/L2/barrier、私有hybrid cache、stable linear、逐层权重同步、KV清零和媒体隔离，并设FSDP pin_memory=0；不表示所有模型运算都变成FP32。
- full会把上述精度开关重新设为1，单独export某项为0不生效。单项对照选择 `HUANYU_PRECISION_PROFILE=custom`；custom保留显式开关和各项默认，并非全部关闭。
- `HUANYU_GRAPH_MODE=0` 为eager推理；full不能搭配graph=1。graph只控制Actor/RM推理，不开启Actor FSDP编译。
- `NUM_NPUS × NNODES` 是Actor池，`RM_N_GPUS_PER_NODE × RM_NNODES` 是RM池。16+16的两池各用一个节点，故双方NNODES=1，整个Ray集群仍有两节点。默认rollout/RM TP4，各4个推理副本；Actor/ref训练使用FSDP2，TP值不表示训练也按TP4切分。
- GenRM长度独立：默认总长32768、输出1024、输入31744，可用 `RM_MAX_MODEL_LEN`、`RM_RESPONSE_LENGTH`、`RM_PROMPT_LENGTH` 调整。
- 学习率、epoch、保存/验证间隔用Hydra，例如末尾加 `actor_rollout_ref.actor.optim.lr=5e-7 trainer.total_epochs=3 trainer.save_freq=20 trainer.test_freq=10`。
- AQA总步数用 `TOTAL_TRAINING_STEPS=2`，正式训练用 `-1` 或 `null`；AQA会在用户参数后追加总步数配置。冻结链短跑直接加 `trainer.total_training_steps=2`。AQA不使用 `DEBUG_MODE` / `DEBUG_SAMPLES`。

### 5.5 STEM纯文本训练

环境和Ray已准备后，仅在主节点执行 `run_huanyu_stem.sh`。默认Actor16 + GenRM16、TP4、full精度、关闭缓存、冻结encoder/projector、micro=1且逐micro同步。数学奖励由GenRM按答案等价性给出二值判断；parquet中的 `style=rule` 不会覆盖这个显式奖励配置。

```bash
# 默认5K：2048输入 + 3072输出；STEM长度变量支持k后缀（1k=1024）。
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem.sh"
# 12K：2048输入 + 10240输出，可设TOTAL_TRAINING_STEPS=3短跑。
STEM_CONTEXT_LENGTH=12k TOTAL_TRAINING_STEPS=3 VAL_BEFORE_TRAIN=false \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem.sh"
```

数据目录默认 `/opt/huawei/dataset/infra_data/data/rl_stem_data`，可用 `STEM_DATA_DIR` 改目录，或 `STEM_TRAIN_FILE` / `STEM_VAL_FILE` 改文件。默认读取train.parquet与val.parquet，后者不存在时尝试validation.parquet。启动时为GenRM在本次运行目录写入带 `extra_info.question` 的parquet副本，源数据保持不变；该运行目录必须能被所有节点读取。

更长测试使用 `STEM_CONTEXT_LENGTH=16k STEM_PROMPT_LENGTH=4096` 等组合；默认遇到EOS可以提前结束，`STEM_IGNORE_EOS=true` 可用于长度压力测试。长度不保证显存可容纳，GenRM的 `RM_MAX_MODEL_LEN` 预算独立配置。默认训练前验证、每10步验证，保存rollout和validation dump；可用 `VAL_BEFORE_TRAIN` / `TEST_FREQ` 调整。

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
| `external_audio.py` | 按需读取外部音频文件、TAR成员或Zarr ZIP数组，返回音频字节并可回填样本的audio_buffer；作为Python模块调用 |
| `export_aqa_validation_wav.py` | 将AQA Parquet按行导出prompt、参考答案、对应WAV及索引；可生成vLLM Chat Completions请求JSON |
| `prepare_aqa_wav_parquet.py` | 按导出manifest匹配源Parquet，复用真实WAV路径，支持固定种子随机抽样；不读取ZIP、不覆盖源文件 |
| `request_aqa_validation_wav.py` | 默认只发送导出的WAV并请求文字回复，逐条调用HTTP Chat Completions，将回复TXT与完整结果JSON保存在对应row目录内 |
| `prepare_stem_jsonl_to_parquet.py` | STEM JSONL转Parquet，默认train.jsonl/val.jsonl；Optional类型注解兼容Python 3.8/3.9 |

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

### 外部音频按需读取：external_audio.py

该文件基于附件收录，补充了AQA采样率组属性及标量数组支持，用于样本只保存音频位置、音频实际存放在外部文件或归档中的场景。调用方先完成rank分配和样本筛选，再读取选中样本的音频；模块本身不读取SSTable/Parquet、不做分布式筛选，也没有命令行入口，目前未接入Huanyu训练调用链。

- `ExternalAudioReference`：记录会话/内容下标、绝对路径、相对路径和归档成员key。优先使用存在的 `absolute_path`；找不到时尝试 `audio_root/relative_path`，未指定root则读环境变量 `AL_AUDIO_ROOT`。
- `read_external_audio()`：普通文件直接返回原始字节；TAR按key提取成员字节；`.zarr.zip`读取成员的 `audio` 数组，采样率优先使用组属性 `sampling_rate` / `sample_rate`，其次使用 `sample_rate` 数组（单元素或标量）。在内存中编码为PCM16 WAV，不解压整个归档到磁盘、不重采样；采样率缺失或无效时明确报错。
- `materialize_external_audio()`：原地将字节填入 `train_unit.conversations[i].content[j].audio_buffer`，要求该内容的 `type` 为 `audio`。数据错误通过带 `error_type` 和路径信息的 `ExternalMediaError` 报告。

依赖 `pydantic`；Zarr ZIP分支另需 `numpy`、`numcodecs`、`soundfile`。该分支支持代码所约定的Zarr v2一维数组结构，并非任意ZIP文件都可读取。按需读取仍会将选中的整段音频载入内存，代码没有显式的字节数或时长上限。

在本目录下的Python程序中，可这样提取一条归档音频用于测试（路径与key替换为真实值）：

```python
from pathlib import Path
from external_audio import ExternalAudioReference, read_external_audio

ref = ExternalAudioReference(
    conversation_index=0,
    content_index=0,
    absolute_path="/opt/huawei/dataset/infra_data/data/aqa_dataset/audios/example.zarr.zip",
    relative_path="example.zarr.zip",
    archive_key="sample_id",
)
wav_bytes = read_external_audio(ref, audio_root=None)
Path("sample.wav").write_bytes(wav_bytes)
```

若原始引用是 `archive.zarr.zip::member_id`，需将两部分分别传入 `absolute_path` 和 `archive_key`；该模块不自动拆分 `::`。

### 从validation抽100条，直接使用已导出的WAV训练

先准备好环境和32卡Ray集群，再仅在主节点执行：

```bash
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_wav100.sh"
```

默认读取 `/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet`，从全部行中以seed=42不放回随机抽100条，复用 `/opt/huawei/dataset/infra_data/data/validation_vllm/manifest.jsonl` 对应的WAV。可分别用 `AQA_WAV_SOURCE_PARQUET`、`AQA_WAV_EXPORT_DIR`、`AQA_WAV_SAMPLE_SEED` 覆盖。若当前导出目录只有前50条，必须先按下一节导出完整validation到新目录，再用 `AQA_WAV_EXPORT_DIR` 指定，不能把缺失行静默换成其他样本。

调用链：`run_huanyu_aqa_wav100.sh → prepare_aqa_wav_parquet.py → run_huanyu_aqa_audio_train.sh → 原有训练链`。新Parquet位于本次 `HUANYU_RUN_DIR/wav100/train.parquet`，只替换 `audios` 中的路径，保持prompt、答案、其他字段和Arrow schema。旁边的 `train.selection.json` 记录原始行号、WAV路径和文件SHA256。准备脚本严格核对原始音频引用、顺序、prompt/答案和WAV格式，不重新解码ZIP或编码WAV；两个节点都必须能访问输出Parquet及这些绝对WAV路径。

该入口默认使用2748模型、Actor16+GenRM16、TP4、冻结encoder/projector、关闭embedding cache、full精度配置、静态micro-batch=1及逐micro-batch梯度同步。输入上限3072、回复上限2048；采样temperature=0、top_p=1、top_k=-1、repetition_penalty=1，ignore_eos=false。它先对这100条做训练前验证，再以batch=4、每题4个rollout训练一轮（25步）；可用 `TOTAL_TRAINING_STEPS=2` 短跑。训练和验证使用同一100条，是输入链路诊断，不是独立验证集效果评估；贪心的组内回答/奖励可能相同，使GRPO优势为零。

先看未更新权重时的 `validation_dump`，再看 `rollout_dump`。此对照保留原训练prompt，仅改变音频读取入口；不能把它当作与纯音频HTTP请求的所有条件已完全一致。旧导出器使用PCM16编码，复用同一WAV有利于服务/训练对照，但不是对源Zarr浮点数组的无损证明。

### 导出AQA验证集，逐条对应prompt、答案和WAV

将 `export_aqa_validation_wav.py` 与 `external_audio.py` 放在同一目录，安装 `pyarrow pydantic numpy numcodecs soundfile` 后执行：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/export_aqa_validation_wav.py" \
  --parquet /opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet \
  --output-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --model huanyu --max-tokens 2048
```

默认导出全部行；先试少量可加 `--limit 5`，输出目录必须不存在。`--model` 填服务实际的模型名称（与 `--served-model-name` 一致）；省略它则只导出数据，不生成请求。脚本不启动或调用推理服务，也不修改源Parquet和归档。

```text
validation_vllm/
  manifest.jsonl                 # 每行包含原Parquet行号、prompt、答案、WAV对应路径
  summary.json                  # status=complete才表示所选行全部完成
  samples/row_000000/
    prompt.json                 # 完整原始消息列表
    ground_truth.txt            # 原始参考答案
    audio_00.wav                # audios[0]，多音频另有audio_01.wav等
    sample.json                 # 本条索引与音频采样率/声道数/时长
    request.json                # 仅指定--model时生成，含base64 WAV
  samples/row_000001/...
```

编号使用Parquet的零基行号；原 `extra_info.index/problem_id` 另行记录，避免源索引重复覆盖。音频按 `audios` 顺序完整导出为PCM16 WAV，保留采样率和声道数。失败会停止，写入 `error.json` 与 `status=failed`；manifest只记录完整导出的行，重跑请换新目录。

请求JSON按原prompt中的 `<audio>` 顺序替换为 `input_audio`，保留system和问题文本，参考答案仅保存在数据文件中。占位符数量与音频数不一致时拒绝生成请求。格式参考[vLLM音频输入文档](https://docs.vllm.ai/en/latest/features/multimodal_inputs/#audio-inputs_1)，需要服务端模型支持音频Chat Completions；生成请求并不验证该服务器的模型适配。base64请求会额外占用磁盘，但无需让服务器访问导出端的WAV路径。

向已经启动的服务测试第一条（地址/端口按实际修改）：

```bash
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/opt/huawei/dataset/infra_data/data/validation_vllm/samples/row_000000/request.json
```

### 使用导出的WAV请求模型，并保存每条回复

`request_aqa_validation_wav.py` 使用 Python 3.8+ 标准库，无需安装OpenAI SDK、PyArrow或启动Ray。在能访问已启动的vLLM服务的机器上运行：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/request_aqa_validation_wav.py" \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --url http://localhost:28889/v1/chat/completions \
  --model qwen3.8 \
  --max-tokens 2048 \
  --timeout-s 300 \
  --limit 50
```

默认地址为 `http://localhost:28889/v1/chat/completions`，模型名为 `qwen3.8`，测试前50条，对应服务启动参数 `--port 28889 --served-model-name qwen3.8`；这三项可省略。请求脚本与服务不在同一机器时，使用 `--url` 将localhost替换为服务机器IP。客户端模型名使用服务别名，不填服务端实际权重目录 `/opt/huawei/dataset/model_gy/downloaded_models/2748`。这里使用HTTP音频问答接口，**默认仅发送WAV，适用于问题描述已包含在音频中的数据**，不发送问题文字、system消息或历史assistant消息；不需要设置回复音色的 `--ref-audio`。服务端必须支持音频Chat Completions。

脚本从 `manifest.jsonl` 取行号与有序WAV路径，仅构造一条user消息，其content中全部为音频项；保留采样率和声道，发送完整文件。`prompt.json` 和 `ground_truth.txt` 仍读取并保存到本地结果，方便对照，**默认均不发送给模型**。无需已有 `request.json`，客户端也不使用导出器生成的旧请求JSON。纯音频模式不依赖prompt中的 `<audio>` 占位符。

只有音频不含问题、确实需要同时发送文字时，才显式添加 `--include-prompt`；此时恢复原始prompt消息，并按 `<audio>` 顺序注入WAV，仍不发送ground_truth。脚本不会自动判断音频是否包含问题。默认控制台打印 `input_mode=audio_only prompt_sent=false modalities=text`，结果JSON同步记录这三项；其中保存的 `prompt` 字段是本地参考，不代表已经发给模型。服务器聊天模板仍会添加角色标记和音频占位符。

客户端默认发送 `audio_url`（`data:audio/wav;base64,...`），并设置 `modalities=["text"]` 请求文字回复。可用 `--audio-input-format input_audio` 切回原传输格式；两种格式都附带真实WAV内容。该参数不控制服务器的聊天模板。

**2748模型的模板兼容配置**：如果 `chat_template.jinja` 只处理image/video/text，音频请求会在模板中触发 `Unexpected item type in content.`。仅改成 `audio_url` 不足以修复，因为vLLM可能将两种音频输入都转换为模板中的audio项。对这套Huanyu thinker-only服务，在原启动命令中增加以下两个参数：

```bash
vllm serve /opt/huawei/dataset/model_gy/downloaded_models/2748 \
  --omni --port 28889 --log-stats --trust-remote-code \
  --stage-init-timeout 600 --served-model-name qwen3.8 \
  --deploy-config /vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml \
  --chat-template-content-format string \
  --interleave-mm-strings \
  2>&1 | tee minicpm.log
```

`string` 使模板接收包含模型音频占位符的字符串，真实音频仍通过多模态输入传递；`--interleave-mm-strings` 保留音频与文本片段的交错顺序。参考[vLLM服务参数](https://docs.vllm.ai/en/latest/cli/serve/#--chat-template-content-format)。这是服务端参数，不能放到本请求脚本中，也无需直接改权重目录中的模板文件。服务器定制分支的实际音频处理仍需通过单条请求验证。

停止原服务、用上述配置重启后，先重试一条；新结果标签保留此前失败日志：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/request_aqa_validation_wav.py" \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --limit 1 --result-tag template_string
```

成功后使用 `--limit 50 --result-tag template_string --skip-existing` 测试前50条；若该标签已有失败结果，用 `--limit 1 --result-tag template_string --overwrite` 重试。不要将音频请求替换为只包含WAV路径的文本，否则模型不会收到音频。旧客户端也可以配合上述服务端参数测试；更新后的客户端会在对应400错误中打印并保存修复提示。

```text
validation_vllm/samples/
  row_000000/
    prompt.json               # 原始问题
    ground_truth.txt          # 参考答案
    audio_00.wav              # 输入音频
    response.txt              # 模型回复正文
    response.json             # 行号、prompt、参考答案、WAV路径、模型回复、完整响应
  row_000001/
    response.txt
    response.json
```

结果保存在对应的 `row_XXXXXX/` **目录内**，与该条数据的prompt、参考答案和WAV放在一起。JSON中另存 `reasoning_content`、`usage`、`finish_reason` 和耗时；`finish_reason=length` 时标记 `truncated=true`，表示服务返回的回复被长度限制截断。正文为空（例如只返回推理内容）记为失败，原始响应仍保留在JSON中。旧版保存在samples下的 `row_XXXXXX.<tag>.txt/json` 不会自动迁移，续跑检查使用新的目录内路径。

- `--limit` 默认50：测试前50条，不足50条则测试全部已有行；`--limit 5` 可先测5条，`--limit 0` 才表示测试全部已导出的行。`--start-row 100` 从原Parquet零基行号100开始。
- `--skip-existing`：续跑时只跳过输入模式、请求地址、模型、参数、原始prompt、WAV字节与参考答案都匹配的成功结果；失败、损坏或输入变化的结果会报错，避免混用。此前带文字的结果不能直接复用于纯音频测试，建议加 `--result-tag audio_only` 另存。
- `--result-tag after_rl`：另存 `row_000000/after_rl.txt/json`，适合比较不同模型；`--result-tag template_string` 对应 `row_000000/template_string.txt/json`。`--overwrite` 则明确替换选中行的旧结果；不能与 `--skip-existing` 同用。标签不能使用 `prompt`、`ground_truth`、`sample`、`request`，以免覆盖导出数据。
- 默认逐条请求、遇错停止，不自动重试；`--continue-on-error` 可记录失败后继续，其最终退出码仍非零。HTTP错误、超时等写入对应结果JSON，不伪造回复。
- 若服务要求API Key，运行前 `export VLLM_API_KEY=...`。该值仅放入请求头，不保存到结果。

先测50条后，使用原命令改为 `--limit 0` 并加 `--skip-existing` 可继续请求其余数据；若要重试失败行，可使用 `--start-row N --limit 1 --overwrite`。客户端CPU测试覆盖请求与文件处理，实际模型适配需要在目标服务上验证。

## 7. 日志与生效确认

- 独立环境准备：`$PROJECT_ROOT/runs/environment/prepare_<时间>_<随机串>/prepare.log`，可用 `HUANYU_PREPARE_LOG_ROOT` 覆盖根目录；`[huanyu-env] READY` 表示检查通过。
- 一键节点日志：`$PROJECT_ROOT/runs/$EXPERIMENT_NAME/cluster/rank_0.log` / `rank_1.log`，可用 `HUANYU_JOB_LOG_DIR` 覆盖。
- 训练控制台：`[huanyu-output] console=...` 指出的 `$PROJECT_ROOT/runs/huanyu_<时间>_<随机串>/console.log`。AQA普通入口及一键入口默认将 `genrm_dump/`、`rollout_dump/`、`validation_dump/` 放在该运行目录下，三个目录同级；`[huanyu-aqa] dump defaults:` 打印目录默认值。旧运行的 `samples_*` 文件不会自动迁移。
- checkpoint/TensorBoard由 `DEFAULT_LOCAL_DIR` / `TENSORBOARD_DIR` 或最终trainer配置决定。共用AQA入口会将共享环境中的固定默认目录改为独立运行目录，并默认保存rollout_dump和validation_dump。显式环境目录与trainer参数优先；验证数据只在实际运行验证时产生。

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

## 8. serve 与训练 rollout 的音频 embedding 对照

`huanyu_audio_probe.py` 是默认关闭的采集工具，**采集两个真实推理引擎，不是拿 Actor 的 teacher-forcing 前向代替 rollout**。它不改变计算公式或返回值，也不为 serve 安装训练精度补丁。每个进程分别保存 JSON 元数据及 NPZ 张量：

- 原生 `get_audio_hidden_states` 的 fbank 特征、真实帧长及最终有效 audio embeddings；
- `thinker.apm`（encoder）的输入/输出；`audio_projection_layer` 的输入/输出，包含 fold 后真正送入 projector 的输入；
- 实际执行时 encoder/projector 的全部参数与 buffers 的 shape、dtype、字节哈希和统一 FP32 值哈希；
- 第一次 Huanyu IPC 同步前、成功同步后的权重指纹及同步计数。

它尚未抓取音频 embedding 插入语言模型后的完整序列/位置，也不在此处记录原始 WAV。先用 WAV 文件及样本清单确定同一输入，再用有效特征的 `feature_id` 配对；这个 ID 不能当作 WAV 哈希。采集会复制张量到 CPU 并同步设备，适合短诊断运行，不适合测吞吐；同名路径下的源代码也记录 SHA256。

**serve 侧**：将本目录放入 `PYTHONPATH`，在原来成功运行的 deploy YAML 副本中，将 stage 0 的 `engine_extras.worker_extension_cls` 设为 `huanyu_audio_probe.ProbeWorkerExtension`。下列命令生成新副本，保留原始 TP、采样、缓存与图配置，不覆盖已有文件或其他 worker extension：

```bash
export PROBE_ROOT=/opt/huawei/dataset/infra_data/data/huanyu_audio_probe
export PYTHONPATH="$PROJECT_ROOT/examples/qwen3_6_huanyu:${PYTHONPATH:-}"
mkdir -p "$PROBE_ROOT"
python3 - <<'PY'
import os
from pathlib import Path
import yaml
src = Path('/vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml')
dst = Path(os.environ['PROBE_ROOT']) / 'serve_probe.yaml'
cfg = yaml.safe_load(src.read_text())
extension = 'huanyu_audio_probe.ProbeWorkerExtension'
patched = 0
for scope in (cfg, cfg.get('platforms', {}).get('npu', {})):
    for stage in scope.get('stages', []):
        if stage.get('stage_id') != 0:
            continue
        extras = stage.setdefault('engine_extras', {})
        if extras.get('worker_extension_cls') not in (None, extension):
            raise RuntimeError('Existing worker extension must be reviewed before adding the probe')
        extras['worker_extension_cls'] = extension
        patched += 1
if not patched:
    raise RuntimeError('No stage 0 in this deploy schema')
with dst.open('x') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY
```

在**原 serve 命令**前加环境变量，并将 `--deploy-config` 改为新副本；其余配置先保持原状。使用一个尚不存在的 ARM 文件，避免初始化假音频耗尽采集次数：

```bash
export HUANYU_AUDIO_PROBE_DIR="$PROBE_ROOT/serve"
export HUANYU_AUDIO_PROBE_MAX_CALLS=2
export HUANYU_AUDIO_PROBE_ARM_FILE="$PROBE_ROOT/serve_ready_$(date +%s).flag"
echo "$HUANYU_AUDIO_PROBE_ARM_FILE"
# 运行原 serve 命令，--deploy-config 改为 "$PROBE_ROOT/serve_probe.yaml"
# 等服务就绪后，在另一终端 touch 上面打印的 ARM 文件，再发送选定 WAV 请求。
```

worker 中必须出现 `[huanyu-audio-probe] phase=installed`，真实请求后必须出现 `phase=executed ... file=...`。只有 `model_hook_installed` 不代表采集成功。不要为采集而给 standalone serve 添加训练的 `engine_hooks`，否则会改变待比较的基线。

**训练侧**：更新 Huanyu 包后，沿用当前 WAV 训练命令，追加下面的 Hydra 配置，将变量明确传入 Ray。不需要更换训练的 worker extension；采集安装在现有音频隔离 wrapper 外层。初次检查不要设置 `HUANYU_AUDIO_PROBE_FEATURE_ID`，以免把前端特征发生变化的样本过滤掉。

```bash
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_audio_train.sh" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_DIR="/opt/huawei/dataset/infra_data/data/huanyu_audio_probe/rollout"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_MAX_CALLS="4"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_ARM_FILE=""' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_FEATURE_ID=""'
```

上面只是追加参数的示例：模型、WAV Parquet、冻结及采样参数继续使用本次对照命令的设置。目录需在两个节点可写；未共享挂载时分别收集文件。每个模型实例最多捕获指定次数，可能包含 warmup；根据元数据的长度、特征、同步计数选择真实样本，不能默认两个 `forward_0001` 是同一 WAV。可设置 ARM 文件在就绪后才采集；已有特征完全相同的证据后，才可使用 `HUANYU_AUDIO_PROBE_FEATURE_ID` 精确筛选。首次 IPC 权重快照不受 ARM/特征筛选影响。

**比较**：从两端日志取得对应 JSON 的完整路径：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_audio_probe.py" \
  /实际serve采集目录/forward_0001.json \
  /实际rollout采集目录/forward_0002.json \
  --out "$PROBE_ROOT/comparison.json"
```

同一命令也能比较 `before_first_reload.json` 和 `after_first_reload.json`。多音频批次按有效特征 ID 唯一匹配；重复音频或特征本身不一致时，用 `--left-group 0 --right-group 0` 显式选择已知的同一 WAV，报告仍会标明输入特征不一致。`huanyu_probe_alignment.py` 按已核实的原生接口验证每次调用的帧长、batch、下采样和 fold 维度，将批处理与逐 chunk 处理还原为可比较的有效区；未知布局会报告不能对齐，不按下标强行配对。保留原始补齐张量，便于继续分析边界效应。

| 最先出现差异的位置 | 接下来检查 |
|---|---|
| 有效 fbank 特征或长度 | 波形解码、重采样、归一化、分块及 frontend 参数 |
| 首次同步前后权重值哈希 | Actor 加载时 dtype 转换、同步导出、native weight loader |
| 相同特征下 encoder 输出 | encoder 权重/buffers、eval 状态、精度、补齐和 batch 形状 |
| encoder 输出相同，projector 输入不同 | 帧折叠、尾帧丢弃、下采样率 |
| projector 输入相同，输出不同 | projector 权重与运算精度 |
| projector 输出相同，最终 audio embeddings 不同 | 有效长度裁剪、chunk 拼接顺序 |
| 最终 audio embeddings 相同 | prompt/template、音频 span 与位置、LLM 权重、GDN/cache/TP |

比较器输出逐元素最大/平均误差、RMSE、相对 L2、cosine 和误差最大的 token 行；不把单一相关系数接近 1 当作正确。字节哈希不同但 FP32 值哈希相同，可能仅是 dtype 不同。CPU 测试验证采集/配对/同步时序，真实 NPU 数值差异必须由上述运行证据确认。

### 8.1 跳过首次 Actor 权重同步，只做验证

`HUANYU_CHECKPOINT_VALIDATION_MODE=disk` 在 Huanyu TaskRunner 中跳过 `on_init_end()` 的 Actor 权重同步，保留 rollout 通过 `safetensors` 从 `MODEL_PATH` 加载的原始权重；`sync` 对照组正常执行一次首次同步。两组都设置 rollout 的 step 元数据为 0，随后只执行 `val_before_train` 并退出，不更新梯度。默认不设置该变量，正常训练行为不变。

不能只把同步函数改为空函数：初始化阶段还有 `sleep_replicas()`。本实验两组都要求 `free_cache_engine=false`，让原始权重和 KV cache 保持驻留，避免依赖同步过程唤醒。必须同时设置 `val_only=true`、`val_before_train=true`、`resume_mode=disable`，使用独立奖励池、非 LoRA 的 Huanyu `omni_sync` 训练入口及音频探针。诊断模式会拒绝不满足这些条件的配置，并禁止后续权重更新或进入训练循环。

在环境和 Ray 已准备好的主节点上运行；两个节点均需更新 Huanyu 包，并能访问同一模型、导出 WAV 和输出目录。下面顺序执行两组，复用 WAV100 的相同抽样种子、冻结 encoder/projector、Actor16+RM16、TP4 和贪心采样配置：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export RM_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B
export AQA_WAV_SAMPLE_SEED=42
export AB_ROOT="/opt/huawei/dataset/infra_data/data/huanyu_sync_ab_$(date +%Y%m%d_%H%M%S)"

for MODE in disk sync; do
  mkdir -p "$AB_ROOT/$MODE"
  HUANYU_CHECKPOINT_VALIDATION_MODE="$MODE" \
  EXPERIMENT_NAME="huanyu_sync_ab_$MODE" \
  bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_wav100.sh" \
    trainer.val_only=true trainer.val_before_train=true trainer.resume_mode=disable \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=false \
    "trainer.validation_data_dir=$AB_ROOT/$MODE/validation_dump" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_CHECKPOINT_VALIDATION_MODE=\"$MODE\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_DIR=\"$AB_ROOT/$MODE/probe\"" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_MAX_CALLS="8"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_ARM_FILE=""' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_FEATURE_ID=""' \
    || break
done
```

核对日志：`[huanyu-checkpoint-validation] mode=disk initial_actor_sync=skipped` 或 `mode=sync initial_actor_sync=completed`。`Initial validation metrics` 后应退出，没有 Actor optimizer step。控制台仍写入启动时打印的 `HUANYU_RUN_DIR/console.log`；完整验证回答在 `$AB_ROOT/{disk,sync}/validation_dump/`，探针在对应的 `probe/`。

选取真实验证样本的 `forward_*.json` 检查 `weight_reload_count`：disk 应为 0，sync 应为 1。disk 没有 `before_first_reload.json` / `after_first_reload.json` 是预期行为；用 forward 文件中的权重哈希对照 checkpoint，尤其是此前发生变化的四个 `conv_layers.0.layers.{1,4}.{weight,bias}`。探针可能先捕获 warmup，必须按有效特征 ID 配对，不能直接比较相同文件序号。

| 对照结果 | 可以得出的结论 |
|---|---|
| disk 正常、sync 再现近音词错误，且权重/embedding 差异同步再现 | 强证据指向 Actor 加载→FSDP 导出→传输→rollout 重载链路；还需边界检查区分具体环节 |
| 两组都异常 | 首次同步不是唯一原因，继续核对与 serve 的输入模板、精度补丁和 TP 等差异 |
| 两组都正常 | 本实验未复现；两组关闭了 sleep，不能据此排除原同步/sleep 交互问题 |

比较同一 WAV 的实际回答和 embedding，不能只比较平均 GenRM 分数。该实验跳过的是整模型同步，包含 LLM，因此不能单独证明四个 LayerNorm 参数就是全部根因；还要结合音频权重和 embedding 对照。当前仅通过 CPU 控制流程测试，真实 NPU 结论需以上两组服务器结果。

### 8.2 定位首次权重同步中最早的数值偏差

在 disk 正常、sync 异常后，启用 `HUANYU_WEIGHT_SYNC_AUDIT_DIR`。探针跟踪 VGG 两个 block 的 `layers.{1,4}.{weight,bias}` 共 8 个 LayerNorm 参数；40 维参数是已发现异常的目标，20 维参数是对照。它记录完整小向量、原 dtype 字节哈希、FP32 数值哈希和源码指纹，避免只因 BF16/FP32 dtype 不同就判为数值变化。

记录链路：`checkpoint → actor_loaded → actor_local_shard → actor_export → receiver_input → receiver_owned → after_finalize`，另记录 `before_reload`、`after_initialize` 和 `after_load_bucket`。`actor_export` 是实际 `OmniFSDPEngine` 原生成器完成 H2D、`full_tensor()` 和 BF16 cast 后的值；当前不能把这三步再拆开。分片检查仅支持一维 `Shard(0)`，其他 mesh 会明确报告证据不足。探针不额外调用 collective，不修改传输值。

两个节点更新代码后，在环境、Ray 已准备好的主节点上运行一次 sync 验证。沿用前面 WAV100 对照的数据和冻结配置：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export RM_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B
export AQA_WAV_SAMPLE_SEED=42
export AUDIT_ROOT="/opt/huawei/dataset/infra_data/data/huanyu_weight_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUDIT_ROOT"

HUANYU_CHECKPOINT_VALIDATION_MODE=sync \
EXPERIMENT_NAME=huanyu_weight_sync_audit \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_aqa_wav100.sh" \
  trainer.val_only=true trainer.val_before_train=true trainer.resume_mode=disable \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine=false \
  "trainer.validation_data_dir=$AUDIT_ROOT/validation_dump" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_CHECKPOINT_VALIDATION_MODE="sync"' \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_DIR=\"$AUDIT_ROOT/audio_probe\"" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_MAX_CALLS="8"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_ARM_FILE=""' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_AUDIO_PROBE_FEATURE_ID=""' \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_WEIGHT_SYNC_AUDIT_DIR=\"$AUDIT_ROOT/weights\"" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES="1"'
```

输出为 `$AUDIT_ROOT/weights/weights-<host>-<pid>-<uuid>.jsonl`。目录不共享时，先收集两个节点的文件到同一目录；不要混入其他实验。同步失败也可比较已落盘的阶段。比较命令：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_weight_sync.py" \
  --audit-dir "$AUDIT_ROOT/weights" --update 1 \
  --output "$AUDIT_ROOT/compare_update1.json"
```

退出码 `0=PASS`、`1=FAIL`、`2=INCOMPLETE`。报告保留每个已观测进程的结果及具体差异下标；缺少前序阶段时不会猜测最早异常边界。首次同步要求未经过 optimizer 更新；这 8 项 checkpoint 原为 BF16，因此到 FP32 再到 BF16 应能精确往返。若换成 FP32 checkpoint，要另行分析合法舍入，不能沿用此结论。

`actor_loaded` 在 FSDP 初始化前采集，非源 rank 的 meta 参数是正常占位。比较器将形状正确的占位记录标为 `DEFERRED_META`，至少需要一份真实参数与 checkpoint 一致才允许该阶段通过；全部为 meta 或缺少进程记录时仍为 `INCOMPLETE`。其他阶段的 meta 参数仍报错。旧版比较器曾将这些占位误报为 `FAIL`；更新比较脚本后可直接重比已有 JSONL，无需重跑训练。

| 最早不一致的阶段 | 下一步定位范围 |
| --- | --- |
| `actor_loaded` | checkpoint 读取、参数绑定或加载时转换 |
| `actor_local_shard` | FSDP 初始化、分片装载及导出前的参数状态 |
| `actor_export` | H2D、FSDP 聚合、导出 cast；尚不能单靠此阶段区分三者 |
| `receiver_input` | Actor 导出后到接收 callback 之前的打包/传输 |
| `receiver_owned` | 接收 buffer 的独立复制 |
| `after_finalize` | layerwise loader、临时参数 materialize、复制回原始存储 |

同时新增正常同步的完整性保护：拒绝未知/重复/形状错误的音频参数；一旦传入任何音频参数，要求 encoder 和 projector 全套完整，缺失时在 finalize 前报错。整套音频均省略的更新仍保留原权重。finalize 后还检查上述 8 个 LayerNorm 是否精确等于接收值转换到目标 dtype 后的结果；不一致时输出参数名和 `bad_indices`，不会继续使用已确认损坏的权重。此保护不从 disk 覆盖 Actor 权重，也不能发现 sender 本身已经输出了错误值。

**验证边界**：CPU 测试覆盖漏参/重复、真实 Torch meta 参数重建、故意破坏长度 40 参数后的报错、各阶段比较和不完整证据处理；尚不能证明 NPU 根因已修复。数值采集及小向量校验需要设备到 CPU 读取，会改变同步时序，尤其是导出前的分片快照；若开启检查后异常消失，应转查异步竞态，不能据此认定原链路正确。该探针只覆盖音频 VGG LayerNorm，不能替代整模型或 LLM 权重一致性验证。

### 8.3 Huanyu FSDP2 尾部分片的 CPU 卸载保护

已观测到的 16 路实验中，四个长度 40 的参数只有 rank 13 持有的索引 39 不一致，长度 20 的对照全部正常；`actor_export`、`receiver_input`、`receiver_owned`、`after_finalize` 哈希一致。最早异常在 Actor 本地分片阶段，后续传输没有新增差异。

PyTorch 2.10 的 `FSDPModule._apply()` 在参数转换后立即执行 `reset_sharded_param()`。非阻塞 D2H 转换尚未完成时，后者可能在 CPU 上复制尾部分片来补齐。40/16 的尾片只有 1 个有效元素、容量为 3，会经过这个分支；20/16 的非空片均为 2，不需要补齐。这个执行顺序提供了与实验相符的竞态机制，实际 NPU 上的因果确认仍需修复前后复测。

Huanyu adapter 现在自动安装 `fsdp_offload.py`：仅对该模型实例在 FSDP2 下的 `.to(cpu, non_blocking=True)` 强制阻塞，确保 CPU 补齐前数据已经就绪。当前默认的非 LoRA 训练中，初始化装载和后续手工卸载都受保护；不会重新从 disk 覆盖权重，解冻后的梯度更新及正常 Actor→rollout 同步保留。代价是 CPU 卸载不再与后续工作异步重叠。额外的 PEFT 外层包装可能绕过该实例的 `.to()`，不能直接套用此覆盖结论。

两节点更新代码、启动新任务后，日志应出现 `[huanyu-fsdp-offload] phase=executed`。用 8.2 的 sync 验证配置写入**新的**审计目录，要求 8 项参数各阶段全部 `PASS`，并与 disk 模式比较同一 WAV 的 embedding 和回答。仅重新比较旧报告不会修复旧进程中的参数。CPU 回归模拟延迟 D2H 与尾片补齐；不等同于 NPU 实测通过。
