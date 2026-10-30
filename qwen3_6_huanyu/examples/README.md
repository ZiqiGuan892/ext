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
| AQA 默认冻结encoder、训练projector，bs32 | `run_huanyu_aqa_projector_5k_bs32_job.sh` | 两节点 | 音频开关可配置；3072+2048，train/mini batch=32，n=4，Actor16 + RM16，关闭缓存 |
| AQA 一键启动，自定义长度/同步 | `HUANYU_JOB_MODE=aqa_unfreeze bash run_huanyu_32card_job.sh` | 两节点 | 直接转入共用AQA入口，长度、精度和同步由参数配置 |
| 已准备环境和 Ray，冻结/缓存训练 | `run_huanyu.sh` | 仅主节点 | 需提前配置冻结与缓存环境；复用通用训练链 |
| 已准备环境和 Ray，AQA 自定义长度/参数 | `run_huanyu_aqa_audio_train.sh` | 仅主节点 | 在线音频，默认4096+2048，16+16，静态 micro=1 |
| 从 validation 抽100条，用已有WAV诊断训练 | `run_huanyu_aqa_wav100.sh` | 仅主节点 | 环境/Ray已准备；两音频组件冻结，16+16，先验证100条再训练 |
| 已准备环境和 Ray，STEM 纯文本 | `run_huanyu_stem.sh` | 仅主节点 | 默认5K、16+16、full、冻结音频、dynamic batching、引擎默认梯度同步、跳过训练前验证；支持12K及更长 |
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

run_huanyu_aqa_projector_5k_bs32_job.sh（默认冻结encoder，训练projector和语言模型；32题×4条回答）
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

AQA 入口限定音频数据，在线编码，关闭 embedding cache；两个音频组件都冻结也不会自动切换为缓存读取。它默认静态 batching，允许通过命令行开启 Actor、rollout/ref log-prob 的 dynamic batching；静态模式固定 micro-batch=1，并保留 FSDP2、sequence parallel=1、视觉塔冻结及 Huanyu 适配参数。

`run_huanyu_aqa_projector_5k_bs32_job.sh` 从环境读取 `HUANYU_TRAIN_AUDIO_ENCODER` / `HUANYU_TRAIN_AUDIO_PROJECTOR`，未设置时分别默认 `false` / `true`（true为训练，false为冻结），两节点需设置相同值。固定train/mini batch=32、n=4及16+16卡池；默认full精度，prompt3072+response2048，engine预算5248。两节点设置各自 `VC_TASK_INDEX` 和相同 `VC_WORKER_HOSTS` 后执行该脚本。数据可用 `HUANYU_AQA_TRAIN_FILE` / `HUANYU_AQA_VAL_FILE` 覆盖；默认训练前验证，`VAL_BEFORE_TRAIN=false` 可跳过。日志搜索 `[huanyu-aqa-projector-job]` 查看入口配置，搜索 `[huanyu-audio-train]` 查看实际模型的开关，默认 `train_encoder=False train_projector=True`。

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

AQA共用入口要求 `PPO_MINI_BATCH_SIZE × ROLLOUT_N` 能被Actor总卡数整除、`ROLLOUT_N >= 2`、mini-batch能被rollout副本数整除、train batch能被mini-batch整除。16+16默认 `4×4=16`，每个Actor rank每次mini-batch一条序列。也支持 `TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=4`：每步64条轨迹，每个rank分到4条，实际micro-batch仍由动态token预算或静态micro-batch配置决定；关闭逐micro-batch同步时，梯度累积可能增加峰值显存。冻结通用默认train32/mini32/n4，Whisper对照为32/16/8，仍需分别核对配置。

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

共用AQA入口默认关闭Actor、rollout log-prob、ref log-prob的dynamic batching，静态模式固定micro=1。可通过 `actor_rollout_ref.actor.use_dynamic_bsz=true`、`actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true`、`actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true` 开启动态分批，此时分批由各自token预算控制。同步开关要求部署模型目录中的 `microbatch_sync.py` 等Python代码；设为 `++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true` 时让每次micro-batch反向传播同步梯度，设为 `false` 时使用引擎默认同步行为，不改变optimizer更新频率。它不保证任意长序列不OOM，也不控制vLLM初始化内存。

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

环境和Ray已准备后，仅在主节点执行 `run_huanyu_stem.sh`。默认Actor16 + GenRM16、TP4、full精度、关闭缓存、冻结encoder/projector。Actor、rollout log-prob、ref log-prob均开启dynamic batching，按各自token预算分批；`huanyu_sync_each_microbatch=false`，使用引擎默认梯度同步。保留的micro-batch=1仅在关闭dynamic batching后生效。数学奖励由GenRM按答案等价性给出二值判断；parquet中的 `style=rule` 不会覆盖这个显式奖励配置。

STEM入口设置并向Ray传递 `HUANYU_STEM_TEXT_TRAIN=1`，复用 `aqa_position_ids.py` 的等价位置轴传输：TQ中保存一维序列，在模型入口恢复四轴，避免动态分批出现 `split_sizes=[4]`。日志搜索 `[huanyu-stem-positions]` 可确认执行。两节点需同时更新该Python文件和启动脚本；包含图像/视频或有效token的位置轴不一致时会报错，不会丢弃空间位置编码。

```bash
# 默认5K：2048输入 + 3072输出；STEM长度变量支持k后缀（1k=1024）。
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem.sh"
# 12K：2048输入 + 10240输出，可设TOTAL_TRAINING_STEPS=3短跑。
STEM_CONTEXT_LENGTH=12k TOTAL_TRAINING_STEPS=3 \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_stem.sh"
```

数据目录默认 `/opt/huawei/dataset/infra_data/data/rl_stem_data`，可用 `STEM_DATA_DIR` 改目录，或 `STEM_TRAIN_FILE` / `STEM_VAL_FILE` 改文件。默认读取train.parquet与val.parquet，后者不存在时尝试validation.parquet。启动时为GenRM在本次运行目录写入带 `extra_info.question` 的parquet副本，源数据保持不变；该运行目录必须能被所有节点读取。

更长测试使用 `STEM_CONTEXT_LENGTH=16k STEM_PROMPT_LENGTH=4096` 等组合；默认遇到EOS可以提前结束，`STEM_IGNORE_EOS=true` 可用于长度压力测试。长度不保证显存可容纳，GenRM的 `RM_MAX_MODEL_LEN` 预算独立配置。默认跳过训练前验证，初始化完成后进入训练；仍每10步验证，保存rollout和validation dump。可用 `VAL_BEFORE_TRAIN` / `TEST_FREQ` 调整，显式设置 `VAL_BEFORE_TRAIN=true` 会恢复初始验证。dynamic batching、逐micro同步和验证配置均可用末尾Hydra参数覆盖。

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

比较同一 WAV 的实际回答和 embedding，不能只比较平均 GenRM 分数。该实验跳过的是整模型同步，包含 LLM，因此不能单独证明四个 LayerNorm 参数就是全部根因；还要结合音频权重和 embedding 对照。2026-10-28 的服务器对照、边界定位与修复后结果见 8.3。

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

**验证边界**：CPU 测试覆盖漏参/重复、真实 Torch meta 参数重建、故意破坏长度 40 参数后的报错、各阶段比较和不完整证据处理；CPU 通过本身不能替代 NPU 复测，后者见 8.3。数值采集及小向量校验需要设备到 CPU 读取，会改变同步时序，尤其是导出前的分片快照；若开启检查后异常消失，应转查异步竞态，不能据此认定原链路正确。该探针只覆盖音频 VGG LayerNorm，不能替代整模型或 LLM 权重一致性验证。

### 8.3 近音词问题复盘：FSDP2 异步 CPU 卸载损坏尾部分片

**记录日期：2026-10-28。** 同一 checkpoint `2748` 在独立 vLLM serve 中回答基本正常，接入 verl 后第一次 rollout 就出现较多近音词混淆。改用导出的 WAV、调整采样参数后仍能复现；跳过首次 Actor 同步的 `disk` 验证正常，正常同步的 `sync` 验证异常。最终定位到 Actor CPU 卸载时的尾部分片竞态。修复后，服务器首次同步的 8 个被跟踪参数全部 `PASS`，用户确认 sync 模式的近音词错误**明显减少**。

本次环境为 `torch 2.10.0+cpu`、`torch-npu 2.10.0.post4`、`accelerate 1.15.0`，Actor FSDP 分片数为 16。以下结论来自用户提供的服务器报告、代码检查及 CPU 回归，并非在本地运行了 NPU 训练。

**定位证据**

| 检查边界 | 修复前的结果 |
| --- | --- |
| checkpoint → `actor_loaded` | rank 0 的 8 个真实参数全部一致；其余 15 个 rank 为正常 meta 占位 |
| `actor_local_shard` | 仅 `model.audio_encoder.conv_layers.0.layers.{1,4}.{weight,bias}` 四个长度 40 参数异常；全部发生在 rank 13、全局索引 39 |
| 长度 20 的 block 1 对照 | 分片、导出、接收、重载均一致 |
| `actor_export` → `receiver_input` → `receiver_owned` → `after_finalize` | 各阶段哈希相同，传输和 rollout 重载继续传递了 Actor 已经损坏的值 |

例如 `conv_layers.0.layers.1.bias[39]` 从 checkpoint 的 `1.578125` 变为 `0.00537109375`。这些 checkpoint 参数本来就是 BF16，BF16→FP32→BF16 可以精确往返；这里不是正常舍入误差。早期比较器把初始 meta 占位误判为 `actor_loaded=FAIL`，修正后最早异常边界为 `actor_local_shard`，不能据旧报告认定 checkpoint 加载错误。

**根本原因与首次 rollout 的关系**

本次根因是异步设备到 CPU 搬运（D2H）与 CPU 尾分片补齐之间缺少完成保证。原卸载调用为 `model.to("cpu", non_blocking=True)`，但 PyTorch 的 `FSDPModule._apply()` 会在转换后立即调用 `reset_sharded_param()`。后者遇到不完整分片时，分配补齐后的 CPU 存储并复制旧分片；此时异步 D2H 可能尚未完成，CPU 复制就读到了未就绪的数据。随后即便同步设备，也只能完成原缓冲区的搬运，无法修复已复制到新存储的错误值。

这个机制解释了异常为何集中于一个 rank：长度 40 按 16 路分片，每片容量 `ceil(40/16)=3`，rank 0–12 各有 3 个元素，rank 13 只有最后 1 个元素，rank 14–15 为空。只有 rank 13 有需要补齐的非空尾片。长度 20 的非空分片均为 2 个元素，没有同样的尾片复制。

当前 AQA 脚本默认 `param_offload=true`、`offload_policy=false`。代码链路是：

```text
Actor 加载 checkpoint → FSDP 分片装载
→ FSDPEngine.initialize() 中手工卸载到 CPU
→ 非阻塞 D2H 返回后，FSDP 立即在 CPU 补齐尾片（竞态位置）
→ on_init_end() 首次 Actor→rollout 权重同步
→ val_before_train / 第一次 rollout 使用损坏的音频 encoder 权重
```

因此问题可以在任何 optimizer 更新之前出现。这个默认配置下，应优先定位初始化后的手工卸载；不能只因 `fsdp2_load_full_state_dict()` 也有条件卸载分支，就认定该分支实际执行过。代码、局部损坏模式、CPU 可控复现和修复后的服务器结果共同支持上述根因；本次没有采集 NPU 算子级时序跟踪。

**解决方式**

修复提交：[c96c88a（GitHub check 分支）](https://github.com/ZiqiGuan892/ext/commit/c96c88aea1ab042217da624c09860a70d6496503)。[thinker_adapter.py](../thinker_adapter.py) 在音频模块加载后安装 [fsdp_offload.py](../fsdp_offload.py) 的实例级保护：Huanyu 根模型进入 FSDP2 后，显式 `.to(cpu, non_blocking=True)` 改为阻塞搬运，确保 CPU 补齐前数据就绪。必须在进入原 `.to()` 前选择阻塞方式，在其返回后补 `synchronize()` 已经太晚。

保护覆盖当前非 LoRA 训练中的初始化及后续手工卸载，保留 dtype、参数及 FSDP 状态管理；不修改磁盘 checkpoint，也不在同步时用 disk 权重覆盖 Actor，encoder/projector 解冻后的更新仍正常同步。代价是 CPU 卸载失去异步重叠。若另加 PEFT 外层包装，需要在最终根对象安装保护，不能直接套用当前覆盖结论。同时修正了比较器对初始 meta 占位的处理，详见 8.2。

**复测结果与复现入口**

- CPU：113 项相关测试通过。其中 [test_fsdp_offload_on_cpu.py](../tests/test_fsdp_offload_on_cpu.py) 使用真实 FSDP 补齐代码和模拟延迟搬运，复现 40/16 仅索引 39 损坏、20/16 正常；阻塞搬运后全部元素正确。
- 服务器：沿用 WAV100、冻结 encoder/projector、Actor16+RM16、TP4、首次 sync 后只验证的配置。`compare_huanyu_weight_sync.py --update 1` 对 8 个目标/对照参数全部输出 `PASS`、`earliest=None`。
- 实际回答：用户确认 sync 模式近音词错误明显减少；未报告错误率数值，也未确认所有近音词错误均消失。

修复前报告为 `/opt/huawei/dataset/infra_data/data/huanyu_weight_sync_20261028_174554/compare_update1.json`。修复后记录目录：

```text
/opt/huawei/dataset/infra_data/data/huanyu_sync_fixed_20261028_204316/
├── compare_update1.json       # 8 项参数全部 PASS
├── weights/                  # 权重边界审计记录
└── validation_dump/0.jsonl    # 本次验证回答
```

再次验证使用 8.2 的 sync 命令，两节点更新代码、启动新进程并使用新的审计目录；检查 `[huanyu-fsdp-offload] phase=executed`、`mode=sync initial_actor_sync=completed`、8 项 `PASS`，再比较同一 WAV 的回答。现有结果验证了这 8 个音频 VGG LayerNorm 的首次同步一致性，尚不代表整模型权重、所有 embedding 或多轮训练都已逐项验证。

## 9. 抽取相同100条音频，比较 serve 与 verl 的文字回答

使用 [aqa_text_ab.py](aqa_text_ab.py) 准备一次样本集、发送 HTTP 请求并配对文字；[launch_huanyu_aqa_text_ab.py](launch_huanyu_aqa_text_ab.py) 从同一份 `case.json` 启动两端。默认 `system_audio` 模式：**保留数据中原有的 system prompt，删除 user 消息中的问题描述，只保留音频及其顺序**。例如保留“你是一个多模态助手”，不再附加音频中已经说过的问题。原始数据没有 system prompt 时，不自动补写。`ground_truth` 只用于结果参照，不发送给模型。

两端使用同一 checkpoint、同一组 WAV、同一份规范化 messages。默认对齐如下：

| 项目 | 两端设置 |
| --- | --- |
| 样本 | 从 validation 随机抽100条，seed=42，原始 `row_XXXXXX` 配对 |
| 生成 | temperature=0、top_p=1、top_k=-1、每条1个回答、请求 seed=42 |
| 惩罚与停止 | repetition_penalty=1、frequency/presence_penalty=0、min_tokens=0、ignore_eos=false |
| 长度 | prompt预算3072、回答最多2048、引擎上下文5248（含128余量），允许正常EOS结束 |
| 推理 | TP4、BF16、eager、显存比例0.5、max_num_seqs=1；关闭prefix/chunked-prefill/MM缓存 |
| 精度与模板 | 两端均加载ext的full精度策略及Huanyu hooks，显式enable_thinking=true；关闭checkpoint默认采样覆盖 |
| verl运行 | Actor16+RM16、冻结encoder/projector、正常首次权重同步、仅val_before_train，无optimizer更新 |

verl保持4个AgentLoop worker和4个TP4推理副本；`val_batch_size=1`逐条验证，每条只生成1个回答。训练配置中的`rollout.n=4`用于满足原入口的批量检查，验证实际使用`val_kwargs.n=1`。新增的`q36_sampling.seed`在请求进入引擎前显式指定seed，避免不同副本的默认seed偏移。普通启动不设置该项时行为不变。

**准备**：两节点部署相同版本的ext与examples；环境及Ray按现有流程准备。WAV导出目录须包含所抽样本，不能只导出前50条。若未完整导出，先用`export_aqa_validation_wav.py --parquet ... --output-dir <新目录> --limit 0`导出全部验证集。下面复用已有完整导出目录，case目录由prepare创建，勿提前`mkdir`该目录。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export RM_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B
export AB_ROOT="$PROJECT_ROOT/runs/aqa_text_ab_$(date +%Y%m%d_%H%M%S)"
echo "$AB_ROOT"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" prepare \
  --parquet /opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --output-dir "$AB_ROOT" --model-path "$MODEL_PATH" \
  --sample-size 100 --seed 42 --prompt-mode system_audio
```

prepare检查原Parquet与WAV导出的来源、prompt、答案、音频顺序；保存WAV哈希与规范化prompt，写入新的`validation.parquet`，不修改源数据。改变回答上限用prepare的`--max-tokens 4096`；关闭思考用`--enable-thinking false`，两端同时读取。要比较包含问题文字的原始prompt，另建case并设`--prompt-mode original`。

**先运行独立serve**（占用4卡，保持前台；同卡不要同时启动verl）：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --port 28889 --served-model qwen3.8
```

此入口在case下写`serve.deploy.yaml`和`serve.launch.json`，运行与rollout相同的thinker-only pipeline及ext精度hooks；不是直接复用旧服务的TP2/图模式。HTTP通过`--chat-template-content-format string --interleave-mm-strings`处理当前checkpoint不接受audio内容项的模板。服务日志在`$AB_ROOT/serve_console.log`。

启动器先在API进程中注册`qwen3_6_whisper_thinker_only`，再在同一Python进程调用原生vLLM CLI；仅设置worker使用的`VLLM_OMNI_EXTERNAL_MODULES`不能保证读取deploy配置前完成注册。日志应先出现`[huanyu-text-ab] API pipeline registered: qwen3_6_whisper_thinker_only`，之后仍需等待服务实际就绪。旧版入口若报`Pipeline ... is not registered to OMNI_PIPELINES`，表示尚未进入模型加载与音频推理。更新`launch_huanyu_aqa_text_ab.py`后，确认失败进程已退出，保留同一批100条样本，只归档失败启动留下的三个文件再重试：

```bash
# AB_ROOT继续使用这次失败实验的绝对路径，不重新prepare。
FAILED_DIR=$(mktemp -d "$AB_ROOT/serve_failed_$(date +%Y%m%d_%H%M%S)_XXXXXX")
for name in serve.launch.json serve.deploy.yaml serve_console.log; do
  if [ -f "$AB_ROOT/$name" ]; then
    mv "$AB_ROOT/$name" "$FAILED_DIR/"
  fi
done
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --port 28889 --served-model qwen3.8
```

此重试方法只适用于请求尚未开始的启动失败；已有回答的实验按下方结果保留规则处理。API注册修复已通过15项启动器CPU测试，包括同进程注册顺序、原生CLI参数传递和注册失败时停止；仍需服务器NPU复测。

服务就绪后，在另一个终端设置**相同的PROJECT_ROOT和AB_ROOT绝对路径**，发送全部100条：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --url http://localhost:28889/v1/chat/completions \
  --model qwen3.8
```

如果旧版A/B脚本报`build_request() got an unexpected keyword argument 'include_prompt'`，说明同目录的`request_aqa_validation_wav.py`接口与调用方不一致；该行尚未发送HTTP请求。更新后的`aqa_text_ab.py`直接从case中的规范化prompt构造`audio_url`请求，保留system原文及WAV顺序、指定`modalities=["text"]`，不再调用辅助脚本的`build_request`；辅助脚本仍用于读取WAV和收发HTTP。无需重启已就绪的服务，也无需重新抽样。

例如仅`row_000026`发生上述构造错误，更新`aqa_text_ab.py`后先归档它的失败记录，再使用`--skip-existing`重试；之前成功的结果会校验后跳过：

```bash
FAILED_DIR=$(mktemp -d "$AB_ROOT/request_failed_$(date +%Y%m%d_%H%M%S)_XXXXXX")
mv "$AB_ROOT/serve/row_000026.json" "$FAILED_DIR/"
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --url http://localhost:28889/v1/chat/completions \
  --model qwen3.8 --skip-existing
```

请求接口修复通过12项工具CPU测试，覆盖旧辅助函数签名、多音频顺序、system原文与问题文字区分、纯文字输出、以及已有成功记录的请求哈希兼容性。真实回答仍需服务器测试。

**完成请求后停止上述独立serve，释放其卡，再运行verl验证**。此入口不拉起Ray、不安装包、不替换vLLM-Omni：

```bash
RAY_ADDRESS=auto python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" rollout \
  --case-dir "$AB_ROOT"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" compare \
  --case-dir "$AB_ROOT"
```

可给启动命令加`--dry-run`仅查看参数。两端启动前检查新Parquet、manifest和checkpoint配置文件哈希；修改后应另建case。已有结果不会被静默覆盖。HTTP中断且已有记录均成功时可加`--skip-existing`继续；失败记录或重新跑完整实验使用新case。收到错误时先看对应行JSON中的`error`。

```text
$AB_ROOT/
├── case.json、manifest.jsonl、validation.parquet
├── serve/row_XXXXXX.txt、row_XXXXXX.json       # HTTP文字、请求参数、原始响应
├── rollout/row_XXXXXX.txt、row_XXXXXX.json     # verl文字、实际解码prompt、样本标识
├── validation_dump/0.jsonl                  # 原有验证dump及奖励
├── comparison.csv、comparison.jsonl         # 同一行的prompt/WAV/答案/两端回复
└── summary.json                            # 完整配对数及文字相同/不同数量
```

比较器按原始样本ID配对，拒绝串实验、错prompt/音频、未知行或重复结果；缺失/失败明确标记。HTTP将思考过程单独放入`reasoning_content`时，两份文字都保留，但不自动拼接猜测“完全相同”。`reply_exact_match`只表示文字一致，不能当作准确率或近音词错误率。`rollout`记录的是AgentLoop合并后的回答，与验证dump采用相同token来源；其请求参数快照也不会冒称引擎最终解析参数。

**验证边界**：旧版 case 只比较文字，没有采集 embedding；规范化 messages 一致不能证明实际模型输入一致。新版 prepare 默认开启第 11 节的完整输入 embedding 检查，必须使用对应启动器重新运行两端，不能给旧文字结果补算或推断 embedding。rollout JSON 仍保留 `rendered_prompt_with_special_tokens`。批处理、副本分配、权重和语言模型内部计算需要与输入 embedding 的比较结果分开判断。

本次本地验证：61项相关CPU测试通过，包含真实小型Parquet/WAV、模拟HTTP和实际捕获wrapper串联的配对测试；新增三个工具/模块通过Python 3.8语法检查。未在本地启动NPU或真实模型服务。

## 10. 首次同步后的全模型权重检查

此前 `compare_huanyu_weight_sync.py` 的 PASS **只覆盖 8 个音频 VGG LayerNorm 参数**，不能证明整个推理模型与 checkpoint 一致。新增的 `full_weight_audit.py` 在真实 rollout worker 内枚举全部注册参数（含别名），对每个参数的完整内容计算 SHA256，并分别保存 shape、dtype、numel。注册 buffer 单独列出，不冒充 checkpoint 参数。

检查采用三个阶段：

1. `initial_disk`：首次 Actor 同步之前，原生 safetensors 加载完成后的实际推理参数。
2. `after_sync`：首次同步及 layerwise finalize 完成后的实际参数。
3. `disk_replay`：再次读取本地 checkpoint，经过安装版本的 `DefaultModelLoader`、模型 `load_weights` 和 layerwise finalize，得到同一 TP rank 的实际推理参数。TP 切分、QKV 合并、MoE 排列由原生加载器处理，不手写猜测映射。Huanyu 音频参数使用已有桥接调用当前原生参数的 deferred `weight_loader`，避免 `load_state_dict` 对 meta 参数加载无效。

重载参考前会把内存参数填为 NaN，防止漏加载的参数保留 sync 数值而误判通过；不会修改磁盘 checkpoint。`initial_disk` 也必须与 `disk_replay` 一致，否则报告 `reference_reload_changes_disk_baseline`，不能把重载本身的差异当成正常。漏加载、meta、非有限值、缺 worker、缺 TP rank、版本接口不支持均不能汇总为 PASS。量化和 LoRA 不支持本次诊断。

**这是独立的权重诊断运行，完成后必须停止，不进行 validation 文字生成或训练。** 磁盘重载已经替换了内存参数，不能把随后生成的回答标成 sync 结果；driver 会以 `AUDIT_ONLY_STOP` 主动退出，是否通过以汇总报告为准。正常训练及普通文字 A/B 不启用此功能。完整哈希及再次读取模型会显著增加初始化时间和磁盘读取量；哈希按块转 CPU，不额外常驻第二份完整模型。

复用第 9 节已准备好的 `AB_ROOT`（含 `case.json` 等文件），保留相同模型和并行配置。先释放独立 serve 占用的卡，保持原来的 Ray 集群；下面的入口沿用 actor16 + RM16、rollout TP4、冻结 encoder/projector。`AUDIT_ROOT` 必须是尚不存在的新目录，由启动器创建，不覆盖已有文字结果：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
# AB_ROOT 保持原来那批 100 条数据的目录。
export AUDIT_ROOT="$PROJECT_ROOT/runs/huanyu_full_weight_audit_$(date +%Y%m%d_%H%M%S)"

RAY_ADDRESS=auto python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" rollout \
  --case-dir "$AB_ROOT" \
  --full-weight-audit-dir "$AUDIT_ROOT"

# 上一步预期以 AUDIT_ONLY_STOP 非零退出；在终端继续单独执行，不用 && 串联。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_full_weights.py" \
  --audit-dir "$AUDIT_ROOT" --expected-workers 16 --tp-size 4 \
  --output "$AUDIT_ROOT/summary.json"
```

每个 worker 写 `full_weights-*.json`，启动配置写 `full_weights.launch.json`。汇总器要求 16 个 worker、4 个完整 TP4 组，以及一致的 checkpoint 内容摘要；不能只查看某一个 rank 的 PASS。返回码 0/1/2 分别表示 PASS/FAIL/INCOMPLETE。参数中的 `sync_differs_from_disk` 定位同步后发生变化，`reference_reload_changes_disk_baseline` 要先检查原生重载与初始加载的差异。缺失证据需先解决，不能降低 expected-workers 来绕过。文件中的 `checkpoint_stream.rounding_tensors` 记录原始浮点权重转 BF16 会发生的舍入；这个统计本身不等于同步错误。

精度口径：Actor 通常保存 FP32 主参数，`OmniFSDPEngine.get_per_tensor_param()` 对 DTensor 执行 `full_tensor().to(torch.bfloat16)`，非 DTensor 原样导出。Huanyu 音频加载还会先将浮点 checkpoint 参数转 BF16，再随 Actor 模型转成 FP32。对于原本为 BF16、尚未优化器更新的有限值，BF16→FP32→BF16 应精确还原；此前 LN 尾元素损坏不是正常舍入。对于真正的 FP32 权重，转 BF16 会舍入，尤其要检查原生推理是否本来保留 FP32。这里保留各阶段实际 dtype，不通过统一强转 BF16 来掩盖差异。FP32 主参数也不代表所有前向计算都是 FP32；混精计算、VGG 内部 FP32 和 GDN FP32 cache 是不同层面的设置。

全量 PASS 只证明这次首同步中全部注册参数与原生 checkpoint 推理表示一致，仍不证明音频特征、prompt token IDs、执行算子或最终生成文字一致；未注册的临时缓存不属于此参数清单。合法的非有限值 buffer（例如 `-inf` 掩码）可按完整字节一致性通过，参数仍严格要求有限值。若 padding 或派生 buffer 导致参考重载与初始加载不一致，也要先检查这项差异，不能直接归因为同步损坏。原生加载器接口参考 [vLLM DefaultModelLoader](https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/default_loader/)；运行时接口不兼容会停止并报告，不退回部分参数检查。

本地验证：95 项相关 CPU 测试通过，其中 16 项使用真实 Torch 2.10 张量验证 TP 融合、参数别名、非 LN 损坏、漏加载、FP32/BF16 舍入、meta deferred loader 和报告汇总联调。真实 Huanyu/NPU 模型尚需在服务器运行此检查。

## 11. 100 条文字 A/B 同时检查完整 prompt embedding

新版 `aqa_text_ab.py prepare` 默认 `--embedding-check true`。两端仍输出文字，同时在真实推理 worker 的 `native.thinker.llm.model.forward` 入口保存**文本与音频合并后的完整 `inputs_embeds`**。服务器 Huanyu thinker 直接调用 `self.llm.model(...)`，因此这里不使用会被绕过的 `self.llm.forward()`，也不重新运行 encoder 构造替代结果。

每条样本使用包含 case ID 与 `row_XXXXXX` 的请求 ID。serve 通过 HTTP `request_id` 发送；rollout 在原本的 AgentLoop→server manager 请求中绑定同一标识，不添加问题文字、不改变采样参数。采集器读取实际 runner 请求的 prompt token IDs、音频占位范围，以及 LLM 调用的 positions。音频 mask 来自 `mm_features.mm_position`（有 `is_embed` 时使用实际 mask），不用 `<unk>` 数量猜测位置。输入已经由 `inputs_embeds` 给出时，外层可能传入未使用的全零 `input_ids`；记录会单独保留该实参，逻辑 token IDs 仍取实际调度请求。

此诊断要求原有 A/B 的 TP4、单请求、eager、关闭 prefix cache 和 chunked prefill。每个 TP rank 都保存完整 prompt，不能只比较 rank0。真实请求未经过采集点、只采到一部分 prompt、缺 rank、重复请求或身份不明都会报错或汇总为 `INCOMPLETE`，不会使用调用先后顺序猜测配对。初始化的无请求 warmup 和非 Huanyu judge 不属于采集范围。**没有“只采前两次”的限制，100 条都必须覆盖。**

旧实验没有完整 embedding，需在新目录重跑两端。源 Parquet、导出 WAV 和 seed 保持与此前实验一致时，仍选中相同的 100 条；保留旧目录用于查看先前回答：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export AB_ROOT="$PROJECT_ROOT/runs/aqa_embedding_ab_$(date +%Y%m%d_%H%M%S)"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" prepare \
  --parquet /opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --output-dir "$AB_ROOT" --model-path "$MODEL_PATH" \
  --sample-size 100 --seed 42 --prompt-mode system_audio --embedding-check true

# 终端一：启动带采集功能的 serve，保持前台。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --port 28889 --served-model qwen3.8
```

服务就绪后，在另一个终端设置相同的 `PROJECT_ROOT`、`AB_ROOT` 绝对路径，发送请求：

```bash
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --url http://localhost:28889/v1/chat/completions --model qwen3.8
```

请求完成后停止独立 serve，释放卡，保持原来的 Ray 集群，再运行 rollout（actor16 + RM16、冻结 encoder/projector、首次 Actor 同步照常执行）：

```bash
RAY_ADDRESS=auto python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" rollout \
  --case-dir "$AB_ROOT"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" compare --case-dir "$AB_ROOT"
```

输出增加以下文件：

| 路径 | 内容 |
| --- | --- |
| `embeddings/serve/*.json`、`*.npz` | serve 每样本、每 TP rank 的真实输入及来源 |
| `embeddings/rollout/*.json`、`*.npz` | rollout 对应输入 |
| `embedding_comparison.jsonl` | 每条的严格比较结果、各 rank 的首个差异位置和误差 |
| `embedding_summary.json` | `equal_samples`、`different_samples`、`incomplete_samples`、总体 `status` |
| `comparison.csv` / `comparison.jsonl` | 原文字结果旁增加 `embedding_status`、`embedding_strict_equal` |

严格相等要求 dtype、shape、完整 embedding 原始字节、逻辑 token IDs、positions、音频 mask 全部一致；不使用 `allclose` 或余弦相似度代替。BF16 按原始字节保存，不先转换 FP32。数值差异另报最大/平均绝对误差、首个差异元素，以及音频区和文本/特殊 token 区的差异数。原始 bytes 和 NPZ 均校验 SHA256，比较器逐样本读取，避免一次加载所有张量。

100 条全部通过、每条两端均有完整 TP4 记录且各 rank 也一致，才汇总 `PASS`。缺失证据为 `INCOMPLETE`；完整证据有差异为 `FAIL`。总 `compare` 返回码：0 为文字配对完整且 embedding 全部通过，1 为 embedding 有差异，2 为文字或 embedding 证据不完整。文字是否完全相同单独统计，不作为 embedding 成败条件。

日志关键词为 `[huanyu-embedding-ab]`；正常每条每个 rank 有 `status=complete scope=llm_inputs_embeds`。完整输入一致只能排除**本次这批样本在此输入边界之前产生输出差异**，不能证明权重、cache、算子和后续生成一定一致。探针会同步复制并写盘，因此此实验不能用于衡量正常吞吐量；普通训练不设置 `HUANYU_EMBEDDING_AB_DIR` 时不会开启。第 10 节全量权重诊断会关闭 embedding 采集，避免将磁盘重载后的结果误标为 sync。

请求 ID 的依据见 [vLLM ChatCompletionRequest](https://docs.vllm.ai/en/v0.28.0/api/vllm/entrypoints/openai/chat_completion/protocol/)；安装版本若改写成无法识别的 ID 或不提供所需运行时元数据，采集器会拒绝继续，而非生成错误配对结果。

本次本地验证：98 项相关 CPU 测试通过，覆盖请求身份传递、启动配置、真实 BF16 张量采集、全部 TP rank 比较，以及采集→NPZ→文字 A/B 汇总的串联测试。改动单个音频 embedding 元素会判为 FAIL；缺少一个 rank 会判为 INCOMPLETE。未运行真实 Huanyu/NPU 模型，100 条是否严格一致仍须以上述服务器运行结果为准。

## 13. serve / rollout 的 prefill、decode logits 和最终采样参数检查

在 embedding 和首次同步权重检查通过后，用 `--logits-check true` 新建一次 A/B 实验。两端仍使用同一 WAV、原始 system 信息和纯音频 user 输入，不额外提供问题文字。该实验只做正常首次权重同步后的验证，不做 optimizer 更新。

探针读取实际 worker 的 `CachedRequestState.sampling_params`，以及采样器收到的 `sampling_metadata`，不是 AgentLoop 的入参或启动命令。每个生成步骤记录三个边界：

| 字段 | 实际观察位置 |
| --- | --- |
| `model_logits` | 原生模型 `compute_logits` 返回后，完整复制当前词表输出 |
| `sampler_input` | 实际 `Sampler.forward` 执行前，可发现模型输出与采样器之间的修改 |
| `greedy_input` | 实际 `greedy_sample` 调用前，即应用 penalty / token mask 等处理后的分数 |
| `sampled_token_ids` | 实际 `SamplerOutput` 返回的 token；同时保存 `greedy_sample` 返回值 |

`step=0, phase=prefill` 是完整 prompt 的最后位置，用于生成第一个 token；`step>=1, phase=decode` 是后续逐 token 解码。**两个阶段均采集**，默认一直到 EOS 或 `max_tokens` 上限。引擎通常只对用于选词的位置计算全词表 logits；这里不会额外重跑 prompt 所有位置或使用另一种 forward 替代实际计算。

每个步骤还保存最终 temperature/top_p/top_k/seed/penalties/stop/EOS/min_tokens/max_tokens 等参数、`all_greedy/all_random`、实际采样元数据、当前生成前缀和词表布局。runtime 记录实际引擎配置、精度开关、执行函数及包装函数的源码指纹。参数既做两端互比，也与 `case.json` 的预期值比较：两端同时被覆盖成相同错误值也不能通过。logprobs 等输出诊断选项单列，避免将“是否返回分数”误认为采样温度差异。

服务器提供的 Ascend 实现存在 `enable_reduce_sample` 分支。探针读取实际开关：开启时保存每个 TP rank 的词表分片并检查全局 argmax；关闭时各 rank 记录完整词表 SHA256，TP0 保存完整向量。两端分片布局不同会明确报告，不能将本地分片直接当作完整词表比较。探针不自动切换此开关，以免掩盖实际差异。

先用4条数据，保留本次2048输出上限（允许EOS）。当前AQA启动预检要求至少一个train batch（4条），即使是val_only也不能只准备2条。新目录不能与旧实验混用；需两节点代码一致，Ray 按已有32卡验证环境启动。两端顺序运行，避免与已有服务争用同一组NPU。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export AB_ROOT="$PROJECT_ROOT/runs/huanyu_logits_ab_$(date +%Y%m%d_%H%M%S)"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" prepare \
  --parquet /opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --model-path "$MODEL_PATH" --output-dir "$AB_ROOT" \
  --sample-size 4 --seed 42 --prompt-mode system_audio \
  --max-tokens 2048 --embedding-check true --logits-check true

# 终端A：启动带相同精度策略和真实探针的serve，等待服务就绪。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --port 28889 --served-model qwen3.8

# 终端B：设置成终端A打印的同一个AB_ROOT绝对路径，再发送请求。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --url http://localhost:28889/v1/chat/completions --model qwen3.8

# serve请求全部完成后，在终端A用Ctrl-C退出本次服务、释放其NPU，再运行rollout。
RAY_ADDRESS=auto python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" rollout \
  --case-dir "$AB_ROOT"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_prompt_embeddings.py" --case-dir "$AB_ROOT"
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_logits.py" --case-dir "$AB_ROOT"
```

`--logits-max-steps 0`（默认）取本次 `max_tokens`，包含全部实际生成步骤。可以显式设为128做短窗口诊断；它只限制采集，不改变引擎生成长度。以约248K词表、FP32为例，一个边界每步约0.95MiB；4条、两端、每条2048步约15.2GiB，多个边界数值不同时还会增加。相同向量在同一步文件内去重，其他TP rank在非分片模式下只存摘要。先不要直接对100条做全程向量导出。设备到CPU复制是阻塞的，会减慢推理；不能用本次耗时衡量正常吞吐。

产物位于 `$AB_ROOT/logits/serve/`、`$AB_ROOT/logits/rollout/`（JSON元数据+无pickle的NPZ向量）；总报告是 `$AB_ROOT/logits_comparison.json`。日志关键词为 `[huanyu-logits-ab]`。读报告时关注：

- `sampling_parameter_differences`：两端最终参数/采样元数据不一致、与case预期不符，或实际贪心选择不成立。
- `runtime_differences`：引擎配置、reduce-sample、精度环境、源码/包装链差异。差异是线索，本身不是根因证明。
- `prefill` / `decode`：相同前缀下，各TP rank的三个边界逐字节SHA256比较；有向量的rank还计算数值误差、mask差异和top10。
- `first_different_logits`：最早发生分歧的步骤和边界；`model_logits` 指向模型前向，`sampler_input` 指向中间处理，`greedy_input` 指向采样前处理。
- `first_different_token`：首次实际选词分歧；`decode_different_prefix` 保留此后的分数摘要比较，但**不能**把不同前缀造成的差异认定为forward错误。若需继续定位，应另做固定同一前缀的实验。

状态 `PASS` 表示全部实际生成（有EOS或max_tokens终止证据）在本次检查范围内一致；`MATCH_CAPTURED_PREFIX` 只表示已采集窗口一致，退出码3；`DIFFERENT` 表示存在数值、选词或设置差异，退出码1；`INCOMPLETE` 表示缺rank/步骤、重复运行混入、缺实际hook、文件损坏或无法完整记录参数，退出码2。缺少decode时不会假装完成整段解码比较，除非首token确实结束了请求。若只有runtime差异而logits/token完全相同，不应据此认定数值错误。

该模式要求单请求、eager、无prefix cache、无chunked prefill、无投机解码，并显式关闭async scheduling，以核对实际CPU前缀与调度步骤。不会修改采样参数来“修正”被覆盖的值，也不会手动调用第二次forward或sampler。本地CPU验证不等同于NPU验证；不支持的服务器接口会报告不完整或停止，而不会产出假PASS。

本次106项CPU回归通过，覆盖探针、比较器、A/B启动器、数据准备及原embedding链路；包含原地修改logits、独立execute/sample生命周期、prefill/decode首次分歧、实际参数被覆盖、TP分片全局argmax、缺步骤/缺rank/损坏文件、提前中断与显式采集窗口的区分。服务器NPU执行尚待验证。

### 13.1 开启 Ascend 批量不变性后对比

准备case时增加 `--batch-invariant true`，两端共同启用 `additional_config.rl_config.enabled=true` 和 `enable_batch_invariant=true`；stage extras和外层参数均携带该配置，避免deploy合并时被覆盖。保持TP4、BF16、贪心、max_num_seqs=1，关闭prefix cache、chunked prefill和async scheduling，block_size=128。保留full精度策略和eager：当前Python reference与完整输入/logits探针不支持图捕获，不能直接叠加官方示例的PIECEWISE。

依照 [Ascend 批量不变性文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html)，两节点须已有匹配当前版本的自定义算子；只在运行时export编译变量不会补齐安装。先在两节点的训练Python环境检查：

```bash
python3 - <<'PY'
import torch
import batch_invariant_ops
print(torch.ops.batch_invariant_ops.npu_matmul_batch_invariant)
PY
```

该检查只验证扩展可导入/接口存在；失败时先按相同版本的Ascend源码安装算子，不能宣称测试已启用。Huanyu自定义模型的效果仍需实测。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export RM_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export NCCL_DETERMINISTIC=1
export LCCL_DETERMINISTIC=1
export AB_ROOT="$PROJECT_ROOT/runs/huanyu_deterministic_ab_$(date +%Y%m%d_%H%M%S)"
echo "$AB_ROOT"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" prepare \
  --parquet /opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet \
  --export-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --model-path "$MODEL_PATH" --output-dir "$AB_ROOT" \
  --sample-size 4 --seed 42 --prompt-mode system_audio --enable-thinking true \
  --max-tokens 2048 --batch-invariant true \
  --embedding-check true --logits-check true --logits-max-steps 0

# 终端A，等待服务就绪。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --port 28889 --served-model qwen3.8

# 终端B设为同一个PROJECT_ROOT和AB_ROOT后执行。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" serve \
  --case-dir "$AB_ROOT" --url http://localhost:28889/v1/chat/completions \
  --model qwen3.8 --timeout-s 3600

# 请求完成后Ctrl-C退出本次serve、释放NPU；Ray沿用已有32卡环境。
set -o pipefail
RAY_ADDRESS=auto python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" rollout \
  --case-dir "$AB_ROOT" 2>&1 | tee "$AB_ROOT/rollout_console.log"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/aqa_text_ab.py" compare --case-dir "$AB_ROOT"
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/compare_huanyu_logits.py" --case-dir "$AB_ROOT"

grep -E 'huanyu-determinism|Enabling batch-invariant|skipping batch-invariant' \
  "$AB_ROOT/serve_console.log" "$AB_ROOT/rollout_console.log"
```

启动器为两端设置HCCL=true、ASCEND_LAUNCH_BLOCKING=1、NCCL=1、LCCL=1，并显式转发Ray runtime_env；底层脚本保留外部ASCEND_LAUNCH_BLOCKING值。`ASCEND_LAUNCH_BLOCKING=1`是算子同步执行/定位异步报错开关，会影响性能，不是硬件数值确定性保证，见[torch_npu说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/ASCEND_LAUNCH_BLOCKING.md)。NPU通信走HCCL；NCCL_DETERMINISTIC按请求保留，不作为NPU确定性依据。

`VLLM_BATCH_INVARIANT`在共享Ray环境先清为0，随后由Actor引擎的RL配置在创建worker前启用；避免把GenRM也切到需要不同cache配置的模式。已启动推理worker的`[huanyu-determinism] actual`与`logits/*/runtime-*.json`才是最终证据：VLLM_BATCH_INVARIANT=1、HCCL_DETERMINISTIC=strict（原生初始化从true升级）、其余三项=1、batch_invariant_registered=true、torch_deterministic_algorithms=true。仅requested日志或launch.json不够。未知版本无法确认注册则INCOMPLETE；两端同时未启用也不能PASS。注册成功并不证明每个自定义算子都满足不变性，最终仍比较prefill和所有decode的实际logits/选词。

135项CPU回归通过，包括最终环境观测、两侧均未生效的反例、Ray传递和生成deploy中的参数保留；尚未在NPU验证。此前HTTP500 `Stage-0 has no live replica`需另查原始worker异常，开启这些参数本身不能证明该故障已修复。

### 13.2 同一个 serve 中，两条音频各重复20次

`repeat_huanyu_serve.py` 固定复用已有case的 `row_000899`、`row_000418`，各20次、共40次。默认仅记录并比较文字，不采集logits或embedding，也不生成张量NPZ。保留原system、音频、thinking、采样设置；不把ground_truth或问题描述加回请求。顺序为899→418→899→418，客户端逐条等待完整响应，服务端max_num_seqs=1。仅请求ID不同；`manifest.jsonl` 的 `source_sample_id/repetition` 保存对应关系。准备时不重新随机抽样，不重新解码/编码WAV。

这次测试的是同一serve内部的文本可重复性，不启动verl训练。环境准备沿用已有环境，需停止占用同一卡/端口的旧服务，再用启动器加载新case。测试期间不要向该服务发送其他请求，不要中途重启。默认模式不观测worker身份、最终采样元数据或数值运算；报告会将这些明确标为NOT_OBSERVED/NOT_COLLECTED，不能由文本PASS推断数值精度相同。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
# AB_ROOT必须是包含上述两条样本的原case目录，其case_id为00c6bf857f4a4b67a1043d719ef6a97f。
: "${AB_ROOT:?请设置原A/B case目录}"
export REPEAT_ROOT="$PROJECT_ROOT/runs/huanyu_serve_repeat_$(date +%Y%m%d_%H%M%S)"

python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/repeat_huanyu_serve.py" prepare \
  --source-case "$AB_ROOT" --output-dir "$REPEAT_ROOT" \
  --samples row_000899 row_000418 --repeats 20 --batch-invariant false --logits-check false
echo "$REPEAT_ROOT"

# 终端A：启动服务，等待就绪。使用case中的checkpoint，TP=4、eager、BS=1。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/launch_huanyu_aqa_text_ab.py" serve \
  --case-dir "$REPEAT_ROOT" --port 28889 --served-model qwen3.8

# 终端B：PROJECT_ROOT和REPEAT_ROOT设置为终端A的相同值。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/repeat_huanyu_serve.py" request \
  --case-dir "$REPEAT_ROOT" --url http://localhost:28889/v1/chat/completions \
  --model qwen3.8 --timeout-s 3600

# 等40次请求结束后比较；缺失请求或请求内容变化不能PASS。
python3 "$PROJECT_ROOT/examples/qwen3_6_huanyu/repeat_huanyu_serve.py" compare \
  --case-dir "$REPEAT_ROOT"
```

默认关闭此前不兼容的VLLM_BATCH_INVARIANT，启动器仍强制HCCL_DETERMINISTIC=true、ASCEND_LAUNCH_BLOCKING=1、NCCL_DETERMINISTIC=1、LCCL_DETERMINISTIC=1，保持eager、单请求、关闭prefix cache、chunked prefill和async scheduling。它们不保证所有算子逐bit确定。若已修复原生batch-invariant算子兼容性，可在准备新case时改成 `--batch-invariant true`。纯文本模式只保留请求配置和启动日志，不宣称验证了worker最终生效设置。

输出均在REPEAT_ROOT：

- `serve/row_XXXXXX.json/.txt`：40次独立响应；此处row编号是请求序号，原数据行号和第几次重复见manifest以及JSON中的 `source_sample_id/repetition`。
- `repeat_summary.json`：`scope=text_only`，两条音频各自的 `distinct_text_variants`（文字变体数）、相对第1次一致的请求数、每次状态；第1次自身也计入20次。`logits_status/embeddings_status=NOT_COLLECTED`。

默认模式的 `PASS` 要求40次HTTP响应完整、请求内容符合冻结的case，且各音频的回复文字和单独reasoning字段完全相同、终止状态一致。`DIFFERENT`为输出或终止状态不同；`INCOMPLETE`为缺记录、失败请求、内容被修改等，不能得出一致结论。文本相同不等于回答正确，也不证明logits相同。纯文本比较使用标准库，不读取旧logits/embedding产物；请创建新REPEAT_ROOT，不要直接复用之前启用张量采集的case。

以后需要数值排查时，可在准备新case时显式使用 `--logits-check true`，才会启用原有完整logits与embedding探针，并生成 `logits/serve/`、`embeddings/serve/`、`repeat_comparisons/`。该模式还检查每个TP rank、完整prefill/decode、实际采样设置和worker身份，报告首个分歧位置。全词表logits落盘可能占用数十至数百GiB；默认纯文本模式没有这项开销。本地CPU测试覆盖40次串行请求及两种比较模式；实际NPU结果需运行后读取报告。
