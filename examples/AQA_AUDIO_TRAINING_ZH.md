# AQA 在线音频训练与冻结开关

更新：2026-11-09。完整脚本清单、环境/Ray 命令、调用链和参数优先级见[启动 Wiki](README.md)。本页只补充 AQA 的数据与音频训练约束。

## 1. 入口与资源

推荐在两节点分别export相同模型、数据与训练参数，设置各自`VC_TASK_INDEX`后，执行`bash "$SHELL_ROOT/run_huanyu_32card_job.sh"`；`export HUANYU_JOB_MODE=aqa_unfreeze`选择AQA入口，只有head提交训练。完整export示例见[启动Wiki](README.md#4-训练命令先-export再启动-job)。已有环境/Ray后，也可在head沿用export执行`bash "$SHELL_ROOT/run_huanyu_aqa_audio_train.sh"`，此直接入口不启动或停止Ray。

```text
run_huanyu_aqa_audio_train.sh
  ├─ job_bootstrap.py check-aqa：数据预检，不解码全部音频
  ├─ HUANYU_REWARD_MODE=RM → run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh
  │                  → run_qwen36_huanyu_gspo_npu.sh
  └─ HUANYU_REWARD_MODE=rule_based → run_qwen36_huanyu_gspo_npu.sh
```

`aqa_unfreeze`是入口模式名，仍可通过音频组件变量选择冻结模式。

默认模型为 `/opt/huawei/dataset/model_gy/downloaded_models/2748`。Actor `NUM_NPUS=16, NNODES=1`、rollout `ROLLOUT_TP=4`；GenRM默认另用16卡，总共32卡。BS4、mini4、N4，静态microbatch默认1；prompt4096、response2048、引擎长度默认6272。`ROLLOUT_NUM_WORKERS`未设置时AgentLoopWorker数等于Actor卡数/rollout TP，默认4。

job前 `export HUANYU_REWARD_MODE=rule_based` 关闭独立奖励池，改用字符 bigram Dice 规则打分，可在16卡运行；`export HUANYU_REWARD_MODE=RM` 使用GenRM。`ruled_based` / `rule` 和 `genrm` 分别是两种模式的兼容写法。统一变量显式设置时优先于旧 `RM_ENABLE=0/1`，未设置时仍接受旧变量。规则分和GenRM语义分是不同训练目标，不能直接比较数值。

在两节点common job中切换规则模式时，默认Actor16保持不变，另16卡不再分配给RM。若要32卡全部做Actor，设置 `export NUM_NPUS=16 NNODES=2`，并调整BS/mini/N满足32卡约束，例如BS8/mini8/N4。独立16卡实验可在已准备的Ray集群上直接运行AQA入口；common job仍按两台16卡物理节点管理环境和Ray。

## 2. 数据选择

默认读取 `/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet` 和 `validation.parquet`。在job前统一`export TRAIN_FILE=... VAL_FILE=...`显式选择文件；通用变量优先，未设置时才回落到`HUANYU_AQA_TRAIN_FILE`/`HUANYU_AQA_VAL_FILE`或默认路径。入口会对实际选择的train/validation两份数据做预检，即使关闭训练前验证，也不能把含图像/视频的混合validation传入AQA模式。

每条样本必须含音频、不能含图片/视频，并保留问答和ground_truth字段。OC-RL 4B全量数据为纯VL，不能使用这个入口。原始OmniMix需先用模型包中的 `extract_aqa.py` 分别筛选 train 和 validation，保留划分；脚本精简不改写既有parquet。

## 3. Encoder / projector 分别冻结或训练

| 期望 | 环境变量 |
|---|---|
| 两者都训练，AQA默认 | `HUANYU_FREEZE_AUDIO=false`，且未显式指定组件变量 |
| 两者都冻结、在线编码 | `HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false` |
| 只训练projector | `HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=true` |
| 只训练encoder | `HUANYU_TRAIN_AUDIO_ENCODER=true HUANYU_TRAIN_AUDIO_PROJECTOR=false` |
| 两者都训练，显式 | `HUANYU_TRAIN_AUDIO_ENCODER=true HUANYU_TRAIN_AUDIO_PROJECTOR=true` |

组件变量优先于旧的整体 `HUANYU_FREEZE_AUDIO`；未指定的组件继承整体开关。只要一个组件参与训练，最终整体冻结标记就是false。shell只接受小写 `true`/`false`，并转换成以下正式配置：

```text
actor_rollout_ref.model.override_config.huanyu_freeze_audio
actor_rollout_ref.model.override_config.huanyu_train_audio_encoder
actor_rollout_ref.model.override_config.huanyu_train_audio_projector
```

这些配置由AQA入口追加为最终约束，不要用矛盾的Hydra参数绕过前置检查。`MINICPMO_TUNE_AUDIO=1` 是旧兼容变量，不是切换模式的入口。视觉塔始终冻结；语言模型仍参与训练。

冻结音频默认在线编码。缓存模式使用`export HUANYU_AQA_USE_PRECOMPUTE=1 HUANYU_AQA_CACHE_DIR=/新的匹配缓存目录`，入口会冻结两者并设置实际`HUANYU_EMBED_CACHE_DIR`。训练中的音频参数变化会让旧embedding失效，而且缓存没有encoder/projector计算图，不能同时缓存并训练音频组件。

只训练projector的两步示例：

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export HUANYU_REWARD_MODE=RM
export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=true
export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=4 TOTAL_TRAINING_STEPS=2
export TRAINER_LOGGER='[console]'
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

## 4. 数据与梯度链路

```text
音频路径/bytes/zarr member → 16kHz单声道波形
 → HuanyuProcessor：30秒分块、零token短尾补静音、40维fbank、有效长度和token槽位
 → 原生Huanyu VGG/Conformer → fold/projector → LLM空间音频embedding
 → 训练forward注入槽位 → 文本主干log_probs → GSPO loss
 → 根据组件冻结设置计算梯度 → optimizer → rollout权重同步
```

训练与rollout使用相同原生音频实现和权重命名。冻结路径使用临时BF16参数视图，训练路径使用可微视图；不替换FSDP或优化器持有的主参数。音频权重加载严格校验键名与shape；不能用忽略missing/unexpected的方式掩盖40/128维或projector错位。

短尾补零策略同时作用于训练、预计算和实际rollout波形，见[短尾说明](docs/audio_short_tails.md)。视频配对及短尾策略更新都会影响缓存契约；旧manifest不自动复用，不手改digest。

## 5. 动态batch、显存和诊断

AQA默认三个阶段的dynamic batching都关闭。在job前统一打开，并设置训练/生成的不同token预算：

```bash
export HUANYU_JOB_MODE=aqa_unfreeze
export AQA_DYNAMIC_BSZ=true MAX_TOKEN_LEN_PER_GPU=8192
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 ROLLOUT_MAX_NUM_SEQS=16
bash "$SHELL_ROOT/run_huanyu_32card_job.sh"
```

`MAX_TOKEN_LEN_PER_GPU`管训练/log-prob microbatch；`ROLLOUT_MAX_NUM_BATCHED_TOKENS`管生成引擎调度。三个阶段也可分别export `ACTOR_DYNAMIC_BSZ`、`ROLLOUT_LOG_PROB_DYNAMIC_BSZ`、`REF_LOG_PROB_DYNAMIC_BSZ`，显式值优先于`AQA_DYNAMIC_BSZ`。BS/mini/N需满足`BS % mini == 0`、`mini×N % Actor卡数 == 0`、`mini % rollout副本数 == 0`，且N至少2。

静态模式下，`PPO_MICRO_BATCH_SIZE_PER_GPU`控制Actor每次microbatch条数，`LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`控制rollout/ref的log-prob microbatch，均默认1。Actor每rank轨迹数`mini×N / Actor卡数`必须能被Actor microbatch整除。动态模式主要使用token预算；这些值不代表生成引擎的并发上限。

AQA默认SP1、FSDP2、gradient checkpointing，默认关闭activation offload及前向/反向prefetch；`param_offload=true`、`optimizer_offload=false`、`offload_policy=false`。offload环境接口为`ACTOR_PARAM_OFFLOAD`、`ACTOR_OPTIMIZER_OFFLOAD`、`ACTOR_OFFLOAD_POLICY`、`REF_PARAM_OFFLOAD`、`ACTIVATION_OFFLOAD`，prefetch用`ACTOR_FORWARD_PREFETCH`/`REF_FORWARD_PREFETCH`。其中参数offload与FSDP CPUOffloadPolicy不是同一个开关，不能仅看“offload关闭”就判断峰值显存。

默认 `ROLLOUT_GPU_MEM_UTIL=0.6`，GenRM独立 `RM_GPU_MEM_UTIL=0.7`。full precision固定多个reference开关；数值/性能对照需先选 `HUANYU_PRECISION_PROFILE=custom`，详见启动Wiki。配置变化后需要新worker，不会改变已经运行的训练进程。

核对 `[huanyu-aqa] freeze_audio=... mode=...`、组件开关、资源池、batch和prompt/response日志，以及最终Hydra配置。`[huanyu-output] console=...` 指明本次console.log；不是固定的旧logs/0。阶段耗时见[gen计时](docs/gen_timing.md)和[forward profile](docs/forward_profile.md)。CPU回归验证接口契约，不能替代NPU梯度、数值对齐与吞吐验收。
