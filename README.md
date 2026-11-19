# AVQA-R1-6K 原生 Qwen3-Omni Thinker 训练

默认模型为 `Qwen/Qwen3-Omni-30B-A3B-Instruct`，与公开 AVQA LoRA 实验一致。Actor 和 rollout 均使用 Thinker，输入为图片、音频和问题，输出为文本。此次对照关闭 route replay。[公开 AVQA 配方](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh)

## 单节点 16 NPU 一键训练

GitHub 的 `aqav` 分支在根目录发布训练配方。首次部署时，可以在已有训练项目中把分支克隆到新的脚本路径：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/ccb_data/code/xy_rl_omni-develop_qwen36_whisper
AQAV_DIR="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open"
git clone --single-branch --branch aqav https://github.com/ZiqiGuan892/ext.git "$AQAV_DIR"
```

在 worker-0 训练容器中，从存放 `launch.sh` 的目录执行下面的命令。启动器按
`PROJECT_ROOT` 查找 `verl_omni_ext/models/qwen3_6_huanyu/aqav_open/run.sh`，
并使用 `examples/qwen3_6_huanyu/` 下的环境与 Ray 脚本。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/ccb_data/code/xy_rl_omni-develop_qwen36_whisper
bash ./launch.sh
```

模型默认从 `/opt/huawei/dataset/model_gy/downloaded_models/Qwen3-Omni-30B-A3B-Instruct` 加载。如果权重放在其他挂载目录，启动前将 `AVQA_MODEL_PATH` 指向含 `config.json`、tokenizer 和权重的本地目录。训练 Python 需能导入 `qwen_omni_utils.process_mm_info`。启动器在启动或复用 Ray 前检查模型类型和音视频处理依赖。

平台可继续使用外置的 `launch.sh`。部署时同步本目录和平台实际执行的启动脚本。平台附加的 `--名称=值` 参数继续透传，`--PROJECT_ROOT=…` 定位训练配方，`--SHELL_ROOT=…` 定位环境和 Ray 脚本。

| 配置 | 一键训练默认值 |
|---|---|
| 训练与采样 | GSPO、LoRA r=32 / alpha=64、学习率 3e-6、batch 128、每题采样 16 次 |
| 资源与显存 | 单节点 16 NPU、FSDP2、rollout TP=2、rollout 图模式、Actor/Ref 参数卸载开启、Actor 优化器卸载关闭 |
| 数据 | 读取 `/opt/huawei/dataset/infra_data/data/avqa_r1_6k_2748_filtered/{train,validation}.parquet` 全部样本，跳过在线长度过滤 |
| 进度与日志 | 从初始模型训练 10 个 epoch，每 10 步验证、每 50 步保存，W&B 离线。启动时打印新建的 `RUN_DIR` 和 `console.log` 路径 |

复用 2748 筛选过的样本可以固定两次实验的题目。原生处理器会重新编码图片、音频和文本，个别样本的 prompt 仍可能超过 4096 token。长度检查按原配方设置为 `data.truncation=error`。原生 Thinker 使用自己的前向实现，2748 适配器专用的 GDN、MoE 和按补齐长度拆微批的补丁仅适用于旧实验。卡死探针和 Host 内存采集均关闭。

## 已核对的来源与尚未验证的内容

- [官方实验说明](https://github.com/verl-project/verl-omni/tree/main/examples/gspo_trainer) 将曲线标为 `release/v0.2.0`、Qwen3-Omni-30B-A3B-Instruct、4×H800、LoRA r=32；报告训练 reward 约从 0.73 到 0.94，验证 reward mean@1 达到 0.877、348 step。
- [公开 GPU AVQA 启动脚本](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh) 是本目录训练参数的来源；当前 `main` 不等于该 W&B run 的固定代码版本。
- [原始数据](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K/tree/main) 是图片 + 音频 + 选择题文本，不是此前的 `aqa_dataset` 或视频数据。原始划分为 4,490 train / 1,911 valid；实际转换与长度过滤后的数量必须以日志为准。[原作者项目](https://github.com/HarryHsing/EchoInk)
- 下载锁定数据 revision `1c380427f66407f9f4ece354d84f804e4a16e522`，归档 SHA256 为 `6ad5d4bc00a3367aa36d2b0ed062a0a53c8d701242110c43826541138522f388`。归档约 5.47 GB，需额外保留解压空间。
- 同目录 `avqa.py` 和 `choice_reward.py` 分别是官方 `examples/gspo_trainer/data_process/avqa.py`、`verl_omni/utils/reward_score/choice_reward.py` 的原样副本，保留版权和许可声明。`prepare.py` 和 `run.sh` 直接引用本目录文件，转换与奖励加载不再依赖仓库根目录的对应文件。保留原 prompt、选项、带 `<answer>` 标签的 ground_truth；比较第一个 answer 标签的内容，完全相同得 1，否则得 0。没有 GenRM、格式奖励或长度奖励。

尚未直接取得 [W&B run](https://wandb.ai/verl-omni/gspo_demo/runs/kzzrq9pr) 的完整 config/history；访问该页面的浏览器流程被自动审批拦截，原因是涉及未授权的 `forge.coreweave.com` 域名，已停止访问。以上曲线数值来自官方 README，不能称作已独立复核的 run 数据。

本地测试只验证命令构造、数据标签、媒体路径和奖励契约。尚未在服务器加载 2748、验证 LoRA 梯度/权重同步或获得 reward 曲线。

### 2748 历史对照与排障记录

以下 2748 命令用于复现旧实验。单独调用 `run.sh` 前在同一终端设置模型族和对应权重即可选择旧适配器：

```bash
export AVQA_MODEL_FAMILY=huanyu
export AVQA_MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
```

#### 2026-11-16 重新测试 2748

本次沿用上述 LoRA 配方，保持 BS128、n16、LoRA32、TP2、16 NPU 和 choice reward，
通过 `AVQA_OPTIMIZED=1` 开启融合 GDN、RMSNorm、MoE 和 rollout 图模式。
此模式使用 `kernel_graph` 精度档案，同时关闭与融合 GDN 冲突的 FP32 参考实现。
不设置该变量时默认 `AVQA_OPTIMIZED=0`，保留原配方的算子路径，作为对照。

route replay 默认关闭（`AVQA_REPLAY=0`）。用户已反馈在其服务器验证 replay 无问题，
现在可显式设置 `AVQA_REPLAY=1 AVQA_OPTIMIZED=1 HUANYU_GRAPH_MODE=1` 开启，保持 LoRA32。
入口同步设置 driver/worker 的 `HUANYU_EXPERT_REPLAY=1`、
`enable_rollout_routing_replay=true`、`enable_return_routed_experts=true`。
回放依赖融合 MoE；`AVQA_REPLAY=1 AVQA_OPTIMIZED=0` 会提前报错。

replay 开启时须已安装 `pyproject.toml` 声明的 `huanyu_r3` general plugin；此脚本不安装依赖。
若已有 `VLLM_PLUGINS` 白名单，只追加 `huanyu_r3`，保留 Ascend 等原有插件；未设置白名单时
继续自动发现。不要只 export `HUANYU_EXPERT_REPLAY=1`，AQAV 使用专用 `AVQA_REPLAY` 开关，
避免继承其他任务的环境。replay debug、卡死与数值探针仍默认关闭。

rollout 图模式未设置时默认开启；`AVQA_OPTIMIZED=1` 要求 `HUANYU_GRAPH_MODE=1`，
显式设置为 0 会提前报错。训练侧 `torch.compile` 仍关闭。
HCCL 明确使用 `aiv / AIV`，调用新探针检查环境一致性；该检查不代表设备通信算法实测。

先同步最新 Huanyu 代码、本次修改的 `run.sh` 和 `verl_omni_ext/probes/hccl_policy.py` 到服务器。
已有原始 parquet 无需重新下载或转换。已离线过滤的数据只有在 tokenizer、模板、
媒体预处理和长度上限一致时才继续复用；未确认一致时保留默认长度过滤。

环境和 Ray 已准备好时，可以先用完整 BS/N 跑 10 step，并对比训练前和第 10 步的验证：

```bash
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open"
export RUN_DIR="$PROJECT_ROOT/runs/avqa_2748_kernel_graph_moe_replay0_step10_$(date +%Y%m%d_%H%M%S)"
AVQA_OPTIMIZED=1 HUANYU_GRAPH_MODE=1 \
bash "$AVQA/run.sh" \
  trainer.total_training_steps=10 \
  trainer.val_before_train=true \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  'trainer.logger=[console]'
```

这个重测窗口在上述算子/图优化之外，只改变总步数、首次验证和保存频率；每个训练 batch 仍有
128×16=2048 条轨迹，并非此前 BS16/n8 的 AQA 测试。10 step 无法保证 reward 上涨。
要继续完整 10 epoch 实验，从原始模型重新执行
`AVQA_OPTIMIZED=1 HUANYU_GRAPH_MODE=1 bash "$AVQA/run.sh"`，不复用重测目录。
验证集、奖励和采样参数保持一致，才能解释 reward 的前后变化。

运行后检查 `console.log` 中的 `[avqa] optimized=1 ... replay=0/1` 与本次开关一致，以及：

```bash
grep -aE '\[avqa\]|\[huanyu-kernel\]|trainable params|step:|Initial validation metrics|val-core/avqa_r1_6k/' "$RUN_DIR/console.log"
```

应有 `actor GDN fused kernel installed`、`actor npu_rms_norm installed`、
`actor fused MoE installed ... replay=off/on` 和 `graph mode requested; resolved cudagraph_mode=`。
replay 开启时还应看到 `[huanyu-replay] engine hook installed`、`server patch installed`
及 EngineCore/worker 的 `plugin applied` 日志。
这些日志用于确认安装和配置；图实际回放、LoRA 梯度、数值精度及吞吐仍需服务器运行验证。
本地 CPU 测试验证原配方参数、优化环境与实际 worker 精度策略一致、replay 开关三处一致、
奖励及数据转换/过滤契约，不代表已在 NPU 上完成训练。

## 参数和适配差异

| 配置 | 本入口 |
|---|---|
| train batch × rollout n | 128 × 16 = 2048 条轨迹/批，增加卡数不增加 global batch |
| PPO mini batch / dynamic token budget | 16 / 20480 |
| 输入 / 输出上限 | 4096 / 12288；引擎 prompt 预算 4160 |
| LoRA | r=32、alpha=64、FP32 adapter；更新前合并给 rollout |
| 目标层 | `q_proj,k_proj,v_proj,o_proj`；视觉、音频 encoder/projector 冻结 |
| GSPO / 学习率 / weight decay | GSPO，3e-6，0.01 |
| clipping / loss aggregation | 3e-4、4e-4、10；seq-mean-token-mean |
| Actor base / Ref / 参数、优化器卸载 | BF16 / BF16 / 开启 |
| KL loss / KL reward | 均关闭 |
| TP / 副本 | 保持 TP=2；16 卡共置 Actor/rollout，理论 8 副本；无奖励卡池 |
| rollout 显存预算 | 0.7，保留该实验设置 |
| prefix cache / graph sizes | 关闭 / `[1,2,4,8,16,32,64,128,256]` |
| validation | n=1、temperature=1、top_p=0.7，保持原脚本显式参数 |
| 训练 / 验证 / 保存 | 10 epoch / 每 10 step / 每 50 step |

未在官方启动脚本指定的字段，继续使用服务器框架默认值。特别是训练 `temperature/top_p/top_k`、验证 `do_sample`、`algorithm.global_std`、optimizer scheduler 等，必须查看 `resolved-config.txt`。例如当前上游验证 `do_sample=false`，不能只看到 `temperature=1` 就认定验证采用随机采样。没有原 run 完整配置时，不声称这些隐含默认值逐项复现。

2748 的 HF 架构是 Qwen3.5-MoE + Huanyu 音频前端。其训练适配器要求 `use_remove_padding=false`；相同 LoRA 名称只匹配该模型实际存在的层，不自动扩展到 GDN 或 MoE experts。`prepare.py check` 只从 checkpoint 索引检查候选名称，真实可训练参数仍需看训练日志。

Ascend 适配使用 Huanyu dataset/architecture/pipeline、`sdpa` 训练后端，关闭训练侧 torch.compile，并分块计算熵。GDN 的 HF fallback 保留 FP32 状态更新保护；这不等于全部模型权重 FP32。推理默认启用图模式，不继承之前 `full` 精度实验的参考算子开关；也不继承 embedding cache、AQA audio-only 限制或 profiler。`Q36_CKPT_DIR` 显式传给 driver 和 Ray workers。

这保留了公开配方的优化目标和显式超参数，但模型、tokenizer、媒体编码、长度过滤、LoRA 覆盖层、设备和运行时均有差异，不能保证达到官方相同 reward。

## 服务器执行

对于 2748 历史配方，在已有环境和 Ray 的 head 容器执行，并使用上节的 `AVQA_MODEL_FAMILY=huanyu`。同步整个 `aqav_open` 目录及现有 Huanyu 插件，所需脚本包括：

```text
run.sh
huanyu_2748.sh
prepare.py
avqa.py
choice_reward.py
filter_data.py
```

不安装或替换服务器的 torch/CANN/vLLM 栈。Python 需要已有 `huggingface_hub`、`pandas`、`pyarrow`；训练还需要镜像的 PEFT/音视频依赖。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export PYTHON_BIN=/usr/local/python3.12.13/bin/python3
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open"

# 1. 下载、校验、解压和转换；已有归档可追加 --archive /path/AVQA_R1.tar.gz。
"$PYTHON_BIN" "$AVQA/prepare.py" prepare

# 2. 检查标签、图片/音频路径和 2748 的 LoRA 候选层，不加载权重。
"$PYTHON_BIN" "$AVQA/prepare.py" check --model "$AVQA_MODEL_PATH"

# 3. 不启动训练，检查服务器当前版本能否解析全部配置。
bash "$AVQA/run.sh" --config > /tmp/avqa_2748_resolved_config.txt
```

数据默认写到 `/opt/huawei/dataset/infra_data/data/avqa_r1_6k`。parquet 记录的是绝对媒体路径，参与训练的容器必须能读取这些路径；下载后不要只移动 parquet。

本入口只接受专用路径变量 `AVQA_DATA_DIR`、`AVQA_TRAIN_FILE`、`AVQA_VAL_FILE`、`AVQA_MODEL_PATH`，避免旧终端导出的通用 `TRAIN_FILE` / `MODEL_PATH` 把本次任务改回视频、AQA 或 Whisper。自定义数据目录时，给 `prepare.py` 的两个子命令也传相同的 `--data-dir`；自定义模型时给 `check` 传相同的 `--model`。

### 一次离线过滤，后续训练复用

`prepare.py` 的转换和媒体路径检查不包含长度过滤。新增的 `run.sh --filter` 复用本脚本的 Hydra 配置，通过训练使用的 `OmniModelConfig` 构造 2748 tokenizer / Huanyu processor，再实例化服务器实际安装的 `Qwen36HuanyuOmniRLHFDataset`。长度判断直接使用该数据集继承的过滤函数，不另外估算图片、音频 token 数；默认保留展开后不超过 4096 token 的样本。

该模式不启动训练、不连接 Ray、不加载 encoder / LLM 权重。默认 8 个 CPU 进程，`AVQA_FILTER_WORKERS` 可调整；音视频解码和前端仍需镜像中的依赖。训练开始时的过滤通常是每次作业初始化执行一次，并非每个 step 扫描整个数据集。

在能访问原 parquet、媒体和 2748 模型目录的容器执行：

```bash
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open"
FILTER_DIR="/opt/huawei/dataset/infra_data/data/avqa_r1_6k_2748_filtered_4096_$(date +%Y%m%d_%H%M%S)"

(
  set -o pipefail
  PYTHON_BIN=/usr/local/python3.12.13/bin/python3 \
  AVQA_FILTER_OUTPUT_DIR="$FILTER_DIR" AVQA_FILTER_WORKERS=8 \
  bash "$AVQA/run.sh" --filter 2>&1 | tee "${FILTER_DIR}.log"
)
```

输入沿用 `run.sh` 的路径变量，默认 `/opt/huawei/dataset/infra_data/data/avqa_r1_6k/{train,validation}.parquet`。自定义路径、模型或模板等配置应在离线和训练命令中一致设置。若不指定 `AVQA_FILTER_OUTPUT_DIR`，输出默认是 `${AVQA_DATA_DIR}_filtered`（未设置数据目录时使用上述默认输入目录加 `_filtered`）。输出目录必须尚不存在；重试使用新目录，原 parquet 和媒体文件保持不变。

输出内容：

- `train.parquet`、`validation.parquet`：从原始文件按筛选索引取行，保留 prompt、媒体路径、奖励标签、原顺序和 split 边界；不写入 embedding，也不将媒体复制到新目录。
- `filter_manifest.json`：运行状态、各 split 的输入/保留/剔除数量、耗时、文件哈希、模型配置文件哈希及实际过滤实现来源。
- `resolved-config.yaml`：本次配置；`*.rows.jsonl`：逐行保留结果及原始行号；`work/*.indexed.parquet`：临时带行号输入，供核对。
- `${FILTER_DIR}.log`：上述命令保存的完整控制台进度及处理异常。

上游过滤函数可能把媒体读取/处理异常也视为不合格样本，因此报告的 `rejected_rows` **不等同于全部超长**；逐行审计统一标为 `upstream_rejected_overlong_or_processing_error`，具体异常看日志。任一 split 全部被剔除时脚本报错，不输出该 split 的空训练文件。

确认两份数据都完成再使用：

```bash
/usr/local/python3.12.13/bin/python3 - "$FILTER_DIR/filter_manifest.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    report = json.load(f)
assert report["status"] == "complete", report
for split in ("train", "validation"):
    s = report["splits"][split]
    print(f'{split}: {s["input_rows"]} -> {s["kept_rows"]}, 剔除 {s["rejected_rows"]}')
PY

# 正式训练：显式指定两份过滤后的文件，关闭重复扫描。
AVQA_TRAIN_FILE="$FILTER_DIR/train.parquet" \
AVQA_VAL_FILE="$FILTER_DIR/validation.parquet" \
bash "$AVQA/run.sh" data.filter_overlong_prompts=false
```

smoke 和基线命令也可使用相同的两个文件变量及 `data.filter_overlong_prompts=false`。这只省掉重复长度过滤；训练仍需逐批解码媒体、生成特征并做模型前向。更换 tokenizer、模板、多模态预处理规则或输入长度上限后应重新过滤，不要对原始未过滤文件直接关闭过滤。离线入口拒绝 `train_max_samples` / `val_max_samples` 的正数限制，避免把调试子集误当作完整数据集。

本地 CPU 测试覆盖导出行、嵌套字段、标签、顺序、文件不覆盖和入口分流；使用模拟数据集筛选结果，不代表已在服务器验证真实媒体长度。服务器完成情况以 `filter_manifest.json` 的 `status=complete` 和日志为准。

### 训练检查和完整运行

先做两步运行检查。下面缩小了数据、BS、N 和输出长度，**仅用于检查图初始化、LoRA 前反向、权重更新及奖励链路，不是正式复现实验**：

```bash
RUN_DIR="$PROJECT_ROOT/runs/avqa_smoke_$(date +%Y%m%d_%H%M%S)" \
bash "$AVQA/run.sh" \
  data.train_max_samples=128 data.val_max_samples=16 \
  data.train_batch_size=16 actor_rollout_ref.rollout.n=2 \
  data.max_response_length=512 \
  trainer.total_training_steps=2 trainer.test_freq=1 trainer.save_freq=-1 \
  'trainer.logger=[console]'
```

确认确实出现两次训练 step、有限的 `actor/grad_norm`、LoRA 可训练参数以及后续 rollout 后，再执行完整验证基线。若日志未显示 LoRA 可训练参数，不能仅凭 checkpoint 索引检查通过就开始长训练。

```bash
RUN_DIR="$PROJECT_ROOT/runs/avqa_baseline_$(date +%Y%m%d_%H%M%S)" \
bash "$AVQA/run.sh" trainer.val_before_train=true ++trainer.val_only=true
```

基线使用完整验证集（仍会经过 2748 的 4096-token 过滤），不做优化；它的验证采样配置与正式训练相同。然后从原始 2748 权重开始正式训练，不加载 smoke 权重：

```bash
RUN_DIR="$PROJECT_ROOT/runs/avqa_train_$(date +%Y%m%d_%H%M%S)" \
bash "$AVQA/run.sh"
```

运行目录保存 `command.sh`、`resolved-config.txt`、`console.log`、checkpoint 和逐样本生成结果。W&B 默认离线，只记录本地文件；如已有账号且希望在线上传，可显式设置 `WANDB_MODE=online`。

如果图捕获 OOM 或不兼容，用 `AVQA_OPTIMIZED=0 HUANYU_GRAPH_MODE=0` 做一次明确标记的原算子/eager 对照；不要自动缩小正式 batch、LoRA 或长度预算后仍称作相同配置。图开关默认 1，正式参数也可以通过末尾 Hydra 参数覆盖，所有覆盖会写入运行记录。

## 首次 old_log_prob 挂起的诊断

2026-11-16 的失败作业在首次训练侧多模态 forward 后出现 HCCL watchdog。
rank 4 / PID 432602 的 Ray 日志是 `12:37:34 ... killed because the job 04000000 finished`，
属于作业退出清理；该记录不能作为“rank 4 提前被 OOM 杀死”的证据。
16 个 Actor 都打印过 `visual unshard ... call=1 n=28/28 failed=(none)`，但这只说明主机侧
unshard 调用走到日志位置，不能证明视觉 forward 或异步设备通信完成。
现有日志没有显式图片解码异常或 NPU OOM；内核日志无权限，因此仍不能排除主机 OOM。
没有时间戳的 `TBE ... main process disappeared` 和 `kernel_meta ... Directory not empty`
也不足以证明最早故障来自编译目录。

下一次复现保持原 batch、n、长度预算和优化开关，只增加诊断并限制为一个训练 step：

```bash
RUN_DIR="$PROJECT_ROOT/runs/avqa_hang_$(date +%Y%m%d_%H%M%S)"
HANG_DIR="/tmp/$(basename "$RUN_DIR")"
printf 'RUN_DIR=%s\nHANG_DIR=%s\n' "$RUN_DIR" "$HANG_DIR"

RUN_DIR="$RUN_DIR" AVQA_OPTIMIZED=1 HUANYU_GRAPH_MODE=1 \
AVQA_TRAIN_FILE=/opt/huawei/dataset/infra_data/data/avqa_r1_6k_2748_filtered/train.parquet \
AVQA_VAL_FILE=/opt/huawei/dataset/infra_data/data/avqa_r1_6k_2748_filtered/validation.parquet \
HUANYU_TRAIN_HANG_DIR="$HANG_DIR" \
bash "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open/run.sh" \
  data.filter_overlong_prompts=false \
  trainer.total_training_steps=1 trainer.test_freq=-1 trainer.save_freq=-1
```

`/tmp` 仅用于逐进程诊断文件，避免共享目录写入干扰；没有修改训练工作目录，不能把它当作修复。
脚本会将三个 `HUANYU_TRAIN_HANG_*` 变量显式传入 Ray worker。先确认 console 中每个 Actor
都有 `[huanyu-train-hang] installed`。默认每进程只观测前两次模型调用，可用
`HUANYU_TRAIN_HANG_CALLS` 调整；不是训练 step 数。每次调用超过 120 秒未退出，就重复保存
Python 线程栈，不杀进程；阈值用 `HUANYU_TRAIN_HANG_SECONDS` 调整。计时从整个模型调用入口
开始，长调用出现线程栈并不自动等于故障。

出现停滞时即可在**同一容器**查看，不必等 HCCL 超时：

```bash
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/train_hang_probe.py" "$HANG_DIR"
# 从上面输出的 python_stacks 路径选择 rank 4 及一个其他 rank，查看完整线程栈。
```

每个 `*.jsonl` 记录视觉/音频/语言模型及视觉块、语言层的进入/返回、输入 shape、调用次数、
主机 RSS 和 cgroup v2 指标，以及 NPU allocator 当前/峰值统计；每 15 秒的 heartbeat 只读主机指标。
探针不复制张量、不读张量值、不增加设备 synchronize。NPU allocator 不代表整张卡的全部占用；
峰值没有清零，包含之前的分配。对 `memory.events` 必须比较同一 cgroup 前后的增量，非零历史
计数不能单独归因到本次任务，读取失败也不表示没有 OOM。

`scope_return` 表示 Python 返回，不保证异步设备完成。安装在探针之后的 FSDP pre/post hook
可能位于这些阶段之外，最终等待位置应以 `*.stacks.txt` 为准。Python 栈也不能代替 NPU 算子栈。
探针使用进程级 `faulthandler.dump_traceback_later` 定时器，本次不要同时启用其他同类 watchdog。
若异常可传播，会记录原 traceback 并重新抛出；若进程被 SIGKILL，无法保证保存最后一条记录。
当前只有 CPU 契约/超时落盘验证，真实 NPU 卡点仍需这次复现结果确认。

### PEFT 直接调用 forward 时的探针覆盖

2026-11-16 的 `avqa_2748_mediafix_step1_20261116_154718` 作业确认服务器 PEFT 0.20.0
的 `BaseTuner.forward` 使用 `self.model.forward(*args, **kwargs)`。它绕过基础模型的
根模块 pre/post hook，因此旧探针可能一直显示 `call=0 last_event=installed`，即使音频
处理和视觉 unshard 已经执行；此时显示的主机内存只是安装快照，超时抓栈也没有启动。

该作业在约 16:03:57 进入训练侧多模态路径，16:14:17–18 的 rank 0/1 报告
`ALLGATHER seq_num 2904/2905 dispatch timeout`，当前算子为 `aclnnInplaceCopy`。
这证明媒体 token 修复后仍存在训练侧等待问题，不能把 `call=0` 解释为仍在等待 rollout。
上述设备日志尚不足以认定 OOM、图片损坏或具体卡死算子，也不能证明某次 all-gather
已经完成；需要 Python 栈与各 rank 的阶段记录继续定位。

同一作业随后通过 `py-spy dump --nonblocking --pid ...` 抓到存活 Actor 的栈：
rank 0/1（PID 619545/619546）在 `compute_log_prob` 的语言模型路径，栈顶为
`torch_chunk_gated_delta_rule_fast`（`gdn_kernels.py:154`）；rank 4（PID 619549）仍在
`get_image_features` 的视觉前向（服务器 `modeling_qwen3_5_moe.py:1057`）。
这缩小了排查范围，但不同 rank 的单次栈快照不等于通信顺序错误的证明，也不能据此
判定 GDN 算子故障或所有图像均已成功处理。下一步核对服务器对应行、视觉塔的
FSDP reshard 策略、各 rank 图像/视觉块调用次数和未完成 collective。

探针现在保留根模块 hook，同时在直接 `forward()` 的包装入口补齐启动/结束逻辑，
两条路径共用采集预算，正常调用不重复计数。直接调用抛错时也会保存原异常并结束计时。
音频阶段覆盖实际的 `model.audio_encoder` / `model.audio_projection`。不修改 PEFT、
模型数值运算、通信配置或训练配方，不增加设备同步。

同步 `train_hang_probe.py` 后需要启动新作业，原来的单步启动命令和三项诊断变量不变。
新版本安装日志应包含 `entrypoints=module_hook,direct_forward`，报告会打印本次实际入口
`entrypoint: direct_forward` 或 `module_hook`。仅有 `installed` 不代表 forward 未执行。
120 秒未返回时，从报告列出的 `*.stacks.txt` 读取完整线程栈，无需等到 HCCL 超时。
本地 16 项 CPU 回归通过，含真实定时抓栈、两类入口计数、梯度和 OOM/视觉异常保留；
发布目录内与媒体边界、forward、注册一起验证共 67 项通过。尚未验证本次 NPU 训练卡点已解决。

### 冻结视觉塔在逐张编码时的 FSDP 参数生命周期

同一作业后续补充的服务器源码和配置确认：`reshard_after_forward=True`，
`apply_fsdp2` 对所选子模块传入这一设置；视觉隔离包装会逐张直接调用原生视觉前向。
原有视觉 pre-hook 只在进入整个视觉塔时 unshard 一次，内部视觉块却会在每张图后
重新分片。因此，一张图的 rank 可以进入语言层，而多张图的 rank 再次聚合视觉块参数；
零图片的 rank 还会跳过整个视觉 pre-hook。这构成实际的集合通信顺序风险。
此前的 Python 栈与这个风险相符，但尚未由 NPU 重跑证实它就是本次卡死的全部原因。

`vision_fsdp.py` 的修复在在线 Huanyu 基础模型的 `forward` 方法上管理冻结视觉参数：

1. 每次前向按相同模块名顺序发现视觉 FSDP 单元，先统一回到分片状态，再 unshard；
   无图片的 rank 也走这一步。发现放在运行时，因为 PEFT/FSDP 包装晚于 adapter 配置。
2. 仅这些冻结视觉单元设置 `set_reshard_after_forward(False, recurse=False)`，
   让同次前向中的多张图片、视频共享已聚合的参数，不在每张图之间重新聚合。
3. 整次基础模型前向结束或抛错时，显式 reshard 释放视觉参数。原始异常继续传播；
   清理失败会附在原异常上，不能用清理错误掩盖图片错误或 OOM。

普通 `model(...)` 和 PEFT 的 `model.forward(...)` 都覆盖。缓存媒体模式不安装此包装，
可训练视觉塔不由它管理。语言层和音频的 FSDP 策略、原生视觉数值计算、GDN/RMS/MoE、
图模式、LoRA 和 replay 配置保持原值；不是全模型关闭 `reshard_after_forward`。
代价是完整冻结视觉参数驻留到基础模型前向结束，无图 rank 也需聚合，可能增加显存峰值。
因此仍须观察本次 NPU 单步的内存及 forward 是否返回，不能仅凭 CPU 测试宣告修复成功。

更新 `thinker_adapter.py`、新增的 `vision_fsdp.py` 后启动新作业，无需新增开关。
前两次受管理的模型调用打印：

```text
[huanyu-vision-fsdp] installed scope=frozen_visual entrypoints=module_call,direct_forward
[huanyu-vision-fsdp] phase=enter rank=... call=1 units=28
[huanyu-vision-fsdp] phase=gathered rank=... call=1 units=28
[huanyu-vision-fsdp] phase=released rank=... call=1 units=28
```

`units` 以实际包装结果为准；`gathered` 表示 Python unshard 调用已返回，不保证全部异步设备
工作已经完成。继续开启 `HUANYU_TRAIN_HANG_DIR`，并从各 rank 的阶段记录和超时栈核对进展。
`released` 在异常清理成功时也会打印，不能单凭这一行判断训练成功。

CPU 测试复现旧实现的一图/多图 gather 序列分歧，并验证修复后的无图、多图、图片加视频、
连续调用、直接 forward、梯度、异常释放、缓存入口及探针兼容性；不模拟 HCCL/设备流。
可运行新增回归和集成检查：

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python3 -m pytest -q \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/tests"/test_{vision_fsdp,registration,train_hang_probe}_on_cpu.py
```

本次与媒体边界、processor、在线/缓存 forward、视频、音频、注册、位置及诊断的相关
回归共 196 项通过，发布目录内也通过同组测试。额外运行 `test_precision_on_cpu.py`
时有一条既有日志字符串断言失败：实际输出增加了 `npu_kernel=0` 字段，旧断言仍要求
`already=1 native=1` 连在一起；在修改前提交 `8bb28f3` 上复现相同失败，未计入上述通过数。
这些结果只代表 CPU 契约验证，尚无本次 16 卡 NPU 成功结果。

### 前向返回后的 NPU log-prob 交叉熵 OOM

`avqa_2748_mediafix_step1_20261116_164155` 的新日志显示第二次模型调用的 16 个 rank
均进入并完成 Python unshard 调用，片段中有 15 个 rank 的 `released`；不能据此认定
所有 rank 都已完成。本次报错的 rank 6 已从模型前向返回，17:04:11 在
`_compute_old_log_prob -> prepare_model_outputs -> logprobs_from_logits_torch_npu ->
npu_cross_entropy_loss` 申请 8.10 GiB 失败。当时设备总量 61.27 GiB，allocated
48.43 GiB、reserved 48.47 GiB、free 6.38 GiB。两项 allocator 数值接近，不支持将
这次失败主要归因于缓存碎片；直接失败是 NPU 分配不足，训练仍为 `0/1` step。

这是一次明确发生在模型输出后处理中的 OOM。它说明程序越过此前观察到的视觉/语言
前向等待位置，但不能反推此前历史超时全部由 OOM 引起，也不能将 `call=2` 当成完成两步训练。
当前阶段尚未执行 actor 的训练反向或梯度归约，因此不能用 BF16 reduce 的开关解释此错误。

原有 entropy 已按 token 分块，但 NPU log-prob helper 会把全部有效 logits 一次提交
原生交叉熵。新增 `log_probs.py` 在 Huanyu 模型配置时包装这个 helper，每块最多 512
个 token，仍使用原来的 `torch_npu.npu_cross_entropy_loss`，保留全部词表、labels、
原生输出 dtype 和 autocast 上下文。dispatcher 的全局 helper 引用被更新，因此引擎
提前 `from ... import logprobs_from_logits` 的别名也能命中；未知接口会明确报错。

训练反向逐块重算原生交叉熵，再调用其原生 autograd，避免保存所有块的中间结果。
仅支持通常 RL 更新所需的一阶梯度。完整 logits、其完整梯度和引擎在调用前打包 logits
的内存仍存在。示例 rank 12 的 BF16 `[13,4922,248320]` logits 本身约 29.6 GiB；
分块处理只限制交叉熵临时内存，不能保证整个训练过程绝无 OOM。

更新 `log_probs.py` 和 `thinker_adapter.py` 后启动新作业，保持原来的单步命令、原配方、
过滤后 parquet、`data.filter_overlong_prompts=false` 和卡死诊断参数。无需新增精度设置。
检查安装日志与首次执行日志：

```text
[huanyu-logprobs] installed ... chunk_size=512 target=verl.utils.torch_functional.logprobs_from_logits_torch_npu
[huanyu-logprobs] active ... chunk_size=512 mode=no_grad backend=native_npu
```

进入策略更新时还应看到 `mode=grad`。这些行表示执行路径，最终仍以训练 `step:1` 和
正常退出判断单步完成。若再次 OOM，检查新的调用栈，不能沿用本次交叉熵定位结论。

本次 246 项 CPU 回归通过，覆盖 log-prob 值/加权梯度、FP16/BF16/FP32/FP64、autocast
重算、非连续 logits、512-token 上限、保存的张量、提前导入的别名、媒体/FSDP/诊断集成。
另用本地 `verl` 的真实 dispatcher/helper AST 及 CPU 算子替身验证了引用和梯度。
尚未在真实 NPU 上验证该分块交叉熵的数值、峰值显存和整步运行结果。

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python3 -m pytest -q \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/tests"/test_{log_probs,entropy,vision_fsdp,media_boundary,processor_bridge,cached_processor,forward,cached_forward,video_pipeline,audio_training,registration,aqa_position_ids,train_hang_probe}_on_cpu.py
```

### 分块已生效但仍 OOM：移除 logits 打包副本

`avqa_2748_full_20261116_173338_612632` 在 `old_log_prob` 再次失败。调用栈已经经过
`log_probs.py -> _ChunkedLogProbs.forward -> backend`，说明 512-token 分块已生效。
这次 NPU 5 申请 244 MiB，而只剩 48.46 MiB；allocated 52.68 GiB、reserved
52.71 GiB。进度为 `0/350`，尚未完成第一个训练 step。

此前修复只限制交叉熵中间量。引擎进入 helper 前仍执行
`torch.cat([t for t in logits.unbind()])`，把有效 logits 从原始 `[B,S,V]` 复制成
完整 `[T,V]`，而 `raw_output` 仍持有原始 logits。这里 T 为有效 token 总数。
这一份副本需要 `T × V × dtype字节数`，与每块 512 token 的交叉熵工作区无关。

新增 `model_outputs.py`，在标记为 Huanyu 的模型上包装
`FSDPEngineWithLMHead.prepare_model_outputs`。直接逐样本、逐 token 块读取原始
logits 的视图，使用引擎原有的 packed/rolled labels，仅拼接 `[T]` 的 log-prob
与 entropy 标量结果；不在样本边界重新 shift labels。反向逐块重算交叉熵，补齐
位置的梯度为零。已分块的原生 helper 直接处理不超过 512 token 的块，避免嵌套重算。

熵计算、温度处理、原生 NPU 后端和输出的 nested offsets 保持原来的语义。
模型前向仍使用 padded 输入。fused-output、remove-padding、蒸馏及 sum-pi-squared
分支调用原始引擎方法。其他模型不启用此 hook；未知引擎接口明确报错。
完整模型 logits 及其梯度仍会分配，非单位温度时原引擎的缩放副本也仍存在；本修复
移除的是有效 logits 的打包副本，不能保证所有后续训练阶段都不再 OOM。

更新后保留完整训练命令与关闭 probe 的设置，无需修改 BS128、n16、LoRA32、TP2、
长度上限或 token 预算。确认出现：

```text
[huanyu-output] installed ... scope=huanyu packing=scalar_only
[huanyu-output] active ... shape=(B,S,V) valid_tokens=T chunk_size=512 mode=no_grad packing=scalar_only
```

`[huanyu-output] ... mode=grad` 表示进入需要梯度的输出处理；仍以真实训练 step
完成和正常退出判断训练是否成功。上述日志为一次性路径说明，不是卡死 probe。

286 项相关 CPU 回归通过，覆盖变长样本、padding 梯度、非连续输入、温度、熵与
log-prob 联合梯度、checkpoint、PEFT 包装、原生 helper 分块及原有多模态适配。
另提取本地真实 verl 的 dispatcher、NPU helper、`FSDPEngineWithLMHead` 输出处理
源码，替换设备算子为 CPU 实现，6 组 dtype/temperature 的数值、梯度及不生成
词表打包副本的检查通过；这不等同于真实 NPU 数值和峰值显存验证。

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python3 -m pytest -q \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/tests"/test_{model_outputs,log_probs,entropy,vision_fsdp,media_boundary,processor_bridge,cached_processor,forward,cached_forward,video_pipeline,audio_training,registration,aqa_position_ids,train_hang_probe}_on_cpu.py
```

### 第一步完成、第二步 Actor backward OOM：限制补齐后的微批次

`avqa_2748_full_20261116_210329_867683` 已完成一个 step，第二步在
`_update_actor -> loss.backward -> log_probs.py` 分配完整 `grad_logits` 时失败。
NPU 1 申请 19.92 GiB，空闲 19.88 GiB，allocated 31.57 GiB、reserved 31.72 GiB。
这是 Actor 反向的明确分配不足；单凭这份日志不能认定跨 step 内存泄漏。

512-token 分块限制交叉熵的中间量，完整 `[B,S,V]` logits 和梯度仍存在。
原动态分批约束 `sum(valid_lengths) <= 20480`，但 Huanyu 使用 padded forward，
实际词表张量按 `B * max(lengths) * V` 分配。长短序列混合可能使 padding 后
的规模远超 20480；第一步能运行不能保证下一步的微批次也能放入显存。

`padded_batching.py` 在 Huanyu FSDP 引擎调用期间额外约束
`B * S_max <= max_token_len_per_gpu * sp_size`。仍使用原 verl 分批器和索引恢复，
按需增加微批次数；各 DP rank 协商统一数量，当前已符合预算的 rank 也参与。
若不可拆分组无法满足共同数量，所有参与 rank 明确报错，不静默丢样本或截断。
旧概率计算和 Actor 更新均覆盖，其他模型、静态分批、remove-padding 路径保持原行为。

BS128、n16、LoRA32、TP2、mini-batch16、4096/12288 长度上限、20480 token 预算、
奖励、优化器 step 及损失归一化均保留；微批次划分和累计顺序可能改变，吞吐与浮点
舍入也可能变化。对 BF16 / V=248320，20480 行词表张量约 9.47 GiB；这不是整个
训练峰值，激活、温度缩放、其他梯度和算子 workspace 仍需额外空间。

同步新文件、`thinker_adapter.py` 后启动新作业，无需新增开关，probe 继续关闭。
首次安装以及发生重新分批时会出现：

```text
[huanyu-padded-batch] installed ... scope=huanyu budget=B*S_max
[huanyu-padded-batch] ... budget=20480 micro_batches=...->... max_padded_tokens=...->... dp_sync=1
```

这属于运行路径说明，不是定时诊断探针。重新分批后的 `max_padded_tokens` 应不超过预算。
CPU 测试覆盖超额 padding、样本及索引完整性、加权梯度、PEFT/外层包装、异常后作用域
恢复，以及模拟 DP 的统一拆分/失败。另使用本地真实 verl 分批与引擎源码验证了
150 组长度分布和推理/反向循环。与已有内存、多模态和诊断用例一起运行共 325 项
CPU 回归通过；模拟 collective 不等同于真实 HCCL/NPU 验证。

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python3 -m pytest -q \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/tests"/test_{padded_batching,model_outputs,log_probs,entropy,vision_fsdp,microbatch_sync,media_boundary,processor_bridge,cached_processor,forward,cached_forward,video_pipeline,audio_training,registration,aqa_position_ids,train_hang_probe}_on_cpu.py
```

### Rollout 后处理中的图像 token 不匹配

另一轮作业在 `_compute_score -> _compute_multi_modal_inputs` 报告
`image token lengths [165, 1] != image grid lengths [165]`，以及 `[880, 2] != [880]`。
这是奖励准备阶段的媒体对齐错误，不能当作训练 Actor forward 的 HCCL 卡点。
完整输入扫描多出一段图像 token，可能来自模型 response；历史日志没有 token 位置，
不能仅据此认定图片损坏或 OOM。

当前修复对 Huanyu 单轮 AR 训练使用显式 prompt 边界：

1. AgentLoop 提供原始 `prompt_ids` 和 `response_ids`，先校验实际输入与二者拼接一致，
   仅允许额外的外侧 padding，再用 prompt 校验图片 grid、视频时序组和音频块。
2. `huanyu_prompt_lengths` 随每条样本的多模态输入传输；RoPE 和训练 forward 根据
   attention mask 中的有效 token 计数恢复边界，支持重新 padding 和微批次重排。
3. HF 的媒体 mask/位置编码使用结构视图，把 response 和 padding 当成普通文本；
   embedding pre-hook 把实际查表参数恢复为原始 token IDs，因此回答中的 `<|image_pad|>`
   等标记仍保留自己的 token embedding 和梯度，不再消耗输入图片的 features。

不删除/替换保存的轨迹，不改变 response mask、log-prob、奖励或采样参数；不把真实 prompt
中的不匹配静默截断。没有显式边界的旧调用继续执行全序列严格校验。此契约用于单轮文本输出，
不支持在 response 中追加需要编码的新媒体（例如多轮工具返回图片）。

须同时同步 `media_boundary.py`、`processor.py` 和 `forward.py`，由新作业的 AgentLoop/Actor
加载；无需新增启动参数。首次成功构造输入会打印：

```text
[huanyu-media-boundary] prompt_only=true response_embeddings=original
```

CPU 回归覆盖 `[165,1]` / `[880,2]`、完整生成媒体标记、左右 padding、混合批次、在线/缓存
媒体注入、原始 token 的 embedding 梯度及真实 prompt 错误。该修复不等于已解决另一轮的
`old_log_prob` HCCL 卡死，仍需在 16 卡 NPU 上验证训练能否继续。

本次本地相关回归共 156 项通过，包含 decoder checkpoint 重算。服务器可执行同一组 CPU 测试：

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python3 -m pytest -q \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/tests"/test_{media_boundary,processor_bridge,cached_processor,forward,cached_forward,video_pipeline,audio_training,registration,aqa_position_ids}_on_cpu.py
```

Huanyu AgentLoop 桥接器现在在输入构建异常时输出一行 `[huanyu-mm-input-error]` JSON，包含：

- `lengths`：prompt、response 和实际 input 的长度。
- `input_equals_prompt_plus_response`：实际输入是否恰好是两者拼接；缺少边界信息时为 null。
- `markers.image_token_id.prompt/response/input`：分别记录图像 token 总数、连续段数和
  `runs_offset_length`（零起始位置、长度；每段位置相对于自己的序列）。最多列出 16 段，
  超出时 `runs_truncated=true`，但总数仍完整。
- 其他视频、视觉边界和音频边界 token 的同类统计，以及原异常类型和消息。

诊断只在异常时运行，不复制设备张量，不打印文本或媒体内容，仍重新抛出原异常。
磁盘文件更新不会自动替换已运行进程中的代码。
在新作业日志中提取：

```bash
grep -aF '[huanyu-mm-input-error]' "$LOG"
```

修复后若仍有错误，依据统计区分 prompt 本身不匹配与输入拼接/元数据错误；不要删除回答
token 或截断 `image_bound` 来绕过校验。

服务器给出的 `_run_prompt` 实现会先等待同组全部 session 结束，再打印异常并写入 `failure`。
进度行是队列状态快照，不是累计异常数；当前 `failure=0` 不能证明没有轨迹失败，也不能在
未核对服务器 ReplayBuffer 的消费/补采逻辑前断定失败组已重试或被标为成功。

### Replay 报告 vLLM prompt 394 / training prompt 396

2026-11-16 的服务器源码确认：原生 `_get_prompt_updates` 的两个图像 target
（`(<image>./</image>)` 和 `<|vision_start|><|image_pad|><|vision_end|>`）
均覆盖完整占位符，但 replacement 仅返回 N 个 `<|image_pad|>`，因此每张图会
丢失两个边界 token。训练侧保留这两个 token；AQAV 单图请求的差值正好是 2。
错误位于 rollout 的 R3 返回校验阶段，不是 Actor OOM，也不能据此认定 LoRA 不兼容。

`video_rollout.py` 的 Huanyu 专用更新器现在也负责图像：完整返回
`<|vision_start|>` + N 个 `<|image_pad|>` + `<|vision_end|>`，通过
`PromptUpdateDetails.select_text` 仅选择 image pad 作为视觉 embedding 注入位置。
`rollout_inputs.py` 对图像或视频请求都在 server dispatch 前确保安装更新器。
音频更新对象和其他模型路径保留原行为；不关闭 replay 校验、不补齐或截断路由行。
该输入修复不依赖 replay 开关，replay=0 同样保留图像边界。

同步后须启动新作业，使 server / EngineCore / worker 加载新代码。无需修改原训练配方、
优化开关或重新制作过滤后的 parquet。安装日志包含：

```text
image_spans=full_boundaries_v1
```

可先在训练容器检查实际安装的 vLLM prompt updater（只加载配置、tokenizer 和图像前端，
使用合成图像，不加载模型权重、不连接 Ray）：

```bash
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/check_image_contract.py" \
  --model /opt/huawei/dataset/model_gy/downloaded_models/2748
```

期望 `status: PASS`，覆盖两种占位符、已展开输入、单图/多图的逐 token 对齐、
完整图像范围和 embedding mask。此检查不包含真实音频前端、NPU 推理或完整 R3 训练。
本地 125 项 CPU 回归通过，另复现了 396 → 394 的旧行为并验证修复后的 token 序列一致；
16 卡完整训练仍需服务器验证。

## 判断是否学到了有效行为

```bash
LOG=/path/to/this/run/console.log
grep -aE 'trainable params|lora_|LoRA|dataset len|filter dataset len' "$LOG"
grep -aE 'step:|Initial validation metrics|val-core/avqa_r1_6k/' "$LOG" | tail -n 30
```

比较完整基线与同一验证样本集合上的后续 `val-core/avqa_r1_6k/.../mean@1`，同时看 `critic/rewards/mean` 的多个 step 窗口，不能用单个高奖励 batch 或 smoke 的 16 条验证样本宣告上涨。保存实际过滤后的样本数；2748 与官方 tokenizer 不同可能改变保留集合和总步数，不能强行将 348 设为停止条件。

若 reward 不涨，先查输出是否有合法 answer 标签、`response_length/clip_ratio` 是否很高、组内奖励是否全同、LoRA 梯度是否非零、更新后的 rollout 权重是否同步，再查 actor/rollout 概率偏差和 KL。`actor/loss` 接近零本身不能证明未学习。不要为制造上升曲线改变答案标签、奖励权重或验证集。

先验证以上链路，再决定是否需要针对 2748 调参；一旦调整学习率、LoRA 范围或采样温度，应记录为后续实验，而不是官方参数复现。

## AQAV 训练故障复盘汇总（截至 2026-11-17）

本节汇总本次 AQAV（数据集名 AVQA-R1-6K）迁移与重测中的实际故障，方便按报错快速定位。
证据来自服务器返回的日志、现场读取的源码和本地 CPU 回归；上文各阶段的“尚未验证”描述
保留为当时记录，当前进展以本节末尾为准。基线为 2748、16 NPU、BS128、n16、TP2、
LoRA32、GSPO/GRPO、choice reward，保留原长度上限与优化配置。

### 故障与处理索引

| 阶段 / 报错 | 定位结论 | 处理及对应代码 | 验证边界 |
| --- | --- | --- | --- |
| 数据准备：`Official converter missing: .../examples/gspo_trainer/data_process/avqa.py` | 入口依赖另一份源码树中的转换脚本，当前环境没有该路径 | 将转换器放入本目录 [avqa.py](avqa.py)，[prepare.py](prepare.py) 从同目录加载 | 消除跨仓库路径依赖；仍需校验原始归档、媒体路径及转换后的 parquet |
| 启动：`缺少官方奖励实现 .../verl_omni/utils/reward_score/choice_reward.py` | 当前基础包未包含入口指定的奖励文件 | 同目录保存 [choice_reward.py](choice_reward.py)，[run.sh](run.sh) 使用该绝对路径 | 保持官方 choice reward，不改成 GenRM 或另写奖励公式 |
| 首次 `old_log_prob`：`ProcessGroupHCCL::Watchdog::run`、`Fatal Python error: Aborted`，随后 Ray `keepalive watchdog timeout` | 已确认 Actor 在训练侧等待并被 watchdog 终止；Ray RPC 错误是后续表现。首轮日志不足以确定唯一底层原因 | 逐 rank 收集阶段、内存和线程栈；随后修复冻结视觉参数的 FSDP 生命周期，见下节 | 不能据此认定网卡故障、图片损坏、OOM 或运行目录有问题 |
| 探针一直 `call=0 last_event=installed`，但 Actor 已在 forward | PEFT 直接调用基础模型 `.forward()`，绕过根模块 pre/post hook | [train_hang_probe.py](../train_hang_probe.py) 同时覆盖模块 hook 和直接 forward，统一计数与异常记录 | `call` 是模型调用次数，不是训练 step；旧 `call=0` 不能证明仍在等 rollout |
| 不同 rank 分别停在 GDN 与视觉前向；`ALLGATHER ... dispatch timeout` | 逐图编码与 `reshard_after_forward=True` 组合，使不同图片数的 rank 存在集合通信调用顺序分歧风险 | [vision_fsdp.py](../vision_fsdp.py)：统一聚合冻结视觉单元，同一次基础模型 forward 内保留完整参数，结束或异常时释放 | 修复有源码和 CPU 序列验证；不能把所有历史超时都归因于此。代价是视觉参数驻留时间延长 |
| `old_log_prob` 原生交叉熵申请 **8.10 GiB** 失败，free **6.38 GiB** | 模型前向已返回，OOM 在输出概率计算；一次处理全部 logits 的交叉熵工作区过大 | [log_probs.py](../log_probs.py)：按 token 分块调用原生 NPU 交叉熵，反向按块重算 | 分块不消除原始 logits、打包副本或完整梯度，不能单独保证整步不 OOM |
| 分块已执行，仍申请 **244 MiB** 失败，free **48.46 MiB** | 分块前已把原始 `[B,S,V]` logits 另行拼成 `[T,V]`，两份大张量同时存在 | [model_outputs.py](../model_outputs.py)：直接读取原始 logits 视图，只打包 log-prob / entropy 标量结果 | 去掉的是词表维打包副本；原始 logits 和完整梯度仍存在 |
| 已完成第一个 step，第二步 `loss.backward -> grad_logits` 申请 **19.92 GiB** 失败，free **19.88 GiB** | 反向需要完整词表梯度；只限制有效 token 总数无法限制长短混合微批次的 padding 内存 | [padded_batching.py](../padded_batching.py)：额外限制 `B * S_max`，各 DP rank 协商一致的微批次数并保留样本/索引 | 单凭第二步 OOM 不能判断内存泄漏；微批次划分会影响吞吐和浮点累计顺序 |
| Rollout 后处理：`image token lengths [165, 1] != image grid lengths [165]`，或 `[880, 2] != [880]` | 构造训练多模态输入时，原逻辑把 response 中的图像标记也当作输入图像占位符 | [media_boundary.py](../media_boundary.py)、[processor.py](../processor.py)、[forward.py](../forward.py)：用真实 prompt 边界限定媒体扫描与 RoPE，保留 response 原始 embedding | 不删除回答 token，不截断真实 prompt 的图片占位符；该错误与训练 forward 的 HCCL 等待不是同一阶段 |
| 开启 replay：`vLLM prompt has 394 tokens but ... training prompt has 396` | 服务器原生更新器将完整图像占位符替换成纯 image pads，每张图丢掉两个视觉边界 token | [video_rollout.py](../video_rollout.py)、[rollout_inputs.py](../rollout_inputs.py)：保留完整视觉边界，仅在 image pad 位置注入 embedding，并在图像请求 dispatch 前安装补丁 | 严格 replay 校验保留；不能补零、随意切片路由行或用 debug 绕过。最新修复尚待完整 NPU 训练验证 |
| 开启 replay 后进入 `old_log_prob`：`expert_replay: actor module has no fused MoE blocks installed` | replay 在 LoRA 包装后的模型上查找 decoder 层，漏解 PEFT 包装，可能将已安装的融合 MoE 误判为不存在 | [engine_hook.py](../expert_replay/engine_hook.py)：通过 `get_base_model()` 与 FSDP/compile 包装链查找真实模型，保留循环保护和原有路由校验 | CPU 已复现“安装成功但查找为空”并修复；此错误不能单独推出 LoRA/replay 不兼容，NPU 完整训练仍待验证 |

### 三次 OOM 为什么需要三层修复

三次失败的分配对象不同，后一次仍 OOM 不等于前一次修复没有执行：

1. **交叉熵中间量**：512-token 分块限制每次原生 NPU CE 的输入和临时空间。
2. **重复的词表张量**：避免在 `[B,S,V]` 原始 logits 仍存活时，再复制完整 `[T,V]`。
3. **原始 logits 及其完整梯度**：用 padding 后的 `B * S_max` 控制微批次规模。

这里 `B` 是当前设备上的微批次序列数，`S_max` 是该微批次补齐后的最大序列长度，
不是全局 `train_batch_size=128`。`V` 是词表大小，`T` 是有效 token 总数。
原始 **logits** 才带有 `[B,S,V]` 词表轴；选定标签的 **log_probs** 和 **entropy**
是每 token 一个标量，通常为 `[B,S]`，去除 padding 后的 values 为 `[T]`。
不能把三者都当作 `[B,S,V]`，也不能认为分块概率计算已经移除了完整 logits 梯度。

示例：BF16、`V=248320` 时，`20480 * V * 2` 字节约为 **9.47 GiB**，只是一份
词表张量的大小；激活、参数、其他梯度、温度缩放及算子 workspace 还要额外显存。
因此微批次预算是分配规模约束，不是整卡峰值显存保证。

上述修复没有把全局 BS128、n16、LoRA32、TP2、mini-batch16 或 4096/12288 长度上限
缩小，也没有改奖励。三次失败的 allocated 与 reserved 数值接近，现有证据不支持
直接把主因归结为 PyTorch 缓存碎片；应先核对失败分配及同时存活的大张量。

### 通信超时与卡死的证据链

首轮作业 `avqa_2748_20261116_115509_430028` 在约 12:34 出现 HCCL watchdog / abort，
约 12:37 主进程才报告 `ActorUnavailableError`。因此应向前找首个 Actor/设备错误，
而不是只看最后一段 Ray traceback。`killed because the job ... finished` 是退出清理，
不能当作此前遭 OOM killer 杀死的证据。

后续非阻塞抓栈确认：rank 0/1 已进入 GDN 语言路径，rank 4 在视觉 attention 的
`lengths.tolist()`。后者读取设备 tensor 时可能等待设备工作；某个 Python 栈顶
既不能证明对应算子就是根因，也不能证明该 rank 正在有效计算。结合实际配置中的
`reshard_after_forward=True` 和逐图调用，才定位到冻结视觉块可能重复 all-gather 的风险。

修复按相同顺序管理所有 rank 的冻结视觉 FSDP 单元，包括没有图片的 rank；仅这些单元
在一次基础模型 forward 中不逐图 reshard，最后显式释放。不是全模型关闭 reshard，
也不是盲目关闭 GDN 或增加通信超时时间。后续作业越过了原等待位置并暴露出明确的
log-prob OOM，之后又有完整第一步成功的日志；这些是进展证据，不是对历史根因的完整证明。

现场只读抓栈使用 `py-spy dump --nonblocking --pid <当前Actor PID>`；PID 必须取自
当前作业，不能复用本文历史 PID。`--nonblocking` 避免暂停进程，但不保证绕过所有权限限制。
只有安装日志而没有阶段记录时，应先确认探针是否覆盖 PEFT 直接 forward。

以下推断在本次排查中没有证据支持，不能作为“已解决方案”写入运行结论：

- 把项目搬到 `/tmp` 就能解决通信超时。`/tmp` 在本次只用来放诊断文件。
- 图片处理失败或主机 OOM 已被彻底排除。原作业 `journalctl` 不存在、`dmesg` 无权限，
  只能记录“缺少内核证据”；`memory.events` 也需比较同一 cgroup 的前后增量。
- 模型输出 helper 的 OOM 是梯度归约精度导致。首次 `old_log_prob` 尚未进入 Actor 更新反向。
- 只要 `failure=0` 就没有轨迹错误。它是队列状态快照；实际 `_run_prompt` 等同组 session
  都结束后才写终态，随后还可能被队列消费，必须同时核对 traceback 和终态处理。

### 启动准备和环境中的相关问题

长度过滤最初只有约 2 条/秒，每次重启重复扫描 4491 条样本会明显延长准备时间。
这属于重复预处理开销；使用 [filter_data.py](filter_data.py) 按同一模型、processor、
模板和长度配置离线过滤，再显式指定过滤后的 train/validation parquet，并设置
`data.filter_overlong_prompts=false`。模型、模板、媒体处理或长度规则改变后需重新验证
过滤结果，不能仅凭文件名含 `filtered` 就跳过过滤。

此前共用 Huanyu 环境还出现过“完成请求缺少 `routed_experts`”的问题。它与本次已拿到
路由记录后发现 `394/396` 的错误不同：前者先查 `vllm.general_plugins` 安装元数据、
`huanyu_r3` 是否在 EngineCore / worker 应用及 `enable_return_routed_experts` 传递。
仅修改 `pyproject.toml` 或设置 `PYTHONPATH` 不代表 entry point 已安装到训练使用的 Python。
需要使用同一解释器重新安装本地扩展，并启动加载新代码的新作业；已有插件白名单要保留
`huanyu_r3` 和原有插件。AQAV 当前显式使用 `AVQA_REPLAY=1` 开启 replay，要求
`AVQA_OPTIMIZED=1`。这类环境问题及 prompt 对齐失败都不能单独推出 LoRA 不兼容。

### LoRA 包装导致 replay 找不到融合 MoE

2026-11-16 23:42 的作业在 `actor_rollout_compute_log_prob` 中报错，调用链为
`prepare_model_inputs -> _arm_micro_batch -> _geometry`。此时是在准备训练输入，
还没有进入该微批次的模型 forward；本段异常没有显示 OOM 或通信超时。

安装与查找发生在不同阶段：`configure_model` 先在裸模型上安装融合 MoE，verl 随后
添加 PEFT/LoRA 包装。裸模型的层路径是 `model.language_model.layers`，但 PEFT 的
`.model` 属性会指向整个基础 causal LM；旧查找只处理 `_fsdp_wrapped_module` 和
`_orig_mod`，少解了一层，返回零个融合块。安装日志即使显示 `total=40 replay=on`，
仍可能触发这个误报。

修复让 replay 的模型标记检测和层数/专家数发现共用支持 `get_base_model()` 的遍历。
仅解析包装，实际前向继续使用原 PEFT 模型，LoRA 参数和融合 MoE forward 都保留。
没有通过硬编码 40 层、跳过路由校验、重新安装算子或关闭 replay 来绕过错误。
如果解包后确实没有融合块，仍然报错；还应检查同一作业启动时的安装日志：

```bash
grep -anE '\[huanyu-kernel\].*fused MoE|\[huanyu-replay\].*engine hook|no fused MoE blocks' "$LOG"
```

本地新增 18 项 CPU 回归覆盖裸模型、LoRA、FSDP/compile 组合、循环引用、ref 门控、
未安装 kernel、路由层数/专家号错误，以及真实 target 布局转换、回放索引、router 梯度
和重复取用 checkpoint targets。连同 AQAV replay 入口和图像展开契约共 **43 项通过**：

```bash
python -m pytest -q \
  verl_omni_ext/models/qwen3_6_huanyu/tests/test_replay_engine_wrappers_on_cpu.py \
  verl_omni_ext/models/qwen3_6_huanyu/aqav_open/tests/test_aqav_replay_launcher_on_cpu.py \
  verl_omni_ext/models/qwen3_6_huanyu/tests/test_image_rollout_on_cpu.py
```

测试使用小型 CPU 模型模拟 PEFT 的公开包装契约，不启动真实 NPU/FSDP 通信。
更新后需启动新作业加载代码；保持原 LoRA、算子、图模式和 `AVQA_REPLAY=1` 配置，
正式训练仍使用过滤后的 parquet、`data.filter_overlong_prompts=false` 和关闭 probe 的入口。

### 最小诊断记录与验收

每次失败保留该次运行的 `console.log`、`command.sh`、`resolved-config.txt`、准确代码
版本、报错时间及 rank/PID；需要时再补同一 Ray session 的 worker 日志、探针文件和设备日志。
先按调用链归类为 rollout、媒体后处理、old_log_prob、Actor backward 或验证阶段，再判断根因。

```bash
# 已设置当前作业的 LOG；先看首批明确错误，再看最新进度。
grep -anEm 8 -B3 -A12 \
  'OutOfMemoryError|NPU out of memory|ValueError: Huanyu|RuntimeError: (R3|expert_replay)|dispatch timeout|Fatal Python error' "$LOG"
grep -anE 'pending:|Error in _run_prompt|step:|Training Progress:' "$LOG" | tail -n 30

# 确认修复路径被安装/执行；这些标记本身不等于训练成功。
grep -anE '\[huanyu-(vision-fsdp|logprobs|output|padded-batch)\]|image_spans=full_boundaries_v1' "$LOG"
```

搜索 OOM 时不要使用不带边界且忽略大小写的裸 `OOM`：它会误匹配 `MiniCPMOOmni`
等类名，使真实错误被大量启动日志淹没。优先使用上面的完整异常名称。

截至本节记录时，服务器已有 **一个完整训练 step 成功**的证据，随后第二步在 Actor
反向 OOM。限制 padded 微批次之后的一轮开启 replay 的作业，又在 rollout 暴露了图像
边界丢失；图像修复之后的新日志已进入训练侧 replay 输入准备，又暴露出上述 LoRA 包装
查找错误。修复后尚无服务器完整训练成功结果。因此，不能写成“所有 step 已跑通”
或“通信/OOM 已全部解决”。正式全量训练按用户选择关闭 probe；专门复现卡死时才启用
诊断并限制训练步数。没有诊断文件不代表没有执行模型前向。

| 修复里程碑 | `new` 分支提交 | 已有验证 |
| --- | --- | --- |
| 区分 prompt 与 response 媒体标记 | `51ef258` | 媒体边界、梯度和多模态 CPU 回归；后续 NPU 作业进入训练侧 |
| 覆盖 PEFT 直接 forward 的探针 | `8bb28f3` | CPU 入口、计数、异常及定时抓栈验证 |
| 冻结视觉 FSDP 生命周期 | `3333bc0` | 196 项相关 CPU 回归；后续 NPU 日志越过原观察位置 |
| 原生 log-prob 分块 | `521278f` | 246 项相关 CPU 回归；后续 NPU traceback 确认 helper 已执行 |
| 仅打包标量输出 | `9434afa` | 286 项相关 CPU 回归；后续服务器完成一个训练 step |
| padded 微批次预算 | `0a60598` | 325 项相关 CPU 回归及真实 verl 分批源码检查；后续训练闭环待验证 |
| 图像完整边界用于 replay | `fb735df` | 125 项相关 CPU 回归；提供 `check_image_contract.py` 供服务器验证，完整 NPU 训练待验证 |

各测试数来自对应提交时的不同回归集合，不能相加作为独立测试总数。CPU 模拟不覆盖实际
NPU 数值、设备流或 HCCL；最终验收需包含连续多个训练 step、验证/保存阶段及正常结束，
并检查是否还出现失败轨迹、OOM 或未完成 collective。

### Host OOM 与高阶页分配失败

11 月 18 日的宿主机记录中，14:38 报 `order:4` 连续页分配失败，15:42 内核
OOM killer 杀了 `ray::AgentLoopW`。两条记录相隔约 63 分钟。回查同一宿主机同一
次启动的内核日志，保留调用栈、`oom-kill:constraint=`、`Mem-Info` 和进程表。
用 `oom_memcg=` 或 `global_oom` 确认触发范围，再将 `task_memcg=` 与作业容器关联。

复现时可在 worker-0 的训练容器内独立采样。将 `PROJECT_ROOT` 指向已有代码的持久化
挂载。每次创建新文件，进程退出前已经写出的样本可用于复盘。

```bash
MEM_MON="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav_open/monitor_host_memory.py"
MEM_DIR="$PROJECT_ROOT/runs/hostmem_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$MEM_DIR"
nohup python3 "$MEM_MON" --interval 15 --out "$MEM_DIR/memory.jsonl" \
  >"$MEM_DIR/monitor.log" 2>&1 &
MEM_MON_PID=$!
```

在同一容器按原配方启动训练。结束后在同一 shell 执行 `kill "$MEM_MON_PID"` 停止
采样。宿主机侧采样可以用 `--pid <训练进程的宿主机PID>` 选定作业 cgroup。默认记录
目标组和子组内可见进程，以及能读取的祖先组水位。加 `--top 6 --pss` 可按 RSS
选出前六个进程，额外读取 PSS 来分摊共享页，读取大进程页表会有开销。

- `buddy_free_blocks_by_order` 按 NUMA node 和 zone 保存，数组索引为 order。
  对照 `zone_watermarks_pages`、`compact_fail` 增量以及当时的分配栈，判断本次请求
  能使用的区域是否仍有空闲小块、却缺少 order ≥ 4 的连续块。
- cgroup v2 的 `memory.current/max`、`memory.events` 和 `memory.stat` 可区分限额、
  匿名内存和共享内存压力。跨采样比较增量。`oom_kill` 单独增加只说明组内进程被杀，
  cgroup v1 可查看 `memory.usage_in_bytes`、`memory.limit_in_bytes` 和 `memory.failcnt`。
- `roles` 汇总进程数和匿名 RSS，`top_kB` 保留 PID、名称与内存拆分。将时间与
  rollout、`_update_actor` 和 `_update_weights` 对齐。Linux swap 活动见
  `pswpin/pswpout`，优化器状态的 CPU offload 需对照训练进程的状态迁移时刻。
  `VmPin`、`VmLck` 和 `Mlocked` 的覆盖取决于 pin 页路径；零值无法排除驱动 pin。
