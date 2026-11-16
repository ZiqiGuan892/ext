# AVQA-R1-6K：2748 / 16 NPU 配方迁移

核对日期：2026-11-12。目标是保留官方公开的 **GPU LoRA 训练配方**，将初始模型替换为 2748、资源替换为单节点 16 张 Ascend NPU。不是仓库另一份 NPU 全参数 AVQA 配方，也不是 Huanyu AQA/GenRM 配方。

发布布局：GitHub `new` 分支中的位置为 `qwen3_6_huanyu/aqav/`；`aqav` 专用分支根目录
只包含本目录文件。服务器使用时均放回 `$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav/`。
专用分支不是独立训练环境，仍需最新 Huanyu 实现及 `verl_omni_ext/probes/hccl_policy.py`。

## 已核对的来源与尚未验证的内容

- [官方实验说明](https://github.com/verl-project/verl-omni/tree/main/examples/gspo_trainer) 将曲线标为 `release/v0.2.0`、Qwen3-Omni-30B-A3B-Instruct、4×H800、LoRA r=32；报告训练 reward 约从 0.73 到 0.94，验证 reward mean@1 达到 0.877、348 step。
- [公开 GPU AVQA 启动脚本](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh) 是本目录训练参数的来源；当前 `main` 不等于该 W&B run 的固定代码版本。
- [原始数据](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K/tree/main) 是图片 + 音频 + 选择题文本，不是此前的 `aqa_dataset` 或视频数据。原始划分为 4,490 train / 1,911 valid；实际转换与长度过滤后的数量必须以日志为准。[原作者项目](https://github.com/HarryHsing/EchoInk)
- 下载锁定数据 revision `1c380427f66407f9f4ece354d84f804e4a16e522`，归档 SHA256 为 `6ad5d4bc00a3367aa36d2b0ed062a0a53c8d701242110c43826541138522f388`。归档约 5.47 GB，需额外保留解压空间。
- 同目录 `avqa.py` 和 `choice_reward.py` 分别是官方 `examples/gspo_trainer/data_process/avqa.py`、`verl_omni/utils/reward_score/choice_reward.py` 的原样副本，保留版权和许可声明。`prepare.py` 和 `run.sh` 直接引用本目录文件，转换与奖励加载不再依赖仓库根目录的对应文件。保留原 prompt、选项、带 `<answer>` 标签的 ground_truth；比较第一个 answer 标签的内容，完全相同得 1，否则得 0。没有 GenRM、格式奖励或长度奖励。

尚未直接取得 [W&B run](https://wandb.ai/verl-omni/gspo_demo/runs/kzzrq9pr) 的完整 config/history；访问该页面的浏览器流程被自动审批拦截，原因是涉及未授权的 `forge.coreweave.com` 域名，已停止访问。以上曲线数值来自官方 README，不能称作已独立复核的 run 数据。

本地测试只验证命令构造、数据标签、媒体路径和奖励契约。尚未在服务器加载 2748、验证 LoRA 梯度/权重同步或获得 reward 曲线。

### 2026-11-16：在同步后的最新代码上重测

本次沿用上述 LoRA 配方，保持 BS128、n16、LoRA32、TP2、16 NPU 和 choice reward，
通过 `AVQA_OPTIMIZED=1` 开启融合 GDN、RMSNorm、MoE 和 rollout 图模式。
此模式使用 `kernel_graph` 精度档案，同时关闭与融合 GDN 冲突的 FP32 参考实现。
不设置该变量时默认 `AVQA_OPTIMIZED=0`，保留原配方的算子路径，作为对照。

**此次暂不启用 route replay**：两种模式均显式设置 `HUANYU_EXPERT_REPLAY=0`、
`enable_rollout_routing_replay=false`、`enable_return_routed_experts=false`。
这不表示 LoRA 与 replay 原理上不兼容；当前回放钩子对 PEFT 包装的识别和完整训练链路尚待验证。
此入口不要求安装 replay 插件，不继承旧终端的 replay、权重同步优化或逐层调试开关。
融合 MoE 本身可以独立运行，关闭 replay 时使用当前模型正常计算的路由。

rollout 图模式未设置时默认开启；`AVQA_OPTIMIZED=1` 要求 `HUANYU_GRAPH_MODE=1`，
显式设置为 0 会提前报错。训练侧 `torch.compile` 仍关闭。
HCCL 明确使用 `aiv / AIV`，调用新探针检查环境一致性；该检查不代表设备通信算法实测。

先同步最新 Huanyu 代码、本次修改的 `run.sh` 和 `verl_omni_ext/probes/hccl_policy.py` 到服务器。
已有原始 parquet 无需重新下载或转换。已离线过滤的数据只有在 tokenizer、模板、
媒体预处理和长度上限一致时才继续复用；未确认一致时保留默认长度过滤。

环境和 Ray 已准备好时，可以先用完整 BS/N 跑 10 step，并对比训练前和第 10 步的验证：

```bash
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav"
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

运行后检查 `console.log` 中的 `[avqa] optimized=1 ... replay=0`，以及：

```bash
grep -aE '\[avqa\]|\[huanyu-kernel\]|trainable params|step:|Initial validation metrics|val-core/avqa_r1_6k/' "$RUN_DIR/console.log"
```

应有 `actor GDN fused kernel installed`、`actor npu_rms_norm installed`、
`actor fused MoE installed ... replay=off` 和 `graph mode requested; resolved cudagraph_mode=`。
这些日志用于确认安装和配置；图实际回放、LoRA 梯度、数值精度及吞吐仍需服务器运行验证。
本地 CPU 测试验证原配方参数、优化环境与实际 worker 精度策略一致、replay 三处关闭、
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

在已有环境和 Ray 的 head 容器执行。目录名统一为服务器使用的 `aqav`。同步整个本目录和现有 Huanyu 插件；本目录的以下五个文件需一并更新：

```text
run.sh
prepare.py
avqa.py
choice_reward.py
filter_data.py
```

不安装或替换服务器的 torch/CANN/vLLM 栈。Python 需要已有 `huggingface_hub`、`pandas`、`pyarrow`；训练还需要镜像的 PEFT/音视频依赖。

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export PYTHON_BIN=/usr/local/python3.12.13/bin/python3
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav"

# 1. 下载、校验、解压和转换；已有归档可追加 --archive /path/AVQA_R1.tar.gz。
"$PYTHON_BIN" "$AVQA/prepare.py" prepare

# 2. 检查标签、图片/音频路径和 2748 的 LoRA 候选层，不加载权重。
"$PYTHON_BIN" "$AVQA/prepare.py" check

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
AVQA="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav"
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
bash "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqav/run.sh" \
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

## 判断是否学到了有效行为

```bash
LOG=/path/to/this/run/console.log
grep -aE 'trainable params|lora_|LoRA|dataset len|filter dataset len' "$LOG"
grep -aE 'step:|Initial validation metrics|val-core/avqa_r1_6k/' "$LOG" | tail -n 30
```

比较完整基线与同一验证样本集合上的后续 `val-core/avqa_r1_6k/.../mean@1`，同时看 `critic/rewards/mean` 的多个 step 窗口，不能用单个高奖励 batch 或 smoke 的 16 条验证样本宣告上涨。保存实际过滤后的样本数；2748 与官方 tokenizer 不同可能改变保留集合和总步数，不能强行将 348 设为停止条件。

若 reward 不涨，先查输出是否有合法 answer 标签、`response_length/clip_ratio` 是否很高、组内奖励是否全同、LoRA 梯度是否非零、更新后的 rollout 权重是否同步，再查 actor/rollout 概率偏差和 KL。`actor/loss` 接近零本身不能证明未学习。不要为制造上升曲线改变答案标签、奖励权重或验证集。

先验证以上链路，再决定是否需要针对 2748 调参；一旦调整学习率、LoRA 范围或采样温度，应记录为后续实验，而不是官方参数复现。
