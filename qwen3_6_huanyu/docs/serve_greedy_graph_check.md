# TP4：greedy 下 eager 与图模式的输出及前 100 步 logits 对比

真实 WAV 输入使用同一个工具的 `prepare-audio` 入口，见
[音频 greedy 图模式对照](serve_audio_greedy_graph_check.md)。

本工具准备同一组 **5 条纯文本输入**，分别启动 eager 和 PIECEWISE 图模式服务；
关闭 prefix cache、强制确定性算法、batch invariant 和 profiler。它用于判断本次实际
输出与原始 logits 是否一致，不是吞吐测试。本地 CPU 测试不代替服务器 NPU 实测。

## 对比范围

| 项目 | 设置 |
| --- | --- |
| 模型 | `/opt/huawei/dataset/model_gy/downloaded_models/2748` |
| 并行 | TP=4、单副本 DP=1，设备 0–3 |
| 数据 | 5 条不同的合成纯文本，沿用文本压测的 prompt 构造；不是 AQA 音频 |
| 输入长度 | 每条含 chat template 精确 2048 tokens |
| 发送方式 | 同样顺序，逐条发送，并发 1；不额外发送预检/预热样本 |
| 解码 | temperature=0、top_p=1、top_k=-1、n=1；repetition_penalty=1，presence/frequency_penalty=0 |
| thinking / prefix cache | 均关闭 |
| 输出 | 默认沿用之前的 3072 tokens，ignore_eos=true、min_tokens=max_tokens，避免提前结束 |
| 服务长度与调度预算 | 两侧 `max_model_len=输入+输出`（默认 5120）；`max_num_batched_tokens` 保留原有效预算，至少覆盖完整上下文 |
| logits | 仅保存生成位置 1–100，每个位置的完整词表向量；不是词表 top-100 |
| 对齐 | 保存实际 prompt token IDs，逐步核验实际生成历史；关闭异步调度、chunked prefill 和 speculative decoding |

请求使用 `top_k=-1`，但 vLLM 0.28 在初始化 greedy 参数时会将其规范化成 `top_k=0`；
两者均表示不做 top-k 截断，参见
[SamplingParams.__post_init__](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/sampling_params.py#L492-L497)。
工具接受这两种表示，并在记录中保留实际值，仍要求 temperature=0、all_greedy=True、
all_random=False，且实际调用 greedy_sample。旧版只接受 -1 会报
`Actual SamplingParams differ from greedy case` 并使 worker 退出；需要更新采集工具、
重启服务，在新的 AB_RUN 中重新采集两组，不能沿用失败目录的部分数据。

输出的 token IDs 和文本比较整个生成结果；只有 logits 的存储限制为前 100 步。
如果也只需输出 100 tokens，在 `prepare` 前设置 `OUTPUT_TOKENS=100`，两组共用该配置。
这是并发 1 下的检查，不能推广成之前并发 16 下也逐位相同。

关闭 chunked prefill 时，vLLM 要求 `max_num_batched_tokens >= max_model_len`。
prepare 会同时修正基础 stage 和 NPU 覆盖层；默认原 YAML 的 32768 上下文改为本次
2048+3072=5120，16384 的调度预算保持不变。仅改变输入或输出长度时会重新计算，
不会把输入输出截短。修复后的配置必须在新 `AB_RUN` 中重新 prepare；旧目录保留作失败记录。

“输出位置 1”的 logits 来自 prefill 最后一个位置，用于预测第一个生成 token，因此需要
保留；prompt 其他位置的 logits 不收集，也不请求 `prompt_logprobs`。
在实际 `model.compute_logits` 返回后、采样器可能原地修改前，将原 dtype 的字节复制到 CPU。
不通过 softmax/logprobs 反推 logits，不重跑模型 forward，也不强制两边采用同一生成历史。
vLLM 的 logits 和 logprobs 处理边界可参见
[v0.28.0 Sampler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/sampler.py)。

每个 TP rank 保存完整返回向量。如果使用词表分片，则依据实际 lm_head 的词表索引核验
四卡覆盖完整词表，比较时排除 padding；无法证明覆盖完整时明确报错。原 dtype 和原始
字节被保留，报告分别给出逐位相同和数值相同，负零/正零也能区分。

## 配套文件

将以下文件放在同一个固定 `serve` 目录，不需要 git：

- `run_serve_greedy_check.sh`
- `serve_greedy_check.py`
- `bench_throughput.py`
- `prepare_benchmark_configs.py`

准备步骤会备份并在已安装 `NPUARModelRunner` 文件末尾加入一个环境变量控制的入口；
不替换模型 forward。`HUANYU_GREEDY_CASE` 未设置时不加载采集工具。
运行前核对该目录与当前 `vllm serve` 实际导入路径一致，启动脚本会自动检查。

## 终端 A：准备并运行 eager

先停止上一轮 serve，确认其 worker 已退出、0–3 卡可用。采集写到本地 `/tmp`。
不要在本次目录重跑同一侧，也不要向该服务发送其他业务请求。

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
export AB_RUN="/tmp/huanyu_greedy_graph_tp4_$(date +%Y%m%d_%H%M%S)_$$"

# 不要提前 mkdir：prepare 要求新目录。
INPUT_TOKENS=2048 OUTPUT_TOKENS=3072 bash "$TOOLS_DIR/run_serve_greedy_check.sh" prepare
bash "$TOOLS_DIR/run_serve_greedy_check.sh" serve eager
```

脚本会清除 profiler、旧 A/B 确定性入口及编译禁用开关，设置：

```text
HUANYU_JOB_DETERMINISM=0
VLLM_BATCH_INVARIANT=0
HCCL_DETERMINISTIC=false
ASCEND_LAUNCH_BLOCKING=0
NCCL_DETERMINISTIC=0
LCCL_DETERMINISTIC=0
```

每个 worker 还会执行并校验 `torch.use_deterministic_algorithms(False)`，并核实实际
prefix cache 关闭、greedy 参数生效。不是只检查启动终端中的环境变量。

## 终端 B：5 条 eager 请求

复制 prepare 打印的 `export AB_RUN=...` 到终端 B，等待终端 A 服务就绪：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
# 先粘贴终端 A 打印的 export AB_RUN=...，两终端必须相同。
bash "$TOOLS_DIR/run_serve_greedy_check.sh" request eager
```

成功后显示 5 条请求完成。若请求失败或出现兼容性错误，保留目录和完整日志；工具不会
跳过错误给出“一致”的结论。

## 重启为图模式，再发送同一 5 条

在终端 A 按 Ctrl+C 停止 eager，确认 worker 退出，保留 `AB_RUN`：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" serve graph
```

图服务完成启动后，在终端 B 执行：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" request graph
bash "$TOOLS_DIR/run_serve_greedy_check.sh" compare
```

四卡均应出现 `SETTINGS_VERIFIED`，每条完成日志显示 `logits=100`。
工具在 `torch_npu.npu.NPUGraph.replay` 的 Python 边界计数实际 replay；eager 要求计数为 0，
图组要求四卡在前 100 个输出位置内均观察到 replay。图模式配置开启但没有观察到 replay，
会报告“未验证”，不当作有效图对照。如果私有分支绕过该 replay 接口，需要按其实际调用
路径适配计数；不能删除这个检查后声称图模式已生效。

## 读取结果

报告：`$AB_RUN/comparison.json`。汇总末行类似下面的字段定义，不是预设结果：

```text
OUTPUT_EQUAL=<True/False> LOGITS_FIRST100_BITWISE_EQUAL=<True/False>
```

每条样本包含：

- `output_token_ids_equal`：整段生成 token IDs 是否完全相同。
- `output_text_equal`、`output_reasoning_equal`：API 返回的文本及 reasoning 是否相同。
- `logits_first100_bitwise_equal`：前 100 个生成位置的完整有效词表 logits 原 dtype 字节是否相同。
- `first_different_output_position`、`first_different_logits_position`：首个差异位置，从 1 开始；无差异为 null。
- `same_context_logits_positions`：两边输入历史相同、可作数值比较的位置数。
- `same_context_metrics`：逐位/数值相同、差异元素数、有限值最大/平均绝对差、mask 位置是否一致。
- `graph_positions_with_replay_per_tp`：每卡在前 100 步中实际发生 replay 的生成位置数。

如果第 k 个输出 token 首次不同，**第 k 步的 logits 仍有相同历史，可以比较**；第 k+1 步
开始历史已分叉，不能把之后的数值差都归因于图模式。报告会标明并排除这些位置的同历史
误差统计。差异元素计数按所有 TP 观察向量累计，完整复制词表时包含各卡重复观察。

输出一致并不要求 logits 逐位一致，微小差异可能不改变 greedy 选中的 token。
本次主动关闭确定性，因此一次跨模式对比只能判断这一次的 5 条数据；若出现差异，还不能
排除同模式自身的运行波动。需要进一步归因时，再对同一 5 条输入做 eager/eager、graph/graph
重复对照；本工具不会擅自增加这些请求。参见
[PyTorch 可重复性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

全词表采集会增加 CPU 复制、同步和磁盘开销。约 25 万词、FP32、每卡完整复制词表时，
两组总数据约 4 GB；请保留在本地 `/tmp`，不要将这里的时延用于吞吐结论。

## 结束后

停止当前 serve 后，可移除条件入口；只移除本工具标记段，保留原文件其他代码：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" remove-hook
```

若保留入口，普通 serve 未设置 `HUANYU_GREEDY_CASE` 时不启用。测试脚本在子进程中设置
变量，不会把采集环境留在调用终端。模型、精度开关和权重未被此工具修改。
