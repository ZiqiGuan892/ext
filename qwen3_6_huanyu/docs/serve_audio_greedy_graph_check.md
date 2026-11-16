# TP4：音频输入的 eager / graph 输出与 logits 对照

沿用纯文本 greedy 对照的采集位置，改成 5 条真实 WAV。每组逐条发送相同音频，
比较完整生成结果和前 100 个输出位置的全词表原始 logits。这里没有训练或 profiler，
也不测吞吐。CPU 测试验证工具逻辑；实际是否一致以服务器两组采集后的报告为准。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 模型 | `/opt/huawei/dataset/model_gy/downloaded_models/2748` |
| 输入 | `validation_vllm` 中按 seed=42 选取 5 条不同 PCM 波形 |
| WAV | 单声道、16 kHz、PCM16、1–30 秒；不裁剪、不重采样 |
| 固定输入 | 准备阶段将 5 条 WAV 快照存到本地 `/tmp`，两组复用同一快照及问题文字 |
| 并行 | TP=4、单副本 DP=1、设备 0–3；请求并发 1 |
| 两组 | eager / PIECEWISE 图模式，分别重启服务 |
| 关闭项 | prefix cache、MM processor cache、强制确定性计算、batch invariant、thinking、profiler |
| 解码 | greedy：temperature=0，top_p=1，n=1，repetition_penalty=1，presence/frequency_penalty=0 |
| 输出 | 默认 3072 tokens，ignore_eos=true，min_tokens=max_tokens；可改为 100 |
| 输入长度 | 以展开后的真实 prompt token IDs 和服务端 usage 为准；2048 是预算，不是固定输入长度 |
| logits | 前 100 个**输出位置**的完整词表，保留原 dtype 的字节；不是 top-100，也不是 logprobs |

位置 1 是 prefill 最后一个位置预测第一个输出 token 的 logits；不采集其他 prompt
位置。输出 token IDs、文本及 reasoning 对比覆盖全部生成结果。图模式组必须在前
100 个位置实际执行图 replay，不能仅凭 YAML 声称图模式生效。

## 使用方法

配套文件放在同一个目录：`run_serve_greedy_check.sh`、`serve_greedy_check.py`、
`prepare_benchmark_configs.py`、`bench_audio_profile.py`、`bench_throughput.py`。
不需要 Git，不修改 checkpoint 或原 deploy YAML。准备命令会备份并安装一个由
`HUANYU_GREEDY_CASE` 环境变量控制的 NPU runner 入口钩子；结束后可以移除。

先停止之前的测试服务。终端 A：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve_audio_greedy
export VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export AUDIO_DIR=/opt/huawei/dataset/infra_data/data/validation_vllm
export AB_RUN="$(mktemp -d /tmp/huanyu_audio_greedy_tp4.XXXXXX)/case"
export OUTPUT_TOKENS=3072
export INPUT_TOKEN_BUDGET=2048
export HUANYU_SERVE_PORT=28889

bash "$TOOLS_DIR/run_serve_greedy_check.sh" prepare-audio
bash "$TOOLS_DIR/run_serve_greedy_check.sh" serve eager
```

准备时优先读 `validation_vllm/manifest.jsonl`，避免遍历 OBS 挂载目录。如果索引在
其他位置，设置 `AUDIO_MANIFEST=/完整路径/manifest.jsonl`。没有索引时才递归扫描 WAV；
日志会提示。输入快照建好后，两组不再扫描源目录。选样结果及 PCM SHA256 在
`case.json` 中。

终端 B，设置相同 TOOLS_DIR，并复制 prepare 打印的真实 `export AB_RUN=...`：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve_audio_greedy
# 在这里粘贴终端 A 打印的 export AB_RUN=...
bash "$TOOLS_DIR/run_serve_greedy_check.sh" request eager
```

看到 `Completed 5 inputs` 后，在终端 A 按 Ctrl+C 停止 eager 服务。确认该服务及
其 worker 已退出，再在终端 A 启动 graph：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" serve graph
```

服务就绪后，终端 B：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" request graph
bash "$TOOLS_DIR/run_serve_greedy_check.sh" compare
```

两组各 5 条请求，无额外预检、预热请求。若想把完整输出也限制为 100 tokens，
在新目录 prepare 前设置 `OUTPUT_TOKENS=100`。失败重试使用新的 AB_RUN，保留旧证据；
不要向这个服务发送其他请求，也不要在两组之间换模型、代码、精度配置或设备。

## 如何读结果

比较结果写入 `$AB_RUN/comparison.json`，每条样本包含：

- `audio_input_verified`：客户端 WAV 快照校验、请求身份和实际音频编码执行均通过。
- `expanded_prompt_token_ids_equal`：两组所有 TP rank 的实际音频占位符展开结果一致。
- `audio_features_bitwise_equal`：两组同 rank 的实际音频特征、长度等输入指纹一致。
- `audio_embeddings_bitwise_equal`：两组同 rank 的音频编码及投影返回值指纹一致。
- `output_token_ids_equal`、`output_text_equal`：完整生成结果是否一致。
- `logits_first100_bitwise_equal`：前 100 个生成位置原始 logits 是否逐位一致。
- `graph_positions_with_replay_per_tp`：各 TP rank 在采集窗口中的实际图 replay 位置数。
- `graph_decode_positions_with_replay_per_tp`：其中输出位置 2–100 的 decode replay 位置数；
  各 rank 均须大于 0，只有音频编码或 prefill replay 不算通过。

末尾 `OUTPUT_EQUAL=True LOGITS_FIRST100_BITWISE_EQUAL=True` 表示这次 5 条音频测试
全部通过。音频特征/embedding 指纹用于定位差异，不用来替代 logits 比较。相同长度
的音频可能具有相同占位符 token IDs，因此工具通过显式 request_id 匹配样本，不能
只按 prompt IDs 匹配。若后端不保留 request_id，会明确报错，不按到达顺序猜测。

如果某一输出 token 开始不同，工具仍报告全部 100 个位置是否相同，但误差统计只
包含生成历史尚一致的位置；后面的差异不能直接归因于同一输入下的图计算误差。
关闭确定性后，单次不一致也不足以证明是图模式导致，必要时再做 eager/eager 重复对照。
单请求的结果不代表高并发或不同 batch shape 下都逐位一致。

采集包含同步拷贝和落盘，耗时不代表纯推理吞吐。全词表 logits 在四 rank 上若为复制
布局，会分别保存并校验；报告中的比较元素数含各 rank。默认 float32、词表约 248K
时，两组 5×100×4 份向量约占 4 GB，另有模型图编译缓存和日志。

测试结束停止服务后，可移除入口钩子：

```bash
bash "$TOOLS_DIR/run_serve_greedy_check.sh" remove-hook
```

工具不会关闭或修改训练模型的音频前端、编码器、投影层或精度开关。
