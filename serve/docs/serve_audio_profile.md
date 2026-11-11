# 2748 真实音频输入：TP=4 eager profile

该版本独立于训练。模型默认 `/opt/huawei/dataset/model_gy/downloaded_models/2748`，
音频默认 `/opt/huawei/dataset/infra_data/data/validation_vllm`，递归查找 WAV。
TP=4、设备 0/1/2/3，沿用原单 stage 的单副本部署（DP=1）。
基础 stage 和 NPU overlay 都写入 TP=4，不能只改客户端的 `--num-devices`。

客户端默认 BS=16 个 HTTP 请求、n=1、输出 3072 tokens、ignore_eos=true、
thinking=false、一次完整预热、一次正式批次；采集 START=1、ACTIVE=10、Level2。
`n=1` 避开该私有 Omni 接口已观察到的多 choice 丢失问题。

音频是实际输入，不是预先计算的 embedding。每条请求带一段 WAV 和固定文字问题，
输出 modalities=text。默认使用 `audio_url` 的 base64 data URI；如服务只支持另一种
输入结构，可显式传 `--audio-input-format input_audio`。不需要 tokenizer 参数。
协议参考 [Omni Chat Completions](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/chat_completions_api/)；
最终以预检结果为准，不假定私有分支与上游完全一致。

## 为什么使用不同音频

正式请求重复预热音频，可能命中 encoder embedding cache，导致实际不再调用音频
encoder；仅关闭 prefix cache 或 mm_processor_cache 不能证明没有这类复用。
参见 [EncoderCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/encoder_cache_manager.py)。
本脚本使用新的服务进程、关闭 prefix 和多模态预处理缓存，并在所有请求之间按原始
PCM SHA256 去重。文件名不同、PCM 相同也只选一次。不会给音频加噪声来伪造新输入。

默认需要 **33 段不同的有效音频**：1 段预检、16 段预热、16 段正式采集。
当前固定接收 **16 kHz、单声道、PCM16 WAV，长度 1–30 秒**；不裁剪、不重采样。
不合格和重复文件跳过并记录原因，数量不足则在发请求前报错。可调整时长筛选或 BS，
但应在比较报告中记录这个负载变化。选中路径、时长和 PCM hash 写入报告，音频 base64
和 API key 不写入报告。该过程只用于性能采集，不是 AQA 正确率评测。

## 终端 A：启动新服务

先停止上一轮文本 serve，确认原服务 worker 退出、0–3 卡可用。保留旧采集目录。
将这些文件放在同一固定的 `serve/` 目录：

- `run_serve_audio_profile.sh`、`bench_audio_profile.py`
- 本次更新后的 `bench_throughput.py`、`serve_forward_profile.py`
- `serve_profile_report.py`

```bash
cd /opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export PROFILE_RUN="$PWD/audio_profile_tp4_$(date +%Y%m%d_%H%M%S)_$$"
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
bash run_serve_audio_profile.sh serve
```

脚本安装幂等的 profiler 条件入口，在新目录生成 `audio_eager_tp4.yaml`，再启动服务。
不会安装依赖、修改 checkpoint、启动训练或 Ray。prepare 保持原来的 max_num_seqs、
max_model_len、max_num_batched_tokens 和显存比例，禁用图模式与 prefix cache；
音频入口额外设置 TP=4、devices=0,1,2,3、mm_processor_cache_gb=0。
启动日志需核对实际 TP=4、enforce_eager=True、cudagraph_mode=NONE。

默认从 `/vllm-workspace/vllm-omni` 查找 runner 和 deploy YAML；自定义安装可设置
`VLLM_OMNI_ROOT` / `DEPLOY_SOURCE`。必须与当前 `vllm serve` 实际导入路径一致，
脚本不会通过 PYTHONPATH 偷换服务版本。

## 终端 B：预热与正式音频请求

粘贴终端 A 打印的 PROFILE_RUN、TOOLS_DIR、HUANYU_SERVE_PORT、SERVED_MODEL。

```bash
cd "$TOOLS_DIR"
export AUDIO_DIR=/opt/huawei/dataset/infra_data/data/validation_vllm
BATCH_SIZE=16 OUTPUT_TOKENS=3072 bash run_serve_audio_profile.sh request
```

等价的独立客户端命令：

```bash
python3 bench_audio_profile.py \
  --host 127.0.0.1 --port 28889 --model qwen3.8 \
  --audio-dir /opt/huawei/dataset/infra_data/data/validation_vllm \
  --batch-size 16 --n 1 --temperature 1.0 \
  --output-tokens 3072 --ignore-eos --no-enable-thinking \
  --min-audio-seconds 1 --max-audio-seconds 30 \
  --seed 42 --timing-source none --num-devices 4 \
  --warmup-batches 1 --num-batches 1 \
  --profile-arm-file "$PROFILE_RUN/armed" \
  --report "$PROFILE_RUN/profile_audio_requests.json"
```

两种请求命令二选一。指定少量 WAV 时用独立客户端的重复 `--audio-file /path/a.wav`
参数代替 `--audio-dir`，不能同时使用；BS=1、一次预热和一次正式请求也需要 3 段不同音频。
如需更小负载，`BATCH_SIZE=4` 需要 9 段不同音频。

先用独立音频做 16-token 短预检，再做整批 3072-token 预热。全部成功才创建 `armed`，
此后发送新的正式音频。不要手工提前创建该文件，也不要同时给此服务发送其他请求。
预检/预热失败不触发采集；已经开始过采集则重启服务并使用新 PROFILE_RUN。

输入 tokens 包含服务端展开的音频占位符、提示词和 chat template，以服务端 usage 为准，
**不再宣称固定输入 2048 tokens**。输出仍逐请求校验为 3072，正式批次总计 49,152。
聚合输出吞吐 = 正式成功输出 token 总数 / 正式 wall time；每卡数再除以 4。
开启 profiler 的吞吐受采集扰动，不与纯文本无 profiler 吞吐直接作性能结论。

## 终端 C：确认音频分支确实被采集

```bash
cd "$TOOLS_DIR"
bash run_serve_audio_profile.sh status
bash run_serve_audio_profile.sh check
```

check 等价于：

```bash
python3 serve_forward_profile.py check \
  --profile-dir "$PROFILE_RUN/capture" --expected-workers 4 --require-audio
```

要求 4 个 TP rank 都有完整采集，并至少有 rank 实际执行了 Huanyu 音频 encoder
（`apm` / `audio_encoder`）和投影层（`audio_projection_layer` / `audio_projection`）。
逐 rank 输出 forward 调用次数，不假定每种媒体并行实现都在全部 rank 上执行编码器。
若缺失则失败，不能把“文件存在”当作音频采集成功。若私有模型改过模块命名，先查看
inventory 确认实际模块，再调整匹配，不要直接绕过检查。

请求结束且 check 通过后，可以停止本次服务，离线解析：

```bash
bash run_serve_audio_profile.sh analyse
```

其余设备算子、模块 CPU inclusive/self、未知模块和覆盖检查沿用
[完整 profile 说明](serve_forward_profile.md)。音频请求新增的 encoder/projector、
后续 GDN/Attention/MoE/HCCL 都在实际执行窗口内采集；不是只筛选音频相关算子。
START=1 窗口包含初始媒体编码和 prefill/早期 decode，不代表所有后续 decode。
运行于 API/processor 进程的 WAV 解码、fbank 等 CPU 预处理不在 TP worker profiler
范围内，不能拿该 trace 声称测量了整个 HTTP 音频链路的全部阶段。

如果仍刷 shared-memory 等待提示，先看 4 个 rank 的 phase。`profiler.stop` 可能等待
设备或数据收尾；缺少真实设备证据时不能据此断言某个 forward 算子卡死。

本地只做 CPU 回归测试和脚本语法检查，未在实际 2748/NPU 上声称跑通。
