# 独立 vLLM-Omni serve 压测与 eager profile

本目录与 `qwen3_6_huanyu/` 同级，可以单独复制到服务容器；无需导入训练插件或启动 Ray/FSDP。
当前 forward profile **只支持 eager**，会拒绝最终 stage 配置 `enforce_eager=False`。

| 文件 | 用途 |
| --- | --- |
| `bench_throughput.py` | 精确输入长度、真实 usage 输出计数、按 choice 计算 TTFT/TPOT；支持预热后触发 profile |
| `bench_audio_profile.py` | 真实 WAV 输入、不同音频预检/预热/采集、按服务端 usage 计数，n=1 |
| `run_serve_audio_profile.sh` | TP=4 音频 eager 的 serve/request/status/check/analyse 入口 |
| `prepare_benchmark_configs.py` | 前次吞吐 A/B 使用的配置生成工具，保留 TP、设备和容量参数 |
| `serve_forward_profile.py` | 安装/卸载独立条件入口、生成 eager 配置、采集进度与完整性检查、离线解析 |
| `serve_profile_report.py` | 汇总全部已导出设备算子、模块 CPU 时间与覆盖核对 |
| `tests/` | 无模型权重、无 NPU 的统计和包装器回归测试 |

## 运行说明

- **当前 eager 全模块采集**：[完整命令及结果解释](docs/serve_forward_profile.md)。
- **音频输入 TP=4 eager 采集**：[validation_vllm 启动与检查命令](docs/serve_audio_profile.md)。
- **独立吞吐测量**：[指标定义、启动及压测命令](docs/serve_throughput.md)。

命令针对已核对的 2748 单 stage 部署：vllm `0.28.0+empty`、
vllm-ascend `0.19.1rc2.dev2002+ge5118d151`、vllm-omni `0.28.0+npu`，
原配置在 NPU 上为 TP=2、设备 0/1。配置生成器读取容器实际 YAML，
不会改写 checkpoint 或已安装的 deploy YAML。

三个采集相关 Python 文件需保留在同一个固定目录。安装器会备份实际 runner，
只追加受 `HUANYU_SERVE_PROFILE_DIR` 控制的入口，保留当前 forward 和视频修复；
不设置该变量时普通 serve 不加载采集器。安装后需要启动新的服务 worker。

设备 profiler 保留采集窗口内实际执行并导出的算子，未知项也不丢弃；模块调用次数
与导出标签次数核对。纯文本不会覆盖媒体编码分支，融合 kernel 无法凭此进一步拆分。
模块 CPU inclusive 时间、设备任务累计时间和端到端 wall time 是不同口径，不能相加
或直接相减。开启采集的请求报告标记 `profiling_run=true`，不作为正常吞吐基准。

## CPU 验证

依赖：`pytest`、`requests`、`PyYAML`。纯文本压测还需要服务环境的 `transformers`
及 checkpoint tokenizer；音频客户端无需 tokenizer。NPU 采集和解析需要对应容器的 `torch_npu` 与 CANN。

从仓库根目录执行：

```bash
python3 -m pytest -q serve/tests
```

本地 CPU 测试不替代目标 NPU 上的采集完整性、实际吞吐或性能结论。
