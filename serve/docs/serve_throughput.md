# Huanyu 2748 单 stage 图模式吞吐测试

独立 eager serve 的全模块/NPU 算子采集见 [serve forward profile](serve_forward_profile.md)。
采集运行会扰动时延，不混入本文的正常吞吐 A/B。

本说明针对用户提供的镜像和配置：vllm `0.28.0+empty`、vllm-ascend
`0.19.1rc2.dev2002+ge5118d151`、vllm-omni `0.28.0+npu`。
本地只验证客户端统计及 YAML 变换，不声称已在该镜像/NPU 上完成性能实测。

## 已确认的原始配置

NPU overlay 覆盖基础 stage，所以实际是 TP=2、devices=`0,1`、
gpu_memory_utilization=0.9、max_num_seqs=16、max_num_batched_tokens=16384、
max_model_len=32768。任务名含 16cards 不代表这个 serve 用了 16 张卡。
原 YAML 的 enforce_eager=false、cudagraph_mode=PIECEWISE，已经请求使用图模式。
原先两个名为“图模式”和“prefix cache”的 CLI 完全相同，无法隔离二者收益。

下面的 A/B 保留上述容量和采样默认项，两组均关闭 prefix cache；只有
enforce_eager 和 cudagraph_mode 不同。graph 保留已配置的 PIECEWISE 模式，
capture sizes 沿用该版本自动推导，不把 2048/3072 输入输出长度当成 capture batch size。
不同时提高 max_num_seqs，以免把并发容量变化算成图模式收益。

## 文件与准备

把本包的 `bench_throughput.py` 和 `prepare_benchmark_configs.py` 放到服务器：
`/opt/huawei/dataset/hfy_guiyang/code/test_throughputs/`。
如果该目录已有旧压测脚本，先以新名字保存旧文件，再放入本版。
客户端使用当前镜像的 requests、transformers、PyYAML，不调用 pip 安装或修改服务源码。

在终端 A：

```bash
cd /opt/huawei/dataset/hfy_guiyang/code/test_throughputs
export BENCH_DIR="$PWD"
export BENCH_RUN="$BENCH_DIR/ab_$(date +%Y%m%d_%H%M%S)_$$"
python3 prepare_benchmark_configs.py \
  --source /vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml \
  --output-dir "$BENCH_RUN"
printf '终端 B 使用相同目录：\nexport BENCH_RUN=%q\n' "$BENCH_RUN"
```

脚本从已安装的 YAML 生成副本，包含基础 stage 和 NPU overlay 的显式设置，
不修改原 YAML；遇到多 stage、旧 schema 或冲突的 engine_extras 时直接报错。
两组服务必须依次运行，使用同样的空闲设备，不能同时占用 0/1 卡。

## 终端 A：启动服务

先设 `MODE=eager` 测基线。完成后在本终端 Ctrl+C 正常停止该服务，
确认该服务的 worker 已退出且端口释放，再将 MODE 改为 graph 执行同一段。
不要用全局 pkill/ray stop 清理其他作业。

```bash
MODE=eager                    # 第二次改为 graph
(
  set -o pipefail
  # 排除从训练终端继承的禁图开关；两组使用相同环境。
  unset TORCH_COMPILE_DISABLE TORCHINDUCTOR_DISABLE
  unset NPU_GRAPH_CAPTURE_DISABLE VLLM_TORCH_COMPILE
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  vllm serve /opt/huawei/dataset/model_gy/downloaded_models/2748 \
    --omni --port 28889 --log-stats --trust-remote-code \
    --stage-init-timeout 1800 --served-model-name qwen3.8 \
    --deploy-config "$BENCH_RUN/huanyu_${MODE}.yaml" \
    2>&1 | tee "$BENCH_RUN/serve_${MODE}.log"
)
```

不再附加 `--enable-prefix-caching` 或顶层 `--compilation-config`，避免覆盖对照配置。
保留需要的原生环境/自定义模型注册，两组不要额外加载不同的 ext precision hooks。

## 终端 B：检查与压测

粘贴终端 A 打印的 `export BENCH_RUN=...`，然后执行：

```bash
cd /opt/huawei/dataset/hfy_guiyang/code/test_throughputs
MODE=eager                    # 必须对应当前服务；第二次改为 graph
curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:28889/health
grep -aiE 'enforce_eager|cudagraph_mode|enable_prefix_caching|[Cc]aptur.*graph|graph.*[Cc]aptur|fall.*back|fallback' \
  "$BENCH_RUN/serve_${MODE}.log" | tail -n 30
```

必须核对最终 engine 配置：eager 为 enforce_eager=true/mode=NONE；graph 为
enforce_eager=false/mode=PIECEWISE，且有捕图成功日志，没有降级为 NONE/eager 的提示。
health 通过、YAML 写了 PIECEWISE 都不能单独证明图已捕获和用于执行。
如需确认实际 replay 而非仅捕图，另开短 profile 查看 ACLGraph replay；不要把开启
profiler 的那次运行纳入性能 A/B。发现 fallback 时，不能把它标为图模式结果。

正式压测：

```bash
python3 bench_throughput.py \
  --host 127.0.0.1 --port 28889 --model qwen3.8 \
  --tokenizer /opt/huawei/dataset/model_gy/downloaded_models/2748 \
  --trust-remote-code \
  --batch-size 16 --n 8 --temperature 1.0 \
  --input-tokens 2048 --output-tokens 3072 --ignore-eos \
  --no-enable-thinking --seed 42 \
  --timing-source token_ids \
  --warmup-batches 1 --num-batches 3 --num-devices 2 \
  --report "$BENCH_RUN/${MODE}.json"
```

客户端先用一条 n=8、每条输出 16 tokens 的短请求验证协议，再做一轮完整负载预热，
都排除在统计之外。正式运行应为 48 个 HTTP 请求、384 条序列、1,179,648 个输出 tokens。
HTTP 并发 16 不等于引擎同时解码 128 条；当前 max_num_seqs=16，其余由服务端排队。
每条消息在本地应用 chat template 后必须恰好 2048 tokens，并逐请求与服务器 usage 核对。
若模板/processor 不一致则失败，不能再称为固定 2K 输入。

客户端兼容 chat template 返回 token ID 列表、`BatchEncoding.input_ids` 和
单条 batch 的 token ID 数组；不能直接对 `BatchEncoding` 取 `len()`，那会数到
字段数而非 tokens。旧版因此可能在发请求前报 `Could not construct exact input length`。
更新 `bench_throughput.py` 后重试原命令即可；这种构造失败尚未发送请求，也未创建
profile arm 文件，无需因此重启 serve。真实模板长度仍由服务端 `usage.prompt_tokens` 校验。

使用非零 temperature 支持 n>1，固定 seed；本命令显式关闭 thinking，两组保持一致。
如需测试 thinking，可把两组都改成 --enable-thinking。新脚本的精确输入和采样口径与
旧脚本不同，因此应重新运行两组，不能直接与旧报告比较百分比。

如果该私有分支不返回逐 choice 的 token_ids，客户端会在短请求阶段报错，不生成虚假 TPOT。
可先加 `--probe-only` 单独查兼容性；或两组均改为 `--timing-source logprobs`，要求接口
支持逐 token logprobs。logprobs 有额外开销，不能与 token_ids 模式跨组比较。
只需要聚合吞吐时，两组可均用 `--timing-source none`，此时明确不报告 TTFT/TPOT。
没有精确的流式 token 元数据时，不回退到文本片段计数或平均分摊 n 路 token 数。

## 指标定义与有效性

- 聚合输出吞吐：所有测量 batch 的成功请求最终 usage.completion_tokens 之和，除以
  所有测量 batch 的 wall_time 之和。既不再乘 n/BS，也不除以各请求耗时之和。
- wall_time：提交该 batch 第一个请求之前，到全部响应解析和验证完成的客户端墙钟时间。
  tokenizer、prompt 构造、短预检、预热、batch 之间打印和 JSON 保存都在计时之外。
- 单卡输出吞吐：上述吞吐除以实际服务占用设备数；本 YAML 对应 2，非任务申请的 16。
- TTFT_i：从 HTTP 请求发出到 choice i 首批输出 token 到达客户端的时间。
- TPOT_i：(choice i 最后输出 token 到达时间 − 首批输出 token 到达时间)/(该 choice 的实际
  token 数 − 1)。对所有成功序列报告均值、median、nearest-rank p95，单位为秒。
  不把 8 路序列合并，不把尾部 usage/[DONE] 等待算作该序列的 decode 时间。
- token_ids / logprobs 从流中计数，并与最终 usage 总数核对；reasoning 和空文本 token
  也按元数据计数。多 token chunk 不等同于单 token，不报告伪造的逐 token ITL。
- 这是客户端观察到的指标，包含流式分组、网络与排队的影响，不是设备 kernel 时间；
  多个 token 一起到达时无法恢复内部生成时刻。整条输出一次性返回时拒绝计算 TPOT=0。
- 缺少 [DONE]/usage/choice finish、HTTP 200 内 error、n 不生效、输出不足 3072、token
  计数矛盾都视为失败。出现失败，报告 valid=false，正常吞吐字段为 null，并退出码 1；
  原始 successful_output_tokens_per_s 仅用于诊断，不能用于性能提升比较。

## 对比两个报告

完成 eager 和 graph 两组后，在终端 B：

```bash
python3 - "$BENCH_RUN/eager.json" "$BENCH_RUN/graph.json" <<'PY'
import json, sys
a, b = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]
for r in (a, b):
    assert r["valid"] and r["summary"]["valid"], "有失败请求，不能比较"
assert a["corpus_sha256"] == b["corpus_sha256"], "输入内容不同"
for key in ("model", "tokenizer", "input_tokens", "output_tokens", "batch_size", "n", "temperature",
            "ignore_eos", "enable_thinking", "seed", "timing_source", "warmup_batches", "num_batches", "num_devices"):
    assert a["config"][key] == b["config"][key], f"测试条件不同: {key}"
assert a["summary"]["total_completion_tokens"] == b["summary"]["total_completion_tokens"], "输出量不同"
for name, r in (("eager", a), ("graph", b)):
    s = r["summary"]
    print(name, "output tok/s=", s["throughput_tokens_per_s"],
          "per-device tok/s=", s["output_tokens_per_s_per_device"],
          "mean TPOT ms=", None if s["tpot_s"] is None else 1000*s["tpot_s"]["mean"])
print("吞吐提升 %=", 100*(b["summary"]["throughput_tokens_per_s"] / a["summary"]["throughput_tokens_per_s"] - 1))
PY
```

这个百分比表示固定负载下 PIECEWISE 相对于 eager 的端到端输出吞吐变化。
稳健比较可交替执行 A/B/A/B；每次均重启对应服务、独立预热、使用新的报告名。
固定相同日志级别、metadata 模式、输入 corpus、sampling、硬件占用和编译环境。

参考：
- [vLLM-Omni v0.28.0 stage 配置源码](https://github.com/vllm-project/vllm-omni/blob/v0.28.0/vllm_omni/config/stage_config.py)
- [vLLM-Omni v0.28.0 流式 token_ids](https://github.com/vllm-project/vllm-omni/blob/v0.28.0/vllm_omni/entrypoints/openai/serving_chat.py)
