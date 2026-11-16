# Huanyu 2748：TP4 纯文本图模式 × Prefix cache 四组对照

记录日期：2026-11-11。四组已完成客户端测量，下表依据用户回传的汇总命令输出记录；
本地没有直接读取服务器的原始 JSON。图执行及 prefix cache 命中仍需服务端日志/指标验收。

原参照文档中“图模式”和“prefix cache”的启动命令相同，都传了
`--enable-prefix-caching` 和 `--compilation-config`；原始 NPU deploy YAML
本身也已配置 `enforce_eager=false`、`cudagraph_mode=PIECEWISE`。
因此原来约 60% 的提升不能直接归因于图模式，需用以下四组分离变量。

| 组别/文件后缀 | 图模式 | Prefix cache | 用途 |
| --- | --- | --- | --- |
| A `baseline` | 关闭：eager / NONE | 关闭 | 本轮基线 |
| B `graph_only` | 开启：PIECEWISE | 关闭 | 测图模式收益 |
| C `prefix_only` | 关闭：eager / NONE | 开启 | 测缓存收益 |
| D `graph_prefix` | 开启：PIECEWISE | 开启 | 测组合收益 |

## 本轮实测结果

四组汇总脚本均通过：客户端参数、模型路径/别名、输入语料 hash 一致，未启用客户端
profile，每组正式 48 请求、147456 个输出 tokens、0 个失败请求，输入均为 2048 tokens、
输出均为 3072 tokens，预检和 1 批预热未计入正式吞吐。
这些检查验证了客户端侧可比性，不代替服务端实际图模式和缓存状态的验证。

| 组别 | 正式总时间（s） | 聚合输出吞吐（tok/s） | 平均单卡（tok/s/卡） | 相对 A 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| A：两者关闭 | 1053.808 | 139.927 | 34.982 | 基线 |
| B：只开图模式 | 313.732 | 470.006 | 117.502 | +235.89% |
| C：只开 prefix cache | 1063.751 | 138.619 | 34.655 | −0.93% |
| D：图模式与缓存都开 | 329.733 | 447.198 | 111.800 | +219.59% |

时间和吞吐保留用户输出的精度；单卡值是聚合吞吐按 4 张卡均摊，不代表 TP=1 的实测性能。

本轮表现最好的是 **B：只开图模式**，吞吐为 A 的 **3.36 倍**，完成相同输出的时间
减少约 **70.23%**。D 的吞吐为 A 的 **3.20 倍**，但相比 B 低 **4.85%**。
C 与 A 接近，低 **0.93%**；D 相比 C 的吞吐高 **222.61%**。
无论缓存开关如何，图模式开启组均明显快于对应关闭组，本轮主要收益集中在图模式配置。

**本次 prefix cache 没有显示总吞吐收益，但还不能判定缓存功能无效或必然拖慢。**
四份报告都未提供 `usage.prompt_tokens_details.cached_tokens`，尚不清楚 C/D 的实际命中情况。
−0.93% 和 −4.85% 只是本轮观测差值；缺少重复实验、逐批波动与服务端命中证据，
不能把它们确认为缓存的固定开销。长输出负载下，prefill 的收益还可能被后续生成时间稀释。

后续先按下方“开关生效验收”读取已有日志和 metrics，确认 B/D 的有效图模式、
C/D 的缓存开启及命中。如果要判定缓存带来的几百分点差异，再用新目录逆序复测四组。
这些吞吐结果不能单独定位 forward 内某个算子、HCCL 或同步等待的耗时变化，
也不提供 TTFT/TPOT 结论；没有将之前的 Level2 profile 耗时混入本表。

## 固定负载及缓存口径

四组均使用模型 `/opt/huawei/dataset/model_gy/downloaded_models/2748`，
单 stage、单副本、TP=4、DP=1，设备 `0,1,2,3`，关闭 profiler。
不启用训练、权重同步或奖励模型。沿用用户提供镜像的版本：
vllm `0.28.0+empty`、vllm-ascend `0.19.1rc2.dev2002+ge5118d151`、
vllm-omni `0.28.0+npu`。

| 参数 | 四组统一设置 |
| --- | --- |
| 输入 | 纯文本，含 chat template 精确 2048 tokens |
| 输出 | 每请求 3072 tokens；ignore_eos=true |
| 请求并发 / 每请求生成数 | 16 / n=1 |
| 采样 | temperature=1.0，thinking=false，基础 seed=42 |
| 请求种子 | seed + request_id；四组一致 |
| 预检 / 预热 / 正式测量 | 1 条短预检 / 1 批完整预热 / 3 批正式请求 |
| 正式数量 | 48 请求，共 147456 个输出 tokens |
| 时延元数据 | timing-source=none；本次不测 TTFT、TPOT |
| 调度容量 | max_num_seqs=16，max_num_batched_tokens=16384 |
| 上下文 / 显存比例 | max_model_len=32768，gpu_memory_utilization=0.9 |

新增客户端参数 `--prompt-mode repeat`：所有请求使用同一条精确 2048-token
输入，包括预检、预热和正式请求。这与原参照脚本重复输入的方式一致，但使用
修正后的精确长度和统计口径。开启缓存的两组测量的是**重复输入、预热后缓存可复用**的吞吐，
不是首次请求或无共享前缀流量的性能。客户端默认仍为 `distinct`，不改变历史测试默认行为。

四组都必须使用 `repeat`，不能只给缓存组改输入。
[之前的 143.520 tok/s 文本结果](serve_throughput_results_20261111.md) 使用了不同 Case 标识的输入，
保留作历史参考；**本轮必须重测 A，不能用该历史值作提升比例的分母**。

Prefix cache 复用匹配前缀的 KV，主要减少 prefill，不直接加速新 token 的 decode。
本负载输出较长，即使命中缓存，总吞吐增益也可能有限，不能预设提升幅度。
参见 [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)。

## 一次性准备

将本次配套的 `bench_throughput.py` 和 `prepare_benchmark_configs.py` 放在同一
`serve` 目录。无需 git，也不修改已安装的原始 YAML。

在终端 A 执行：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export BENCH_RUN=$(mktemp -d /tmp/huanyu_text_graph_prefix_tp4.XXXXXX)

python3 "$TOOLS_DIR/prepare_benchmark_configs.py" \
  --source /vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml \
  --output-dir "$BENCH_RUN/configs" \
  --four-way --tp 4 --devices 0,1,2,3

printf '复制到终端 B：\nexport BENCH_RUN=%q\n' "$BENCH_RUN"
```

生成器同时设置基础 stage 和 NPU 覆盖项的图模式、缓存和 TP，其他配置保持一致。
核对打印出的四份有效配置均为上表容量、TP=4、devices=0,1,2,3；如果源 YAML
已经变更，应先统一容量，再重生成整套配置。不要给某一组单独增加并发或显存预算。
图组沿用本版本的 capture size 推导，不把输入/输出长度 2048/3072 当作 capture batch size。

## 终端 A：逐组启动独立服务

先在原服务终端按 Ctrl+C 停止旧 serve，确认其 worker 已退出、0–3 卡空闲。
四组顺序运行，每组都重启服务，不并行占用同一组设备。

在终端 A 定义以下函数；每次都在同一 shell 中调用以保持其余环境一致：

```bash
start_variant() (
  set -euo pipefail
  case "${1:-}" in
    baseline|graph_only|prefix_only|graph_prefix) variant="$1" ;;
    *) echo '用法：start_variant baseline|graph_only|prefix_only|graph_prefix' >&2; exit 2 ;;
  esac
  : "${BENCH_RUN:?请先运行一次性准备}"
  test -f "$BENCH_RUN/configs/huanyu_${variant}.yaml"
  mkdir -p "$BENCH_RUN/$variant"
  if [ -e "$BENCH_RUN/$variant/serve.log" ]; then
    echo '该组已有日志；重测请新建整轮 BENCH_RUN，保留现有结果。' >&2
    exit 2
  fi
  for name in ${!HUANYU_SERVE_PROFILE_@} ${!HUANYU_FORWARD_PROFILE_@}; do
    unset "$name"
  done
  unset HUANYU_GEN_TIMING VLLM_TORCH_PROFILER_DIR VLLM_OMNI_EXTERNAL_MODULES
  unset TORCH_COMPILE_DISABLE TORCHINDUCTOR_DISABLE NPU_GRAPH_CAPTURE_DISABLE VLLM_TORCH_COMPILE
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

  vllm serve "$MODEL_PATH" \
    --omni --port 28889 --log-stats --trust-remote-code \
    --stage-init-timeout 1800 --served-model-name qwen3.8 \
    --deploy-config "$BENCH_RUN/configs/huanyu_${variant}.yaml" \
    2>&1 | tee "$BENCH_RUN/$variant/serve.log"
)

start_variant baseline
```

图模式和 prefix cache 只由各自 YAML 控制，不再在四组共用的 CLI 中追加
`--enforce-eager` 或 `--enable-prefix-caching`。图编译、capture 和客户端预热均不计入正式吞吐。

## 终端 B：统一客户端

粘贴终端 A 打印的 `export BENCH_RUN=...`，然后执行：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748

bench_variant() (
  set -euo pipefail
  case "${1:-}" in
    baseline|graph_only|prefix_only|graph_prefix) variant="$1" ;;
    *) echo '用法：bench_variant baseline|graph_only|prefix_only|graph_prefix' >&2; exit 2 ;;
  esac
  : "${BENCH_RUN:?请粘贴终端 A 打印的路径}"
  test -f "$BENCH_RUN/$variant/serve.log"
  curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:28889/health >/dev/null
  curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:28889/metrics \
    -o "$BENCH_RUN/$variant/metrics.before.txt" || true

  python3 -u "$TOOLS_DIR/bench_throughput.py" \
    --host 127.0.0.1 --port 28889 --model qwen3.8 \
    --tokenizer "$MODEL_PATH" --trust-remote-code \
    --prompt-mode repeat \
    --batch-size 16 --n 1 --temperature 1.0 \
    --input-tokens 2048 --output-tokens 3072 --ignore-eos \
    --no-enable-thinking --seed 42 --timing-source none \
    --warmup-batches 1 --num-batches 3 --num-devices 4 \
    --timeout-s 1800 --report "$BENCH_RUN/$variant/text_throughput.json" \
    2>&1 | tee "$BENCH_RUN/$variant/benchmark.log"

  curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:28889/metrics \
    -o "$BENCH_RUN/$variant/metrics.after.txt" || true
)

bench_variant baseline
```

客户端应打印 `prompt_mode=repeat, unique_prompts=1`。每组结束后，在终端 A
按 Ctrl+C 停止该服务，确认 worker 退出，再继续下一行。终端 B 的组名必须对应当前服务。

| 顺序 | 终端 A（等待服务就绪） | 终端 B |
| --- | --- | --- |
| A | `start_variant baseline` | `bench_variant baseline` |
| B | `start_variant graph_only` | `bench_variant graph_only` |
| C | `start_variant prefix_only` | `bench_variant prefix_only` |
| D | `start_variant graph_prefix` | `bench_variant graph_prefix` |

上方两个代码块已分别启动 A，不要把表中 A 再执行一次。

## 开关生效验收

`/health` 成功只说明服务可访问，不能证明图执行或缓存命中。
每组检查实际引擎配置、图 capture/回退提示、缓存统计，保存完整日志：

```bash
for variant in baseline graph_only prefix_only graph_prefix; do
  printf '\n===== %s =====\n' "$variant"
  grep -aiE 'enforce_eager|cudagraph|graph.*(captur|replay)|prefix.*cach|fallback|falling back' \
    "$BENCH_RUN/$variant/serve.log" | tail -n 25
  for file in "$BENCH_RUN/$variant"/metrics.{before,after}.txt; do
    [ -f "$file" ] || continue
    printf '%s\n' "$file"
    grep -iE 'prefix.*cache|cache.*hit|cached.*token' "$file" | head -n 20 || true
  done
done
```

图组需确认有效配置未回退 eager，预热已完成 capture，并结合该版本提供的日志/计数
确认测量负载使用了图。仅生成 PIECEWISE 配置不证明 replay；Ascend 可因后端支持情况
调整最终模式，见 [Ascend Graph Mode](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)。

缓存组需确认实际开启且存在命中。若接口提供 `usage.prompt_tokens_details.cached_tokens`，
下方汇总会显示；若只提供服务端 cache hit/query counters，则比较前后增量。
Omni 私有分支可能不导出这些字段或指标，缺失时标为“未验证”，不能当成命中为零。
如果该模型/后端不支持 APC 或图模式，保留报错与有效配置，这一组不能冒充功能已生效的结果。

## 汇总及提升计算

聚合输出吞吐 = 正式输出 token 总数 / 三批 wall time 之和；平均单卡 = 聚合值 / 4。
不再乘并发 16，也不按各请求时延求和；不把预检、预热、组间重启时间计入分母。

四组完成后执行，校验客户端控制变量、语料 hash 和完整性，再重新计算吞吐：

```bash
python3 - "$BENCH_RUN" <<'PY'
import json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
names = ('baseline', 'graph_only', 'prefix_only', 'graph_prefix')
expected = dict(batch_size=16, n=1, input_tokens=2048, output_tokens=3072,
                temperature=1.0, ignore_eos=True, enable_thinking=False,
                seed=42, timing_source='none', warmup_batches=1,
                num_batches=3, num_devices=4, prompt_mode='repeat')
hashes, identities, rates = set(), set(), {}
for name in names:
    r = json.loads((root/name/'text_throughput.json').read_text())
    c, s = r['config'], r['summary']
    assert all(c.get(k) == v for k, v in expected.items()), (name, '客户端配置不一致')
    assert r.get('valid') and s and s['valid'] and s['num_errors'] == 0, name
    assert r.get('profiling_run') is False and r['probe']['success'], name
    assert len(r['warmup']) == 1 and len(r['batches']) == 3, name
    assert all(x['success'] for x in r['warmup'][0]['requests']), name
    hashes.add(r['corpus_sha256'])
    identities.add((c['tokenizer'], c['model'], c['trust_remote_code']))
    requests = [x for b in r['batches'] for x in b['requests']]
    assert all(len(b['requests']) == 16 for b in r['batches']), name
    assert all(x['success'] and x['prompt_tokens'] == 2048 and x['completion_tokens'] == 3072
               for x in requests), name
    total = sum(x['completion_tokens'] for x in requests)
    assert total == s['total_completion_tokens'] == 147456 and s['num_requests'] == 48, name
    wall = sum(b['wall_time_s'] for b in r['batches'])
    assert math.isfinite(wall) and wall > 0 and math.isclose(wall, s['total_wall_clock_s']), name
    rates[name] = total / wall
    assert math.isclose(rates[name], s['throughput_tokens_per_s']), name
    cached = [(x.get('usage', {}).get('prompt_tokens_details') or {}).get('cached_tokens')
              for x in requests]
    reported = [v for v in cached if type(v) is int and v >= 0]
    cache_note = (f'usage缓存字段覆盖 {len(reported)}/48 请求；合计 {sum(reported)} tokens'
                  if reported else 'usage未提供缓存命中，需结合服务端指标核实')
    print(f'{name:13s} wall={wall:.3f}s 聚合={rates[name]:.3f} tok/s '
          f'单卡={rates[name]/4:.3f} tok/s/卡；{cache_note}')

assert len(hashes) == len(identities) == 1, '四组输入或模型设置不同，不计算提升'
a, b, c, d = (rates[name] for name in names)
for label, numerator, denominator in (
    ('B/A：只开图', b, a), ('C/A：只开缓存', c, a), ('D/A：两者都开', d, a),
    ('D/B：图上再加缓存', d, b), ('D/C：缓存上再加图', d, c),
):
    print(f'{label}，吞吐提升={(numerator/denominator-1)*100:+.2f}%')
print('以上为测得比值；功能收益结论仍需通过图执行和缓存命中的服务端验收。')
PY
```

只有完成开关验收后，才能分别把 B/A、C/A、D/A 解读为该固定负载下的功能收益。
两项单独提升不能相加代替组合提升。输出长度、并发和重复前缀比例改变后，应重测。
三批仍属于一次实验；若差异很小，换新目录、逆序再跑一轮，先检查重复性再下结论。

本地验证覆盖配置四组仅开关不同、TP 同步覆盖、原文件不变、重复 prompt 精确长度、
预热排除及音频客户端兼容性；不代表已在 NPU 上验证图执行或实际缓存命中。
