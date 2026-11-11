# 2748 独立 eager serve：forward 全模块与 NPU 算子采集

本流程只采集 eager，使用用户已提供的 vllm 0.28.0+empty、
vllm-ascend 0.19.1rc2.dev2002+ge5118d151、vllm-omni 0.28.0+npu 部署。
默认保留原部署的 TP=2、卡 0/1、max_num_seqs=16、max_num_batched_tokens=16384、
显存比例 0.9。实际 engine 必须 enforce_eager=True，否则采集入口拒绝运行。
本地验证了 CPU 包装器和报告逻辑，尚未在该私有镜像/NPU 上实测。

## 采集范围与边界

- 不启动训练、FSDP、GenRM、AgentLoop 或权重同步；工具按独立文件加载，不 import
  `verl_omni_ext` 的训练注册入口，不安装 precision/video/weight_reload 等训练插件。
  如果当前 serve 原本依赖自定义注册，应保留必要注册，并检查 manifest 中实际已加载的包和环境。
- 只在已安装 runner 模块末尾追加带环境开关的入口。不替换任何 forward 函数体；
  模型加载后包装当前实例，所以已更新的视频 forward 参数、返回值和异常保持原样。
  实际文件路径、行号、SHA256、`__wrapped__` 链写入 inventory。
- 遍历所有 `model.named_modules()`，不是只挑 GDN、Attention、MoE。Embedding、Norm、
  RoPE、投影、router/expert、媒体编码器、未知新模块等均保留标签；还标记存在的
  `_forward_core`、`compute_logits`、runner 前后处理和独立 `sample_tokens`。
- 模块外的函数式算子由 CPU+NPU profiler 采集；Level2 请求 CANN Runtime、AI CPU、
  HCCL 等信息。所有导出的 kernel CSV 行和 trace 完整时长事件均保留，不按 Top-K、
  kernel 名称或类别删除未知项。原始 trace 的 flow、metadata、通信和拷贝视图保持原样。
- 不新增逐层 synchronize 或 collective；只标记已有的显式 synchronize。profiler
  自身 start/stop 仍可能同步及扰动执行。默认关闭硬件性能计数器、stack、shapes，减少附加开销。
- “全部”指采集窗口中实际执行且由采集器导出的事件。纯文本不会执行音视频编码器；
  decode 窗口不会覆盖所有 prefill 分支。未命中模块列入 coverage，不能当作零耗时。
  融合 kernel 的内部子算子不能凭这份时间线再拆分，也不能保证驱动永不丢事件。

## 1. 安装一次独立入口，生成 eager 配置

把本包的 `serve_forward_profile.py`、`serve_profile_report.py`、新版
`bench_throughput.py` 放到同一个服务器目录，例如原来的 `test_throughputs`。
使用新目录或先备份旧压测脚本。工具文件必须在服务运行期间保留在安装时的绝对路径。

在终端 A：

```bash
cd /opt/huawei/dataset/hfy_guiyang/code/test_throughputs
export TOOLS_DIR="$PWD"
export PROFILE_RUN="$PWD/eager_profile_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PROFILE_RUN"

python3 serve_forward_profile.py install \
  --runner-file /vllm-workspace/vllm-omni/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py

python3 serve_forward_profile.py prepare \
  --source /vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml \
  --output "$PROFILE_RUN/eager.yaml"

printf '复制到同一节点终端 B/C：\nexport PROFILE_RUN=%q\nexport TOOLS_DIR=%q\n' \
  "$PROFILE_RUN" "$TOOLS_DIR"
```

install 先备份 runner，再追加条件入口；重复执行不会叠加包装。prepare 只生成 eager
副本，关闭 prefix cache，原始 YAML 不动。没有设置 HUANYU_SERVE_PROFILE_DIR 的普通
serve 不导入此采集器。需要启动新服务，已经运行的 worker 不会自动获得入口。

## 2. 终端 A：启动独立 eager serve

确认 0/1 卡没有其他训练或独立服务占用。不能仅凭不启动训练就认为卡上无其他进程。
下面命令不改变原部署的资源容量，也不清理其他作业。

```bash
(
  set -o pipefail
  unset HUANYU_FORWARD_PROFILE_DIR VLLM_TORCH_PROFILER_DIR
  unset VLLM_OMNI_EXTERNAL_MODULES HUANYU_GEN_TIMING
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  export HUANYU_SERVE_PROFILE_DIR="$PROFILE_RUN/capture"
  export HUANYU_SERVE_PROFILE_ARM_FILE="$PROFILE_RUN/armed"
  export HUANYU_SERVE_PROFILE_START=1
  export HUANYU_SERVE_PROFILE_ACTIVE=10
  export HUANYU_SERVE_PROFILE_LEVEL=2
  export HUANYU_SERVE_PROFILE_SHAPES=0
  export HUANYU_SERVE_PROFILE_STACK=0

  vllm serve /opt/huawei/dataset/model_gy/downloaded_models/2748 \
    --omni --port 28889 --log-stats --trust-remote-code \
    --stage-init-timeout 1800 --served-model-name qwen3.8 \
    --deploy-config "$PROFILE_RUN/eager.yaml" \
    2>&1 | tee "$PROFILE_RUN/serve.log"
)
```

先核对启动日志的最终 TP/devices/enforce_eager/mode，以及实际导入 vllm-omni 路径。
代码会拒绝非 eager runner。此次默认一级 trace 标签覆盖所有模块，Level2 覆盖更多
底层任务；若需要输入形状或 Python 调用栈，可在一次单独采集中将 SHAPES/STACK 设为 1。
它们会增加开销，不能拿开启后的绝对时间与无 profiler 吞吐直接比较。

## 3. 终端 B：预热后触发采集

先粘贴终端 A 打印的 PROFILE_RUN 和 TOOLS_DIR，再执行：

```bash
cd "$TOOLS_DIR"
curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:28889/health

python3 bench_throughput.py \
  --host 127.0.0.1 --port 28889 --model qwen3.8 \
  --tokenizer /opt/huawei/dataset/model_gy/downloaded_models/2748 \
  --trust-remote-code \
  --batch-size 16 --n 8 --temperature 1.0 \
  --input-tokens 2048 --output-tokens 3072 --ignore-eos \
  --no-enable-thinking --seed 42 --timing-source none \
  --warmup-batches 1 --num-batches 1 --num-devices 2 \
  --profile-arm-file "$PROFILE_RUN/armed" \
  --report "$PROFILE_RUN/profile_requests.json"
```

使用本包新版客户端，`--help` 应含 `--profile-arm-file`。它先完成短协议预检和一轮
完整负载预热，再创建 armed 文件；任一预检/预热失败不会触发采集。不要提前 touch armed。
报告标记 profiling_run=true；这次请求耗时包含插桩和收尾，不作为正常吞吐基准。
`--timing-source none` 不要求私有服务支持流式 token_ids，只验证真实输入/输出 token 数。

START 从 armed 后的真实有 token 的 execute_model 调用开始计数，排除初始化 dummy 和预热。
START=1、ACTIVE=10 捕获请求初始窗口，可能同时包含 prefill/decode，不能把整段称为纯 prefill。
结束点是第 11 次 execute_model 的入口，保证第 10 次 forward 后的独立 sample_tokens
仍被覆盖。窗口结束后不再插桩。请求须有足够的迭代数触发这个结束点；过短请求或中途退出
会留下 incomplete，不能把它当作完整结果。

为覆盖另一类路径，第一轮完成后停止本测试服务，用新的 PROFILE_RUN 重复同一流程，
将 START 改成 100、ACTIVE 保持 10，采集较稳定的后续 decode 窗口。输入、N、并发、
模型和其他参数保持一致。用 manifest 的 scheduled_tokens/requests、可用的 attention
metadata 及 trace 核实实际阶段；采集器不会仅凭迭代号猜测 prefill/decode。

纯文本压测无法覆盖音视频算子。如果要分析实际视频/音频请求，可使用原来的真实媒体
请求客户端：先预热，确认两个 TP worker 的 READY；然后在同一节点创建 armed 文件，
再发送正式媒体请求。模块清单会区分媒体分支未执行与已执行但缺少 trace。

## 4. 终端 C：确认完整性，离线解析全部 rank

```bash
python3 "$TOOLS_DIR/serve_forward_profile.py" status \
  --profile-dir "$PROFILE_RUN/capture" --expected-workers 2

python3 "$TOOLS_DIR/serve_forward_profile.py" check \
  --profile-dir "$PROFILE_RUN/capture" --expected-workers 2
```

必须两个 TP rank 都 completed=10/10、status=complete，且框架、设备原始数据和结束
元数据完整。READY 会在第一次实际请求进入 runner 后出现，健康检查本身不会产生 manifest。
如果长期停在 `phase=profiler.stop`，说明 native profiler 收尾没有返回；不能归因于
forward 一直算不完。此前训练采集曾卡在最后 profiler.step；这里改成显式有边界的 stop
并取消在线解析，但不能保证修复 CANN/torch-npu 自身的收尾问题。

请求结束、check 通过后，可正常 Ctrl+C 停止本测试服务，再离线解析，避免 CPU 解析
竞争影响服务测量：

```bash
python3 "$TOOLS_DIR/serve_forward_profile.py" analyse \
  --profile-dir "$PROFILE_RUN/capture" --expected-workers 2
```

逐 worker 串行 analyse，解析器 max_process_number=1。只有生成/更新 trace_view.json、
kernel_details.csv 且 coverage 核对通过才输出 PASS，不能只看解析库打印的成功字样。
捕获完成时默认只有原始数据，没有 trace_view.json 是正常的；解析失败或采集未完成则不正常。

## 5. 查看哪些结果

每个 `*_ascend_pt/serve_summary/` 下：

| 文件 | 含义 |
| --- | --- |
| all_device_ops.csv | 按 Name/Type/Core 汇总每一种已导出设备任务：次数、累计、平均、最小、最大 μs；包括未知项和零时长行 |
| all_trace_ranges.csv | trace 中全部 X 型时长事件，按 PID/TID/category/name 汇总；包含 CPU、Runtime 和设备事件，不能与设备 CSV 再相加 |
| module_cpu.csv | 每个模块/runner 标签的 CPU inclusive 和 self 时间；self 只扣除嵌套的 serve 标签，不是设备执行时间 |
| module_coverage.csv | 全部已发现模块与方法、是否包装、运行调用次数和导出时长事件次数；二者不符会报错 |
| summary.json | rank、原始行数、零时长行数、设备任务累计与时间区间并集、coverage 问题 |

原始 `ASCEND_PROFILER_OUTPUT/trace_view.json` 用 MindStudio Insight 打开，沿框架到设备的
关联看每层 GDN/Attention/MoE 的 kernel、HCCL 及等待。若采集器生成了 communication.json、
communication_matrix.json、api_statistic.csv、data_preprocess.csv，也保留使用，不拼接成重复总时长。
原始 trace 中的同步 API 和空档结合两个 rank 对齐，才能区分主机供给不足、通信传输及慢 rank 等待。

```bash
python3 - "$PROFILE_RUN/capture" <<'PY'
import csv, json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).rglob('serve_summary/summary.json')):
    s = json.loads(p.read_text())
    print('\nTP', s['identity']['tp_rank'], 'device audit:', s['device'])
    with p.with_name('all_device_ops.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows[:30]:
        print(r['name'], r['core'], 'count='+r['count'], 'sum_us='+r['sum_us'])
    print('这里只展示前30项，完整算子表:', p.with_name('all_device_ops.csv'))
PY
```

HCCL duration 可以包括等待，不能全部叫作网络传输。父子 CPU range、多个 NPU stream、
通信父任务可能重叠，不能将 inclusive 时长直接求和，更不能用 CPU 总时间减设备任务累计
时长当作 CPU 开销。不同 TP rank 分开报告；不能把 rank0+rank1 的耗时作为单次 forward 延迟。
工具不凭 kernel 名称强行给出 GDN/Attention/MoE 的设备占比，未建立关联的事件仍完整保留。

## 6. 关闭或移除

停止本次服务后，不设置 HUANYU_SERVE_PROFILE_DIR 再启动，即不加载采集器。
也可删除条件入口，保留之后其他人对 runner 的修复：

```bash
python3 "$TOOLS_DIR/serve_forward_profile.py" uninstall \
  --runner-file /vllm-workspace/vllm-omni/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py
```

参考：
- [Ascend profiler Level1/Level2 采集范围](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0033.html)
- [torch-npu v2.10.0 profile 生命周期及离线 analyse](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/profiler.py)
- [torch-npu 实验配置](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/experimental_config.py)
