# 2748：纯文本 eager TP=4，进一步拆分 MoE 主机耗时

保持 Level2、采集 10 次 execute、纯文本输入、eager、TP=4。采集文件写入新建的
本地 `/tmp/huanyu_text_moe_tp4.*`，不再写 OBS 挂载目录。本工具没有在用户的 NPU
镜像中运行；CPU 测试验证包装、恢复、异常传播和报告归因，实际路径由采集结果核实。

## 此次要验证什么

已有稳态窗口 execute 6–9 中，TP0/TP2 的 forward 主机耗时分别为 172.683/169.100 ms。
`vllm::moe_forward_shared` 的 recorded self 分别为 33.313/32.237 ms/iter。
这些是带 Level2 插桩的主机范围，不能当作 NPU GEMM 计算时间。

用户提供的代码表明，该自定义算子调用实际实例的 `AscendMoERunner._forward_impl`。
其中 `routed_experts.forward_impl()` 是直接方法调用，旧的模块 `forward` 标记无法
单独划出这个边界。shared experts 也可能是普通辅助对象，不在 `named_modules()` 中。
基类 `MoERunner._forward_impl` 的实现不能代替这个 override 解释当前执行路径。

新增 `HUANYU_SERVE_PROFILE_MOE_DETAIL=1` 后：

- 包装当前实例的 runner `_forward_impl`、routed `forward_impl`、shared `forward`
  以及存在的 shared 辅助方法，形成 `serve/moe/...` 范围。
- 单独包装 runner 中输出归约、输入/输出变换等方法，区分自定义算子内部和返回后的
  all-reduce。不能把外层归约的设备等待直接计入那 33 ms 自定义算子的 CPU self。
- 第一层记录实际 helper 调用的参数和结果，包括 overlap 判定、归约布尔参数、
  当时的 `moe_comm_type`。只记录标量或类型，不读取 Tensor 内容，不额外执行判定函数。
  `stored flags` 只表示实例已存储字段，缺失不代表对应能力关闭。
- 所有层记录命中次数、实际包装的源码文件及 SHA256；缺失的可选方法列入 inventory。
  只捕获执行到的分支，未命中的方法不按零耗时解释。
- 保留原有全模块和 CPU+NPU 算子采集，不按 Top-K 过滤原始事件，不增加逐层同步。
  不替换 forward 实现，因此当前视频修复和已安装的参数/返回值契约仍被使用。

该开关默认关闭。此次细分会增加观测开销，并改变原来 self 时间的归属；新旧 self
差值不是加速收益。现在只能提出需要核实的主机下发/跨 rank 到达偏差，尚不能宣称
已证明 CPU、GEMM、网络带宽或某个 stream 等待是唯一瓶颈。

## 文件与部署

独立包中的以下文件放在同一个目录，服务运行期间保持该目录不动：

```text
run_serve_text_moe_profile.sh
serve_forward_profile.py
serve_profile_report.py
diagnose_forward_ops.py
moe_detail_report.py
bench_throughput.py
README.md
```

可以使用一个新目录，例如：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve/moe_detail
```

将解压后的文件放入该目录。脚本自动以自身目录寻找工具，不需要把训练插件加入
PYTHONPATH。`serve` 会备份已安装的 NPU runner，并在文件末尾安装/更新带环境开关的
采集入口。它不会替换模型或 runner 的函数体。无 `HUANYU_SERVE_PROFILE_DIR` 的普通
serve 不加载采集器。已经启动的 worker 不会自动得到新插桩，须启动本次新服务。

## 终端 A：启动服务

在先前测试服务已停止、卡 0–3 和端口 28889 可用时运行；脚本不会清理其他进程。

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve/moe_detail
export MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748
export VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
unset PROFILE_RUN
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" serve
```

脚本打印供终端 B/C 复制的 `export` 行。其配置为：

| 项目 | 值 |
| --- | --- |
| TP / 可见卡 | 4 / 0,1,2,3 |
| eager / prefix cache | true / false |
| profiler | CPU+NPU，Level2，START=1，ACTIVE=10 |
| shapes / stack | false / false |
| 请求 | 纯文本，2048 输入 token，3072 输出 token，ignore_eos |
| HTTP 并发 / n | 16 / 1 |
| 预热 / 采集请求批次 | 1 / 1 |
| temperature / thinking | 1.0 / false |
| 客户端时序来源 | none，不伪造 TPOT |

使用 `n=1` 是因为此前该后端未通过 `n=8` 的 choices/usage 预检。与后续成功的 TP4
纯文本采集保持一致。源 YAML 的显存比例、max_num_seqs、max_num_batched_tokens 等
容量参数保留，只生成 TP4 eager 副本。修改源 YAML 会影响容量，核对服务最终启动配置。
如果实际安装目录不同，通过 `VLLM_OMNI_ROOT` 指定；`DEPLOY_SOURCE` 可指定源 YAML。

## 终端 B：发送压测

先粘贴终端 A 打印的全部 export 行，待服务就绪后运行：

```bash
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" request
```

客户端先完成协议预检与完整预热，随后自动创建 armed 文件，不能手动提前 touch。
10 次是有 token 的 execute 调用，不是 10 条请求。第 11 次调用入口执行收尾，覆盖
第 10 次 forward 后的 sample_tokens。此请求批次的总耗时含 profiler 收尾，不能作为
正常推理吞吐。原始文件即使已存在，也必须等所有 worker 完整结束再解析。

## 查看状态、正常吞吐与离线报告

终端 C 同样粘贴 export 行，可随时查看：

```bash
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" status
```

请求完成后检查四个 worker 都 `complete`、`completed=10/10`，并且直接 MoE 方法命中：

```bash
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" check
```

如需正常吞吐参考，在服务仍运行、没有离线解析竞争时执行：

```bash
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" baseline
```

此命令先确认采集结束，再用相同负载预热 1 批、测量 3 批。所有模块/方法包装已恢复，
原生 profiler 已停止，只有 execute 入口的完成状态检查仍在。输出
`$PROFILE_RUN/post_capture_requests.json`，不是完全移除入口的干净服务基准。
不要用采集批次与此批次的 wall time 差值估算逐 token 的插桩开销，因为采集批次包含
一次性的 start/stop。若要求严格的无入口基线，需要另启未设置采集环境变量的服务。

之后可在终端 A 正常 Ctrl+C 结束本测试服务，再离线运行：

```bash
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" analyse
bash "$TOOLS_DIR/run_serve_text_moe_profile.sh" report
```

analyse 按 rank 串行解析，保留完整设备算子表并核对已执行标记。report 默认比较
execute 6,7,8,9；四个 rank 的 token/request 数必须一致，含 prefill 的窗口会报错。
若 metadata 无法确认 decode，则标记 `phase=unknown`，不能据此宣称纯 decode。
确认另一组相同工作量的 decode 调用后，可设 `PROFILE_STEPS=7,8,9,10` 再运行 report。

## 需要返回的结果及读法

首先返回 `$PROFILE_RUN/moe_detail_report.txt` 的完整控制台文本即可，通常不需要上传
完整 trace。报告依次输出各 rank 的 forward CPU 时间、各 MoE 方法 inclusive/exclusive、
自定义算子的剩余 recorded self、设备关联、第一层的实际分支参数和结果。

- `runner/_forward_impl` 包括内部 routed/shared；不能与它们的 inclusive 相加。
- `routed/forward_impl` 划出路由专家路径；shared 分支单独统计。
- runner 输出归约方法独立于 `_forward_impl`，用于分清调用入口慢和归约等待。
- `exclusive` 只扣除嵌套的细分 MoE 方法范围，仍包含 CPU 框架下发和等待，不是
  Python 纯计算或 NPU 纯计算。原有 CPU 算子的更细 self 另存 CSV。
- 设备事件按 CPU flow 关联到最近的细分方法，嵌套普通模块不会抹掉归属。
  `EVENT_WAIT` 单列同步事件；HCCL 范围与 AICPU kernel 可能重叠，不可相加为通信耗时。
  设备事件 duration 的和也不是设备墙钟时间或整卡空闲率。
- `moe_forward_shared` 剩余 self 下降，只表示原先未细分的时间现在进入子范围；
  若仍较大，查看完整 CPU 表中的未覆盖范围和来源，而非直接归因于 GEMM。

完整结果位于 `$PROFILE_RUN/moe_detail_diag`：

```text
summary.json
tp*_moe_stages.csv      各阶段累计调用数，以及每迭代 inclusive/exclusive
tp*_moe_layers.csv      各层明细
tp*_moe_device_ops.csv  按细分阶段和 CPU flow 关联的设备事件
tp*_branches.json      第一层的实际参数、结果、通信类型
tp*_cpu.csv            全部 CPU 归因
tp*_devices.csv        全部已关联设备事件聚合
tp*_flow_audit.csv     所有 flow 端点的匹配/未匹配状态
```

CSV 的 calls 是所选窗口总次数，时间已除以所选迭代数；默认窗口为 4 次，不能再除以 4。
未匹配或有歧义的 flow 不猜测归属，完整原始事件仍在各 rank 的 trace 中。

移除条件入口（服务停止后）可执行：

```bash
python3 "$TOOLS_DIR/serve_forward_profile.py" uninstall \
  --runner-file /vllm-workspace/vllm-omni/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py
```
