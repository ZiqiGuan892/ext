# Huanyu rollout forward 的 NPU 细分采集

`../forward_profile.py` 是独立诊断工具。先将该文件同步到各节点当前项目的
`verl_omni_ext/models/qwen3_6_huanyu/`，不需要替换 native vLLM-Omni 的任何 forward。
下面命令在服务器执行；本地 Windows 只做 CPU 包装器测试，没有验证 NPU 运行。

工具在现有 `HuanyuColocateWorkerExtension` 安装完 precision/video hooks 后接入，
在模型构造完毕后包装实际实例的 `_model_forward`、`linear_attn`、`self_attn`、`mlp`
及 GDN `_forward_core`。原函数参数、结果和异常继续传递；不新增 collective 或逐层
`torch.npu.synchronize()`。源码路径、包装链和 SHA256 保存在每个 worker 的 manifest 中。
当前流程是 eager；图执行可能绕过 Python 模块标签，不能将未命中的标签解释成零耗时。

## 1. 安装 opt-in 入口

在每个节点执行一次（共享代码目录只需一次）：

```bash
P="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py"
python3 "$P" install
```

这条命令仅给 sibling `worker_extension.py` 追加条件导入，并打印备份路径。
没有设置 `HUANYU_FORWARD_PROFILE_DIR` 的作业不导入/安装 profiler；重复安装不重复追加。
不需要重启 Ray，但已有 worker 不会获得新钩子，必须启动新的训练作业。

## 2. 先用短窗口复测采集、结束与解析

环境和 Ray 已准备好，只在 head 启动。当前服务器入口是
`$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_train.sh`；本地开发对应
`verl_omni_ext/models/qwen3_6_huanyu/examples/run_huanyu_train.sh`。
下面用 **AQA 的 aqa_data 路径**、2748、Actor16 + GenRM16 和现有精度默认值。
缩小 batch/输出长度只为先验证采集生命周期，不能与之前的吞吐量结果直接比较。

```bash
export PROFILE_DIR="$PROJECT_ROOT/runs/huanyu_forward_$(date +%Y%m%d_%H%M%S)"
P="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py"
printf 'PROFILE_DIR=%s\n' "$PROFILE_DIR"

TRAIN_FILE=/opt/huawei/dataset/infra_data/data/aqa_data/train.parquet \
VAL_FILE=/opt/huawei/dataset/infra_data/data/aqa_data/validation.parquet \
TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 \
MAX_RESPONSE_LENGTH=128 ROLLOUT_IGNORE_EOS=true \
TOTAL_TRAINING_STEPS=2 SAVE_FREQ=-1 TEST_FREQ=-1 \
HUANYU_GEN_TIMING=0 HUANYU_FORWARD_PROFILE_DIR="$PROFILE_DIR" \
HUANYU_FORWARD_PROFILE_START=5 HUANYU_FORWARD_PROFILE_ACTIVE=5 \
HUANYU_RUN_DIR="$PROFILE_DIR/train" \
bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/run_huanyu_train.sh" \
  data.train_max_samples=8 data.val_max_samples=4
```

新入口会把 profiler 变量显式写入 Ray runtime_env；此处不重新准备环境或启动 Ray。
默认 N=4，所以 BS=4 对应 16 条轨迹，适配 Actor16 的每卡 microbatch=1。
`ignore_eos=true` 在这次采集复测中避免很短的响应提前结束，让每个使用中的引擎能推进窗口。
数据或模型在采集前报错、worker 未收到请求、进程被终止等情况仍会导致采集不完整；
下面检查会明确失败，不能把这个命令当成 NPU 成功的事先保证。

这次跳过 4 次非空调度，第 5 次为 warmup，再记录 5 次。首个媒体编码仍可能很慢，
未到阈值时没有 manifest，不应将其判断成 profiler.start 卡住。
这里的次数是 **engine execute_model 调用**，不是 PPO step、请求数或固定的 decode token 数。
attention metadata 会记录可读取的 Python 计数；字段缺失时阶段未知，不能只按 token 数猜 decode。

短跑通过后再恢复原来 BS/N/响应长度等配置，使用 `START=101 ACTIVE=30` 做正式诊断。
匹配 checkpoint basename `2748` 的所有 rollout TP worker 都会采集，GenRM 不匹配。
保留 eager 和 Level1 默认；先不要同时改变精度策略、TP 或并发。

## 3. 实时进度与成功验收

运行中另开终端，把 `PROFILE_DIR` 设为上面打印的目录。此命令只读文件、不导入 torch：

```bash
P="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py"
python3 "$P" status --profile-dir "$PROFILE_DIR"
# 持续查看（Ctrl-C 只结束查看，不影响训练）
watch -n 5 python3 "$P" status --profile-dir "$PROFILE_DIR"
```

每个 worker 的 `pid*_tp*_t*.json` 在关键边界前后原子更新，记录迭代编号、已返回的
forward 数、已返回的 `profiler.step()` 数、当前阶段和时间、最后完成阶段、错误阶段及 trace 路径。
不在每个 Transformer 层写文件。窗口内增加的文件写入有诊断开销，不应使用这次训练测吞吐量。

| phase | 含义 |
| --- | --- |
| `profiler.create` / `profiler.start` | 正在构造或启动采集器 |
| `execute_model` | runner 前后处理，当前不在被包装的 forward 内 |
| `forward` | 实际 `_model_forward`（或模型 forward）尚未返回 |
| `profiler.step` | forward 已返回，正在推进采集状态；最后一次可能包含原生 stop/finalize |
| `profiler.stop` | 正在显式调用采集器 stop |
| `profiler.synchronize` | step/start/stop 内原有的 NPU 同步；`phase_stack` 显示外层阶段 |
| `waiting_for_execute` | 上次 step 已返回，等待下一次调度推进窗口 |
| `capture.verify` | stop 已返回，正在核对采集元数据和原始文件 |
| `finished` | 查看 status：complete 或 incomplete |

`phase_age_s` 是最近记录的当前阶段持续时间，不是自动死锁判定；强杀或节点失联也会留下旧状态。
结合该记录的 host/PID 在对应容器检查进程与 `py-spy dump`，不直接操作同名的其他作业。
`iterations` 在每次 execute 返回后落盘，不再只能看到最初的 `starting/0`。
原生 step 的状态切换和 stop 的同步/元数据导出行为可参见
[torch-npu profiler.py](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/profiler.py) 与
[profiler_interface.py](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/profiler_interface.py)。

训练结束后，在安装了同版本 torch-npu/CANN、能读到采集文件的容器中执行：

```bash
(
set -euo pipefail
P="$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py"
python3 "$P" check --profile-dir "$PROFILE_DIR" --expected-workers 16
python3 -u "$P" analyse --profile-dir "$PROFILE_DIR" --expected-workers 16 \
  2>&1 | tee "$PROFILE_DIR/analyse.log"
)
```

上述默认 Actor16 对应 16 个 TP worker；如改了 Actor 卡数，要同步修改 expected-workers。
使用新目录避免混入旧作业。若各节点路径不是共享存储，分别对本节点数据检查，并以本节点
预期 worker 数替换 16，不能把未看到的远端记录当作已完成。

- `check` 要求预期数量的 worker 均 complete、warmup+active 步数完整、有 forward 命中，
  且存在有效 `profiler_info*.json`、非空 FRAMEWORK 和 CANN 设备原始数据。
- `analyse` 逐 worker 调用 `analyse(..., max_process_number=1, export_type="text")`，
  第一个失败即停止；要求新生成的 `trace_view.json` 包含有时长的 `huanyu/forward`，
  `kernel_details.csv` 包含有时长的设备 kernel。不凭解析函数返回或 “All profiling data parsed” 判成功。
- 成功时输出 `CAPTURE_PASS workers=16` 和 `ANALYSIS_PASS workers=16`，
  汇总写入 `$PROFILE_DIR/forward_profile_analysis.json`。它验证可用于 forward 分析的产物，
  不代表数值正确性、所有通信视图或训练整体均已验证。
- 已有解析结果可用 `python3 "$P" verify --profile-dir "$PROFILE_DIR" --expected-workers 16`
  再验收，不重新运行解析。

`stop()` 抛异常、被内部吞掉错误导致元数据缺失、窗口不足或正常退出清理未完成，均不能标 complete。
强杀无法自动补齐数据；不伪造 profiler_info，也不借用其他 rank 的时钟元数据。

`GDN`、`Attention`、`MoE` 命中才说明相应模块入口已覆盖。`MLP` 表示未识别为 MoE 的 MLP，
需查看 manifest 记录的实际类；未命中时先核对当前版本的模块名，不报告“耗时为零”。
hits 包含 warmup；GDN forward/core 相互嵌套，命中数或 inclusive 耗时不能相加。

profiler 也会记录采样器、空调度和两次 execute 之间的活动；不能把整个 ProfilerStep 当作 forward。
窗口可能跨越生成阶段，须查看迭代 metadata。

同一副本需要完整 TP rank，例如 TP=4 时各有一个 0、1、2、3。
工具按 hostname/parent PID 建候选分组；forkserver 等模式可能共享父进程，须结合 manifest 的
MASTER_PORT、TP 信息和原 console.log 的 Worker_TP/服务进程映射核对。
**不能将几个独立副本的 rank 0 混在同一通信分析目录。**

初次使用 Level1（CPU、NPU、HCCL/AscendCL）；不采 stack/modules/shapes/memory，且
`analyse_flag=False`，避免 TP worker 在训练中同时解析数据。若需要细看 Runtime 同步调用，
另起短跑，将 LEVEL 改为 `2`、ACTIVE 改为 `10`，不要同时改变训练/数值配置。

## 4. 怎么归因

在 MindStudio Insight 或支持 trace JSON 的查看器中打开生成的 `trace_view.json`：

| 现象 | 需要的证据 |
| --- | --- |
| GDN 计算重 | `huanyu/GDN/...` 关联的 NPU kernel 占用长；区分 prefill、decode reference |
| Attention 计算重 | `huanyu/Attention/...` 关联的 attention/QKV kernel 长 |
| MoE 计算重 | `huanyu/MoE/...` 内 expert GEMM、routing/dispatch kernel 长；通信另外看 |
| HCCL 瓶颈 | 同一 TP 组非重叠通信长，再分传输、等待、同步，并对齐各 rank 到达时间 |
| 同步/主机发射瓶颈 | forward CPU 区间长而 NPU 有空洞，或显式同步、D2H、scalar/.item 等待长 |

`huanyu/explicit_npu_synchronize` 只标注原来存在的 Python synchronize；`.item()`/`.cpu()`
造成的同步需要看 PyTorch 算子、AscendCL/Runtime 和设备流。某个 rank 先到 HCCL 并等待别的
rank，不能据其通信耗时长直接归因于网络。NPU 忙于大量小算子和 CPU 逐个发射的情况，也不等于
大矩阵算力已用满。

准备好**只包含同一个 TP 组**数据的目录后（确认安装了 msprof-analyze）：

```bash
msprof-analyze cluster -m all -d "$TP_GROUP_PROFILE_DIR" -o "$TP_GROUP_PROFILE_DIR/analysis"
```

看 `cluster_step_trace_time.csv` 的 Computing、Communication(Not Overlapped)、Overlapped、
Free；进一步检查通信 JSON 的等待/传输/同步和各 rank 时间线。
通信分析针对整个采集窗口；精确拆 forward 仍需关联 `huanyu/forward` 范围。
不要把 GDN/Attention/MoE 的 inclusive 范围再加 HCCL，也不要累加并行 TP rank 后当作墙钟时间。

当前代码中待验证的候选是 GDN mixed cache barrier 前后同步，以及 reference 的
`.all().item()`、长度 `.cpu().tolist()` 和 token 循环。现有约 96% 的 host forward 比例
还不能证明它们是主要原因。

## 5. 关闭

下次启动不传 `HUANYU_FORWARD_PROFILE_DIR` 即关闭。可在各节点删除条件入口，保留其他新增代码：

```bash
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/forward_profile.py" uninstall
```

参考：[CANN profiler API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html)、
[Ascend 集群分析](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/user_guide/cluster_analyse_instruct.md)。
