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

## 2. 保留原来的训练配置，只增加采集参数

下面是 AQA 入口的追加方式。保持上次分析用的 BS/N/TP/模型/精度/并发等环境变量及
已有 Hydra overrides；不要为采集同时调整它们。只把总训练步数限定为 2。

```bash
PROFILE_DIR="/tmp/huanyu_forward_$(date +%Y%m%d_%H%M%S)"
TOTAL_TRAINING_STEPS=2 HUANYU_GEN_TIMING=0 \
bash "$SHELL_ROOT/run_huanyu_aqa_audio_train.sh" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FORWARD_PROFILE_DIR=\"$PROFILE_DIR\"" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FORWARD_PROFILE_MODEL="2748"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FORWARD_PROFILE_START="101"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FORWARD_PROFILE_ACTIVE="30"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_FORWARD_PROFILE_LEVEL="1"' \
  'trainer.logger=[console]'
```

上述新变量显式传入 Ray runtime_env，不能只在已有 Ray 集群外 export。
匹配 checkpoint 目录 basename `2748` 的所有 rollout TP worker 都会采集，GenRM 不匹配。
`HUANYU_GEN_TIMING=0` 关闭旧的逐迭代控制台计时，减少重复日志干扰，不改变数值精度。
如果上次使用其他入口，把这些 profiler overrides 追加到原命令，沿用该入口限制训练步数的方式。

采集行为：跳过 100 次非空调度调用，第 101 次用于 profiler warmup，再记录 30 次。
这里的次数是 **engine execute_model 调用**，不是 PPO step、请求数或固定的 decode token 数。
attention metadata 会记录可读取的 Python 计数；字段缺失时阶段未知，不能只按 token 数猜 decode。

profiler 在采集窗口内也会记录采样器、空调度以及两次 execute 之间的活动。
分析 forward 时以 `huanyu/forward` 和其算子关联为准，不把整个 ProfilerStep 当成 forward。
窗口由后续调用推进：若请求过短或中途停止而不足 31 次，可能延续到下一轮生成。
必须检查 manifest `status=complete`；正常退出时未完成窗口标记为 `incomplete`，强杀不能保证导出。
若只想分析一个生成阶段，选取确认有连续 31 次执行的位置；不要将跨阶段的窗口混为稳定 decode。

## 3. 确认采集命中，训练结束后离线解析

```bash
grep -aF '[huanyu-forward-profile]' "$LOG"
find "$PROFILE_DIR" -type d -name '*_ascend_pt'

python3 - "$PROFILE_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for p in sorted(root.glob('*/*.json')):
    j = json.loads(p.read_text()); i = j['identity']
    groups = {}
    for label, count in j['hits'].items():
        name = label.split('/')[1]
        groups[name] = groups.get(name, 0) + count
    print(p.parent.name, 'pid=', i['pid'], 'tp=', i.get('tp_rank'),
          'status=', j['status'], 'calls=', len(j['iterations']), 'hits=', groups)
PY
```

同一副本需要完整 TP rank，例如 TP=4 时各有一个 0、1、2、3。
工具按 hostname/parent PID 建候选分组；forkserver 等模式可能共享父进程，须结合 manifest 的
MASTER_PORT、TP 信息和原 console.log 的 Worker_TP/服务进程映射核对。
**不能将几个独立副本的 rank 0 混在同一通信分析目录。**

`GDN`、`Attention`、`MoE` 命中才说明相应模块入口已覆盖。`MLP` 表示未识别为 MoE 的 MLP，
需查看 manifest 记录的实际类；未命中时先核对当前版本的模块名，不报告“耗时为零”。
hits 包含 warmup；GDN forward/core 相互嵌套，命中数或 inclusive 耗时不能相加。

在各采集节点、训练结束后解析其本地文件：

```bash
python3 - "$PROFILE_DIR" <<'PY'
import sys
from pathlib import Path
from torch_npu.profiler.profiler import analyse
for p in sorted(Path(sys.argv[1]).rglob('*_ascend_pt')):
    if p.is_dir():
        print('ANALYSE', p, flush=True)
        analyse(str(p))
PY
```

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
