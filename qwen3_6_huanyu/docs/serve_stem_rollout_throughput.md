# 16 卡 STEM rollout 配置的纯推理吞吐测试

日期：2026-11-11。用户已返回服务器实测报告：16 卡聚合输出 **73.921 tok/s**，
按卡均摊 **4.620 tok/s/卡**。完整原始输出、配置核对范围及统计口径已追加到
[吞吐实验记录第 7 节](serve_throughput_results_20261111.md)。本地 CPU 测试仍不代表 NPU 性能实测。

## 要解决的问题和比较口径

之前 TP4 单副本 eager 测量为聚合 143.520 tok/s、按四卡均摊 35.880 tok/s/卡；四组
图/缓存对照中，重新测得的 eager 基线为 139.927 tok/s。这些都是**输出 token 数除以纯推理
测量窗口**，不是训练 step 吞吐。不同输入的实验不能混用基线计算收益。

训练的 `perf/throughput` 使用 `perf/total_num_tokens / timing_s/step / 统计采用的卡数`，
通常包含有效输入和输出 token，分母包含 rollout、log_prob、训练更新、权重同步等整个 step。
另外，有独立 RM 资源池时还需查清卡数分母。仅凭“2–3 tok/s/卡”不能定位瓶颈，也不能断定
其与当前的纯推理结果矛盾。若训练日志齐全，可另算：

```text
生成阶段输出 tok/s/卡 = 该 step 实际生成的输出 token 总数 / timing_s/gen / rollout 占用卡数
```

`timing_s/gen` 仍可能包含训练框架的准备、请求调度、唤醒等开销。输出总数应从本次有效轨迹
或明确的计数得到，不把配置 BS×N 强行当成失败/过滤后的样本数。

## 核对过的配置

来源为本地 `examples/run_huanyu_stem_job.sh` → `run_huanyu_stem.sh` →
`run_qwen36_huanyu_gspo_npu.sh`（规则奖励分支）。以 job 入口传入值为准，直接运行下层脚本的
独立默认值不同；最后的 Hydra override 也可能修改结果。

| 项目 | 之前 TP4 serve | 本次 STEM 配置对照 |
| --- | --- | --- |
| 模型 | 2748 | 同一 2748 |
| 总推理卡数 | 4 | 16 |
| 推理拓扑 | 单副本 TP4 | 两个独立副本，每个 TP8 |
| 推理并行解释 | engine DP1 | 每个 engine DP1/PP1，副本数2（相当于外部 DP2） |
| Actor 训练 DP | 不适用 | STEM 训练 DP16；本测试没有 Actor 训练 |
| 每副本 max_num_seqs | 16 | 8，全局最多16条调度序列 |
| HTTP 活跃并发 | 16 | 每副本8，全局16 |
| 显存预算 | 0.9 | 0.6 |
| max_model_len | 32768 | 12288 |
| max_num_batched_tokens | 16384 | 12288，可覆盖为16384 |
| chunked prefill | 以当时服务配置为准 | false，与 STEM 配置一致 |
| graph / prefix cache / profiler | 均关闭 | 均关闭 |
| precision hooks | 原生 serve，无训练插件入口 | 默认 `stem_full`，载入 STEM worker extension 和参考实现 |
| HCCL_OP_EXPANSION_MODE | 以当时进程环境为准 | AIV，与训练脚本默认一致 |
| sampling | temperature1、top_p1 | temperature1、top_p0.95、top_k20、repetition_penalty1.05 |

`full` 包括 `MINICPMO_GDN_FP32_REFERENCE=1`、`MINICPMO_GDN_PREFILL_REFERENCE=1`、
GDN cache barrier、L2Norm/conv/gating reference、private hybrid cache、stable linear 等。
启动时使用真实的 `HuanyuColocateWorkerExtension`，不只是设置同名环境变量。
客户端要求每个副本日志中至少有8次成功的 `rollout installed: gdn_fp32=True`，否则拒绝开始测量。

**输入负载是控制变量，不宣称复现旧训练数据：**默认是合成纯文本，chat template 后精确2048
tokens，输出强制3072，thinking关闭；这两项长度沿用刚才的纯文本基准。每批4条不同prompt、
每条生成4次，共16条轨迹，使用不同seed。这取自用户贴过的 BS4/N4 小规模 STEM 配置；
当前 job 文件默认 BS128/N16 不是这次测试的默认负载。各阶段使用不同prompt，prefix cache关闭。
训练脚本默认 thinking 开启，最新默认允许 EOS；旧慢作业曾显式禁止 EOS 并输出10240。
下面给出相应覆盖方法。

本次保留长上下文容量12288，但默认输出3072。要比较完整训练配置，请同时对齐真实prompt分布、
thinking、EOS、输出长度、采样参数和实际训练checkpoint。这里使用磁盘checkpoint，没有训练更新、
sleep/wakeup、IPC权重传输、Ray AgentLoop、奖励打分或FSDP显存竞争。

## 文件及部署

将配套文件放在同一个服务器目录（例如 `$PROJECT_ROOT/serve`）：

- `bench_stem_rollout.py`
- `bench_throughput.py`
- `prepare_benchmark_configs.py`
- `run_stem_rollout_throughput.sh`

需要现有服务环境里的 transformers、requests、PyYAML 和已能训练的 `verl_omni_ext` 插件。
脚本不执行依赖安装、不启动Ray、不修改训练脚本、模型配置或已安装vLLM源码。需要先停止占用这16卡的
旧测试服务/训练任务，并核对 `npu-smi info`。这里不提供全局进程清理命令。

生成文件保存在新的 `/tmp` 目录，`case.json` 锁定参数。prepare 后改变环境变量不会改变已准备的
实验；修改参数要重新 prepare 到新目录。`serve_0.log` / `serve_1.log` 记录服务输出；
`launch_*.json` 记录实际导入源码路径和环境设置。

## 完整命令

先在准备终端执行，复制最后打印的 `BENCH_RUN` 到另两个终端：

```bash
export PROJECT_ROOT=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new
export TOOLS_DIR="$PROJECT_ROOT/serve"
export BENCH_RUN="$(mktemp -d /tmp/huanyu_stem_rollout_tp8x2.XXXXXX)"

MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748 \
ROLLOUT_TP=8 ROLLOUT_MAX_NUM_SEQS=8 ROLLOUT_GPU_MEM_UTIL=0.6 \
ROLLOUT_MAX_MODEL_LEN=12288 ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 BENCH_CONCURRENCY_PER_REPLICA=8 \
BENCH_INPUT_TOKENS=2048 BENCH_OUTPUT_TOKENS=3072 \
BENCH_IGNORE_EOS=true BENCH_ENABLE_THINKING=false BENCH_PRECISION=stem_full \
BENCH_WARMUP_BATCHES=1 BENCH_NUM_BATCHES=3 \
bash "$TOOLS_DIR/run_stem_rollout_throughput.sh" prepare
```

终端A：保留上述环境，启动副本0，占物理卡0–7，端口28889。它保持前台运行。

```bash
bash "$TOOLS_DIR/run_stem_rollout_throughput.sh" serve 0
```

终端B：设置同一个目录，启动副本1，占物理卡8–15，端口28890：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export BENCH_RUN=/tmp/huanyu_stem_rollout_tp8x2.替换成刚才的目录
bash "$TOOLS_DIR/run_stem_rollout_throughput.sh" serve 1
```

两个服务都使用 YAML 的逻辑卡0–7，分别通过 `ASCEND_RT_VISIBLE_DEVICES` 映射到各自的物理卡组。
这一映射遵循 [vLLM-Omni stage devices 规则](https://docs.vllm.ai/projects/vllm-omni/en/v0.28.0/api/vllm_omni/entrypoints/stage_utils/)。
不能在可见卡已经限制为8–15时再把 YAML 逻辑卡写成8–15。

终端C：设置相同 `TOOLS_DIR` 和 `BENCH_RUN`。待两边服务就绪后检查，再同时向两边压测：

```bash
export TOOLS_DIR=/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/serve
export BENCH_RUN=/tmp/huanyu_stem_rollout_tp8x2.替换成刚才的目录

for port in 28889 28890; do
  curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$port/health" || break
done
grep -aE 'tensor_parallel_size|max_num_seqs|max_num_batched_tokens|enforce_eager|huanyu-precision.*rollout installed' \
  "$BENCH_RUN"/serve_*.log | tail -n 24
npu-smi info

bash "$TOOLS_DIR/run_stem_rollout_throughput.sh" bench
bash "$TOOLS_DIR/run_stem_rollout_throughput.sh" report
```

服务没有就绪时先查看服务日志。客户端本身会做每副本一次短预检；失败会保存错误报告而不计算有效吞吐。
请核对实际 engine 为 TP8 / eager / max_num_seqs8，并且16张不同卡各有对应worker。
`launch_*.json` 是启动配置记录，不替代实际设备占用检查。

## 吞吐计算和结果读取

每个副本有独立的8线程请求池，两个池在同一个进程内同时调度。同一批从提交第一个请求开始计时，
直到两个副本所有请求完成，得到一个共同 `wall_time_s`；正式三批的时间相加。线程池排队、引擎排队、
prefill、decode、HTTP传输均在此窗口内。预检、预热、tokenizer和构造prompt不计入。

```text
总输出 tokens = 3 × 4 prompts × 4 trajectories × 3072 = 147456
16卡聚合输出吞吐 = 总输出 tokens / 三批共同 wall_time 之和
平均单卡输出吞吐 = 聚合输出吞吐 / 16
每副本贡献 = 该副本总输出 tokens / 同一个三批共同 wall_time 之和
```

每副本各24576 tokens/批，两边合计49152。两个副本贡献之和等于聚合值，不累加两个副本的运行时间。
这里的“单卡”是 TP8 各卡分摊的数值，不是把35B模型放到单卡独立推理的性能。
失败请求、输出提前结束（禁止EOS的场景）、输入长度不一致会使该报告无效。
最终数据在 `throughput.json`；`report` 命令重新计算而不是只打印保存的平均值。

`ROLLOUT_N=4` 用4个独立 `n=1` HTTP请求实现，避免此前后端忽略 `n>1`。这个展开不完全复现
训练的请求封装，但保持轨迹数、每副本并发、每条轨迹的独立seed。128个AgentLoopWorker并不意味着
同时有128条GPU序列；本测试省略这些生产者进程。

## 必要的后续对照

1. 首先运行默认 `stem_full`。若纯推理也低，继续对比精度参考实现和推理配置；若纯推理明显高，
   回查旧训练的 `timing_s/gen`、`update_actor`、`update_weights` 及吞吐卡数分母。
2. 要分离参考实现影响，停止两边服务，用新 `BENCH_RUN`、**其余 prepare 参数完全相同**，只把
   `BENCH_PRECISION=stem_full` 改成 `BENCH_PRECISION=native` 重跑。native不加载训练插件和参考开关。
   此时两组会得到相同 `corpus_sha256`。这也是独立的速度对照，不是推荐未经正确性验证就关闭训练参考计算。
3. 要逼近曾贴出的旧长输出作业，将 `BENCH_OUTPUT_TOKENS=10240 BENCH_ENABLE_THINKING=true
   BENCH_IGNORE_EOS=true` 传给 prepare（其余 BS4/N4、context12288 保留）。prompt仍是合成2048，
   不是原数据集的实际长度分布。不能用3072长度那组作为唯一变量的对照。
4. 要用后来的 BS16/N8：设置 `TRAIN_BATCH_SIZE=16 ROLLOUT_N=8`；每批128条轨迹，
   每副本仍最多8个活跃HTTP请求，分批排队完成。`max_num_seqs`、client concurrency、总轨迹数
   是三个不同量。若要复现 context4096，需同时将输出上限改成2048或更小，并设置
   `ROLLOUT_MAX_MODEL_LEN=4096 ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384`。

测试证据：本地 CPU 测试覆盖配置覆盖层、双副本同时运行、N展开、公共分母计算、失败处理、
precision/profiler环境清理和不可覆盖报告。没有本地NPU/真实模型吞吐实测。
