# Huanyu 训练 Rollout 与 Forward 性能分析记录

更新日期：2026-11-10。覆盖此前提供的 AQA 训练指标、gen 主机计时、逐迭代引擎拆分，以及 2748 模型 NPU profiler 的采集故障和部分设备数据。所有数值来自已有日志或其摘要，没有重新运行训练。

目前形成的定位顺序是：**训练 step 主要耗时在 gen；后续有效引擎迭代主要耗时在 forward 主机调用；最新设备采集的已导出任务中，HCCL 通信任务累计时长最突出。** 最后一步仍缺少完整框架时间线和同一 TP 组的跨 rank 对照，尚不能认定网络带宽、Attention、GDN 或 MoE 中某一项是最终根因。

## 1. 实验来源和统计口径

| 记录 | 来源与范围 | 可以回答的问题 | 不能据此回答的问题 |
| --- | --- | --- | --- |
| AQA BS=4 基线 | 历史训练记录，step 1–6；Actor16 + GenRM16，N=4，固定生成 10240 token | gen 占训练 step 的比例、端到端吞吐 | gen 内部各阶段、具体设备算子耗时 |
| AQA BS=16 | 用户确认属于 AQA BS=16 作业的完整 step 7 指标，TaskRunnerV1 PID 755187 | gen、old_log_prob、Actor 更新、权重更新的 step 占比 | 所有 step 的平均占比；不能改称 STEM 实验 |
| 11 月 7 日 gen 计时 | `huanyu_20261107_154813_aXLqrE/console.log`；首次及每 100 次引擎迭代采样 | 请求等待区间、首次媒体编码长尾 | 完整 gen 墙钟分解、未观测迭代的媒体编码开销 |
| 11 月 9 日引擎拆分 | `huanyu_20261109_103419_kx4OJY/console.log`；2748，16 个进程，`sample_every=1` | 各进程有效对齐的 execute_model 调用内部占比 | 未对齐迭代、整轮 gen 墙钟、设备 kernel 占比 |
| 11 月 10 日 NPU profiler | `runs/huanyu_forward_1110`；16 个 worker 卡在最终 profiler.step；仅导出 PID 1635029 的 CANN 数据 | 已落盘设备任务的数量和累计时长构成 | 完整采集成功、完整 forward 模块占比、其他 rank 的性能 |

服务器项目根目录为 `/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new`。上表相对作业目录位于该目录的 `runs/` 下。BS=16 step 7 未保留可确认的作业绝对路径，不用其他实验目录补填。

证据归档：

- [AQA BS=4 历史训练记录](profile_evidence/aqa_bs4_20261028.md)：保留当时配置与舍入后的六步指标，里面的命令不是当前启动建议。
- [11 月 9 日完整引擎拆分输出](profile_evidence/engine_breakdown_20261109.txt)：保留 16 个进程的 first/later 报告、有效样本数、跳过原因和日志质量信息。
- 本文其余表格转录会话中已提供的指标；NPU 原始数据仍在服务器，不将本地文档整理视为重新验证了设备采集。

### 1.1 四种分母不能混用

| 统计 | 分子 / 分母 | 是否可以相加成 100% |
| --- | --- | --- |
| 训练 step 占比 | 同一 step 的 `timing_s/阶段 ÷ timing_s/step` | 此处列出的顺序阶段加剩余项可以闭合 |
| Agent 累计调用时间比 | 某 phase 的累计时间 / trajectory 累计时间 | 不可以；嵌套和并发调用会重叠 |
| 引擎内部占比 | 同一进程、有效对齐迭代的阶段独占时间 / execute_model 总时间 | 可以近似闭合，舍入可能产生微小误差；采样器单独列出 |
| CANN 算子构成 | 某类任务累计 duration / 已导出所有任务累计 duration | 数学上可闭合，但不是 forward 墙钟占比；不同流可能重叠 |

**不得将 81.84% × 96% × 98.86% 当作整步的通信占比。** 它们来自不同作业、不同采样范围、不同分母。

## 2. 整个训练 step 的阶段耗时

这里的“整个训练”是已记录训练 step 的计时范围，不包括环境准备、Ray 启动、模型加载等作业启动开销。

### 2.1 AQA BS=16 的 step 7

分母：`timing_s/step = 2773.7188045680523 s`，约 **46.23 分钟**。

| 阶段 | 日志指标 | 耗时 s | 占该 step |
| --- | --- | ---: | ---: |
| 生成、完整 batch 等待及采样收尾 | `timing_s/gen` | 2270.008695 | **81.8399%** |
| Actor 更新 | `timing_s/update_actor` | 432.852568 | **15.6055%** |
| 更新 rollout 权重 | `timing_s/update_weights` | 41.891637 | 1.5103% |
| 旧策略 logprob | `timing_s/old_log_prob` | 28.637098 | 1.0324% |
| 优势计算 | `timing_s/adv` | 0.204870 | 0.0074% |
| 上述字段之外的剩余计时 | step 减上述阶段 | 0.123937 | 0.0045% |
| 合计 | `timing_s/step` | **2773.718805** | **100%** |

`gen` 在该同步训练链路中包含生成结果、评分及数据准备完成前的等待，以及采样结束处理。它不是纯 LLM forward 时间。未单独出现的 reward/ref 等字段不按零耗时处理；上表也没有将剩余项任意命名为通信、反向或 CPU 开销。

| 同一步的关联指标 | 数值 |
| --- | ---: |
| `perf/total_num_tokens` | 661464 |
| `perf/throughput` | 7.452359614 token/s/卡 |
| response 长度 min / mean / max | 10240 / 10240 / 10240 |
| response clip ratio | 1.0 |
| prompt 均长 | 95.375 |
| rollout / Actor 概率 Pearson | 0.9993267655 |
| `rollout_corr/kl` | 0.0005509746 |
| Actor 最大 allocated / reserved 显存 | 49.146 / 57.002 GB |

吞吐复算：`661464 ÷ 2773.7188045680523 ÷ 32 = 7.452359614`，与按 32 卡归一化吻合。这不是单个 vLLM 副本的生成 token/s，也不是只除以 Actor 卡数的指标。

分析：gen 是首要端到端优化对象，Actor 更新是第二大项。Pearson 接近 1、KL 较小支持该 batch 两端概率较接近，不能证明精度配置没有性能代价，也不能用相关性指标证明数值链路完全正确。

### 2.2 AQA BS=4 基线和 BS=16 的比较

BS=4 基线配置为 N=4、每步 16 条轨迹、Actor16 + GenRM16、TP=4，2K prompt + 固定 10K response，full precision、图模式关闭。三条 logprob/训练路径启用 dynamic batching；更完整配置见归档。

| BS=4 step | gen s | step s | gen / step | 日志吞吐 token/s/卡 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2275.8 | 2551.9 | 89.18% | 2.02 |
| 2 | 2149.1 | 2298.8 | 93.49% | 2.25 |
| 3 | 2152.1 | 2283.9 | 94.23% | 2.26 |
| 4 | 2143.9 | 2272.1 | 94.36% | 2.27 |
| 5 | 2282.0 | 2425.0 | 94.10% | 2.13 |
| 6 | 2273.9 | 2427.9 | 93.66% | 2.12 |
| 六步合计 | **13276.8** | **14259.6** | **93.11%** | 六步日志算术平均 **2.175** |

非 gen 占六步总耗时 **6.89%**，平均每步 163.8 s；缺少该基线的完整阶段字段，无法进一步拆成 Actor 更新、logprob 和权重更新。

BS=16 step 7 的吞吐比上述六步日志吞吐均值约高 **3.43 倍**，但其 step 时间仍约 46 分钟，gen 仍需约 37.83 分钟。“吞吐增加”不等于“单步更快”。更大的有效 batch 可以提高并发和固定开销摊销效率；这两组不是只改变 BS 的受控对照，不能将全部增益归因于 BS 或 dynamic batching。

## 3. Rollout 请求级计时和首次长尾

### 3.1 11 月 7 日累计调用统计

来源：`runs/huanyu_20261107_154813_aXLqrE/console.log`。以下均为 `host_wall`；`work_s` 是各调用时长相加，可以跨进程、请求和 TP rank 重叠。

| phase | status | calls | mean s | max s | work s |
| --- | --- | ---: | ---: | ---: | ---: |
| agent.acquire_server | ok | 160 | 0.010682 | 0.035497 | 1.709089 |
| agent.generate_request | ok | 160 | 160.322423 | 474.224030 | 25651.587746 |
| agent.postprocess | ok | 160 | 1.362648 | 9.261944 | 218.023740 |
| agent.queue_put | ok | 160 | 0.034073 | 0.049438 | 5.451613 |
| agent.trajectory | ok | 160 | 163.112002 | 477.215468 | 26097.920262 |
| server.generate_request | ok | 160 | 160.300353 | 474.185074 | 25648.056468 |
| engine.compute_logits | error | 1114 | 0.000009 | 0.000043 | 0.009840 |
| engine.compute_logits | ok | 1092 | 0.000274 | 0.001166 | 0.299376 |
| engine.execute_model | ok | 1082 | 1.693809 | 102.135706 | 1832.700889 |
| engine.mm_encoder | ok | 1061 | 1.356709 | 94.498903 | 1439.468116 |
| engine.preprocess | ok | 1089 | 1.293996 | 94.570085 | 1409.162050 |
| engine.sample_tokens | ok | 1025 | 0.004858 | 0.098908 | 4.979405 |
| trainer.sample_begin | ok | 5 | 0.000002 | 0.000004 | 0.000010 |
| trainer.sample_end | ok | 5 | 7.141187 | 8.642221 | 35.705934 |
| trainer.submit | ok | 5 | 7.382609 | 36.503156 | 36.913043 |
| trainer.update_cycle | ok | 4 | 532.724594 | 771.940122 | 2130.898377 |

在这份聚合表中，`agent.generate_request / agent.trajectory` 的累计时间比为 **98.29%**，`agent.postprocess / agent.trajectory` 为 **0.84%**。这是请求累计调用时间比，不是整轮 gen 墙钟占比，也没有通过逐 uid 父子区间验证形成完整的 100% 分解。

`generate_request` 包括引擎排队、执行、RPC 和返回；客户端与服务端耗时不能相加。`acquire_server` 嵌套于客户端生成调用，`queue_put` 属于后处理。`trainer.submit` 在原生 gen 计时前，`update_cycle` 包含训练，不能作为 gen 子阶段参与求和。

### 3.2 首次调用和后续调用必须分开

同一日志进一步按 `engine_iteration == 1` 分组，模型路径均为 `/opt/huawei/dataset/model_gy/downloaded_models/2748`：

| 阶段 | 分组 | calls | mean s | P50 s | P95 s | max s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| execute_model | first | 16 | 101.9558 | 102.0788 | 102.1357 | 102.1357 |
| execute_model | later | 1066 | 0.1889 | 0.1845 | 0.2352 | 0.2503 |
| mm_encoder | first | 16 | 89.9665 | 91.9660 | 94.4989 | 94.4989 |
| mm_encoder | later | 1045 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| preprocess | first | 15 | 93.8759 | 94.1378 | 94.5701 | 94.5701 |
| preprocess | later | 1074 | 0.0010 | 0.0009 | 0.0013 | 0.0027 |

结论：约 90–94 秒的媒体编码长尾集中于首次被计时的调用。直接用全部样本的 mm_encoder 均值 1.36 s 判断稳态媒体编码最慢，会被首次长尾误导。

限制：默认每 100 次迭代采样可能漏掉后续稀疏媒体编码；表中的 0.0000 是输出精度内接近零，不证明后续所有输入都不需要编码。first 不自动等于纯预热，其耗时还可能包含真实媒体计算、首次初始化或编译，当前没有再拆分这些来源。preprocess 包含 mm_encoder，不能相加；三个阶段的调用数量不同，也不能直接用均值相减推出 forward。

1114 个 `engine.compute_logits error` 均为 `TypeError`。已核对原生代码会先带 `sampling_metadata` 调用，捕获 TypeError 后不带该参数重试；不能将这批兼容性重试等同于 1114 次生成失败。

## 4. Rollout 引擎内部各阶段占比

### 4.1 11 月 9 日逐迭代对齐报告

来源：`runs/huanyu_20261109_103419_kx4OJY/console.log`，`gen_timing.py --engine-breakdown`，checkpoint 2748，runner 为 NPUARModelRunner。

报告使用单调时钟，按进程和 first/later 分组，用父子 span 对齐并扣除已观测的嵌套区间。这是**主机调用独占区间**：forward 里面的设备等待、通信和 Python 执行仍算在 forward，不能将其解读为纯 NPU 计算。

16 个进程的 later 共观察到 141751 次 execute，100592 次通过对齐，保留率 **70.96%**。单进程保留率为 56.72%–82.05%；其余因缺失/多余子 span 或未关联嵌套 span 被排除。总计数是“进程×迭代”样本数，不是生成 token 数。日志另报告 `malformed_records=3`、`orphan_spans=81181`；孤立 span 数不等于丢失迭代数。

因此下面结论适用于有效对齐子集，缺失可能存在选择偏差。16 个进程得到相近的 forward 占比增强了这一定位的可信度，但没有消除日志缺失问题。

### 4.2 后续迭代的完整拆分

代表进程 PID 725373：有效迭代 6749 次，分母 `observed_execute_s=1820.357716 s`。`mean/iteration` 按有效 execute 次数计算；某阶段未调用的迭代计零。各阶段有效 calls 不一定相同。

| 阶段 | 独占累计 s | 平均 ms/有效迭代 | 占该进程 execute | 16 进程各自占比范围 |
| --- | ---: | ---: | ---: | ---: |
| forward | 1748.141578 | 259.022 | **96.03%** | **95.93%–96.30%** |
| preprocess.other | 26.395148 | 3.911 | 1.45% | 1.35%–1.51% |
| 未归因部分 | 19.628896 | 2.908 | 1.08% | 0.94%–1.12% |
| prepare_inputs | 10.084856 | 1.494 | 0.55% | 0.46%–0.56% |
| mm_encoder | 8.082569 | 1.198 | 0.44% | 0.40%–0.58% |
| update_states | 5.031330 | 0.745 | 0.28% | 0.26%–0.33% |
| compute_logits | 2.129125 | 0.315 | 0.12% | 0.11%–0.12% |
| gather_mm_embeddings | 0.864215 | 0.128 | 0.05% | 0.04%–0.05% |

`preprocess.other` 已扣除 mm_encoder 和 embedding 汇集等被观测子区间；未归因部分不命名为 forward、排队或通信。

采样器位于 execute_model 外，不能写成 0%：PID 725373 的 `sample_tokens_outside_execute` 有 8604 次，均值 **11.713 ms**、P95 **17.693 ms**、最大 **26.696 ms**。其样本集与 6749 次有效 execute 不同，不将均值直接拼成严格完整的引擎迭代时间。各进程采样器和完整分位数见原始报告。

结论：后续有效迭代中，应优先继续拆 forward；预处理和媒体编码在这个统计口径下不是最大的常驻耗时项。优化只有约 1% 的预处理区间，难以单独解释或解决主要吞吐瓶颈。

### 4.3 首次迭代的瓶颈不同

16 个进程中，13 个 first 样本通过对齐，3 个被排除。有效 first 的 mm_encoder 占各自 execute 的 **82.35%–96.38%**，forward 仅占 **0.40%–1.95%**。

以同一代表进程 PID 725373 为例：

| first 阶段 | 耗时 s | 占首次 execute |
| --- | ---: | ---: |
| mm_encoder | 111.940702 | **96.34%** |
| prepare_inputs | 2.577532 | 2.22% |
| preprocess.other | 1.126345 | 0.97% |
| forward | 0.467784 | 0.40% |
| update_states | 0.044966 | 0.04% |
| 未归因 | 0.029185 | 0.03% |
| compute_logits | 0.001048 | <0.01% |
| gather_mm_embeddings | 0.000444 | <0.01% |
| 首次 execute 合计 | **116.188007** | **100%** |

这里“首次媒体编码约占 96%”和“后续 forward 约占 96%”是两个不同结论，不能相互替代。11 月 7 日的 90 秒与本次约 112 秒属于不同作业，不由此断言媒体编码退化。

### 4.4 各进程的后续 forward 结果

共同 host：`ma-job-a27663aa-e237-4c73-8eed-1175976dc05a-vj-0-worker-0`。不将这些并发进程的累计时间当作作业墙钟时间，也不从相邻 PID 自动推定 TP 组。

| PID | execute_seen | aligned | 保留率 | execute 累计 s | forward 平均 ms/有效迭代 | forward 最大 s | forward 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 725373 | 8883 | 6749 | 75.98% | 1820.357716 | 259.022 | 11.850359 | 96.03% |
| 725374 | 8692 | 6326 | 72.78% | 1699.831464 | 258.379 | 11.843425 | 96.16% |
| 725375 | 8408 | 4769 | 56.72% | 1322.529592 | 266.021 | 11.849139 | 95.93% |
| 725376 | 8556 | 5096 | 59.56% | 1405.142795 | 264.580 | 11.844695 | 95.95% |
| 725460 | 8409 | 6609 | 78.59% | 2045.554391 | 297.602 | 10.546013 | 96.15% |
| 725461 | 8381 | 6713 | 80.10% | 2100.483286 | 300.906 | 10.505362 | 96.17% |
| 725462 | 8390 | 6145 | 73.24% | 1903.086337 | 297.879 | 2.814365 | 96.18% |
| 725463 | 8132 | 5565 | 68.43% | 1735.313411 | 300.291 | 10.381628 | 96.30% |
| 725566 | 9841 | 8075 | 82.05% | 2202.631897 | 261.710 | 10.417248 | 95.94% |
| 725567 | 9161 | 6202 | 67.70% | 1701.814490 | 263.762 | 10.406008 | 96.12% |
| 725568 | 9521 | 6064 | 63.69% | 1665.162985 | 263.523 | 10.411854 | 95.97% |
| 725569 | 9336 | 6028 | 64.57% | 1674.177007 | 266.722 | 10.400042 | 96.04% |
| 725574 | 9005 | 6815 | 75.68% | 1917.237911 | 270.351 | 16.499112 | 96.10% |
| 725575 | 8940 | 6602 | 73.85% | 1890.544547 | 275.403 | 16.726670 | 96.17% |
| 725576 | 9066 | 6331 | 69.83% | 1791.163188 | 272.152 | 16.514534 | 96.19% |
| 725577 | 9030 | 6503 | 72.02% | 1846.716103 | 273.245 | 16.499861 | 96.22% |

各进程 forward 平均值约 258–301 ms，最大值最高 16.727 s。长尾需要关联具体迭代、请求和 rank，不能由单个最大值推定 HCCL 故障或全部发生于 decode。

## 5. Forward 设备采集及算子耗时

### 5.1 两次采集故障和可用数据

较早的 `runs/huanyu_forward_1109` 采集中，PID 1090560 的旧格式 manifest 只有 `status=starting`、`iterations=0`，原始目录缺少 `profiler_info*.json`。当时 torch 为 `2.10.0+cpu`，torch-npu 为 `2.10.0.post4`，msprof 位于 CANN 9.1.0，SQLite 动态库可加载。多级 parser 报错后仍打印 “All profiling data parsed”，不能据此判为解析成功。

加入进度记录后的 `runs/huanyu_forward_1110` 提供了明确的停止边界：

| 配置或状态 | 实际结果 |
| --- | --- |
| `HUANYU_FORWARD_PROFILE_START` | 5 |
| `HUANYU_FORWARD_PROFILE_ACTIVE` | 5 |
| 采集计划 | iteration 5 为 1 次 warmup；6–10 为 5 次 active |
| 16 个 worker 的共同状态 | `status=capturing phase=profiler.step iteration=10 forward=6 step=5/6` |
| 最后完成的边界 | `last=profiler.synchronize` |
| 当前 phase stack | `['profiler.step']` |
| profiler 元数据 | PID 1635029 目录无 `profiler_info*.json` |
| 引擎配置 | 2748、TP=4、bf16、eager、graph NONE，max_seq_len=3328 |
| 后续错误 | `sample_tokens` RPC 超时，继而引擎退出和副本淘汰 |

可确认：6 次 forward 调用已经返回；第 6 次 profiler.step 未记录完成；已记录的 profiler.synchronize 已返回；还没有进入包装器后续的显式 profiler.stop。最后一次 step 内部可能执行停止、finalize 和回调，当前状态不能进一步定位到某一个 CANN/C++ 函数。

`phase_age_s≈4990–4999` 是持久化状态的年龄，不证明进程当时仍存活或已阻塞同样长时间。Ray runtime_env 中 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=None` 只说明未在该记录显式设置；源码默认 300 s，但仍可能继承其他环境值，未确认本次最终有效超时。

### 5.2 已返回的 execute_model 主机时间

PID 1635029、TP rank 0；每次 `scheduled_tokens=4`、`scheduled_requests=4`。结合此前 scheduler 输出已有生成 token，可视为本次小批量 decode 窗口；一般情况下不能仅凭这两个计数给所有作业分类。

| engine iteration | warmup | tokens | requests | execute_model 主机时间 s |
| --- | --- | ---: | ---: | ---: |
| 5 | 是 | 4 | 4 | 3.213809116 |
| 6 | 否 | 4 | 4 | 5.422820201 |
| 7 | 否 | 4 | 4 | 3.657187227 |
| 8 | 否 | 4 | 4 | 6.087048108 |
| 9 | 否 | 4 | 4 | 4.477864381 |
| 10 | 否 | 4 | 4 | 3.328111349 |
| 5 次 active 合计 | — | — | — | **22.973031266** |
| active 平均 | — | — | — | **4.594606253** |

这是 execute_model 包装器的主机时间，不是单独的纯 forward kernel 时间；后续 profiler.step/stop 不在该字段内，但开启 profiler 会影响被采集执行本身。不能直接与 11 月 9 日不同负载下的 forward 均值相除，声称 profiler 固定拖慢了多少倍。

### 5.3 CANN 独立导出来源

原始目录位于：

```text
/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new/runs/huanyu_forward_1110/
  ma-job-a27663aa-e237-4c73-8eed-1175976dc05a-vj-1-worker-0_parent1634911/
    pid1635029_tp0_t1794295312949874108_1635029_20261110152212767_ascend_pt/
```

实际分析副本和结果：

```text
/tmp/huanyu_partial_profile_imubo9ga/
  PROF_000001_20261110152215145_01635029NPOMGJFC/
    mindstudio_profiler_output/op_summary_20261110170446.csv
```

对复制的 `PROF_*` 执行 `msprof --export=on --output=<PROF目录>`，返回码为 0；随后实际读取到 **9637 条有正时长的算子记录，跳过 0 行，累计 duration=6.064927 s**。日志中的部分 “no summary data” 不否定这张已生成的算子表，也不证明其他视图完整。临时分析目录应在清理 `/tmp` 前归档。

这属于**部分数据导出可用**，不是完整 PyTorch profiler 采集/解析成功。缺少结束元数据，不能断言所有 active step 的设备事件齐全。采集器也可能记录 forward 之外的 logits、采样和其他活动，不能将全部 CSV 行都归入某个 forward 模块。

### 5.4 已导出设备任务的类别占比

分母固定为 **6064.927 ms 的任务 duration 之和**，不是 22.973 s 主机时间，也不是设备时间轴区间并集。

| 类别 | 次数 | 累计 ms | 平均每次 ms | 最大一次 ms | 累计 duration 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HCCL AllReduce | 406 | 4777.846 | 11.768 | 680.198 | **78.7783%** |
| HCCL AllGather | 4 | 1217.959 | 304.490 | 327.231 | **20.0820%** |
| 其他任务合计 | 9227 | 69.122 | — | — | **1.1397%** |
| 总计 | **9637** | **6064.927** | — | — | **100%** |

通信任务合计 **5995.805 ms，98.8603%**。HCCL 时长可能包含同步、等待和真正传输；不能将 5.996 s 全部算成网络搬运时间。并发流、通信父任务与设备事件也可能重叠，累计 duration 不是暴露在关键路径上的通信时间。

### 5.5 已提供的全部 Top 25 算子

| Task Type | Op Type | 次数 | 累计 ms | 平均 μs | 最大 μs | 占已导出任务 duration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COMMUNICATION | hcom_allReduce_ | 406 | 4777.846 | 11768.094 | 680197.870 | 78.7783% |
| COMMUNICATION | hcom_allGather_ | 4 | 1217.959 | 304489.793 | 327230.918 | 20.0820% |
| AI_VECTOR_CORE | Index | 630 | 9.685 | 15.374 | 48.600 | 0.1597% |
| MIX_AIC | GroupedMatmul | 200 | 8.084 | 40.418 | 51.040 | 0.1333% |
| AI_VECTOR_CORE | Mul | 959 | 6.754 | 7.042 | 20.000 | 0.1114% |
| AI_CORE | linear_persistent_kernel | 577 | 6.586 | 11.414 | 28.242 | 0.1086% |
| AI_VECTOR_CORE | ScatterUpdate | 75 | 5.344 | 71.257 | 76.542 | 0.0881% |
| MIX_AIV | ReduceSum | 318 | 3.403 | 10.702 | 27.360 | 0.0561% |
| AI_VECTOR_CORE | Cast | 1089 | 2.349 | 2.157 | 11.800 | 0.0387% |
| AI_VECTOR_CORE | Add | 659 | 1.801 | 2.734 | 13.442 | 0.0297% |
| AI_VECTOR_CORE | IndexCheck | 630 | 1.738 | 2.759 | 5.940 | 0.0287% |
| AI_VECTOR_CORE | Transpose | 78 | 1.613 | 20.684 | 28.022 | 0.0266% |
| MIX_AIV | MoeInitRoutingV3 | 100 | 1.413 | 14.129 | 15.080 | 0.0233% |
| AI_VECTOR_CORE | RepeatInterleave | 150 | 1.255 | 8.365 | 10.520 | 0.0207% |
| AI_VECTOR_CORE | ConcatD | 75 | 1.238 | 16.507 | 19.160 | 0.0204% |
| AI_VECTOR_CORE | MemSet | 153 | 1.238 | 8.088 | 19.120 | 0.0204% |
| AI_VECTOR_CORE | Cumsum | 3 | 1.226 | 408.655 | 409.888 | 0.0202% |
| AI_VECTOR_CORE | Slice | 580 | 1.190 | 2.052 | 3.580 | 0.0196% |
| AI_VECTOR_CORE | MaskedFill | 90 | 1.111 | 12.340 | 28.360 | 0.0183% |
| AI_CORE | MatMulV2 | 102 | 1.082 | 10.607 | 232.124 | 0.0178% |
| MIX_AIV | Sort | 3 | 0.950 | 316.547 | 317.366 | 0.0157% |
| AI_VECTOR_CORE | NonZero | 150 | 0.764 | 5.092 | 5.940 | 0.0126% |
| MIX_AIC | FusedInferAttentionScore | 25 | 0.689 | 27.575 | 29.480 | 0.0114% |
| AI_VECTOR_CORE | SwiGlu | 200 | 0.664 | 3.318 | 4.720 | 0.0109% |
| MIX_AIC | CausalConv1d | 76 | 0.650 | 8.555 | 10.060 | 0.0107% |
| 未列出的其他算子合计 | 由总数和 Top 25 相减 | 2305 | 8.295 | — | — | 0.1368% |

其他算子累计值按已打印的毫秒数相减，存在原始表舍入误差。非通信项占比很小，只表示在这份已导出数据中如此，不能据此声称整个模型计算只需 69 ms。

### 5.6 尚不能给出的 forward 模块占比

| 模块 | 已看到的相关任务举例 | 仍缺少的证据 |
| --- | --- | --- |
| Attention | FusedInferAttentionScore、线性投影相关 kernel | QKV/输出投影、通信及等待与具体层的关联 |
| MoE | GroupedMatmul、MoeInitRoutingV3、SwiGlu 等 | routing、expert 计算、共享 expert、通信的完整范围 |
| GDN | CausalConv1d、索引、逐元素运算等候选 | reference 路径、状态更新、同步和具体模块的关联 |
| 主机开销 | execute_model 主机时间明显大于导出任务累计值 | 同一窗口内 CPU 栈、设备流和空档的对齐 |

不能把一个 `FusedInferAttentionScore` kernel 的 0.0114% 当作完整 Attention 模块占比，也不能把 GroupedMatmul 当作整个 MoE。`22.973 - 6.065` 不是合法的“CPU 耗时”估计：二者覆盖、并发和时钟口径不一致，设备事件还可能缺失。

## 6. 通信含义和 TP 性能判断

### 6.1 AllReduce 和 AllGather 传递什么

当前窗口来自 rollout 推理 worker，频繁 AllReduce 很符合 TP 各层汇总局部输出的模式；AllGather 可能汇总分片激活或词表 logits。但算子名称本身不含通信组、调用栈和张量形状，不能将全部调用唯一归类。

vLLM 的 RowParallelLinear 会将各卡的局部结果求和。例如 TP=4 时，`Y = X0W0 + X1W1 + X2W2 + X3W3`；通常通信的是输出激活的局部贡献，权重分片保留在各卡。LogitsProcessor 也支持在 TP 组内 gather/all-gather 词表分片。它们分别见 [vLLM 0.28.0 linear.py](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/linear.py) 和 [logits_processor.py](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/logits_processor.py)。这些源码解释典型路径，不替代对本次原生适配器调用栈的确认。

不能把这里的集合通信直接解释为 Actor 反向梯度同步、FSDP 参数聚合或 Actor 向 rollout 更新权重。

### 6.2 是否意味着单卡比多卡快

这是成立的可能性，尚不是实测结论。每次只有 4 个 decode token 时，计算量较小，TP 每层通信的固定延迟容易突出。如果不能被计算覆盖的通信/等待成本超过分片节省的计算和访存时间，TP=4 的同批次延迟可能高于 TP=1。

但当前 98.86% 是部分采集的任务累计时长比例；它不能证明“正常训练 98.86% 的时间用于通信”。profiler 自身及诊断写文件会扰动速度，尤其不能将这次最终收尾失败的短窗口当作正常吞吐基准。[vLLM profiling 文档](https://docs.vllm.ai/en/stable/contributing/profiling/) 明确区分 profiling 和性能基准测试。

需要区分三种比较：

| 对比 | 衡量什么 | 条件 |
| --- | --- | --- |
| 单副本 TP=1 vs TP=4 总吞吐 | 同一模型副本增加卡数是否加速 | 同一精度、输入输出长度和实际并发；TP=1 能容纳完整模型与 KV |
| TP=4 总吞吐除以 4 vs TP=1 吞吐 | 单卡资源效率 | 不能把总吞吐和每卡吞吐混比 |
| 4 个 TP=1 副本 vs 1 个 TP=4 副本 | 固定总卡数的部署吞吐 | 有足够请求、每个单卡副本都能装下模型 |

TP=4 总吞吐提高而平均单卡吞吐下降并不矛盾。若单卡需要 CPU offload 或降低精度才装得下，则已经改变了实验条件，不能用于证明纯 TP 通信的得失。

## 7. 结论强度和后续验证

| 判断 | 当前证据强度 |
| --- | --- |
| 固定长输出 AQA 的 gen 是端到端主要耗时 | 已支持：BS=4 六步约 93.11%；BS=16 step 7 为 81.84% |
| 首次媒体编码有显著长尾 | 已支持：两个计时作业均观察到；具体初始化/编译/编码构成未知 |
| 后续有效 execute 中 forward 主机调用占主导 | 已支持：16 个进程为 95.93%–96.30%，但只覆盖对齐子集 |
| 新 profiler 收尾阻塞比 forward 未返回更符合状态 | 已支持：forward=6、step=5/6、最后完成 profiler.synchronize |
| PID 1635029 部分 CANN 数据可用 | 已支持：9637 条有效任务，实际读到 op_summary |
| 已导出任务中 HCCL 累计时长最突出 | 已支持：98.8603%；不等于关键路径通信占比 |
| 网络带宽不足是根因 | 未证明：缺少传输大小、带宽、Wait/Transit 分解和跨 rank 对齐 |
| GDN reference 或 CPU 下发导致其他 rank 等待 | 候选：需要实际时间线和调用栈，不由现有比例直接推出 |
| TP=1 比 TP=4 更快 | 未实测：需关闭 profiler 后做受控对照 |

后续工作按以下顺序推进：

1. **保存当前可用证据。** 归档 manifest、原始 PROF/FRAMEWORK、导出 CSV 和日志；不修改 manifest 为 complete，不借其他 rank 的 profiler_info 补齐时间基准。
2. **先利用现有设备数据。** 计算同一时间轴内任务区间并集和间隙；若有通信明细，查看 AllReduce/AllGather 的等待、同步、传输时长及字节数。缺失事件可能伪造“空闲”，空档不能直接判成 CPU 瓶颈。
3. **补同一 TP 组的比较。** 核对组身份后对比 PID 1635029、1635030、1635031、1635032；看匹配的 collective 谁晚到、谁等待，不能仅用所有进程的第 N 行强行对应。传输慢与等待慢的分析口径见 [Ascend 通信分析](https://github.com/Ascend/msinsight/blob/master/docs/zh/user_guide/system_tuning.md)。
4. **定位并完成 profiler 收尾。** 若进程仍存活，取得实际线程栈；进一步区分 stop_trace、finalize_trace、写元数据及回调。延长 RPC timeout 只能增加等待预算，不保证死锁解除。
5. **恢复模块与 kernel 关联。** 完整 trace 中将 `huanyu/forward`、GDN、Attention、MoE 范围与设备 kernel/HCCL 关联，计算关键路径和排除嵌套后的占比。通过元数据、forward 范围和 kernel 文件校验后才称完整解析成功。
6. **关闭 profiler 做吞吐对照。** 先保持 TP=4、当前 batch 测正常基准，再比较可运行的 TP=1/2/4；固定精度、模型、数据和长度，先比较实际并发 4，再提高并发。记录总 token/s、token/s/卡、请求延迟与显存，不同时改变精度和图模式。

计时工具和运行命令分别见 [gen 计时说明](gen_timing.md)、[forward NPU profiler 说明](forward_profile.md) 和[当前启动入口 Wiki](../README.md)。
