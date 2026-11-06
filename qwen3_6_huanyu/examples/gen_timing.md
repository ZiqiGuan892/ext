# Huanyu gen 阶段计时

从本仓库部署时，除了更新 `qwen3_6_huanyu/`，还需要应用
[`gen-timing-integration.patch`](../../patches/gen-timing-integration.patch)
中的 ext 根目录和 trainer 接入；具体命令见
[`patches/README.md`](../../patches/README.md)。目标已包含这些接入时无需重复应用。

在已有训练命令前设置以下变量，启动新的训练进程即可，不需要重新启动 Ray：

```bash
HUANYU_GEN_TIMING=1 HUANYU_GEN_TIMING_ENGINE_EVERY=100 \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8 \
bash "$SHELL_ROOT/run_huanyu_aqa_audio_train.sh"
```

同样支持 Huanyu 的 job/STEM 入口。默认 `HUANYU_GEN_TIMING=0`，不安装计时包装。
`precision_mode.sh` 会保留这两个环境变量并显式传给 Ray worker。
已有作业不会因在另一个 shell 执行 export 而获得这些钩子；部署修改后需启动新作业。
不修改模型、采样参数、精度配置、checkpoint 或依赖仓库源码。

日志为 `[huanyu-gen-timing]` 加一条 JSON。`hook` / `engine_hook_installed` 说明安装，
实际 `span` 才说明执行；`unavailable` 表示该版本接口未命中，不能解释为耗时为零。
首次导入可因循环依赖暂时不可用，processor/worker 初始化会重试。

| phase | 实际测量范围 |
|---|---|
| `trainer.submit` | 提交采样任务；该调用发生在原生 `gen` 计时开始之前 |
| `trainer.update_cycle` | `_step_once` 整体，包含训练；仅用于建立 step 上下文，绝不是 gen |
| `trainer.sample_begin` | 原生 gen 内采样前处理 |
| `trainer.wait_batch` | ReplayBuffer 等待完整训练 batch；包含生成、评分、后处理等待，不是纯队列读开销 |
| `trainer.sample_end` | 原生 gen 内结束处理；omni_sync 包含休眠 rollout 副本 |
| `agent.trajectory` | 一个 uid/session 的整个 agent loop |
| `agent.generate_request` | 客户端选择副本、RPC、引擎排队/执行和返回 |
| `agent.acquire_server` | 客户端选择/获取 server 的调用，包含于 generate_request |
| `server.generate_request` | 服务端 generate 调用，仍包含引擎排队；model 字段区分 actor / judge |
| `agent.postprocess` | 评分、teacher、媒体输入、位置、结果打包和写队列的总区间 |
| `agent.reward` | 评分准备、远程 GenRM 等待及返回处理；不是单独 judge forward |
| `agent.teacher` | teacher logprob 调用，未启用时可以很短 |
| `agent.media_inputs` / `agent.positions` | 训练输入准备及位置编码构造 |
| `agent.queue_put` | 实际 async_kv_batch_put 的调用；调用前的 TensorDict 转换未包含 |
| `engine.execute_model` | 采样到的 runner 迭代，包含下面的部分区间 |
| `engine.preprocess` / `engine.mm_encoder` | runner 预处理 / 多模态编码调用 |
| `engine.forward` / `engine.compute_logits` | 实际顶层模型 forward / logits 调用 |
| `engine.sample_tokens` | 采样器调用；有些版本在 execute_model 返回后单独执行 |

引擎只采样第 1 次和每 N 次迭代，N 默认 100；设为 1 才是逐迭代记录，会产生大量日志。
支持当前 Huanyu NPUARModelRunner；不声称覆盖其他 runner。标记 scheduled_requests / scheduled_tokens
仅来自 CPU 调度元数据，不能靠 token 数自动认定 prefill/decode。引擎迭代不直接知道训练 step，
也不自动关联请求 uid；服务端 weight_step 是权重版本，不能等同于 trainer.step。
ContextVar 不跨 Ray RPC 传播。verl 在非 full_determinism 模式下还会重写发给服务端的
request_id，因此客户端与服务端记录不能直接按 request_id 联表；当前应分别分析两端的
调用区间，这套计时尚不是跨进程完整 trace。

## 查看结果

```bash
grep -aF '[huanyu-gen-timing]' "$LOG" | tail -n 40

/usr/local/python3.12.13/bin/python3 \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/gen_timing.py" "$LOG"

# 仅筛选能明确关联到训练 step 的记录；引擎/RPC 记录可能没有 step。
/usr/local/python3.12.13/bin/python3 \
  "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/gen_timing.py" "$LOG" --step 2
```

汇总显示每个 phase 的次数、均值、最大值、work_s。不同进程/rank 合并统计，不能将 work_s
视为该阶段对整步的墙钟贡献；Actor 与 GenRM 的同名 server/engine phase 也会合并，
判断具体模型瓶颈时须按原始记录的 model/model_path、host、pid 区分。
保留原始 JSON 的 host/pid/step/uid/session_id/request_id、
span_id/parent_span_id 和起止时间，才可比较慢请求或嵌套区间。跨机器比较绝对时间要求时钟同步。
Ray 日志去重、截断或缺失会影响汇总；它不是独立的完整性审计记录。

## 时间的含义

全部记录使用 perf_counter 测量 `host_wall`，无 NPU synchronize、额外 collective 或张量读回。
forward 的异步 kernel 可能在函数返回后仍执行；阻塞调用也可能包含通信、设备等待。
这能定位主机调用和等待区间，但不能精确分解 NPU kernel / HCCL / 访存的设备耗时；后者仍需 profiler。
嵌套区间和并发请求互相重叠，**不能将各 phase 的 sum 相加后称为 gen 总耗时或百分比**。
日志输出本身有开销，诊断运行不应直接当作无注入的吞吐基线。

例如 wait_batch 很长而 sample_end 很短，应继续看请求生成、评分及后处理；若 agent.reward
明显晚于生成结束，需要分析 judge；若引擎 forward 主机耗时长，仍应通过设备 trace 区分
计算、通信和同步等待。单次 hook 安装成功或单次采样，均不证明某阶段是整个作业瓶颈。
