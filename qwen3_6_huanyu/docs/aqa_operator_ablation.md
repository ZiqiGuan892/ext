# AQA 16 卡算子与图模式吞吐对照

Last updated: 11/16/2026.

本入口使用同步后的 `run_qwen36_huanyu_gspo_npu.sh` 内置精度策略，不使用明确拒绝
kernel 档案的 `run_huanyu_train.sh`。环境和 Ray 由调用者提前准备；测试不安装软件、不重启 Ray。

| 组名 | 精度档案 | GDN / RMS 融合算子 | rollout 图模式 | 融合 MoE | route replay |
| --- | --- | --- | --- | --- | --- |
| full | full | 关 / 关 | 关 | 关 | 关 |
| kernel | kernel | 开 / 开 | 关 | 关 | 关 |
| graph | kernel_graph | 开 / 开 | 开 | 关 | 关 |
| moe | kernel_graph | 开 / 开 | 开 | 开 | 关 |
| replay | kernel_graph | 开 / 开 | 开 | 开 | 开 |
| gdn_off（额外） | kernel | 关 / 开 | 关 | 关 | 关 |
| rms_off（额外） | kernel | 开 / 关 | 关 | 关 | 关 |

默认负载：2748、AQA `aqa_dataset/train.parquet` 和 `validation.parquet`，单节点 16 卡，
TP4 × 4 个 rollout 副本，BS16、PPO minibatch16、n8，输入 3072 / 输出 2048 token 上限，
动态 batch，encoder/projector 冻结、在线编码、不使用 embedding 缓存、规则奖励、10 个训练 step。
各组固定 AIV、FP32 梯度归约，关闭预取和新增权重同步优化；关闭验证、保存、profiler 以及逐层调试日志。
需要 Ray 使用正常的默认训练引擎，不能沿用启用了 `VERL_ENGINE_VENDOR=huanyu_wsync` 的集群环境。

`full → kernel` 同时改变 GDN/RMS 实现、数值策略和 Mamba cache 精度，不能把总收益归因于某一个算子。
用 `gdn_off` / `rms_off` 辅助归因。10 step 吞吐测试与 Pearson/KL 指标不能替代算子误差测试或长期训练精度验证。

## 部署与插件安装

发布仓库是扩展源码布局，不是完整训练项目：

- `qwen3_6_huanyu/` 对应 `$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/`。
- 其中 `examples/` 同时可部署到 `$PROJECT_ROOT/examples/qwen3_6_huanyu/`，同一目录内保留后端脚本和策略文件。
- `probes/hccl_policy.py` 对应 `$PROJECT_ROOT/verl_omni_ext/probes/hccl_policy.py`，本次已恢复两个调用点。
- `patches/huanyu-r3-entrypoint.patch` 将 `huanyu_r3` 声明加入完整项目根目录的 `pyproject.toml`。
  如果服务器已经有同名、同目标入口，不重复添加。仅设置 `PYTHONPATH` 不能替代 entry point 安装。

在完整训练项目的 `pyproject.toml` 中确认有：

```toml
[project.entry-points."vllm.general_plugins"]
huanyu_r3 = "verl_omni_ext.models.qwen3_6_huanyu.expert_replay.plugin:register"
```

更新安装元数据（每个使用的节点/容器都需要；不升级依赖）：

```bash
python3 -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqa_ablation.py" plugins
```

有 `VLLM_PLUGINS` 白名单时需包含 `huanyu_r3`，并保留原插件。启动器会将白名单传给新 worker。
不需要为安装元数据主动重启已准备好的 Ray；此前失败作业的模型进程必须退出，新作业使用新进程读取代码。
HCCL 探针只证明声明与进程环境一致，不能证明设备实际使用了某种通信算法。

## 启动

`SHELL_ROOT` 指向刚部署的 examples 目录。先预览，再顺序运行全部五组：

```bash
export BENCH_ROOT="$PROJECT_ROOT/runs/aqa_ablation_$(date +%Y%m%d_%H%M%S)"
bash "$SHELL_ROOT/run_huanyu_aqa_ablation.sh" all --dry-run
bash "$SHELL_ROOT/run_huanyu_aqa_ablation.sh" all
```

或单独运行 `bash "$SHELL_ROOT/run_huanyu_aqa_ablation.sh" replay`，额外二分使用 `gdn_off` / `rms_off`。
`all` 先检查 replay 元数据，任何一组失败就停止；不会在后台同时运行多个训练作业。
已有同名组目录会拒绝覆盖。日志：`$BENCH_ROOT/aqa_16card_<组名>_step10/console.log`。

公共负载可通过 `TRAIN_FILE`、`VAL_FILE`、`MODEL_PATH`、`Q36_CKPT_DIR`、`TRAIN_BATCH_SIZE`、
`PPO_MINI_BATCH_SIZE`、`ROLLOUT_N`、`MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH`、
`TOTAL_TRAINING_STEPS` 覆盖。所有组须使用相同值；不要把不同负载的数字当成严格 A/B。
`SOCKET_IFNAME` 沿用已准备环境，未设置时后端使用 `eth0`；需要符合服务器实际网卡。

## 汇总与证据范围

```bash
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqa_ablation.py" report "$BENCH_ROOT"
```

默认使用 step3–10；吞吐为 `sum(perf/total_num_tokens) / sum(timing_s/step) / 16`，不是输出 token 吞吐。
同时输出 gen、old_log_prob、update_actor、update_weights 均值、Pearson。
`report.json` 保留日志中的响应长度、截断比例、KL、reward、梯度等所有数值指标均值，以及路径/负载配置。
缺 step、进程非零退出、非有限指标或非正耗时的组不会得到可排名的吞吐值。

运行证据分开记录：GDN/RMS/MoE 安装、原生 rollout GDN 校验、graph 模式解析、
replay worker/scheduler/actor 钩子、router capture 绑定、实际路由存储、HCCL 环境一致性。
`complete=true` 只表示该训练作业完整结束，**不等于所有优化已在所有 rank 得到验证**。
graph 的 resolved 或捕图完成日志仍不足以证明稳态每一步实际回放图；需要设备 trace/runner 运行证据进一步核实。

本次本地验证覆盖启动参数与真实后端策略透传、HCCL 契约、报告完整性和吞吐分母。
本地没有 NPU，未产生新的真实性能数据；服务器运行结果才是本次 A/B 结论的依据。
