# 主仓库及其他插件脚本补丁

## new 分支：同步的算子优化、HCCL 探针与 replay 插件

本次同步包括 `qwen3_6_huanyu/` 和其 `examples/`。examples 中不再存在的旧
可执行入口已删除；历史 Markdown 记录保留。新的 16 卡 AQA 五组吞吐入口与部署说明见
[算子对照说明](../qwen3_6_huanyu/docs/aqa_operator_ablation.md)。

此外必须将本仓库 `probes/hccl_policy.py` 放到训练项目的
`verl_omni_ext/probes/hccl_policy.py`，训练与 rollout 的校验调用均已保留。
它检查环境声明一致性，不测量设备通信算法。

`huanyu-r3-entrypoint.patch` 为完整训练项目根目录的 `pyproject.toml` 添加
`vllm.general_plugins.huanyu_r3`。服务器若已有同名同目标声明，不重复应用。
检查、应用后使用训练所用的 Python 更新安装元数据：

```bash
cd "$PROJECT_ROOT"
git apply --check /path/to/ext/patches/huanyu-r3-entrypoint.patch
git apply /path/to/ext/patches/huanyu-r3-entrypoint.patch
python3 -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
python3 "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/aqa_ablation.py" plugins
```

若服务器 pyproject 内容与补丁上下文不同，按上述算子对照说明中的 TOML 小节添加，
仍需执行安装与 metadata 检查。单纯复制 Python 源码或设置 PYTHONPATH 不会生成入口元数据。

## o5 视觉 checkpoint 环境变量

`o5-q36-checkpoint-env.patch` 修改两个 o5 基础启动脚本：

- `verl_omni_ext/examples/run_minicpmo_thinker_gspo_npu.sh`
- `verl_omni_ext/examples/run_minicpmo_thinker_gspo_npu_avqa.sh`

两者显式导出 `Q36_CKPT_DIR` 并写入 Ray `runtime_env.env_vars`，保留调用者
传入的值，启动时打印实际路径。GenRM 和 OPD 包装入口调用基础脚本，自动继承。
这两个脚本不在本分支的源码布局内，仅拉取 profile 不会修改服务器上的脚本。
在包含上述文件的训练项目根目录执行（替换为下载后的实际补丁路径）：

```bash
git apply --check /path/to/o5-q36-checkpoint-env.patch
git apply /path/to/o5-q36-checkpoint-env.patch
```

若脚本已经包含修改，无需重复应用。启动时使用
`Q36_CKPT_DIR=/path/to/vision/checkpoint bash <o5启动脚本>`；各 worker 节点
需要能读取该目录。此补丁保证变量显式传递，不验证 checkpoint 的视觉权重内容。

## gen 阶段计时（profile 分支）

`qwen3_6_huanyu/` 包含计时模块、引擎接入、启动变量透传及测试。
ext 包根目录和 trainer 不在本仓库布局内，两处接入作为
`gen-timing-integration.patch` 一并发布。部署时将 `qwen3_6_huanyu/`
更新到 `verl_omni_ext/models/qwen3_6_huanyu/`，再在训练项目根目录运行：

```bash
git apply --check /path/to/gen-timing-integration.patch
git apply /path/to/gen-timing-integration.patch
```

用本仓库 `patches/gen-timing-integration.patch` 的实际路径替换上述路径。
补丁让 ext 激活时安装 trainer/agent/server 计时，并透传两个计时环境变量。
若目标文件已包含该接入，无需再次应用；不能仅复制计时模块而遗漏接入。
启用方式、日志汇总和 host_wall 测量范围见
[gen_timing.md](../qwen3_6_huanyu/examples/gen_timing.md)。

## 训练显存默认值

本分支直接发布 `qwen3_6_huanyu/`。另外 8 个本地训练脚本不在本分支中，
对应修改保存在 `other-training-memory-defaults.patch`，路径相对于 verl-omni 项目根目录。
它将 Actor rollout 默认显存预算改为 0.6，保留 GenRM 和 AB/serve 设置。

在包含这些脚本的项目根目录使用补丁文件的实际路径，先执行
`git apply --check /path/to/other-training-memory-defaults.patch`，
检查通过后执行 `git apply /path/to/other-training-memory-defaults.patch`。
如果该工作区已经应用过本次修改，无需再次应用。

## 使用已安装的 vLLM-Omni

`qwen3_6_huanyu/` 下的 prepare、job 和训练入口已直接包含此次修改。
另外 5 个历史入口不属于本分支的目录布局，其改动放在
`installed-vllm-source-policy.patch`，路径相对于 verl-omni 项目根目录：

- `verl_omni_ext/env_export.sh`
- `verl_omni_ext/env_export_huanyu.sh`
- `verl_omni_ext/examples/run_huanyu.sh`
- `verl_omni_ext/examples/run_qwen36_huanyu_32card.sh`
- `verl_omni_ext/examples/run_qwen36_whisper_32card.sh`

补丁移除这些入口对 vLLM-Omni 的 PYTHONPATH 注入，保留 ext/CANN 等依赖路径；
默认使用已安装的 `/vllm-workspace/vllm-omni`。在包含上述文件的项目根目录运行：

```bash
git apply --check /path/to/installed-vllm-source-policy.patch
git apply /path/to/installed-vllm-source-policy.patch
```

用下载后的实际补丁路径替换 `/path/to/installed-vllm-source-policy.patch`。
若这些文件已更新，无需重复应用；仅拉取本分支不会自动修改外部目录中的文件。
环境更新后重新 prepare，并使用新生成的激活文件；旧 Ray/worker 不会自动切换安装来源。
