# Qwen3.6-Huanyu 训练适配

本目录接入 Huanyu 的 GSPO/GenRM 链路。本次实现、测试、启动入口和文档均限定在此目录，没有修改 Whisper、RM、公共 verl-omni 或 vLLM-Omni 源文件。

状态：训练接口已实现，CPU 合约测试通过；本地没有服务器 checkpoint、Transformers/verl/vLLM 运行栈或 NPU，尚未完成真实模型的分布式训练验证。

## 原因与确认的接口

之前只切换 MODEL_PATH 和推理后端路径，训练端仍由原注册项选择 Whisper adapter，造成 Huanyu encoder 权重全部 unexpected，以及 projector 的 `[2048,3072]` 对 `[2048,1024]` 形状错误。

用户提供的真实服务器构造代码和 audio_preprocessor_config.json 已确认：

| 项目 | 本次模型 |
|---|---|
| HF 主体 | Qwen3_5MoeForConditionalGeneration |
| 音频前端 | 16 kHz、40 维 fbank；n_fft=400、hop=160 为运行时默认值 |
| Encoder | VGG + 8 层 Conformer，hidden=1536 |
| VGG 时间下采样 | 4；40 维版 flat Linear 输入为 1920，不能当作 projector 输入 |
| Projector | 两帧拼接：1536×2=3072 → linear1 → ReLU → linear2 → 2048 |
| 有效音频 token 数 | `ceil(真实 fbank 帧数 / 4) // 2`，按 30 秒 chunk 计算 |
| checkpoint 前缀 | `model.audio_encoder.*`、`model.audio_projection.*` |

旧 processing_huanyu.py 是 128 维版本，不用于当前 40 维链路。本目录的 audio_frontend.py 和 load_huanyu_processor() 保留为旧版检查组件；当前训练使用 build_huanyu_processor()。

## 实现分工与数据流

| 文件 | 职责 |
|---|---|
| thinker_adapter.py | 独立注册 Qwen36HuanyuForTraining + thinker，挂载 Huanyu 音频模块 |
| audio_runtime.py | 复用当前 vLLM-Omni 的 encoder、projector、fold；严格加载音频权重 |
| runtime_imports.py | 检查实际进程的 vLLM-Omni 来源、Huanyu 文件和接口；提供独立导入预检 |
| processor.py | 40 维 fbank、Qwen 图像处理、媒体 token 展开、原生 HF mRoPE、AgentLoop 输入桥接 |
| dataset.py | 读取原始图片、音频、视频帧；不读取 Whisper cache |
| forward.py | 保留 HF 图像/文本前向，在经过验证的音频 token 位置注入 embedding |
| rollout_inputs.py | 按媒体元数据收缩推理占位符，兼容独立预处理方法及内联预处理的 async generate 接口 |
| rollout_adapter.py | 独立注册 qwen3_6_huanyu，复用现有单 llm stage 拓扑 |
| run_32card.sh | 调用原 32 卡入口，追加 Huanyu 配置、关闭长度预筛，显式传递 Ray 导入路径 |
| check_runtime.py | 不加载模型权重的真实依赖/processor/mRoPE CPU 接口检查 |

```text
音频路径/bytes/zarr member
 → 16 kHz 单声道波形 → 按原音频分 30 秒 chunk
 → 40 维 fbank [40,T] + 未 padding 的真实帧数
 → Huanyu VGG/Conformer [T_encoder,1536]
 → 相邻两帧拼接 [T_tokens,3072]
 → 原生 projector [T_tokens,2048]
 → 替换 <|audio_start|><unk>...<|audio_end|> 中的 <unk> embedding
 → HF Qwen3.5-MoE 文本主干 → logits/log_probs → 原 GSPO 更新
```

图像交给 Qwen image processor 和 HF 原生视觉模块；视频用已有帧采样 helper 转为有序图片。奖励函数、GenRM、优势和 GSPO 公式不由本目录改写。

新增训练字段为 huanyu_audio_features、huanyu_audio_lengths、huanyu_audio_bounds、huanyu_audio_chunk_counts。每条样本始终带 chunk_count（无音频为 0），进入 batch 后逐条验证，避免跨样本错配。音频 bounds 从实际 input_ids 重建，包含左 padding 后的位置，不依赖 decode 后的文本。

推理侧仍接收原始图片/波形，由 vLLM-Omni 自己编码。31 秒音频在训练侧分两个 chunk，回到推理侧仍是一个原始 audio 条目，不能按相邻标记猜条目边界。两个运行时 hook 仅处理 Huanyu processor，其他模型转交原方法；没有修改公共源码文件。

## 复用的服务器实现

```text
/vllm-workspace/vllm-omni/vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py
  HuanyuFbankFeatureExtractor / HuanyuAudioEncoder / huanyu_project_and_fold
/vllm-workspace/vllm-omni/vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py
  MultiModalProjector
```

Python 实际导入路径决定使用哪套实现，加载日志会打印 native_source。推理继续使用现有 minicpmo_5_0_thinker_only pipeline ID，其模型根据 audio_config.model_type 选择 Huanyu；verl 的 rollout 注册名另设为 qwen3_6_huanyu。

训练别名仅用于 verl 注册表，**不改写 checkpoint 的 HF architectures**，不覆盖 Whisper/RM 的原注册项。model.external_lib 显式导入本目录，不需要修改 pyproject.toml。

## 服务器命令

将本目录完整同步到 gzq 副本。环境加载与各节点 Ray 启动仍由操作者完成；所有节点需要看到相同插件及包含 Huanyu 的 vLLM-Omni。

在**每个节点**检查原生模块（特别是本次报错的 `172.16.3.168`）：

```bash
PYTHONPATH="/vllm-workspace/vllm-omni:/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq${PYTHONPATH:+:$PYTHONPATH}" \
python3 -B /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py \
  --check --expected-root /vllm-workspace/vllm-omni
```

成功时输出 `[huanyu-runtime] OK` 和解析符号链接后的实际源码位置。预检只导入依赖，不加载 checkpoint 权重。若失败，按后面的故障记录处理。

先运行低成本接口检查（不启动 Ray、不读取模型 tensor、不占 NPU；不是 encoder 数值验收）：

```bash
cd /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq
python3 -B verl_omni_ext/models/qwen3_6_huanyu/check_runtime.py \
  --model-path /opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens
```

训练入口：

```bash
bash /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/verl_omni_ext/models/qwen3_6_huanyu/run_32card.sh
```

入口固定使用上述 Huanyu checkpoint、原 35B OmniMix 的 train/validation parquet、Qwen3.5-35B-A3B RM，调用原 examples/qwen3_6_whisper/run_qwen36_whisper_32card.sh。其余 batch、采样、长度和奖励参数沿用既有后端，可通过原有环境变量/CLI 参数控制。训练日志仍由原后端写入 run_qwen36_whisper_gspo_npu.log。

核心覆盖如下，入口已包含：

```text
actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu
actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining
actor_rollout_ref.model.model_stage=thinker
actor_rollout_ref.model.use_remove_padding=false
data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset
data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu
data.filter_overlong_prompts=false
MINICPMO_USE_PRECOMPUTE=0
MINICPMO_TUNE_AUDIO=0
HUANYU_VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
```

入口还会把包含指定 vLLM-Omni 和 gzq 插件目录的 `PYTHONPATH`，通过 `++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH` 显式传入 Ray；同样传递 `HUANYU_VLLM_OMNI_ROOT` 供 worker 检查。这是新任务的启动环境，不会修改已经运行的 worker，也不会自动启停 Ray。

不需要重新生成 parquet。本次使用在线编码，不能复用 Whisper 音频 cache。共享脚本可能仍打印原 cache 路径，但本目录 dataset 不访问该缓存。关闭长度预筛只跳过全量预扫描，真实请求仍受上下文长度限制。

## 2026-10-19：TaskRunner 找不到 huanyu_audio

日志中的异常是：

```text
TaskRunnerV1 host=172.16.3.168 pid=1142861
qwen3_6_huanyu/thinker_adapter.py → build_huanyu_processor
ModuleNotFoundError:
vllm_omni.model_executor.models.minicpmo_5_0.huanyu_audio
```

调用栈证明 Huanyu 训练 adapter 已生效。故障位于 processor 初始化的 Python 导入阶段，尚未开始音频网络权重加载。它与上一轮“Whisper 结构加载 Huanyu 权重”的错误不同。

该日志没有打印 worker 实际导入路径，因此只能确认该进程找不到原生模块，不能直接认定它加载了某个具体旧目录。主要需区分：Ray worker 导入了不含 Huanyu 的旧版本；或该 worker 所在节点的正确目录本身缺少文件/挂载。

本次修复全部位于本目录：

1. 启动器置顶 `/vllm-workspace/vllm-omni` 与 gzq 插件目录，清除已知旧 vLLM-Omni 路径，并在启动训练前执行真实导入检查。
2. 通过 Ray `runtime_env.env_vars` 显式传递导入路径。已有 Ray 集群启动时的环境与当前 shell 环境可能不同，仅在主节点 `find_spec` 成功不足以验证 worker。
3. processor 与 encoder 共用原生模块导入检查。失败时报告主机/PID、实际包与模块来源、模型包搜索路径、预期文件是否存在、缺失模块名以及 PYTHONPATH，保留原异常链。已经加载了另一份代码时直接拒绝，不在进程内替换 `sys.modules` 或混用两个版本。

如何读取新错误：

| 输出 | 处理 |
|---|---|
| `source mismatch`，实际来源不属于指定目录 | 检查新任务的 `ray init kwargs` 是否包含上述两个环境变量；检查服务器额外的环境重写代码，然后重新启动训练任务 |
| `expected_native_file=... exists=False` | 在报错节点核对 `/vllm-workspace/vllm-omni` 的链接/挂载，部署完整且一致的 Huanyu 版 vLLM-Omni；单改 ext 导入路径不能补出缺失文件 |
| 文件存在但仍无法导入 | 结合 `vllm_omni_origin` 与 `model_package_paths` 检查包搜索路径，保留原异常链继续定位 |
| `missing_module` 为其他依赖名 | 安装/校正该版本要求的依赖，不误认为缺少 Huanyu 模型代码 |
| `runtime API mismatch` | 当前 Huanyu 模块缺少适配器所需接口，需核对完整版本，不能使用旧 Whisper 组件代替 |

不要只复制一个 `huanyu_audio.py` 到旧推理仓库：推理模型的 Huanyu 分支及 processor 分派也必须同版本。`/vllm-workspace/vllm-omni` 可指向此前确认的 `/opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit`，检查按解析后的真实路径比较。

本地未提供服务器 V1 的 `verl.trainer.main_ppo.run_ppo` 及额外 `worker_env.py`，无法本地证明服务器是否再次覆盖 Ray 环境。最终以新任务打印的配置和 worker 导入结果验收；本地没有执行 NPU 训练。

## 2026-10-19：安装 rollout hook 时没有 _preprocess_input

新日志已越过原生 Huanyu 导入和 `build_huanyu_processor()`，随后在 `configure_processor()` 安装推理输入 hook 时失败：

```text
TaskRunnerV1 pid=1390206 ip=172.16.3.226
rollout_inputs.py → _install_on_server_class(vLLMOmniHttpServer)
AttributeError: type object 'vLLMOmniHttpServer' has no attribute '_preprocess_input'
```

问题在本插件对 server 接口的假设。旧实现直接读取 `_preprocess_input`，只兼容本地所见版本；新日志中的实际导入对象没有该属性。错误发生在 trainer 初始化阶段，尚未产生 rollout，不能解释为 Huanyu encoder 权重或音频特征出错。

本次只修改 Huanyu 目录里的 hook、测试和本说明：

- 若导出对象是 Ray ActorClass，先取其 `__ray_metadata__.modified_class`，对实际 Python 类安装 hook。
- 有同步 `_preprocess_input(self, prompt_ids, ..., multi_modal_data, ...)` 时沿用此入口。
- 没有独立预处理方法时，支持带 `prompt_ids/image_data/audio_data` 的异步 `generate` 入口：从该请求的媒体参数构造元数据，收缩 prompt 后 `await` 原方法。请求 ID、采样参数、媒体对象和其他参数继续传给原方法；不重写引擎生成逻辑。
- 只选择一个入口，重复安装不会二次包裹；其他模型仍调用原方法。音频条目与 30 秒 chunk 的校验继续保留，不通过跳过 hook 或删除媒体检查绕过报错。
- 方法缺失或签名不匹配时明确报出不支持的接口，不能把“安装函数返回了”当作适配成功。
- 包装器保留 Huanyu 模块内的局部函数身份，不复制原 server 方法的 `__module__/__qualname__`，避免序列化时只记录服务器原方法路径、远端恢复成未安装 hook 的方法；签名与原方法调用关系仍保留。

安装成功的日志包含 `Huanyu rollout input hook installed`、实际方法名（`_preprocess_input` 或 `generate`）和源码路径。服务器尚需核实真实方法实现并重跑验证，不能仅凭 AttributeError 判断一定是 Ray 包装导致。

同步更新后的 `rollout_inputs.py` 到所有节点的 `verl_omni_ext/models/qwen3_6_huanyu/`，继续使用同一个 `run_32card.sh`。无需修改训练参数、重算 cache 或重建 parquet。

## 2026-10-19：generate 被 autoregressive server mode 检查拦截

15:18 的日志已进入 `SingleTurnAgentLoop → server_manager.generate → vLLMOmniHttpServer.generate`，随后在 Huanyu 包装器内报错：

```text
rollout_inputs.py → contract_arguments
RuntimeError: Huanyu rollout requires the autoregressive server mode
```

这说明异步 `generate` hook 已安装并被远端调用。本次失败在调用原始生成方法之前，不是 encoder 算子、模型生成或 RM 打分报错。调用栈中的 `qwen3_6_whisper/rollout_sampling.py` 是复用的采样包装器，仅凭这个路径不能认定模型又走回 Whisper 音频适配器。

旧检查 `getattr(self, "_ar_mode", False)` 将“私有属性不存在”和“显式 False”合并为同一个错误。该日志没有打印属性值，因此不能据此断言引擎实际使用了非 AR 模式；此前启动参数已包含 `engine_kwargs.vllm_omni.output_mode=ar`，但仍须在报错 server 进程中确认。

随后提供的服务器源码搜索结果补齐了版本差异：`vllm_omni_async_server.py:70` 使用 `strategy_cls = ARStrategy if omni_kwargs.get("output_mode", "diffusion") == "ar" else DiffusionStrategy`，整个文件未找到 `_ar_mode`。该版本改为策略类分派，插件继续依赖旧私有字段是不兼容的。应按实际 rollout 配置确认 AR，而不是把缺少旧字段解释为配置选中了 diffusion；原启动脚本中的 `output_mode=ar` 无需更改。

修复只在本目录：

- 同时读取 server 的 `_ar_mode` 与 `self.config.engine_kwargs.vllm_omni.output_mode`，配置支持字典和属性对象。
- `_ar_mode` 缺失且配置明确为 `ar` 时允许继续；旧版 server 的 `_ar_mode=True` 且配置未设置时仍兼容。
- 显式 `_ar_mode=False`、非 AR 配置、两者冲突或没有模式依据时仍拒绝。不会通过设置 `_ar_mode=True`、修改配置或移除检查强行切换 server。
- 通过时每个 server 仅打印一次 `Huanyu rollout mode verified`，包含两项实际值；失败信息也包含两项值，以区分接口缺失与真正的模式冲突。媒体条目与占位符校验保持不变，其他模型不经过该检查。

同步新版 `rollout_inputs.py` 到两节点当前 ext 目录后，启动新的训练任务即可，不修改原训练参数、parquet 或 cache。若仍报错，请保留完整的新错误行：`_ar_mode=False` 表示需要核查实际 server 初始化分支；`<missing>` 配合配置缺失表示该服务器版本的配置保存接口仍需核实。不要再用旧错误推断具体字段值。

## 验证与边界

```bash
python3 -B -m unittest discover \
  -s verl_omni_ext/models/qwen3_6_huanyu/tests -p 'test_*_on_cpu.py' -v
bash -n verl_omni_ext/models/qwen3_6_huanyu/run_32card.sh
```

本次本地验证：91 项 CPU 测试全部通过，启动脚本此前的 Bash 语法检查通过且本轮未改动。本轮新增 5 项模式检查回归测试，覆盖缺少私有字段时根据 AR 配置处理图像及 31 秒音频、字典/属性混合配置、异步 generate、缺失证据、模式冲突和 RM 透传；同时验证不修改输入、配置或实际模式。此前 8 项接口回归测试覆盖异步 generate 的真实执行、纯文本、异常传播、Ray 类解包、入口优先级与不支持接口的拒绝，并检查包装器保留局部身份与原签名。另有 11 项测试验证原生模块缺失、内部依赖缺失、错误包来源、文件存在但搜索路径错误以及接口缺失。原有测试覆盖严格权重加载、FP32 Actor 主参数不被 BF16 临时视图修改、真实长度与占位符、混合 batch/左 padding、文本梯度、长音频推理条目数、独立注册和其他模型的 hook 分流。真实模型依赖使用测试替身，不能据此证明 NPU 上数值一致。

Windows 本地沙箱中的 asyncio 测试初始化曾阻塞，允许本地 IPC 后全部 86 项测试完成；这不是服务器训练测试。未在本地安装 Ray/cloudpickle，真实 Ray 远端序列化与服务器接口仍需服务器验收。

当前只支持冻结 audio encoder/projector，采用 padded [B,S] 前向（use_remove_padding=false），可能比 packed 路径占用更多显存。音频计算保持原生 VGG FP32、Conformer BF16；参数仍未解分片或音频模块被独立 FSDP 包装时明确报错，不能在不同 rank 上临时发起不对称 collective 来绕过。

服务器验收顺序：接口检查通过；Huanyu 严格加载成功；纯文本、图像、短音频、跨 30 秒音频和视频样本完成 rollout/log_probs/loss；再对同一音频比较训练与推理的 token 数、embedding 和 log_probs。若真实服务器接口不同，保留原始错误继续在本目录适配，不能用补零、裁剪或 strict=False 掩盖。
