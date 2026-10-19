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
| processor.py | 40 维 fbank、Qwen 图像处理、媒体 token 展开、原生 HF mRoPE、AgentLoop 输入桥接及可选严格 cache 读取 |
| dataset.py | 读取原始图片、音频、视频帧；不读取 Whisper cache |
| forward.py | 保留 HF 图像/文本前向，在经过验证的音频 token 位置注入 embedding |
| rollout_inputs.py | 按媒体元数据收缩推理占位符，兼容独立预处理方法及内联预处理的 async generate 接口 |
| rollout_adapter.py | 独立注册 qwen3_6_huanyu，复用现有单 llm stage 拓扑 |
| run_32card.sh | 调用原 32 卡入口，追加 Huanyu 配置、关闭长度预筛，显式传递 Ray 导入路径 |
| check_runtime.py | 不加载模型权重的真实依赖/processor/mRoPE CPU 接口检查 |
| precompute.py / cache.py | Huanyu 专用音频、图片和视频帧 embedding 预计算、内容键与模型身份校验 |

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

入口固定使用上述 Huanyu checkpoint、原 35B OmniMix 的 train/validation parquet、Qwen3.5-35B-A3B RM，调用原 examples/qwen3_6_whisper/run_qwen36_whisper_32card.sh。batch、采样和奖励参数沿用既有后端，可通过原有环境变量/CLI 参数控制。长度预算由本目录入口统一设置，使用下文列出的环境变量调整，避免共享 DEBUG 分支覆盖。训练日志仍由原后端写入 run_qwen36_whisper_gspo_npu.log。

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

不需要重新生成 parquet。默认使用在线编码；可按下节生成并启用 Huanyu 专用 cache，不能复用 Whisper cache。共享脚本可能仍打印原 cache 路径，但本目录不访问旧缓存。关闭长度预筛只跳过全量预扫描，真实请求仍受上下文长度限制。

## Huanyu 预计算原 35B OmniMix 数据集

需先将本目录全部更新同步到服务器。环境加载与 Ray 启停仍由操作者负责。以下预计算命令不使用 Ray，只使用执行节点的一张 NPU（`npu:0`），应在该卡空闲时运行。原始 parquet、原始媒体及所有旧 Whisper/MiniCPM cache 均不修改。

```bash
cd /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq
mkdir -p /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/logs/huanyu_precompute

(
set -o pipefail
python3 -u /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/verl_omni_ext/models/qwen3_6_huanyu/precompute.py \
  --parquet \
    /opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/train.parquet \
    /opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/validation.parquet \
  --model_path /opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens \
  --cache_dir /opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128 \
  --device npu:0 \
  --sampling_rate 16000 \
  --read-batch-size 16 \
  --log-interval 10 \
  2>&1 | tee -a /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/logs/huanyu_precompute/precompute.log
)
```

一条命令先处理 train，再处理 validation，共用同一个 cache。也可以将 `--parquet` 后分别只保留其中一个文件，分两次运行。成功以最后的 `[huanyu-precompute] COMPLETE` 为准；中途异常带 parquet 路径和行号。每约 10 秒（在一条样本完成后）输出累计行数、音频 chunk 数、图片数、已创建/复用条目数、速率和估算剩余时间。

`--read-batch-size` 是 parquet 的读取批大小，不是 NPU 推理批大小。当前为每个音频 chunk、每张图片分别编码，限制峰值显存，并使音频计算不受同批其他样本 padding 的影响；没有旧脚本的 `--batch_size`、`--num_workers`、`--max_slice_nums` 选项。视频使用与训练相同的采样 helper，采样帧计入 vision cache，启用视频音轨时也处理音轨。预计算和训练的 `MINICPMO_VIDEO_*`、`MINICPMO_USE_AUDIO_IN_VIDEO` 等解码设置必须一致；否则内容键可能不命中并明确报错。

缓存保存的是音频 fold/projector 和视觉 merger 后的 BF16 `[tokens, 2048]`，不是 fbank 或原始像素。预计算不加载 35B 文本主干。音频编码复用服务器 Huanyu 实现，视觉编码使用当前训练侧 HF 类与严格匹配的 `model.visual.*` 权重。视觉 eager attention 每块最多计算 128 个 query，每块仍使用完整 K/V；不缩图、不截断视觉 token。非空 DeepStack 配置不支持并报错。

当前预计算使用新的 `_q128` 目录，避免与旧版完整矩阵 eager 缓存混用。需要同步本目录的 `precompute.py`、`vision_attention.py` 和 `cache.py`；训练所有节点也需更新以读取相同 contract。原 `embed_cache_huanyu_sampleall` 目录保留，不删除、不修改 manifest。新目录生成后，再次运行相同命令仍会校验并复用其完整条目。

新目录结构为 `manifest.json`、`audio/<内容哈希>.pt`、`vision/<内容哈希>.pt`。键来自真实 fbank/像素及 grid，图片绝对路径变化本身不会造成旧式路径键错配。manifest 校验 checkpoint 路径、配置、权重分片元数据/头部、前处理设置及运行时源码版本；不把整个 35B 权重读入内存做哈希。checkpoint 必须保持不可变，多节点挂载/复制需保持一致的分片元数据。缓存不兼容时需使用新目录，不能手改 manifest 绕过校验。

重复执行相同命令会校验并复用已有完整条目；损坏条目报错，不覆盖。不将没有 Huanyu manifest 的旧 cache 目录接管为新缓存。发布使用同目录临时文件、目录锁及 rename，不要求支持硬链接；文件系统需支持原子 mkdir 和同目录 rename。所有并发预计算进程必须使用此更新后的发布协议。

若仅要复核完整性，给上述 python 命令添加 `--verify-only`（可去掉日志管道）。它重新扫描两个 parquet 并检查每个需要的条目，不启动 encoder、不写入 cache。这是覆盖和完整性检查，不是 NPU 数值验收。

**缓存生成完成后，训练时必须显式启用读取：**

```bash
HUANYU_USE_PRECOMPUTE=1 \
HUANYU_EMBED_CACHE_DIR=/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall_q128 \
bash /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/verl_omni_ext/models/qwen3_6_huanyu/run_32card.sh
```

入口把这两个变量传入新 Ray 任务；旧 `MINICPMO_USE_PRECOMPUTE` 保持 0。训练侧应打印 `[huanyu-cache] strict audio+vision cache reader`，模型适配器日志应为 `cache=strict`。processor 保留媒体 token/grid 和 RoPE，读 cache 替代训练侧 encoder 输入；forward 校验逐样本数量、位置和长度后注入 embedding，跳过音频/视觉编码。缺项、版本不匹配或未传入缓存标记即失败，不在部分 rank 上在线回退。rollout 仍携带原始媒体，推理侧仍自行编码；缓存也不消除 LLM/FSDP 本身的通信。

Huanyu 在线音频现在也逐 chunk 编码并按真实长度独立 padding，与离线输入形状一致。未改变 encoder/projector 网络或权重。CPU 回归验证缓存读写、缺项/损坏拒绝、混合 batch 插入、梯度与 checkpoint 重算；真实 checkpoint 的 NPU 数值一致性和分布式性能仍需服务器小样本验证。不能仅凭 cache 已生成就断言之前的 AllGather 超时已解决。

预计算功能初次发布时的本地验证：`python -B -m unittest discover -s verl_omni_ext/models/qwen3_6_huanyu/tests -p 'test_*_on_cpu.py' -q`，133 项通过；`precompute.py --help` 和 `bash -n run_32card.sh` 通过。非重入 checkpoint 测试仅绕过编译装饰器，实际执行反向重算。Huanyu 目录外的 79 个插件文件哈希与修改前一致。

## 2026-10-19：预计算在发布 manifest 时被挂载拒绝

日志为 `cache.py → _publish → os.link(temporary, manifest.json)` 抛出 `PermissionError: [Errno 1] Operation not permitted`。临时文件创建、写入和 fsync 已完成，失败的是硬链接操作，说明该挂载拒绝此操作；不能直接据此认定整个目录没有写权限。此时尚未开始样本编码，与 Huanyu 网络、通信超时或 `_version.py` 警告无关。

修复限定于本目录的缓存发布逻辑：先完整写入临时文件并 fsync，再用 `.huanyu-<目标文件名>.lock` 目录锁串行化检查和 rename。锁内再次确认目标不存在，已有条目由原有完整性校验处理，不覆盖。manifest 的旧目录接管检查也放在同一把锁内，避免多个初始化进程互相误判。manifest 和所有 audio/vision 条目共用这一机制，均不再调用 `os.link`。

更新服务器 `verl_omni_ext/models/qwen3_6_huanyu/cache.py` 后，直接重跑上面的完整预计算命令，继续使用 `/opt/huawei/dataset/ccb_data/data/omni/embed_cache_huanyu_sampleall`。本修复不改变缓存格式、内容键或 contract；本次失败留下的空目录可直接使用，不必删除目录、改路径或重新转换 parquet。如果存在之前成功写出的完整条目，会按原规则校验后复用。

目录锁最多等待 30 秒；异常退出遗留的锁不会被自动删除，以免打断另一个节点正在发布的缓存。若报 `publication lock timed out`，日志会包含具体锁路径，应先确认所有节点均没有使用该锁的写入进程，再处理该锁；不要删除整个缓存目录。并发进程须统一更新，目录锁不保护不遵守此协议的旧程序或外部写入者。

附件末尾的 `bash: syntax error near unexpected token ')'` 是另一个问题：只粘贴了命令尾部的 `)`，没有对应的开头 `(`。复制本节完整命令块时保留成对括号；仅执行 `python3 ... | tee ...` 时不要额外追加 `)`。第二次执行已经没有此语法错误，但仍因硬链接失败，二者无因果关系。

本次修复新增 9 项 CPU 回归，覆盖硬链接被禁用、并发发布/初始化、已有目标保护及异常清理；完整 Huanyu CPU 测试共 142 项通过。测试使用本地文件系统和故障注入，服务器实际挂载及 NPU 预计算仍需重跑验证。

## 2026-10-19：第 9 条样本的视觉 eager attention OOM

新日志已出现 `mode=populate rows=124016`、`created=1` 和音频 `290/4 tensors; features=40`，说明 manifest 发布及音频权重加载已通过。真正失败在 `train.parquet row 8 (zero-based)`：`FrozenEncoders.vision → Qwen3_5MoeVisionModel → eager_attention_forward → torch.matmul(query, key.T) * scaling`，单次申请 31.13 GiB，设备当时仅余 24.31 GiB。不是 Huanyu 音频网络加载失败，也不是 Ray/HCCL 通信超时。`33.60 GiB already allocated` 包含运行时中间张量，不能解释成预计算加载了 35B 文本主干。

原脚本虽然每次只编码一张图片，但 eager 会构造形如 `[batch, heads, S, S]` 的注意力分数，缩放和 FP32 softmax 还需额外空间。`--read-batch-size 16` 控制 CPU parquet 读取；调成 1 仍不能消除单张大图的平方矩阵。该 HF attention 的接口和分段行为可参照 [Transformers Qwen3.5 MoE 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)，服务器的具体失败点以本次 traceback 为准。

修复在本目录新增 `vision_attention.py`：仅在离线视觉前向期间拦截该模型实例的 HF eager 调用，将 query 轴按 128 分块；每块仍对完整 K/V 做原实现的缩放、FP32 softmax 和 BF16 输出。不切 key 轴，不将图片裁成互相隔离的小图，也不改变 HF 已有的图片/视频帧 `cu_seqlens` 边界。逐块释放 attention weights，按原顺序拼回 query 输出。HF 原函数在异常时也会恢复，不安装到训练或 rollout 的模型前向。

guard 要求 `eval + no_grad + dropout=0 + noncausal`，并验证输出布局、dtype 和 device；关闭 attention/hidden-state 收集。若实际 HF 前向未经过 hook，则报错，不保存未经验证的输出。正常日志包含 `attention=eager_query_chunked query_chunk=128`、图片 `grid/patches/merged_tokens` 和 `vision attention verified` 的调用数、最大 query 分块及完整 key 长度。只在首张图和较大的新尺寸记录这些信息，避免逐层刷屏。

分块保留完整 attention 数学语义与输入，但不同 NPU GEMM 形状不保证逐 bit 一致。manifest 因此新增 helper 源码哈希和执行标识 `single_image_hf_bf16_eager_q128_v1`，不把旧缓存冒充新执行结果。本节上方的预计算和训练命令已改用 `embed_cache_huanyu_sampleall_q128`。已有旧目录不受影响，不能手改旧 manifest 绕过检查。该方案降低 score/softmax 峰值显存，不降低全量 attention 的平方计算量；实际 NPU 数值、峰值内存和耗时仍需服务器验证。

本次本地完整 CPU 回归共 157 项通过。新增测试以 FP32/BF16 小张量对照完整 eager，并覆盖 query mask、GQA、完整 K/V、不同帧不交叉、异常恢复和 hook 未命中时拒绝保存。测试不代表已在服务器 NPU 上完成预计算。

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

## 2026-10-19：AgentLoop 缺少 mm_token_type_ids，以及展开 prompt 超长

本轮日志包含两个独立故障，不能只修其中一个：

1. 18 处 `Qwen3_5MoeModel.get_rope_index() missing 1 required positional argument: 'mm_token_type_ids'`。部分请求已完成 rollout，在 `AgentLoopWorkerTQ._agent_loop_postprocess → _compute_score → _compute_position_ids → HuanyuProcessor.get_rope_index` 失败。该调用栈中的 `_compute_score` 不等于 RM 模型故障，具体失败点是位置编码的接口适配。
2. 4 处 `Multimodal prompt produced 3107 tokens, exceeding rollout.prompt_length=3104`，发生在其他请求生成前的 `ct_build_initial_tokens`。关闭长度预筛不取消运行时容量限制。

日志中个别请求的 `[mm-merge] ok features=... placeholders=...` 只说明相应请求的数量对齐，不能证明整个训练链路或 encoder 数值已经正确。

### RoPE 参数修复

新 Qwen 接口要求形状为 `[B,S]` 的 `mm_token_type_ids`，文本为 0、图片为 1、视频为 2。旧 Huanyu 适配器只转发 AgentLoop 提供的参数，没有生成这个新增字段。本目录的 `processor.py` 现按已安装原生函数的签名处理：

- 旧版不要求该参数时，继续原调用。
- 新版缺少该参数或值为 None 时，从实际 `input_ids` 构造同形状、同设备的整数类型标记，兼容 padding；不从 decode 文本或原始 problem 猜位置。
- `image_pad` 为 1；原生 `video_token_id` 为 2。本目录的视频已拆为图片帧，实际走 image token 和 image grid，因此这些帧为 1。
- Huanyu 音频 `<unk>` 槽位、文本和媒体边界符为 0，padding 为 0。这里的 0 表示使用顺序位置编码，并不意味着丢弃音频：音频 embedding 仍由 Huanyu forward 在对应槽位注入。
- 调用方已经明确传入类型 tensor 时原样转发。原生 HF 函数继续负责图像三维位置及 delta；没有把整条多模态序列改成普通 arange，也没有改动 token、图片网格或 attention mask。

用户更正后提供的真实实现来自 `/usr/local/python3.12.13/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`，与报错环境一致。它除了要求上述模态标记，还会将 `video_grid_thw` 按时间维重复展开，并将每帧时间维置为 1，适配帧间时间戳；本补丁继续调用这份原生函数，保留该处理。先前误贴的 Python 3.11 环境不作为本次修复依据。

更新后的 `check_runtime.py` 会打印实际 Python 和 Transformers 路径/版本，并在训练环境中用图像加 1 秒/31 秒音频调用原生 RoPE。请在训练使用的容器里 source 对应环境后执行本文件前面的接口检查命令，无需为检查重新启动训练或加载模型权重。

### 长度预算修复

只调整本目录 `run_32card.sh`，长度预筛仍关闭；不删除样本、不截断媒体槽位。无环境变量覆盖时的默认值：

| 环境变量 | 默认值 | 最终对应配置 |
|---|---:|---|
| MAX_PROMPT_LENGTH | 4096 | data.max_prompt_length |
| ROLLOUT_PROMPT_LENGTH | 同 MAX_PROMPT_LENGTH | rollout.prompt_length |
| MAX_RESPONSE_LENGTH | 2048 | data.max_response_length、rollout.response_length |
| ROLLOUT_MAX_MODEL_LEN | prompt + response + 128，即 6272 | rollout.max_model_len、max_num_batched_tokens |
| MAX_TOKEN_LEN_PER_GPU | 同 ROLLOUT_MAX_MODEL_LEN | actor 训练及 actor/ref log_prob token 预算 |

入口保存这些值并作为最终 Hydra 覆盖传给后端，防止共享脚本的 DEBUG 分支重新把 prompt 设回 3072；不要再通过该入口的 CLI 参数覆盖上述长度键，应使用表中环境变量。已有 shell 若仍设置旧的 `ROLLOUT_PROMPT_LENGTH=3104`，会保留该显式值，需同时更新。

启动前校验正整数及预算关系：rollout prompt 不大于训练 prompt；model length 至少覆盖 rollout prompt + response；训练 token 预算至少覆盖训练 prompt + response。4096 能容纳本次已知 3107-token 请求，不保证未预筛的全量数据没有更长样本；过长时继续报出真实长度，不能通过截断破坏对齐。更长上下文可能增加显存占用，本轮不调整 batch、TP 或 dtype。

两节点同步本目录后，在已加载环境、Ray 已就绪的主节点运行（显式值避免继承旧 shell 的长度设置）：

```bash
MAX_PROMPT_LENGTH=4096 \
ROLLOUT_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=2048 \
ROLLOUT_MAX_MODEL_LEN=6272 \
MAX_TOKEN_LEN_PER_GPU=6272 \
bash /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/verl_omni_ext/models/qwen3_6_huanyu/run_32card.sh
```

如继续使用 4096-token response，将 `MAX_RESPONSE_LENGTH` 改为 4096，同时将 `ROLLOUT_MAX_MODEL_LEN` 和 `MAX_TOKEN_LEN_PER_GPU` 设为 8320。无需重建 parquet 或重算 cache。

## 验证与边界

```bash
python3 -B -m unittest discover \
  -s verl_omni_ext/models/qwen3_6_huanyu/tests -p 'test_*_on_cpu.py' -v
bash -n verl_omni_ext/models/qwen3_6_huanyu/run_32card.sh
```

本轮 97 项 CPU 测试全部通过，Bash 语法检查通过；用外部命令替身执行启动器的默认、DEBUG、response=4096、prompt=5120、非法模型预算和非法训练预算六组检查均通过，未启动训练。新增 6 项 RoPE 回归测试覆盖必填模态参数、图像/音频混合序列、左右 padding、纯文本、显式参数透传、None 补齐、视频类型及无效 mask；旧接口与原生位置精度检查保留。

另在内存中直接执行用户更正后的服务器 `get_rope_index` 函数，检查图片三轴位置、图像加音频及 padding、2055-token 纯文本、纯音频、带时间戳分隔的两帧视频，共五组均通过。该附件未包含 `get_vision_position_ids`，此项辅助函数使用空间网格测试替身；生产适配器绑定的是安装版本自己的 helper，因此这次检查不等于完成服务器完整 Transformers/NPU 数值验收。

此前 91 项测试覆盖 AR 模式检查、异步 generate、Ray 类解包、原生导入来源、严格权重加载、FP32 Actor 主参数、真实长度与占位符、文本梯度、长音频和其他模型的 hook 分流。真实模型依赖使用测试替身，不能据此证明 NPU 上数值一致。

Windows 本地沙箱中的 asyncio 测试初始化曾阻塞，允许本地 IPC 后全部 86 项测试完成；这不是服务器训练测试。未在本地安装 Ray/cloudpickle，真实 Ray 远端序列化与服务器接口仍需服务器验收。

当前只支持冻结 audio encoder/projector，采用 padded [B,S] 前向（use_remove_padding=false），可能比 packed 路径占用更多显存。音频计算保持原生 VGG FP32、Conformer BF16；参数仍未解分片或音频模块被独立 FSDP 包装时明确报错，不能在不同 rank 上临时发起不对称 collective 来绕过。

服务器验收顺序：接口检查通过；Huanyu 严格加载成功；纯文本、图像、短音频、跨 30 秒音频和视频样本完成 rollout/log_probs/loss；再对同一音频比较训练与推理的 token 数、embedding 和 log_probs。若真实服务器接口不同，保留原始错误继续在本目录适配，不能用补零、裁剪或 strict=False 掩盖。
