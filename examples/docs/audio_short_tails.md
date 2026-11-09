# 音频短尾处理（40 维 Huanyu RL）

训练 processor、离线 precompute 和 RL rollout 请求统一使用
`audio_chunking.py` 的 `right_zero_pad_full_token_v1` 策略。30 秒分块边界保持不变；
非空末块若不足一个 token，保留全部采样点并在波形右侧补零至完整的 encoder/projector
帧组。当前 16 kHz、FFT400、hop160、encoder /4、projector /2 下，补至 **1520 个采样点、
8 帧、1 个 token**。不足一个 token 的整段短音频也采用相同策略；空音频、NaN/Inf、
非单声道和错误采样率仍报错。原本已有 token 的短块（例如 1040、1600 个采样点）不改变。

例如原始 960749 个采样点分成 `480000 + 480000 + 749`，原先最后一块只有 3 帧、
0 token，processor 抛异常后上游长度过滤会丢掉整条 QA。现在仅尾部补 771 个零，
处理后的帧数为 `2998, 2998, 8`，token 数为 `375, 375, 1`。占位符、训练 feature
长度和实际交给原生 rollout 的波形使用相同处理；输入数组不原地修改。这里的有效长度
包含主动补入的静音，不包含 feature 的 batch padding。`[huanyu-audio-padding]`
日志记录原始长度、补零数量和末块帧/token 数。

此变更会使旧 embedding cache 的 manifest 校验失败。使用**新的缓存目录**重新运行
precompute；不要只修改旧 manifest。外部 `filter_audit.py` 的失败缓存不在本仓库，
也必须更换审计缓存命名空间并重新过滤。恢复音频处理后仍按完整 prompt 长度过滤，
不能据此保证历史 606 条失败 QA 全部保留。新 worker 才会加载修改后的 Python 代码。

此修复覆盖使用 `HuanyuProcessor` 和 Huanyu RL server hook 的链路；没有修改外部
vLLM-Omni checkout，也没有修改独立 native serve 或旧 128 维 frontend。
CPU 回归验证分块、占位符、实际请求 payload 和缓存契约，不能代替真实编码器前向。
部署后可在空闲 NPU 上用下面的音频模块检查验证 8 帧可执行；它只加载音频权重，
不启动 Ray、不加载语言模型。`MODEL_PATH` 必须指向本次训练的 40 维 checkpoint。

```bash
MODEL_PATH=/opt/huawei/dataset/model_gy/downloaded_models/2748 python3 - <<'PY'
import importlib, json, os, sys, types
from pathlib import Path
import numpy as np
import torch
import torch_npu

root = Path(os.environ['PROJECT_ROOT']) / 'verl_omni_ext/models/qwen3_6_huanyu'
pkg = types.ModuleType('_huanyu_tail_check'); pkg.__path__ = [str(root)]
sys.modules[pkg.__name__] = pkg
runtime = importlib.import_module(pkg.__name__ + '.audio_runtime')
chunking = importlib.import_module(pkg.__name__ + '.audio_chunking')
checkpoint = Path(os.environ['MODEL_PATH'])
cfg = json.loads((checkpoint / 'config.json').read_text())
front = json.loads((checkpoint / 'audio_preprocessor_config.json').read_text())
assert cfg['audio_config']['input_feat_per_channel'] == front['feature_size'] == 40
assert front['sampling_rate'] == 16000
host = torch.nn.Module(); host.model = torch.nn.Module(); host.config = cfg
runtime.attach_huanyu_audio(host, {'path': str(checkpoint)})
assert host.model.audio_encoder.fwd_downsample == 4
assert host.model._huanyu_audio_runtime['downsample_rate'] == 2
torch.npu.set_device(0); host.to('npu:0')
native = runtime.load_huanyu_runtime()
fft, hop = front.get('n_fft', 400), front.get('hop_length', 160)
extractor = native.HuanyuFbankFeatureExtractor(
    feature_size=40, sampling_rate=16000, n_fft=fft, hop_length=hop)
for size in (1, 6, 58, 749, 833, 1039, 1040, 1600):
    wav, plan = chunking.prepare_audio_waveform(np.ones(size, np.float32) * .01,
                                              n_fft=fft, hop_length=hop)
    feature, length = extractor.extract_fbank(wav)
    assert length == plan[0].frames
    output = runtime.encode_huanyu_audio(host, feature.unsqueeze(0), [length])[0]
    assert output.shape[0] == plan[0].tokens and torch.isfinite(output).all()
    print(f'PASS original={size} processed={len(wav)} frames={length} tokens={len(output)}')
PY
```
