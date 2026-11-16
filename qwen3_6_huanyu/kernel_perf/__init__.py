"""Huanyu NPU 融合 kernel 包（GDN fwd/bwd + npu_rms_norm），eager 模式专用。

设计约束：
- CPU import 契约——import 本包无任何副作用，不 import torch_npu/triton/vllm；
  所有安装动作都通过显式函数调用完成（actor_kernels / rollout_kernels）。
- 模型无关——gdn_kernels / rms_norm 只接受显式模块与张量，qwen3_5 系列
  （qwen3_5_moe / qwen3_5_dense）可直接复用；huanyu 专名只出现在
  actor_kernels.install_actor_kernels 与 rollout_kernels 的接线层。
- 与 qwen_precision 的 FP32 参考实现互斥（同一批包装点，二选一）。
"""

__all__ = [
    "actor_kernels",
    "gdn_kernels",
    "rms_norm",
    "rollout_kernels",
]
