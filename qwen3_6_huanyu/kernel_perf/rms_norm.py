"""torch_npu.npu_rms_norm 融合归一化包装（训练侧），模型无关可复用。

替换对象是按类名匹配的 HF RMSNorm 实例（默认 ``Qwen3_5MoeRMSNorm`` 与
``Qwen3_5MoeRMSNormGated``），qwen3_5 系列模型类名一致，可直接复用。

精度边界与仓内 ``qwen_precision/norm_precision.py`` 的既有参考实现完全对齐：
归一化与权重乘法都在 FP32 域完成（``npu_rms_norm(x.float(), gamma.float())``），
仅在出口处 cast 回输入 dtype。普通 RMSNorm 遵循 HF 的 (1+weight) 零初始化
约定；Gated 变体复用 npu_rms_norm 完成 norm+weight 后再乘 silu(gate)。

反向走显式 ``torch_npu.npu_rms_norm_backward(dy, x, gamma, rstd)``，避免依赖
torch_npu 是否为该算子注册了自动微分。

CPU import 契约：模块级只 import torch；torch_npu 延迟到安装/执行时导入。
"""

from types import MethodType

import torch
import torch.nn.functional as F


class _NpuRMSNormFn(torch.autograd.Function):
    """FP32 域 npu_rms_norm 前向 + 显式 npu_rms_norm_backward 反向。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float):
        import torch_npu

        with torch.autocast(device_type=x.device.type, enabled=False):
            out, rstd = torch_npu.npu_rms_norm(x.float(), gamma.float(), epsilon=eps)
        ctx.save_for_backward(x, gamma, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        import torch_npu

        x, gamma, rstd = ctx.saved_tensors
        with torch.autocast(device_type=x.device.type, enabled=False):
            dx, dgamma = torch_npu.npu_rms_norm_backward(
                dy.float().contiguous(), x.float(), gamma.float(), rstd
            )
        return dx, dgamma, None


def npu_rms_norm_fp32(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 归一化 + 权重乘法，返回 dtype 与 x 一致（对齐 HF 出口 cast 边界）。"""
    return _NpuRMSNormFn.apply(x, gamma, eps).type_as(x)


def _npu_forward(self, x):
    # 对齐 HF Qwen3_5MoeRMSNorm：FP32 norm，(1+weight) 约定，出口 cast。
    return npu_rms_norm_fp32(x, 1.0 + self.weight.float(), self.eps)


def _npu_gated_forward(self, hidden_states, gate=None):
    # 对齐 HF Qwen3_5MoeRMSNormGated：norm → weight → silu(gate) 逐段边界。
    weighted = _NpuRMSNormFn.apply(hidden_states, self.weight, self.variance_epsilon)
    if gate is not None:
        weighted = weighted * F.silu(gate.float())
    return weighted.to(hidden_states.dtype)


def install_npu_rms_norm(module, class_names=("Qwen3_5MoeRMSNorm", "Qwen3_5MoeRMSNormGated")) -> int:
    """把 module 子树中指定类名的 norm 实例前向替换为 npu_rms_norm 融合实现。

    幂等：已包装的实例跳过。返回新包装的实例数；一个都没匹配到时返回 0
    （调用方决定是否需要 raise）。
    """
    import torch_npu  # noqa: F401  # 安装期 fail-fast：无 torch_npu 即环境错误

    if not hasattr(torch_npu, "npu_rms_norm") or not hasattr(torch_npu, "npu_rms_norm_backward"):
        raise RuntimeError("kernel 档案需要 torch_npu.npu_rms_norm / npu_rms_norm_backward")
    count = 0
    for child in module.modules():
        if getattr(child, "_huanyu_npu_rms_norm", False):
            continue
        name = type(child).__name__
        if name == "Qwen3_5MoeRMSNorm" and "Qwen3_5MoeRMSNorm" in class_names:
            child.forward = MethodType(_npu_forward, child)
            child._huanyu_npu_rms_norm = True
            count += 1
        elif name == "Qwen3_5MoeRMSNormGated" and "Qwen3_5MoeRMSNormGated" in class_names:
            child.forward = MethodType(_npu_gated_forward, child)
            child._huanyu_npu_rms_norm = True
            count += 1
    return count
