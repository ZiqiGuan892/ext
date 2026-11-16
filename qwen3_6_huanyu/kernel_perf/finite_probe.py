"""非有限值溯源探针（``HUANYU_FINITE_PROBE=1``，调试用，默认关）。

给子模块挂 forward hook，报告**第一个「输入全有限、输出含非有限」的模块**
——即 NaN/Inf 的产生点，而不是传播点。kernel 档案（融合 GDN + npu_rms_norm）
下训练侧 entropy/grad_norm 全 NaN，而 FP32 参考档案有限；逐 flag 重跑 16 卡
每次 40 分钟，所以改为一次运行直接定位算子。

- 只打印产生点（origin），传播点静默，避免刷屏；总条数由
  ``HUANYU_FINITE_PROBE_LIMIT`` 控制（默认 40）。
- 每个 rank 独立计数并带 pid（NaN 是 rank 相关的）。
- isfinite 全量扫描有开销，仅用于短跑调试，不进生产路径。

CPU import 契约：模块级只 import os/torch。
"""

import os

import torch

_STATE = {"left": None, "found": set()}


def _enabled() -> bool:
    return os.environ.get("HUANYU_FINITE_PROBE", "0") == "1"


def _tensors(value):
    """递归取出可检查的浮点张量。"""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _nonfinite_count(value) -> int:
    total = 0
    for tensor in _tensors(value):
        if tensor.numel() == 0:
            continue
        total += int((~torch.isfinite(tensor)).sum())
    return total


def _absmax(value) -> float:
    best = 0.0
    for tensor in _tensors(value):
        if tensor.numel() == 0:
            continue
        best = max(best, float(tensor.detach().abs().max()))
    return best


def _make_hook(path):
    def hook(module, args, kwargs, output):
        if _STATE["left"] is not None and _STATE["left"] <= 0:
            return
        bad_out = _nonfinite_count(output)
        if bad_out == 0:
            return
        bad_in = _nonfinite_count(args) + _nonfinite_count(kwargs)
        if bad_in > 0:
            return  # 传播点，不是产生点
        key = (path, os.getpid())
        if key in _STATE["found"]:
            return
        _STATE["found"].add(key)
        _STATE["left"] -= 1
        print(
            f"[huanyu-finite-origin] path={path} type={type(module).__name__} "
            f"nonfinite_out={bad_out} in_absmax={_absmax(args):.4g} pid={os.getpid()}",
            flush=True,
        )

    return hook


def install_finite_probe(module, limit=None) -> int:
    """给 module 子树挂非有限值溯源 hook。幂等；未开开关时返回 0。"""
    if not _enabled():
        return 0
    if limit is None:
        limit = int(os.environ.get("HUANYU_FINITE_PROBE_LIMIT", "40"))
    _STATE["left"] = limit
    count = 0
    for path, child in module.named_modules():
        if getattr(child, "_huanyu_finite_probe", False):
            continue
        child.register_forward_hook(_make_hook(path or "<root>"), with_kwargs=True)
        child._huanyu_finite_probe = True
        count += 1
    print(f"[huanyu-finite-origin] probe installed on {count} modules (limit={limit})", flush=True)
    return count
