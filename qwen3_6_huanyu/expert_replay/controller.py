"""R3 路由回放的 batch / micro-batch 级 targets 注册表（纯 torch，CPU 可测）。

生命周期对齐 verl veomni 引擎（``veomni/transformer_impl.py`` 的
forward_backward_batch）：

- ``begin_batch()`` / ``end_batch()`` 包住整个 ``forward_backward_batch``
  （含所有 micro-batch 的 forward **与 backward**）；
- 每个 micro-batch 前向前 ``arm()`` 覆盖为本 micro-batch 的 targets，
  **不在 micro-batch 之间 clear**：verl 循环顺序是 forward(i) → backward(i)
  → forward(i+1)，backward 同步执行，因此 gradient checkpointing 的反向重算
  看到的一定是第 i 个 micro-batch 的 targets。

状态是**进程全局**（不是 threading.local）：autograd 反向在设备线程上执行，
线程局部状态在重算时不可见，会静默退回原生路由。

无记录行的约定（vLLM 捕获 / verl 组包一致）：该行为全零。top-k 合法值互不
相同，重复值行（含全零）→ kernel 回退原生路由（pad 位、最后一个未前向的
生成 token、abort 样本）。
"""

import torch

# 每层每个 micro-batch 最多调用次数：一次前向 + 一次 checkpoint 重算。
MAX_CALLS_PER_LAYER = 2

_STATE = {
    "in_batch": False,
    "targets": None,   # [T, L, K] int64
    "fallback": None,  # [T, L] bool，True → 该行该层回退原生路由
    "generation": 0,
    "calls": {},       # layer → 本 generation 调用次数
}


def begin_batch() -> None:
    if _STATE["in_batch"]:
        raise RuntimeError("expert_replay: begin_batch() while a batch scope is active; missing end_batch()")
    _reset()
    _STATE["in_batch"] = True


def end_batch() -> None:
    _reset()


def _reset() -> None:
    _STATE.update(in_batch=False, targets=None, fallback=None, generation=0, calls={})


def in_batch() -> bool:
    return _STATE["in_batch"]


def arm(targets: torch.Tensor) -> None:
    """注册当前 micro-batch 的 targets（[T, L, K] int64），覆盖上一个 micro-batch。"""
    if not _STATE["in_batch"]:
        raise RuntimeError("expert_replay: arm() outside begin_batch()/end_batch() scope")
    if targets.dtype != torch.int64 or targets.dim() != 3:
        raise RuntimeError(
            f"expert_replay: targets must be int64 [T, L, K], got {targets.dtype} {tuple(targets.shape)}"
        )
    _STATE["targets"] = targets
    _STATE["fallback"] = has_duplicate_rows(targets)
    _STATE["generation"] += 1
    _STATE["calls"] = {}


def is_armed() -> bool:
    return _STATE["targets"] is not None


def num_layers() -> int:
    targets = _STATE["targets"]
    return 0 if targets is None else targets.shape[1]


def generation() -> int:
    return _STATE["generation"]


def layer_target(layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """取本层 (target [T, K], fallback [T, 1])，并记录调用次数。"""
    targets = _STATE["targets"]
    if targets is None:
        raise RuntimeError("expert_replay: layer_target() called while not armed")
    if not 0 <= layer_idx < targets.shape[1]:
        raise RuntimeError(
            f"expert_replay REPLAY: layer {layer_idx} outside armed layers [0, {targets.shape[1]}); "
            "routed_experts layer count disagrees with the model"
        )
    calls = _STATE["calls"]
    calls[layer_idx] = calls.get(layer_idx, 0) + 1
    if calls[layer_idx] > MAX_CALLS_PER_LAYER:
        raise RuntimeError(
            f"expert_replay: layer {layer_idx} called {calls[layer_idx]} times in one micro-batch "
            f"(max {MAX_CALLS_PER_LAYER}: forward + checkpoint recompute); targets are stale"
        )
    return targets[:, layer_idx, :], _STATE["fallback"][:, layer_idx].unsqueeze(-1)


def assert_all_layers_fired() -> None:
    """micro-batch 前向后：每个已 arm 的层都必须至少消费过一次 target。"""
    expected = num_layers()
    fired = sorted(_STATE["calls"])
    if fired != list(range(expected)):
        missing = sorted(set(range(expected)) - set(fired))
        raise RuntimeError(
            f"expert_replay REPLAY: {len(fired)}/{expected} MoE layers consumed targets "
            f"(missing {missing[:8]}); a MoE block bypassed the fused replay kernel"
        )


def has_duplicate_rows(target: torch.Tensor) -> torch.Tensor:
    """沿最后一维（top-k）判断是否有重复值（含全零行）；返回去掉该维的 bool。"""
    sorted_target, _ = target.sort(dim=-1)
    return (sorted_target[..., 1:] == sorted_target[..., :-1]).any(dim=-1)
