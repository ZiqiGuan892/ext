"""R3 scheduler 注入（EngineCore 进程；零 vllm-omni 源码改动）。

omni ``OmniARScheduler.update_from_output`` 覆盖了上游实现，丢了两件事：
1. 从不 ``routed_experts_mgr.store_batch``（routing 没进槽位缓冲）；
2. 停时提取 ``omni_routed_experts_for_request`` 依赖 vllm 0.28 已不存在的
   ``request.block_table``，恒为 None；且读取发生在 ``_free_request`` 之前
   但上游 ``_re_block_ids`` 快照从不被消费（泄漏）。

注入（三个 ``_patchkit`` 类级包装）：

- ``update_from_output`` 前置：本步 routing 行按**调度时**窗口
  ``[num_computed_tokens, +num_scheduled)``（取自 ``scheduler_output``；异步
  调度下 ``request.num_computed_tokens`` 在 ``schedule()`` 里已前推，不能用）
  与 ``_re_block_ids`` 调度时块表算 slot（attention 组 ``block*bs + off``，与
  ``RoutedExpertsManager.get`` 同公式；不依赖 runner 只为 gid 0 填的
  slot mapping——混合模型的 gid 0 不是 full-attention 组）。已释放请求的行丢弃。
  **落盘前校验 int32 原值**：id ∈ [0, E)、每行 top-K 互异；-1（runner 填充、
  本步某层没写）或 poison 一律 raise——uint8 窄化会把它们静默变成合法 id。
- ``_handle_stopped_request`` 前置：块尚未释放时 ``mgr.get(blocks,
  num_tokens - 1, prompt_start)`` 读出整段，暂存；后置在 ``update_from_output``
  返回前挂到该请求的 EngineCoreOutput。流式输入（resumable）请求 raise：
  分段停时的 routing 语义未定义。
- ``_free_request`` 后置：清 ``_re_block_ids`` 与暂存。
"""

import os
from functools import wraps

import numpy as np

_FINGERPRINT = "vllm-omni-xy@deployed"
_STOPPED = "_huanyu_r3_stopped_routing"


def _schedule_starts(scheduler_output) -> dict:
    starts = {new.req_id: int(new.num_computed_tokens) for new in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    starts.update(zip(cached.req_ids, map(int, cached.num_computed_tokens)))
    return starts


def _num_experts(scheduler) -> int:
    value = scheduler.__dict__.get("_huanyu_r3_num_experts")
    if value is None:
        value = int(scheduler.vllm_config.model_config.get_num_experts())
        scheduler.__dict__["_huanyu_r3_num_experts"] = value
    return value


def validate_rows(rows: np.ndarray, num_experts: int, where: str) -> None:
    if rows.ndim != 3:
        raise RuntimeError(f"R3 {where}: routing rows must be [N, L, K], got {rows.shape}")
    if rows.shape[0] == 0:
        return
    bad = (rows < 0) | (rows >= num_experts)
    if bad.any():
        row, layer, _ = map(int, np.argwhere(bad)[0])
        raise RuntimeError(
            f"R3 {where}: {int(bad.any(axis=(1, 2)).sum())}/{rows.shape[0]} rows carry expert ids outside "
            f"[0, {num_experts}) (row={row} layer={layer}: {rows[row, layer].tolist()}); -1 = layer not "
            "captured this step"
        )
    ordered = np.sort(rows, axis=-1)
    duplicate = (ordered[..., 1:] == ordered[..., :-1]).any(axis=-1)
    if duplicate.any():
        row, layer = map(int, np.argwhere(duplicate)[0])
        raise RuntimeError(f"R3 {where}: repeated expert ids at row={row} layer={layer}: "
                           f"{rows[row, layer].tolist()}")


def store_step(scheduler, scheduler_output, model_runner_output) -> int:
    """把本步 routing 行落进槽位缓冲；返回落盘行数。"""
    routed = model_runner_output.routed_experts
    if routed is None:
        return 0
    data = np.asarray(routed.routing_data)
    mgr = scheduler.routed_experts_mgr
    block_size = mgr.block_size
    snapshot = getattr(scheduler, "_re_block_ids", {})
    starts = _schedule_starts(scheduler_output)
    keep, slots = [], []
    offset = 0
    for rid in model_runner_output.req_ids:
        n = int(scheduler_output.num_scheduled_tokens.get(rid, 0))
        if n <= 0:
            continue
        rows = np.arange(offset, offset + n)
        offset += n
        if rid not in scheduler.requests:
            continue
        start = starts.get(rid)
        if start is None:
            raise RuntimeError(f"R3: request {rid} has routing rows but no schedule-time window")
        block_ids = snapshot.get(rid)
        if block_ids is None:
            block_ids = scheduler.kv_cache_manager.get_blocks(rid).get_block_ids()[mgr.attn_gid]
        block_ids = np.asarray(block_ids, dtype=np.int64)
        if block_ids.size * block_size < start + n:
            continue
        position = np.arange(start, start + n, dtype=np.int64)
        keep.append(rows)
        slots.append(block_ids[position // block_size] * block_size + position % block_size)
    if offset != data.shape[0]:
        raise RuntimeError(f"R3: scheduled tokens {offset} != routing rows {data.shape[0]} "
                           "(model_runner_output.req_ids order disagrees with the runner batch)")
    if not keep:
        return 0
    rows = data[np.concatenate(keep)]
    validate_rows(rows, _num_experts(scheduler), "store_batch")
    mgr.store_batch(rows, np.concatenate(slots))
    return int(rows.shape[0])


def read_stopped(scheduler, request) -> None:
    """块释放前读出整段 routing 并暂存（_handle_stopped_request 入口）。"""
    if request.resumable:
        raise NotImplementedError(
            "R3 routing replay does not support streaming-input (resumable) requests: a segment stop "
            "has no defined routing span"
        )
    mgr = scheduler.routed_experts_mgr
    block_ids = scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()[mgr.attn_gid]
    num_tokens = request.num_tokens - 1
    if len(block_ids) * mgr.block_size < num_tokens:
        raise RuntimeError(f"R3 request {request.request_id}: {len(block_ids)} blocks cannot hold "
                           f"{num_tokens} routed tokens at stop")
    prompt_start = 0
    params = request.sampling_params
    if params is not None and getattr(params, "routed_experts_prompt_start", None) is not None:
        prompt_start = int(params.routed_experts_prompt_start)
    scheduler.__dict__.setdefault(_STOPPED, {})[request.request_id] = mgr.get(
        block_ids, num_tokens, token_start=prompt_start)


def _iter_outputs(outputs):
    for value in outputs.values():
        items = getattr(value, "outputs", value)
        yield from (items if isinstance(items, (list, tuple)) else [items])


def attach_stopped(scheduler, outputs) -> None:
    stopped = scheduler.__dict__.get(_STOPPED)
    if not stopped:
        return
    for output in _iter_outputs(outputs):
        routed = stopped.pop(output.request_id, None)
        if routed is not None:
            output.routed_experts = routed
    if stopped:
        leftover = sorted(stopped)
        stopped.clear()
        raise RuntimeError(f"R3: stopped requests emitted no EngineCoreOutput: {leftover[:4]}")


def install_scheduler_hooks(module) -> bool:
    """post-import(vllm_omni.core.sched.omni_ar_scheduler)。"""
    from verl_omni_ext._patchkit import idempotent_patch

    common = dict(target_module=module.__name__, target_attr="OmniARScheduler",
                  fingerprint=_FINGERPRINT, probe_signature=True)
    reported = {}

    @idempotent_patch(name="huanyu_r3_sched_update", expected_signature="update_from_output", **common)
    def _wrap_update(original, __patch_signature__=None):
        @wraps(original)
        def update_from_output(self, scheduler_output, model_runner_output):
            if not getattr(self, "enable_return_routed_experts", False):
                return original(self, scheduler_output, model_runner_output)
            stored = store_step(self, scheduler_output, model_runner_output)
            if stored and not reported:
                reported["store"] = True
                print(f"[huanyu-replay] scheduler storing routed experts: first step rows={stored} "
                      f"pid={os.getpid()}", flush=True)
            outputs = original(self, scheduler_output, model_runner_output)
            attach_stopped(self, outputs)
            return outputs

        return update_from_output

    @idempotent_patch(name="huanyu_r3_sched_stop", expected_signature="_handle_stopped_request", **common)
    def _wrap_stop(original, __patch_signature__=None):
        @wraps(original)
        def _handle_stopped_request(self, request):
            if getattr(self, "enable_return_routed_experts", False):
                read_stopped(self, request)
            return original(self, request)

        return _handle_stopped_request

    @idempotent_patch(name="huanyu_r3_sched_free", expected_signature="_free_request", **common)
    def _wrap_free(original, __patch_signature__=None):
        @wraps(original)
        def _free_request(self, request, *args, **kwargs):
            result = original(self, request, *args, **kwargs)
            getattr(self, "_re_block_ids", {}).pop(request.request_id, None)
            return result

        return _free_request

    if not (_wrap_update() and _wrap_stop() and _wrap_free()):
        raise RuntimeError("HUANYU_EXPERT_REPLAY=1 but the OmniARScheduler R3 patches were not applied")
    return True
