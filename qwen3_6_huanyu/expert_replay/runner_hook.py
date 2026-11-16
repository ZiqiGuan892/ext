"""R3 runner 侧注入（vLLM worker 进程；零 vllm-omni 源码改动）。

两个缺口（2026-11-12 对照部署树确认）：

1. **异步调度无快照**：omni ``NPUARModelRunner.sample_tokens`` 的异步返回构造
   ``AsyncGPUModelRunnerOutput`` 时没有传 ``routed_experts=``（vllm-ascend
   ``NPUModelRunner`` 自己的同款路径是传的）。于是 scheduler 拿到的是
   ``_omni_extract_routed_experts`` 包出来的 pinned ``torch.empty`` 缓冲视图——
   异步模式下从没有 D2H 写进去，读到的就是 int32 poison（-1077936128）。
   修复：``sample_tokens`` 入口（本步 forward 已在默认流排队、下一步尚未开始）
   克隆 ``device_buffer[:total]`` 与 slot mapping 成私有快照，经
   ``AsyncGPUModelRunnerOutput.__init__`` 注入；上游原有的 copy-stream D2H 与
   ``get_output`` 覆盖逻辑即生效。快照未被消费 → raise。
2. **残留行不可见**：capturer 设备缓冲跨步复用、从不清零；某层没写到的 token
   会带着上一步的合法 id 混过所有校验。``execute_model`` 入口把本步
   ``[:total]`` 行填 -1，scheduler 在落盘前拒收任何 -1 行。

同步调度路径不受影响（vllm-ascend ``_bookkeeping_sync`` 已做 D2H）。
"""

from functools import wraps

_FINGERPRINT = "vllm-omni-xy@deployed"
_PENDING: dict = {}


def _scheduler_output(args, kwargs):
    if "scheduler_output" in kwargs:
        return kwargs["scheduler_output"]
    return args[0] if args else None


def _poison_step_rows(runner, scheduler_output) -> None:
    if not getattr(runner, "routed_experts_initialized", False) or scheduler_output is None:
        return
    total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    if total > 0:
        runner.routed_experts_capturer.device_buffer[:total].fill_(-1)


def _take_step_snapshot(runner, tensors_cls) -> None:
    state = getattr(runner, "execute_model_state", None)
    if state is None or not getattr(runner, "use_async_scheduling", False):
        return
    if not getattr(runner, "routed_experts_initialized", False):
        return
    scheduler_output = getattr(state, "scheduler_output", None) or state[0]
    total = int(scheduler_output.total_num_scheduled_tokens)
    if total <= 0:
        return
    if _PENDING:
        raise RuntimeError("R3: previous step's routed-experts snapshot was never consumed")
    _PENDING["snapshot"] = tensors_cls(
        routing_data=runner.routed_experts_capturer.device_buffer[:total].clone(),
        slot_mapping=runner.routed_experts_slot_mapping_device[:total].clone(),
    )


def install_async_output_injection(module) -> bool:
    """post-import(vllm.v1.worker.gpu_model_runner)：AsyncGPUModelRunnerOutput 接收快照。"""
    from verl_omni_ext._patchkit import idempotent_patch

    @idempotent_patch(
        name="huanyu_r3_async_output",
        target_module=module.__name__,
        target_attr="AsyncGPUModelRunnerOutput",
        fingerprint="vllm==0.28.0+xy",
        expected_signature="__init__",
        probe_signature=True,
    )
    def _wrap(original, __patch_signature__=None):
        @wraps(original)
        def __init__(self, *args, **kwargs):
            snapshot = _PENDING.pop("snapshot", None)
            if snapshot is not None and len(args) <= 6 and kwargs.get("routed_experts") is None:
                kwargs["routed_experts"] = snapshot
            original(self, *args, **kwargs)

        return __init__

    if not _wrap():
        raise RuntimeError("HUANYU_EXPERT_REPLAY=1 but the AsyncGPUModelRunnerOutput R3 patch was not applied")
    return True


def install_runner_hooks(module) -> bool:
    """post-import(npu_ar_model_runner)：execute_model 填 -1、sample_tokens 快照。"""
    from verl_omni_ext._patchkit import idempotent_patch

    common = dict(target_module=module.__name__, target_attr="NPUARModelRunner",
                  fingerprint=_FINGERPRINT, probe_signature=True)

    @idempotent_patch(name="huanyu_r3_runner_execute", expected_signature="execute_model", **common)
    def _wrap_execute(original, __patch_signature__=None):
        @wraps(original)
        def execute_model(self, *args, **kwargs):
            _poison_step_rows(self, _scheduler_output(args, kwargs))
            return original(self, *args, **kwargs)

        return execute_model

    @idempotent_patch(name="huanyu_r3_runner_sample", expected_signature="sample_tokens", **common)
    def _wrap_sample(original, __patch_signature__=None):
        from vllm.v1.outputs import RoutedExpertsTensors
        from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

        @wraps(original)
        def sample_tokens(self, *args, **kwargs):
            _take_step_snapshot(self, RoutedExpertsTensors)
            try:
                result = original(self, *args, **kwargs)
            finally:
                leftover = _PENDING.pop("snapshot", None)
            if leftover is not None and isinstance(result, AsyncGPUModelRunnerOutput):
                raise RuntimeError("R3: async output was built without consuming the routed-experts snapshot")
            return result

        return sample_tokens

    if not (_wrap_execute() and _wrap_sample()):
        raise RuntimeError("HUANYU_EXPERT_REPLAY=1 but the NPUARModelRunner R3 patches were not applied")
    return True
