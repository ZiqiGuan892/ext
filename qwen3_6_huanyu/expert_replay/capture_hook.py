"""R3 rollout 捕获绑定（vLLM worker 进程；零 vllm / vllm-ascend 源码改动）。

断链根因：vLLM 的 ``bind_routed_experts_capturer`` 对 Ascend 栈走 monolithic
分支（``fused_experts.set_capture_fn``），该回调在 Ascend MoE 路径上从不触发；
而 Ascend 实际的选路 ``AscendRoutedExperts._select_experts`` →
``router._select_experts`` 正是 vLLM ``BaseRouter._select_experts`` 模板方法，
其中 ``if self.capture_fn is not None: self.capture_fn(topk_ids)`` 在 EPLB
映射**之前**捕获逻辑专家 id——只是 router 的 capture_fn 没人设置。

修复：后置包装 ``GPUModelRunner.init_routed_experts_capturer``（NPU runner 继承
此方法；它在 capture_model 之前执行，图捕获时回调已就位），在原绑定之后对每个
MoE 层的 ``BaseRouter`` 设置 ``partial(capturer.capture, layer_id)``，并断言
绑定的层号恰好覆盖 ``range(num_layers)``——少绑一层就等于那一层永远读到
上一步的残留，必须启动期 raise。
"""

import os
from functools import partial, wraps

_FINGERPRINT = "vllm==0.28.0+xy"


def _layer_id(module):
    layer_id = getattr(module, "layer_id", None)
    if isinstance(layer_id, int) and layer_id >= 0:
        return layer_id
    layer_name = getattr(module, "layer_name", None)
    if layer_name:
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(layer_name)
    return None


def bind_router_capture(model, capturer) -> dict:
    """把 capturer 绑到每个 MoE 层的 BaseRouter；返回 {layer_id: router}。"""
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    num_layers = int(capturer.device_buffer.shape[1])
    by_router: dict[int, tuple] = {}
    for module in model.modules():
        router = getattr(module, "router", None)
        if not isinstance(router, BaseRouter):
            continue
        layer_id = _layer_id(module)
        if layer_id is None:
            continue
        known = by_router.get(id(router))
        if known is not None and known[0] != layer_id:
            raise RuntimeError(f"R3 capture: router shared by layers {known[0]} and {layer_id}")
        by_router[id(router)] = (layer_id, router)

    bound: dict[int, object] = {}
    for layer_id, router in by_router.values():
        if layer_id in bound and bound[layer_id] is not router:
            raise RuntimeError(f"R3 capture: two distinct routers claim layer {layer_id}")
        bound[layer_id] = router
    if sorted(bound) != list(range(num_layers)):
        missing = sorted(set(range(num_layers)) - set(bound))
        extra = sorted(set(bound) - set(range(num_layers)))
        raise RuntimeError(
            f"R3 capture: MoE routers cover layers {sorted(bound)[:4]}...; missing={missing} "
            f"out_of_range={extra} for a {num_layers}-layer capture buffer"
        )
    for layer_id, router in bound.items():
        router.set_capture_fn(partial(capturer.capture, layer_id))
    return bound


def install_capture_binding(module) -> bool:
    """post-import(vllm.v1.worker.gpu_model_runner)：包装 init_routed_experts_capturer。"""
    from verl_omni_ext._patchkit import idempotent_patch

    @idempotent_patch(
        name="huanyu_r3_capture_binding",
        target_module=module.__name__,
        target_attr="GPUModelRunner",
        fingerprint=_FINGERPRINT,
        expected_signature="init_routed_experts_capturer",
        probe_signature=True,
    )
    def _wrap(original, __patch_signature__=None):
        @wraps(original)
        def init_routed_experts_capturer(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            capturer = getattr(self, "routed_experts_capturer", None)
            if capturer is None:
                raise RuntimeError("R3: init_routed_experts_capturer did not create routed_experts_capturer")
            bound = bind_router_capture(self.model, capturer)
            print(f"[huanyu-replay] capture bound: {len(bound)} routers -> RoutedExpertsCapturer "
                  f"buffer={tuple(capturer.device_buffer.shape)} pid={os.getpid()}", flush=True)
            return result

        return init_routed_experts_capturer

    if not _wrap():
        raise RuntimeError("HUANYU_EXPERT_REPLAY=1 but the R3 capture binding patch was not applied")
    return True
