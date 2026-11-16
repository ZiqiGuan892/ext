"""R3 replay 的 verl FSDP 引擎接线（verl FSDP 后端无原生 router replay）。

三个类级包装（``_patchkit.idempotent_patch``，Huanyu 模型标记门控）：

- ``FSDPEngine.forward_backward_batch``（基类）：``controller.begin_batch()``
  / ``end_batch()`` 包住全部 micro-batch 的 forward **与 backward**——
  ``loss.backward()`` 在 ``forward_step`` 返回之后才执行，checkpoint 重算
  必须仍看得到 targets。包基类而不是 ``FSDPEngineWithLMHead``：legacy
  ``microbatch_sync`` 包的是基类同名方法，子类覆盖会把它静默旁路。
- ``FSDPEngineWithLMHead.prepare_model_inputs``：原函数返回 padded
  ``input_ids [B, S]`` 后（此时 micro_batch 已在设备上），按同一布局构造
  targets 并 ``arm``（不 clear，见 controller 生命周期）。
- ``FSDPEngineWithLMHead.forward_step``：前向后断言每个 MoE 层都消费过 target。

ref 引擎（construction-time ``engine_config.forward_only=True``）不回放；
actor 的 old_log_prob 与 update 共用同一路径，同一钩覆盖。

fail-closed：replay 开启且模型带标记，但 micro_batch 缺 routed_experts、
层数/专家号/样本对齐不符 → raise。
"""

from functools import wraps
import os

from . import controller
from .layout import build_padded_targets
from .moe_kernel import _FLAG as _MOE_FLAG, _language_layers

_MARK = "_huanyu_expert_replay"
_IMPL = "verl.workers.engine.fsdp.transformer_impl"
_FINGERPRINT = "verl==0.10.0.dev0"


def _model_chain(module):
    seen = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        yield module
        wrapped = getattr(module, "_fsdp_wrapped_module", None)
        module = wrapped if wrapped is not None else getattr(module, "_orig_mod", None)


def _replay_engine(engine) -> bool:
    if getattr(getattr(engine, "engine_config", None), "forward_only", False):
        return False
    return any(getattr(candidate, _MARK, False) for candidate in _model_chain(getattr(engine, "module", None)))


def _moe_geometry(module) -> tuple[int, int]:
    """(融合 MoE 层数, 专家数)；按 decoder 层号连续计数。"""
    for candidate in _model_chain(module):
        try:
            layers = _language_layers(candidate)
        except RuntimeError:
            continue
        fused = [layer.mlp for layer in layers if getattr(getattr(layer, "mlp", None), _MOE_FLAG, False)]
        if fused:
            return len(fused), int(fused[0].experts.num_experts)
    return 0, 0


def _geometry(engine) -> tuple[int, int]:
    cached = getattr(engine, "_huanyu_replay_geometry", None)
    if cached is None:
        cached = _moe_geometry(engine.module)
        if cached[0] == 0:
            raise RuntimeError("expert_replay: actor module has no fused MoE blocks installed")
        engine._huanyu_replay_geometry = cached
    return cached


def _arm_micro_batch(engine, micro_batch, model_inputs) -> None:
    routed = micro_batch.get("routed_experts", None)
    if routed is None:
        raise RuntimeError(
            "HUANYU_EXPERT_REPLAY=1 but micro_batch has no 'routed_experts': rollout must run with "
            "actor_rollout_ref.rollout.enable_rollout_routing_replay=true (fail-closed R3)"
        )
    num_layers, num_experts = _geometry(engine)
    targets, stats = build_padded_targets(
        routed, model_inputs["input_ids"], num_layers=num_layers, num_experts=num_experts)
    controller.arm(targets)
    if stats["unrecorded_samples"] or os.environ.get("HUANYU_REPLAY_DEBUG", "0") == "1":
        print(f"[huanyu-replay] micro_batch gen={controller.generation()} samples={stats['samples']} "
              f"unrecorded_samples={stats['unrecorded_samples']} recorded_rows={stats['recorded_rows']}"
              f"/{stats['tokens']} pid={os.getpid()}", flush=True)


def install_engine_hook(module) -> bool:
    """标记模型并安装三个类级包装（幂等）。configure_model 期调用（FSDP 包装前）。"""
    setattr(module, _MARK, True)
    from verl_omni_ext._patchkit import idempotent_patch

    @idempotent_patch(
        name="huanyu_r3_forward_backward_batch",
        target_module=_IMPL,
        target_attr="FSDPEngine",
        fingerprint=_FINGERPRINT,
        expected_signature="forward_backward_batch",
        probe_signature=True,
    )
    def _wrap_batch(original, __patch_signature__=None):
        @wraps(original)
        def forward_backward_batch(engine, *args, **kwargs):
            if not _replay_engine(engine):
                return original(engine, *args, **kwargs)
            controller.begin_batch()
            try:
                return original(engine, *args, **kwargs)
            finally:
                controller.end_batch()

        return forward_backward_batch

    @idempotent_patch(
        name="huanyu_r3_prepare_model_inputs",
        target_module=_IMPL,
        target_attr="FSDPEngineWithLMHead",
        fingerprint=_FINGERPRINT,
        expected_signature="prepare_model_inputs",
        probe_signature=True,
    )
    def _wrap_inputs(original, __patch_signature__=None):
        @wraps(original)
        def prepare_model_inputs(engine, micro_batch, *args, **kwargs):
            result = original(engine, micro_batch, *args, **kwargs)
            if controller.in_batch() and _replay_engine(engine):
                _arm_micro_batch(engine, micro_batch, result[0])
            return result

        return prepare_model_inputs

    @idempotent_patch(
        name="huanyu_r3_forward_step",
        target_module=_IMPL,
        target_attr="FSDPEngineWithLMHead",
        fingerprint=_FINGERPRINT,
        expected_signature="forward_step",
        probe_signature=True,
    )
    def _wrap_step(original, __patch_signature__=None):
        @wraps(original)
        def forward_step(engine, *args, **kwargs):
            result = original(engine, *args, **kwargs)
            if controller.in_batch() and _replay_engine(engine):
                controller.assert_all_layers_fired()
            return result

        return forward_step

    applied = [_wrap_batch(), _wrap_inputs(), _wrap_step()]
    if not all(applied):
        raise RuntimeError(f"expert_replay engine hook not fully applied: {applied}")
    print(f"[huanyu-replay] engine hook installed: FSDPEngine.forward_backward_batch + "
          f"FSDPEngineWithLMHead.prepare_model_inputs/forward_step pid={os.getpid()}", flush=True)
    return True
