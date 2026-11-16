"""Server actor 侧 R3 传输补丁：ARStrategy.process_output 提取并校验 routed_experts。

verl-omni 的 ``ARStrategy.process_output`` 构建 ``TokenOutput`` 时丢弃了
``CompletionOutput.routed_experts``（verl 自身的 ``vllm_async_server.py``
已有同款提取，verl-omni 未跟上）。按 ``features/opd/_server_patch.py``
先例做**运行时注入**（``_patchkit.idempotent_patch``）：env 门控、幂等、
fail-closed，非 replay 运行零行为变化，不改 verl-omni 源码。

安装点：``Qwen36HuanyuThinkerAdapter.configure_processor``（与
``rollout_inputs`` 的 prompt 收缩钩子同处 Ray server actor），不进 vLLM
EngineCore / worker 进程。

校验（在 int16 组包把垃圾值抹成 0 之前）：
- 自然完成的请求必须带 routed_experts；abort/preempt → 透传 None（trainer 按
  无记录样本整体回退原生路由并计数）；
- 形状 ``[len(prompt_token_ids) + len(token_ids) - 1, L, K]``；
- 每行 top-K 互不相同（全零/重复行 = 捕获链没写入）；
- vLLM 展开后的 prompt 长度 == 训练侧展开 prompt 长度（``rollout_inputs``
  收缩前登记）；不等说明 vLLM 重展开与训练不一致，路由行会整体错位。
  ``HUANYU_REPLAY_DEBUG=1`` 时只打印不 raise，用于在真实数据上测量。
"""

from contextvars import ContextVar
import os

import numpy as np

_ENABLED_ENV = "HUANYU_EXPERT_REPLAY"
# rollout_inputs 收缩前登记的训练侧展开 prompt 长度（同一 generate 协程上下文）。
TRAINING_PROMPT_LEN: ContextVar = ContextVar("huanyu_r3_training_prompt_len", default=None)
_REPORTED: dict = {}


def _debug() -> bool:
    return os.environ.get("HUANYU_REPLAY_DEBUG", "0") == "1"


def validate_routed_experts(routed, *, vllm_prompt_len: int, num_output_tokens: int,
                            training_prompt_len: int | None, request_id=None) -> np.ndarray:
    """返回 numpy 化的 routed_experts；任何不一致 raise（debug 下 prompt 长度只打印）。"""
    array = np.asarray(routed)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(f"R3 request {request_id!r}: routed_experts must be integer [N, L, K], "
                           f"got {array.dtype} {array.shape}")
    expected_rows = vllm_prompt_len + num_output_tokens - 1
    if array.shape[0] != expected_rows:
        raise RuntimeError(f"R3 request {request_id!r}: routed_experts rows {array.shape[0]} != "
                           f"prompt {vllm_prompt_len} + output {num_output_tokens} - 1")
    if array.shape[0]:
        ordered = np.sort(array, axis=-1)
        duplicate = (ordered[..., 1:] == ordered[..., :-1]).any(axis=-1)  # [N, L]
        if duplicate.any():
            row, layer = map(int, np.argwhere(duplicate)[0])
            raise RuntimeError(
                f"R3 request {request_id!r}: {int(duplicate.any(axis=-1).sum())}/{array.shape[0]} routing rows "
                f"have repeated expert ids (first row={row} layer={layer}: {array[row, layer].tolist()}); "
                "the capture chain did not write these tokens"
            )
    if training_prompt_len is not None and training_prompt_len != vllm_prompt_len:
        message = (f"R3 request {request_id!r}: vLLM prompt has {vllm_prompt_len} tokens but the training "
                   f"prompt has {training_prompt_len}; routing rows would be shifted by "
                   f"{training_prompt_len - vllm_prompt_len}")
        if not _debug():
            raise RuntimeError(message)
        print(f"[huanyu-replay-debug] prompt_len_mismatch {message}", flush=True)
    return array


def _r3_process_output(original, self, final_res, params, sampling_params):
    output = original(self, final_res, params, sampling_params)
    req_output = getattr(final_res, "request_output", None) or final_res
    outputs = getattr(req_output, "outputs", None)
    routed = getattr(outputs[0], "routed_experts", None) if outputs else None
    if routed is None:
        if getattr(output, "stop_reason", None) == "completed":
            raise RuntimeError(
                "HUANYU_EXPERT_REPLAY=1 但完成的请求缺少 routed_experts：检查 vLLM 进程的 "
                "huanyu_r3 general plugin 是否在 EngineCore 与所有 worker 日志中 applied，"
                "以及 enable_return_routed_experts 是否到达 stage ModelConfig"
            )
        return output
    training_prompt_len = TRAINING_PROMPT_LEN.get()
    if training_prompt_len is None and not _REPORTED.get("no_training_len"):
        # 收缩钩子未在本协程上下文登记（非 Huanyu processor 或上下文未传递）：
        # prompt 长度校验退化为 trainer 侧样本级对齐校验（layout.build_padded_targets）。
        _REPORTED["no_training_len"] = True
        print("[huanyu-replay] WARNING: training prompt length not registered in this request context; "
              "server-side prompt-length check skipped (trainer alignment check still applies)", flush=True)
    routed = validate_routed_experts(
        routed,
        vllm_prompt_len=len(req_output.prompt_token_ids),
        num_output_tokens=len(outputs[0].token_ids),
        training_prompt_len=training_prompt_len,
        request_id=getattr(final_res, "request_id", None),
    )
    return output.model_copy(update={"routed_experts": routed})


def install_server_patch() -> bool:
    """包装 ARStrategy.process_output（幂等；目标缺失 → strict raise）。"""
    from verl_omni_ext._patchkit import idempotent_patch

    @idempotent_patch(
        name="huanyu_r3_ar_process_output",
        target_module="verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy",
        target_attr="ARStrategy",
        fingerprint="verl-omni==0.2.0rc1",
        expected_signature="process_output",
        probe_signature=True,
    )
    def _wrap(original, __patch_signature__=None):
        def process_output(self, final_res, params, sampling_params):
            return _r3_process_output(original, self, final_res, params, sampling_params)

        process_output.__name__ = "process_output"
        process_output.__wrapped__ = original
        return process_output

    if not _wrap():
        raise RuntimeError("HUANYU_EXPERT_REPLAY=1 but the ARStrategy.process_output R3 patch was not applied")
    print(f"[huanyu-replay] server patch installed: ARStrategy.process_output "
          f"routed_experts extraction pid={os.getpid()}", flush=True)
    return True


def maybe_install_server_patch() -> bool:
    """仅在 HUANYU_EXPERT_REPLAY=1 的运行中安装；返回是否已激活。"""
    value = os.environ.get(_ENABLED_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_ENABLED_ENV} must be 0 or 1, got {value!r}")
    if value != "1":
        return False
    return install_server_patch()
