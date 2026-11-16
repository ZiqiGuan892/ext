"""rollout 侧 R3 开关（L1）：向 vllm-omni stage engine extras 透传捕获开关。

``enable_return_routed_experts`` 打开 vLLM 的 RoutedExpertsCapturer / 调度侧
RoutedExpertsManager；Ascend 栈上的捕获绑定、异步调度快照与调度侧存取由
``expert_replay.plugin``（vllm.general_plugins）在 vLLM 进程内运行时注入，
server actor 侧的 TokenOutput 传输由 ``server_patch`` 补齐。

仅依赖 stdlib：rollout_adapter 在 CPU 契约检查期也会 import，保持轻量。
"""

import os


def expert_replay_engine_extras() -> dict:
    value = os.environ.get("HUANYU_EXPERT_REPLAY", "0")
    if value not in ("0", "1"):
        raise ValueError(f"HUANYU_EXPERT_REPLAY must be 0 or 1, got {value!r}")
    if value != "1":
        return {}
    return {"enable_return_routed_experts": True}
