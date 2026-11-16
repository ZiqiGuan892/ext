"""vLLM general plugin：R3 rollout 侧钩子的唯一安装入口。

``pyproject.toml`` 的 ``vllm.general_plugins`` 入口点 ``huanyu_r3`` 指向
``register``；vLLM 在前端 / EngineCore / 每个 worker 进程里都会调用它（可能多次，
必须幂等）。``HUANYU_EXPERT_REPLAY`` 未开时为零成本 no-op。

开启时强制 ``VERL_OMNI_EXT_PATCH_STRICT=1``，并为三个目标模块登记 post-import
钩子（目标模块在哪个进程被 import，就在哪个进程安装；前端进程不会因此
拉起 NPU runner）：

| 模块 | 安装 | 进程 |
|------|------|------|
| ``vllm.v1.worker.gpu_model_runner`` | capture 绑定 + 异步输出快照注入 | worker |
| ``vllm_omni.platforms.npu.worker.npu_ar_model_runner`` | 步首填 -1 + 步末快照 | worker |
| ``vllm_omni.core.sched.omni_ar_scheduler`` | store / stop 读取 / free 清理 | EngineCore |
"""

import os

_ENABLED_ENV = "HUANYU_EXPERT_REPLAY"
_REGISTERED: dict = {}

TARGETS = {
    "vllm.v1.worker.gpu_model_runner": ("capture_hook.install_capture_binding",
                                        "runner_hook.install_async_output_injection"),
    "vllm_omni.platforms.npu.worker.npu_ar_model_runner": ("runner_hook.install_runner_hooks",),
    "vllm_omni.core.sched.omni_ar_scheduler": ("scheduler_hook.install_scheduler_hooks",),
}


def _installer(path):
    from importlib import import_module

    module_name, func_name = path.rsplit(".", 1)
    func = getattr(import_module(f"{__package__}.{module_name}"), func_name)

    def run(module):
        func(module)
        print(f"[huanyu-replay] plugin applied {path} on {module.__name__} pid={os.getpid()}", flush=True)

    return run


def register() -> None:
    value = os.environ.get(_ENABLED_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_ENABLED_ENV} must be 0 or 1, got {value!r}")
    if value != "1" or _REGISTERED:
        return
    strict = os.environ.get("VERL_OMNI_EXT_PATCH_STRICT")
    if strict not in (None, "1"):
        raise RuntimeError(f"{_ENABLED_ENV}=1 requires VERL_OMNI_EXT_PATCH_STRICT=1, got {strict!r}")
    os.environ["VERL_OMNI_EXT_PATCH_STRICT"] = "1"
    _REGISTERED["pid"] = os.getpid()

    from .import_hooks import when_imported

    for target, installers in TARGETS.items():
        for path in installers:
            when_imported(target, _installer(path))
