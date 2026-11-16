"""Explicit inference-worker entry point for Huanyu precision and IPC hooks.

Keep this filename distinct from the dependency package ``vllm_omni``:
direct Huanyu CLI scripts put this directory first on Python's import path.
"""

# Keep the existing native model and pipeline registration, then apply the
# Huanyu-specific hooks. No shared source files or checkpoint configs are edited.
from verl_omni_ext.models.qwen3_6_whisper import vllm_omni as _registration  # noqa: F401

from .cache_precision import install_huanyu_cache_precision
from .kernel_perf.rollout_kernels import install_huanyu_rollout_kernels
from .processor_compat import install_huanyu_processor_compat
from .rollout_precision import install_huanyu_rollout_precision
from .weight_reload import install_huanyu_weight_reload
from .video_rollout import install_huanyu_video_rollout

# HCCL 策略自检：本模块在每个 rollout server 进程（VLLM_OMNI_EXTERNAL_MODULES）
# 加载。TP 小 allReduce 在 AICPU RPC 路径是 1.9ms/次、AIV 是 0.5ms/次
# （2026-11-10 微基准）——profile 声明与进程 env 不一致必须立刻炸，不能带进
# decode 关键路径。未声明 HUANYU_HCCL_PROFILE 时为 no-op。
from verl_omni_ext.probes.hccl_policy import assert_hccl_policy

assert_hccl_policy()

# Dummy multimodal budgeting calls the processor before model initialization.
install_huanyu_processor_compat()
install_huanyu_video_rollout()
install_huanyu_rollout_precision()
# kernel 档案（MINICPMO_GDN_NPU_KERNEL=1）：互斥校验 + 原生 GDN 算子断言。
# 未开启时为 no-op，不影响既有 precision 安装链路。
install_huanyu_rollout_kernels()
install_huanyu_cache_precision()
# Best-effort compatibility bind may defer during the native registry's
# circular import. The rollout adapter also selects an explicit worker class;
# that worker factory import is the mandatory, non-skippable binding point.
install_huanyu_weight_reload()
# R3 的 vLLM 侧钩子不在这里：由 vllm.general_plugins 入口 huanyu_r3
# （expert_replay/plugin.py）在每个 vLLM 进程安装。

from .gen_timing_engine import install_engine as _install_gen_timing_engine

_install_gen_timing_engine()

# Selected only by the text/embedding A/B launcher, before runner construction.
import os as _os
if _os.environ.get("HUANYU_EMBEDDING_AB_DIR", "").strip():
    from .prompt_embedding_probe import install_runtime_probe
    install_runtime_probe()
