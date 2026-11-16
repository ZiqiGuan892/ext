#!/usr/bin/env python3
"""Run the native import check without importing the ext package initializer."""

import importlib.util
import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    runtime = project_root / "verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py"
    if not runtime.is_file():
        raise SystemExit(f"Huanyu runtime module not found: {runtime}; check PROJECT_ROOT")

    # HCCL 策略预检（fail-loud）：声明的 HUANYU_HCCL_PROFILE 与本进程 env 不一致时
    # 在启动 3 秒内炸掉——「AIV 设了但被静默回退」曾是 69% kernel 时间的形态
    # （2026-11-10 诊断，perf_report/analysis_11_10）。按文件路径加载，不触发包初始化。
    hccl_policy = project_root / "verl_omni_ext/probes/hccl_policy.py"
    if hccl_policy.is_file():
        spec = importlib.util.spec_from_file_location("_hccl_policy_preflight", hccl_policy)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.assert_hccl_policy()

    runpy.run_path(str(runtime), run_name="__main__")