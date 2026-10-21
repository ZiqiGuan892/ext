#!/usr/bin/env python3
"""Run the native import check without importing the ext package initializer."""

import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    runtime = project_root / "verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py"
    if not runtime.is_file():
        raise SystemExit(f"Huanyu runtime module not found: {runtime}; check PROJECT_ROOT")
    runpy.run_path(str(runtime), run_name="__main__")
