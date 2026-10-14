"""
通用 Feature 区域

跨多个注册表的特性按功能域组织在这里。
每个 feature 子目录是一个自包含的特性（可能同时需要 trainer + worker + reward）。

单注册表的扩展仍放在对应目录（trainer/ reward/ algos/ workers/）。

组织规则：
  - 一个特性只涉及 1 个注册表 → 放对应目录（trainer/ 或 reward/ 等）
  - 一个特性涉及 2+ 注册表 → 放 features/<feature_name>/
"""
from . import fullduplex  # noqa: F401