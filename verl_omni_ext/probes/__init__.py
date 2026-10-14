"""
探针模块（从 examples/probes/ 内化）

可 import 调用的测量工具，用于在写适配器之前测量关键事实。
也用于 tests/ 里做契约验证。

用法：
  python -m verl_omni_ext.probes.forward_signature --model_path /path/to/model
  python -m verl_omni_ext.probes.processor_whitelist --model_path /path/to/model
"""