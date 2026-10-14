"""槽位④: Qwen3.5-MoE 不需要 custom dataset——使用上游 RLHFDataset 即可。"""
# Qwen3.5-MoE 的 processor 类名在 verl.utils.hf_processor 白名单内，
# 不会走到 except Exception 吞异常的路径，所以不需要子类化数据集。