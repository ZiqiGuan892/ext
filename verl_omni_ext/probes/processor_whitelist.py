"""
Processor 白名单探针

verl.utils.hf_processor 用白名单 match processor 类名：
  类名不在已知列表里 → raise ValueError → 外层 except 吞成 None
  → processor 退化为 None → 走 tokenizer 分支 → 静默丢多模态信息

本探针在写 configure_processor 之前测量：
  - 你的 processor 类名是否在 verl 白名单里？
  - 如果不在 → configure_processor 必须自建 processor（绕过白名单）
"""
import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def probe_processor_whitelist(model_path: str) -> dict:
    """检查 processor 类名是否在 verl 白名单里

    Args:
        model_path: 模型路径

    Returns:
        dict: 白名单检查结果
    """
    from transformers import AutoProcessor

    # 1. 获取 processor 类名
    processor = AutoProcessor.from_pretrained(model_path)
    processor_cls_name = type(processor).__name__

    # 2. 检查 verl 白名单
    try:
        from verl.utils.hf_processor import _HF_PROCESSOR_REGISTRY  # 或实际的变量名
        in_whitelist = processor_cls_name in _HF_PROCESSOR_REGISTRY
        whitelist_names = list(_HF_PROCESSOR_REGISTRY.keys()) if hasattr(_HF_PROCESSOR_REGISTRY, '__iter__') else []
    except ImportError:
        # verl 版本不同，白名单可能不是公开 API
        # 用硬编码的已知白名单
        known_whitelist = {
            "Qwen2AudioProcessor",
            "Qwen2VLProcessor",
            "Qwen2VLMProcessor",
            "AutoProcessor",
            "OmniProcessor",
        }
        in_whitelist = processor_cls_name in known_whitelist
        whitelist_names = list(known_whitelist)

    # 3. 检查关键方法
    has_get_rope_index = hasattr(processor, "get_rope_index")
    has_dedup_pad_tokens = hasattr(processor, "dedup_pad_tokens")

    result = {
        "processor_cls_name": processor_cls_name,
        "in_verl_whitelist": in_whitelist,
        "whitelist_names": whitelist_names,
        "has_get_rope_index": has_get_rope_index,
        "has_dedup_pad_tokens": has_dedup_pad_tokens,
        "needs_custom_configure": not in_whitelist,
    }

    if not in_whitelist:
        logger.warning(
            f"Processor {processor_cls_name!r} NOT in verl whitelist. "
            f"You MUST implement configure_processor() to bypass the whitelist. "
            f"Otherwise processor will silently become None and multimodal data will be lost."
        )

    logger.info(f"Processor whitelist check: {json.dumps(result, indent=2)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe processor whitelist")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    result = probe_processor_whitelist(args.model_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()