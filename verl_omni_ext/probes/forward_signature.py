"""
Forward 签名探针

在写 configure_model 的 forward 适配器之前，测量模型的 forward 签名：
  - 有没有 labels 参数？
  - position_ids 是位置参数还是 kwargs？
  - 返回值是 tuple 还是 ModelOutput？

测量结果决定 forward 适配器的写法——不能猜。
"""
import argparse
import inspect
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def probe_forward_signature(model_path: str, architecture: str = None) -> dict:
    """测量模型 forward 签名

    Args:
        model_path: 模型路径
        architecture: 可选的 architecture 名（从 config.json 读取如果不指定）

    Returns:
        dict: forward 签名信息
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    arch = architecture or config.architectures[0]
    logger.info(f"Probing forward signature for: {arch}")

    # 尝试 import 模型类
    try:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception:
        # 用 trust_remote_code
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_cls = get_class_from_dynamic_module(arch, model_path)

    # 获取 forward 签名
    sig = inspect.signature(model_cls.forward)
    params = {}
    for name, param in sig.parameters.items():
        params[name] = {
            "kind": str(param.kind),
            "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
        }

    # 返回值注解
    return_annotation = str(sig.return_annotation) if sig.return_annotation is not inspect.Signature.empty else None

    result = {
        "architecture": arch,
        "forward_params": params,
        "return_annotation": return_annotation,
        "has_labels": "labels" in params,
        "has_position_ids": "position_ids" in params,
        "has_attention_mask": "attention_mask" in params,
    }

    logger.info(f"Forward signature: {json.dumps(result, indent=2)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe model forward signature")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--architecture", default=None, help="Architecture name (auto from config if not specified)")
    args = parser.parse_args()

    result = probe_forward_signature(args.model_path, args.architecture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()