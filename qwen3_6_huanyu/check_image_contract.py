"""Check installed vLLM image prompt updates without model weights or Ray.

Uses synthetic images and the checkpoint tokenizer/frontend. This validates
token expansion and embedding ranges, not NPU execution or end-to-end replay.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


if not __package__:
    _package = "_huanyu_image_check"
    _module = ModuleType(_package)
    _module.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package] = _module
    __package__ = _package

from .processor import HuanyuProcessor, IMAGE_PATTERN
from .video_rollout import _NATIVE, install_huanyu_video_rollout


def verify_updates(proc, native, config, inputs):
    """Run the installed updater for both compact spellings and expanded IDs."""
    boundary = object.__new__(native.MiniCPMO50OmniLLMMultiModalProcessor)
    boundary.info = SimpleNamespace(
        get_hf_config=lambda: config, get_tokenizer=lambda: proc.tokenizer,
        audio_pattern="(<audio>./</audio>)",
        ctx=SimpleNamespace(model_config=SimpleNamespace(model=config._name_or_path)),
    )
    grids = inputs["image_grid_thw"]
    expected_ids = inputs["input_ids"][0].tolist()
    out = {"image": [{"image_grid_thw": SimpleNamespace(data=g)} for g in grids]}
    updates = boundary._get_prompt_updates({}, {}, out)
    resolved = boundary._bind_and_group_updates(updates, {"image": len(grids)})
    cases = []
    for target in ("<|vision_start|><|image_pad|><|vision_end|>", IMAGE_PATTERN):
        text = "check " + " next ".join([target] * len(grids)) + " end"
        cases.append((target, proc.tokenizer.encode(text, add_special_tokens=False)))
    cases.append(("preexpanded", expected_ids))
    for label, source_ids in cases:
        ids, placeholders = boundary._apply_prompt_updates(source_ids, resolved)
        if list(ids) != expected_ids:
            raise AssertionError(f"Image prompt update differs from training for {label!r}: "
                                 f"rollout={len(ids)} training={len(expected_ids)}")
        images = placeholders.get("image", [])
        if len(images) != len(grids):
            raise AssertionError("Image prompt update returned the wrong number of media items")
        last_end = 0
        for item, grid in zip(images, grids):
            position = item.to_range()
            count = int(grid.prod()) // proc.spatial_merge_size**2
            expected_span = [proc.vision_start_token_id] + [proc.image_token_id] * count + [proc.vision_end_token_id]
            begin, end = position.offset, position.offset + position.length
            if begin < last_end or list(ids[begin:end]) != expected_span:
                raise AssertionError("Native image range lost boundaries or overlaps another item")
            mask = getattr(position, "is_embed", None)
            if mask is None:
                raise AssertionError("Native image range lost its embedding mask")
            torch.testing.assert_close(torch.as_tensor(mask).cpu(), torch.tensor(expected_span) == proc.image_token_id)
            last_end = end


def check(model_path):
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    frontend = Qwen2VLImageProcessor.from_pretrained(model_path)
    proc = HuanyuProcessor(tokenizer, frontend, None, config, {"sampling_rate": 16000, "feature_size": 40})
    install_huanyu_video_rollout()
    native = importlib.import_module(_NATIVE)
    images = [np.zeros((224, 224, 3), dtype=np.uint8), np.full((336, 224, 3), 127, dtype=np.uint8)]
    results = []
    for count in (1, 2):
        text = "check " + " next ".join([IMAGE_PATTERN] * count) + " end"
        inputs = proc(text, images=images[:count])
        verify_updates(proc, native, config, inputs)
        results.append({"images": count, "prompt_tokens": inputs["input_ids"].shape[1],
                        "grid_thw": inputs["image_grid_thw"].tolist()})
    print(json.dumps({"status": "PASS", "checks": results, "weights_loaded": False,
                      "checked": "native_image_prompt_ids_ranges_and_embedding_masks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    check(parser.parse_args().model)
