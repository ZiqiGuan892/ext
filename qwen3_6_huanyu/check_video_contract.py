"""Check the installed checkpoint's paired-video pixels, tokens and M-RoPE.

Run directly in the training container. This loads configs/tokenizer only,
not model weights, and does not connect to Ray. --native also imports the
installed vLLM-Omni and exercises its patched processing/position boundaries;
it is not a substitute for a real request through a running engine.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace

import numpy as np
import torch


if not __package__:
    _package = "_huanyu_video_check"
    _module = ModuleType(_package)
    _module.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package] = _module
    __package__ = _package

from .processor import HuanyuProcessor, VIDEO_PATTERN, _bind_qwen_rope
from .video_frontend import video_placeholder


def check(model_path, *, native=False):
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    frontend = Qwen2VLImageProcessor.from_pretrained(model_path)
    proc = HuanyuProcessor(tokenizer, frontend, None, config, {"sampling_rate": 16000, "feature_size": 40})
    _bind_qwen_rope(proc)
    # Distinct colors expose accidental AA duplication even after resizing.
    frames = np.zeros((8, 448, 448, 3), dtype=np.uint8)
    frames[::2, :, :, 0] = 255
    frames[1::2, :, :, 1] = 255
    temporal = frontend.temporal_patch_size
    if temporal != 2:
        raise ValueError(f"This A/B two-frame check expects temporal_patch_size=2, checkpoint declares {temporal}")
    outputs = []
    for count in (1, 2, 3, 8):
        clip = frames[:count]
        inputs = proc(VIDEO_PATTERN, videos=[clip])
        grid = inputs["video_grid_thw"][0]
        packed = inputs["pixel_values_videos"]
        slots = packed.reshape(-1, 3, temporal, frontend.patch_size, frontend.patch_size)
        if count > 1 and torch.equal(slots[:, :, 0], slots[:, :, 1]):
            raise AssertionError("Different input frames were duplicated into AA temporal slots")
        if count == 1:
            torch.testing.assert_close(slots[:, :, 0], slots[:, :, 1], rtol=0, atol=0)
        single = proc._vision(list(clip))
        old_tokens = int(single["image_grid_thw"].prod(-1).sum()) // proc.spatial_merge_size ** 2
        new_tokens = int(grid.prod()) // proc.spatial_merge_size ** 2
        assert new_tokens * count == old_tokens * ((count + 1) // 2)
        assert int((inputs["input_ids"] == proc.video_token_id).sum()) == new_tokens
        if native:
            verify_native(proc, clip, inputs, config)
        outputs.append({"frames": count, "grid": grid.tolist(), "old_visual_tokens": old_tokens,
                        "paired_visual_tokens": new_tokens, "native_boundaries_checked": native})
    print(json.dumps({"status": "PASS", "checkpoint": str(Path(model_path).resolve()),
                      "checks": outputs, "weights_loaded": False}, ensure_ascii=False, indent=2))


def verify_native(proc, clip, inputs, config):
    from .video_rollout import _NATIVE, install_huanyu_video_rollout

    install_huanyu_video_rollout()
    native = importlib.import_module(_NATIVE)
    # Adapt only the container of parsed media; run the actual installed
    # processor method and actual native M-RoPE implementation, without init.
    parsed = SimpleNamespace(get=lambda index: (clip, {}))
    class Parsed:
        def __len__(self):
            return 1

        get = staticmethod(parsed.get)

    info = SimpleNamespace(get_hf_config=lambda: config,
                           ctx=SimpleNamespace(model_config=SimpleNamespace(model=config._name_or_path)))
    runner = SimpleNamespace(info=info, data_parser=SimpleNamespace(
        parse_mm_data=lambda data: SimpleNamespace(get_items=lambda *args: Parsed())))
    processor_cls = native.MiniCPMO50OmniLLMMultiModalProcessor
    actual = processor_cls.process_videos(runner, {"videos": [clip]}, {}, {})
    for key in ("pixel_values_videos", "video_grid_thw"):
        torch.testing.assert_close(actual[key], inputs[key], rtol=0, atol=0)

    ids = inputs["input_ids"][0].tolist()
    grid = inputs["video_grid_thw"][0]
    # Real PromptReplacement expansion is exercised by CPU integration tests;
    # this verifies the installed tokenizer agrees with the shared rendering.
    rendered = video_placeholder(grid, proc.spatial_merge_size)
    assert tokenizer_ids(proc.tokenizer, rendered) == ids
    feature = SimpleNamespace(modality="video", data={"video_grid_thw": SimpleNamespace(data=grid)},
                              mm_position=SimpleNamespace(offset=0, length=len(ids)))
    model_cls = native.MiniCPMO50OmniLLMForConditionalGeneration
    model = SimpleNamespace(config=config, vision_start_token_id=proc.vision_start_token_id,
                            vision_end_token_id=proc.vision_end_token_id, video_token_id=proc.video_token_id)
    model._q36_iter_mm_grid_hw = MethodType(model_cls._q36_iter_mm_grid_hw, model)
    actual_positions, actual_delta = model_cls.get_mrope_input_positions(model, ids, [feature])
    expected_positions, expected_delta = proc.get_rope_index(
        input_ids=inputs["input_ids"], video_grid_thw=inputs["video_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    torch.testing.assert_close(actual_positions.cpu(), expected_positions[:, 0].cpu(), rtol=0, atol=0)
    assert actual_delta == int(expected_delta.reshape(-1)[0])


def tokenizer_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native", action="store_true", help="Also check installed vLLM-Omni boundary methods")
    args = parser.parse_args()
    check(args.model, native=args.native)
