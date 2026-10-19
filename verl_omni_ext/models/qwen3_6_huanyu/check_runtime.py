"""CPU-only check of installed Huanyu imports, processor and native HF mRoPE.

Does not load model weights, generate text, modify checkpoints or start Ray.
"""

import argparse
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    for name in ("verl_omni_ext", "verl_omni", "vllm_omni"):
        spec = find_spec(name)
        print(f"{name} -> {spec.origin if spec else '<missing>'}", flush=True)
    from verl_omni_ext.models.qwen3_6_huanyu.audio_runtime import load_huanyu_runtime
    from verl_omni_ext.models.qwen3_6_huanyu.processor import build_huanyu_processor
    from verl_omni_ext.models.qwen3_6_huanyu.rollout_inputs import contract_rollout_prompt

    native = load_huanyu_runtime()
    print(f"Huanyu native audio -> {Path(native.__file__).resolve()}", flush=True)
    proc = build_huanyu_processor(args.model_path, SimpleNamespace(trust_remote_code=True))
    for seconds in (1, 31):
        waveform = np.zeros(seconds * proc.sampling_rate, dtype=np.float32)
        image = Image.new("RGB", (64, 64))
        text = proc.apply_chat_template([
            {"role": "user", "content": [
                {"type": "image"}, {"type": "audio"},
                {"type": "text", "text": "Describe the image and audio."},
            ]},
        ], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=text, images=[image], audios=[waveform], return_tensors="pt")
        positions, _ = proc.get_rope_index(
            input_ids=inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        expected = (3, 1, inputs["input_ids"].shape[-1])
        if tuple(positions.shape) != expected or positions.dtype != torch.long:
            raise ValueError(f"Native mRoPE returned {tuple(positions.shape)}/{positions.dtype}, expected {expected}/int64")
        proc.build_multi_modal_inputs(inputs["input_ids"], {"images": [image], "audios": [waveform]})
        contracted = contract_rollout_prompt(
            proc, inputs["input_ids"][0].tolist(), {"image": [image], "audio": [waveform]}
        )
        print(f"audio={seconds}s features={[tuple(x.shape) for x in inputs['huanyu_audio_features']]} "
              f"chunks={inputs['huanyu_audio_chunk_counts'].tolist()} "
              f"tokens={inputs['input_ids'].shape[-1]} rope={tuple(positions.shape)} "
              f"rollout_prompt_tokens={len(contracted)}", flush=True)
    print("CPU processor/import checks passed; NPU/FSDP/rollout numerical checks remain.", flush=True)


if __name__ == "__main__":
    main()
