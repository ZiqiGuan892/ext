"""Keep native Huanyu audio math while isolating each real chunk's shape."""

from functools import wraps
import os
from types import MethodType

import torch


def install_audio_isolation(native):
    if os.environ.get("MINICPMO_ISOLATE_MEDIA_ENCODERS", "1") != "1":
        print("[huanyu-audio] component=rollout_encoder phase=disabled", flush=True)
        return False
    thinker = getattr(native, "thinker", native)
    original = getattr(thinker, "get_audio_hidden_states", None)
    if not callable(original):
        raise RuntimeError("Huanyu audio isolation requires native get_audio_hidden_states")
    if getattr(original, "_huanyu_isolated_audio", False):
        return True
    reported = False

    @wraps(original)
    def audio(self, data):
        nonlocal reported
        raw = data["audio_features"]
        chunks = list(raw.unbind(0)) if isinstance(raw, torch.Tensor) else list(raw)
        raw_lengths = data["audio_feature_lens"]
        groups = list(raw_lengths.unbind(0)) if isinstance(raw_lengths, torch.Tensor) else list(raw_lengths)
        lengths = [torch.as_tensor(group).reshape(-1) for group in groups]
        flat = torch.cat(lengths) if lengths else torch.empty(0, dtype=torch.long)
        if len(chunks) != len(flat):
            raise ValueError("Huanyu rollout feature/chunk length count mismatch")
        outputs = []
        for chunk, length_tensor in zip(chunks, flat):
            length = int(length_tensor.item())
            if length <= 0 or length > chunk.shape[-1]:
                raise ValueError(f"Invalid Huanyu rollout frame count {length} for {tuple(chunk.shape)}")
            # Native Huanyu owns VGG padding, Conformer, fold and valid lengths.
            # Crop only cross-item padding, as the Whisper MR's audio_items does.
            result = original({"audio_features": [chunk[..., :length]],
                               "audio_feature_lens": [length_tensor.reshape(1)]})
            if len(result) != 1:
                raise RuntimeError("Native Huanyu audio must return one group for one input chunk")
            outputs.append(result[0])
        regrouped, start = [], 0
        for group in lengths:
            stop = start + len(group)
            if stop == start:
                raise ValueError("Huanyu rollout audio groups must contain at least one chunk")
            regrouped.append(torch.cat(outputs[start:stop], dim=0))
            start = stop
        if outputs and not reported:
            reported = True
            print(f"[huanyu-audio] component=rollout_encoder phase=executed pid={os.getpid()} "
                  f"chunks={len(outputs)} groups={len(regrouped)} dtype={outputs[0].dtype} "
                  "native_huanyu=True per_chunk_shape=True", flush=True)
        return regrouped

    audio._huanyu_isolated_audio = True
    thinker.get_audio_hidden_states = MethodType(audio, thinker)
    print(f"[huanyu-audio] component=rollout_encoder phase=installed pid={os.getpid()} "
          "native_huanyu=True per_chunk_shape=True", flush=True)
    return True
