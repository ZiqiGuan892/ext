"""Raw-media dataset for Huanyu; no Whisper embedding cache is consumed."""

from io import BytesIO
import os

import numpy as np
from verl.utils.dataset.rl_dataset import RLHFDataset


def _video_helpers():
    # Media decoding is independent of the audio encoder. Keep the exact same
    # frame sampler for message expansion and actual frame extraction.
    from verl_omni_ext.models.minicpmo_5_0 import minicpmo_5_0_video
    return minicpmo_5_0_video


def video_prompt_parts(item, sampling_rate):
    from .processor import AUDIO_PATTERN, IMAGE_PATTERN
    helper = _video_helpers()
    spec = helper.video_spec_from_item(item)
    parts = [IMAGE_PATTERN] * helper.count_frames(spec)
    if helper.use_audio_in_video() and helper.extract_video_audio(spec, sampling_rate) is not None:
        parts.append(AUDIO_PATTERN)
    return parts


def _image(item):
    from PIL import Image
    value = item.get("image")
    if value is None and item.get("bytes") is not None:
        value = BytesIO(item["bytes"])
    if value is None:
        raise ValueError("Huanyu image message has no image payload")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    with Image.open(value) as source:
        return source.convert("RGB")


def _audio(item, sampling_rate):
    value = item.get("audio")
    if value is None:
        value = item.get("audio_url")
    if value is None:
        value = item.get("bytes")
    if value is None:
        raise ValueError("Huanyu audio message has no audio payload")
    if isinstance(value, np.ndarray):
        actual_rate = int(item.get("sampling_rate", sampling_rate))
        array = value
    elif isinstance(value, str) and "::" in value:
        import zarr
        archive_path, member_id = value.split("::", 1)
        store = zarr.storage.ZipStore(archive_path, mode="r")
        try:
            group = zarr.open(store=store, mode="r")[f"{member_id}.wav"]
            array = np.asarray(group["audio"], dtype=np.float32)
            actual_rate = int(group.attrs.get("sampling_rate", group.attrs.get("sample_rate", sampling_rate)))
        finally:
            store.close()
    else:
        import librosa
        source = BytesIO(value) if isinstance(value, (bytes, bytearray)) else os.fspath(value)
        array, actual_rate = librosa.load(source, sr=sampling_rate, mono=True)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=0 if array.shape[0] < array.shape[1] else 1)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Huanyu audio must decode to finite mono samples")
    if actual_rate != sampling_rate:
        import librosa
        array = librosa.resample(array, orig_sr=actual_rate, target_sr=sampling_rate)
    return np.ascontiguousarray(array, dtype=np.float32)


class Qwen36HuanyuOmniRLHFDataset(RLHFDataset):
    """Keep upstream schema and decode images/audio/video without old caches."""

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        kwargs = config.get("mm_processor_kwargs", {}) if config is not None else {}
        sampling_rate = int((kwargs or {}).get("sampling_rate", 16000))
        if sampling_rate != 16000:
            raise ValueError("Verified Huanyu checkpoint expects sampling_rate=16000")
        images, audios = [], []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                kind = item.get("type")
                if kind == "image":
                    images.append(_image(item))
                elif kind == "audio":
                    audios.append(_audio(item, sampling_rate))
                elif kind == "video":
                    helper = _video_helpers()
                    spec = helper.video_spec_from_item(item)
                    images.extend(helper.sample_frames(spec))
                    if helper.use_audio_in_video():
                        waveform = helper.extract_video_audio(spec, sampling_rate)
                        if waveform is not None:
                            audios.append(waveform)
        return images or None, None, audios or None
