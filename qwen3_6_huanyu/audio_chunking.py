"""Shared waveform/length policy for Huanyu training and RL rollout.

Only a nonempty chunk producing zero tokens is extended with trailing silence.
All original samples and the existing 30-second boundaries are preserved. The
extension supplies a complete encoder/projector frame group, rather than fake
lengths or an invented placeholder. Empty/invalid waveforms still fail.
"""

from dataclasses import dataclass
from operator import index


AUDIO_CHUNK_POLICY = "right_zero_pad_full_token_v1"


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Huanyu {name} must be a positive integer")
    value = index(value)
    if value <= 0:
        raise ValueError(f"Huanyu {name} must be a positive integer")
    return value


def audio_chunking_contract(*, sampling_rate=16000, n_fft=400, hop_length=160,
                            chunk_seconds=30, encoder_downsample=4, projector_downsample=2):
    values = dict(sampling_rate=sampling_rate, n_fft=n_fft, hop_length=hop_length,
                  chunk_seconds=chunk_seconds, encoder_downsample=encoder_downsample,
                  projector_downsample=projector_downsample)
    values = {name: _positive(value, name) for name, value in values.items()}
    full_token_frames = values["encoder_downsample"] * values["projector_downsample"]
    padded_samples = values["n_fft"] + (full_token_frames - 1) * values["hop_length"]
    if padded_samples > values["sampling_rate"] * values["chunk_seconds"]:
        raise ValueError("Huanyu chunk duration cannot accommodate one complete audio token")
    return {"policy": AUDIO_CHUNK_POLICY, **values,
            "short_chunk_target_samples": padded_samples,
            "standalone_short_audio": "pad_nonempty", "empty_audio": "reject",
            "frame_lengths": "after_waveform_padding_before_feature_batch_padding"}


@dataclass(frozen=True)
class AudioChunk:
    offset: int
    original_samples: int
    processed_samples: int
    frames: int
    tokens: int

    @property
    def padding_samples(self):
        return self.processed_samples - self.original_samples


def plan_audio_chunks(num_samples, *, sampling_rate=16000, n_fft=400, hop_length=160,
                      chunk_seconds=30, encoder_downsample=4, projector_downsample=2):
    contract = audio_chunking_contract(
        sampling_rate=sampling_rate, n_fft=n_fft, hop_length=hop_length,
        chunk_seconds=chunk_seconds, encoder_downsample=encoder_downsample,
        projector_downsample=projector_downsample,
    )
    if isinstance(num_samples, bool):
        raise ValueError("Huanyu audio sample count must be an integer")
    num_samples = index(num_samples)
    if num_samples <= 0:
        raise ValueError("Huanyu audio waveform is empty" if num_samples == 0 else "Negative Huanyu audio sample count")
    limit = contract["chunk_seconds"] * contract["sampling_rate"]
    fft, hop = contract["n_fft"], contract["hop_length"]
    encoder, projector = contract["encoder_downsample"], contract["projector_downsample"]
    chunks = []
    for offset in range(0, num_samples, limit):
        original = min(limit, num_samples - offset)
        processed = original
        frames = (processed - fft) // hop + 1
        tokens = ((frames + encoder - 1) // encoder) // projector
        if frames < 1 or tokens < 1:
            processed = contract["short_chunk_target_samples"]
            frames = (processed - fft) // hop + 1
            tokens = ((frames + encoder - 1) // encoder) // projector
        chunks.append(AudioChunk(offset, original, processed, frames, tokens))
    return tuple(chunks)


def prepare_audio_waveform(waveform, **kwargs):
    """Return finite float32 mono audio and its explicit per-chunk plan.

    No input array is modified. The only possible padded chunk is the final
    one, so padding the source waveform once gives native rollout's fixed
    chunker exactly the same slices as the training processor.
    """
    import numpy as np

    if np.iscomplexobj(waveform):
        raise ValueError("Huanyu audio must be a real finite mono waveform [samples]")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Huanyu audio must be a finite mono waveform [samples]")
    plan = plan_audio_chunks(len(waveform), **kwargs)
    padding = sum(chunk.padding_samples for chunk in plan)
    if padding:
        waveform = np.pad(waveform, (0, padding), mode="constant")
    return np.ascontiguousarray(waveform), plan
