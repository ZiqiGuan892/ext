"""Huanyu audio preprocessing, based on the supplied processing_huanyu.py.

This component has no vLLM or Transformers dependency. It does not instantiate
the encoder/projector or change any existing processor class. Integration into
the full multimodal training path is separate.
"""

from operator import index
from pathlib import Path

import numpy as np
import torch


SAMPLING_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SAMPLES = 30 * SAMPLING_RATE


def fbank_frame_count(num_samples: int) -> int:
    num_samples = index(num_samples)
    if num_samples < N_FFT:
        raise ValueError(f"Audio too short: {num_samples} samples < n_fft={N_FFT}")
    return (num_samples - N_FFT) // HOP_LENGTH + 1


def tokens_from_frames(frames: int) -> int:
    frames = index(frames)
    if frames < 1:
        raise ValueError("Expected a positive, unpadded fbank frame count")
    return ((frames + 3) // 4) // 2


def chunk_sample_counts(num_samples: int, *, chunk_input: bool = True) -> tuple[int, ...]:
    """Preserve every sample; reject tails with no computable fbank frame."""
    num_samples = index(num_samples)
    fbank_frame_count(num_samples)
    if not chunk_input:
        return (num_samples,)
    full, remainder = divmod(num_samples, CHUNK_SAMPLES)
    counts = (CHUNK_SAMPLES,) * full + ((remainder,) if remainder else ())
    for i, count in enumerate(counts):
        if count < N_FFT:
            raise ValueError(
                f"Audio chunk {i} has only {count} samples (<{N_FFT}); "
                "the supplied Huanyu frontend cannot extract a frame. "
                "No tail was dropped or silently padded."
            )
    return counts


def chunk_token_counts(num_samples: int, *, chunk_input: bool = True) -> tuple[int, ...]:
    return tuple(
        tokens_from_frames(fbank_frame_count(count))
        for count in chunk_sample_counts(num_samples, chunk_input=chunk_input)
    )


def audio_placeholder(
    num_samples: int,
    *,
    audio_start: str,
    audio_end: str,
    placeholder: str = "<unk>",
    chunk_input: bool = True,
) -> str:
    """Use the same segmentation as feature extraction, one span per chunk.

    Very short chunks can have valid fbank frames but zero encoder tokens. The
    source processor can produce an empty span (or omit it in chunk mode). This
    training component rejects that ambiguous case pending an explicit model
    policy instead of dropping the audio or inventing a token.
    """
    counts = chunk_token_counts(num_samples, chunk_input=chunk_input)
    _require_nonempty_tokens(counts)
    if not audio_start or not audio_end or not placeholder:
        raise ValueError("Audio boundary and placeholder strings must be non-empty")
    return "".join(audio_start + placeholder * count + audio_end for count in counts)


def _require_nonempty_tokens(counts) -> None:
    for i, count in enumerate(counts):
        if count <= 0:
            raise ValueError(
                f"Audio chunk {i} produces zero Huanyu tokens; "
                "an explicit short-audio policy is required. No audio was dropped."
            )


class HuanyuAudioFrontend:
    """128-bin fbank: periodic Hann, center=False, power spectrum, log10.

    ``mel_filter_path`` must identify the exported SFT matrix, shape [201,128].
    Do not regenerate it with a library default: the matrix is part of the
    preprocessing contract. ``from_pretrained`` resolves it under the selected
    checkpoint, avoiding the server processor's hardcoded checkpoint path.
    """

    def __init__(self, mel_filter_path):
        self.mel_filter_path = Path(mel_filter_path).resolve(strict=True)
        filters = torch.load(self.mel_filter_path, map_location="cpu", weights_only=True)
        filters = torch.as_tensor(filters, dtype=torch.float32)
        if tuple(filters.shape) != (201, N_MELS) or not torch.isfinite(filters).all():
            raise ValueError("Huanyu mel filters must be finite float32 data with shape [201,128]")
        self._mel_filters = filters.contiguous().numpy().copy()
        self._window = (
            0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
        ).astype(np.float32)

    @classmethod
    def from_pretrained(cls, model_path):
        return cls(Path(model_path) / "fb128_mel_filters.pt")

    @staticmethod
    def _waveform(audio):
        if np.iscomplexobj(audio):
            raise ValueError("Expected a real mono waveform")
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("Expected a finite mono waveform with shape [samples]")
        fbank_frame_count(len(waveform))
        return np.ascontiguousarray(waveform)

    def extract(self, waveform, *, sampling_rate=SAMPLING_RATE) -> torch.Tensor:
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(f"Huanyu expects {SAMPLING_RATE} Hz; resample before this frontend")
        x = self._waveform(waveform)
        # This is the same frame/window/FFT arithmetic as the supplied source;
        # sliding_window_view gives a bounds-checked view instead of as_strided.
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP_LENGTH]
        spectrum = np.fft.rfft(frames * self._window[None, :], axis=1)
        magnitudes = (np.abs(spectrum) ** 2).astype(np.float32)
        mel = magnitudes @ self._mel_filters
        mel = (np.log10(np.clip(mel, 1e-10, None)) + 4.0) / 4.0
        return torch.from_numpy(np.ascontiguousarray(mel.T, dtype=np.float32))

    def prepare(self, audios, *, sampling_rate=SAMPLING_RATE, chunk_input=True) -> dict:
        """Process one prompt's flat list of raw audio waveforms.

        Returns features [num_chunks,128,padded_frames], nested per-prompt true
        lengths (matching the supplied processor), token counts for alignment,
        and a chunk count per original audio. Padding never enters token counts.
        Extra alignment metadata must not be passed blindly to the HF model.
        """
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(f"Huanyu expects {SAMPLING_RATE} Hz; resample before this frontend")
        if isinstance(audios, np.ndarray):
            audios = [audios]
        features, lengths, token_lengths, groups = [], [], [], []
        for audio in audios:
            waveform = self._waveform(audio)
            sample_counts = chunk_sample_counts(len(waveform), chunk_input=chunk_input)
            counts = tuple(tokens_from_frames(fbank_frame_count(n)) for n in sample_counts)
            _require_nonempty_tokens(counts)
            groups.append(len(sample_counts))
            offset = 0
            for n, tokens in zip(sample_counts, counts, strict=True):
                feature = self.extract(waveform[offset:offset + n], sampling_rate=sampling_rate)
                length = feature.shape[-1]
                features.append(feature)
                lengths.append(length)
                token_lengths.append(tokens)
                offset += n
        if not features:
            raise ValueError("No audio supplied; omit the audio branch for text/image-only samples")
        pad_to = (max(lengths) + 3) // 4 * 4
        padded = torch.stack([
            torch.nn.functional.pad(feature, (0, pad_to - feature.shape[-1]))
            for feature in features
        ])
        return {
            "audio_features": padded,
            "audio_feature_lens": [torch.tensor(lengths, dtype=torch.int64)],
            "audio_token_lens": tuple(token_lengths),
            "audio_chunks_per_input": tuple(groups),
        }
