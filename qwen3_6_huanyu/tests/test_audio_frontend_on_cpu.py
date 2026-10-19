"""Numerical and length checks against the supplied Huanyu preprocessing rules."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_frontend_test", ROOT / "audio_frontend.py")
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)


class LengthTests(unittest.TestCase):
    def test_known_duration_and_padding_cases(self):
        # 3 s => 298 true frames, 300 padded frames, 37 output tokens.
        self.assertEqual(frontend.fbank_frame_count(48000), 298)
        self.assertEqual(frontend.chunk_token_counts(48000), (37,))
        self.assertEqual(frontend.chunk_token_counts(480000), (375,))
        self.assertEqual(frontend.chunk_token_counts(960000), (375, 375))
        # Padding 5 true frames to 8 must not change its token count from 1.
        self.assertEqual(frontend.tokens_from_frames(5), 1)
        self.assertEqual(frontend.tokens_from_frames(8), 1)

    def test_placeholder_and_features_share_chunk_boundaries(self):
        text = frontend.audio_placeholder(
            960000, audio_start="<a>", audio_end="</a>", placeholder="X"
        )
        self.assertEqual(text, ("<a>" + "X" * 375 + "</a>") * 2)

    def test_no_silent_tail_dropping_or_zero_token_audio(self):
        with self.assertRaisesRegex(ValueError, "only 200 samples"):
            frontend.chunk_sample_counts(480200)
        with self.assertRaisesRegex(ValueError, "zero Huanyu tokens"):
            frontend.audio_placeholder(400, audio_start="<a>", audio_end="</a>")


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.filters = torch.linspace(0.001, 0.01, 201 * 128).reshape(201, 128)
        torch.save(self.filters, self.root / "fb128_mel_filters.pt")
        self.frontend = frontend.HuanyuAudioFrontend.from_pretrained(self.root)

    def test_matches_reference_stft_without_center_padding(self):
        # Independently express the supplied NumPy FFT frontend through STFT.
        samples = torch.arange(3200, dtype=torch.float32)
        waveform = (0.3 * torch.sin(samples * 0.12) + 0.1 * torch.cos(samples * 0.31)).numpy()
        features = self.frontend.extract(waveform)
        spectrum = torch.stft(
            torch.from_numpy(waveform), n_fft=400, hop_length=160,
            window=torch.hann_window(400, periodic=True), center=False, return_complex=True,
        )
        expected = (torch.log10((spectrum.abs().square().T @ self.filters).clamp_min(1e-10)) + 4) / 4
        torch.testing.assert_close(features, expected.T, rtol=1e-5, atol=2e-6)

    def test_heterogeneous_batch_keeps_true_lengths(self):
        result = self.frontend.prepare([np.zeros(48000, np.float32), np.zeros(16000, np.float32)])
        self.assertEqual(tuple(result["audio_features"].shape), (2, 128, 300))
        self.assertEqual(result["audio_feature_lens"][0].tolist(), [298, 98])
        self.assertEqual(result["audio_token_lens"], (37, 12))
        self.assertEqual(result["audio_chunks_per_input"], (1, 1))
        self.assertEqual(result["audio_features"][1, :, 98:].count_nonzero().item(), 0)

    def test_rejects_invalid_waveform_and_rate(self):
        for waveform in (np.zeros((2, 1600)), np.full(1600, np.nan)):
            with self.assertRaisesRegex(ValueError, "finite mono waveform"):
                self.frontend.extract(waveform)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            self.frontend.prepare([np.zeros(1600)], sampling_rate=8000)

    def test_missing_or_wrong_filter_matrix_does_not_fallback(self):
        with self.assertRaises(FileNotFoundError):
            frontend.HuanyuAudioFrontend.from_pretrained(self.root / "missing")
        torch.save(torch.zeros(201, 80), self.root / "fb128_mel_filters.pt")
        with self.assertRaisesRegex(ValueError, "shape"):
            frontend.HuanyuAudioFrontend.from_pretrained(self.root)


if __name__ == "__main__":
    unittest.main()
