"""Exercise Huanyu's shipped video reader without the MiniCPM-o package."""

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_video_decode_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
base = ModuleType("verl.utils.dataset.rl_dataset")
base.RLHFDataset = type("RLHFDataset", (), {})
with patch.dict(sys.modules, {base.__name__: base}):
    dataset = importlib.import_module(f"{PACKAGE}.dataset")


class VideoDecodeTests(unittest.TestCase):
    def test_training_and_precompute_decode_without_minicpmo_package(self):
        # Block the exact deployment dependency that was missing on the server.
        # Do not stub _video_helpers: both callers must reach the bundled reader.
        with patch.dict(sys.modules, {"verl_omni_ext.models.minicpmo_5_0": None}):
            helper = dataset._video_helpers()
            self.assertEqual(Path(helper.__file__).resolve(), ROOT / "video_decode.py")
            pixels = [Image.fromarray(np.full((4, 8, 3), value, dtype=np.uint8))
                      for value in (11, 23, 37)]
            item = {"video": "clip.mp4", "fps": 1.0, "max_frames": 3, "max_pixels": 32}

            def decode(spec, indices):
                self.assertEqual(spec.path, "clip.mp4")
                self.assertEqual(indices, [0, 30, 59])
                return [frame.copy() for frame in pixels]

            with patch.dict(os.environ, {"MINICPMO_USE_AUDIO_IN_VIDEO": "0"}), \
                    patch.object(helper, "_probe", return_value=(60, 20.0)), \
                    patch.object(helper, "_decode", side_effect=decode) as reader:
                messages = [{"role": "user", "content": [{"type": "video", **item}]}]
                images, videos, audios = dataset.Qwen36HuanyuOmniRLHFDataset._process_multi_modal_info(
                    messages, 16, {"mm_processor_kwargs": {"sampling_rate": 16000}},
                )
                cached_images, cached_videos, cached_audios = dataset.decode_huanyu_media({"videos": [item]})
                self.assertIsNone(images)
                self.assertIsNone(audios)
                self.assertEqual(cached_images, [])
                self.assertEqual(cached_audios, [])
                self.assertEqual(len(videos), 1)
                self.assertEqual(reader.call_count, 2)
                np.testing.assert_array_equal(videos[0], np.stack([np.asarray(f) for f in pixels]))
                np.testing.assert_array_equal(cached_videos[0], videos[0])
                self.assertEqual(dataset.video_prompt_parts(item, 16000), ["(<video>./</video>)"])

    def test_parquet_sampling_hints_override_environment_defaults(self):
        helper = dataset._video_helpers()
        with patch.dict(os.environ, {"MINICPMO_VIDEO_FPS": "2", "MINICPMO_VIDEO_MAX_FRAMES": "8",
                                     "MINICPMO_VIDEO_MAX_PIXELS": "200704"}):
            spec = helper.video_spec_from_item({"video": "clip.mp4", "fps": 1.0,
                                               "max_frames": 192, "max_pixels": 262144})
            self.assertEqual((spec.fps, spec.max_frames, spec.max_pixels), (1.0, 192, 262144))
            indices = helper.frame_indices(spec, 9000, 30.0)
            self.assertEqual(len(indices), 192)
            self.assertEqual((indices[0], indices[-1]), (0, 8999))
            defaults = helper.video_spec_from_item({"video": "other.mp4"})
            self.assertEqual((defaults.fps, defaults.max_frames, defaults.max_pixels), (2.0, 8, 200704))

    def test_av_reader_selects_requested_frames_and_closes_container(self):
        helper = dataset._video_helpers()
        frames = [Image.fromarray(np.full((2, 2, 3), i, dtype=np.uint8)) for i in range(4)]
        stream = SimpleNamespace(frames=4, average_rate=2, guessed_rate=None)
        closed = []
        container = SimpleNamespace(
            streams=SimpleNamespace(video=[stream]), duration=2000000,
            decode=lambda selected: iter([SimpleNamespace(to_image=lambda f=f: f) for f in frames]),
            close=lambda: closed.append(True),
        )
        av = ModuleType("av")
        av.open = lambda path: container
        spec = helper.VideoSpec("clip.mp4", 1.0, 2, 4)
        with patch.dict(os.environ, {"MINICPMO_VIDEO_BACKEND": "av"}), patch.dict(sys.modules, {"av": av}):
            decoded = helper.sample_frames(spec)
        self.assertEqual([np.asarray(frame)[0, 0, 0] for frame in decoded], [0, 3])
        self.assertEqual(len(closed), 2)


if __name__ == "__main__":
    unittest.main()
