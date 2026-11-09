"""Check real temporal pixel grouping without model weights or a video decoder."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_video_frontend_test", ROOT / "video_frontend.py")
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)


class ImageProcessor:
    """Small RGB image frontend with Qwen's public patch layout contract."""

    patch_size = 2
    temporal_patch_size = 2
    merge_size = 2

    def __init__(self):
        self.calls = []
        self.outputs = []

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        self.calls.append(list(images))
        rows, grids = [], []
        patch, merge = self.patch_size, self.merge_size
        for frame in images:
            frame = torch.as_tensor(np.array(frame)).to(torch.float64) / 127.5 - 1
            height, width, _ = frame.shape
            grid_h, grid_w = height // patch, width // patch
            grids.append([1, grid_h, grid_w])
            # Group spatial patches in merge blocks, as the image processor
            # does. Use explicit coordinates instead of the production reshape.
            for block_y in range(0, grid_h, merge):
                for block_x in range(0, grid_w, merge):
                    for y in range(block_y, block_y + merge):
                        for x in range(block_x, block_x + merge):
                            channels = []
                            for channel in range(3):
                                cell = frame[y * patch:(y + 1) * patch, x * patch:(x + 1) * patch, channel]
                                channels.extend([cell.flatten()] * self.temporal_patch_size)
                            rows.append(torch.cat(channels))
        result = {"pixel_values": torch.stack(rows), "image_grid_thw": torch.tensor(grids)}
        self.outputs.append({key: value.clone() for key, value in result.items()})
        self.last_result = result
        return result


def rgb_frame(seed, height=8, width=8):
    return (np.arange(height * width * 3).reshape(height, width, 3) + seed).astype(np.uint8)


class VideoFrontendTests(unittest.TestCase):
    def test_size_geometry_measures_one_frame_once_and_keeps_temporal_count_separate(self):
        proc = ImageProcessor()
        self.assertEqual(frontend.video_grid_for_size(proc, 8, 4, 7), [4, 4, 2])
        self.assertEqual(frontend.video_grid_for_size(proc, 8, 4, 1), [1, 4, 2])
        self.assertEqual(len(proc.calls), 1)
        self.assertEqual(len(proc.calls[0]), 1)
        self.assertEqual(frontend.video_grid_for_size(proc, 4, 8, 2), [1, 2, 4])
        self.assertEqual(len(proc.calls), 2)
        for height, width, count in ((0, 8, 1), (8, -1, 1), (8, 8, 0), (True, 8, 1)):
            with self.subTest(height=height, width=width, count=count), self.assertRaises(ValueError):
                frontend.video_grid_for_size(proc, height, width, count)

    def test_placeholder_uses_one_bounded_span_per_temporal_group(self):
        span = "<|vision_start|>" + "<|video_pad|>" * 4 + "<|vision_end|>"
        self.assertEqual(frontend.video_placeholder(torch.tensor([[3, 4, 4]]), 2), span * 3)
        for grid, merge in (([1, 3, 4], 2), ([0, 4, 4], 2), ([1.0, 4, 4], 2), ([1, 4], 2), ([1, 4, 4], True)):
            with self.subTest(grid=grid, merge=merge), self.assertRaises(ValueError):
                frontend.video_placeholder(grid, merge)

    def test_two_real_frames_keep_channels_and_spatial_patches(self):
        proc = ImageProcessor()
        frames = [rgb_frame(7), rgb_frame(53)]
        result = frontend.preprocess_video_frames(proc, [frames])
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 4, 4]])
        pixels = result["pixel_values_videos"]
        self.assertEqual(pixels.shape, (16, 24))
        self.assertEqual(pixels.dtype, torch.float64)
        cells = pixels.reshape(16, 3, 2, 2, 2)
        # Probe across a merge-block boundary, where raster ordering differs.
        for row, (y, x) in {0: (0, 0), 3: (1, 1), 4: (0, 2), 8: (2, 0), 15: (3, 3)}.items():
            for time, frame in enumerate(frames):
                expected = torch.tensor(frame[y * 2:y * 2 + 2, x * 2:x * 2 + 2], dtype=torch.float64)
                expected = expected.permute(2, 0, 1) / 127.5 - 1
                torch.testing.assert_close(cells[row, :, time], expected)
        self.assertGreater((cells[:, :, 0] - cells[:, :, 1]).abs().max().item(), 0)
        self.assertEqual(len(proc.calls), 1)
        self.assertEqual(len(proc.calls[0]), 2)

    def test_odd_tail_repeats_only_last_frame_and_single_frame_is_unchanged(self):
        proc = ImageProcessor()
        frames = [rgb_frame(0), rgb_frame(10), rgb_frame(20)]
        result = frontend.preprocess_video_frames(proc, [frames])
        self.assertEqual(result["video_grid_thw"].tolist(), [[2, 4, 4]])
        cells = result["pixel_values_videos"].reshape(2, 16, 3, 2, 2, 2)
        torch.testing.assert_close(cells[1, :, :, 0], cells[1, :, :, 1])
        self.assertGreater((cells[0, :, :, 0] - cells[0, :, :, 1]).abs().max().item(), 0)
        single = frontend.preprocess_video_frames(proc, [[frames[-1]]])
        torch.testing.assert_close(single["pixel_values_videos"], proc.last_result["pixel_values"])
        torch.testing.assert_close(result["pixel_values_videos"][16:], single["pixel_values_videos"])

    def test_multiple_videos_keep_separate_grids_and_never_pair_across_clips(self):
        proc = ImageProcessor()
        videos = [[rgb_frame(11)], [rgb_frame(99, 4, 8), rgb_frame(123, 4, 8)]]
        result = frontend.preprocess_video_frames(proc, videos)
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 4, 4], [1, 2, 4]])
        self.assertEqual(result["pixel_values_videos"].shape, (24, 24))
        for index, video in enumerate(videos):
            expected = frontend.preprocess_video_frames(ImageProcessor(), [video])["pixel_values_videos"]
            start = 0 if index == 0 else 16
            torch.testing.assert_close(result["pixel_values_videos"][start:start + len(expected)], expected)

    def test_even_frame_count_halves_vision_tokens(self):
        proc = ImageProcessor()
        result = frontend.preprocess_video_frames(proc, [[rgb_frame(index) for index in range(8)]])
        image_tokens = int(proc.last_result["image_grid_thw"].prod(dim=1).sum()) // proc.merge_size**2
        video_tokens = int(result["video_grid_thw"].prod(dim=1).sum()) // proc.merge_size**2
        self.assertEqual((image_tokens, video_tokens), (32, 16))
        self.assertEqual(result["pixel_values_videos"].shape[0] * 2, proc.last_result["pixel_values"].shape[0])

    def test_temporal_size_comes_from_processor_including_size_one(self):
        for temporal, expected_groups in ((1, 4), (3, 2)):
            with self.subTest(temporal=temporal):
                proc = ImageProcessor()
                proc.temporal_patch_size = temporal
                result = frontend.preprocess_video_frames(proc, [[rgb_frame(i * 10) for i in range(4)]])
                self.assertEqual(result["video_grid_thw"].tolist(), [[expected_groups, 4, 4]])
                self.assertEqual(result["pixel_values_videos"].shape, (expected_groups * 16, 12 * temporal))
                cells = result["pixel_values_videos"].reshape(expected_groups, 16, 3, temporal, 2, 2)
                if temporal == 3:
                    torch.testing.assert_close(cells[-1, :, :, 0], cells[-1, :, :, 2])

    def test_inputs_and_processor_outputs_are_unchanged(self):
        proc = ImageProcessor()
        video = np.stack([rgb_frame(5), rgb_frame(17), rgb_frame(29)])
        original = video.copy()
        result = frontend.preprocess_video_frames(proc, [video])
        np.testing.assert_array_equal(video, original)
        for key, value in proc.last_result.items():
            torch.testing.assert_close(value, proc.outputs[-1][key])
        result["pixel_values_videos"].zero_()
        torch.testing.assert_close(proc.last_result["pixel_values"], proc.outputs[-1]["pixel_values"])

    def test_empty_batch_is_allowed_but_empty_clip_and_ambiguous_input_are_rejected(self):
        proc = ImageProcessor()
        self.assertEqual(frontend.preprocess_video_frames(proc, None), {})
        self.assertEqual(frontend.preprocess_video_frames(proc, []), {})
        self.assertEqual(proc.calls, [])
        with self.assertRaisesRegex(ValueError, "no frames"):
            frontend.preprocess_video_frames(proc, [[]])
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            frontend.preprocess_video_frames(proc, np.stack([rgb_frame(0)]))
        with self.assertRaisesRegex(TypeError, "frame sequence"):
            frontend.preprocess_video_frames(proc, [rgb_frame(0)])

    def test_different_resized_grids_within_one_video_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "share one resized spatial grid"):
            frontend.preprocess_video_frames(ImageProcessor(), [[rgb_frame(0), rgb_frame(0, 4, 8)]])

    def test_invalid_checkpoint_geometry_is_rejected_before_processing(self):
        for name in ("patch_size", "temporal_patch_size", "merge_size"):
            for value in (None, True, 0, -1, 2.0, "2"):
                with self.subTest(name=name, value=value):
                    proc = ImageProcessor()
                    setattr(proc, name, value)
                    with self.assertRaisesRegex(ValueError, name + " must be a positive integer"):
                        frontend.preprocess_video_frames(proc, [[rgb_frame(0)]])
                    self.assertEqual(proc.calls, [])

    def test_invalid_patch_width_grid_or_row_count_cannot_silently_repack(self):
        class BrokenProcessor(ImageProcessor):
            def __call__(self, **kwargs):
                result = super().__call__(**kwargs)
                return self.break_result(result)

        cases = (
            (lambda r: {**r, "pixel_values": r["pixel_values"][:, :-1]}, "RGB pixel_values"),
            (lambda r: {**r, "pixel_values": r["pixel_values"][:-1]}, "row count"),
            (lambda r: {**r, "image_grid_thw": torch.tensor([[2, 4, 4]])}, "single-image grids"),
            (lambda r: {**r, "image_grid_thw": torch.tensor([[1, 3, 4]])}, "divisible by merge_size"),
            (lambda r: {**r, "image_grid_thw": torch.tensor([[1.0, 4.0, 4.0]])}, "integer image_grid_thw"),
            (lambda r: {**r, "image_grid_thw": torch.empty((0, 3), dtype=torch.long)}, "one integer"),
        )
        for break_result, message in cases:
            with self.subTest(message=message):
                proc = BrokenProcessor()
                proc.break_result = break_result
                with self.assertRaisesRegex(ValueError, message):
                    frontend.preprocess_video_frames(proc, [[rgb_frame(0)]])


if __name__ == "__main__":
    unittest.main()
