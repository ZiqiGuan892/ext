"""Pack real video frames with the checkpoint's existing Qwen image frontend.

The image frontend owns resizing, normalization and spatial patch ordering.
Its single-image temporal slots repeat the same frame; replace those slots
with consecutive frames before the vision tower consumes the pixel patches.
This does not depend on Transformers' version-specific video processor API.
"""

from numbers import Integral
from weakref import WeakKeyDictionary

import torch


_SIZE_GRIDS = WeakKeyDictionary()


def video_placeholder(grid, merge_size):
    """Render the native Huanyu video protocol, one bounded span per group."""
    if isinstance(merge_size, bool) or not isinstance(merge_size, Integral) or merge_size <= 0:
        raise ValueError("Huanyu video merge_size must be a positive integer")
    if isinstance(grid, torch.Tensor):
        grid = grid.reshape(-1).tolist()
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        raise ValueError("Huanyu video grid must contain exactly T, H and W")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in grid):
        raise ValueError("Huanyu video grid T, H and W must be positive integers")
    temporal, height, width = map(int, grid)
    if height % merge_size or width % merge_size:
        raise ValueError("Huanyu video grid H and W must be divisible by merge_size")
    pads = "<|video_pad|>" * (height * width // merge_size**2)
    return ("<|vision_start|>" + pads + "<|vision_end|>") * temporal


def _positive_size(image_processor, name):
    value = getattr(image_processor, name, None)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"Huanyu video image processor {name} must be a positive integer, got {value!r}")
    return int(value)


def _frames(video, index):
    if isinstance(video, (list, tuple)):
        frames = list(video)
    elif getattr(video, "ndim", None) == 4:
        frames = list(video)
    else:
        raise TypeError(f"Huanyu video #{index} must be a frame sequence or a four-dimensional video array")
    if not frames:
        raise ValueError(f"Huanyu video #{index} has no frames")
    return frames


def video_grid_for_size(image_processor, height, width, num_frames):
    """Measure one frame through the real frontend, caching its spatial grid.

    Dummy prompt sizing must use the same checkpoint resize rules as pixels.
    This costs one CPU image preprocessing call per size, without decoding a
    clip or relying on a Transformers-version-specific size formula.
    """
    for name, value in (("height", height), ("width", width), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"Huanyu video {name} must be a positive integer")
    patch = _positive_size(image_processor, "patch_size")
    temporal = _positive_size(image_processor, "temporal_patch_size")
    merge = _positive_size(image_processor, "merge_size")
    # Keep runtime caches outside the HF processor's serializable __dict__.
    cache = _SIZE_GRIDS.setdefault(image_processor, {})
    key = (height, width, patch, temporal, merge)
    if key not in cache:
        from PIL import Image

        image = Image.new("RGB", (int(width), int(height)))
        try:
            result = preprocess_video_frames(image_processor, [[image]])
        finally:
            image.close()
        _, grid_h, grid_w = result["video_grid_thw"][0].tolist()
        if len(cache) >= 128:
            cache.clear()
        cache[key] = (grid_h, grid_w)
    grid_h, grid_w = cache[key]
    return [(int(num_frames) + temporal - 1) // temporal, grid_h, grid_w]


def preprocess_video_frames(image_processor, videos):
    """Return packed video pixels and one ``(T, H, W)`` grid per clip.

    ``videos`` is a list/tuple of clips, each containing ordered RGB frames or
    a four-dimensional frame array. ``None`` and an empty clip list return
    ``{}``; an empty individual clip is invalid. Frames in a clip must resize
    to the same spatial grid. Different clips may have different grids.

    Only the final temporal group is padded, by repeating the last frame.
    Pixel dtype/device, RGB channel order and the frontend's spatial ordering
    are preserved. No input frame or image-processor output is modified.
    """
    if videos is None:
        return {}
    if not isinstance(videos, (list, tuple)):
        raise TypeError("Huanyu videos must be a list or tuple of frame sequences")
    if not videos:
        return {}

    patch = _positive_size(image_processor, "patch_size")
    temporal = _positive_size(image_processor, "temporal_patch_size")
    merge = _positive_size(image_processor, "merge_size")
    width = 3 * temporal * patch * patch
    packed_videos, video_grids = [], []
    for index, video in enumerate(videos):
        frames = _frames(video, index)
        result = image_processor(images=frames, return_tensors="pt")
        pixels, grids = result.get("pixel_values"), result.get("image_grid_thw")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 2 or pixels.shape[1] != width:
            raise ValueError(
                f"Huanyu video #{index} requires RGB pixel_values with shape [patches, {width}] "
                f"for patch_size={patch}, temporal_patch_size={temporal}; got {getattr(pixels, 'shape', None)}"
            )
        if (
            not isinstance(grids, torch.Tensor)
            or tuple(grids.shape) != (len(frames), 3)
            or grids.dtype == torch.bool
            or grids.is_floating_point()
            or grids.is_complex()
        ):
            raise ValueError(f"Huanyu video #{index} requires one integer image_grid_thw row per frame")
        rows = grids.tolist()
        if any(t != 1 or h <= 0 or w <= 0 or h % merge or w % merge for t, h, w in rows):
            raise ValueError(
                f"Huanyu video #{index} requires single-image grids [1, H, W] "
                f"with positive H/W divisible by merge_size={merge}; got {rows}"
            )
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(f"Huanyu video #{index} frames must share one resized spatial grid; got {rows}")
        _, height, grid_width = rows[0]
        frame_patches = height * grid_width
        if pixels.shape[0] != len(frames) * frame_patches:
            raise ValueError(f"Huanyu video #{index} pixel_values row count does not match its frame grids")

        # Qwen patch columns are [channel, temporal slot, patch row, patch col].
        # Preserve the already-packed spatial row order and take each real
        # frame once, instead of averaging or concatenating its embeddings.
        real_frames = pixels.reshape(len(frames), frame_patches, 3, temporal, patch, patch)[:, :, :, 0]
        missing = (-len(frames)) % temporal
        if missing:
            real_frames = torch.cat((real_frames, real_frames[-1:].expand(missing, -1, -1, -1, -1)), dim=0)
        groups = (len(frames) + missing) // temporal
        packed = real_frames.reshape(groups, temporal, frame_patches, 3, patch, patch)
        packed_videos.append(packed.permute(0, 2, 3, 1, 4, 5).reshape(groups * frame_patches, width))
        video_grids.append(grids.new_tensor([[groups, height, grid_width]], dtype=torch.long))

    return {
        "pixel_values_videos": torch.cat(packed_videos, dim=0),
        "video_grid_thw": torch.cat(video_grids, dim=0),
    }
