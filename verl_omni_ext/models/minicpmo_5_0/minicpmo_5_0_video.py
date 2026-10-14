# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frame sampling for MiniCPM-o 5.0 video RL, shared by two callers that must agree.

Why this module exists at all
-----------------------------
MiniCPM-o has no video *input* path of its own.  ``MiniCPMOProcessor.__call__``
is ``(text, images, audios, audio_parts, ...)`` -- there is no ``videos``
parameter to bind frames to.  vLLM-Omni does accept video, but look at how:

    # vllm_omni/.../minicpmo_5_0_omni_llm.py
    def get_video_prompt_texts(self, image_size, num_frames):
        return self.info.get_slice_image_placeholder(
            image_size, image_idx=0,
            max_slice_nums=self.info.get_video_max_slice_num(),   # == 1
            use_image_id=False,
        ) * num_frames

    def process_videos(self, mm_data, mm_kwargs, tok_kwargs):
        ...
        prompts=[self.info.image_pattern * len(video) for video in parsed_videos],
        mm_data={"images": list(parsed_videos)},
        mm_kwargs={**mm_kwargs, "max_slice_nums": self.info.get_video_max_slice_num()},

Video *is* the image path, repeated once per frame, with slicing turned off.
So the way to give MiniCPM-o a video is to hand it N frames as ``images`` and
N ``(<image>./</image>)`` placeholders -- which is also exactly the multi-image
form ModelBest's own ``minicpmo5-infer_trans_orig.py`` writes.  No new
processor plumbing, no ``videos=`` anywhere.

Two things have to be pinned for the two sides to produce identical ids::

    ++data.mm_processor_kwargs.max_slice_nums=1
    ++data.mm_processor_kwargs.use_image_id=false

``max_slice_nums`` because it decides how many ``<unk>`` slots the expansion
writes, and ``use_image_id`` because the HF processor defaults it to *True* --
which would prefix every frame with ``<image_id>N</image_id>`` tokens that
vLLM-Omni's video path (``use_image_id=False``, hardcoded) never produces.  Both
keys reach both sides: ``build_multimodal_processor_inputs`` splices
``data.mm_processor_kwargs`` into the HF processor call
(``verl/utils/tokenizer/tokenizer.py:130``) and ``AgentLoopWorker`` forwards the
same dict to ``generate()``, where ``get_image_replacement`` reads
``max_slice_nums`` / ``use_image_id`` per request.

The agreement problem this module solves
----------------------------------------
Two callers need the *same* answer about one video:

* :meth:`MiniCPMOOmniRLHFDataset._process_multi_modal_info` needs the frames;
* ``minicpmo_5_0_processor._flatten_messages`` needs only **how many** frames,
  because it writes one placeholder per frame and the processor asserts
  ``len(image_tags) == len(image_sizes[index])``.

If those two numbers ever disagree the failure is not subtle -- the processor
raises, or worse, the prompt carries placeholders for media that is not there.
So both go through this module, and the count is derived from a single
:func:`frame_indices` on a single metadata probe.  Decoding is cached, so the
second caller in an agent-loop turn pays nothing:
``single_turn_agent_loop`` calls ``process_multi_modal_info(messages)`` and then
``apply_chat_template(messages)``, in that order, on the same list.

Sampling contract
-----------------
Deterministic and self-contained: ``fps`` frames per second of wall-clock
duration, capped at ``max_frames``, spread evenly across the whole clip.  Both
values come from the parquet row when present (``examples/gspo_trainer/
data_process/nextqa.py`` writes ``fps`` / ``max_frames`` / ``max_pixels`` into
each video item) and from the env fallbacks below when not.  Determinism
matters beyond reproducibility: the actor recompute reads the *decoded frames*
out of ``multi_modal_data``, but a cache eviction between the two callers has to
re-derive the same count from the file, and a random or time-dependent sampler
would silently break that.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "VideoSpec",
    "count_frames",
    "extract_video_audio",
    "frame_indices",
    "sample_frames",
    "video_spec_from_item",
]

# Sampling defaults for rows whose video item does not carry its own hints.
# 1 FPS / 32 frames is what the NExT-QA converter writes and what the Qwen3-Omni
# NExT-QA recipe was validated with; keeping the fallback identical means a
# hand-written parquet without the hint columns trains on the same footprint.
_FPS_ENV = "MINICPMO_VIDEO_FPS"
_MAX_FRAMES_ENV = "MINICPMO_VIDEO_MAX_FRAMES"
_MAX_PIXELS_ENV = "MINICPMO_VIDEO_MAX_PIXELS"
_DEFAULT_FPS = 1.0
_DEFAULT_MAX_FRAMES = 32
# 448*448: MiniCPM-o's own source-image grid at max_slice_nums=1, so frames
# larger than this are resized by the image processor anyway. Doing it here
# instead bounds the PIL objects that Ray has to pickle to the rollout replica
# and back -- a 1920x1080 frame is 6.2 MB, and 32 of them per trajectory is the
# difference between a working host and one paging.
_DEFAULT_MAX_PIXELS = 448 * 448

# Whether to append the clip's audio track as one (<audio>./</audio>) segment
# after the frame placeholders. Off by default: see extract_video_audio.
_USE_AUDIO_ENV = "MINICPMO_USE_AUDIO_IN_VIDEO"

# Frame counts already established by a decode, keyed by VideoSpec. Only ints,
# so this is a few hundred bytes per distinct clip and never worth evicting
# during a run; it exists so ``_flatten_messages`` does not re-open the file
# that ``_process_multi_modal_info`` just read.
_COUNT_CACHE: dict[VideoSpec, int] = {}
# Ceiling only as a leak guard for a long run over a large corpus.
_COUNT_CACHE_MAX = 65536


@dataclass(frozen=True)
class VideoSpec:
    """Everything that determines which frames a clip yields.

    Frozen and hashable because it is the cache key, and because being the
    cache key is the whole point: two callers that build a spec from the same
    message item get the same object and therefore the same answer.
    """

    path: str
    fps: float
    max_frames: int
    max_pixels: int


def video_spec_from_item(item: dict) -> VideoSpec:
    """Build a :class:`VideoSpec` from one verl ``{"type": "video", ...}`` item.

    ``RLHFDataset._build_messages`` splices the parquet's video dict into the
    content item wholesale (``{"type": "video", **video}``), so the converter's
    ``fps`` / ``max_frames`` / ``max_pixels`` hints arrive here untouched.

    Raises:
        ValueError: if the item carries no single video path.  A list of frame
            paths -- which ``_build_messages`` also accepts -- is refused
            explicitly rather than half-supported, because pre-extracted frames
            need no sampling and should be written as ``<image>`` segments.
    """
    path = item.get("video")
    if isinstance(path, list | tuple):
        raise ValueError(
            "MiniCPM-o video sampling received a list of frame paths. Pre-extracted frames need "
            "no sampling: write them as one <image> placeholder per frame in the prompt and put "
            "them in the images column, which the image path already handles."
        )
    if path is None:
        raise ValueError(
            f"A <video> segment carried no path: keys={sorted(item)}. Expected a 'video' key holding a filesystem path."
        )
    return VideoSpec(
        path=os.fspath(path),
        fps=float(item.get("fps") or _env_float(_FPS_ENV, _DEFAULT_FPS)),
        max_frames=int(item.get("max_frames") or _env_int(_MAX_FRAMES_ENV, _DEFAULT_MAX_FRAMES)),
        max_pixels=int(item.get("max_pixels") or _env_int(_MAX_PIXELS_ENV, _DEFAULT_MAX_PIXELS)),
    )


def frame_indices(spec: VideoSpec, total_frames: int, video_fps: float) -> list[int]:
    """Return the frame indices to sample, evenly spread over the whole clip.

    The single source of truth for "how many frames does this clip become".
    :func:`sample_frames` and :func:`count_frames` both route through it so
    they cannot drift apart -- and drifting apart is the failure this module is
    built to prevent, because a placeholder count that disagrees with the
    decoded frame count trips ``_convert_omni_to_inputs``'s
    ``assert len(image_tags) == len(image_sizes[index])`` in the best case and
    misplaces every subsequent media embedding in the worst.
    """
    if total_frames <= 0:
        raise ValueError(f"{spec.path}: decoder reported {total_frames} frames; nothing to sample.")
    if video_fps <= 0:
        # Some containers report no frame rate. Falling back to "sample the cap"
        # is better than dividing by zero, and the cap is the same on both sides.
        wanted = spec.max_frames
    else:
        duration = total_frames / video_fps
        wanted = int(round(duration * spec.fps))
    wanted = max(1, min(wanted, spec.max_frames, total_frames))

    if wanted == 1:
        return [0]
    step = (total_frames - 1) / (wanted - 1)
    return [int(round(i * step)) for i in range(wanted)]

def sample_frames(spec: VideoSpec) -> list:
    """Decode ``spec`` into a list of RGB ``PIL.Image`` frames.

    Records the resulting count in the module cache so a subsequent
    :func:`count_frames` for the same spec answers without touching the file.
    """
    total_frames, video_fps = _probe(spec)
    indices = frame_indices(spec, total_frames, video_fps)
    frames = _decode(spec, indices)

    expected_count = len(indices)

    if len(frames) != expected_count:
        if len(frames) == 0:
            raise RuntimeError(f"{spec.path}: decoder returned 0 frames.")
        if len(frames) < expected_count:
            diff = expected_count - len(frames)
            last_frame = frames[-1]
            for _ in range(diff):
                frames.append(last_frame.copy() if hasattr(last_frame, "copy") else last_frame)
        elif len(frames) > expected_count:
            frames = frames[:expected_count]

    frames = [_bound_pixels(frame, spec.max_pixels) for frame in frames]
    _remember_count(spec, len(frames))
    return frames


def count_frames(spec: VideoSpec) -> int:
    """Return how many frames ``spec`` yields, without decoding when avoidable.

    Answers from the cache populated by :func:`sample_frames` -- the normal
    case, since the agent loop extracts media before it renders the prompt.  On
    a miss it reads container metadata only (no pixel decode) and re-derives the
    count through :func:`frame_indices`, which is the same arithmetic the decode
    used.
    """
    cached = _COUNT_CACHE.get(spec)
    if cached is not None:
        return cached
    total_frames, video_fps = _probe(spec)
    count = len(frame_indices(spec, total_frames, video_fps))
    _remember_count(spec, count)
    return count


def use_audio_in_video() -> bool:
    """Whether to carry the clip's audio track alongside its frames.

    Read from the environment rather than from ``data.mm_processor_kwargs``
    because both callers need it and only one of them is handed a config:
    ``_flatten_messages`` is a module-level function invoked from
    ``processor.apply_chat_template``, several frames away from anything holding
    the trainer config.  ``MINICPMO_ENABLE_MM_FORWARD`` and
    ``MINICPMO_DISABLE_MM_DEDUP`` already work this way for the same reason.

    Default off.  MiniCPM-o's trained omni video format interleaves audio
    chunks with frames on a fixed cadence; what this module can offer is one
    audio segment after all the frames, which is structurally what vLLM-Omni
    itself does for a ``{"video": ..., "audio": ...}`` request but is *not* the
    duplex layout the checkpoint was trained on.  Worth trying, not worth
    defaulting to.
    """
    return os.environ.get(_USE_AUDIO_ENV, "0") == "1"


def extract_video_audio(spec: VideoSpec, sampling_rate: int):
    """Decode ``spec``'s audio track to a float32 mono waveform, or None.

    Returns None -- rather than raising -- when the clip has no decodable audio
    stream, because a corpus is allowed to be partly silent and a missing track
    is not a configuration error.  The caller drops the audio placeholder in
    that case, so prompt and media stay in step.

    ``sampling_rate`` is enforced here for the reason spelled out in
    ``minicpmo_5_0_dataset._load_audio``: MiniCPM-o's audio tower is Whisper,
    ``audio_feature_extract`` does not resample, and handing it 44.1 kHz audio
    while calling it 16 kHz produces no error at all -- only features of audio
    playing at the wrong speed.
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            f"{_USE_AUDIO_ENV}=1 needs librosa to decode the video's audio track to the 16 kHz "
            "mono waveform MiniCPM-o's Whisper encoder requires. Install it "
            "(`pip install librosa soundfile`) or unset the flag to train on frames only."
        ) from e

    import numpy as np

    try:
        waveform, _ = librosa.load(spec.path, sr=sampling_rate, mono=True)
    except Exception as e:
        logger.warning(
            "%s: could not decode an audio track (%r); training on frames only for this row.",
            spec.path,
            e,
        )
        return None
    if waveform is None or waveform.size == 0:
        logger.warning("%s: audio track decoded to zero samples; training on frames only.", spec.path)
        return None
    return np.asarray(waveform, dtype=np.float32)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _probe(spec: VideoSpec) -> tuple[int, float]:
    """Read (total_frames, average_fps) from container metadata only."""
    backend = _backend()
    if backend == "torchcodec":
        decoder = _torchcodec_decoder(spec.path)
        metadata = decoder.metadata
        total = getattr(metadata, "num_frames", None)
        fps = getattr(metadata, "average_fps", None)
        if total:
            return int(total), float(fps or 0.0)
        # Some containers have no frame count in the header. torchcodec still
        # knows the duration, which with the frame rate is enough.
        duration = getattr(metadata, "duration_seconds", None)
        if duration and fps:
            return int(round(float(duration) * float(fps))), float(fps)
        raise RuntimeError(
            f"{spec.path}: torchcodec reported neither a frame count nor a duration, so the frame "
            "grid cannot be computed. Re-encode the clip or force the PyAV backend with "
            "MINICPMO_VIDEO_BACKEND=av."
        )

    container, stream = _av_open(spec.path)
    try:
        total = int(stream.frames or 0)
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        if not total:
            duration = container.duration
            if duration and fps:
                # AV_TIME_BASE is microseconds.
                total = int(round(duration / 1_000_000 * fps))
        if not total:
            raise RuntimeError(
                f"{spec.path}: PyAV reported neither a frame count nor a duration for the video "
                "stream, so the frame grid cannot be computed."
            )
        return total, fps
    finally:
        container.close()


def _decode(spec: VideoSpec, indices: list[int]) -> list:
    from PIL import Image

    backend = _backend()
    if backend == "torchcodec":
        decoder = _torchcodec_decoder(spec.path)
        batch = decoder.get_frames_at(indices=list(indices))
        data = getattr(batch, "data", batch)
        return [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()).convert("RGB") for frame in data]

    wanted = set(indices)
    container, stream = _av_open(spec.path)
    try:
        # Sequential decode with a membership test rather than a seek per frame:
        # seeking is keyframe-accurate, not frame-accurate, so a seek-per-index
        # loop can return a different frame than the index names -- and the two
        # callers would then disagree about content while agreeing about count.
        by_index = {}
        for position, frame in enumerate(container.decode(stream)):
            if position in wanted:
                by_index[position] = frame.to_image().convert("RGB")
                if len(by_index) == len(wanted):
                    break
    finally:
        container.close()
    return [by_index[i] for i in indices if i in by_index]


def _backend() -> str:
    """Pick the video reader: torchcodec by default, PyAV on request or fallback.

    torchcodec first because that is the reader the NExT-QA recipe was validated
    with on this stack -- ``FORCE_QWENVL_VIDEO_READER=torchcodec`` in the
    Qwen3-Omni launcher exists because the torchvision/PyAV path showed severe
    transient native-thread growth on a 320-core host and could fail with
    ``Resource temporarily unavailable`` mid-decode.  See
    ``examples/gspo_trainer/README.md`` for the Ascend build recipe.
    """
    requested = os.environ.get("MINICPMO_VIDEO_BACKEND")
    if requested:
        if requested not in {"torchcodec", "av"}:
            raise ValueError(f"MINICPMO_VIDEO_BACKEND must be 'torchcodec' or 'av', got {requested!r}.")
        return requested
    try:
        import torchcodec.decoders  # noqa: F401
    except ImportError:
        return "av"
    return "torchcodec"


def _torchcodec_decoder(path: str):
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise ImportError(
            "MiniCPM-o video training needs a video reader. Install torchcodec (see the Ascend "
            "build recipe in examples/gspo_trainer/README.md) or PyAV, and select the latter with "
            "MINICPMO_VIDEO_BACKEND=av."
        ) from e
    return VideoDecoder(path)


def _av_open(path: str):
    try:
        import av
    except ImportError as e:
        raise ImportError(
            "MINICPMO_VIDEO_BACKEND=av needs PyAV (`pip install av`). torchcodec is the reader "
            "this recipe was validated with; see examples/gspo_trainer/README.md."
        ) from e
    container = av.open(path)
    if not container.streams.video:
        container.close()
        raise RuntimeError(f"{path}: no video stream.")
    stream = container.streams.video[0]
    # Frame-accurate sequential decode; without this PyAV may hand back frames
    # in presentation order interleaved across threads.
    stream.thread_type = "AUTO"
    return container, stream


def _bound_pixels(frame, max_pixels: int):
    """Downscale ``frame`` in place of the image processor, preserving aspect ratio.

    Only shrinks.  Upscaling a small frame to fill ``max_pixels`` would invent
    detail and cost tokens for it, and MiniCPM-o's image processor already
    resizes up to its own grid when it needs to.
    """
    width, height = frame.size
    if width * height <= max_pixels or width <= 0 or height <= 0:
        return frame
    from PIL import Image

    scale = (max_pixels / (width * height)) ** 0.5
    return frame.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)


def _remember_count(spec: VideoSpec, count: int) -> None:
    if len(_COUNT_CACHE) >= _COUNT_CACHE_MAX:
        _COUNT_CACHE.clear()
    _COUNT_CACHE[spec] = count


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s.", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s.", name, raw, default)
        return default