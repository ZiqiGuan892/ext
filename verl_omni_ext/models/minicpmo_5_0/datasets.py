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
"""RL dataset for MiniCPM-o 5.0 thinker training.

Two problems, both learned the expensive way, both fixed here.

Problem 1: the processor cannot render a chat template
------------------------------------------------------
``RLHFDataset.maybe_filter_out_long_prompts`` picks its length-measuring
function by asking one question -- ``is self.processor None?`` -- and the
processor arm opens with::

    raw_prompt = self.processor.apply_chat_template(messages, ...)

MiniCPM-o's processor has no chat template of its own.  V0-10 measured the file
layout on the checkpoint and found the split is by *file*, not by oversight:
``chat_template.jinja`` is present (which is where the **tokenizer** looks) and
``chat_template.json`` is absent (which is where the **processor** looks), with
no ``chat_template`` key in ``tokenizer_config.json`` either.  So the tokenizer
renders and the processor raises, on the same checkpoint, forever.

Because the whole of ``doc2len`` sits inside ``except Exception: ... return
self.max_prompt_length + 1``, that raise is not an error.  It is a vote to
discard the row.  Every row votes the same way.  The run continues with an
empty dataset.  This is pitfall 5, and it has now cost two 16-card launches.

``minicpmo_5_0_processor.build_minicpmo_processor`` attacks this from the other
end -- it tries four sources for a template and then *renders a smoke message*
to prove the processor works before handing it over.  That is a good guard, but
it is a guard on a path we do not need: **for MiniCPM-o the processor should
never be rendering chat templates in the first place.**

The working reference agrees.  ``MB_NPU/verl_mm-feature-one-step-offline_
half_duplex_omni`` -- a verl-based RL stack that actually trains MiniCPM-o 4.5
on GPU -- routes around the processor twice:

* its ``doc2len`` catches the failure and falls back to the tokenizer
  (``verl/utils/dataset/rl_dataset.py:205-213``, comment: *"Some processors
  (e.g., MiniCPMVProcessor) don't expose chat_template on the processor but do
  on the underlying tokenizer"*);
* and its production dataset does not go near ``doc2len`` at all -- the launch
  script sets ``data.custom_cls.path=pkg://verl.utils.dataset.minicpmo_dataset``
  and the class it names is a plain ``Dataset``, so ``filter_overlong_prompts``
  is silently inert and truncation happens per-row instead.  Prompt text is
  rendered with ``self.tokenizer.apply_chat_template`` and handed to the
  rollout as ``raw_prompt_ids``; the processor is used only to turn pixels into
  tensors.

The division of labour that falls out of that, and that this class implements:

    tokenizer -> text, chat template, length
    processor -> pixels and audio, nothing else

Problem 2: an empty dataset is not an error
--------------------------------------------
Filtering every row produces no exception, no non-zero exit and no summary --
only ``Error processing one of the samples, skipping...`` once per row, and
then a training loop with nothing in it.  The implementation plan records this
twice, as pitfall 31 and pitfall 32, precisely because the two have *identical*
symptoms and unrelated causes (a processor with no chat template; a parquet
with the wrong column names).  Telling them apart has cost a 16-card launch
more than once, and the acceptance checklist's row 0 exists only to remind a
human to go and count the rows by eye.

So this class counts what the filter ate and refuses to hand back an empty
dataset, naming the candidate causes and where to look.  A silent
misconfiguration becomes a named startup failure.

Wiring
------
Set both keys; verl-omni's ``get_dataset_class`` delegates to upstream's
loader as soon as ``custom_cls.path`` is present
(``verl_omni/utils/dataset/rl_dataset.py:94-97``)::

    data.custom_cls.path=pkg://verl_omni_ext.models.minicpmo_5_0.dataset
    data.custom_cls.name=MiniCPMOThinkerRLHFDataset

The launch script does this when ``MINICPMO_USE_CUSTOM_DATASET=1``.
"""

import hashlib
import logging
import os

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

__all__ = ["MiniCPMOThinkerRLHFDataset", "MiniCPMOOmniRLHFDataset"]


# ---------------------------------------------------------------------------
# Precomputed embedding cache support
# ---------------------------------------------------------------------------

class _EmbedList(list):
    """A list subclass that can carry precomputed embedding attributes.

    Python lists don't support custom attributes, so we wrap them in a subclass
    that does. ``_process_multi_modal_info`` attaches precomputed embeds here,
    and the monkey-patched ``_compute_multi_modal_inputs`` reads them back.
    """
    pass


def _attach_precomputed(lst, embeds, modality):
    """Wrap a list in _EmbedList and attach precomputed embeds as attribute."""
    wrapped = _EmbedList(lst)
    if modality == "vision":
        wrapped._precomputed_vision_embeds = embeds
    else:
        wrapped._precomputed_audio_embeds = embeds
    return wrapped


def _path_hash(path: str) -> str:
    """SHA-256 of a file path string. Used as cache key.

    Path-based keys are deterministic regardless of how the image/audio is
    loaded, which dtype the processor uses, or how slices/chunks are
    arranged — eliminating the content-hash mismatch problem (logs 16-28).
    """
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _load_precomputed(file_path: str, cache_dir: str, modality: str) -> "list | None":
    """Load precomputed embedding from cache by file path hash.

    Returns a list of per-chunk (audio) or per-slice (vision) embedding tensors,
    or None if the cache file doesn't exist.
    """
    h = _path_hash(file_path)
    cache_file = os.path.join(cache_dir, modality, f"{h}.pt")
    if os.path.isfile(cache_file):
        import torch
        return torch.load(cache_file, map_location="cpu")
    return None


# ---------------------------------------------------------------------------
# value is also what the launch script passes as
# ``++data.mm_processor_kwargs.sampling_rate``; this is the fallback for when it
# does not.
_DEFAULT_SAMPLING_RATE = 16000

# Fraction of rows the length filter may discard before this class treats the
# result as a misconfiguration rather than a dataset property. Losing a few
# genuinely long prompts is normal; losing most of them is not.
_DROP_RATIO_ENV = "MINICPMO_MAX_FILTER_DROP_RATIO"
_DEFAULT_MAX_DROP_RATIO = 0.5

# Columns whose presence means the tokenizer-only length estimate is an
# undercount. Names follow the keys RLHFDataset itself reads.
_MODALITY_CONFIG_KEYS = ("image_key", "video_key", "audio_key")


class MiniCPMOThinkerRLHFDataset(RLHFDataset):
    """``RLHFDataset`` that measures prompt length with the tokenizer, always.

    Everything else -- download, parquet reading, ``__getitem__``, multimodal
    handling, resume -- is inherited untouched.  The only behaviour that
    changes is which of the two ``doc2len`` implementations upstream selects,
    and what happens when the filter empties the dataset.
    """

    def maybe_filter_out_long_prompts(self, dataframe=None):
        before = _safe_len(dataframe)

        # Whether upstream will actually filter. With filter_overlong_prompts
        # False it returns the dataframe untouched, and "kept 40/40 rows" would
        # be a lie about work that never happened -- worse, an empty parquet
        # would then be blamed on a filter that did not run.
        filtered = bool(getattr(self, "filter_overlong_prompts", False))

        if filtered and self.processor is not None:
            self._warn_if_multimodal()

        # Upstream branches on `self.processor is not None` to choose between
        # the processor arm (which calls processor.apply_chat_template) and the
        # tokenizer arm. Hiding the processor for the duration of the call is
        # what selects the tokenizer arm -- it is not a trick to avoid
        # subclassing doc2len, it is the whole intent: for this model family
        # the tokenizer is the component that owns the chat template.
        #
        # Restored in `finally` because __getitem__ needs the real processor
        # for pixels; only length measurement goes without it.
        saved_processor = self.processor
        self.processor = None
        try:
            dataframe = super().maybe_filter_out_long_prompts(dataframe)
        finally:
            self.processor = saved_processor

        self._report_filter_outcome(before, _safe_len(dataframe), filtered)
        return dataframe

    def _warn_if_multimodal(self):
        """Say out loud that the tokenizer-only estimate ignores modality tokens.

        Measuring a prompt that will later carry 200 image tokens by counting
        only its text undercounts it, so a row that will overflow
        ``max_prompt_length`` at training time can survive the filter here.
        That is a real limitation and it is stated rather than hidden -- but it
        is strictly better than the alternative on this checkpoint, where the
        processor arm does not undercount, it raises, and every row is dropped.

        The working GPU reference has the same property for the same reason:
        its production dataset bypasses length filtering entirely and truncates
        per row inside ``preprocess(..., max_length, truncation)``.
        """
        present = []
        for cfg_key in _MODALITY_CONFIG_KEYS:
            column = getattr(self, cfg_key, None)
            if column and column in getattr(self.dataframe, "column_names", []):
                present.append(column)
        if present:
            logger.warning(
                "%s: measuring prompt length with the tokenizer only, but the data carries %s. "
                "Modality tokens are NOT counted, so a row near max_prompt_length=%s may still "
                "overflow at training time. This is deliberate: on this checkpoint the processor "
                "cannot render a chat template (V0-10: chat_template.jinja present, "
                "chat_template.json absent), so the processor arm would not undercount, it would "
                "drop every row. Set data.truncation appropriately for the multimodal phase.",
                type(self).__name__,
                present,
                self.max_prompt_length,
            )

    def _report_filter_outcome(self, before, after, filtered=True):
        """Turn 'the dataset is empty' from a silent state into a loud one."""
        if after is None:
            # Upstream handed back nothing at all. Nothing to count and nothing
            # to train on either, but this is a caller bug rather than a filter
            # outcome, so it is reported as itself.
            raise RuntimeError(
                f"{type(self).__name__}: maybe_filter_out_long_prompts returned None. "
                "It was called without a dataframe and filter_overlong_prompts is off; "
                "upstream's contract is that the caller passes self.dataframe in."
            )

        if not filtered:
            logger.info(
                "%s: filter_overlong_prompts is off; %d rows passed through unmeasured.",
                type(self).__name__,
                after,
            )
        elif before is None:
            logger.info("%s: %d rows after length filtering.", type(self).__name__, after)
        else:
            dropped = before - after
            ratio = (dropped / before) if before else 0.0
            logger.info(
                "%s: length filter kept %d/%d rows (dropped %d, %.1f%%) with "
                "max_prompt_length=%s via the tokenizer path.",
                type(self).__name__,
                after,
                before,
                dropped,
                ratio * 100.0,
                self.max_prompt_length,
            )

        if after == 0:
            raise RuntimeError(
                f"{type(self).__name__}: the dataset is empty. Upstream treats this as normal and "
                "would have continued into a training loop with no data -- see implementation plan "
                "pitfalls 31 and 32. " + (
                    f"The length filter discarded all {before} rows. doc2len swallows every "
                    "exception and votes 'too long', so the cause is one of: (1) the prompt column "
                    "named by data.prompt_key is missing or is not a list of chat messages; (2) the "
                    "tokenizer cannot render the chat template with data.apply_chat_template_kwargs "
                    "as given; (3) the prompts genuinely exceed "
                    f"max_prompt_length={self.max_prompt_length}; (4) the parquet was written with "
                    "the wrong schema. Read the traceback printed above the first "
                    "'Error processing one of the samples' line -- it names which."
                    if filtered and before else
                    "No filtering ran, so the parquet itself has no rows -- check "
                    "data.train_files and the conversion step, not the filter."
                ) + " To inspect the data without launching, run the dataset-conversion script's "
                "--check mode."
            )

        if not filtered or not before:
            return

        ratio = (before - after) / before
        max_ratio = _max_drop_ratio()
        if ratio > max_ratio:
            raise RuntimeError(
                f"{type(self).__name__}: the length filter discarded {before - after}/{before} rows "
                f"({ratio:.1%}), above the {max_ratio:.0%} threshold. Losing a few over-long "
                "prompts is expected; losing most of them means max_prompt_length is too small, "
                "the chat template is rendering something unexpected, or the prompt column is "
                "malformed for part of the corpus. Raise the threshold with "
                f"{_DROP_RATIO_ENV}=<0..1> if this really is the data."
            )


class MiniCPMOOmniRLHFDataset(MiniCPMOThinkerRLHFDataset):
    """Omni (image + audio) RL dataset for MiniCPM-o 5.0 thinker.

    Structural precedent: ``QwenOmniRLHFDataset``
    (``verl_omni/utils/dataset/omni_rl_datasets.py``), which overrides exactly
    this one classmethod and lets ``RLHFDataset`` own everything else.  The
    difference is what the media has to be *decoded into*, because the two
    models' processors want different things:

    ==============  ========================  =============================
    Qwen3-Omni      ``process_mm_info(...)``  from ``qwen_omni_utils``
    MiniCPM-o 5.0   PIL RGB + float32 numpy   this method
    ==============  ========================  =============================

    ``qwen_omni_utils`` is not usable here.  It returns audio already resampled
    to *Qwen's* feature-extractor rate and videos as decoded frame tensors, and
    MiniCPM-o's ``MiniCPMOProcessor.__call__`` accepts neither -- it takes
    ``audios=`` as a list of raw waveforms plus an explicit ``sampling_rate=``,
    and has **no ``videos`` parameter at all**.

    Why the AVQA parquet works unchanged
    ------------------------------------
    The Qwen3-Omni AVQA data (``avqa_r1_6k/{train,validation}.parquet``) is
    model-agnostic: the ``prompt`` column carries chat messages whose text holds
    bare ``<image>`` / ``<audio>`` placeholders, and the media live in separate
    ``images`` / ``audios`` columns.  ``RLHFDataset._build_messages`` splits on
    those placeholders and rewrites the content into verl's structured
    ``{"type": "image", ...}`` / ``{"type": "audio", ...}`` lists.  Nothing in
    that chain is Qwen-specific, so the same rows feed both models; only the
    decode step below and the processor shim differ.

    Why the NExT-QA parquet works unchanged too
    -------------------------------------------
    Same reason, one step further.  ``examples/gspo_trainer/data_process/
    nextqa.py`` writes ``<video>`` placeholders and a ``videos`` column of
    ``{"video": path, "fps": ..., "max_frames": ...}`` dicts, and
    ``_build_messages`` splices each dict into a ``{"type": "video", **video}``
    item.  This method turns that item into *frames in the ``images`` list*,
    because that is the only door MiniCPM-o has for video -- see
    ``verl_omni/utils/dataset/minicpmo_5_0_video.py`` for why, and for the two
    ``mm_processor_kwargs`` that have to be pinned for the rollout and training
    expansions to agree.  So one parquet feeds Qwen3-Omni and MiniCPM-o.
    """

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        """Decode messages into ``(images, videos, audios)`` shaped for MiniCPM-o.

        Returns:
            A 3-tuple.  ``images`` is a flat list of RGB ``PIL.Image`` -- video
            frames included, in message order -- or None; ``videos`` is always
            None; ``audios`` is a flat list of float32 mono numpy waveforms at
            ``sampling_rate`` Hz, or None.

        Flat lists, not per-sample nesting: this runs once per trajectory, and
        ``MiniCPMOProcessor`` reads a flat list as "the media belonging to the
        single prompt in ``text``" (``audio_feature_extract`` branches on
        ``isinstance(audios[0], np.ndarray)`` to wrap it as one sample).

        ``videos`` stays None on purpose, and that is not a limitation being
        worked around -- it is the model's actual input contract.
        ``MiniCPMOProcessor.__call__`` is ``(text, images, audios,
        audio_parts, ...)`` with no ``videos`` parameter, and vLLM-Omni's own
        ``process_videos`` routes frames through the *image* path at
        ``max_slice_nums=1``.  Both sides read video as "N images", so this
        method produces N images and ``_flatten_messages`` writes N
        ``(<image>./</image>)`` placeholders for the same item.  The two counts
        come from one shared sampler for exactly that reason.

        A video's audio track is carried only when
        ``MINICPMO_USE_AUDIO_IN_VIDEO=1``; it then becomes one ``audios`` entry
        appended after the clip's frames, matching the single
        ``(<audio>./</audio>)`` that ``_flatten_messages`` writes.  Ordering
        within ``audios`` is message order, so a standalone ``<audio>`` segment
        later in the prompt still lines up.
        """
        from verl_omni_ext.models.minicpmo_5_0.minicpmo_5_0_video import (
            extract_video_audio,
            sample_frames,
            use_audio_in_video,
            video_spec_from_item,
        )

        images, audios = [], []
        sampling_rate = _configured_sampling_rate(config)
        with_video_audio = use_audio_in_video()

        # Precompute cache: if MINICPMO_EMBED_CACHE_DIR is set, look up
        # precomputed embeddings by file path hash. Attach results to the
        # images/audios list objects via _EmbedList, for the monkey-patched
        # _compute_multi_modal_inputs to pick up and inject into multi_modal_inputs.
        cache_dir = os.environ.get("MINICPMO_EMBED_CACHE_DIR", "")
        precomputed_vision = []
        precomputed_audio = []

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "image":
                    images.append(_load_image(item))
                    p = item.get("image")
                    if isinstance(p, (str, os.PathLike)):
                        p = os.fspath(p)
                        if cache_dir:
                            precomputed_vision.append(_load_precomputed(p, cache_dir, "vision"))
                elif kind == "audio":
                    audios.append(_load_audio(item, sampling_rate))
                    p = item.get("audio") or item.get("audio_url")
                    if isinstance(p, (str, os.PathLike)):
                        p = os.fspath(p)
                        if cache_dir:
                            precomputed_audio.append(_load_precomputed(p, cache_dir, "audio"))
                    else:
                        print(f"[verl-omni] DEBUG audio: item keys={sorted(item.keys())}, "
                              f"audio={item.get('audio')}, audio_url={item.get('audio_url')}, "
                              f"p={p}, type(p)={type(p).__name__}",
                              flush=True)
                elif kind == "video":
                    spec = video_spec_from_item(item)
                    images.extend(sample_frames(spec))
                    # video 的 cache key 是视频文件路径（precompute 时也用视频路径）
                    if cache_dir and spec.path:
                        precomputed_vision.append(_load_precomputed(spec.path, cache_dir, "vision"))
                    if with_video_audio:
                        waveform = extract_video_audio(spec, sampling_rate)
                        # None means the clip has no decodable audio stream.
                        # `_flatten_messages` reaches the same verdict through
                        # the same call, so the placeholder is dropped with it.
                        if waveform is not None:
                            audios.append(waveform)
                            if cache_dir and spec.path:
                                precomputed_audio.append(_load_precomputed(spec.path, cache_dir, "audio"))

        # Attach precomputed embeddings to the images/audios list objects
        # via a wrapper class. Python lists don't support custom attributes,
        # so we wrap them in a subclass that does.
        if cache_dir:
            if images:
                images = _attach_precomputed(images, precomputed_vision if precomputed_vision else None, "vision")
            if audios:
                audios = _attach_precomputed(audios, precomputed_audio if precomputed_audio else None, "audio")

        # Diagnostic: log cache lookup results
        if cache_dir:
            n_vision_cached = sum(1 for x in precomputed_vision if x is not None)
            n_vision_none = sum(1 for x in precomputed_vision if x is None)
            n_audio_cached = sum(1 for x in precomputed_audio if x is not None)
            n_audio_none = sum(1 for x in precomputed_audio if x is None)
            print(
                f"[verl-omni] cache lookup: "
                f"vision_cached={n_vision_cached} vision_none={n_vision_none} "
                f"audio_cached={n_audio_cached} audio_none={n_audio_none} "
                f"cache_dir={cache_dir}",
                flush=True,
            )

        return (images or None, None, audios or None)


def _configured_sampling_rate(config):
    """Read the sampling rate the launch script configured, else 16 kHz.

    ``config`` is the ``data`` config, so the value the script sets as
    ``++data.mm_processor_kwargs.sampling_rate=16000`` is visible here.  Reading
    it rather than hardcoding keeps one number in one place: the *same* value
    reaches ``MiniCPMOProcessor`` through ``mm_processor_kwargs``, and a
    waveform decoded at one rate but described to the feature extractor as
    another is a silent corruption -- no exception, just audio the encoder reads
    at the wrong speed.
    """
    try:
        kwargs = config.get("mm_processor_kwargs", None) if config is not None else None
        rate = kwargs.get("sampling_rate", None) if kwargs else None
    except Exception:  # a plain dict, a DictConfig, or None -- none worth crashing over
        rate = None
    return int(rate) if rate else _DEFAULT_SAMPLING_RATE


def _load_image(item):
    """Turn one verl image content item into an RGB ``PIL.Image``.

    ``RLHFDataset._build_messages`` has already normalised most of this: a PIL
    input arrives converted to RGB, and a dict carrying ``bytes`` arrives with
    an ``image`` key already opened.  What it does *not* do is convert the
    opened-from-bytes and loaded-from-path cases to RGB, and MiniCPM-o's image
    processor assumes 3 channels -- a palettised PNG or a grayscale JPEG would
    otherwise reach it with 1 channel and fail on a shape it never explains.
    """
    from io import BytesIO

    from PIL import Image

    image = item.get("image")
    if image is None and item.get("bytes") is not None:
        image = Image.open(BytesIO(item["bytes"]))
    if isinstance(image, str | os.PathLike):
        image = Image.open(os.fspath(image))
    if image is None:
        raise ValueError(
            f"An <image> segment carried no usable payload: keys={sorted(item)}. Expected one of "
            "'image' (PIL or path) or 'bytes'."
        )
    return image.convert("RGB")


def _load_audio(item, sampling_rate):
    """Turn one verl audio content item into a float32 mono numpy waveform.

    Accepts what a parquet realistically holds: an already-decoded array, raw
    encoded ``bytes``, or a path/URL under ``audio`` or ``audio_url``.

    Resampling is mandatory, not opportunistic.  MiniCPM-o's audio tower is
    Whisper, which is only defined at 16 kHz, and
    ``MiniCPMOProcessor.audio_feature_extract`` does not resample -- it takes
    ``sampling_rate`` as a *description* of the waveform it was handed and
    computes the 30-second chunk boundary from it.  Handing it 44.1 kHz audio
    while saying 16 kHz produces no error at all: the features are simply of
    audio playing at the wrong rate.  So the rate is enforced at decode time,
    which is the last point where it can still be checked.
    """
    import numpy as np

    audio = item.get("audio")
    if audio is None:
        audio = item.get("audio_url")
    if audio is None and item.get("bytes") is not None:
        audio = item["bytes"]
    if audio is None:
        raise ValueError(
            f"An <audio> segment carried no usable payload: keys={sorted(item)}. Expected one of "
            "'audio', 'audio_url' or 'bytes'."
        )

    if isinstance(audio, np.ndarray):
        # Already decoded. Trust the caller's rate: there is no sample rate
        # attached to a bare array, so there is nothing to check against and
        # resampling on a guess would be worse than not resampling.
        waveform = audio
    else:
        waveform = _decode_audio(audio, sampling_rate)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        # Whisper wants mono. Averaging channels is what librosa's mono=True
        # does; doing it here covers the pre-decoded-array path too.
        waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[-1] else -1)
    return waveform


def _decode_audio(source, sampling_rate):
    """Decode bytes / path / URL to a mono waveform at ``sampling_rate``.

    ``librosa`` is the dependency the Qwen3-Omni path already pulls in through
    ``qwen_omni_utils``, so requiring it here adds nothing new to the image.
    The import is local and the failure is spelled out because a missing
    ``librosa`` in a Ray worker otherwise surfaces as a bare
    ``ModuleNotFoundError`` inside a dataset ``__getitem__``, several frames
    away from anything naming audio.

    Supports zarr.zip::uuid format: ``/path/to/archive.zarr.zip::member-uuid``.
    """
    from io import BytesIO

    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "MiniCPMOOmniRLHFDataset needs librosa to decode audio to the 16 kHz mono waveforms "
            "MiniCPM-o's Whisper encoder requires. Install it (`pip install librosa soundfile`) "
            "or pre-decode the audio column into numpy arrays, which this class accepts as-is."
        ) from e

    # zarr.zip::uuid format
    if isinstance(source, str) and "::" in source:
        return _load_zarr_audio(source, sampling_rate)

    handle = BytesIO(source) if isinstance(source, bytes | bytearray) else os.fspath(source)
    waveform, _ = librosa.load(handle, sr=sampling_rate, mono=True)
    return waveform


def _load_zarr_audio(zarr_path, sampling_rate):
    """Load audio from a zarr.zip::uuid member path.

    Uses zarr.storage.ZipStore (v3 API) to handle compressed chunk data.
    """
    import numpy as np
    import zarr

    archive_path, member_id = zarr_path.split("::", 1)

    store = zarr.storage.ZipStore(archive_path, mode="r")
    root = zarr.open(store=store, mode="r")
    member = root[f"{member_id}.wav"]
    audio_np = np.array(member["audio"], dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0 if audio_np.shape[0] < audio_np.shape[-1] else -1)
    return audio_np


def _safe_len(obj):
    """``len(obj)`` where obj may legitimately be None."""
    return None if obj is None else len(obj)


def _max_drop_ratio():
    raw = os.environ.get(_DROP_RATIO_ENV)
    if raw is None:
        return _DEFAULT_MAX_DROP_RATIO
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.2f", _DROP_RATIO_ENV, raw, _DEFAULT_MAX_DROP_RATIO)
        return _DEFAULT_MAX_DROP_RATIO
    if not 0.0 <= value <= 1.0:
        logger.warning("%s=%r is outside [0,1]; using %.2f", _DROP_RATIO_ENV, raw, _DEFAULT_MAX_DROP_RATIO)
        return _DEFAULT_MAX_DROP_RATIO
    return value


# ---------------------------------------------------------------------------
# Monkey-patch AgentLoopWorker._compute_multi_modal_inputs to inject
# precomputed embeddings into multi_modal_inputs after convert_to_tensors.
#
# _process_multi_modal_info stores precomputed embeddings as attributes on the
# images/audios _EmbedList objects. This patch reads them and injects into the
# returned multi_modal_inputs dict as NonTensorData-wrapped values, which
# survive extract_multi_modal_inputs and reach the forward adapter.
#
# Ray @ray.remote wraps classes — patch via modified_class. Module-level cache
# fails under asyncio — use _EmbedList attributes (travels with the output
# through multi_modal_data). See memory:
#   feedback_extract_multi_modal_inputs_skips_none.md
# ---------------------------------------------------------------------------

def _patch_compute_multi_modal_inputs():
    """Patch _compute_multi_modal_inputs to inject precomputed embeddings."""
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    except ImportError as e:
        print(f"[verl-omni] _patch_compute_multi_modal_inputs: ImportError: {e}", flush=True)
        return

    if getattr(AgentLoopWorker._compute_multi_modal_inputs, "_omni_patched", False):
        return

    _original = AgentLoopWorker._compute_multi_modal_inputs

    def _patched(self, output, input_ids):
        result = _original(self, output, input_ids)
        # Read precomputed embeddings from the images/audios list objects
        # (attached by _process_multi_modal_info). These travel with the
        # output through multi_modal_data, avoiding async concurrency issues.
        vision_embeds = None
        audio_embeds = None
        mmd = output.multi_modal_data or {}
        images = mmd.get("images")
        if images is not None and hasattr(images, "_precomputed_vision_embeds"):
            vision_embeds = images._precomputed_vision_embeds
        audios = mmd.get("audios")
        if audios is not None and hasattr(audios, "_precomputed_audio_embeds"):
            audio_embeds = audios._precomputed_audio_embeds
        if vision_embeds is not None:
            from tensordict import NonTensorData
            result["_precomputed_vision_embeds"] = NonTensorData(vision_embeds)
        if audio_embeds is not None:
            from tensordict import NonTensorData
            result["_precomputed_audio_embeds"] = NonTensorData(audio_embeds)
        if vision_embeds is not None or audio_embeds is not None:
            print(
                f"[verl-omni] _compute_multi_modal_inputs patched: "
                f"cls={type(self).__name__} vision={vision_embeds is not None} "
                f"audio={audio_embeds is not None} result_keys={list(result.keys())}",
                flush=True,
            )
        return result

    _patched._omni_patched = True
    AgentLoopWorker._compute_multi_modal_inputs = _patched
    print("[verl-omni] _patch_compute_multi_modal_inputs: patched AgentLoopWorker successfully", flush=True)

    # Also try to patch the Ray actor class directly. AgentLoopWorkerTQ is
    # decorated with @ray.remote, so importing it gives an ActorClass object.
    # The original class is in __ray_metadata__.modified_class.
    try:
        from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ as _TQ_actor
        _TQ_cls = _TQ_actor.__ray_metadata__.modified_class
        if not getattr(_TQ_cls._compute_multi_modal_inputs, "_omni_patched", False):
            _TQ_cls._compute_multi_modal_inputs = _patched
            print(f"[verl-omni] _patch_compute_multi_modal_inputs: patched "
                  f"{_TQ_cls.__name__} (via modified_class) successfully", flush=True)
    except Exception as e:
        print(f"[verl-omni] _patch_compute_multi_modal_inputs: AgentLoopWorkerTQ patch skipped: {e}", flush=True)


_patch_compute_multi_modal_inputs()