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
"""Build MiniCPM-o's ``multi_modal_inputs`` without re-decoding the prompt.

The defect this replaces
------------------------
``AgentLoopWorker._compute_multi_modal_inputs``
(``verl/experimental/agent_loop/agent_loop.py:880``) rebuilds the processor
inputs for training like this::

    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
    multi_modal_inputs = build_multimodal_processor_inputs(self.processor, text=[current_text], ...)

For a Qwen-style processor that round-trips.  For MiniCPM-o it silently
destroys exactly the thing training needs.

``skip_special_tokens=True`` strips ``<image>``, ``</image>``, ``<unk>``,
``<audio>`` and ``</audio>`` -- every sentinel
``MiniCPMOProcessor._convert`` scans for.  The re-processed text therefore
contains no media markers at all, so::

    start_cond = (input_ids == im_start_id) | (input_ids == slice_start_id)

matches nothing, and ``image_bound`` / ``audio_bounds`` come back **empty**.
The tensors still have the right dtype and a plausible shape, nothing raises,
and ``get_vllm_embedding`` then scatters zero vision embeddings into the
sequence.  The model trains on a prompt whose image is described only by
``<unk>`` runs it never fills in -- a wrong-answer machine that reports no
error.

There is a second, quieter problem even if the sentinels survived: the decode
happens on the **padded** training ``input_ids``, while the bounds returned by
a fresh ``processor(...)`` call would be in the coordinates of *that* call's
own padding.  The two disagree by the left-padding length, and
``image_bound`` is an index into the training sequence.

The fix
-------
Do not re-derive the text.  The expanded ``input_ids`` are already the ground
truth, so scan them directly for the sentinels and emit bounds in the padded
training coordinate system by construction.  Pixels and audio features are
obtained from the two processor entry points that are **text-independent** --
``process_image`` and ``audio_feature_extract`` -- so nothing depends on a
decode round-trip.

Scope
-----
Installed from ``MiniCPMO50ThinkerAdapter.configure_processor`` and gated on the
processor being MiniCPM-o's, so other models in the same process keep upstream
behaviour.

Which class to patch is the part that bit us.  ``Qwen3OmniThinkerAdapter``
patches ``AgentLoopWorkerTQ.__ray_metadata__.modified_class`` (Ray copies
inherited methods onto that backing class), and run 15 showed a patch installed
that way never dispatched: the training forward received MiniCPM-o's seven-key
*empty* sentinel set, including ``image_sizes`` and ``spk_bounds`` -- keys this
builder never emits -- so upstream's method ran instead.  Whether the actor's
class is the Ray backing class, the decorated class, or the base differs by Ray
version and by how the worker was constructed, and none of it is introspectable
without Ray.

So install on **every** class an actor could resolve the method on -- the
defining ``AgentLoopWorker``, ``AgentLoopWorkerTQ``, and the Ray backing class
-- all delegating to a single pristine original, which makes double-wrapping
impossible no matter how the MRO resolves.

Reporting goes through ``print(..., flush=True)``, not ``logger``: run 15 also
confirmed that ``logger.info``/``logger.warning`` from a Ray worker never
reaches driver output on this stack, which is why "did the patch install?" was
unanswerable from a 5000-line log.
"""

import logging
import os

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_verl_minicpmo_mm_inputs_patch"
_DISABLE_ENV = "MINICPMO_DISABLE_MM_INPUTS_PATCH"
_METHOD_NAME = "_compute_multi_modal_inputs"
_LOG_PREFIX = "[verl-omni] MiniCPM-o multi_modal_inputs"

# One line per event, not per sample: enough to answer "did it install, and did
# it dispatch" from the next log without drowning it.
_first_call_reported = False
_mismatch_reported = False

__all__ = ["apply_minicpmo_mm_inputs_patch", "build_minicpmo_multi_modal_inputs"]


def _is_minicpmo_processor(processor) -> bool:
    """True when this processor is MiniCPM-o's, shim subclass included.

    ``_bless_processor_for_verl`` rebless the instance to a generated subclass
    named ``MiniCPMOProcessorVerlShim``, so an equality test on the class name
    would stop recognising the very processors this patch exists for.
    """
    return any(base.__name__.startswith("MiniCPMOProcessor") for base in type(processor).__mro__)


def _describe(cls) -> str:
    """Name plus identity, because the three candidates can share a name.

    ``AgentLoopWorkerTQ`` and its Ray backing class are both called
    ``AgentLoopWorkerTQ``.  Printing the id is what makes the next log able to
    say whether they were in fact distinct objects -- the question run 15 could
    not answer.
    """
    name = (
        getattr(cls, "__name__", None)
        or getattr(cls, "_actor_name", None)
        or getattr(getattr(cls, "_modified_class", None), "__name__", None)
        or type(cls).__name__
    )
    return f"{name}#{id(cls):x}"


def _worker_class_candidates():
    """Every class an agent-loop actor might resolve the method on.

    Returns:
        ``(classes, problems)`` -- classes that expose ``_compute_multi_modal_inputs``,
        and human-readable notes for the lookups that failed.  A partial result
        is useful: patching two of three targets still wins if the actor's class
        is one of the two.
    """
    found, problems = [], []
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        found.append(AgentLoopWorker)
    except Exception as e:
        problems.append(f"AgentLoopWorker import failed: {e!r}")
    try:
        from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

        found.append(AgentLoopWorkerTQ)
        modified = getattr(getattr(AgentLoopWorkerTQ, "__ray_metadata__", None), "modified_class", None)
        if modified is not None:
            found.append(modified)
        else:
            problems.append("AgentLoopWorkerTQ exposes no __ray_metadata__.modified_class")
    except Exception as e:
        problems.append(f"AgentLoopWorkerTQ import failed: {e!r}")

    unique, seen = [], set()
    for cls in found:
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if hasattr(cls, _METHOD_NAME):
            unique.append(cls)
        else:
            problems.append(f"{_describe(cls)} has no {_METHOD_NAME}")
    return unique, problems


def _pristine_method(candidates):
    """The unpatched original, preferring a class's own copy over an inherited one.

    All targets delegate to this single function, so installing on a base and on
    a subclass that carries its own copy cannot nest two wrappers.  None means
    every candidate is already patched.
    """
    for lookup in (lambda cls: cls.__dict__.get(_METHOD_NAME), lambda cls: getattr(cls, _METHOD_NAME, None)):
        for cls in candidates:
            fn = lookup(cls)
            if fn is not None and not getattr(fn, _PATCH_FLAG, False):
                return fn
    return None


def apply_minicpmo_mm_inputs_patch(processor) -> bool:
    """Replace ``_compute_multi_modal_inputs`` on every candidate worker class.

    Returns:
        True if the patch is now installed on at least one class (or already
        was), False if it was declined -- not MiniCPM-o, switched off by env, or
        no worker class could be reached.

    Idempotent via a flag on the installed function.  Declining is never fatal
    in the sense that nothing raises, which is exactly the danger: multimodal
    training then gets empty ``image_bound`` tensors and trains as text-only.
    """
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        print(
            f"{_LOG_PREFIX}: declined, disable switch {_DISABLE_ENV} is on. Upstream's "
            "_compute_multi_modal_inputs stays, which rebuilds inputs from a "
            "skip_special_tokens decode; image_bound/audio_bounds will be empty and no media "
            "embedding reaches the sequence. Text-only runs are unaffected.",
            flush=True,
        )
        return False

    if not _is_minicpmo_processor(processor):
        # Reached only from MiniCPMO50ThinkerAdapter.configure_processor, so this
        # firing at all means the processor is not the one we think it is.
        print(
            f"{_LOG_PREFIX}: declined, {type(processor).__name__} is not MiniCPM-o's processor; "
            "upstream's builder stays in place for it.",
            flush=True,
        )
        return False

    candidates, problems = _worker_class_candidates()
    if not candidates:
        print(
            f"{_LOG_PREFIX}: declined, no agent-loop worker class could be reached "
            f"({'; '.join(problems)}). Multimodal training would be handed empty bounds.",
            flush=True,
        )
        return False

    original = _pristine_method(candidates)
    if original is None:
        print(f"{_LOG_PREFIX}: already installed on {[_describe(c) for c in candidates]}.", flush=True)
        return True

    def _compute_multi_modal_inputs(self, output, input_ids):
        global _first_call_reported
        if self.processor is None or not _is_minicpmo_processor(self.processor):
            return original(self, output, input_ids)
        result = build_minicpmo_multi_modal_inputs(
            self.processor,
            input_ids,
            output.multi_modal_data or {},
            output.mm_processor_kwargs,
        )
        if not _first_call_reported:
            _first_call_reported = True
            shapes = {k: tuple(getattr(v, "shape", ())) or type(v).__name__ for k, v in result.items()}
            print(
                f"{_LOG_PREFIX} builder, first call: dispatched on {_describe(type(self))} "
                f"seq_len={int(input_ids.shape[-1])} built={shapes or '(no media in this sample)'}",
                flush=True,
            )
        return result

    setattr(_compute_multi_modal_inputs, _PATCH_FLAG, True)
    installed = []
    for cls in candidates:
        # ``AgentLoopWorkerTQ`` is @ray.remote-decorated, so the imported name is
        # an ActorClass proxy rather than a class; assigning through it may be
        # refused. One refused target must not cost us the other two.
        try:
            setattr(cls, _METHOD_NAME, _compute_multi_modal_inputs)
            installed.append(_describe(cls))
        except Exception as e:
            problems.append(f"{_describe(cls)} rejected the assignment: {e!r}")

    if not installed:
        print(
            f"{_LOG_PREFIX}: declined, no candidate accepted the patch ({'; '.join(problems)}). "
            "Multimodal training would be handed empty bounds.",
            flush=True,
        )
        return False

    print(
        f"{_LOG_PREFIX}: installed on {installed}; bounds are scanned from the training input_ids "
        "instead of a skip_special_tokens decode."
        + (f" Notes: {'; '.join(problems)}" if problems else ""),
        flush=True,
    )
    return True


def build_minicpmo_multi_modal_inputs(processor, input_ids, multi_modal_data, mm_processor_kwargs=None):
    """Build one sample's ``multi_modal_inputs`` from already-expanded ``input_ids``.

    Args:
        processor: The MiniCPM-o processor.
        input_ids: ``(1, seq_len)`` padded training ids -- the coordinate system
            the returned bounds are expressed in.
        multi_modal_data: ``{"images": [...], "audios": [...]}`` as the agent
            loop assembled it.
        mm_processor_kwargs: Carries ``sampling_rate``; the waveforms were
            decoded at that rate by the dataset.

    Returns:
        A dict carrying only the keys this sample actually has payload for.
        Every value is **unwrapped to per-sample shape**: ``extract_multi_modal_inputs``
        takes its ``has_image_bound`` branch as soon as any sample carries
        ``image_bound`` and then keeps values as a per-sample *list* without
        ``torch.cat``, which is exactly what ``get_vllm_embedding``'s ``[i]``
        indexing wants.  Returning the processor's own ``[[...]]`` nesting here
        would produce a 3-deep structure where the model expects 2.
    """
    images = multi_modal_data.get("images") or None
    audios = multi_modal_data.get("audios") or None
    if images is None and audios is None:
        return {}

    _require_media_entrypoints(processor, images, audios)

    kwargs = dict(mm_processor_kwargs or {})
    inputs = {}
    ids = input_ids.squeeze(0)

    if images is not None:
        # max_slice_nums has to match whatever the prompt expansion used, or the
        # slice count in pixel_values disagrees with the number of spans found
        # in input_ids -- the model would then index past the end of its own
        # image list. Forwarding it from mm_processor_kwargs is what keeps the
        # two calls describing the same image; absent that, both default.
        image_kwargs = {}
        if "max_slice_nums" in kwargs:
            image_kwargs["max_slice_nums"] = kwargs["max_slice_nums"]
        image_inputs = processor.process_image(images=images, return_tensors="pt", **image_kwargs)
        # process_image returns batch-of-one nesting ([[...]]); unwrap it.
        for key in ("pixel_values", "tgt_sizes"):
            value = image_inputs.get(key)
            if value is not None:
                inputs[key] = value[0] if len(value) == 1 else value
        # image_sizes is bookkeeping the data-dict forward does not accept; the
        # forward adapter drops it anyway, so never emit it.

    if audios is not None:
        sampling_rate = int(kwargs.get("sampling_rate", 16000))
        audio_features, audio_feature_lens, _ = processor.audio_feature_extract(
            audios=[audios],
            sampling_rate=sampling_rate,
        )
        if len(audio_features):
            # Already padded across this sample's chunks: [n_chunks, mel, T].
            # Samples still disagree on T, which the forward adapter re-pads.
            inputs["audio_features"] = audio_features
            inputs["audio_feature_lens"] = audio_feature_lens[0]

    bounds = _scan_bounds(ids, processor)
    if images is not None:
        inputs["image_bound"] = bounds["image_bound"]
    if audios is not None:
        inputs["audio_bounds"] = bounds["audio_bounds"]

    _warn_on_bound_mismatch(inputs, images, audios, processor)
    return inputs


def _require_media_entrypoints(processor, images, audios):
    """Fail with an explanation if the processor lacks the text-independent entry points.

    ``process_image`` and ``audio_feature_extract`` were read on the **4.5**
    processor (``MB_NPU/.../MiniCPM_o_4_5/processing_minicpmo.py``); the 5.0
    processor ships inside the checkpoint and could not be read on the authoring
    machine.  The two are the same processor lineage and vLLM-Omni drives 5.0
    through the same ``__call__`` contract, so they are expected to be present --
    but "expected" is not "measured", and a bare ``AttributeError`` five frames
    into a Ray worker would not say any of this.

    If this ever fires, the fix is not to reinstate the upstream path (it
    produces empty bounds); it is to look at what the 5.0 processor calls these
    two operations and rename them here.
    """
    missing = [
        name
        for name, needed in (("process_image", images is not None), ("audio_feature_extract", audios is not None))
        if needed and not hasattr(processor, name)
    ]
    if missing:
        raise AttributeError(
            f"{type(processor).__name__} has no {missing}. This builder needs the processor's "
            "text-independent media entry points, because the whole point is to avoid re-deriving "
            "text: upstream's _compute_multi_modal_inputs rebuilds inputs from a "
            "skip_special_tokens decode, which strips the sentinels image_bound/audio_bounds are "
            "scanned from and yields empty bounds with no error. These names were verified on the "
            "MiniCPM-o 4.5 processor; if 5.0 renamed them, map the new names here rather than "
            f"falling back to upstream. Set {_DISABLE_ENV}=1 to take the (silently wrong) "
            "upstream path anyway."
        )


def _scan_bounds(ids, processor):
    """Locate media spans in ``ids``, mirroring ``MiniCPMOProcessor._convert``.

    The scan is deliberately identical to the processor's own -- same sentinel
    sets, same ``+1`` on the start index -- because the bounds it produces are
    consumed by the same model code.  The only difference is the input: the
    padded training sequence rather than a freshly tokenised string, which is
    the whole point.
    """
    import torch

    tokenizer = processor.tokenizer

    start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
    end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
    image_start = torch.where(start_cond)[0] + 1
    image_end = torch.where(end_cond)[0]
    valid = min(len(image_start), len(image_end))
    image_bound = (
        torch.hstack([image_start[:valid].unsqueeze(-1), image_end[:valid].unsqueeze(-1)])
        if valid
        else torch.zeros((0, 2), dtype=torch.long)
    )

    audio_start = torch.where(ids == tokenizer.audio_start_id)[0] + 1
    audio_end = torch.where(ids == tokenizer.audio_end_id)[0]
    valid_audio = min(len(audio_start), len(audio_end))
    audio_bounds = (
        torch.hstack([audio_start[:valid_audio].unsqueeze(-1), audio_end[:valid_audio].unsqueeze(-1)])
        if valid_audio
        else torch.zeros((0, 2), dtype=torch.long)
    )

    return {"image_bound": image_bound, "audio_bounds": audio_bounds}


def _warn_on_bound_mismatch(inputs, images, audios, processor):
    """Say so when the span count does not match the media count.

    This is the check that would have caught the upstream defect immediately:
    an empty ``image_bound`` next to a non-empty ``images`` list is precisely
    what the ``skip_special_tokens`` decode produced, and nothing downstream
    raises on it.

    A mismatch is not always a bug -- ``max_prompt_length`` truncation can cut a
    span off the front of a prompt, and UHD slicing legitimately yields more
    spans than images -- so this warns rather than raises.  What matters is that
    the number reaches the log at all.
    """
    global _mismatch_reported

    messages = []
    image_bound = inputs.get("image_bound")
    if images is not None and image_bound is not None and len(image_bound) == 0:
        messages.append(
            f"{len(images)} image(s) in this sample but 0 image spans in the training input_ids. "
            "No vision embedding will be scattered into the sequence and nothing downstream will "
            "raise. Most likely the prompt ids were not built by the HF processor (so the "
            "image sentinels are absent), or truncation removed the span."
        )

    audio_bounds = inputs.get("audio_bounds")
    if audios is not None and audio_bounds is not None and len(audio_bounds) == 0:
        messages.append(
            f"{len(audios)} audio clip(s) in this sample but 0 audio spans in the training "
            "input_ids. The audio encoder's output has nowhere to be placed."
        )

    for message in messages:
        logger.warning("MiniCPM-o: %s", message)
    # First occurrence also goes to stdout: this is the one diagnostic that must
    # survive a Ray worker, since an empty bound is the failure that reports
    # itself in no other way.
    if messages and not _mismatch_reported:
        _mismatch_reported = True
        print(f"{_LOG_PREFIX} bound mismatch, first occurrence: {' | '.join(messages)}", flush=True)