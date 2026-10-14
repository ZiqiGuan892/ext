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
"""Processor fallback for MiniCPM-o 5.0.

Why a fallback is needed
------------------------
``verl.utils.tokenizer.hf_processor`` recognises a fixed list of processor
classes (``verl/utils/tokenizer/tokenizer.py:205-231``)::

    match processor.__class__.__name__:
        case "Qwen2VLProcessor":   ...
        case "Qwen2_5_VLProcessor": ...
        case "Qwen3VLProcessor":   ...
        case "Glm4vProcessor":     ...
        case "MllamaProcessor":    pass
        case "Gemma4Processor":    ...
        case _:
            raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

V0-3 measured this checkpoint's processor class as ``MiniCPMOProcessor``, which
falls through to ``case _``.  The ``ValueError`` is then swallowed by the
enclosing ``except Exception`` -- the function sets ``processor = None`` and
emits a single ``warnings.warn``.  So ``hf_processor`` **always** returns None
for MiniCPM-o, and it does so quietly.

Downstream that means ``OmniModelConfig.get_processor()``
(``verl_omni/workers/config/omni/model.py:189``) falls back to the *tokenizer*.
Text-only RL survives that -- ``RLHFDataset`` takes its no-processor branch and
tokenises with ``tokenizer.apply_chat_template`` -- but any row carrying an
``images`` / ``videos`` / ``audios`` column trips
``RLHFDataset._build_messages``'s ``assert self.processor is not None``.

The precedent is ``patch_hf_processor_for_qwen3_omni``
(``verl_omni/models/transformers/qwen3_omni_thinker.py:118``), which wraps
``hf_processor`` globally.  This module deliberately does **not** monkey-patch:
for the omni path, ``configure_processor`` is the single call site that matters
(``main_omni.py:174`` returns ``omni_model_config.get_processor()``, which is
populated from ``adapter_cls.configure_processor`` at
``workers/config/omni/model.py:186``).  Building the processor there keeps the
blast radius to MiniCPM-o runs instead of every model in the process.

What is deliberately NOT bound: ``get_rope_index``
--------------------------------------------------
The Qwen-family branches of ``hf_processor`` bind the model's
``get_rope_index`` onto the processor so verl can precompute 4-channel m-RoPE
position ids in the dataset.  MiniCPM-o must not get that treatment:

* V0-3 measured ``hasattr(processor, "get_rope_index") == False`` -- there is
  no method to bind.
* V0-7 read ``MiniCPMO._compute_position_ids`` and found the model computes its
  own canvas m-RoPE from ``input_ids`` / ``image_bound`` / ``tgt_sizes``
  whenever ``config.mrope_mode`` is a canvas mode, **ignoring**
  ``data["position_ids"]`` entirely on that branch.  Position ids fabricated on
  the verl side would be dead weight at best.
* If ``mrope_mode`` is *not* canvas, the model passes ``data["position_ids"]``
  straight through -- and verl's own 1-D positions are then exactly right.

Either way, binding a Qwen-style ``get_rope_index`` would be wrong.  Leaving it
unbound makes ``AgentLoopWorker._compute_multi_modal_position_ids`` take its
``not hasattr(...)`` branch and emit standard ``(1, seq_len)`` positions, which
is what both cases want.  See implementation plan section 4.7 / V0-8.

What IS bound, since V3 run 2: the chat template
------------------------------------------------
``AutoProcessor.from_pretrained`` returns a ``MiniCPMOProcessor`` whose
``chat_template`` is None -- the template lives in the *tokenizer*, not in a
``chat_template.json`` next to the processor.  That is fatal in a way that only
shows up 500 lines into a run::

    rl_dataset.py:214 in doc2len -> processor.apply_chat_template(...)
    ValueError: Cannot use apply_chat_template because this processor does not
                have a chat template.

``doc2len`` catches it, prints "Error processing one of the samples,
skipping..." and returns ``max_prompt_length + 1``, so **every** row is filtered
out as too long and the dataset silently empties (measured 2026-08-29 14:26,
7473/7473 rows).  ``_ensure_chat_template`` copies the tokenizer's template onto
the processor, which is the same template ``tokenizer.apply_chat_template``
would have used on the no-processor branch -- so prompt rendering is unchanged,
the processor just stops throwing.
"""

import logging
import os

logger = logging.getLogger(__name__)

_EXPECTED_PROCESSOR_CLASS = "MiniCPMOProcessor"

from verl_omni_ext.models.qwen3_6_whisper import apply_qwen36_vllm_prompt_updates_patch
apply_qwen36_vllm_prompt_updates_patch()

def _provenance() -> str:
    """Which copy of this file is running, and how old it is.

    Pitfall 22 has now cost three separate investigations: the training machine
    imports verl-omni from a different tree than the one the code is written in,
    so "I fixed that" and "the fix ran" are independent facts.  Every log line
    this module emits about a failure carries the answer, so the next reader
    never has to ask.
    """
    try:
        import datetime
        import os

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(__file__)).strftime("%Y-%m-%d %H:%M")
        return f"{__file__} (mtime {mtime})"
    except Exception:
        return __file__


def _ensure_chat_template(processor, model_path: str, trust_remote_code: bool) -> str:
    """Give the processor a chat template, taking the tokenizer's if it has none.

    ``transformers.ProcessorMixin.apply_chat_template`` reads ``self.chat_template``
    and raises ``ValueError: Cannot use apply_chat_template because this processor
    does not have a chat template`` when it is None.

    **Where this checkpoint actually keeps its template (V0-10, measured).**
    Not where the first version of this function assumed:

    ===========================  ==========  ==============================
    ``chat_template.json``       absent      the file processors look for
    ``chat_template.jinja``      **present** the file tokenizers look for
    ``tokenizer_config.json``    present, but **no** ``chat_template`` key
    ===========================  ==========  ==============================

    So the split is by *file*, not by "the processor forgot".  The tokenizer
    renders fine (V0-11 section 1b) and the processor cannot, on the same
    checkpoint, for that reason alone.  ``chat_template.jinja`` is therefore read
    off disk as a last resort: it is the file that definitely exists, and going
    through ``AutoTokenizer`` to reach it costs a full tokenizer load.

    Also attaches ``.tokenizer`` if it is missing, because ``doc2len`` calls
    ``processor.tokenizer(text=...)`` right after ``apply_chat_template``
    (``rl_dataset.py:222``) -- one missing attribute there produces the same
    silent empty dataset.

    Returns:
        Where the template came from -- ``"native"``, ``"processor.tokenizer"``,
        ``"AutoTokenizer"``, ``"chat_template.jinja"`` -- or ``"missing"``.
        The caller does not branch on this; it is logged, and the real check is
        ``_rejects_doc2len``.
    """
    if getattr(processor, "chat_template", None):
        return "native"

    source = "processor.tokenizer"
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None

    if not template:
        source = "AutoTokenizer"
        try:
            from transformers import AutoTokenizer

            loaded = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.warning(
                "Could not load a tokenizer from %s to source a chat template: %s: %s",
                model_path,
                type(e).__name__,
                e,
            )
            loaded = None
        if loaded is not None:
            template = getattr(loaded, "chat_template", None)
            if getattr(processor, "tokenizer", None) is None:
                # doc2len needs this one line later; without it the failure looks
                # identical to a missing template.
                processor.tokenizer = loaded

    if not template:
        source = "chat_template.jinja"
        try:
            import os

            jinja = os.path.join(model_path, "chat_template.jinja")
            if os.path.exists(jinja):
                with open(jinja, encoding="utf-8") as fh:
                    template = fh.read()
        except Exception as e:
            logger.warning("Could not read chat_template.jinja from %s: %s: %s", model_path, type(e).__name__, e)

    if not template:
        return "missing"

    processor.chat_template = template
    return source


# doc2len's two calls, verbatim (rl_dataset.py:214 and :222).  Anything that
# throws here throws there, and there it is swallowed.
_SMOKE_MESSAGES = [{"role": "user", "content": "ping"}]


def _rejects_doc2len(processor) -> str:
    """Do what ``doc2len`` does; return why it failed, or "" if it works.

    **Why an attribute check is not enough.**  V0-10 (2026-08-29, real
    checkpoint) reported ``chat_template : None`` on a processor that
    ``_ensure_chat_template`` had just been asked to fix.  Whatever the reason --
    a stale copy of this file on the training machine (pitfall 22), or a
    transformers version whose ``apply_chat_template`` resolves the template from
    somewhere other than ``self.chat_template`` -- the lesson is the same as
    pitfall 33: **verify the guess by doing the actual thing**, not by reading an
    attribute that is merely correlated with it.

    So this calls the exact two lines ``doc2len`` calls.  If they work here they
    work there; if they do not, we hand back None and let the tokenizer branch
    take over, which is what text-only training wanted anyway.  The alternative
    -- returning a processor that throws on every row -- costs a full 16-card
    launch to diagnose, because ``rl_dataset.py:241`` turns it into
    ``filter dataset len: 0`` with no message naming the cause.
    """
    try:
        raw = processor.apply_chat_template(_SMOKE_MESSAGES, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        return f"processor.apply_chat_template raised {type(e).__name__}: {e} (rl_dataset.py:214)"
    if not isinstance(raw, str) or not raw:
        return f"processor.apply_chat_template returned {type(raw).__name__} {raw!r}, expected a non-empty str"

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return "processor has no .tokenizer (rl_dataset.py:222 calls processor.tokenizer(text=...))"
    try:
        ids = tokenizer(text=raw, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    except Exception as e:
        return f"processor.tokenizer(text=...) raised {type(e).__name__}: {e} (rl_dataset.py:222)"
    if not len(ids):
        return "processor.tokenizer returned 0 input_ids for a non-empty prompt"
    return ""


"""Placeholder literals, and the one asymmetry between the two consumers.

``(<image>./</image>)`` / ``<audio>./</audio>`` are what ModelBest's own
inference script writes into the flattened message text
(``MiniCPM/minicpmo5-infer_trans_orig.py:25-110``), so they are the strings this
checkpoint was actually driven with.

The two sides do not agree on the parentheses, and it is worth writing down
because it is invisible until it costs a run:

============================  ======================================
HF ``_convert_omni_to_inputs``  ``image_pattern = "<image>./</image>"``
vLLM-Omni ``_get_prompt_updates``  ``image_pattern = "(<image>./</image>)"``
============================  ======================================

The HF processor's ``re.split`` target has **no** parens, so the ``(`` and
``)`` survive as ordinary text tokens either side of the expanded image span.
vLLM-Omni's ``PromptReplacement`` target **includes** them, so vLLM consumes
them.  Audio is the mirror image: vLLM's target carries parens and ModelBest's
script does not write them.

Consequence: after the contraction below, the ids vLLM re-expands can differ
from the ids the HF side produced by the two paren tokens per media item.  That
is a real open question, not a solved one -- it cannot be settled on the
authoring machine because base vLLM's ``_apply_prompt_updates`` is not
installed here.  The signal to check on the first real run is
``training/rollout_actor_probs_pearson_corr``: the text-only run measured
0.986, so a materially lower value on the omni run points here first.
"""
_IMAGE_PLACEHOLDER = "(<image>./</image>)"
# Parens, deliberately, even though ModelBest's inference script writes audio
# WITHOUT them. The paren form is the only one both consumers accept: the HF
# processor's re.split target is the bare inner literal, so it finds and expands
# `<audio>./</audio>` whether or not parens surround it (they survive as two
# ordinary text tokens), whereas vLLM-Omni's PromptReplacement target is the
# paren-wrapped string and matches nothing without them. Writing the bare form
# would work in training and silently fail to place audio embeddings in rollout.
_AUDIO_PLACEHOLDER = "(<audio>./</audio>)"

_VLLM_IMAGE_PATTERN = "(<image>./</image>)"
_VLLM_AUDIO_PATTERN = "(<audio>./</audio>)"

_CALL_SHIM_FLAG = "_verl_minicpmo_call_shim"
_DEDUP_DISABLE_ENV = "MINICPMO_DISABLE_MM_DEDUP"

# Whisper's only defined rate; the launch script also passes it as
# ``++data.mm_processor_kwargs.sampling_rate``. Mirrors
# ``minicpmo_5_0_dataset._DEFAULT_SAMPLING_RATE`` -- duplicated rather than
# imported because that module imports ``verl.utils.dataset.rl_dataset``, and
# the processor is built before the dataset on the rollout replica.
_DEFAULT_SAMPLING_RATE = 16000


def _bless_processor_for_verl(processor):
    """Give the processor a ``__call__`` that speaks verl's keyword vocabulary."""
    cls = type(processor)
    if getattr(cls, _CALL_SHIM_FLAG, False):
        return processor
    orig_convert = getattr(cls, "_convert_omni_to_inputs")

    _PARALLEL_FIELDS = (
        "grids",
        "source_image_visual_tokens",
        "patch_visual_tokens",
    )

    def _read_field(images, field):
        """从属性 / dict / data 三种存储里读，返回 (值, 来源)。"""
        try:
            if hasattr(images, field):
                v = getattr(images, field)
                if v is not None:
                    return v, "attr"
        except Exception:
            pass
        try:
            if field in images:
                return images[field], "dict"
        except Exception:
            pass
        try:
            if hasattr(images, "data") and isinstance(images.data, dict):
                if field in images.data:
                    return images.data[field], "data"
        except Exception:
            pass
        return None, "absent"

    def _write_field(images, field, value):
        """三路写回，确保 orig_convert 无论读哪个存储都能拿到新值。"""
        results = {}
        try:
            setattr(images, field, value)
            results["attr"] = "OK"
        except Exception as e:
            results["attr"] = f"FAIL: {e}"
        try:
            images[field] = value
            results["dict"] = "OK"
        except Exception as e:
            results["dict"] = f"FAIL: {e}"
        try:
            if hasattr(images, "data") and isinstance(images.data, dict):
                images.data[field] = value
                results["data"] = "OK"
            else:
                results["data"] = "SKIP"
        except Exception as e:
            results["data"] = f"FAIL: {e}"
        return results

    def _to_size_tuple(size):
        """把 tensor([224, 224]) / [224, 224] / (224, 224) 统一成 (224, 224)。"""
        if hasattr(size, "tolist"):
            size = size.tolist()
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        return None

    def safe_convert_omni_to_inputs(self, images, *args, **kwargs):
        # ---- 读取 image_sizes（三种存储都试）----
        image_sizes, src = _read_field(images, "image_sizes")

        batch_size = len(image_sizes) if image_sizes is not None else 1

        # ---- 防御 audio_phs[index] 越界（修复 processing_minicpmo line 1682 崩溃）----
        audio_phs, audio_src = _read_field(images, "audio_phs")
        if audio_phs is None or not isinstance(audio_phs, (list, tuple)):
            _write_field(images, "audio_phs", [[] for _ in range(batch_size)])
        elif len(audio_phs) < batch_size:
            audio_phs = list(audio_phs) + [[] for _ in range(batch_size - len(audio_phs))]
            _write_field(images, "audio_phs", audio_phs)

        if image_sizes is None:
            return orig_convert(self, images, *args, **kwargs)

        # ---- 逐字段修补视觉相关元数据 ----
        for field in _PARALLEL_FIELDS:
            old, old_src = _read_field(images, field)
            new_val = []
            for idx, sizes in enumerate(image_sizes):
                expected_len = len(sizes)
                cur = []

                if field == "grids":
                    for s in sizes:
                        t = _to_size_tuple(s)
                        cur.append(t)
                else:
                    if old is not None:
                        try:
                            if idx < len(old):
                                g = old[idx]
                                if isinstance(g, (list, tuple)):
                                    cur = list(g)
                                else:
                                    cur = [g]
                        except Exception:
                            cur = []
                    if len(cur) < expected_len:
                        cur.extend([None] * (expected_len - len(cur)))
                    elif len(cur) > expected_len:
                        cur = cur[:expected_len]

                new_val.append(cur)

            _write_field(images, field, new_val)

        return orig_convert(self, images, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            # ---------------- 1. 注入字符串类型的模态边界 Token ----------------
            # processing_minicpmo.py 的 get_audio_placeholder 用 self.tokenizer.audio_start
            # 构造 audio placeholder 字符串，然后 tokenize。
            # 必须用真实的特殊 token 字符串，不能用 <audio>（=unk_token_id）。
            if not hasattr(tok, "audio_start"):
                tok.audio_start = "<|audio_start|>"
            if not hasattr(tok, "audio_end"):
                tok.audio_end = "<|audio_end|>"

            if not hasattr(tok, "im_start"):
                tok.im_start = "<|vision_start|>"
            if not hasattr(tok, "im_end"):
                tok.im_end = "<|vision_end|>"

            if not hasattr(tok, "slice_start"):
                tok.slice_start = "<|image_pad|>"
            if not hasattr(tok, "slice_end"):
                tok.slice_end = "<|image_pad|>"

            if not hasattr(tok, "spk_start"):
                tok.spk_start = "<|audio_start|>"
            if not hasattr(tok, "spk_end"):
                tok.spk_end = "<|audio_end|>"

            # ---------------- 2. 注入数字 ID 类型的模态边界属性 ----------------
            if not hasattr(tok, "im_start_id"):
                for name in ["<|vision_start|>", "<image>", "<im_start>"]:
                    try:
                        tid = tok.convert_tokens_to_ids(name)
                        if tid is not None and tid != getattr(tok, "unk_token_id", None):
                            tok.im_start_id = tid
                            break
                    except Exception:
                        pass
                if not hasattr(tok, "im_start_id"):
                    tok.im_start_id = -1

            if not hasattr(tok, "im_end_id"):
                for name in ["<|vision_end|>", "</image>", "<im_end>"]:
                    try:
                        tid = tok.convert_tokens_to_ids(name)
                        if tid is not None and tid != getattr(tok, "unk_token_id", None):
                            tok.im_end_id = tid
                            break
                    except Exception:
                        pass
                if not hasattr(tok, "im_end_id"):
                    tok.im_end_id = -1

            if not hasattr(tok, "slice_start_id"):
                for name in ["<|vision_pad|>", "<slice>"]:
                    try:
                        tid = tok.convert_tokens_to_ids(name)
                        if tid is not None and tid != getattr(tok, "unk_token_id", None):
                            tok.slice_start_id = tid
                            break
                    except Exception:
                        pass
                if not hasattr(tok, "slice_start_id"):
                    tok.slice_start_id = -1

            if not hasattr(tok, "slice_end_id"):
                for name in ["<|vision_pad|>", "</slice>"]:
                    try:
                        tid = tok.convert_tokens_to_ids(name)
                        if tid is not None and tid != getattr(tok, "unk_token_id", None):
                            tok.slice_end_id = tid
                            break
                    except Exception:
                        pass
                if not hasattr(tok, "slice_end_id"):
                    tok.slice_end_id = -1

            # 音频与说话人数字 ID 兜底
            for prefix in ("audio", "spk"):
                for suffix in ("start_id", "end_id"):
                    attr = f"{prefix}_{suffix}"
                    if not hasattr(tok, attr):
                        str_attr = f"{prefix}_{suffix.split('_')[0]}"
                        target_str = getattr(tok, str_attr, None)
                        tid = None
                        if target_str:
                            try:
                                tid = tok.convert_tokens_to_ids(target_str)
                            except Exception:
                                pass
                        if tid is None or tid == getattr(tok, "unk_token_id", None):
                            tid = tok.im_end_id
                        setattr(tok, attr, tid)

        # -------------------- 模态参数清洗 --------------------
        videos = kwargs.pop("videos", None)
        kwargs.pop("video_metadata", None)
        kwargs.pop("do_sample_frames", None)
        if videos:
            raise NotImplementedError(
                "MiniCPMOProcessor cannot take video: its __call__ signature is "
                "(text, images, audios, audio_parts, ...) with no `videos` parameter. This is not "
                "a gap -- vLLM-Omni's own process_videos routes frames through the image path at "
                "max_slice_nums=1, so on this model video *is* N images. The supported path is "
                "MiniCPMOOmniRLHFDataset._process_multi_modal_info, which samples frames into "
                "`images` and returns videos=None; reaching here means something upstream put "
                "frames in `videos` instead. See MiniCPMOOmniRLHFDataset._process_multi_modal_info."
            )

        # 统一 audio 命名 -> audios
        if "audio" in kwargs:
            kwargs["audios"] = kwargs.pop("audio")

        raw_images = kwargs.get("images", None)
        images_in_args = False
        raw_text = kwargs.get("text", None)

        if raw_images is None and len(args) > 1:
            raw_images = args[1]
            images_in_args = True

        if raw_text is None and len(args) > 0:
            raw_text = args[0]

        # -------------------- 图像维度规整 (batch x items) --------------------
        if raw_images is not None:
            if hasattr(raw_images, "size") and hasattr(raw_images, "convert"):
                adapted_images = [[raw_images]]
            elif isinstance(raw_images, (list, tuple)) and len(raw_images) > 0:
                if not isinstance(raw_images[0], (list, tuple)):
                    if isinstance(raw_text, str) or (
                        isinstance(raw_text, (list, tuple)) and len(raw_text) == 1
                    ):
                        adapted_images = [list(raw_images)]
                    else:
                        adapted_images = raw_images
                else:
                    adapted_images = raw_images
            else:
                adapted_images = raw_images

            if images_in_args:
                args = (args[0], adapted_images) + args[2:]
            else:
                kwargs["images"] = adapted_images

        # -------------------- 音频维度规整 (batch x items) --------------------
        raw_audios = kwargs.get("audios", None)
        if raw_audios is not None:
            if not isinstance(raw_audios, (list, tuple)):
                kwargs["audios"] = [[raw_audios]]
            elif len(raw_audios) > 0 and not isinstance(raw_audios[0], (list, tuple)):
                if isinstance(raw_text, str) or (
                    isinstance(raw_text, (list, tuple)) and len(raw_text) == 1
                ):
                    kwargs["audios"] = [list(raw_audios)]
                else:
                    kwargs["audios"] = [[a] for a in raw_audios]

        return super(shim, self).__call__(*args, **kwargs)

    shim = type(
        f"{cls.__name__}VerlShim",
        (cls,),
        {
            "__call__": __call__,
            "_convert_omni_to_inputs": safe_convert_omni_to_inputs,
            _CALL_SHIM_FLAG: True,
        },
    )
    processor.__class__ = shim
    return processor


def _flatten_messages(messages):
    """Collapse verl's structured content lists into MiniCPM-o's plain-string form.

    ``RLHFDataset._build_messages`` rewrites ``"...<image>...<audio>..."`` into
    ``[{"type": "text", ...}, {"type": "image", "image": PIL}, ...]`` because
    that is what a Qwen-style ``processor.apply_chat_template`` consumes.
    MiniCPM-o's chat template does not: ModelBest's own script joins the parts
    with newlines into one string carrying inline ``(<image>./</image>)`` /
    ``<audio>./</audio>`` literals, and the processor's
    ``_convert_omni_to_inputs`` then finds those literals by regex.

    Returns new message dicts; the input is not mutated.  That matters because
    the agent loop calls ``process_multi_modal_info(messages)`` and
    ``apply_chat_template(messages)`` on the *same* list, and rewriting content
    in place under the first caller would change what the second one sees.

    One placeholder is emitted per media item, in order, which is what keeps
    ``_convert_omni_to_inputs``'s ``assert len(image_tags) ==
    len(image_sizes[index])`` true: the dataset's
    ``_process_multi_modal_info`` walks the same items in the same order.

    Video is the one item type that is not one-to-one.  MiniCPM-o has no video
    input path -- vLLM-Omni's ``process_videos`` routes frames through the
    *image* path at ``max_slice_nums=1``, and the HF processor has no ``videos``
    parameter at all -- so a ``<video>`` item becomes ``N`` image placeholders,
    one per sampled frame, plus one audio placeholder when the clip's track is
    being carried.  ``N`` comes from the same sampler the dataset used
    (``minicpmo_5_0_video.count_frames``, cache-backed), because a count that
    disagrees with the decoded frames is precisely the assertion above firing.
    """
    flattened = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            flattened.append(message)
            continue

        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            kind = item.get("type")
            if kind == "text":
                parts.append(item.get("text", ""))
            elif kind == "image":
                parts.append(_IMAGE_PLACEHOLDER)
            elif kind == "audio":
                parts.append(_AUDIO_PLACEHOLDER)
            elif kind == "video":
                parts.append(_video_placeholders(item))
            else:
                raise NotImplementedError(
                    f"MiniCPM-o message flattening met content type {kind!r}, which has no "
                    "placeholder literal on this checkpoint. Supported: text, image, video, audio."
                )
        flattened.append({**message, "content": "\n".join(parts)})
    return flattened


def _video_placeholders(item):
    """Render one ``<video>`` item as its frame (and optional audio) placeholders.

    Frames are joined with newlines rather than concatenated, matching how
    ModelBest's own script separates consecutive media in a message.  The
    separator is ordinary text on both sides -- the HF processor's ``re.split``
    target is the inner literal and vLLM-Omni's ``PromptReplacement`` target is
    the paren-wrapped literal, and neither includes surrounding whitespace -- so
    it changes the token count by one per frame boundary and nothing else.

    The audio placeholder is emitted only when the clip actually yielded a
    waveform.  ``extract_video_audio`` returns None for a silent or
    undecodable track and the dataset drops the entry, so asking it here (the
    decode is cached) is what keeps the placeholder count and the media count
    equal for a partly-silent corpus.
    """
    from verl_omni_ext.models.qwen3_6_whisper.minicpmo_5_0_video import (
        count_frames,
        extract_video_audio,
        use_audio_in_video,
        video_spec_from_item,
    )

    spec = video_spec_from_item(item)
    parts = [_IMAGE_PLACEHOLDER] * count_frames(spec)
    if use_audio_in_video():
        sampling_rate = int(item.get("sampling_rate") or _DEFAULT_SAMPLING_RATE)
        if extract_video_audio(spec, sampling_rate) is not None:
            parts.append(_AUDIO_PLACEHOLDER)
    return "\n".join(parts)


def _bind_chat_template(processor):
    """Route ``processor.apply_chat_template`` through the tokenizer, after flattening.

    Two independent reasons this cannot be the inherited implementation:

    * ``ProcessorMixin.apply_chat_template`` renders whatever content it is
      given, and this checkpoint's template does not understand verl's
      structured lists -- it expects the flat string form.
    * ModelBest's own script calls ``processor.tokenizer.apply_chat_template``,
      not ``processor.apply_chat_template``.  The tokenizer is where the
      template lives on this checkpoint (``chat_template.jinja`` is present,
      ``chat_template.json`` is absent -- see ``_ensure_chat_template``).

    Plain-string content passes through untouched, so ``doc2len``'s length
    measurement and ``_rejects_doc2len``'s smoke test behave exactly as before.
    """
    import types

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(_flatten_messages(messages), **kwargs)

    processor.apply_chat_template = types.MethodType(apply_chat_template, processor)
    return processor


def _bind_dedup_pad_tokens(processor):
    """Contract expanded media spans back to their placeholder literals.

    The mismatch this bridges
    ------------------------
    ``AgentLoopWorker.apply_chat_template`` takes ``prompt_ids`` from the *HF
    processor*, so they are **already expanded**: each media item has become
    ``im_start unk*N im_end`` (plus slice structure for UHD images) or
    ``audio_start unk*N audio_end``.  Training needs exactly that -- ``image_bound``
    indexes into those ids.

    Rollout needs the opposite.  ``_preprocess_input`` hands vLLM
    ``{"prompt_token_ids": prompt_ids}``, and vLLM-Omni's ``_get_prompt_updates``
    installs ``PromptReplacement`` objects whose *target* is the placeholder
    literal ``(<image>./</image>)`` / ``(<audio>./</audio>)``.  Against
    already-expanded ids that target matches nothing, so no media embedding gets
    placed.

    So the contraction is not an optimisation -- it is what makes the rollout
    prompt and the training prompt describe the same thing.  ``verl-omni`` calls
    it at ``vllm_omni_async_server.py:539`` for exactly this purpose.

    Why this differs from the Qwen implementation
    ---------------------------------------------
    ``Qwen3OmniThinkerAdapter``'s version collapses *runs of one repeated pad
    token*, because Qwen's replacement target is a single ``<|image_pad|>``.
    MiniCPM-o's target is a multi-token text pattern, and its expanded span is
    heterogeneous (start/end sentinels, unk runs, per-slice sentinels).  So this
    rewrites whole spans into the pattern's token ids rather than thinning a run.

    What is NOT verified
    --------------------
    Base vLLM's ``_apply_prompt_updates`` is not installed on the authoring
    machine, so the claim "vLLM re-expands this to exactly the length the HF
    processor produced" is reasoned, not measured.  Hence: a loud one-time report
    of what was contracted, and ``MINICPMO_DISABLE_MM_DEDUP=1`` to turn the whole
    thing off.  If rollout logprobs and training logprobs disagree on the omni
    run -- watch ``training/rollout_actor_probs_pearson_corr``, which the
    text-only run measured at 0.986 -- this function is the first suspect.
    """
    import types

    reported = []

    def dedup_pad_tokens(self, prompt_ids):
        if os.environ.get(_DEDUP_DISABLE_ENV, "0") == "1":
            return prompt_ids

        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return prompt_ids

        spec = _media_span_spec(tokenizer)
        if spec is None:
            return prompt_ids

        contracted, counts = _contract_media_spans(list(prompt_ids), spec)

        if not reported:
            reported.append(True)
            # print, not logger: logger.info from a Ray worker does not reach the
            # driver's captured output on this stack (proven twice).
            print(
                "[verl-omni] MiniCPM-o rollout prompt contraction, first call: "
                f"{len(prompt_ids)} -> {len(contracted)} ids; "
                f"image spans={counts['image']} audio spans={counts['audio']}. "
                "vLLM-Omni re-expands these placeholder literals itself; training keeps the "
                f"expanded ids. Set {_DEDUP_DISABLE_ENV}=1 to disable.",
                flush=True,
            )

        return contracted

    processor.dedup_pad_tokens = types.MethodType(dedup_pad_tokens, processor)
    return processor


def _media_span_spec(tokenizer):
    """Collect the token ids that delimit MiniCPM-o's expanded media spans.

    Returns None when the tokenizer does not expose them, which is the signal to
    leave the prompt alone rather than guess.  All six sentinel ids are set by
    the checkpoint's own tokenizer (``im_start_id`` / ``slice_start_id`` /
    ``audio_start_id`` and their ``_end_`` counterparts) and read the same way by
    ``MiniCPMOProcessor._convert``.
    """
    names = (
        "im_start_id",
        "im_end_id",
        "slice_start_id",
        "slice_end_id",
        "audio_start_id",
        "audio_end_id",
    )
    ids = {}
    for name in names:
        value = getattr(tokenizer, name, None)
        if value is None:
            logger.warning(
                "MiniCPM-o prompt contraction disabled: tokenizer has no %s, so the expanded "
                "media spans cannot be located. Rollout will receive already-expanded prompt ids "
                "and vLLM-Omni's placeholder replacement will match nothing.",
                name,
            )
            return None
        ids[name] = int(value)

    def encode(pattern):
        return list(tokenizer.encode(pattern, add_special_tokens=False))

    try:
        image_pattern_ids = encode(_VLLM_IMAGE_PATTERN)
        audio_pattern_ids = encode(_VLLM_AUDIO_PATTERN)
    except Exception as e:
        logger.warning("MiniCPM-o prompt contraction disabled: could not encode placeholders: %r", e)
        return None

    return {
        **ids,
        "image_pattern_ids": image_pattern_ids,
        "audio_pattern_ids": audio_pattern_ids,
    }


def _contract_media_spans(ids, spec):
    """Replace each expanded media span with its placeholder literal's token ids.

    An image span runs from an ``im_start`` to the last ``im_end`` / ``slice_end``
    that still belongs to the same image.  UHD images expand to a *group* --
    a global view followed by slice rows, with ordinary text tokens (newlines)
    between them -- and the whole group corresponds to one
    ``(<image>./</image>)`` in the source text, so the group is contracted as a
    unit.  A span is only continued across a gap if the very next sentinel is a
    ``slice_start``; anything else closes it.

    Audio spans are simple: ``audio_start`` .. ``audio_end``, no slicing.
    """
    image_starts = {spec["im_start_id"], spec["slice_start_id"]}
    image_ends = {spec["im_end_id"], spec["slice_end_id"]}
    audio_start, audio_end = spec["audio_start_id"], spec["audio_end_id"]

    out, counts = [], {"image": 0, "audio": 0}
    i, n = 0, len(ids)

    while i < n:
        token = ids[i]

        if token in image_starts:
            j = _scan_image_group(ids, i, image_starts, image_ends, spec["slice_start_id"])
            if j is not None:
                out.extend(spec["image_pattern_ids"])
                counts["image"] += 1
                i = j
                continue

        if token == audio_start:
            j = _scan_to(ids, i, audio_end)
            if j is not None:
                out.extend(spec["audio_pattern_ids"])
                counts["audio"] += 1
                i = j
                continue

        out.append(token)
        i += 1

    return out, counts


def _scan_image_group(ids, start, image_starts, image_ends, slice_start_id):
    """Index just past a full image group, or None if it never closes.

    Returning None on an unterminated span is deliberate: the token is then
    emitted verbatim and the prompt is left as it was.  A truncated prompt
    (``max_prompt_length`` cutting through an image) is the realistic way to get
    here, and passing it through unchanged is safer than contracting half a span.
    """
    n = len(ids)
    i, end = start, None

    while i < n:
        if ids[i] in image_ends:
            end = i + 1
            # A slice row may follow, separated by newline text tokens. Look
            # ahead for the next sentinel; only slice_start continues this group.
            k = end
            while k < n and ids[k] not in image_starts and ids[k] not in image_ends:
                k += 1
            if k < n and ids[k] == slice_start_id:
                i = k
                continue
            return end
        i += 1

    return end


def _scan_to(ids, start, end_id):
    """Index just past ``end_id``, or None if it does not occur after ``start``."""
    for i in range(start + 1, len(ids)):
        if ids[i] == end_id:
            return i + 1
    return None


def build_minicpmo_processor(model_path: str, trust_remote_code: bool = True):
    """Construct MiniCPM-o's remote-code processor directly.

    Use only as a fallback after ``hf_processor`` has returned None.

    Args:
        model_path: Local checkpoint directory.
        trust_remote_code: Must be True for this checkpoint; threaded from the
            trainer config rather than hardcoded so a future natively-supported
            MiniCPM-o still works.

    Returns:
        The processor, or None if this checkpoint is not MiniCPM-o, the
        remote-code import failed, or no chat template could be found for it.
        Returning None (rather than raising) keeps the caller's existing "warn
        and carry on" behaviour intact -- text-only training does not need a
        processor at all.
    """
    try:
        from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase
    except ImportError:  # pragma: no cover - transformers is always present in training
        return None

    try:
        # No use_fast=True: this is remote code with no fast/slow variants, and
        # an unexpected kwarg into remote code is a TypeError for no benefit.
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as e:
        logger.warning(
            "Could not build MiniCPM-o processor from %s: %s: %s. If this is an "
            "AttributeError from AutoImageProcessor.register, the register guard did not run -- "
            "make sure verl_omni was imported (VERL_USE_EXTERNAL_MODULES=verl_omni).",
            model_path,
            type(e).__name__,
            e,
        )
        return None

    if isinstance(processor, PreTrainedTokenizerBase):
        # Newer transformers can legitimately fall back to a tokenizer backend.
        # That is "no multimodal processor", same as hf_processor's own check.
        return None

    actual = processor.__class__.__name__
    if actual != _EXPECTED_PROCESSOR_CLASS:
        logger.warning(
            "Expected %s for MiniCPM-o but AutoProcessor produced %s. Not using it: this "
            "fallback exists specifically to cover hf_processor's unsupported-class path, and "
            "an unexpected class means the assumption behind it no longer holds.",
            _EXPECTED_PROCESSOR_CLASS,
            actual,
        )
        return None

    # hf_processor attaches the config for every class it supports; several verl
    # call sites read processor.config, so keep parity.
    try:
        processor.config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as e:
        logger.warning("Could not attach config to the MiniCPM-o processor: %s: %s", type(e).__name__, e)

    template_source = _ensure_chat_template(processor, model_path, trust_remote_code)

    # Order matters: _rejects_doc2len calls processor.apply_chat_template, so the
    # rebind has to be in place before the smoke test, or the test would verify
    # an implementation that never runs in production.
    #
    # All three shims are installed unconditionally, including for text-only
    # runs. They are inert there -- flattening leaves plain-string content
    # untouched, the __call__ shim only renames kwargs that a text-only call does
    # not send, and dedup_pad_tokens is reached only when multi_modal_data is
    # non-empty (vllm_omni_async_server.py:539). Installing them always is what
    # keeps the text-only path and the omni path the same code path, so a
    # regression in one shows up in both rather than hiding until V5.
    _bless_processor_for_verl(processor)
    _bind_chat_template(processor)
    _bind_dedup_pad_tokens(processor)

    # Do not trust `template_source`: verify by rendering. See _rejects_doc2len.
    rejection = _rejects_doc2len(processor)
    if rejection:
        # Deliberately give back None rather than a processor that cannot render
        # a prompt. With None, RLHFDataset takes its tokenizer branch and
        # text-only training works; with a broken processor, doc2len throws on
        # every row, the exception is swallowed, and the dataset silently comes
        # back empty. A loud assert later beats a silent zero now.
        logger.warning(
            "Built %s from %s (chat_template source=%s) but it cannot do what doc2len does: %s. "
            "Returning None instead -- otherwise RLHFDataset would swallow that on every row "
            "(rl_dataset.py:241) and report `filter dataset len: 0` with nothing naming the "
            "cause. Text-only training still works via the tokenizer branch; multimodal will "
            "fail loudly on `assert self.processor is not None`. This is exactly what V3 run 2 "
            "and run 3 died of; run V0-10 section (5) to see it in seconds instead of 16 cards. "
            "Running from %s -- if that is not the tree you edited, you are hitting pitfall 22.",
            _EXPECTED_PROCESSOR_CLASS,
            model_path,
            template_source,
            rejection,
            _provenance(),
        )
        return None

    logger.info(
        "Built MiniCPM-o processor (%s) directly and verified it renders: verl's hf_processor "
        "rejects this class (`case _: raise ValueError` in verl/utils/tokenizer/tokenizer.py) "
        "and returns None. chat_template source=%s; get_rope_index is intentionally left "
        "unbound -- the model computes its own canvas m-RoPE (V0-7/V0-8). Running from %s.",
        actual,
        template_source,
        _provenance(),
    )
    return processor