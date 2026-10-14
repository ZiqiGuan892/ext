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
"""Forward-signature adapter for MiniCPM-o 5.0 thinker training.

The problem
-----------
verl calls the training module with **keyword arguments only**
(``verl/workers/engine/fsdp/transformer_impl.py:1526``)::

    raw_output = self.module(**model_inputs, use_cache=False)

where ``model_inputs`` is ``{"input_ids", "attention_mask", "position_ids",
*multi_modal_inputs}``, and then reads ``raw_output.logits``
(``transformer_impl.py:1348`` packed / ``:1439`` dense).

MiniCPM-o's remote code does not have that signature.  V0-4 (2026-08-28,
measured) reports::

    class MiniCPMO.forward(self, data, **kwargs)

That is the MiniCPM-V/o family convention: a **single positional dict**.  Called
the way verl calls it, ``data`` never gets bound, and Python raises
``TypeError: forward() missing 1 required positional argument: 'data'`` on the
very first micro-batch -- after the model has loaded, so it fails deep into V3
rather than at startup.

The fix
-------
``configure_model`` installs ``build_minicpmo_forward_adapter(module)``, which
re-exposes a verl-shaped keyword signature and dispatches:

* **Text-only batch** (no multimodal keys) -> call the LLM backbone directly.
  V0-6 confirmed ``MiniCPMO.__init__`` does
  ``self.llm = AutoModelForCausalLM.from_config(text_config)``, so the backbone
  is a stock CausalLM: keyword-compatible with verl and already returning
  ``.logits``.  For MiniCPM-o 5.0 that backbone is Qwen3.5-MoE -- the same code
  path the validated Qwen3.5-MoE GSPO run exercises.  This is the path V1-V3
  take.
* **Multimodal batch** -> repack into the ``data`` dict and call the original
  ``MiniCPMO.forward`` so the vision/audio embedding merge still happens.

Precedent for rebinding forward from ``configure_model`` is
``verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py:47-64``
(``module.forward = module.thinker.forward``).  This module has to do more than
rebind because the two signatures are not merely differently-named, they are
differently-*shaped*.

Status of the multimodal branch
-------------------------------
V0-6 (2026-08-28) read the ``data`` contract out of the checkpoint's remote
code, so the key set below is measured, not guessed::

    MiniCPMO.forward(self, data, **kwargs):
        vllm_embedding, _ = self.get_vllm_embedding(data)          # input_ids, image_bound
        vllm_embedding = self.get_omni_embedding(data, ...)        # audio_bounds, audio_features?
        position_ids = self._compute_position_ids(data)            # input_ids, image_bound,
                                                                   # position_ids, attention_mask?, tgt_sizes?
        return self.llm(input_ids=None, position_ids=position_ids,
                        inputs_embeds=vllm_embedding, **kwargs)

Note ``image_bound`` and ``audio_bounds`` are read with ``data[...]``, not
``data.get(...)``: they are **required even for a text-only batch**.

What the upstream trainer actually puts in ``data`` (2026-08-29, measured)
-------------------------------------------------------------------------
Read out of ModelBest's own HF-side SFT trainer, which drives this very
checkpoint family through this very ``forward``.  Paths are on the training
machine under ``final_minicpm/sft-tunnel/``::

    model_vlu_minicpm/train_minicpmv.py:905    output = vllm_engine(data=batch)
                                       :907    logits = output.logits...

    multimodal_common/dataset/dataloader.py:268-295  # the collator's batch dict
        data['input_ids']    (1, S)          data['target']      (1, S)
        data['position_ids'] (1, S)   <-- 2-D, NOT (3,B,S) or (4,B,S)
        data['cu_seqlens']   (nseq+1,) int32          # packed, like verl
        data['max_seqlen']   int
        data['pixel_values'] [ list[Tensor(3,H,W)] ]  # outer list is per-sample
        data['image_bound']  [ Tensor(N,2) ]  or  [[]]      # :285 / :287
        data['tgt_sizes']    [ Tensor(N,2) ]  or  [[]]      # :290 / :292
        # plus bookkeeping the model ignores: context/length/spans/raw_data/
        # source/keys/temproal_ids/use_4x_downsample
    dataloader_audio.py adds: audio_features (Ba,80,mel), audio_feature_lens,
        audio_bounds, and the tts_* family.

Three consequences, each of which changes this file:

1. **Absent modality is ``[[]]``, not absent and not ``None``.**  The collator
   always emits the key; a packed sequence carrying no image gets a one-element
   outer list holding an empty list.  So "required even for a text-only batch"
   is satisfied by a *sentinel*, not by real data -- which is why the repack
   below synthesises ``[[]]`` instead of refusing the batch.

2. **The caller passes plain 2-D ``position_ids``.**  Upstream never builds a
   4-channel m-RoPE tensor, not even for image batches.

3. **Upstream packs too** (``cu_seqlens`` with a leading zero, ``B=1``, several
   samples concatenated into one 8192-token row), and it still supplies
   ``position_ids`` explicitly.

An earlier revision of this docstring concluded from 2 that "canvas mode
discards verl's position ids" was a contract rather than a defect, and closed
the question.  **That was wrong, and V0-11 measured it wrong** (2026-08-29,
section (2), on this checkpoint)::

    读 data['position_ids'] : False
    调 _text_position_ids   : True

MiniCPM-o **5.0** ignores the position ids it is handed and recomputes them
from ``input_ids``/``attention_mask``.  What upstream passes is therefore not
evidence about what 5.0 consumes -- see pitfall 34b.

Cross-check on a working GPU RL run (2026-08-29)
------------------------------------------------
``MB_NPU/verl_mm-feature-one-step-offline_half_duplex_omni`` is a *working*
verl-based RL stack for MiniCPM-o **4.5** -- same architecture, the LLM
backbone swapped from Qwen3-8B to Qwen3.5.  Its shipped remote code
(``scripts/MiniCPM_o_4_5/modeling_minicpmo.py:783-800``) is the direct
contrast::

    def forward(self, data, **kwargs):
        vllm_embedding, _ = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(data, input_embeddings=..., ...)
        position_ids = data["position_ids"]          # <-- read, then passed through
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()
        return self.llm(input_ids=None, position_ids=position_ids,
                        inputs_embeds=vllm_embedding, **kwargs)

No ``_compute_position_ids``, no canvas branch: **4.5 lets the caller own the
positions; 5.0 took that ownership away.**  And the caller in that stack builds
them as a plain 1-D ramp even for image batches
(``verl/utils/dataset/minicpmo_dataset.py:886``)::

    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

So the hazard is real, it is specific to 5.0, and it is not something the
working reference had to solve.  Concretely, with
``use_remove_padding=True`` verl passes ``(1, total_nnz)`` and
``attention_mask=None``; ``_text_position_ids`` then has nothing to restart on
and lays one monotonic ramp across the whole pack, mis-positioning every
sub-sequence after the first -- silently.  ``_reject_packed_mm_batch`` below
refuses that combination rather than let it train.

What the working reference *does* solve is the other half of packing: how the
multimodal payload is rebased into packed coordinates.  Its recipe, for when
this branch is enabled (``verl/utils/dataset/vision_utils.py:105-141`` and
``verl/utils/model.py:733-741``)::

    # 1. per-sample lists, NOT torch.cat -- concatenating destroys the
    #    per-sample structure get_vllm_embedding indexes with [i]
    # 2. image_bound[i] -= left_padding_length[i]; image_bound[i] += cu_seqlens[i]
    # 3. pixel_values -> [flat_list]      (one packed "sample")
    #    image_bound  -> [vstack(bounds)] (None if empty)
    #    tgt_sizes    -> [vstack(sizes)]  (None if empty)
    # 4. multi_modal_inputs.pop("image_sizes")  -- bookkeeping the LLM rejects

That is deliberately *not* implemented here: without a 5.0 position-id answer
it would produce a batch that is right about pixels and wrong about positions,
which is the most expensive kind of wrong, because it does not raise.  It is
recorded so the mm phase starts from a known recipe.

The multimodal branch therefore stays behind ``MINICPMO_ENABLE_MM_FORWARD=1``.
Text-only RL (V1-V3) is unaffected: it goes straight to the backbone with
verl's own packed, per-sequence-restarting positions, which are correct by
construction.  See implementation plan §4.7.
"""

import logging
import os

import torch

# from verl_omni.utils import hang_probe as _hang_probe

logger = logging.getLogger(__name__)

_FORWARD_FLAG = "_verl_omni_minicpmo_forward_adapter"

# Marks the module as already having the vision position-embedding device hook
# installed, so repeated calls to build_minicpmo_forward_adapter don't
# double-register it.
_DEVICE_HOOK_FLAG = "_verl_omni_minicpmo_vision_pos_emb_device_hook"

# Marks the module as already having the audio encoder FSDP unshard hook
# installed, so repeated calls to build_minicpmo_forward_adapter don't
# double-register it.
_AUDIO_UNSHARD_FLAG = "_verl_omni_minicpmo_audio_fsdp_unshard_hook"

# How many times _apm_unshard_pre_hook has run in this process.  Printed with
# every trace line so a per-rank divergence is visible by comparing counters
# across ranks in one log.
_APM_UNSHARD_CALL_COUNT = 0

# Per-call tracing for the multimodal path: one line per audio-encoder unshard
# and one per multimodal forward, on every rank.  Off by default because a
# 700-step run would emit hundreds of thousands of lines; turn it on for a
# diagnostic run.  Failures and first calls print regardless of this setting.
_MM_TRACE = os.environ.get("MINICPMO_MM_TRACE", "0") == "1"

# Trace the first N calls unconditionally, not just the first one.
#
# This is the single change that would have saved logs 20-26.  The old gate was
# `call_no == 1`, and compute_log_prob runs exactly 2 micro-batches per rank
# (train_batch_size 8 x rollout.n 8 / 32 ranks / micro_batch_size_per_gpu 1).
# So round 1 was traced, round 2 -- the one every run dies in -- was silent, and
# the "32 ranks all report n=1, therefore the collective counts match" reading
# of log 24 was an artefact of the gate rather than a measurement.
#
# 8 covers both compute_log_prob rounds plus the actor update rounds with room
# to spare, at a cost of a few hundred lines per run.
_MM_TRACE_FIRST_N = int(os.environ.get("MINICPMO_MM_TRACE_FIRST_N", "8"))

# How many multimodal (data-dict route) forwards have run in this process.
_MM_FORWARD_CALL_COUNT = 0

# Candidate attribute names for the LLM backbone inside MiniCPMO, most likely
# first.  V0-6 confirmed ``llm`` on the released checkpoint --
# ``self.llm = AutoModelForCausalLM.from_config(text_config)`` in
# ``MiniCPMO.__init__`` -- so it is an ``AutoModelForCausalLM``, which already
# returns an object carrying ``.logits``.  The rest are defensive.
_LLM_ATTR_CANDIDATES = ("llm", "language_model", "model", "thinker")

# Candidate attribute names for the vision tower inside MiniCPM-o.  V0-6
# measured ``self.vpm`` on this checkpoint -- ``MiniCPMO.__init__`` does
# ``self.vpm = MiniCPMVision(...)``.  The rest are defensive.
_VPM_ATTR_CANDIDATES = ("vpm", "vision_tower", "vision_model", "vision")

# Candidate attribute names for the audio encoder inside MiniCPM-o.  The
# checkpoint's ``MiniCPMO.__init__`` does ``self.apm = WhisperEncoder(...)``
# (the audio projection module).  The rest are defensive.
_APM_ATTR_CANDIDATES = ("apm", "audio_encoder", "audio_model", "audio_tower")

# Keys verl puts in model_inputs that are *not* multimodal payload.  Anything
# else arriving alongside them came from multi_modal_inputs.
_CORE_KEYS = frozenset({"input_ids", "attention_mask", "position_ids", "inputs_embeds"})

# Kwargs verl adds that the LLM backbone understands directly.
_PASSTHROUGH_KEYS = frozenset({"use_cache", "cu_seqlens", "cu_seqlens_cpu", "return_dict"})

# `data` keys the remote code reads with data[...] -- absent means KeyError.
# Measured by V0-6; see the module docstring.
_DATA_REQUIRED_KEYS = ("input_ids", "image_bound", "audio_bounds", "position_ids")

# Of those, the ones that must carry a real tensor.  The other two are modality
# bounds, and upstream satisfies them with a sentinel when the modality is
# absent (dataloader.py:287 / :292), so this adapter does the same.
_DATA_REQUIRED_TENSOR_KEYS = ("input_ids", "position_ids")

# Upstream's "this packed row has no image/audio" value.  Note the shape: an
# outer per-sample list (length B) whose entry is an empty list, *not* None and
# *not* an empty outer list -- ``get_vllm_embedding`` indexes ``[i]`` per sample.
_MODALITY_BOUND_KEYS = ("image_bound", "audio_bounds")

# `data` keys the remote code reads with data.get(...) -- safe to omit.
_DATA_OPTIONAL_KEYS = ("attention_mask", "tgt_sizes", "audio_features")

# Escape hatch for the packed-multimodal refusal. Deliberately a *different*
# variable from MINICPMO_ENABLE_MM_FORWARD, because they gate different risks:
# that one gates "unvalidated", this one gates "measured to be wrong".
_PACKED_MM_OVERRIDE = "MINICPMO_ALLOW_PACKED_MM_FORWARD"

# Bookkeeping keys MiniCPM* processors emit that the LLM forward does not
# accept. verl_mm drops this one explicitly, with the comment "MiniCPMV model
# forward does not consume image_sizes; leaving it will be passed through to
# the LLM and can error" (verl/utils/model.py:738-740,
# verl/utils/dataset/vision_utils.py:139-140).
_MM_DROP_KEYS = ("image_sizes",)


def _empty_bound_sentinel(batch_size):
    """Upstream's empty-modality value: one empty list per sample.

    ``dataloader.py:285-292`` writes ``[torch.vstack(bounds)]`` when the packed
    row carries images and ``[[]]`` when it does not -- always the key, never
    ``None``.  Upstream always packs to ``B=1``; verl may not, so this scales
    with the batch instead of hard-coding ``[[]]``.
    """
    return [[] for _ in range(max(int(batch_size), 1))]


def _pad_and_concat_audio_features(value):
    """Collapse a per-sample list of ``[n_chunks, mel, T]`` into one padded 3-D tensor.

    ``get_audio_embedding`` reads ``batch_size, _, max_mel_seq_len =
    wavforms.shape`` and then builds its attention mask from ``max_mel_seq_len``,
    so it needs a single tensor whose first dimension counts audio chunks across
    the whole batch.  What arrives instead is a Python list -- one entry per
    sample -- because ``extract_multi_modal_inputs`` skips ``torch.cat`` on the
    ``has_image_bound`` branch.

    Each sample's entry is already padded across *its own* chunks by
    ``audio_feature_extract``, but two samples' padding lengths need not agree,
    so a bare ``torch.cat`` on the time axis would raise.  Pad to the batch-wide
    maximum first, then concatenate.  Zero padding is what the processor itself
    pads with, and ``audio_feature_lens`` carries the true lengths, so the
    padding is masked out downstream rather than read as signal.

    A tensor that is already 3-D passes through untouched.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value

    parts = [v for v in value if v is not None and len(v)]
    if not parts:
        return value

    parts = [v if v.dim() == 3 else v.unsqueeze(0) for v in parts]
    max_len = max(v.shape[-1] for v in parts)
    padded = [
        v if v.shape[-1] == max_len else torch.nn.functional.pad(v, (0, max_len - v.shape[-1]))
        for v in parts
    ]
    return torch.cat(padded, dim=0)


def _concat_audio_feature_lens(value):
    """Flatten the per-sample audio lengths into one 1-D tensor.

    Must stay in step with ``_pad_and_concat_audio_features``: the model pairs
    each row of the concatenated features with one entry here, so a per-sample
    list would misalign every chunk after the first sample's.

    Returns a **list of 1-D tensors**, not a single 1-D tensor. The remote
    ``get_audio_embedding`` calls ``torch.hstack(audio_feature_lens_raw)``,
    which requires a sequence of tensors -- a single tensor raises
    ``TypeError: hstack(): argument 'tensors' must be tuple of Tensors, not
    Tensor``. The vllm_omni fork handles a tensor input by ``unbind(0)``
    first, but the checkpoint's HF-cached copy does not, so the adapter must
    hand back a list. Each entry is flattened so a per-sample ``(1, n_chunks)``
    shape becomes ``(n_chunks,)`` -- what hstack expects from each piece.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return [value.flatten()]

    parts = [torch.as_tensor(v).flatten() for v in value if v is not None and len(v)]
    if not parts:
        return value

    return parts



def _is_empty_payload(value):
    """True if *value* carries no actual modality data.

    The distinction this function draws is the whole reason it exists.  verl's
    ``_compute_multi_modal_inputs`` (agent_loop.py:880-892) calls
    ``build_multimodal_processor_inputs`` **unconditionally** whenever a
    processor exists -- with ``images=None, videos=None, audio=None`` for a
    text-only row.  Qwen2.5-VL's processor answers that call by omitting
    ``pixel_values`` entirely, so upstream's presence-based reasoning ("this key
    is here, therefore there is an image") happens to hold.  MiniCPM-o's
    processor answers it by emitting the *full* key set with empty containers,
    per the ``[[]]`` sentinel convention documented in this module's docstring.

    So on this checkpoint a text-only gsm8k batch arrives wearing
    ``pixel_values``, ``image_bound``, ``audio_features`` and five more.
    Presence means nothing here; only content does.

    Conservative by construction: anything this function does not recognise is
    reported as *non*-empty, so an unrecognised payload takes the refusal path
    rather than being silently waved through to the text bypass.
    """
    if value is None:
        return True
    numel = getattr(value, "numel", None)
    if callable(numel):  # torch.Tensor
        try:
            return numel() == 0
        except Exception:
            return False
    shape = getattr(value, "shape", None)
    if shape is not None:  # numpy and friends; () stays non-empty
        try:
            return 0 in tuple(shape)
        except TypeError:
            return False
    if isinstance(value, (list, tuple)):
        # all(()) is True, so both [] and the [[]] sentinel report empty.
        return all(_is_empty_payload(v) for v in value)
    if isinstance(value, dict):
        return all(_is_empty_payload(v) for v in value.values())
    return False


def _describe(value):
    """Shape if it has one, else the type name. Never raises; mocks pass strings."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)}"
    if value is None:
        return "None"
    return type(value).__name__


def _position_embedding_device_hook(module, args, kwargs):
    """forward_pre_hook: move position-embedding indices onto the weight's device.

    The checkpoint's remote ``SiglipVisionEmbeddings.forward`` computes
    ``pos_ids`` on CPU (``torch.arange`` / ``torch.bucketize`` inside
    ``_create_position_ids``) and feeds them straight to
    ``self.position_embedding(pos_ids)``.  After FSDP shard-up the embedding
    weight sits on NPU:0, so ``nn.Embedding`` raises
    ``Expected all tensors to be on the same device, but got indices is on cpu``.

    The vllm_omni fork fixes this with ``.to(weight.device)`` at the end of
    ``_create_position_ids``; the HF-cached copy this checkpoint ships does
    not.  Rather than edit the cached remote file, this hook moves the input
    onto the weight's device before the lookup runs.
    """
    weight_device = module.weight.device
    moved = False
    new_args = list(args)
    if new_args and isinstance(new_args[0], torch.Tensor) and new_args[0].device != weight_device:
        new_args[0] = new_args[0].to(weight_device)
        moved = True
    for _k in ("input", "indices"):
        _v = kwargs.get(_k)
        if isinstance(_v, torch.Tensor) and _v.device != weight_device:
            kwargs[_k] = _v.to(weight_device)
            moved = True
    if moved:
        return tuple(new_args), kwargs
    return None


def patch_vision_position_embedding_device(module):
    """Install the device-fix hook on the vision tower's position embeddings.

    Idempotent: re-applying to an already-patched module is a no-op (the
    ``_DEVICE_HOOK_FLAG`` attribute guards against double-registration).

    Scoped to ``nn.Embedding`` submodules whose dotted name contains
    "position", so the LLM backbone's ``embed_tokens`` is untouched.
    """
    if getattr(module, _DEVICE_HOOK_FLAG, False):
        return
    vpm = None
    for name in _VPM_ATTR_CANDIDATES:
        sub = getattr(module, name, None)
        if sub is not None and hasattr(sub, "forward"):
            vpm = sub
            break
    if vpm is None:
        # print, not logger: muted on the vLLM-driver ranks (see the note in
        # patch_audio_encoder_fsdp_unshard). A silently skipped patch is how
        # log 18 lost three hours to a device mismatch.
        print(
            "[verl-omni] MiniCPM-o vision tower not found under "
            f"{_VPM_ATTR_CANDIDATES}; skipping position-embedding device patch. "
            "If training fails with 'indices is on cpu, different from other "
            "tensors on npu', this is why.",
            flush=True,
        )
        return

    patched = []
    for child_name, child in vpm.named_modules():
        if isinstance(child, torch.nn.Embedding) and "position" in child_name:
            child.register_forward_pre_hook(_position_embedding_device_hook, with_kwargs=True)
            patched.append(child_name)
    if patched:
        setattr(module, _DEVICE_HOOK_FLAG, True)
        print(
            "[verl-omni] MiniCPM-o vision position-embedding device patch "
            f"installed on: {patched}",
            flush=True,
        )
    else:
        print(
            "[verl-omni] No position_embedding nn.Embedding found on the vision "
            f"tower ({type(vpm).__name__}); skipping device patch. If training "
            "fails with 'indices is on cpu', this is why.",
            flush=True,
        )


def _dist_rank():
    """Global rank, or -1 when torch.distributed is not up yet.

    Only used to label diagnostics, so it must never raise.
    """
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _apm_unshard_pre_hook(module, args):
    """forward_pre_hook: unshard all FSDP2-wrapped nn.Embedding children on the
    audio encoder before its forward runs.

    The problem: verl's ``_select_fsdp2_wrap_targets`` (fsdp_utils.py:554)
    wraps EVERY ``nn.Embedding`` individually with ``fully_shard``.  That
    includes the Whisper audio encoder's position embedding.  When the
    Whisper forward reads ``self.embeddings.position_embedding.weight``
    directly (instead of calling it through ``__call__``), the FSDP2
    all-gather hook never fires, and ``.weight`` is still a DTensor.  Adding
    it to ``inputs_embeds`` (a plain tensor) raises::

        RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor

    The vision tower does not hit this because its forward calls
    ``self.position_embedding(pos_ids)`` via ``__call__``, which triggers the
    FSDP2 all-gather.  The audio encoder reads ``.weight`` directly, so we
    must unshard proactively here.

    This hook iterates over the audio encoder's submodules, finds FSDP2-wrapped
    ones (detected by having a callable ``unshard`` method, which only
    ``FSDPModule`` instances expose), and calls ``unshard()`` on each.  The
    reshard happens automatically when the parent FSDP unit's post-forward
    hook fires.
    """
    global _APM_UNSHARD_CALL_COUNT
    _APM_UNSHARD_CALL_COUNT += 1
    call_no = _APM_UNSHARD_CALL_COUNT

    # Sorted by dotted name so every rank walks the same modules in the same
    # order.  The module tree is structurally identical across ranks, so this
    # makes the *sequence* of unshard() calls rank-invariant.  named_modules()
    # is already deterministic in practice; sorting removes the assumption,
    # which matters because each unshard() is an all-gather.
    targets = sorted(
        (
            (_name, child)
            for _name, child in module.named_modules()
            if callable(getattr(child, "unshard", None))
        ),
        key=lambda name_child: name_child[0],
    )

    unsharded, failed = [], []
    for _name, child in targets:
        # Per-unshard heartbeat, not per-hook: each unshard() is an all-gather,
        # so if we wedge inside this loop the watchdog dump has to name the
        # exact module we wedged on -- "apm unshard call=2" alone would not.
        #_hang_probe.beat(f"apm_unshard call={call_no} -> {_name}")
        try:
            child.unshard()
            unsharded.append(_name)
        except Exception as exc:
            # Do NOT swallow this.  The old version passed silently on the
            # theory that a failure meant "already unsharded", but an
            # already-unsharded module returns cleanly -- so anything landing
            # here is a real problem that was previously invisible.
            failed.append(f"{_name}: {type(exc).__name__}: {exc}")
    #_hang_probe.beat(f"apm_unshard call={call_no} done")

    # print, not logger: Ray workers swallow logger output on the vLLM-driver
    # ranks (0/4/8/12 under TP=4), which is why log 20 had zero visibility
    # here.  unshard() issues an all-gather, so if this count differs across
    # ranks the next collective deadlocks with no Python error anywhere --
    # exactly the 174s silent hang in log 20.  Compare `call=` and `n=` across
    # ranks: they must match rank for rank at every call number.
    if failed or call_no <= _MM_TRACE_FIRST_N or _MM_TRACE:

        print(
            f"[verl-omni] apm unshard rank={_dist_rank()} call={call_no} "
            f"n={len(unsharded)}/{len(targets)} failed={failed or '(none)'}",
            flush=True,
        )


def patch_audio_encoder_fsdp_unshard(module):
    """Install the FSDP unshard hook on the audio encoder.

    Idempotent: re-applying to an already-patched module is a no-op (the
    ``_AUDIO_UNSHARD_FLAG`` attribute guards against double-registration).
    """
    if getattr(module, _AUDIO_UNSHARD_FLAG, False):
        return
    apm = None
    for name in _APM_ATTR_CANDIDATES:
        sub = getattr(module, name, None)
        if sub is not None and hasattr(sub, "forward"):
            apm = sub
            break
    if apm is None:
        # print, not logger: Ray workers suppress logger.warning (proven
        # multiple times in earlier logs). This diagnostic must survive.
        print(
            "[verl-omni] MiniCPM-o audio encoder not found under "
            f"{_APM_ATTR_CANDIDATES}; skipping FSDP unshard patch. "
            "If training fails with 'mixed torch.Tensor and DTensor' in "
            "the audio encoder, this is why.",
            flush=True,
        )
        return

    apm.register_forward_pre_hook(_apm_unshard_pre_hook)
    setattr(module, _AUDIO_UNSHARD_FLAG, True)

    # Detect FSDP-wrapped children by their unshard method (FSDPModule instances
    # have it; plain nn.Module does not).
    fsdp_children = [
        n for n, m in apm.named_modules()
        if callable(getattr(m, "unshard", None))
    ][:20]

    # print, not logger: Ray workers suppress logger.info (proven by the fact
    # that the vision position-embedding patch's logger.info never appeared
    # in logs 21-23 despite the patch demonstrably working).
    print(
        "[verl-omni] MiniCPM-o audio encoder FSDP unshard hook installed on "
        f"{type(apm).__name__} (FSDP children with unshard: {fsdp_children}).",
        flush=True,
    )


def find_llm_backbone(module):
    """Return ``(attr_name, submodule)`` for MiniCPM-o's LLM backbone.

    The candidate list is ordered, so this is a *guess* -- and pitfall 33 (V0-9
    resolved Whisper instead of the LLM and produced a syntactically fine,
    entirely wrong analysis) says every guess must be followed by a check that
    what came back is what was wanted.  Here that check is ``lm_head``: the
    caller is about to read ``.logits`` off this submodule's output, which only
    a CausalLM produces.  A vision or audio tower would sail through the
    ``hasattr(sub, "forward")`` test and fail much later with a shape error.

    The expected resolution on this checkpoint is
    ``llm -> Qwen3_5MoeForCausalLM``: ``MiniCPMO.__init__`` does
    ``self.llm = AutoModelForCausalLM.from_config(text_config)``, and the
    hf2mcore converter's parameter tree confirms the sub-keys
    (``llm.model.embed_tokens.weight`` / ``llm.model.norm.weight`` /
    ``llm.lm_head.weight`` / ``llm.mtp.*``).

    Raises:
        AttributeError: if no candidate attribute is present, or none of the
            present ones looks like a CausalLM -- listing what the module
            actually exposes so the caller can extend ``_LLM_ATTR_CANDIDATES``.
    """
    rejected = []
    for name in _LLM_ATTR_CANDIDATES:
        sub = getattr(module, name, None)
        if sub is None or not hasattr(sub, "forward"):
            continue
        if not (hasattr(sub, "lm_head") or hasattr(sub, "get_output_embeddings")):
            # Present, callable, but headless -- a tower, not the backbone.
            rejected.append(f"{name}({type(sub).__name__}, no lm_head)")
            continue
        logger.info(
            "MiniCPM-o LLM backbone resolved: module.%s -> %s%s",
            name,
            type(sub).__name__,
            f" (rejected earlier candidates: {', '.join(rejected)})" if rejected else "",
        )
        return name, sub

    available = sorted(n for n, _ in module.named_children())
    raise AttributeError(
        f"Could not find the LLM backbone on {type(module).__name__}. Tried "
        f"{_LLM_ATTR_CANDIDATES}"
        + (f"; rejected {rejected} for having no lm_head" if rejected else "")
        + f"; the module exposes {available}. Add the right name to "
        "_LLM_ATTR_CANDIDATES in verl_omni/models/transformers/minicpmo_5_0_forward.py, "
        "or relax the lm_head check if this checkpoint keeps the head outside the backbone."
    )


def build_minicpmo_forward_adapter(module):
    """Give *module* a verl-shaped keyword ``forward``. Returns the module.

    Idempotent: re-applying to an already-adapted module is a no-op.

    The adapter is bound to the **instance**, not the class, matching the
    Qwen3-Omni precedent.  Binding to the class would leak across every
    MiniCPMO instance in the process, including the rollout engine's.
    """
    if getattr(module, _FORWARD_FLAG, False):
        return module

    llm_attr, llm = find_llm_backbone(module)
    original_forward = module.forward
    allow_mm = os.environ.get("MINICPMO_ENABLE_MM_FORWARD", "0") == "1"
    logged_first_call = []

    # The vision tower's position_embedding weight is on NPU after model load,
    # but the HF-cached SiglipVisionEmbeddings.forward computes pos_ids on CPU
    # and feeds them straight to nn.Embedding without a .to(weight.device). The
    # vllm_omni fork has that .to(); the checkpoint's remote code does not.
    # Install a forward pre-hook to bridge the gap -- see
    # patch_vision_position_embedding_device for the rationale.
    patch_vision_position_embedding_device(module)

    # The audio encoder's nn.Embedding submodules are individually FSDP2-wrapped
    # by verl's _select_fsdp2_wrap_targets (it wraps ALL nn.Embedding). The
    # Whisper forward reads position_embedding.weight directly (not via __call__),
    # so the FSDP2 all-gather never fires and .weight stays a DTensor. Install a
    # forward pre-hook on the audio encoder to unshard FSDP-wrapped children
    # before the forward runs. See patch_audio_encoder_fsdp_unshard.
    #
    # Always install unshard hook, even when precompute cache is active:
    # compute_log_prob / ref_log_prob paths re-forward the rollout's input_ids
    # WITHOUT precomputed embeds (NonTensorData only travels through
    # _compute_multi_modal_inputs in the rollout path). When cache miss
    # occurs in these paths, Whisper forward runs and needs the unshard hook
    # to avoid DTensor mismatch (aten.add.Tensor got mixed Tensor+DTensor).
    _precompute_cache_dir = os.environ.get("MINICPMO_EMBED_CACHE_DIR", "")
    patch_audio_encoder_fsdp_unshard(module)

    # Precomputed embedding injection: if MINICPMO_EMBED_CACHE_DIR is set,
    # freeze the encoders and patch get_audio_embedding to use cached
    # embeddings. This bypasses Whisper + ViT during forward, eliminating the
    # multimodal load imbalance that causes FSDP2 per-layer AllGather timeout.
    # See precompute.py for the offline cache generation script.
    if _precompute_cache_dir:
        from .precompute import patch_precomputed_embeddings
        patch_precomputed_embeddings(module, _precompute_cache_dir)

    def _forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        mm_keys = sorted(k for k in kwargs if k not in _CORE_KEYS and k not in _PASSTHROUGH_KEYS)
        # Route on payload, not on presence -- see _is_empty_payload. The two
        # lists differ exactly when the processor emitted its empty sentinels
        # for a text-only row, which on this checkpoint is every V1-V3 batch.
        mm_payload_keys = [k for k in mm_keys if not _is_empty_payload(kwargs[k])]
        empty_mm_keys = [k for k in mm_keys if k not in mm_payload_keys]

        if not logged_first_call:
            # Once per adapted module. The first real batch is the only place
            # the actual tensor shapes become visible, and this port is being
            # developed against logs from a machine we cannot attach to.
            #
            # print, not logger.info: the logger.info version of this line did
            # not appear anywhere in the 12.txt run even though this function
            # demonstrably executed (it raised three lines below). Ray worker
            # stdout is captured; this module's logger level is not ours to
            # assume. A diagnostic that silently does not print is worse than
            # none, because its absence reads as "the code did not run".
            logged_first_call.append(True)
            print(
                "[verl-omni] MiniCPM-o forward adapter, first call: "
                f"route={'backbone' if not mm_payload_keys else ('data-dict' if allow_mm else 'REJECTED')} "
                f"backbone={type(llm).__name__} "
                f"input_ids={_describe(input_ids)} "
                f"attention_mask={_describe(attention_mask)} "
                f"position_ids={_describe(position_ids)} "
                f"multimodal_keys_with_payload={mm_payload_keys or '(none)'} "
                f"multimodal_keys_empty={empty_mm_keys or '(none)'} "
                f"passthrough={sorted(k for k in kwargs if k in _PASSTHROUGH_KEYS)}",
                flush=True,
            )

        if not mm_payload_keys and not allow_mm:
            # Text-only: straight to the LLM backbone, which already speaks
            # verl's calling convention and returns .logits.
            #
            # Strictly, the data-dict path could now take a text-only batch too
            # -- the empty-modality sentinel below is exactly what upstream
            # feeds it for an image-free row. The bypass is still the right
            # default, for a reason that has nothing to do with convenience:
            # going through MiniCPMO.forward means going through
            # _compute_position_ids, and under canvas mrope_mode that
            # substitutes self-computed positions for verl's. For a packed
            # text batch verl's positions are known-correct (they restart per
            # sequence), so the bypass is the path that preserves them. It is
            # also, byte for byte, the path the validated Qwen3.5-MoE GSPO run
            # exercises.
            #
            # kwargs may still hold the processor's empty sentinels here; they
            # are dropped rather than forwarded, because the backbone is a stock
            # CausalLM that has never heard of image_bound.
            passthrough = {k: v for k, v in kwargs.items() if k in _PASSTHROUGH_KEYS}
            return llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **passthrough,
            )

        if not allow_mm:
            raise NotImplementedError(
                "MiniCPM-o 5.0 multimodal training forward is not validated yet. This batch "
                f"carries real payload under {mm_payload_keys} (empty sentinels, ignored: "
                f"{empty_mm_keys or '(none)'}). The remote code's data-dict contract is now "
                "known and cross-checked against ModelBest's own HF SFT collator (required: "
                f"{list(_DATA_REQUIRED_KEYS)}, of which {list(_MODALITY_BOUND_KEYS)} accept the "
                f"empty sentinel [[]]; optional: {list(_DATA_OPTIONAL_KEYS)}). What is NOT settled "
                "is position ids: V0-11 measured that 5.0's canvas branch ignores "
                "data['position_ids'] and calls _text_position_ids(input_ids, attention_mask) "
                "instead. MiniCPM-o 4.5 -- the version the working GPU RL stack trains -- reads "
                "data['position_ids'] and passes it straight through, so that reference does not "
                "cover this. With use_remove_padding=True verl passes a packed (1,total_nnz) batch "
                "and attention_mask=None, and one monotonic ramp then covers the whole pack: every "
                "sub-sequence after the first is positioned wrong, silently. Set "
                "actor_rollout_ref.model.use_remove_padding=False to sidestep it. See "
                "implementation plan section 4.7. Two ways forward: keep the dataset text-only for "
                "V1-V3 RL, or set MINICPMO_ENABLE_MM_FORWARD=1 to try the repack below and report "
                "what breaks."
            )

        if attention_mask is None and os.environ.get(_PACKED_MM_OVERRIDE, "0") != "1":
            # Pitfall 29, made unbypassable. attention_mask=None is verl's
            # signature for "this batch is packed" (use_remove_padding=True):
            # transformer_impl passes the (1, total_nnz) row and relies on
            # position_ids alone to mark the sequence boundaries. 5.0 does not
            # read those position ids, and _text_position_ids has no mask to
            # restart on, so it lays a single ramp across the pack.
            #
            # This is the one failure mode in this file that produces no error,
            # no shape mismatch and no NaN -- just a model trained on
            # mis-positioned tokens. It gets a hard refusal rather than a
            # warning for exactly that reason.
            raise NotImplementedError(
                "Refusing a packed multimodal batch: attention_mask is None, which means verl "
                "packed the sequences (use_remove_padding=True), and MiniCPM-o 5.0 recomputes "
                "position ids via _text_position_ids instead of reading data['position_ids'] "
                "(V0-11, measured). With no attention mask there is nothing to restart the ramp "
                "on, so every packed sub-sequence after the first would be positioned wrong -- and "
                "nothing would raise. Fix by setting "
                "actor_rollout_ref.model.use_remove_padding=False (verified to work; this is what "
                "the text-only V1-V3 runs use), or port the rebase recipe quoted in this module's "
                "docstring from verl_mm's process_multi_modal_inputs_for_minicpmo AND resolve the "
                "5.0 position-id question first. To override anyway -- only for a shape-debugging "
                f"run whose loss you will not trust -- set {_PACKED_MM_OVERRIDE}=1."
            )

        # Repack into the single `data` dict the remote code expects, in the
        # shape ModelBest's own collator produces (dataloader.py:268-295).
        data = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        dropped = []
        for k in mm_keys:
            if k in _MM_DROP_KEYS:
                dropped.append(k)
                continue
            data[k] = kwargs[k]

        # audio_features is the one key the per-sample-list collation gets wrong.
        #
        # extract_multi_modal_inputs takes its has_image_bound branch as soon as
        # any sample carries image_bound, and that branch keeps every value as a
        # per-sample list with no torch.cat. For pixel_values / image_bound /
        # tgt_sizes that is exactly right -- get_vllm_embedding indexes them
        # with [i]. For audio it is not: get_audio_embedding opens with
        #
        #     batch_size, _, max_mel_seq_len = wavforms.shape
        #
        # i.e. it wants ONE padded 3-D tensor, and a list has no .shape.
        # Samples also disagree on the mel time axis, so they cannot simply be
        # concatenated -- they have to be padded to a common length first.
        if data.get("audio_features") is not None:
            data["audio_features"] = _pad_and_concat_audio_features(data["audio_features"])
        if data.get("audio_feature_lens") is not None:
            data["audio_feature_lens"] = _concat_audio_feature_lens(data["audio_feature_lens"])

        # verl's training stack only moves input_ids / attention_mask /
        # position_ids to the NPU; the extra mm kwargs stay on CPU.
        # pixel_values / image_bound / tgt_sizes are per-sample lists of tensors
        # (get_vllm_embedding indexes them with [i]); audio_features is a single
        # padded 3-D tensor after the reshape above; audio_feature_lens is a
        # list of 1-D tensors (one per sample) because the remote code calls
        # torch.hstack on it.
        if input_ids is not None and input_ids.device.type != "cpu":
            _dev = input_ids.device
            for _k, _v in list(data.items()):
                if isinstance(_v, torch.Tensor):
                    data[_k] = _v.to(_dev)
                elif isinstance(_v, list) and _v and isinstance(_v[0], torch.Tensor):
                    data[_k] = [_t.to(_dev) for _t in _v]

        if data.get("pixel_values") is None:
            data["pixel_values"] = []
        if data.get("tgt_sizes") is None:
            data["tgt_sizes"] = []

        missing = [k for k in _DATA_REQUIRED_TENSOR_KEYS if data.get(k) is None]
        if missing:
            raise NotImplementedError(
                f"MiniCPM-o's data-dict forward requires {missing}, which this batch does not "
                "carry. The remote code reads these with data[...] (not .get), so they are "
                "mandatory. The dataset's multi_modal_inputs must supply them -- see "
                "implementation plan section 4.6 (data.custom_cls) and section 4.7."
            )

        # image_bound / audio_bounds are read with data[...] too, but upstream
        # never omits them: a packed row with no image gets [[]] rather than a
        # missing key (dataloader.py:287, :292). Synthesising the same sentinel
        # is what lets a mixed batch -- some samples with images, some without --
        # go through one code path instead of two.
        batch_size = getattr(input_ids, "shape", (1,))[0] if input_ids is not None else 1
        synthesised = []
        for k in _MODALITY_BOUND_KEYS:
            if data.get(k) is None:
                data[k] = _empty_bound_sentinel(batch_size)
                synthesised.append(k)

        # Compact per-rank forward trace, via print so it survives on the
        # vLLM-driver ranks (0/4/8/12 under TP=4) whose logger is muted -- in
        # log 20 only 12 of 16 ranks were visible here, which is why lockstep
        # had to be inferred rather than read off.
        #
        # `slices` is the number of image slices in this micro-batch (tgt_sizes
        # rows). It varies per sample (1 vs 7 observed in log 20) and is the
        # prime suspect for a rank-dependent collective count: if the vision
        # tower is invoked per slice rather than once per micro-batch, unequal
        # slice totals mean unequal all-gathers and a silent deadlock.
        # Compare, per call number, that every rank reports the same `n=`.
        global _MM_FORWARD_CALL_COUNT
        _MM_FORWARD_CALL_COUNT += 1
        _call_no = _MM_FORWARD_CALL_COUNT
        if _call_no <= _MM_TRACE_FIRST_N or _MM_TRACE:
            _tgt = data.get("tgt_sizes")
            if isinstance(_tgt, list):
                _slices = sum(
                    t.shape[0] for t in _tgt if isinstance(t, torch.Tensor) and t.dim() > 0
                )
            elif isinstance(_tgt, torch.Tensor) and _tgt.dim() > 0:
                _slices = _tgt.shape[0]
            else:
                _slices = -1
            _audio = data.get("audio_features")
            _frames = _audio.shape[-1] if isinstance(_audio, torch.Tensor) else -1
            # has_image / has_audio are the fields to diff across ranks first.
            #
            # Under FSDP2 with reshard_after_forward=true, every forward of a
            # wrapped submodule issues an all-gather.  The remote code runs the
            # vision tower only when pixel_values is non-empty and the audio
            # encoder only when audio_features is non-empty, so a micro-batch
            # that happens to carry no image (or no audio) skips that tower --
            # and skips its all-gathers.  Ranks then disagree on how many
            # collectives to issue and the group deadlocks with no error on any
            # rank, which is precisely the observed failure.  A single rank
            # printing has_image=0 while the others print 1, at the same n=,
            # is the smoking gun; `slices` alone does not show it because a
            # rank with no image at all reports slices=0 or -1 ambiguously.
            print(
                f"[verl-omni] mm forward rank={_dist_rank()} "
                f"n={_call_no} seq_len={getattr(input_ids, 'shape', ('?', '?'))[-1]} "
                f"slices={_slices} audio_frames={_frames} "
                f"has_image={int(not _is_empty_payload(data.get('pixel_values')))} "
                f"has_audio={int(not _is_empty_payload(data.get('audio_features')))}",
                flush=True,
            )

        passthrough = {k: v for k, v in kwargs.items() if k in _PASSTHROUGH_KEYS}
        logger.warning(
                    f"Calling MiniCPMO.forward with a repacked `data` dict (keys: {sorted(data)}; "
                    f"empty-sentinel: {synthesised or '(none)'}; dropped bookkeeping: {dropped or '(none)'}).")

        # Enter/exit markers around the remote-code call.
        #
        # The warning above is emitted immediately before original_forward, and
        # in every hung run so far it is the last thing any head rank ever says
        # -- which left "wedged inside the remote forward" and "wedged on the
        # collective after it returned" indistinguishable.  These two beats
        # separate them: the watchdog dump names whichever one was last.
        #_hang_probe.beat(f"mm_forward n={_call_no} enter remote MiniCPMO.forward")

        # Inject precomputed embeddings if cache is enabled. This sets
        # data["_precomputed_audio_embeds"] (consumed by patched
        # get_audio_embedding) and data["vision_hidden_states"] (consumed by
        # the existing check in get_vision_embedding). On cache miss, raises
        # RuntimeError (no silent fallback — see _inject_precomputed_embeddings).
        if _precompute_cache_dir:
            from .precompute import _inject_precomputed_embeddings
            _inject_precomputed_embeddings(data, _precompute_cache_dir, _call_no)

        try:
            # Keyword, matching upstream's `vllm_engine(data=batch)`
            # (train_minicpmv.py:905). Positional would also bind, but only as
            # long as `data` stays the first parameter.
            _out = original_forward(data=data, **passthrough)
        finally:
            #_hang_probe.beat(f"mm_forward n={_call_no} exit remote MiniCPMO.forward")
            pass
        if _call_no <= _MM_TRACE_FIRST_N or _MM_TRACE:
            print(
                f"[verl-omni] mm forward rank={_dist_rank()} n={_call_no} returned",
                flush=True,
            )
        return _out

    module.forward = _forward
    setattr(module, _FORWARD_FLAG, True)
    logger.info(
        "Installed MiniCPM-o forward adapter: verl keyword calls now route to `%s` "
        "(text-only) or the original data-dict forward (multimodal, gated).",
        llm_attr,
    )
    return module