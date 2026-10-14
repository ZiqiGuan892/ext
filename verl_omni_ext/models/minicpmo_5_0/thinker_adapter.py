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
"""OmniModelBase adapter for MiniCPM-o 5.0 thinker training.

Registration key
----------------
``("MiniCPMO", "thinker")``.  The architecture string is what the checkpoint's
``config.json`` actually reports -- ``architectures: ["MiniCPMO"]``, confirmed
against the released MiniCPM-o-5_0_RL checkpoint -- *not* the vLLM-side class
name ``MiniCPMO50OmniForConditionalGeneration``.  Three distinct names are in
play and mixing them up is the easiest way to get this wrong:

===========================  ==========================================
HF ``architectures[0]``      ``MiniCPMO``                (this file)
vLLM ``model_arch``          ``MiniCPMO50OmniForConditionalGeneration``
vLLM ``model_stage``         ``llm``
verl-omni stage key          ``thinker``                 (this file)
===========================  ==========================================

Without this adapter ``OmniModelConfig.__post_init__`` raises
``NotImplementedError`` from ``OmniModelBase.get_class_by_name`` -- the
architecture is simply absent from verl-omni's registry.

Weight-sync note
----------------
No key remapping is needed anywhere in this port.  The HF side
(``module.state_dict()``) and the vLLM side
(``MiniCPMO50OmniForConditionalGeneration.load_weights``) agree on the prefixes
``vpm. / resampler. / vit_merger. / llm. / apm. / audio_projection_layer. /
tts.``, because both read the same checkpoint's safetensors keys.
"""

import logging
import os
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


def _report_mtp(module):
    """Log whether the backbone carries an unused MTP head, and optionally freeze it.

    The checkpoint's backbone config declares ``mtp_num_hidden_layers: 1`` and
    the hf2mcore converter's parameter tree lists ``llm.mtp.*`` -- i.e. a
    multi-token-prediction head that a plain causal-LM forward never touches.
    verl reads ``output.logits``, the main head, so MTP contributes nothing to
    the loss.

    This is deliberately a **report, not a fix**.  Three things are true and
    they point in different directions:

    * It is not a correctness problem.  FSDP2 tolerates unused parameters (no
      DDP-style ``find_unused_parameters``), and ``torch.optim`` skips any
      parameter whose ``.grad`` is None, so no optimizer state is allocated for
      them and ``fsdp2_clip_grad_norm_`` skips them too.  Uniformly unused
      across every rank means no collective can go asymmetric.
    * It is not free either.  ``build_optimizer`` is handed
      ``module.parameters()`` with no ``requires_grad`` filter
      (``transformer_impl.py:480``), the weights are still sharded, still
      offloaded, and still pushed through every rollout weight sync.
    * Freezing is cheap and reversible -- ``requires_grad_(False)`` leaves the
      state dict byte-identical, which is the same property
      ``get_strip_modules`` is protecting -- but it would be a blind change
      until V0-11 section (3) confirms ``llm.mtp`` exists on *this* checkpoint
      and that ``forward`` really ignores it.

    So: always log, freeze only when asked.  ``MINICPMO_FREEZE_MTP=1``.

    **Answered on hardware: this checkpoint has no ``llm.mtp`` at all.**  The
    run that first printed this line took the ``mtp is None`` branch on all 16
    ranks.  The backbone config's ``mtp_num_hidden_layers: 1`` and the hf2mcore
    converter's parameter tree both pointed at a head that the released weights
    do not contain, so ``MINICPMO_FREEZE_MTP`` is a dead switch.  Kept anyway:
    it costs one ``getattr`` and it is the only thing that would notice if a
    future checkpoint started shipping the head.

    Uses ``print(..., flush=True)``, not ``logger.info``.  The logger version of
    these lines produced **nothing** in the runs that mattered -- ``logger.info``
    from inside a Ray worker does not reach the driver's captured output on this
    stack, which is how the MTP question stayed open across three 16-card
    launches while this function was running correctly every time.
    """
    llm = getattr(module, "llm", None)
    mtp = getattr(llm, "mtp", None) if llm is not None else None
    if mtp is None:
        print("[verl-omni] MiniCPM-o MTP check: no `llm.mtp` on this checkpoint, nothing to do.", flush=True)
        return module

    try:
        params = list(mtp.parameters())
        n_elem = sum(p.numel() for p in params)
    except Exception:  # a stub, a meta module, anything unexpected
        params, n_elem = [], 0

    if os.environ.get("MINICPMO_FREEZE_MTP", "0") == "1":
        for p in params:
            p.requires_grad_(False)
        print(
            f"[verl-omni] MiniCPM-o MTP check: froze llm.mtp ({len(params)} tensors, "
            f"{n_elem / 1e6:.1f}M elements). State dict is unchanged, so rollout weight sync "
            f"still sees them; only optimizer state is saved (~{n_elem * 8 / 2**30:.2f} GiB "
            "fp32 Adam moments, before sharding).",
            flush=True,
        )
    else:
        print(
            f"[verl-omni] MiniCPM-o MTP check: llm.mtp present ({len(params)} tensors, "
            f"{n_elem / 1e6:.1f}M elements) and NOT used by the training forward -- verl reads "
            f"output.logits, the main head. Harmless but not free: ~{n_elem * 8 / 2**30:.2f} GiB "
            "of fp32 Adam moments and one extra slice of every weight sync. "
            "Set MINICPMO_FREEZE_MTP=1 to freeze it (state dict stays identical).",
            flush=True,
        )
    return module


def _report_param_census(module):
    """Print a per-prefix parameter census, plus the node's host-memory counters.

    Why this is worth a few lines of output.  Under FSDP2 ``CPUOffloadPolicy``
    every parameter shard is materialised on the host, and (until the patch in
    ``verl_omni.utils.fsdp_cpu_offload_pin``) page-locked there.  The host-side
    cost is therefore driven by *total* parameters, not by the ones the
    thinker-only forward actually touches -- and this checkpoint carries six
    towers that a text-only step never enters: ``vpm. / resampler. /
    vit_merger. / apm. / audio_projection_layer. / tts.``

    Freezing them would not reclaim that host memory (pinning happens per
    parameter at FSDP init, regardless of ``requires_grad``), which is exactly
    why it is worth printing rather than silently "optimising": the number tells
    you whether a tower is worth stripping at load time, and stripping is a
    different and much bigger change than freezing.

    **Measured, so nobody has to guess again**::

        total=35.95B  llm=34.66B  tts=0.42B  vpm=0.42B  apm=0.31B
                      vit_merger=0.11B  resampler=0.03B  audio_projection_layer=0.01B

    The six towers a text-only step never enters come to 1.30B -- 3.6% of the
    model.  That settles it: neither freezing nor stripping them is worth the
    risk to the rollout weight sync, and the host-memory pressure this census
    was written to investigate is essentially all ``llm.``
    """
    from verl_omni_ext.models.minicpmo_5_0.fsdp_cpu_offload_pin import report_host_memory

    buckets = {}
    for name, p in module.named_parameters():
        prefix = name.split(".", 1)[0]
        total, trainable = buckets.get(prefix, (0, 0))
        buckets[prefix] = (total + p.numel(), trainable + (p.numel() if p.requires_grad else 0))

    rendered = " ".join(
        f"{prefix}={total / 1e9:.2f}B{'' if total == trainable else f'(trainable {trainable / 1e9:.2f}B)'}"
        for prefix, (total, trainable) in sorted(buckets.items(), key=lambda kv: -kv[1][0])
    )
    grand = sum(t for t, _ in buckets.values())
    print(f"[verl-omni] MiniCPM-o param census: total={grand / 1e9:.2f}B  {rendered}", flush=True)
    report_host_memory("after configure_model")
    return module


@OmniModelBase.register("MiniCPMO", stage="thinker")
class MiniCPMO50ThinkerAdapter(OmniModelBase):
    """verl-omni adapter for MiniCPM-o 5.0 thinker-only GSPO training.

    The thinker's LLM backbone *is* Qwen3.5-MoE, so the memory, precision and
    micro-batch conclusions from the Qwen3.5-MoE run transfer unchanged (fp32
    master weights, one micro-batch per device, ``weight_nz_mode=0``).  What
    does not transfer is the vision stack: MiniCPM-o wraps its own UHD NaViT
    encoder rather than ``Qwen3_5MoeVisionModel``.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Strip nothing -- in particular, do not strip ``tts``.

        The ``tts.*`` weights (0.84 GB of the checkpoint's 71.9 GB, against
        71.07 GB for the thinker) are unused in thinker-only rollout, and
        removing them looks like free memory.  It is not worth it:

        * The saving is ~1% of the checkpoint, well inside the noise of the
          fp32 master-weight and optimizer-state budget that actually decides
          whether this fits.
        * Stripping changes ``module.state_dict()``, and RL weight sync pushes
          that dict straight at the rollout engine.  Today
          ``MiniCPMO50OmniForConditionalGeneration.load_weights`` routes
          ``tts.*`` to a talker list that thinker-only never consumes, so their
          absence is harmless -- but that is a property of the *current* vLLM
          side, not a contract.  Keeping the state dict identical to the
          checkpoint keeps the two sides trivially comparable.
        * A stripped module diverges from what the HF ``modeling_minicpmo.py``
          expects to find, and the failure mode is a mid-training
          ``AttributeError``, not a startup error.

        Revisit only if profiling shows the thinker genuinely does not fit.
        """
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        """Install the forward-signature adapter, then defer to the base.

        **Why this is mandatory.**  V0-4 measured MiniCPM-o's remote-code
        signature as ``MiniCPMO.forward(self, data, **kwargs)`` -- the
        MiniCPM-V/o family's single-positional-dict convention.  verl calls the
        training module keyword-only
        (``transformer_impl.py:1526``, ``self.module(**model_inputs,
        use_cache=False)``) and then reads ``raw_output.logits``.  Those two do
        not meet: ``data`` never binds, and the first micro-batch dies with
        ``TypeError: forward() missing 1 required positional argument: 'data'``.
        Because the model loads fine, this surfaces in V3, not at startup.

        ``build_minicpmo_forward_adapter`` re-exposes a verl-shaped keyword
        ``forward`` and routes text-only batches straight to the Qwen3.5-MoE
        backbone -- which already speaks verl's convention and already returns
        ``.logits``.  See ``verl_omni/models/transformers/minicpmo_5_0_forward.py``.

        The precedent is ``Qwen3OmniThinkerAdapter.configure_model``, which does
        ``module.forward = module.thinker.forward``.  MiniCPM-o needs more than
        a rebind because the signatures differ in *shape*, not just in name.

        **What this does NOT do.**  It does not call
        ``apply_qwen3_5_vision_device_fix``.  That patch targets
        ``Qwen3_5MoeVisionModel``, and MiniCPM-o does not use it: the vision
        stack is the checkpoint's own UHD NaViT encoder (``vpm.`` /
        ``resampler.`` / ``vit_merger.``, confirmed by V0-3 reporting
        ``MiniCPMVImageProcessor``).  Applying it here would either no-op
        silently or patch the wrong thing.

        MiniCPM-o's vision tower may well need its own device fix under FSDP2
        ``CPUOffloadPolicy`` -- the same conditions that broke Qwen3.5 apply to
        the ref model here too.  That cannot be written blind; it needs the
        first multimodal failure to point at the offending line.  Text-only
        training does not enter the vision branch at all, so this is not on the
        critical path to a working loop.

        The two transformers compatibility shims this checkpoint needs are
        applied at *package import* time (see this package's ``__init__``), not
        from here: ``omni_impl._build_module`` calls
        ``AutoModelForMultimodalLM.from_pretrained`` before ``configure_model``
        ever runs, so patching there would be too late.  The forward adapter is
        different -- it patches the *instance* that ``from_pretrained``
        returned, so here is exactly the right place.
        """
        from verl_omni_ext.models.minicpmo_5_0.forward import build_minicpmo_forward_adapter

        module = super().configure_model(module, model_config)
        cls._configure_multimodal_encoders(module)
        module = build_minicpmo_forward_adapter(module)
        _report_mtp(module)
        _report_param_census(module)
        return module

    @classmethod
    def _configure_multimodal_encoders(cls, module):
        """Configure freeze/tune switches for visual (vpm) and audio (apm) encoders."""
        tune_vision = os.environ.get("MINICPMO_TUNE_VISION", "0") == "1"
        tune_audio = os.environ.get("MINICPMO_TUNE_AUDIO", "0") == "1"

        for name in ["vpm", "resampler", "vit_merger"]:
            sub = getattr(module, name, None)
            if sub is not None:
                sub.train() if tune_vision else sub.eval()
                for p in sub.parameters():
                    p.requires_grad_(tune_vision)

        for name in ["apm", "audio_projection_layer"]:
            sub = getattr(module, name, None)
            if sub is not None:
                sub.train() if tune_audio else sub.eval()
                for p in sub.parameters():
                    p.requires_grad_(tune_audio)

        print(
            f"[verl-omni] MiniCPM-o multimodal encoder status: "
            f"VPM(vision)={'TUNE' if tune_vision else 'FREEZE'}, "
            f"APM(audio)={'TUNE' if tune_audio else 'FREEZE'}",
            flush=True,
        )
    
    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Standard HF tokenizer, loaded through remote code.

        ``trust_remote_code`` must be True for this checkpoint; it is threaded
        from the trainer config rather than hardcoded so a future
        natively-supported MiniCPM-o still works.
        """
        from verl.utils import hf_tokenizer

        return hf_tokenizer(
            model_path,
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
        )

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Return the real HF processor -- ``hf_processor`` alone cannot.

        ``OmniPPOTrainerSync._init_tokenizer`` assigns ``self.processor =
        model_config.processor`` and hands it to the dataset;
        ``RLHFDataset._build_messages`` then asserts it is not None whenever a
        row carries a non-empty ``images`` / ``videos`` / ``audios`` column,
        regardless of whether the model is being trained text-only.

        **Why the fallback is mandatory, not defensive.**
        ``verl.utils.tokenizer.hf_processor`` dispatches on the processor class
        name through a ``match`` whose default arm is
        ``raise ValueError(f"Unsupported processor type: ...")``
        (``verl/utils/tokenizer/tokenizer.py:205-231``).  V0-3 measured this
        checkpoint's class as ``MiniCPMOProcessor``, which is not one of the
        six recognised names, so the raise always fires -- and the enclosing
        ``except Exception`` swallows it into ``processor = None`` plus one
        ``warnings.warn``.  This is not an edge case that might happen; it is
        what happens on every run.

        With None, ``OmniModelConfig.get_processor()`` degrades to the
        tokenizer.  Text-only V1-V3 survive that (``RLHFDataset`` takes its
        no-processor branch), which is why the loop can be brought up before
        this is fixed -- but V5 multimodal cannot.

        ``build_minicpmo_processor`` constructs the processor directly.  It
        deliberately does not bind ``get_rope_index``: MiniCPM-o computes its
        own canvas m-RoPE inside ``_compute_position_ids`` (V0-7), so
        verl-side position ids are either ignored or already correct.  See
        ``verl_omni/models/transformers/minicpmo_5_0_processor.py``.

        Unlike the Qwen3.5 adapter, ``use_fast=True`` is **not** passed.  This
        is a remote-code processor; it has no fast/slow variants, and
        forwarding an unexpected kwarg into remote code risks a ``TypeError``
        at construction for no benefit.

        **The behaviour change this causes, and the switch to undo it.**
        A non-None processor moves ``RLHFDataset.maybe_filter_out_long_prompts``
        from its tokenizer branch to its processor branch
        (``rl_dataset.py:204``), i.e. length is measured with
        ``processor.apply_chat_template`` instead of
        ``tokenizer.apply_chat_template``.  Both read the same chat template, so
        they should agree -- but that branch wraps everything in
        ``except Exception: return max_prompt_length + 1``, so anything the
        remote-code processor throws silently filters every sample out and the
        dataset comes back empty.

        V3 (2026-08-29) confirmed the fallback fires on the real checkpoint: the
        traceback came from ``rl_dataset.py:208``, which only exists inside the
        ``if processor is not None:`` arm.  The exception there was
        ``KeyError: 'prompt'`` -- an unpreprocessed dataset, not a processor
        fault (see ``examples/gspo_trainer/minicpmo_5_0/prepare_gsm8k_text_only.py``)
        -- but it demonstrated the swallow.

        V3 run 2 (same day, after fixing the dataset) then hit the swallow for a
        *processor* reason: ``MiniCPMOProcessor.chat_template`` is None, so
        ``rl_dataset.py:214`` raised ``ValueError: Cannot use apply_chat_template
        because this processor does not have a chat template`` on all 7473 rows.
        ``build_minicpmo_processor`` now copies the tokenizer's template onto the
        processor, and returns None outright if no template can be found
        anywhere -- a template-less processor is strictly worse than no
        processor, because it turns a loud assert into a silent empty dataset.

        Because text-only RL does not need a processor at all, set
        ``MINICPMO_DISABLE_PROCESSOR_FALLBACK=1`` to go back to the tokenizer
        branch in one line.  That is the way to *prove* whether a
        ``filter dataset len: 0`` is the processor's doing: if the count is
        still 0 with the fallback off, the fault is in the data.

        **The omni path.**  Whatever produced the processor, it then gets
        ``apply_minicpmo_mm_inputs_patch``, which replaces
        ``AgentLoopWorker._compute_multi_modal_inputs`` for MiniCPM-o only.
        Upstream's version rebuilds the processor inputs from
        ``tokenizer.decode(..., skip_special_tokens=True)``, and that decode
        strips ``<image>`` / ``</image>`` / ``<unk>`` / ``<audio>`` --
        precisely the sentinels ``image_bound`` and ``audio_bounds`` are scanned
        from.  The bounds come back empty, no media embedding is scattered into
        the sequence, and **nothing raises**.  The patch is applied here rather
        than at import time because it needs the processor instance to decide
        whether this run is MiniCPM-o's at all.  Text-only runs reach the same
        line and are unaffected: the builder returns ``{}`` when a sample
        carries no media.
        """
        from verl.utils import hf_processor

        from verl_omni_ext.models.minicpmo_5_0.mm_inputs import apply_minicpmo_mm_inputs_patch
        from verl_omni_ext.models.minicpmo_5_0.processor import build_minicpmo_processor

        trust_remote_code = getattr(model_config, "trust_remote_code", False)
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code)
        if processor is not None:
            # A future transformers/verl may add MiniCPMOProcessor to the match.
            apply_minicpmo_mm_inputs_patch(processor)
            return processor

        if os.environ.get("MINICPMO_DISABLE_PROCESSOR_FALLBACK", "0") == "1":
            logger.warning(
                "MINICPMO_DISABLE_PROCESSOR_FALLBACK=1: leaving the processor as None. "
                "RLHFDataset will measure prompt length with the tokenizer instead of the "
                "processor. Text-only training is unaffected; any row with an "
                "images/videos/audios column will trip `assert self.processor is not None`."
            )
            return None

        processor = build_minicpmo_processor(model_path, trust_remote_code=trust_remote_code)
        if processor is None:
            logger.warning(
                "No processor could be built for %s. Text-only training still works, but any "
                "dataset row with an image/video/audio column will trip RLHFDataset's "
                "`assert self.processor is not None`. Run V0-9 section (5) to see what "
                "hf_processor actually returned and why the fallback declined.",
                model_path,
            )
        else:
            apply_minicpmo_mm_inputs_patch(processor)
        return processor