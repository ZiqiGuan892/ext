"""Keep vLLM's full multimodal ranges and embedding masks for Huanyu.

The legacy Whisper safe_apply fallback scans pad runs and can keep only the
longest run of a video. Bypass that specific wrapper on the Huanyu processor,
retaining the original updater of the installed vLLM version.
"""

from inspect import getclosurevars


def native_prompt_updater(method):
    # The deployed legacy wrapper does not expose __wrapped__. Its closure
    # holds orig_apply (both known server revisions use this exact name).
    # Do not unwrap unrelated decorators or replace the shared base class.
    seen = set()
    while (
        getattr(method, "__module__", None) == "verl_omni_ext.models.qwen3_6_whisper"
        and getattr(method, "__name__", None) == "safe_apply"
    ):
        if id(method) in seen:
            raise RuntimeError("Huanyu found a cyclic legacy prompt-update wrapper")
        seen.add(id(method))
        original = getclosurevars(method).nonlocals.get("orig_apply")
        if not callable(original):
            raise RuntimeError("Huanyu cannot resolve legacy safe_apply.orig_apply; refusing pad-run recovery")
        method = original
    return method


def _complete_ranges(placeholders, updates):
    return all(len(placeholders.get(modality, ())) == len(items) for modality, items in updates.items())


def apply_huanyu_prompt_updates(processor, native_apply, token_ids, mm_prompt_updates):
    # Some native frontends already expand the prompt. Recognize only exact
    # replacement contents, using vLLM's own finder so it retains is_embed.
    # One video may span many separated pad runs; never count runs as items.
    placeholders = processor._find_mm_placeholders(token_ids, mm_prompt_updates)
    if _complete_ranges(placeholders, mm_prompt_updates):
        return token_ids, placeholders

    # Compact prompts still use the installed vLLM token/text replacement
    # implementation. Let its original exception surface instead of inventing
    # a shortened range and failing later inside an NPU worker.
    ids, placeholders = native_apply(processor, token_ids, mm_prompt_updates)
    if not _complete_ranges(placeholders, mm_prompt_updates):
        counts = {modality: (len(placeholders.get(modality, ())), len(items))
                  for modality, items in mm_prompt_updates.items()}
        raise ValueError(f"Huanyu native prompt-update item counts differ (actual, expected): {counts}")
    return ids, placeholders
