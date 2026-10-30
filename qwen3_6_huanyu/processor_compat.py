"""Compatibility for the native MiniCPMO processing-info entry point.

The deployed get_hf_processor normalizes MiniCPM image ``mean/std`` arrays,
but Qwen processors name those fields ``image_mean/image_std``. Port that
small getter with optional normalization fields; keep native processor
construction, tokenizer tokens, and audio pooling behavior intact. Install
before model-runner construction, which already builds dummy multimodal inputs.
"""

from functools import wraps
from importlib import import_module

import numpy as np


_NATIVE = "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni_llm"
_CLASS = "MiniCPMO50OmniLLMProcessingInfo"
_MARKER = "_huanyu_image_normalization_compat"
_TOKEN_PIECES = (
    ("audio_start", "<|audio_start|>"),
    ("audio_end", "<|audio_end|>"),
    ("im_start", "<image>"),
    ("im_end", "</image>"),
    ("slice_start", "<slice>"),
    ("slice_end", "</slice>"),
    ("spk_start", "<|spk_bos|>"),
    ("spk_end", "<|spk_eos|>"),
)


def _get_hf_processor(self, **kwargs):
    # Same lazy-tokenizer branch as the native getter. In workers using
    # skip_tokenizer_init, ctx.get_hf_processor alone is not sufficient.
    if self.ctx.tokenizer is None:
        from vllm.transformers_utils.processor import cached_processor_from_config

        tokenizer = self.get_tokenizer()
        hf_processor = cached_processor_from_config(
            self.ctx.model_config, tokenizer=tokenizer, **kwargs,
        )
    else:
        hf_processor = self.ctx.get_hf_processor(**kwargs)

    tokenizer = getattr(hf_processor, "tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "audio_start"):
        for attr, piece in _TOKEN_PIECES:
            setattr(tokenizer, attr, piece)
            setattr(tokenizer, attr + "_id", tokenizer.convert_tokens_to_ids(piece))

    image_processor = hf_processor.image_processor
    # Only normalize existing fields, retaining their configured values. Qwen
    # consumes image_mean/image_std, so creating unused mean/std aliases would
    # leave NumPy values in the fields its own preprocessing actually reads.
    for attr in ("mean", "std", "image_mean", "image_std"):
        val = getattr(image_processor, attr, None)
        if isinstance(val, np.ndarray):
            setattr(image_processor, attr, val.tolist())

    config = self.get_hf_config()
    pool_step = getattr(config, "audio_pool_step", None)
    if pool_step is not None and hasattr(hf_processor, "pool_step"):
        hf_processor.pool_step = pool_step
    return hf_processor


def install_huanyu_processor_compat():
    """Bind once in both registry and explicit-worker startup paths."""
    native = import_module(_NATIVE)
    info_cls = getattr(native, _CLASS)
    original = info_cls.get_hf_processor
    if getattr(original, _MARKER, False):
        return True

    @wraps(original)
    def get_hf_processor(self, **kwargs):
        return _get_hf_processor(self, **kwargs)

    setattr(get_hf_processor, _MARKER, True)
    info_cls.get_hf_processor = get_hf_processor
    print(
        "[huanyu-processor] native image normalization compatibility installed; "
        "fields=mean,std,image_mean,image_std; configured values retained",
        flush=True,
    )
    return True
