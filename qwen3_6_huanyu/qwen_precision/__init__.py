"""Local snapshot of the common Qwen precision fixes from qwen3_6_whisper.

Ported from the sibling qwen3_6_whisper implementation in this repository.
The arithmetic modules are copied without numerical changes so deployments
that synchronize only qwen3_6_huanyu carry their own matching implementations.
Source attribution and existing notices remain in each copied module.

packed.py retains only the HF torch GDN fallback guard; media_precision.py
retains only frozen-vision processing; cache_precision.py retains only cache
algorithms. Huanyu owns model selection and hook installation separately.
Whisper audio processing and packed forwards are intentionally excluded.
"""
