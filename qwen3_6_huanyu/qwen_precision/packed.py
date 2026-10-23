"""HF torch GDN guard ported from Whisper's packed module, without packed forward."""

import torch


def isolate_torch_gdn_fallbacks(module):
    """Do not advertise **kwargs as varlen support for HF's torch fallback.

    HF accepts arbitrary kwargs but its torch_chunk_gated_delta_rule ignores
    cu_seqlens. verl detects **kwargs as support and otherwise fails to split
    samples. An explicit signature selects verl's existing per-sequence path.
    Real FLA kernels keep their native cu_seqlens support.
    """
    count = 0
    for layer in module.model.language_model.layers:
        attention = getattr(layer, "linear_attn", None)
        kernel = getattr(attention, "chunk_gated_delta_rule", None)
        if (getattr(kernel, "__name__", "") != "torch_chunk_gated_delta_rule"
                or not getattr(kernel, "__module__", "").startswith("transformers.")):
            continue

        def wrap(reference):
            def torch_delta_rule(query, key, value, g, beta, chunk_size=64,
                                 initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False):
                # FSDP wraps the model in BF16 autocast. Explicit .float() calls
                # inside HF's reference do not prevent autocast from reducing
                # its state-update matmuls back to BF16. Keep the reference's
                # FP32 recurrence and its original Q/K normalization boundaries.
                with torch.autocast(device_type=query.device.type, enabled=False):
                    return reference(query, key, value, g=g, beta=beta, chunk_size=chunk_size,
                                     initial_state=initial_state, output_final_state=output_final_state,
                                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
            return torch_delta_rule

        attention.chunk_gated_delta_rule = wrap(kernel)
        count += 1
    return count
