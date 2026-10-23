"""FP32 HF torch GDN guard for Huanyu's padded training forward."""

import torch


_EXECUTION_REPORTED = False


def _tensor_metadata(value):
    if value is None:
        return "none"
    if isinstance(value, torch.Tensor):
        return f"dtype={value.dtype},shape={tuple(value.shape)}"
    return type(value).__name__


def isolate_torch_gdn_fallbacks(module):
    """Keep HF's padded-call API while isolating the torch fallback from autocast.

    Newer HF forwards pass cu_seqlens=None even for padded inputs. Accept that
    argument without forwarding it to older reference kernels. Actual packed
    boundaries must never be dropped: Huanyu requires use_remove_padding=false
    and this guard does not implement a packed forward. Real FLA kernels keep
    their native implementation and cu_seqlens support.
    """
    count = 0
    for layer in module.model.language_model.layers:
        attention = getattr(layer, "linear_attn", None)
        kernel = getattr(attention, "chunk_gated_delta_rule", None)
        if getattr(kernel, "_huanyu_fp32_guard", False) is True:
            continue
        if (getattr(kernel, "__name__", "") != "torch_chunk_gated_delta_rule"
                or not getattr(kernel, "__module__", "").startswith("transformers.")):
            continue

        def wrap(reference):
            def torch_delta_rule(query, key, value, g, beta, chunk_size=64,
                                 initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False, cu_seqlens=None):
                global _EXECUTION_REPORTED
                if cu_seqlens is not None:
                    raise ValueError(
                        "Huanyu's FP32 torch GDN guard supports padded inputs only "
                        "(cu_seqlens=None). Keep actor_rollout_ref.model.use_remove_padding=false; "
                        "packed sequence boundaries cannot be ignored."
                    )
                # FSDP wraps the model in BF16 autocast. Explicit .float() calls
                # inside HF's reference do not prevent autocast from reducing
                # its state-update matmuls back to BF16. Keep the reference's
                # FP32 recurrence and its original Q/K normalization boundaries.
                with torch.autocast(device_type=query.device.type, enabled=False):
                    result = reference(query, key, value, g=g, beta=beta, chunk_size=chunk_size,
                                       initial_state=initial_state, output_final_state=output_final_state,
                                       use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                    if not _EXECUTION_REPORTED:
                        _EXECUTION_REPORTED = True
                        output, state = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
                        print(
                            "[huanyu-precision] training GDN executed: "
                            f"kernel={reference.__module__}.{reference.__name__} "
                            f"autocast=false grad_enabled={torch.is_grad_enabled()} "
                            f"query=({_tensor_metadata(query)}) output=({_tensor_metadata(output)}) "
                            f"state=({_tensor_metadata(state)})",
                            flush=True,
                        )
                    return result
            torch_delta_rule._huanyu_fp32_guard = True
            torch_delta_rule._huanyu_fp32_reference = reference
            return torch_delta_rule

        attention.chunk_gated_delta_rule = wrap(kernel)
        count += 1
    return count
