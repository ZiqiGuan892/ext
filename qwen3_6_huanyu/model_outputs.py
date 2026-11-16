"""Avoid the padded FSDP engine's additional full-vocabulary packing copy.

Huanyu forwards stay padded. Compute log-probs from views into those logits,
then pack scalar statistics only. This preserves the engine's label shift,
temperature, entropy implementation and nested output contract.
"""

from functools import wraps
import importlib
import inspect
import os

import torch

from .log_probs import DEFAULT_CHUNK_SIZE, chunked_ragged_logprobs_from_logits


_MODEL_MARKER = "_huanyu_scalar_output_packing"
_PATCH_MARKER = "_huanyu_scalar_output_packing_patch"


def _enabled(module):
    pending, seen = [module], set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if getattr(candidate, _MODEL_MARKER, False) is True:
            return True
        for attr in ("_fsdp_wrapped_module", "_orig_mod"):
            pending.append(getattr(candidate, attr, None))
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            pending.append(get_base_model())
    return False


def _verify_engine(original):
    expected = ("self", "output", "output_args", "micro_batch", "logits_processor_func")
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("Unsupported FSDP prepare_model_outputs signature for Huanyu scalar packing")
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise RuntimeError("Cannot inspect FSDP output packing before installing the Huanyu hook") from error
    required = (
        "torch.cat([t for t in logits.unbind()])",
        'output_args["input_ids_rmpad_rolled"]',
        "_scale_logits_by_temperature(",
        "torch.nested.nested_tensor_from_jagged(",
    )
    if any(part not in source for part in required):
        raise RuntimeError("Unknown FSDP padded-output implementation; refusing Huanyu scalar packing hook")


def install_huanyu_scalar_output_packing(module):
    """Install a model-gated hook after entropy and native log-prob chunking.

    Fused outputs, remove-padding, distillation and sum-pi-squared use the
    original engine method. No installed verl files are edited.
    """
    impl = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    engine_cls = getattr(impl, "FSDPEngineWithLMHead", None)
    original = getattr(engine_cls, "prepare_model_outputs", None)
    if not callable(original):
        raise RuntimeError("Cannot find FSDPEngineWithLMHead.prepare_model_outputs for Huanyu scalar packing")
    if not getattr(original, _PATCH_MARKER, False):
        _verify_engine(original)
        for name in ("tu", "DatasetPadMode", "DTensor", "_scale_logits_by_temperature", "verl_F",
                     "logprobs_from_logits"):
            if not hasattr(impl, name):
                raise RuntimeError(f"FSDP output packing dependency missing: {name}")
        reported = set()

        @wraps(original)
        def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
            def flag(key, default=False):
                return impl.tu.get_non_tensor_data(data=micro_batch, key=key, default=default)

            if (not _enabled(getattr(self, "module", None))
                    or flag("use_remove_padding", True) or flag("use_fused_kernels")
                    or flag("pad_mode", impl.DatasetPadMode.NO_PADDING) != impl.DatasetPadMode.NO_PADDING
                    or flag("distillation_use_topk") or flag("distillation_only")
                    or flag("calculate_sum_pi_squared")):
                return original(self, output, output_args, micro_batch, logits_processor_func)

            logits = output.logits
            if isinstance(logits, impl.DTensor):
                logits = logits.full_tensor()
            logits = impl._scale_logits_by_temperature(
                logits, output_args["temperature"].unsqueeze(-1).unsqueeze(-1),
                is_unit_temperature=output_args.get("temperature_is_one", False),
            )
            offsets = micro_batch["input_ids"].offsets()
            # Small metadata transfer; never transfer or flatten the logits.
            lengths = offsets.diff().tolist()
            labels = output_args["input_ids_rmpad_rolled"]
            mode = "grad" if torch.is_grad_enabled() and logits.requires_grad else "no_grad"
            if mode not in reported:
                print(f"[huanyu-output] active pid={os.getpid()} shape={tuple(logits.shape)} "
                      f"valid_tokens={labels.numel()} chunk_size={DEFAULT_CHUNK_SIZE} mode={mode} "
                      "packing=scalar_only", flush=True)
                reported.add(mode)

            result = {}
            if flag("calculate_entropy"):
                if self.engine_config.entropy_checkpointing:
                    entropy = torch.utils.checkpoint.checkpoint(impl.verl_F.entropy_from_logits, logits)
                else:
                    entropy = impl.verl_F.entropy_from_logits(logits)
                # This concatenation contains one scalar per token, no V axis.
                entropy = torch.cat([entropy[b, :length] for b, length in enumerate(lengths)])
                result["entropy"] = torch.nested.nested_tensor_from_jagged(entropy, offsets)

            log_probs = chunked_ragged_logprobs_from_logits(
                logits, labels, lengths, impl.logprobs_from_logits,
            )
            result["log_probs"] = torch.nested.nested_tensor_from_jagged(log_probs, offsets)
            return result

        setattr(prepare_model_outputs, _PATCH_MARKER, True)
        engine_cls.prepare_model_outputs = prepare_model_outputs
        print(f"[huanyu-output] installed pid={os.getpid()} scope=huanyu packing=scalar_only", flush=True)
    setattr(module, _MODEL_MARKER, True)
    return module
