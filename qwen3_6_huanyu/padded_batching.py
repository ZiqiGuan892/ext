"""Bound Huanyu's padded logits allocation while retaining verl's DP batching.

The upstream budget counts valid tokens. Huanyu's dense forward allocates
B * max(sequence_length) vocabulary rows, including padding. Repartition with
a common microbatch count on all DP ranks until that actual size also fits.
"""

from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import os

import torch

from .model_outputs import _enabled


_ACTIVE = ContextVar("huanyu_padded_batching", default=False)
_PATCH_MARKER = "_huanyu_padded_batching_patch"


def _padded_tokens(batch):
    ids = batch["input_ids"]
    if ids.is_nested:
        lengths = ids.offsets().diff()
        return lengths.numel() * int(lengths.max().item()) if lengths.numel() else 0
    return ids.shape[0] * ids.shape[1]


def _prepare(original, impl, data, dp_group=None, num_batches_divided_by=None,
             same_micro_num_in_dp=True, min_num_micro_batch=None, use_dynamic_bsz_balance=True):
    options = dict(dp_group=dp_group, num_batches_divided_by=num_batches_divided_by,
                   same_micro_num_in_dp=same_micro_num_in_dp,
                   min_num_micro_batch=min_num_micro_batch,
                   use_dynamic_bsz_balance=use_dynamic_bsz_balance)

    def flag(key, default):
        return impl.tu.get_non_tensor_data(data=data, key=key, default=default)

    if (not _ACTIVE.get() or not flag("use_dynamic_bsz", True)
            or flag("use_remove_padding", True)):
        return original(data, **options)

    budget = int(data["max_token_len_per_gpu"]) * int(flag("sp_size", 1))
    group_size = int(flag("force_group_size", 1))
    micro_batches, indices = original(data, **options)
    initial_count = len(micro_batches)
    initial_peak = max(map(_padded_tokens, micro_batches), default=0)
    sync = (torch.distributed.is_initialized() and same_micro_num_in_dp and dp_group is not None)
    while True:
        peak = max(map(_padded_tokens, micro_batches), default=0)
        count = len(micro_batches)
        next_count = count + int(peak > budget)
        max_count = len(data["input_ids"]) // group_size
        if sync:
            # Every rank participates, including ranks whose local batch fits.
            # Communicate on the engine device: CPU tensors cannot use HCCL.
            limits = torch.tensor([count, -count, int(peak > budget), -max_count],
                                  dtype=torch.int64, device=impl.get_device_name())
            torch.distributed.all_reduce(limits, op=torch.distributed.ReduceOp.MAX, group=dp_group)
            common_count, neg_min_count, needs_split, neg_max_count = limits.tolist()
            if common_count != -neg_min_count:
                raise RuntimeError("verl returned different DP microbatch counts before Huanyu padding checks")
            next_count = common_count + needs_split
            max_count = -neg_max_count
        if next_count == count:
            break
        divisor = num_batches_divided_by or 1
        next_count = ((next_count + divisor - 1) // divisor) * divisor
        if next_count > max_count:
            raise ValueError(
                "Huanyu padded-token budget cannot fit an indivisible group with a common DP microbatch count: "
                f"budget={budget}, required_micro_batches={next_count}, available_groups={max_count}. "
                "No samples were dropped or truncated."
            )
        options["min_num_micro_batch"] = next_count
        micro_batches, indices = original(data, **options)
        if len(micro_batches) < next_count:
            raise RuntimeError("verl ignored min_num_micro_batch required by the Huanyu padded-token budget")

    if len(micro_batches) != initial_count:
        print(f"[huanyu-padded-batch] pid={os.getpid()} budget={budget} "
              f"micro_batches={initial_count}->{len(micro_batches)} "
              f"max_padded_tokens={initial_peak}->{peak} dp_sync={int(sync)}", flush=True)
    return micro_batches, indices


def install_huanyu_padded_batching(module):
    """Scope engine batching to models already marked by scalar-output packing.

    The ContextVar is reset even when a forward fails. Optimizer stepping,
    loss normalization, sample indexing and the original batch object remain
    under the upstream engine's control.
    """
    impl = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    engine = getattr(impl, "FSDPEngine", None)
    original = getattr(engine, "forward_backward_batch", None)
    prepare = getattr(impl, "prepare_micro_batches", None)
    if not _enabled(module):
        raise RuntimeError("Install Huanyu scalar-output packing before padded batching")
    if not callable(original) or not callable(prepare):
        raise RuntimeError("Cannot find FSDP batching hooks for the Huanyu padded-token budget")
    if getattr(original, _PATCH_MARKER, False) and getattr(prepare, _PATCH_MARKER, False):
        return module
    expected = ("data", "dp_group", "num_batches_divided_by", "same_micro_num_in_dp",
                "min_num_micro_batch", "use_dynamic_bsz_balance")
    if tuple(inspect.signature(prepare).parameters) != expected:
        raise RuntimeError("Unsupported prepare_micro_batches signature for Huanyu padded batching")
    base = inspect.unwrap(original)
    if ("prepare_micro_batches" not in base.__code__.co_names
            or base.__globals__.get("prepare_micro_batches") is not prepare):
        raise RuntimeError("FSDP forward_backward_batch does not use the expected global batching helper")

    @wraps(prepare)
    def prepare_micro_batches(data, dp_group=None, num_batches_divided_by=None,
                              same_micro_num_in_dp=True, min_num_micro_batch=None, use_dynamic_bsz_balance=True):
        return _prepare(prepare, impl, data, dp_group, num_batches_divided_by,
                        same_micro_num_in_dp, min_num_micro_batch, use_dynamic_bsz_balance)

    @wraps(original)
    def forward_backward_batch(self, *args, **kwargs):
        token = _ACTIVE.set(_enabled(getattr(self, "module", None)))
        try:
            return original(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    setattr(prepare_micro_batches, _PATCH_MARKER, True)
    setattr(forward_backward_batch, _PATCH_MARKER, True)
    impl.prepare_micro_batches = prepare_micro_batches
    engine.forward_backward_batch = forward_backward_batch
    print(f"[huanyu-padded-batch] installed pid={os.getpid()} scope=huanyu budget=B*S_max", flush=True)
    return module
