"""Finish Huanyu FSDP2 host copies before FSDP re-pads uneven shards.

``FSDPModule._apply`` calls ``reset_sharded_param`` after the tensor conversion.
For an uneven shard, that reset copies the newly offloaded CPU tensor into a
padded CPU allocation. An asynchronous device-to-host copy must finish before
that host read. Synchronizing after ``model.to`` returns is already too late.

This instance-only wrapper makes explicit FSDP2 CPU offloads blocking. It does
not reload a checkpoint, replace parameters, or change device-side transfers.
"""

import inspect
import os
from types import MethodType


_ORIGINAL_TO = "_huanyu_fsdp_original_to"
_COUNT = "_huanyu_fsdp_blocking_offload_count"


def _safe_to(self, *args, **kwargs):
    original = getattr(self, _ORIGINAL_TO)
    # The adapter runs before fully_shard changes the root object's class.
    # Keep normal model construction and FSDP1 conversion behavior untouched.
    if not callable(getattr(self, "_get_fsdp_state", None)):
        return original(*args, **kwargs)

    import torch

    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    if device is None or device.type != "cpu" or not non_blocking:
        return original(*args, **kwargs)

    converted = dict(device=device, dtype=dtype, non_blocking=False)
    if memory_format is not None:
        converted["memory_format"] = memory_format
    # nn.Module.to forwards this blocking conversion through the original
    # FSDPModule._apply, preserving its sharding bookkeeping and parameter IDs.
    result = original(**converted)
    count = getattr(self, _COUNT, 0) + 1
    setattr(self, _COUNT, count)
    if count == 1:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown"
        finally:
            del frame
        print(
            f"[huanyu-fsdp-offload] phase=executed pid={os.getpid()} "
            f"reason=host_read_before_shard_repad caller={caller} "
            "non_blocking=False count=1; subsequent offloads use the same guard",
            flush=True,
        )
    return result


def install_huanyu_fsdp_offload(module):
    """Guard a Huanyu root retained by FSDP2, once per model.

    Fully sharded full finetuning keeps this root object. A caller introducing
    a new outer model (for example PEFT) must install on that final root too;
    a parent's ``to`` traverses child ``_apply`` methods, not child ``to``.
    """
    if hasattr(module, _ORIGINAL_TO):
        return module
    setattr(module, _ORIGINAL_TO, module.to)
    setattr(module, _COUNT, 0)
    module.to = MethodType(_safe_to, module)
    print(
        f"[huanyu-fsdp-offload] phase=installed pid={os.getpid()} scope=huanyu "
        "policy=blocking_cpu_offload; execution requires FSDP2 and an explicit CPU transfer",
        flush=True,
    )
    return module
