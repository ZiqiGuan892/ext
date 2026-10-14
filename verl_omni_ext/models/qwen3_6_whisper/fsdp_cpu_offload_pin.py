
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
"""Force FSDP2's ``CPUOffloadPolicy`` to ``pin_memory=False`` on Ascend.

The failure this exists for
---------------------------
16 ranks, one node, first ``loss.backward()`` of the first training step.
13 of the 16 die simultaneously with::

    File ".../torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py", line 625
      new_sharded_grad = new_sharded_grad.to(
    RuntimeError: ... srclen=134217728, dstlen=134217728, kind=2.
    Invalid_Argument_Null_Pointer(EH0002): Argument [dst] must not be null.

``kind=2`` is Device->Host.  ``dst`` is the *host* destination pointer.  So the
host-side allocation backing that copy came back null.

Why it is the pinned pool and not plain RAM
-------------------------------------------
The decisive clue is that the failure is **size-selective**.  Across the 13
tracebacks only two ``srclen`` values ever appear, 134217728 and 67108864 --
2**29 and 2**28 elements at fp32.  Those are exactly the fused MoE expert
shapes, ``[256, 2048, 1024]`` and ``[256, 2048, 512]``.  Every smaller
parameter in the same ``foreach_reduce`` loop -- norms, biases, the attention
projections -- copied out fine.

If async D2H into host memory were simply unsupported here, the *first* and
smallest tensor would have failed.  A failure that only bites the two largest
buffers is an allocator running out of a contiguous region, not an API
contract being violated.

And it is the *pinned* region, not ordinary RAM, because nothing in the log
looks like ordinary RAM exhaustion: no kernel OOM-killer line, no
``DefaultCPUAllocator: can't allocate memory``, no ``MemoryError``.  Pinned
(page-locked) host memory is a separate, much smaller, non-swappable budget,
and its allocator signals exhaustion by handing back a null pointer -- which is
precisely the message above.

Where the pinned memory goes
----------------------------
``verl/workers/engine/fsdp/transformer_impl.py:443`` hardcodes::

    offload_policy = CPUOffloadPolicy(pin_memory=True)

with no config knob, for the actor (``offload_policy=true``) *and* for the ref
worker (the same branch fires on ``forward_only``).  That flag has two effects,
both of which this patch removes:

1. **Init-time pinning of every parameter shard.**
   ``_fsdp_param.py:239``  ``self.pin_memory = offload_to_cpu and
   offload_policy.pin_memory``, then ``:388`` ``padded_sharded_param =
   padded_sharded_param.pin_memory(...)``.  The census printed by
   ``_report_param_census`` measures this checkpoint at **35.95B** parameters,
   so ``35.95e9 / 16 ranks * 4 B`` = ~9 GiB pinned per rank for the fp32 actor,
   times 16 ranks on the node, plus the bf16 ref worker's ~4.5 GiB per rank.
   Order 200 GiB of page-locked host memory before a single gradient exists.

2. **The async D2H itself.**
   ``_fsdp_collectives.py:478``  ``non_blocking = fsdp_param.pin_memory and not
   to_accumulate_grad``.  Note ``to_accumulate_grad`` is ``sharded_param.grad
   is not None``, so ``non_blocking`` is True exactly on the **first**
   micro-batch of the step -- which is why this crashes on step 1 rather than
   drifting into it.  With ``pin_memory=False`` that expression is False, the
   copy becomes synchronous, and the failing path is not taken at all.

Confirmed on hardware
---------------------
Same 16-card launch, same config, this one boolean the only change: step 1
completed end to end (``training/global_step:1``, zero tracebacks) where the
previous run lost 13 of 16 ranks in the first ``loss.backward()``.  The run
reports ``actor/perf/cpu_memory_used_gb: 1709`` against a 2010 GiB node -- so
ordinary host RAM was never the binding constraint, but the page-locked subset
of it was.

torch_npu ships its own FSDP2 patch (``torch_npu/distributed/fsdp/
_add_fsdp_patch.py``) whose ``foreach_reduce`` carries this block **verbatim**
-- line 452 is the same ``non_blocking = fsdp_param.pin_memory and not
to_accumulate_grad``.  So whether or not that patch is installed changes
nothing here; both roads lead to the same line.

What it costs
-------------
Real but modest: the D2H copy of sharded gradients no longer overlaps with the
reduce-scatter stream, so each backward pays serialised copy time.  Correctness
is unaffected -- the same bytes reach the same CPU tensors, and the
``grad_offload_event`` that a non-blocking copy would have recorded exists only
to make the CPU wait for that copy, which a blocking copy has already done.

Scope
-----
Applied from ``verl_omni_ext.models.minicpmo_5_0.__init__``, so it is scoped to
MiniCPM-o runs; Qwen3.5-MoE and Qwen3-Omni runs import a different pipeline
package and keep stock behaviour.  Set ``MINICPMO_FSDP2_PIN_MEMORY=1`` to
restore ``pin_memory=True`` and get the stock crash back.
"""

import os

_ENV = "MINICPMO_FSDP2_PIN_MEMORY"
_TARGET_MODULE = "verl.workers.engine.fsdp.transformer_impl"

_applied = False


def apply_fsdp2_cpu_offload_pin_memory_patch():
    """Rebind ``CPUOffloadPolicy`` inside verl's FSDP engine to force ``pin_memory=False``.

    Patches the *name in verl's module namespace* rather than the torch class,
    so nothing outside verl's engine changes and ``isinstance(policy,
    CPUOffloadPolicy)`` checks inside torch keep working -- the object handed
    back is a genuine ``CPUOffloadPolicy``, just with the flag flipped.

    Idempotent, and a no-op when ``MINICPMO_FSDP2_PIN_MEMORY=1``.

    Import order is safe by construction: ``OmniModelConfig.__post_init__``
    resolves the adapter through the registry, which imports this package, and
    that happens before ``_build_module`` -- let alone ``_build_fsdp_module``,
    which is where line 443 runs.
    """
    global _applied
    if _applied:
        return
    if os.environ.get(_ENV, "0") == "1":
        print(
            f"[verl-omni] FSDP2 CPUOffloadPolicy left at stock pin_memory=True ({_ENV}=1). "
            "Expect a null host destination pointer in the first loss.backward() if the "
            "node's page-locked memory budget is the binding constraint.",
            flush=True,
        )
        _applied = True
        return

    try:
        import importlib

        impl = importlib.import_module(_TARGET_MODULE)
        real_policy = impl.CPUOffloadPolicy
    except Exception as e:  # verl not importable yet, or upstream moved the symbol
        # Deliberately not fatal. A missing patch shows up as a decodable
        # crash in backward; a hard failure here would take down model
        # loading with an error that names none of the above.
        print(
            f"[verl-omni] Could not patch FSDP2 CPUOffloadPolicy in {_TARGET_MODULE}: {e!r}. "
            "Falling back to stock pin_memory=True.",
            flush=True,
        )
        return

    if real_policy is None:  # verl's own guard: torch < 2.4 has no FSDP2
        print(
            "[verl-omni] FSDP2 CPUOffloadPolicy is None in this torch build; nothing to patch.",
            flush=True,
        )
        _applied = True
        return

    def _cpu_offload_policy_no_pin(*args, **kwargs):
        kwargs["pin_memory"] = False
        return real_policy(*args, **kwargs)

    # Keep the class visible on the factory so anything reaching for the type
    # through verl's namespace still finds it.
    _cpu_offload_policy_no_pin.cls = real_policy

    impl.CPUOffloadPolicy = _cpu_offload_policy_no_pin
    _applied = True
    # Deliberately does NOT spell out the Ascend error code this prevents.
    # An earlier version did, and since every one of the 16 workers prints this
    # line, `grep -c` for that code in a healthy run returned 51 -- a diagnostic
    # that lies about the thing it is diagnosing. See the docstring instead.
    print(
        "[verl-omni] Patched FSDP2 CPUOffloadPolicy -> pin_memory=False for actor and ref "
        f"(hardcoded True at {_TARGET_MODULE}:443). Drops the node-wide page-locked host "
        "memory reservation and makes the sharded-gradient D2H copy in foreach_reduce "
        f"synchronous. Set {_ENV}=1 to restore stock behaviour.",
        flush=True,
    )


def report_host_memory(tag):
    """Print the node's host-memory counters, including the pinned total.

    ``Mlocked`` in ``/proc/meminfo`` is the page-locked total for the whole
    node, which is the number this patch exists to reduce and the one number no
    part of the training log otherwise reports.  Printed rather than logged:
    ``logger.info`` from inside a Ray worker does not reach the driver's output
    on this stack -- proven twice now, most recently by the forward adapter's
    first-call line being absent from an entire 4968-line run.
    """
    wanted = ("MemTotal", "MemAvailable", "Mlocked", "Cached", "SwapTotal", "SwapFree")
    try:
        with open("/proc/meminfo") as fh:
            fields = dict(
                (parts[0].rstrip(":"), parts[1] + " " + (parts[2] if len(parts) > 2 else ""))
                for parts in (line.split() for line in fh)
                if parts and parts[0].rstrip(":") in wanted
            )
    except OSError as e:
        print(f"[verl-omni] host memory ({tag}): unavailable, {e!r}", flush=True)
        return

    rendered = " ".join(f"{k}={fields.get(k, '?').strip()}" for k in wanted)
    print(f"[verl-omni] host memory ({tag}, rank {os.environ.get('RANK', '?')}): {rendered}", flush=True)