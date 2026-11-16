"""Keep frozen vision FSDP units resident across all media in one forward.

Per-image encoding may call a vision block a different number of times on each
rank. Its FSDP post-forward must not reshard between those local calls. Gather
the same vision units before the base model body, including on image-free ranks,
and release them locally after the body. This also covers PEFT's direct forward
call, which skips hooks on the base model itself.
"""

from functools import wraps
from types import MethodType

import torch


_FLAG = "_huanyu_frozen_vision_fsdp"


def _vision_units(visual):
    # FSDP wrapping happens AFTER adapter configuration, so discover at runtime.
    units = sorted(
        ((name, child) for name, child in visual.named_modules()
         if callable(getattr(child, "unshard", None))),
        key=lambda item: item[0],
    )
    for name, child in units:
        for method in ("reshard", "set_reshard_after_forward"):
            if not callable(getattr(child, method, None)):
                raise RuntimeError(
                    f"Huanyu frozen vision FSDP unit {name!r} lacks {method}; "
                    "cannot safely retain parameters across media items"
                )
    return units


def _release(units):
    errors = []
    for name, child in units:
        try:
            child.reshard()
        except Exception as error:
            errors.append((name, error))
    return errors


def _report(phase, call, units):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    print(f"[huanyu-vision-fsdp] phase={phase} rank={rank} call={call} units={len(units)}", flush=True)


def install_frozen_vision_fsdp(module):
    """Install on the online Huanyu base model, after its forward adapter.

    Only frozen visual parameters are managed; audio and language FSDP policies
    remain owned by the engine. Cached-media callers must not install this wrapper.
    Input pixels are raw data, not outputs of a trainable upstream image model.
    """
    visual = getattr(getattr(module, "model", None), "visual", None)
    if visual is None or getattr(module, _FLAG, False):
        return False
    if any(parameter.requires_grad for parameter in visual.parameters()):
        return False
    original = module.forward.__func__
    active = False
    calls = 0

    @wraps(original)
    def forward(self, *args, **kwargs):
        nonlocal active, calls
        if active:
            return original(self, *args, **kwargs)
        tower = self.model.visual
        # PEFT may add trainable modules after configure_model. Do not take over
        # their forward/backward parameter lifecycle.
        if any(parameter.requires_grad for parameter in tower.parameters()):
            if calls:
                raise RuntimeError("Huanyu vision became trainable after frozen FSDP management started")
            return original(self, *args, **kwargs)
        units = _vision_units(tower)
        if not units:
            return original(self, *args, **kwargs)
        calls += 1
        active = True
        failure = None
        try:
            if calls <= 2:
                _report("enter", calls, units)
            for _, child in units:
                # Non-recursive: never change policies outside these units.
                child.set_reshard_after_forward(False, recurse=False)
            # Equalize residual state even when a previous call had no images.
            for _, child in units:
                child.reshard()
            for _, child in units:
                child.unshard()
            if calls <= 2:
                _report("gathered", calls, units)
            return original(self, *args, **kwargs)
        except BaseException as error:
            failure = error
            raise
        finally:
            # reshard is local and must run on both success and failure. Keeping
            # the policy False is intentional: this wrapper owns the release
            # boundary of these frozen units, also on the next model call.
            try:
                errors = _release(units)
                if errors:
                    detail = "; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in errors)
                    message = f"Huanyu frozen vision FSDP release failed: {detail}"
                    if failure is not None:
                        failure.add_note(message)
                    else:
                        raise RuntimeError(message) from errors[0][1]
                elif calls <= 2:
                    _report("released", calls, units)
            finally:
                active = False

    module.forward = MethodType(forward, module)
    setattr(module, _FLAG, True)
    print("[huanyu-vision-fsdp] installed scope=frozen_visual entrypoints=module_call,direct_forward", flush=True)
    return True
