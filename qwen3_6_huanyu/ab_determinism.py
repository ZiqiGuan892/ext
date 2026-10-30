"""Enable and verify both A/B inference workers, independently of batch invariance.

The launcher sets communication flags before process creation. Do not repair
missing flags inside an initialized worker: the backend may have cached them.
This gate is separate from normal training and the existing repeat experiments.
"""

from functools import wraps
import importlib
import json
import os
import socket
import sys


_GATE = "HUANYU_TEXT_AB_DETERMINISM"
_MARKER = "_huanyu_ab_determinism_check"
_WORKER = "vllm_omni.platforms.npu.worker.npu_ar_worker"


def _expected_environment(*, initialized):
    batch = os.environ.get("HUANYU_TEXT_AB_BATCH_INVARIANT", "0") == "1"
    return {
        "VLLM_BATCH_INVARIANT": "1" if batch and initialized else "0",
        "HCCL_DETERMINISTIC": "true" if batch and not initialized else "strict",
        "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1", "LCCL_DETERMINISTIC": "1",
    }


def verify_ab_determinism(worker, *, initialized=True):
    import torch

    expected = _expected_environment(initialized=initialized)
    environment = {key: os.environ.get(key) for key in expected}
    errors = [key + " must be " + value for key, value in expected.items() if environment[key] != value]
    side = os.environ.get("HUANYU_TEXT_AB_SIDE")
    if side not in ("serve", "rollout"):
        errors.append("HUANYU_TEXT_AB_SIDE must identify serve or rollout")
    native = sys.modules.get("vllm_ascend.batch_invariant")
    registered = (native._batch_invariant_LIB is not None
                  if native is not None and hasattr(native, "_batch_invariant_LIB") else None)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if initialized:
        batch = expected["VLLM_BATCH_INVARIANT"] == "1"
        if (batch and registered is not True) or (not batch and registered):
            errors.append("native batch-invariant registration disagrees with requested mode")
        if not enabled or warn_only:
            errors.append("torch deterministic algorithms must be enabled with warn_only=False")
    if initialized or errors:
        state = dict(host=socket.gethostname(), pid=os.getpid(), rank=getattr(worker, "rank", None),
                     side=side, environment=environment, batch_invariant_registered=registered,
                     torch_deterministic_algorithms=enabled, torch_deterministic_warn_only=warn_only,
                     errors=errors)
        print("[huanyu-determinism] actual scope=ab_worker_init status="
              + ("FAILED" if errors else "SETTINGS_VERIFIED") + " "
              + json.dumps(state, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Huanyu A/B determinism check failed: " + "; ".join(errors))


def install_ab_determinism_check():
    if os.environ.get(_GATE) != "1":
        return False
    worker_class = importlib.import_module(_WORKER).NPUARWorker
    original = worker_class.init_device
    if getattr(original, _MARKER, False):
        return True

    @wraps(original)
    def checked(self, *args, **kwargs):
        if os.environ.get(_GATE) != "1":
            return original(self, *args, **kwargs)
        import torch

        verify_ab_determinism(self, initialized=False)
        # Before native device/distributed initialization, not merely in the
        # driver: torch settings are process-local under spawn/Ray.
        torch.use_deterministic_algorithms(True, warn_only=False)
        result = original(self, *args, **kwargs)
        verify_ab_determinism(self)
        return result

    setattr(checked, _MARKER, True)
    worker_class.init_device = checked
    return True
