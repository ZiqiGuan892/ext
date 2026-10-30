"""Verify native repeat-worker initialization without recording any tensors.

Only the standalone repeat launcher enables this check. It observes the native
registration after init_device; it must not initialize kernels itself, which
would hide a skipped native initialization. Verified settings are not a proof
of bitwise reproducibility of every operator.
"""

from functools import wraps
import importlib
import json
import os
import socket
import sys


_GATE = "HUANYU_SERVE_REPEAT_DETERMINISM"
_MARKER = "_huanyu_repeat_determinism_check"
_WORKER = "vllm_omni.platforms.npu.worker.npu_ar_worker"
_EXPECTED = {
    "VLLM_BATCH_INVARIANT": "1", "HCCL_DETERMINISTIC": "strict",
    "LCCL_DETERMINISTIC": "1", "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1",
}


def verify_worker_determinism(worker):
    import torch

    # Inspect only the already-imported module: importing it here could run
    # registration and turn missing native setup into a false success.
    module = sys.modules.get("vllm_ascend.batch_invariant")
    registered = getattr(module, "_batch_invariant_LIB", None) is not None
    environment = {key: os.environ.get(key) for key in _EXPECTED}
    deterministic = torch.are_deterministic_algorithms_enabled()
    errors = [key + " must be " + value for key, value in _EXPECTED.items() if environment[key] != value]
    if not registered:
        errors.append("native Ascend batch-invariant registration is missing or unsupported")
    if not deterministic:
        errors.append("torch deterministic algorithms are disabled")
    state = dict(host=socket.gethostname(), pid=os.getpid(), rank=getattr(worker, "rank", None),
                 environment=environment, batch_invariant_registered=registered,
                 torch_deterministic_algorithms=deterministic,
                 torch_deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                 errors=errors)
    status = "FAILED" if errors else "SETTINGS_VERIFIED"
    print("[huanyu-determinism] actual scope=worker_init status=" + status + " "
          + json.dumps(state, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Huanyu repeat determinism check failed: " + "; ".join(errors)
                           + ". Check native batch-invariant kernel/build support; no fallback is allowed.")


def install_repeat_determinism_check():
    if os.environ.get(_GATE) != "1":
        return False
    worker_class = importlib.import_module(_WORKER).NPUARWorker
    original = worker_class.init_device
    if getattr(original, _MARKER, False):
        return True

    @wraps(original)
    def checked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if os.environ.get(_GATE) == "1":
            verify_worker_determinism(self)
        return result

    setattr(checked, _MARKER, True)
    worker_class.init_device = checked
    return True
