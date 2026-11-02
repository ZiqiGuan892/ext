"""Verify native repeat-worker initialization without recording any tensors.

Only the standalone repeat launcher enables this check. Batch-invariant kernels
must remain disabled. Off mode disables HCCL/LCCL/NCCL determinism and
ASCEND_LAUNCH_BLOCKING. Torch deterministic algorithms are observed, never
changed here; off mode rejects an enabled setting. Verified settings do not
prove bitwise reproducibility or require repeated outputs to differ.
"""

from functools import wraps
import importlib
import json
import os
import socket
import sys


_GATE = "HUANYU_SERVE_REPEAT_DETERMINISM"
_MODE = "HUANYU_SERVE_REPEAT_DETERMINISM_MODE"
_MARKER = "_huanyu_repeat_determinism_check"
_WORKER = "vllm_omni.platforms.npu.worker.npu_ar_worker"
_EXPECTED = {
    "VLLM_BATCH_INVARIANT": "0", "HCCL_DETERMINISTIC": "strict",
    "LCCL_DETERMINISTIC": "1", "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1",
}


def verify_worker_determinism(worker):
    import torch

    mode = os.environ.get(_MODE, "on")
    if mode not in ("on", "off"):
        raise RuntimeError("invalid repeat determinism mode: " + repr(mode))
    expected = dict(_EXPECTED)
    if mode == "off":
        expected.update(HCCL_DETERMINISTIC="false", ASCEND_LAUNCH_BLOCKING="0",
                        NCCL_DETERMINISTIC="0", LCCL_DETERMINISTIC="0")
    # Inspect only the already-imported module: importing it here could run
    # registration and change the very state being checked.
    module = sys.modules.get("vllm_ascend.batch_invariant")
    registered = (module._batch_invariant_LIB is not None
                  if module is not None and hasattr(module, "_batch_invariant_LIB") else None)
    environment = {key: os.environ.get(key) for key in expected}
    deterministic = torch.are_deterministic_algorithms_enabled()
    errors = [key + " must be " + value for key, value in expected.items() if environment[key] != value]
    if mode == "off" and deterministic:
        errors.append("torch deterministic algorithms must be disabled in off mode")
    if registered:
        errors.append("native Ascend batch-invariant kernels are still registered; restart with the option disabled")
    state = dict(host=socket.gethostname(), pid=os.getpid(), rank=getattr(worker, "rank", None),
                 mode=mode, environment=environment, batch_invariance_requested=False, batch_invariant_registered=registered,
                 torch_deterministic_algorithms=deterministic,
                 torch_deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                 errors=errors)
    status = "FAILED" if errors else "SETTINGS_VERIFIED"
    print("[huanyu-determinism] actual scope=worker_init status=" + status + " "
          + json.dumps(state, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Huanyu repeat determinism check failed: " + "; ".join(errors)
                           + ". Restart with the current repeat launcher and batch_invariant=false.")


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
