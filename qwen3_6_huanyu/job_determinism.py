"""Opt-in deterministic computation for Huanyu training and inference workers."""

from functools import wraps
import importlib
import json
import os
import socket
import sys


_GATE = "HUANYU_JOB_DETERMINISM"
_EXPECTED = {"HCCL_DETERMINISTIC": "strict", "ASCEND_LAUNCH_BLOCKING": "1",
             "NCCL_DETERMINISTIC": "1", "LCCL_DETERMINISTIC": "1", "VLLM_BATCH_INVARIANT": "0"}
_MARKER = "_huanyu_job_determinism_check"


def _verify(scope, *, initialized):
    import torch

    environment = {key: os.environ.get(key) for key in _EXPECTED}
    errors = [f"{key} must be {value}" for key, value in _EXPECTED.items() if environment[key] != value]
    native = sys.modules.get("vllm_ascend.batch_invariant")
    registered = (native._batch_invariant_LIB is not None
                  if native is not None and hasattr(native, "_batch_invariant_LIB") else None)
    if registered:
        errors.append("batch-invariant kernels must be disabled")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if initialized and (not enabled or warn_only):
        errors.append("Torch deterministic algorithms must be enabled with warn_only=False")
    if initialized or errors:
        print("[huanyu-determinism] actual scope=" + scope + " status="
              + ("FAILED" if errors else "SETTINGS_VERIFIED") + " "
              + json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), mode="on",
                                rank=os.environ.get("RANK"), environment=environment,
                                torch_deterministic_algorithms=enabled,
                                torch_deterministic_warn_only=warn_only,
                                batch_invariant_registered=registered, errors=errors), sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Huanyu job determinism check failed: " + "; ".join(errors))


def configure_job_determinism(scope="actor_ref_model_config"):
    if os.environ.get(_GATE) != "1":
        return False
    import torch

    _verify(scope, initialized=False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    _verify(scope, initialized=True)
    return True


def install_job_determinism_check():
    if os.environ.get(_GATE) != "1":
        return False
    worker = importlib.import_module("vllm_omni.platforms.npu.worker.npu_ar_worker").NPUARWorker
    original = worker.init_device
    if getattr(original, _MARKER, False):
        return True

    @wraps(original)
    def checked(self, *args, **kwargs):
        if os.environ.get(_GATE) != "1":
            return original(self, *args, **kwargs)
        import torch

        scope = "rollout_worker_init_rank_" + str(getattr(self, "rank", "unknown"))
        _verify(scope, initialized=False)
        torch.use_deterministic_algorithms(True, warn_only=False)
        result = original(self, *args, **kwargs)
        _verify(scope, initialized=True)
        return result

    setattr(checked, _MARKER, True)
    worker.init_device = checked
    return True
