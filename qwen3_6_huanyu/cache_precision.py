"""Apply the shared hybrid-cache arithmetic only to Huanyu rollout workers.

Importing this module does not import vLLM. The external engine entry point
calls the installer after the native worker classes are available. Selection is
based on checkpoint configuration, never a model-directory naming convention.
"""

from collections.abc import Mapping
from functools import wraps
import importlib
import os
import socket


_REPORTED = set()
_AUDIO_TYPE = "huanyu_conformer_0_6b"


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def is_huanyu_policy(owner):
    model_config = _field(owner, "model_config")
    if model_config is None:
        model_config = _field(_field(owner, "vllm_config"), "model_config")
    audio = _field(_field(model_config, "hf_config"), "audio_config")
    return _field(audio, "model_type") == _AUDIO_TYPE


def _shared():
    # Reuse the verified logical/page mapping and allocation calculations. Do
    # not reuse the Whisper installer's path-based policy predicate.
    return importlib.import_module("verl_omni_ext.models.qwen3_6_huanyu.qwen_precision.cache_precision")


def _report_once(event, component, target, **details):
    # Include PID in the key so a forked child emits its own execution proof.
    key = (os.getpid(), event, component)
    if key in _REPORTED:
        return
    fields = " ".join(f"{name}={value}" for name, value in details.items())
    print(
        f"[huanyu-cache-precision] event={event} component={component} "
        f"host={socket.gethostname()} pid={os.getpid()} module={target} scope=huanyu {fields}",
        flush=True,
    )
    _REPORTED.add(key)


def _install_zero():
    if os.environ.get("MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE", "1") != "1":
        return False
    try:
        runner = importlib.import_module("vllm_omni.worker.gpu_model_runner").OmniGPUModelRunner
        original = runner._torch_zero_attention_block_ids
        if not callable(original):
            raise AttributeError("native _torch_zero_attention_block_ids must be callable")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE=1 requests Huanyu logical-to-physical KV zeroing, "
            "but the selected vLLM-Omni does not expose a callable "
            "OmniGPUModelRunner._torch_zero_attention_block_ids"
        ) from error
    if getattr(original, "_huanyu_logical_blocks", False):
        return True
    shared = _shared()

    @wraps(original)
    def zero(self, block_ids):
        if is_huanyu_policy(self):
            result = shared.zero_attention_blocks(self, block_ids)
            # This per-call evidence is committed only after every index_fill_
            # returned successfully. A previous successful call cannot make an
            # empty/no-cache call look like an execution. No device sync needed.
            evidence = getattr(self, "_huanyu_zero_execution", None)
            if evidence:
                _report_once(
                    "executed", "logical_zero",
                    "vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner._torch_zero_attention_block_ids",
                    **evidence,
                )
            return result
        return original(self, block_ids)

    zero._huanyu_logical_blocks = True
    runner._torch_zero_attention_block_ids = zero
    _report_once(
        "installed", "logical_zero",
        "vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner._torch_zero_attention_block_ids",
    )
    return True


def _install_private():
    if os.environ.get("MINICPMO_PRIVATE_HYBRID_CACHE", "0") != "1":
        return False
    try:
        worker = importlib.import_module("vllm_ascend.worker.worker").NPUWorker
        runner = importlib.import_module("vllm_ascend.worker.model_runner_v1").NPUModelRunner
        get_groups = importlib.import_module("vllm.v1.core.kv_cache_utils").get_kv_cache_groups
        original_budget = worker._scale_kv_cache_memory_for_multi_group
        original_allocate = runner._allocate_kv_cache_tensors
        if not all(callable(method) for method in (get_groups, original_budget, original_allocate)):
            raise AttributeError("private-cache planner/allocation APIs must be callable")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "MINICPMO_PRIVATE_HYBRID_CACHE=1 explicitly requests Huanyu private buffers, "
            "but the selected vLLM-Ascend does not expose the required planner/allocation APIs"
        ) from error
    budget_installed = getattr(original_budget, "_huanyu_private_cache", False)
    allocation_installed = getattr(original_allocate, "_huanyu_private_cache", False)
    if budget_installed != allocation_installed:
        raise RuntimeError("Huanyu private-cache budget/allocation hooks are only partially installed")
    if budget_installed:
        return True
    shared = _shared()

    @wraps(original_budget)
    def budget(self, available):
        if not is_huanyu_policy(self):
            return original_budget(self, available)
        self.model_runner._q36_private_cache_budget_applied = False
        self.model_runner.supports_standardized_shared_kv_backing = False
        groups = list(get_groups(self.vllm_config, self.get_kv_cache_spec()))
        result = shared.planner_budget(available, groups)
        if groups:
            self.model_runner._q36_private_cache_budget_applied = True
            self.model_runner._q36_private_cache_available_bytes = available
            _report_once(
                "executed", "private_budget",
                "vllm_ascend.worker.worker.NPUWorker._scale_kv_cache_memory_for_multi_group",
                available_bytes=available, planner_bytes=result, groups=len(groups),
            )
        return result

    @wraps(original_allocate)
    def allocate(self, config):
        if is_huanyu_policy(self):
            buffers = shared.private_buffers(self, config)
            allocated = sum(value.numel() * value.element_size() for value in buffers.values())
            if allocated:
                _report_once(
                    "executed", "private_buffers",
                    "vllm_ascend.worker.model_runner_v1.NPUModelRunner._allocate_kv_cache_tensors",
                    layers=len(buffers), blocks=config.num_blocks, allocated_bytes=allocated,
                )
            return buffers
        return original_allocate(self, config)

    budget._huanyu_private_cache = True
    allocate._huanyu_private_cache = True
    worker._scale_kv_cache_memory_for_multi_group = budget
    runner._allocate_kv_cache_tensors = allocate
    _report_once(
        "installed", "private_budget",
        "vllm_ascend.worker.worker.NPUWorker._scale_kv_cache_memory_for_multi_group",
    )
    _report_once(
        "installed", "private_buffers",
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner._allocate_kv_cache_tensors",
    )
    return True


def install_huanyu_cache_precision():
    """Return whether any enabled cache hook is installed in this process."""
    zero_installed = _install_zero()
    private_installed = _install_private()
    return zero_installed or private_installed
