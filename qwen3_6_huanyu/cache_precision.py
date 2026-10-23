"""Apply the shared hybrid-cache arithmetic only to Huanyu rollout workers.

Importing this module does not import vLLM. The external engine entry point
calls the installer after the native worker classes are available. Selection is
based on checkpoint configuration, never a model-directory naming convention.
"""

from collections.abc import Mapping
from functools import wraps
import importlib
import os


_UNAVAILABLE_REPORTED = set()
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


def _unavailable(stage, error):
    if stage not in _UNAVAILABLE_REPORTED:
        _UNAVAILABLE_REPORTED.add(stage)
        print(f"[huanyu-cache-precision] unavailable stage={stage}: {error}", flush=True)


def _install_zero():
    if os.environ.get("MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE", "1") != "1":
        return False
    try:
        runner = importlib.import_module("vllm_omni.worker.gpu_model_runner").OmniGPUModelRunner
        original = runner._torch_zero_attention_block_ids
    except (ImportError, AttributeError) as error:
        # Older native checkouts have no zeroing hook. Make the absence visible
        # without claiming a fix was installed or breaking their default path.
        _unavailable("logical_zero", error)
        return False
    if not callable(original):
        _unavailable("logical_zero", "native _torch_zero_attention_block_ids is not callable")
        return False
    if getattr(original, "_huanyu_logical_blocks", False):
        return True
    shared = _shared()

    @wraps(original)
    def zero(self, block_ids):
        if is_huanyu_policy(self):
            return shared.zero_attention_blocks(self, block_ids)
        return original(self, block_ids)

    zero._huanyu_logical_blocks = True
    runner._torch_zero_attention_block_ids = zero
    print("[huanyu-cache-precision] installed logical-to-physical KV zeroing; scope=huanyu", flush=True)
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
        self.model_runner.supports_standardized_shared_kv_backing = False
        groups = get_groups(self.vllm_config, self.get_kv_cache_spec())
        result = shared.planner_budget(available, groups)
        self.model_runner._q36_private_cache_budget_applied = True
        self.model_runner._q36_private_cache_available_bytes = available
        print(f"[huanyu-cache-precision] physical cache budget: available={available}, planner={result}", flush=True)
        return result

    @wraps(original_allocate)
    def allocate(self, config):
        if is_huanyu_policy(self):
            return shared.private_buffers(self, config)
        return original_allocate(self, config)

    budget._huanyu_private_cache = True
    allocate._huanyu_private_cache = True
    worker._scale_kv_cache_memory_for_multi_group = budget
    runner._allocate_kv_cache_tensors = allocate
    print("[huanyu-cache-precision] installed private hybrid buffers and matching planner; scope=huanyu", flush=True)
    return True


def install_huanyu_cache_precision():
    """Return whether any enabled cache hook is installed in this process."""
    zero_installed = _install_zero()
    private_installed = _install_private()
    return zero_installed or private_installed
