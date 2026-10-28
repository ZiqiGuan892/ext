"""Opt-in, eight-parameter Huanyu sync audit; never performs a collective.

Set HUANYU_WEIGHT_SYNC_AUDIT_DIR on actor and rollout processes. Updates are
one-based and HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES defaults to 1. Initial load
records have round_id=None. Device-to-CPU reads synchronize the selected tiny
tensors; CPU snapshots also synchronize already-initialized device backends to
finish pending asynchronous offloads. Pre-export observations can mask a race.
"""

from collections.abc import Mapping
from functools import wraps
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import uuid


NAMES = frozenset(
    f"model.audio_encoder.conv_layers.{block}.layers.{layer}.{leaf}"
    for block in (0, 1) for layer in (1, 4) for leaf in ("weight", "bias")
)
_LOCK = threading.Lock()
_FILES = {}


def enabled(round_id=None):
    """True only for explicitly enabled initialization or a bounded update."""
    if not os.environ.get("HUANYU_WEIGHT_SYNC_AUDIT_DIR", "").strip():
        return False
    maximum = int(os.environ.get("HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES", "1"))
    if maximum < 1:
        raise ValueError("HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES must be >= 1")
    return round_id is None or round_id <= maximum


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _canonical(name):
    name = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
    for prefix in ("thinker.apm.", "apm.", "audio_encoder."):
        if name.startswith(prefix):
            name = "model.audio_encoder." + name[len(prefix):]
            break
    return name if name in NAMES else None


def source_info(obj):
    """Fingerprint a Python class, function or module without importing it."""
    path = inspect.getsourcefile(obj)
    return {
        "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__qualname__)),
        "file": path,
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
        "method_sha256": hashlib.sha256(inspect.getsource(obj).encode()).hexdigest(),
    }


def _tensor_info(tensor):
    import torch

    result = {"global_shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if hasattr(tensor, "to_local"):
        result["placements"] = [str(p) for p in tensor.placements]
        mesh = tensor.device_mesh
        result["mesh_shape"] = list(mesh.shape)
        result["mesh_coordinate"] = mesh.get_coordinate()
        tensor = tensor.to_local()  # Local access only; never full_tensor().
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("weight audit expects a Tensor or DTensor")
    result.update(shape=list(tensor.shape), device=str(tensor.device), is_meta=tensor.is_meta)
    if tensor.is_meta:
        return result
    result["diagnostic_sync"] = []
    if tensor.device.type == "cpu":
        # .cpu() is a no-op on an unfinished asynchronous offload destination.
        for name in ("npu", "cuda"):
            if name == "npu" and "torch_npu" not in sys.modules:
                continue
            backend = getattr(torch, name, None)
            if backend is not None and backend.is_initialized():
                backend.synchronize()
                result["diagnostic_sync"].append(name)
    local = tensor.detach().cpu().contiguous()
    values = local.float().contiguous()
    result.update(
        raw_sha256=hashlib.sha256(local.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        float32_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        finite=bool(torch.isfinite(values).all()), numel=local.numel(),
    )
    if local.numel() <= 256:
        result["values"] = [v if math.isfinite(v) else str(v) for v in values.reshape(-1).tolist()]
    return result


def record(stage, weights, *, round_id=None, extra=None):
    """Append selected tensors without consuming weights when disabled.

    Callers must not pass a shared generator that must subsequently be consumed;
    use a singleton list while iterating it, or a separate named_parameters view.
    Opt-in failures propagate instead of silently leaving incomplete evidence.
    """
    if not enabled(round_id):
        return
    directory = Path(os.environ["HUANYU_WEIGHT_SYNC_AUDIT_DIR"]).resolve()
    for name, tensor in weights:
        canonical = _canonical(name)
        if canonical is None:
            continue
        entry = dict(schema_version=1, stage=stage, round_id=round_id,
                     name=canonical, source_name=name, host=socket.gethostname(),
                     pid=os.getpid(), rank=os.environ.get("RANK"), time_ns=time.time_ns(),
                     tensor=_tensor_info(tensor), extra=extra or {})
        entry["versions"] = {name: str(getattr(sys.modules[name], "__version__", "unknown"))
                             for name in ("torch", "torch_npu") if name in sys.modules}
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
        with _LOCK:
            key = (str(directory), os.getpid())
            if key not in _FILES:
                directory.mkdir(parents=True, exist_ok=True)
                _FILES[key] = directory / f"weights-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}.jsonl"
            with _FILES[key].open("a", encoding="utf-8") as stream:
                stream.write(line)


def _checkpoint(path):
    from safetensors import safe_open

    root = Path(path).resolve()
    index = root / "model.safetensors.index.json"
    mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.is_file() else None
    shards = sorted({mapping[n] for n in NAMES if n in mapping}) if mapping is not None else sorted(
        p.name for p in root.glob("*.safetensors"))
    found = set()
    for shard in shards:
        with safe_open(str(root / shard), framework="pt", device="cpu") as reader:
            for name in sorted(NAMES.intersection(reader.keys())):
                if name in found:
                    raise ValueError(f"Duplicate checkpoint audit weight: {name}")
                found.add(name)
                record("checkpoint", [(name, reader.get_tensor(name))],
                       extra={"checkpoint": str(root), "shard": shard})
    if found != NAMES:
        raise ValueError(f"Checkpoint missing Huanyu audit weights: {sorted(NAMES - found)}")


def _huanyu(config):
    hf = _field(config, "hf_config", config)
    return _field(_field(hf, "audio_config"), "model_type") == "huanyu_conformer_0_6b"


def _install_export_audit():
    cls = importlib.import_module("verl_omni.workers.engine.fsdp.omni_impl").OmniFSDPEngine
    original = cls.get_per_tensor_param
    if getattr(original, "_huanyu_weight_sync_audit", False):
        return
    source = source_info(original)

    @wraps(original)
    def get_per_tensor_param(self, *args, **kwargs):
        if not enabled() or not _huanyu(self.model_config):
            return original(self, *args, **kwargs)
        round_id = getattr(self, "_huanyu_weight_sync_audit_round", 0) + 1
        self._huanyu_weight_sync_audit_round = round_id
        if not enabled(round_id):
            return original(self, *args, **kwargs)
        extra = {"export_source": source}
        record("actor_local_shard", self.module.named_parameters(), round_id=round_id, extra=extra)
        weights, peft_config = original(self, *args, **kwargs)

        def observed():
            for pair in weights:
                # Original generator has already gathered/cast this tensor.
                record("actor_export", [pair], round_id=round_id, extra=extra)
                yield pair

        return observed(), peft_config

    get_per_tensor_param._huanyu_weight_sync_audit = True
    cls.get_per_tensor_param = get_per_tensor_param


def start_actor_audit(module, model_config):
    """Call after audio checkpoint loading and before FSDP wraps the module."""
    if not enabled() or not _huanyu(model_config):
        return module
    _checkpoint(_field(model_config, "path"))
    record("actor_loaded", module.named_parameters(),
           extra={"checkpoint": str(_field(model_config, "path")), "source": source_info(type(module))})
    _install_export_audit()
    return module
