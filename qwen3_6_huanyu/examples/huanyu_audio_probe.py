"""Opt-in, neutral capture of the native Huanyu inference audio boundary.

Import as ``huanyu_audio_probe`` for a standalone worker extension, or call
``attach_audio_probe`` after installing the training rollout's existing hooks.
This module never imports verl_omni_ext or changes model arithmetic. Capturing
copies tensors to CPU and therefore synchronizes the device; use a diagnostic
run, not performance measurements. NPZ files contain no pickled objects.
"""

from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
from types import MethodType
import uuid


_ACTIVE = ContextVar("huanyu_audio_probe_active", default=None)
_NATIVE = "vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni"


def _digest(array):
    # Flatten before exposing bytes: memoryview cannot cast zero-sized 2-D
    # buffers, while native modules may register empty or scalar buffers.
    return hashlib.sha256(memoryview(array.reshape(-1)).cast("B")).hexdigest()


def _source(obj):
    cls = type(obj)
    try:
        path = inspect.getfile(cls)
    except (TypeError, OSError):
        path = None
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except OSError:
        digest = None
    return {"class": cls.__module__ + "." + cls.__qualname__, "file": path, "sha256": digest}


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _weights(module):
    import torch

    result = {"source": _source(module), "training": module.training}
    for kind, iterator in (("parameters", module.named_parameters()), ("buffers", module.named_buffers())):
        values = {}
        for name, tensor in iterator:
            cpu = tensor.detach().to(device="cpu").contiguous()
            raw = cpu.reshape(-1).view(torch.uint8).numpy()
            canonical = cpu.float().numpy() if cpu.is_floating_point() else cpu.numpy()
            values[name] = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                            "sha256": _digest(raw), "values_sha256": _digest(canonical),
                            "requires_grad": tensor.requires_grad}
        result[kind] = values
    return result


class _Capture:
    def __init__(self, metadata):
        self.metadata = metadata
        self.arrays = {}
        self.metadata.update(tensors={}, scalars={})

    def record(self, path, value):
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            cpu = value.detach().to(device="cpu").contiguous()
            # BF16/FP16 values are exact in FP32; preserve FP64 so the capture
            # cannot hide a difference by rounding double-precision values.
            array = (cpu.float().numpy().copy()
                     if cpu.is_floating_point() and cpu.dtype != torch.float64 else cpu.numpy().copy())
            key = "tensor_%06d" % len(self.arrays)
            self.arrays[key] = array
            self.metadata["tensors"][path] = {
                "array_key": key, "shape": list(value.shape), "dtype": str(value.dtype),
                "stored_dtype": str(array.dtype), "device": str(value.device),
                "finite": bool(np.isfinite(array).all()),
            }
        elif isinstance(value, dict):
            for key, item in value.items():
                self.record(path + "/" + str(key), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.record(path + "/" + str(index), item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            self.metadata["scalars"][path] = value
        else:
            self.metadata["scalars"][path] = {"unsupported_type": type(value).__name__}

    def features(self, data):
        import numpy as np
        import torch

        raw, lens = data["audio_features"], data["audio_feature_lens"]
        chunks = list(raw.unbind(0)) if isinstance(raw, torch.Tensor) else list(raw)
        groups = list(lens.unbind(0)) if isinstance(lens, torch.Tensor) else list(lens)
        lengths = [torch.as_tensor(group).reshape(-1).tolist() for group in groups]
        if sum(map(len, lengths)) != len(chunks):
            raise ValueError("Huanyu probe: feature chunks and lengths disagree")
        self.record("features", chunks)
        self.record("feature_lens", lens)
        audio_groups, offset = [], 0
        for index, group in enumerate(lengths):
            identity = hashlib.sha256()
            paths = []
            if not group:
                raise ValueError("Huanyu probe: empty audio group")
            for length in group:
                chunk = chunks[offset]
                if int(length) != length or length <= 0 or length > chunk.shape[-1]:
                    raise ValueError("Huanyu probe: invalid audio feature length")
                array = chunk[..., :int(length)].detach().float().cpu().contiguous().numpy()
                if not np.isfinite(array).all():
                    raise ValueError("Huanyu probe: nonfinite audio feature")
                identity.update(json.dumps(list(array.shape)).encode("ascii") + b"\n")
                identity.update(memoryview(array).cast("B"))
                paths.append("features/%d" % offset)
                offset += 1
            audio_groups.append({"feature_id": identity.hexdigest(), "chunk_lengths": group,
                                 "feature_paths": paths, "output_path": "audio_embeddings/%d" % index})
        self.metadata["audio_groups"] = audio_groups

    def save(self, folder, stem):
        import numpy as np

        folder.mkdir(parents=True, exist_ok=True)
        if self.arrays:
            with (folder / (stem + ".npz")).open("xb") as output:
                np.savez(output, **self.arrays)
        self.metadata["status"] = "complete"
        with (folder / (stem + ".json")).open("x", encoding="utf-8") as output:
            json.dump(self.metadata, output, ensure_ascii=False, indent=2, allow_nan=False)
        return folder / (stem + ".json")


def attach_audio_probe(native):
    """Attach without reading weights; weights are captured at execution time."""
    directory = os.environ.get("HUANYU_AUDIO_PROBE_DIR")
    if not directory:
        return False
    thinker = getattr(native, "thinker", native)
    if getattr(thinker, "_huanyu_audio_probe_installed", False):
        return True
    maximum = int(os.environ.get("HUANYU_AUDIO_PROBE_MAX_CALLS", "2"))
    if maximum <= 0:
        raise ValueError("HUANYU_AUDIO_PROBE_MAX_CALLS must be positive")
    selected_feature = os.environ.get("HUANYU_AUDIO_PROBE_FEATURE_ID", "")
    if selected_feature and not re.fullmatch(r"[a-fA-F0-9]{64}", selected_feature):
        raise ValueError("HUANYU_AUDIO_PROBE_FEATURE_ID must be a SHA256 hex digest")
    selected_feature = selected_feature.lower()
    arm_file = os.environ.get("HUANYU_AUDIO_PROBE_ARM_FILE")
    original = getattr(thinker, "get_audio_hidden_states", None)
    modules = {"encoder": getattr(thinker, "apm", None),
               "projector": getattr(thinker, "audio_projection_layer", None)}
    if not callable(original) or any(not hasattr(module, "register_forward_hook") for module in modules.values()):
        raise RuntimeError("Huanyu probe requires get_audio_hidden_states, apm and audio_projection_layer")
    host = socket.gethostname()
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
    instance = re.sub(r"[^a-zA-Z0-9_.-]", "_", "%s-pid%d-rank%s" % (host, os.getpid(), rank))
    folder = Path(directory) / (instance + "-" + uuid.uuid4().hex[:12])
    captured, snapshots = 0, set()

    def metadata(phase):
        config = getattr(native, "_huanyu_audio_probe_config",
                         getattr(native, "vllm_config", getattr(native, "config", None)))
        model_config = _value(config, "model_config", config)
        parallel = _value(config, "parallel_config")
        multimodal = _value(model_config, "multimodal_config")
        hf = _value(model_config, "hf_config", model_config)
        audio_config = _value(hf, "audio_config", {})
        if hasattr(audio_config, "to_dict"):
            audio_config = audio_config.to_dict()
        return {
            "format": "huanyu_audio_probe_v1", "phase": phase, "host": host,
            "pid": os.getpid(), "rank": rank, "source": _source(thinker),
            "checkpoint": str(_value(model_config, "model", "unknown")),
            "audio_config": audio_config if isinstance(audio_config, dict) else str(audio_config),
            "encoder_downsample": getattr(modules["encoder"], "fwd_downsample", None),
            "projector_downsample": _value(audio_config, "projector_downsample_rate"),
            "tensor_parallel_size": _value(parallel, "tensor_parallel_size"),
            "enforce_eager": _value(model_config, "enforce_eager"),
            "mm_processor_cache_gb": _value(multimodal, "mm_processor_cache_gb"),
            "weight_reload_count": getattr(native, "_huanyu_audio_probe_reload_count", 0),
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith(("MINICPMO_", "HUANYU_"))},
            "weights": {name: _weights(module) for name, module in modules.items()},
        }

    def snapshot_weights(phase):
        if phase in snapshots:
            return
        if phase not in {"before_first_reload", "after_first_reload"}:
            raise ValueError("Unsupported Huanyu weight probe phase: " + str(phase))
        path = _Capture(metadata(phase)).save(folder, phase)
        snapshots.add(phase)
        print("[huanyu-audio-probe] phase=%s file=%s" % (phase, path), flush=True)

    @wraps(original)
    def audio(self, data, *args, **kwargs):
        nonlocal captured
        if captured >= maximum or (arm_file and not Path(arm_file).is_file()):
            return original(data, *args, **kwargs)
        call_index = captured + 1
        hooks = []
        token = None
        try:
            # Select by the actual valid features before reading ~0.5B audio
            # parameters. Unselected/dummy calls do not consume the limit.
            capture = _Capture({})
            capture.features(data)
            if not capture.metadata["audio_groups"] or (selected_feature and not any(
                    group["feature_id"] == selected_feature for group in capture.metadata["audio_groups"])):
                return original(data, *args, **kwargs)
            captured += 1
            capture.metadata.update(metadata("audio_forward"), call_index=call_index,
                                    hook_counts={name: 0 for name in modules})
            token = _ACTIVE.set(capture)
            for name, module in modules.items():
                counter = [0]

                def before(mod, inputs, kwinputs, name=name, counter=counter):
                    if _ACTIVE.get() is capture:
                        path = name + "/%d/input" % counter[0]
                        capture.record(path + "/args", inputs)
                        capture.record(path + "/kwargs", kwinputs)

                def after(mod, inputs, kwinputs, result, name=name, counter=counter):
                    if _ACTIVE.get() is capture:
                        capture.record(name + "/%d/output" % counter[0], result)
                        counter[0] += 1
                        capture.metadata["hook_counts"][name] = counter[0]

                hooks.append(module.register_forward_pre_hook(before, with_kwargs=True))
                hooks.append(module.register_forward_hook(after, with_kwargs=True))
            result = original(data, *args, **kwargs)
            if not isinstance(result, (list, tuple)) or len(result) != len(capture.metadata["audio_groups"]):
                raise RuntimeError("Huanyu probe: unexpected audio embedding group count/layout")
            import torch

            if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in result):
                raise RuntimeError("Huanyu probe: expected each audio embedding to be [tokens, hidden]")
            capture.record("audio_embeddings", result)
            path = capture.save(folder, "forward_%04d" % call_index)
            print("[huanyu-audio-probe] phase=executed call=%d file=%s" % (call_index, path), flush=True)
            return result
        except Exception as exc:
            print("[huanyu-audio-probe] phase=failed call=%d error=%s: %s" %
                  (call_index, type(exc).__name__, exc), flush=True)
            raise
        finally:
            for hook in hooks:
                hook.remove()
            if token is not None:
                _ACTIVE.reset(token)

    thinker.get_audio_hidden_states = MethodType(audio, thinker)
    thinker._huanyu_audio_probe_installed = True
    native._huanyu_audio_probe_snapshot_weights = snapshot_weights
    print("[huanyu-audio-probe] phase=installed max_calls=%d directory=%s" % (maximum, folder), flush=True)
    return True


def install_model_probe():
    """For standalone workers only; no precision/registration package imports."""
    if not os.environ.get("HUANYU_AUDIO_PROBE_DIR"):
        return False
    cls = importlib.import_module(_NATIVE).MiniCPMO50OmniForConditionalGeneration
    original = cls.__init__
    if getattr(original, "_huanyu_neutral_probe", False):
        return True

    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = kwargs.get("vllm_config", kwargs.get("config", args[0] if args else None))
        config = getattr(self, "vllm_config", config)
        model_config = _value(config, "model_config", config)
        hf = _value(model_config, "hf_config", getattr(self, "config", model_config))
        if _value(_value(hf, "audio_config"), "model_type") == "huanyu_conformer_0_6b":
            self._huanyu_audio_probe_config = config
            attach_audio_probe(self)

    init._huanyu_neutral_probe = True
    cls.__init__ = init
    print("[huanyu-audio-probe] phase=model_hook_installed neutral=True", flush=True)
    return True


class ProbeWorkerExtension:
    """Standalone vLLM worker extension; installation happens on module import."""


if __name__ == "huanyu_audio_probe" and os.environ.get("HUANYU_AUDIO_PROBE_DIR"):
    install_model_probe()
