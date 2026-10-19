"""Strict, content-keyed frozen Huanyu embeddings; independent of Whisper caches.

Entries contain final BF16 LLM-space tokens. Inputs are hashed before device/dtype
conversion, so training and precomputation share exactly the same CPU frontend.
Missing/corrupt entries are fatal: never silently run an encoder on one rank.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch


FORMAT = "huanyu_frozen_embeddings_v1"
_PUBLISH_LOCK_TIMEOUT = 30.0


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _digest_tensor(value):
    header = _json({"shape": list(value.shape), "dtype": str(value.dtype)}).encode()
    return hashlib.sha256(header + b"\0" + _tensor_bytes(value)).hexdigest()


def _cpu_float(value, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} cache keys require original CPU float32 frontend output")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} cache input contains NaN/Inf")
    return value.contiguous()


def audio_key(feature, length):
    feature = _cpu_float(feature, "audio")
    length = int(length)
    if feature.ndim != 2 or length <= 0 or length > feature.shape[1]:
        raise ValueError("Audio cache key requires [feature_bins,frames] and a valid real length")
    return _digest_tensor(feature[:, :length])


def vision_key(pixels, grid):
    pixels = _cpu_float(pixels, "vision")
    grid = torch.as_tensor(grid)
    if grid.device.type != "cpu" or grid.shape != (3,) or grid.is_floating_point() or torch.any(grid <= 0):
        raise ValueError("Vision cache key requires a positive integer CPU [t,h,w] grid")
    if pixels.ndim != 2 or pixels.shape[0] != int(grid.prod()):
        raise ValueError("Vision pixels must contain exactly prod(grid) patches")
    return hashlib.sha256((_digest_tensor(pixels) + _digest_tensor(grid.long())).encode()).hexdigest()


def split_vision_inputs(pixels, grids):
    """Split concatenated Qwen processor patches without changing their order."""
    pixels = _cpu_float(pixels, "vision")
    grids = torch.as_tensor(grids)
    if pixels.ndim != 2 or grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError("Vision cache expects pixel_values [patches,dim], grids [items,3]")
    if grids.device.type != "cpu" or grids.is_floating_point() or torch.any(grids <= 0):
        raise ValueError("Vision grids must be positive CPU integers")
    sizes = [int(grid.prod()) for grid in grids]
    if sum(sizes) != pixels.shape[0]:
        raise ValueError("Vision grid patch counts do not match pixel_values")
    return list(zip(pixels.split(sizes), grids, strict=True))


def _file_identity(path):
    path = Path(path)
    return {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def make_cache_contract(model_path, processor):
    """Describe the checkpoint and actual frontend/encoder implementation.

    Shards use size + nanosecond mtime and header hash instead of reading the
    entire 35B model. Keep a checkpoint immutable; copied mounts must preserve
    metadata, or build a new cache contract for that checkpoint installation.
    """
    from .runtime_imports import load_huanyu_native

    root = Path(model_path).resolve(strict=True)
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    metadata = {}
    for name in ("config.json", "audio_preprocessor_config.json", "preprocessor_config.json",
                 "processor_config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            metadata[name] = _file_identity(path)
    shards = {}
    for path in sorted(root.glob("*.safetensors")):
        stat = path.stat()
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            size = int.from_bytes(raw, "little")
            if size <= 0 or size > min(stat.st_size - 8, 128 * 1024 * 1024):
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = handle.read(size)
        shards[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "header_sha256": hashlib.sha256(header).hexdigest()}
    if not shards:
        raise ValueError(f"No safetensors checkpoint shards in {root}")
    native = load_huanyu_native()
    hf = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    llm = importlib.import_module("vllm_omni.model_executor.models.minicpmo_5_0.minicpmo_5_0_omni_llm")
    sources = {"native_huanyu": _file_identity(native.__file__),
               "native_projector": _file_identity(llm.__file__), "hf_qwen35": _file_identity(hf.__file__)}
    for name in ("processor.py", "audio_runtime.py", "vision_attention.py"):
        sources[name] = _file_identity(Path(__file__).parent / name)
    image_cfg = processor.image_processor.to_dict()
    # JSON roundtrip rejects opaque runtime objects rather than stringifying them.
    return json.loads(_json({"format": FORMAT, "model_path": str(root), "metadata": metadata,
                           "shards": shards, "sources": sources,
                           "audio_preprocessor": processor.audio_preprocessor,
                           "image_processor": image_cfg, "audio_execution": "single_chunk_bf16_v1",
                           "vision_execution": "single_image_hf_bf16_eager_q128_v1"}))


def _publish(path, writer, validate_new=None):
    """Publish a complete file, serializing cooperating writers without hard links.

    The directory lock covers the existence check and same-directory rename.
    All concurrent cache writers must use this protocol. A crashed owner's lock
    is never automatically stolen: an operator must first rule out live writers.
    """
    lock = path.with_name(f".huanyu-{path.name}.lock")
    fd, temporary = tempfile.mkstemp(prefix=".huanyu-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + _PUBLISH_LOCK_TIMEOUT
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if os.path.lexists(path):
                    return False
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Huanyu cache publication lock timed out: {lock}; "
                        "check for active precompute writers before removing a leftover lock"
                    ) from None
                time.sleep(0.05)
        try:
            if os.path.lexists(path):
                return False
            if validate_new is not None:
                validate_new()
            try:
                os.rename(temporary, path)
            except FileExistsError:
                return False
            return True
        finally:
            lock.rmdir()  # Only the process that successfully created it releases it.
    finally:
        Path(temporary).unlink(missing_ok=True)


class HuanyuEmbeddingCache:
    def __init__(self, root, contract, create=False):
        self.root = Path(root).resolve()
        self.contract = json.loads(_json(contract))
        self.contract_digest = hashlib.sha256(_json(self.contract).encode()).hexdigest()
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                def validate_new():
                    # Run under the manifest lock: another initializer may have
                    # published a manifest since our initial existence check.
                    if any((path.name in ("audio", "vision") and
                            (not path.is_dir() or any(path.iterdir()))) or
                           (path.name not in ("audio", "vision") and not path.name.startswith(".huanyu-"))
                           for path in self.root.iterdir()):
                        raise ValueError(f"Refusing to adopt an existing non-Huanyu cache: {self.root}")

                payload = {"format": FORMAT, "contract_digest": self.contract_digest, "contract": self.contract}
                _publish(manifest, lambda handle: handle.write((_json(payload) + "\n").encode()), validate_new)
            for modality in ("audio", "vision"):
                (self.root / modality).mkdir(exist_ok=True)
        if not manifest.is_file():
            raise FileNotFoundError(f"Huanyu cache manifest missing: {manifest}; precompute first")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload != {"format": FORMAT, "contract_digest": self.contract_digest, "contract": self.contract}:
            raise ValueError("Huanyu cache contract differs from checkpoint/frontend/runtime; use a new cache directory")

    def path(self, modality, key):
        if modality not in ("audio", "vision") or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid Huanyu cache modality or digest")
        return self.root / modality / f"{key}.pt"

    def contains(self, modality, key):
        return self.path(modality, key).is_file()

    @staticmethod
    def _validate(embedding, expected_tokens, hidden_size):
        if not isinstance(embedding, torch.Tensor) or embedding.device.type != "cpu":
            raise ValueError("Cached embedding must be a CPU Tensor")
        if embedding.dtype != torch.bfloat16 or tuple(embedding.shape) != (int(expected_tokens), int(hidden_size)):
            raise ValueError(f"Huanyu cached embedding shape/dtype mismatch: {embedding.shape}/{embedding.dtype}")
        if int(expected_tokens) <= 0 or int(hidden_size) <= 0 or not torch.isfinite(embedding).all():
            raise ValueError("Huanyu cached embedding is empty or contains NaN/Inf")

    def get(self, modality, key, expected_tokens, hidden_size):
        path = self.path(modality, key)
        if not path.is_file():
            raise FileNotFoundError(f"Huanyu {modality} cache miss: {path}; no online fallback is allowed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or payload.get("format") != FORMAT
                or payload.get("contract_digest") != self.contract_digest
                or payload.get("key") != key or payload.get("modality") != modality):
            raise ValueError(f"Invalid Huanyu cache entry metadata: {path}")
        embedding = payload.get("embedding")
        self._validate(embedding, expected_tokens, hidden_size)
        if payload.get("tensor_sha256") != _digest_tensor(embedding):
            raise ValueError(f"Huanyu cache tensor checksum mismatch: {path}")
        return embedding

    def put(self, modality, key, embedding, expected_tokens, hidden_size):
        path = self.path(modality, key)
        value = embedding.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        self._validate(value, expected_tokens, hidden_size)
        payload = {"format": FORMAT, "contract_digest": self.contract_digest, "modality": modality,
                   "key": key, "tensor_sha256": _digest_tensor(value), "embedding": value}
        if not _publish(path, lambda handle: torch.save(payload, handle)):
            existing = self.get(modality, key, expected_tokens, hidden_size)
            if _digest_tensor(existing) != payload["tensor_sha256"]:
                raise ValueError(f"Existing cache differs for identical input: {path}; refusing overwrite")
        return path
