"""Precompute frozen Huanyu audio + Qwen vision embeddings without loading the LLM.

Run this file directly after sourcing the server environment. Existing items are
validated and reused, while a changed checkpoint/frontend requires a new cache
directory. Cache misses never fall back online when --verify-only is selected.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import torch


# Direct execution must not import the ext package's global registration hooks.
if not __package__:
    _package = "_huanyu_precompute_cli"
    _module = ModuleType(_package)
    _module.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package] = _module
    __package__ = _package

from .cache import HuanyuEmbeddingCache, audio_key, make_cache_contract, split_vision_inputs, vision_key


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def load_vision_weights(model_path):
    """Read only model.visual.* tensors, never allocate checkpoint LLM weights."""
    from safetensors import safe_open

    root = Path(model_path).resolve()
    index = root / "model.safetensors.index.json"
    expected = None
    prefix = "model.visual."
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        expected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
        files = sorted(set(expected.values()))
    else:
        files = sorted(path.name for path in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No model.visual.* checkpoint tensors in {root}")
    result = {}
    for name in files:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Vision shard escapes model directory: {name}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith(prefix):
                    continue
                short = key[len(prefix):]
                if short in result or (expected is not None and expected.get(key) != name):
                    raise ValueError(f"Vision tensor/index duplicate or mismatch: {key}")
                result[short] = handle.get_tensor(key)
    if not result or (expected is not None and {prefix + name for name in result} != set(expected)):
        raise ValueError("Vision checkpoint tensors are missing; no partial load is allowed")
    return result


def vision_output_tokens(output, expected_tokens, hidden_size):
    """Accept only the post-merger [tokens,LLM hidden] output, never patch features."""
    deepstack = getattr(output, "deepstack_features", None)
    if deepstack is not None and len(deepstack):
        raise ValueError("Huanyu cache does not support vision DeepStack feature injection")
    if isinstance(output, torch.Tensor):
        result = output
    elif getattr(output, "pooler_output", None) is not None:
        result = output.pooler_output
    elif getattr(output, "last_hidden_state", None) is not None:
        result = output.last_hidden_state
    elif isinstance(output, (tuple, list)) and output:
        result = output[0]
        if len(output) > 1 and output[1] is not None and len(output[1]):
            raise ValueError("Huanyu cache received additional vision features; DeepStack must be empty")
    else:
        raise TypeError(f"Unsupported HF vision output: {type(output).__name__}")
    if (not isinstance(result, torch.Tensor)
            or tuple(result.shape) != (int(expected_tokens), int(hidden_size))
            or not torch.isfinite(result).all()):
        raise ValueError("HF vision output is not finite post-merger [tokens,text_hidden] embeddings")
    return result


class FrozenEncoders:
    """Load each frozen tower lazily, leaving the 35B text backbone unloaded."""

    def __init__(self, model_path, processor, device):
        self.model_path = model_path
        self.processor = processor
        if str(device).startswith("npu"):
            import torch_npu  # noqa: F401
        self.device = torch.device(device)
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
        if self.device.type not in ("npu", "cuda", "cpu"):
            raise ValueError(f"Unsupported precompute device: {self.device}")
        self.audio_model = None
        self.vision_model = None
        self.hidden_size = int(_field(_field(processor.config, "text_config"), "hidden_size"))

    def audio(self, feature, length):
        from .audio_runtime import attach_huanyu_audio, encode_huanyu_audio

        if self.audio_model is None:
            model = torch.nn.Module()
            model.model = torch.nn.Module()
            model.config = self.processor.config
            attach_huanyu_audio(model, SimpleNamespace(path=self.model_path))
            self.audio_model = model.to(device=self.device).eval()
        # The online Huanyu bridge also encodes chunks independently. Real length
        # controls padding and folding; neighboring samples cannot change output.
        length = int(length)
        feature = feature[:, :length]
        padded = torch.nn.functional.pad(feature, (0, -length % 4)).unsqueeze(0)
        return encode_huanyu_audio(self.audio_model, padded, torch.tensor([length], dtype=torch.long))[0]

    def vision(self, pixels, grid):
        if self.vision_model is None:
            hf = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
            cls = next((getattr(hf, name) for name in (
                "Qwen3_5MoeVisionModel", "Qwen3_5MoeVisionTransformerPretrainedModel"
            ) if hasattr(hf, name)), None)
            if cls is None:
                raise RuntimeError("Installed Transformers has no supported Qwen3.5 MoE vision class")
            vision_config = _field(self.processor.config, "vision_config")
            if _field(vision_config, "deepstack_visual_indexes", []):
                raise ValueError("Huanyu offline embeddings require an empty DeepStack configuration")
            vision_config._attn_implementation = "eager"
            with torch.device("cpu"):
                model = cls(vision_config)
            states = load_vision_weights(self.model_path)
            expected = model.state_dict()
            if set(states) != set(expected):
                raise ValueError(f"Vision state mismatch: missing={sorted(set(expected) - set(states))[:5]} "
                                 f"unexpected={sorted(set(states) - set(expected))[:5]}")
            for key, value in states.items():
                if value.shape != expected[key].shape:
                    raise ValueError(f"Vision checkpoint shape mismatch for {key}: {value.shape} != {expected[key].shape}")
            model.load_state_dict({key: value.bfloat16() if value.is_floating_point() else value
                                   for key, value in states.items()}, strict=True, assign=True)
            model.requires_grad_(False)
            self.vision_model = model.to(device=self.device).eval()
            print(f"[huanyu-precompute] loaded frozen vision only: {len(states)} tensors; {cls.__name__}", flush=True)
        merge = self.processor.spatial_merge_size ** 2
        count = int(grid.prod()) // merge
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            output = self.vision_model(pixels.to(device=self.device, dtype=torch.bfloat16),
                                       grid_thw=grid.to(device=self.device).reshape(1, 3))
        return vision_output_tokens(output, count, self.hidden_size)


def process_row(row, processor, cache, encoders=None, *, verify_only=False, sampling_rate=16000):
    """Populate/verify one parquet row with exactly the training media decoder."""
    from .dataset import decode_huanyu_media

    images, audios = decode_huanyu_media(row, sampling_rate=sampling_rate)
    vision = processor._vision(images)
    features, lengths, _ = processor._audio(audios, sampling_rate)
    hidden = int(_field(_field(processor.config, "text_config"), "hidden_size"))
    counts = {"audio": 0, "vision": 0, "created": 0, "reused": 0}
    items = []
    for feature, length in zip(features, lengths.tolist(), strict=True):
        tokens = ((length + 3) // 4) // 2
        items.append(("audio", audio_key(feature, length), tokens, feature, length))
    if vision:
        for pixels, grid in split_vision_inputs(vision["pixel_values"], vision["image_grid_thw"]):
            patches = int(grid.prod())
            if patches % processor.spatial_merge_size ** 2:
                raise ValueError("Vision grid cannot be divided by spatial merge size")
            items.append(("vision", vision_key(pixels, grid), patches // processor.spatial_merge_size ** 2, pixels, grid))
    for modality, key, tokens, feature, size in items:
        counts[modality] += 1
        if verify_only or cache.contains(modality, key):
            cache.get(modality, key, tokens, hidden)
            counts["reused"] += 1
            continue
        if encoders is None:
            raise ValueError("Encoder instance is required to create missing cache entries")
        output = getattr(encoders, modality)(feature, size)
        cache.put(modality, key, output, tokens, hidden)
        counts["created"] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", required=True, help="One or more existing GenRM OmniMix parquet files")
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--cache_dir", "--cache-dir", required=True)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--read-batch-size", type=int, default=16)
    parser.add_argument("--log-interval", type=float, default=10.0)
    parser.add_argument("--verify-only", action="store_true", help="Verify all row embeddings; never run an encoder")
    args = parser.parse_args(argv)
    if args.read_batch_size < 1 or args.log_interval <= 0 or args.sampling_rate != 16000:
        parser.error("Use a positive read batch/log interval and sampling_rate=16000")

    import pyarrow.parquet as pq
    from .processor import build_huanyu_processor

    inputs = [Path(value).resolve(strict=True) for value in args.parquet]
    processor = build_huanyu_processor(args.model_path, SimpleNamespace(trust_remote_code=True),
                                      install_bridge=False, use_cache=False)
    contract = make_cache_contract(args.model_path, processor)
    cache = HuanyuEmbeddingCache(args.cache_dir, contract, create=not args.verify_only)
    encoders = None if args.verify_only else FrozenEncoders(args.model_path, processor, args.device)
    started = last_log = time.monotonic()
    totals = {"rows": 0, "audio": 0, "vision": 0, "created": 0, "reused": 0}
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in inputs)
    print(f"[huanyu-precompute] mode={'verify' if args.verify_only else 'populate'} rows={total_rows} "
          f"device={args.device} cache={cache.root} contract={cache.contract_digest}", flush=True)
    for path in inputs:
        row_index = 0
        parquet = pq.ParquetFile(path)
        print(f"[huanyu-precompute] start {path} rows={parquet.metadata.num_rows}", flush=True)
        try:
            for batch in parquet.iter_batches(batch_size=args.read_batch_size):
                for row in batch.to_pylist():
                    counts = process_row(row, processor, cache, encoders, verify_only=args.verify_only,
                                         sampling_rate=args.sampling_rate)
                    row_index += 1
                    totals["rows"] += 1
                    for key, value in counts.items():
                        totals[key] += value
                    now = time.monotonic()
                    if now - last_log >= args.log_interval:
                        elapsed = now - started
                        rate = totals["rows"] / max(elapsed, 1e-9)
                        eta = (total_rows - totals["rows"]) / max(rate, 1e-9)
                        print(f"[huanyu-precompute] {totals['rows']}/{total_rows} rows "
                              f"audio_chunks={totals['audio']} images={totals['vision']} "
                              f"created={totals['created']} reused={totals['reused']} "
                              f"elapsed={elapsed:.0f}s rate={rate:.2f}rows/s eta={eta:.0f}s", flush=True)
                        last_log = now
        except Exception as exc:
            raise RuntimeError(f"Huanyu precompute failed at {path}, row {row_index} (zero-based): {exc}. "
                               "Validated complete entries remain reusable; rerun the same command to resume.") from exc
    print(f"[huanyu-precompute] COMPLETE {json.dumps(totals, sort_keys=True)} "
          f"elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
