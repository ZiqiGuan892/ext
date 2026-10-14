"""Read checkpoint tensor names and shapes without loading weights or model code.

Only Safetensors headers are inspected. This does not validate tensor contents,
architecture compatibility, or whether a loader marks each parameter as loaded.
"""

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


VISION_COMPONENTS = {"visual", "vpm", "resampler", "vit_merger", "vision_model", "vision_tower"}


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated Safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid Safetensors header size {size}: {path}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def inspect_checkpoint(directory: Path, sample_keys: int = 5) -> dict:
    directory = directory.resolve(strict=True)
    index_path = directory / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Empty or invalid weight_map: {index_path}")
        names = sorted(set(weight_map.values()))
        files = [directory / name for name in names]
    elif (directory / "model.safetensors").is_file():
        files = [directory / "model.safetensors"]
    else:
        files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No Safetensors files found: {directory}")

    tensors = {}
    for path in files:
        for key, metadata in read_header(path).items():
            if key in tensors:
                raise ValueError(f"Duplicate tensor key {key!r} in {path}")
            tensors[key] = {"file": str(path.relative_to(directory)), **metadata}
    if weight_map is not None:
        missing = [key for key, name in weight_map.items() if tensors.get(key, {}).get("file") != name]
        if missing:
            raise ValueError(f"Index keys missing from declared shards ({len(missing)}): {missing[:5]}")

    vision = {}
    for key, metadata in sorted(tensors.items()):
        parts = key.split(".")
        for position, component in enumerate(parts):
            if component not in VISION_COMPONENTS:
                continue
            prefix = ".".join(parts[: position + 1]) + "."
            group = vision.setdefault(prefix, {"key_count": 0, "samples": []})
            group["key_count"] += 1
            if len(group["samples"]) < sample_keys:
                group["samples"].append({"key": key, "shape": metadata["shape"], "dtype": metadata["dtype"]})
            break

    return {
        "checkpoint": str(directory),
        "inspection": "safetensors_headers_only_not_model_compatibility_validation",
        "index": str(index_path) if weight_map is not None else None,
        "files": [str(path.relative_to(directory)) for path in files],
        "tensor_count": len(tensors),
        "top_level_prefix_counts": dict(sorted(Counter(key.split(".")[0] for key in tensors).items())),
        "vision_prefixes": vision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--q36-path", type=Path, help="Optional actual auxiliary checkpoint directory")
    parser.add_argument("--sample-keys", type=int, default=5)
    args = parser.parse_args()
    if args.sample_keys < 1:
        parser.error("--sample-keys must be positive")
    try:
        result = {"model": inspect_checkpoint(args.model_path, args.sample_keys)}
        if args.q36_path is not None:
            result["q36"] = inspect_checkpoint(args.q36_path, args.sample_keys)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Checkpoint inspection failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
