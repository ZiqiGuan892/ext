"""Read local Huanyu metadata and safetensors headers without loading weights.

Run this file by path, not via ``python -m``: this avoids importing the ext
package and activating its existing process-wide patches during inspection.
"""

import argparse
from collections import Counter
import json
from pathlib import Path


PREFIXES = ("model.audio_encoder.", "model.audio_projection.")
METADATA_FILES = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "audio_config.json",
    "audio_preprocessor_config.json",
)


def read_tensor_headers(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        if length <= 0 or length > min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header length {length}: {path}")
        header = json.loads(stream.read(length))
    return {name: spec for name, spec in header.items() if name != "__metadata__"}


def inspect_checkpoint(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected a checkpoint directory: {root}")
    metadata = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        if (root / name).is_file() else None
        for name in METADATA_FILES
    }
    # An index lets us avoid opening language/vision-only shards. Read actual
    # tensor headers for shapes rather than treating index presence as proof.
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = {
            shard for key, shard in index["weight_map"].items()
            if key.startswith(PREFIXES)
        }
        shards = [root / name for name in sorted(names)]
    else:
        shards = sorted(root.glob("*.safetensors"))
    counts = Counter()
    encoder_samples, projection, seen = [], [], set()
    for shard in shards:
        resolved = shard.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"Shard resolves outside checkpoint directory: {shard}")
        for key, spec in sorted(read_tensor_headers(resolved).items()):
            if not key.startswith(PREFIXES):
                continue
            if key in seen:
                raise ValueError(f"Duplicate audio tensor in shards: {key}")
            seen.add(key)
            prefix = next(p for p in PREFIXES if key.startswith(p))
            counts[prefix] += 1
            item = {"key": key, "shape": spec["shape"], "dtype": spec["dtype"]}
            if prefix == PREFIXES[1]:
                if len(projection) < 20:
                    projection.append(item)
            elif len(encoder_samples) < 12:
                encoder_samples.append(item)
    return {
        "checkpoint": str(root),
        "metadata": metadata,
        "python_files": [str(p.relative_to(root)) for p in sorted(root.rglob("*.py"))],
        "audio_tensor_counts": dict(counts),
        "encoder_tensor_samples": encoder_samples,
        "projection_tensor_samples": projection,
        "shards_inspected": len(shards),
        "note": "Metadata inspection only; this does not establish Huanyu runtime support.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.model_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Huanyu checkpoint inspection failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
