# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Select a DEBUG subset from a completed omni_mix_4b conversion.

Keep the original train/validation membership and absolute media paths, allowing
reuse of the full dataset's embedding cache. Write only to a NEW directory.
Use per-data_source sampling to cover all retained datasets. The legacy
first-N-shards selection remains available but does not ensure source coverage.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def select_shards(dataset_dir: str | Path, output_dir: str | Path, num_shards: int = 10) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError("num_shards must be a positive integer")
    dataset = Path(dataset_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("ground_truth_source") != "answer.solution":
        raise ValueError("Expected a complete omni_mix_4b reference-answer dataset")
    candidates = sorted(item["path"] for item in manifest["inputs"] if item.get("kept_rows", item["rows"]) > 0)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate source paths in input manifest")
    if num_shards > len(candidates):
        raise ValueError(f"Requested {num_shards} shards, but only {len(candidates)} retained data")
    selected = candidates[:num_shards]
    tables = {}
    seen = set()
    for split in ("train", "validation"):
        table = pq.read_table(dataset / f"{split}.parquet")
        if table.num_rows != manifest["splits"][split]["rows"]:
            raise ValueError(f"{split} row count disagrees with the completed manifest")
        source_files = pc.struct_field(table["extra_info"], "source_file")
        subset = table.filter(pc.is_in(source_files, value_set=pa.array(selected)))
        if not subset.num_rows:
            raise ValueError(f"Selected shards have an empty {split} split; select more shards")
        seen.update(pc.struct_field(subset["extra_info"], "source_file").to_pylist())
        tables[split] = subset
    if seen != set(selected):
        raise ValueError("Some selected source shards have no rows in the converted files")
    result = {
        "status": "complete",
        "selector": "select_omni_mix_4b_shards_v1",
        "source_dataset": str(dataset),
        "selection": "first N sorted source paths with retained rows",
        "selected_source_files": selected,
        "source_shards": len(selected),
        "total_rows": sum(table.num_rows for table in tables.values()),
        "splits": {split: {"rows": table.num_rows} for split, table in tables.items()},
        "media_and_cache": "Original absolute media paths preserved; reuse source dataset embedding cache",
    }
    output.mkdir(parents=True, exist_ok=False)
    for split, table in tables.items():
        with (output / f"{split}.parquet").open("xb") as handle:
            pq.write_table(table, handle, use_dictionary=False)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def select_sources(
    dataset_dir: str | Path,
    output_dir: str | Path,
    train_per_source: int = 8,
    val_per_source: int = 2,
    seed: int = 42,
    expected_sources: int | None = None,
) -> dict:
    """Sample each retained source in BOTH splits, or fail without writing output.

    Never copy training rows into validation to fill missing source coverage.
    Exact quotas make DEBUG size predictable and avoid silently omitting a rare
    source. All selected rows and their media paths are kept unchanged.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for name, value in (("train_per_source", train_per_source), ("val_per_source", val_per_source)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    dataset = Path(dataset_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("ground_truth_source") != "answer.solution":
        raise ValueError("Expected a complete omni_mix_4b reference-answer dataset")
    sources = sorted(set().union(*(stats["data_sources"] for stats in manifest["splits"].values())))
    if not sources:
        raise ValueError("The dataset has no retained data sources")
    if expected_sources is not None and len(sources) != expected_sources:
        raise ValueError(f"Expected {expected_sources} retained data sources, found {len(sources)}: {sources}")
    tables, split_stats = {}, {}
    rng = random.Random(seed)
    selected_files = set()
    for split, quota in (("train", train_per_source), ("validation", val_per_source)):
        table = pq.read_table(dataset / f"{split}.parquet")
        column = table["data_source"].to_pylist()
        counts = dict(Counter(column))
        if table.num_rows != manifest["splits"][split]["rows"] or counts != manifest["splits"][split]["data_sources"]:
            raise ValueError(f"{split} row/source counts disagree with the completed manifest")
        positions = defaultdict(list)
        for index, source in enumerate(column):
            positions[source].append(index)
        shortages = {source: len(positions[source]) for source in sources if len(positions[source]) < quota}
        if shortages:
            raise ValueError(
                f"{split} needs {quota} rows per source; insufficient rows: {shortages}. "
                "Lower the quota if possible; do not move rows across train/validation."
            )
        indices = [index for source in sources for index in rng.sample(positions[source], quota)]
        subset = table.take(pa.array(indices, type=pa.int64()))
        selected_files.update(pc.struct_field(subset["extra_info"], "source_file").to_pylist())
        tables[split] = subset
        split_stats[split] = {
            "rows": subset.num_rows,
            "data_sources": dict(Counter(subset["data_source"].to_pylist())),
            "reward_types": dict(Counter(pc.struct_field(subset["extra_info"], "source_reward_type").to_pylist())),
            "modalities": dict(Counter(subset["ability"].to_pylist())),
        }
    result = {
        "status": "complete",
        "selector": "select_omni_mix_4b_sources_v1",
        "source_dataset": str(dataset),
        "selection": "seeded fixed quota per retained data_source in each original split",
        "seed": seed,
        "train_per_source": train_per_source,
        "val_per_source": val_per_source,
        "data_sources": sources,
        "source_count": len(sources),
        "source_shards": len(selected_files),
        "selected_source_files": sorted(selected_files),
        "total_rows": sum(table.num_rows for table in tables.values()),
        "splits": split_stats,
        "skipped_by_data_source_in_full_conversion": manifest.get("skipped_by_data_source", {}),
        "media_and_cache": "Original absolute media paths preserved; reuse source dataset embedding cache",
    }
    output.mkdir(parents=True, exist_ok=False)
    for split, table in tables.items():
        with (output / f"{split}.parquet").open("xb") as handle:
            pq.write_table(table, handle, use_dictionary=False)
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Completed full conversion directory")
    parser.add_argument("--output-dir", required=True, help="New DEBUG subset directory; must not exist")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--num-shards", type=int, help="Legacy first-N-shards mode; no source coverage guarantee")
    selection.add_argument(
        "--train-per-source", type=int, help="Cover each retained data_source with this many train rows"
    )
    parser.add_argument("--val-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sources", type=int, help="Expected retained data_source count, e.g. 28")
    args = parser.parse_args()
    try:
        if args.num_shards is not None:
            result = select_shards(args.dataset_dir, args.output_dir, args.num_shards)
        else:
            result = select_sources(
                args.dataset_dir,
                args.output_dir,
                args.train_per_source,
                args.val_per_source,
                args.seed,
                args.expected_sources,
            )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Selection failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
