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
"""Convert OC-RL reference-answer Parquet shards to MiniCPM-o GenRM input.

Input: problem, answer (JSON), data_source, images (bytes), reward_type.
The answer.solution value becomes reward_model.ground_truth. Rows without the
solution key are skipped and audited. Present values must be nonempty strings
or integers; malformed answers and invalid present values still fail conversion.
With a valid reference answer, rows whose problem is null, blank, or not a string
are skipped and audited rather than being given an invented prompt.
Rows containing an MPO image are also skipped in their entirety and audited;
other unsupported formats, animated images, and corrupt images still fail.
Original reward_type values are provenance, not reward dispatch instructions:
training uses the existing reference-answer GenRM scorer for all accepted rows.
Original image bytes are validated and materialized in a NEW
output directory, so the existing path-based embedding cache can consume them.
No model imports, training, or embedding computation take place here.

Example (run on the machine where final media paths will be accessible)::

    python3 verl_omni_ext/examples/omni_mix_4b.py \
        --input '/path/to/OC-RL/cleaned_rl_dataset_v3/data/train-*-of-00199.parquet' \
        --expected-shards 199 --output-dir /path/to/NEW/oc_rl_4b_genrm_v1

Inputs must be passed explicitly; do not glob a directory containing both raw
shards and existing RL Parquet. Existing output directories are always refused.
A successful conversion writes manifest.json last; a failed conversion may
leave extracted images in its newly created directory, without a success manifest.
Progress and stage logs are flushed to stderr (including when piped through tee).
The final JSON summary stays on stdout. --log-interval controls progress frequency
in seconds; each shard also reports on completion. ETA covers row conversion only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

SYSTEM_PROMPT = "你是一个多模态助手。请结合图像、音频、视频等信息分析问题，给出准确、详细的回答。"
REQUIRED_COLUMNS = ("problem", "answer", "data_source", "images", "reward_type")
IMAGE_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif"}
IMAGE_MARKER = re.compile(r"<image(?:_([^>]*))?>")


class MissingSolutionError(ValueError):
    """A valid answer object lacks the reference-answer key; skip this row only."""


class InvalidProblemError(ValueError):
    """No usable question text; skip the row and retain the precise failure category."""

    def __init__(self, value):
        self.problem_type = type(value).__name__
        if value is None:
            self.problem_issue = "missing_or_null"
        elif not isinstance(value, str):
            self.problem_issue = "non_string"
        else:
            self.problem_issue = "empty_or_whitespace"
        super().__init__(
            f"problem must be a non-empty string (issue={self.problem_issue}, type={self.problem_type}); "
            "skip entire row"
        )


class MPOImageError(ValueError):
    """An embedded image is MPO; skip the entire row without writing its media."""


def _log(stage: str, message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] [{stage}] {message}", file=sys.stderr, flush=True)


def _duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ConversionProgress:
    """Line-oriented progress, readable in a terminal, tee log, or redirected file."""

    def __init__(self, total: int, shards: int, interval: float):
        self.total = total
        self.shards = shards
        self.interval = interval
        self.started = self.last_report = time.monotonic()

    def report(self, kept, skipped, images, shard, source_row, shard_rows, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_report < self.interval:
            return False
        processed = kept + skipped
        elapsed = now - self.started
        rate = processed / elapsed if elapsed > 0 else 0
        fraction = min(processed / self.total, 1.0) if self.total else 0.0
        filled = int(fraction * 20)
        bar = "#" * filled + "-" * (20 - filled)
        eta = _duration((self.total - processed) / rate) if rate > 0 else "unknown"
        _log(
            "convert",
            f"[{bar}] {fraction:.1%} rows={processed:,}/{self.total:,} "
            f"kept={kept:,} skipped={skipped:,} images={images:,} "
            f"shard={shard}/{self.shards} shard_rows={source_row:,}/{shard_rows:,} "
            f"elapsed={_duration(elapsed)} rate={rate:.1f} rows/s ETA_rows={eta}",
        )
        self.last_report = now
        return True


def resolve_inputs(patterns: list[str]) -> list[Path]:
    """Expand explicit files/globs without accidentally taking other Parquet files."""
    paths = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        matches = glob.glob(expanded)
        if not matches:
            raise ValueError(f"No input files match: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix.lower() != ".parquet":
                raise ValueError(f"Expected a Parquet file: {path}")
            paths.add(path)
    if not paths:
        raise ValueError("At least one input shard is required")
    return sorted(paths)


def _required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _image_bytes(entry: dict) -> tuple[bytes, dict]:
    if not isinstance(entry, dict) or not isinstance(entry.get("bytes"), (bytes, bytearray, memoryview)):
        raise ValueError("images entries require actual binary bytes, not a printed bytes string or an old path")
    payload = bytes(entry["bytes"])
    try:
        with Image.open(BytesIO(payload)) as image:
            image_format = image.format
            if image_format == "MPO":
                raise MPOImageError(f"MPO image detected (frames={getattr(image, 'n_frames', 1)}); skip entire row")
            if image_format not in IMAGE_EXTENSIONS or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Unsupported image format or animated image: {image_format}")
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            width, height = rgb.size
            # Shared pixels stay in one split even when PNG metadata differs.
            pixel_hash = hashlib.sha256(f"RGB:{width}:{height}:".encode() + rgb.tobytes()).hexdigest()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image bytes cannot be decoded: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    return payload, {
        "sha256": digest,
        "pixel_sha256": pixel_hash,
        "filename": digest + IMAGE_EXTENSIONS[image_format],
        "size_bytes": len(payload),
        "width": width,
        "height": height,
    }


def _normalize_image_references(problem: str, image_count: int, single_image_one_based: bool):
    """Map every image marker to its original media index, preserving repeated references."""
    markers = list(IMAGE_MARKER.finditer(problem))
    if not markers:
        prefix = "<image>" * image_count + "\n" if image_count else ""
        return prefix + problem, list(range(image_count))
    numbered = [marker.group(1) is not None for marker in markers]
    if any(numbered) and not all(numbered):
        raise ValueError("Mixed numbered and unnumbered image placeholders need an explicit source mapping")
    if not any(numbered):
        if len(markers) != image_count:
            raise ValueError(f"Image placeholder count {len(markers)} does not match image count {image_count}")
        return problem, list(range(image_count))
    indices = []
    for marker in markers:
        value = marker.group(1)
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"Invalid image placeholder: {marker.group(0)}")
        reference = int(value)
        if single_image_one_based and image_count == 1 and reference == 1:
            reference = 0
        if reference >= image_count:
            raise ValueError(
                f"Image placeholder {marker.group(0)} is out of range for image count {image_count}; "
                "single-image <image_1> requires --single-image-one-based"
            )
        indices.append(reference)
    if set(indices) != set(range(image_count)):
        raise ValueError("Image placeholders do not reference every embedded image")
    return IMAGE_MARKER.sub("<image>", problem), indices


def build_row(
    raw: dict,
    *,
    source_file: str,
    source_row: int,
    index: int,
    image_prefix: str,
    single_image_one_based: bool = False,
):
    """Validate a row before writing its images. Return (RL row, media payloads)."""
    answer_json = _required_text(raw.get("answer"), "answer JSON")
    try:
        answer = json.loads(answer_json)
    except json.JSONDecodeError as exc:
        raise ValueError("answer must contain valid JSON") from exc
    if not isinstance(answer, dict):
        raise ValueError("answer JSON must be an object")
    if "solution" not in answer:
        raise MissingSolutionError(
            "answer.solution is missing; a reference answer is required for GenRM, including text-only rows"
        )
    data_source = _required_text(raw.get("data_source"), "data_source")
    reward_type = _required_text(raw.get("reward_type"), "reward_type")
    solution = answer["solution"]
    if isinstance(solution, int) and not isinstance(solution, bool):
        solution = str(solution)
    ground_truth = _required_text(solution, "answer.solution")
    problem = raw.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise InvalidProblemError(problem)
    sub_dataset = answer.get("sub_dataset")
    if sub_dataset is not None and not isinstance(sub_dataset, str):
        raise ValueError("answer.sub_dataset must be a string when present")
    orig_index = answer.get("orig_index")
    if orig_index is not None and (isinstance(orig_index, bool) or not isinstance(orig_index, (int, str))):
        raise ValueError("answer.orig_index must be an integer or string when present")
    if "<audio>" in problem or "<video>" in problem:
        raise ValueError("Audio/video placeholders require a separate media adapter")
    entries = raw.get("images")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise ValueError("images must be a list of binary image objects or null")
    metadata_images = answer.get("images")
    has_historical_paths = isinstance(metadata_images, list) and any(
        isinstance(path, str) and path.strip() for path in metadata_images
    )
    if has_historical_paths and not entries:
        raise ValueError("answer metadata declares images but the row is missing embedded image bytes")
    user_prompt, image_indices = _normalize_image_references(problem, len(entries), single_image_one_based)
    original_media = [_image_bytes(entry) for entry in entries]
    media = [original_media[i] for i in image_indices]
    modality = "image" if media else "text"
    identity = hashlib.sha256(f"{source_file}\0{source_row}".encode()).hexdigest()
    row = {
        "data_source": data_source,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        "images": [{"image": image_prefix + "/" + info["filename"]} for _, info in media],
        "videos": [],
        "audios": [],
        "ability": modality,
        "reward_model": {"style": "model", "ground_truth": ground_truth},
        "extra_info": {
            "split": "",
            "index": index,
            "problem_id": "oc_rl_4b_" + identity,
            "dataset": "OC-RL-4B",
            "modality": modality,
            "raw_source": data_source,
            "raw_answer": ground_truth,
            "question": problem,
            "source_file": source_file,
            "source_row": source_row,
            "source_reward_type": reward_type,
            "reward_type": reward_type,
            "source_answer_json": answer_json,
            "orig_index": str(orig_index) if orig_index is not None else None,
            "sub_dataset": sub_dataset,
            "image_pixel_hashes": [info["pixel_sha256"] for _, info in media],
            "image_reference_indices": image_indices,
        },
    }
    return row, media


def split_rows(rows: list[dict], train_ratio: float, seed: int):
    """Keep connected components of shared images/source IDs in one split."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        info = row["extra_info"]
        keys = [("image", value) for value in info["image_pixel_hashes"]]
        if info["orig_index"] is not None:
            keys.append(("source_id", row["data_source"], info["sub_dataset"], info["orig_index"]))
        if not info["image_pixel_hashes"]:
            keys.append(("text", " ".join(info["question"].split())))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[find(i)].append(row)
    groups = list(grouped.values())
    if len(groups) < 2:
        raise ValueError("Need at least two independent sample/media groups to form non-leaking train/validation sets")
    random.Random(seed).shuffle(groups)
    target = len(rows) * train_ratio
    cumulative = 0
    cuts = []
    for cut, group in enumerate(groups[:-1], 1):
        cumulative += len(group)
        cuts.append((abs(cumulative - target), cut))
    cut = min(cuts)[1]
    splits = {
        "train": [row for group in groups[:cut] for row in group],
        "validation": [row for group in groups[cut:] for row in group],
    }
    for name, split in splits.items():
        for row in split:
            row["extra_info"]["split"] = name
    return splits, len(groups)


def output_schema():
    """Explicitly type empty media lists so image-only shards remain compatible."""
    import pyarrow as pa

    def strings(names):
        return [(name, pa.string()) for name in names]

    extra = strings(["split"])
    extra += [("index", pa.int64())]
    extra += strings(["problem_id", "dataset", "modality", "raw_source", "raw_answer", "question", "source_file"])
    extra += [("source_row", pa.int64())]
    extra += strings(["source_reward_type", "reward_type", "source_answer_json", "orig_index", "sub_dataset"])
    extra += [("image_pixel_hashes", pa.list_(pa.string()))]
    extra += [("image_reference_indices", pa.list_(pa.int64()))]
    video = pa.struct(
        [
            ("video", pa.string()),
            ("fps", pa.float64()),
            ("min_pixels", pa.int64()),
            ("max_pixels", pa.int64()),
            ("max_frames", pa.int64()),
        ]
    )
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct(strings(["role", "content"])))),
            ("images", pa.list_(pa.struct(strings(["image"])))),
            ("videos", pa.list_(video)),
            ("audios", pa.list_(pa.string())),
            ("ability", pa.string()),
            ("reward_model", pa.struct(strings(["style", "ground_truth"]))),
            ("extra_info", pa.struct(extra)),
        ]
    )


def convert_dataset(
    patterns: list[str],
    output_dir: str | Path,
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
    expected_shards: int | None = None,
    image_path_prefix: str | None = None,
    single_image_one_based: bool = False,
    read_batch_size: int = 16,
    log_interval: float = 10.0,
) -> dict:
    started = time.monotonic()
    _log("start", f"Starting OC-RL conversion; output={output_dir}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if not isinstance(read_batch_size, int) or isinstance(read_batch_size, bool) or read_batch_size <= 0:
        raise ValueError("read_batch_size must be a positive integer")
    if (
        isinstance(log_interval, bool)
        or not isinstance(log_interval, (int, float))
        or not math.isfinite(log_interval)
        or log_interval <= 0
    ):
        raise ValueError("log_interval must be a finite positive number of seconds")
    inputs = resolve_inputs(patterns)
    if expected_shards is not None and len(inputs) != expected_shards:
        raise ValueError(f"Expected {expected_shards} input shards, found {len(inputs)}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    media_dir = output / "media"
    prefix = image_path_prefix or media_dir.as_posix()
    if not (PurePosixPath(prefix).is_absolute() or PureWindowsPath(prefix).is_absolute()):
        raise ValueError("image_path_prefix must be an absolute filesystem path, not a URL")
    prefix = prefix.replace("\\", "/").rstrip("/")
    input_stats = []
    _log("scan", f"Checking schemas and row counts for {len(inputs)} input shards")
    for shard_number, path in enumerate(inputs, 1):
        _log("scan", f"Checking shard {shard_number}/{len(inputs)}: {path}")
        with pq.ParquetFile(path) as parquet:
            columns = set(parquet.schema_arrow.names)
            if columns != set(REQUIRED_COLUMNS):
                raise ValueError(f"{path}: expected exactly {REQUIRED_COLUMNS}, found {sorted(columns)}")
            input_stats.append(
                {
                    "path": str(path),
                    "rows": parquet.metadata.num_rows,
                    "size_bytes": path.stat().st_size,
                    "kept_rows": 0,
                    "skipped_rows": 0,
                }
            )
    input_rows = sum(item["rows"] for item in input_stats)
    input_bytes = sum(item["size_bytes"] for item in input_stats)
    _log("scan", f"Complete: shards={len(inputs)}, rows={input_rows:,}, input_size={input_bytes / 1024**3:.2f} GiB")
    output.mkdir(parents=True, exist_ok=False)
    media_dir.mkdir()
    rows = []
    media_manifest = {}
    skipped_by_source = Counter()
    skipped_by_reason = Counter({"missing_answer_solution": 0, "mpo_image": 0, "invalid_problem": 0})
    progress = _ConversionProgress(input_rows, len(inputs), log_interval)
    _log("convert", "Reading rows and writing images; ETA_rows excludes the later split/write stages")
    progress.report(0, 0, 0, 0, 0, 0, force=True)
    with (output / "skipped_rows.jsonl").open("x", encoding="utf-8") as skipped_handle:
        for shard_number, (path, stats) in enumerate(zip(inputs, input_stats, strict=True), 1):
            _log("shard", f"Starting {shard_number}/{len(inputs)}: {path.name}, rows={stats['rows']:,}")
            with pq.ParquetFile(path) as parquet:
                source_row = 0
                for batch in parquet.iter_batches(batch_size=read_batch_size, columns=list(REQUIRED_COLUMNS)):
                    for raw in batch.to_pylist():
                        row_number = source_row
                        source_row += 1
                        try:
                            row, media = build_row(
                                raw,
                                source_file=str(path),
                                source_row=row_number,
                                index=len(rows),
                                image_prefix=prefix,
                                single_image_one_based=single_image_one_based,
                            )
                        except (MissingSolutionError, InvalidProblemError, MPOImageError) as exc:
                            if isinstance(exc, MissingSolutionError):
                                reason = "missing_answer_solution"
                            elif isinstance(exc, InvalidProblemError):
                                reason = "invalid_problem"
                            else:
                                reason = "mpo_image"
                            stats["skipped_rows"] += 1
                            skipped_by_source[str(raw.get("data_source"))] += 1
                            skipped_by_reason[reason] += 1
                            skipped_record = {
                                "source_file": str(path),
                                "source_row": row_number,
                                "data_source": raw.get("data_source"),
                                "reward_type": raw.get("reward_type"),
                                "reason": reason,
                            }
                            if isinstance(exc, InvalidProblemError):
                                skipped_record.update(problem_issue=exc.problem_issue, problem_type=exc.problem_type)
                            skipped_handle.write(json.dumps(skipped_record, ensure_ascii=False) + "\n")
                            if reason != "missing_answer_solution":
                                skipped_handle.flush()
                                _log(
                                    "skip",
                                    f"{path} row {row_number}: {exc}; data_source={raw.get('data_source')}; "
                                    f"{reason}_rows={skipped_by_reason[reason]:,}",
                                )
                            if progress.report(
                                len(rows),
                                sum(skipped_by_source.values()),
                                len(media_manifest),
                                shard_number,
                                source_row,
                                stats["rows"],
                            ):
                                skipped_handle.flush()
                            continue
                        except ValueError as exc:
                            raise ValueError(f"{path} row {row_number}: {exc}") from exc
                        for payload, info in media:
                            name = info["filename"]
                            if name not in media_manifest:
                                with (media_dir / name).open("xb") as handle:
                                    handle.write(payload)
                                runtime_path = prefix + "/" + name
                                media_manifest[name] = {
                                    **info,
                                    "relative_path": "media/" + name,
                                    "runtime_path": runtime_path,
                                    "embedding_cache_key": hashlib.sha256(runtime_path.encode()).hexdigest(),
                                }
                        rows.append(row)
                        stats["kept_rows"] += 1
                        if progress.report(
                            len(rows),
                            sum(skipped_by_source.values()),
                            len(media_manifest),
                            shard_number,
                            source_row,
                            stats["rows"],
                        ):
                            skipped_handle.flush()
            skipped_handle.flush()
            progress.report(
                len(rows),
                sum(skipped_by_source.values()),
                len(media_manifest),
                shard_number,
                source_row,
                stats["rows"],
                force=True,
            )
    _log("filter", f"Skipped rows by reason: {dict(skipped_by_reason)}")
    if not rows:
        raise ValueError("No usable rows remain after filtering; no training dataset was generated")
    split_started = time.monotonic()
    _log("split", f"Grouping {len(rows):,} retained rows by shared media/source IDs and splitting train/validation")
    splits, groups = split_rows(rows, train_ratio, seed)
    _log(
        "split",
        f"Complete: groups={groups:,}, train={len(splits['train']):,}, validation={len(splits['validation']):,} "
        f"elapsed={_duration(time.monotonic() - split_started)}",
    )
    split_stats = {}
    for name, split in splits.items():
        write_started = time.monotonic()
        _log("write", f"Building and writing {name}.parquet: rows={len(split):,}")
        table = pa.Table.from_pylist(split, schema=output_schema())
        # The directory is owned by this invocation; never overwrite an existing file.
        with (output / f"{name}.parquet").open("xb") as handle:
            pq.write_table(table, handle, use_dictionary=False)
        _log(
            "write",
            f"Complete: {name}.parquet, size={(output / f'{name}.parquet').stat().st_size / 1024**2:.2f} MiB, "
            f"elapsed={_duration(time.monotonic() - write_started)}",
        )
        split_stats[name] = {
            "rows": len(split),
            "data_sources": dict(Counter(row["data_source"] for row in split)),
            "modalities": dict(Counter(row["ability"] for row in split)),
            "reward_types": dict(Counter(row["extra_info"]["source_reward_type"] for row in split)),
        }
    manifest = {
        "status": "complete",
        "converter": "omni_mix_4b_v1",
        "ground_truth_source": "answer.solution",
        "single_image_one_based": single_image_one_based,
        "read_batch_size": read_batch_size,
        "inputs": input_stats,
        "input_shards": len(inputs),
        "input_rows": input_rows,
        "total_rows": len(rows),
        "dropped_rows": sum(skipped_by_source.values()),
        "missing_solution_policy": "skip",
        "invalid_problem_policy": "skip_row",
        "mpo_image_policy": "skip_row",
        "skipped_by_reason": dict(skipped_by_reason),
        "skipped_by_data_source": dict(skipped_by_source),
        "skipped_rows_file": "skipped_rows.jsonl",
        "split_method": "seeded shuffle of connected shared-pixel/source-ID groups; nearest nonempty row-count split",
        "seed": seed,
        "requested_train_ratio": train_ratio,
        "actual_train_ratio": len(splits["train"]) / len(rows),
        "independent_groups": groups,
        "splits": split_stats,
        "unique_image_files": len(media_manifest),
        "image_references": sum(len(row["images"]) for row in rows),
        "image_path_prefix": prefix,
        "images": list(media_manifest.values()),
        "embedding_cache": "not computed; use the CURRENT actor checkpoint and a NEW cache directory",
    }
    _log("manifest", f"Writing {output / 'manifest.json'}")
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _log(
        "done",
        f"Conversion complete: input={input_rows:,}, kept={len(rows):,}, "
        f"skipped={sum(skipped_by_source.values()):,}, images={len(media_manifest):,}, "
        f"elapsed={_duration(time.monotonic() - started)}, output={output}",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True, help="Explicit raw shard paths/globs, not RL train.parquet"
    )
    parser.add_argument("--output-dir", required=True, help="Must not exist; all output stays beneath this directory")
    parser.add_argument(
        "--expected-shards", type=int, help="Refuse an incomplete selection, e.g. 199 for the full dataset"
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-image-one-based",
        action="store_true",
        help="Explicitly allow <image_1> to refer to the sole image; default indexed references are zero-based",
    )
    parser.add_argument("--read-batch-size", type=int, default=16, help="Rows to read per batch, default: 16")
    parser.add_argument(
        "--log-interval",
        type=float,
        default=10.0,
        help="Progress interval in seconds (default: 10); always logs stages and shard completion to stderr",
    )
    parser.add_argument(
        "--image-path-prefix", help="Optional final absolute media directory after copying to the training server"
    )
    args = parser.parse_args()
    try:
        result = convert_dataset(
            args.input,
            args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            expected_shards=args.expected_shards,
            image_path_prefix=args.image_path_prefix,
            single_image_one_based=args.single_image_one_based,
            read_batch_size=args.read_batch_size,
            log_interval=args.log_interval,
        )
    except (ValueError, OSError) as exc:
        _log("error", f"Conversion failed: {exc}; existing outputs were not reused; inspect any NEW partial directory")
        parser.exit(1)
    summary = {key: value for key, value in result.items() if key not in ("images", "inputs")}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
