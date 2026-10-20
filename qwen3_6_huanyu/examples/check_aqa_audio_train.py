"""Read-only AQA split checks before online Huanyu audio training.

No training modules are imported and media are not decoded. Files and zarr
archive references are checked for existence; archive members/decoded waveforms
are still validated by the dataset and processor on every node during training.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq


SOURCE = "aqa_real"
REQUIRED_COLUMNS = ("extra_info", "prompt", "images", "videos", "audios", "reward_model")


def _audio_reference_exists(item, checked_paths):
    if isinstance(item, Mapping):
        value = next((item[key] for key in ("audio", "audio_url", "bytes") if item.get(key) is not None), None)
    else:
        value = item
    if isinstance(value, (bytes, bytearray, memoryview)):
        if not value:
            raise ValueError("empty embedded audio bytes")
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio must contain a non-empty local path/archive reference or bytes")
    archive, separator, member = value.partition("::")
    if separator and not member.strip():
        raise ValueError(f"empty zarr archive member: {value!r}")
    if archive not in checked_paths:
        if not Path(archive).is_file():
            raise ValueError(f"audio file/archive does not exist on this node: {archive}")
        checked_paths.add(archive)


def validate_aqa_row(row, checked_paths):
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or extra.get("raw_source") != SOURCE:
        raise ValueError("expected extra_info.raw_source='aqa_real'; mixed sources are not allowed")
    if row.get("images") or row.get("videos"):
        raise ValueError("AQA audio training forbids non-empty images/videos")
    if row.get("ability") not in (None, "audio"):
        raise ValueError(f"expected ability='audio', got {row['ability']!r}")
    audios = row.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError("audios must be a non-empty list")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty message list")
    if any(not isinstance(message, Mapping) or not isinstance(message.get("content"), str)
           or not message["content"].strip() for message in prompt):
        raise ValueError("every prompt message must contain non-empty string content")
    if not any(message.get("role") == "user" for message in prompt):
        raise ValueError("prompt must contain a user message")
    reward = row.get("reward_model")
    answer = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("reward_model.ground_truth must be a non-empty string")
    for item in audios:
        _audio_reference_exists(item, checked_paths)


def check_split(path, minimum_rows=1, batch_size=1024):
    path = Path(path).resolve(strict=True)
    if minimum_rows < 1 or batch_size < 1:
        raise ValueError("minimum_rows and batch_size must be positive")
    checked_paths = set()
    scanned = 0
    with pq.ParquetFile(path) as parquet:
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        total = parquet.metadata.num_rows
        if total < minimum_rows:
            raise ValueError(f"{path}: rows={total} is smaller than required minimum={minimum_rows}")
        columns = list(REQUIRED_COLUMNS)
        if "ability" in parquet.schema_arrow.names:
            columns.append("ability")
        print(f"[huanyu-aqa-check] scanning={path} rows={total}", flush=True)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                try:
                    validate_aqa_row(row, checked_paths)
                except ValueError as error:
                    raise ValueError(f"{path} row={scanned} (zero-based): {error}") from error
                scanned += 1
    print(f"[huanyu-aqa-check] OK rows={scanned} source={SOURCE} audio_only=true "
          f"media_files_or_archives={len(checked_paths)} path={path}", flush=True)
    return scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--train-batch-size", type=int, default=4)
    args = parser.parse_args()
    check_split(args.train, minimum_rows=args.train_batch_size)
    check_split(args.validation)


if __name__ == "__main__":
    main()
