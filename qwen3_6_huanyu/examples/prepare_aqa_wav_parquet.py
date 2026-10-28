#!/usr/bin/env python3
"""Reuse serving-export WAVs in a new AQA Parquet, optionally sampling rows.

Python 3.8+, pyarrow, numpy and soundfile. Never reads ZIPs or re-encodes WAVs.
Use the manifest from export_aqa_validation_wav.py for exact row/audio matching.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def audio_reference(item):
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        for key in ("audio", "audio_url"):
            if item.get(key) is not None:
                if not isinstance(item[key], str):
                    break
                return item[key], key
    raise ValueError("audios must contain paths or dicts with an audio/audio_url path")


def load_manifest(root):
    records = {}
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("row_index")
            if type(index) is not int or index < 0 or index in records:
                raise ValueError("manifest line {}: invalid/duplicate row_index={!r}".format(line_number, index))
            if record.get("sample_id") != "row_{:06d}".format(index):
                raise ValueError("manifest line {}: sample_id does not match row_index".format(line_number))
            records[index] = record
    return records


def check_wav(root, entry, checked, sample_id):
    relative = Path(entry["wav_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("wav_path must stay inside export-dir")
    path = (root / relative).resolve(strict=True)
    sample_dir = (root / "samples" / sample_id).resolve(strict=True)
    sample_dir.relative_to(root)
    path.relative_to(sample_dir)
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError("expected a real WAV file: {}".format(path))
    if path not in checked:
        with sf.SoundFile(str(path)) as audio:
            if audio.format not in ("WAV", "WAVEX", "RF64") or not len(audio):
                raise ValueError("empty or non-WAV audio: {}".format(path))
            info = {"sample_rate": audio.samplerate, "num_frames": len(audio), "channels": audio.channels}
            for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("non-finite samples: {}".format(path))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checked[path] = dict(info, sha256=digest.hexdigest())
    info = checked[path]
    for key in ("sample_rate", "num_frames", "channels"):
        if entry.get(key) != info[key]:
            raise ValueError("WAV {} differs from manifest: {}".format(key, path))
    return str(path), info


def convert_parquet(parquet_path, export_dir, output_parquet, limit=0, batch_size=32,
                    sample_size=None, seed=42):
    if limit < 0 or batch_size < 1 or (sample_size is not None and sample_size < 1):
        raise ValueError("limit must be nonnegative; batch-size and sample-size must be positive")
    if limit and sample_size is not None:
        raise ValueError("use either limit or sample-size, not both")
    source = Path(parquet_path).expanduser().resolve(strict=True)
    root = Path(export_dir).expanduser().resolve(strict=True)
    output = Path(output_parquet).expanduser().resolve()
    report = output.with_suffix(".selection.json")
    if output.exists() or report.exists() or output == source:
        raise FileExistsError("output Parquet or selection JSON already exists; choose a new output path")
    records = load_manifest(root)
    checked, selected_records, written, created = {}, [], 0, []
    try:
        with pq.ParquetFile(source) as reader:
            schema = reader.schema_arrow
            if not {"audios", "prompt", "reward_model"}.issubset(schema.names):
                raise ValueError("Parquet requires audios, prompt and reward_model columns")
            total = reader.metadata.num_rows
            if not total:
                raise ValueError("source Parquet is empty")
            if sample_size is not None:
                if sample_size > total:
                    raise ValueError("sample-size exceeds source rows={}".format(total))
                selected = sorted(random.Random(seed).sample(range(total), sample_size))
            else:
                selected = list(range(min(limit, total) if limit else total))
            missing = set(selected) - records.keys()
            if missing:
                raise ValueError("export manifest lacks {} selected rows (first: {}); export all source rows first".format(
                    len(missing), sorted(missing)[:8]))
            selected_set = set(selected)
            print("[aqa-wav-parquet] source={} rows={} selected={} seed={} export={} output={}".format(
                source, total, len(selected), seed, root, output), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                created.append(output)
                with pq.ParquetWriter(handle, schema) as writer:
                    offset = 0
                    for batch in reader.iter_batches(batch_size=batch_size):
                        indices = [i for i in range(batch.num_rows) if offset + i in selected_set]
                        if indices:
                            table = pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64()))
                            replaced = []
                            for local_index, row in zip(indices, table.to_pylist()):
                                index = offset + local_index
                                record = records[index]
                                if Path(record["source_parquet"]).resolve() != source:
                                    raise ValueError("row {}: manifest belongs to a different source Parquet".format(index))
                                if (record.get("prompt") != row["prompt"] or
                                        record.get("ground_truth") != row["reward_model"].get("ground_truth")):
                                    raise ValueError("row {}: prompt/ground_truth mismatch".format(index))
                                original = row["audios"]
                                entries = record.get("audios")
                                if not original or not isinstance(entries, list) or len(original) != len(entries):
                                    raise ValueError("row {}: audio count mismatch".format(index))
                                paths, audit = [], []
                                for audio_index, (item, entry) in enumerate(zip(original, entries)):
                                    reference, key = audio_reference(item)
                                    if entry.get("audio_index") != audio_index or entry.get("source_reference") != reference:
                                        raise ValueError("row {} audio {}: source_reference/order mismatch".format(index, audio_index))
                                    path, info = check_wav(root, entry, checked, record["sample_id"])
                                    paths.append(path if key is None else dict(item, **{key: path}))
                                    audit.append(dict(info, audio_index=audio_index, source_reference=reference, wav_path=path))
                                replaced.append(paths)
                                selected_records.append({"output_row_index": written, "source_row_index": index,
                                                         "sample_id": record["sample_id"], "audios": audit})
                                written += 1
                            field = schema.field("audios")
                            table = table.set_column(schema.get_field_index("audios"), field,
                                                     pa.array(replaced, type=field.type))
                            writer.write_table(table)
                            print("[aqa-wav-parquet] checked/written={}/{}".format(written, len(selected)), flush=True)
                        offset += batch.num_rows
            if written != len(selected):
                raise RuntimeError("source row count changed while reading")
        with report.open("x", encoding="utf-8") as handle:
            created.append(report)
            json.dump({"source_parquet": str(source), "source_rows": total,
                       "export_dir": str(root), "output_parquet": str(output),
                       "selected_rows": written, "sample_size": sample_size, "seed": seed,
                       "rows": selected_records}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except BaseException:
        # Remove only the two files exclusively created by this invocation.
        for path in created:
            path.unlink()
        raise
    print("[aqa-wav-parquet] COMPLETE rows={} WAVs={} ZIP_reads=0 report={}".format(
        written, len(checked), report), flush=True)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="Original AQA split used to export the WAVs")
    parser.add_argument("--export-dir", required=True, help="Existing serving export with manifest.jsonl")
    parser.add_argument("--output-parquet", required=True, help="New file; original source is never overwritten")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=0, help="First N rows; 0 keeps all")
    selection.add_argument("--sample-size", type=int, help="Random sample without replacement, sorted in source order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        convert_parquet(args.parquet, args.export_dir, args.output_parquet, args.limit,
                        args.batch_size, args.sample_size, args.seed)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("[aqa-wav-parquet] ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
