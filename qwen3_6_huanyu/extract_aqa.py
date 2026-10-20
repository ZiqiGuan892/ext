"""Extract only extra_info.raw_source == 'aqa_real' from an OmniMix parquet.

Run this file directly; it needs pyarrow, not the training/runtime imports.
The input split, schema, media paths, ground truths and row order are preserved.
The output must not exist. No media are decoded and no cache is modified.
"""

import argparse
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE = "aqa_real"


def extract_aqa(input_path, output_path, batch_size=8192):
    if batch_size < 1:
        raise ValueError("read-batch-size must be positive")
    source = Path(input_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().resolve()
    if source == target or target.exists():
        raise FileExistsError(f"Refusing to overwrite input or existing output: {target}")

    with pq.ParquetFile(source) as parquet:
        schema = parquet.schema_arrow
        if "extra_info" not in schema.names:
            raise ValueError("Expected OmniMix extra_info.raw_source; extra_info is missing")
        extra_type = schema.field("extra_info").type
        if not pa.types.is_struct(extra_type) or extra_type.get_field_index("raw_source") < 0:
            raise ValueError("Expected extra_info to be a struct with a raw_source field")
        source_type = extra_type.field("raw_source").type
        if not (pa.types.is_string(source_type) or pa.types.is_large_string(source_type)):
            raise ValueError(f"Expected string extra_info.raw_source, got {source_type}")

        total = parquet.metadata.num_rows
        scanned = kept = 0
        started = time.monotonic()
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[aqa] input={source} rows={total:,} filter=extra_info.raw_source=={SOURCE!r}", flush=True)
        # Exclusive creation also protects an output created after the check.
        # On a normal error/interrupt remove only the file created by this run.
        handle = target.open("xb")
        try:
            with handle, pq.ParquetWriter(handle, schema, use_dictionary=False) as writer:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    sources = pc.struct_field(batch.column(schema.get_field_index("extra_info")), "raw_source")
                    mask = pc.fill_null(pc.equal(sources, SOURCE), False)
                    selected = batch.filter(mask)
                    if selected.num_rows:
                        writer.write_batch(selected)
                    scanned += batch.num_rows
                    kept += selected.num_rows
                    print(f"[aqa] scanned={scanned:,}/{total:,} kept={kept:,} skipped={scanned-kept:,}", flush=True)
                if kept == 0:
                    raise ValueError("No aqa_real rows found; no output dataset was retained")
            with pq.ParquetFile(target) as result:
                if result.metadata.num_rows != kept or not result.schema_arrow.equals(schema, check_metadata=True):
                    raise RuntimeError("Output row count or schema verification failed")
        except BaseException:
            handle.close()
            target.unlink(missing_ok=True)
            raise
    print(f"[aqa] COMPLETE output={target} rows={kept:,} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing OmniMix train.parquet (or validation.parquet)")
    parser.add_argument("--output", required=True, help="New parquet path; existing files are never overwritten")
    parser.add_argument("--read-batch-size", type=int, default=8192)
    args = parser.parse_args()
    extract_aqa(args.input, args.output, args.read_batch_size)


if __name__ == "__main__":
    main()
