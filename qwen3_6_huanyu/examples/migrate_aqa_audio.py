#!/usr/bin/env python3
"""Copy AQA audio archives and write a parquet with relocated audio references.

Run this file directly with Python + pyarrow; no training dependencies are used.
Defaults migrate train.parquet and validation.parquet together. Use --parquet
to select a single split, including validation after train was already migrated.
Only referenced files are copied, once per archive across both splits;
ZIP/Zarr contents are not extracted, and the ``::audio_id`` suffix is unchanged.
The source tree and source parquets are never modified.

The CLI reads source paths directly under /opt/huawei/dataset, without prefix
conversion. Parquets are written to destination-root; audio archives go below
its audios/ subdirectory, retaining their relative paths. The new parquet stores
the actual destination paths. Prefix mapping and a separate output reference
root remain optional CLI overrides for machines with different mount paths.
Scanning deduplicates archive paths before filesystem checks. Destination
preflight reuses parent checks, including missing directories on FUSE mounts.
Progress includes a periodic heartbeat with the current operation and path.

Existing identical media are reused after SHA256 verification. Existing output
parquets or conflicting media are never overwritten. Interrupted runs can reuse
completed media. Publishing prefers a hard link (Windows: non-overwriting
rename); mounts that reject this use exclusive creation and a verified copy.
The copy fallback is not atomic: consume output only after COMPLETE. A forced
termination during that copy may leave an incomplete target, which is reported
as a conflict on retry rather than silently reused or overwritten.
"""

import argparse
from collections.abc import Mapping
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
import time
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PARQUET = "/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/train.parquet"
DEFAULT_VALIDATION_PARQUET = "/opt/huawei/dataset/ccb_data/data/omni/omni_mix_aqa_only/validation.parquet"
DEFAULT_SOURCE_ROOT = "/opt/huawei/dataset/data_lyj_guiyang/dataset/al_data/audios"
DEFAULT_DESTINATION_ROOT = "/opt/huawei/dataset/hfy-infra-gy/data/aqa_dataset"
DEFAULT_REFERENCE_PREFIX = None
DEFAULT_LOCAL_PREFIX = None
DEFAULT_OUTPUT_REFERENCE_ROOT = None  # Store the actual copied-file paths by default.
AUDIO_KEYS = ("audio", "audio_url", "path", "url", "bytes")
BLOCK_SIZE = 8 * 1024 * 1024
_COPY_PUBLISH_NOTICE_SHOWN = False


class _Progress:
    """Report the current operation even while the main thread waits on a mount."""

    def __init__(self, interval):
        if not 0 < interval < float("inf"):
            raise ValueError("progress_interval must be positive and finite")
        self.interval = interval
        self.started = time.monotonic()
        self.context = "initializing"
        self.operation = ("starting", self.started)
        self.stopped = threading.Event()

    def set_operation(self, text):
        self.operation = (text, time.monotonic())

    def report(self):
        operation, since = self.operation
        now = time.monotonic()
        print(f"[aqa-migrate] {self.context} elapsed={now - self.started:.1f}s "
              f"operation={operation} operation_elapsed={now - since:.1f}s", flush=True)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            self.report()

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=1)


def _destination_size(path, data_root, parents):
    """Check each shared parent once during preflight; copying rechecks paths."""
    if path.parent not in parents:
        resolved = path.parent.resolve()
        if not resolved.is_relative_to(data_root):
            raise ValueError(f"Destination escapes data directory (possibly a symlink): {path}")
        try:
            parent_info = path.parent.stat()
        except FileNotFoundError:
            parents[path.parent] = False
        else:
            if not stat.S_ISDIR(parent_info.st_mode):
                raise NotADirectoryError(path.parent)
            parents[path.parent] = True
    # A fresh dataset has many absent directories. Avoid a remote negative
    # lookup for every archive below each absent parent. Copying still checks
    # for any files/symlinks created concurrently after this preflight.
    if not parents[path.parent]:
        return None
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        _inside(path, data_root)
        info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise FileExistsError(f"Destination media is not a regular file: {path}")
    return info.st_size


def rewrite_audio(value, transform):
    """Visit all audio aliases, preserving bytes, metadata and container shape."""
    if value is None or isinstance(value, (bytes, bytearray, memoryview)):
        return value
    if isinstance(value, str):
        return transform(value) if value.strip() else value
    if isinstance(value, list):
        return [rewrite_audio(item, transform) for item in value]
    if isinstance(value, Mapping):
        keys = [key for key in AUDIO_KEYS if key in value]
        if not keys:
            raise ValueError(f"Unsupported audio object keys: {list(value)}")
        result = dict(value)
        for key in keys:
            result[key] = rewrite_audio(value[key], transform)
        return result
    raise ValueError(f"Unsupported audio value type: {type(value).__name__}")


def _check_prompt(value):
    # This script migrates the audios column, not arbitrary question text.
    # Refuse alternate schemas with media embedded in structured messages.
    if isinstance(value, Mapping):
        if value.get("type") in ("audio", "audio_url", "input_audio"):
            raise ValueError("Structured prompt audio found; this script expects audio references in audios")
        for item in value.values():
            _check_prompt(item)
    elif isinstance(value, list):
        for item in value:
            _check_prompt(item)


def _snapshot(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _unchanged(path, expected):
    if _snapshot(path) != expected:
        raise RuntimeError(f"Source changed during migration: {path}")


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            result.update(block)
    return result.digest()


def _inside(path, root):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes data directory (possibly a symlink): {path}")


def _publish_copy(temporary, destination):
    """Publish on mounts without hard links, without truncating existing files."""
    created = None
    try:
        digest = hashlib.sha256()
        count = 0
        with temporary.open("rb") as source, destination.open("xb") as output:
            created = os.fstat(output.fileno())
            for block in iter(lambda: source.read(BLOCK_SIZE), b""):
                output.write(block)
                digest.update(block)
                count += len(block)
            output.flush()
            os.fsync(output.fileno())
        if (count != temporary.stat().st_size or destination.stat().st_size != count
                or _digest(destination) != digest.digest()):
            raise IOError(f"Published file failed size/SHA256 verification: {destination}")
        shutil.copystat(temporary, destination)
    except BaseException:
        # An existing file that made open('xb') fail must never be deleted.
        # Also avoid deleting a replacement installed by another process.
        if created is not None:
            try:
                if os.path.samestat(created, destination.lstat()):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise


def _publish_new(temporary, destination):
    # Unlike replace()/POSIX rename(), link() atomically fails if dest exists.
    # Windows rename also rejects an existing destination, without needing links.
    global _COPY_PUBLISH_NOTICE_SHOWN
    try:
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite destination: {destination}") from None
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP,
                             errno.ENOSYS, errno.EXDEV, errno.EMLINK):
            raise
        if not _COPY_PUBLISH_NOTICE_SHOWN:
            print(f"[aqa-migrate] hard-link/rename unavailable (errno={exc.errno}); "
                  "using exclusive copy with SHA256 verification. Wait for COMPLETE before using output.",
                  flush=True)
            _COPY_PUBLISH_NOTICE_SHOWN = True
        _publish_copy(temporary, destination)


def _copy_media(source, destination, expected, data_root):
    _unchanged(source, expected)
    _inside(destination, data_root)
    if os.path.lexists(destination):
        if (not destination.is_file() or destination.stat().st_size != expected[0]
                or _digest(source) != _digest(destination)):
            raise FileExistsError(f"Existing media differs from source; not overwritten: {destination}")
        _unchanged(source, expected)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    _inside(destination, data_root)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    temporary = Path(name)
    try:
        source_hash = hashlib.sha256()
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            for block in iter(lambda: input_file.read(BLOCK_SIZE), b""):
                output.write(block)
                source_hash.update(block)
            output.flush()
            os.fsync(output.fileno())
        _unchanged(source, expected)
        if temporary.stat().st_size != expected[0] or _digest(temporary) != source_hash.digest():
            raise IOError(f"Copied media failed size/SHA256 verification: {source}")
        shutil.copystat(source, temporary)
        _publish_new(temporary, destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet(source, destination, schema, rows, expected, rewrite, batch_size):
    _unchanged(source, expected)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        count = 0
        with pq.ParquetFile(source) as reader, pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
            index = schema.get_field_index("audios")
            field = schema.field(index)
            for batch in reader.iter_batches(batch_size=batch_size):
                values = [rewrite_audio(value, rewrite) for value in batch.column(index).to_pylist()]
                # All other Arrow arrays, field metadata and row order stay intact.
                output = batch.set_column(index, field, pa.array(values, type=field.type))
                writer.write_batch(output)
                count += batch.num_rows
        _unchanged(source, expected)
        with pq.ParquetFile(temporary) as result:
            if count != rows or result.metadata.num_rows != rows or not result.schema_arrow.equals(
                schema, check_metadata=True
            ):
                raise RuntimeError(f"Output parquet row count/schema mismatch: {destination}")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _publish_new(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_root(value, option):
    # Server paths are POSIX strings, not paths to resolve on this machine.
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{option} must be an absolute POSIX path without '..': {value}")
    return path


def migrate(parquets, source_root, destination_root, *, dry_run=False, batch_size=512, log_every=100,
            reference_prefix=None, local_prefix=None, output_reference_root=None, progress_interval=10):
    with _Progress(progress_interval) as progress:
        return _migrate(
            parquets, source_root, destination_root, dry_run=dry_run, batch_size=batch_size,
            log_every=log_every, reference_prefix=reference_prefix, local_prefix=local_prefix,
            output_reference_root=output_reference_root, progress=progress,
        )


def _migrate(parquets, source_root, destination_root, *, dry_run, batch_size, log_every,
             reference_prefix, local_prefix, output_reference_root, progress):
    if batch_size < 1 or log_every < 1:
        raise ValueError("batch_size and log_every must be positive")
    if (reference_prefix is None) != (local_prefix is None):
        raise ValueError("reference_prefix and local_prefix must be supplied together")
    if reference_prefix is not None:
        reference_prefix = _reference_root(reference_prefix, "--reference-prefix")
        progress.set_operation(f"resolve_local_prefix {local_prefix}")
        local_prefix = Path(local_prefix).expanduser().resolve(strict=True)
        if not local_prefix.is_dir():
            raise NotADirectoryError(local_prefix)
        print(f"[aqa-migrate] read_prefix={reference_prefix} -> local_prefix={local_prefix}", flush=True)
    if output_reference_root is not None:
        output_reference_root = _reference_root(output_reference_root, "--output-reference-root")
    progress.set_operation(f"resolve_source_root {source_root}")
    source_root = Path(source_root).expanduser().resolve(strict=True)
    progress.set_operation(f"resolve_destination_root {destination_root}")
    destination_root = Path(destination_root).expanduser().resolve()
    data_root = destination_root / "audios"
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if data_root.resolve() != data_root:
        raise ValueError(f"Destination data directory must not be a symlink: {data_root}")
    if data_root.is_relative_to(source_root) or source_root.is_relative_to(data_root):
        raise ValueError("Source audio root and destination data root must not overlap")
    if not parquets:
        raise ValueError("At least one input parquet is required")

    inputs, names = [], set()
    for name in parquets:
        progress.set_operation(f"check_parquet_paths source={name} destination={destination_root}")
        source = Path(name).expanduser().resolve(strict=True)
        destination = destination_root / source.name
        if source.name in names:
            raise ValueError(f"Input parquets have duplicate output name: {source.name}")
        names.add(source.name)
        if source == destination or os.path.lexists(destination):
            raise FileExistsError(f"Output parquet already exists; not overwritten: {destination}")
        inputs.append((source, destination, _snapshot(source)))

    # Dictionaries are bounded by the number of unique archives/references, not
    # the entire dataset. Parquet rows and non-audio columns remain streaming.
    files, references, targets, archives = {}, {}, {}, {}

    def plan(reference):
        if reference in references:
            return references[reference]
        archive, separator, member = reference.partition("::")
        if separator and not member.strip():
            raise ValueError(f"Empty audio archive member: {reference}")
        if archive.startswith("file://"):
            uri = urlparse(archive)
            if uri.netloc not in ("", "localhost"):
                raise ValueError(f"Remote audio URI is unsupported: {reference}")
            archive = unquote(uri.path)
        elif "://" in archive:
            raise ValueError(f"Remote audio URI is unsupported: {reference}")
        original = Path(archive)
        if reference_prefix is not None:
            try:
                relative = PurePosixPath(archive).relative_to(reference_prefix)
            except ValueError:
                pass  # Already-local references are still checked against source_root below.
            else:
                if ".." in relative.parts:
                    raise ValueError(f"Audio path traverses outside its reference prefix: {reference}")
                original = local_prefix.joinpath(*relative.parts)
        if not original.is_absolute():
            raise ValueError(f"Audio path must be absolute: {reference}")
        # One archive can contain thousands of distinct ::audio_id references.
        # Cache by archive path BEFORE any source/destination filesystem calls.
        if original in archives:
            rewritten = archives[original] + (separator + member if separator else "")
            references[reference] = rewritten
            return rewritten
        progress.set_operation(f"check_source_archive {original}")
        try:
            source = original.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Audio file not found: parquet_reference={reference!r}; local_read_path={original}. "
                "Check the audio mount and source path; prefix mapping is disabled unless explicitly supplied."
            ) from exc
        if not source.is_file():
            raise ValueError(f"Audio archive is not a file: {source}")
        if not source.is_relative_to(source_root):
            raise ValueError(f"Audio path is outside --source-root: {source}")
        relative_audio = source.relative_to(source_root)
        destination = data_root / relative_audio
        # No destination filesystem calls in this per-reference scan. Validate
        # unique targets below, and recheck symlink containment before copying.
        if destination in targets and targets[destination] != source:
            raise ValueError(f"Multiple sources map to one destination: {destination}")
        targets[destination] = source
        if source not in files:
            expected = _snapshot(source)
            files[source] = (destination, expected)
        output_path = (
            (output_reference_root / "audios" / relative_audio.as_posix()).as_posix()
            if output_reference_root is not None else destination.as_posix()
        )
        archives[original] = output_path
        rewritten = output_path + (separator + member if separator else "")
        references[reference] = rewritten
        return rewritten

    jobs, total_rows = [], 0
    for source, destination, expected in inputs:
        progress.context = f"scan split={source.name} rows=0/?"
        progress.set_operation(f"open_parquet {source}")
        print(f"[aqa-migrate] scanning={source}", flush=True)
        with pq.ParquetFile(source) as reader:
            schema = reader.schema_arrow
            if schema.get_field_index("audios") < 0:
                raise ValueError(f"Missing audios column: {source}")
            rows = reader.metadata.num_rows
            scanned = 0
            progress.context = f"scan split={source.name} rows=0/{rows} unique_archives={len(files)}"
            columns = ["audios"] + (["prompt"] if "prompt" in schema.names else [])
            progress.set_operation(f"read_parquet_batch {source}")
            for batch in reader.iter_batches(batch_size=batch_size, columns=columns):
                for value in batch.column(0).to_pylist():
                    rewrite_audio(value, plan)
                    scanned += 1
                    progress.context = (f"scan split={source.name} rows={scanned}/{rows} "
                                        f"unique_archives={len(files)} unique_refs={len(references)}")
                    if scanned % log_every == 0:
                        progress.report()
                if len(columns) == 2:
                    progress.set_operation(f"check_prompt_batch {source}")
                    for value in batch.column(1).to_pylist():
                        _check_prompt(value)
                progress.set_operation(f"read_parquet_batch {source}")
            progress.set_operation(f"scan_complete {source}")
            progress.report()
            _unchanged(source, expected)
            total_rows += rows
            jobs.append((source, destination, schema, rows, expected))

    total_bytes = sum(snapshot[0] for _, snapshot in files.values())
    missing_sizes, checked_parents = [], {}
    for index, (dest, snapshot) in enumerate(files.values(), 1):
        progress.context = f"check_destinations files={index}/{len(files)}"
        progress.set_operation(f"stat_destination {dest}")
        if index == 1 or index % log_every == 0:
            progress.report()
        size = _destination_size(dest, data_root, checked_parents)
        if size is None:
            missing_sizes.append(snapshot[0])
        elif size != snapshot[0]:
            raise FileExistsError(f"Conflicting destination media: {dest}")
    missing_bytes = sum(missing_sizes)
    parquet_sizes = [item[2][0] for item in inputs]
    # Exclusive-copy fallback temporarily holds both the prepared and final
    # file. Copies are sequential, so reserve one extra largest file, not two
    # complete datasets. Parquet sizes remain estimates until rewriting.
    temporary_bytes = max(missing_sizes + parquet_sizes + [0])
    progress.context = "check_disk_space"
    progress.set_operation(f"disk_usage {destination_root}")
    ancestor = destination_root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    print(f"[aqa-migrate] rows={total_rows:,} unique_files={len(files):,} "
          f"archive_GiB={total_bytes / 2**30:.3f} to_copy_GiB={missing_bytes / 2**30:.3f} "
          f"free_GiB={free / 2**30:.3f}", flush=True)
    print(f"[aqa-migrate] audio_root={data_root}", flush=True)
    print(f"[aqa-migrate] parquet_audio_root="
          f"{output_reference_root / 'audios' if output_reference_root is not None else data_root}", flush=True)
    if missing_bytes + sum(parquet_sizes) + temporary_bytes > free:
        raise OSError("Insufficient free space for new archives, estimated output parquets and a temporary copy")
    summary = {"rows": total_rows, "unique_files": len(files), "copied": 0, "reused": 0,
               "outputs": [str(job[1]) for job in jobs]}
    if dry_run:
        print("[aqa-migrate] DRY RUN: no writes; existing equal-size media will be SHA256-checked during copying.",
              flush=True)
        for output in summary["outputs"]:
            print(f"[aqa-migrate] planned_parquet={output}", flush=True)
        return summary

    progress.set_operation(f"mkdir {destination_root}")
    destination_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, (source, (destination, expected)) in enumerate(files.items(), 1):
        progress.context = f"copy files={index}/{len(files)} copied={summary['copied']} reused={summary['reused']}"
        progress.set_operation(f"copy_or_verify source={source} destination={destination}")
        if index == 1 or index % log_every == 0:
            print(f"[aqa-migrate] checking/copying {index}/{len(files)}: {source.name}", flush=True)
        copied = _copy_media(source, destination, expected, data_root)
        summary["copied" if copied else "reused"] += 1
    for source, destination, schema, rows, expected in jobs:
        progress.context = f"write_parquet split={source.name} rows={rows}"
        progress.set_operation(f"write_and_verify {destination}")
        # Indexing (rather than re-planning) rejects any changed/new reference.
        _write_parquet(source, destination, schema, rows, expected, references.__getitem__, batch_size)
        print(f"[aqa-migrate] parquet={destination} rows={rows:,}", flush=True)
    print(f"[aqa-migrate] COMPLETE copied={summary['copied']} reused={summary['reused']} "
          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", default=[DEFAULT_PARQUET, DEFAULT_VALIDATION_PARQUET],
                        help="Source parquet file(s); defaults to train and validation. "
                             "Select validation only if train has already been migrated.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--destination-root", default=DEFAULT_DESTINATION_ROOT,
                        help="Local output directory for train/validation parquets and the audios/ subdirectory")
    parser.add_argument("--reference-prefix", default=DEFAULT_REFERENCE_PREFIX,
                        help="Optional old audio prefix to remap; requires --local-prefix. Disabled by default.")
    parser.add_argument("--local-prefix", default=DEFAULT_LOCAL_PREFIX,
                        help="Optional replacement mount for reading files only; requires --reference-prefix")
    parser.add_argument("--output-reference-root", default=DEFAULT_OUTPUT_REFERENCE_ROOT,
                        help="Optional reference root with an audios/ subdirectory; defaults to actual copied paths")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Report every N scanned rows or checked/copied archives")
    parser.add_argument("--progress-interval", type=float, default=10,
                        help="Print current operation every N seconds, including while waiting on storage")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and estimate size without writing")
    args = parser.parse_args()
    migrate(args.parquet, args.source_root, args.destination_root,
            dry_run=args.dry_run, batch_size=args.batch_size, log_every=args.log_every,
            reference_prefix=args.reference_prefix, local_prefix=args.local_prefix,
            output_reference_root=args.output_reference_root, progress_interval=args.progress_interval)


if __name__ == "__main__":
    main()
