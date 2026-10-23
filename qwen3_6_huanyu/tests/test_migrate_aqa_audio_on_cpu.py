"""Exercise audio relocation with real, tiny Arrow parquet files and media bytes."""

import errno
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "migrate_aqa_audio.py"
_spec = importlib.util.spec_from_file_location("huanyu_audio_migration_fixture", SCRIPT)
migration = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = migration
_spec.loader.exec_module(migration)


class AudioMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original audio"
        self.source.mkdir()
        self.destination = self.root / "relocated dataset"
        self.input = self.root / "input" / "train.parquet"

    def media(self, relative="常识类知识/20240801/audio/samples.zarr.zip", content=b"archive bytes"):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def mapped(self, source):
        return self.destination / "data" / source.relative_to(self.source)

    def parquet(self, audio_rows, path=None):
        path = path or self.input
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("audios", pa.list_(pa.string()), metadata={b"meaning": b"media references"}),
                pa.field("prompt", pa.string()),
                pa.field("extra_info", pa.struct([pa.field("raw_source", pa.string())])),
            ],
            metadata={b"dataset": b"aqa_real", b"user_metadata": "保持原样".encode("utf-8")},
        )
        rows = [
            {
                "id": index,
                "audios": audios,
                "prompt": "Question containing a path that must not change: " + self.source.as_posix(),
                "extra_info": {"raw_source": "aqa_real"},
            }
            for index, audios in enumerate(audio_rows)
        ]
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, row_group_size=1)
        # Parquet normalizes nested list child names from item to element.
        # Preserve the actual source-file schema, including all its metadata.
        return pq.read_table(path)

    def migrate(self, parquets=None, **kwargs):
        return migration.migrate(
            parquets or [self.input], self.source, self.destination,
            batch_size=1, log_every=100, **kwargs,
        )

    def test_preserves_structure_members_metadata_and_non_audio_columns(self):
        first = self.media(content=b"first archive")
        second = self.media("常识类知识/20240926/audio/samples.zarr.zip", b"second archive")
        rows = [
            [first.as_posix() + "::first-uuid", first.as_posix() + "::second-uuid"],
            [second.as_posix() + "::third-uuid"],
            [],
            None,
        ]
        original = self.parquet(rows)
        input_bytes = self.input.read_bytes()

        result = self.migrate()

        output = self.destination / "train.parquet"
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["unique_files"], 2)
        self.assertEqual(result["copied"], 2)
        self.assertEqual(result["reused"], 0)
        self.assertEqual([Path(path) for path in result["outputs"]], [output])
        migrated = pq.read_table(output)
        self.assertTrue(migrated.schema.equals(original.schema, check_metadata=True))
        for name in ("id", "prompt", "extra_info"):
            self.assertEqual(migrated[name].to_pylist(), original[name].to_pylist())
        self.assertEqual(
            migrated["audios"].to_pylist(),
            [
                [self.mapped(first).as_posix() + "::first-uuid", self.mapped(first).as_posix() + "::second-uuid"],
                [self.mapped(second).as_posix() + "::third-uuid"], [], None,
            ],
        )
        self.assertEqual(self.mapped(first).read_bytes(), b"first archive")
        self.assertEqual(self.mapped(second).read_bytes(), b"second archive")
        self.assertEqual(self.input.read_bytes(), input_bytes)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())

    def test_multiple_parquets_share_one_copy_of_same_archive(self):
        archive = self.media()
        validation = self.input.with_name("validation.parquet")
        self.parquet([[archive.as_posix() + "::train"]])
        self.parquet([[archive.as_posix() + "::validation"]], validation)

        result = self.migrate([self.input, validation])

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["unique_files"], 1)
        self.assertEqual(result["copied"], 1)
        self.assertEqual({Path(path).name for path in result["outputs"]}, {"train.parquet", "validation.parquet"})
        self.assertEqual(
            pq.read_table(self.destination / "validation.parquet")["audios"].to_pylist(),
            [[self.mapped(archive).as_posix() + "::validation"]],
        )

    def test_dry_run_checks_without_creating_destination(self):
        archive = self.media()
        self.parquet([[archive.as_posix() + "::uuid"]])

        result = self.migrate(dry_run=True)

        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["unique_files"], 1)
        self.assertFalse(self.destination.exists())

    def test_old_server_paths_read_from_local_mount_but_output_uses_server_root(self):
        archive = self.media(content=b"local archive bytes")
        old = "/opt/huawei/dataset/" + archive.relative_to(self.root).as_posix()
        source_table = self.parquet([[old + "::server-uuid", archive.as_posix() + "::local-uuid"]])
        original_parquet = self.input.read_bytes()

        result = self.migrate(
            reference_prefix="/opt/huawei/dataset", local_prefix=self.root,
            output_reference_root="/opt/huawei/dataset/hfy-infra-gy",
        )

        self.assertEqual(result["unique_files"], 1)
        self.assertEqual(result["copied"], 1)
        self.assertEqual(self.mapped(archive).read_bytes(), b"local archive bytes")
        expected = "/opt/huawei/dataset/hfy-infra-gy/data/" + archive.relative_to(self.source).as_posix()
        migrated = pq.read_table(self.destination / "train.parquet")
        self.assertEqual(migrated["audios"].to_pylist(), [[expected + "::server-uuid", expected + "::local-uuid"]])
        self.assertEqual(migrated["prompt"].to_pylist(), source_table["prompt"].to_pylist())
        self.assertTrue(migrated.schema.equals(source_table.schema, check_metadata=True))
        self.assertEqual(self.input.read_bytes(), original_parquet)

    def test_cli_defaults_keep_training_server_prefix(self):
        archive = self.media()
        self.parquet([["/opt/huawei/dataset/" + archive.relative_to(self.root).as_posix() + "::uuid"]])

        result = subprocess.run([
            sys.executable, "-B", str(SCRIPT), "--parquet", str(self.input),
            "--source-root", str(self.source), "--destination-root", str(self.destination),
            "--local-prefix", str(self.root),
        ], capture_output=True, text=True, encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mapped(archive).read_bytes(), archive.read_bytes())
        self.assertEqual(
            pq.read_table(self.destination / "train.parquet")["audios"].to_pylist(),
            [["/opt/huawei/dataset/hfy-infra-gy/data/" + archive.relative_to(self.source).as_posix() + "::uuid"]],
        )

    def test_cli_without_parquet_migrates_train_and_validation(self):
        archive = self.media(content=b"one shared audio archive")
        validation = self.input.with_name("validation.parquet")
        original_reference = "/opt/huawei/dataset/" + archive.relative_to(self.root).as_posix()
        self.parquet([[original_reference + "::train-uuid"]])
        self.parquet([[original_reference + "::validation-uuid"]], validation)
        arguments = [
            str(SCRIPT), "--source-root", str(self.source),
            "--destination-root", str(self.destination), "--local-prefix", str(self.root),
        ]

        with mock.patch.object(migration, "DEFAULT_PARQUET", str(self.input)):
            with mock.patch.object(migration, "DEFAULT_VALIDATION_PARQUET", str(validation)):
                with mock.patch.object(sys, "argv", arguments):
                    migration.main()

        expected_reference = (
            "/opt/huawei/dataset/hfy-infra-gy/data/" + archive.relative_to(self.source).as_posix()
        )
        for split, member in (("train", "train-uuid"), ("validation", "validation-uuid")):
            with self.subTest(split=split):
                output = pq.read_table(self.destination / (split + ".parquet"))
                self.assertEqual(output["audios"].to_pylist(), [[expected_reference + "::" + member]])
        copied = [path for path in (self.destination / "data").rglob("*") if path.is_file()]
        self.assertEqual(copied, [self.mapped(archive)])
        self.assertEqual(copied[0].read_bytes(), b"one shared audio archive")

    def test_missing_mapped_audio_reports_reference_and_local_path(self):
        old = "/opt/huawei/dataset/original audio/missing.zip::uuid"
        self.parquet([[old]])

        with self.assertRaises(FileNotFoundError) as caught:
            self.migrate(reference_prefix="/opt/huawei/dataset", local_prefix=self.root)

        self.assertIn(old, str(caught.exception))
        self.assertIn(str(self.source / "missing.zip"), str(caught.exception))
        self.assertFalse(self.destination.exists())

    def test_mapping_requires_directory_boundary(self):
        local_mount = self.root / "mount"
        local_mount.mkdir()
        self.source = self.root / "mount2" / "audios"
        archive = self.media()
        old = "/opt/huawei/dataset2/audios/" + archive.relative_to(self.source).as_posix()
        self.parquet([[old + "::uuid"]])

        with self.assertRaises((ValueError, FileNotFoundError)):
            self.migrate(reference_prefix="/opt/huawei/dataset", local_prefix=local_mount)

        self.assertFalse(self.destination.exists())

    def test_traversal_and_relative_output_root_are_rejected_before_copy(self):
        archive = self.media()
        self.parquet([["/opt/huawei/dataset/../original audio/" + archive.name]])
        with self.assertRaisesRegex(ValueError, "traverses"):
            self.migrate(reference_prefix="/opt/huawei/dataset", local_prefix=self.root)

        self.parquet([[archive.as_posix()]])
        with self.assertRaisesRegex(ValueError, "absolute POSIX"):
            self.migrate(output_reference_root="relative/output")
        self.assertFalse(self.destination.exists())

    def test_missing_source_is_detected_before_any_output_is_created(self):
        archive = self.media()
        missing = self.source / "missing.zip"
        self.parquet([[archive.as_posix()], [missing.as_posix() + "::uuid"]])

        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            self.migrate()

        self.assertFalse(self.destination.exists())

    def test_source_outside_declared_root_is_rejected_before_copy(self):
        archive = self.media()
        outside = self.root / "outside.zip"
        outside.write_bytes(b"outside source root")
        self.parquet([[archive.as_posix()], [outside.as_posix() + "::uuid"]])

        with self.assertRaises((ValueError, RuntimeError)):
            self.migrate()

        self.assertFalse(self.destination.exists())

    def test_remote_and_relative_references_are_not_silently_migrated(self):
        archive = self.media()
        for invalid in ("https://example.test/audio.zip::uuid", "relative/audio.wav"):
            with self.subTest(reference=invalid):
                self.parquet([[archive.as_posix()], [invalid]])
                with self.assertRaises((ValueError, RuntimeError)):
                    self.migrate()
                self.assertFalse(self.destination.exists())

    def test_existing_identical_media_is_reused(self):
        archive = self.media(content=b"identical bytes")
        target = self.mapped(archive)
        target.parent.mkdir(parents=True)
        target.write_bytes(archive.read_bytes())
        self.parquet([[archive.as_posix() + "::uuid"]])

        result = self.migrate()

        self.assertEqual(result["copied"], 0)
        self.assertEqual(result["reused"], 1)
        self.assertEqual(target.read_bytes(), b"identical bytes")
        self.assertTrue((self.destination / "train.parquet").is_file())

    def test_same_size_different_content_is_rejected_without_overwrite(self):
        archive = self.media(content=b"source")
        target = self.mapped(archive)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"target")
        self.parquet([[archive.as_posix()]])

        with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            self.migrate()

        self.assertEqual(target.read_bytes(), b"target")
        self.assertFalse((self.destination / "train.parquet").exists())
        self.assertEqual({path for path in self.destination.rglob("*") if path.is_file()}, {target})

    def test_existing_parquet_is_never_overwritten(self):
        archive = self.media()
        self.parquet([[archive.as_posix()]])
        self.destination.mkdir()
        output = self.destination / "train.parquet"
        output.write_bytes(b"existing output must survive")

        with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            self.migrate()

        self.assertEqual(output.read_bytes(), b"existing output must survive")
        self.assertFalse((self.destination / "data").exists())

    def test_mount_rejecting_links_still_copies_media_and_parquet(self):
        archive = self.media(content=b"audio on a mount without hard links")
        original = self.parquet([[archive.as_posix() + "::uuid"]])
        atomic_call = "rename" if migration.os.name == "nt" else "link"
        with mock.patch.object(migration.os, atomic_call, side_effect=OSError(errno.EPERM, "not permitted")) as rejected:
            result = self.migrate()

        self.assertEqual(rejected.call_count, 2)  # Media and final parquet both publish via fallback.
        self.assertEqual(result["copied"], 1)
        self.assertEqual(self.mapped(archive).read_bytes(), archive.read_bytes())
        output = pq.read_table(self.destination / "train.parquet")
        self.assertTrue(output.schema.equals(original.schema, check_metadata=True))
        self.assertEqual(output["audios"].to_pylist(), [[self.mapped(archive).as_posix() + "::uuid"]])
        self.assertEqual(output["prompt"].to_pylist(), original["prompt"].to_pylist())
        self.assertEqual(list(self.destination.rglob("*.partial")), [])

    def test_publish_copy_never_overwrites_or_deletes_existing_target(self):
        temporary = self.root / "ready.partial"
        target = self.root / "existing.zip"
        temporary.write_bytes(b"new bytes")
        target.write_bytes(b"existing bytes")
        atomic_call = "rename" if migration.os.name == "nt" else "link"
        with mock.patch.object(migration.os, atomic_call, side_effect=OSError(errno.EOPNOTSUPP, "unsupported")):
            with self.assertRaises(FileExistsError):
                migration._publish_new(temporary, target)
        self.assertEqual(target.read_bytes(), b"existing bytes")
        self.assertEqual(temporary.read_bytes(), b"new bytes")

    def test_failed_or_interrupted_publish_copy_cleans_own_target(self):
        temporary = self.root / "ready.partial"
        target = self.root / "result.zip"
        temporary.write_bytes(b"verified bytes")
        for failure in (OSError(errno.ENOSPC, "disk full"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                with mock.patch.object(migration.os, "fsync", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        migration._publish_copy(temporary, target)
                self.assertFalse(target.exists())
                self.assertEqual(temporary.read_bytes(), b"verified bytes")

    def test_publish_copy_checksum_mismatch_is_rejected_and_cleaned(self):
        temporary = self.root / "ready.partial"
        target = self.root / "result.zip"
        temporary.write_bytes(b"verified bytes")
        with mock.patch.object(migration, "_digest", return_value=b"wrong hash"):
            with self.assertRaisesRegex(IOError, "SHA256"):
                migration._publish_copy(temporary, target)
        self.assertFalse(target.exists())
        self.assertTrue(temporary.is_file())

    def test_disk_full_does_not_trigger_publish_fallback(self):
        temporary = self.root / "ready.partial"
        target = self.root / "result.zip"
        temporary.write_bytes(b"verified bytes")
        atomic_call = "rename" if migration.os.name == "nt" else "link"
        with mock.patch.object(migration.os, atomic_call, side_effect=OSError(errno.ENOSPC, "disk full")):
            with mock.patch.object(migration, "_publish_copy") as fallback:
                with self.assertRaises(OSError) as caught:
                    migration._publish_new(temporary, target)
                self.assertEqual(caught.exception.errno, errno.ENOSPC)
                fallback.assert_not_called()
        self.assertFalse(target.exists())

    def test_duplicate_output_basenames_are_rejected_before_copy(self):
        archive = self.media()
        other = self.root / "different input" / "train.parquet"
        self.parquet([[archive.as_posix()]])
        self.parquet([[archive.as_posix()]], other)

        with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            self.migrate([self.input, other])

        self.assertFalse(self.destination.exists())

    def test_embedded_audio_bytes_and_struct_metadata_are_preserved(self):
        archive = self.media()
        schema = pa.schema([
            pa.field("audios", pa.list_(pa.struct([
                pa.field("path", pa.string()), pa.field("bytes", pa.binary()),
                pa.field("sampling_rate", pa.int32()),
            ]))),
            pa.field("ground_truth", pa.string()),
        ], metadata={b"format": b"embedded-or-path"})
        original = pa.Table.from_pylist([
            {"audios": [{"path": None, "bytes": b"embedded waveform", "sampling_rate": 16000}], "ground_truth": "hello"},
            {"audios": [{"path": archive.as_posix() + "::uuid", "bytes": None, "sampling_rate": 16000}], "ground_truth": "world"},
        ], schema=schema)
        self.input.parent.mkdir(parents=True)
        pq.write_table(original, self.input)
        original = pq.read_table(self.input)

        result = self.migrate()

        migrated = pq.read_table(self.destination / "train.parquet")
        self.assertEqual(result["unique_files"], 1)
        self.assertTrue(migrated.schema.equals(original.schema, check_metadata=True))
        values = migrated["audios"].to_pylist()
        self.assertEqual(values[0], original["audios"].to_pylist()[0])
        self.assertEqual(values[1], [{"path": self.mapped(archive).as_posix() + "::uuid", "bytes": None, "sampling_rate": 16000}])
        self.assertEqual(migrated["ground_truth"].to_pylist(), ["hello", "world"])


class AudioValueRewriteTests(unittest.TestCase):
    def test_supported_wrappers_preserve_unrelated_fields(self):
        for key in ("audio", "audio_url", "path", "url"):
            with self.subTest(wrapper=key):
                value = {key: ["/original.zip::uuid"], "duration": 12.5}
                rewritten = migration.rewrite_audio(value, lambda path: path.replace("/original", "/new"))
                self.assertEqual(rewritten, {key: ["/new.zip::uuid"], "duration": 12.5})
                self.assertEqual(value, {key: ["/original.zip::uuid"], "duration": 12.5})

    def test_embedded_bytes_do_not_reach_path_transform(self):
        def forbidden(_):
            self.fail("Embedded bytes must not be interpreted as a path")

        for value in (b"wave bytes", {"bytes": b"wave bytes", "path": None}, None, []):
            with self.subTest(value=value):
                self.assertEqual(migration.rewrite_audio(value, forbidden), value)

    def test_unknown_nonempty_audio_values_raise(self):
        for value in (123, {"unrecognized_media_field": "/original.wav"}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    migration.rewrite_audio(value, lambda path: path)


if __name__ == "__main__":
    unittest.main()
