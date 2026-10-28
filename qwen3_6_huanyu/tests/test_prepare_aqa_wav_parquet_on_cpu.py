"""Exercise WAV-backed AQA subsets using real tiny WAVs and Parquet files."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_spec = importlib.util.spec_from_file_location("aqa_wav_parquet_fixture", EXAMPLES / "prepare_aqa_wav_parquet.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


class PrepareAqaWavParquetTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("HUANYU_TEST_TMPDIR", tempfile.gettempdir())).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="aqa_wav_parquet_", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "validation.parquet"
        self.export = self.root / "export"
        self.export.mkdir()
        self.output = self.root / "subset.parquet"

    def fixture(self, count=2, dict_items=False, audios_per_row=1, materialize_indices=None):
        rows, records = [], []
        for index in range(count):
            sample_id = "row_{:06d}".format(index)
            sample_dir = self.export / "samples" / sample_id
            materialize = materialize_indices is None or index in materialize_indices
            if materialize:
                sample_dir.mkdir(parents=True)
            items, entries = [], []
            for audio_index in range(audios_per_row):
                reference = str(self.root / "absent.zarr.zip") + "::{}_{}".format(index, audio_index)
                key = "audio" if audio_index % 2 == 0 else "audio_url"
                items.append({"audio": None, "audio_url": None, key: reference,
                              "bytes": None, "sampling_rate": 16000, "label": "keep"}
                             if dict_items else reference)
                wav = sample_dir / "audio_{:02d}.wav".format(audio_index)
                if materialize:
                    sf.write(str(wav), np.linspace(-0.25, 0.25, 16 + audio_index, dtype=np.float32),
                             16000, subtype="PCM_16")
                entries.append({"audio_index": audio_index, "source_reference": reference,
                                "wav_path": wav.relative_to(self.export).as_posix(),
                                "sample_rate": 16000, "num_frames": 16 + audio_index, "channels": 1})
            prompt = [{"role": "user", "content": "Question {} {}".format(index, "<audio>" * audios_per_row)}]
            answer = "answer_{}".format(index)
            rows.append({"prompt": prompt, "audios": items, "images": [], "videos": [],
                         "reward_model": {"style": "model", "ground_truth": answer},
                         "data_source": "omni_mix", "ability": "audio",
                         "extra_info": {"raw_source": "aqa_real", "index": index, "split": "validation"},
                         "untouched": {"tags": ["one", "two"], "score": 1.5}})
            records.append({"row_index": index, "sample_id": sample_id, "source_parquet": str(self.source),
                            "prompt": prompt, "ground_truth": answer, "audios": entries})
        table = pa.Table.from_pylist(rows).replace_schema_metadata({b"fixture": b"preserve-metadata"})
        field_index = table.schema.get_field_index("audios")
        field = table.schema.field("audios").with_metadata({b"purpose": b"raw-audio"})
        table = table.set_column(field_index, field, table.column("audios"))
        pq.write_table(table, self.source, row_group_size=3)
        self.write_manifest(records)
        return rows, records

    def write_manifest(self, records):
        (self.export / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def report(self, output=None):
        return json.loads((output or self.output).with_suffix(".selection.json").read_text(encoding="utf-8"))

    def test_random_100_is_reproducible_and_preserves_only_selected_rows(self):
        expected = sorted(random.Random(42).sample(range(2100), 100))
        other = sorted(random.Random(43).sample(range(2100), 100))
        rows, records = self.fixture(count=2100, materialize_indices=set(expected) | set(other))
        original = self.source.read_bytes()
        wav_before = {entry["wav_path"]: (self.export / entry["wav_path"]).read_bytes()
                      for record in records if record["row_index"] in set(expected) | set(other)
                      for entry in record["audios"]}
        # Only the selected rows need a manifest entry or a materialized WAV.
        self.write_manifest([records[index] for index in expected])
        self.assertFalse((self.root / "absent.zarr.zip").exists())
        self.assertEqual(prepare.convert_parquet(self.source, self.export, self.output,
                                                sample_size=100, seed=42, batch_size=17), 100)
        self.assertNotEqual(expected, list(range(100)))
        actual = pq.read_table(self.output)
        self.assertTrue(actual.schema.equals(pq.read_schema(self.source), check_metadata=True))
        self.assertEqual(actual.num_rows, 100)
        report = self.report()
        self.assertEqual(report["selected_rows"], 100)
        self.assertEqual([row["source_row_index"] for row in report["rows"]], expected)
        for output_index, (result, source_index) in enumerate(zip(actual.to_pylist(), expected)):
            source = copy.deepcopy(rows[source_index])
            source["audios"] = [str((self.export / records[source_index]["audios"][0]["wav_path"]).resolve())]
            self.assertEqual(result, source)
            audit = report["rows"][output_index]
            self.assertEqual(audit["output_row_index"], output_index)
            self.assertEqual(audit["audios"][0]["sha256"], hashlib.sha256(Path(result["audios"][0]).read_bytes()).hexdigest())
        repeated = self.root / "repeated.parquet"
        prepare.convert_parquet(self.source, self.export, repeated, sample_size=100, seed=42, batch_size=31)
        self.assertEqual(pq.read_table(repeated).to_pylist(), actual.to_pylist())
        different = self.root / "different.parquet"
        self.write_manifest([records[index] for index in other])
        prepare.convert_parquet(self.source, self.export, different, sample_size=100, seed=43)
        self.assertNotEqual([row["source_row_index"] for row in self.report(different)["rows"]], expected)
        self.assertEqual(self.source.read_bytes(), original)
        for relative, contents in wav_before.items():
            self.assertEqual((self.export / relative).read_bytes(), contents)

    def test_multiple_dict_audios_keep_order_and_unrelated_fields(self):
        rows, records = self.fixture(dict_items=True, audios_per_row=2)
        self.assertEqual(prepare.convert_parquet(self.source, self.export, self.output, batch_size=1), 2)
        actual = pq.read_table(self.output)
        self.assertTrue(actual.schema.equals(pq.read_schema(self.source), check_metadata=True))
        expected = copy.deepcopy(rows)
        for index, row in enumerate(expected):
            for audio_index, item in enumerate(row["audios"]):
                key = "audio" if audio_index == 0 else "audio_url"
                item[key] = str((self.export / records[index]["audios"][audio_index]["wav_path"]).resolve())
        self.assertEqual(actual.to_pylist(), expected)

    def test_manifest_mismatch_after_first_written_batch_cleans_outputs(self):
        _, records = self.fixture()
        mutations = {
            "prompt": lambda record: record.update(prompt=[{"role": "user", "content": "wrong"}]),
            "truth": lambda record: record.update(ground_truth="wrong"),
            "source": lambda record: record.update(source_parquet=str(self.root / "another.parquet")),
            "audio_reference": lambda record: record["audios"][0].update(source_reference="wrong"),
            "audio_order": lambda record: record["audios"][0].update(audio_index=1),
            "missing_audio": lambda record: record.update(audios=[]),
            "cross_row": lambda record: record["audios"][0].update(wav_path=records[0]["audios"][0]["wav_path"]),
            "rate": lambda record: record["audios"][0].update(sample_rate=8000),
            "frames": lambda record: record["audios"][0].update(num_frames=1),
            "channels": lambda record: record["audios"][0].update(channels=2),
        }
        original = self.source.read_bytes()
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(records)
                mutate(changed[1])
                self.write_manifest(changed)
                with self.assertRaises((ValueError, OSError)):
                    prepare.convert_parquet(self.source, self.export, self.output, batch_size=1)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.output.with_suffix(".selection.json").exists())
        self.assertEqual(self.source.read_bytes(), original)

    def test_partial_export_requires_explicit_limit_and_rejects_bad_manifest_indices(self):
        rows, records = self.fixture(count=4)
        self.write_manifest(records[:2])
        with self.assertRaises(ValueError):
            prepare.convert_parquet(self.source, self.export, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(prepare.convert_parquet(self.source, self.export, self.output, limit=2), 2)
        self.assertEqual([row["extra_info"]["index"] for row in pq.read_table(self.output).to_pylist()], [0, 1])
        for altered in (records + [records[0]], [dict(records[0], row_index=-1)],
                        [dict(records[0], sample_id="row_999999")]):
            self.write_manifest(altered)
            with self.assertRaises(ValueError):
                prepare.convert_parquet(self.source, self.export, self.root / "bad.parquet")
            self.assertFalse((self.root / "bad.parquet").exists())

    def test_existing_destinations_and_invalid_selection_never_overwrite(self):
        self.fixture()
        original = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.convert_parquet(self.source, self.export, self.source)
        for destination in (self.output, self.output.with_suffix(".selection.json")):
            with self.subTest(path=destination):
                destination.write_bytes(b"do not replace")
                with self.assertRaises(FileExistsError):
                    prepare.convert_parquet(self.source, self.export, self.output)
                self.assertEqual(destination.read_bytes(), b"do not replace")
                destination.unlink()
        for options in ({"sample_size": 3}, {"sample_size": 0}, {"sample_size": 1, "limit": 1},
                        {"limit": -1}, {"batch_size": 0}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    prepare.convert_parquet(self.source, self.export, self.output, **options)
                self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), original)

    def test_invalid_wav_contents_or_missing_wav_are_rejected_without_partial_output(self):
        _, records = self.fixture()
        wav = self.export / records[1]["audios"][0]["wav_path"]
        for kind in ("flac", "nonfinite", "missing"):
            with self.subTest(kind=kind):
                if kind == "flac":
                    sf.write(str(wav), np.zeros(16, dtype=np.float32), 16000, format="FLAC")
                elif kind == "nonfinite":
                    sf.write(str(wav), np.full(16, np.nan, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
                else:
                    wav.unlink()
                with self.assertRaises((ValueError, OSError, RuntimeError)):
                    prepare.convert_parquet(self.source, self.export, self.output, batch_size=1)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.output.with_suffix(".selection.json").exists())


if __name__ == "__main__":
    unittest.main()
