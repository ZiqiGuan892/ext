"""Export real tiny Zarr ZIP/Parquet fixtures without a model or serving endpoint."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from numcodecs import Zlib
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
# The scripts deliberately import their sibling helper instead of the training
# package, which would import the NPU runtime during CPU-only verification.
sys.path.insert(0, str(EXAMPLES))
try:
    _spec = importlib.util.spec_from_file_location("aqa_validation_export_fixture", EXAMPLES / "export_aqa_validation_wav.py")
    export = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(export)
finally:
    sys.path.pop(0)


class ValidationAudioExportTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("HUANYU_TEST_TMPDIR", tempfile.gettempdir())).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="aqa_export_", dir=base)
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, base)
        self.addCleanup(self.temporary.cleanup)
        self.source = self.root / "validation.parquet"
        self.output = self.root / "export"

    @staticmethod
    def write_array(archive, name, array, scalar=False):
        array = np.asarray(array)
        codec = Zlib(level=1)
        chunk_size = 8 if array.size > 1 else 1
        metadata = {
            "zarr_format": 2,
            "shape": [] if scalar else [array.size],
            "chunks": [] if scalar else [chunk_size],
            "dtype": array.dtype.str,
            "compressor": codec.get_config(),
            "filters": None,
            "order": "C",
            "fill_value": 0,
        }
        archive.writestr(name + "/.zarray", json.dumps(metadata))
        flat = array.reshape(-1)
        for index, start in enumerate(range(0, len(flat), chunk_size)):
            chunk = np.zeros(chunk_size, dtype=array.dtype)
            values = flat[start:start + chunk_size]
            chunk[:len(values)] = values
            archive.writestr(name + "/" + str(index), codec.encode(chunk.tobytes()))

    def archive(self, name, key, rate=16000, rate_kind="attrs", amplitude=0.25):
        path = self.root / name
        values = np.linspace(-amplitude, amplitude, 13, dtype=np.float32)
        group = key if key.endswith(".wav") else key + ".wav"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(".zgroup", '{"zarr_format": 2}')
            archive.writestr(group + "/.zgroup", '{"zarr_format": 2}')
            self.write_array(archive, group + "/audio", values)
            if rate_kind == "attrs":
                # AQA group attributes take precedence over a stale array.
                archive.writestr(group + "/.zattrs", json.dumps({"sampling_rate": rate}))
                self.write_array(archive, group + "/sample_rate", np.array([8000], dtype="<i8"))
            elif rate_kind in ("array", "scalar"):
                self.write_array(archive, group + "/sample_rate", np.array([rate], dtype="<i8"), scalar=rate_kind == "scalar")
            elif rate_kind == "sample_rate_attr":
                archive.writestr(group + "/.zattrs", json.dumps({"sample_rate": rate}))
            elif rate_kind != "missing":
                raise AssertionError(rate_kind)
        return str(path) + "::" + key, values

    @staticmethod
    def row(audios, answer="这是独立保存的参考答案", text=None):
        return {
            "data_source": "omni_mix",
            "prompt": [
                {"role": "system", "content": "请根据音频回答。"},
                {"role": "user", "content": text or "逐段听：" + "，然后".join(["<audio>"] * len(audios)) + "。描述声音。"},
            ],
            "audios": audios,
            "images": [],
            "videos": [],
            "ability": "audio",
            "reward_model": {"style": "model", "ground_truth": answer},
            "extra_info": {"raw_source": "aqa_real", "problem_id": "duplicate-id-is-allowed"},
        }

    def parquet(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.source, row_group_size=1)
        return self.source.read_bytes()

    def manifest(self):
        return [json.loads(line) for line in (self.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def summary(self):
        return json.loads((self.output / "summary.json").read_text(encoding="utf-8"))

    def test_rows_multiple_audio_and_requests_preserve_correspondence(self):
        first, audio_a = self.archive("甲.zarr.zip", "same-key", 16000)
        second, audio_b = self.archive("乙.zarr.zip", "same-key", 24000, "array", 0.5)
        third, audio_c = self.archive("丙.zarr.zip", "third.wav", 22050, "scalar", 0.125)
        rows = [self.row([first, second], "REFERENCE_ONLY_A"), self.row([third], "REFERENCE_ONLY_B")]
        before = self.parquet(rows)
        self.assertEqual(export.export_validation(self.source, self.output, batch_size=1, model="huanyu-test", max_tokens=17), 2)
        self.assertEqual(self.source.read_bytes(), before)
        records = self.manifest()
        self.assertEqual([record["row_index"] for record in records], [0, 1])
        self.assertEqual([record["sample_id"] for record in records], ["row_000000", "row_000001"])
        for index, (record, row) in enumerate(zip(records, rows)):
            self.assertEqual(record["prompt"], row["prompt"])
            self.assertEqual(record["ground_truth"], row["reward_model"]["ground_truth"])
            self.assertEqual([audio["source_reference"] for audio in record["audios"]], row["audios"])
            sample_dir = self.output / "samples" / record["sample_id"]
            self.assertEqual(json.loads((sample_dir / "sample.json").read_text(encoding="utf-8")), record)
            self.assertEqual(json.loads((sample_dir / "prompt.json").read_text(encoding="utf-8")), row["prompt"])
            self.assertEqual((sample_dir / "ground_truth.txt").read_text(encoding="utf-8"), record["ground_truth"])
            request_text = (self.output / record["request_path"]).read_text(encoding="utf-8")
            request = json.loads(request_text)
            self.assertNotIn("REFERENCE_ONLY", request_text)
            self.assertEqual(request["model"], "huanyu-test")
            self.assertEqual(request["max_tokens"], 17)
            self.assertEqual(request["messages"][0], row["prompt"][0])
            parts = request["messages"][1]["content"]
            payloads = [part["input_audio"] for part in parts if part["type"] == "input_audio"]
            reconstructed = "".join(part["text"] if part["type"] == "text" else "<audio>" for part in parts)
            self.assertEqual(reconstructed, row["prompt"][1]["content"])
            self.assertEqual(len(payloads), len(record["audios"]))
            expected = [(audio_a, 16000), (audio_b, 24000)] if index == 0 else [(audio_c, 22050)]
            for position, (audio, payload, (waveform, rate)) in enumerate(zip(record["audios"], payloads, expected)):
                wav_path = self.output / audio["wav_path"]
                self.assertEqual(audio["wav_path"], "samples/{}/audio_{:02d}.wav".format(record["sample_id"], position))
                self.assertEqual(base64.b64decode(payload["data"]), wav_path.read_bytes())
                self.assertEqual(payload["format"], "wav")
                decoded, actual_rate = sf.read(wav_path, dtype="float32")
                np.testing.assert_allclose(decoded, waveform, rtol=0, atol=1 / 32768)
                self.assertEqual(actual_rate, rate)
                self.assertEqual(audio["sample_rate"], rate)
                self.assertEqual(audio["num_frames"], len(waveform))
                self.assertEqual(audio["channels"], 1)
                self.assertAlmostEqual(audio["duration_seconds"], len(waveform) / rate)
                self.assertEqual(sf.info(wav_path).subtype, "PCM_16")
        self.assertEqual(self.summary()["status"], "complete")
        self.assertEqual(self.summary()["exported_rows"], 2)

    def test_limit_keeps_original_row_number_and_does_not_read_later_bad_audio(self):
        first, _ = self.archive("good.zarr.zip", "first", rate_kind="sample_rate_attr")
        self.parquet([self.row([first]), self.row([str(self.root / "absent.zarr.zip") + "::later"])])
        self.assertEqual(export.export_validation(self.source, self.output, limit=1, batch_size=1), 1)
        self.assertEqual(self.manifest()[0]["row_index"], 0)
        self.assertNotIn("request_path", self.manifest()[0])
        self.assertFalse((self.output / "samples/row_000000/request.json").exists())
        self.assertEqual(self.summary()["source_rows"], 2)
        self.assertEqual(self.summary()["selected_rows"], 1)

    def test_missing_key_stops_at_the_right_row_and_does_not_claim_complete(self):
        first, _ = self.archive("first.zarr.zip", "present")
        missing = first.split("::", 1)[0] + "::not-there"
        self.parquet([self.row([first]), self.row([missing])])
        with self.assertRaises(RuntimeError):
            export.export_validation(self.source, self.output)
        self.assertEqual([record["row_index"] for record in self.manifest()], [0])
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["exported_rows"], 1)
        error = json.loads((self.output / "error.json").read_text(encoding="utf-8"))
        self.assertEqual(error["row_index"], 1)
        self.assertEqual(error["audio_index"], 0)
        self.assertEqual(error["source_reference"], missing)
        self.assertIn("not-there", error["message"])

    def test_existing_export_is_never_overwritten(self):
        first, _ = self.archive("source.zarr.zip", "first")
        self.parquet([self.row([first])])
        self.output.mkdir()
        marker = self.output / "manifest.jsonl"
        marker.write_bytes(b"keep this export")
        with self.assertRaises(FileExistsError):
            export.export_validation(self.source, self.output)
        self.assertEqual(marker.read_bytes(), b"keep this export")

    def test_bad_second_reference_does_not_report_the_first_audio_path(self):
        first, _ = self.archive("first.zarr.zip", "present")
        self.parquet([self.row([first, "invalid-relative.zip::key"])])
        with self.assertRaises(RuntimeError):
            export.export_validation(self.source, self.output)
        error = json.loads((self.output / "error.json").read_text(encoding="utf-8"))
        self.assertEqual(error["row_index"], 0)
        self.assertEqual(error["audio_index"], 1)
        self.assertIsNone(error["source_reference"])
        self.assertIn("invalid-relative.zip", error["message"])
        self.assertEqual(self.manifest(), [])

    def test_zarr_nonfinite_audio_is_rejected_before_pcm_encoding(self):
        path = self.root / "nonfinite.zarr.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("bad.wav/.zgroup", '{"zarr_format": 2}')
            archive.writestr("bad.wav/.zattrs", '{"sampling_rate": 16000}')
            self.write_array(archive, "bad.wav/audio", np.array([0, np.nan, np.inf], dtype="<f4"))
        self.parquet([self.row([str(path) + "::bad"])])
        with self.assertRaises(RuntimeError):
            export.export_validation(self.source, self.output)
        self.assertEqual(self.manifest(), [])
        self.assertIn("finite", self.summary()["error"]["message"])

    def test_absent_sample_rate_fails_instead_of_assuming_16khz(self):
        reference, _ = self.archive("missing-rate.zarr.zip", "first", rate_kind="missing")
        self.parquet([self.row([reference])])
        with self.assertRaises(RuntimeError):
            export.export_validation(self.source, self.output)
        self.assertEqual(self.manifest(), [])
        self.assertEqual(self.summary()["status"], "failed")
        error = json.loads((self.output / "error.json").read_text(encoding="utf-8"))
        self.assertIn("sample_rate", error["message"])

    def test_request_placeholder_mismatch_does_not_silently_drop_audio(self):
        reference, _ = self.archive("audio.zarr.zip", "first")
        self.parquet([self.row([reference], text="这条问题没有占位符。")])
        with self.assertRaises(RuntimeError):
            export.export_validation(self.source, self.output, model="huanyu-test")
        self.assertEqual(self.manifest(), [])
        self.assertEqual(self.summary()["status"], "failed")

    def test_direct_flac_is_converted_to_pcm16_wav_without_resampling(self):
        source = self.root / "source.flac"
        expected = np.linspace(-0.25, 0.25, 15, dtype=np.float32)
        sf.write(source, expected, 24000, format="FLAC", subtype="PCM_16")
        self.parquet([self.row([str(source)])])
        export.export_validation(self.source, self.output)
        wav = self.output / self.manifest()[0]["audios"][0]["wav_path"]
        info = sf.info(wav)
        self.assertEqual(info.format, "WAV")
        self.assertEqual(info.subtype, "PCM_16")
        self.assertEqual(info.samplerate, 24000)
        actual, _ = sf.read(wav, dtype="float32")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 32768)


if __name__ == "__main__":
    unittest.main()
