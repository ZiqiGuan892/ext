"""Strict prompt comparison must reject missing provenance and partial captures."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prompt_embedding_compare_cpu", ROOT / "examples/compare_huanyu_prompt_embeddings.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class PromptEmbeddingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [{"sample_id": "row_000655", "case_id": "case-1", "prompt": [{"role": "user", "content": "<audio>"}],
                      "wav_sha256": ["a" * 64]}]
        self.case = {"schema_version": 1, "case_id": "case-1", "sample_size": 1,
                     "engine": {"tensor_parallel_size": 2}}
        self.records = {}
        for side in compare.SIDES:
            for rank in range(2):
                self.write(side, rank)

    def write(self, side, rank, values=None, dtype="torch.float32", changes=None, arrays_changes=None, suffix=""):
        directory = self.root / "embeddings" / side / "worker"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("row_000655_rank%d%s.json" % (rank, suffix))
        if values is None:
            values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        if dtype == "torch.bfloat16":
            storage = (np.asarray(values, dtype="<f4").view("<u4") >> 16).astype("<u2")
        else:
            storage = np.asarray(values, dtype=compare.DTYPES[dtype][0])
        shape = storage.shape
        itemsize = compare.DTYPES[dtype][1]
        raw = storage.view(np.uint8).reshape(shape + (itemsize,))
        arrays = {"embedding_bytes": raw, "input_ids": np.asarray([10, 11, 12], dtype=np.int64),
                  "positions": np.asarray([0, 1, 2], dtype=np.int64),
                  "audio_mask": np.asarray([False, True, False], dtype=bool)}
        if arrays_changes:
            arrays.update(arrays_changes)
        tensor_path = path.with_suffix(".npz")
        np.savez(tensor_path, **arrays)
        record = {"format": compare.FORMAT, "case_id": "case-1", "sample_id": "row_000655",
                  "side": side, "tp_rank": rank, "tp_size": 2, "status": "complete",
                  "request_id": "chatcmpl-huanyuab.case-1.row_000655-stage0",
                  "prompt_sha256": compare._canonical_hash(self.rows[0]["prompt"]), "wav_sha256": ["a" * 64],
                  "tensor_file": tensor_path.name, "tensor_file_sha256": compare._sha(tensor_path),
                  "embeddings": {"dtype": dtype, "shape": list(shape), "sha256": hashlib.sha256(raw.tobytes()).hexdigest()},
                  "capture_scope": "llm_inputs_embeds", "prompt_token_count": 3, "captured_token_count": 3,
                  "num_computed_tokens": 0, "prefill_complete": True}
        if changes:
            record.update(changes)
        path.write_text(json.dumps(record), encoding="utf-8")
        self.records[side, rank] = path
        return path

    def result(self):
        return compare.compare_case_embeddings(self.root, self.case, self.rows)

    def test_complete_all_ranks_strictly_equal(self):
        report = self.result()
        self.assertEqual(report["summary"]["status"], "PASS")
        self.assertTrue(report["samples"][0]["strict_equal"])
        self.assertEqual(len(report["samples"][0]["ranks"]), 2)

    def test_missing_one_rank_is_incomplete(self):
        self.records["rollout", 1].unlink()
        result = self.result()
        self.assertEqual(result["summary"]["status"], "INCOMPLETE")
        self.assertIsNone(result["samples"][0]["strict_equal"])

    def test_missing_all_records_is_not_equal(self):
        for path in self.records.values():
            path.unlink()
        result = self.result()
        self.assertEqual(result["summary"]["equal_samples"], 0)
        self.assertEqual(result["summary"]["incomplete_samples"], 1)

    def test_one_bit_difference_is_fail_even_with_tiny_numeric_error(self):
        values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        values[1, 0] = np.nextafter(values[1, 0], np.float32(4))
        self.write("rollout", 1, values)
        report = self.result()
        self.assertEqual(report["summary"]["status"], "FAIL")
        rank = report["samples"][0]["ranks"][1]
        self.assertEqual(rank["first_different_element"], [1, 0])
        self.assertEqual(rank["audio"]["mismatched_elements"], 1)
        self.assertEqual(rank["text_and_special_tokens"]["mismatched_elements"], 0)
        self.assertGreater(rank["max_abs_difference"], 0)

    def test_bfloat16_decodes_without_torch_and_is_exact(self):
        for side in compare.SIDES:
            for rank in range(2):
                self.write(side, rank, dtype="torch.bfloat16")
        self.assertEqual(self.result()["summary"]["status"], "PASS")
        self.write("rollout", 1, values=[[1, 2], [3.125, 4], [5, 6]], dtype="torch.bfloat16")
        rank = self.result()["samples"][0]["ranks"][1]
        self.assertEqual(rank["max_abs_difference"], 0.125)

    def test_same_values_different_dtype_is_not_strictly_equal(self):
        for rank in range(2):
            self.write("rollout", rank, dtype="torch.float64")
        report = self.result()
        self.assertEqual(report["summary"]["status"], "FAIL")
        self.assertEqual(report["samples"][0]["ranks"][0]["max_abs_difference"], 0)

    def test_signed_zero_difference_does_not_hide_behind_numeric_equality(self):
        values = np.ones((3, 2), dtype=np.float32)
        values[0, 0] = 0
        for side in compare.SIDES:
            for rank in range(2):
                self.write(side, rank, values=values)
        values[0, 0] = -0.0
        self.write("rollout", 1, values=values)
        rank = self.result()["samples"][0]["ranks"][1]
        self.assertFalse(rank["strict_equal"])
        self.assertEqual(rank["max_abs_difference"], 0)

    def test_token_positions_and_audio_placement_are_separate_checks(self):
        for key, value in (("input_ids", np.asarray([10, 13, 12], dtype=np.int64)),
                           ("positions", np.asarray([0, 2, 3], dtype=np.int64)),
                           ("audio_mask", np.asarray([True, False, False]))):
            with self.subTest(key=key):
                self.write("rollout", 1, arrays_changes={key: value})
                report = self.result()
                self.assertEqual(report["summary"]["status"], "FAIL")
                self.assertTrue(report["samples"][0]["ranks"][1]["embedding_bytes_equal"])

    def test_equal_mrope_positions_are_supported(self):
        for side in compare.SIDES:
            for rank in range(2):
                self.write(side, rank, arrays_changes={"positions": np.asarray([[0, 1, 2]] * 3, dtype=np.int64)})
        self.assertEqual(self.result()["summary"]["status"], "PASS")

    def test_duplicate_capture_is_incomplete(self):
        self.write("serve", 0, suffix="_duplicate")
        self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_unknown_sample_is_visible_and_prevents_global_pass(self):
        self.write("serve", 0, suffix="_foreign", changes={"sample_id": "row_999999"})
        result = self.result()
        self.assertEqual(result["summary"]["status"], "INCOMPLETE")
        self.assertEqual(len(result["summary"]["global_issues"]), 1)

    def test_incomplete_record_and_partial_prefill_never_pass(self):
        cases = [{"status": "INCOMPLETE", "error": "unsupported runtime"}, {"prefill_complete": False},
                 {"captured_token_count": 2}, {"prompt_token_count": 4}, {"num_computed_tokens": 1},
                 {"num_computed_tokens": False}, {"capture_scope": "get_audio_hidden_states"}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.write("serve", 0, changes=changes)
                self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_wrong_case_wav_prompt_side_or_request_is_incomplete(self):
        cases = [{"case_id": "old-case"}, {"wav_sha256": ["b" * 64]}, {"prompt_sha256": "c" * 64},
                 {"side": "rollout"}, {"request_id": "huanyuab.case-1.row_0006550"},
                 {"request_id": "untagged"}, {"tp_rank": 9}, {"tp_size": 4}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.write("serve", 0, changes=changes)
                self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_same_side_different_requests_cannot_form_rank_set(self):
        self.write("serve", 1, changes={"request_id": "chatcmpl-huanyuab.case-1.row_000655-retry"})
        self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_corrupt_or_missing_tensor_file_is_incomplete(self):
        tensor = self.records["serve", 0].with_suffix(".npz")
        with tensor.open("ab") as stream:
            stream.write(b"changed")
        self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")
        tensor.unlink()
        self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_npz_bytes_hash_and_metadata_are_validated(self):
        for changes in ({"embeddings": {"dtype": "torch.float32", "shape": [3, 2], "sha256": "a" * 64}},
                        {"embeddings": None}, {"tensor_file": "../outside.npz"}):
            with self.subTest(changes=changes):
                self.write("serve", 0, changes=changes)
                self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_invalid_archive_with_matching_file_hash_is_incomplete(self):
        path = self.records["serve", 0]
        tensor = path.with_suffix(".npz")
        tensor.write_bytes(b"PK\x03\x04broken ZIP archive")
        record = json.loads(path.read_text())
        record["tensor_file_sha256"] = compare._sha(tensor)
        path.write_text(json.dumps(record))
        self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_nonfinite_embeddings_never_pass(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.write("serve", 0, values=[[value, 2], [3, 4], [5, 6]])
                self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_invalid_npz_shapes_and_dtypes_never_pass(self):
        cases = [{"input_ids": np.asarray([1, 2, 3], dtype=np.int32)},
                 {"positions": np.asarray([[0, 1, 2]], dtype=np.int64)},
                 {"audio_mask": np.zeros(3, dtype=bool)},
                 {"embedding_bytes": np.zeros((3, 2), dtype=np.uint8)}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.write("serve", 0, arrays_changes=changes)
                self.assertEqual(self.result()["summary"]["status"], "INCOMPLETE")

    def test_all_samples_are_required_not_only_captured_subset(self):
        row = copy.deepcopy(self.rows[0])
        row["sample_id"] = "row_000418"
        self.rows.append(row)
        self.case["sample_size"] = 2
        result = self.result()
        self.assertEqual(result["summary"]["equal_samples"], 1)
        self.assertEqual(result["summary"]["incomplete_samples"], 1)
        self.assertEqual(result["summary"]["status"], "INCOMPLETE")

    def test_cli_writes_report_and_preserves_prior_outputs(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")
        self.case["manifest_sha256"] = compare._sha(manifest)
        (self.root / "case.json").write_text(json.dumps(self.case), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(["--case-dir", str(self.root)]), 0)
            original = (self.root / "embedding_summary.json").read_bytes()
            self.assertEqual(compare.main(["--case-dir", str(self.root)]), 2)
        self.assertEqual((self.root / "embedding_summary.json").read_bytes(), original)

    def test_tensors_are_released_between_samples(self):
        row = copy.deepcopy(self.rows[0])
        row["sample_id"] = "row_000418"
        self.rows.append(row)
        self.case["sample_size"] = 2
        for path in self.records.values():
            record = json.loads(path.read_text())
            record["sample_id"] = row["sample_id"]
            record["request_id"] = record["request_id"].replace("row_000655", "row_000418")
            path.with_name(path.name.replace("row_000655", "row_000418")).write_text(json.dumps(record))
        original = compare._load_arrays
        first_sample_arrays = []

        def load(path, record, root):
            if record["sample_id"] == "row_000418":
                self.assertTrue(first_sample_arrays)
                self.assertTrue(all(reference() is None for reference in first_sample_arrays))
            result = original(path, record, root)
            if record["sample_id"] == "row_000655":
                first_sample_arrays.append(weakref.ref(result["embedding_bytes"]))
            return result

        with patch.object(compare, "_load_arrays", load):
            self.assertEqual(self.result()["summary"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
