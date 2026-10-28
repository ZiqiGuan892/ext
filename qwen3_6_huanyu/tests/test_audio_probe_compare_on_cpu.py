"""Audio evidence comparison must not manufacture sample or call alignment."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "examples/compare_huanyu_audio_probe.py"
spec = importlib.util.spec_from_file_location("huanyu_audio_compare_cpu", SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def capture(*identities):
    metadata = {"format": compare.FORMAT, "status": "complete", "phase": "executed",
                "weight_reload_count": 0, "weights": {}, "audio_groups": [], "tensors": {}}
    arrays = {}
    for index, identity in enumerate(identities):
        feature_path, output_path = "features/%d" % index, "audio_embeddings/%d" % index
        metadata["audio_groups"].append({"feature_id": identity, "chunk_lengths": [3],
                                         "feature_paths": [feature_path], "output_path": output_path})
        arrays[feature_path] = np.arange(6, dtype=np.float32).reshape(2, 3)
        arrays[output_path] = np.arange(12, dtype=np.float32).reshape(3, 4)
    for index, (name, array) in enumerate(arrays.items()):
        metadata["tensors"][name] = {"shape": list(array.shape), "dtype": "torch.float32",
                                      "array_key": "tensor_%06d" % index}
    return metadata, arrays


def add_module_tensor(metadata, arrays, name, value):
    arrays[name] = np.asarray(value, dtype=np.float32)
    metadata["tensors"][name] = {"shape": list(arrays[name].shape), "dtype": "torch.float32"}


class TensorComparisonTests(unittest.TestCase):
    def test_real_probe_files_compare_projector_change_after_same_input(self):
        import test_audio_probe_on_cpu as recording
        import torch
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR")) as directory:
            env = {"HUANYU_AUDIO_PROBE_DIR": directory, "HUANYU_AUDIO_PROBE_MAX_CALLS": "2",
                   "HUANYU_AUDIO_PROBE_ARM_FILE": "", "HUANYU_AUDIO_PROBE_FEATURE_ID": ""}
            with patch.dict(os.environ, env):
                native = SimpleNamespace(thinker=recording.Thinker())
                recording.probe.attach_audio_probe(native)
                native.thinker.get_audio_hidden_states(recording.data())
                with torch.no_grad():
                    native.thinker.audio_projection_layer.bias.add_(1)
                native.thinker.get_audio_hidden_states(recording.data())
            left, la = compare.read_capture(next(Path(directory).rglob("forward_0001.json")))
            right, ra = compare.read_capture(next(Path(directory).rglob("forward_0002.json")))
            result = compare.compare(left, la, right, ra)
            self.assertTrue(all(group["same_valid_features"] for group in result["groups"]))
            self.assertTrue(result["module_tensors"]["encoder/0/output/0"]["values_exact"])
            self.assertFalse(result["module_tensors"]["projector/0/output"]["values_exact"])
            self.assertTrue(result["weights"]["projector"]["different_tensors"])
            self.assertFalse(result["weights"]["encoder"]["different_tensors"])

    def test_comparator_aligns_native_batched_and_isolated_valid_regions(self):
        import test_audio_probe_alignment_on_cpu as alignment

        def full_capture(partition):
            meta, arrays = alignment.capture(partition)
            chunks, reason = alignment.MODULE.align_native_chunks(meta, arrays)
            self.assertIsNone(reason)
            offset = 0
            for index, group in enumerate(meta["audio_groups"]):
                count = len(group["chunk_lengths"])
                group.update(feature_id="sample-%d" % index, feature_paths=[],
                             output_path="audio_embeddings/%d" % index)
                for j in range(count):
                    name = "features/%d" % (offset + j)
                    group["feature_paths"].append(name)
                    arrays[name] = chunks[offset + j]["encoder_input"]
                arrays[group["output_path"]] = np.concatenate(
                    [chunk["projector_output"] for chunk in chunks[offset:offset + count]])
                offset += count
            meta["tensors"] = {name: {"dtype": "torch.float64"} for name in arrays}
            return meta, arrays

        left, la = full_capture([3])
        right, ra = full_capture([1, 1, 1])
        result = compare.compare(left, la, right, ra)
        self.assertEqual(result["valid_chunk_alignment"], {"left": None, "right": None})
        self.assertIn("module_comparison", result)
        for group in result["groups"]:
            for chunk in group["valid_module_chunks"]:
                self.assertTrue(all(stage["values_exact"] for stage in chunk.values()))

    def test_exact_and_changed_embedding_values_have_numeric_evidence(self):
        left, la = capture("sample-a")
        right, ra = copy.deepcopy((left, la))
        exact = compare.compare(left, la, right, ra)["groups"][0]
        self.assertTrue(exact["same_valid_features"])
        self.assertTrue(exact["audio_embeddings"]["values_exact"])
        self.assertEqual(exact["audio_embeddings"]["max_abs"], 0)
        ra["audio_embeddings/0"][1, 2] += 2
        changed = compare.compare(left, la, right, ra)["groups"][0]
        self.assertTrue(changed["valid_feature_chunks"][0]["values_exact"])
        evidence = changed["audio_embeddings"]
        self.assertFalse(evidence["values_exact"])
        self.assertEqual(evidence["max_abs"], 2)
        self.assertAlmostEqual(evidence["mean_abs"], 2 / 12)
        self.assertEqual(evidence["worst_token_rows"][0]["index"], 1)

    def test_shape_mismatch_and_nonfinite_are_not_comparable(self):
        x = np.ones((2, 3))
        self.assertEqual(compare.tensor_difference(x, x.T)["reason"], "shape_mismatch")
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                y = x.copy()
                y[0, 0] = invalid
                result = compare.tensor_difference(x, y)
                self.assertFalse(result["comparable"])
                self.assertEqual(result["reason"], "nonfinite_values")
                self.assertNotIn("cosine", result)

    def test_zero_norm_and_empty_arrays_serialize_without_nan(self):
        for shape in ((0, 3), (2, 3)):
            with self.subTest(shape=shape):
                result = compare.tensor_difference(np.zeros(shape), np.zeros(shape))
                self.assertTrue(result["values_exact"])
                self.assertIsNone(result["cosine"])
                self.assertIsNone(result["relative_l2"])
                json.dumps(result, allow_nan=False)

    def test_valid_feature_comparison_ignores_padding_tail(self):
        left, la = capture("sample-a")
        right, ra = copy.deepcopy((left, la))
        ra["features/0"] = np.pad(ra["features/0"], ((0, 0), (0, 4)), constant_values=999)
        result = compare.compare(left, la, right, ra)
        self.assertTrue(result["groups"][0]["valid_feature_chunks"][0]["values_exact"])


class PairingTests(unittest.TestCase):
    def test_unique_feature_identity_matches_reordered_samples(self):
        left, la = capture("a", "b", "left-only")
        right, ra = capture("b", "a", "right-only")
        result = compare.compare(left, la, right, ra)
        self.assertEqual([(g["left_group"], g["right_group"]) for g in result["groups"]], [(0, 1), (1, 0)])
        self.assertEqual(result["unmatched_left_groups"], [2])
        self.assertEqual(result["unmatched_right_groups"], [2])
        self.assertNotIn("module_tensors", result)

    def test_no_known_matching_sample_refuses_automatic_pairing(self):
        left, la = capture("a", "b")
        right, ra = capture("x", "y")
        with self.assertRaisesRegex(ValueError, "No unique matching"):
            compare.compare(left, la, right, ra)

    def test_repeated_identical_samples_require_explicit_selection(self):
        left, la = capture("same", "same")
        right, ra = capture("same", "same")
        with self.assertRaisesRegex(ValueError, "No unique matching"):
            compare.compare(left, la, right, ra)
        selected = compare.compare(left, la, right, ra, left_group=1, right_group=0)
        self.assertEqual(selected["groups"][0]["left_group"], 1)
        self.assertEqual(selected["unmatched_left_groups"], [0])
        self.assertEqual(selected["unmatched_right_groups"], [1])

    def test_single_changed_feature_group_warns_instead_of_claiming_same_audio(self):
        left, la = capture("a")
        right, ra = capture("b")
        result = compare.compare(left, la, right, ra)["groups"][0]
        self.assertFalse(result["same_valid_features"])
        self.assertIn("cannot isolate encoder/projector", result["interpretation"])

    def test_explicit_selection_requires_two_valid_indices(self):
        left, la = capture("a")
        for indices in ((0, None), (None, 0), (-1, 0), (0, 1)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                compare.compare(left, la, left, la, *indices)

    def test_module_call_shape_partition_change_is_not_paired_by_index(self):
        left, la = capture("a")
        right, ra = copy.deepcopy((left, la))
        name = "encoder/0/input/args/0"
        add_module_tensor(left, la, name, np.ones((1, 3, 2)))
        add_module_tensor(right, ra, name, np.ones((3, 1, 2)))
        add_module_tensor(left, la, "encoder/0/output/0", np.ones((3, 4)))
        add_module_tensor(right, ra, "encoder/0/output/0", np.ones((3, 4)))
        result = compare.compare(left, la, right, ra)
        self.assertNotIn("module_tensors", result)
        self.assertIn("call partition", result["module_comparison"].lower())
        self.assertEqual(result["module_shapes"]["left"][name], [1, 3, 2])

    def test_module_call_count_change_is_not_paired_by_index(self):
        left, la = capture("a")
        right, ra = copy.deepcopy((left, la))
        add_module_tensor(left, la, "projector/0/output", np.ones((3, 4)))
        add_module_tensor(right, ra, "projector/0/output", np.ones((1, 4)))
        add_module_tensor(right, ra, "projector/1/output", np.ones((2, 4)))
        result = compare.compare(left, la, right, ra)
        self.assertNotIn("module_tensors", result)
        self.assertIn("module_comparison", result)

    def test_same_module_partition_reports_changed_input_values(self):
        left, la = capture("a")
        right, ra = copy.deepcopy((left, la))
        name = "encoder/0/input/args/0"
        add_module_tensor(left, la, name, np.ones((1, 3, 2)))
        add_module_tensor(right, ra, name, np.ones((1, 3, 2)) * 2)
        result = compare.compare(left, la, right, ra)
        self.assertFalse(result["module_tensors"][name]["values_exact"])
        self.assertEqual(result["module_tensors"][name]["max_abs"], 1)


class WeightAndFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_dtype_only_and_actual_value_changes_are_distinguishable(self):
        record = {"shape": [2], "dtype": "torch.bfloat16", "sha256": "bf16-bytes", "values_sha256": "canonical"}
        left = {"encoder": {"parameters": {"weight": record}, "buffers": {}}}
        right = copy.deepcopy(left)
        right["encoder"]["parameters"]["weight"].update(dtype="torch.float32", sha256="fp32-bytes")
        changed = compare.weight_difference(left, right)["encoder"]["different_tensors"][0]["differences"]
        self.assertIn("dtype", changed)
        self.assertIn("sha256", changed)
        self.assertNotIn("values_sha256", changed)
        right["encoder"]["parameters"]["weight"]["values_sha256"] = "changed-values"
        changed = compare.weight_difference(left, right)["encoder"]["different_tensors"][0]["differences"]
        self.assertEqual(changed["values_sha256"], ["canonical", "changed-values"])

    def test_missing_weight_or_buffer_is_reported(self):
        result = compare.weight_difference({"encoder": {"buffers": {"inv_freq": {"dtype": "torch.float32"}}}}, {})
        missing = result["encoder"]["different_tensors"][0]
        self.assertEqual(missing["kind"], "buffers")
        self.assertEqual(missing["reason"], "missing")
        self.assertTrue(missing["left_present"])
        self.assertFalse(missing["right_present"])

    def test_weight_snapshot_without_arrays_does_not_open_npz(self):
        metadata = {"format": compare.FORMAT, "status": "complete", "phase": "before_first_reload",
                    "weights": {}, "tensors": {}}
        path = self.root / "weights.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch.object(compare.np, "load", side_effect=AssertionError("no NPZ for weights")):
            read, arrays = compare.read_capture(path)
        self.assertEqual(arrays, {})
        result = compare.compare(read, arrays, read, arrays)
        self.assertIn("weight snapshots only", result["activation_comparison"])

    def test_array_reader_uses_adjacent_npz_and_disallows_pickle(self):
        metadata, arrays = capture("a")
        path = self.root / "capture.json"
        metadata["npz_path"] = "../../untrusted.npz"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        archive = {metadata["tensors"][name]["array_key"]: value for name, value in arrays.items()}
        np.savez(path.with_suffix(".npz"), **archive)
        with patch.object(compare.np, "load", wraps=np.load) as reader:
            _, actual = compare.read_capture(path)
        reader.assert_called_once_with(str(path.with_suffix(".npz")), allow_pickle=False)
        np.testing.assert_array_equal(actual["audio_embeddings/0"], arrays["audio_embeddings/0"])
        archive[metadata["tensors"]["audio_embeddings/0"]["array_key"]] = np.array([{"unsafe": True}], dtype=object)
        np.savez(path.with_suffix(".npz"), **archive)
        with self.assertRaisesRegex(ValueError, "Object arrays cannot be loaded"):
            compare.read_capture(path)

    def test_incomplete_or_wrong_shape_capture_is_rejected(self):
        metadata, arrays = capture("a")
        path = self.root / "capture.json"
        metadata["status"] = "failed"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete Huanyu"):
            compare.read_capture(path)
        metadata["status"] = "complete"
        metadata["tensors"]["audio_embeddings/0"]["shape"] = [99]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        archive = {metadata["tensors"][name]["array_key"]: value for name, value in arrays.items()}
        np.savez(path.with_suffix(".npz"), **archive)
        with self.assertRaisesRegex(ValueError, "shape does not match"):
            compare.read_capture(path)


if __name__ == "__main__":
    unittest.main()
