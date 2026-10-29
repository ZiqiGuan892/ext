"""Offline audit aggregation: incomplete topology/evidence must never pass."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_huanyu_full_weights_test", ROOT / "examples" / "compare_huanyu_full_weights.py")
compare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_module)


def tensor(shape=(4,), digest="a" * 64):
    numel = 1
    for size in shape:
        numel *= size
    return {"dtype": "torch.bfloat16", "shape": list(shape), "numel": numel,
            "sha256": digest, "finite": True, "is_meta": False, "device": "npu:0", "stride": [1]}


def record(kind="parameter"):
    return {"status": "PASS", "initial_disk": tensor(), "after_sync": tensor(), "disk_replay": tensor(),
            "comparison_scope": "registered_buffer_threeway_not_poisoned" if kind == "buffer" else "parameter"}


def worker(index, tp_size=2):
    return {"schema_version": 1, "status": "PASS", "checkpoint": "/models/2748", "run_id": "run-1",
            "audit_only": True, "generation_forbidden": True, "comparison": compare_module.COMPARISON,
            "checkpoint_stream": {"tensor_count": 1, "nonfinite_elements": 0, "manifest_sha256": "c" * 64},
            "provenance": {"checkpoint": "/models/2748", "load_format": "safetensors",
                           "checkpoint_headers": {"model.safetensors": {"header_sha256": "d" * 64,
                                                                        "file_bytes": 128, "tensor_count": 1}}},
            "process": {"host": "worker-host", "pid": 100 + index, "rank": index % tp_size,
                        "tp_rank": index % tp_size, "tp_world_size": tp_size,
                        "engine_group": f"replica-{index // tp_size}"},
            "parameters": {"embedding.weight": record(), "lm_head.weight": record()},
            "buffers": {"rotary.inv_freq": record("buffer")},
            "summary": {"parameter_names": 2, "parameter_numel_with_aliases": 8,
                        "unique_parameter_objects": 1, "unique_parameter_numel": 4},
            "errors": []}


class CompareFullWeightsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def write(self, reports):
        # Each subtest gets a fresh directory, mirroring a timestamped audit run.
        folder = self.directory / str(len(list(self.directory.iterdir())))
        folder.mkdir()
        for index, report in enumerate(reports):
            (folder / f"full_weights-{index}.json").write_text(json.dumps(report), encoding="utf-8")
        return folder

    def compare(self, reports, expected=4, tp_size=2):
        return compare_module.compare(self.write(reports), expected, tp_size)

    def test_complete_replicas_and_alias_counts(self):
        result = self.compare([worker(i) for i in range(4)])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["observed_unique_workers"], 4)
        self.assertEqual(result["summary"]["parameter_entries"], 8)
        self.assertEqual(result["summary"]["parameter_numel"], 32)
        self.assertEqual(result["summary"]["unique_parameter_objects"], 4)
        self.assertEqual(result["summary"]["unique_parameter_numel"], 16)
        # Global ranks may repeat between engine replicas; TP-group identity matters.
        self.assertTrue(all(group["complete"] for group in result["topology"].values()))

    def test_missing_or_extra_workers_never_pass(self):
        for count in (0, 1, 2, 3, 5):
            with self.subTest(count=count):
                result = self.compare([worker(i) for i in range(count)])
                self.assertEqual(result["status"], "INCOMPLETE")

    def test_duplicate_worker_and_duplicate_tp_rank(self):
        for mutate in (lambda reports: reports.__setitem__(3, copy.deepcopy(reports[2])),
                       lambda reports: reports[3]["process"].update(tp_rank=0)):
            reports = [worker(i) for i in range(4)]
            mutate(reports)
            self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_missing_topology_or_wrong_tp_size(self):
        for changes in ({"engine_group": None}, {"tp_rank": None}, {"tp_world_size": 4},
                        {"tp_rank": -1}, {"tp_rank": 2}):
            with self.subTest(changes=changes):
                reports = [worker(i) for i in range(4)]
                reports[0]["process"].update(changes)
                self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_mixed_runs_and_checkpoints_are_incomplete(self):
        for key, value in (("run_id", "another-run"), ("checkpoint", "/models/other"),
                           ("run_id", None), ("checkpoint", "")):
            with self.subTest(key=key, value=value):
                reports = [worker(i) for i in range(4)]
                reports[0][key] = value
                self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_corruption_is_fail_with_complete_coverage(self):
        reports = [worker(i) for i in range(4)]
        reports[1]["status"] = "FAIL"
        entry = reports[1]["parameters"]["embedding.weight"]
        entry["status"] = "FAIL"
        entry["after_sync"]["sha256"] = "b" * 64
        result = self.compare(reports)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(row["name"] == "embedding.weight" for row in result["failures"]))
        self.assertEqual(result["errors"], [])

    def test_misleading_pass_flag_is_not_accepted(self):
        reports = [worker(i) for i in range(4)]
        reports[0]["parameters"]["embedding.weight"]["after_sync"]["sha256"] = "b" * 64
        result = self.compare(reports)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("contradicts" in error for error in result["errors"]))

    def test_fingerprint_requires_concrete_finite_values_and_shape(self):
        for change in ({"finite": False}, {"is_meta": True}, {"numel": 5}, {"sha256": ""},
                       {"shape": [True]}, {"dtype": None}, {"numel": True}):
            with self.subTest(change=change):
                reports = [worker(i) for i in range(4)]
                reports[0]["parameters"]["embedding.weight"]["disk_replay"].update(change)
                self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_dtype_mismatch_and_buffer_mismatch_are_not_ignored(self):
        for kind, name in (("parameters", "embedding.weight"), ("buffers", "rotary.inv_freq")):
            with self.subTest(kind=kind):
                reports = [worker(i) for i in range(4)]
                reports[0]["status"] = "FAIL"
                reports[0][kind][name]["status"] = "FAIL"
                reports[0][kind][name]["after_sync"]["dtype"] = "torch.float32"
                self.assertEqual(self.compare(reports)["status"], "FAIL")

    def test_identical_nonfinite_attention_mask_buffer_passes(self):
        reports = [worker(i) for i in range(4)]
        for report in reports:
            report["buffers"] = {"attention.mask": record("buffer")}
            for stage in compare_module.STAGES:
                report["buffers"]["attention.mask"][stage]["finite"] = False
        self.assertEqual(self.compare(reports)["status"], "PASS")

    def test_changed_nonfinite_attention_mask_buffer_fails(self):
        reports = [worker(i) for i in range(4)]
        for report in reports:
            report["buffers"] = {"attention.mask": record("buffer")}
            for stage in compare_module.STAGES:
                report["buffers"]["attention.mask"][stage]["finite"] = False
        reports[0]["status"] = "FAIL"
        entry = reports[0]["buffers"]["attention.mask"]
        entry["status"] = "FAIL"
        entry["after_sync"]["sha256"] = "b" * 64
        self.assertEqual(self.compare(reports)["status"], "FAIL")

    def test_identical_nan_parameter_remains_incomplete(self):
        reports = [worker(i) for i in range(4)]
        for stage in compare_module.STAGES:
            reports[0]["parameters"]["embedding.weight"][stage]["finite"] = False
        result = self.compare(reports)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("missing/invalid fingerprints" in error for error in result["errors"]))

    def test_missing_inventory_or_summary_cannot_hide_parameters(self):
        changes = ({"parameters": {}}, {"buffers": None}, {"summary": {}},
                   {"errors": ["checkpoint loading failed"]}, {"schema_version": 2},
                   {"schema_version": True}, {"status": []}, {"process": []}, {"summary": []})
        for change in changes:
            with self.subTest(change=change):
                reports = [worker(i) for i in range(4)]
                reports[0].update(change)
                self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")
        reports = [worker(i) for i in range(4)]
        reports[0]["parameters"].pop("lm_head.weight")
        reports[0]["summary"].update(parameter_names=1, parameter_numel_with_aliases=4)
        self.assertTrue(any("inventories differ" in error for error in self.compare(reports)["errors"]))

    def test_malformed_tensor_status_is_incomplete_without_crashing(self):
        reports = [worker(i) for i in range(4)]
        reports[0]["parameters"]["embedding.weight"]["status"] = []
        self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_disk_replay_provenance_is_required(self):
        changes = ({"audit_only": False}, {"generation_forbidden": False}, {"comparison": "sync_vs_sync"},
                   {"checkpoint_stream": {}}, {"checkpoint_stream": {"tensor_count": 0, "nonfinite_elements": 0}},
                   {"checkpoint_stream": {"tensor_count": 1, "nonfinite_elements": 1}},
                   {"provenance": {}},
                   {"provenance": {"checkpoint": "/models/2748", "load_format": "safetensors",
                                   "checkpoint_headers": {}}})
        for change in changes:
            with self.subTest(change=change):
                reports = [worker(i) for i in range(4)]
                reports[0].update(change)
                self.assertEqual(self.compare(reports)["status"], "INCOMPLETE")

    def test_broken_json_and_duplicate_json_keys(self):
        for content in ('{"schema_version":', '{"schema_version":1,"schema_version":1}'):
            with self.subTest(content=content):
                folder = self.write([worker(i) for i in range(4)])
                (folder / "full_weights-0.json").write_text(content, encoding="utf-8")
                self.assertEqual(compare_module.compare(folder, 4, 2)["status"], "INCOMPLETE")

    def test_same_missing_parameter_on_entire_replica_is_incomplete(self):
        reports = [worker(i) for i in range(4)]
        for report in reports[2:]:
            report["parameters"].pop("lm_head.weight")
            report["summary"].update(parameter_names=1, parameter_numel_with_aliases=4)
        result = self.compare(reports)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("between policy workers/replicas" in error for error in result["errors"]))

    def test_same_checkpoint_path_with_different_replica_weights_is_incomplete(self):
        reports = [worker(i) for i in range(4)]
        # Each local three-way comparison passes, but replica 1's disk differs.
        for report in reports[2:]:
            for stage in compare_module.STAGES:
                report["parameters"]["embedding.weight"][stage]["sha256"] = "b" * 64
        result = self.compare(reports)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("baseline differs between replicas" in error for error in result["errors"]))

    def test_raw_checkpoint_manifests_must_agree_even_when_runtime_values_match(self):
        reports = [worker(i) for i in range(4)]
        reports[0]["checkpoint_stream"]["manifest_sha256"] = "e" * 64
        result = self.compare(reports)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("checkpoint content manifest" in error for error in result["errors"]))

    def test_different_tp_shards_are_valid_when_corresponding_replicas_agree(self):
        reports = [worker(i) for i in range(4)]
        for index in (1, 3):
            for stage in compare_module.STAGES:
                reports[index]["parameters"]["embedding.weight"][stage]["sha256"] = "b" * 64
        self.assertEqual(self.compare(reports)["status"], "PASS")

    def test_cli_exit_codes_and_written_report(self):
        for target, exit_code in (("PASS", 0), ("FAIL", 1), ("INCOMPLETE", 2)):
            with self.subTest(target=target):
                reports = [worker(i) for i in range(4)]
                if target == "FAIL":
                    reports[0]["status"] = "FAIL"
                    reports[0]["parameters"]["embedding.weight"]["status"] = "FAIL"
                elif target == "INCOMPLETE":
                    reports.pop()
                folder = self.write(reports)
                output = folder / "summary.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    actual = compare_module.main(["--audit-dir", str(folder), "--expected-workers", "4",
                                                  "--tp-size", "2", "--output", str(output)])
                self.assertEqual(actual, exit_code)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], target)

    def test_invalid_counts_rejected(self):
        for expected, tp_size in ((0, 2), (4, 0), (3, 2), (True, 1)):
            with self.subTest(expected=expected, tp_size=tp_size):
                with self.assertRaises(ValueError):
                    compare_module.compare(self.directory, expected, tp_size)


if __name__ == "__main__":
    unittest.main()
