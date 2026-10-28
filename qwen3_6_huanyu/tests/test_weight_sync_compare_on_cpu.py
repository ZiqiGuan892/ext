"""CPU fixtures for sync audit classification; no distributed runtime required."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "examples/compare_huanyu_weight_sync.py"
SPEC = importlib.util.spec_from_file_location("weight_sync_compare", PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def tensor(values, **kwargs):
    return dict(shape=[len(values)], global_shape=[len(values)], dtype="torch.bfloat16", finite=True,
                values=values, float32_sha256=hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest(),
                **kwargs)


def fixture(update=1):
    result = []
    for name in compare.NAMES:
        size = 20 if ".conv_layers.1." in name else 40
        values = [float(i) for i in range(size)]
        for stage in compare.STAGES:
            for rank in range(2):
                current = values if update == 1 or stage in compare.STAGES[:2] else [v + 1 for v in values]
                result.append(dict(name=name, stage=stage, tensor=tensor(current),
                                   process=f"{'receiver' if stage in compare.STAGES[3:] else 'actor'}{rank}",
                                   location="fixture", schema_version=1,
                                   round_id=None if stage in compare.STAGES[:2] else update))
        for rank in range(2):
            current = values if update == 1 else [v + 1 for v in values]
            item = tensor(current[rank * (size // 2):(rank + 1) * (size // 2)],
                          placements=["S(0)"], mesh_shape=[2], mesh_coordinate=[rank])
            item["global_shape"] = [size]
            result.append(dict(name=name, stage="actor_local_shard", tensor=item,
                               process=f"actor{rank}", location="fixture", schema_version=1, round_id=update))
    return result


class ComparisonTests(unittest.TestCase):
    def test_equal_values_ignore_storage_dtype_and_pass_all_eight(self):
        entries = fixture()
        entries[2]["tensor"]["dtype"] = "torch.float32"
        report = compare.compare(entries)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["parameters"]), 8)

    def test_receiver_corruption_localizes_and_records_values(self):
        entries = fixture()
        bad = next(e for e in entries if e["stage"] == "receiver_input")
        values = bad["tensor"]["values"][:]
        values[3] += 7
        bad["tensor"] = tensor(values)
        item = compare.compare(entries)["parameters"][bad["name"]]
        self.assertEqual(item["earliest_observed_bad_boundary"], "receiver_input")
        stage = item["stages"]["receiver_input"]
        self.assertTrue(stage["process_divergence"])
        self.assertEqual(stage["records"][0]["comparison"]["differences"],
                         [{"index": 3, "expected": 3.0, "actual": 10.0}])

    def test_missing_predecessor_prevents_false_localization(self):
        entries = [e for e in fixture() if e["stage"] != "actor_export"]
        next(e for e in entries if e["stage"] == "receiver_input")["tensor"] = tensor([999.0] * 40)
        report = compare.compare(entries)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(all(p["earliest_observed_bad_boundary"] is None for p in report["parameters"].values()))

    def test_partial_process_or_control_is_incomplete(self):
        entries = fixture()
        entries = [e for e in entries if not (e["stage"] == "after_finalize" and e["process"] == "receiver1")]
        report = compare.compare(entries)
        self.assertEqual(report["status"], "INCOMPLETE")
        entries = [e for e in fixture() if e["name"] != compare.NAMES[-1]]
        self.assertEqual(compare.compare(entries)["status"], "INCOMPLETE")

    def test_receiver_failed_before_input_is_not_omitted(self):
        for stage in ("before_reload", "after_initialize"):
            with self.subTest(stage=stage):
                entries = fixture()
                entries.append(dict(entries[0], stage=stage, process="receiver_failed", round_id=1))
                report = compare.compare(entries)
                self.assertEqual(report["status"], "INCOMPLETE")
                item = report["parameters"][compare.NAMES[0]]["stages"]["receiver_input"]
                self.assertIn("receiver_failed", item["missing_processes"])

    def test_stage_and_shard_reports_keep_source_and_version_evidence(self):
        entries = fixture()
        extra = {"export_source": {"file_sha256": "source_hash"}}
        versions = {"torch": "server_version"}
        for entry in entries:
            entry.update(extra=extra, versions=versions)
        stages = compare.compare(entries)["parameters"][compare.NAMES[0]]["stages"]
        for stage in ("actor_export", "actor_local_shard"):
            self.assertEqual(stages[stage]["records"][0]["extra"], extra)
            self.assertEqual(stages[stage]["records"][0]["versions"], versions)

    def test_unsupported_shard_mesh_is_not_pass(self):
        entries = fixture()
        next(e for e in entries if e["stage"] == "actor_local_shard")["tensor"]["mesh_shape"] = [1, 2]
        self.assertEqual(compare.compare(entries)["status"], "INCOMPLETE")

    def test_uneven_40_element_shards_and_empty_ranks(self):
        baseline = tensor([float(i) for i in range(40)])
        entries = []
        for rank in range(16):
            local = tensor(baseline["values"][min(rank * 3, 40):min((rank + 1) * 3, 40)],
                           placements=["S(0)"], mesh_shape=[16], mesh_coordinate=[rank])
            local["global_shape"] = [40]
            entries.append(dict(tensor=local, process=str(rank), location="fixture"))
        expected = {str(rank) for rank in range(16)}
        self.assertEqual(compare.shard_report(entries, expected, baseline)["status"], "PASS")
        entries[13]["tensor"]["values"] = [999.0]
        entries[13]["tensor"]["float32_sha256"] = tensor([999.0])["float32_sha256"]
        report = compare.shard_report(entries, expected, baseline)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["records"][13]["offset"], 39)

    def test_missing_whole_mesh_rank_is_incomplete(self):
        entries = [e for e in fixture() if e["process"] != "actor1"]
        self.assertEqual(compare.compare(entries)["status"], "INCOMPLETE")

    def test_later_round_uses_current_export_not_checkpoint(self):
        report = compare.compare(fixture(update=2), update=2)
        self.assertEqual(report["status"], "PASS")
        self.assertIn("informational", report["checkpoint_baseline"])

    def test_empty_or_nonfinite_never_passes(self):
        self.assertEqual(compare.compare([])["status"], "INCOMPLETE")
        entries = fixture()
        entries[0]["tensor"]["finite"] = False
        self.assertEqual(compare.compare(entries)["status"], "FAIL")

    def test_load_filters_rounds_and_keeps_initial_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights-host-1.jsonl"
            entries = fixture() + fixture(2)
            path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
            selected = compare.load_entries(temporary, 2)
            self.assertEqual({e["round_id"] for e in selected}, {None, 2})
            self.assertTrue(all(str(path) in e["process"] for e in selected))


if __name__ == "__main__":
    unittest.main()
