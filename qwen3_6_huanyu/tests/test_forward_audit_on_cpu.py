"""Check partial-capture attribution without importing torch or loading a model."""

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_forward_audit.py"
SPEC = importlib.util.spec_from_file_location("_forward_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

GDN = "huanyu/GDN/thinker.llm.model.layers.0.linear_attn:QwenGatedDeltaNetAttention"
CHUNK = "huanyu/GDN/thinker.llm.model.layers.0.linear_attn.chunk_gated_delta_rule:ChunkGatedDeltaRule"
ATTN = "huanyu/Attention/thinker.llm.model.layers.1.self_attn:Attention"
MOE = "huanyu/MoE/thinker.llm.model.layers.0.mlp:SparseMoeBlock"


def event(name, start, end, tid=1):
    return dict(name=name, start=start, end=end, pid=42, tid=tid)


def fixture():
    manifest = dict(identity={"pid": 42}, targets=[{"label": x} for x in (GDN, CHUNK, ATTN, MOE)],
                    iterations=[dict(engine_iteration=6, warmup=False, host_seconds=0.12)])
    events = [event("ProfilerStep#1", 90, 220), event("huanyu/execute/iteration_6", 95, 210),
              event("huanyu/forward", 100, 200), event(GDN, 100, 140), event(ATTN, 140, 160),
              event(MOE, 160, 190), event(GDN + "/core", 105, 135)]
    return manifest, events


class ForwardAuditTests(unittest.TestCase):
    def test_optional_chunk_branch_and_nested_core_are_not_required_or_double_counted(self):
        expected, rows = audit.audit_iterations(*fixture())
        self.assertEqual(expected, {"GDN": 1, "Attention": 1, "MoE": 1})
        self.assertEqual(rows[0]["step_id"], "1")
        self.assertEqual(rows[0]["missing_or_repeated"], [])
        self.assertEqual(rows[0]["host_fractions"], {
            "GDN": 40.0, "Attention": 20.0, "MoE": 30.0, "MLP": 0.0,
            "other": 10.0, "cross_category_overlap": 0.0,
        })

    def test_missing_required_module_or_decoder_warning_prevents_fractions(self):
        manifest, events = fixture()
        _, rows = audit.audit_iterations(manifest, [e for e in events if e["name"] != ATTN])
        self.assertEqual(rows[0]["missing_or_repeated"], [ATTN])
        self.assertIsNone(rows[0]["host_fractions"])
        _, rows = audit.audit_iterations(manifest, events, ["The collected data has been lost"])
        self.assertIsNone(rows[0]["host_fractions"])

    def test_ambiguous_or_unclosed_forward_is_not_accepted(self):
        manifest, events = fixture()
        for extra in [event("huanyu/forward", 101, 199), event("huanyu/forward", 100, 200, tid=2)]:
            modified = events + [extra] if extra["tid"] == 1 else [e for e in events if e["name"] != "huanyu/forward"] + [extra]
            _, rows = audit.audit_iterations(manifest, modified)
            self.assertIsNone(rows[0]["host_fractions"])
        _, rows = audit.audit_iterations(manifest, [e for e in events if not e["name"].startswith("ProfilerStep#")])
        self.assertIsNone(rows[0]["host_fractions"])

    def test_overlapping_categories_are_reported_instead_of_summed_as_wall_time(self):
        manifest, events = fixture()
        for e in events:
            if e["name"] == MOE:
                e.update(start=150, end=190)
        _, rows = audit.audit_iterations(manifest, events)
        self.assertEqual(rows[0]["host_fractions"]["cross_category_overlap"], 10.0)
        self.assertEqual(rows[0]["host_fractions"]["other"], 10.0)
        self.assertEqual(audit.union_length([(1, 5), (2, 3), (3, 7), (8, 10)]), 8)

    def test_csv_preserves_microsecond_precision_and_reports_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_summary.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Task Start Time(us)", "Task Duration(us)"])
                writer.writerows([
                    ["1794295361468119.010\t", "3.360"],
                    ["1794295361468212.852\t", "1.940"],
                    ["N/A", "1"], ["1", "NaN"], ["1", "-1"],
                ])
            result = audit.csv_coverage(path)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["invalid_rows"], 3)
            self.assertEqual(result["start_us"], Decimal("1794295361468119.010"))
            self.assertEqual(result["span_s"], Decimal("0.000095782"))

    def test_manifest_selection_never_picks_an_arbitrary_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("pid42_tp0_t123.json", "pid43_tp1_t124.json"):
                (root / name).write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.select_manifest(root)
            self.assertEqual(audit.select_manifest(root, pid=42).name, "pid42_tp0_t123.json")

    def test_export_inventory_ignores_other_captures_and_reports_broken_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "capture"
            (trace / "PROF_current").mkdir(parents=True)
            exported = root / "exports"
            for name in ("PROF_current", "PROF_unrelated"):
                p = exported / name
                p.mkdir(parents=True)
                (p / "msprof_1.json").write_text("[", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                audit.print_device_evidence(exported, trace)
            self.assertIn("TIMELINE_READ_ERROR", output.getvalue())
            self.assertNotIn("PROF_unrelated", output.getvalue())

    def test_clock_inventory_does_not_require_profiler_info(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "PROF_current/host"
            host.mkdir(parents=True)
            (host / "info.json").write_text(json.dumps({"CPU": [{"Frequency": "100"}]}), encoding="utf-8")
            (host / "host_start.log").write_text("[Host]\nclock_monotonic_raw=123\ncntvct=456\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                audit.print_clock_anchors(root)
            self.assertIn("CPU_FREQUENCIES ['100']", output.getvalue())
            self.assertIn("'cntvct': '456'", output.getvalue())
            self.assertIn("exists= False", output.getvalue())
            (host / "host_start.log").write_text("incomplete config", encoding="utf-8")
            with redirect_stdout(output):
                audit.print_clock_anchors(root)
            self.assertIn("CLOCK_READ_ERROR", output.getvalue())

    def test_cli_help_and_argument_errors_work_without_npu_import(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", str(SCRIPT)], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"profile_dir", result.stderr)

    def test_main_reports_partial_capture_without_changing_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "capture"
            framework = trace / "FRAMEWORK"
            framework.mkdir(parents=True)
            source = framework / "torch.op_range"
            source.write_bytes(b"mock framework bytes")
            (trace / "PROF_current").mkdir()
            exported = root / "export"
            exported.mkdir()
            manifest, events = fixture()
            manifest.update(trace_dir=str(trace), active=1, status="capturing", phase="profiler.step")
            (root / "pid42_tp0_t123.json").write_text(json.dumps(manifest), encoding="utf-8")
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = io.StringIO()
            with patch.object(audit, "load_framework_events", return_value=(events, 100, [], source)):
                with redirect_stdout(output):
                    result = audit.main([str(root), str(exported), "--pid", "42", "--tp-rank", "0"])
            self.assertEqual(result, 0)
            self.assertIn("HOST_SHARE_PCT", output.getvalue())
            self.assertIn("PROFILER_INFO []", output.getvalue())
            self.assertIn("不代表设备事件完整", output.getvalue())
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
