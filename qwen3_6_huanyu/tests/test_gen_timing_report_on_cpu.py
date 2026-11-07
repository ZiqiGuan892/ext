"""Engine attribution uses paired intervals, never unrelated phase averages."""

from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_huanyu_timing_report", ROOT / "gen_timing_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def span(name, phase, start, end, parent=None, **extra):
    return dict(event="span", host="node-a", pid=42, model_path="/models/actor", runner_class="Runner",
                engine_iteration=100, sample_every=100, scheduled_encoder_requests=1,
                span_id=name, parent_span_id=parent, phase=phase, status="ok", error_type=None,
                elapsed_s=end - start, start_monotonic_s=start, end_monotonic_s=end,
                start_unix_ns=int(start * 1e9), end_unix_ns=int(end * 1e9), **extra)


def example():
    root = span("root", "engine.execute_model", 100, 110)
    root["nested_span_counts"] = {"engine.preprocess": 1, "engine.mm_encoder": 1,
                                 "engine.gather_mm_embeddings": 1, "engine.forward": 1,
                                 "engine.compute_logits": 1}
    return [root, span("prep", "engine.preprocess", 100, 106, "root"),
            span("mm", "engine.mm_encoder", 101, 105, "prep"),
            span("embeds", "engine.gather_mm_embeddings", 105, 105.5, "prep"),
            span("fwd", "engine.forward", 106, 109, "root"),
            span("logits", "engine.compute_logits", 109, 109.2, "root"),
            span("sample", "engine.sample_tokens", 110, 111)]


def analyze(rows, **filters):
    lines = ["(Worker pid=42) " + report.PREFIX + json.dumps(row) + " [repeated 1x]\n" for row in rows]
    return report.engine_breakdown(lines, **filters)


class EngineReportTests(unittest.TestCase):
    def test_nested_preprocessing_is_subtracted_and_external_sampling_is_separate(self):
        result = analyze(example())
        group, = result["groups"]
        record, = group["accepted"]
        expected = {"engine.preprocess.other": 1.5, "engine.mm_encoder": 4.0,
                    "engine.gather_mm_embeddings": 0.5, "engine.forward": 3.0,
                    "engine.compute_logits": 0.2, "engine.unattributed": 0.8}
        for phase, value in expected.items():
            self.assertAlmostEqual(record["totals"][phase], value)
        self.assertAlmostEqual(sum(record["totals"].values()), 10.0)
        self.assertEqual(group["verified"], 1)
        self.assertEqual(len(group["outside"]), 1)
        self.assertNotIn("engine.sample_tokens", record["totals"])
        with redirect_stdout(io.StringIO()) as output:
            report.print_engine_breakdown(result)
        self.assertIn("engine.mm_encoder\t1\t4.000000", output.getvalue())
        self.assertIn("\t40.00", output.getvalue())
        self.assertIn("sample_tokens_outside_execute", output.getvalue())

    def test_processes_models_and_first_iteration_are_never_merged(self):
        rows = example()
        for pid, model, iteration in ((43, "/models/actor", 100), (42, "/models/judge", 200),
                                       (42, "/models/actor", 1)):
            for row in copy.deepcopy(example()):
                row.update(pid=pid, model_path=model, engine_iteration=iteration)
                row["span_id"] += f"-{pid}-{iteration}"
                if row["parent_span_id"]:
                    row["parent_span_id"] += f"-{pid}-{iteration}"
                rows.append(row)
        self.assertEqual(len(analyze(rows)["groups"]), 4)
        selected = analyze(rows, pid=43, host="node-a", model="actor")
        self.assertEqual(len(selected["groups"]), 1)
        self.assertEqual(selected["groups"][0]["seen"], 1)

    def test_external_sampler_with_non_engine_ancestors_stays_outside_execute(self):
        rows = example()
        rows[0]["parent_span_id"] = "request"
        rows[-1]["parent_span_id"] = "request"
        outer = span("request", "server.generate_request", 99, 112, "rpc")
        outer.pop("model_path")
        rows.extend([outer, span("rpc", "rpc.call", 98, 113)])
        result = analyze(rows, model="actor")
        group, = result["groups"]
        self.assertEqual(len(group["accepted"]), 1)
        self.assertEqual(group["verified"], 1)
        self.assertEqual(len(group["outside"]), 1)
        self.assertEqual(group["skipped"], {})
        self.assertNotIn("orphan_spans", result["diagnostics"])

    def test_sampler_with_missing_or_engine_ancestry_is_not_marked_outside(self):
        for parent in ("missing", "inside"):
            with self.subTest(parent=parent):
                rows = example()
                rows[0].pop("nested_span_counts")  # Legacy logs still fail closed.
                rows[-1] = span("sample", "engine.sample_tokens", 109.3, 109.5, parent)
                rows.append(span("inside", "custom.processing", 109.2, 109.6, "root"))
                group, = analyze(rows)["groups"]
                self.assertEqual(group["accepted"], [])
                self.assertEqual(group["outside"], [])
                self.assertEqual(group["skipped"]["unattached_nested_spans"], 1)

    def test_missing_child_counters_and_unattached_legacy_spans_fail_closed(self):
        rows = [r for r in example() if r["span_id"] != "fwd"]
        group, = analyze(rows)["groups"]
        self.assertEqual(group["accepted"], [])
        self.assertEqual(group["skipped"]["missing_or_extra_child_spans"], 1)
        rows = [r for r in example() if r["span_id"] != "prep"]
        rows[0].pop("nested_span_counts")
        group, = analyze(rows)["groups"]
        self.assertEqual(group["accepted"], [])
        self.assertEqual(group["skipped"]["unattached_nested_spans"], 1)

    def test_legacy_clock_jumps_are_rejected_but_monotonic_records_survive(self):
        rows = example()
        rows[0]["end_unix_ns"] += int(3600e9)
        self.assertEqual(len(analyze(rows)["groups"][0]["accepted"]), 1)
        for row in rows:
            row.pop("start_monotonic_s")
            row.pop("end_monotonic_s")
        group, = analyze(rows)["groups"]
        self.assertEqual(group["skipped"]["interval_clock_mismatch"], 1)

    def test_old_logs_show_unverified_completeness_and_missing_forward_as_unknown(self):
        rows = [r for r in example() if r["span_id"] != "fwd"]
        rows[0].pop("nested_span_counts")
        for row in rows:
            row.pop("start_monotonic_s")
            row.pop("end_monotonic_s")
        result = analyze(rows)
        group, = result["groups"]
        self.assertEqual(group["verified"], 0)
        record, = group["accepted"]
        self.assertAlmostEqual(record["totals"]["engine.unattributed"], 3.8)
        with redirect_stdout(io.StringIO()) as output:
            report.print_engine_breakdown(result)
        self.assertIn("LEGACY:", output.getvalue())
        self.assertIn("engine.forward\tUNOBSERVED", output.getvalue())

    def test_duplicates_dont_double_time_and_conflicting_duplicates_invalidate_tree(self):
        rows = example()
        result = analyze(rows + [copy.deepcopy(rows[2])])
        self.assertEqual(result["diagnostics"]["duplicate_records"], 1)
        self.assertEqual(len(result["groups"][0]["accepted"]), 1)
        conflicting = dict(rows[2], elapsed_s=123)
        group, = analyze(rows + [conflicting])["groups"]
        self.assertEqual(group["skipped"]["conflicting_duplicates"], 1)

    def test_sibling_overlap_and_child_outside_parent_are_rejected(self):
        for start, end, reason in ((104, 107, "overlapping_siblings"), (105, 111, "child_outside_parent")):
            rows = example()
            rows[4] = span("fwd", "engine.forward", start, end, "root")
            group, = analyze(rows)["groups"]
            self.assertEqual(group["accepted"], [])
            self.assertEqual(group["skipped"][reason], 1)

    def test_caught_typeerror_is_reported_without_discarding_successful_iteration(self):
        rows = example()
        rows[5].update(status="error", error_type="TypeError")
        group, = analyze(rows)["groups"]
        self.assertEqual(len(group["accepted"]), 1)
        self.assertEqual(group["errors"][("engine.compute_logits", "TypeError")], 1)

    def test_cli_keeps_old_summary_and_supports_filtered_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "console.log"
            log.write_text("\n".join(report.PREFIX + json.dumps(r) for r in example()), encoding="utf-8")
            base = [sys.executable, str(ROOT / "gen_timing.py"), str(log)]
            for args, expected in (([], "work_s (overlapping, not gen wall time)"),
                                   (["--engine-breakdown", "--pid", "42"], "observed_execute_%")):
                run = subprocess.run(base + args, capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn(expected, run.stdout)
            run = subprocess.run(base + ["--engine-breakdown", "--step", "2"],
                                 capture_output=True, text=True, timeout=15)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("no reliable training step", run.stderr)


if __name__ == "__main__":
    unittest.main()
