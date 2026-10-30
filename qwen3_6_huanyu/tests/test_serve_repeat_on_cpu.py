"""Serial HTTP repeats plus real CPU sampler captures; no vLLM/NPU required."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repeat = load("serve_repeat_cpu", ROOT / "examples/repeat_huanyu_serve.py")
text_fixture = load("serve_repeat_text_fixture", ROOT / "tests/test_aqa_text_ab_on_cpu.py")
probe_fixture = load("serve_repeat_probe_fixture", ROOT / "tests/test_logits_probe_on_cpu.py")
ENV = {"HCCL_DETERMINISTIC": "strict", "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1",
       "LCCL_DETERMINISTIC": "1", "VLLM_BATCH_INVARIANT": "1"}


class ServeRepeatTests(unittest.TestCase):
    def setUp(self):
        self.fixture = text_fixture.TextCaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.fixtures(count=2)
        _, self.source, _ = self.fixture.prepare(size=2, embedding=True, logits=True)
        self.source["sampling"]["max_tokens"] = 3
        self.source["logits_check"]["max_steps"] = 3
        self.source["engine"].update(max_model_len=3203, max_num_batched_tokens=3203)
        (self.fixture.case / "case.json").write_text(json.dumps(self.source), encoding="utf-8")
        self.root = self.fixture.root / "repeat"

    def prepare(self, count=2, logits=True):
        with redirect_stdout(io.StringIO()):
            repeat.prepare(NS(source_case=self.fixture.case, output_dir=self.root, repeats=count,
                              samples=["row_000000", "row_000001"], batch_invariant="true",
                              logits_check="true" if logits else "false"))
        _, self.case, self.rows = repeat.load_case(self.root)

    def send(self):
        ab = repeat.sibling("aqa_text_ab")
        client = ab.sibling("request_aqa_validation_wav")
        calls = []

        def post(url, payload, timeout, key):
            body = json.loads(payload)
            calls.append(body)
            # Assert the preceding HTTP call has finished and its response is
            # on disk before the next call starts, not merely n=1 in a request.
            self.assertEqual(len(list((self.root / "serve").glob("*.json"))), len(calls) - 1)
            response = self.fixture.response()
            response["id"] = "chatcmpl-" + body["request_id"]
            return response

        original_sibling = repeat.sibling
        with patch.object(client, "post_json", side_effect=post), patch.object(ab, "sibling", return_value=client), \
                patch.object(repeat, "sibling", side_effect=lambda name: ab if name == "aqa_text_ab" else original_sibling(name)), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(repeat.request(self.fixture.server(root=self.root)), 0)
        return calls

    def captures(self, difference=None):
        helper = probe_fixture.LogitsProbeTests()
        helper.root, helper.case, helper.rows = self.root, self.case, self.rows
        actual = dict(environment=ENV, batch_invariant_registered=True, torch_deterministic_algorithms=True)
        for rank in range(4):
            first = self.rows[0]["sample_id"]
            first_id = "huanyuab.{}.{}".format(self.case["case_id"], first)
            with patch.object(probe_fixture, "SAMPLE", first), patch.object(probe_fixture, "REQUEST", first_id), \
                    patch.object(probe_fixture.probe, "_determinism_state", return_value=actual):
                runner = helper.runner("serve", rank=rank, tp=4)
            state = getattr(runner.model, probe_fixture.probe._MARKER)
            state["rows"] = {r["sample_id"]: r for r in self.rows}
            request = runner.requests[first_id]
            for row in self.rows:
                request_id = "huanyuab.{}.{}".format(self.case["case_id"], row["sample_id"])
                request.output_token_ids = []
                runner.requests = {request_id: request}
                runner.input_batch.req_ids = [request_id]
                with patch.object(probe_fixture, "REQUEST", request_id):
                    for step in range(3):
                        vector = [1, 2, 3.125, 0] if difference == (row["sample_id"], step) else None
                        helper.step(runner, vector)
                self.embedding(row, rank)

    def embedding(self, row, rank):
        helper = repeat.sibling("compare_huanyu_prompt_embeddings")
        directory = self.root / "embeddings" / "serve"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (row["sample_id"] + "-{}.json".format(rank))
        raw = np.arange(6, dtype=np.float32).reshape(3, 2).view(np.uint8).reshape(3, 2, 4)
        np.savez(path.with_suffix(".npz"), embedding_bytes=raw, input_ids=np.array([8, 9, 10], dtype=np.int64),
                 positions=np.arange(3, dtype=np.int64), audio_mask=np.array([False, True, False]))
        data = dict(format=helper.FORMAT, case_id=self.case["case_id"], sample_id=row["sample_id"],
                    side="serve", tp_rank=rank, tp_size=4, status="complete", checkpoint=self.case["model_path"],
                    request_id="huanyuab.{}.{}".format(self.case["case_id"], row["sample_id"]),
                    prompt_sha256=row["prompt_sha256"], wav_sha256=row["wav_sha256"],
                    tensor_file=path.with_suffix(".npz").name, tensor_file_sha256=helper._sha(path.with_suffix(".npz")),
                    embeddings=dict(dtype="torch.float32", shape=[3, 2], sha256=hashlib.sha256(raw.tobytes()).hexdigest()),
                    capture_scope="llm_inputs_embeds", prompt_token_count=3, captured_token_count=3,
                    num_computed_tokens=0, prefill_complete=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def compare(self, expected):
        with redirect_stdout(io.StringIO()):
            code = repeat.compare(NS(case_dir=self.root))
        report = json.loads((self.root / "repeat_summary.json").read_text())
        self.assertEqual(report["status"], expected)
        self.assertEqual(code, {"PASS": 0, "DIFFERENT": 1, "INCOMPLETE": 2}[expected])
        return report

    def test_twenty_each_forty_serial_calls_same_bodies_unique_ids(self):
        self.prepare(count=20, logits=False)
        calls = self.send()
        self.assertEqual(len(calls), 40)
        self.assertEqual(len({c["request_id"] for c in calls}), 40)
        for index, body in enumerate(calls):
            body.pop("request_id")
            self.assertEqual(body, calls[index % 2])
            self.assertEqual(body["n"], 1)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["seed"], 42)
            self.assertEqual(body["messages"][0]["content"], " 系统原文\n只输出中文。\t")
            self.assertEqual([p["type"] for p in body["messages"][1]["content"]], ["audio_url"])
        plan = repeat.sibling("launch_huanyu_aqa_text_ab").build_launch(self.root, self.case, ROOT, "serve")
        for key, value in ENV.items():
            self.assertEqual(plan["env"][key], value)
        self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_DIR"], "")
        self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_SIDE"], "")
        self.assertIn("--no-async-scheduling", plan["command"])
        with self.assertRaisesRegex(ValueError, "one serving engine"):
            repeat.sibling("launch_huanyu_aqa_text_ab").build_launch(self.root, self.case, ROOT, "rollout")
        report = self.compare("PASS")
        self.assertEqual(report["scope"], "text_only")
        self.assertEqual(report["expected_requests"], 40)
        self.assertEqual(report["logits_status"], "NOT_COLLECTED")
        self.assertEqual(report["final_sampling_status"], "NOT_OBSERVED")
        self.assertFalse((self.root / "logits").exists())
        self.assertFalse((self.root / "embeddings").exists())
        for sample in report["samples"]:
            self.assertEqual(sample["distinct_text_variants"], 1)
            self.assertEqual(sample["matching_baseline_text_requests"], 20)

    def test_cli_defaults_disable_inherited_tensor_probes(self):
        source_before = (self.fixture.case / "case.json").read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(repeat.main([
                "prepare", "--source-case", str(self.fixture.case), "--output-dir", str(self.root),
                "--samples", "row_000000", "row_000001", "--repeats", "2",
            ]), 0)
        _, case, _ = repeat.load_case(self.root)
        self.assertFalse(case["logits_check"]["enabled"])
        self.assertFalse(case["embedding_check"]["enabled"])
        self.assertTrue(case["engine"]["batch_invariant"])
        self.assertEqual((self.fixture.case / "case.json").read_bytes(), source_before)

    def test_source_determinism_never_controls_repeat(self):
        for enabled in (None, False, True):
            with self.subTest(source_batch_invariant=enabled):
                self.source["engine"].pop("batch_invariant", None)
                if enabled is not None:
                    self.source["engine"]["batch_invariant"] = enabled
                path = self.fixture.case / "case.json"
                path.write_text(json.dumps(self.source), encoding="utf-8")
                before = path.read_bytes()
                self.root = self.fixture.root / ("repeat-" + str(enabled))
                self.prepare(logits=False)
                self.assertTrue(self.case["engine"]["batch_invariant"])
                self.assertEqual(path.read_bytes(), before)

    def test_explicit_disable_rejected_before_creating_case(self):
        with self.assertRaisesRegex(ValueError, "requires --batch-invariant true"):
            repeat.prepare(NS(batch_invariant="false"))
        self.assertFalse(self.root.exists())

    def test_old_case_remains_readable_but_cannot_send_new_requests(self):
        self.prepare(logits=False)
        self.send()
        self.case["engine"]["batch_invariant"] = False
        (self.root / "case.json").write_text(json.dumps(self.case), encoding="utf-8")
        self.compare("PASS")  # Old text comparisons still describe old outputs.
        with self.assertRaisesRegex(ValueError, "prepare a new repeat case"):
            repeat.request(self.fixture.server(root=self.root))

    def test_text_difference_without_tensor_files_is_reported(self):
        self.prepare(logits=False)
        self.send()
        path = self.root / "serve" / (self.rows[2]["sample_id"] + ".json")
        r = json.loads(path.read_text(encoding="utf-8"))
        r["reply"] = "different reply"
        r["raw_response"]["choices"][0]["message"]["content"] = r["reply"]
        path.write_text(json.dumps(r), encoding="utf-8")
        path.with_suffix(".txt").write_text(r["reply"], encoding="utf-8")
        report = self.compare("DIFFERENT")
        self.assertEqual(report["samples"][0]["distinct_text_variants"], 2)

    def test_text_compare_never_imports_tensor_comparison(self):
        self.prepare(logits=False)
        self.send()
        original = repeat.sibling

        def no_tensors(name):
            self.assertNotIn(name, ("compare_huanyu_logits", "compare_huanyu_prompt_embeddings"))
            return original(name)

        with patch.object(repeat, "sibling", side_effect=no_tensors):
            self.compare("PASS")

    def test_text_missing_response_is_incomplete(self):
        self.prepare(logits=False)
        self.send()
        (self.root / "serve" / (self.rows[2]["sample_id"] + ".json")).unlink()
        self.compare("INCOMPLETE")

    def test_text_request_drift_cannot_pass_for_equal_replies(self):
        self.prepare(logits=False)
        self.send()
        for path in (self.root / "serve").glob("*.json"):
            r = json.loads(path.read_text(encoding="utf-8"))
            r["request"]["temperature"] = 1
            path.write_text(json.dumps(r), encoding="utf-8")
        self.compare("INCOMPLETE")

    def test_full_prefill_decode_and_same_worker_pass(self):
        self.prepare()
        self.send()
        self.captures()
        report = self.compare("PASS")
        self.assertEqual(report["expected_requests"], 4)
        for sample in report["samples"]:
            for entry in sample["repetitions"]:
                self.assertTrue(entry["text_exact_equal"])
                self.assertTrue(entry["inputs_exact_equal"])
                self.assertTrue(entry["logits_exact_equal"])

    def test_same_text_but_decode_logits_different_is_not_pass(self):
        self.prepare()
        self.send()
        self.captures(difference=(self.rows[2]["sample_id"], 1))
        report = self.compare("DIFFERENT")
        result = report["samples"][0]["repetitions"][1]
        self.assertTrue(result["text_exact_equal"])
        self.assertEqual(result["first_different_logits"]["step"], 1)
        self.assertEqual(result["first_different_logits"]["phase"], "decode")

    def test_missing_decode_tail_is_incomplete_even_with_same_text(self):
        self.prepare()
        self.send()
        self.captures()
        for path in (self.root / "logits" / "serve").glob("*.json"):
            r = json.loads(path.read_text())
            if r.get("sample_id") == self.rows[2]["sample_id"] and r.get("step") == 2:
                path.unlink()
        self.compare("INCOMPLETE")

    def test_runtime_determinism_override_detected(self):
        self.prepare()
        self.send()
        self.captures()
        for path in (self.root / "logits" / "serve").glob("*.json"):
            r = json.loads(path.read_text())
            if r["kind"] == "runtime":
                r["determinism"]["environment"]["ASCEND_LAUNCH_BLOCKING"] = "0"
                path.write_text(json.dumps(r))
        report = self.compare("DIFFERENT")
        self.assertTrue(report["samples"][0]["repetitions"][0]["control_differences"])

    def test_worker_restart_is_not_same_engine_test(self):
        self.prepare()
        self.send()
        self.captures()
        folder = self.root / "logits" / "serve"
        runtime = next(json.loads(p.read_text()) for p in folder.glob("*.json") if json.loads(p.read_text())["kind"] == "runtime")
        rank = runtime["tp_rank"]
        runtime.update(runtime_id="restarted", pid=12345)
        (folder / "restart-runtime.json").write_text(json.dumps(runtime))
        for path in folder.glob("*.json"):
            r = json.loads(path.read_text())
            if r.get("sample_id") == self.rows[2]["sample_id"] and r["tp_rank"] == rank:
                r["runtime_id"] = "restarted"
                path.write_text(json.dumps(r))
        self.compare("INCOMPLETE")

    def test_sampler_override_detected_even_if_both_repeats_match(self):
        self.prepare()
        self.send()
        self.captures()
        for path in (self.root / "logits" / "serve").glob("*.json"):
            r = json.loads(path.read_text())
            if r["kind"] == "step":
                r["sampling_params"]["repetition_penalty"] = 1.05
                path.write_text(json.dumps(r))
        report = self.compare("DIFFERENT")
        self.assertTrue(report["samples"][0]["repetitions"][1]["logits_exact_equal"])

    def test_missing_sample_rejected_without_random_fallback(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            repeat.prepare(NS(source_case=self.fixture.case, output_dir=self.root, repeats=20,
                              samples=["row_000899", "row_000418"], batch_invariant="true"))
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
