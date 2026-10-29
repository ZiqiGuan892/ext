"""Real producer -> BF16 NPZ -> strict comparator -> text A/B output integration."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture_module = load("embedding_roundtrip_fixture", ROOT / "tests/test_prompt_embedding_probe_on_cpu.py")
comparator = load("embedding_roundtrip_compare", ROOT / "examples/compare_huanyu_prompt_embeddings.py")
text_ab = load("embedding_roundtrip_text_ab", ROOT / "examples/aqa_text_ab.py")


class PromptEmbeddingRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ProbeTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.fixture.row.update(source_row_index=1, prompt_mode="system_audio", ground_truth="answer",
                                wav_paths=["/test/audio.wav"])
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(self.fixture.row, ensure_ascii=False) + "\n", encoding="utf-8")
        self.fixture.case.update(schema_version=1, engine={"tensor_parallel_size": 4},
                                 manifest_sha256=fixture_module.probe._file_hash(manifest))
        self.fixture.case["embedding_check"]["expected_tp_size"] = 4
        self.fixture.write_case()
        self.fixture.config.parallel_config.tensor_parallel_size = 4
        with patch.object(fixture_module.probe, "_tp_identity", return_value=(0, 4)), \
                patch.object(fixture_module.probe, "install_runtime_probe", return_value=True):
            fixture_module.probe.attach_model_probe(self.fixture.native, self.fixture.config)
        for side in ("serve", "rollout"):
            folder = self.root / side
            folder.mkdir()
            result = {key: self.fixture.row[key] for key in text_ab.IDENTITY_KEYS}
            # Text output need not match merely because prompt inputs match.
            result.update(status="complete", reply="serve answer" if side == "serve" else "rollout answer")
            (folder / (fixture_module.SAMPLE + ".json")).write_text(json.dumps(result), encoding="utf-8")
            (folder / (fixture_module.SAMPLE + ".txt")).write_text(result["reply"], encoding="utf-8")

    def emit(self, changed=False, missing=None):
        state = self.fixture.native._huanyu_embedding_probe_state
        original = self.fixture.tensor.clone()
        for side in ("serve", "rollout"):
            for rank in range(4):
                if (side, rank) == missing:
                    continue
                state.update(side=side, tp_rank=rank)
                state["seen"].clear()
                self.fixture.tensor = original.clone()
                if changed and side == "rollout":
                    self.fixture.tensor[1, 2] += 1
                self.fixture.execute()

    def run_text_compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = text_ab.compare_case(SimpleNamespace(case_dir=str(self.root)))
        summary = json.loads((self.root / "summary.json").read_text())
        row = json.loads((self.root / "comparison.jsonl").read_text().splitlines()[0])
        embeddings = json.loads((self.root / "embedding_comparison.jsonl").read_text().splitlines()[0])
        return code, summary, row, embeddings

    def test_actual_producer_schema_passes_all_four_tp_ranks(self):
        self.emit()
        result = comparator.compare_case_embeddings(self.root, self.fixture.case, [self.fixture.row])
        self.assertEqual(result["summary"]["status"], "PASS")
        self.assertEqual(result["summary"]["observed_records"], {"serve": 4, "rollout": 4})
        code, summary, row, embeddings = self.run_text_compare()
        self.assertEqual(code, 0)
        self.assertEqual(summary["embedding_check"]["status"], "PASS")
        self.assertTrue(row["embedding_strict_equal"])
        self.assertFalse(row["reply_exact_match"])
        self.assertEqual(len(embeddings["ranks"]), 4)

    def test_real_audio_element_difference_propagates_to_text_ab_report(self):
        self.emit(changed=True)
        code, summary, row, embeddings = self.run_text_compare()
        self.assertEqual(code, 1)
        self.assertEqual(summary["embedding_check"]["status"], "FAIL")
        self.assertFalse(row["embedding_strict_equal"])
        for rank in embeddings["ranks"]:
            self.assertEqual(rank["first_different_element"], [1, 2])
            self.assertEqual(rank["max_abs_difference"], 1)
            self.assertEqual(rank["audio"]["mismatched_elements"], 1)
            self.assertEqual(rank["text_and_special_tokens"]["mismatched_elements"], 0)

    def test_missing_rank_cannot_be_hidden_by_complete_text_outputs(self):
        self.emit(missing=("rollout", 3))
        code, summary, row, embeddings = self.run_text_compare()
        self.assertEqual(code, 2)
        self.assertEqual(summary["complete_pairs"], 1)
        self.assertEqual(summary["embedding_check"]["status"], "INCOMPLETE")
        self.assertIsNone(row["embedding_strict_equal"])
        self.assertEqual(embeddings["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
