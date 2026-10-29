"""Real CPU tensors at a tiny native-like model/runner boundary; no model download."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_prompt_embedding_probe_cpu", ROOT / "prompt_embedding_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
CASE = "a" * 32
SAMPLE = "row_000001"
REQUEST = "huanyuab.%s.%s" % (CASE, SAMPLE)


class ActualLanguageModel(torch.nn.Module):
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.received = inputs_embeds
        return inputs_embeds + 1


class CausalWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ActualLanguageModel()

    def forward(self, *args, **kwargs):
        raise AssertionError("native Huanyu bypasses CausalLM.forward")


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Module()
        self.thinker.llm = CausalWrapper()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "<audio>"}]
        self.row = dict(case_id=CASE, sample_id=SAMPLE, prompt=prompt, wav_sha256=["1" * 64])
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        self.case = dict(case_id=CASE, sample_size=1, model_path="/checkpoint",
                         manifest_sha256=probe._file_hash(manifest), embedding_check=dict(
                             enabled=True, strict=True, scope="llm_inputs_embeds", expected_tp_size=2))
        self.write_case()
        self.env = patch.dict(os.environ, {probe.DIRECTORY_ENV: str(self.root), probe.SIDE_ENV: "serve"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)
        self.config = NS(model_config=NS(enforce_eager=True, model="/checkpoint", hf_config=NS(
            audio_config=NS(model_type="huanyu_conformer_0_6b"))), scheduler_config=NS(
                max_num_seqs=1, enable_chunked_prefill=False), cache_config=NS(enable_prefix_caching=False),
            parallel_config=NS(tensor_parallel_size=2), speculative_config=None)
        self.native = Native()
        self.ids = [11, 12, 13, 14]
        self.tensor = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        self.positions = torch.arange(6, dtype=torch.long)
        self.features = [NS(modality="audio", mm_position=NS(offset=1, length=2, is_embed=None))]
        self.computed = 0
        self.count = 4
        self.request_id = REQUEST
        self.calls = 1
        self.omit = False
        self.direct = False
        self.llm_ids = None
        test = self

        class Runner:
            def __init__(self):
                self.model = test.native
                self.input_batch = NS(req_ids=[], num_computed_tokens_cpu=np.array([0]))
                self.requests = {}

            def execute_model(self, scheduler_output, intermediate_tensors=None):
                # Native runners create this state inside execute_model.
                self.input_batch.req_ids = [test.request_id]
                self.input_batch.num_computed_tokens_cpu[0] = test.computed
                self.requests[test.request_id] = NS(prompt_token_ids=test.ids, mm_features=test.features)
                if test.omit:
                    return "no_llm_call"
                output = None
                for _ in range(test.calls):
                    target = self.model.thinker.llm.model
                    if test.direct:
                        output = target.forward(test.llm_ids, test.positions, None, test.tensor)
                    else:
                        output = target(test.llm_ids, test.positions, inputs_embeds=test.tensor)
                return output

        self.runner = Runner()
        probe.bind_runner_class(Runner)

    def write_case(self):
        (self.root / "case.json").write_text(json.dumps(self.case), encoding="utf-8")

    def attach(self):
        with patch.object(probe, "_tp_identity", return_value=(0, 2)), patch.object(probe, "install_runtime_probe", return_value=True):
            return probe.attach_model_probe(self.native, self.config)

    def execute(self):
        scheduler = NS(num_scheduled_tokens={self.request_id: self.count}, total_num_scheduled_tokens=self.count)
        return self.runner.execute_model(scheduler)

    def records(self):
        return [json.loads(p.read_text()) for p in (self.root / "embeddings" / "serve").glob("*.json")]

    def test_actual_inner_model_prefill_preserves_bf16_bytes_and_output(self):
        self.attach()
        before = self.tensor.clone()
        result = self.execute()
        torch.testing.assert_close(result, before + 1, rtol=0, atol=0)
        torch.testing.assert_close(self.tensor, before, rtol=0, atol=0)
        self.assertIs(self.native.thinker.llm.model.received, self.tensor)
        record, = self.records()
        self.assertEqual(record["embeddings"]["shape"], [4, 4])
        self.assertEqual(record["embeddings"]["dtype"], "torch.bfloat16")
        path = self.root / "embeddings" / "serve" / record["tensor_file"]
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["embedding_bytes"], before[:4].view(torch.uint8).reshape(4, 4, 2).numpy())
            np.testing.assert_array_equal(arrays["audio_mask"], [False, True, True, False])
            np.testing.assert_array_equal(arrays["input_ids"], self.ids)
            np.testing.assert_array_equal(arrays["positions"], [0, 1, 2, 3])
        self.assertEqual(record["tensor_file_sha256"], probe._file_hash(path))

    def test_direct_forward_and_mrope_positions(self):
        self.direct = True
        self.positions = torch.arange(18, dtype=torch.int32).reshape(3, 6)
        self.attach()
        self.execute()
        record, = self.records()
        with np.load(self.root / "embeddings" / "serve" / record["tensor_file"]) as arrays:
            np.testing.assert_array_equal(arrays["positions"], self.positions[:, :4].numpy())

    def test_native_dummy_zero_ids_are_diagnostic_not_scheduler_prompt_ids(self):
        self.llm_ids = torch.zeros(6, dtype=torch.long)
        self.attach()
        self.execute()
        record, = self.records()
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["llm_input_ids"], [0, 0, 0, 0])
        self.assertFalse(record["llm_input_ids_match_prompt"])
        with np.load(self.root / "embeddings" / "serve" / record["tensor_file"]) as arrays:
            np.testing.assert_array_equal(arrays["input_ids"], self.ids)

    def test_decode_skips_and_warmup_does_not_claim_sample(self):
        self.attach()
        self.native.thinker.llm.model(None, self.positions, inputs_embeds=self.tensor)
        self.assertEqual(self.records(), [])
        self.execute()
        self.computed, self.count = 4, 1
        self.execute()
        self.assertEqual(len(self.records()), 1)

    def test_request_tag_and_documented_random_suffix(self):
        for suffix in ("", "-abcdef12"):
            for prefix in ("", "chatcmpl-"):
                self.assertEqual(probe.parse_request_id(prefix + REQUEST + suffix), (CASE, SAMPLE))
        for value in ("wrong" + REQUEST, REQUEST + "-other", "another", None):
            with self.assertRaises(ValueError):
                probe.parse_request_id(value)

    def test_audio_is_embed_mask_does_not_include_marker_slots(self):
        self.features[0].mm_position.is_embed = torch.tensor([True, False])
        self.attach()
        self.execute()
        record, = self.records()
        self.assertEqual(record["audio_token_count"], 1)
        with np.load(self.root / "embeddings" / "serve" / record["tensor_file"]) as arrays:
            np.testing.assert_array_equal(arrays["audio_mask"], [False, True, False, False])

    def test_partial_or_cached_prefill_is_incomplete(self):
        self.count = 2
        self.attach()
        with self.assertRaisesRegex(ValueError, "entire uncached prompt"):
            self.execute()
        self.assertEqual(self.records()[0]["status"], "incomplete")
        self.assertIsNone(probe._ACTIVE.get())

    def test_no_boundary_call_cannot_pass(self):
        self.omit = True
        self.attach()
        with self.assertRaisesRegex(RuntimeError, "never reached"):
            self.execute()
        self.assertEqual(self.records()[0]["status"], "incomplete")

    def test_second_prefill_capture_is_not_silently_overwritten(self):
        self.attach()
        self.execute()
        with self.assertRaisesRegex(ValueError, "repeated or partial"):
            self.execute()
        self.assertEqual(sorted(r["status"] for r in self.records()), ["complete", "incomplete"])

    def test_nonfinite_embeddings_are_not_a_complete_capture(self):
        self.tensor[0, 0] = float("nan")
        self.attach()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.execute()
        self.assertEqual(self.records()[0]["status"], "incomplete")

    def test_unknown_request_does_not_fall_back_to_ordinal_matching(self):
        self.request_id = "a-random-engine-id"
        self.attach()
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            self.execute()
        self.assertEqual(self.records()[0]["status"], "incomplete")

    def test_actual_config_guards_and_modified_manifest(self):
        self.config.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "enable_prefix_caching"):
            self.attach()
        self.config.cache_config.enable_prefix_caching = False
        (self.root / "manifest.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.attach()

    def test_rm_or_disabled_probe_does_not_intercept_generation(self):
        self.config.model_config.hf_config.audio_config.model_type = "other"
        self.assertFalse(self.attach())
        self.request_id = "judge-random-id"
        self.execute()
        self.assertEqual(self.records(), [])
        with patch.dict(os.environ, {probe.DIRECTORY_ENV: ""}):
            self.assertFalse(probe.attach_model_probe(self.native, self.config))
            self.assertFalse(probe.install_runtime_probe())

    def test_missing_positions_or_audio_mask_fails_visibly(self):
        self.positions = None
        self.attach()
        with self.assertRaisesRegex(ValueError, "positions"):
            self.execute()
        self.positions = torch.arange(6)
        self.features = []
        with self.assertRaisesRegex(ValueError, "no multimodal"):
            self.execute()
        self.assertTrue(all(r["status"] == "incomplete" for r in self.records()))

    def test_raw_bytes_distinguish_signed_zero_without_float_conversion(self):
        self.tensor[0, 0] = -0.0
        self.attach()
        self.execute()
        record, = self.records()
        raw = self.tensor[:4].view(torch.uint8).contiguous().numpy().tobytes()
        self.assertEqual(record["embeddings"]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_cross_case_id_fails(self):
        self.request_id = "huanyuab.%s.%s" % ("b" * 32, SAMPLE)
        self.attach()
        with self.assertRaisesRegex(ValueError, "another case"):
            self.execute()

    def test_no_implicit_two_call_limit(self):
        self.attach()
        for index in range(100):
            sample = "row_%06d" % index
            self.native._huanyu_embedding_probe_state["rows"][sample] = dict(self.row, sample_id=sample)
            self.request_id = "huanyuab.%s.%s" % (CASE, sample)
            self.execute()
        self.assertEqual(len(self.records()), 100)


if __name__ == "__main__":
    unittest.main()
