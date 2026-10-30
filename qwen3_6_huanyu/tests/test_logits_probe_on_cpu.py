"""Actual CPU forward/sampler calls exercise observation and report boundaries."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = "_huanyu_logits_cpu"
package = ModuleType(PKG)
package.__path__ = [str(ROOT)]
sys.modules[PKG] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load(PKG + ".logits_probe", ROOT / "logits_probe.py")
compare = load("huanyu_logits_compare_cpu", ROOT / "examples/compare_huanyu_logits.py")
launcher = load("huanyu_logits_launcher_cpu", ROOT / "examples/launch_huanyu_aqa_text_ab.py")
CASE, SAMPLE = "b" * 32, "row_000001"
REQUEST = "huanyuab." + CASE + "." + SAMPLE


class GreedySampler(torch.nn.Module):
    @staticmethod
    def greedy_sample(logits):
        return logits.argmax(dim=-1).view(-1)

    def sample(self, logits, sampling_metadata):
        if sampling_metadata.all_random:
            return torch.tensor([0]), None
        greedy = self.greedy_sample(logits)
        return greedy, None

    def forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        self.received = logits
        logits = logits.float()
        # Exercise the real risk: processors mutate the same input storage.
        logits[:, 0] += sampling_metadata.frequency_penalties[0]
        tokens, _ = self.sample(logits, sampling_metadata)
        return NS(sampled_token_ids=tokens[:, None].int())


class NativeModel(torch.nn.Module):
    def compute_logits(self, hidden_states, sampling_metadata=None):
        return hidden_states


_ascend_fixture = NS(enabled=False, rank=0, global_values=[3.0, 5.0], global_ids=[2, 5])


def get_ascend_config():
    return NS(enable_reduce_sample=_ascend_fixture.enabled)


def get_tp_group():
    class Group:
        rank_in_group = _ascend_fixture.rank

        def all_gather(self, value, dim):
            return torch.tensor([_ascend_fixture.global_values if value.dtype.is_floating_point
                                 else _ascend_fixture.global_ids], dtype=value.dtype)
    return Group()


class SuppliedAscendSampler(GreedySampler):
    @staticmethod
    def greedy_sample(logits):
        # Same decision algorithm as the server source supplied by the user.
        if get_ascend_config().enable_reduce_sample:
            tp_group = get_tp_group()
            _, local_width = logits.shape
            local_max_logits, local_max_indices = logits.max(dim=-1)
            local_global_idx = local_max_indices + tp_group.rank_in_group * local_width
            gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
            gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
            global_max_rank = gathered_logits.argmax(dim=-1)
            return gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1)).squeeze(-1)
        return logits.argmax(dim=-1).view(-1)


class LogitsProbeTests(unittest.TestCase):
    def setUp(self):
        base = ROOT.parents[2] / "artifacts/logits_probe_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = {"schema_version": 1, "case_id": CASE, "model_path": str(self.root / "model"),
                     "sample_size": 1, "embedding_check": {"enabled": True, "expected_tp_size": 1},
                     "logits_check": {"enabled": True, "max_steps": 4},
                     "engine": {"tensor_parallel_size": 1},
                     "sampling": {"temperature": 0, "top_p": 1, "top_k": -1, "n": 1,
                                  "seed": 42, "max_tokens": 4, "min_tokens": 0, "ignore_eos": False,
                                  "repetition_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0}}
        self.rows = [{"sample_id": SAMPLE, "case_id": CASE}]

    def runner(self, side, *, rank=0, tp=1, reduced=False, split=False, max_steps=None,
               penalty=0.0, random=False, params=None, bridge=0.0, sampler_type=GreedySampler):
        native = NativeModel()
        state = {"root": self.root, "side": side, "case": self.case, "rows": {SAMPLE: self.rows[0]},
                 "tp_rank": rank, "tp_size": tp, "history": {},
                 "max_steps": max_steps or self.case["logits_check"]["max_steps"],
                 "runtime_id": side + str(rank), "config": {"dtype": "bfloat16"}}
        setattr(native, probe._MARKER, state)
        sampling = dict(self.case["sampling"], stop=[], stop_token_ids=[], all_stop_token_ids=[])
        sampling.update(params or {})
        request = NS(prompt_token_ids=[8, 9, 10], output_token_ids=[], sampling_params=NS(**sampling))
        test = self

        class Runner:
            def __init__(self):
                self.model = native
                self.sampler = sampler_type()
                self.input_batch = NS(req_ids=[REQUEST], num_computed_tokens_cpu=np.array([0]))
                self.requests = {REQUEST: request}
                self.metadata = NS(all_greedy=not random, all_random=random, temperature=torch.tensor([0.0 if not random else 1.0]),
                                   top_p=None, top_k=None, frequency_penalties=torch.tensor([penalty]),
                                   generators={}, max_num_logprobs=None, logitsprocs=NS(argmax_invariant=[], non_argmax_invariant=[]))
                self.vector = torch.tensor([[1.0, 2.0, 3.0, 0.0]])

            def execute_model(self, scheduler_output):
                logits = self.model.compute_logits(self.vector)
                logits[:, 1] += bridge
                if split:
                    self.pending = logits
                    return None
                return self._do_sample(logits)

            def sample_tokens(self):
                return self._do_sample(self.pending)

            def _do_sample(self, logits):
                result = self.sampler(logits, self.metadata)
                request.output_token_ids.extend(result.sampled_token_ids.reshape(-1).tolist())
                return result

        probe.bind_runner_class(Runner)
        runner = Runner()
        with patch.object(probe, "_reduce_sample", return_value=reduced):
            probe._bind_sampler(runner, state)
        return runner

    def step(self, runner, vector=None, split=False):
        request = runner.requests[REQUEST]
        prefix = len(request.output_token_ids)
        count = len(request.prompt_token_ids) if prefix == 0 else 1
        runner.input_batch.num_computed_tokens_cpu[0] = 0 if prefix == 0 else len(request.prompt_token_ids) + prefix - 1
        runner.vector = torch.tensor([vector or [1.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
        result = runner.execute_model(NS(num_scheduled_tokens={REQUEST: count}))
        return runner.sample_tokens() if split else result

    def events(self, side):
        return [(p, compare.read(p)) for p in (self.root / "logits" / side).glob("*.json")
                if compare.read(p)["kind"] == "step"]

    def report(self):
        return compare.compare_case(self.root, self.case, self.rows)

    def both(self, n=4):
        for side in ("serve", "rollout"):
            runner = self.runner(side)
            for _ in range(n):
                self.step(runner)

    def test_prefill_and_every_decode_step_match(self):
        self.both()
        report = self.report()
        self.assertEqual(report["summary"]["status"], "PASS")
        item = report["samples"][0]
        self.assertEqual(len(item["prefill"]), 1)
        self.assertEqual(len(item["decode"]), 3)
        self.assertTrue(item["full_generation_captured"])

    def test_observer_does_not_change_actual_forward_and_preserves_pre_mutation_bytes(self):
        runner = self.runner("serve", penalty=4.0)
        output = self.step(runner)
        self.assertEqual(output.sampled_token_ids.item(), 0)
        self.assertIs(runner.sampler.received, runner.vector)
        path, event = self.events("serve")[0]
        arrays = compare.load_vectors(path, event)
        np.testing.assert_array_equal(arrays["model_logits"], [[1, 2, 3, 0]])
        np.testing.assert_array_equal(arrays["sampler_input"], [[1, 2, 3, 0]])
        np.testing.assert_array_equal(arrays["greedy_input"], [[5, 2, 3, 0]])

    def test_separate_sample_tokens_lifecycle(self):
        runner = self.runner("serve", split=True)
        self.step(runner, split=True)
        self.step(runner, split=True)
        events = sorted([r for _, r in self.events("serve")], key=lambda r: r["step"])
        self.assertEqual([r["phase"] for r in events], ["prefill", "decode"])
        self.assertEqual(events[1]["prefix_token_ids"], [2])

    def test_prefill_model_difference_is_identified(self):
        self.step(self.runner("serve"))
        self.step(self.runner("rollout"), [1, 2, 3.1, 0])
        item = self.report()["samples"][0]
        self.assertEqual(item["first_different_logits"]["boundary"], "model_logits")
        self.assertEqual(item["first_different_logits"]["phase"], "prefill")
        self.assertIsNone(item["first_different_token"])

    def test_first_decode_divergence_and_later_prefixes_not_compared(self):
        for side in ("serve", "rollout"):
            runner = self.runner(side)
            self.step(runner)
            self.step(runner, [1, 2, 3, 0] if side == "serve" else [1, 4, 3, 0])
            self.step(runner)
        item = self.report()["samples"][0]
        self.assertEqual(item["first_different_logits"]["step"], 1)
        self.assertEqual(item["first_different_logits"]["phase"], "decode")
        self.assertEqual(item["first_different_token"]["step"], 1)
        self.assertEqual(item["post_divergence_steps"], 1)
        self.assertEqual([r["step"] for r in item["decode"]], [1])

    def test_logits_processing_difference(self):
        self.step(self.runner("serve"))
        self.step(self.runner("rollout", penalty=0.25))
        item = self.report()["samples"][0]
        self.assertEqual(item["first_different_logits"]["boundary"], "greedy_input")
        self.assertTrue(item["sampling_parameter_differences"])

    def test_between_model_and_sampler_difference(self):
        self.step(self.runner("serve"))
        self.step(self.runner("rollout", bridge=0.25))
        self.assertEqual(self.report()["samples"][0]["first_different_logits"]["boundary"], "sampler_input")

    def test_both_sides_overridden_same_way_still_detected(self):
        for side in ("serve", "rollout"):
            runner = self.runner(side, params={"repetition_penalty": 1.1})
            for _ in range(4):
                self.step(runner)
        item = self.report()["samples"][0]
        diff = item["sampling_parameter_differences"][0]
        self.assertFalse(diff["between_sides"])
        self.assertIn("repetition_penalty", diff["vs_case"]["serve"])
        self.assertEqual(item["status"], "DIFFERENT")

    def test_logprobs_only_difference_not_misreported_as_sampling_drift(self):
        self.step(self.runner("serve", params={"logprobs": None}))
        self.step(self.runner("rollout", params={"logprobs": 0}))
        item = self.report()["samples"][0]
        self.assertFalse(item["sampling_parameter_differences"])
        self.assertIn("logprobs", item["prefill"][0]["diagnostic_parameter_differences"])

    def test_random_sampling_override_never_passes(self):
        self.step(self.runner("serve"))
        self.step(self.runner("rollout", random=True, params={"temperature": 1.0}))
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")
        self.assertTrue(self.report()["summary"]["errors"])

    def test_capture_limit_does_not_claim_full_generation(self):
        self.case["logits_check"]["max_steps"] = 2
        self.both(4)
        self.assertEqual(len(self.events("serve")), 2)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "MATCH_CAPTURED_PREFIX")
        self.assertFalse(item["full_generation_captured"])

    def test_eos_is_full_generation_evidence(self):
        for side in ("serve", "rollout"):
            self.step(self.runner(side, params={"all_stop_token_ids": [2]}))
        self.assertEqual(self.report()["summary"]["status"], "PASS")

    def test_missing_decode_step_is_incomplete(self):
        self.both()
        for path, event in self.events("rollout"):
            if event["step"] == 1:
                path.unlink()
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_missing_tail_not_confused_with_explicit_capture_limit(self):
        self.both(3)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "INCOMPLETE")
        self.assertFalse(item["full_generation_captured"])
        self.assertTrue(any("termination evidence" in x["error"] for x in item["issues"]))

    def test_duplicate_step_is_incomplete(self):
        self.both(1)
        path, _ = self.events("serve")[0]
        path.with_name("duplicate.json").write_bytes(path.read_bytes())
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_corrupt_payload_is_incomplete(self):
        self.both(1)
        path, event = self.events("serve")[0]
        (path.parent / event["tensor_file"]).write_bytes(b"not original")
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_missing_tp_rank_is_incomplete(self):
        self.case["engine"]["tensor_parallel_size"] = 2
        for side in ("serve", "rollout"):
            self.step(self.runner(side, tp=2))
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_unequal_vocab_layout_not_misdiagnosed_as_model_math(self):
        self.step(self.runner("serve"))
        self.step(self.runner("rollout", reduced=True))
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "INCOMPLETE")
        self.assertIsNone(item["first_different_logits"])
        self.assertTrue(item["runtime_differences"])

    def test_stale_async_prefix_rejected(self):
        runner = self.runner("serve")
        self.step(runner)
        runner.requests[REQUEST].output_token_ids.clear()
        with self.assertRaisesRegex(ValueError, "actual prefix differs"):
            self.step(runner)

    def test_bf16_negative_infinity_and_signed_zero_preserved(self):
        summary, raw = probe.capture_vector(torch.tensor([[-0.0, float("-inf"), 2.0]], dtype=torch.bfloat16))
        self.assertEqual(summary["negative_inf_count"], 1)
        self.assertEqual(summary["argmax_local"], 2)
        self.assertEqual(raw.shape, (1, 6))
        other, _ = probe.capture_vector(torch.tensor([[0.0, float("-inf"), 2.0]], dtype=torch.bfloat16))
        self.assertNotEqual(summary["sha256"], other["sha256"])

    def test_nan_and_bad_batch_rejected(self):
        for tensor in (torch.tensor([[float("nan"), 1.0]]), torch.zeros(2, 4), torch.tensor([[float("-inf")]])):
            with self.assertRaises(ValueError):
                probe.capture_vector(tensor)

    def test_disable_does_not_import_runner_or_change_model(self):
        state = {"case": {}}
        model = NativeModel()
        with patch.object(probe.importlib, "import_module", side_effect=AssertionError("must not import")):
            self.assertFalse(probe.attach_model_probe(model, None, state))

    def test_model_attachment_installs_actual_runner_and_is_idempotent(self):
        runner = self.runner("serve")
        embedding_state = getattr(runner.model, probe._MARKER)
        delattr(runner.model, probe._MARKER)
        runner.sampler = GreedySampler()
        self.case["logits_check"]["enabled"] = True
        config = NS(scheduler_config=NS(async_scheduling=False))
        with patch.object(probe.importlib, "import_module", return_value=NS(NPUModelRunner=type(runner))):
            self.assertTrue(probe.attach_model_probe(runner.model, config, embedding_state))
            self.assertFalse(probe.attach_model_probe(runner.model, config, embedding_state))
        self.step(runner)
        self.step(runner)
        self.assertEqual(len(self.events("serve")), 2)

    def test_async_scheduling_rejected_at_attachment(self):
        runner = self.runner("serve")
        state = getattr(runner.model, probe._MARKER)
        delattr(runner.model, probe._MARKER)
        self.case["logits_check"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "async_scheduling"):
            probe.attach_model_probe(runner.model, NS(scheduler_config=NS(async_scheduling=True)), state)

    def test_supplied_ascend_reduce_sample_flag_is_read_from_real_function_globals(self):
        with patch.object(_ascend_fixture, "enabled", True):
            self.assertTrue(probe._reduce_sample(SuppliedAscendSampler()))
        self.assertFalse(probe._reduce_sample(SuppliedAscendSampler()))

    def test_tp_sharded_vocab_uses_global_argmax_and_saves_every_rank(self):
        self.case["engine"]["tensor_parallel_size"] = 2
        self.case["sampling"]["max_tokens"] = 1
        with patch.object(_ascend_fixture, "enabled", True):
            for side in ("serve", "rollout"):
                for rank in range(2):
                    with patch.object(_ascend_fixture, "rank", rank):
                        runner = self.runner(side, rank=rank, tp=2, reduced=True, sampler_type=SuppliedAscendSampler)
                        self.assertEqual(self.step(runner, [1, 2, 3, 0] if rank == 0 else [1, 5, 2, 0]).sampled_token_ids.item(), 5)
        self.assertEqual(self.report()["summary"]["status"], "PASS")
        self.assertTrue(all(r.get("tensor_file") for _, r in self.events("serve")))

    def test_wrong_greedy_choice_is_reported_even_when_both_engines_agree(self):
        class WrongSampler(GreedySampler):
            @staticmethod
            def greedy_sample(logits):
                return torch.tensor([0])
        for side in ("serve", "rollout"):
            runner = self.runner(side, sampler_type=WrongSampler)
            for _ in range(4):
                self.step(runner)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "DIFFERENT")
        self.assertTrue(any(x.get("expected_global_argmax") == 2 for x in item["sampling_parameter_differences"]))

    def test_opaque_final_parameter_never_passes(self):
        for side in ("serve", "rollout"):
            self.step(self.runner(side, params={"custom": object()}))
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
