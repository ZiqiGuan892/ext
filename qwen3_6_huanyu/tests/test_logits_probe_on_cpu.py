"""Actual CPU forward/sampler calls exercise observation and report boundaries."""

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
                     "logits_check": {"enabled": True, "max_steps": 4, "top_k": 0},
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
        records = [(p, json.loads(p.read_text())) for p in (self.root / "logits" / side).glob("*.json")]
        return [(p, record) for p, record in records if record["kind"] == "step"]

    def both(self, n=4):
        for side in ("serve", "rollout"):
            runner = self.runner(side)
            for _ in range(n):
                self.step(runner)

    def test_prefill_and_every_decode_step_are_captured(self):
        self.both()
        for side in ("serve", "rollout"):
            records = sorted((record for _, record in self.events(side)), key=lambda record: record["step"])
            self.assertEqual([record["phase"] for record in records], ["prefill", "decode", "decode", "decode"])
            self.assertEqual([record["sampled_token_ids"] for record in records], [[2]] * 4)
            self.assertTrue(all(record["status"] == "complete" for record in records))
            self.assertEqual(records[-1]["terminal_evidence"], "max_tokens")

    def test_worker_determinism_is_observed_without_importing_or_mutating_ascend(self):
        module = ModuleType("vllm_ascend.batch_invariant")
        module._batch_invariant_LIB = object()
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {
            "HCCL_DETERMINISTIC": "strict", "ASCEND_LAUNCH_BLOCKING": "1"}):
            actual = probe._determinism_state()
        self.assertTrue(actual["batch_invariant_registered"])
        self.assertEqual(actual["environment"]["HCCL_DETERMINISTIC"], "strict")
        self.assertEqual(actual["environment"]["ASCEND_LAUNCH_BLOCKING"], "1")

    def test_observer_does_not_change_actual_forward_and_preserves_pre_mutation_bytes(self):
        runner = self.runner("serve", penalty=4.0)
        output = self.step(runner)
        self.assertEqual(output.sampled_token_ids.item(), 0)
        self.assertIs(runner.sampler.received, runner.vector)
        path, event = self.events("serve")[0]
        expected = {"model_logits": [[1, 2, 3, 0]], "sampler_input": [[1, 2, 3, 0]], "greedy_input": [[5, 2, 3, 0]]}
        with np.load(path.parent / event["tensor_file"], allow_pickle=False) as arrays:
            for stage, values in expected.items():
                summary = event["logits"][stage]
                self.assertEqual(summary["dtype"], "torch.float32")
                captured = arrays[event["tensor_keys"][stage]].view(np.float32).reshape(summary["shape"])
                np.testing.assert_array_equal(captured, values)

    def test_separate_sample_tokens_lifecycle(self):
        runner = self.runner("serve", split=True)
        self.step(runner, split=True)
        self.step(runner, split=True)
        events = sorted([r for _, r in self.events("serve")], key=lambda r: r["step"])
        self.assertEqual([r["phase"] for r in events], ["prefill", "decode"])
        self.assertEqual(events[1]["prefix_token_ids"], [2])

    def test_stale_async_prefix_rejected(self):
        runner = self.runner("serve")
        self.step(runner)
        runner.requests[REQUEST].output_token_ids.clear()
        with self.assertRaisesRegex(ValueError, "actual prefix differs"):
            self.step(runner)

    def test_bf16_negative_infinity_and_signed_zero_preserved(self):
        summary, raw = probe.capture_vector(torch.tensor([[-0.0, float("-inf"), 2.0]], dtype=torch.bfloat16), top_k=0)
        self.assertEqual(summary["negative_inf_count"], 1)
        self.assertEqual(summary["argmax_local"], 2)
        self.assertEqual(raw.shape, (1, 6))
        other, _ = probe.capture_vector(torch.tensor([[0.0, float("-inf"), 2.0]], dtype=torch.bfloat16))
        self.assertNotEqual(summary["sha256"], other["sha256"])

    def test_default_top100_bounded_storage_and_full_vocab_hash(self):
        tensor = torch.arange(248320, dtype=torch.float32).reshape(1, -1)
        summary, raw = probe.capture_vector(tensor)
        self.assertIsNone(raw)
        self.assertEqual(summary["stored_shape"], [1, 100])
        self.assertEqual(summary["shape"], [1, 248320])
        self.assertEqual(summary["topk"]["token_ids_local"], list(range(248319, 248219, -1)))
        self.assertLess(len(json.dumps(summary)), 5000)
        full, raw = probe.capture_vector(tensor, top_k=0)
        self.assertEqual(summary["sha256"], full["sha256"])
        self.assertEqual(raw.size, 248320 * 4)

    def test_compact_prefill_decode_capture_preserves_sampling_without_npz(self):
        self.case["logits_check"]["top_k"] = 100
        self.both()
        self.assertFalse(list((self.root / "logits").rglob("*.npz")))
        for side in ("serve", "rollout"):
            self.assertEqual(len(self.events(side)), 4)
            for _, record in self.events(side):
                self.assertEqual(record["tensor_storage"], "topk_json")
                self.assertEqual(record["sampling_params"]["top_k"], -1)
                self.assertEqual(record["sampled_token_ids"], [2])
                self.assertEqual(record["logits"]["model_logits"]["topk"]["token_ids_local"], [2, 1, 0, 3])

    def test_ties_at_topk_cutoff_choose_lowest_ids(self):
        tensor = torch.tensor([[2., 2., 2., 1., float("-inf")]], dtype=torch.bfloat16)
        summary, _ = probe.capture_vector(tensor, top_k=2)
        self.assertEqual(summary["topk"]["token_ids_local"], [0, 1])
        self.assertEqual(summary["max_tie_count"], 3)
        self.assertEqual(summary["top1_top2_margin"], 0)
        summary, _ = probe.capture_vector(tensor, top_k=100)
        self.assertEqual(summary["stored_shape"], [1, 5])
        self.assertEqual(summary["topk"]["logits"][-1], "-inf")

    def test_compact_sharded_vocab_still_verifies_global_argmax(self):
        self.case["logits_check"]["top_k"] = 100
        self.case["engine"]["tensor_parallel_size"] = 2
        self.case["sampling"]["max_tokens"] = 1
        with patch.object(_ascend_fixture, "enabled", True):
            for side in ("serve", "rollout"):
                for rank in range(2):
                    with patch.object(_ascend_fixture, "rank", rank):
                        runner = self.runner(side, rank=rank, tp=2, reduced=True, sampler_type=SuppliedAscendSampler)
                        self.step(runner, [1, 2, 3, 0] if rank == 0 else [1, 5, 2, 0])
        for side in ("serve", "rollout"):
            records = [record for _, record in self.events(side)]
            self.assertEqual({record["tp_rank"] for record in records}, {0, 1})
            self.assertEqual([record["sampled_token_ids"] for record in records], [[5], [5]])
        self.assertFalse(list((self.root / "logits").rglob("*.npz")))

    def test_invalid_capture_topk_rejected(self):
        for value in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "top_k"):
                probe.capture_vector(torch.ones(1, 4), top_k=value)

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
        for side in ("serve", "rollout"):
            records = [record for _, record in self.events(side)]
            self.assertEqual({record["tp_rank"] for record in records}, {0, 1})
            self.assertEqual([record["sampled_token_ids"] for record in records], [[5], [5]])
        self.assertTrue(all(r.get("tensor_file") for _, r in self.events("serve")))


if __name__ == "__main__":
    unittest.main()
