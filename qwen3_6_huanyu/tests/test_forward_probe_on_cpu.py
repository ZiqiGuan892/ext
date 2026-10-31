"""Exercise real boundary calls, owned snapshots, and fail-closed comparison."""

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = "_huanyu_forward_cpu"
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
forward = load(PKG + ".forward_probe", ROOT / "forward_probe.py")
embedding = sys.modules.get(PKG + ".prompt_embedding_probe") or load(
    PKG + ".prompt_embedding_probe", ROOT / "prompt_embedding_probe.py")
compare = load("huanyu_forward_compare_cpu", ROOT / "examples/compare_huanyu_forward.py")
CASE, SAMPLE = "a" * 32, "row_000001"
REQUEST = "huanyuab." + CASE + "." + SAMPLE


class Norm(torch.nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states * 0.5
        result = hidden_states + residual
        return result * 0.5, result


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = 0.0
        self.calls = 0

    def forward(self, hidden_states, positions=None):
        self.calls += 1
        return hidden_states + self.delta


class Layer(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.layer_type = kind
        self.use_attn_reduce_scatter_for_moe = False
        self.layer_scale = False
        self.input_layernorm = Norm()
        self.post_attention_layernorm = Norm()
        if kind == "linear_attention":
            self.linear_attn = Attention()
        else:
            self.self_attn = Attention()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden_states, residual, positions=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class LLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_sequence_parallel = False
        self.start_layer, self.end_layer = 0, 2
        self.layers = torch.nn.ModuleList([Layer("linear_attention"), Layer("full_attention")])
        self.norm = Norm()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.received = (input_ids, positions, intermediate_tensors, inputs_embeds)
        hidden_states, residual = inputs_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
        return self.norm(hidden_states, residual)[0]


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = NS(llm=NS(model=LLM()))

    def compute_logits(self, hidden_states):
        return hidden_states[-1:, :]


class GreedySampler(torch.nn.Module):
    @staticmethod
    def greedy_sample(logits):
        return logits.argmax(dim=-1).view(-1)

    def sample(self, logits, sampling_metadata):
        return self.greedy_sample(logits), None

    def forward(self, logits, sampling_metadata, predict_bonus_token=False):
        selected, _ = self.sample(logits.float(), sampling_metadata)
        return NS(sampled_token_ids=selected[:, None].int())


class ForwardProbeTests(unittest.TestCase):
    def setUp(self):
        base = ROOT.parents[2] / "artifacts/forward_probe_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = {"case_id": CASE, "engine": {"tensor_parallel_size": 1},
                     "logits_check": {"enabled": True, "max_steps": 2},
                     "forward_check": {"enabled": True, "max_steps": 2}}
        self.rows = [{"sample_id": SAMPLE}]

    def runner(self, side, *, rank=0, delta=0.0, sequence_parallel=False):
        native = Native()
        native.thinker.llm.model.layers[1].self_attn.delta = delta
        native.thinker.llm.model.use_sequence_parallel = sequence_parallel
        state = {"root": self.root, "side": side, "case": self.case, "rows": {SAMPLE: {}},
                 "tp_rank": rank, "tp_size": self.case["engine"]["tensor_parallel_size"],
                 "runtime_id": side + str(rank), "history": {}, "max_steps": self.case["logits_check"]["max_steps"]}
        setattr(native, probe._MARKER, state)
        runner = NS(model=native, input_batch=NS(req_ids=[REQUEST]),
                    requests={REQUEST: NS(prompt_token_ids=[8, 9], output_token_ids=[])})
        probe._bind_model(runner)
        return runner, state

    def step(self, runner, state, prefix=()):
        runner.requests[REQUEST].output_token_ids = list(prefix)
        state["history"][REQUEST] = list(prefix)
        context = {"runner": runner, "state": state}
        token = probe._ACTIVE.set(context)
        try:
            x = torch.arange(8, dtype=torch.float32).reshape(2, 4) if not prefix else torch.ones(1, 4)
            llm = runner.model.thinker.llm.model
            y = llm(None, torch.arange(x.shape[0]), inputs_embeds=x)
            runner.model.compute_logits(y)
            runner._huanyu_pending_logits = None
            return y
        finally:
            probe._ACTIVE.reset(token)

    def both(self, delta=0.0, sequence_parallel=False):
        for side in ("serve", "rollout"):
            runner, state = self.runner(side, delta=delta if side == "rollout" else 0.0,
                                        sequence_parallel=sequence_parallel if side == "rollout" else False)
            self.step(runner, state)
            self.step(runner, state, [1])

    def report(self):
        return compare.compare_case(self.root, self.case, self.rows)

    def test_equal_prefill_and_decode(self):
        self.both()
        r = self.report()
        self.assertEqual(r["summary"]["status"], "MATCH_CAPTURED_BOUNDARIES")
        self.assertEqual([s["phase"] for s in r["samples"][0]["steps"]], ["prefill", "decode"])
        self.assertFalse(list((self.root / "forward").rglob("*.npz")))

    def test_embedding_forward_and_sampler_probes_compose_with_native_call_signature(self):
        self.full_case(limit=2)
        self.case.update(model_path="/fixture", embedding_check={"expected_tp_size": 1})
        config = NS(model_config=NS(model="/fixture", enforce_eager=True, hf_config=NS(
            audio_config=NS(model_type="huanyu_conformer_0_6b"))),
            scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False),
            cache_config=NS(enable_prefix_caching=False), parallel_config=NS(tensor_parallel_size=1))
        for style in ("native", "positional", "keyword", "direct"):
            self.root = self.root / style
            self.root.mkdir()
            for side in ("serve", "rollout"):
                native = Native()
                llm = native.thinker.llm.model
                expected_signature = inspect.signature(llm.forward)
                # Install the actual embedding wrapper first, as the worker
                # does. Stub environment discovery, not capture or forwarding.
                state = {"root": self.root, "side": side, "case": self.case,
                         "rows": {SAMPLE: {"wav_sha256": ["1" * 64]}}, "history": {},
                         "tp_rank": 0, "tp_size": 1, "runtime_id": side, "max_steps": 2, "config": {}}
                setattr(native, probe._MARKER, state)
                with patch.object(embedding, "_enabled", return_value=True), \
                        patch.object(embedding, "_load_case", return_value=(self.root, side, self.case, state["rows"])), \
                        patch.object(embedding, "_tp_identity", return_value=(0, 1)), \
                        patch.object(embedding, "install_runtime_probe", return_value=True):
                    embedding.attach_model_probe(native, config)
                # Before this fix inspect.signature drops input_ids here.
                self.assertEqual(inspect.signature(llm.forward), expected_signature)
                request = NS(prompt_token_ids=[8, 9], output_token_ids=[],
                             mm_features=[NS(modality="audio", mm_position=NS(offset=0, length=1, is_embed=None))],
                             sampling_params=NS(temperature=0, top_p=1, top_k=-1, n=1, seed=42, max_tokens=2,
                                 min_tokens=0, ignore_eos=False, repetition_penalty=1, frequency_penalty=0,
                                 presence_penalty=0, stop=[], stop_token_ids=[], all_stop_token_ids=[]))
                runner = NS(model=native, sampler=GreedySampler(), execute_model=native.compute_logits,
                            input_batch=NS(req_ids=[REQUEST], num_computed_tokens_cpu=[0]), requests={REQUEST: request})
                signatures = [inspect.signature(method) for method in (
                    native.compute_logits, runner.sampler.forward, runner.sampler.greedy_sample)]
                probe._bind_model(runner)
                probe._bind_sampler(runner, state)
                self.assertEqual(inspect.signature(llm.forward), expected_signature)
                self.assertEqual([inspect.signature(method) for method in (
                    native.compute_logits, runner.sampler.forward, runner.sampler.greedy_sample)], signatures)
                # Wrappers must preserve normal Python errors, not swallow
                # genuinely duplicated arguments to work around inspection.
                with self.assertRaisesRegex(TypeError, "multiple values.*inputs_embeds"):
                    llm(None, None, None, object(), inputs_embeds=object())
                for step in range(2):
                    x = torch.arange(8.).reshape(2, 4) if step == 0 else torch.ones(1, 4)
                    ids, positions = torch.zeros(x.shape[0], dtype=torch.long), torch.arange(x.shape[0])
                    scheduler = NS(num_scheduled_tokens={REQUEST: x.shape[0]}, total_num_scheduled_tokens=x.shape[0])
                    runner.input_batch.num_computed_tokens_cpu[0] = 0 if step == 0 else 2
                    ctx = {"runner": runner, "state": state, "scheduler": scheduler}
                    tokens = (probe._ACTIVE.set(ctx), embedding._ACTIVE.set(dict(ctx)))
                    try:
                        if style == "native":
                            output = llm(ids, positions, None, inputs_embeds=x)
                        elif style == "positional":
                            output = llm(ids, positions, None, x)
                        elif style == "keyword":
                            output = llm(input_ids=ids, positions=positions, intermediate_tensors=None, inputs_embeds=x)
                        else:
                            output = llm.forward(ids, positions, None, inputs_embeds=x)
                        self.assertIs(llm.received[0], ids)
                        self.assertIs(llm.received[1], positions)
                        self.assertIsNone(llm.received[2])
                        self.assertIs(llm.received[3], x)
                        torch.testing.assert_close(output, LLM()(ids, positions, None, x), rtol=0, atol=0)
                        logits = native.compute_logits(output)
                        result = runner.sampler(logits, NS(all_greedy=True, all_random=False, temperature=None,
                                                          top_k=None, top_p=None))
                        request.output_token_ids.extend(result.sampled_token_ids.reshape(-1).tolist())
                    finally:
                        embedding._ACTIVE.reset(tokens[1])
                        probe._ACTIVE.reset(tokens[0])
                self.assertEqual(llm.layers[0].linear_attn.calls, 2)
                records = list((self.root / "embeddings" / side).glob("*.json"))
                self.assertEqual(len(records), 1)
                self.assertEqual(json.loads(records[0].read_text())["status"], "complete")
            report = self.report()
            self.assertEqual(report["summary"]["status"], "MATCH_CAPTURED_BOUNDARIES")
            self.assertTrue(report["samples"][0]["full_generation_captured"])

    def test_first_difference_is_attention_output_not_outer_layer(self):
        self.both(delta=0.125)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "DIFFERENT")
        first = item["first_difference"]
        self.assertEqual(first["step"], 0)
        self.assertEqual(first["first_difference"]["name"], "layers.1.self_attn")
        self.assertEqual(first["first_difference"]["boundary"], "output")
        self.assertTrue(first["first_difference"]["module_input_equal"])
        self.assertEqual(first["first_difference"]["last_equal_boundary"],
                         {"name": "layers.1.self_attn", "boundary": "input"})

    def full_case(self, limit=4):
        self.case["sampling"] = {"max_tokens": limit}
        self.case["logits_check"]["max_steps"] = limit
        self.case["forward_check"].update(max_steps=limit, until_eos=True)

    def terminate(self, state, prefix, *, reason="stop_token", selected=9, status="complete"):
        prompt = [8, 9]
        event = dict(request_id=REQUEST, sample_id=SAMPLE, step=len(prefix),
                     phase="decode" if prefix else "prefill", prompt_token_ids=prompt,
                     prompt_token_sha256=compare.canonical_hash(prompt), prefix_token_ids=prefix,
                     prefix_token_sha256=compare.canonical_hash(prefix), sampled_token_ids=[selected],
                     terminal_evidence=reason, status=status,
                     sampling_params={"max_tokens": self.case["sampling"]["max_tokens"],
                                      "ignore_eos": False, "min_tokens": 0, "all_stop_token_ids": [9]})
        forward.record_termination(state, event)

    def full_both(self, lengths=(2, 2)):
        for side, count in zip(("serve", "rollout"), lengths):
            runner, state = self.runner(side)
            for step in range(count):
                self.step(runner, state, [1] * step)
            reason = "max_tokens" if count == self.case["sampling"]["max_tokens"] else "stop_token"
            self.terminate(state, [1] * (count - 1), reason=reason)

    def test_early_eos_is_complete_without_fictitious_missing_decode(self):
        self.full_case()
        self.full_both()
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "MATCH_CAPTURED_BOUNDARIES")
        self.assertEqual(item["issues"], [])
        self.assertTrue(item["full_generation_captured"])
        self.assertEqual([c["generated_steps"] for c in item["coverage"]], [2, 2])

    def test_real_sampler_closes_trace_after_separate_execute_and_sample_calls(self):
        self.full_case()
        for side in ("serve", "rollout"):
            runner, state = self.runner(side)
            state["config"] = {}
            runner.execute_model = runner.model.compute_logits
            runner.sampler = GreedySampler()
            runner.requests[REQUEST].sampling_params = NS(
                temperature=0, top_p=1, top_k=-1, n=1, seed=42, max_tokens=4,
                min_tokens=0, ignore_eos=False, repetition_penalty=1,
                frequency_penalty=0, presence_penalty=0, stop=[],
                stop_token_ids=[3], all_stop_token_ids=[3])
            metadata = NS(all_greedy=True, all_random=False, temperature=None,
                          top_p=None, top_k=None)
            probe._bind_sampler(runner, state)
            token = probe._ACTIVE.set({"runner": runner, "state": state})
            try:
                hidden = runner.model.thinker.llm.model(
                    None, torch.arange(2), inputs_embeds=torch.arange(8.).reshape(2, 4))
                logits = runner.model.compute_logits(hidden)
            finally:
                probe._ACTIVE.reset(token)
            self.assertFalse(list((self.root / "forward" / side).glob("*termination*.json")))
            # Real NPU runners may invoke sample_tokens in a separate context.
            token = probe._ACTIVE.set({"runner": runner, "state": state})
            try:
                result = runner.sampler(logits, metadata)
            finally:
                probe._ACTIVE.reset(token)
            self.assertEqual(result.sampled_token_ids.tolist(), [[3]])
            marker = next((self.root / "forward" / side).glob("*termination*.json"))
            self.assertEqual(json.loads(marker.read_text())["terminal_evidence"], "stop_token")
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "MATCH_CAPTURED_BOUNDARIES")
        self.assertTrue(item["full_generation_captured"])

    def test_first_token_eos_and_length_limit_are_both_valid(self):
        for count in (1, 4):
            with self.subTest(count=count):
                # Each subtest uses a separate case directory, not overwrites.
                self.root = self.root / str(count)
                self.root.mkdir()
                self.full_case()
                self.full_both((count, count))
                self.assertTrue(self.report()["samples"][0]["full_generation_captured"])

    def test_deletion_or_invalid_terminal_evidence_cannot_pass(self):
        self.full_case()
        self.full_both()
        marker = next((self.root / "forward/serve").glob("*termination*.json"))
        original = marker.read_bytes()
        for mode in ("missing", "bad_stop", "runtime", "history"):
            with self.subTest(mode=mode):
                if mode == "missing":
                    marker.unlink()
                else:
                    record = json.loads(original)
                    if mode == "bad_stop":
                        record["sampled_token_ids"] = [8]
                    elif mode == "runtime":
                        record["runtime_id"] = "other"
                    else:
                        record["prefix_token_ids"] = [7]
                        record["prefix_token_sha256"] = compare.canonical_hash([7])
                    marker.write_text(json.dumps(record), encoding="utf-8")
                self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")
                marker.write_bytes(original)

    def test_missing_middle_forward_and_duplicate_terminal_are_incomplete(self):
        self.full_case()
        self.full_both((4, 4))
        marker = next((self.root / "forward/serve").glob("*termination*.json"))
        duplicate = marker.with_name("duplicate.json")
        duplicate.write_bytes(marker.read_bytes())
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")
        duplicate.unlink()
        next((self.root / "forward/serve").glob("*-step00001-*.json")).unlink()
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_different_termination_lengths_are_reported_without_false_missing_steps(self):
        self.full_case()
        self.full_both((1, 2))
        item = self.report()["samples"][0]
        self.assertTrue(item["full_generation_captured"])
        self.assertEqual(item["unpaired_after_termination"], 1)

    def test_late_decode_difference_identifies_module_and_equal_input(self):
        self.full_case()
        for side in ("serve", "rollout"):
            runner, state = self.runner(side)
            for step in range(4):
                if side == "rollout" and step == 3:
                    runner.model.thinker.llm.model.layers[0].linear_attn.delta = 0.25
                self.step(runner, state, [1] * step)
            self.terminate(state, [1, 1, 1], reason="max_tokens")
        first = self.report()["samples"][0]["first_difference"]
        self.assertEqual(first["step"], 3)
        self.assertEqual(first["first_difference"]["name"], "layers.0.linear_attn")
        self.assertTrue(first["first_difference"]["module_input_equal"])

    def test_earliest_module_across_tp_ranks_wins_over_rank_zero(self):
        self.case["engine"]["tensor_parallel_size"] = 2
        for side in ("serve", "rollout"):
            for rank in range(2):
                runner, state = self.runner(side, rank=rank)
                if side == "rollout":
                    llm = runner.model.thinker.llm.model
                    target = llm.layers[0].linear_attn if rank == 1 else llm.layers[1].self_attn
                    target.delta = 0.25
                self.step(runner, state)
                self.step(runner, state, [1])
        first = self.report()["samples"][0]["first_difference"]
        self.assertEqual(first["tp_rank"], 1)
        self.assertEqual(first["first_difference"]["name"], "layers.0.linear_attn")

    def test_input_difference_is_not_attributed_to_module_internal_arithmetic(self):
        self.both()
        path = next((self.root / "forward/rollout").glob("*-step00000-*.json"))
        record = compare.read(path)
        entry = next(e for e in record["events"] if e["name"] == "layers.1.self_attn" and e["boundary"] == "input")
        entry["value"]["hidden_states"]["sha256"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")
        first = self.report()["samples"][0]["first_difference"]["first_difference"]
        self.assertEqual(first["boundary"], "input")
        self.assertFalse(first["module_input_equal"])

    def test_runtime_flags_not_hidden_by_identical_forward_source(self):
        self.both(sequence_parallel=True)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "DIFFERENT")
        self.assertIsNone(item["first_difference"])
        self.assertIn("use_sequence_parallel", item["steps"][0]["runtime_difference"]["path"])

    def test_observation_preserves_calls_and_results(self):
        expected = LLM()(None, torch.arange(2), inputs_embeds=torch.arange(8, dtype=torch.float32).reshape(2, 4))
        runner, state = self.runner("serve")
        result = self.step(runner, state)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(runner.model.thinker.llm.model.layers[0].linear_attn.calls, 1)
        self.assertEqual(runner.model.thinker.llm.model.layers[1].self_attn.calls, 1)

    def test_tensor_snapshot_owns_pre_mutation_bytes(self):
        tensor = torch.ones(2, 3, dtype=torch.bfloat16)
        before = forward.tensor_tree({"x": tensor})
        saved = copy.deepcopy(before)
        tensor.add_(1)
        self.assertEqual(before, saved)
        self.assertNotEqual(before["x"]["sha256"], forward.tensor_tree(tensor)["sha256"])

    def test_missing_tp_rank_is_incomplete(self):
        self.case["engine"]["tensor_parallel_size"] = 2
        self.both()
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_duplicate_record_is_incomplete(self):
        self.both()
        p = next((self.root / "forward/serve").glob("*.json"))
        p.with_name("duplicate.json").write_bytes(p.read_bytes())
        self.assertEqual(self.report()["summary"]["status"], "INCOMPLETE")

    def test_missing_internal_hook_is_incomplete(self):
        runner, state = self.runner("serve")
        layer = runner.model.thinker.llm.model.layers[0]
        # Simulate a bypassed instance hook in a different native version.
        layer.linear_attn.forward = layer.linear_attn.forward.__wrapped__
        self.step(runner, state)
        record = compare.read(next((self.root / "forward/serve").glob("*.json")))
        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertIn(["layers.0.linear_attn", "output"], record["missing_boundaries"])

    def test_different_prefix_not_used_to_accuse_forward(self):
        for side, prefix in (("serve", [1]), ("rollout", [2])):
            runner, state = self.runner(side)
            self.step(runner, state)
            self.step(runner, state, prefix)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "INCOMPLETE")
        self.assertIsNone(item["first_difference"])
        self.assertFalse(item["steps"][1]["same_prefix"])

    def test_disabled_probe_does_not_wrap_llm(self):
        self.case["forward_check"]["enabled"] = False
        runner, state = self.runner("serve")
        self.assertNotIn("forward_manifest", state)
        self.assertEqual(runner.model.thinker.llm.model.forward.__func__, LLM.forward)

    def test_requested_window_is_bounded(self):
        self.case["forward_check"]["max_steps"] = 1
        runner, state = self.runner("serve")
        self.step(runner, state)
        self.step(runner, state, [1])
        self.assertEqual(len(list((self.root / "forward/serve").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
