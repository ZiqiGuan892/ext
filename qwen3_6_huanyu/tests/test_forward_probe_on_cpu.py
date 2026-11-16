"""Exercise real boundary calls, owned snapshots, and capture completeness."""

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

    def test_prefill_and_decode_boundaries_are_captured(self):
        self.both()
        for side in ("serve", "rollout"):
            records = [json.loads(p.read_text()) for p in (self.root / "forward" / side).glob("*.json")]
            records.sort(key=lambda record: record["step"])
            self.assertEqual([record["phase"] for record in records], ["prefill", "decode"])
            self.assertEqual([record["status"] for record in records], ["CAPTURED", "CAPTURED"])
            self.assertTrue(all(not record["missing_boundaries"] for record in records))
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
            self.assertEqual(len(list((self.root / "forward").rglob("*termination*.json"))), 2)

    def full_case(self, limit=4):
        self.case["sampling"] = {"max_tokens": limit}
        self.case["logits_check"]["max_steps"] = limit
        self.case["forward_check"].update(max_steps=limit, until_eos=True)

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
        self.assertEqual(len(list((self.root / "forward").rglob("*termination*.json"))), 2)

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

    def test_missing_internal_hook_is_incomplete(self):
        runner, state = self.runner("serve")
        layer = runner.model.thinker.llm.model.layers[0]
        # Simulate a bypassed instance hook in a different native version.
        layer.linear_attn.forward = layer.linear_attn.forward.__wrapped__
        self.step(runner, state)
        record = json.loads(next((self.root / "forward/serve").glob("*.json")).read_text())
        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertIn(["layers.0.linear_attn", "output"], record["missing_boundaries"])

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
