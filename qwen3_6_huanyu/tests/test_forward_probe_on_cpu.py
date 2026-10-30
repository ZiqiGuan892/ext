"""Exercise real boundary calls, owned snapshots, and fail-closed comparison."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest

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

    def forward(self, input_ids, positions, inputs_embeds=None):
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
                 "runtime_id": side + str(rank), "history": {}, "max_steps": 2}
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

    def test_first_difference_is_attention_output_not_outer_layer(self):
        self.both(delta=0.125)
        item = self.report()["samples"][0]
        self.assertEqual(item["status"], "DIFFERENT")
        first = item["first_difference"]
        self.assertEqual(first["step"], 0)
        self.assertEqual(first["first_difference"]["name"], "layers.1.self_attn")
        self.assertEqual(first["first_difference"]["boundary"], "output")

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
