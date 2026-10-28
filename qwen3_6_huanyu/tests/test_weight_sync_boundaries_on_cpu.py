"""Exercise receiver auditing around real Torch meta/materialization boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_sync_boundaries_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.weight_reload", ROOT / "weight_reload.py")
reload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reload)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Module()
        self.thinker.apm = torch.nn.Module()
        self.thinker.apm.conv_layers = torch.nn.ModuleList()
        for size in (40, 20):
            block = torch.nn.Module()
            block.layers = torch.nn.ModuleList([
                torch.nn.Identity(), torch.nn.LayerNorm(size), torch.nn.Identity(),
                torch.nn.Identity(), torch.nn.LayerNorm(size),
            ])
            self.thinker.apm.conv_layers.append(block)
        self.thinker.audio_projection_layer = torch.nn.Linear(2, 2)
        self.to(torch.bfloat16)

    def load_weights(self, weights):
        raise AssertionError("fixture contains only audio weights")


class BoundaryTests(unittest.TestCase):
    def run_reload(self, directory, corrupt=False):
        model = Model()
        incoming = []
        for prefix, module in (("model.audio_encoder.", model.thinker.apm),
                               ("model.audio_projection.", model.thinker.audio_projection_layer)):
            for name, param in module.named_parameters():
                incoming.append((prefix + name, torch.full_like(param, 0.25)))
        originals, pending = {}, []
        api = ModuleType(reload._RELOAD_MODULE)

        def initialize(current):
            for path, param in list(current.named_parameters()):
                parent, _, leaf = path.rpartition(".")
                layer = current.get_submodule(parent)
                originals[path] = param
                meta = torch.nn.Parameter(torch.empty_like(param, device="meta"), requires_grad=False)
                meta.weight_loader = lambda p, value, path=path: pending.append((path, value))
                setattr(layer, leaf, meta)

        def finalize(current, config):
            with torch.no_grad():
                for path, value in pending:
                    originals[path].copy_(value)
                for path, param in originals.items():
                    parent, _, leaf = path.rpartition(".")
                    setattr(current.get_submodule(parent), leaf, param)
                if corrupt:
                    current.thinker.apm.conv_layers[0].layers[1].weight[39] += 1

        api.initialize_layerwise_reload = initialize
        api.finalize_layerwise_reload = finalize

        def receive(on_bucket_received):
            on_bucket_received(incoming[:1], False)
            on_bucket_received(incoming[1:], True)

        env = {"HUANYU_WEIGHT_SYNC_AUDIT_DIR": directory,
               "HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES": "1"}
        with patch.dict(sys.modules, {api.__name__: api}), patch.dict(os.environ, env):
            reload.receive_layerwise_weights(model, SimpleNamespace(),
                                             SimpleNamespace(receive_weights=receive))
        return model

    def records(self, directory):
        return [json.loads(line) for p in Path(directory).glob("weights-*.jsonl")
                for line in p.read_text(encoding="utf-8").splitlines()]

    def test_real_meta_cycle_has_complete_boundary_evidence(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR")) as directory:
            model = self.run_reload(directory)
            records = self.records(directory)
        for stage in ("before_reload", "after_initialize", "receiver_input", "receiver_owned", "after_finalize"):
            selected = [r for r in records if r["stage"] == stage]
            self.assertEqual(len(selected), 8, stage)
            self.assertTrue(all(r["round_id"] == 1 for r in selected))
        self.assertTrue(all(r["tensor"]["is_meta"] for r in records if r["stage"] == "after_initialize"))
        expected = {r["name"]: r["tensor"]["float32_sha256"] for r in records if r["stage"] == "receiver_owned"}
        actual = {r["name"]: r["tensor"]["float32_sha256"] for r in records if r["stage"] == "after_finalize"}
        self.assertEqual(expected, actual)
        self.assertEqual(model._huanyu_weight_sync_update_id, 1)

    def test_bad_finalize_is_recorded_before_guard_rejects_it(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR")) as directory:
            with self.assertRaisesRegex(RuntimeError, "conv_layers.0.layers.1.weight"):
                self.run_reload(directory, corrupt=True)
            records = self.records(directory)
        target = "model.audio_encoder.conv_layers.0.layers.1.weight"
        values = {r["stage"]: r["tensor"]["values"] for r in records
                  if r["name"] == target and not r["tensor"]["is_meta"]}
        self.assertEqual(values["receiver_owned"][39], 0.25)
        self.assertEqual(values["after_finalize"][39], 1.25)


if __name__ == "__main__":
    unittest.main()
