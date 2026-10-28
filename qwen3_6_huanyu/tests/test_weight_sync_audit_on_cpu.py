"""Tiny real-tensor boundary tests; no distributed group or model download."""

import hashlib
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
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_sync_audit_test", ROOT / "weight_sync_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
NAME = "model.audio_encoder.conv_layers.0.layers.1.weight"


class TinyModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def named_parameters(self):
        return iter(self.weights.items())


class Engine:
    def __init__(self, module, config):
        self.module, self.model_config = module, config
        self.consumed = []
        self.peft = object()
        self.pairs = [(name, value.clone()) for name, value in module.named_parameters()]

    def get_per_tensor_param(self, *args, **kwargs):
        def generator():
            for pair in self.pairs:
                self.consumed.append(pair[0])
                yield pair
        return generator(), self.peft


class LocalShard:
    """Mock DTensor boundary; values are real torch tensors, no collectives."""
    shape = (40,)
    placements = ("Shard(dim=0)",)
    device_mesh = SimpleNamespace(shape=(16,), get_coordinate=lambda: [13])

    def __init__(self, tensor):
        self.local, self.dtype = tensor, tensor.dtype

    def to_local(self):
        return self.local

    def full_tensor(self):
        raise AssertionError("Audit must not collect a full tensor")


class WeightSyncAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {
            "HUANYU_WEIGHT_SYNC_AUDIT_DIR": str(self.directory / "audit"),
            "HUANYU_WEIGHT_SYNC_AUDIT_MAX_UPDATES": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        audit._FILES.clear()

    def entries(self):
        return [json.loads(line) for p in (self.directory / "audit").glob("*.jsonl")
                for line in p.read_text(encoding="utf-8").splitlines()]

    def checkpoint(self):
        values = {name: torch.arange(40 if ".0." in name else 20, dtype=torch.float32) / 7
                  for name in sorted(audit.NAMES)}
        path = self.directory / "checkpoint"
        path.mkdir()
        save_file(values, str(path / "selected.safetensors"))
        # Unrelated shards must not even be opened for an indexed checkpoint.
        mapping = {name: "selected.safetensors" for name in values}
        mapping["unrelated.large_weight"] = "absent-unrelated.safetensors"
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        config = SimpleNamespace(path=str(path), hf_config=SimpleNamespace(
            audio_config={"model_type": "huanyu_conformer_0_6b"}))
        return values, config

    def test_disabled_and_excess_round_do_not_consume_input(self):
        def forbidden():
            raise AssertionError("should not consume")
            yield
        with patch.dict(os.environ, {"HUANYU_WEIGHT_SYNC_AUDIT_DIR": ""}):
            audit.record("receiver_input", forbidden())
            marker = object()
            self.assertIs(audit.start_actor_audit(marker, None), marker)
        audit.record("receiver_input", forbidden(), round_id=2)
        self.assertEqual(self.entries(), [])

    def test_aliases_hashes_and_values_capture_mutation_without_changing_input(self):
        value = torch.arange(40, dtype=torch.bfloat16)
        original = value.clone()
        aliases = ["apm." + NAME.removeprefix("model.audio_encoder."),
                   "thinker.apm." + NAME.removeprefix("model.audio_encoder.")]
        audit.record("before_reload", [(aliases[0], value), ("unrelated", value)], round_id=1)
        torch.testing.assert_close(value, original)
        value[0] = 9
        audit.record("receiver_input", [(aliases[1], value)], round_id=1)
        first, second = self.entries()
        self.assertEqual(first["name"], NAME)
        self.assertEqual(first["tensor"]["dtype"], "torch.bfloat16")
        self.assertEqual(first["tensor"]["values"], list(range(40)))
        expected = hashlib.sha256(original.float().numpy().tobytes()).hexdigest()
        self.assertEqual(first["tensor"]["float32_sha256"], expected)
        self.assertNotEqual(first["tensor"]["raw_sha256"], second["tensor"]["raw_sha256"])

    def test_meta_and_uneven_or_empty_local_shards_need_no_collective(self):
        audit.record("after_initialize", [(NAME, torch.empty(40, device="meta"))], round_id=1)
        audit.record("actor_local_shard", [(NAME, LocalShard(torch.tensor([39.])))], round_id=1)
        audit.record("actor_local_shard", [(NAME, LocalShard(torch.empty(0)))], round_id=1)
        meta, local, empty = [row["tensor"] for row in self.entries()]
        self.assertTrue(meta["is_meta"])
        self.assertNotIn("raw_sha256", meta)
        self.assertEqual(local["global_shape"], [40])
        self.assertEqual(local["shape"], [1])
        self.assertEqual(local["values"], [39.])
        self.assertEqual(local["mesh_coordinate"], [13])
        self.assertEqual(empty["values"], [])
        self.assertTrue(empty["finite"])

    def test_nonfinite_values_are_valid_json_and_large_values_are_bounded(self):
        audit.record("receiver_input", [(NAME, torch.tensor([float("nan"), float("inf")]))])
        audit.record("receiver_input", [(NAME, torch.zeros(257))])
        first, second = self.entries()
        self.assertFalse(first["tensor"]["finite"])
        self.assertEqual(first["tensor"]["values"], ["nan", "inf"])
        self.assertNotIn("values", second["tensor"])

    def test_cpu_snapshot_waits_for_initialized_offload_backends(self):
        npu = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: None)
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}), \
                patch.object(torch, "npu", npu, create=True), \
                patch.object(npu, "synchronize") as npu_sync, \
                patch.object(torch.cuda, "is_initialized", return_value=True), \
                patch.object(torch.cuda, "synchronize") as cuda_sync:
            audit.record("actor_local_shard", [(NAME, torch.tensor([1.]))], round_id=1)
            npu_sync.assert_called_once_with()
            cuda_sync.assert_called_once_with()
        self.assertEqual(self.entries()[0]["tensor"]["diagnostic_sync"], ["npu", "cuda"])

    def test_actor_checkpoint_export_stages_preserve_raw_dtype_and_lazy_pairs(self):
        weights, config = self.checkpoint()
        model = TinyModel({name: value.to(torch.bfloat16) for name, value in weights.items()})
        impl = ModuleType("verl_omni.workers.engine.fsdp.omni_impl")
        impl.OmniFSDPEngine = type("AuditedEngine", (Engine,), {})
        with patch.dict(sys.modules, {impl.__name__: impl}):
            self.assertIs(audit.start_actor_audit(model, config), model)
            self.assertEqual(len(self.entries()), 16)
            rows = self.entries()
            self.assertTrue(all(r["tensor"]["dtype"] == "torch.float32"
                                for r in rows if r["stage"] == "checkpoint"))
            self.assertTrue(all(r["tensor"]["dtype"] == "torch.bfloat16"
                                for r in rows if r["stage"] == "actor_loaded"))
            engine = impl.OmniFSDPEngine(model, config)
            generator, peft = engine.get_per_tensor_param()
            self.assertIs(peft, engine.peft)
            self.assertEqual(engine.consumed, [])
            exported = list(generator)
            for observed, original in zip(exported, engine.pairs):
                self.assertIs(observed, original)
                self.assertIs(observed[1], original[1])
            entries = self.entries()
            self.assertEqual(len(entries), 32)
            self.assertEqual([r["name"] for r in entries if r["stage"] == "actor_export"], list(weights))
            self.assertTrue(all(r["round_id"] == 1 for r in entries if r["stage"] == "actor_export"))
            self.assertTrue(entries[-1]["extra"]["export_source"]["method_sha256"])
            list(engine.get_per_tensor_param()[0])
            self.assertEqual(len(self.entries()), 32)
            engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(audio_config={"model_type": "whisper"}))
            list(engine.get_per_tensor_param()[0])
            self.assertEqual(len(self.entries()), 32)

    def test_opt_in_checkpoint_missing_parameters_fails_visibly(self):
        _, config = self.checkpoint()
        (Path(config.path) / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        with self.assertRaisesRegex(ValueError, "missing Huanyu audit weights"):
            audit.start_actor_audit(TinyModel({}), config)


if __name__ == "__main__":
    unittest.main()
