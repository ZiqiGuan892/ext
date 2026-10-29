"""Whole-model audit with real tiny CPU tensors and a native-loader stand-in."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "huanyu_full_weight_audit_cpu_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(PACKAGE + ".full_weight_audit", ROOT / "full_weight_audit.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)
TMP = Path(os.environ.get("HUANYU_TEST_TMPDIR", ROOT.parents[2] / "artifacts" / "full-weight-audit-tests"))
TMP.mkdir(parents=True, exist_ok=True)


class FusedTPModel(nn.Module):
    """A fused shard that cannot be compared to a naive whole-checkpoint slice."""

    def __init__(self, weights, *, dtype=torch.bfloat16, rank=1, tie=True):
        super().__init__()
        self.rank = rank
        self.fused = nn.Parameter(torch.empty(4, 3, dtype=dtype))
        self.audio = nn.Parameter(torch.empty(3, dtype=dtype))
        if tie:
            self.alias = self.audio
        self.register_buffer("position", torch.arange(3), persistent=False)
        self.omit_audio = False
        self.load_error = False
        self.load_weights(iter(weights.items()))

    def load_weights(self, iterator):
        if self.load_error:
            raise RuntimeError("synthetic loader failure")
        values = dict(iterator)
        # Native fusion: shard the gate and up matrices individually, then fuse.
        begin = self.rank * 2
        with torch.no_grad():
            self.fused.copy_(torch.cat((values["gate"][begin:begin + 2],
                                        values["up"][begin:begin + 2]), dim=0))
            if not self.omit_audio:
                self.audio.copy_(values["audio"])
        return {"fused", "audio", "alias"}


class FullWeightAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = self.root / "checkpoint"
        self.ckpt.mkdir()
        self.weights = {"gate": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                        "up": torch.arange(12, 24, dtype=torch.float32).reshape(4, 3),
                        "audio": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)}
        self.write_checkpoint()
        self.model_config = SimpleNamespace(model=str(self.ckpt), dtype=torch.bfloat16, quantization=None)
        self.worker = SimpleNamespace(
            rank=0, local_rank=0,
            vllm_config=SimpleNamespace(load_config=SimpleNamespace(load_format="safetensors"),
                                         quant_config=None, lora_config=None))
        self.environment = patch.dict(os.environ, {
            audit.ENV: str(self.root / "reports"), audit.RUN_ENV: "run-unit-test",
            "HUANYU_CHECKPOINT_VALIDATION_MODE": "sync"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.events = []
        default = ModuleType("vllm.model_executor.model_loader.default_loader")
        config = ModuleType("vllm.config")
        owner = self

        class Loader:
            def __init__(self, load_config):
                owner.assertEqual(load_config.load_format, "safetensors")

            def get_all_weights(self, model_config, model):
                owner.assertIs(model_config, owner.model_config)
                owner.events.append("checkpoint_iterator")
                return iter(owner.weights.items())

        default.DefaultModelLoader = Loader
        config.LoadConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        self.modules = patch.dict(sys.modules, {default.__name__: default, config.__name__: config})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def write_checkpoint(self):
        header, chunks, offset = {}, [], 0
        for name, tensor in self.weights.items():
            chunk = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            dtype = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float64: "F64"}[tensor.dtype]
            header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(chunk)]}
            offset += len(chunk)
            chunks.append(chunk)
        raw = json.dumps(header).encode("utf-8")
        (self.ckpt / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))

    def initialize(self, model):
        self.events.append("initialize")
        self.assertTrue(all(bool(torch.isnan(p).all()) for p in model.parameters()))

    def finalize(self, model, config):
        self.assertIs(config, self.model_config)
        self.events.append("finalize")

    def execute(self, model, initial, finalize=None):
        path = audit.audit_first_sync(model, self.model_config, self.worker,
                                     self.initialize, finalize or self.finalize, initial=initial)
        self.assertTrue(model._huanyu_full_weight_audit_complete)
        self.assertTrue(model._huanyu_full_weight_audit_generation_forbidden)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_native_tp_fusion_tied_aliases_and_expected_rounding_pass(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        # Lossless BF16 -> FP32 Actor storage -> BF16 first export.
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(param.float().to(torch.bfloat16))
        result = self.execute(model, initial)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(set(result["parameters"]), {"fused", "audio", "alias"})
        self.assertEqual(result["summary"]["unique_parameter_objects"], 2)
        self.assertEqual(result["aliases"]["initial_disk"], {"alias": "audio"})
        self.assertEqual(result["checkpoint_stream"]["tensor_count"], 3)
        self.assertEqual(len(result["checkpoint_stream"]["manifest_sha256"]), 64)
        self.assertGreater(result["checkpoint_stream"]["bf16_rounding_changed_elements"], 0)
        self.assertEqual(self.events, ["initialize", "checkpoint_iterator", "finalize"])
        self.assertEqual(result["buffers"]["position"]["comparison_scope"],
                         "registered_buffer_threeway_not_poisoned")

    def test_non_audio_parameter_corruption_is_detected_and_reference_restored(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        with torch.no_grad():
            model.fused[3, 2] = 99
        result = self.execute(model, initial)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["parameters"]["fused"]["reason"], "sync_differs_from_disk")
        self.assertEqual(result["parameters"]["audio"]["status"], "PASS")
        self.assertEqual(result["parameters"]["fused"]["initial_disk"]["sha256"],
                         result["parameters"]["fused"]["disk_replay"]["sha256"])

    def test_omitted_parameter_cannot_retain_synchronized_value(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        model.omit_audio = True
        result = self.execute(model, initial)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["parameters"]["audio"]["disk_replay"]["finite"])
        self.assertFalse(result["parameters"]["alias"]["disk_replay"]["finite"])

    def test_replay_corruption_is_not_misreported_as_sync_problem(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)

        def broken_finalize(current, config):
            self.finalize(current, config)
            current.fused[0, 0] = -100

        result = self.execute(model, initial, broken_finalize)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["parameters"]["fused"]["reason"], "reference_reload_changes_disk_baseline")

    def test_retained_fp32_parameter_detects_lossy_bf16_roundtrip(self):
        model = FusedTPModel(self.weights, dtype=torch.float32)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        with torch.no_grad():
            model.audio.copy_(model.audio.to(torch.bfloat16).to(torch.float32))
        result = self.execute(model, initial)
        self.assertEqual(result["parameters"]["audio"]["status"], "FAIL")
        self.assertEqual(result["parameters"]["audio"]["after_sync"]["dtype"], "torch.float32")
        self.assertEqual(result["parameters"]["audio"]["disk_replay"]["dtype"], "torch.float32")
        self.assertEqual(result["checkpoint_stream"]["rounding_tensors"]["audio"]["changed_elements"], 3)

    def test_loader_exception_writes_incomplete_and_forbids_generation(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        model.load_error = True
        with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
            self.execute(model, initial)
        result = json.loads(Path(model._huanyu_full_weight_audit_report_path).read_text())
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(model._huanyu_full_weight_audit_generation_forbidden)
        self.assertIn("synthetic loader failure", result["errors"][0])
        self.assertIsNotNone(result["parameters"]["audio"]["after_sync"])
        self.assertIsNone(result["parameters"]["audio"]["disk_replay"])

    def test_initial_meta_and_unsupported_configs_rejected(self):
        model = FusedTPModel(self.weights)
        self.worker.vllm_config.load_config.load_format = "dummy"
        with self.assertRaisesRegex(ValueError, "load_format"):
            audit.snapshot_initial_disk(model, self.model_config, self.worker)
        self.worker.vllm_config.load_config.load_format = "safetensors"
        self.model_config.quantization = "fp8"
        with self.assertRaisesRegex(ValueError, "quantized"):
            audit.snapshot_initial_disk(model, self.model_config, self.worker)
        self.model_config.quantization = None
        model.audio = nn.Parameter(torch.empty(3, device="meta"))
        with self.assertRaisesRegex(ValueError, "meta or nonfinite"):
            audit.snapshot_initial_disk(model, self.model_config, self.worker)

    def test_missing_baseline_and_later_sync_rejected(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.execute(model, None)
        model._huanyu_weight_sync_update_id = 1
        with self.assertRaisesRegex(RuntimeError, "first sync"):
            self.execute(model, initial)

    def test_bounded_chunks_preserve_noncontiguous_logical_bytes(self):
        tensor = torch.arange(2 * 5 * 17, dtype=torch.float32).reshape(2, 5, 17).transpose(0, 2)
        pieces = list(audit._chunks(tensor, 16))
        self.assertTrue(all(part.numel() * part.element_size() <= 16 for part in pieces))
        actual = b"".join(part.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() for part in pieces)
        expected = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        self.assertEqual(actual, expected)
        self.assertEqual(audit._fingerprint(tensor)["sha256"], hashlib.sha256(expected).hexdigest())

    def test_nonfinite_sync_and_missing_inventory_are_incomplete(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        with torch.no_grad():
            model.audio[0] = float("nan")
        result = self.execute(model, initial)
        self.assertEqual(result["parameters"]["audio"]["status"], "INCOMPLETE")
        self.assertEqual(audit._compare({"missing": initial["parameters"]["audio"]}, {}, {})["missing"]["status"],
                         "INCOMPLETE")

    def test_registered_buffer_changes_are_reported_separately(self):
        model = FusedTPModel(self.weights)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        model.position[0] = 99
        result = self.execute(model, initial)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(all(row["status"] == "PASS" for row in result["parameters"].values()))
        self.assertEqual(result["buffers"]["position"]["status"], "FAIL")
        self.assertFalse(result["summary"]["buffers_independently_loaded_from_checkpoint"])

    def test_identical_infinite_attention_buffer_is_not_a_parameter_failure(self):
        model = FusedTPModel(self.weights)
        model.register_buffer("attention_mask", torch.tensor([float("-inf"), 0.0]), persistent=False)
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        result = self.execute(model, initial)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["buffers"]["attention_mask"]["status"], "PASS")
        self.assertFalse(result["buffers"]["attention_mask"]["after_sync"]["finite"])

    def test_audio_replay_uses_deferred_loader_after_real_meta_restore(self):
        self.weights = {"model.audio_encoder.weight": torch.tensor([1.0, 2.0, 3.0]),
                        "model.audio_encoder.bias": torch.tensor([0.1, 0.2, 0.3]),
                        "text": torch.tensor([4.0, 5.0])}
        self.write_checkpoint()
        owner = self

        class AudioModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text = nn.Parameter(owner.weights["text"].clone())
                self.thinker = nn.Module()
                self.thinker.apm = nn.LayerNorm(3)
                self.thinker.apm.load_state_dict({
                    "weight": owner.weights["model.audio_encoder.weight"],
                    "bias": owner.weights["model.audio_encoder.bias"]})
                self.direct_audio_loads = 0

            def load_weights(self, weights):
                for name, tensor in weights:
                    if name.startswith("model.audio_encoder."):
                        self.direct_audio_loads += 1
                        # This native path is ineffective after restore-to-meta.
                        self.thinker.apm.load_state_dict({name.rsplit(".", 1)[1]: tensor}, strict=False)
                    else:
                        self.text.weight_loader(self.text, tensor)

        model = AudioModel()
        initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
        original, pending = {}, []

        def initialize(current):
            for layer in current.modules():
                for name, param in list(layer.named_parameters(recurse=False)):
                    owner.assertTrue(bool(torch.isnan(param).all()))
                    original[(id(layer), name)] = param
                    meta = nn.Parameter(torch.empty_like(param, device="meta"))

                    def load(param, value, *, layer=layer, name=name):
                        pending.append((layer, name, value.clone()))

                    meta.weight_loader = load
                    layer.register_parameter(name, meta)

        def finalize(current, config):
            for layer, name, value in pending:
                param = original[(id(layer), name)]
                param.copy_(value.to(param.dtype))
                layer.register_parameter(name, param)

        path = audit.audit_first_sync(model, self.model_config, self.worker,
                                     initialize, finalize, initial=initial)
        result = json.loads(Path(path).read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(model.direct_audio_loads, 0)
        self.assertEqual(len(pending), 3)
        self.assertTrue(all(not p.is_meta for p in model.parameters()))

    def test_process_group_endpoint_unwraps_prefix_store_and_stops_cycles(self):
        from torch.distributed import distributed_c10d
        group = SimpleNamespace(cpu_group=object())
        store = SimpleNamespace(underlying_store=SimpleNamespace(host="actual-engine", port=1234))
        with patch.object(distributed_c10d, "_get_process_group_store", return_value=store, create=True):
            self.assertEqual(audit._process_group_endpoint(group), "actual-engine:1234")
        store.underlying_store = store
        with patch.object(distributed_c10d, "_get_process_group_store", return_value=store, create=True):
            self.assertIsNone(audit._process_group_endpoint(group))

    def test_engine_identity_prefers_worker_rendezvous_over_actor_environment(self):
        module = ModuleType("vllm.distributed")
        module.get_tp_group = lambda: SimpleNamespace(ranks=[0, 1, 2, 3])
        module.get_tensor_model_parallel_rank = lambda: 1
        module.get_tensor_model_parallel_world_size = lambda: 4
        self.worker.distributed_init_method = "tcp://actual-engine:1234"
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch.dict(os.environ, {"MASTER_ADDR": "actor-master", "MASTER_PORT": "9999"}):
            process = audit._identity(self.worker)
        self.assertEqual(process["engine_group"], "tcp://actual-engine:1234/tp=0,1,2,3")
        self.assertEqual(process["engine_group_source"], "worker_distributed_init_method")

    def test_actual_worker_report_passes_actual_aggregator(self):
        spec = importlib.util.spec_from_file_location(
            "huanyu_actual_full_weight_compare_cpu", ROOT / "examples" / "compare_huanyu_full_weights.py")
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        model = FusedTPModel(self.weights)
        process = {"host": "cpu-test", "pid": 123, "rank": 0, "tp_rank": 0,
                   "tp_world_size": 1, "engine_group": "cpu-test:1234/tp=0"}
        with patch.object(audit, "_identity", return_value=process):
            initial = audit.snapshot_initial_disk(model, self.model_config, self.worker)
            self.execute(model, initial)
        report = compare.compare(self.root / "reports", expected_workers=1, tp_size=1)
        self.assertEqual(report["status"], "PASS", report)


if __name__ == "__main__":
    unittest.main()
