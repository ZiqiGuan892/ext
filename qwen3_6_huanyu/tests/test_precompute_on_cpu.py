"""Offline encoder/cache boundaries with small CPU tensors and fake frontends."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "huanyu_precompute_cpu_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.precompute", ROOT / "precompute.py")
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


class FakeProcessor:
    config = SimpleNamespace(text_config={"hidden_size": 6})
    spatial_merge_size = 2

    def _vision(self, images):
        return {"pixel_values": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]])} if images else {}

    def _audio(self, audios, sampling_rate):
        if not audios:
            return [], torch.tensor([], dtype=torch.long), []
        return [torch.ones(4, 8)], torch.tensor([8]), [[1]]


class FakeEncoders:
    def __init__(self):
        self.calls = []

    def audio(self, feature, length):
        self.calls.append("audio")
        return torch.ones(1, 6)

    def vision(self, pixels, grid):
        self.calls.append("vision")
        return torch.full((1, 6), 2.)


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = cli.HuanyuEmbeddingCache(self.root / "cache", {"model": "huanyu"}, create=True)
        fake_dataset = ModuleType(f"{PACKAGE}.dataset")
        fake_dataset.decode_huanyu_media = lambda row, sampling_rate: (row.get("images"), row.get("audios"))
        self.stub = patch.dict(sys.modules, {fake_dataset.__name__: fake_dataset})
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def test_populate_resume_and_verify_use_same_keys_without_reencoding(self):
        encoders = FakeEncoders()
        row = {"images": ["image"], "audios": ["audio"]}
        result = cli.process_row(row, FakeProcessor(), self.cache, encoders)
        self.assertEqual(result, {"audio": 1, "vision": 1, "created": 2, "reused": 0})
        result = cli.process_row(row, FakeProcessor(), self.cache, encoders)
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["reused"], 2)
        self.assertEqual(encoders.calls, ["audio", "vision"])
        result = cli.process_row(row, FakeProcessor(), self.cache, verify_only=True)
        self.assertEqual(result["reused"], 2)

    def test_verify_missing_never_encodes(self):
        encoders = FakeEncoders()
        with self.assertRaisesRegex(FileNotFoundError, "cache miss"):
            cli.process_row({"audios": ["audio"]}, FakeProcessor(), self.cache, encoders, verify_only=True)
        self.assertEqual(encoders.calls, [])

    def test_plain_text_row_has_no_encoder_work(self):
        result = cli.process_row({}, FakeProcessor(), self.cache)
        self.assertEqual(sum(result.values()), 0)

    def test_vision_uses_merger_output_and_rejects_deepstack(self):
        result = torch.ones(2, 6)
        output = SimpleNamespace(pooler_output=result, last_hidden_state=torch.ones(8, 4))
        torch.testing.assert_close(cli.vision_output_tokens(output, 2, 6), result)
        torch.testing.assert_close(cli.vision_output_tokens((result, []), 2, 6), result)
        with self.assertRaisesRegex(ValueError, "post-merger"):
            cli.vision_output_tokens(torch.ones(8, 4), 2, 6)
        with self.assertRaisesRegex(ValueError, "DeepStack"):
            cli.vision_output_tokens((result, [torch.ones(2, 6)]), 2, 6)

    def test_weight_loader_selects_visual_only_and_checks_index(self):
        save_file({"model.visual.proj.weight": torch.ones(6, 4),
                   "model.language_model.weight": torch.ones(3, 2)}, str(self.root / "model.safetensors"))
        result = cli.load_vision_weights(self.root)
        self.assertEqual(set(result), {"proj.weight"})
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "model.visual.proj.weight": "model.safetensors", "model.visual.missing": "model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing"):
            cli.load_vision_weights(self.root)

    def test_tiny_hf_vision_only_loads_once_and_matches_expected_tokens(self):
        class TinyVision(torch.nn.Module):
            constructions = 0

            def __init__(self, config):
                super().__init__()
                self.proj = torch.nn.Linear(4, 6, bias=False)
                TinyVision.constructions += 1

            def forward(self, pixels, grid_thw):
                result = self.proj(pixels).reshape(-1, 4, 6).mean(1)
                return SimpleNamespace(pooler_output=result)

        save_file({"model.visual.proj.weight": torch.ones(6, 4)}, str(self.root / "model.safetensors"))
        processor = FakeProcessor()
        processor.config = SimpleNamespace(text_config={"hidden_size": 6},
                                           vision_config=SimpleNamespace(deepstack_visual_indexes=[]))
        encoders = cli.FrozenEncoders(str(self.root), processor, "cpu")
        with patch.object(cli.importlib, "import_module", return_value=SimpleNamespace(Qwen3_5MoeVisionModel=TinyVision)):
            for _ in range(2):
                output = encoders.vision(torch.ones(4, 4), torch.tensor([1, 2, 2]))
                torch.testing.assert_close(output, torch.full((1, 6), 4., dtype=torch.bfloat16))
        self.assertEqual(TinyVision.constructions, 1)
        self.assertIsNone(encoders.audio_model)

    def test_audio_pads_each_real_chunk_and_never_loads_text_model(self):
        seen = []
        audio_runtime = ModuleType(f"{PACKAGE}.audio_runtime")

        def attach(module, config):
            self.assertEqual(config.path, str(self.root))
            return module

        def encode(module, features, lengths):
            seen.append((features.clone(), lengths.clone()))
            return [torch.ones(1, 6)]

        audio_runtime.attach_huanyu_audio = attach
        audio_runtime.encode_huanyu_audio = encode
        encoders = cli.FrozenEncoders(str(self.root), FakeProcessor(), "cpu")
        with patch.dict(sys.modules, {audio_runtime.__name__: audio_runtime}):
            encoders.audio(torch.ones(4, 16), 10)
        self.assertEqual(tuple(seen[0][0].shape), (1, 4, 12))
        self.assertEqual(seen[0][1].tolist(), [10])
        torch.testing.assert_close(seen[0][0][:, :, 10:], torch.zeros(1, 4, 2))
        self.assertIsNone(encoders.vision_model)


if __name__ == "__main__":
    unittest.main()
