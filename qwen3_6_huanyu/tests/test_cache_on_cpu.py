"""Cache integrity, identity and exact frontend keys without an inference engine."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "huanyu_cache_cpu_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.cache", ROOT / "cache.py")
cache_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache_module
spec.loader.exec_module(cache_module)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "new-cache"
        self.contract = {"checkpoint": "huanyu", "frontend": "40-bin", "revision": 1}
        self.cache = cache_module.HuanyuEmbeddingCache(self.root, self.contract, create=True)
        self.feature = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        self.key = cache_module.audio_key(self.feature, 8)

    def test_audio_key_ignores_padding_and_changes_with_real_features(self):
        padded = torch.nn.functional.pad(self.feature, (0, 4), value=999)
        self.assertEqual(self.key, cache_module.audio_key(padded, 8))
        self.assertNotEqual(self.key, cache_module.audio_key(padded, 9))
        changed = self.feature.clone()
        changed[0, 0] += 0.01
        self.assertNotEqual(self.key, cache_module.audio_key(changed, 8))
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            cache_module.audio_key(self.feature.bfloat16(), 8)

    def test_vision_keys_include_grid_geometry_and_validate_patch_count(self):
        pixels = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        self.assertNotEqual(cache_module.vision_key(pixels, [1, 2, 4]),
                            cache_module.vision_key(pixels, [1, 4, 2]))
        pieces = cache_module.split_vision_inputs(pixels, torch.tensor([[1, 2, 2], [1, 2, 2]]))
        self.assertEqual(len(pieces), 2)
        torch.testing.assert_close(pieces[1][0], pixels[4:])
        with self.assertRaisesRegex(ValueError, "patch counts"):
            cache_module.split_vision_inputs(pixels, torch.tensor([[1, 2, 2]]))

    def test_roundtrip_reopen_and_strict_missing(self):
        values = torch.randn(2, 6).bfloat16()
        self.cache.put("audio", self.key, values, 2, 6)
        reopened = cache_module.HuanyuEmbeddingCache(self.root, self.contract)
        torch.testing.assert_close(reopened.get("audio", self.key, 2, 6), values)
        with self.assertRaisesRegex(FileNotFoundError, "no online fallback"):
            reopened.get("audio", "0" * 64, 2, 6)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            reopened.get("audio", self.key, 3, 6)

    def test_contract_change_refuses_reuse(self):
        with self.assertRaisesRegex(ValueError, "contract differs"):
            cache_module.HuanyuEmbeddingCache(self.root, {**self.contract, "revision": 2}, create=True)
        with self.assertRaisesRegex(FileNotFoundError, "precompute first"):
            cache_module.HuanyuEmbeddingCache(self.root / "absent", self.contract)

    def test_existing_old_cache_is_not_adopted(self):
        old = self.root / "old"
        old.mkdir()
        (old / "legacy.pt").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "non-Huanyu cache"):
            cache_module.HuanyuEmbeddingCache(old, self.contract, create=True)
        self.assertEqual((old / "legacy.pt").read_bytes(), b"old")
        legacy = self.root / "legacy-tree"
        (legacy / "audio").mkdir(parents=True)
        (legacy / "audio" / "old.pt").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "non-Huanyu cache"):
            cache_module.HuanyuEmbeddingCache(legacy, self.contract, create=True)

    def test_existing_different_embedding_cannot_be_overwritten(self):
        values = torch.ones(2, 6)
        self.cache.put("audio", self.key, values, 2, 6)
        self.cache.put("audio", self.key, values, 2, 6)
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.cache.put("audio", self.key, values + 1, 2, 6)
        torch.testing.assert_close(self.cache.get("audio", self.key, 2, 6), values.bfloat16())
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_corruption_detected_before_use(self):
        path = self.cache.put("audio", self.key, torch.ones(2, 6), 2, 6)
        payload = torch.load(path, weights_only=True)
        payload["embedding"][0, 0] = 2
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.cache.get("audio", self.key, 2, 6)

    def test_nonfinite_and_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            self.cache.put("audio", self.key, torch.full((2, 6), float("nan")), 2, 6)
        with self.assertRaisesRegex(ValueError, "modality or digest"):
            self.cache.get("../escape", self.key, 2, 6)

    def test_contract_tracks_native_code_and_checkpoint_metadata(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
        save_file({"model.visual.weight": torch.ones(1, 1)}, str(model / "model.safetensors"))
        source = model / "native.py"
        source.write_text("# native encoder v1\n")
        runtime = ModuleType(f"{PACKAGE}.runtime_imports")
        runtime.load_huanyu_native = lambda: type("Native", (), {"__file__": str(source)})
        processor = type("Processor", (), {
            "audio_preprocessor": {"feature_size": 40},
            "image_processor": type("Image", (), {"to_dict": lambda self: {"patch_size": 16}})(),
        })()
        fake_module = type("HF", (), {"__file__": str(source)})
        with patch.dict(sys.modules, {runtime.__name__: runtime}), \
                patch.object(cache_module.importlib, "import_module", return_value=fake_module):
            first = cache_module.make_cache_contract(model, processor)
            self.assertEqual(first, cache_module.make_cache_contract(model, processor))
            source.write_text("# native encoder v2\n")
            self.assertNotEqual(first, cache_module.make_cache_contract(model, processor))


if __name__ == "__main__":
    unittest.main()
