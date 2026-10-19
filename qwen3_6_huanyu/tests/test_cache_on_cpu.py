"""Cache integrity, identity and exact frontend keys without an inference engine."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import errno
import json
from pathlib import Path
import sys
import tempfile
import threading
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

    def test_manifest_and_embeddings_work_without_hardlink_support(self):
        root = self.root / "no-hardlinks"
        with patch.object(cache_module.os, "link", side_effect=PermissionError(errno.EPERM, "unsupported")) as link:
            cache = cache_module.HuanyuEmbeddingCache(root, self.contract, create=True)
            values = torch.arange(12, dtype=torch.float32).reshape(2, 6).bfloat16()
            cache.put("audio", self.key, values, 2, 6)
            reopened = cache_module.HuanyuEmbeddingCache(root, self.contract)
            torch.testing.assert_close(reopened.get("audio", self.key, 2, 6), values)
            link.assert_not_called()
        self.assertFalse(list(root.rglob(".huanyu-*")))

    def test_concurrent_publish_has_one_winner_and_complete_file(self):
        target = self.cache.path("audio", self.key)
        barrier = threading.Barrier(2)
        payloads = (b"first" * 4096, b"second" * 4096)

        def publish(index):
            def writer(handle):
                handle.write(payloads[index])
                barrier.wait(timeout=5)
            return cache_module._publish(target, writer)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(publish, i) for i in range(2)]
            results = [future.result(timeout=10) for future in futures]
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)
        self.assertEqual(target.read_bytes(), payloads[results.index(True)])
        self.assertFalse(list(target.parent.glob(".huanyu-*")))

    def _initialize_concurrently(self, contracts):
        root = self.root / "concurrent-init"
        barrier = threading.Barrier(2)
        publish = cache_module._publish

        def coordinated_publish(path, writer, validate_new=None):
            # Both constructors have observed a missing manifest before either
            # publishes it, reproducing the directory-adoption race.
            barrier.wait(timeout=5)
            return publish(path, writer, validate_new)

        def initialize(contract):
            try:
                return cache_module.HuanyuEmbeddingCache(root, contract, create=True)
            except Exception as error:
                return error

        with patch.object(cache_module, "_publish", side_effect=coordinated_publish):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(initialize, contract) for contract in contracts]
                results = [future.result(timeout=10) for future in futures]
        self.assertFalse(list(root.glob(".huanyu-*")))
        return root, results

    def test_concurrent_initializers_accept_the_same_contract(self):
        root, results = self._initialize_concurrently([self.contract, self.contract])
        for result in results:
            self.assertIsInstance(result, cache_module.HuanyuEmbeddingCache, str(result))
            self.assertEqual(result.root, root.resolve())
        self.assertEqual(json.loads((root / "manifest.json").read_text())["contract"], self.contract)

    def test_concurrent_initializers_reject_a_different_contract(self):
        root, results = self._initialize_concurrently([self.contract, {**self.contract, "revision": 2}])
        accepted = [result for result in results if isinstance(result, cache_module.HuanyuEmbeddingCache)]
        rejected = [result for result in results if isinstance(result, Exception)]
        self.assertEqual(len(accepted), 1, str(results))
        self.assertEqual(len(rejected), 1, str(results))
        self.assertIsInstance(rejected[0], ValueError)
        self.assertIn("contract differs", str(rejected[0]))
        self.assertEqual(json.loads((root / "manifest.json").read_text())["contract"], accepted[0].contract)

    def test_writer_failure_does_not_leave_a_partial_file_or_lock(self):
        target = self.root / "failed-entry.pt"

        def writer(handle):
            handle.write(b"partial")
            raise RuntimeError("encoder output serialization failed")

        with self.assertRaisesRegex(RuntimeError, "serialization failed"):
            cache_module._publish(target, writer)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".huanyu-*")))

    def test_rename_failure_cleans_its_temporary_file_and_owned_lock(self):
        target = self.root / "failed-entry.pt"
        with patch.object(cache_module.os, "rename", side_effect=PermissionError(errno.EPERM, "rename denied")):
            with self.assertRaisesRegex(PermissionError, "rename denied"):
                cache_module._publish(target, lambda handle: handle.write(b"complete"))
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".huanyu-*")))

    def test_abandoned_lock_times_out_without_deleting_another_owners_lock(self):
        target = self.root / "blocked-entry.pt"
        lock = target.with_name(f".huanyu-{target.name}.lock")
        lock.mkdir()
        with patch.object(cache_module, "_PUBLISH_LOCK_TIMEOUT", 0):
            with self.assertRaisesRegex(TimeoutError, "check for active precompute writers") as error:
                cache_module._publish(target, lambda handle: handle.write(b"complete"))
        self.assertIn(str(lock), str(error.exception))
        self.assertTrue(lock.is_dir())
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".huanyu-*")), [lock])

    def test_existing_destination_is_never_renamed_over(self):
        target = self.root / "already-published.pt"
        target.write_bytes(b"original complete entry")
        with patch.object(cache_module.os, "rename") as rename:
            self.assertFalse(cache_module._publish(target, lambda handle: handle.write(b"replacement")))
            rename.assert_not_called()
        self.assertEqual(target.read_bytes(), b"original complete entry")
        self.assertFalse(list(self.root.glob(".huanyu-*")))

    def test_dangling_symlink_destination_is_not_overwritten(self):
        target = self.root / "dangling-entry.pt"
        # Windows may not permit creating symlinks; model their defining state:
        # exists() is False, while lexists() is True.
        self.assertFalse(target.exists())
        with patch.object(cache_module.os.path, "lexists", return_value=True) as lexists, \
                patch.object(cache_module.os, "rename") as rename:
            self.assertFalse(cache_module._publish(target, lambda handle: handle.write(b"replacement")))
            lexists.assert_called_once_with(target)
            rename.assert_not_called()
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".huanyu-*")))

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
