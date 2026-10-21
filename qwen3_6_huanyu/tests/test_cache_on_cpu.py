"""Cache integrity, identity and exact frontend keys without an inference engine."""

import importlib.util
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import errno
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

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

    def _source_contracts(self):
        current = {
            **self.contract,
            "model_path": "/fake/huanyu/model",
            "metadata": {"config.json": {"size": 20, "sha256": "a" * 64}},
            "shards": {"model.safetensors": {
                "size": 100, "mtime_ns": 1234567, "header_sha256": "b" * 64,
            }},
            "audio_preprocessor": {"feature_size": 40, "sampling_rate": 16000},
            "image_processor": {"patch_size": 16},
            "audio_execution": "single_chunk_bf16_v1",
            "vision_execution": "single_image_hf_bf16_eager_q128_v1",
        }
        old = copy.deepcopy(current)
        old["sources"] = {
            name: {"size": 10 + i, "sha256": str(i) * 64}
            for i, name in enumerate(("native_huanyu", "native_projector", "hf_qwen35",
                                      "processor.py", "audio_runtime.py", "vision_attention.py"))
        }
        return old, current

    def test_all_source_hash_and_size_changes_are_compatible(self):
        old, _ = self._source_contracts()
        for name in old["sources"]:
            for field, replacement in (("sha256", "f" * 64), ("size", 987654)):
                with self.subTest(source=name, field=field):
                    current = copy.deepcopy(old)
                    current["sources"][name][field] = replacement
                    payload = cache_module._manifest_payload(old)
                    snapshot = copy.deepcopy((payload, current))
                    self.assertFalse(cache_module._manifest_matches(payload, current))
                    self.assertTrue(cache_module._source_compatible(payload, current))
                    self.assertEqual((payload, current), snapshot)

    def test_sources_only_update_reads_original_entries_without_rewriting(self):
        old, current = self._source_contracts()
        root = self.root / "old-source-cache"
        writer = cache_module.HuanyuEmbeddingCache(root, old, create=True)
        values = torch.ones(2, 6).bfloat16()
        entry = writer.put("audio", self.key, values, 2, 6)
        original = {path: path.read_bytes() for path in (root / "manifest.json", entry)}
        with self.assertWarnsRegex(UserWarning, "source fingerprints ignored"):
            reader = cache_module.HuanyuEmbeddingCache(root, current)
        self.assertEqual(reader.contract_digest, writer.contract_digest)
        self.assertEqual(reader.contract, old)
        torch.testing.assert_close(reader.get("audio", self.key, 2, 6), values)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        # Compatibility never disables entry digest validation.
        payload = torch.load(entry, weights_only=True)
        payload["contract_digest"] = cache_module._manifest_payload(current)["contract_digest"]
        torch.save(payload, entry)
        with self.assertRaisesRegex(ValueError, "entry metadata"):
            reader.get("audio", self.key, 2, 6)

    def test_precompute_can_resume_old_source_cache_using_its_original_digest(self):
        old, current = self._source_contracts()
        root = self.root / "old-source-cache"
        original = cache_module.HuanyuEmbeddingCache(root, old, create=True)
        existing = original.put("audio", self.key, torch.ones(2, 6), 2, 6)
        before = {path: path.read_bytes() for path in (root / "manifest.json", existing)}
        (root / "vision").rmdir()
        with self.assertWarnsRegex(UserWarning, "source fingerprints ignored"):
            resumed = cache_module.HuanyuEmbeddingCache(root, current, create=True)
        self.assertEqual(resumed.contract_digest, original.contract_digest)
        self.assertTrue((root / "vision").is_dir())
        new_key = cache_module.audio_key(self.feature + 1, 8)
        values = torch.full((3, 6), 2.0).bfloat16()
        new_entry = resumed.put("audio", new_key, values, 3, 6)
        self.assertEqual(torch.load(new_entry, weights_only=True)["contract_digest"], original.contract_digest)
        torch.testing.assert_close(original.get("audio", new_key, 3, 6), values)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_manifest_numeric_types_must_match_the_digest_and_contract(self):
        manifest = self.root / "manifest.json"
        original = manifest.read_text()
        for value in (True, 1.0):
            for update_digest in (False, True):
                with self.subTest(value=value, update_digest=update_digest):
                    payload = json.loads(original)
                    payload["contract"]["revision"] = value
                    if update_digest:
                        payload = cache_module._manifest_payload(payload["contract"])
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertFalse(cache_module._manifest_matches(payload, self.contract))
                    with self.assertRaisesRegex(ValueError, "contract differs"):
                        cache_module.HuanyuEmbeddingCache(self.root, self.contract)
                    self.assertIn("status=mismatch", cache_module.cache_contract_report(
                        payload, self.contract, manifest,
                    ))

    def test_tampered_manifest_digest_cannot_use_compatibility(self):
        old, current = self._source_contracts()
        root = self.root / "old-processor-cache"
        cache_module.HuanyuEmbeddingCache(root, old, create=True)
        manifest = root / "manifest.json"
        payload = json.loads(manifest.read_text())
        payload["contract_digest"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        before = manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "manifest_integrity=INVALID"):
            cache_module.HuanyuEmbeddingCache(root, current)
        self.assertEqual(manifest.read_bytes(), before)

    def test_non_source_contract_changes_still_report_and_reject(self):
        old, current = self._source_contracts()
        root = self.root / "changed-config"
        cache_module.HuanyuEmbeddingCache(root, old, create=True)
        mutations = (
            (("model_path",), "/other/model"),
            (("metadata", "config.json", "sha256"), "e" * 64),
            (("metadata", "config.json", "size"), 21),
            (("shards", "model.safetensors", "size"), 101),
            (("shards", "model.safetensors", "mtime_ns"), 1234568),
            (("shards", "model.safetensors", "header_sha256"), "e" * 64),
            (("audio_preprocessor", "feature_size"), 128),
            (("image_processor", "patch_size"), 32),
            (("audio_execution",), "single_chunk_bf16_v2"),
            (("vision_execution",), "single_image_hf_bf16_eager_q64_v1"),
            (("future_contract_field",), "must-not-be-ignored"),
        )
        before = (root / "manifest.json").read_bytes()
        for keys, value in mutations:
            with self.subTest(field=keys):
                changed = copy.deepcopy(current)
                cursor = changed
                for key in keys[:-1]:
                    cursor = cursor[key]
                cursor[keys[-1]] = value
                self.assertFalse(cache_module._source_compatible(cache_module._manifest_payload(old), changed))
                for create in (False, True):
                    with self.assertRaisesRegex(ValueError, "contract differs") as error:
                        cache_module.HuanyuEmbeddingCache(root, changed, create=create)
                    self.assertIn(keys[-1], str(error.exception))
        self.assertEqual((root / "manifest.json").read_bytes(), before)

    def test_sources_compatibility_keeps_numeric_types_strict(self):
        old, current = self._source_contracts()
        for value in (True, 1.0):
            with self.subTest(value=value):
                changed = {**current, "revision": value}
                self.assertFalse(cache_module._source_compatible(cache_module._manifest_payload(old), changed))

    def test_sources_compatibility_reports_accepted_existing_cache(self):
        old, current = self._source_contracts()
        report = cache_module.cache_contract_report(
            cache_module._manifest_payload(old), current, self.root / "manifest.json",
        )
        self.assertIn("status=compatible_without_source_hashes", report)
        self.assertIn("manifest_integrity=valid", report)

    def test_difference_diagnostics_are_bounded_and_show_missing_nested_fields(self):
        saved = {"values": {f"field{i:03d}": "a" * 500 for i in range(100)}}
        current = {"values": {f"field{i:03d}": "b" * 500 for i in range(100)}}
        current["new_field"] = 1
        report = cache_module.cache_contract_report(
            cache_module._manifest_payload(saved), current, self.root / "manifest.json", max_differences=4,
        )
        self.assertIn("differences=102 shown=4", report)
        self.assertIn("manifest.contract.new_field", report)
        self.assertIn("<missing>", report)
        self.assertNotIn("field099", report)
        self.assertLess(len(report), 2400)
        self.assertEqual(saved["values"]["field000"], "a" * 500)

    def test_read_only_inspector_disables_cache_reader_and_never_builds_encoders(self):
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.inspect_cache", ROOT / "inspect_cache.py")
        inspector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspector)
        processor_module = ModuleType(f"{PACKAGE}.processor")
        processor_module.build_huanyu_processor = Mock(return_value=object())
        before = (self.root / "manifest.json").read_bytes()
        with patch.dict(sys.modules, {processor_module.__name__: processor_module}), \
                patch.object(cache_module, "make_cache_contract", return_value=self.contract), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(inspector.inspect_cache("/fake/model", self.root), 0)
        self.assertIn("status=exact", output.getvalue())
        kwargs = processor_module.build_huanyu_processor.call_args.kwargs
        self.assertFalse(kwargs["install_bridge"])
        self.assertFalse(kwargs["use_cache"])
        with patch.dict(sys.modules, {processor_module.__name__: processor_module}), \
                patch.object(cache_module, "make_cache_contract", return_value={**self.contract, "revision": 3}), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(inspector.inspect_cache("/fake/model", self.root), 2)
        self.assertIn("status=mismatch", output.getvalue())
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)

    def test_read_only_inspector_accepts_legacy_source_metadata(self):
        old, current = self._source_contracts()
        root = self.root / "legacy-inspection"
        cache_module.HuanyuEmbeddingCache(root, old, create=True)
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.inspect_cache", ROOT / "inspect_cache.py")
        inspector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspector)
        processor_module = ModuleType(f"{PACKAGE}.processor")
        processor_module.build_huanyu_processor = Mock(return_value=object())
        before = (root / "manifest.json").read_bytes()
        with patch.dict(sys.modules, {processor_module.__name__: processor_module}), \
                patch.object(cache_module, "make_cache_contract", return_value=current), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(inspector.inspect_cache("/fake/model", root), 0)
        self.assertIn("status=compatible_without_source_hashes", output.getvalue())
        self.assertEqual((root / "manifest.json").read_bytes(), before)

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

    def test_concurrent_source_compatible_initializers_use_one_producer_digest(self):
        old, current = self._source_contracts()
        with patch.object(cache_module.warnings, "warn"):
            root, results = self._initialize_concurrently([old, current])
        manifest = json.loads((root / "manifest.json").read_text())
        for result in results:
            self.assertIsInstance(result, cache_module.HuanyuEmbeddingCache, str(result))
            self.assertEqual(result.contract_digest, manifest["contract_digest"])
            self.assertEqual(result.contract, manifest["contract"])
        values = torch.ones(2, 6).bfloat16()
        results[0].put("audio", self.key, values, 2, 6)
        torch.testing.assert_close(results[1].get("audio", self.key, 2, 6), values)

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

    def test_contract_ignores_source_changes_without_importing_model_implementations(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
        save_file({"model.visual.weight": torch.ones(1, 1)}, str(model / "model.safetensors"))
        source = model / "native.py"
        source.write_text("# native encoder v1\n")
        runtime = ModuleType(f"{PACKAGE}.runtime_imports")
        runtime.load_huanyu_native = Mock(side_effect=AssertionError("must not load native encoder"))
        processor = type("Processor", (), {
            "audio_preprocessor": {"feature_size": 40},
            "image_processor": type("Image", (), {"to_dict": lambda self: {"patch_size": 16}})(),
        })()
        with patch.dict(sys.modules, {runtime.__name__: runtime}), \
                patch("importlib.import_module", side_effect=AssertionError("must not import model implementations")):
            first = cache_module.make_cache_contract(model, processor)
            self.assertNotIn("sources", first)
            self.assertEqual(first, cache_module.make_cache_contract(model, processor))
            source.write_text("# native encoder v2\n")
            self.assertEqual(first, cache_module.make_cache_contract(model, processor))
            runtime.load_huanyu_native.assert_not_called()
            (model / "config.json").write_text('{"model_type":"qwen3_5_moe","revision":2}')
            self.assertNotEqual(first, cache_module.make_cache_contract(model, processor))


if __name__ == "__main__":
    unittest.main()
