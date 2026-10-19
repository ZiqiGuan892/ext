"""Exercise native import diagnostics without importing ext or the GPU stack."""

import importlib.util
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_runtime_imports_tested", ROOT / "runtime_imports.py")
runtime_imports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_imports)

NATIVE_MODULE = "vllm_omni.model_executor.models.minicpmo_5_0.huanyu_audio"
PARENT_MODULE = NATIVE_MODULE.rsplit(".", 1)[0]
SYMBOLS = ("HuanyuFbankFeatureExtractor", "HuanyuAudioEncoder", "huanyu_project_and_fold")
VALID_NATIVE = "\n".join(f"def {name}(*args, **kwargs):\n    return None\n" for name in SYMBOLS)


class RuntimeImportsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.checkout = Path(self.temporary.name) / "selected-checkout"
        self.native_path = self.checkout.joinpath(*NATIVE_MODULE.split(".")).with_suffix(".py")
        self.native_path.parent.mkdir(parents=True)

        modules_patch = patch.dict(sys.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        for name in list(sys.modules):
            if name == "vllm_omni" or name.startswith("vllm_omni."):
                del sys.modules[name]
        self.packages = {}
        parts = PARENT_MODULE.split(".")
        for index in range(1, len(parts) + 1):
            name = ".".join(parts[:index])
            path = self.checkout.joinpath(*parts[:index])
            package = ModuleType(name)
            package.__file__ = str(path / "__init__.py")
            package.__path__ = [str(path)]
            self.packages[name] = package
            sys.modules[name] = package

        environment_patch = patch.dict(os.environ, {"PYTHONPATH": "selected-runtime-path"})
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        os.environ.pop("HUANYU_VLLM_OMNI_ROOT", None)
        self.original_path = list(sys.path)

    def write_native(self, source=VALID_NATIVE):
        self.native_path.write_text(source, encoding="utf-8")
        importlib.invalidate_caches()

    def assert_selection_unchanged(self):
        self.assertEqual(sys.path, self.original_path)
        for name, package in self.packages.items():
            self.assertIs(sys.modules[name], package)

    def assert_diagnostics(self, message, exists):
        for detail in (
            f"host={socket.gethostname()}",
            f"pid={os.getpid()}",
            str(self.checkout / "vllm_omni" / "__init__.py"),
            "model_package_paths=",
            str(self.native_path),
            f"exists={exists}",
            "PYTHONPATH=selected-runtime-path",
        ):
            self.assertIn(detail, message)

    def test_valid_native_uses_selected_package_without_changing_import_paths(self):
        self.write_native()
        native = runtime_imports.load_huanyu_native(self.checkout)
        self.assertEqual(Path(native.__file__), self.native_path)
        self.assertTrue(all(callable(getattr(native, name)) for name in SYMBOLS))
        self.assertIs(runtime_imports.load_huanyu_native(self.checkout), native)
        self.assert_selection_unchanged()

    def test_environment_selects_root_and_explicit_argument_takes_precedence(self):
        self.write_native()
        os.environ["HUANYU_VLLM_OMNI_ROOT"] = str(self.checkout)
        native = runtime_imports.load_huanyu_native()
        os.environ["HUANYU_VLLM_OMNI_ROOT"] = str(self.checkout / "another-checkout")
        self.assertIs(runtime_imports.load_huanyu_native(self.checkout), native)
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            runtime_imports.load_huanyu_native()
        self.assertIs(sys.modules[NATIVE_MODULE], native)
        self.assert_selection_unchanged()

    def test_without_expected_root_valid_installed_native_is_accepted(self):
        self.write_native()
        native = runtime_imports.load_huanyu_native()
        self.assertEqual(Path(native.__file__), self.native_path)
        self.assert_selection_unchanged()

    def test_missing_native_preserves_cause_and_reports_worker_and_source(self):
        with self.assertRaises(RuntimeError) as raised:
            runtime_imports.load_huanyu_native(self.checkout)
        error = raised.exception
        self.assertIsInstance(error.__cause__, ModuleNotFoundError)
        self.assertEqual(error.__cause__.name, NATIVE_MODULE)
        self.assertIn("does not expose the Huanyu module", str(error))
        self.assertNotIn("a dependency is missing", str(error))
        self.assert_diagnostics(str(error), exists=False)
        self.assert_selection_unchanged()

    def test_missing_ancestor_is_reported_as_missing_huanyu_checkout(self):
        for missing in ("vllm_omni", "vllm_omni.model_executor", PARENT_MODULE):
            with self.subTest(missing=missing):
                cause = ModuleNotFoundError(f"No module named {missing!r}", name=missing)
                with patch.object(runtime_imports.importlib, "import_module", side_effect=cause):
                    with self.assertRaises(RuntimeError) as raised:
                        runtime_imports.load_huanyu_native(self.checkout)
                self.assertIs(raised.exception.__cause__, cause)
                self.assertIn("does not expose the Huanyu module", str(raised.exception))
                self.assertIn(f"missing_module={missing!r}", str(raised.exception))
        self.assert_selection_unchanged()

    def test_missing_internal_dependency_is_not_misdiagnosed_as_missing_native(self):
        dependency = "_huanyu_runtime_test_missing_dependency_31bf"
        self.write_native(f"import {dependency}\n" + VALID_NATIVE)
        with self.assertRaises(RuntimeError) as raised:
            runtime_imports.load_huanyu_native(self.checkout)
        error = raised.exception
        self.assertIsInstance(error.__cause__, ModuleNotFoundError)
        self.assertEqual(error.__cause__.name, dependency)
        self.assertIn("a dependency is missing", str(error))
        self.assertNotIn("does not expose the Huanyu module", str(error))
        self.assert_diagnostics(str(error), exists=True)
        self.assert_selection_unchanged()

    def test_existing_file_does_not_trigger_fallback_outside_active_package_paths(self):
        self.write_native()
        hidden_path = self.checkout / "different-model-package"
        self.packages[PARENT_MODULE].__path__ = [str(hidden_path)]
        with self.assertRaises(RuntimeError) as raised:
            runtime_imports.load_huanyu_native(self.checkout)
        self.assert_diagnostics(str(raised.exception), exists=True)
        self.assertIn(str(hidden_path).replace("\\", "\\\\"), str(raised.exception))
        self.assertNotIn(NATIVE_MODULE, sys.modules)
        self.assertEqual(self.packages[PARENT_MODULE].__path__, [str(hidden_path)])
        self.assert_selection_unchanged()

    def test_wrong_top_package_is_rejected_before_importing_native(self):
        self.write_native("raise AssertionError('wrong checkout must not be imported')\n")
        expected_root = self.checkout / "another-checkout"
        with self.assertRaisesRegex(RuntimeError, "source mismatch") as raised:
            runtime_imports.load_huanyu_native(expected_root)
        self.assertIn(str(self.checkout / "vllm_omni" / "__init__.py"), str(raised.exception))
        self.assertNotIn(NATIVE_MODULE, sys.modules)
        self.assert_selection_unchanged()

    def test_cached_native_from_sibling_package_is_rejected_without_replacement(self):
        self.write_native()
        native = runtime_imports.load_huanyu_native(self.checkout)
        native.__file__ = str(self.checkout / "vllm_omni_shadow" / "huanyu_audio.py")
        with self.assertRaisesRegex(RuntimeError, "source mismatch") as raised:
            runtime_imports.load_huanyu_native(self.checkout)
        self.assertIn(native.__file__, str(raised.exception))
        self.assertIs(sys.modules[NATIVE_MODULE], native)
        self.assert_selection_unchanged()

    def test_missing_or_noncallable_native_api_is_rejected(self):
        self.write_native()
        native = runtime_imports.load_huanyu_native(self.checkout)
        for name in SYMBOLS:
            original = getattr(native, name)
            for noncallable in (None, object()):
                with self.subTest(symbol=name, value=noncallable):
                    setattr(native, name, noncallable)
                    with self.assertRaisesRegex(RuntimeError, "API mismatch") as raised:
                        runtime_imports.load_huanyu_native(self.checkout)
                    self.assertIn(name, str(raised.exception))
            delattr(native, name)
            with self.subTest(symbol=name, absent=True):
                with self.assertRaisesRegex(RuntimeError, "API mismatch") as raised:
                    runtime_imports.load_huanyu_native(self.checkout)
                self.assertIn(name, str(raised.exception))
            setattr(native, name, original)
        self.assertIs(sys.modules[NATIVE_MODULE], native)
        self.assert_selection_unchanged()

    def test_missing_native_without_root_reports_default_checkout_candidate(self):
        with self.assertRaises(RuntimeError) as raised:
            runtime_imports.load_huanyu_native()
        self.assertIn("vllm-workspace", str(raised.exception))
        self.assertIn("vllm-omni", str(raised.exception))
        self.assertIn("expected_native_file=", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)
        self.assert_selection_unchanged()


if __name__ == "__main__":
    unittest.main()
