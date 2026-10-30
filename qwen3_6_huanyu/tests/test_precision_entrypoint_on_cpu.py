"""Import the real Huanyu engine entry point with lightweight registrations."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "verl_omni_ext.models.qwen3_6_huanyu"
ENTRY = PACKAGE + ".engine_hooks"
INSTALLERS = (
    ("processor_compat", "install_huanyu_processor_compat"),
    ("rollout_precision", "install_huanyu_rollout_precision"),
    ("cache_precision", "install_huanyu_cache_precision"),
    ("weight_reload", "install_huanyu_weight_reload"),
)


def package(name, path=()):
    module = ModuleType(name)
    module.__path__ = list(path)
    return module


@contextmanager
def fake_registrations(fail_at=None):
    events, installed = [], set()
    root = package("verl_omni_ext")
    models = package("verl_omni_ext.models")
    whisper = package("verl_omni_ext.models.qwen3_6_whisper")
    huanyu = package(PACKAGE, [str(ROOT)])
    root.models = models
    models.qwen3_6_whisper = whisper
    models.qwen3_6_huanyu = huanyu
    modules = {module.__name__: module for module in (root, models, whisper, huanyu)}

    def register(name):
        if name != "vllm_omni":
            raise AttributeError(name)
        events.append("whisper-registration")
        if fail_at == "whisper-registration":
            raise RuntimeError("registration failed")
        registration = ModuleType(whisper.__name__ + ".vllm_omni")
        whisper.vllm_omni = registration
        installed.add("whisper-registration")
        return registration

    whisper.__getattr__ = register

    def installer(label):
        def install():
            if "whisper-registration" not in installed:
                raise AssertionError("precision installer ran before native registration")
            events.append(label)
            if fail_at == label:
                raise RuntimeError(label + " failed")
            if label not in installed:
                events.append("installed:" + label)
                installed.add(label)
            return True

        return install

    for submodule, function in INSTALLERS:
        module = ModuleType(PACKAGE + "." + submodule)
        setattr(module, function, installer(submodule))
        modules[module.__name__] = module
    # import_module uses the actual entry file on disk, with only its external
    # registrations/installers stubbed. It must neither swallow their errors nor
    # substitute the older Whisper-only external entry point.
    previous = sys.modules.pop(ENTRY, None)
    try:
        with patch.dict(sys.modules, modules):
            yield events, installed
    finally:
        sys.modules.pop(ENTRY, None)
        if previous is not None:
            sys.modules[ENTRY] = previous


class PrecisionEntrypointTests(unittest.TestCase):
    def test_real_entry_registers_whisper_pipeline_before_all_hooks(self):
        with fake_registrations() as (events, installed):
            entry = importlib.import_module(ENTRY)
            self.assertEqual(Path(entry.__file__).resolve(), ROOT / "engine_hooks.py")
            self.assertEqual(events, [
                "whisper-registration", "processor_compat", "installed:processor_compat",
                "rollout_precision", "installed:rollout_precision",
                "cache_precision", "installed:cache_precision", "weight_reload", "installed:weight_reload",
            ])
            self.assertEqual(installed, {"whisper-registration", *(name for name, _ in INSTALLERS)})

    def test_reload_delegates_idempotency_to_installers_and_keeps_registration_once(self):
        with fake_registrations() as (events, _):
            entry = importlib.import_module(ENTRY)
            importlib.reload(entry)
            self.assertEqual(events.count("whisper-registration"), 1)
            for name, _ in INSTALLERS:
                self.assertEqual(events.count(name), 2)
                self.assertEqual(events.count("installed:" + name), 1)
            self.assertEqual(events[-len(INSTALLERS):], [name for name, _ in INSTALLERS])

    def test_installation_failures_propagate_and_do_not_run_later_hooks(self):
        names = [name for name, _ in INSTALLERS]
        for index, failed in enumerate(names):
            with self.subTest(installer=failed), fake_registrations(failed) as (events, installed):
                with self.assertRaisesRegex(RuntimeError, failed + " failed"):
                    importlib.import_module(ENTRY)
                self.assertNotIn(ENTRY, sys.modules)
                self.assertNotIn(failed, installed)
                for later in names[index + 1:]:
                    self.assertNotIn(later, events)

    def test_native_registration_failure_propagates_before_any_precision_hook(self):
        with fake_registrations("whisper-registration") as (events, installed):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                importlib.import_module(ENTRY)
            self.assertEqual(events, ["whisper-registration"])
            self.assertEqual(installed, set())
            self.assertNotIn(ENTRY, sys.modules)


if __name__ == "__main__":
    unittest.main()
