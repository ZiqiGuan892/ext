"""Exercise mandatory worker binding without relying on the registry gate."""

import ast
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_worker_extension_tests"
BASE_MODULE = "verl_omni.workers.rollout.vllm_rollout.utils"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def worker_modules(*, fail=None):
    events = []
    base = ModuleType(BASE_MODULE)

    class Base:
        def __new__(cls):
            events.append("base-new")
            return super().__new__(cls)

        def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False):
            events.append("original-update")
            return "original"

    base.vLLMOmniColocateWorkerExtension = Base
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    reload = load_file(PACKAGE + ".weight_reload", ROOT / "weight_reload.py")
    modules = {BASE_MODULE: base, PACKAGE: package, reload.__name__: reload}
    for module_name, installer in (
        ("ab_determinism", "install_ab_determinism_check"),
        ("repeat_determinism", "install_repeat_determinism_check"),
        ("processor_compat", "install_huanyu_processor_compat"),
        ("rollout_precision", "install_huanyu_rollout_precision"),
        ("cache_precision", "install_huanyu_cache_precision"),
    ):
        module = ModuleType(PACKAGE + "." + module_name)

        def install(name=module_name):
            events.append(name)
            if fail == name:
                raise RuntimeError(name + " unavailable")
            return True

        setattr(module, installer, install)
        modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        yield Base, events, reload


class WorkerExtensionTests(unittest.TestCase):
    def test_ab_check_is_mandatory_without_embedding_or_logits_probe(self):
        for fail in (None, "ab_determinism"):
            with self.subTest(fail=fail), patch.dict(os.environ, {
                    "HUANYU_TEXT_AB_DETERMINISM": "1", "HUANYU_SERVE_REPEAT_DETERMINISM": "",
                    "HUANYU_EMBEDDING_AB_DIR": ""}), worker_modules(fail=fail) as (_, events, _), \
                    redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "ab_determinism unavailable"):
                        load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                else:
                    load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                self.assertEqual(events[0], "ab_determinism")

    def test_repeat_check_installs_at_worker_factory_without_tensor_probes(self):
        for fail in (None, "repeat_determinism"):
            with self.subTest(fail=fail), patch.dict(os.environ, {
                    "HUANYU_SERVE_REPEAT_DETERMINISM": "1", "HUANYU_EMBEDDING_AB_DIR": ""}), \
                    worker_modules(fail=fail) as (_, events, _), redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "repeat_determinism unavailable"):
                        load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                else:
                    module = load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                    self.assertNotIn("init_device", module.HuanyuColocateWorkerExtension.__dict__)
                    module.HuanyuColocateWorkerExtension()
                self.assertEqual(events[0], "repeat_determinism")

    def test_factory_import_installs_all_hooks_without_external_registry(self):
        with worker_modules() as (base, events, reload), redirect_stdout(io.StringIO()) as output:
            module = load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
            cls = module.HuanyuColocateWorkerExtension
            self.assertIn("update_weights_from_ipc", cls.__dict__)
            self.assertTrue(getattr(cls.update_weights_from_ipc, reload._PATCH_MARKER))
            self.assertIs(cls.__new__, base.__new__)
            cls()
        self.assertEqual(events, ["processor_compat", "rollout_precision", "cache_precision", "base-new"])
        self.assertIn("[huanyu-worker] explicit precision extension selected", output.getvalue())
        self.assertIn("ipc_signature=", output.getvalue())

    def test_false_reload_flag_delegates_but_worker_still_installs_numerical_hooks(self):
        with patch.dict(os.environ, {"MINICPMO_HUANYU_LAYERWISE_RELOAD": "0"}), worker_modules() as (_, events, _):
            with redirect_stdout(io.StringIO()) as output:
                module = load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                worker = module.HuanyuColocateWorkerExtension()
                self.assertEqual(worker.update_weights_from_ipc(), "original")
        self.assertEqual(events, ["processor_compat", "rollout_precision", "cache_precision", "base-new", "original-update"])
        self.assertIn("enabled=0", output.getvalue())
        self.assertIn("delegated to original IPC handler", output.getvalue())

    def test_factory_import_propagates_installer_failure(self):
        for failed in ("processor_compat", "rollout_precision", "cache_precision"):
            with self.subTest(failed=failed), worker_modules(fail=failed), redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(RuntimeError, failed + " unavailable"):
                    load_file(PACKAGE + ".worker_extension", ROOT / "worker_extension.py")
                self.assertNotIn("explicit precision extension selected", output.getvalue())

    def test_actual_server_deploy_writer_receives_huanyu_stage_extension(self):
        registry = {}
        model_base = ModuleType("verl_omni.pipelines.model_base")

        def register(name):
            def decorator(cls):
                registry[name] = cls
                return cls
            return decorator

        model_base.OmniRolloutPipelineBase = SimpleNamespace(register=register)
        whisper = ModuleType("verl_omni_ext.models.qwen3_6_whisper.rollout_adapter")
        parent_extras = {"model_stage": "llm", "additional_config": {"weight_nz_mode": 0}}

        class Parent:
            @classmethod
            def get_stage_engine_extras(cls, stage_id, pipeline_mode="thinker_only"):
                return parent_extras if stage_id == 0 else {}

            @classmethod
            def ensure_pipeline_registered(cls, mode):
                pass

            @classmethod
            def build_stage_configs(cls, pipeline_mode):
                return [SimpleNamespace(stage_id=0)]

            @classmethod
            def get_pipeline_id(cls, mode):
                return "qwen3_6_whisper_thinker_only"

        whisper.Qwen36WhisperRolloutAdapter = Parent
        with patch.dict(sys.modules, {model_base.__name__: model_base, whisper.__name__: whisper}):
            adapter = load_file(PACKAGE + ".rollout_adapter", ROOT / "rollout_adapter.py").Qwen36HuanyuRolloutAdapter
        expected = "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
        self.assertEqual(adapter.get_stage_engine_extras(0), {**parent_extras, "worker_extension_cls": expected})
        self.assertNotIn("worker_extension_cls", parent_extras)
        self.assertEqual(adapter.get_stage_engine_extras(1), {})
        self.assertIs(registry["qwen3_6_huanyu"], adapter)
        with patch.dict(os.environ, {"HUANYU_TEXT_AB_BATCH_INVARIANT": "1"}):
            extra = adapter.get_stage_engine_extras(0)
            self.assertEqual(extra["additional_config"]["rl_config"],
                             {"enabled": True, "enable_batch_invariant": True})
            self.assertEqual(extra["additional_config"]["weight_nz_mode"], 0)
            self.assertEqual(adapter.get_stage_engine_extras(1), {})
        self.assertEqual(parent_extras["additional_config"], {"weight_nz_mode": 0})
        with patch.dict(os.environ, {"HUANYU_TEXT_AB_BATCH_INVARIANT": "0"}):
            self.assertEqual(adapter.get_stage_engine_extras(0)["additional_config"]["rl_config"],
                             {"enabled": True, "enable_batch_invariant": False})

        # Execute the real local server's deploy writer with only its external
        # IO dependencies stubbed. This checks the rendered per-stage contract,
        # not merely the adapter return value. Native config merge needs NPU CI.
        server_path = ROOT.parents[2] / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
        if not server_path.is_file():
            self.skipTest("The separate verl-omni checkout is unavailable in this Huanyu-only export")
        tree = ast.parse(server_path.read_text(encoding="utf-8"))
        server_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniHttpServer")
        method = next(n for n in server_class.body if isinstance(n, ast.FunctionDef) and n.name == "_write_deploy_config")
        compiled = ast.Module(body=[method], type_ignores=[])
        scope = {"os": os, "tempfile": tempfile, "yaml": SimpleNamespace(dump=json.dumps),
                 "get_visible_devices_keyword": lambda: "TEST_DEVICES", "logger": Mock()}
        exec(compile(compiled, str(server_path), "exec"), scope)
        server = SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=4))
        kwargs = {}
        with patch.dict(os.environ, {"TEST_DEVICES": "4,5,6,7", "HUANYU_TEXT_AB_BATCH_INVARIANT": "1"}):
            scope["_write_deploy_config"](server, kwargs, "qwen3_6_huanyu", adapter, "thinker_only")
        try:
            saved = json.loads(Path(kwargs["deploy_config"]).read_text())
            self.assertEqual(saved["stages"][0]["engine_extras"]["worker_extension_cls"], expected)
            self.assertEqual(saved["stages"][0]["devices"], "0,1,2,3")
            self.assertEqual(saved["stages"][0]["tensor_parallel_size"], 4)
            self.assertTrue(saved["stages"][0]["engine_extras"]["additional_config"]["rl_config"]["enable_batch_invariant"])
        finally:
            server._temp_deploy_ctx.cleanup()


if __name__ == "__main__":
    unittest.main()
