"""IPC ownership, reload lifecycle and Huanyu dispatch without NPU dependencies."""

from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_weight_reload_cpu_tests", ROOT / "weight_reload.py")
reload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reload)


class Tensor:
    def __init__(self, value, *, is_meta=False):
        self.value = value
        self.is_meta = is_meta

    def clone(self):
        return Tensor(self.value, is_meta=self.is_meta)


class Layer:
    def __init__(self):
        self.weight = Tensor(1)
        self.bias = Tensor(2)

    def get_submodule(self, path):
        if path == "layer":
            return self
        raise ValueError(path)


class Model:
    def __init__(self):
        self.thinker = SimpleNamespace(apm=Layer(), audio_projection_layer=Layer())
        self.text = Tensor(3)
        self.loaded = []

    def load_weights(self, weights):
        self.loaded.extend(weights)

    def named_parameters(self):
        for name in ("apm", "audio_projection_layer"):
            layer = getattr(self.thinker, name)
            for leaf in ("weight", "bias"):
                yield f"thinker.{name}.{leaf}", getattr(layer, leaf)
        yield "llm.weight", self.text


def config(audio_type="huanyu_conformer_0_6b"):
    return SimpleNamespace(hf_config=SimpleNamespace(audio_config={"model_type": audio_type}))


@contextmanager
def fake_reload_api(model, *, leave_meta=False):
    api = ModuleType(reload._RELOAD_MODULE)
    events, pending = [], []

    def initialize(current):
        assert current is model
        events.append("initialize")
        for name in ("apm", "audio_projection_layer"):
            layer = getattr(current.thinker, name)
            for leaf in ("weight", "bias"):
                param = getattr(layer, leaf)
                param.weight_loader = lambda param, value: pending.append((param, value))

    def finalize(current, model_config):
        assert current is model
        events.append("finalize")
        for param, value in pending:
            param.value = value.value
            param.is_meta = leave_meta
        for name, value in current.loaded:
            current.text.value = value.value

    api.initialize_layerwise_reload = initialize
    api.finalize_layerwise_reload = finalize
    with patch.dict(sys.modules, {api.__name__: api}):
        yield events, pending


def extension_class():
    class Extension:
        _pending_lora_peft_config = None

        def __init__(self, model=None, model_config=None):
            self.model_runner = SimpleNamespace(model=model or Model(), model_config=model_config or config())
            self.device = "npu:0"
            self.delegated = []

        def _get_standard_weight_model_and_config(self):
            return self.model_runner.model, self.model_runner.model_config

        def _get_zmq_handle(self):
            return "ipc:///tmp/replica-0.sock"

        def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False, zmq_update_id=None):
            self.delegated.append((peft_config, base_sync_done, use_shm, zmq_update_id))
            return "original"

    return Extension


@contextmanager
def fake_worker_api(extension):
    module = ModuleType(reload._EXTENSION_MODULE)
    module.vLLMOmniColocateWorkerExtension = extension
    receiver_module = ModuleType(reload._RECEIVER_MODULE)
    receiver_calls = []

    class Receiver:
        def __init__(self, **kwargs):
            receiver_calls.append(kwargs)

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("llm.weight", Tensor(99))], True)

    receiver_module.BucketedWeightReceiver = Receiver
    zmq_module = ModuleType("verl_omni.workers.rollout.vllm_rollout.zmq_utils")
    zmq_module.make_update_zmq_handle = lambda handle, update_id: f"{handle}:{update_id}"
    with patch.dict(sys.modules, {
        module.__name__: module,
        receiver_module.__name__: receiver_module,
        zmq_module.__name__: zmq_module,
    }):
        yield receiver_calls


class WeightReloadTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {reload._ENV_FLAG: "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        reload._REPORTED_FALLBACKS.clear()

    def test_deferred_loaders_keep_owned_tensor_after_ipc_buffer_reuse(self):
        model = Model()
        shared = Tensor(10)

        def receive(on_bucket_received):
            on_bucket_received([("model.audio_encoder.layer.weight", shared)], False)
            shared.value = 20
            on_bucket_received([("model.audio_projection.layer.bias", shared), ("llm.weight", shared)], True)
            shared.value = -999  # receiver releases/reuses memory after callback

        with fake_reload_api(model) as (events, pending), redirect_stdout(io.StringIO()) as output:
            reload.receive_layerwise_weights(model, config(), SimpleNamespace(receive_weights=receive))
        self.assertEqual(events, ["initialize", "finalize"])
        self.assertEqual(model.thinker.apm.weight.value, 10)
        self.assertEqual(model.thinker.audio_projection_layer.bias.value, 20)
        self.assertEqual(model.text.value, 20)
        self.assertTrue(all(value is not shared for _, value in pending))
        self.assertEqual([name for name, _ in model.loaded], ["llm.weight"])
        self.assertIn("buckets=2 audio_tensors=2 meta_parameters=0", output.getvalue())

    def test_one_argument_callback_and_mapping_supported_frozen_audio_unchanged(self):
        model = Model()

        def receive(on_bucket_received):
            on_bucket_received({"llm.weight": Tensor(8)})

        with fake_reload_api(model) as (events, pending), redirect_stdout(io.StringIO()):
            reload.receive_layerwise_weights(model, config(), SimpleNamespace(receive_weights=receive))
        self.assertEqual(events, ["initialize", "finalize"])
        self.assertEqual(pending, [])
        self.assertEqual(model.thinker.apm.weight.value, 1)
        self.assertEqual(model.text.value, 8)

    def test_audio_parameters_are_resolved_after_layer_materializes(self):
        model = Model()
        layer = model.thinker.apm
        seen = []
        old_bias = layer.bias
        old_bias.weight_loader = lambda *args: self.fail("stale parameter used")

        def load_weight(param, value):
            self.assertIs(param, layer.weight)
            layer.weight = value
            layer.bias = Tensor(0)
            layer.bias.weight_loader = lambda param, value: seen.append((param, value.value))

        layer.weight.weight_loader = load_weight
        remaining, count = reload._load_audio_weights(model, [
            ("apm.weight", Tensor(10)), ("apm.bias", Tensor(20)),
        ])
        self.assertEqual((remaining, count), ([], 2))
        self.assertEqual(seen, [(layer.bias, 20)])
        self.assertIsNot(layer.bias, old_bias)

    def test_all_native_and_checkpoint_audio_prefixes(self):
        model = Model()
        prefixes = ("model.audio_encoder.", "model.audio_projection.", "apm.", "audio_projection_layer.")
        with fake_reload_api(model) as (_, pending):
            api = sys.modules[reload._RELOAD_MODULE]
            api.initialize_layerwise_reload(model)
            remaining, count = reload._load_audio_weights(model, [(p + "layer.weight", Tensor(i)) for i, p in enumerate(prefixes)])
        self.assertEqual((remaining, count), ([], 4))
        self.assertEqual(len(pending), 4)

    def test_missing_audio_module_or_loader_fails_instead_of_meta_copy(self):
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "No layerwise weight loader"):
            reload._load_audio_weights(model, [("apm.weight", Tensor(5))])
        model.thinker.apm = None
        with self.assertRaisesRegex(ValueError, "apm is absent"):
            reload._load_audio_weights(model, [("apm.weight", Tensor(5))])

    def test_finalize_left_meta_is_rejected_without_success_log(self):
        model = Model()
        receiver = SimpleNamespace(receive_weights=lambda on_bucket_received: on_bucket_received([("apm.weight", Tensor(9))]))
        with fake_reload_api(model, leave_meta=True), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "left meta parameters.*apm.weight"):
                reload.receive_layerwise_weights(model, config(), receiver)
        self.assertNotIn("completed", output.getvalue())

    def test_receive_failure_does_not_finalize_partial_update(self):
        model = Model()

        def failed(on_bucket_received):
            on_bucket_received([("apm.weight", Tensor(9))])
            raise RuntimeError("IPC disconnected")

        with fake_reload_api(model) as (events, _), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "IPC disconnected"):
                reload.receive_layerwise_weights(model, config(), SimpleNamespace(receive_weights=failed))
        self.assertEqual(events, ["initialize"])
        self.assertNotIn("completed", output.getvalue())

    def test_missing_reload_api_fails_before_receiving(self):
        incomplete = ModuleType(reload._RELOAD_MODULE)
        receiver = SimpleNamespace(receive_weights=lambda **kw: self.fail("receiver must not run"))
        with patch.dict(sys.modules, {incomplete.__name__: incomplete}):
            with self.assertRaisesRegex(RuntimeError, "requires vLLM initialize_layerwise_reload"):
                reload.receive_layerwise_weights(Model(), config(), receiver)

    def test_huanyu_full_update_preserves_shared_memory_and_update_id(self):
        extension = extension_class()
        worker = extension()
        with fake_worker_api(extension) as receivers, fake_reload_api(worker.model_runner.model) as (events, _):
            with redirect_stdout(io.StringIO()):
                reload.install_huanyu_weight_reload()
                worker.update_weights_from_ipc(None, False, True, "step-1-seq-2")
        self.assertEqual(worker.delegated, [])
        self.assertEqual(events, ["initialize", "finalize"])
        self.assertEqual(receivers, [{
            "zmq_handle": "ipc:///tmp/replica-0.sock:step-1-seq-2", "device": "npu:0", "use_shm": True,
        }])

    def test_other_models_and_explicit_lora_delegate_original(self):
        for audio_type, peft in (("whisper", None), ("huanyu_conformer_0_6b", {"r": 8}), ("huanyu_conformer_0_6b", {})):
            with self.subTest(audio_type=audio_type, peft=peft):
                extension = extension_class()
                worker = extension(model_config=config(audio_type))
                with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
                    reload.install_huanyu_weight_reload()
                    result = worker.update_weights_from_ipc(peft_config=peft, base_sync_done=True, use_shm=True)
                self.assertEqual(result, "original")
                self.assertEqual(worker.delegated, [(peft, True, True, None)])
                self.assertEqual(receivers, [])

    def test_pending_lora_is_owned_by_original_and_not_consumed_by_wrapper(self):
        for peft in ({"r": 8}, {}):
            extension = extension_class()
            worker = extension()
            worker._pending_lora_peft_config = peft
            with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
                reload.install_huanyu_weight_reload()
                result = worker.update_weights_from_ipc(zmq_update_id="later")
            self.assertEqual(result, "original")
            self.assertIs(worker._pending_lora_peft_config, peft)
            self.assertEqual(worker.delegated, [(None, False, False, "later")])
            self.assertEqual(receivers, [])

    def test_older_worker_without_helper_uses_runner_get_model(self):
        extension = extension_class()
        del extension._get_standard_weight_model_and_config
        worker = extension()
        model = worker.model_runner.model
        worker.model_runner = SimpleNamespace(get_model=lambda: model, model_config=config())
        with fake_worker_api(extension) as receivers, fake_reload_api(model), redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            worker.update_weights_from_ipc()
        self.assertEqual(len(receivers), 1)
        self.assertEqual(worker.delegated, [])

    def test_non_ar_worker_delegates_without_receiver(self):
        extension = extension_class()
        del extension._get_standard_weight_model_and_config
        worker = extension()
        worker.model_runner = None
        with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            self.assertEqual(worker.update_weights_from_ipc(), "original")
        self.assertEqual(receivers, [])

    def test_new_signature_fails_instead_of_silently_using_legacy_loading(self):
        extension = extension_class()

        def future_update(worker, peft_config=None, new_format="v2"):
            worker.delegated.append(new_format)
            return new_format

        extension.update_weights_from_ipc = future_update
        worker = extension()
        with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()) as output:
            reload.install_huanyu_weight_reload()
            with self.assertRaisesRegex(RuntimeError, "unsupported arguments=.*new_format"):
                worker.update_weights_from_ipc(new_format="v3")
            with self.assertRaisesRegex(RuntimeError, "unsupported arguments=.*new_format"):
                worker.update_weights_from_ipc()
        self.assertEqual(receivers, [])
        self.assertEqual(worker.delegated, [])
        self.assertNotIn("completed", output.getvalue())

    def test_nonempty_variadic_kwargs_fail_without_receiving(self):
        extension = extension_class()

        def future_update(worker, peft_config=None, **kwargs):
            worker.delegated.append(kwargs)
            return kwargs

        extension.update_weights_from_ipc = future_update
        worker = extension()
        with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            with self.assertRaisesRegex(RuntimeError, "unsupported arguments=.*kwargs"):
                worker.update_weights_from_ipc(protocol="future")
        self.assertEqual(receivers, [])
        self.assertEqual(worker.delegated, [])

    def test_disabled_before_install_imports_nothing_and_can_disable_after(self):
        with patch.dict("os.environ", {reload._ENV_FLAG: "0"}):
            with patch.object(reload.importlib, "import_module", side_effect=AssertionError("unexpected import")):
                self.assertFalse(reload.install_huanyu_weight_reload())
        extension = extension_class()
        worker = extension()
        with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            with patch.dict("os.environ", {reload._ENV_FLAG: "false"}):
                self.assertEqual(worker.update_weights_from_ipc(), "original")
        self.assertEqual(receivers, [])

    def test_install_is_idempotent_and_does_not_run_global_postload(self):
        extension = extension_class()
        worker = extension()
        with fake_worker_api(extension), fake_reload_api(worker.model_runner.model) as (events, _):
            with redirect_stdout(io.StringIO()) as output:
                reload.install_huanyu_weight_reload()
                installed = extension.update_weights_from_ipc
                reload.install_huanyu_weight_reload()
                self.assertIs(extension.update_weights_from_ipc, installed)
                worker.update_weights_from_ipc()
        self.assertEqual(output.getvalue().count("installed Huanyu-only"), 1)
        self.assertEqual(events, ["initialize", "finalize"])
        self.assertEqual(worker.delegated, [])  # original's global postload never runs

    def test_missing_device_and_invalid_flag_fail_clearly(self):
        extension = extension_class()
        worker = extension()
        worker.device = None
        with fake_worker_api(extension), redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            with self.assertRaisesRegex(RuntimeError, "initialized worker device"):
                worker.update_weights_from_ipc()
        with patch.dict("os.environ", {reload._ENV_FLAG: "maybe"}):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                reload.install_huanyu_weight_reload()

    def test_known_huanyu_worker_without_runner_cannot_silently_delegate(self):
        extension = extension_class()
        del extension._get_standard_weight_model_and_config
        worker = extension()
        worker.model_runner = None
        worker.model_config = config()
        with fake_worker_api(extension) as receivers, redirect_stdout(io.StringIO()):
            reload.install_huanyu_weight_reload()
            with self.assertRaisesRegex(RuntimeError, "cannot resolve model_runner"):
                worker.update_weights_from_ipc()
        self.assertEqual(receivers, [])
        self.assertEqual(worker.delegated, [])

    def test_circular_registry_import_defers_only_partially_initialized_base(self):
        module = ModuleType(reload._EXTENSION_MODULE)
        module.__spec__ = SimpleNamespace(_initializing=True)
        with patch.dict(sys.modules, {module.__name__: module}), redirect_stdout(io.StringIO()) as output:
            with patch.object(reload.importlib, "import_module", side_effect=AssertionError("recursive import")):
                self.assertFalse(reload.install_huanyu_weight_reload())
        self.assertIn("deferred: base worker module is initializing", output.getvalue())
        module.__spec__._initializing = False
        with patch.dict(sys.modules, {module.__name__: module}), redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(AttributeError):
                reload.install_huanyu_weight_reload()
        self.assertIn("installation failed: AttributeError", output.getvalue())

    def test_explicit_subclass_owns_inherited_wrapper_for_method_copy_factories(self):
        base = extension_class()
        with fake_worker_api(base), redirect_stdout(io.StringIO()) as output:
            reload.install_huanyu_weight_reload()

            class ExplicitExtension(base):
                pass

            reload.bind_huanyu_weight_reload(ExplicitExtension)
            method = ExplicitExtension.__dict__["update_weights_from_ipc"]
            self.assertIs(method, base.update_weights_from_ipc)
            # Old worker factories copy methods into a concrete class rather
            # than inheriting from the extension. The actual copied callable
            # must be the patched one, without waiting for native model init.
            copied = type("CopiedWorker", (base,), {"update_weights_from_ipc": method})
            worker = copied()
            with fake_reload_api(worker.model_runner.model):
                worker.update_weights_from_ipc()
                worker.update_weights_from_ipc()
        self.assertEqual(worker.delegated, [])
        self.assertIn("bound existing reload wrapper on explicit extension", output.getvalue())
        self.assertEqual(output.getvalue().count("entered full Huanyu IPC update"), 1)
        self.assertEqual(output.getvalue().count("weight update completed"), 2)
        self.assertIn("host=", output.getvalue())
        self.assertIn("pid=", output.getvalue())

    def test_frozen_audio_zero_updates_are_valid_and_reported(self):
        model = Model()
        receiver = SimpleNamespace(receive_weights=lambda on_bucket_received: on_bucket_received({"llm.weight": Tensor(9)}))
        with patch.dict("os.environ", {"HUANYU_FREEZE_AUDIO": "true"}), fake_reload_api(model), redirect_stdout(io.StringIO()) as output:
            reload.receive_layerwise_weights(model, config(), receiver)
        self.assertIn("audio_mode=frozen audio_encoder_tensors=0 audio_projection_tensors=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
