"""CPU regressions for Huanyu actor arithmetic and independent audio autograd."""

import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


precision = load_file("huanyu_precision_test", ROOT / "precision.py")
packed = load_file("qwen_packed_test", ROOT / "qwen_precision" / "packed.py")
linear = load_file("qwen_linear_test", ROOT / "qwen_precision" / "linear_precision.py")


class PrecisionSnapshotTests(unittest.TestCase):
    def test_precision_bundle_imports_without_sibling_whisper_helpers(self):
        name = "huanyu_local_precision_snapshot_test"
        package = ModuleType(name)
        package.__path__ = [str(ROOT / "qwen_precision")]
        with patch.dict(sys.modules, {name: package}):
            modules = {
                path.stem: importlib.import_module(f"{name}.{path.stem}")
                for path in (ROOT / "qwen_precision").glob("*.py")
                if path.stem != "__init__"
            }
            self.assertEqual(len(modules), 12)
            for module in modules.values():
                self.assertEqual(Path(module.__file__).resolve().parent, (ROOT / "qwen_precision").resolve())
            self.assertFalse(hasattr(modules["packed"], "install_packed_forward"))
            self.assertFalse(hasattr(modules["media_precision"], "audio_items"))
            self.assertFalse(hasattr(modules["media_precision"], "install_rollout"))
            self.assertFalse(hasattr(modules["cache_precision"], "is_policy"))
            self.assertFalse(hasattr(modules["cache_precision"], "install"))
        self.assertEqual(precision._SHARED, "verl_omni_ext.models.qwen3_6_huanyu.qwen_precision")


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                               initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=False, **kwargs):
    # A deliberately small recurrence-like matmul exposes BF16 autocast even
    # when its inputs explicitly use .float(), matching the bug being fixed.
    return query.float() @ key.float().T + value.float()


torch_chunk_gated_delta_rule.__module__ = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"


class Layer(torch.nn.Module):
    block_type = "linear_attention"

    def __init__(self):
        super().__init__()
        self.linear_attn = SimpleNamespace(chunk_gated_delta_rule=torch_chunk_gated_delta_rule)

    def forward(self, x):
        return x


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer()])

    def forward(self, x):
        return self.layers[0](x)


def model():
    result = torch.nn.Module()
    result.model = torch.nn.Module()
    result.model.language_model = Language()
    result.model.audio_encoder = torch.nn.Linear(3, 3)
    result.model.audio_projection = torch.nn.Linear(3, 3)
    result.model.visual = None
    result.lm_head = torch.nn.Linear(3, 3)
    return result


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.media = SimpleNamespace(install_actor=Mock())
        self.imports = []

        def shared(name):
            self.imports.append(name)
            return {"packed": packed, "media_precision": self.media, "linear_precision": linear}[name]

        self.shared = patch.object(precision, "_shared", side_effect=shared)
        self.shared.start()
        self.addCleanup(self.shared.stop)

    def test_gdn_default_disables_autocast_and_keeps_audio_gradients(self):
        torch.manual_seed(7)
        actor = model()
        audio_ids = [id(p) for p in actor.model.audio_encoder.parameters()]
        precision.install_huanyu_actor_precision(actor)
        kernel = actor.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule
        self.assertNotIn(inspect.Parameter.VAR_KEYWORD,
                         [item.kind for item in inspect.signature(kernel).parameters.values()])
        audio = torch.randn(2, 3, requires_grad=True)
        encoded = actor.model.audio_projection(actor.model.audio_encoder(audio))
        key = torch.randn(2, 3)
        value = torch.zeros(2, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = kernel(encoded, key, value, g=None, beta=None)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, encoded @ key.T)
        output.square().sum().backward()
        for component in (actor.model.audio_encoder, actor.model.audio_projection):
            for parameter in component.parameters():
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(audio_ids, [id(p) for p in actor.model.audio_encoder.parameters()])
        self.assertEqual(actor.model.language_model.layers[0].layer_type, "linear_attention")
        self.assertEqual(self.imports, ["packed", "media_precision"])

    def test_guard_is_idempotent_and_does_not_override_native_fla(self):
        actor = model()
        layer = actor.model.language_model.layers[0]
        precision.install_huanyu_actor_precision(actor)
        first = layer.linear_attn.chunk_gated_delta_rule
        precision.install_huanyu_actor_precision(actor)
        self.assertIs(first, layer.linear_attn.chunk_gated_delta_rule)
        native = Mock(__name__="chunk_gated_delta_rule", __module__="fla.ops.gated_delta_rule")
        layer.linear_attn.chunk_gated_delta_rule = native
        layer.layer_type = "already-set"
        precision.install_huanyu_actor_precision(actor)
        self.assertIs(layer.linear_attn.chunk_gated_delta_rule, native)
        self.assertEqual(layer.layer_type, "already-set")

    def test_core_optout_preserves_kernel(self):
        actor = model()
        with patch.dict(os.environ, {"MINICPMO_HUANYU_GDN_FP32_TRAINING": "0",
                                     "MINICPMO_ISOLATE_MEDIA_ENCODERS": "0"}):
            precision.install_huanyu_actor_precision(actor)
        self.assertEqual(self.imports, [])
        self.assertIs(actor.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule,
                      torch_chunk_gated_delta_rule)

    def test_optional_references_touch_language_only_and_match_source_order(self):
        actor = model()
        calls = []

        def reference(name):
            return SimpleNamespace(install_actor=lambda target: calls.append((name, target)))

        flags = {name: "1" for name in ("MINICPMO_MOE_MATCH_ROLLOUT",
                                          "MINICPMO_VISION_POSITION_REFERENCE")}
        flags.update(MINICPMO_MOE_TP_REFERENCE="4", MINICPMO_ROW_TP_REFERENCE="4",
                     MINICPMO_HUANYU_ROLLOUT_TP="4", MINICPMO_HUANYU_GDN_FP32_TRAINING="0")
        with patch.dict(os.environ, flags), patch.object(precision, "_shared", side_effect=reference):
            precision.install_huanyu_actor_precision(actor)
        names = [name for name, _ in calls]
        self.assertEqual(names, ["moe_precision", "moe_tp_reference", "row_tp_reference",
                                 "vision_position_precision", "media_precision"])
        for _, target in calls[:3]:
            self.assertIs(target, actor.model.language_model)
        for _, target in calls[3:]:
            self.assertIs(target, actor)

    def test_packed_attention_reference_cannot_discard_huanyu_padding_mask(self):
        with patch.dict(os.environ, {"MINICPMO_ISOLATE_FULL_ATTENTION": "1"}):
            with self.assertRaisesRegex(ValueError, "requires packed forward; Huanyu uses padded"):
                precision.install_huanyu_actor_precision(model())
        self.assertEqual(self.imports, [])

    def test_tp4_reference_rejects_missing_or_other_rollout_tp_before_patching(self):
        for tp in ("", "2", "8"):
            with self.subTest(tp=tp), patch.dict(os.environ, {
                "MINICPMO_MOE_TP_REFERENCE": "4", "MINICPMO_HUANYU_ROLLOUT_TP": tp,
            }):
                with self.assertRaisesRegex(ValueError, "actual rollout TP=4"):
                    precision.install_huanyu_actor_precision(model())
        self.assertEqual(self.imports, [])

    def test_invalid_switches_fail_instead_of_silently_disabling(self):
        for name, value in (("MINICPMO_HUANYU_GDN_FP32_TRAINING", "false"),
                            ("MINICPMO_MOE_TP_REFERENCE", "8")):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, name):
                    precision.install_huanyu_actor_precision(model())

    def test_later_actor_gets_stable_linear_scope_and_audio_remains_outside(self):
        first, second = model(), model()
        with patch.object(F, "linear", F.linear), patch.dict(os.environ, {"MINICPMO_STABLE_LINEAR": "1"}):
            precision.install_huanyu_actor_precision(first)
            precision.install_huanyu_actor_precision(second)
            forward_before = second.model.language_model.forward
            precision.install_huanyu_actor_precision(second)
            self.assertIs(forward_before, second.model.language_model.forward)
            # Observe inside forward itself because forward hooks run outside
            # the scope by design; recomputation can bypass hooks entirely.
            for actor in (first, second):
                with patch.object(actor.model.language_model.layers[0], "forward",
                                  side_effect=lambda x: x + linear._actor_depth.get()):
                    torch.testing.assert_close(actor.model.language_model.forward(torch.zeros(1)), torch.ones(1))
                self.assertEqual(linear._actor_depth.get(), 0)
                actor.model.audio_encoder(torch.ones(1, 3))
                self.assertEqual(linear._actor_depth.get(), 0)

    def test_vision_policy_rejects_unversioned_cache_or_opposite_reference(self):
        for extra, message in (({"HUANYU_USE_PRECOMPUTE": "1"}, "precompute caches"),
                               ({"MINICPMO_NATIVE_VISION_REFERENCE": "1"}, "conflict")):
            with self.subTest(extra=extra), patch.dict(os.environ, {
                "MINICPMO_VISION_POSITION_REFERENCE": "1", **extra,
            }):
                with self.assertRaisesRegex(ValueError, message):
                    precision.install_huanyu_actor_precision(model())
        self.assertEqual(self.imports, [])

    def test_foreign_stable_linear_cannot_silently_use_the_wrong_scope(self):
        def foreign_linear(*args, **kwargs):
            raise AssertionError("must reject before executing the foreign linear")

        foreign_linear._q36_stable_linear = True
        foreign_linear.__module__ = "whisper.linear_precision"
        with patch.object(F, "linear", foreign_linear), patch.dict(os.environ, {
            "MINICPMO_STABLE_LINEAR": "1",
        }):
            with self.assertRaisesRegex(RuntimeError, "another stable-linear precision implementation"):
                precision.install_huanyu_actor_precision(model())

    def test_scope_resets_after_forward_exception(self):
        actor = model()
        with patch.object(F, "linear", F.linear), patch.dict(os.environ, {"MINICPMO_STABLE_LINEAR": "1"}):
            precision.install_huanyu_actor_precision(actor)
            with patch.object(actor.model.language_model.layers[0], "forward", side_effect=RuntimeError("forward failed")):
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    actor.model.language_model(torch.ones(1))
            self.assertEqual(linear._actor_depth.get(), 0)

    def test_requested_installer_failure_propagates(self):
        with patch.dict(os.environ, {"MINICPMO_MOE_MATCH_ROLLOUT": "1"}), \
                patch.object(precision, "_shared", side_effect=ImportError("missing kernel")):
            with self.assertRaisesRegex(ImportError, "missing kernel"):
                precision.install_huanyu_actor_precision(model())


if __name__ == "__main__":
    unittest.main()
