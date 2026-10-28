"""Full audio-update checks using real CPU PyTorch modules and tensors."""

import gc
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import weakref

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_audio_reload_guard_tests", ROOT / "audio_reload_guard.py")
guard_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard_module)
AudioReloadGuard = guard_module.AudioReloadGuard


def tiny_model():
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.apm = torch.nn.Sequential(torch.nn.LayerNorm(40), torch.nn.LayerNorm(20))
    model.thinker.audio_projection_layer = torch.nn.Linear(20, 4)
    return model.bfloat16()


def tiny_vgg_model():
    model = tiny_model()
    model.thinker.apm = torch.nn.Module()
    model.thinker.apm.conv_layers = torch.nn.ModuleList()
    for features in (40, 20):
        block = torch.nn.Module()
        block.layers = torch.nn.ModuleList([
            torch.nn.Identity(), torch.nn.LayerNorm(features), torch.nn.ReLU(),
            torch.nn.Identity(), torch.nn.LayerNorm(features),
        ])
        model.thinker.apm.conv_layers.append(block)
    return model.bfloat16()


def audio_weights(model):
    return [
        (prefix + name, param.detach().clone())
        for module, prefix in (
            (model.thinker.apm, "model.audio_encoder."),
            (model.thinker.audio_projection_layer, "model.audio_projection."),
        )
        for name, param in module.named_parameters()
    ]


class AudioReloadGuardTests(unittest.TestCase):
    def test_complete_update_with_split_layer_across_buckets(self):
        model = tiny_model()
        guard = AudioReloadGuard(model)
        weights = audio_weights(model)
        self.assertEqual(guard.ingest(weights[:1]), 1)
        self.assertEqual(guard.ingest(dict(weights[1:3])), 2)
        self.assertEqual(guard.ingest(weights[3:] + [("model.language_model.weight", torch.ones(2, 3))]), 3)
        guard.check_complete()

    def test_missing_bias_rejected_before_finalize(self):
        model = tiny_model()
        guard = AudioReloadGuard(model)
        finalize = Mock()
        missing_name = "model.audio_encoder.0.bias"
        weights = [(name, value) for name, value in audio_weights(model) if name != missing_name]
        guard.ingest(weights[:1])
        guard.ingest(weights[1:])
        with self.assertRaisesRegex(RuntimeError, r"Incomplete.*model.audio_encoder.0.bias"):
            guard.check_complete()
            finalize(model)
        finalize.assert_not_called()

    def test_wholly_omitted_audio_can_keep_original_frozen_weights(self):
        model = tiny_model().requires_grad_(False)
        guard = AudioReloadGuard(model)
        self.assertEqual(guard.ingest([("model.language_model.weight", torch.ones(2, 3))]), 0)
        guard.check_complete()
        guard.verify(model)

    def test_duplicate_name_within_bucket(self):
        model = tiny_model()
        weight = audio_weights(model)[0]
        guard = AudioReloadGuard(model)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            guard.ingest([weight, weight])

    def test_alias_duplicate_across_buckets(self):
        model = tiny_model()
        guard = AudioReloadGuard(model)
        name, value = audio_weights(model)[0]
        guard.ingest([(name, value)])
        with self.assertRaisesRegex(ValueError, "Duplicate.*model.audio_encoder.0.weight"):
            guard.ingest([("apm.0.weight", value)])

    def test_native_aliases_form_a_complete_update(self):
        model = tiny_model()
        guard = AudioReloadGuard(model)
        native_weights = [
            (name.replace("model.audio_encoder.", "apm.").replace(
                "model.audio_projection.", "audio_projection_layer."), value)
            for name, value in audio_weights(model)
        ]
        self.assertEqual(guard.ingest(native_weights), 6)
        guard.check_complete()

    def test_unknown_audio_parameter_rejected(self):
        guard = AudioReloadGuard(tiny_model())
        with self.assertRaisesRegex(ValueError, "Unknown.*apm.missing.weight"):
            guard.ingest([("apm.missing.weight", torch.ones(40))])

    def test_same_numel_wrong_shape_rejected(self):
        guard = AudioReloadGuard(tiny_model())
        with self.assertRaisesRegex(ValueError, r"shape mismatch.*expected \(40,\).*received \(2, 20\)"):
            guard.ingest([("apm.0.weight", torch.ones(2, 20))])

    def test_guard_retains_neither_tensors_nor_model(self):
        model = tiny_model()
        guard = AudioReloadGuard(model)
        weights = audio_weights(model)
        model_ref = weakref.ref(model)
        tensor_refs = [weakref.ref(value) for _, value in weights]
        guard.ingest(weights)
        del weights, model
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertTrue(all(reference() is None for reference in tensor_refs))
        guard.check_complete()

    def test_missing_audio_module_rejected(self):
        model = tiny_model()
        del model.thinker.audio_projection_layer
        with self.assertRaisesRegex(RuntimeError, "audio_projection_layer is absent"):
            AudioReloadGuard(model)

    def test_verify_detects_partial_corruption_of_shape40_with_shape20_intact(self):
        model = tiny_vgg_model()
        guard = AudioReloadGuard(model)
        guard.ingest(audio_weights(model))
        guard.check_complete()
        guard.verify(model)
        norm40 = model.thinker.apm.conv_layers[0].layers[1]
        norm20 = model.thinker.apm.conv_layers[1].layers[1]
        control = norm20.weight.detach().clone()
        with torch.no_grad():
            norm40.weight[32:35].add_(1)
        with self.assertRaisesRegex(
            RuntimeError, r"conv_layers.0.layers.1.weight.*mismatched_elements=3/40.*bad_indices=\[32, 33, 34\]"
        ):
            guard.verify(model)
        torch.testing.assert_close(norm20.weight, control, rtol=0, atol=0)

    def test_verify_allows_expected_fp32_to_bf16_cast(self):
        model = tiny_vgg_model()
        weights = [(name, value.float() + 0.123456) for name, value in audio_weights(model)]
        guard = AudioReloadGuard(model)
        guard.ingest(weights)
        guard.check_complete()
        with torch.no_grad():
            for name, value in weights:
                if name.startswith("model.audio_encoder."):
                    module = model.thinker.apm
                    relative = name[len("model.audio_encoder."):]
                else:
                    module = model.thinker.audio_projection_layer
                    relative = name[len("model.audio_projection."):]
                path, _, leaf = relative.rpartition(".")
                layer = module.get_submodule(path) if path else module
                getattr(layer, leaf).copy_(value)
        guard.verify(model)

    def test_norm_snapshots_are_independent_and_bounded(self):
        model = tiny_vgg_model()
        weights = audio_weights(model)
        guard = AudioReloadGuard(model)
        guard.ingest(weights)
        for _, value in weights:
            value.zero_()  # Simulate later reuse or mutation of source tensors.
        guard.verify(model)
        self.assertEqual(len(guard._norm_values), 8)
        self.assertEqual(sum(value.numel() for value in guard._norm_values.values()), 240)
        self.assertTrue(all(value.device.type == "cpu" for value in guard._norm_values.values()))


if __name__ == "__main__":
    unittest.main()
