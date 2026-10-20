"""Check online Huanyu gradients and the frozen/trainable boundary on CPU."""

import copy
import importlib
import io
import logging
import os
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_audio_runtime_on_cpu as fixture


training = importlib.import_module(f"{fixture.PACKAGE}.audio_training")
forward = importlib.import_module(f"{fixture.PACKAGE}.forward")


class TinyHFModel(torch.nn.Module):
    """Use an actual embedding lookup so scatter participates in autograd."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.nn.Module()
        self.embedding = torch.nn.Embedding(32, 6)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return self.embedding(input_ids)


class AudioTrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "HUANYU_USE_PRECOMPUTE": "0", "MINICPMO_USE_PRECOMPUTE": "0",
            "MINICPMO_TUNE_AUDIO": "0", "MINICPMO_TUNE_VISION": "0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_freeze_default_and_override_precedence(self):
        self.assertTrue(training.resolve_freeze_audio(SimpleNamespace()))
        self.assertFalse(training.resolve_freeze_audio(
            SimpleNamespace(hf_config=SimpleNamespace(huanyu_freeze_audio=False))))
        self.assertTrue(training.resolve_freeze_audio({
            "override_config": {"huanyu_freeze_audio": True},
            "hf_config": {"huanyu_freeze_audio": False},
        }))
        self.assertFalse(training.resolve_freeze_audio(
            SimpleNamespace(override_config={"huanyu_freeze_audio": False})))

    def test_options_are_booleans_not_truthy_strings_or_numbers(self):
        for resolver, option in (
            (training.resolve_freeze_audio, "huanyu_freeze_audio"),
            (training.resolve_audio_checkpointing, "huanyu_audio_gradient_checkpointing"),
        ):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, "must be a boolean"):
                        resolver(SimpleNamespace(override_config={option: value}))

    def test_chunk_checkpointing_defaults_true_and_can_be_disabled(self):
        self.assertTrue(training.resolve_audio_checkpointing(SimpleNamespace()))
        self.assertFalse(training.resolve_audio_checkpointing({
            "override_config": {"huanyu_audio_gradient_checkpointing": False},
        }))

    def test_cache_is_compatible_only_with_frozen_audio(self):
        unfrozen = SimpleNamespace(override_config={"huanyu_freeze_audio": False})
        frozen = SimpleNamespace(override_config={"huanyu_freeze_audio": True})
        self.assertFalse(training.validate_audio_mode(unfrozen))
        for cache_flag in ("HUANYU_USE_PRECOMPUTE", "MINICPMO_USE_PRECOMPUTE"):
            with self.subTest(flag=cache_flag), patch.dict(os.environ, {cache_flag: "1"}):
                with self.assertRaisesRegex(ValueError, "precomputed embeddings cannot train audio"):
                    training.validate_audio_mode(unfrozen)
        with patch.dict(os.environ, {"HUANYU_USE_PRECOMPUTE": "1"}):
            self.assertTrue(training.validate_audio_mode(frozen))

    def test_legacy_environment_cannot_silently_unfreeze_default(self):
        with patch.dict(os.environ, {"MINICPMO_TUNE_AUDIO": "1"}):
            self.assertTrue(training.resolve_freeze_audio(SimpleNamespace()))
            with self.assertRaisesRegex(ValueError, "conflicts with frozen audio"):
                training.validate_audio_mode(SimpleNamespace())
            self.assertFalse(training.validate_audio_mode({
                "override_config": {"huanyu_freeze_audio": False},
            }))

    def test_audio_training_rejects_vision_training(self):
        with patch.dict(os.environ, {"MINICPMO_TUNE_VISION": "1"}):
            with self.assertRaisesRegex(ValueError, "frozen vision"):
                training.validate_audio_mode({
                    "override_config": {"huanyu_freeze_audio": False},
                })


class AudioTrainingGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        fixture.AudioRuntimeTests.setUp(self)
        self.module = TinyHFModel(self.module.config)
        fixture.runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.module.float()
        self.features = torch.randn(3, 4, 24)
        self.lengths = torch.tensor([16, 9, 24])

    def configure(self, *, checkpoint_chunks=False, model=None):
        model = self.module if model is None else model
        training.configure_trainable_audio(model, checkpoint_chunks=checkpoint_chunks)
        return model

    def assert_audio_gradients(self, model):
        for name in ("audio_encoder", "audio_projection"):
            parameters = list(getattr(model.model, name).parameters())
            self.assertTrue(all(parameter.grad is not None for parameter in parameters), name)
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters), name)
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in parameters), 0, name)

    def test_trainable_and_frozen_compute_match_without_changing_master_parameters(self):
        frozen = fixture.runtime.encode_huanyu_audio(self.module, self.features, self.lengths)
        before = {name: (id(value), value.detach().clone())
                  for name, value in self.module.named_parameters()}
        state_keys = set(self.module.state_dict())
        self.configure()
        trainable = training.encode_trainable_huanyu_audio(self.module, self.features, self.lengths)
        self.assertEqual([tuple(chunk.shape) for chunk in trainable], [(2, 6), (1, 6), (3, 6)])
        for expected, actual in zip(frozen, trainable):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(actual.requires_grad)
        self.assertEqual(set(self.module.state_dict()), state_keys)
        self.assertTrue(any(key.startswith("model.audio_encoder.") for key in state_keys))
        self.assertTrue(any(key.startswith("model.audio_projection.") for key in state_keys))
        for name, value in self.module.named_parameters():
            self.assertEqual(id(value), before[name][0])
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, before[name][1], rtol=0, atol=0)

    def test_embedding_scatter_backpropagates_to_encoder_projector_and_optimizer(self):
        self.configure(checkpoint_chunks=True)
        forward.install_huanyu_forward(
            self.module, start_token_id=2, end_token_id=3, placeholder_token_id=4,
            encode_audio=training.encode_trainable_huanyu_audio, require_audio_only=True,
        )
        parameters = dict(self.module.named_parameters())
        before = {key: value.detach().clone() for key, value in parameters.items()}
        optimizer = torch.optim.SGD(self.module.parameters(), lr=0.1)
        output = self.module(
            torch.tensor([[9, 2, 4, 4, 3, 2, 4, 3]]),
            huanyu_audio_features=[self.features[0, :, :16], self.features[1, :, :9]],
            huanyu_audio_lengths=[16, 9], huanyu_audio_chunk_counts=[2],
        )
        with self.assertLogs(training.logger, level="INFO") as logs:
            output.square().sum().backward()
        messages = "\n".join(logs.output)
        self.assertIn("component=audio_encoder parameter=input.weight", messages)
        self.assertIn("component=audio_projection parameter=linear1.weight", messages)
        self.assertEqual(messages.count("backward component="), 2)
        self.assertNotIn("parameter=inv_freq", messages)
        self.assert_audio_gradients(self.module)
        self.assertGreater(self.module.embedding.weight.grad[9].abs().sum().item(), 0)
        torch.testing.assert_close(self.module.embedding.weight.grad[4], torch.zeros(6))
        optimizer.step()
        for prefix in ("model.audio_encoder.", "model.audio_projection."):
            self.assertTrue(any(not torch.equal(value, before[key])
                                for key, value in parameters.items() if key.startswith(prefix)), prefix)

    def test_training_evidence_is_visible_once_under_warning_root(self):
        # Capture the real console handler, without assertLogs changing levels.
        stream = io.StringIO()
        root = logging.getLogger()
        old_level, old_handlers = root.level, root.handlers[:]
        root_stream = io.StringIO()
        root_handler = logging.StreamHandler(root_stream)
        root_handler.setLevel(logging.WARNING)
        root.handlers = [root_handler]
        root.setLevel(logging.WARNING)
        try:
            handlers = [
                handler for handler in training.logger.handlers
                if isinstance(handler, logging.StreamHandler)
            ]
            self.assertTrue(handlers, "Huanyu training needs its own console handler")
            with ExitStack() as stack:
                for handler in handlers:
                    stack.enter_context(patch.object(handler, "stream", stream))
                self.configure(checkpoint_chunks=True)
                with torch.no_grad():
                    training.encode_trainable_huanyu_audio(self.module, self.features, self.lengths)
                for _ in range(2):
                    self.module.zero_grad()
                    outputs = training.encode_trainable_huanyu_audio(
                        self.module, self.features, self.lengths)
                    sum(value.float().square().sum() for value in outputs).backward()
                    self.assert_audio_gradients(self.module)
            messages = stream.getvalue()
            for marker in (
                "[huanyu-audio-train] enabled",
                "forward mode=no_grad",
                "forward mode=grad",
                "backward component=audio_encoder",
                "backward component=audio_projection",
            ):
                self.assertEqual(messages.count(marker), 1, messages)
            self.assertIn("output_requires_grad=True", messages)
            self.assertEqual(root_stream.getvalue(), "")
            self.assertEqual(root.level, logging.WARNING)
            self.assertEqual(root.handlers, [root_handler])
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)
            root_handler.close()

    def test_nonreentrant_checkpoint_preserves_outputs_and_all_audio_gradients(self):
        direct = self.configure()
        checked = self.configure(model=copy.deepcopy(direct), checkpoint_chunks=True)
        direct_outputs = training.encode_trainable_huanyu_audio(direct, self.features, self.lengths)
        checked_outputs = training.encode_trainable_huanyu_audio(checked, self.features, self.lengths)
        for expected, actual in zip(direct_outputs, checked_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for model, outputs in ((direct, direct_outputs), (checked, checked_outputs)):
            sum(chunk.float().square().sum() for chunk in outputs).backward()
            self.assert_audio_gradients(model)
        direct_params = dict(direct.model.named_parameters())
        for name, parameter in checked.model.named_parameters():
            torch.testing.assert_close(parameter.grad, direct_params[name].grad, rtol=0, atol=0)

    def test_no_grad_logprob_and_frozen_reference_keep_online_audio_functional(self):
        self.configure(checkpoint_chunks=True)
        with torch.no_grad():
            outputs = training.encode_trainable_huanyu_audio(self.module, self.features, self.lengths)
        self.assertTrue(all(not output.requires_grad for output in outputs))
        self.assertTrue(all(parameter.requires_grad for parameter in self.module.model.parameters()))
        self.module.requires_grad_(False).eval()
        with torch.no_grad():
            reference_outputs = training.encode_trainable_huanyu_audio(
                self.module, self.features, self.lengths)
        for expected, actual in zip(outputs, reference_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(self.module.model.audio_encoder.training)
        with self.assertRaisesRegex(RuntimeError, "frozen after adapter configuration"):
            training.encode_trainable_huanyu_audio(self.module, self.features, self.lengths)

    def test_variable_chunk_counts_use_plain_audio_calls_and_detect_nested_fsdp(self):
        self.configure()
        observed = []
        handle = self.module.model.audio_encoder.register_forward_pre_hook(
            lambda part, args: observed.append((tuple(args[0].shape), args[1].tolist())))
        self.addCleanup(handle.remove)
        training.encode_trainable_huanyu_audio(self.module, self.features, self.lengths)
        self.assertEqual(observed, [
            ((1, 4, 16), [16]), ((1, 4, 12), [9]), ((1, 4, 24), [24]),
        ])
        child = self.module.model.audio_encoder.input
        child.unshard = Mock(side_effect=AssertionError("collective must not run"))
        child.reshard = Mock(side_effect=AssertionError("collective must not run"))
        with self.assertRaisesRegex(RuntimeError, "independently FSDP"):
            training.encode_trainable_huanyu_audio(self.module, self.features[:1], self.lengths[:1])
        child.unshard.assert_not_called()
        child.reshard.assert_not_called()
        self.assertEqual(len(observed), 3)

    def test_audio_wrap_policy_rejected_before_enabling_parameter_gradients(self):
        self.module._no_split_modules = ["TinyEncoder"]
        with self.assertRaisesRegex(ValueError, "independently FSDP"):
            self.configure()
        self.assertTrue(all(not value.requires_grad for value in self.module.model.parameters()))

    def test_invalid_lengths_and_feature_width_fail_before_native_forward(self):
        self.configure()
        with self.assertRaisesRegex(ValueError, "fbank mismatch"):
            training.encode_trainable_huanyu_audio(self.module, torch.ones(1, 128, 16), [16])
        for lengths in ([0], [25], [16.0], [True], [[16]]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                training.encode_trainable_huanyu_audio(self.module, self.features[:1], lengths)


class AudioOnlyForwardTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyHFModel(SimpleNamespace())
        self.encode = Mock(side_effect=AssertionError("invalid inputs must fail before encoding"))
        forward.install_huanyu_forward(
            self.model, start_token_id=2, end_token_id=3, placeholder_token_id=4,
            encode_audio=self.encode, require_audio_only=True,
        )

    def test_no_audio_or_one_text_only_sample_fail_before_encoding(self):
        with self.assertRaisesRegex(ValueError, "audio in every AQA sample"):
            self.model(torch.tensor([[9, 10, 11]]))
        with self.assertRaisesRegex(ValueError, "audio in every AQA sample"):
            self.model(
                torch.tensor([[2, 4, 3], [9, 10, 11]]),
                huanyu_audio_features=[torch.ones(4, 8)], huanyu_audio_lengths=[8],
                huanyu_audio_chunk_counts=[1, 0],
            )
        self.encode.assert_not_called()

    def test_visual_inputs_fail_before_encoding(self):
        for key, value in (
            ("pixel_values", torch.ones(4, 8)),
            ("image_grid_thw", torch.tensor([[1, 2, 2]])),
            ("pixel_values_videos", torch.ones(4, 8)),
            ("video_grid_thw", torch.tensor([[1, 2, 2]])),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "AQA audio-only"):
                self.model(
                    torch.tensor([[2, 4, 3]]), huanyu_audio_features=[torch.ones(4, 8)],
                    huanyu_audio_lengths=[8], **{key: value},
                )
        self.encode.assert_not_called()

    def test_cache_mode_cannot_be_combined_with_trainable_audio_only_forward(self):
        with self.assertRaisesRegex(ValueError, "cache|cached|precompute"):
            forward.install_huanyu_forward(
                TinyHFModel(SimpleNamespace()), start_token_id=2, end_token_id=3,
                placeholder_token_id=4, require_audio_only=True,
                use_precompute=True, image_token_id=5,
            )


if __name__ == "__main__":
    unittest.main()
