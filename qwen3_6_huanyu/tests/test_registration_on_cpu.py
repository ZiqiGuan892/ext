"""Check independent training registration and admission guards without verl."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_registration_tests"
HF_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"


class Tokenizer:
    pieces = {"<|audio_start|>": 10, "<|audio_end|>": 11, "<unk>": 12}

    def encode(self, text, add_special_tokens=False):
        return [self.pieces[text]]

    def convert_tokens_to_ids(self, text):
        return self.pieces[text]


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.original_training_adapter = object()
        self.rollout_registry = {"qwen3_6_whisper": object(), "qwen3_5_moe": object()}
        self.original_rollouts = dict(self.rollout_registry)
        registry = {(HF_ARCHITECTURE, "thinker"): self.original_training_adapter}
        self.registry = registry
        self.base_configure = Mock(side_effect=lambda module, config: module)
        configure = self.base_configure

        class FakeOmniModelBase:
            @classmethod
            def register(cls, architecture, stage):
                def decorate(adapter):
                    registry[(architecture, stage)] = adapter
                    return adapter
                return decorate

            @classmethod
            def configure_model(cls, module, config):
                return configure(module, config)

        base = ModuleType("verl_omni.pipelines.model_base")
        base.OmniModelBase = FakeOmniModelBase
        # The Huanyu training adapter must not register under either existing
        # inference name, regardless of the order Whisper/RM were imported.
        base.OmniRolloutPipelineBase = SimpleNamespace(registry=self.rollout_registry)
        package = ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        audio_runtime = ModuleType(f"{PACKAGE}.audio_runtime")
        self.attach = Mock(side_effect=self.attach_audio)
        audio_runtime.attach_huanyu_audio = self.attach
        forward = ModuleType(f"{PACKAGE}.forward")
        self.install_forward = Mock()
        forward.install_huanyu_forward = self.install_forward
        visual = ModuleType("verl_omni_ext.models.qwen3_6_whisper.forward")
        self.visual_patch = Mock()
        visual.patch_visual_fsdp_unshard = self.visual_patch
        self.modules_patch = patch.dict(sys.modules, {
            PACKAGE: package,
            base.__name__: base,
            audio_runtime.__name__: audio_runtime,
            forward.__name__: forward,
            visual.__name__: visual,
        })
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.thinker_adapter", ROOT / "thinker_adapter.py")
        self.adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter_module)
        self.adapter = self.adapter_module.Qwen36HuanyuThinkerAdapter
        self.environment = patch.dict(os.environ, {
            "MINICPMO_TUNE_AUDIO": "0", "MINICPMO_TUNE_VISION": "0", "MINICPMO_USE_PRECOMPUTE": "0",
            "HUANYU_USE_PRECOMPUTE": "0", "HUANYU_EMBED_CACHE_DIR": "",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.visual = torch.nn.Linear(2, 2)
        self.model.config = SimpleNamespace(
            model_type="qwen3_5_moe", architectures=[HF_ARCHITECTURE],
            audio_config={"model_type": "huanyu_conformer_0_6b"}, text_config={"hidden_size": 6},
            audio_start_token_id=10, audio_end_token_id=11,
        )
        self.config = SimpleNamespace(
            hf_config=self.model.config, use_remove_padding=False,
            tokenizer=Tokenizer(), path="unused-checkpoint", trust_remote_code=True,
        )

    @staticmethod
    def attach_audio(module, model_config):
        # Strict loading itself is tested by test_audio_runtime_on_cpu.py.
        # Here, check that the adapter delegates to it before wiring forward.
        module.model.audio_encoder = torch.nn.Linear(2, 2)
        module.model.audio_projection = torch.nn.Linear(2, 2)
        return module

    def test_alias_registration_leaves_whisper_and_reward_names_unchanged(self):
        self.assertIs(self.registry[(HF_ARCHITECTURE, "thinker")], self.original_training_adapter)
        self.assertIs(self.registry[("Qwen36HuanyuForTraining", "thinker")], self.adapter)
        self.assertEqual(len(self.registry), 2)
        self.assertEqual(self.rollout_registry, self.original_rollouts)
        self.assertEqual(self.model.config.architectures, [HF_ARCHITECTURE])

    def test_valid_model_uses_huanyu_loader_freezes_audio_and_installs_new_forward(self):
        result = self.adapter.configure_model(self.model, self.config)
        self.assertIs(result, self.model)
        self.base_configure.assert_called_once_with(self.model, self.config)
        self.attach.assert_called_once_with(self.model, self.config)
        self.visual_patch.assert_called_once_with(self.model)
        self.install_forward.assert_called_once_with(
            self.model, start_token_id=10, end_token_id=11, placeholder_token_id=12)
        for part in (self.model.model.audio_encoder, self.model.model.audio_projection, self.model.model.visual):
            self.assertFalse(part.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in part.parameters()))
        self.assertEqual(self.model.config.architectures, [HF_ARCHITECTURE])
        self.assertEqual(self.model.config.hidden_size, 6)

    def test_whisper_checkpoint_is_rejected_before_loading(self):
        self.model.config.audio_config = {"model_type": "whisper", "d_model": 1024}
        with self.assertRaisesRegex(ValueError, "huanyu_conformer_0_6b"):
            self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()
        self.install_forward.assert_not_called()

    def test_huanyu_cache_mode_uses_separate_switch_and_injection(self):
        self.model.config.image_token_id = 13
        self.model.config.vision_config = SimpleNamespace(spatial_merge_size=2, deepstack_visual_indexes=[])
        with patch.dict(os.environ, {"HUANYU_USE_PRECOMPUTE": "1", "HUANYU_EMBED_CACHE_DIR": "huanyu-cache"}):
            self.adapter.configure_model(self.model, self.config)
        self.install_forward.assert_called_once_with(
            self.model, start_token_id=10, end_token_id=11, placeholder_token_id=12,
            use_precompute=True, image_token_id=13, spatial_merge_size=2,
        )

    def test_cached_vision_cannot_be_tuned_or_require_deepstack(self):
        self.model.config.vision_config = SimpleNamespace(deepstack_visual_indexes=[1])
        with patch.dict(os.environ, {"HUANYU_USE_PRECOMPUTE": "1", "HUANYU_EMBED_CACHE_DIR": "huanyu-cache"}):
            with self.assertRaisesRegex(ValueError, "deepstack"):
                self.adapter.configure_model(self.model, self.config)
            with patch.dict(os.environ, {"MINICPMO_TUNE_VISION": "1"}):
                with self.assertRaisesRegex(ValueError, "frozen vision"):
                    self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()

    def test_huanyu_cache_mode_requires_explicit_cache_root(self):
        with patch.dict(os.environ, {"HUANYU_USE_PRECOMPUTE": "1"}):
            with self.assertRaisesRegex(ValueError, "HUANYU_EMBED_CACHE_DIR"):
                self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()

    def test_incorrect_backbone_is_rejected(self):
        self.model.config.model_type = "unrelated_model"
        with self.assertRaisesRegex(ValueError, "Qwen3.5-MoE HF backbone"):
            self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()

    def test_remove_padding_is_rejected_before_loading(self):
        self.config.use_remove_padding = True
        with self.assertRaisesRegex(ValueError, "use_remove_padding=false"):
            self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()

    def test_whisper_cache_and_audio_finetuning_are_rejected(self):
        for variable, message in (("MINICPMO_USE_PRECOMPUTE", "Whisper caches"),
                                  ("MINICPMO_TUNE_AUDIO", "frozen audio")):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "1"}):
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()
        self.install_forward.assert_not_called()

    def test_strict_loader_failure_does_not_install_a_working_looking_forward(self):
        self.attach.side_effect = ValueError("Huanyu projector: incompatible checkpoint")
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
            self.adapter.configure_model(self.model, self.config)
        self.install_forward.assert_not_called()
        self.assertEqual(self.model.config.architectures, [HF_ARCHITECTURE])

    def test_tokenizer_marker_mismatch_stops_before_audio_load(self):
        self.model.config.audio_start_token_id = 99
        with self.assertRaisesRegex(ValueError, "disagrees with config"):
            self.adapter.configure_model(self.model, self.config)
        self.attach.assert_not_called()


if __name__ == "__main__":
    unittest.main()
