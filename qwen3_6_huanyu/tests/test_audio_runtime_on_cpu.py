"""Native-runtime boundary tests without importing an inference engine."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "huanyu_audio_runtime_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.audio_runtime", ROOT / "audio_runtime.py")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class TinyEncoder(torch.nn.Module):
    def __init__(self, input_feat, embed_dim, ffn_dim, num_heads, num_conformer_layers, flat_dim):
        super().__init__()
        self.input = torch.nn.Linear(input_feat, embed_dim)
        self.register_buffer("inv_freq", torch.tensor([0.123456789], dtype=torch.float32), persistent=False)
        self.fwd_downsample = 4

    def forward(self, features, lengths):
        if self.input.weight.dtype != torch.bfloat16:
            raise AssertionError("Native Huanyu expects BF16 Conformer parameters")
        if self.inv_freq.dtype != torch.float32:
            raise AssertionError("Rotary constants lost FP32 precision")
        result = self.input(features[:, :, ::4].transpose(1, 2).bfloat16())
        return result, result.shape[1]


class TinyProjector(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, out_dim)
        self.linear2 = torch.nn.Linear(out_dim, out_dim)

    def forward(self, values):
        return self.linear2(self.linear1(values).relu())


def native_fold(values, lengths, projector, enc_downsample, downsample_rate):
    batch, seq, hidden = values.shape
    seq -= seq % downsample_rate
    projected = projector(values[:, :seq].reshape(batch, seq // downsample_rate, hidden * downsample_rate))
    valid = ((lengths + enc_downsample - 1) // enc_downsample // downsample_rate).clamp(max=projected.shape[1])
    return projected, valid


class AudioRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.audio_cfg = dict(
            model_type="huanyu_conformer_0_6b", projector_type="firered_conv2d",
            input_feat_per_channel=4, encoder_hidden_size=8, encoder_ffn_dim=16,
            encoder_num_heads=2, loaded_transformer_module_indices=[0, 1],
            projector_downsample_rate=2, projector_output_size=6,
        )
        self.module = torch.nn.Module()
        self.module.model = torch.nn.Module()
        self.module.config = SimpleNamespace(audio_config=self.audio_cfg, text_config={"hidden_size": 6})
        self.model_cfg = SimpleNamespace(path=str(self.path))
        self.encoder = TinyEncoder(4, 8, 16, 2, 1, None)
        self.projector = TinyProjector(16, 6)
        state = {f"model.audio_encoder.{key}": value for key, value in self.encoder.state_dict().items()}
        state.update({f"model.audio_projection.{key}": value for key, value in self.projector.state_dict().items()})
        state["model.language_model.weight"] = torch.ones(2, 3)
        save_file(state, str(self.path / "model.safetensors"))
        self.native = SimpleNamespace(
            HuanyuAudioEncoder=TinyEncoder, huanyu_project_and_fold=native_fold, __file__="native/huanyu_audio.py"
        )
        self.native_patch = patch.object(runtime, "load_huanyu_native", return_value=self.native)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.import_patch = patch.object(runtime, "import_module", return_value=
                                        SimpleNamespace(MultiModalProjector=TinyProjector))
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def test_strict_loading_preserves_checkpoint_namespace(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.assertTrue(all(not item.requires_grad for item in self.module.parameters()))
        self.assertEqual(set(self.module.state_dict()),
                         {f"model.audio_encoder.{key}" for key in self.encoder.state_dict()} |
                         {f"model.audio_projection.{key}" for key in self.projector.state_dict()})
        torch.testing.assert_close(self.module.model.audio_projection.linear1.weight,
                                   self.projector.linear1.weight.bfloat16())
        self.assertIs(runtime.attach_huanyu_audio(self.module, self.model_cfg), self.module)

    def test_fp32_actor_master_weights_are_not_mutated_by_bf16_audio(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.module.float()
        before = {key: (id(value), value.detach().clone()) for key, value in self.module.named_parameters()}
        chunks = runtime.encode_huanyu_audio(self.module, torch.ones(2, 4, 16), torch.tensor([16, 9]))
        self.assertEqual([tuple(item.shape) for item in chunks], [(2, 6), (1, 6)])
        self.assertTrue(all(item.dtype == torch.bfloat16 and not item.requires_grad for item in chunks))
        for key, value in self.module.named_parameters():
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(id(value), before[key][0])
            torch.testing.assert_close(value, before[key][1])

    def test_bf16_engine_cast_does_not_round_rotary_constants_during_forward(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        original = self.module.model.audio_encoder.inv_freq.clone()
        self.module.bfloat16()
        observed = []
        handle = self.module.model.audio_encoder.register_forward_pre_hook(
            lambda part, args: observed.append(part.inv_freq.clone()))
        self.addCleanup(handle.remove)
        runtime.encode_huanyu_audio(self.module, torch.ones(1, 4, 16), [16])
        torch.testing.assert_close(observed[0], original)
        self.assertEqual(self.module.model.audio_encoder.inv_freq.dtype, torch.bfloat16)

    def test_bad_checkpoint_fails_before_attachment(self):
        self.audio_cfg["encoder_hidden_size"] = 16
        with self.assertRaisesRegex(ValueError, "shape_mismatches"):
            runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.assertFalse(hasattr(self.module.model, "audio_encoder"))

    def test_each_chunk_has_same_padding_online_and_offline(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        observed = []
        hook = self.module.model.audio_encoder.register_forward_pre_hook(
            lambda part, args: observed.append((tuple(args[0].shape), args[1].tolist())))
        self.addCleanup(hook.remove)
        features = torch.randn(2, 4, 24)
        together = runtime.encode_huanyu_audio(self.module, features, [16, 9])
        self.assertEqual(observed, [((1, 4, 16), [16]), ((1, 4, 12), [9])])
        separately = runtime.encode_huanyu_audio(self.module, features[1:2, :, :9], [9])
        torch.testing.assert_close(together[1], separately[0], rtol=0, atol=0)

    def test_meta_outer_construction_is_materialized(self):
        with torch.device("meta"):
            runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.assertTrue(all(not item.is_meta for item in self.module.parameters()))
        self.assertTrue(all(not item.is_meta for item in self.module.buffers()))

    def test_missing_audio_and_corrupt_shard_index_are_not_silently_accepted(self):
        (self.path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "model.audio_encoder.nonexistent.weight": "model.safetensors"
        }}))
        with self.assertRaisesRegex(ValueError, "index mismatch"):
            runtime.load_huanyu_audio_weights(self.path)

    def test_wrong_feature_width_and_zero_token_chunks_are_rejected(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        with self.assertRaisesRegex(ValueError, "fbank mismatch"):
            runtime.encode_huanyu_audio(self.module, torch.ones(1, 128, 16), [16])
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            runtime.encode_huanyu_audio(self.module, torch.ones(1, 4, 8), [4])

    def test_audio_finetuning_and_independent_fsdp_do_not_silently_bypass_safety(self):
        runtime.attach_huanyu_audio(self.module, self.model_cfg)
        self.module.model.audio_encoder.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen audio only"):
            runtime.encode_huanyu_audio(self.module, torch.ones(1, 4, 16), [16])
        self.module.model.audio_encoder.requires_grad_(False)
        self.module.model.audio_encoder.unshard = lambda: None
        self.module.model.audio_encoder.reshard = lambda: None
        with self.assertRaisesRegex(RuntimeError, "independently FSDP"):
            runtime.encode_huanyu_audio(self.module, torch.ones(1, 4, 16), [16])


if __name__ == "__main__":
    unittest.main()
