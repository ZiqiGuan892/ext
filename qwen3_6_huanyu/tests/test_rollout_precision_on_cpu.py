"""Exercise Huanyu worker installation and real GDN state math on CPU."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


precision = load("huanyu_rollout_precision_tested", ROOT / "rollout_precision.py")
gdn = load("shared_gdn_precision_tested", ROOT / "qwen_precision" / "gdn_precision.py")
media = load("shared_media_precision_tested", ROOT / "qwen_precision" / "media_precision.py")


def config(huanyu=True):
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(
        audio_config={"model_type": "huanyu_conformer_0_6b" if huanyu else "whisper"})))


class RolloutPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.raw = Mock(return_value="legacy")
        self.raw._huanyu_fp32_state = False
        self.ops = SimpleNamespace(npu_recurrent_gated_delta_rule=self.raw)
        ops_patch = patch.object(torch.ops, "_C_ascend", self.ops)
        ops_patch.start()
        self.addCleanup(ops_patch.stop)
        self.helpers = {}

        def shared(name):
            if name == "gdn_precision":
                return SimpleNamespace(recurrent_gdn_fp32=gdn.recurrent_gdn_fp32,
                                       install_reference_prefill=Mock(), install_mixed_cache_barrier=Mock())
            if name == "media_precision":
                return media
            return self.helpers.setdefault(name, SimpleNamespace(install_rollout=Mock()))

        shared_patch = patch.object(precision, "_shared", side_effect=shared)
        shared_patch.start()
        self.addCleanup(shared_patch.stop)

    def tensors(self):
        generator = torch.Generator().manual_seed(71)
        return dict(
            query=torch.randn(2, 1, 3, generator=generator).bfloat16(),
            key=torch.randn(2, 1, 3, generator=generator).bfloat16(),
            value=torch.randn(2, 2, 4, generator=generator).bfloat16(),
            state=torch.randn(5, 2, 4, 3, generator=generator),
            beta=torch.rand(2, 2, generator=generator).bfloat16(),
            g=-torch.rand(2, 2, generator=generator),
            actual_seq_lengths=torch.tensor([0, 1, 1]),
            ssm_state_indices=torch.tensor([1, 3]),
        )

    def test_fp32_decode_matches_independent_delta_rule_and_keeps_other_slots(self):
        values = self.tensors()
        original = values["state"].clone()
        expected = original.clone()
        output = torch.empty_like(values["value"])
        for token, slot in enumerate([1, 3]):
            for head in range(2):
                state = expected[slot, head] * values["g"][token, head].exp()
                key = values["key"][token, 0].float()
                residual = values["value"][token, head].float() - state @ key
                state += torch.outer(residual * values["beta"][token, head].float(), key)
                expected[slot, head] = state
                output[token, head] = state @ (values["query"][token, 0].float() * (3 ** -0.5))

        class Native:
            def forward(self, **kwargs):
                return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**kwargs)

        native = Native()
        precision.configure_huanyu_rollout(native, config())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = native.forward(**values)
        torch.testing.assert_close(actual, output, rtol=0, atol=0)
        torch.testing.assert_close(values["state"], expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(values["state"][[0, 2, 4]], original[[0, 2, 4]]))
        self.raw.assert_not_called()
        self.assertFalse(precision._active.get())
        self.assertIs(self.ops.npu_recurrent_gated_delta_rule.default,
                      self.ops.npu_recurrent_gated_delta_rule)

    def test_scope_excludes_rm_and_resets_after_failure(self):
        class Native:
            def forward(self, **kwargs):
                return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**kwargs)

        native, rm = Native(), Native()
        precision.configure_huanyu_rollout(native, config())
        self.assertFalse(precision.configure_huanyu_rollout(rm, config(False)))
        values = self.tensors()
        self.assertEqual(rm.forward(**values), "legacy")
        values["state"] = values["state"].bfloat16()
        with self.assertRaisesRegex(RuntimeError, "float32 recurrent cache"):
            native.forward(**values)
        self.assertFalse(precision._active.get())
        self.assertEqual(rm.forward(**values), "legacy")

    def test_padding_slot_does_not_round_or_modify_other_cache(self):
        values = self.tensors()
        values["ssm_state_indices"] = torch.tensor([1, -1])
        before = values["state"].clone()
        precision.install_recurrent_fp32()
        token = precision._active.set(True)
        try:
            result = self.ops.npu_recurrent_gated_delta_rule(**values)
        finally:
            precision._active.reset(token)
        self.assertTrue(torch.equal(before[[0, 2, 3, 4]], values["state"][[0, 2, 3, 4]]))
        self.assertTrue(torch.equal(result[1], torch.zeros_like(result[1])))

    def test_initializer_runs_legacy_first_then_installs_and_is_idempotent(self):
        class Native:
            def __init__(self, *, vllm_config):
                # Simulate the inherited initializer retrying the legacy hook.
                self.initialized = True
                torch.ops._C_ascend.npu_recurrent_gated_delta_rule = self_raw

            def forward(self):
                return self.initialized

        self_raw = self.raw
        with patch.object(precision, "import_module", return_value=SimpleNamespace(
                MiniCPMO50OmniForConditionalGeneration=Native)):
            precision.install_huanyu_rollout_precision()
            first = Native.__init__
            precision.install_huanyu_rollout_precision()
            self.assertIs(first, Native.__init__)
        model = Native(vllm_config=config())
        self.assertTrue(model.forward())
        self.assertTrue(self.ops.npu_recurrent_gated_delta_rule._huanyu_fp32_state)
        self.assertTrue(self.ops.npu_recurrent_gated_delta_rule._is_bf16_patched)

    def test_each_image_is_isolated_without_wrapping_audio(self):
        calls = []

        class Thinker:
            visual = object()

            def _q36_encode_vision(self, pixels, grids):
                calls.append(tuple(pixels.shape))
                return (pixels.clone(),)

            def get_audio_hidden_states(self):
                return "native huanyu audio"

        class Native:
            thinker = Thinker()

            def forward(self):
                return None

        native = Native()
        audio = native.thinker.get_audio_hidden_states.__func__
        precision.configure_huanyu_rollout(native, config())
        outputs = native.thinker._q36_encode_vision(torch.arange(12).reshape(6, 2),
                                                  torch.tensor([[1, 2, 2], [1, 1, 2]]))
        self.assertEqual(calls, [(4, 2), (2, 2)])
        self.assertEqual(len(outputs), 2)
        self.assertIs(native.thinker.get_audio_hidden_states.__func__, audio)

    def test_optout_missing_operator_and_bad_cache_mode(self):
        with patch.dict(os.environ, {"MINICPMO_GDN_FP32_REFERENCE": "0"}):
            self.assertFalse(precision.install_recurrent_fp32())
        with patch.object(torch.ops, "_C_ascend", SimpleNamespace()):
            self.assertFalse(precision.install_recurrent_fp32())
            with self.assertRaisesRegex(RuntimeError, "after native init"):
                precision.configure_huanyu_rollout(SimpleNamespace(forward=lambda: None), config())
        with patch.dict(os.environ, {"MINICPMO_NATIVE_VISION_REFERENCE": "1"}):
            with self.assertRaisesRegex(ValueError, "cache producer"):
                precision.configure_huanyu_rollout(SimpleNamespace(), config())

    def test_saved_overload_packet_alias_and_legacy_fallback(self):
        packet = SimpleNamespace()

        class Raw:
            _overloadpacket = packet

            def __call__(self, *args, **kwargs):
                return "raw native"

        class Legacy:
            _is_bf16_patched = True

            def __init__(self):
                self._raw_op = Raw()

            def __call__(self, *args, **kwargs):
                return self._raw_op(*args, **kwargs)

        original = Legacy()
        packet.default = original
        self.ops.npu_recurrent_gated_delta_rule = original
        precision.install_recurrent_fp32()
        self.assertIs(packet.default, self.ops.npu_recurrent_gated_delta_rule)
        self.assertEqual(packet.default(**self.tensors()), "raw native")
        token = precision._active.set(True)
        try:
            self.assertIsInstance(packet.default(**self.tensors()), torch.Tensor)
        finally:
            precision._active.reset(token)

    def test_initializer_can_use_constructed_config(self):
        class Native:
            def __init__(self):
                self.vllm_config = config()

            def forward(self):
                return None

        with patch.object(precision, "import_module", return_value=SimpleNamespace(
                MiniCPMO50OmniForConditionalGeneration=Native)):
            precision.install_huanyu_rollout_precision()
        self.assertTrue(Native()._huanyu_precision_configured)

    def test_architecture_selection_does_not_rely_on_filename(self):
        self.assertTrue(precision.is_huanyu_config({"model_config": {"model": "/renamed/checkpoint",
            "hf_config": {"audio_config": {"model_type": "huanyu_conformer_0_6b"}}}}))
        self.assertFalse(precision.is_huanyu_config({"model_config": {"model": "/Qwen3.6-huanyu"}}))


if __name__ == "__main__":
    unittest.main()
