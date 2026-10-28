"""The neutral probe records actual calls without changing tensors or gradients."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "examples/huanyu_audio_probe.py"
spec = importlib.util.spec_from_file_location("huanyu_audio_probe_cpu", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 3)
        self.register_buffer("inv_freq", torch.tensor([0.25, 0.5]), persistent=False)

    def forward(self, values, *, length):
        return self.proj(values), length


class Thinker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.apm = Encoder()
        self.audio_projection_layer = torch.nn.Linear(3, 4)
        self.calls = 0
        self.error = None

    def get_audio_hidden_states(self, data):
        self.calls += 1
        result, index = [], 0
        for group in data["audio_feature_lens"]:
            encoded = []
            for length in group:
                value, _ = self.apm(data["audio_features"][index][:, :int(length)].t(), length=length)
                if self.error:
                    raise self.error
                encoded.append(self.audio_projection_layer(value))
                index += 1
            result.append(torch.cat(encoded))
        self.last_result = result
        return result


def data():
    return {"audio_features": [torch.arange(12, dtype=torch.float32).reshape(2, 6),
                               torch.ones(2, 6), torch.full((2, 6), 2.0)],
            "audio_feature_lens": [torch.tensor([3, 4]), torch.tensor([5])]}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"HUANYU_AUDIO_PROBE_DIR": self.tmp.name,
                                          "HUANYU_AUDIO_PROBE_MAX_CALLS": "2",
                                          "HUANYU_AUDIO_PROBE_ARM_FILE": "",
                                          "HUANYU_AUDIO_PROBE_FEATURE_ID": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.native = SimpleNamespace(thinker=Thinker())

    def read(self, name="forward_0001"):
        file = next(Path(self.tmp.name).rglob(name + ".json"))
        return json.loads(file.read_text(encoding="utf-8")), file

    def test_disabled_does_not_install_or_read_parameters(self):
        with patch.dict(os.environ, {"HUANYU_AUDIO_PROBE_DIR": ""}):
            original = self.native.thinker.get_audio_hidden_states
            self.assertFalse(probe.attach_audio_probe(self.native))
            self.assertFalse(probe.install_model_probe())
            self.assertEqual(self.native.thinker.get_audio_hidden_states, original)
            self.assertFalse(list(Path(self.tmp.name).iterdir()))

    def test_real_capture_preserves_output_identity_gradients_and_bounds(self):
        thinker = self.native.thinker
        expected = thinker.get_audio_hidden_states(data())
        probe.attach_audio_probe(self.native)
        original_hook = thinker.get_audio_hidden_states
        self.assertTrue(probe.attach_audio_probe(self.native))
        self.assertEqual(thinker.get_audio_hidden_states, original_hook)
        actual = thinker.get_audio_hidden_states(data())
        self.assertIs(actual, thinker.last_result)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        sum(x.sum() for x in actual).backward()
        self.assertIsNotNone(thinker.apm.proj.weight.grad)
        thinker.get_audio_hidden_states(data())
        thinker.get_audio_hidden_states(data())
        self.assertEqual(len(list(Path(self.tmp.name).rglob("forward_*.json"))), 2)
        meta, file = self.read()
        self.assertEqual(meta["status"], "complete")
        self.assertEqual(meta["audio_groups"][0]["chunk_lengths"], [3, 4])
        self.assertEqual(meta["audio_groups"][1]["output_path"], "audio_embeddings/1")
        self.assertIn("inv_freq", meta["weights"]["encoder"]["buffers"])
        self.assertIn("encoder/0/input/kwargs/length", meta["tensors"])
        self.assertIn("encoder/0/output/0", meta["tensors"])
        self.assertIn("projector/2/output", meta["tensors"])
        self.assertEqual(meta["hook_counts"], {"encoder": 3, "projector": 3})
        with np.load(file.with_suffix(".npz"), allow_pickle=False) as saved:
            key = meta["tensors"]["audio_embeddings/0"]["array_key"]
            np.testing.assert_array_equal(saved[key], actual[0].detach().numpy())
        self.assertFalse(thinker.apm._forward_hooks)
        self.assertFalse(thinker.apm._forward_pre_hooks)

    def test_feature_identity_ignores_batch_order_and_padding_not_values(self):
        first = probe._Capture({})
        first.features(data())
        swapped = data()
        swapped["audio_features"] = [swapped["audio_features"][2], *swapped["audio_features"][:2]]
        swapped["audio_feature_lens"] = [torch.tensor([5]), torch.tensor([3, 4])]
        for chunk in swapped["audio_features"]:
            chunk[:, -1] = 999
        second = probe._Capture({})
        second.features(swapped)
        self.assertEqual(first.metadata["audio_groups"][0]["feature_id"],
                         second.metadata["audio_groups"][1]["feature_id"])
        self.assertEqual(first.metadata["audio_groups"][1]["feature_id"],
                         second.metadata["audio_groups"][0]["feature_id"])
        swapped["audio_features"][0][:, 0] = -100
        third = probe._Capture({})
        third.features(swapped)
        self.assertNotEqual(third.metadata["audio_groups"][0]["feature_id"],
                            second.metadata["audio_groups"][0]["feature_id"])

    def test_capture_hashes_current_weights_and_snapshots_include_buffers(self):
        probe.attach_audio_probe(self.native)
        self.native._huanyu_audio_probe_snapshot_weights("before_first_reload")
        self.native._huanyu_audio_probe_snapshot_weights("before_first_reload")
        with torch.no_grad():
            self.native.thinker.apm.proj.weight.add_(0.5)
            self.native.thinker.apm.inv_freq.add_(0.25)
        self.native._huanyu_audio_probe_reload_count = 1
        self.native._huanyu_audio_probe_snapshot_weights("after_first_reload")
        self.native.thinker.get_audio_hidden_states(data())
        before, _ = self.read("before_first_reload")
        after, _ = self.read("after_first_reload")
        forward, _ = self.read()
        self.assertNotEqual(before["weights"], after["weights"])
        self.assertEqual(after["weights"], forward["weights"])
        self.assertEqual(forward["weight_reload_count"], 1)
        self.assertEqual(len(list(Path(self.tmp.name).rglob("before_first_reload.json"))), 1)

    def test_weight_value_hash_distinguishes_dtype_from_value_changes(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.register_buffer("empty", torch.empty(0, 2))
        layer.register_buffer("scalar", torch.tensor(3))
        with torch.no_grad():
            layer.weight.fill_(0.5)
        fp32 = probe._weights(layer)["parameters"]["weight"]
        bf16 = probe._weights(layer.bfloat16())["parameters"]["weight"]
        self.assertNotEqual(fp32["sha256"], bf16["sha256"])
        self.assertEqual(fp32["values_sha256"], bf16["values_sha256"])

    def test_invalid_capture_configuration_fails_explicitly(self):
        with patch.dict(os.environ, {"HUANYU_AUDIO_PROBE_MAX_CALLS": "0"}):
            with self.assertRaisesRegex(ValueError, "positive"):
                probe.attach_audio_probe(self.native)
        with self.assertRaisesRegex(RuntimeError, "requires"):
            probe.attach_audio_probe(SimpleNamespace())
        probe.attach_audio_probe(self.native)
        invalid = data()
        invalid["audio_feature_lens"] = [torch.tensor([3, 4])]
        with self.assertRaisesRegex(ValueError, "chunks and lengths"):
            self.native.thinker.get_audio_hidden_states(invalid)
        self.assertEqual(self.native.thinker.calls, 0)
        self.assertFalse(list(Path(self.tmp.name).rglob("*.json")))

    def test_arm_and_feature_filter_skip_dummy_calls_without_consuming_limit(self):
        capture = probe._Capture({})
        capture.features(data())
        selected = capture.metadata["audio_groups"][0]["feature_id"]
        marker = Path(self.tmp.name) / "armed"
        with patch.dict(os.environ, {"HUANYU_AUDIO_PROBE_ARM_FILE": str(marker),
                                    "HUANYU_AUDIO_PROBE_FEATURE_ID": selected,
                                    "HUANYU_AUDIO_PROBE_MAX_CALLS": "1"}):
            probe.attach_audio_probe(self.native)
            # Snapshot fingerprints remain available before warmup completes.
            self.native._huanyu_audio_probe_snapshot_weights("before_first_reload")
            self.native.thinker.get_audio_hidden_states(data())
            marker.touch()
            unselected = data()
            for chunk in unselected["audio_features"]:
                chunk.add_(100)
            self.native.thinker.get_audio_hidden_states(unselected)
            self.assertFalse(list(Path(self.tmp.name).rglob("forward_*.json")))
            self.native.thinker.get_audio_hidden_states(data())
            self.native.thinker.get_audio_hidden_states(data())
        self.assertEqual(len(list(Path(self.tmp.name).rglob("forward_*.json"))), 1)
        meta, _ = self.read()
        self.assertEqual(meta["call_index"], 1)
        self.assertEqual(meta["audio_groups"][0]["feature_id"], selected)

    def test_double_precision_capture_preserves_values(self):
        capture = probe._Capture({})
        value = torch.tensor([1.0 + 1e-10], dtype=torch.float64)
        capture.record("output", value)
        array = capture.arrays[capture.metadata["tensors"]["output"]["array_key"]]
        self.assertEqual(array.dtype, np.dtype("float64"))
        np.testing.assert_array_equal(array, value.numpy())

    def test_native_failure_preserved_hooks_removed_no_success_file(self):
        error = RuntimeError("native audio computation failed")
        self.native.thinker.error = error
        probe.attach_audio_probe(self.native)
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as caught:
                self.native.thinker.get_audio_hidden_states(data())
        self.assertIs(caught.exception, error)
        self.assertIn("phase=failed", output.getvalue())
        self.assertNotIn("phase=executed", output.getvalue())
        self.assertFalse(list(Path(self.tmp.name).rglob("*.json")))
        self.assertFalse(self.native.thinker.apm._forward_hooks)
        self.assertFalse(self.native.thinker.apm._forward_pre_hooks)
        self.assertIsNone(probe._ACTIVE.get())

    def test_model_hook_only_attaches_to_huanyu_and_reads_no_initial_weights(self):
        class Native:
            def __init__(self, vllm_config):
                self.thinker = Thinker()
                self.vllm_config = vllm_config

        module = SimpleNamespace(MiniCPMO50OmniForConditionalGeneration=Native)
        with patch.object(probe.importlib, "import_module", return_value=module):
            self.assertTrue(probe.install_model_probe())
            hook = Native.__init__
            self.assertTrue(probe.install_model_probe())
            self.assertIs(hook, Native.__init__)
            other = Native({"model_config": {"hf_config": {"audio_config": {"model_type": "whisper"}}}})
            self.assertFalse(hasattr(other.thinker, "_huanyu_audio_probe_installed"))
            huanyu = Native({"model_config": {"hf_config": {"audio_config": {"model_type": "huanyu_conformer_0_6b"}}}})
            self.assertTrue(huanyu.thinker._huanyu_audio_probe_installed)
        self.assertFalse(list(Path(self.tmp.name).iterdir()))


if __name__ == "__main__":
    unittest.main()
