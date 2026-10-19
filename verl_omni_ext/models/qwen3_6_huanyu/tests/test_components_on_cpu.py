"""CPU boundary tests; all test artifacts stay in this Huanyu directory."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_local(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


weights = load_local("audio_weights")
inspect = load_local("inspect_checkpoint")
processor_loader = load_local("processor")


def tensor_shape(*shape):
    return SimpleNamespace(shape=shape)


class AudioWeightsTests(unittest.TestCase):
    def test_wrong_encoder_rejected_before_any_loading(self):
        target = Mock()
        target.state_dict.return_value = {"conv1.weight": tensor_shape(4, 3)}
        checkpoint = {"conv_layers.0.layers.0.weight": tensor_shape(4, 3)}
        with self.assertRaisesRegex(ValueError, "missing=1.*unexpected=1"):
            weights.load_audio_state_dict(target, checkpoint, component="encoder")
        target.load_state_dict.assert_not_called()

    def test_projection_shape_error_prevents_loading_even_matching_keys(self):
        target = Mock()
        target.state_dict.return_value = {"linear1.weight": tensor_shape(2048, 1024)}
        checkpoint = {"linear1.weight": tensor_shape(2048, 3072)}
        with self.assertRaisesRegex(ValueError, "shape_mismatches=1"):
            weights.load_audio_state_dict(target, checkpoint, component="projection")
        target.load_state_dict.assert_not_called()

    def test_matching_weights_use_strict_loading(self):
        target = Mock()
        state = {"linear1.weight": tensor_shape(4, 6)}
        target.state_dict.return_value = state
        weights.load_audio_state_dict(target, state, component="projection")
        target.load_state_dict.assert_called_once_with(state, strict=True)

    def test_empty_state_rejected(self):
        target = Mock()
        target.state_dict.return_value = {}
        with self.assertRaisesRegex(ValueError, "empty"):
            weights.load_audio_state_dict(target, {}, component="encoder")
        target.load_state_dict.assert_not_called()


class ProcessorTests(unittest.TestCase):
    def test_native_huanyu_auto_map_result_accepted(self):
        class HuanyuMiniCPMOProcessor:
            tokenizer = object()

            def __call__(self):
                pass

        processor = HuanyuMiniCPMOProcessor()
        auto = Mock()
        auto.from_pretrained.return_value = processor
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoProcessor=auto)}):
            result = processor_loader.load_huanyu_processor("model", trust_remote_code=True)
        self.assertIs(result, processor)
        auto.from_pretrained.assert_called_once_with("model", trust_remote_code=True)

    def test_loader_exception_retains_cause_instead_of_tokenizer_fallback(self):
        auto = Mock()
        cause = FileNotFoundError("fb128_mel_filters.pt")
        auto.from_pretrained.side_effect = cause
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoProcessor=auto)}):
            with self.assertRaisesRegex(RuntimeError, "fb128_mel_filters.pt") as ctx:
                processor_loader.load_huanyu_processor("model", trust_remote_code=True)
        self.assertIs(ctx.exception.__cause__, cause)

    def test_wrong_processor_not_silently_accepted(self):
        auto = Mock()
        auto.from_pretrained.return_value = SimpleNamespace(tokenizer=object())
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoProcessor=auto)}):
            with self.assertRaisesRegex(TypeError, "Expected checkpoint Huanyu"):
                processor_loader.load_huanyu_processor("model", trust_remote_code=True)


class InspectionTests(unittest.TestCase):
    def test_reads_headers_without_tensor_library_or_tensor_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"audio_config": {"model_type": "huanyu_conformer_0_6b"}}')
            header = {
                "model.audio_encoder.conv_layers.0.weight": {"shape": [2, 3], "dtype": "BF16"},
                "model.audio_projection.linear1.weight": {"shape": [2048, 3072], "dtype": "BF16"},
                "model.language_model.fake": {"shape": [1], "dtype": "BF16"},
            }
            raw = json.dumps(header).encode()
            (root / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
            report = inspect.inspect_checkpoint(root)
        self.assertEqual(report["audio_tensor_counts"], {
            "model.audio_encoder.": 1, "model.audio_projection.": 1,
        })
        self.assertEqual(report["projection_tensor_samples"][0]["shape"], [2048, 3072])
        self.assertIsNone(report["metadata"]["processor_config.json"])

    def test_truncated_header_fails_explicitly(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "broken.safetensors"
            path.write_bytes(b"oops")
            with self.assertRaisesRegex(ValueError, "Truncated"):
                inspect.read_tensor_headers(path)


if __name__ == "__main__":
    unittest.main()
