"""Reproduce native startup with Qwen normalization fields, without model IO."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("huanyu_processor_compat_tested", ROOT / "processor_compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class Tokenizer:
    def convert_tokens_to_ids(self, piece):
        pieces = ("<|audio_start|>", "<|audio_end|>", "<image>", "</image>",
                  "<slice>", "</slice>", "<|spk_bos|>", "<|spk_eos|>")
        return 100 + pieces.index(piece)


def native_info_class():
    class NativeInfo:
        # Native getter supplied from the failing server. Keep this independent
        # from the compatibility implementation so the old failure is exercised.
        def get_hf_processor(self, **kwargs: object):
            if self.ctx.tokenizer is None:
                from vllm.transformers_utils.processor import cached_processor_from_config

                tokenizer = self.get_tokenizer()
                hf_processor = cached_processor_from_config(
                    self.ctx.model_config, tokenizer=tokenizer, **kwargs,
                )
            else:
                hf_processor = self.ctx.get_hf_processor(**kwargs)
            tokenizer = getattr(hf_processor, "tokenizer", None)
            if tokenizer is not None and not hasattr(tokenizer, "audio_start"):
                for attr, piece in (
                    ("audio_start", "<|audio_start|>"), ("audio_end", "<|audio_end|>"),
                    ("im_start", "<image>"), ("im_end", "</image>"),
                    ("slice_start", "<slice>"), ("slice_end", "</slice>"),
                    ("spk_start", "<|spk_bos|>"), ("spk_end", "<|spk_eos|>"),
                ):
                    setattr(tokenizer, attr, piece)
                    setattr(tokenizer, attr + "_id", tokenizer.convert_tokens_to_ids(piece))
            image_processor = hf_processor.image_processor
            for attr in ("mean", "std"):
                val = getattr(image_processor, attr)
                if isinstance(val, np.ndarray):
                    setattr(image_processor, attr, val.tolist())
            config = self.get_hf_config()
            pool_step = getattr(config, "audio_pool_step", None)
            if pool_step is not None and hasattr(hf_processor, "pool_step"):
                hf_processor.pool_step = pool_step
            return hf_processor

    return NativeInfo


def qwen_processor():
    # Actual model preprocessor config from the server; no mean/std aliases.
    return SimpleNamespace(
        tokenizer=Tokenizer(), pool_step=2,
        image_processor=SimpleNamespace(
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
            patch_size=16, merge_size=2,
        ),
    )


class ProcessorCompatTests(unittest.TestCase):
    def setUp(self):
        self.cls = native_info_class()
        self.native = SimpleNamespace(MiniCPMO50OmniLLMProcessingInfo=self.cls)
        importer = patch.object(compat, "import_module", return_value=self.native)
        importer.start()
        self.addCleanup(importer.stop)

    def info(self, processor, *, lazy=False, pool_step=5):
        info = self.cls()
        info.ctx = SimpleNamespace(
            tokenizer=None if lazy else processor.tokenizer,
            model_config=object(), get_hf_processor=Mock(return_value=processor),
        )
        info.get_tokenizer = Mock(return_value=processor.tokenizer)
        info.get_hf_config = Mock(return_value=SimpleNamespace(audio_pool_step=pool_step))
        return info

    def install(self):
        with redirect_stdout(io.StringIO()):
            compat.install_huanyu_processor_compat()

    def test_reproduces_startup_error_then_preserves_qwen_processor_and_values(self):
        processor = qwen_processor()
        info = self.info(processor)
        with self.assertRaisesRegex(AttributeError, "mean"):
            info.get_hf_processor()
        info.ctx.get_hf_processor.reset_mock()
        self.install()
        result = info.get_hf_processor(chunk_input=False, custom=17)
        self.assertIs(result, processor)
        self.assertEqual(vars(result.image_processor), {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "patch_size": 16, "merge_size": 2,
        })
        info.ctx.get_hf_processor.assert_called_once_with(chunk_input=False, custom=17)
        info.get_tokenizer.assert_not_called()
        self.assertEqual(result.pool_step, 5)

    def test_worker_lazy_tokenizer_uses_cached_factory_and_forwards_kwargs(self):
        processor = qwen_processor()
        info = self.info(processor, lazy=True)
        module = ModuleType("vllm.transformers_utils.processor")
        module.cached_processor_from_config = Mock(return_value=processor)
        self.install()
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertIs(info.get_hf_processor(chunk_input=False), processor)
        module.cached_processor_from_config.assert_called_once_with(
            info.ctx.model_config, tokenizer=processor.tokenizer, chunk_input=False,
        )
        info.get_tokenizer.assert_called_once_with()
        info.ctx.get_hf_processor.assert_not_called()
        self.assertEqual(processor.tokenizer.audio_start, "<|audio_start|>")
        self.assertEqual(processor.tokenizer.audio_start_id, 100)
        self.assertEqual(processor.tokenizer.spk_end_id, 107)

    def test_legacy_minicpm_behavior_matches_native_getter(self):
        processor = qwen_processor()
        processor.image_processor = SimpleNamespace(
            mean=np.array([0.2, 0.3, 0.4]), std=np.array([0.6, 0.7, 0.8]),
        )
        baseline = self.info(deepcopy(processor)).get_hf_processor()
        self.install()
        actual = self.info(processor).get_hf_processor()
        self.assertEqual(vars(actual.image_processor), vars(baseline.image_processor))
        self.assertEqual(vars(actual.tokenizer), vars(baseline.tokenizer))
        self.assertEqual(actual.pool_step, baseline.pool_step)

    def test_numpy_fields_are_converted_without_replacing_configured_values(self):
        processor = qwen_processor()
        processor.image_processor.image_mean = np.array([0.1, 0.2, 0.3])
        processor.image_processor.image_std = np.array([0.7, 0.8, 0.9])
        # A processor exposing both conventions keeps each one's own value.
        processor.image_processor.mean = np.array([1.0, 2.0, 3.0])
        processor.image_processor.std = [4.0, 5.0, 6.0]
        original_std = processor.image_processor.std
        self.install()
        self.info(processor).get_hf_processor()
        self.assertEqual(processor.image_processor.image_mean, [0.1, 0.2, 0.3])
        self.assertEqual(processor.image_processor.image_std, [0.7, 0.8, 0.9])
        self.assertEqual(processor.image_processor.mean, [1.0, 2.0, 3.0])
        self.assertIs(processor.image_processor.std, original_std)

    def test_existing_tokenizer_fields_are_not_overwritten(self):
        processor = qwen_processor()
        processor.tokenizer.audio_start = "custom"
        processor.tokenizer.audio_start_id = 900
        self.install()
        self.info(processor).get_hf_processor()
        self.assertEqual(vars(processor.tokenizer), {"audio_start": "custom", "audio_start_id": 900})

    def test_optional_pool_setting_keeps_native_semantics(self):
        self.install()
        for configured, expected in ((None, 2), (0, 0), (5, 5)):
            with self.subTest(configured=configured):
                processor = qwen_processor()
                self.info(processor, pool_step=configured).get_hf_processor()
                self.assertEqual(processor.pool_step, expected)
        processor = qwen_processor()
        del processor.pool_step
        self.info(processor).get_hf_processor()
        self.assertFalse(hasattr(processor, "pool_step"))

    def test_repeated_install_and_cached_processor_calls_are_idempotent(self):
        original = self.cls.get_hf_processor
        self.install()
        installed = self.cls.get_hf_processor
        self.install()
        self.assertIs(self.cls.get_hf_processor, installed)
        self.assertIs(installed.__wrapped__, original)
        processor = qwen_processor()
        info = self.info(processor)
        self.assertIs(info.get_hf_processor(), info.get_hf_processor())
        self.assertEqual(info.ctx.get_hf_processor.call_count, 2)

    def test_factory_errors_are_not_swallowed_or_retried(self):
        self.install()
        info = self.info(qwen_processor())
        info.ctx.get_hf_processor.side_effect = ValueError("invalid processor config")
        with self.assertRaisesRegex(ValueError, "invalid processor config"):
            info.get_hf_processor()
        info.ctx.get_hf_processor.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
