"""Huanyu rollout item/chunk contracts, without Ray, vLLM or model weights."""

import asyncio
from collections import UserDict
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import inspect
from io import StringIO
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


_PATH = Path(__file__).resolve().parents[1] / "rollout_inputs.py"
_PACKAGE = ModuleType("huanyu_rollout_inputs_tested")
_PACKAGE.__path__ = [str(_PATH.parent)]
sys.modules[_PACKAGE.__name__] = _PACKAGE
_SPEC = importlib.util.spec_from_file_location(f"{_PACKAGE.__name__}.rollout_inputs", _PATH)
rollout = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rollout
_SPEC.loader.exec_module(rollout)


class Waveform:
    ndim = 1

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"(<image>./</image>)": [100, 101], "(<audio>./</audio>)": [200, 201]}[text]


class Processor:
    _huanyu_training_processor = True
    vision_start_token_id, vision_end_token_id, image_token_id = 10, 11, 12
    audio_start_token_id, audio_end_token_id, audio_placeholder_token_id = 20, 21, 22
    sampling_rate, n_fft, hop_length = 16000, 400, 160
    audio_chunk_seconds, audio_encoder_downsample, audio_projector_downsample = 30, 4, 2
    tokenizer = Tokenizer()

    def dedup_pad_tokens(self, ids):
        if self.audio_start_token_id in ids or self.vision_start_token_id in ids:
            raise RuntimeError("metadata contraction was not performed")
        return ids


def audio_span(length):
    return [20] + [22] * length + [21]


class RolloutInputsTest(unittest.TestCase):
    def test_long_audio_is_one_item_and_adjacent_second_audio_stays_separate(self):
        # A 31s waveform has 375+12 tokens; an adjacent 1s waveform has 12.
        # Their source item boundary cannot be inferred from token adjacency.
        ids = [7] + audio_span(375) + audio_span(12) + audio_span(12) + [8]
        original = list(ids)
        actual = rollout.contract_rollout_prompt(
            Processor(), ids, {"audio": [Waveform(31 * 16000), Waveform(16000)]}
        )
        self.assertEqual(actual, [7, 200, 201, 200, 201, 8])
        self.assertEqual(ids, original)

    def test_interleaved_images_text_and_audio_preserve_order(self):
        ids = [10, 12, 12, 11, 7] + audio_span(12) + [8, 10, 12, 11]
        actual = rollout.contract_rollout_prompt(
            Processor(), ids, {"image": [object(), object()], "audio": [(Waveform(16000), 16000)]}
        )
        self.assertEqual(actual, [100, 101, 7, 200, 201, 8, 100, 101])

    def test_audio_counts_lengths_and_missing_short_tail_span_fail(self):
        for ids, media in (
            (audio_span(11), {"audio": [Waveform(16000)]}),
            (audio_span(12), {"audio": [Waveform(31 * 16000)]}),
            (audio_span(12) * 2, {"audio": [Waveform(16000)]}),
            ([7], {"audio": [Waveform(16000)]}),
            (audio_span(375), {"audio": [Waveform(30 * 16000 + 100)]}),
            (audio_span(12), {"audio": [(Waveform(16000), 8000)]}),
        ):
            with self.subTest(ids_count=len(ids), media=media), self.assertRaises(ValueError):
                rollout.contract_rollout_prompt(Processor(), ids, media)

    def test_short_tails_keep_one_original_audio_item_and_have_real_token_spans(self):
        for whole_chunks in (0, 1, 2):
            for tail in (1, 6, 58, 749, 833, 1039):
                with self.subTest(whole_chunks=whole_chunks, tail=tail):
                    waveform = Waveform(whole_chunks * 480000 + tail)
                    spans = audio_span(375) * whole_chunks + audio_span(1)
                    actual = rollout.contract_rollout_prompt(Processor(), spans, {"audio": [waveform]})
                    self.assertEqual(actual, [200, 201])

    def test_video_server_installs_native_hooks_before_dispatch_and_only_once(self):
        class VideoProcessor(Processor):
            video_token_id = 13
            spatial_merge_size = 2
            tokenizer = SimpleNamespace(encode=lambda text, **kw: {
                "(<image>./</image>)": [100, 101], "(<audio>./</audio>)": [200, 201],
                "(<video>./</video>)": [300, 301],
            }[text])

            def _video(self, videos):
                return {"video_grid_thw": [np.array([2, 2, 2])]}

        events = []

        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                events.append("dispatch")
                return prompt_ids, multi_modal_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=VideoProcessor())
        server._ar_mode = True
        module = ModuleType(_PACKAGE.__name__ + ".video_rollout")
        module.install_huanyu_video_rollout = Mock(side_effect=lambda: events.append("install"))
        media = {"video": [object()]}
        with patch.dict(sys.modules, {module.__name__: module}):
            for _ in range(2):
                ids, forwarded = server._preprocess_input([10, 13, 11] * 2, media)
                self.assertEqual(ids, [300, 301])
                self.assertIs(forwarded, media)
        self.assertEqual(events, ["install", "dispatch", "dispatch"])
        module.install_huanyu_video_rollout.assert_called_once_with()
        # A failed installation must never dispatch to an unpatched server.
        server._huanyu_video_hooks_ready = False
        module.install_huanyu_video_rollout.side_effect = RuntimeError("unsupported native hooks")
        with patch.dict(sys.modules, {module.__name__: module}), self.assertRaisesRegex(RuntimeError, "native hooks"):
            server._preprocess_input([10, 13, 11] * 2, media)
        self.assertEqual(events, ["install", "dispatch", "dispatch"])
        self.assertFalse(server._huanyu_video_hooks_ready)

    def test_server_forwards_padded_short_tail_without_mutating_shared_requests(self):
        class Server:
            def _preprocess_input(self, prompt_ids, sampling_params, multi_modal_data):
                return prompt_ids, sampling_params, multi_modal_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        source = np.arange(960006, dtype=np.float32)
        source_before = source.copy()
        rate = np.int64(16000)
        item = (source, rate)
        untouched = np.zeros(16000, dtype=np.float32)
        audios = [item, untouched]
        images = [object()]
        media = UserDict(audio=audios, image=images, request_metadata=object())
        ids = [10, 12, 11] + audio_span(375) * 2 + audio_span(1) + audio_span(12)
        params = {"max_tokens": 17}
        result = server._preprocess_input(ids, params, media)
        forwarded = result[2]
        self.assertEqual(result[0], [100, 101, 200, 201, 200, 201])
        self.assertIs(result[1], params)
        self.assertIsNot(forwarded, media)
        self.assertIsNot(forwarded["audio"], audios)
        self.assertIs(forwarded["image"], images)
        self.assertIs(forwarded["request_metadata"], media["request_metadata"])
        self.assertIs(forwarded["audio"][0][1], rate)
        self.assertIs(forwarded["audio"][1], untouched)
        padded = forwarded["audio"][0][0]
        self.assertEqual(padded.shape, (961520,))
        np.testing.assert_array_equal(padded[:len(source)], source_before)
        np.testing.assert_array_equal(padded[len(source):], 0)
        self.assertEqual(rollout._audio_chunk_lengths(forwarded["audio"][0], Processor()), [375, 375, 1])
        self.assertIs(media["audio"], audios)
        self.assertIs(audios[0], item)
        np.testing.assert_array_equal(source, source_before)
        # A second generation can reuse the already-prepared request unchanged.
        repeated = server._preprocess_input(ids, params, forwarded)
        self.assertIs(repeated[2], forwarded)

    def test_async_server_forwards_short_audio_preserving_tuple_container_and_rate(self):
        class Server:
            async def generate(self, prompt_ids, image_data=None, audio_data=None, video_data=None):
                return prompt_ids, image_data, audio_data, video_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        source = np.ones(833, dtype=np.float32)
        audios = ((source, 16000),)
        images = [object()]
        result = asyncio.run(server.generate([10, 12, 11] + audio_span(1), images, audios))
        self.assertEqual(result[0], [100, 101, 200, 201])
        self.assertIs(result[1], images)
        self.assertIsInstance(result[2], tuple)
        self.assertIsNot(result[2], audios)
        self.assertEqual(result[2][0][1], 16000)
        padded = result[2][0][0]
        self.assertEqual(padded.shape, (1520,))
        np.testing.assert_array_equal(padded[:833], source)
        np.testing.assert_array_equal(padded[833:], 0)
        self.assertIs(audios[0][0], source)
        self.assertEqual(len(source), 833)
        self.assertIsNone(result[3])

    def test_existing_positive_token_tails_keep_waveform_and_container_identity(self):
        for tail in (1040, 1520, 16000):
            with self.subTest(tail=tail):
                waveform = np.zeros(480000 + tail, dtype=np.float32)
                item = (waveform, 16000)
                media = {"audio": [item]}
                self.assertIs(rollout._prepare_audio_media(Processor(), media), media)
                self.assertIs(media["audio"][0], item)
                self.assertIs(item[0], waveform)

    def test_invalid_short_audio_is_not_hidden_by_zero_padding(self):
        for waveform in (np.array([], dtype=np.float32), np.array([np.nan]), np.array([np.inf]),
                         np.array([1j]), np.ones((2, 3))):
            with self.subTest(waveform=waveform), self.assertRaises((TypeError, ValueError)):
                rollout._prepare_audio_media(Processor(), {"audio": [waveform]})

    def test_malformed_or_missing_image_markers_fail(self):
        for ids, media in (
            ([10, 11], {"image": [object()]}),
            ([10, 12, 99, 11], {"image": [object()]}),
            ([10, 12, 11], {}),
            ([11], {}),
            ([12], {}),
            ([7], {"image": [object()]}),
            ([100, 101, 10, 12, 11], {"image": [object()]}),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rollout.contract_rollout_prompt(Processor(), ids, media)

    def test_text_only_unchanged_and_video_requires_frame_conversion(self):
        self.assertEqual(rollout.contract_rollout_prompt(Processor(), [7, 22, 8], {}), [7, 22, 8])
        with self.assertRaises(ValueError):
            rollout.contract_rollout_prompt(Processor(), [7], {"video": [object()]})
        with self.assertRaises(TypeError):
            rollout.contract_rollout_prompt(Processor(), [True], {})

    def test_gated_server_wrapper_preserves_other_models_and_is_idempotent(self):
        class Server:
            def _preprocess_input(self, prompt_ids, sampling_params, multi_modal_data, *, sentinel=None):
                processor = self.model_config.processor
                if getattr(processor, "_huanyu_training_processor", False):
                    prompt_ids = processor.dedup_pad_tokens(prompt_ids)
                return prompt_ids, sampling_params, multi_modal_data, sentinel

        rollout._install_on_server_class(Server)
        installed = Server._preprocess_input
        rollout._install_on_server_class(Server)
        self.assertIs(Server._preprocess_input, installed)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        params = {"max_tokens": 17}
        media = {"audio": [Waveform(16000)]}
        result = server._preprocess_input(
            prompt_ids=audio_span(12), sampling_params=params, multi_modal_data=media, sentinel="same"
        )
        self.assertEqual(result[0], [200, 201])
        self.assertIs(result[1], params)
        self.assertIs(result[2], media)
        self.assertEqual(result[3], "same")
        server.model_config = SimpleNamespace(processor=object())
        result = server._preprocess_input("opaque_other_model_ids", params, None)
        self.assertEqual(result[0], "opaque_other_model_ids")
        self.assertIsNone(result[2])

    def test_incompatible_server_signature_rejected(self):
        class Server:
            def _preprocess_input(self, unrelated):
                return unrelated

        with self.assertRaises(RuntimeError):
            rollout._install_on_server_class(Server)

    def test_config_ar_without_private_flag_contracts_media_for_mapping_and_attribute_configs(self):
        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                return self.model_config.processor.dedup_pad_tokens(prompt_ids), multi_modal_data

        rollout._install_on_server_class(Server)
        # Exercise Mapping/attribute lookup independently at every config level.
        for mask in range(8):
            config = {"output_mode": "ar"}
            if mask & 1:
                config = SimpleNamespace(**config)
            config = {"vllm_omni": config}
            if mask & 2:
                config = SimpleNamespace(**config)
            config = {"engine_kwargs": config}
            config = SimpleNamespace(**config) if mask & 4 else UserDict(config)
            with self.subTest(attribute_levels=mask):
                server = Server()
                server.model_config = SimpleNamespace(processor=Processor())
                server.config = config
                before_config = deepcopy(config)
                ids = [7, 10, 12, 12, 11] + audio_span(375) + audio_span(12) + [8]
                before_ids = list(ids)
                media = {"image": [object()], "audio": [Waveform(31 * 16000)]}
                result = server._preprocess_input(ids, media)
                self.assertEqual(result[0], [7, 100, 101, 200, 201, 8])
                self.assertIs(result[1], media)
                self.assertEqual(ids, before_ids)
                self.assertIs(server.config, config)
                self.assertEqual(config, before_config)
                self.assertFalse(hasattr(server, "_ar_mode"))

    def test_missing_or_unconfirmed_mode_rejected_before_native_preprocessing(self):
        calls = []

        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                calls.append(prompt_ids)

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        with self.assertRaisesRegex(RuntimeError, "missing") as caught:
            server._preprocess_input([7], {})
        self.assertIn("_ar_mode=<missing>", str(caught.exception))
        self.assertIn("output_mode=<missing>", str(caught.exception))
        for config in (
            None, {}, {"engine_kwargs": None}, {"engine_kwargs": {}},
            {"engine_kwargs": {"vllm_omni": None}},
            *({"engine_kwargs": {"vllm_omni": {"output_mode": mode}}}
              for mode in (None, "", "unknown", "diffusion")),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(RuntimeError, "autoregressive"):
                server.config = config
                server._preprocess_input([7], {})
        self.assertEqual(calls, [])
        self.assertFalse(hasattr(server, "_ar_mode"))

    def test_explicit_private_mode_and_config_conflicts_cannot_override_each_other(self):
        calls = []

        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                calls.append(prompt_ids)

        rollout._install_on_server_class(Server)
        for flag, mode in ((False, "ar"), (True, "diffusion"), (True, "unknown"),
                           (None, "ar"), (0, "ar"), ("True", "ar")):
            with self.subTest(private_flag=flag, output_mode=mode):
                server = Server()
                server.model_config = SimpleNamespace(processor=Processor())
                server._ar_mode = flag
                server.config = {"engine_kwargs": {"vllm_omni": {"output_mode": mode}}}
                before_config = deepcopy(server.config)
                with self.assertRaisesRegex(RuntimeError, "autoregressive"):
                    server._preprocess_input([7], {})
                self.assertIs(server._ar_mode, flag)
                self.assertEqual(server.config, before_config)
        self.assertEqual(calls, [])

    def test_async_generate_accepts_config_ar_without_private_flag(self):
        class Server:
            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                return self.model_config.processor.dedup_pad_tokens(prompt_ids), image_data, audio_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server.config = SimpleNamespace(engine_kwargs={"vllm_omni": {"output_mode": "ar"}})
        audio = [Waveform(16000)]
        result = asyncio.run(server.generate(audio_span(12), audio_data=audio))
        self.assertEqual(result[0], [200, 201])
        self.assertIsNone(result[1])
        self.assertIs(result[2], audio)
        self.assertFalse(hasattr(server, "_ar_mode"))

    def test_non_huanyu_preprocessing_ignores_missing_and_conflicting_mode_evidence(self):
        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                return prompt_ids, multi_modal_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=object())
        ids, media = object(), object()
        for config in ({}, {"engine_kwargs": {"vllm_omni": {"output_mode": "diffusion"}}}):
            with self.subTest(config=config):
                server.config = config
                result = server._preprocess_input(ids, media)
                self.assertIs(result[0], ids)
                self.assertIs(result[1], media)
                self.assertFalse(hasattr(server, "_ar_mode"))

    def test_async_generate_contracts_long_audio_and_image_without_mutating_inputs(self):
        class Server:
            async def generate(self, prompt_ids, sampling_params, request_id,
                               image_data=None, video_data=None, audio_data=None, *, priority=0):
                await asyncio.sleep(0)
                self.model_config.processor.dedup_pad_tokens(prompt_ids)
                return prompt_ids, sampling_params, request_id, image_data, video_data, audio_data, priority

        original_generate = Server.generate
        rollout._install_on_server_class(Server)
        installed = Server.generate
        self.assertEqual(installed.__module__, rollout.__name__)
        self.assertIn("<locals>", installed.__qualname__)
        self.assertEqual(inspect.signature(installed), inspect.signature(original_generate))
        rollout._install_on_server_class(Server)
        self.assertIs(Server.generate, installed)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        ids = [7, 10, 12, 12, 11] + audio_span(375) + audio_span(12) + [8]
        before = list(ids)
        params = {"max_tokens": 19}
        images = [object()]
        audio = [Waveform(31 * 16000)]
        request_id = object()
        result = asyncio.run(server.generate(
            ids, params, request_id, image_data=images, audio_data=audio, priority=3
        ))
        self.assertEqual(result[0], [7, 100, 101, 200, 201, 8])
        self.assertEqual(ids, before)
        self.assertIsNot(result[0], ids)
        self.assertIs(result[1], params)
        self.assertIs(result[2], request_id)
        self.assertIs(result[3], images)
        self.assertIsNone(result[4])
        self.assertIs(result[5], audio)
        self.assertEqual(result[6], 3)
        self.assertEqual(params, {"max_tokens": 19})
        self.assertEqual(len(audio[0]), 31 * 16000)

    def test_async_generate_without_video_parameter_handles_text_only(self):
        class Server:
            async def generate(self, prompt_ids, sampling_params, *, image_data=None, audio_data=None):
                return prompt_ids, sampling_params, image_data, audio_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        ids = [7, 22, 8]
        params = object()
        result = asyncio.run(server.generate(prompt_ids=ids, sampling_params=params))
        self.assertEqual(result[0], ids)
        self.assertIs(result[1], params)
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_async_generate_non_huanyu_rm_is_delegated_unchanged(self):
        class Server:
            async def generate(self, prompt_ids, sampling_params, image_data=None, audio_data=None):
                return prompt_ids, sampling_params, image_data, audio_data

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=object())
        server._ar_mode = False  # Other models must not run the Huanyu mode/metadata guards.
        ids, params, images, audios = (object() for _ in range(4))
        result = asyncio.run(server.generate(ids, params, images, audios))
        for actual, expected in zip(result, (ids, params, images, audios)):
            self.assertIs(actual, expected)

    def test_async_generate_requires_ar_mode_and_rejects_bad_media_before_call(self):
        calls = []

        class Server:
            async def generate(self, prompt_ids, image_data=None, audio_data=None, video_data=None):
                calls.append(prompt_ids)

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = False
        with self.assertRaisesRegex(RuntimeError, "autoregressive"):
            asyncio.run(server.generate([7]))
        server._ar_mode = True
        with self.assertRaises(ValueError):
            asyncio.run(server.generate(audio_span(11), audio_data=[Waveform(16000)]))
        with self.assertRaises(ValueError):
            asyncio.run(server.generate([7], video_data=[object()]))
        self.assertEqual(calls, [])

    def test_async_generate_preserves_original_exception(self):
        original_error = RuntimeError("native generation failed")

        class Server:
            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                await asyncio.sleep(0)
                raise original_error

        rollout._install_on_server_class(Server)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(server.generate([7]))
        self.assertIs(caught.exception, original_error)

    def test_ray_actor_wrapper_patches_only_modified_class(self):
        class ModifiedServer:
            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                return prompt_ids

        wrapper_generate = object()
        actor_export = SimpleNamespace(
            __ray_metadata__=SimpleNamespace(modified_class=ModifiedServer),
            generate=wrapper_generate,
        )
        original_generate = ModifiedServer.generate
        rollout._install_on_server_class(actor_export)
        installed = ModifiedServer.generate
        self.assertIsNot(installed, original_generate)
        self.assertIs(actor_export.generate, wrapper_generate)
        rollout._install_on_server_class(actor_export)
        self.assertIs(ModifiedServer.generate, installed)
        server = ModifiedServer()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        self.assertEqual(asyncio.run(server.generate(
            [10, 12, 11], image_data=[object()]
        )), [100, 101])

    def test_preprocess_is_preferred_over_generate_to_avoid_double_contraction(self):
        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                return self.model_config.processor.dedup_pad_tokens(prompt_ids)

            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                return self._preprocess_input(prompt_ids, {"image": image_data, "audio": audio_data})

        original_generate = Server.generate
        rollout._install_on_server_class(Server)
        self.assertIs(Server.generate, original_generate)
        server = Server()
        server.model_config = SimpleNamespace(processor=Processor())
        server._ar_mode = True
        result = asyncio.run(server.generate(
            audio_span(375) + audio_span(12), audio_data=[Waveform(31 * 16000)]
        ))
        self.assertEqual(result, [200, 201])

    def test_unsupported_server_api_is_rejected_explicitly(self):
        class NoEntry:
            pass

        class SyncGenerate:
            def generate(self, prompt_ids, image_data=None, audio_data=None):
                return prompt_ids

        class MissingAudio:
            async def generate(self, prompt_ids, image_data=None):
                return prompt_ids

        class AsyncPreprocess:
            async def _preprocess_input(self, prompt_ids, multi_modal_data):
                return prompt_ids

        class InvalidPreprocessWithValidGenerate:
            def _preprocess_input(self, unrelated):
                return unrelated

            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                return prompt_ids

        class OpaqueGenerate:
            async def generate(self, *args, **kwargs):
                return args

        candidates = (NoEntry, SyncGenerate, MissingAudio, AsyncPreprocess,
                      InvalidPreprocessWithValidGenerate, OpaqueGenerate,
                      SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=object())))
        for candidate in candidates:
            with self.subTest(server=candidate), self.assertRaises(RuntimeError):
                rollout._install_on_server_class(candidate)


class SamplingControlsTest(unittest.TestCase):
    @staticmethod
    def server(config, *, asynchronous=False, huanyu=True):
        class Server:
            def _preprocess_input(self, prompt_ids, sampling_params, multi_modal_data):
                self.calls += 1
                return prompt_ids, sampling_params

        class AsyncServer:
            async def generate(self, prompt_ids, sampling_params, image_data=None, audio_data=None):
                self.calls += 1
                return prompt_ids, sampling_params

        cls = AsyncServer if asynchronous else Server
        rollout._install_on_server_class(cls)
        server = cls()
        server.calls = 0
        server.model_config = SimpleNamespace(processor=Processor() if huanyu else object())
        server._ar_mode = True
        server.config = config
        return server

    @staticmethod
    def invoke(server, params, *, asynchronous=False):
        if asynchronous:
            return asyncio.run(server.generate(audio_span(12), params, audio_data=[Waveform(16000)]))
        return server._preprocess_input(audio_span(12), params, {"audio": [Waveform(16000)]})

    def test_length_controls_reach_both_server_apis_without_mutating_requests(self):
        for asynchronous in (False, True):
            # The serialized config may expose mapping or dataclass attributes.
            config = {"ignore_eos": True, "response_length": 10240,
                      "engine_kwargs": {"q36_sampling": {"min_tokens": 10240}}}
            if asynchronous:
                config = SimpleNamespace(**config)
            with self.subTest(asynchronous=asynchronous):
                server = self.server(config, asynchronous=asynchronous)
                params = {"temperature": 1.0, "max_tokens": 10240}
                before = deepcopy(params)
                with redirect_stdout(StringIO()) as logs:
                    ids, actual = self.invoke(server, params, asynchronous=asynchronous)
                    self.invoke(server, params, asynchronous=asynchronous)
                self.assertEqual(ids, [200, 201])
                self.assertEqual(actual, {**before, "ignore_eos": True, "min_tokens": 10240})
                self.assertIsNot(actual, params)
                self.assertEqual(params, before)
                self.assertEqual(logs.getvalue().count("[huanyu-sampling]"), 1)
                self.assertIn("ignore_eos=True min_tokens=10240 response_length=10240", logs.getvalue())
                self.assertEqual(server.calls, 2)

    def test_request_controls_override_server_defaults(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                server = self.server(
                    {"ignore_eos": True, "engine_kwargs": {"q36_sampling": {"min_tokens": 10240}}},
                    asynchronous=asynchronous,
                )
                params = {"ignore_eos": False, "min_tokens": 0, "max_tokens": 8}
                with redirect_stdout(StringIO()):
                    _, actual = self.invoke(server, params, asynchronous=asynchronous)
                self.assertEqual(actual, params)
                self.assertIsNot(actual, params)

    def test_unconfigured_or_false_defaults_preserve_parameter_identity(self):
        for config in (None, {}, {"ignore_eos": False}, {"engine_kwargs": {"q36_sampling": {}}}):
            for asynchronous in (False, True):
                with self.subTest(config=config, asynchronous=asynchronous):
                    server = self.server(config, asynchronous=asynchronous)
                    params = {"max_tokens": 31}
                    with redirect_stdout(StringIO()) as logs:
                        _, actual = self.invoke(server, params, asynchronous=asynchronous)
                    self.assertIs(actual, params)
                    self.assertNotIn("[huanyu-sampling]", logs.getvalue())

    def test_min_tokens_can_be_selected_without_enabling_ignore_eos(self):
        config = UserDict(engine_kwargs=SimpleNamespace(q36_sampling=UserDict(min_tokens=0)))
        server = self.server(config)
        params = {"max_tokens": 17}
        with redirect_stdout(StringIO()):
            _, actual = self.invoke(server, params)
        self.assertEqual(actual, {"max_tokens": 17, "min_tokens": 0})
        self.assertNotIn("ignore_eos", actual)

    def test_explicit_seed_reaches_both_apis_and_preserves_request_override(self):
        for asynchronous in (False, True):
            server = self.server({"engine_kwargs": {"q36_sampling": {"seed": 42}}},
                                 asynchronous=asynchronous)
            for params, expected in (({"temperature": 0}, 42), ({"seed": 7}, 7)):
                before = dict(params)
                with redirect_stdout(StringIO()):
                    _, actual = self.invoke(server, params, asynchronous=asynchronous)
                self.assertEqual(actual["seed"], expected)
                self.assertEqual(params, before)

    def test_other_model_servers_ignore_even_invalid_huanyu_controls(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                server = self.server(
                    {"ignore_eos": "bad", "engine_kwargs": {"q36_sampling": {"min_tokens": -1}}},
                    asynchronous=asynchronous, huanyu=False,
                )
                params = {"max_tokens": 1024}
                with redirect_stdout(StringIO()) as logs:
                    ids, actual = self.invoke(server, params, asynchronous=asynchronous)
                self.assertEqual(ids, audio_span(12))
                self.assertIs(actual, params)
                self.assertNotIn("[huanyu-sampling]", logs.getvalue())

    def test_invalid_config_values_fail_before_the_native_server_call(self):
        configs = [{"ignore_eos": value} for value in ("true", 1, None)]
        configs += [{"engine_kwargs": {"q36_sampling": {"min_tokens": value}}}
                    for value in (-1, 1.5, "10240", True, None)]
        configs += [{"engine_kwargs": {"q36_sampling": {"seed": value}}}
                    for value in (-1, 2**63, 1.5, "42", True, None)]
        for config in configs:
            for asynchronous in (False, True):
                with self.subTest(config=config, asynchronous=asynchronous):
                    server = self.server(config, asynchronous=asynchronous)
                    with self.assertRaises(ValueError):
                        self.invoke(server, {}, asynchronous=asynchronous)
                    self.assertEqual(server.calls, 0)

    def test_configured_controls_require_a_mapping_request(self):
        for asynchronous in (False, True):
            for params in (None, object(), []):
                with self.subTest(asynchronous=asynchronous, params=params):
                    server = self.server({"ignore_eos": True}, asynchronous=asynchronous)
                    with self.assertRaisesRegex(RuntimeError, "sampling_params mapping"):
                        self.invoke(server, params, asynchronous=asynchronous)
                    self.assertEqual(server.calls, 0)

    def test_configured_controls_reject_an_api_without_sampling_params(self):
        class Server:
            def _preprocess_input(self, prompt_ids, multi_modal_data):
                raise AssertionError("must reject before native preprocessing")

        class AsyncServer:
            async def generate(self, prompt_ids, image_data=None, audio_data=None):
                raise AssertionError("must reject before native generation")

        for cls in (Server, AsyncServer):
            with self.subTest(server=cls.__name__):
                rollout._install_on_server_class(cls)
                server = cls()
                server.model_config = SimpleNamespace(processor=Processor())
                server._ar_mode = True
                server.config = {"ignore_eos": True}
                with self.assertRaisesRegex(RuntimeError, "sampling_params mapping"):
                    if cls is Server:
                        server._preprocess_input([7], {})
                    else:
                        asyncio.run(server.generate([7]))


if __name__ == "__main__":
    unittest.main()
