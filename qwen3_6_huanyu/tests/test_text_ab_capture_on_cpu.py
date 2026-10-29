"""Sample-keyed text capture with fake AgentLoop/tokenizer; no Ray or models."""

import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_text_capture_test", ROOT / "text_ab_capture.py")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.calls.append((list(ids), skip_special_tokens, clean_up_tokenization_spaces))
        values = {1: "<system>", 2: "原始系统提示", 3: "<audio>", 4: "回答", 5: "  保留空格", 6: "<eos>"}
        return "".join(values[i] for i in ids if not skip_special_tokens or i not in (1, 3, 6))


class TextCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HUANYU_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wav = self.root / "same.wav"
        self.wav.write_bytes(b"same exported WAV bytes")
        self.digest = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        self.prompt = [{"role": "system", "content": "原始系统提示"}, {"role": "user", "content": "<audio>"}]
        self.metadata = {"sample_id": "row_000123", "case_id": "case-test-42", "prompt_mode": "system_audio",
                         "prompt": deepcopy(self.prompt), "wav_paths": [str(self.wav)], "wav_sha256": [self.digest]}
        self.kwargs = {
            "raw_prompt": [{"role": "system", "content": [{"type": "text", "text": "原始系统提示"}]},
                           {"role": "user", "content": [{"type": "audio", "audio": str(self.wav)}]}],
            "extra_info": {"huanyu_text_ab": self.metadata}, "session_id": 0,
        }
        self.sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "logprobs": True}
        self.env = patch.dict(os.environ, {"HUANYU_TEXT_AB_DIR": str(self.root / "case")})
        self.env.start()
        self.addCleanup(self.env.stop)
        class Loop:
            def __init__(self):
                self.processor = SimpleNamespace(_huanyu_training_processor=True)
                self.tokenizer = Tokenizer()
                self.rollout_config = SimpleNamespace(val_kwargs=SimpleNamespace(n=1))
                self.config = SimpleNamespace(
                    trainer=SimpleNamespace(val_only=True, val_before_train=True, resume_mode="disable"),
                    actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path="/checkpoint/2748")),
                )
                self.calls = []
                self.result = SimpleNamespace(prompt_ids=[1, 2, 3], response_ids=[4, 5, 6])
                self.error = None

            async def run(self, sampling_params, priority=0, **kwargs):
                self.calls.append((sampling_params, priority, kwargs))
                await asyncio.sleep(0)
                if self.error:
                    raise self.error
                return self.result

        self.Loop = Loop
        self.original = Loop.run
        self.module = SimpleNamespace(SingleTurnAgentLoop=Loop)
        self.importer = patch.object(capture.importlib, "import_module", return_value=self.module)
        self.importer.start()
        self.addCleanup(self.importer.stop)

    def install(self):
        self.assertTrue(capture.install_text_ab_capture())
        return self.Loop()

    def read_record(self):
        return json.loads((self.root / "case/rollout/row_000123.json").read_text(encoding="utf-8"))

    def test_capture_matches_validation_text_without_mutating_request_or_result(self):
        loop = self.install()
        original_kwargs, original_sampling = deepcopy(self.kwargs), deepcopy(self.sampling)
        result = asyncio.run(loop.run(self.sampling, priority=9, **self.kwargs))
        self.assertIs(result, loop.result)
        self.assertIs(loop.calls[0][0], self.sampling)
        self.assertIs(loop.calls[0][2]["raw_prompt"], self.kwargs["raw_prompt"])
        self.assertEqual(loop.calls[0][1], 9)
        self.assertEqual(self.kwargs, original_kwargs)
        self.assertEqual(self.sampling, original_sampling)
        record = self.read_record()
        self.assertEqual(record["reply"], "回答  保留空格")
        self.assertEqual(record["reply_with_special_tokens"], "回答  保留空格<eos>")
        self.assertEqual(record["rendered_prompt_with_special_tokens"], "<system>原始系统提示<audio>")
        self.assertEqual(record["prompt"], self.prompt)
        self.assertEqual(record["requested_sampling_params"], self.sampling)
        self.assertEqual(record["sampling_scope"], "incoming_agent_loop_not_final_engine")
        self.assertEqual(record["representation"], "post_continuous_token_merge")
        self.assertEqual(record["response_token_count"], 3)
        self.assertEqual(record["prompt_token_count"], 3)
        self.assertEqual(record["wav_sha256"], [self.digest])
        self.assertEqual(record["status"], "complete")
        self.assertEqual((self.root / "case/rollout/row_000123.txt").read_text(encoding="utf-8"), record["reply"])
        self.assertTrue(all(not args[2] for args in loop.tokenizer.calls))

    def test_install_disabled_has_no_import_and_repeated_install_is_idempotent(self):
        with patch.dict(os.environ, {"HUANYU_TEXT_AB_DIR": ""}):
            self.assertFalse(capture.install_text_ab_capture())
            capture.importlib.import_module.assert_not_called()
        self.assertTrue(capture.install_text_ab_capture())
        installed = self.Loop.run
        self.assertFalse(capture.install_text_ab_capture())
        self.assertIs(self.Loop.run, installed)
        self.assertEqual(installed.__name__, self.original.__name__)
        self.assertEqual(installed.__module__, self.original.__module__)

    def test_unrelated_processor_and_disabled_runtime_delegate_without_metadata(self):
        loop = self.install()
        loop.processor = object()
        self.assertIs(asyncio.run(loop.run(self.sampling)), loop.result)
        loop.processor = SimpleNamespace(_huanyu_training_processor=True)
        with patch.dict(os.environ, {"HUANYU_TEXT_AB_DIR": ""}):
            self.assertIs(asyncio.run(loop.run(self.sampling)), loop.result)
        self.assertFalse((self.root / "case").exists())

    def test_reject_training_resume_or_multiple_validation_generations_before_request(self):
        loop = self.install()
        for field, value in (("val_only", False), ("val_before_train", False), ("resume_mode", "auto")):
            with self.subTest(field=field), patch.object(loop.config.trainer, field, value):
                with self.assertRaisesRegex(ValueError, "requires val_only"):
                    asyncio.run(loop.run(self.sampling, **self.kwargs))
        loop.rollout_config.val_kwargs.n = 4
        with self.assertRaisesRegex(ValueError, "val_kwargs.n=1"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(loop.calls, [])

    def test_per_sample_generation_override_cannot_publish_a_second_reply(self):
        loop = self.install()
        self.kwargs["session_id"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(loop.calls, [])

    def test_missing_identity_changed_prompt_or_wav_fail_before_inference(self):
        loop = self.install()
        cases = []
        missing = deepcopy(self.kwargs)
        missing["extra_info"] = {}
        cases.append(missing)
        changed = deepcopy(self.kwargs)
        changed["raw_prompt"][0]["content"][0]["text"] = "伪造系统提示"
        cases.append(changed)
        bad_id = deepcopy(self.kwargs)
        bad_id["extra_info"]["huanyu_text_ab"]["sample_id"] = "../escape"
        cases.append(bad_id)
        for kwargs in cases:
            with self.assertRaises(ValueError):
                asyncio.run(loop.run(self.sampling, **kwargs))
        self.wav.write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "WAV changed"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(loop.calls, [])

    def test_system_audio_drops_no_existing_system_and_rejects_user_text(self):
        loop = self.install()
        self.metadata["prompt"][1]["content"] = "描述问题<audio>"
        self.kwargs["raw_prompt"][1]["content"].insert(0, {"type": "text", "text": "描述问题"})
        with self.assertRaisesRegex(ValueError, "system_audio accepts"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.metadata["prompt_mode"] = "original"
        asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(self.read_record()["prompt"][1]["content"], "描述问题<audio>")

    def test_absent_system_is_not_synthesized(self):
        loop = self.install()
        self.metadata["prompt"] = self.metadata["prompt"][1:]
        self.kwargs["raw_prompt"] = self.kwargs["raw_prompt"][1:]
        asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(self.read_record()["prompt"], [{"role": "user", "content": "<audio>"}])

    def test_identical_audio_values_at_different_paths_do_not_hide_order_mismatch(self):
        loop = self.install()
        second = self.root / "second.wav"
        second.write_bytes(self.wav.read_bytes())
        self.kwargs["raw_prompt"][1]["content"][0]["audio"] = str(second)
        with self.assertRaisesRegex(ValueError, "paths/order differ"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(loop.calls, [])

    def test_duplicate_concurrent_sample_reservation_prevents_second_inference(self):
        loop = self.install()

        async def run_both():
            return await asyncio.gather(loop.run(self.sampling, **self.kwargs),
                                        loop.run(self.sampling, **self.kwargs), return_exceptions=True)

        results = asyncio.run(run_both())
        self.assertEqual(len(loop.calls), 1)
        self.assertEqual(sum(isinstance(x, FileExistsError) for x in results), 1)
        self.assertEqual(self.read_record()["status"], "complete")

    def test_failed_generation_stays_failed_and_is_not_silently_retried(self):
        loop = self.install()
        loop.error = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(self.read_record()["status"], "failed")
        self.assertEqual(self.read_record()["error"]["type"], "RuntimeError")
        with self.assertRaises(FileExistsError):
            asyncio.run(loop.run(self.sampling, **self.kwargs))
        self.assertEqual(len(loop.calls), 1)


if __name__ == "__main__":
    unittest.main()
