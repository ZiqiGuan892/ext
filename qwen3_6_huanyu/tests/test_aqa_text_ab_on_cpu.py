"""Shared AQA text cases: real tiny Parquet/WAVs and mocked HTTP replies."""

from contextlib import redirect_stderr, redirect_stdout
import asyncio
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "examples/aqa_text_ab.py"
spec = importlib.util.spec_from_file_location("aqa_text_ab_test", SCRIPT)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class TextCaseTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("HUANYU_TEST_TMPDIR", tempfile.gettempdir())).resolve()
        self.tmp = tempfile.TemporaryDirectory(prefix="aqa_text_ab_", dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "source.parquet"
        self.export = self.root / "export"
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"huanyu"}', encoding="utf-8")
        self.case = self.root / "case"

    def fixtures(self, count=5, audio_count=1):
        rows, manifest = [], []
        for index in range(count):
            sample = "row_{:06d}".format(index)
            directory = self.export / "samples" / sample
            directory.mkdir(parents=True)
            prompt = [{"role": "system", "content": " 系统原文\n只输出中文。\t"},
                      {"role": "user", "content": "文字问题不能发送" + "之间描述".join(["<audio>"] * audio_count)}]
            truth = "GROUND_TRUTH_SECRET_{}".format(index)
            references, audios = [], []
            for audio_index in range(audio_count):
                reference = "absent.zarr.zip::{}-{}".format(index, audio_index)
                references.append(reference)
                wav = directory / "audio_{:02d}.wav".format(audio_index)
                with wave.open(str(wav), "wb") as handle:
                    handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    handle.writeframes(bytes([index % 128, 0]) * (16 + audio_index))
                audios.append({"audio_index": audio_index, "wav_path": wav.relative_to(self.export).as_posix(),
                               "source_reference": reference, "channels": 1, "num_frames": 16 + audio_index,
                               "sample_rate": 16000})
            row = {"data_source": "omni_mix", "prompt": prompt, "audios": references, "images": [], "videos": [],
                   "reward_model": {"ground_truth": truth, "style": "model"},
                   "extra_info": {"index": index, "raw_source": "aqa_real", "other": ["keep"]}}
            rows.append(row)
            record = {"sample_id": sample, "row_index": index, "source_parquet": str(self.source),
                      "prompt": prompt, "ground_truth": truth, "audios": audios}
            manifest.append(record)
            ab.write_json(directory / "prompt.json", prompt)
            (directory / "ground_truth.txt").write_text(truth, encoding="utf-8")
        pq.write_table(pa.Table.from_pylist(rows), self.source)
        (self.export / "manifest.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest), encoding="utf-8")
        return rows

    def prepare(self, size=3, mode="system_audio", destination=None, embedding=False, logits=False, max_steps=0):
        args = SimpleNamespace(parquet=str(self.source), export_dir=str(self.export),
                               output_dir=str(destination or self.case), sample_size=size, seed=42,
                               prompt_mode=mode, model_path=str(self.model), enable_thinking="true", tp=4,
                               max_tokens=2048, embedding_check="true" if embedding else "false",
                               logits_check="true" if logits else "false", logits_max_steps=max_steps)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ab.prepare_case(args), 0)
        return ab.load_case(destination or self.case)

    def test_logits_case_covers_full_generation_by_default_and_can_limit_window(self):
        self.fixtures(count=3)
        _, case, _ = self.prepare(embedding=True, logits=True)
        self.assertEqual(case["logits_check"]["max_steps"], case["sampling"]["max_tokens"])
        _, limited, _ = self.prepare(embedding=True, logits=True, max_steps=128, destination=self.root / "limited")
        self.assertEqual(limited["logits_check"]["max_steps"], 128)

    def test_logits_without_embedding_identity_rejected_before_case_creation(self):
        self.fixtures(count=3)
        with self.assertRaisesRegex(ValueError, "embedding-check"):
            self.prepare(logits=True)
        self.assertFalse(self.case.exists())

    def server(self, root=None, **kwargs):
        args = SimpleNamespace(case_dir=str(root or self.case), url="http://localhost:28889/v1/chat/completions",
                               model="qwen3.8", timeout_s=600, skip_existing=False, continue_on_error=False)
        args.__dict__.update(kwargs)
        return args

    @staticmethod
    def response(reply="正确回答", reasoning=None, finish="stop"):
        return {"choices": [{"message": {"content": reply, "reasoning_content": reasoning},
                             "finish_reason": finish}], "usage": {"completion_tokens": 3}}

    def test_random100_preserves_system_strips_user_text_and_original_files(self):
        originals = self.fixtures(count=103, audio_count=2)
        source_before = self.source.read_bytes()
        root, case, rows = self.prepare(size=100)
        expected = sorted(random.Random(42).sample(range(103), 100))
        self.assertEqual([row["source_row_index"] for row in rows], expected)
        self.assertNotEqual(expected, list(range(100)))
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(case["engine"]["max_model_len"], 5248)
        self.assertEqual(case["sampling"]["seed"], 42)
        self.assertEqual(case["checkpoint_file_sha256"]["config.json"], ab.sha256(self.model / "config.json"))
        subset = pq.read_table(case["subset_parquet"]).to_pylist()
        for rewritten, row in zip(subset, rows):
            original = originals[row["source_row_index"]]
            self.assertEqual(row["prompt"][0], original["prompt"][0])
            self.assertEqual(row["prompt"][1], {"role": "user", "content": "<audio><audio>"})
            self.assertEqual(rewritten["prompt"], row["prompt"])
            identity = rewritten["extra_info"].pop("huanyu_text_ab")
            self.assertEqual(identity, dict({key: row[key] for key in ab.IDENTITY_KEYS},
                                           prompt_sha256=row["prompt_sha256"]))
            self.assertEqual(rewritten["extra_info"], original["extra_info"])
            self.assertEqual(rewritten["reward_model"], original["reward_model"])
            self.assertEqual(rewritten["audios"], row["wav_paths"])
            self.assertEqual([ab.sha256(p) for p in row["wav_paths"]], row["wav_sha256"])

    def test_prompt_original_mode_and_absent_system_are_not_rewritten(self):
        originals = self.fixtures()
        _, _, rows = self.prepare(mode="original")
        self.assertEqual(rows[0]["prompt"], originals[rows[0]["source_row_index"]]["prompt"])
        self.assertEqual(ab.canonical_prompt([{"role": "user", "content": "听<audio>问题"}], 1, "system_audio"),
                         [{"role": "user", "content": "<audio>"}])
        invalid = [
            [{"role": "system", "content": "<audio>"}],
            [{"role": "user", "content": "文字"}],
            [{"role": "assistant", "content": "<audio>"}],
            [{"role": "user", "content": "<audio>"}, {"role": "system", "content": "late"}],
            [{"role": "user", "content": "<audio><image>"}],
        ]
        for prompt in invalid:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                ab.canonical_prompt(prompt, 1, "system_audio")

    def test_http_body_has_same_prompt_explicit_sampling_and_no_ground_truth(self):
        self.fixtures()
        _, case, rows = self.prepare()
        client = ab.sibling("request_aqa_validation_wav")
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            post.return_value = self.response()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 0)
            self.assertEqual(post.call_count, len(rows))
            for call, row in zip(post.call_args_list, rows):
                body = json.loads(call.args[1])
                self.assertNotIn("GROUND_TRUTH", call.args[1].decode())
                self.assertNotIn("文字问题不能发送", call.args[1].decode())
                self.assertEqual(body["messages"][0], row["prompt"][0])
                self.assertEqual([p["type"] for p in body["messages"][1]["content"]], ["audio_url"])
                self.assertEqual({key: body[key] for key in case["sampling"]}, case["sampling"])
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
                result = ab.read_json(self.case / "serve" / (row["sample_id"] + ".json"))
                self.assertEqual(result["status"], "complete")
                self.assertNotIn("base64,", json.dumps(result["request"]))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server(skip_existing=True)), 0)
            self.assertEqual(post.call_count, len(rows))
            with self.assertRaises(FileExistsError):
                ab.serve_case(self.server())

    def test_reasoning_only_length_response_is_complete_and_preserved(self):
        self.fixtures()
        _, _, rows = self.prepare(size=1)
        client = ab.sibling("request_aqa_validation_wav")
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            post.return_value = self.response("", reasoning="思考到长度上限", finish="length")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 0)
        result = ab.read_json(self.case / "serve" / (rows[0]["sample_id"] + ".json"))
        self.assertTrue(result["truncated"])
        self.assertFalse(result["answer_text_present"])
        self.assertEqual(result["reasoning_content"], "思考到长度上限")

    def test_legacy_client_request_signature_does_not_control_case_messages(self):
        originals = self.fixtures(count=1, audio_count=2)
        _, case, rows = self.prepare(size=1)
        client = ab.sibling("request_aqa_validation_wav")
        # The old deployed helper had exactly these five arguments. A call with
        # include_prompt would reproduce the reported TypeError before HTTP.
        def legacy_build_request(prompt, payloads, model, max_tokens, temperature):
            raise AssertionError("case construction must not use the standalone helper")
        client.build_request = legacy_build_request
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            post.return_value = self.response()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 0)
            post.assert_called_once()
            body = json.loads(post.call_args.args[1])
        self.assertEqual(body["messages"][0], originals[0]["prompt"][0])
        self.assertEqual(body["modalities"], ["text"])
        self.assertEqual({key: body[key] for key in case["sampling"]}, case["sampling"])
        parts = body["messages"][1]["content"]
        self.assertEqual([part["type"] for part in parts], ["audio_url", "audio_url"])
        for part, path in zip(parts, rows[0]["wav_paths"]):
            data = part["audio_url"]["url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(data), Path(path).read_bytes())
        self.assertNotIn("文字问题不能发送", json.dumps(body, ensure_ascii=False))
        self.assertNotIn("GROUND_TRUTH", json.dumps(body))

    def test_request_serialization_stays_compatible_with_completed_cases(self):
        self.fixtures()
        _, case, _ = self.prepare(size=1)
        client = ab.sibling("request_aqa_validation_wav")
        source = [{"role": "system", "content": " 系统原文\n\t"},
                  {"role": "user", "content": "前<audio>中<audio>后"},
                  {"role": "user", "content": "另一个<audio>问题"}]
        payloads = [b"wav1", b"wav2", b"wav3"]
        for mode in ("system_audio", "original"):
            for with_system in (True, False):
                with self.subTest(mode=mode, with_system=with_system):
                    prompt = ab.canonical_prompt(source if with_system else source[1:], 3, mode)
                    before = copy.deepcopy(prompt)
                    expected = client.build_request(prompt, payloads, "qwen3.8", 2048, 0, include_prompt=True)
                    expected.update(case["sampling"])
                    expected["chat_template_kwargs"] = {"enable_thinking": True}
                    actual = ab.build_case_request(prompt, payloads, "qwen3.8", case["sampling"], True)
                    # Identical serialization retains request_sha256 and permits
                    # --skip-existing for already completed rows after upgrading.
                    self.assertEqual(json.dumps(actual, ensure_ascii=False), json.dumps(expected, ensure_ascii=False))
                    self.assertEqual(prompt, before)
                    self.assertEqual(payloads, [b"wav1", b"wav2", b"wav3"])

    def test_case_request_rejects_mismatched_audio_before_http(self):
        self.fixtures()
        _, case, _ = self.prepare(size=1)
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            ab.build_case_request([{"role": "user", "content": "<audio><audio>"}],
                                  [b"wav"], "qwen3.8", case["sampling"], True)

    def test_embedding_case_tags_http_request_without_adding_prompt_text(self):
        self.fixtures(count=1)
        _, case, rows = self.prepare(size=1, embedding=True)
        client = ab.sibling("request_aqa_validation_wav")
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            post.return_value = self.response()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 0)
        body = json.loads(post.call_args.args[1])
        tag = "huanyuab.{}.{}".format(case["case_id"], rows[0]["sample_id"])
        self.assertEqual(body["request_id"], tag)
        self.assertEqual(body["messages"][0], rows[0]["prompt"][0])
        self.assertEqual([part["type"] for part in body["messages"][1]["content"]], ["audio_url"])
        record = ab.read_json(self.case / "serve" / (rows[0]["sample_id"] + ".json"))
        self.assertEqual(record["embedding_request_id"], tag)

    def test_text_success_cannot_hide_missing_embedding_evidence(self):
        self.fixtures(count=1)
        _, _, rows = self.prepare(size=1, embedding=True)
        for side in ("serve", "rollout"):
            self.save_result(side, rows[0], "相同回答")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ab.compare_case(SimpleNamespace(case_dir=str(self.case))), 2)
        summary = ab.read_json(self.case / "summary.json")
        self.assertEqual(summary["identical_text_pairs"], 1)
        self.assertEqual(summary["embedding_check"]["status"], "INCOMPLETE")
        self.assertEqual(summary["embedding_check"]["incomplete_samples"], 1)
        records = [json.loads(line) for line in (self.case / "comparison.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["embedding_status"], "INCOMPLETE")
        self.assertIsNone(records[0]["embedding_strict_equal"])
        self.assertTrue((self.case / "embedding_summary.json").is_file())

    def test_prepare_cli_enables_strict_embedding_check_by_default(self):
        self.fixtures(count=1)
        with redirect_stdout(io.StringIO()):
            result = ab.main(["prepare", "--parquet", str(self.source), "--export-dir", str(self.export),
                              "--output-dir", str(self.case), "--model-path", str(self.model),
                              "--sample-size", "1"])
        self.assertEqual(result, 0)
        check = ab.read_json(self.case / "case.json")["embedding_check"]
        self.assertEqual(check, {"enabled": True, "scope": "llm_inputs_embeds", "strict": True,
                                "expected_tp_size": 4})

    def test_wav_changed_after_prepare_fails_before_request(self):
        self.fixtures()
        _, _, rows = self.prepare(size=1)
        path = Path(rows[0]["wav_paths"][0])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        client = ab.sibling("request_aqa_validation_wav")
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 1)
            post.assert_not_called()

    def save_result(self, side, row, reply, reasoning=None):
        directory = self.case / side
        directory.mkdir(exist_ok=True)
        result = {key: row[key] for key in ab.IDENTITY_KEYS}
        result.update(schema_version=1, status="complete", reply=reply, reasoning_content=reasoning,
                      response_token_count=3)
        ab.write_json(directory / (row["sample_id"] + ".json"), result)
        (directory / (row["sample_id"] + ".txt")).write_text(reply, encoding="utf-8")

    def test_compare_joins_ids_not_order_and_does_not_claim_accuracy(self):
        self.fixtures()
        _, _, rows = self.prepare(size=3)
        for i, row in enumerate(rows):
            self.save_result("serve", row, "回答{}".format(i), "单独思考" if i == 2 else None)
        for i, row in reversed(list(enumerate(rows))):
            self.save_result("rollout", row, "回答{}".format(i) if i != 1 else "另一种回答")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ab.compare_case(SimpleNamespace(case_dir=str(self.case))), 0)
        report = ab.read_json(self.case / "summary.json")
        self.assertEqual(report["complete_pairs"], 3)
        self.assertEqual(report["identical_text_pairs"], 1)
        self.assertEqual(report["different_text_pairs"], 1)
        self.assertEqual(report["representation_comparable_pairs"], 2)
        self.assertIn("not accuracy", report["note"])
        self.assertTrue((self.case / "comparison.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        records = [json.loads(x) for x in (self.case / "comparison.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIsNone(records[2]["reply_exact_match"])
        self.assertEqual(records[2]["serve_reasoning"], "单独思考")

    def test_missing_pairs_report_incomplete_and_foreign_identity_rejected(self):
        self.fixtures()
        _, _, rows = self.prepare(size=1)
        self.save_result("serve", rows[0], "文本")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ab.compare_case(SimpleNamespace(case_dir=str(self.case))), 2)
        self.assertEqual(ab.read_json(self.case / "summary.json")["incomplete_pairs"], 1)
        second = self.root / "second"
        _, _, second_rows = self.prepare(size=1, destination=second)
        self.case = second
        wrong = dict(second_rows[0], case_id="foreign-case")
        self.save_result("serve", wrong, "文本")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ab.compare_case(SimpleNamespace(case_dir=str(second)))
        self.assertFalse((second / "summary.json").exists())

    def test_prepare_and_manifest_do_not_overwrite_or_accept_modified_case(self):
        self.fixtures()
        self.prepare(size=1)
        with self.assertRaises(FileExistsError):
            self.prepare(size=1)
        manifest = self.case / "manifest.jsonl"
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            ab.load_case(self.case)

    def test_prepared_parquet_to_actual_capture_wrapper_http_and_comparison(self):
        self.fixtures(count=6, audio_count=2)
        _, case, records = self.prepare(size=3)
        capture_path = SCRIPT.parent.parent / "text_ab_capture.py"
        capture_spec = importlib.util.spec_from_file_location("capture_roundtrip", capture_path)
        capture = importlib.util.module_from_spec(capture_spec)
        capture_spec.loader.exec_module(capture)

        class Loop:
            processor = SimpleNamespace(_huanyu_training_processor=True)
            rollout_config = SimpleNamespace(val_kwargs=SimpleNamespace(n=1))
            config = SimpleNamespace(
                trainer=SimpleNamespace(val_only=True, val_before_train=True, resume_mode="disable"),
                actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path=case["model_path"])),
            )
            tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "正确回答" if ids == [3] else "<prompt>")

            async def run(self, sampling_params, **kwargs):
                return SimpleNamespace(prompt_ids=[1, 2], response_ids=[3])

        with patch.dict(os.environ, {"HUANYU_TEXT_AB_DIR": str(self.case)}):
            with patch.object(capture.importlib, "import_module", return_value=SimpleNamespace(SingleTurnAgentLoop=Loop)):
                self.assertTrue(capture.install_text_ab_capture())
            # These values are read back from the actual prepared Parquet, not
            # recreated from the manifest. Expand <audio> like RLHFDataset.
            for row in reversed(pq.read_table(case["subset_parquet"]).to_pylist()):
                raw, audio_index = [], 0
                for message in row["prompt"]:
                    parts = []
                    pieces = message["content"].split("<audio>")
                    for index, text in enumerate(pieces):
                        parts.append({"type": "text", "text": text})
                        if index < len(pieces) - 1:
                            parts.append({"type": "audio", "audio": row["audios"][audio_index]})
                            audio_index += 1
                    raw.append({"role": message["role"], "content": parts})
                arguments = {"raw_prompt": raw, "extra_info": row["extra_info"], "session_id": 0}
                before = copy.deepcopy(arguments)
                with redirect_stdout(io.StringIO()):
                    asyncio.run(Loop().run(dict(case["sampling"]), **arguments))
                self.assertEqual(arguments, before)

        client = ab.sibling("request_aqa_validation_wav")
        with patch.object(ab, "sibling", return_value=client), patch.object(client, "post_json") as post:
            post.return_value = self.response("正确回答")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ab.serve_case(self.server()), 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ab.compare_case(SimpleNamespace(case_dir=str(self.case))), 0)
        summary = ab.read_json(self.case / "summary.json")
        self.assertEqual(summary["identical_text_pairs"], len(records))
        self.assertEqual(summary["complete_pairs"], len(records))


if __name__ == "__main__":
    unittest.main()
