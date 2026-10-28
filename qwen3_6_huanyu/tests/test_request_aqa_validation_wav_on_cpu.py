"""Exercise the standalone audio HTTP client without vLLM, NPU or a server."""

import base64
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import wave


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "request_aqa_validation_wav.py"
_spec = importlib.util.spec_from_file_location("aqa_wav_request", SCRIPT)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


class Reply:
    status = 200

    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class AqaWavRequestTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("HUANYU_TEST_TMPDIR", tempfile.gettempdir())).resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="aqa_http_", dir=base)
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, base)
        self.addCleanup(self.temporary.cleanup)
        (self.root / "samples").mkdir()
        self.rows = []
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"VLLM_API_KEY": "test-only-key"}).start()
        self.urlopen = patch.object(client.request, "urlopen").start()
        self.urlopen.return_value = self.answer()
        self.args = ["--export-dir", str(self.root)]

    @staticmethod
    def answer(content="有鸟鸣声。", finish="stop"):
        return Reply({"id": "test-response", "model": "qwen3.8", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": content,
                                    "reasoning_content": "声音有周期性的高音。"},
            "finish_reason": finish,
        }], "usage": {"prompt_tokens": 123, "completion_tokens": 10, "total_tokens": 133}})

    def add_row(self, index=0, audio_count=1):
        row_id = "row_{:06d}".format(index)
        folder = self.root / "samples" / row_id
        folder.mkdir()
        prompt = [{"role": "system", "content": "根据音频回答问题。"},
                  {"role": "user", "content": "请听：" + "，然后".join(["<audio>"] * audio_count) + "是什么声音？"}]
        (folder / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False), encoding="utf-8")
        (folder / "ground_truth.txt").write_text("秘密参考答案_不应发给模型", encoding="utf-8")
        audios = []
        for i in range(audio_count):
            wav_path = folder / "audio_{:02d}.wav".format(i)
            with wave.open(str(wav_path), "wb") as wav:
                wav.setparams((1, 2, 16000 + i * 8000, 0, "NONE", "not compressed"))
                wav.writeframes(bytes([i, 0]) * 16)
            audios.append({"audio_index": i, "wav_path": wav_path.relative_to(self.root).as_posix()})
        row = {"row_index": index, "sample_id": row_id, "audios": audios}
        self.rows.append(row)
        (self.root / "manifest.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in self.rows), encoding="utf-8")
        return folder

    def run_client(self, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return client.main(self.args + list(extra))

    def result(self, index=0, tag="response"):
        return json.loads((self.root / "samples" / "row_{:06d}".format(index) / (tag + ".json")).read_text(encoding="utf-8"))

    def test_explicit_prompt_mode_and_outputs_inside_row(self):
        folder = self.add_row(audio_count=2)
        self.assertEqual(self.run_client("--include-prompt", "--max-tokens", "4096", "--temperature", "0.3", "--timeout-s", "77"), 0)
        req = self.urlopen.call_args.args[0]
        sent = json.loads(req.data)
        self.assertEqual(req.full_url, client.DEFAULT_URL)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-only-key")
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], 77)
        self.assertNotIn("秘密参考答案", req.data.decode("utf-8"))
        self.assertEqual(sent["model"], "qwen3.8")
        self.assertEqual(sent["max_tokens"], 4096)
        self.assertEqual(sent["temperature"], 0.3)
        self.assertEqual(sent["modalities"], ["text"])
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["messages"][0]["content"], "根据音频回答问题。")
        parts = sent["messages"][1]["content"]
        self.assertEqual([p["text"] for p in parts if p["type"] == "text"], ["请听：", "，然后", "是什么声音？"])
        blobs = [p["audio_url"]["url"] for p in parts if p["type"] == "audio_url"]
        self.assertEqual(len(blobs), 2)
        for i, blob in enumerate(blobs):
            prefix, encoded = blob.split(",", 1)
            self.assertEqual(prefix, "data:audio/wav;base64")
            self.assertEqual(base64.b64decode(encoded), (folder / "audio_{:02d}.wav".format(i)).read_bytes())
        result = self.result()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ground_truth"], "秘密参考答案_不应发给模型")
        self.assertEqual(result["reply"], "有鸟鸣声。")
        self.assertEqual(result["usage"]["prompt_tokens"], 123)
        self.assertEqual(result["raw_response"]["id"], "test-response")
        self.assertNotIn("test-only-key", json.dumps(result))
        self.assertEqual((folder / "response.txt").read_text(encoding="utf-8"), "有鸟鸣声。")
        self.assertFalse((folder.parent / "row_000000.response.txt").exists())
        self.assertFalse((folder.parent / "row_000000.response.json").exists())

    def test_default_request_is_audio_only_and_keeps_original_prompt_locally(self):
        folder = self.add_row(audio_count=2)
        prompt = [
            {"role": "system", "content": "SYSTEM_ONLY_LOCAL"},
            {"role": "user", "content": "QUESTION_ONLY_LOCAL"},
            {"role": "assistant", "content": "ASSISTANT_ONLY_LOCAL"},
        ]
        # Audio-only mode does not need source <audio> text placeholders.
        (folder / "prompt.json").write_text(json.dumps(prompt), encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(client.main(self.args), 0)
        sent = json.loads(self.urlopen.call_args.args[0].data)
        self.assertEqual([m["role"] for m in sent["messages"]], ["user"])
        parts = sent["messages"][0]["content"]
        self.assertEqual([p["type"] for p in parts], ["audio_url", "audio_url"])
        for i, part in enumerate(parts):
            encoded = part["audio_url"]["url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), (folder / "audio_{:02d}.wav".format(i)).read_bytes())
        self.assertNotIn("ONLY_LOCAL", json.dumps(sent))
        self.assertNotIn("秘密参考答案", json.dumps(sent, ensure_ascii=False))
        self.assertEqual(sent["modalities"], ["text"])
        self.assertEqual(self.result()["prompt"], prompt)
        self.assertEqual(self.result()["input_mode"], "audio_only")
        self.assertFalse(self.result()["prompt_sent"])
        self.assertIn("input_mode=audio_only prompt_sent=false modalities=text", stdout.getvalue())

    def test_resume_does_not_mix_audio_only_and_prompt_results(self):
        folder = self.add_row()
        self.assertEqual(self.run_client("--include-prompt"), 0)
        before = (folder / "response.json").read_bytes()
        self.assertEqual(self.run_client("--skip-existing"), 1)
        self.assertEqual((folder / "response.json").read_bytes(), before)
        self.assertEqual(self.run_client("--skip-existing", "--include-prompt"), 0)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_input_audio_compatibility_and_format_aware_resume(self):
        folder = self.add_row()
        self.assertEqual(self.run_client("--audio-input-format", "input_audio"), 0)
        sent = json.loads(self.urlopen.call_args.args[0].data)
        self.assertEqual([m["role"] for m in sent["messages"]], ["user"])
        parts = sent["messages"][0]["content"]
        self.assertTrue(all(p["type"] == "input_audio" for p in parts))
        blobs = [p["input_audio"] for p in parts if p["type"] == "input_audio"]
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0]["format"], "wav")
        self.assertEqual(base64.b64decode(blobs[0]["data"]), (folder / "audio_00.wav").read_bytes())
        self.assertEqual(sent["modalities"], ["text"])
        self.assertEqual(self.result()["audio_input_format"], "input_audio")
        self.assertEqual(self.run_client("--skip-existing"), 1)
        self.assertEqual(self.run_client("--skip-existing", "--audio-input-format", "input_audio"), 0)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_template_error_explains_server_fix_without_retrying_or_dropping_audio(self):
        self.add_row()
        body = b'{"error":{"message":"Unexpected item type in content. Unexpected item type in content."}}'
        self.urlopen.side_effect = HTTPError(client.DEFAULT_URL, 400, "Bad Request", {}, io.BytesIO(body))
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            self.assertEqual(client.main(self.args), 1)
        result = self.result()
        self.assertEqual(result["error"]["response_body"], body.decode())
        self.assertIn("--chat-template-content-format string", result["error"]["hint"])
        self.assertIn("--interleave-mm-strings", stderr.getvalue())
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertTrue(any(p["type"] == "audio_url" for p in json.loads(
            self.urlopen.call_args.args[0].data)["messages"][0]["content"]))
        self.assertIsNone(client.server_error_hint(client.ServerError("bad", 400, "unrelated error")))

    def test_resume_verifies_wav_bytes_and_does_not_overwrite(self):
        folder = self.add_row()
        self.assertEqual(self.run_client(), 0)
        before = (folder / "response.json").read_bytes()
        self.assertEqual(self.run_client(), 1)
        self.assertEqual(self.run_client("--skip-existing"), 0)
        self.assertEqual(self.urlopen.call_count, 1)
        wav_path = folder / "audio_00.wav"
        payload = wav_path.read_bytes()
        wav_path.write_bytes(payload[:-2] + b"\x01\x00")
        self.assertEqual(self.run_client("--skip-existing"), 1)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual((folder / "response.json").read_bytes(), before)

    def test_http_failure_preserves_body_and_clears_old_reply_on_overwrite(self):
        folder = self.add_row()
        self.assertEqual(self.run_client(), 0)
        self.urlopen.side_effect = HTTPError(client.DEFAULT_URL, 400, "Bad Request", {}, io.BytesIO(b'{"error":"audio unsupported"}'))
        self.assertEqual(self.run_client("--overwrite"), 1)
        result = self.result()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["http_status"], 400)
        self.assertIn("audio unsupported", result["error"]["response_body"])
        self.assertFalse((folder / "response.txt").exists())

    def test_empty_answer_keeps_reasoning_and_original_response(self):
        self.add_row()
        self.urlopen.return_value = self.answer(None, "length")
        self.assertEqual(self.run_client(), 1)
        result = self.result()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reply"], "")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["reasoning_content"], "声音有周期性的高音。")
        self.assertIn("raw_response", result)

    def test_content_parts_and_truncation(self):
        self.add_row()
        self.urlopen.return_value = self.answer([{"type": "text", "text": "鸟"}, {"type": "text", "text": "鸣"}], "length")
        self.assertEqual(self.run_client(), 0)
        self.assertEqual(self.result()["reply"], "鸟鸣")
        self.assertTrue(self.result()["truncated"])

    def test_invalid_inputs_fail_before_http(self):
        folder = self.add_row()
        prompt_path = folder / "prompt.json"
        original = prompt_path.read_bytes()
        prompt_path.write_text('[{"role":"user","content":"missing placeholder"}]', encoding="utf-8")
        self.assertEqual(self.run_client("--include-prompt"), 1)
        self.assertIn("count", self.result()["error"]["message"])
        prompt_path.write_bytes(original)
        wav_path = folder / "audio_00.wav"
        wav_path.write_bytes(wav_path.read_bytes()[:-4])
        self.assertEqual(self.run_client("--overwrite"), 1)
        self.assertIn("PCM16", self.result()["error"]["message"])
        self.urlopen.assert_not_called()

    def test_row_selection_tags_and_continue_after_timeout(self):
        self.add_row(0)
        self.add_row(1)
        self.add_row(2)
        self.urlopen.side_effect = [TimeoutError("socket timed out"), self.answer()]
        self.assertEqual(self.run_client("--start-row", "1", "--limit", "2", "--result-tag", "modelB", "--continue-on-error"), 1)
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertFalse((self.root / "samples" / "row_000000" / "modelB.json").exists())
        self.assertEqual(self.result(1, "modelB")["status"], "failed")
        self.assertEqual(self.result(2, "modelB")["status"], "complete")

    def test_result_tags_cannot_overwrite_exported_inputs(self):
        folder = self.add_row()
        for name in ("sample.json", "request.json"):
            (folder / name).write_text('{}', encoding="utf-8")
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        for tag in ("prompt", "ground_truth", "sample", "request"):
            self.assertEqual(self.run_client("--result-tag", tag, "--overwrite"), 1)
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)
        self.urlopen.assert_not_called()

    def test_non_json_http_success_is_reported(self):
        self.add_row()
        self.urlopen.return_value = Reply(b"<html>wrong endpoint</html>")
        self.assertEqual(self.run_client(), 1)
        self.assertEqual(self.result()["error"]["response_body"], "<html>wrong endpoint</html>")

    def test_missing_choices_and_api_error_keep_raw_response(self):
        self.add_row()
        for response in ({"choices": []}, {"error": {"message": "server error"}}):
            self.urlopen.return_value = Reply(response)
            self.assertEqual(self.run_client("--overwrite"), 1)
            self.assertEqual(self.result()["raw_response"], response)

    def test_global_validation_and_path_escape(self):
        self.add_row()
        for extra in (("--url", "ws://localhost:2862/v1/realtime?duplex=1"),
                      ("--result-tag", "../escape"), ("--timeout-s", "nan")):
            self.assertEqual(self.run_client(*extra), 1)
        self.rows[0]["audios"][0]["wav_path"] = "manifest.jsonl"
        (self.root / "manifest.jsonl").write_text(json.dumps(self.rows[0]) + "\n", encoding="utf-8")
        self.assertEqual(self.run_client(), 1)
        self.assertIn("ValueError", self.result()["error"]["type"])
        self.urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
