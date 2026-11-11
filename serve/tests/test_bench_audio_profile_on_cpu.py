"""Real WAV bytes and fake HTTP responses: no tokenizer, NPU or training imports."""

import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_audio_profile_test", ROOT / "bench_audio_profile.py")
audio = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT))
try:
    spec.loader.exec_module(audio)
finally:
    sys.path.pop(0)


def wav_file(root, index, rate=16000, seconds=1):
    path = root / f"audio_{index}.wav"
    pcm = int(index).to_bytes(2, "little", signed=True) * int(rate * seconds)
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        f.writeframes(pcm)
    return path, pcm


def arguments(root, *extra):
    return audio.parse_args([
        "--audio-dir", str(root), "--batch-size", "2", "--output-tokens", "32",
        "--profile-arm-file", str(root / "armed"), "--report", str(root / "report.json"), *extra])


@pytest.mark.parametrize("wire_format", ["audio_url", "input_audio"])
def test_payload_carries_real_pcm_and_text_output_only(tmp_path, wire_format):
    path, pcm = wav_file(tmp_path, 7)
    item = audio.read_audio(path, 1, 30)
    payload = audio.audio_payload(arguments(tmp_path, "--audio-input-format", wire_format), item, 0)
    part = payload["messages"][1]["content"][0]
    if wire_format == "audio_url":
        prefix, data = part["audio_url"]["url"].split(",", 1)
        assert prefix == "data:audio/wav;base64"
    else:
        assert part["input_audio"]["format"] == "wav"
        data = part["input_audio"]["data"]
    with wave.open(io.BytesIO(base64.b64decode(data))) as f:
        assert f.getframerate() == 16000
        assert f.readframes(f.getnframes()) == pcm
    assert payload["n"] == 1
    assert payload["modalities"] == ["text"]
    assert payload["min_tokens"] == payload["max_tokens"] == 32
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_selection_deduplicates_pcm_and_does_not_reuse_for_warmup(tmp_path):
    first, _ = wav_file(tmp_path, 1)
    (tmp_path / "same_audio_other_name.wav").write_bytes(first.read_bytes())
    wav_file(tmp_path, 2, rate=8000)
    wav_file(tmp_path, 3, seconds=0.5)
    wav_file(tmp_path, 4)
    selected, _ = audio.select_audio(arguments(tmp_path), 2)
    assert len({x["pcm_sha256"] for x in selected}) == 2
    assert all(x["duration_s"] == 1 and x["sampling_rate"] == 16000 for x in selected)
    again, _ = audio.select_audio(arguments(tmp_path), 2)
    assert [x["path"] for x in selected] == [x["path"] for x in again]
    with pytest.raises(audio.bench.BenchmarkError, match="Need 3 distinct valid WAVs"):
        audio.select_audio(arguments(tmp_path), 3)


@pytest.mark.parametrize("failed_phase", [None, "probe", "warmup", "batches"])
def test_audio_run_only_arms_after_successful_warmup_and_uses_fresh_audio(tmp_path, monkeypatch, failed_phase):
    for i in range(1, 6):
        wav_file(tmp_path, i)
    args = arguments(tmp_path)
    sent = []

    def send(args, payload, request_id, expected_input_tokens):
        assert expected_input_tokens is None
        assert args.profile_arm_file.exists() == (request_id >= 3)
        phase = "probe" if request_id == 0 else "warmup" if request_id < 3 else "batches"
        sent.append(payload["messages"][1]["content"][0]["audio_url"]["url"])
        if phase == failed_phase:
            return {"request_id": request_id, "success": False, "error": "simulated failure"}
        meter = audio.bench.StreamMetrics(1, "none")
        # Multimodal prompt lengths may vary; never demand a fake 2048.
        meter.feed(json.dumps({"choices": [{"index": 0, "finish_reason": "length"}],
                              "usage": {"prompt_tokens": 140 + request_id,
                                        "completion_tokens": payload["max_tokens"]}}), 1)
        meter.feed("[DONE]", 1)
        return {"request_id": request_id, **meter.finish(1, None, payload["max_tokens"], True)}

    monkeypatch.setattr(audio.bench, "send_payload", send)
    assert audio.run(args) == (0 if failed_phase is None else 1)
    assert len(set(sent)) == len(sent)
    assert args.profile_arm_file.exists() == (failed_phase in (None, "batches"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["profiling_run"] is True
    assert "api_key" not in report["config"]
    assert "data:audio" not in args.report.read_text(encoding="utf-8")
    assert report["valid"] is (failed_phase is None)
    if failed_phase is None:
        assert report["summary"]["num_requests"] == 2
        assert report["summary"]["total_completion_tokens"] == 64
        assert report["summary"]["total_prompt_tokens"] == 143 + 144
        assert report["summary"]["output_tokens_per_s_per_device"] == pytest.approx(
            64 / report["summary"]["total_wall_clock_s"] / 4)
    elif failed_phase == "batches":
        assert report["summary"]["throughput_tokens_per_s"] is None
    else:
        assert report["summary"] is None
        assert report["batches"] == []


def test_missing_audio_or_existing_capture_fails_before_http(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("No network call should happen")
    monkeypatch.setattr(audio.bench, "send_payload", unexpected)
    args = arguments(tmp_path)
    with pytest.raises(audio.bench.BenchmarkError, match="distinct valid WAVs"):
        audio.run(args)
    assert not args.profile_arm_file.exists()
    args.profile_arm_file.touch()
    with pytest.raises(audio.bench.BenchmarkError, match="arm file already exists"):
        audio.run(args)


def test_audio_request_uses_shared_streaming_transport(tmp_path, monkeypatch):
    path, _ = wav_file(tmp_path, 1)
    item = audio.read_audio(path, 1, 30)
    wire = (b'data: {"choices":[{"index":0,"finish_reason":"length"}],'
            b'"usage":{"prompt_tokens":153,"completion_tokens":32}}\n\n'
            b'data: [DONE]\n\n')

    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def iter_content(self, chunk_size):
            assert chunk_size is None
            yield wire

    class Session:
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def post(self, url, **kwargs):
            assert self.trust_env is False
            assert kwargs["stream"] is True
            assert kwargs["json"]["n"] == 1
            return Response()

    monkeypatch.setattr(audio.bench.requests, "Session", Session)
    result = audio.send_audio(arguments(tmp_path), item, 0)
    assert result["success"] is True
    assert result["prompt_tokens"] == 153
    assert result["completion_tokens"] == 32
    assert result["choices"][0]["tpot_s"] is None


@pytest.mark.parametrize("flags", [("--n", "8"), ("--warmup-batches", "0"),
                                  ("--max-audio-seconds", "nan")])
def test_invalid_audio_profile_parameters_fail(tmp_path, flags):
    with pytest.raises(SystemExit):
        arguments(tmp_path, *flags)
