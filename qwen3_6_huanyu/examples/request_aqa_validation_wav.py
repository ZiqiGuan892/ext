#!/usr/bin/env python3
"""Send exported AQA WAVs to vLLM for text replies (audio-only input by default).

Standard library only (Python 3.8+). Results are saved inside samples/row_XXXXXX/.
This uses HTTP question answering, not the realtime duplex voice protocol.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, parse, request
import wave


DEFAULT_URL = "http://localhost:28889/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_rows(root, start_row, limit):
    rows = []
    seen = set()
    with (root / "manifest.jsonl").open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("row_index")
            sample_id = row.get("sample_id")
            if (type(index) is not int or index < 0 or sample_id != "row_{:06d}".format(index)
                    or sample_id in seen):
                raise ValueError("invalid/duplicate row identity at manifest line {}".format(line_number))
            seen.add(sample_id)
            if index >= start_row:
                rows.append(row)
                if limit and len(rows) == limit:
                    break
    if not rows:
        raise ValueError("no exported rows selected; check --export-dir and --start-row")
    return rows


def load_inputs(root, row):
    sample_dir = (root / "samples" / row["sample_id"]).resolve(strict=True)
    sample_dir.relative_to(root)
    prompt = read_json(sample_dir / "prompt.json")
    truth = (sample_dir / "ground_truth.txt").read_text(encoding="utf-8")
    audios = row.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError("manifest row has no audios")
    paths, payloads = [], []
    for index, audio in enumerate(audios):
        if audio.get("audio_index") != index:
            raise ValueError("audios must be ordered by audio_index starting at 0")
        relative = Path(audio["wav_path"])
        if relative.is_absolute():
            raise ValueError("manifest wav_path must be relative to --export-dir")
        path = (root / relative).resolve(strict=True)
        path.relative_to(sample_dir)
        payload = path.read_bytes()
        # Exporter writes PCM16 WAVs. Reject mislabeled/compressed or truncated
        # files before a request; preserve the exported sample rate/channels.
        with wave.open(io.BytesIO(payload), "rb") as wav:
            frames = wav.getnframes()
            frame_bytes = wav.getnchannels() * wav.getsampwidth()
            if (wav.getsampwidth() != 2 or not frames or wav.getcomptype() != "NONE"
                    or len(wav.readframes(frames)) != frames * frame_bytes):
                raise ValueError("expected a non-empty, complete PCM16 WAV: {}".format(path))
        paths.append(str(path))
        payloads.append(payload)
    return prompt, truth, paths, payloads


def build_request(prompt, payloads, model, max_tokens, temperature, audio_input_format="audio_url",
                  include_prompt=False):
    # Standalone client: do not import the Parquet exporter and its dependencies.
    if audio_input_format not in ("audio_url", "input_audio"):
        raise ValueError("unsupported audio input format: {}".format(audio_input_format))
    if not payloads:
        raise ValueError("at least one WAV is required")
    audio_parts = []
    for payload in payloads:
        encoded = base64.b64encode(payload).decode("ascii")
        if audio_input_format == "audio_url":
            audio_parts.append({"type": "audio_url", "audio_url": {
                "url": "data:audio/wav;base64," + encoded,
            }})
        else:
            audio_parts.append({"type": "input_audio", "input_audio": {
                "data": encoded, "format": "wav",
            }})
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, "n": 1, "modalities": ["text"]}
    if not include_prompt:
        # Do not copy any source message, role, text part or metadata. The WAV
        # order comes from the manifest; prompt.json is local reference only.
        body["messages"] = [{"role": "user", "content": audio_parts}]
        return body
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt.json must contain a non-empty message list")
    for message in prompt:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)):
            raise ValueError("each prompt message must contain role and string content")
    count = sum(message["content"].count("<audio>") for message in prompt)
    if count != len(payloads):
        raise ValueError("<audio> count={} differs from WAV count={}".format(count, len(payloads)))
    messages = []
    audio_index = 0
    for message in prompt:
        result = dict(message)
        pieces = message["content"].split("<audio>")
        if len(pieces) > 1:
            if message["role"] != "user":
                raise ValueError("audio placeholders must belong to user messages")
            parts = []
            for index, text in enumerate(pieces):
                if text:
                    parts.append({"type": "text", "text": text})
                if index < len(pieces) - 1:
                    parts.append(audio_parts[audio_index])
                    audio_index += 1
            result["content"] = parts
        messages.append(result)
    body["messages"] = messages
    return body


class ServerError(RuntimeError):
    def __init__(self, message, status, body):
        super().__init__(message)
        self.status = status
        self.body = body


def server_error_hint(exc):
    if exc.status == 400 and "Unexpected item type in content" in exc.body:
        return (
            "The server chat template rejected a content part. For the Huanyu checkpoint with a "
            "Qwen image/video/text-only template, restart vllm serve with "
            "--chat-template-content-format string --interleave-mm-strings. "
            "Changing input_audio to audio_url alone does not fix this template error: "
            "vLLM can normalize both to an audio part. Keep the WAV attached; do not "
            "replace it with a text path. See examples/README.md."
        )
    return None


def post_json(url, payload, timeout_s, api_key):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ServerError("HTTP {}: {}".format(exc.code, body[:500]), exc.code, body) from exc
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ServerError("server returned non-JSON content", status, body) from exc


def extract_reply(response):
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("server returned an error or invalid response object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices; check the Chat Completions endpoint")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response choices[0] has no message")
    content = message.get("content")
    if content is None:
        content = ""
    if isinstance(content, list):
        pieces = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("text", "output_text"):
                raise ValueError("response contains a non-text content part; inspect raw_response")
            if not isinstance(part.get("text"), str):
                raise ValueError("response text part is not a string")
            pieces.append(part["text"])
        content = "".join(pieces)
    if not isinstance(content, str):
        raise ValueError("response message.content is not text")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    return content, reasoning, choices[0].get("finish_reason")


def run(args):
    parsed = parse.urlsplit(args.url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.path.rstrip("/") != "/v1/chat/completions" or parsed.query or parsed.fragment):
        raise ValueError("--url must be an HTTP(S) /v1/chat/completions endpoint, not a duplex WebSocket URL")
    if not args.model.strip():
        raise ValueError("--model must match the server's served model name")
    if (args.limit < 0 or args.start_row < 0 or args.max_tokens < 1
            or not math.isfinite(args.timeout_s) or args.timeout_s <= 0
            or not math.isfinite(args.temperature) or args.temperature < 0):
        raise ValueError("invalid limit/start-row/max-tokens/timeout-s/temperature")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.result_tag):
        raise ValueError("--result-tag may only contain letters, digits, _ and -")
    if args.result_tag in ("prompt", "ground_truth", "sample", "request"):
        raise ValueError("--result-tag is reserved for exported input files: {}".format(args.result_tag))
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip-existing cannot be combined")
    root = Path(args.export_dir).expanduser().resolve(strict=True)
    samples = (root / "samples").resolve(strict=True)
    samples.relative_to(root)
    rows = select_rows(root, args.start_row, args.limit)
    # Check collisions for all selected rows before paying for inference.
    output_paths = {}
    for row in rows:
        sample_dir = (samples / row["sample_id"]).resolve(strict=True)
        sample_dir.relative_to(samples)
        paths = (sample_dir / (args.result_tag + ".json"), sample_dir / (args.result_tag + ".txt"))
        for path in paths:
            path.resolve().relative_to(sample_dir)
            if path.exists() and not (args.overwrite or args.skip_existing):
                raise FileExistsError("{} exists; use --skip-existing, --overwrite or a new --result-tag".format(path))
        output_paths[row["sample_id"]] = paths
    input_mode = "audio_and_prompt" if args.include_prompt else "audio_only"
    print("[aqa-request] selected={} model={} url={} audio_format={} input_mode={} prompt_sent={} modalities=text output={}".format(
        len(rows), args.model, args.url, args.audio_input_format, input_mode,
        str(args.include_prompt).lower(), samples), flush=True)
    completed, failed, skipped = 0, 0, 0
    for row in rows:
        sample_id = row["sample_id"]
        output_json, output_text = output_paths[sample_id]
        result = {"sample_id": sample_id, "row_index": row["row_index"], "status": "failed",
                  "model": args.model, "url": args.url, "max_tokens": args.max_tokens,
                  "audio_input_format": args.audio_input_format, "modalities": ["text"],
                  "input_mode": input_mode, "prompt_sent": args.include_prompt,
                  "temperature": args.temperature, "created_at": datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        try:
            prompt, truth, paths, payloads = load_inputs(root, row)
            result.update(prompt=prompt, ground_truth=truth, wav_paths=paths)
            body = build_request(prompt, payloads, args.model, args.max_tokens, args.temperature,
                                 args.audio_input_format, args.include_prompt)
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            fingerprint = hashlib.sha256(args.url.encode("utf-8") + b"\n" + payload).hexdigest()
            result["request_sha256"] = fingerprint
            if args.skip_existing and (output_json.exists() or output_text.exists()):
                previous = read_json(output_json)
                if (previous.get("status") != "complete" or previous.get("request_sha256") != fingerprint
                        or previous.get("input_mode") != input_mode or previous.get("prompt") != prompt
                        or previous.get("ground_truth") != truth
                        or output_text.read_text(encoding="utf-8") != previous.get("reply")):
                    raise FileExistsError("existing result is incomplete or inputs changed; use a new --result-tag or --overwrite")
                skipped += 1
                print("[aqa-request] {} SKIP matching complete result".format(sample_id), flush=True)
                continue
            print("[aqa-request] {} sending wavs={} ...".format(sample_id, len(paths)), flush=True)
            response = post_json(args.url, payload, args.timeout_s, os.environ.get("VLLM_API_KEY", ""))
            result["raw_response"] = response
            reply, reasoning, finish = extract_reply(response)
            result.update(reply=reply, reasoning_content=reasoning, finish_reason=finish,
                          truncated=(finish == "length"), usage=response.get("usage"))
            if not reply.strip():
                raise ValueError("server returned no answer text; inspect raw_response/reasoning_content and max-tokens")
            output_text.write_text(reply, encoding="utf-8")
            result["status"] = "complete"
            completed += 1
            print("[aqa-request] {} OK finish_reason={} reply={}".format(sample_id, finish, output_text), flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            # Never replace an earlier result while trying to resume a run.
            if args.skip_existing and (output_json.exists() or output_text.exists()):
                raise RuntimeError("{}: cannot reuse existing result: {}".format(sample_id, exc)) from exc
            failed += 1
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, ServerError):
                result["error"].update(http_status=exc.status, response_body=exc.body)
                hint = server_error_hint(exc)
                if hint:
                    result["error"]["hint"] = hint
                    print("[aqa-request] HINT: " + hint, file=sys.stderr, flush=True)
            # If --overwrite failed, an old successful answer must not survive.
            if args.overwrite and output_text.exists():
                output_text.unlink()
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(output_json, result)
            print("[aqa-request] {} ERROR: {}; details={}".format(sample_id, exc, output_json), file=sys.stderr, flush=True)
            if isinstance(exc, KeyboardInterrupt) or not args.continue_on_error:
                return 1
            continue
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output_json, result)
    print("[aqa-request] DONE completed={} skipped={} failed={}".format(completed, skipped, failed), flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, help="Export root containing manifest.jsonl and samples/")
    parser.add_argument("--url", default=DEFAULT_URL, help="Full HTTP Chat Completions endpoint (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name, same as --served-model-name (default: %(default)s)")
    parser.add_argument("--audio-input-format", choices=("audio_url", "input_audio"), default="audio_url",
                        help="WAV transport format (default: %(default)s with a base64 data URL); does not override the server chat template")
    parser.add_argument("--include-prompt", action="store_true",
                        help="Also send original prompt messages/text (default: only WAVs, no system or text messages)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-s", type=float, default=300, help="HTTP socket timeout in seconds")
    parser.add_argument("--start-row", type=int, default=0, help="First original Parquet row index, inclusive")
    parser.add_argument("--limit", type=int, default=50, help="Selected row count (default: %(default)s); 0 requests all remaining exported rows")
    parser.add_argument("--result-tag", default="response", help="Result name: samples/row_XXXXXX/<tag>.txt/json")
    parser.add_argument("--skip-existing", action="store_true", help="Skip only matching, complete results")
    parser.add_argument("--overwrite", action="store_true", help="Replace results for the selected rows/tag")
    parser.add_argument("--continue-on-error", action="store_true", help="Record errors and continue; exit status stays nonzero")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print("[aqa-request] ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
