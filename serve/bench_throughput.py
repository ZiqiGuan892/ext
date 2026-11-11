#!/usr/bin/env python3
"""Client-observed text throughput benchmark for a single-stage Omni server.

Token counts come from server usage, never from SSE/text chunk counts.
Per-choice TPOT requires streamed token_ids (default) or logprobs. Unsupported
or buffered timing data fails explicitly; --timing-source none measures only
usage-based throughput. No model weights, Ray, or training plugins are loaded.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import random
import statistics
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral
from pathlib import Path

import requests


class BenchmarkError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BenchmarkError(message)


def positive_int(value):
    require(type(value) is int and value > 0, f"Expected positive integer, got {value!r}")
    return value


def messages_for(prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def tokenized_chat_length(encoded):
    """Count one chat's IDs across Transformers list/BatchEncoding APIs."""
    # Transformers 5 defaults to BatchEncoding: len(encoded) counts fields,
    # not tokens. Custom tokenizers may also retain a singleton batch axis.
    if isinstance(encoded, Mapping):
        require("input_ids" in encoded, "Tokenized chat has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    require(isinstance(encoded, (list, tuple)) and len(encoded) > 0,
            f"Expected nonempty token IDs, got {type(encoded).__name__}")
    if isinstance(encoded[0], (list, tuple)):
        require(len(encoded) == 1, "Expected one tokenized chat, got multiple sequences")
        encoded = encoded[0]
    require(len(encoded) > 0 and all(isinstance(t, Integral) and not isinstance(t, bool) for t in encoded),
            "Expected a flat sequence of integer token IDs for one chat")
    return len(encoded)


def make_prompts(tokenizer, count, input_tokens, seed, enable_thinking):
    """Fit each *templated* prompt exactly; never silently round its length."""
    rng = random.Random(seed)
    words = "apple river music garden paper cloud green table winter story light book".split()

    def length(text):
        return tokenized_chat_length(tokenizer.apply_chat_template(
            messages_for(text), tokenize=True, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ))

    prompts = []
    for i in range(count):
        prefix = f"Case {rng.getrandbits(64):016x}-{i}: Explain the following ideas."
        require(length(prefix) <= input_tokens, "--input-tokens is smaller than the chat template/prefix")
        fillers = [rng.choice(words) for _ in range(input_tokens * 2)]
        lo, hi = 0, len(fillers)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            text = prefix + "".join(" " + w for w in fillers[:mid])
            if length(text) <= input_tokens:
                lo = mid
            else:
                hi = mid - 1
        text = prefix + "".join(" " + w for w in fillers[:lo])
        # Normally each common English word adds one token. Handle tokenizer
        # boundary differences explicitly instead of assuming that property.
        while (current := length(text)) < input_tokens:
            for suffix in (" a", " the", " hello", " x", ".", "\n"):
                proposed = length(text + suffix)
                if current < proposed <= input_tokens:
                    text += suffix
                    break
            else:
                raise BenchmarkError(
                    f"Could not construct exact input length with {type(tokenizer).__name__}: "
                    f"target={input_tokens}, current={current}; tried all filler suffixes"
                )
        require(length(text) == input_tokens, "Prompt length construction failed")
        prompts.append(text)
    return prompts


def make_payload(args, prompt, output_tokens=None, seed=None):
    limit = args.output_tokens if output_tokens is None else output_tokens
    payload = {
        "model": args.model,
        "messages": messages_for(prompt),
        "max_tokens": limit,
        "temperature": args.temperature,
        "top_p": 1.0,
        "n": args.n,
        "stream": True,
        "stream_options": {"include_usage": True},
        # These are HTTP JSON fields, NOT a nested SDK extra_body object.
        "ignore_eos": args.ignore_eos,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if args.ignore_eos:
        payload["min_tokens"] = limit
    if seed is not None:
        payload["seed"] = seed
    if args.timing_source == "token_ids":
        payload["return_token_ids"] = True
    elif args.timing_source == "logprobs":
        payload.update(logprobs=True, top_logprobs=0)
    return payload


def sse_events(lines):
    """Parse SSE records, including comments/multiline data; reject truncation."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        # Other SSE fields and heartbeats do not represent generated tokens.
    require(not data, "SSE stream ended inside an event")


def stream_lines(chunks):
    """Decode HTTP chunks without reading byte-by-byte or waiting for 512 bytes.

    Uvicorn serves SSE using HTTP chunked transfer. iter_content(None) preserves
    its chunk delivery; SSE events may still span chunks or share a chunk.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending = ""
    for chunk in chunks:
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            yield line.removesuffix("\r")
    pending += decoder.decode(b"", final=True)
    require(not pending, "SSE stream ended inside a line")


def new_choice():
    return {"tokens": 0, "token_chunks": 0, "first_token_s": None,
            "last_token_s": None, "finish_reason": None, "content_chunks": 0,
            "max_tokens_per_chunk": 0}


class StreamMetrics:
    def __init__(self, n, source):
        self.n = n
        self.source = source
        self.choices = {i: new_choice() for i in range(n)}
        self.usage = None
        self.done = False

    def feed(self, event, elapsed):
        require(not self.done, "Event after [DONE]")
        if event.strip() == "[DONE]":
            self.done = True
            return
        chunk = json.loads(event)  # Malformed JSON is a failed request.
        require(isinstance(chunk, dict), "Invalid SSE JSON object")
        require(not chunk.get("error"), f"Server stream error: {chunk.get('error')}")
        if chunk.get("usage") is not None:
            require(isinstance(chunk["usage"], dict), "Invalid usage object")
            self.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            index = choice.get("index")
            require(type(index) is int and index in self.choices, f"Unexpected choice index: {index!r}")
            state = self.choices[index]
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                state["content_chunks"] += 1
            token_items = None
            if self.source == "token_ids":
                token_items = choice.get("token_ids")
                if token_items is not None:
                    require(isinstance(token_items, list) and all(type(t) is int for t in token_items),
                            "choice.token_ids must be a list of integer delta token IDs")
            elif self.source == "logprobs":
                token_items = (choice.get("logprobs") or {}).get("content")
                if token_items is not None:
                    require(isinstance(token_items, list) and all(isinstance(t, dict) for t in token_items),
                            "choice.logprobs.content must contain per-token objects")
            if token_items:
                require(state["finish_reason"] is None, f"Tokens received after choice {index} finished")
                require(state["last_token_s"] is None or elapsed >= state["last_token_s"],
                        "Non-monotonic token arrival time")
                state["tokens"] += len(token_items)
                state["token_chunks"] += 1
                state["max_tokens_per_chunk"] = max(state["max_tokens_per_chunk"], len(token_items))
                if state["first_token_s"] is None:
                    state["first_token_s"] = elapsed
                state["last_token_s"] = elapsed
            if choice.get("finish_reason") is not None:
                reason = choice["finish_reason"]
                require(reason in ("length", "stop"), f"Unexpected finish_reason: {reason!r}")
                state["finish_reason"] = reason

    def finish(self, latency, input_tokens, output_tokens, fixed_output):
        require(self.done, "Missing [DONE]: truncated stream")
        require(self.usage is not None, "Missing final usage; token throughput cannot be measured")
        prompt_tokens = positive_int(self.usage.get("prompt_tokens"))
        completion_tokens = positive_int(self.usage.get("completion_tokens"))
        # Audio clients cannot infer the expanded multimodal token count from
        # a text tokenizer. None means record positive server usage instead.
        if input_tokens is not None:
            require(prompt_tokens == input_tokens,
                    f"Server prompt_tokens={prompt_tokens}, expected {input_tokens}; tokenizer/chat template mismatch")
        require(all(c["finish_reason"] is not None for c in self.choices.values()),
                f"Not all {self.n} choices finished; n may be ignored by this backend")
        if fixed_output:
            require(completion_tokens == self.n * output_tokens,
                    f"completion_tokens={completion_tokens}, expected n*max_tokens={self.n * output_tokens}")
            # A backend may report stop on the final allowed token. Exact
            # counts, not the spelling of that terminal reason, prove length.
        require(completion_tokens <= self.n * output_tokens, "Usage exceeds requested output limit")
        if self.source != "none":
            total = sum(c["tokens"] for c in self.choices.values())
            require(total == completion_tokens,
                    f"Streamed {self.source} count={total}, usage={completion_tokens}. "
                    "Missing/cumulative/incomplete token metadata: TPOT is unavailable. "
                    "Try --timing-source logprobs if supported, or --timing-source none for throughput only.")
        choices = []
        for index, state in self.choices.items():
            result = {"index": index, **state, "ttft_s": None, "tpot_s": None}
            if self.source != "none":
                require(0 < state["tokens"] <= output_tokens, f"Invalid token count for choice {index}")
                require(state["last_token_s"] <= latency, "Last token is later than request end")
                if fixed_output:
                    require(state["tokens"] == output_tokens, f"Choice {index} has wrong output length")
                result["ttft_s"] = state["first_token_s"]
                if state["tokens"] > 1:
                    require(state["token_chunks"] > 1 and state["last_token_s"] > state["first_token_s"],
                            f"Choice {index} output is buffered in one instant; cannot measure TPOT. "
                            "Use --timing-source none for throughput only.")
                    result["tpot_s"] = ((state["last_token_s"] - state["first_token_s"])
                                        / (state["tokens"] - 1))
            else:
                result["tokens"] = None  # Never divide request usage equally across n choices.
            choices.append(result)
        return {"success": True, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "request_latency_s": latency,
                "choices": choices, "usage": self.usage}


def send_request(args, prompt, request_id, output_tokens=None):
    limit = args.output_tokens if output_tokens is None else output_tokens
    payload = make_payload(args, prompt, limit, seed=args.seed + request_id)
    return send_payload(args, payload, request_id, expected_input_tokens=args.input_tokens)


def send_payload(args, payload, request_id, expected_input_tokens):
    """Shared SSE transport; payload creation/file I/O precede request timing."""
    meter = StreamMetrics(payload["n"], args.timing_source)
    start = time.perf_counter()
    try:
        # A session belongs to one request, not multiple concurrent threads.
        # trust_env=False keeps localhost benchmark traffic out of HTTP proxies.
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                f"http://{args.host}:{args.port}/v1/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {args.api_key}"}, stream=True,
                timeout=(10, args.timeout_s),
            ) as response:
                if response.status_code != 200:
                    raise BenchmarkError(f"HTTP {response.status_code}: {response.text[:1200]}")
                for event in sse_events(stream_lines(response.iter_content(chunk_size=None))):
                    meter.feed(event, time.perf_counter() - start)
                    if meter.done:
                        break
                latency = time.perf_counter() - start
        result = meter.finish(latency, expected_input_tokens, payload["max_tokens"], args.ignore_eos)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}",
                  "request_latency_s": time.perf_counter() - start,
                  "observed_usage": meter.usage}
    return {"request_id": request_id, **result}


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": ordered[0], "max": ordered[-1],
            "p95": ordered[math.ceil(0.95 * len(values)) - 1]}


def summarize(batches, num_devices=None):
    wall = sum(b["wall_time_s"] for b in batches)
    require(wall > 0, "Measured wall time must be positive")
    results = [r for b in batches for r in b["requests"]]
    good = [r for r in results if r["success"]]
    choices = [c for r in good for c in r["choices"]]
    outputs = sum(r["completion_tokens"] for r in good)
    inputs = sum(r["prompt_tokens"] for r in good)
    valid = bool(results) and len(good) == len(results)
    output_rate = outputs / wall
    return {
        "valid": valid, "num_requests": len(results), "num_successful_requests": len(good),
        "num_errors": len(results) - len(good), "num_successful_sequences": len(choices),
        "total_wall_clock_s": wall, "total_completion_tokens": outputs,
        "total_prompt_tokens": inputs,
        # No multiplying by BS/n, no dividing by the sum of request latencies.
        "successful_output_tokens_per_s": output_rate,
        "throughput_tokens_per_s": output_rate if valid else None,
        "total_tokens_per_s": (inputs + outputs) / wall if valid else None,
        "requests_per_s": len(good) / wall, "sequences_per_s": len(choices) / wall,
        "output_tokens_per_s_per_device": output_rate / num_devices if valid and num_devices else None,
        "ttft_s": distribution([c["ttft_s"] for c in choices if c["ttft_s"] is not None]),
        "tpot_s": distribution([c["tpot_s"] for c in choices if c["tpot_s"] is not None]),
        "request_latency_s": distribution([r["request_latency_s"] for r in good]),
        "multi_token_chunks": any(c["max_tokens_per_chunk"] > 1 for c in choices),
    }


def run_batch(args, executor, prompts, start_id):
    start = time.perf_counter()  # Immediately before submitting the first request.
    futures = [executor.submit(send_request, args, p, start_id + i) for i, p in enumerate(prompts)]
    results = [future.result() for future in as_completed(futures)]
    wall = time.perf_counter() - start  # All responses parsed and validated; no per-request printing inside.
    return {"wall_time_s": wall, "requests": sorted(results, key=lambda r: r["request_id"])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28889)
    parser.add_argument("--model", default="qwen3.8")
    parser.add_argument("--tokenizer", required=True, help="Same checkpoint/tokenizer as the serving stage")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=2048, help="Exact length INCLUDING chat template")
    parser.add_argument("--output-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=16, help="Concurrent HTTP requests, not engine batch size")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timing-source", choices=("token_ids", "logprobs", "none"), default="token_ids")
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--probe-only", action="store_true", help="One request, up to 16 output tokens per choice")
    parser.add_argument("--profile-arm-file", type=Path,
                        help="Create this NEW file after preflight/warmup to trigger the standalone serve profiler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=1800, help="HTTP socket read timeout, not total run deadline")
    parser.add_argument("--num-devices", type=int, help="Actual devices occupied by THIS server; no inferred divisor")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--report", type=Path, default=Path(f"bench_report_{time.time_ns()}.json"))
    args = parser.parse_args(argv)
    try:
        for name in ("input_tokens", "output_tokens", "batch_size", "n", "num_batches"):
            positive_int(getattr(args, name))
        require(args.warmup_batches >= 0, "--warmup-batches must be >= 0")
        require(math.isfinite(args.temperature) and args.temperature >= 0, "Invalid temperature")
        require(args.n == 1 or args.temperature >= 1e-5,
                "n>1 requires sampling; use --temperature 1.0, not greedy temperature=0")
        require(math.isfinite(args.timeout_s) and args.timeout_s > 0, "Invalid timeout")
        require(args.seed >= 0, "--seed must be nonnegative")
        require(not (args.probe_only and args.profile_arm_file), "--probe-only cannot arm a profile")
        if args.num_devices is not None:
            positive_int(args.num_devices)
    except BenchmarkError as exc:
        parser.error(str(exc))
    return args


def run(args):
    # Import lazily: tests and --help do not require torch/transformers.
    from transformers import AutoTokenizer

    require(not args.report.exists(), f"Report already exists: {args.report}")
    if args.profile_arm_file:
        require(not args.profile_arm_file.exists(), f"Profile arm file already exists: {args.profile_arm_file}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    count = 1 if args.probe_only else 1 + args.batch_size * (args.warmup_batches + args.num_batches)
    prompts = make_prompts(tokenizer, count, args.input_tokens, args.seed, args.enable_thinking)
    corpus = json.dumps([messages_for(p) for p in prompts], ensure_ascii=False, separators=(",", ":"))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "api_key"}
    report = {"config": config, "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
              "profiling_run": args.profile_arm_file is not None,
              "timing_definition": "client receipt; TPOT=(last token arrival-first token arrival)/(choice tokens-1)",
              "warmup": [], "batches": [], "summary": None}
    print(f"Exact templated input={args.input_tokens}, output={args.output_tokens}, "
          f"HTTP concurrency={args.batch_size}, n={args.n}, timing={args.timing_source}", flush=True)
    print("Preflight: verifying n, usage, input length, fixed output and timing metadata", flush=True)
    report["probe"] = send_request(args, prompts[0], 0, output_tokens=min(16, args.output_tokens))
    valid = report["probe"]["success"]
    if not valid:
        print(report["probe"]["error"], flush=True)
    elif not args.probe_only:
        cursor = 1
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            for phase, number in (("warmup", args.warmup_batches), ("batches", args.num_batches)):
                if not valid:
                    break
                if phase == "batches" and args.profile_arm_file:
                    args.profile_arm_file.parent.mkdir(parents=True, exist_ok=True)
                    # O_EXCL prevents silently reusing another capture's trigger.
                    with args.profile_arm_file.open("x", encoding="utf-8") as trigger:
                        json.dump({"created_at": time.time(), "report": str(args.report)}, trigger)
                    print(f"Profile armed: {args.profile_arm_file}; timings include instrumentation overhead", flush=True)
                for i in range(number):
                    batch = run_batch(args, executor, prompts[cursor:cursor + args.batch_size], cursor)
                    cursor += args.batch_size
                    report[phase].append(batch)
                    summary = summarize([batch], args.num_devices)
                    print(f"{phase} {i + 1}: wall={batch['wall_time_s']:.3f}s "
                          f"output_tokens={summary['total_completion_tokens']} "
                          f"output_tok/s={summary['throughput_tokens_per_s']} "
                          f"errors={summary['num_errors']}", flush=True)
                    valid = summary["valid"]
                    if not valid:
                        for result in batch["requests"]:
                            if not result["success"]:
                                print(f"request {result['request_id']}: {result['error']}", flush=True)
                        break
        if report["batches"]:
            report["summary"] = summarize(report["batches"], args.num_devices)
    report["valid"] = valid
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report["summary"] or report["probe"], indent=2, ensure_ascii=False))
    print(f"Report: {args.report.resolve()}")
    return 0 if valid else 1


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (BenchmarkError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
