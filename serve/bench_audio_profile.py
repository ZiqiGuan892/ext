#!/usr/bin/env python3
"""Real WAV input for standalone eager profiling; no training/tokenizer imports.

Use distinct PCM waveforms for probe, warmup and measurement so audio encoding
is not silently skipped by the server's cross-request embedding cache.
"""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import wave

import bench_throughput as bench


def read_audio(path, min_seconds, max_seconds):
    # Inspect before reading a potentially long recording into memory.
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, frames, compression, _ = wav.getparams()
        bench.require((channels, width, rate, compression) == (1, 2, 16000, "NONE"),
                      "requires mono PCM16 WAV at 16000 Hz (no implicit conversion)")
        seconds = frames / rate
        bench.require(min_seconds <= seconds <= max_seconds,
                      f"duration {seconds:.3f}s outside [{min_seconds}, {max_seconds}]")
        pcm = wav.readframes(frames)
        bench.require(len(pcm) == frames * channels * width, "truncated WAV data")
    # Repack the original PCM unchanged; ignore file metadata when deduplicating.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return {"path": str(path.resolve()), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "duration_s": seconds, "sampling_rate": rate, "frames": frames,
            "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def select_audio(args, count):
    if args.audio_dir is not None:
        bench.require(args.audio_dir.is_dir(), f"Audio directory not found: {args.audio_dir}")
        paths = sorted(p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    else:
        paths = list(args.audio_file)
    random.Random(args.seed).shuffle(paths)
    selected, hashes, skipped = [], set(), Counter()
    for path in paths:
        try:
            item = read_audio(path, args.min_audio_seconds, args.max_audio_seconds)
        except (OSError, EOFError, wave.Error, bench.BenchmarkError) as exc:
            skipped[str(exc)] += 1
            continue
        if item["pcm_sha256"] in hashes:
            skipped["duplicate PCM waveform"] += 1
            continue
        hashes.add(item["pcm_sha256"])
        selected.append(item)
        if len(selected) == count:
            break
    bench.require(len(selected) == count,
                  f"Need {count} distinct valid WAVs for probe+warmup+capture, found {len(selected)}; "
                  f"scanned={len(paths)}, skipped={dict(skipped)}. Supply more audio or reduce --batch-size. "
                  "Audio is never repeated, cropped, resampled or replaced with noise.")
    return selected, dict(skipped)


def audio_payload(args, item, request_id, output_tokens=None):
    payload = bench.make_payload(args, args.prompt, output_tokens, seed=args.seed + request_id)
    if args.audio_input_format == "audio_url":
        part = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + item["data"]}}
    else:
        part = {"type": "input_audio", "input_audio": {"data": item["data"], "format": "wav"}}
    payload["messages"][1]["content"] = [part, {"type": "text", "text": args.prompt}]
    payload["modalities"] = ["text"]
    return payload


def send_audio(args, item, request_id, output_tokens=None):
    result = bench.send_payload(args, audio_payload(args, item, request_id, output_tokens),
                                request_id, expected_input_tokens=None)
    result["audio"] = {k: v for k, v in item.items() if k != "data"}
    return result


def run_batch(args, executor, items, start_id):
    start = time.perf_counter()
    futures = [executor.submit(send_audio, args, item, start_id + i) for i, item in enumerate(items)]
    results = [future.result() for future in as_completed(futures)]
    return {"wall_time_s": time.perf_counter() - start,
            "requests": sorted(results, key=lambda r: r["request_id"])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=Path, help="Recursively select distinct WAV files")
    source.add_argument("--audio-file", type=Path, action="append", help="Repeat for each WAV path")
    parser.add_argument("--min-audio-seconds", type=float, default=1)
    parser.add_argument("--max-audio-seconds", type=float, default=30)
    parser.add_argument("--audio-input-format", choices=("audio_url", "input_audio"), default="audio_url")
    parser.add_argument("--prompt", default="请根据音频内容回答其中的问题。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28889)
    parser.add_argument("--model", default="qwen3.8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n", type=int, choices=(1,), default=1, help="This Omni path requires n=1")
    parser.add_argument("--output-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timing-source", choices=("none", "token_ids", "logprobs"), default="none")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=4, help="Statistical divisor, not server TP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--profile-arm-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for key in ("batch_size", "output_tokens", "num_batches", "num_devices"):
            bench.positive_int(getattr(args, key))
        bench.require(args.warmup_batches >= 1, "At least one warmup batch is required")
        bench.require(args.seed >= 0, "Seed must be nonnegative")
        bench.require(math.isfinite(args.temperature) and args.temperature >= 0, "Invalid temperature")
        bench.require(math.isfinite(args.timeout_s) and args.timeout_s > 0, "Invalid timeout")
        bench.require(math.isfinite(args.min_audio_seconds) and math.isfinite(args.max_audio_seconds)
                      and 0 < args.min_audio_seconds <= args.max_audio_seconds, "Invalid audio duration range")
        bench.require(args.report.resolve() != args.profile_arm_file.resolve(), "Report and arm file must differ")
    except bench.BenchmarkError as exc:
        parser.error(str(exc))
    return args


def run(args):
    bench.require(not args.report.exists(), f"Report already exists: {args.report}")
    bench.require(not args.profile_arm_file.exists(), f"Profile arm file already exists: {args.profile_arm_file}")
    count = 1 + args.batch_size * (args.warmup_batches + args.num_batches)
    items, skipped = select_audio(args, count)
    print(f"Selected {count} distinct audio waveforms; skipped={skipped}", flush=True)
    print(f"HTTP concurrency={args.batch_size}, n=1, output={args.output_tokens}; "
          "multimodal input tokens come from server usage, not a fixed text length", flush=True)
    config = {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "audio_file" and v else v
              for k, v in vars(args).items() if k != "api_key"}
    report = {"config": config, "profiling_run": True, "input_modality": "audio",
              "selection_skipped": skipped, "warmup": [], "batches": [], "summary": None,
              "audio_selection": [{k: v for k, v in item.items() if k != "data"} for item in items],
              "note": "Client timings include profiling overhead; capture completion needs a separate check."}
    print("Preflight: one fresh audio, up to 16 output tokens", flush=True)
    report["probe"] = send_audio(args, items[0], 0, output_tokens=min(16, args.output_tokens))
    valid = report["probe"]["success"]
    if not valid:
        print(report["probe"]["error"], flush=True)
    else:
        cursor = 1
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            for phase, number in (("warmup", args.warmup_batches), ("batches", args.num_batches)):
                if not valid:
                    break
                if phase == "batches":
                    args.profile_arm_file.parent.mkdir(parents=True, exist_ok=True)
                    with args.profile_arm_file.open("x", encoding="utf-8") as stream:
                        json.dump({"created_at": time.time(), "report": str(args.report), "modality": "audio"}, stream)
                    print(f"Profile armed: {args.profile_arm_file}", flush=True)
                for i in range(number):
                    batch = run_batch(args, executor, items[cursor:cursor + args.batch_size], cursor)
                    cursor += args.batch_size
                    report[phase].append(batch)
                    summary = bench.summarize([batch], args.num_devices)
                    print(f"{phase} {i + 1}: wall={batch['wall_time_s']:.3f}s "
                          f"output_tokens={summary['total_completion_tokens']} "
                          f"output_tok/s={summary['throughput_tokens_per_s']} errors={summary['num_errors']}", flush=True)
                    valid = summary["valid"]
                    if not valid:
                        for result in batch["requests"]:
                            if not result["success"]:
                                print(f"request {result['request_id']}: {result['error']}", flush=True)
                        break
    if report["batches"]:
        report["summary"] = bench.summarize(report["batches"], args.num_devices)
    report["valid"] = valid
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Report: {args.report.resolve()}", flush=True)
    return 0 if valid else 1


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (bench.BenchmarkError, OSError, wave.Error) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
