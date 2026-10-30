#!/usr/bin/env python3
"""Prepare identical AQA inputs, request serving text, and join rollout replies.

Python 3.8+. Only prepare needs pyarrow/numpy/soundfile. No model weights are
loaded here. Both engines consume a single immutable case manifest and WAVs.
"""

import argparse
import base64
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
import uuid


DEFAULT_MODEL = "/opt/huawei/dataset/model_gy/downloaded_models/2748"
IDENTITY_KEYS = ("sample_id", "case_id", "prompt_mode", "prompt", "wav_paths", "wav_sha256")


def sibling(name):
    spec = importlib.util.spec_from_file_location("aqa_ab_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_prompt(prompt, audio_count, mode):
    """Keep original system text, remove only user text in system_audio mode."""
    if mode not in ("system_audio", "original"):
        raise ValueError("unsupported prompt mode")
    if not isinstance(prompt, list) or not prompt or audio_count < 1:
        raise ValueError("a nonempty message list and at least one audio are required")
    result, count, seen_user = [], 0, False
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("prompt messages must have string content")
        role, content = message.get("role"), message["content"]
        if role not in ("system", "user") or (role == "system" and seen_user):
            raise ValueError("comparison supports leading system messages followed by user audio messages only")
        if "<image>" in content or "<video>" in content:
            raise ValueError("audio comparison cannot contain visual placeholders")
        n_audio = content.count("<audio>")
        if role == "system" and n_audio:
            raise ValueError("system messages cannot contain audio placeholders")
        if role == "user":
            seen_user = True
            if not n_audio:
                raise ValueError("each user message must contain audio; ambiguous text-only history is unsupported")
            count += n_audio
            if mode == "system_audio":
                content = "<audio>" * n_audio
        result.append({"role": role, "content": content})
    if count != audio_count:
        raise ValueError("prompt/audio count mismatch: {} != {}".format(count, audio_count))
    return result


def prepare_case(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.sample_size < 1 or args.max_tokens < 1 or args.tp != 4 or not 0 <= args.seed < 2**63:
        raise ValueError("positive sample size/max tokens, nonnegative seed and TP=4 required")
    logits_check = getattr(args, "logits_check", "false") == "true"
    logits_max_steps = getattr(args, "logits_max_steps", 0)
    if logits_max_steps < 0:
        raise ValueError("logits-max-steps must be nonnegative")
    logits_max_steps = logits_max_steps or args.max_tokens
    logits_top_k = getattr(args, "logits_top_k", 100)
    if type(logits_top_k) is not int or logits_top_k < 0:
        raise ValueError("logits-top-k must be a nonnegative integer (0=full)")
    if logits_check and getattr(args, "embedding_check", "false") != "true":
        raise ValueError("logits-check requires embedding-check=true for request/input identity")
    forward_check = getattr(args, "forward_check", "false") == "true"
    forward_steps = getattr(args, "forward_max_steps", 2)
    forward_until_eos = forward_steps == 0
    if forward_until_eos:
        forward_steps = args.max_tokens
    if forward_check and (not logits_check or not 1 <= forward_steps <= logits_max_steps):
        raise ValueError("forward-check requires logits-check and 1 <= forward-max-steps <= logits-max-steps")
    source = Path(args.parquet).expanduser().resolve(strict=True)
    export = Path(args.export_dir).expanduser().resolve(strict=True)
    model = Path(args.model_path).expanduser().resolve(strict=True)
    if not model.is_dir():
        raise ValueError("model-path must be the checkpoint directory")
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    helper = sibling("prepare_aqa_wav_parquet")
    original_wav_subset = root / "source_wav.parquet"
    helper.convert_parquet(source, export, original_wav_subset, sample_size=args.sample_size, seed=args.seed)
    selection = read_json(original_wav_subset.with_suffix(".selection.json"))
    exports = helper.load_manifest(export)
    table = pq.read_table(original_wav_subset)
    case_id, rows, prompts, extras = uuid.uuid4().hex, [], [], []
    for original, selected in zip(table.to_pylist(), selection["rows"]):
        if original.get("images") or original.get("videos"):
            raise ValueError("selected rows must be audio-only data")
        prompt = canonical_prompt(original["prompt"], len(selected["audios"]), args.prompt_mode)
        identity = {
            "sample_id": selected["sample_id"], "case_id": case_id,
            "prompt_mode": args.prompt_mode, "prompt": prompt,
            "wav_paths": [a["wav_path"] for a in selected["audios"]],
            "wav_sha256": [a["sha256"] for a in selected["audios"]],
        }
        identity["prompt_sha256"] = hashlib.sha256(json.dumps(
            prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        extra = dict(original.get("extra_info") or {})
        if "huanyu_text_ab" in extra:
            raise ValueError("source already contains reserved huanyu_text_ab metadata")
        extra["huanyu_text_ab"] = identity
        extras.append(extra)
        prompts.append(prompt)
        record = exports[selected["source_row_index"]]
        rows.append(dict(identity, row_index=selected["source_row_index"],
                         source_row_index=selected["source_row_index"],
                         original_prompt=original["prompt"], ground_truth=record["ground_truth"],
                         audios=record["audios"]))
    prompt_field = table.schema.field("prompt")
    table = table.set_column(table.schema.get_field_index("prompt"), prompt_field,
                             pa.array(prompts, type=prompt_field.type))
    extra_array = pa.array(extras)
    extra_index = table.schema.get_field_index("extra_info")
    if extra_index < 0:
        table = table.append_column("extra_info", extra_array)
    else:
        old_field = table.schema.field("extra_info")
        table = table.set_column(extra_index, pa.field("extra_info", extra_array.type,
                                                      metadata=old_field.metadata), extra_array)
    subset = root / "validation.parquet"
    with subset.open("xb") as stream:
        pq.write_table(table, stream)
    manifest = root / "manifest.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fingerprints = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
                 "generation_config.json", "special_tokens_map.json"):
        path = model / name
        fingerprints[name] = sha256(path) if path.is_file() else None
    context = 3072 + args.max_tokens + 128
    case = {
        "schema_version": 1, "case_id": case_id, "model_path": str(model),
        "subset_parquet": str(subset), "subset_parquet_sha256": sha256(subset),
        "source_parquet": str(source), "export_dir": str(export),
        "sample_size": len(rows), "selection_seed": args.seed, "prompt_mode": args.prompt_mode,
        "manifest_sha256": sha256(manifest), "checkpoint_file_sha256": fingerprints,
        "enable_thinking": args.enable_thinking == "true",
        "determinism_check": {"enabled": True, "torch_deterministic_algorithms": True,
                              "torch_deterministic_warn_only": False},
        "embedding_check": {"enabled": getattr(args, "embedding_check", "false") == "true",
                            "scope": "llm_inputs_embeds", "strict": True, "expected_tp_size": args.tp},
        "logits_check": {"enabled": logits_check, "max_steps": logits_max_steps, "top_k": logits_top_k,
                         "scope": "model_logits_sampler_input_greedy_input", "strict": True},
        "forward_check": {"enabled": forward_check, "max_steps": forward_steps, "until_eos": forward_until_eos},
        "sampling": {"max_tokens": args.max_tokens, "temperature": 0, "top_p": 1, "top_k": -1,
                     "n": 1, "seed": args.seed, "min_tokens": 0, "ignore_eos": False,
                     "repetition_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0},
        "engine": {"tensor_parallel_size": args.tp, "prompt_length": 3072,
                   "max_model_len": context, "max_num_batched_tokens": context,
                   "max_num_seqs": 1, "dtype": "bfloat16", "gpu_memory_utilization": 0.5,
                   "enforce_eager": True,
                   "batch_invariant": getattr(args, "batch_invariant", "false") == "true"},
    }
    # Written last: a failed preparation never looks like a completed case.
    write_json(root / "case.json", case)
    print("[aqa-text-ab] prepared={} selected={} prompt_mode={}".format(root, len(rows), args.prompt_mode))
    return 0


def load_case(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    case = read_json(root / "case.json")
    if case.get("schema_version") != 1 or not case.get("case_id"):
        raise ValueError("unsupported or incomplete case.json")
    manifest = root / "manifest.jsonl"
    if sha256(manifest) != case["manifest_sha256"]:
        raise ValueError("case manifest changed after preparation")
    rows, seen = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = row.get("sample_id", "")
        if not re.fullmatch(r"row_[0-9]{6,}", sample) or sample in seen or row.get("case_id") != case["case_id"]:
            raise ValueError("invalid, duplicate or foreign sample identity")
        seen.add(sample)
        rows.append(row)
    if len(rows) != case["sample_size"]:
        raise ValueError("case sample count mismatch")
    return root, case, rows


def sanitized_request(body):
    copied = copy.deepcopy(body)
    for message in copied["messages"]:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "audio_url":
                    value = part["audio_url"]["url"].encode("ascii")
                    part["audio_url"]["url"] = "<WAV data URL sha256={}>".format(hashlib.sha256(value).hexdigest())
    return copied


def build_case_request(prompt, payloads, model, sampling, enable_thinking):
    """Encode the frozen case messages, independent of the standalone client's mode.

    prepare already removed user question text in system_audio mode. Preserve
    those messages here, including system text, and only expand audio markers.
    Older request_aqa_validation_wav.py versions lack include_prompt and use a
    different audio format; neither should determine this comparison's inputs.
    """
    messages = canonical_prompt(prompt, len(payloads), "original")
    audio_index = 0
    for message in messages:
        pieces = message["content"].split("<audio>")
        if len(pieces) == 1:
            continue
        parts = []
        for index, text in enumerate(pieces):
            if text:
                parts.append({"type": "text", "text": text})
            if index < len(pieces) - 1:
                encoded = base64.b64encode(payloads[audio_index]).decode("ascii")
                parts.append({"type": "audio_url", "audio_url": {
                    "url": "data:audio/wav;base64," + encoded,
                }})
                audio_index += 1
        message["content"] = parts
    body = {"model": model, "max_tokens": sampling["max_tokens"], "temperature": sampling["temperature"],
            "stream": False, "n": 1, "modalities": ["text"], "messages": messages}
    body.update(sampling)
    body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return body


def serve_case(args):
    root, case, rows = load_case(args.case_dir)
    parsed = urlsplit(args.url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1/chat/completions"):
        raise ValueError("url must be an HTTP(S) /v1/chat/completions endpoint")
    if not args.model.strip() or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("nonempty model and positive finite timeout required")
    client = sibling("request_aqa_validation_wav")
    destination = root / "serve"
    destination.mkdir(exist_ok=True)
    # Fail before sending any request if a later row would collide.
    if not args.skip_existing:
        for row in rows:
            for suffix in (".json", ".txt"):
                if (destination / (row["sample_id"] + suffix)).exists():
                    raise FileExistsError("serve result already exists; choose a new case or --skip-existing")
    failures = 0
    for row in rows:
        sample = row["sample_id"]
        output = destination / (sample + ".json")
        text_path = destination / (sample + ".txt")
        result = {key: row[key] for key in IDENTITY_KEYS}
        result.update(schema_version=1, status="failed", model=args.model, url=args.url,
                      model_path=case["model_path"], representation="http_message_content",
                      sampling=case["sampling"], enable_thinking=case["enable_thinking"])
        started = time.monotonic()
        try:
            export_row = row
            if case.get("serve_repeat"):
                # Unique probe IDs identify repetitions; WAVs remain in the
                # original exported row directory and are never re-encoded.
                export_row = dict(row, sample_id=row["source_sample_id"])
                result.update(source_sample_id=row["source_sample_id"], repetition=row["repetition"])
            original_prompt, truth, paths, payloads = client.load_inputs(Path(case["export_dir"]), export_row)
            hashes = [hashlib.sha256(payload).hexdigest() for payload in payloads]
            if (original_prompt != row["original_prompt"] or truth != row["ground_truth"]
                    or paths != row["wav_paths"] or hashes != row["wav_sha256"]):
                raise ValueError("exported prompt/truth/WAV differs from prepared case")
            body = build_case_request(row["prompt"], payloads, args.model,
                                      case["sampling"], case["enable_thinking"])
            if case.get("embedding_check", {}).get("enabled", False) or case.get("serve_repeat"):
                body["request_id"] = "huanyuab.{}.{}".format(case["case_id"], sample)
                result["request_id"] = body["request_id"]
                if case.get("embedding_check", {}).get("enabled", False):
                    result["embedding_request_id"] = body["request_id"]
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            digest = hashlib.sha256(args.url.encode("utf-8") + b"\n" + payload).hexdigest()
            result.update(request_sha256=digest, request=sanitized_request(body))
            if args.skip_existing and (output.exists() or text_path.exists()):
                prior = read_json(output)
                if (prior.get("status") != "complete" or prior.get("request_sha256") != digest
                        or any(prior.get(k) != row[k] for k in IDENTITY_KEYS)
                        or text_path.read_text(encoding="utf-8") != prior.get("reply")):
                    raise FileExistsError("existing response is incomplete or inputs differ")
                print("[aqa-text-ab] {} SKIP".format(sample), flush=True)
                continue
            response = client.post_json(args.url, payload, args.timeout_s, os.environ.get("VLLM_API_KEY", ""))
            result["raw_response"] = response
            reply, reasoning, finish = client.extract_reply(response)
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("structured reasoning content is not a string")
            if not reply and not reasoning:
                raise ValueError("response has neither answer nor reasoning text")
            result.update(status="complete", reply=reply, reasoning_content=reasoning,
                          answer_text_present=bool(reply), finish_reason=finish, truncated=finish == "length",
                          usage=response.get("usage"))
            with text_path.open("x", encoding="utf-8") as stream:
                stream.write(reply)
        except Exception as exc:
            # Never overwrite an existing successful or failed record on resume.
            if output.exists() or text_path.exists():
                raise
            failures += 1
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(output, result)
            print("[aqa-text-ab] {} ERROR {}".format(sample, exc), file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return 1
            continue
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output, result)
        print("[aqa-text-ab] {} COMPLETE finish={} text={}".format(sample, finish, text_path), flush=True)
    return 1 if failures else 0


def _read_result(root, side, row):
    path = root / side / (row["sample_id"] + ".json")
    if not path.exists():
        return None, "missing"
    result = read_json(path)
    for key in IDENTITY_KEYS:
        if result.get(key) != row[key]:
            raise ValueError("{} {} identity mismatch: {}".format(side, row["sample_id"], key))
    if result.get("status") != "complete":
        return result, "failed"
    if not isinstance(result.get("reply"), str):
        raise ValueError("{} {} has no text reply".format(side, row["sample_id"]))
    text_path = path.with_suffix(".txt")
    if not text_path.exists() or text_path.read_text(encoding="utf-8") != result["reply"]:
        raise ValueError("{} {} text/JSON mismatch".format(side, row["sample_id"]))
    return result, "complete"


def compare_case(args):
    root, case, rows = load_case(args.case_dir)
    expected = {row["sample_id"] for row in rows}
    for side in ("serve", "rollout"):
        for path in (root / side).glob("row_*.json"):
            if path.stem not in expected:
                raise ValueError("unexpected sample result: {}".format(path))
    outputs = [root / name for name in ("comparison.jsonl", "comparison.csv", "summary.json")]
    embedding_enabled = case.get("embedding_check", {}).get("enabled", False)
    embedding_outputs = [root / name for name in ("embedding_comparison.jsonl", "embedding_summary.json")]
    if embedding_enabled and any(path.exists() for path in embedding_outputs):
        raise FileExistsError("embedding comparison outputs already exist; preserve previous results")
    if any(path.exists() for path in outputs):
        raise FileExistsError("comparison outputs already exist; preserve them and choose a new case for another run")
    embedding_report = None
    embedding_samples = {}
    if embedding_enabled:
        embedding_report = sibling("compare_huanyu_prompt_embeddings").compare_case_embeddings(root, case, rows)
        embedding_samples = {item["sample_id"]: item for item in embedding_report["samples"]}
    comparisons = []
    for row in rows:
        serve, serve_status = _read_result(root, "serve", row)
        rollout, rollout_status = _read_result(root, "rollout", row)
        complete = serve_status == rollout_status == "complete"
        reasoning = serve.get("reasoning_content") if serve else None
        comparable = complete and not reasoning
        comparisons.append({
            "sample_id": row["sample_id"], "source_row_index": row["source_row_index"],
            "case_id": case["case_id"], "prompt_mode": row["prompt_mode"], "prompt": row["prompt"],
            "ground_truth": row["ground_truth"], "wav_paths": row["wav_paths"],
            "serve_status": serve_status, "rollout_status": rollout_status,
            "serve_reply": serve.get("reply", "") if serve else "",
            "serve_reasoning": reasoning, "rollout_reply": rollout.get("reply", "") if rollout else "",
            "serve_finish_reason": serve.get("finish_reason") if serve else None,
            "serve_truncated": serve.get("truncated") if serve else None,
            "rollout_token_count": rollout.get("response_token_count") if rollout else None,
            "representation_comparable": comparable,
            "reply_exact_match": (serve["reply"] == rollout["reply"]) if comparable else None,
            "embedding_status": embedding_samples.get(row["sample_id"], {}).get("status", "NOT_COLLECTED"),
            "embedding_strict_equal": embedding_samples.get(row["sample_id"], {}).get("strict_equal"),
            "comparison_note": ("HTTP separates reasoning; raw texts retained, no reconstructed equality claim"
                                if complete and reasoning else "incomplete pair" if not complete
                                else "exact text equality only; not answer correctness"),
        })
    with outputs[0].open("x", encoding="utf-8") as stream:
        for item in comparisons:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    with outputs[1].open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        for item in comparisons:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in item.items()})
    completed = sum(item["serve_status"] == item["rollout_status"] == "complete" for item in comparisons)
    comparable_count = sum(item["representation_comparable"] for item in comparisons)
    identical = sum(item["reply_exact_match"] is True for item in comparisons)
    summary = {"case_id": case["case_id"], "expected": len(rows), "complete_pairs": completed,
               "incomplete_pairs": len(rows) - completed, "representation_comparable_pairs": comparable_count,
               "identical_text_pairs": identical, "different_text_pairs": comparable_count - identical,
               "embedding_check": embedding_report["summary"] if embedding_report else {"status": "NOT_COLLECTED"},
               "note": "Text equality is not accuracy. Embedding equality is reported separately; "
                       "same canonical messages alone do not prove identical model inputs or numerics."}
    if embedding_report is not None:
        with embedding_outputs[0].open("x", encoding="utf-8") as stream:
            for item in embedding_report["samples"]:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(embedding_outputs[1], embedding_report["summary"])
    write_json(outputs[2], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if completed != len(rows) or (embedding_report and embedding_report["summary"]["status"] == "INCOMPLETE"):
        return 2
    return 1 if embedding_report and embedding_report["summary"]["status"] == "FAIL" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Sample once and freeze a shared WAV/prompt manifest")
    prepare.add_argument("--parquet", required=True)
    prepare.add_argument("--export-dir", required=True)
    prepare.add_argument("--output-dir", required=True, help="New directory; never reuse an older case")
    prepare.add_argument("--sample-size", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--prompt-mode", choices=("system_audio", "original"), default="system_audio")
    prepare.add_argument("--model-path", default=DEFAULT_MODEL)
    prepare.add_argument("--enable-thinking", choices=("true", "false"), default="true")
    prepare.add_argument("--tp", type=int, default=4)
    prepare.add_argument("--max-tokens", type=int, default=2048)
    prepare.add_argument("--embedding-check", choices=("true", "false"), default="true",
                         help="Collect actual LLM input embeddings on both engines (default: true)")
    prepare.add_argument("--logits-check", choices=("true", "false"), default="false",
                         help="Capture final worker sampling controls and actual logits/greedy decisions")
    prepare.add_argument("--batch-invariant", choices=("true", "false"), default="false",
                         help="Enable Ascend RL batch invariance on both engines; requires installed custom ops")
    prepare.add_argument("--logits-max-steps", type=int, default=0,
                         help="0 captures through max-tokens/EOS; positive values limit capture. Start with sample-size 4")
    prepare.add_argument("--logits-top-k", type=int, default=100,
                         help="Store top-K logits/IDs plus full-vocab SHA256; default 100, 0 stores full NPZ vectors")
    prepare.add_argument("--forward-check", choices=("true", "false"), default="false",
                         help="Hash actual LLM layer/submodule boundaries; requires logits-check")
    prepare.add_argument("--forward-max-steps", type=int, default=2,
                         help="0=through EOS/max-tokens; 1=prefill; 2=prefill plus first decode; hashes only")
    prepare.set_defaults(function=prepare_case)
    serve = commands.add_parser("serve", help="Send the shared case to an already-running HTTP service")
    serve.add_argument("--case-dir", required=True)
    serve.add_argument("--url", default="http://localhost:28889/v1/chat/completions")
    serve.add_argument("--model", default="qwen3.8")
    serve.add_argument("--timeout-s", type=float, default=600)
    serve.add_argument("--skip-existing", action="store_true")
    serve.add_argument("--continue-on-error", action="store_true")
    serve.set_defaults(function=serve_case)
    compare = commands.add_parser("compare", help="Join saved text by original sample identity")
    compare.add_argument("--case-dir", required=True)
    compare.set_defaults(function=compare_case)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("[aqa-text-ab] ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
