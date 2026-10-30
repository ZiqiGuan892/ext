#!/usr/bin/env python3
"""Repeat exact AQA requests and compare text (full logits are opt-in).

Python 3.8+. prepare uses pyarrow; optional logits comparison uses numpy. No training or extra
model forward is performed. Probe row IDs are unique invocation IDs, with an
explicit source_sample_id/repetition mapping. Original WAVs are not copied.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid


def sibling(name):
    spec = importlib.util.spec_from_file_location("huanyu_repeat_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    ab = sibling("aqa_text_ab")
    source, original, source_rows = ab.load_case(args.source_case)
    if original.get("serve_repeat"):
        raise ValueError("source must be the original A/B case, not a repeat case")
    if args.repeats < 2 or not args.samples or len(set(args.samples)) != len(args.samples):
        raise ValueError("require distinct sample IDs and at least two repetitions")
    selected = {row["sample_id"]: row for row in source_rows}
    missing = set(args.samples) - set(selected)
    if missing:
        raise ValueError("requested samples absent from source case: " + str(sorted(missing)))
    if ab.sha256(original["subset_parquet"]) != original["subset_parquet_sha256"]:
        raise ValueError("source subset Parquet changed")
    source_table = pq.read_table(original["subset_parquet"])
    table_rows = {r["extra_info"]["huanyu_text_ab"]["sample_id"]: r for r in source_table.to_pylist()}
    root = Path(args.output_dir).expanduser().resolve()
    case = copy.deepcopy(original)
    case.update(case_id=uuid.uuid4().hex, sample_size=len(args.samples) * args.repeats)
    case["engine"]["batch_invariant"] = args.batch_invariant == "true"
    logits_check = getattr(args, "logits_check", "false") == "true"
    case["embedding_check"] = dict(enabled=logits_check, scope="llm_inputs_embeds", strict=True,
                                   expected_tp_size=case["engine"]["tensor_parallel_size"])
    case["logits_check"] = dict(enabled=logits_check, max_steps=case["sampling"]["max_tokens"] if logits_check else 0,
                                scope="model_logits_sampler_input_greedy_input", strict=True)
    case["serve_repeat"] = dict(source_case=str(source), source_case_id=original["case_id"],
                                source_case_sha256=ab.sha256(source / "case.json"),
                                samples=args.samples, repetitions=args.repeats, order="round_robin",
                                concurrency=1, baseline_repetition=1)
    # Validate engine/sampling controls before creating any output directory.
    sibling("launch_huanyu_aqa_text_ab").build_launch(root, case, Path("/unused"), "serve")
    client = ab.sibling("request_aqa_validation_wav")
    for sample in args.samples:
        row = selected[sample]
        prompt, truth, paths, payloads = client.load_inputs(Path(case["export_dir"]), row)
        if (prompt != row["original_prompt"] or truth != row["ground_truth"] or paths != row["wav_paths"]
                or [hashlib.sha256(p).hexdigest() for p in payloads] != row["wav_sha256"]):
            raise ValueError("source export changed: " + sample)
        saved = table_rows[sample]["extra_info"]["huanyu_text_ab"]
        if any(saved.get(k) != row[k] for k in ab.IDENTITY_KEYS):
            raise ValueError("source manifest and subset disagree: " + sample)
    rows, rewritten = [], []
    for repetition in range(1, args.repeats + 1):
        for sample in args.samples:
            row = copy.deepcopy(selected[sample])
            row.update(sample_id="row_{:06d}".format(len(rows)), case_id=case["case_id"],
                       source_sample_id=sample, repetition=repetition)
            record = copy.deepcopy(table_rows[sample])
            record["extra_info"]["huanyu_text_ab"] = {
                k: row[k] for k in ab.IDENTITY_KEYS + ("prompt_sha256",)
            }
            rows.append(row)
            rewritten.append(record)
    root.mkdir(parents=True, exist_ok=False)
    subset = root / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(rewritten, schema=source_table.schema), subset)
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    case.update(subset_parquet=str(subset), subset_parquet_sha256=ab.sha256(subset),
                manifest_sha256=ab.sha256(manifest))
    ab.write_json(root / "case.json", case)
    print("[huanyu-repeat] case={} samples={} repeats={} requests={} BS=1 logits_check={}".format(
        root, args.samples, args.repeats, len(rows), logits_check))
    return 0


def load_case(directory):
    root, case, rows = sibling("aqa_text_ab").load_case(directory)
    spec = case.get("serve_repeat") or {}
    if spec.get("concurrency") != 1 or spec.get("order") != "round_robin" or spec.get("baseline_repetition") != 1:
        raise ValueError("not a serial serve repeat case")
    expected = [(sample, i) for i in range(1, spec["repetitions"] + 1) for sample in spec["samples"]]
    if [(r["source_sample_id"], r["repetition"]) for r in rows] != expected:
        raise ValueError("repeat mapping/count/order mismatch")
    for sample in spec["samples"]:
        group = [r for r in rows if r["source_sample_id"] == sample]
        for row in group:
            if any(row[k] != group[0][k] for k in ("prompt", "prompt_sha256", "wav_paths", "wav_sha256")):
                raise ValueError("repetitions must have identical prompt and WAV: " + sample)
    logits_check = case.get("logits_check", {}).get("enabled")
    if type(logits_check) is not bool or case.get("embedding_check", {}).get("enabled") is not logits_check:
        raise ValueError("repeat case must enable both tensor probes or neither")
    if logits_check and case["logits_check"]["max_steps"] != case["sampling"]["max_tokens"]:
        raise ValueError("logits mode requires capture through the full generation")
    return root, case, rows


def request(args):
    root, case, rows = load_case(args.case_dir)
    # The shared client makes blocking calls one at a time; no retries, resume
    # or parallel tasks can silently mix different serving processes.
    args.skip_existing = False
    args.continue_on_error = False
    result = sibling("aqa_text_ab").serve_case(args)
    print("[huanyu-repeat] responses=" + str(root / "serve"))
    return result


def _embeddings(root, case, rows):
    helper = sibling("compare_huanyu_prompt_embeddings")
    wanted = {row["sample_id"]: row for row in rows}
    captures, errors = {}, []
    for path in sorted((root / "embeddings" / "serve").glob("*.json")):
        try:
            record = helper._read(path)
            row = wanted[record["sample_id"]]
            rank = helper._validate_record(path, record, root, case, row, "serve", case["engine"]["tensor_parallel_size"])
            arrays = helper._load_arrays(path, record, root)
            key = (row["sample_id"], rank)
            if key in captures:
                raise ValueError("duplicate repeat embedding capture")
            captures[key] = dict(dtype=arrays["dtype"], shape=arrays["shape"], **{
                k: hashlib.sha256(arrays[k].tobytes()).hexdigest()
                for k in ("embedding_bytes", "input_ids", "positions", "audio_mask")})
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(dict(file=str(path), error=str(error)))
    return captures, errors


def _labels(value):
    """Name the two requests honestly in a reused A/B report; no files change."""
    if isinstance(value, dict):
        return {k.replace("serve", "baseline").replace("rollout", "repeat"): _labels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_labels(v) for v in value]
    return {"serve": "baseline", "rollout": "repeat"}.get(value, value) if isinstance(value, str) else value


def compare(args):
    root, case, rows = load_case(args.case_dir)
    if not case["logits_check"]["enabled"]:
        return compare_text(root, case, rows)
    ab, logits = sibling("aqa_text_ab"), sibling("compare_huanyu_logits")
    target = root / "repeat_summary.json"
    if target.exists():
        raise FileExistsError("repeat_summary.json already exists; preserve the previous report")
    records, runtimes, errors = logits._load_records(root, case, rows, sides=("serve",))
    embeddings, embedding_errors = _embeddings(root, case, rows)
    errors.extend(embedding_errors)
    tp, groups, summaries = case["engine"]["tensor_parallel_size"], defaultdict(list), []
    for rank in range(tp):
        workers = {r["runtime_id"] for row in rows
                   for _, r in records.get((row["sample_id"], "serve", rank), {}).values()}
        if len(workers) > 1:
            errors.append(dict(tp_rank=rank, error="the repeat run crossed live worker instances",
                               runtime_ids=sorted(workers)))
    for row in rows:
        groups[row["source_sample_id"]].append(row)
    details = root / "repeat_comparisons"
    details.mkdir(exist_ok=False)
    for sample, group in groups.items():
        baseline = group[0]
        entries = []
        for row in group:
            issues, control_differences = [], []
            paired, runtime_pairs = {}, {}
            input_equal = True
            for rank in range(tp):
                left = records.get((baseline["sample_id"], "serve", rank), {})
                right = records.get((row["sample_id"], "serve", rank), {})
                paired[(sample, "serve", rank)] = left
                paired[(sample, "rollout", rank)] = right
                observed_workers = []
                for label, steps in (("serve", left), ("rollout", right)):
                    ids = {rec["runtime_id"] for _, rec in steps.values()}
                    for runtime_id in ids:
                        runtime = runtimes.get(("serve", runtime_id))
                        if runtime is None:
                            continue  # Shared comparator reports missing runtime.
                        runtime_pairs[(label, runtime_id)] = runtime
                        observed_workers.append((runtime_id, runtime.get("host"), runtime.get("pid")))
                        actual = (runtime.get("determinism") or {}).get("environment") or {}
                        required = {"ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1",
                                    "LCCL_DETERMINISTIC": "1", "VLLM_BATCH_INVARIANT":
                                    "1" if case["engine"].get("batch_invariant") else "0"}
                        diff = {k: dict(expected=v, actual=actual.get(k)) for k, v in required.items() if actual.get(k) != v}
                        if actual.get("HCCL_DETERMINISTIC") not in ("true", "strict"):
                            diff["HCCL_DETERMINISTIC"] = dict(expected="true or strict", actual=actual.get("HCCL_DETERMINISTIC"))
                        if diff:
                            control_differences.append(dict(tp_rank=rank, differences=diff))
                if len(set(observed_workers)) != 1:
                    issues.append(dict(tp_rank=rank, error="requests did not use one verified live worker instance"))
                a = embeddings.get((baseline["sample_id"], rank))
                b = embeddings.get((row["sample_id"], rank))
                if a is None or b is None:
                    issues.append(dict(tp_rank=rank, error="missing actual prompt embedding capture"))
                    input_equal = False
                elif a != b:
                    input_equal = False
            report = logits.compare_case(root, case, [dict(sample_id=sample)], loaded=(paired, runtime_pairs, []))
            item = report["samples"][0]
            outputs = []
            for request_row in (baseline, row):
                try:
                    response, status = ab._read_result(root, "serve", request_row)
                    if status != "complete":
                        raise ValueError("missing or failed HTTP response")
                    expected_id = "huanyuab.{}.{}".format(case["case_id"], request_row["sample_id"])
                    if response.get("request", {}).get("request_id") != expected_id:
                        raise ValueError("HTTP/probe request ID mismatch")
                    observed = response.get("raw_response", {}).get("id")
                    if not observed or expected_id not in observed:
                        raise ValueError("HTTP response does not identify the captured request")
                    outputs.append(response)
                except (ValueError, KeyError, OSError) as error:
                    issues.append(dict(request=request_row["sample_id"], error=str(error)))
            text_equal = None
            if len(outputs) == 2:
                text_equal = all(outputs[0].get(k) == outputs[1].get(k) for k in
                                 ("reply", "reasoning_content", "finish_reason", "truncated"))
                left_body, right_body = [dict(o["request"]) for o in outputs]
                for body in (left_body, right_body):
                    body.pop("request_id", None)
                if left_body != right_body:
                    control_differences.append(dict(error="HTTP bodies differ beyond request ID"))
            status = item["status"]
            if errors or issues or status in ("INCOMPLETE", "MATCH_CAPTURED_PREFIX"):
                status = "INCOMPLETE"
            elif control_differences or not input_equal or text_equal is False:
                status = "DIFFERENT"
            # Keep numeric equality separate from the combined audit verdict:
            # identical vectors with an overridden sampling control still fail
            # the audit, but are not reported as different numeric vectors.
            logits_equal = None
            steps = item["prefill"] + item["decode"] + item["decode_different_prefix"]
            if any(not stage.get("bytes_equal", False) for step in steps for stage in step["logits"].values()):
                logits_equal = False
            elif item["full_generation_captured"] and not errors:
                logits_equal = True
            entry = dict(source_sample_id=sample, repetition=row["repetition"], request_sample_id=row["sample_id"],
                         baseline_request_sample_id=baseline["sample_id"], status=status,
                         text_exact_equal=text_equal, inputs_exact_equal=input_equal,
                         logits_status=item["status"], logits_exact_equal=logits_equal,
                         first_different_logits=item["first_different_logits"],
                         first_different_token=_labels(item["first_different_token"]),
                         issues=issues, control_differences=control_differences,
                         response=str(root / "serve" / (row["sample_id"] + ".json")))
            if len(outputs) == 2:
                entry["response_content_sha256"] = logits.canonical_hash({
                    k: outputs[1].get(k) for k in ("reply", "reasoning_content", "finish_reason", "truncated")})
            entries.append(entry)
            ab.write_json(details / (sample + "-{:02d}.json".format(row["repetition"])),
                          dict(comparison=entry, logits=_labels(item)))
            print("[huanyu-repeat] {} repeat={:02d} {} text={} logits={} first={}".format(
                sample, row["repetition"], status, text_equal, item["status"], item["first_different_logits"]), flush=True)
        statuses = [e["status"] for e in entries]
        summaries.append(dict(sample_id=sample, requests=len(group),
                              status="INCOMPLETE" if "INCOMPLETE" in statuses else "DIFFERENT" if "DIFFERENT" in statuses else "PASS",
                              distinct_response_variants=len({e["response_content_sha256"] for e in entries
                                                              if "response_content_sha256" in e}),
                              matching_baseline_text_requests=sum(e["text_exact_equal"] is True for e in entries),
                              matching_baseline_logits_requests=sum(e["logits_exact_equal"] is True for e in entries),
                              repetitions=entries))
    statuses = [s["status"] for s in summaries]
    status = "INCOMPLETE" if errors or "INCOMPLETE" in statuses else "DIFFERENT" if "DIFFERENT" in statuses else "PASS"
    ab.write_json(target, dict(case_id=case["case_id"], status=status, expected_requests=len(rows), samples=summaries,
                               errors=errors, scope="Same live serve engine; each sample compared with its first request. "
                               "Text equality is not correctness. After token divergence, logits have different prefixes."))
    print("[huanyu-repeat] {} report={}".format(status, target))
    return {"PASS": 0, "DIFFERENT": 1, "INCOMPLETE": 2}[status]


def compare_text(root, case, rows):
    """Check recorded HTTP bodies and outputs, without importing tensor readers.

    This intentionally makes no claim about live worker identity, final sampler
    metadata, numerical inputs or logits. Those require the opt-in probes.
    """
    ab = sibling("aqa_text_ab")
    client = ab.sibling("request_aqa_validation_wav")
    target = root / "repeat_summary.json"
    if target.exists():
        raise FileExistsError("repeat_summary.json already exists; preserve the previous report")
    wanted = {row["sample_id"] for row in rows}
    errors = [{"file": str(p), "error": "unexpected response ID"}
              for p in (root / "serve").glob("*.json") if p.stem not in wanted]
    groups, samples = defaultdict(list), []
    for row in rows:
        groups[row["source_sample_id"]].append(row)
    for sample, group in groups.items():
        outputs, entries, variants = {}, [], set()
        for row in group:
            entry = dict(source_sample_id=sample, repetition=row["repetition"], request_sample_id=row["sample_id"],
                         status="INCOMPLETE", text_exact_equal=None, termination_equal=None,
                         response=str(root / "serve" / (row["sample_id"] + ".json")), issues=[])
            try:
                response, state = ab._read_result(root, "serve", row)
                if state != "complete":
                    raise ValueError("missing or failed HTTP response")
                payloads = [Path(p).read_bytes() for p in row["wav_paths"]]
                if [hashlib.sha256(p).hexdigest() for p in payloads] != row["wav_sha256"]:
                    raise ValueError("WAV changed since preparation")
                body = ab.build_case_request(row["prompt"], payloads, response["model"],
                                             case["sampling"], case["enable_thinking"])
                body["request_id"] = "huanyuab.{}.{}".format(case["case_id"], row["sample_id"])
                payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                expected_hash = hashlib.sha256(response["url"].encode("utf-8") + b"\n" + payload).hexdigest()
                if response.get("request") != ab.sanitized_request(body) or response.get("request_sha256") != expected_hash:
                    raise ValueError("recorded HTTP request differs from the frozen case")
                raw = response.get("raw_response") or {}
                if body["request_id"] not in (raw.get("id") or ""):
                    raise ValueError("HTTP response request ID mismatch")
                reply, reasoning, finish = client.extract_reply(raw)
                if ((reply, reasoning, finish) != tuple(response.get(k) for k in
                                                       ("reply", "reasoning_content", "finish_reason"))
                        or response.get("truncated") != (finish == "length")):
                    raise ValueError("raw HTTP response differs from recorded reply")
                outputs[row["repetition"]] = response
                variants.add(json.dumps([reply, reasoning], ensure_ascii=False))
            except (ValueError, KeyError, TypeError, OSError) as error:
                entry["issues"].append(str(error))
            entries.append(entry)
        baseline = outputs.get(1)
        for entry in entries:
            current = outputs.get(entry["repetition"])
            if baseline is not None and current is not None:
                if any(current.get(k) != baseline.get(k) for k in ("model", "url")):
                    entry["issues"].append("HTTP model name or endpoint changed across requests")
                entry["text_exact_equal"] = all(current.get(k) == baseline.get(k) for k in ("reply", "reasoning_content"))
                entry["termination_equal"] = all(current.get(k) == baseline.get(k) for k in ("finish_reason", "truncated"))
                entry["status"] = ("INCOMPLETE" if entry["issues"] or errors else
                                   "PASS" if entry["text_exact_equal"] and entry["termination_equal"] else "DIFFERENT")
            print("[huanyu-repeat] {} repeat={:02d} {} text={} logits=NOT_COLLECTED".format(
                sample, entry["repetition"], entry["status"], entry["text_exact_equal"]), flush=True)
        statuses = [e["status"] for e in entries]
        status = "INCOMPLETE" if "INCOMPLETE" in statuses else "DIFFERENT" if "DIFFERENT" in statuses else "PASS"
        samples.append(dict(sample_id=sample, requests=len(group), complete_http_responses=len(outputs),
                            distinct_text_variants=len(variants), status=status,
                            matching_baseline_text_requests=sum(e["text_exact_equal"] is True for e in entries),
                            repetitions=entries))
    statuses = [s["status"] for s in samples]
    status = "INCOMPLETE" if errors or "INCOMPLETE" in statuses else "DIFFERENT" if "DIFFERENT" in statuses else "PASS"
    ab.write_json(target, dict(case_id=case["case_id"], status=status, scope="text_only",
                               expected_requests=len(rows), samples=samples, errors=errors,
                               logits_status="NOT_COLLECTED", embeddings_status="NOT_COLLECTED",
                               worker_identity_status="NOT_OBSERVED", final_sampling_status="NOT_OBSERVED",
                               note="PASS means HTTP replies/reasoning and termination match for identical requested inputs. "
                               "It does not prove final engine settings, weights, embeddings, logits or answer correctness."))
    print("[huanyu-repeat] {} scope=text_only report={}".format(status, target))
    return {"PASS": 0, "DIFFERENT": 1, "INCOMPLETE": 2}[status]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-case", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--samples", nargs="+", default=["row_000899", "row_000418"])
    prep.add_argument("--repeats", type=int, default=20)
    prep.add_argument("--batch-invariant", choices=("true", "false"), default="false")
    prep.add_argument("--logits-check", choices=("true", "false"), default="false",
                      help="Default false: compare only text; true enables full logits and input embeddings")
    prep.set_defaults(func=prepare)
    send = commands.add_parser("request")
    send.add_argument("--case-dir", required=True)
    send.add_argument("--url", default="http://localhost:28889/v1/chat/completions")
    send.add_argument("--model", default="qwen3.8")
    send.add_argument("--timeout-s", type=float, default=3600)
    send.set_defaults(func=request)
    check = commands.add_parser("compare")
    check.add_argument("--case-dir", required=True)
    check.set_defaults(func=compare)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("[huanyu-repeat] ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
