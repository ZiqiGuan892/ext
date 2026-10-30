#!/usr/bin/env python3
"""Locate the first unequal observed LLM boundary on identical token prefixes.

Hashes cover complete tensors, not sampled elements. Equality applies only to
captured boundaries/steps. Attention metadata (including physical cache slots)
is retained as diagnostic evidence, not treated as a causal explanation.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k != "file"}
    return [portable(v) for v in value] if isinstance(value, list) else value


def validate_tree(value):
    if isinstance(value, dict):
        if "unserialized" in value:
            raise ValueError("unobserved forward input/output: " + str(value))
        if value.get("tensor"):
            if (not isinstance(value.get("dtype"), str) or not isinstance(value.get("shape"), list)
                    or any(type(n) is not int or n < 0 for n in value["shape"])
                    or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
                raise ValueError("invalid tensor fingerprint")
        for v in value.values():
            validate_tree(v)
    elif isinstance(value, list):
        for v in value:
            validate_tree(v)


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return {"path": path, "serve": a, "rollout": b}
    if isinstance(a, dict):
        if set(a) != set(b):
            return {"path": path, "serve_keys": sorted(a), "rollout_keys": sorted(b)}
        for key in a:
            result = first_difference(a[key], b[key], path + "." + key)
            if result:
                return result
    elif isinstance(a, list):
        if len(a) != len(b):
            return {"path": path, "serve_length": len(a), "rollout_length": len(b)}
        for i, (x, y) in enumerate(zip(a, b)):
            result = first_difference(x, y, path + "[%d]" % i)
            if result:
                return result
    elif a != b:
        return {"path": path, "serve": a, "rollout": b}
    return None


def compare_case(root, case, rows):
    check = case.get("forward_check", {})
    if not check.get("enabled"):
        raise ValueError("case requires --forward-check true")
    steps, tp = check["max_steps"], case["engine"]["tensor_parallel_size"]
    expected = {r["sample_id"] for r in rows}
    until_eos = check.get("until_eos", False)
    records, terminals, errors = {}, {}, []
    streams = defaultdict(dict)
    for side in ("serve", "rollout"):
        for path in sorted((Path(root) / "forward" / side).glob("*.json")):
            try:
                r = read(path)
                terminal = r.get("kind") == "termination"
                expected_format = "huanyu_forward_termination_v1" if terminal else "huanyu_forward_v1"
                if (r.get("format") != expected_format or r.get("case_id") != case["case_id"]
                        or r.get("side") != side or r.get("tp_size") != tp
                        or r.get("sample_id") not in expected or r.get("status") != "CAPTURED"
                        or type(r.get("tp_rank")) is not int or not 0 <= r["tp_rank"] < tp
                        or type(r.get("step")) is not int or not 0 <= r["step"] < steps):
                    raise ValueError("invalid/incomplete record identity or capture")
                if (r["step"] != len(r["prefix_token_ids"])
                        or r["phase"] != ("prefill" if r["step"] == 0 else "decode")
                        or r["prompt_token_sha256"] != canonical_hash(r["prompt_token_ids"])
                        or r["prefix_token_sha256"] != canonical_hash(r["prefix_token_ids"])):
                    raise ValueError("prefix/phase mismatch")
                if any(type(v) is not int or v < 0 for v in r["prompt_token_ids"] + r["prefix_token_ids"]):
                    raise ValueError("invalid token IDs")
                if terminal:
                    params, selected = r["sampling_params"], r["sampled_token_ids"]
                    if (not until_eos or len(selected) != 1 or type(selected[0]) is not int or selected[0] < 0
                            or params["max_tokens"] != case["sampling"]["max_tokens"]
                            or type(params["ignore_eos"]) is not bool or type(params["min_tokens"]) is not int):
                        raise ValueError("invalid termination sampling evidence")
                    stop_ids = params.get("all_stop_token_ids", params.get("stop_token_ids", [])) or []
                    reason = ("max_tokens" if r["step"] + 1 == params["max_tokens"] else
                              "stop_token" if not params["ignore_eos"] and r["step"] + 1 >= params["min_tokens"]
                              and selected[0] in stop_ids else None)
                    if reason is None or r["terminal_evidence"] != reason:
                        raise ValueError("invalid termination reason")
                    key = (r["sample_id"], side, r["tp_rank"])
                    if key in terminals:
                        raise ValueError("duplicate termination marker")
                    terminals[key] = r
                    continue
                observed = [(e["name"], e["boundary"]) for e in r["events"]]
                if (not r["required"] or not {"llm", "norm", "compute_logits"}.issubset(r["required"])
                        or len(set(observed)) != len(observed)
                        or any((name, boundary) not in observed for name in r["required"] for boundary in ("input", "output"))):
                    raise ValueError("missing/duplicate boundaries")
                for e in r["events"]:
                    validate_tree(e["value"])
                key = (r["sample_id"], side, r["step"], r["tp_rank"])
                if key in records:
                    raise ValueError("duplicate sample/step/rank: use a new case directory")
                # Full-generation traces can contain hundreds of thousands of
                # module trees. Keep a small index and load only one pair of
                # trees for the actual comparison, rather than retaining all
                # activation fingerprints in RAM.
                records[key] = {"path": path, "runtime_id": r["runtime_id"],
                                "prompt_token_sha256": r["prompt_token_sha256"],
                                "prefix_token_sha256": r["prefix_token_sha256"]}
                streams[(r["sample_id"], side, r["tp_rank"])][r["step"]] = records[key]
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({"file": str(path), "error": str(error)})
    samples = []
    for sample in sorted(expected):
        item = {"sample_id": sample, "status": "MATCH_CAPTURED_BOUNDARIES", "steps": [],
                "first_difference": None, "issues": [], "coverage": [], "unpaired_after_termination": 0}
        runtimes = defaultdict(set)
        ends = {}
        if until_eos:
            for side in ("serve", "rollout"):
                endings = set()
                prefix_hashes = {}
                for rank in range(tp):
                    terminal = terminals.get((sample, side, rank))
                    observed_steps = sorted(streams[(sample, side, rank)])
                    if terminal is None:
                        item["issues"].append("missing termination evidence side=%s rank=%d" % (side, rank))
                        continue
                    count = terminal["step"] + 1
                    ends[(side, rank)] = count
                    endings.add((count, terminal["prefix_token_sha256"], tuple(terminal["sampled_token_ids"])))
                    complete = observed_steps == list(range(count))
                    item["coverage"].append({"side": side, "tp_rank": rank, "generated_steps": count,
                                             "captured_steps": len(observed_steps), "complete": complete,
                                             "terminal_evidence": terminal["terminal_evidence"]})
                    if not complete:
                        item["issues"].append("non-contiguous/truncated forward capture side=%s rank=%d" % (side, rank))
                    for index in observed_steps:
                        rec = records[(sample, side, index, rank)]
                        history_key = (terminal["prefix_token_sha256"], index)
                        if history_key not in prefix_hashes:
                            prefix_hashes[history_key] = canonical_hash(terminal["prefix_token_ids"][:index])
                        if (rec["runtime_id"] != terminal["runtime_id"]
                                or rec["prompt_token_sha256"] != terminal["prompt_token_sha256"]
                                or rec["prefix_token_sha256"] != prefix_hashes[history_key]):
                            item["issues"].append("forward history disagrees with termination side=%s rank=%d step=%d" % (side, rank, index))
                if len(endings) > 1:
                    item["issues"].append("TP ranks disagree on termination side=" + side)
        compared_steps = (max([index + 1 for side in ("serve", "rollout") for rank in range(tp)
                              for index in streams[(sample, side, rank)]] + list(ends.values()) + [0])
                          if until_eos else steps)
        for step in range(compared_steps):
            rank_prefixes = defaultdict(set)
            for rank in range(tp):
                a, b = (records.get((sample, side, step, rank)) for side in ("serve", "rollout"))
                if a is None or b is None:
                    if until_eos and all(r is not None or step >= ends.get((side, rank), steps)
                                         for side, r in (("serve", a), ("rollout", b))):
                        if a is not None or b is not None:
                            item["unpaired_after_termination"] += 1
                        continue
                    item["issues"].append("missing step=%d rank=%d" % (step, rank))
                    continue
                try:
                    a, b = read(a["path"]), read(b["path"])
                except (OSError, ValueError) as error:
                    item["issues"].append("cannot reload step=%d rank=%d: %s" % (step, rank, error))
                    continue
                for side, r in (("serve", a), ("rollout", b)):
                    runtimes[(side, rank)].add(r["runtime_id"])
                    rank_prefixes[side].add((r["prompt_token_sha256"], r["prefix_token_sha256"]))
                same_prefix = (a["prompt_token_ids"], a["prefix_token_ids"]) == (b["prompt_token_ids"], b["prefix_token_ids"])
                row = {"step": step, "phase": a["phase"], "tp_rank": rank, "same_prefix": same_prefix,
                       "first_difference": None, "runtime_difference": first_difference(portable(a["manifest"]), portable(b["manifest"])),
                       "attention_metadata_equal": a.get("attention_context") == b.get("attention_context")}
                if a["prompt_token_ids"] != b["prompt_token_ids"]:
                    item["issues"].append("prompt mismatch rank=%d" % rank)
                if same_prefix:
                    if a["required"] != b["required"]:
                        row["first_difference"] = {"reason": "required boundary topology differs"}
                    else:
                        equal_inputs, last_equal = set(), None
                        for event_index, (x, y) in enumerate(zip(a["events"], b["events"])):
                            if (x["name"], x["boundary"]) != (y["name"], y["boundary"]):
                                row["first_difference"] = {"reason": "execution order differs", "serve": x["name"] + "." + x["boundary"],
                                                           "rollout": y["name"] + "." + y["boundary"],
                                                           "event_index": event_index, "last_equal_boundary": last_equal}
                                break
                            delta = first_difference(x["value"], y["value"])
                            if delta:
                                row["first_difference"] = {"name": x["name"], "boundary": x["boundary"], "difference": delta,
                                                           "event_index": event_index, "last_equal_boundary": last_equal,
                                                           "module_input_equal": x["name"] in equal_inputs}
                                break
                            last_equal = {"name": x["name"], "boundary": x["boundary"]}
                            if x["boundary"] == "input":
                                equal_inputs.add(x["name"])
                        if row["first_difference"] is None and len(a["events"]) != len(b["events"]):
                            row["first_difference"] = {"reason": "executed boundary counts differ"}
                    if row["first_difference"]:
                        candidate = {k: row[k] for k in ("step", "phase", "tp_rank", "first_difference")}
                        def order(value):
                            return (value["step"], value["first_difference"].get("event_index", -1), value["tp_rank"])
                        if item["first_difference"] is None or order(candidate) < order(item["first_difference"]):
                            item["first_difference"] = candidate
                item["steps"].append(row)
            if any(len(v) > 1 for v in rank_prefixes.values()):
                item["issues"].append("TP ranks disagree on prefix step=%d" % step)
        if any(len(v) > 1 for v in runtimes.values()):
            item["issues"].append("mixed worker runtimes for one sample/rank")
        if item["first_difference"] or any(r["runtime_difference"] for r in item["steps"]):
            item["status"] = "DIFFERENT"
        if any(not r["same_prefix"] for r in item["steps"]) and item["first_difference"] is None:
            item["issues"].append("different decode prefixes: cannot compare forward outputs")
        if errors or item["issues"]:
            item["status"] = "INCOMPLETE"
        item["full_generation_captured"] = bool(until_eos and not errors and not item["issues"]
                                                and len(item["coverage"]) == 2 * tp
                                                and all(c["complete"] for c in item["coverage"]))
        samples.append(item)
    status = "INCOMPLETE" if errors or not samples or any(s["status"] == "INCOMPLETE" for s in samples) else (
        "DIFFERENT" if any(s["status"] == "DIFFERENT" for s in samples) else "MATCH_CAPTURED_BOUNDARIES")
    return {"summary": {"case_id": case["case_id"], "status": status, "samples": len(samples),
                        "max_steps": steps, "until_eos": until_eos, "errors": errors,
                        "scope": "Full tensor hashes at observed boundaries on common prefixes; no activation values stored. "
                                 "Matching hashes do not prove unobserved kernels/cache states or later steps equal. "
                                 "Runtime/metadata differences alone do not establish causation."}, "samples": samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("aqa_forward_case", Path(__file__).with_name("aqa_text_ab.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        root, case, rows = module.load_case(args.case_dir)
        report = compare_case(root, case, rows)
        path = root / "forward_comparison.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        for row in report["samples"]:
            print(row["sample_id"], row["status"], "first=", row["first_difference"])
        print("report=" + str(path))
        return {"MATCH_CAPTURED_BOUNDARIES": 0, "DIFFERENT": 1, "INCOMPLETE": 2}[report["summary"]["status"]]
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("[huanyu-forward-ab] ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
