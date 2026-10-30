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
    records, errors = {}, []
    for side in ("serve", "rollout"):
        for path in sorted((Path(root) / "forward" / side).glob("*.json")):
            try:
                r = read(path)
                if (r.get("format") != "huanyu_forward_v1" or r.get("case_id") != case["case_id"]
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
                records[key] = r
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({"file": str(path), "error": str(error)})
    samples = []
    for sample in sorted(expected):
        item = {"sample_id": sample, "status": "MATCH_CAPTURED_BOUNDARIES", "steps": [],
                "first_difference": None, "issues": []}
        runtimes = defaultdict(set)
        for step in range(steps):
            rank_prefixes = defaultdict(set)
            for rank in range(tp):
                a, b = (records.get((sample, side, step, rank)) for side in ("serve", "rollout"))
                if a is None or b is None:
                    item["issues"].append("missing step=%d rank=%d" % (step, rank))
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
                        for x, y in zip(a["events"], b["events"]):
                            if (x["name"], x["boundary"]) != (y["name"], y["boundary"]):
                                row["first_difference"] = {"reason": "execution order differs", "serve": x["name"] + "." + x["boundary"],
                                                           "rollout": y["name"] + "." + y["boundary"]}
                                break
                            delta = first_difference(x["value"], y["value"])
                            if delta:
                                row["first_difference"] = {"name": x["name"], "boundary": x["boundary"], "difference": delta}
                                break
                        if row["first_difference"] is None and len(a["events"]) != len(b["events"]):
                            row["first_difference"] = {"reason": "executed boundary counts differ"}
                    if row["first_difference"] and item["first_difference"] is None:
                        item["first_difference"] = {k: row[k] for k in ("step", "phase", "tp_rank", "first_difference")}
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
        samples.append(item)
    status = "INCOMPLETE" if errors or not samples or any(s["status"] == "INCOMPLETE" for s in samples) else (
        "DIFFERENT" if any(s["status"] == "DIFFERENT" for s in samples) else "MATCH_CAPTURED_BOUNDARIES")
    return {"summary": {"case_id": case["case_id"], "status": status, "samples": len(samples),
                        "max_steps": steps, "errors": errors,
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
