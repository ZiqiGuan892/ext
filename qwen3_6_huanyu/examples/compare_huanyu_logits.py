#!/usr/bin/env python3
"""Compare actual prefill/decode logits and sampling controls on common prefixes.

CPU/numpy only. Missing ranks, duplicate steps, corrupt files and unobserved
boundaries never pass. A limited capture is reported as MATCH_CAPTURED_PREFIX,
not as proof that the complete generations match.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np


STAGES = ("model_logits", "sampler_input", "greedy_input")
PARAM_DIAGNOSTICS = {"logprobs", "prompt_logprobs", "logprob_token_ids", "detokenize",
                     "skip_special_tokens", "spaces_between_special_tokens", "output_kind",
                     "include_stop_str_in_output", "skip_clone"}
META_DIAGNOSTICS = {"max_num_logprobs", "logprob_token_ids", "generators"}
DTYPES = {"torch.bfloat16": (np.dtype("<u2"), 2), "torch.float16": (np.dtype("<f2"), 2),
          "torch.float32": (np.dtype("<f4"), 4), "torch.float64": (np.dtype("<f8"), 8)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def differences(left, right, excluded=()):
    return {key: {"serve": left.get(key, "<absent>"), "rollout": right.get(key, "<absent>")}
            for key in sorted(set(left) | set(right))
            if key not in excluded and left.get(key, "<absent>") != right.get(key, "<absent>")}


def sampling_value(key, value):
    """Normalize only vLLM's two documented disabled-top-k spellings.

    Keep nonzero top-k changes visible even in greedy runs, and retain the
    original observed params in capture files and representation diagnostics.
    """
    if key == "top_k" and type(value) is int and value in (-1, 0):
        return 0
    return value


def sampling_differences(left, right):
    return differences({k: sampling_value(k, v) for k, v in left.items()},
                       {k: sampling_value(k, v) for k, v in right.items()}, PARAM_DIAGNOSTICS)


def _portable(value):
    # Source paths can differ across hosts. The actual code SHA/class must match.
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items() if k != "file"}
    return [_portable(x) for x in value] if isinstance(value, list) else value


def _opaque(value):
    if isinstance(value, dict):
        return "unserialized" in value or any(_opaque(v) for v in value.values())
    return any(_opaque(v) for v in value) if isinstance(value, list) else False


def _load_topk(summary):
    shape, count = summary.get("shape"), summary.get("requested_top_k")
    if (summary.get("storage") != "topk_json" or summary.get("dtype") not in DTYPES
            or not isinstance(shape, list) or len(shape) != 2 or shape[0] != 1
            or type(shape[1]) is not int or shape[1] < 1 or type(count) is not int or count < 1
            or summary.get("sha256_scope") != "full_vocab_original_dtype_bytes"
            or not re.fullmatch(r"[0-9a-f]{64}", summary.get("sha256", ""))):
        raise ValueError("invalid top-K shape/dtype/full-vocab fingerprint")
    payload = summary["topk"]
    ids, values = payload["token_ids_local"], payload["logits"]
    expected = min(count, shape[1])
    if (not isinstance(ids, list) or not isinstance(values, list) or len(ids) != expected
            or len(values) != expected or summary.get("stored_shape") != [1, expected]
            or any(type(i) is not int or not 0 <= i < shape[1] for i in ids) or len(set(ids)) != len(ids)
            or canonical_hash(payload) != summary.get("topk_sha256")):
        raise ValueError("invalid/missing/corrupt top-K payload")
    if any(type(v) not in (int, float) and v != "-inf" for v in values):
        raise ValueError("invalid top-K value type")
    numeric = np.array([float(v) for v in values], dtype=np.float64)
    if np.isnan(numeric).any() or np.isposinf(numeric).any() or not np.isfinite(numeric[0]):
        raise ValueError("invalid top-K logits")
    if sorted(zip(ids, numeric), key=lambda x: (-x[1], x[0])) != list(zip(ids, numeric)):
        raise ValueError("top-K not in score/token-ID order")
    top10 = [{"token_id_local": i, "logit": v} for i, v in zip(ids[:10], values[:10])]
    if summary.get("argmax_local") != ids[0] or summary.get("top10") != top10:
        raise ValueError("top-K does not match observed argmax/top10")
    return {"token_ids": np.array(ids, dtype=np.int64), "values": numeric}


def load_vectors(path, record):
    if record.get("tensor_storage") == "topk_json":
        if record.get("tensor_file"):
            raise ValueError("unexpected dense payload in top-K record")
        return {stage: _load_topk(record["logits"][stage]) for stage in STAGES}
    if any(s.get("storage") == "topk_json" for s in record["logits"].values()):
        raise ValueError("top-K record missing tensor_storage tag")
    name = record.get("tensor_file")
    if not name:
        if record["tp_rank"] == 0 or record["enable_reduce_sample"]:
            raise ValueError("missing full logits payload")
        return {}
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("tensor_file must be a sibling filename")
    tensor_path = path.parent / name
    tensor_path.resolve().relative_to(path.parent.resolve())
    if sha(tensor_path.read_bytes()) != record.get("tensor_file_sha256"):
        raise ValueError("tensor file SHA256 mismatch")
    result = {}
    with np.load(tensor_path, allow_pickle=False) as arrays:
        for stage in STAGES:
            summary = record["logits"][stage]
            dtype, width = DTYPES[summary["dtype"]]
            shape = summary["shape"]
            raw = arrays[record["tensor_keys"][stage]]
            if (len(shape) != 2 or shape[0] != 1 or shape[1] < 1 or raw.dtype != np.uint8
                    or raw.shape != (1, shape[1] * width) or sha(raw.tobytes()) != summary["sha256"]):
                raise ValueError("invalid vector shape/dtype/bytes for " + stage)
            values = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
            if summary["dtype"] == "torch.bfloat16":
                values = (values.astype(np.uint32) << 16).view(np.float32)
            if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
                raise ValueError("invalid logits values")
            result[stage] = values.astype(np.float64)
    return result


def vector_difference(left, right, a=None, b=None):
    equal = all(left[key] == right[key] for key in ("dtype", "shape", "sha256"))
    result = {"bytes_equal": equal, "serve_dtype": left["dtype"], "rollout_dtype": right["dtype"],
              "serve_shape": left["shape"], "rollout_shape": right["shape"],
              "serve_top10": left["top10"], "rollout_top10": right["top10"],
              "serve_top1_top2_margin": left.get("top1_top2_margin"),
              "rollout_top1_top2_margin": right.get("top1_top2_margin"),
              "equality_scope": "full_vocab_bytes_sha256",
              "numeric_comparison_scope": "not_computed"}
    if a is not None and b is not None and (isinstance(a, dict) or isinstance(b, dict)):
        if left["shape"] != right["shape"]:
            return result
        def sparse(value):
            if isinstance(value, dict):
                return value["token_ids"], value["values"]
            flat = value.reshape(-1)
            return np.arange(flat.size), flat
        ai, av = sparse(a)
        bi, bv = sparse(b)
        common, aix, bix = np.intersect1d(ai, bi, assume_unique=True, return_indices=True)
        x, y = av[aix], bv[bix]
        finite = np.isfinite(x) & np.isfinite(y)
        delta = np.abs(x[finite] - y[finite])
        a_only, b_only = np.setdiff1d(ai, bi), np.setdiff1d(bi, ai)
        result.update(numeric_comparison_scope="stored_token_id_intersection",
                      compared_token_count=int(common.size), serve_stored_token_count=int(ai.size),
                      rollout_stored_token_count=int(bi.size), stored_token_ids_equal=not a_only.size and not b_only.size,
                      serve_only_token_count=int(a_only.size), rollout_only_token_count=int(b_only.size),
                      serve_only_token_ids_first100=a_only[:100].tolist(), rollout_only_token_ids_first100=b_only[:100].tolist(),
                      topk_max_abs_difference_finite=float(delta.max()) if delta.size else None,
                      topk_mean_abs_difference_finite=float(delta.mean()) if delta.size else None,
                      topk_numeric_mismatched_elements=int(np.count_nonzero(x != y)),
                      topk_masking_differences=int(np.count_nonzero(np.isneginf(x) != np.isneginf(y))))
        return result
    if a is not None and b is not None and a.shape == b.shape:
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.abs(a[finite] - b[finite])
        indices = np.argwhere(a != b)
        result.update(numeric_comparison_scope="full_vocab", max_abs_difference_finite=float(delta.max()) if delta.size else 0.0,
                      mean_abs_difference_finite=float(delta.mean()) if delta.size else 0.0,
                      masking_differences=int(np.count_nonzero(np.isneginf(a) != np.isneginf(b))),
                      numeric_mismatched_elements=int(indices.shape[0]),
                      first_different_token_id=int(indices[0, 1]) if indices.size else None)
    return result


def _load_records(root, case, rows, sides=("serve", "rollout")):
    expected = {r["sample_id"] for r in rows}
    tp = case["engine"]["tensor_parallel_size"]
    records, runtimes, errors = defaultdict(dict), {}, []
    for side in sides:
        for path in sorted((root / "logits" / side).glob("*.json")):
            try:
                r = read(path)
                if (r.get("format") != "huanyu_logits_v1" or r.get("case_id") != case["case_id"]
                        or r.get("side") != side or r.get("tp_size") != tp
                        or type(r.get("tp_rank")) is not int or not 0 <= r["tp_rank"] < tp):
                    raise ValueError("record identity/TP mismatch")
                if r.get("status") != "complete":
                    raise ValueError("incomplete capture: " + str(r.get("error")))
                if r["kind"] == "runtime":
                    key = (side, r["runtime_id"])
                    if key in runtimes:
                        raise ValueError("duplicate runtime")
                    runtimes[key] = r
                    continue
                if r["kind"] != "step" or r.get("sample_id") not in expected:
                    raise ValueError("unknown record kind/sample")
                step = r["step"]
                if (type(step) is not int or not 0 <= step < case["logits_check"]["max_steps"]
                        or step != len(r["prefix_token_ids"])
                        or r.get("phase") != ("prefill" if step == 0 else "decode")
                        or r["prompt_token_sha256"] != canonical_hash(r["prompt_token_ids"])
                        or r["prefix_token_sha256"] != canonical_hash(r["prefix_token_ids"])
                        or len(r["sampled_token_ids"]) != 1):
                    raise ValueError("invalid step/prefix/phase")
                if any(type(x) is not int or x < 0 for x in r["prompt_token_ids"] + r["prefix_token_ids"] + r["sampled_token_ids"]):
                    raise ValueError("invalid token IDs")
                key = (r["sample_id"], side, r["tp_rank"])
                if step in records[key]:
                    raise ValueError("duplicate sample/TP/step (mixed reruns)")
                records[key][step] = (path, r)
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append({"file": str(path), "error": str(error)})
    return records, runtimes, errors


def compare_case(root, case, rows, *, loaded=None):
    root = Path(root)
    if not case.get("logits_check", {}).get("enabled"):
        raise ValueError("case was not prepared with --logits-check true")
    # Repeatability tests reuse this validation on two requests from one serve
    # process. Paths still refer to the original, unmodified capture files.
    records, runtimes, errors = _load_records(root, case, rows) if loaded is None else loaded
    tp = case["engine"]["tensor_parallel_size"]
    samples = []
    for row in rows:
        sample = row["sample_id"]
        item = {"sample_id": sample, "status": "MATCH_CAPTURED_PREFIX", "issues": [],
                "prefill": [], "decode": [], "decode_different_prefix": [], "first_different_logits": None,
                "first_different_token": None, "post_divergence_steps": 0,
                "sampling_parameter_differences": [], "runtime_differences": []}
        full = True
        for rank in range(tp):
            sides = [records.get((sample, side, rank), {}) for side in ("serve", "rollout")]
            for side, entries in zip(("serve", "rollout"), sides):
                if not entries or sorted(entries) != list(range(len(entries))):
                    item["issues"].append({"side": side, "tp_rank": rank, "error": "missing prefill or non-contiguous decode steps"})
                    full = False
                    continue
                history, prompt = [], None
                for step, (path, record) in sorted(entries.items()):
                    if record["prefix_token_ids"] != history or (prompt is not None and prompt != record["prompt_token_ids"]):
                        item["issues"].append({"side": side, "tp_rank": rank, "error": "broken actual token history"})
                    history += record["sampled_token_ids"]
                    prompt = record["prompt_token_ids"]
                    try:
                        # Validate all payloads, including those after divergence.
                        load_vectors(path, record)
                    except (ValueError, KeyError, TypeError, OSError) as error:
                        item["issues"].append({"file": str(path), "error": str(error)})
                terminal = entries[max(entries)][1].get("terminal_evidence")
                if terminal not in ("max_tokens", "stop_token"):
                    full = False
                    if len(entries) < case["logits_check"]["max_steps"]:
                        item["issues"].append({"side": side, "tp_rank": rank,
                                               "error": "capture ended before its limit without generation termination evidence"})
            left, right = sides
            if not left or not right:
                continue
            if len(left) != len(right):
                full = False
            runtime_keys = [{r["runtime_id"] for _, r in entries.values()} for entries in sides]
            if any(len(keys) != 1 for keys in runtime_keys):
                item["issues"].append({"tp_rank": rank, "error": "sample moved between runtime instances"})
                continue
            a_runtime = runtimes.get(("serve", next(iter(runtime_keys[0]))))
            b_runtime = runtimes.get(("rollout", next(iter(runtime_keys[1]))))
            if a_runtime is None or b_runtime is None:
                item["issues"].append({"tp_rank": rank, "error": "missing live worker runtime record"})
                continue
            runtime_keys = ("config", "enable_reduce_sample", "sampler_forward", "greedy_sample", "sampler_sample", "compute_logits", "llm_forward", "precision_env", "determinism")
            runtime_diff = differences(_portable({k: a_runtime.get(k) for k in runtime_keys}),
                                       _portable({k: b_runtime.get(k) for k in runtime_keys}))
            if runtime_diff:
                item["runtime_differences"].append({"tp_rank": rank, "differences": runtime_diff})
            if case["engine"].get("batch_invariant"):
                expected_env = {"VLLM_BATCH_INVARIANT": "1", "HCCL_DETERMINISTIC": "strict",
                                "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1",
                                "LCCL_DETERMINISTIC": "1"}
                for side, runtime in (("serve", a_runtime), ("rollout", b_runtime)):
                    actual = runtime.get("determinism") or {}
                    if actual.get("batch_invariant_registered") is None:
                        item["issues"].append({"side": side, "tp_rank": rank,
                                               "error": "cannot verify native batch-invariant registration"})
                    requested = dict(environment=expected_env, batch_invariant_registered=True,
                                     torch_deterministic_algorithms=True)
                    diff = differences(requested, {k: actual.get(k) for k in requested})
                    if diff:
                        item["runtime_differences"].append({"side": side, "tp_rank": rank,
                                                            "vs_case_determinism": diff})
            for step in sorted(set(left) & set(right)):
                lp, a = left[step]
                rp, b = right[step]
                params = sampling_differences(a["sampling_params"], b["sampling_params"])
                sampler_call_diff = differences(a.get("sampler_call", {}), b.get("sampler_call", {}),
                                                {"logprobs_mode_override"})
                diagnostics = differences(a["sampling_params"], b["sampling_params"])
                diagnostics = {k: v for k, v in diagnostics.items() if k not in params}
                requested = {side: {k: {"requested": v, "actual": rec["sampling_params"].get(k, "<absent>")}
                                    for k, v in case["sampling"].items()
                                    if sampling_value(k, rec["sampling_params"].get(k, "<absent>")) != sampling_value(k, v)}
                             for side, rec in (("serve", a), ("rollout", b))}
                normalizations = {side: {k: {"requested": v, "actual": rec["sampling_params"][k]}
                                         for k, v in case["sampling"].items()
                                         if k in rec["sampling_params"] and rec["sampling_params"][k] != v
                                         and k not in requested[side]}
                                  for side, rec in (("serve", a), ("rollout", b))}
                greedy_ok = all(r.get("all_greedy") is True and r.get("all_random") is False
                                and r.get("greedy_output_matches_sampled") is True for r in (a, b))
                bonus = any(r.get("sampler_call", {}).get("predict_bonus_token", False) for r in (a, b))
                if params or any(requested.values()) or not greedy_ok or sampler_call_diff or bonus:
                    item["sampling_parameter_differences"].append({"step": step, "tp_rank": rank,
                                                                  "between_sides": params, "vs_case": requested,
                                                                  "sampler_call": sampler_call_diff,
                                                                  "unexpected_bonus_token": bonus,
                                                                  "actual_greedy_verified": greedy_ok})
                if _opaque(a["sampling_params"]) or _opaque(b["sampling_params"]):
                    item["issues"].append({"step": step, "tp_rank": rank, "error": "opaque sampling parameter; equality not proven"})
                if (a["prompt_token_ids"] != b["prompt_token_ids"] or a["prefix_token_ids"] != b["prefix_token_ids"]):
                    item["post_divergence_steps"] += 1
                    item["decode_different_prefix"].append({
                        "step": step, "tp_rank": rank, "same_prefix": False,
                        "serve_prefix_sha256": a["prefix_token_sha256"],
                        "rollout_prefix_sha256": b["prefix_token_sha256"],
                        "serve_token": a["sampled_token_ids"][0], "rollout_token": b["sampled_token_ids"][0],
                        "logits": {stage: vector_difference(a["logits"][stage], b["logits"][stage]) for stage in STAGES},
                        "note": "Different conditioning prefix: differences are recorded but cannot locate a forward error."})
                    if step == 0:
                        item["issues"].append({"tp_rank": rank, "error": "initial prompt token IDs differ"})
                    continue
                result = {"step": step, "tp_rank": rank, "same_prefix": True,
                          "serve_token": a["sampled_token_ids"][0], "rollout_token": b["sampled_token_ids"][0],
                          "sampling_equal": not params and not any(requested.values()) and greedy_ok
                                            and not sampler_call_diff and not bonus,
                          "diagnostic_parameter_differences": diagnostics,
                          "requested_parameter_normalizations": normalizations,
                          "metadata_differences": differences(_portable(a["sampling_metadata"]),
                                                              _portable(b["sampling_metadata"]), META_DIAGNOSTICS),
                          "logits": {}}
                if result["metadata_differences"]:
                    item["sampling_parameter_differences"].append({"step": step, "tp_rank": rank,
                                                                  "actual_metadata": result["metadata_differences"]})
                if _opaque(a["sampling_metadata"]) or _opaque(b["sampling_metadata"]):
                    item["issues"].append({"step": step, "tp_rank": rank,
                                           "error": "opaque sampling metadata; full equality not proven"})
                if a["logits_layout"] != b["logits_layout"]:
                    result["layout_difference"] = [a["logits_layout"], b["logits_layout"]]
                    item["issues"].append({"step": step, "tp_rank": rank,
                                           "error": "different vocab layouts; align reduce_sample before elementwise comparison"})
                else:
                    try:
                        av, bv = load_vectors(lp, a), load_vectors(rp, b)
                        for stage in STAGES:
                            result["logits"][stage] = vector_difference(a["logits"][stage], b["logits"][stage], av.get(stage), bv.get(stage))
                            if not result["logits"][stage]["bytes_equal"]:
                                candidate = {"step": step, "phase": a["phase"], "boundary": stage, "tp_rank": rank}
                                old = item["first_different_logits"]
                                if old is None or (step, STAGES.index(stage)) < (old["step"], STAGES.index(old["boundary"])):
                                    item["first_different_logits"] = candidate
                    except (ValueError, KeyError, OSError, TypeError) as error:
                        item["issues"].append({"step": step, "tp_rank": rank, "error": str(error)})
                if result["serve_token"] != result["rollout_token"]:
                    old = item["first_different_token"]
                    if old is None or step < old["step"]:
                        item["first_different_token"] = {k: result[k] for k in ("step", "tp_rank", "serve_token", "rollout_token")}
                item[a["phase"]].append(result)
        # All TP ranks must agree on the actual prefix and emitted token, even
        # when their logits are deliberately vocabulary-sharded.
        for side in ("serve", "rollout"):
            base = records.get((sample, side, 0), {})
            for rank in range(1, tp):
                other = records.get((sample, side, rank), {})
                if set(base) != set(other):
                    item["issues"].append({"side": side, "tp_rank": rank, "error": "TP step coverage differs"})
                for step in set(base) & set(other):
                    a, b = base[step][1], other[step][1]
                    if any(a[k] != b[k] for k in ("prompt_token_ids", "prefix_token_ids", "sampled_token_ids")):
                        item["issues"].append({"side": side, "tp_rank": rank, "step": step, "error": "TP ranks disagree on actual token history"})
            for step, (_, base_record) in base.items():
                rank_records = [records.get((sample, side, rank), {}).get(step) for rank in range(tp)]
                if any(pair is None for pair in rank_records):
                    continue
                candidates = []
                reduced = base_record["enable_reduce_sample"]
                width = base_record["logits"]["greedy_input"]["shape"][1]
                for rank, (_, rec) in enumerate(rank_records):
                    desc = rec["logits"]["greedy_input"]
                    if rec["enable_reduce_sample"] != reduced or desc["shape"][1] != width:
                        item["issues"].append({"side": side, "step": step, "error": "TP vocab layouts disagree"})
                    local = desc["top10"][0]
                    candidates.append((local["logit"], desc["argmax_local"] + (rank * width if reduced else 0)))
                    if not reduced and desc["sha256"] != base_record["logits"]["greedy_input"]["sha256"]:
                        item["issues"].append({"side": side, "step": step, "tp_rank": rank,
                                               "error": "replicated greedy logits differ across TP ranks"})
                # Mirrors the supplied Ascend implementation: first TP rank
                # wins an equal global maximum; local argmax chooses first ID.
                expected = max(candidates, key=lambda pair: pair[0])[1]
                actual = base_record["sampled_token_ids"][0]
                if base_record["all_greedy"] and expected != actual:
                    item["sampling_parameter_differences"].append({"side": side, "step": step,
                                                                  "expected_global_argmax": expected,
                                                                  "actual_sampled_token": actual})
        mismatch = bool(item["first_different_logits"] or item["first_different_token"]
                        or item["sampling_parameter_differences"] or item["runtime_differences"])
        if item["issues"]:
            item["status"] = "INCOMPLETE"
        elif mismatch:
            item["status"] = "DIFFERENT"
        elif full:
            item["status"] = "PASS"
        item["full_generation_captured"] = full and not item["issues"]
        samples.append(item)
    statuses = [x["status"] for x in samples]
    status = ("INCOMPLETE" if errors or "INCOMPLETE" in statuses else "DIFFERENT" if "DIFFERENT" in statuses
              else "MATCH_CAPTURED_PREFIX" if "MATCH_CAPTURED_PREFIX" in statuses else "PASS")
    return {"summary": {"case_id": case["case_id"], "status": status, "samples": len(samples),
                         "max_capture_steps": case["logits_check"]["max_steps"], "errors": errors,
                         "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
                         "scope": "Actual model output, sampler input and greedy input; same prefix only. "
                                  "Prefill predicts token 0; decode predicts tokens 1 onward. "
                                  "Byte equality uses full-vocab SHA256 even with top-K storage; "
                                  "topk_* numeric metrics cover only stored token-ID intersections. "
                                  "Runtime differences are evidence, not automatic causal attribution."},
            "samples": samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args(argv)
    try:
        spec = importlib.util.spec_from_file_location("aqa_logits_case", Path(__file__).with_name("aqa_text_ab.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root, case, rows = module.load_case(args.case_dir)
        report = compare_case(root, case, rows)
        path = root / "logits_comparison.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        for item in report["samples"]:
            print(item["sample_id"], item["status"], "first_logits=", item["first_different_logits"],
                  "first_token=", item["first_different_token"])
        print("report=" + str(path))
        return {"PASS": 0, "MATCH_CAPTURED_PREFIX": 3, "DIFFERENT": 1, "INCOMPLETE": 2}[report["summary"]["status"]]
    except (ValueError, KeyError, OSError, TypeError) as error:
        print("[huanyu-logits-ab] ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
