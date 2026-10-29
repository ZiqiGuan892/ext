#!/usr/bin/env python3
"""Compare actual LLM prompt inputs captured by both sides of an AQA case.

PASS requires every sample and TP rank, identical dtype/shape/token IDs/positions/
audio mask, and identical embedding bytes. Missing evidence never counts as equal.
This module uses NumPy only; it does not load model weights or recompute embeddings.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


FORMAT = "huanyu_prompt_embedding_v1"
SIDES = ("serve", "rollout")
DTYPES = {"torch.bfloat16": ("<u2", 2), "torch.float16": ("<f2", 2),
          "torch.float32": ("<f4", 4), "torch.float64": ("<f8", 8)}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inside(path, root):
    path = path.resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    return path


def _canonical_hash(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_arrays(path, record, root):
    filename = record.get("tensor_file")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or "/" in filename or "\\" in filename or not filename.endswith(".npz")):
        raise ValueError("tensor_file must be a relative NPZ basename")
    target = _inside(path.parent / filename, root)
    if not _is_hash(record.get("tensor_file_sha256")) or _sha(target) != record["tensor_file_sha256"]:
        raise ValueError("tensor file SHA256 mismatch")
    with target.open("rb") as stream, np.load(stream, allow_pickle=False) as source:
        if set(source.files) != {"embedding_bytes", "input_ids", "positions", "audio_mask"}:
            raise ValueError("NPZ must contain exactly embedding_bytes, input_ids, positions, audio_mask")
        arrays = {key: source[key] for key in source.files}
    metadata = record.get("embeddings", {})
    if not isinstance(metadata, dict):
        raise ValueError("invalid embedding metadata")
    shape, dtype = metadata.get("shape"), metadata.get("dtype")
    if (not isinstance(shape, list) or len(shape) != 2 or not all(_integer(n, 1) for n in shape)
            or dtype not in DTYPES or not _is_hash(metadata.get("sha256"))):
        raise ValueError("invalid embedding shape/dtype/hash metadata")
    count = shape[0]
    if (record.get("capture_scope") != "llm_inputs_embeds" or record.get("prefill_complete") is not True
            or not _integer(record.get("prompt_token_count"), 1)
            or record["prompt_token_count"] != count or record.get("num_computed_tokens") != 0
            or not _integer(record.get("num_computed_tokens"))
            or not _integer(record.get("captured_token_count"), 1)
            or record.get("captured_token_count") != count):
        raise ValueError("record does not prove a complete actual LLM prompt capture")
    raw, ids, positions, mask = (arrays[key] for key in
                                 ("embedding_bytes", "input_ids", "positions", "audio_mask"))
    storage_dtype, itemsize = DTYPES[dtype]
    if raw.dtype != np.uint8 or raw.shape != tuple(shape) + (itemsize,):
        raise ValueError("embedding raw bytes shape/dtype mismatch")
    raw_bytes = raw.tobytes(order="C")
    if hashlib.sha256(raw_bytes).hexdigest() != metadata["sha256"]:
        raise ValueError("embedding content SHA256 mismatch")
    if ids.dtype != np.int64 or ids.shape != (count,) or np.any(ids < 0):
        raise ValueError("invalid full prompt token IDs")
    if (positions.dtype != np.int64 or positions.shape not in ((count,), (3, count))
            or np.any(positions < 0)):
        raise ValueError("invalid full prompt positions")
    if mask.dtype != np.bool_ or mask.shape != (count,) or not np.any(mask):
        raise ValueError("invalid or empty audio placement mask")
    values = np.frombuffer(raw_bytes, dtype=storage_dtype).reshape(shape)
    if dtype == "torch.bfloat16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite prompt embeddings")
    arrays.update(dtype=dtype, shape=shape, values=values, path=str(path))
    return arrays


def _validate_record(path, record, root, case, row, side, tp):
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("unsupported embedding record format")
    if record.get("case_id") != case["case_id"] or record.get("sample_id") != row["sample_id"]:
        raise ValueError("case/sample identity mismatch")
    if record.get("side") != side or record.get("status") != "complete":
        raise ValueError("wrong side or capture incomplete: " + str(record.get("error", record.get("status"))))
    rank = record.get("tp_rank")
    if not _integer(rank) or rank >= tp or not _integer(record.get("tp_size"), 1) or record["tp_size"] != tp:
        raise ValueError("invalid TP rank or size")
    request_id = record.get("request_id")
    tag = "huanyuab.{}.{}".format(case["case_id"], row["sample_id"])
    if (not isinstance(request_id, str)
            or re.search(r"(?<![A-Za-z0-9_])" + re.escape(tag) + r"(?=$|[^A-Za-z0-9_])", request_id) is None):
        raise ValueError("actual request ID lacks this case/sample tag")
    if record.get("prompt_sha256") != row.get("prompt_sha256", _canonical_hash(row["prompt"])):
        raise ValueError("canonical prompt identity mismatch")
    if record.get("wav_sha256") != row["wav_sha256"]:
        raise ValueError("WAV identity/order mismatch")
    return rank


def _array_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def _subset(raw_mismatch, difference, mask):
    count = int(np.count_nonzero(mask))
    return {"tokens": count, "mismatched_elements": int(np.count_nonzero(raw_mismatch[mask])),
            "max_abs_difference": float(np.max(difference[mask])) if count else 0.0}


def compare_tensors(left, right):
    """Strict bytes decide equality; numeric summaries are diagnostic only."""
    fields = {
        "dtype_equal": left["dtype"] == right["dtype"],
        "shape_equal": left["shape"] == right["shape"],
        "input_ids_equal": _array_equal(left["input_ids"], right["input_ids"]),
        "positions_equal": _array_equal(left["positions"], right["positions"]),
        "audio_mask_equal": _array_equal(left["audio_mask"], right["audio_mask"]),
        "embedding_bytes_equal": _array_equal(left["embedding_bytes"], right["embedding_bytes"]),
    }
    result = dict(fields, strict_equal=all(fields.values()))
    result.update(left_dtype=left["dtype"], right_dtype=right["dtype"],
                  left_shape=left["shape"], right_shape=right["shape"])
    for key in ("input_ids", "positions", "audio_mask"):
        if left[key].shape == right[key].shape and not fields[key + "_equal"]:
            indices = np.argwhere(left[key] != right[key])
            if indices.size:
                index = tuple(indices[0])
                result[key + "_first_difference"] = {"index": list(map(int, index)),
                                                      "left": left[key][index].item(),
                                                      "right": right[key][index].item()}
    if fields["shape_equal"]:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(left["values"].astype(np.float64) - right["values"].astype(np.float64))
        finite_difference = np.isfinite(difference).all()
        result["max_abs_difference"] = float(np.max(difference)) if finite_difference else None
        result["mean_abs_difference"] = float(np.sum(difference / difference.size)) if finite_difference else None
        if not finite_difference:
            result["numeric_summary_note"] = "difference exceeds float64 range; strict bytes comparison remains valid"
        if fields["dtype_equal"]:
            mismatches = np.any(left["embedding_bytes"] != right["embedding_bytes"], axis=-1)
            indices = np.argwhere(mismatches)
            result["mismatched_elements"] = int(indices.shape[0])
            result["first_different_element"] = indices[0].tolist() if indices.size else None
            if fields["audio_mask_equal"] and finite_difference:
                result["audio"] = _subset(mismatches, difference, left["audio_mask"])
                result["text_and_special_tokens"] = _subset(mismatches, difference, ~left["audio_mask"])
    return result


def _compare_sample(root, case, row, records, issues, tp):
    """Bound tensor memory to one sample instead of retaining the whole case."""
    evidence = {side: {} for side in SIDES}
    item = {"sample_id": row["sample_id"], "case_id": case["case_id"], "status": "INCOMPLETE",
            "strict_equal": None, "issues": list(issues), "ranks": [], "within_side": {}}
    for side in SIDES:
        for rank, (path, record) in records[side].items():
            try:
                evidence[side][rank] = (record, _load_arrays(path, record, root))
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                item["issues"].append({"side": side, "record": str(path), "error": str(error)})
        missing = sorted(set(range(tp)) - set(evidence[side]))
        if missing:
            item["issues"].append({"side": side, "error": "missing TP ranks", "ranks": missing})
        requests = {record["request_id"] for record, _ in evidence[side].values()}
        if len(requests) > 1:
            item["issues"].append({"side": side, "error": "TP ranks belong to different requests"})
        if 0 in evidence[side]:
            base = evidence[side][0][1]
            item["within_side"][side] = [dict(tp_rank=rank, **compare_tensors(base, value[1]))
                                         for rank, value in sorted(evidence[side].items()) if rank != 0]
    for rank in sorted(set(evidence["serve"]) & set(evidence["rollout"])):
        item["ranks"].append(dict(tp_rank=rank, **compare_tensors(
            evidence["serve"][rank][1], evidence["rollout"][rank][1])))
    if not item["issues"]:
        equal = (all(value["strict_equal"] for value in item["ranks"])
                 and all(value["strict_equal"] for group in item["within_side"].values() for value in group))
        item.update(status="PASS" if equal else "FAIL", strict_equal=equal)
    return item


def compare_case_embeddings(root, case, rows):
    """Return a complete sample inventory without writing or modifying any input."""
    root = Path(root).resolve(strict=True)
    tp = case.get("engine", {}).get("tensor_parallel_size")
    if not _integer(tp, 1) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("case requires a case_id and positive TP size")
    expected = {row["sample_id"]: row for row in rows}
    if (not rows or len(expected) != len(rows) or not _integer(case.get("sample_size"), 1)
            or case["sample_size"] != len(rows)):
        raise ValueError("case sample inventory is empty, duplicated or inconsistent")
    for row in rows:
        if (not isinstance(row["sample_id"], str) or re.fullmatch(r"row_[0-9]{6,}", row["sample_id"]) is None
                or row.get("case_id") != case["case_id"]
                or not isinstance(row.get("wav_sha256"), list) or not row["wav_sha256"]
                or not all(_is_hash(value) for value in row["wav_sha256"])
                or ("prompt_sha256" in row and row["prompt_sha256"] != _canonical_hash(row["prompt"]))):
            raise ValueError("invalid case sample identity, WAV hashes or canonical prompt hash")
    configured_tp = case.get("embedding_check", {}).get("expected_tp_size", tp)
    if not _integer(configured_tp, 1) or configured_tp != tp:
        raise ValueError("embedding check TP size differs from case engine")
    entries = defaultdict(lambda: defaultdict(dict))
    issues = defaultdict(list)
    global_issues = []
    observed = Counter()
    for side in SIDES:
        directory = root / "embeddings" / side
        for path in sorted(directory.rglob("*.json")):
            sample = None
            try:
                _inside(path, root)
                record = _read(path)
                sample = record.get("sample_id") if isinstance(record, dict) else None
                if sample not in expected:
                    raise ValueError("unknown or missing sample ID")
                observed[side] += 1
                rank = _validate_record(path, record, root, case, expected[sample], side, tp)
                if rank in entries[sample][side]:
                    raise ValueError("duplicate capture for sample/side/TP rank")
                entries[sample][side][rank] = (path, record)
            except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
                issue = {"side": side, "record": str(path), "error": str(error)}
                if sample in expected:
                    issues[sample].append(issue)
                else:
                    global_issues.append(issue)
    samples = []
    for row in rows:
        sample = row["sample_id"]
        samples.append(_compare_sample(root, case, row, entries[sample], issues[sample], tp))
    counts = Counter(item["status"] for item in samples)
    status = ("INCOMPLETE" if global_issues or counts["INCOMPLETE"] else "FAIL" if counts["FAIL"] else "PASS")
    summary = {"format": FORMAT, "case_id": case["case_id"], "status": status,
               "expected_samples": len(rows), "expected_tp_size": tp,
               "equal_samples": counts["PASS"], "different_samples": counts["FAIL"],
               "incomplete_samples": counts["INCOMPLETE"], "observed_records": dict(observed),
               "global_issues": global_issues,
               "note": "Strict equality requires all TP ranks and complete actual LLM prompt inputs, "
                       "including audio placements. Numerical summaries use no tolerance. "
                       "Equal inputs do not prove identical model weights, cache state, operators or outputs."}
    return {"summary": summary, "samples": samples}


def load_case(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    case = _read(root / "case.json")
    manifest = root / "manifest.jsonl"
    if case.get("schema_version") != 1 or _sha(manifest) != case.get("manifest_sha256"):
        raise ValueError("invalid case schema or changed manifest")
    rows = [json.loads(line, object_pairs_hook=_object) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return root, case, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args(argv)
    try:
        root, case, rows = load_case(args.case_dir)
        paths = [root / "embedding_comparison.jsonl", root / "embedding_summary.json"]
        if any(path.exists() for path in paths):
            raise FileExistsError("embedding comparison outputs already exist; preserve them before comparing again")
        report = compare_case_embeddings(root, case, rows)
        with paths[0].open("x", encoding="utf-8") as stream:
            for item in report["samples"]:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        with paths[1].open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report["summary"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["summary"]["status"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("[huanyu-prompt-embedding] ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
