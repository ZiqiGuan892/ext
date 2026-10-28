#!/usr/bin/env python3
"""Compare bounded Huanyu weight audit JSONL files without torch or collectives."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct


NAMES = tuple(sorted(
    f"model.audio_encoder.conv_layers.{block}.layers.{layer}.{leaf}"
    for block in (0, 1) for layer in (1, 4) for leaf in ("weight", "bias")
))
STAGES = ("checkpoint", "actor_loaded", "actor_export", "receiver_input", "receiver_owned", "after_finalize")


def fingerprint(tensor):
    if tensor.get("is_meta") or not tensor.get("finite") or not tensor.get("float32_sha256"):
        return None
    return (tuple(tensor.get("shape", [])), tensor["float32_sha256"])


def difference(expected, actual):
    result = {"expected_shape": expected.get("shape"), "actual_shape": actual.get("shape")}
    result["match"] = fingerprint(expected) is not None and fingerprint(expected) == fingerprint(actual)
    left, right = expected.get("values"), actual.get("values")
    if left is not None and right is not None and max(len(left), len(right)) <= 256:
        result["differences"] = [
            {"index": i, "expected": left[i] if i < len(left) else None,
             "actual": right[i] if i < len(right) else None}
            for i in range(max(len(left), len(right)))
            if i >= len(left) or i >= len(right) or left[i] != right[i]
        ]
    return result


def consensus(entries):
    hashes = {fingerprint(e["tensor"]) for e in entries}
    return entries[0]["tensor"] if entries and None not in hashes and len(hashes) == 1 else None


def load_entries(directory, update):
    entries = []
    for path in sorted(Path(directory).rglob("weights-*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("schema_version") != 1:
                raise ValueError(f"Unsupported schema: {path}:{number}")
            stage = entry.get("stage")
            initial = stage in ("checkpoint", "actor_loaded") and entry.get("round_id") is None
            if entry.get("name") in NAMES and (initial or entry.get("round_id") == update):
                entry["process"] = f"{entry.get('host')}:{entry.get('pid')}:rank={entry.get('rank')}:{path}"
                entry["location"] = f"{path}:{number}"
                entries.append(entry)
    return entries


def stage_report(entries, expected_processes, baseline, *, allow_initial_meta=False):
    present = {e["process"] for e in entries}
    missing = sorted(expected_processes - present)
    fingerprints = defaultdict(list)
    records, concrete, deferred = [], [], []
    bad = False
    for e in entries:
        tensor = e["tensor"]
        row = {"process": e["process"], "location": e["location"], "tensor": tensor,
               "extra": e.get("extra", {}), "versions": e.get("versions", {}),
               "comparison": None}
        if allow_initial_meta and tensor.get("is_meta"):
            # Non-source ranks deliberately remain meta inside init_empty_weights.
            # FSDP's later state-dict broadcast, not this snapshot, supplies values.
            # Check their structure here and require concrete source-rank evidence.
            deferred.append(e["process"])
            shape_matches = baseline is not None and tensor.get("shape") == baseline.get("shape")
            if baseline is not None and "global_shape" in tensor:
                shape_matches = shape_matches and tensor["global_shape"] == baseline.get("shape")
            row["status"] = "DEFERRED_META" if shape_matches else "INCOMPLETE" if baseline is None else "FAIL"
            row["comparison"] = {
                "match": None, "deferred": True, "shape_match": shape_matches if baseline else None,
                "expected_shape": baseline.get("shape") if baseline else None,
                "actual_shape": tensor.get("shape"),
                "reason": "Initial meta rank has no values until FSDP initialization/broadcast",
            }
            bad = bad or row["status"] == "FAIL"
        else:
            concrete.append(e)
            key = fingerprint(tensor)
            fingerprints[str(key)].append(e["process"])
            row["comparison"] = difference(baseline, tensor) if baseline else None
            bad = bad or key is None or (baseline is not None and not row["comparison"]["match"])
            row["status"] = ("FAIL" if key is None or (baseline is not None and not row["comparison"]["match"])
                             else "PASS" if baseline is not None else "INCOMPLETE")
        records.append(row)
    bad = bad or len(fingerprints) > 1
    status = "FAIL" if bad else "PASS" if concrete and not missing and baseline is not None else "INCOMPLETE"
    return {"status": status, "missing_processes": missing, "fingerprints": dict(fingerprints),
            "process_divergence": len(fingerprints) > 1, "records": records,
            "concrete_record_count": len(concrete), "deferred_meta_count": len(deferred),
            "deferred_meta_processes": sorted(set(deferred))}


def shard_report(entries, expected_processes, baseline):
    result = {"records": [], "missing_processes": sorted(expected_processes - {e["process"] for e in entries})}
    statuses, coordinates, mesh_sizes = [], [], set()
    for entry in entries:
        tensor = entry["tensor"]
        row = {"process": entry["process"], "location": entry["location"], "tensor": tensor,
               "extra": entry.get("extra", {}), "versions": entry.get("versions", {}),
               "status": "UNSUPPORTED"}
        mesh, coord = tensor.get("mesh_shape", []), tensor.get("mesh_coordinate", [])
        placement = tensor.get("placements", [])
        values = baseline.get("values") if baseline else None
        supported = (len(mesh) == 1 and mesh[0] > 0 and coord is not None and len(coord) == 1
                     and 0 <= coord[0] < mesh[0] and placement in (["S(0)"], ["Shard(dim=0)"])
                     and len(tensor.get("global_shape", [])) == 1)
        if supported and values is not None and len(values) <= 256:
            size = tensor["global_shape"][0]
            width = (size + mesh[0] - 1) // mesh[0]
            start = min(coord[0] * width, size)
            selected = values[start:min(start + width, size)]
            expected = {"shape": [len(selected)], "values": selected, "finite": True,
                        "float32_sha256": hashlib.sha256(struct.pack(f"<{len(selected)}f", *selected)).hexdigest()}
            row.update(offset=start, comparison=difference(expected, tensor))
            row["status"] = "PASS" if size == len(values) and row["comparison"]["match"] else "FAIL"
            coordinates.append(coord[0])
            mesh_sizes.add(mesh[0])
        elif supported:
            row["status"] = "INCOMPLETE"
        statuses.append(row["status"])
        result["records"].append(row)
    coverage = len(mesh_sizes) == 1 and set(coordinates) == set(range(next(iter(mesh_sizes))))
    result["mesh_coverage_complete"] = coverage
    result["status"] = ("FAIL" if "FAIL" in statuses else "PASS"
                        if statuses and set(statuses) == {"PASS"} and not result["missing_processes"] and coverage
                        else "INCOMPLETE")
    return result


def compare(entries, update=1):
    groups = defaultdict(list)
    processes = defaultdict(set)
    for e in entries:
        groups[(e["name"], e["stage"])].append(e)
        processes[e["stage"]].add(e["process"])
    actors = processes["actor_export"] | processes["actor_local_shard"]
    receivers = set().union(*(processes[s] for s in (*STAGES[3:], "before_reload", "after_initialize")))
    # An actor observed at export must also have initial audit evidence. Do not
    # silently accept a missing checkpoint/load pair just because both are absent.
    initial = processes["checkpoint"] | processes["actor_loaded"] | actors
    report = {"update": update, "scope": "eight Huanyu LayerNorm parameters; observed processes only",
              "baseline_note": "First-update exact equality assumes these eight checkpoint tensors are BF16; "
                               "an FP32 checkpoint rounded by actor loading needs separate dtype analysis.",
              "checkpoint_baseline": "required_equal" if update == 1 else "informational_after_optimizer_updates",
              "parameters": {}}
    for name in NAMES:
        checkpoint = consensus(groups[(name, "checkpoint")])
        exported = consensus(groups[(name, "actor_export")])
        stages = {}
        for stage in STAGES:
            expected = initial if stage in STAGES[:2] else actors if stage == "actor_export" else receivers
            baseline = checkpoint if update == 1 or stage in STAGES[:2] else exported
            stages[stage] = stage_report(groups[(name, stage)], expected, baseline,
                                         allow_initial_meta=stage == "actor_loaded")
        shards = shard_report(groups[(name, "actor_local_shard")], actors,
                              checkpoint if update == 1 else exported)
        stages["actor_local_shard"] = shards
        ordered = ("checkpoint", "actor_loaded", "actor_local_shard", "actor_export", *STAGES[3:])
        earliest = None
        # Later rounds' shards are compared to the gathered export, not an independent reference.
        if update == 1:
            for stage in ordered:
                if stages[stage]["status"] != "PASS":
                    if stages[stage]["status"] == "FAIL" and stage != "checkpoint":
                        earliest = stage
                    break
        elif stages["actor_export"]["status"] == "PASS":
            for stage in STAGES[3:]:
                if stages[stage]["status"] != "PASS":
                    if stages[stage]["status"] == "FAIL":
                        earliest = stage
                    break
        statuses = [item["status"] for item in stages.values()]
        status = "FAIL" if "FAIL" in statuses else "PASS" if set(statuses) == {"PASS"} else "INCOMPLETE"
        report["parameters"][name] = {"status": status, "control": ".conv_layers.1." in name,
                                      "earliest_observed_bad_boundary": earliest, "stages": stages}
    statuses = {p["status"] for p in report["parameters"].values()}
    report["status"] = "FAIL" if "FAIL" in statuses else "PASS" if statuses == {"PASS"} else "INCOMPLETE"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--update", type=int, default=1, help="One-based sync update (default: 1)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.update < 1:
        parser.error("--update must be >= 1")
    report = compare(load_entries(args.audit_dir, args.update), args.update)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"{report['status']}: update={args.update}; report={args.output}")
        for name, item in report["parameters"].items():
            print(f"{item['status']} {name}: earliest={item['earliest_observed_bad_boundary']}")
    else:
        print(payload)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
