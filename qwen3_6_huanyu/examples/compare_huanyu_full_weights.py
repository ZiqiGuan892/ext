#!/usr/bin/env python3
"""Summarize every rollout worker's first-sync checkpoint audit, without torch.

A PASS requires the requested worker count and complete tensor-parallel groups.
Use a fresh audit directory per run; incomplete or mixed evidence never passes.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


STATUSES = {"PASS", "FAIL", "INCOMPLETE"}
STAGES = ("initial_disk", "after_sync", "disk_replay")
COMPARISON = "initial_native_disk_vs_first_sync_vs_poisoned_native_disk_replay"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_status(value):
    return isinstance(value, str) and value in STATUSES


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _fingerprint(tensor, allow_nonfinite=False):
    if not isinstance(tensor, dict):
        return None
    shape, dtype, digest = tensor.get("shape"), tensor.get("dtype"), tensor.get("sha256")
    if (not isinstance(shape, list) or not all(_integer(size) for size in shape)
            or not _nonempty(dtype) or not _sha256(digest)
            or not isinstance(tensor.get("finite"), bool)
            or (not allow_nonfinite and tensor["finite"] is not True)
            or tensor.get("is_meta") is not False
            or not _integer(tensor.get("numel")) or tensor["numel"] != _numel(shape)):
        return None
    return tuple(shape), dtype, digest


def _numel(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def _check_provenance(report, worker, issues):
    if (report.get("audit_only") is not True or report.get("generation_forbidden") is not True
            or report.get("comparison") != COMPARISON):
        issues.append(f"{worker}: missing audit-only three-way checkpoint replay provenance")
    stream = report.get("checkpoint_stream")
    if (not isinstance(stream, dict) or not _integer(stream.get("tensor_count"), 1)
            or not _integer(stream.get("nonfinite_elements")) or stream["nonfinite_elements"] != 0
            or not _sha256(stream.get("manifest_sha256"))):
        issues.append(f"{worker}: missing, empty or nonfinite checkpoint stream")
    provenance = report.get("provenance")
    if (not isinstance(provenance, dict) or provenance.get("checkpoint") != report.get("checkpoint")
            or provenance.get("load_format") != "safetensors"):
        issues.append(f"{worker}: missing/mismatched local safetensors provenance")
        return
    headers = provenance.get("checkpoint_headers")
    if not isinstance(headers, dict) or not headers:
        issues.append(f"{worker}: missing local checkpoint headers")


def _check_tensors(report, worker, issues, failures):
    counts = {}
    for kind in ("parameters", "buffers"):
        tensors = report.get(kind)
        if not isinstance(tensors, dict) or (kind == "parameters" and not tensors):
            issues.append(f"{worker}: missing or empty {kind} inventory")
            counts[kind] = {"entries": 0, "numel": 0, "statuses": {}}
            continue
        statuses = Counter()
        total_numel = 0
        for name, entry in tensors.items():
            label = f"{worker}:{kind}:{name}"
            if not _nonempty(name) or not isinstance(entry, dict) or not _valid_status(entry.get("status")):
                issues.append(f"{label}: invalid tensor record/status")
                continue
            status = entry["status"]
            statuses[status] += 1
            fingerprints = {stage: _fingerprint(entry.get(stage), allow_nonfinite=kind == "buffers")
                            for stage in STAGES}
            missing = [stage for stage, fingerprint in fingerprints.items() if fingerprint is None]
            if missing:
                issues.append(f"{label}: missing/invalid fingerprints: {', '.join(missing)}")
            else:
                total_numel += _numel(fingerprints["after_sync"][0])
                if len(set(fingerprints.values())) != 1:
                    failures.append({"worker": worker, "kind": kind, "name": name,
                                     "reason": "initial_disk/after_sync/disk_replay differ"})
                    if status == "PASS":
                        issues.append(f"{label}: declared PASS contradicts fingerprints")
            if status == "FAIL":
                failures.append({"worker": worker, "kind": kind, "name": name,
                                 "reason": entry.get("reason", "worker reported FAIL")})
            elif status == "INCOMPLETE":
                issues.append(f"{label}: worker reported INCOMPLETE")
        counts[kind] = {"entries": len(tensors), "numel": total_numel, "statuses": dict(statuses)}
    return counts


def compare(audit_dir, expected_workers, tp_size):
    """Read reports and validate both numerical evidence and job coverage."""
    if not _integer(expected_workers, 1) or not _integer(tp_size, 1):
        raise ValueError("expected_workers and tp_size must be positive integers")
    if expected_workers % tp_size:
        raise ValueError("expected_workers must be divisible by tp_size")
    directory = Path(audit_dir)
    paths = sorted(directory.rglob("full_weights-*.json")) if directory.is_dir() else []
    issues, failures, workers = [], [], []
    identities, runs, checkpoints = set(), set(), set()
    groups = defaultdict(list)
    if len(paths) != expected_workers:
        issues.append(f"Expected {expected_workers} worker reports, found {len(paths)}")
    for path in paths:
        location = str(path)
        try:
            report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (OSError, ValueError) as exc:
            issues.append(f"{location}: cannot read report: {exc}")
            continue
        if (not isinstance(report, dict) or not _integer(report.get("schema_version"), 1)
                or report["schema_version"] != 1):
            issues.append(f"{location}: unsupported or missing schema_version")
            continue
        process = report.get("process")
        if not isinstance(process, dict):
            issues.append(f"{location}: missing process metadata")
            continue
        host, pid = process.get("host"), process.get("pid")
        if not _nonempty(host) or not _integer(pid, 1):
            issues.append(f"{location}: invalid host/pid")
            continue
        worker = f"{host}:{pid}"
        if worker in identities:
            issues.append(f"{location}: duplicate worker {worker}")
        identities.add(worker)
        run_id, checkpoint = report.get("run_id"), report.get("checkpoint")
        for key, value, collection in (("run_id", run_id, runs), ("checkpoint", checkpoint, checkpoints)):
            if not _nonempty(value):
                issues.append(f"{worker}: missing {key}")
            else:
                collection.add(value)
        status = report.get("status")
        errors = report.get("errors")
        if not _valid_status(status):
            issues.append(f"{worker}: invalid status")
        elif status == "INCOMPLETE":
            issues.append(f"{worker}: worker reported INCOMPLETE")
        elif status == "FAIL":
            failures.append({"worker": worker, "kind": "report", "name": None,
                             "reason": "worker reported FAIL"})
        if not isinstance(errors, list):
            issues.append(f"{worker}: missing errors list")
        elif errors:
            issues.extend(f"{worker}: {error}" for error in errors)
        _check_provenance(report, worker, issues)
        group, rank, size = process.get("engine_group"), process.get("tp_rank"), process.get("tp_world_size")
        if (not _nonempty(group) or not _integer(rank) or rank >= tp_size
                or not _integer(size, 1) or size != tp_size):
            issues.append(f"{worker}: invalid/missing TP topology: group={group!r}, rank={rank!r}, size={size!r}")
        else:
            groups[group].append((rank, worker, report))
        counts = _check_tensors(report, worker, issues, failures)
        summary = report.get("summary")
        if not isinstance(summary, dict):
            issues.append(f"{worker}: missing summary")
            summary = {}
        expected_counts = {"parameter_names": counts["parameters"]["entries"],
                           "parameter_numel_with_aliases": counts["parameters"]["numel"]}
        for key, expected in expected_counts.items():
            if not _integer(summary.get(key)) or summary[key] != expected:
                issues.append(f"{worker}: summary.{key} does not match tensor inventory")
        for key, upper in (("unique_parameter_objects", counts["parameters"]["entries"]),
                           ("unique_parameter_numel", counts["parameters"]["numel"])):
            if not _integer(summary.get(key)) or not 0 < summary[key] <= upper:
                issues.append(f"{worker}: invalid summary.{key}")
        workers.append({"worker": worker, "path": location, "process": process, "status": status,
                        "counts": counts, "summary": summary,
                        "checkpoint_stream": report.get("checkpoint_stream", {}),
                        "provenance": report.get("provenance", {})})
    if len(identities) != expected_workers:
        issues.append(f"Expected {expected_workers} unique workers, found {len(identities)}")
    if len(runs) != 1:
        issues.append(f"Expected one run_id, found {len(runs)}: {sorted(runs)}")
    if len(checkpoints) != 1:
        issues.append(f"Expected one checkpoint, found {len(checkpoints)}: {sorted(checkpoints)}")
    manifests = {row["checkpoint_stream"]["manifest_sha256"] for row in workers
                 if isinstance(row["checkpoint_stream"], dict)
                 and _sha256(row["checkpoint_stream"].get("manifest_sha256"))}
    if len(manifests) != 1:
        issues.append(f"Expected one full checkpoint content manifest, found {len(manifests)}")
    if len(groups) != expected_workers // tp_size:
        issues.append(f"Expected {expected_workers // tp_size} TP groups, found {len(groups)}")
    topology = {}
    for group, entries in sorted(groups.items()):
        ranks = [rank for rank, _, _ in entries]
        complete = sorted(ranks) == list(range(tp_size))
        topology[group] = {"tp_ranks": sorted(ranks), "workers": [worker for _, worker, _ in entries],
                           "complete": complete}
        if not complete:
            issues.append(f"TP group {group}: expected ranks 0..{tp_size - 1}, found {sorted(ranks)}")
        for kind in ("parameters", "buffers"):
            inventories = [set(report.get(kind, {})) for _, _, report in entries
                           if isinstance(report.get(kind), dict)]
            if inventories and any(inventory != inventories[0] for inventory in inventories[1:]):
                issues.append(f"TP group {group}: {kind} inventories differ across workers")
    for kind in ("parameters", "buffers"):
        inventories = [set(report[kind]) for entries in groups.values() for _, _, report in entries
                       if isinstance(report.get(kind), dict)]
        if inventories and any(inventory != inventories[0] for inventory in inventories[1:]):
            issues.append(f"{kind} inventories differ between policy workers/replicas")
    # A shared path can point to different files on different hosts. Compare the
    # native disk baselines at the same TP coordinate across engine replicas.
    by_tp_rank = defaultdict(list)
    for entries in groups.values():
        for rank, worker, report in entries:
            by_tp_rank[rank].append((worker, report))
    for rank, replicas in by_tp_rank.items():
        for kind in ("parameters", "buffers"):
            for stage in ("initial_disk", "disk_replay"):
                baseline = None
                for worker, report in replicas:
                    inventory = report.get(kind)
                    if not isinstance(inventory, dict):
                        continue
                    current = {name: _fingerprint(entry.get(stage), allow_nonfinite=kind == "buffers")
                               for name, entry in inventory.items()
                               if isinstance(entry, dict)}
                    if baseline is None:
                        baseline = current
                    elif current != baseline:
                        issues.append(f"TP rank {rank}: {kind} {stage} baseline differs between replicas ({worker})")
    # A numerical failure and missing coverage are both recorded. INCOMPLETE has
    # priority because this invocation cannot establish a whole-job conclusion.
    status = "INCOMPLETE" if issues else "FAIL" if failures else "PASS"
    return {"schema_version": 1, "status": status, "audit_dir": str(directory),
            "expected_workers": expected_workers, "tp_size": tp_size,
            "run_ids": sorted(runs), "checkpoints": sorted(checkpoints),
            "checkpoint_manifest_sha256": sorted(manifests),
            "observed_reports": len(paths), "observed_unique_workers": len(identities),
            "topology": topology, "workers": workers,
            "summary": {"parameter_entries": sum(row["counts"]["parameters"]["entries"] for row in workers),
                        "parameter_numel": sum(row["counts"]["parameters"]["numel"] for row in workers),
                        "unique_parameter_objects": sum(row["summary"].get("unique_parameter_objects", 0)
                                                        for row in workers
                                                        if _integer(row["summary"].get("unique_parameter_objects"))),
                        "unique_parameter_numel": sum(row["summary"].get("unique_parameter_numel", 0)
                                                      for row in workers
                                                      if _integer(row["summary"].get("unique_parameter_numel"))),
                        "buffer_entries": sum(row["counts"]["buffers"]["entries"] for row in workers),
                        "buffer_numel": sum(row["counts"]["buffers"]["numel"] for row in workers)},
            "failures": failures, "errors": issues,
            "note": "Counts include every worker. Unique parameter counts deduplicate object aliases per worker, "
                    "not shared storage or global TP replicas. "
                    "Equality covers runtime tensors after native checkpoint loading, not raw checkpoint names."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--expected-workers", required=True, type=int,
                        help="Total policy rollout worker processes; exclude actor/ref/reward workers")
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.audit_dir, args.expected_workers, args.tp_size)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"{report['status']}: workers={report['observed_unique_workers']}/{args.expected_workers}; "
              f"report={args.output}")
        for error in report["errors"]:
            print("INCOMPLETE:", error)
        for failure in report["failures"]:
            print("FAIL:", failure["worker"], failure["kind"], failure["name"], failure["reason"])
    else:
        print(payload, end="")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
