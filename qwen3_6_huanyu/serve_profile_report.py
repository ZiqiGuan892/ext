"""Audit and summarize ALL exported kernel rows and timed trace ranges.

Keep module CPU durations separate from device task durations; neither their
sum nor their difference is a wall-clock critical-path decomposition.
Unknown kernels remain in the report. No top-K truncation of the CSV output.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid time value: {value!r}")
    return result


def union_length(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Negative interval")
        total += stop - start if end is None or start >= end else max(0, stop - end)
        end = stop if end is None else max(end, stop)
    return total


def stats(values):
    return dict(count=len(values), sum_us=sum(values), mean_us=sum(values) / len(values),
                min_us=min(values), max_us=max(values))


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def kernel_report(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        normalized = {k.replace("μ", "u").replace("µ", "u"): k for k in columns}
        duration = normalized.get("Duration(us)")
        start = normalized.get("Start Time(us)")
        if "Name" not in columns or duration is None:
            raise ValueError(f"Unrecognized kernel CSV schema: {columns}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty kernel_details.csv")
    groups, spans = defaultdict(list), []
    total = 0.0
    zero = 0
    for index, row in enumerate(rows, 2):
        try:
            elapsed = number(row[duration])
            if start is not None:
                begin = number(row[start])
                spans.append((begin, begin + elapsed))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid kernel row {index}; no rows silently dropped: {row}") from exc
        kind = row.get("Type") or row.get("Op Type") or "<unknown>"
        core = row.get("Accelerator Core") or row.get("Task Type") or "<unknown>"
        name = row.get("Name") or "<unnamed>"
        groups[(name, kind, core)].append(elapsed)
        total += elapsed
        zero += elapsed == 0
    if total <= 0:
        raise ValueError("No positive-duration device tasks")
    result = [dict(name=name, type=kind, core=core, **stats(values))
              for (name, kind, core), values in groups.items()]
    result.sort(key=lambda r: r["sum_us"], reverse=True)
    assert sum(r["count"] for r in result) == len(rows)
    return result, dict(rows=len(rows), zero_duration_rows=zero, invalid_rows=0,
                        total_task_duration_us=total,
                        device_task_interval_union_us=union_length(spans) if spans else None,
                        device_task_span_us=max(b for _, b in spans) - min(a for a, _ in spans) if spans else None)


def trace_report(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data if isinstance(data, list) else data["traceEvents"]
    groups, modules, timelines = defaultdict(list), [], defaultdict(list)
    non_complete = 0
    for event in events:
        if event.get("ph") != "X":
            non_complete += 1
            continue
        # No category/name whitelist: keep CPU, runtime, device and unknown
        # complete events, but NEVER sum this mixed table as device time.
        start, duration = number(event["ts"]), number(event["dur"])
        name = str(event.get("name", "<unnamed>"))
        pid, tid = str(event.get("pid", "")), str(event.get("tid", ""))
        category = str(event.get("cat", "<unknown>"))
        groups[(pid, tid, category, name)].append(duration)
        if name.startswith("serve/"):
            row = dict(pid=pid, tid=tid, name=name, start=start, end=start + duration, duration=duration)
            modules.append(row)
            timelines[(pid, tid)].append(row)
    if not any(m["name"].startswith("serve/module/") for m in modules):
        raise ValueError("No timed module ranges; this trace cannot establish eager module coverage")
    module_groups = defaultdict(list)
    for thread, ranges in timelines.items():
        ranges.sort(key=lambda r: (r["start"], -r["end"]))
        stack = []
        for row in ranges:
            while stack and not (stack[-1]["start"] <= row["start"] and row["end"] <= stack[-1]["end"]):
                stack.pop()
            row["children"] = []
            if stack:
                stack[-1]["children"].append((row["start"], row["end"]))
            stack.append(row)
        for row in ranges:
            exclusive = row["duration"] - union_length(row["children"])
            if exclusive < -1e-6:
                raise ValueError("Invalid nested CPU ranges")
            module_groups[(*thread, row["name"])].append((row["duration"], max(0, exclusive)))
    host = []
    for (pid, tid, name), values in module_groups.items():
        host.append(dict(pid=pid, tid=tid, name=name, count=len(values),
                         inclusive_cpu_us=sum(a for a, _ in values),
                         self_cpu_us=sum(b for _, b in values)))
    host.sort(key=lambda r: r["inclusive_cpu_us"], reverse=True)
    all_ranges = [dict(pid=p, tid=t, category=c, name=n, **stats(v)) for (p, t, c, n), v in groups.items()]
    all_ranges.sort(key=lambda r: r["sum_us"], reverse=True)
    return host, all_ranges, dict(complete_events=sum(len(v) for v in groups.values()),
                                  metadata_flow_or_other_events=non_complete,
                                  marker_events=len(modules))


def report_capture(manifest):
    if manifest["status"] != "complete" or manifest["errors"] or manifest["completed"] != manifest["active"]:
        raise ValueError("Capture did not finalize; do not label partial results as a complete profile")
    trace = Path(manifest["trace_dir"])
    source = trace / "ASCEND_PROFILER_OUTPUT"
    kernels, device = kernel_report(source / "kernel_details.csv")
    host, ranges, timeline = trace_report(source / "trace_view.json")
    out = trace / "serve_summary"
    out.mkdir(exist_ok=True)
    save_csv(out / "all_device_ops.csv", kernels, ("name", "type", "core", "count", "sum_us", "mean_us", "min_us", "max_us"))
    save_csv(out / "all_trace_ranges.csv", ranges, ("pid", "tid", "category", "name", "count", "sum_us", "mean_us", "min_us", "max_us"))
    save_csv(out / "module_cpu.csv", host, ("pid", "tid", "name", "count", "inclusive_cpu_us", "self_cpu_us"))
    inventory = json.loads(Path(manifest["inventory"]).read_text(encoding="utf-8"))
    observed = Counter()
    for row in host:
        observed[row["name"]] += row["count"]
    coverage = [dict(label=t["label"], wrapped=t["wrapped"],
                     calls=manifest["hits"].get(t["label"], 0), timed_range=t["label"] in observed,
                     timed_calls=observed[t["label"]],
                     reason=t.get("reason", "")) for t in inventory["targets"]]
    # Zero calls can mean an inactive branch (e.g. audio on text requests).
    # Positive calls without any exported range is an explicit audit failure.
    missing = [r["label"] for r in coverage if r["calls"] != r["timed_calls"]]
    save_csv(out / "module_coverage.csv", coverage, ("label", "wrapped", "calls", "timed_range", "timed_calls", "reason"))
    summary = dict(identity=manifest["identity"], device=device, timeline=timeline,
                   capture_complete=True, missing_executed_module_ranges=missing,
                   all_exported_kernel_rows_accounted_for=True,
                   note="CPU self excludes nested serve markers only. Device task sums can overlap; "
                        "no critical-path or automatic GDN/Attention/MoE device attribution. "
                        "Original trace retains flows, metadata, memory and communication views.")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tp={manifest['identity']['tp_rank']} all_kernel_rows={device['rows']} "
          f"unique_ops={len(kernels)} summary={out}", flush=True)
    if missing:
        raise ValueError(f"{len(missing)} executed markers missing in parsed trace; see module_coverage.csv")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    report_capture(json.loads(args.manifest.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
