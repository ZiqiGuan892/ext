"""Pair sampled engine spans before attributing host wall time to phases.

Fractions describe observed execute_model calls in one process, not gen wall
time or device utilization. Missing hooks remain unattributed. First iterations
are separated because periodic sampling otherwise overweights cold calls.
"""

from collections import Counter, defaultdict
import json
import math
import statistics


PREFIX = "[huanyu-gen-timing] "
ROOT = "engine.execute_model"
OTHER = "engine.unattributed"
PHASES = (
    "engine.update_states", "engine.prepare_inputs", "engine.preprocess.other",
    "engine.mm_encoder", "engine.gather_mm_embeddings", "engine.forward",
    "engine.compute_logits", "engine.sample_tokens", OTHER,
)


def _label(phase):
    return "engine.preprocess.other" if phase == "engine.preprocess" else phase


def _scope(row):
    return row.get("host", "<unknown>"), row.get("pid", "<unknown>")


def _iteration(row):
    return (*_scope(row), row.get("model_path"), row.get("engine_iteration"))


def _group(row):
    iteration = row.get("engine_iteration")
    period = "first" if iteration == 1 else "later" if type(iteration) is int else "unknown"
    return (*_scope(row), row.get("model_path", "<unknown>"), row.get("runner_class", "<unknown>"), period)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _partition(root, descendants, children):
    """Return exclusive phase seconds only for a consistent, non-overlapping tree."""
    monotonic = "start_monotonic_s" in root
    start_key, end_key, divisor = (
        ("start_monotonic_s", "end_monotonic_s", 1.0) if monotonic else
        ("start_unix_ns", "end_unix_ns", 1e9)
    )
    origin = root.get(start_key)
    if not _number(origin):
        raise ValueError("missing_interval")

    def interval(row):
        start, end, elapsed = row.get(start_key), row.get(end_key), row.get("elapsed_s")
        if not all(_number(v) for v in (start, end, elapsed)) or end < start or elapsed < 0:
            raise ValueError("invalid_interval")
        duration = (end - start) / divisor
        # Old logs have wall-clock endpoints but monotonic elapsed_s. Do not
        # turn a clock jump into an apparent preprocessing or forward cost.
        tolerance = 1e-6 if monotonic else max(0.005, elapsed * 0.01)
        if abs(duration - elapsed) > tolerance:
            raise ValueError("interval_clock_mismatch")
        return (start - origin) / divisor, (end - origin) / divisor

    bounds = {id(row): interval(row) for row in [root, *descendants]}
    width = bounds[id(root)][1]
    if width <= 0:
        raise ValueError("empty_root_interval")
    scale = root["elapsed_s"] / width
    totals = Counter()
    calls = Counter()

    def visit(row):
        start, end = bounds[id(row)]
        nested = sorted(children.get((*_scope(row), row["span_id"]), ()),
                        key=lambda r: bounds[id(r)][0])
        previous_end, covered = start, 0.0
        for child in nested:
            lo, hi = bounds[id(child)]
            if lo < start - 1e-6 or hi > end + 1e-6:
                raise ValueError("child_outside_parent")
            if lo < previous_end - 1e-6:
                raise ValueError("overlapping_siblings")
            covered += max(0.0, min(hi, end) - max(lo, previous_end, start))
            previous_end = max(previous_end, hi)
            visit(child)
        label = OTHER if row is root else _label(row["phase"])
        totals[label] += max(0.0, end - start - covered) * scale
        calls[label] += 1

    visit(root)
    return dict(totals), dict(calls), "monotonic" if monotonic else "legacy_wall_checked"


def engine_breakdown(log, *, model=None, host=None, pid=None):
    rows, conflicts, diagnostics = {}, set(), Counter()
    outer_parents, outer_conflicts = {}, set()
    for line in log:
        if PREFIX not in line:
            continue
        try:
            row, _ = json.JSONDecoder().raw_decode(line.split(PREFIX, 1)[1].lstrip())
            if not isinstance(row, dict):
                raise ValueError("not an object")
        except (ValueError, TypeError):
            diagnostics["malformed_records"] += 1
            continue
        if row.get("event") != "span":
            continue
        if not str(row.get("phase", "")).startswith("engine."):
            # A separate sampler RPC may inherit a non-engine timing context.
            # Keep its ancestry even when model filtering excludes that context.
            if isinstance(row.get("span_id"), str):
                key = (*_scope(row), row["span_id"])
                parent = row.get("parent_span_id")
                if key in outer_parents and outer_parents[key] != parent:
                    outer_conflicts.add(key)
                outer_parents[key] = parent
            continue
        if ((model and model not in str(row.get("model_path", "")))
                or (host and row.get("host") != host) or (pid is not None and row.get("pid") != pid)):
            continue
        if not isinstance(row.get("span_id"), str):
            diagnostics["missing_span_id"] += 1
            continue
        key = (*_scope(row), row["span_id"])
        if key in rows:
            diagnostics["duplicate_records"] += 1
            if rows[key] != row:
                conflicts.add(key)
            continue
        rows[key] = row

    children, iterations, groups = defaultdict(list), defaultdict(list), {}
    for row in rows.values():
        parent = row.get("parent_span_id")
        if parent:
            children[(*_scope(row), parent)].append(row)
        iterations[_iteration(row)].append(row)

    def group_for(row):
        key = _group(row)
        if key not in groups:
            groups[key] = dict(host=key[0], pid=key[1], model_path=key[2], runner_class=key[3],
                               period=key[4], seen=0, accepted=[], skipped=Counter(),
                               outside=[], errors=Counter(), verified=0, sample_every=set())
        return groups[key]

    def outside_sampler(row):
        if row["phase"] != "engine.sample_tokens":
            return False
        parent, seen = row.get("parent_span_id"), set()
        while parent:
            key = (*_scope(row), parent)
            if key in seen or key in rows or key in outer_conflicts or key not in outer_parents:
                return False  # An engine ancestor or an incomplete parent chain.
            seen.add(key)
            parent = outer_parents[key]
        return True

    assigned = set()
    for root in rows.values():
        if root["phase"] != ROOT:
            continue
        group = group_for(root)
        group["seen"] += 1
        group["sample_every"].add(str(root.get("sample_every", "unknown")))
        root_key = (*_scope(root), root["span_id"])
        seen, descendants, stack = {root_key}, [], list(children.get(root_key, ()))
        try:
            if root.get("status", "ok") != "ok":
                raise ValueError("execute_model_error")
            while stack:
                row = stack.pop()
                key = (*_scope(row), row["span_id"])
                if key in seen:
                    raise ValueError("cyclic_span_tree")
                seen.add(key)
                if _iteration(row) != _iteration(root):
                    raise ValueError("iteration_mismatch")
                descendants.append(row)
                stack.extend(children.get(key, ()))
            if conflicts & seen:
                raise ValueError("conflicting_duplicates")
            for row in iterations[_iteration(root)]:
                key = (*_scope(row), row["span_id"])
                if key not in seen and row.get("parent_span_id") and not outside_sampler(row):
                    raise ValueError("unattached_nested_spans")
            expected = root.get("nested_span_counts")
            observed = Counter(row["phase"] for row in descendants)
            if expected is not None:
                if not isinstance(expected, dict) or expected != dict(observed):
                    raise ValueError("missing_or_extra_child_spans")
            totals, calls, clock = _partition(root, descendants, children)
            group["accepted"].append(dict(iteration=root.get("engine_iteration"),
                                           elapsed_s=root["elapsed_s"], totals=totals, calls=calls,
                                           interval_clock=clock,
                                           scheduled_encoder_requests=root.get("scheduled_encoder_requests")))
            group["verified"] += expected is not None
        except ValueError as error:
            group["skipped"][str(error)] += 1
        assigned.update(seen)

    for key, row in rows.items():
        if row.get("status") == "error":
            group_for(row)["errors"][(row["phase"], row.get("error_type", "unknown"))] += 1
        if key in assigned:
            continue
        if outside_sampler(row):
            if _number(row.get("elapsed_s")) and row["elapsed_s"] >= 0 and key not in conflicts:
                group_for(row)["outside"].append(row)
        else:
            diagnostics["orphan_spans"] += 1
    return dict(groups=list(groups.values()), diagnostics=dict(diagnostics))


def _p95(values):
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def print_engine_breakdown(report):
    print("Exclusive HOST-WALL time within sampled execute_model calls; NOT gen wall or NPU kernel fractions.")
    print("Processes/models and first/later iterations are separate. Unobserved hooks remain unattributed.")
    for group in sorted(report["groups"], key=lambda g: (str(g["model_path"]), str(g["host"]),
                                                         str(g["pid"]), g["period"])):
        print("\nmodel=%s host=%s pid=%s runner=%s iterations=%s" %
              tuple(group[k] for k in ("model_path", "host", "pid", "runner_class", "period")))
        accepted = group["accepted"]
        print("execute_seen=%d aligned=%d child_counts_verified=%d sample_every=%s skipped=%s" %
              (group["seen"], len(accepted), group["verified"], ",".join(sorted(group["sample_every"])),
               dict(group["skipped"])))
        if accepted:
            denominator = sum(row["elapsed_s"] for row in accepted)
            encoder_calls = sum((row["scheduled_encoder_requests"] or 0) > 0 for row in accepted)
            print("observed_execute_s=%.6f iterations_with_encoder_inputs=%d interval_clock=%s" %
                  (denominator, encoder_calls, sorted({r["interval_clock"] for r in accepted})))
            if group["verified"] != len(accepted):
                print("LEGACY: child-span completeness cannot be verified; percentages cover recorded intervals only.")
            print("phase\tcalls\tmean_s/iteration\tp50_s\tp95_s\tmax_s\texclusive_s\tobserved_execute_%")
            phases = sorted(set(PHASES) | {p for r in accepted for p in r["totals"]})
            for phase in phases:
                count = sum(row["calls"].get(phase, 0) for row in accepted)
                if count == 0:
                    print(f"{phase}\tUNOBSERVED")
                    continue
                values = [row["totals"].get(phase, 0.0) for row in accepted]
                total = sum(values)
                print("%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f" %
                      (phase, count, statistics.mean(values), statistics.median(values), _p95(values),
                       max(values), total, 100 * total / denominator if denominator else 0))
        if group["outside"]:
            values = [row["elapsed_s"] for row in group["outside"]]
            print("sample_tokens_outside_execute: calls=%d mean_s=%.6f p95_s=%.6f max_s=%.6f "
                  "(excluded from execute fractions)" % (len(values), statistics.mean(values), _p95(values), max(values)))
        if group["errors"]:
            print("error_types (including caught/retried calls):", dict(group["errors"]))
    if not report["groups"]:
        print("No matching engine spans.")
    if report["diagnostics"]:
        print("\nlog_diagnostics:", report["diagnostics"])
