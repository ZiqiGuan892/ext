#!/usr/bin/env python3
"""Summarize fine MoE host ranges and retain all flow-associated device tables.

Default input is $PROFILE_RUN/capture. Select matching execute IDs across TP
ranks; never compare an unlabelled mixture of prefill and decode as pure decode.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import diagnose_forward_ops as diag


def moe_ranges(events, steps):
    xs = [diag.complete(e) for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    stages = defaultdict(lambda: [0, 0, 0])
    for step in steps:
        execute = [r for r in xs if r["name"] == f"serve/execute/{step}"]
        if len(execute) != 1:
            raise ValueError(f"Missing/ambiguous execute/{step}")
        fw = [r for r in xs if r["name"] == "serve/runner/_model_forward" and diag.contains(execute[0], r)]
        if len(fw) != 1:
            raise ValueError(f"Missing/ambiguous forward in execute/{step}")
        root, _ = diag.tree(fw[0], xs)

        def visit(node, owner=None):
            if node["name"].startswith("serve/moe/"):
                owner = node["name"]
                stages[owner][0] += node["b"]-node["a"]
                stages[owner][2] += 1
            if owner is not None:
                stages[owner][1] += node["own"]
            for child in node["children"]:
                visit(child, owner)

        visit(root)
    return stages


def phase_rows(manifest, steps):
    by_step = {row["iteration"]: row for row in manifest["iterations"]}
    rows = []
    for step in steps:
        row = by_step[step]
        metadata = row.get("attention_metadata") or []
        if any((m.get("num_prefills") or 0) > 0 or (m.get("num_prefill_tokens") or 0) > 0 for m in metadata):
            raise ValueError(f"execute/{step} contains prefill; select other --steps for decode comparison")
        confirmed = bool(metadata) and all(
            (m.get("num_prefills") == 0 or m.get("num_prefill_tokens") == 0)
            and ((m.get("num_decodes") or 0) > 0 or (m.get("num_decode_tokens") or 0) > 0)
            for m in metadata)
        rows.append(dict(iteration=step, phase="decode" if confirmed else "unknown",
                         tokens=row.get("scheduled_tokens"), requests=row.get("scheduled_requests")))
    return rows


def summarize(profile_dir, steps, expected):
    manifests = {}
    for path in profile_dir.glob("serve_pid*_tp*.json"):
        if path.name.endswith(".inventory.json"):
            continue
        m = diag.read(path)
        rank = int(m["identity"]["tp_rank"])
        if rank in manifests:
            raise ValueError(f"Duplicate TP{rank}; use one capture directory")
        if m.get("status") != "complete" or m.get("errors") or m.get("completed") != m.get("active"):
            raise ValueError(f"TP{rank}: capture incomplete")
        if not m.get("moe_detail"):
            raise ValueError(f"TP{rank}: MOE_DETAIL was not enabled")
        manifests[rank] = m
    if set(manifests) != set(range(expected)):
        raise ValueError(f"Expected TP0..{expected-1}, found {sorted(manifests)}")
    signatures = {tuple((row["tokens"], row["requests"]) for row in phase_rows(m, steps))
                  for m in manifests.values()}
    if len(signatures) != 1:
        raise ValueError("Selected iterations have different token/request counts across ranks")
    out = profile_dir.parent / "moe_detail_diag"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for rank, m in sorted(manifests.items()):
        phases = phase_rows(m, steps)
        print(f"\nTP{rank}: {json.dumps(phases, ensure_ascii=False)}", flush=True)
        raw = diag.read(Path(m["trace_dir"]) / "ASCEND_PROFILER_OUTPUT/trace_view.json")
        events = raw if isinstance(raw, list) else raw["traceEvents"]
        # Check every executed fine marker, including iterations outside the selected subset.
        observed = Counter(e.get("name") for e in events if e.get("ph") == "X")
        expected_hits = {k: v for k, v in m["hits"].items() if k.startswith("serve/moe/")}
        if not expected_hits or any(observed[k] != v for k, v in expected_hits.items()):
            raise ValueError(f"TP{rank}: fine marker call counts do not match capture; parse complete trace first")
        stages = moe_ranges(events, steps)
        if not stages:
            raise ValueError(f"TP{rank}: no fine MoE calls in selected iterations")
        result = diag.analyse(events, steps=steps)
        if result["cpu_errors"]:
            raise ValueError(f"TP{rank}: {result['cpu_errors']}")
        diag.save(rank, result, out, iterations=len(steps))
        scale = len(steps)*1_000_000
        device_stages = defaultdict(lambda: [0, 0])
        for row in result["audit"]:
            if row["status"] != "attributed" or not row["moe_stage"]:
                continue
            parts = row["moe_stage"].split("/")
            kind = "synchronization_event" if row["name"] == "EVENT_WAIT" else row["kind"]
            bucket = device_stages[(parts[2]+"/"+parts[-1], kind, row["name"])]
            bucket[0] += row["duration_ns"]
            bucket[1] += 1
        device_rows = [dict(stage=k[0], kind=k[1], name=k[2], sum_ms_per_iter=v[0]/scale,
                            calls=v[1]) for k, v in device_stages.items()]
        device_rows.sort(key=lambda r: r["sum_ms_per_iter"], reverse=True)
        diag.write_csv(out / f"tp{rank}_moe_device_ops.csv", device_rows,
                       ("stage", "kind", "name", "sum_ms_per_iter", "calls"))
        rows = [dict(label=label, inclusive_ms_per_iter=v[0]/scale,
                     exclusive_ms_per_iter=v[1]/scale, calls=v[2]) for label, v in stages.items()]
        diag.write_csv(out / f"tp{rank}_moe_layers.csv", rows,
                       ("label", "inclusive_ms_per_iter", "exclusive_ms_per_iter", "calls"))
        groups = defaultdict(lambda: [0, 0, 0])
        for label, values in stages.items():
            parts = label.split("/")
            group = parts[2]+"/"+parts[-1]
            for i in range(3):
                groups[group][i] += values[i]
        grouped = [dict(stage=k, inclusive_ms_per_iter=v[0]/scale,
                        exclusive_ms_per_iter=v[1]/scale, calls=v[2]) for k, v in groups.items()]
        grouped.sort(key=lambda row: row["inclusive_ms_per_iter"], reverse=True)
        diag.write_csv(out / f"tp{rank}_moe_stages.csv", grouped,
                       ("stage", "inclusive_ms_per_iter", "exclusive_ms_per_iter", "calls"))
        print(f"forward_CPU={result['total']/scale:.6f} ms/iter")
        print("MoE CPU: inclusive_ms/iter exclusive_ms/iter calls stage")
        for row in grouped:
            print(f"  {row['inclusive_ms_per_iter']:.6f} {row['exclusive_ms_per_iter']:.6f} "
                  f"{row['calls']} {row['stage']}")
        residual = sum(v[0] for k, v in result["cpu"].items()
                       if k[2] == "vllm::moe_forward_shared")/scale
        print(f"moe_forward_shared remaining recorded self={residual:.6f} ms/iter")
        print("按 CPU flow 来源关联的 MoE 设备事件（完整表已保存；时长可能重叠）：")
        for row in device_rows[:8]:
            print(f"  {row['sum_ms_per_iter']:.6f} ms/iter calls={row['calls']} "
                  f"{row['stage']} | {row['kind']} | {row['name']}")
        branch_rows = [dict(iteration=row["iteration"], **observation) for row in m["iterations"]
                       if row["iteration"] in steps for observation in row.get("moe_observations", [])]
        (out / f"tp{rank}_branches.json").write_text(json.dumps(branch_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        branch_counts = Counter((row["label"].rsplit("/", 1)[-1], json.dumps(row["args"], default=str),
                                 json.dumps(row.get("result"), default=str), json.dumps(row.get("comm_type"), default=str))
                                for row in branch_rows if row["label"].endswith((
                                    "/_can_overlap_sp_shared_with", "/_maybe_reduce_final_output",
                                    "/_maybe_reduce_shared_expert_output", "/_forward_impl",
                                    "/parallel_mode", "/_should_reduce_routed_before_combine")))
        print("代表层的实际调用参数/结果（Tensor只记录类型）：", m.get("moe_probe_path"))
        for key, count in sorted(branch_counts.items()):
            print(f"  calls={count} method={key[0]} args={key[1]} result={key[2]} comm={key[3]}")
        inventory = diag.read(Path(m["inventory"]))
        roots = inventory.get("moe_roots", [])
        print("MoE layers:", len(roots), "first layer stored flags:",
              json.dumps({k: roots[0].get(k) for k in ("runner", "routed", "shared")}, ensure_ascii=False, default=str) if roots else "missing")
        summary[rank] = dict(phases=phases, forward_cpu_ms_per_iter=result["total"]/scale,
                             moe_stages=grouped, custom_op_residual_ms_per_iter=residual)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整 CPU/设备/flow 审计和 MoE 分层表：{out}")
    print("exclusive 只扣除嵌套的细分 MoE 范围，仍含 CPU 下发及等待；inclusive 不能逐层相加。")
    print("设备事件累计时间可能重叠；本报告没有把 Event 等待当作计算或网络传输时间。")
    print("新增标记改变了 self 时间的归属且增加观测开销，不能把旧 self 与新 self 的差值当作加速。")
    print("phase=unknown 表示当前 metadata 无法确认纯 decode；不能把它标作已确认的 decode 性能。")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--steps", default="6,7,8,9")
    parser.add_argument("--expected-workers", type=int, default=4)
    args = parser.parse_args()
    root = args.profile_dir or (Path(os.environ["PROFILE_RUN"])/"capture" if os.environ.get("PROFILE_RUN") else None)
    if root is None:
        parser.error("Set PROFILE_RUN or --profile-dir")
    try:
        steps = tuple(int(x) for x in args.steps.split(","))
        if not steps or len(set(steps)) != len(steps) or min(steps) < 1 or args.expected_workers < 1:
            raise ValueError("Steps must be unique positive integers; expected-workers must be positive")
        summarize(root, steps, args.expected_workers)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"诊断失败：{exc}\n")


if __name__ == "__main__":
    main()
