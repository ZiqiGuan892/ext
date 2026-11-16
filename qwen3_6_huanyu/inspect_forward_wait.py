#!/usr/bin/env python3
"""Inspect existing EVENT_WAIT flows and installed MoE source, without importing NPU libraries.

Run after diagnose_forward_ops.py (which may have been renamed test.py).
All paths are fixed; this file has no dependencies beyond the standard library.
It does not pair reused event IDs or infer dependencies from temporal proximity.
"""
import ast
import csv
import hashlib
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

PROFILE_DIR = Path("/tmp/huanyu_text_eager_tp4.Ogud1q/capture")
AUDIT_DIR = PROFILE_DIR.parent / "forward_ops_diag"
OUTPUT_DIR = PROFILE_DIR.parent / "forward_wait_diag"
STEPS = (6, 7, 8, 9)
RANKS = (0, 2)
SOURCE_ROOTS = (
    Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe"),
    Path("/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe"),
)


def ns(value):
    return int(Decimal(str(value)) * 1000)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def key(event):
    return str(event["pid"]), str(event["tid"]), ns(event["ts"])


def dump(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def union_ns(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("Negative duration")
        total += max(0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total


def flow_key(event):
    ident = event.get("id2", event.get("id"))
    local = str(event.get("pid")) if isinstance(ident, dict) and "local" in ident else ""
    return event.get("cat"), event.get("name"), event.get("scope"), local, dump(ident)


def inspect_waits(events, audit):
    points, lanes, metadata = defaultdict(list), defaultdict(list), {}
    starts, h2d = defaultdict(list), defaultdict(set)
    for event in events:
        if event.get("ph") == "M" and event.get("name") in ("process_name", "thread_name"):
            meta_key = (str(event.get("pid")), str(event.get("tid")))
            if event["name"] == "process_name":
                meta_key = (str(event.get("pid")), "*")
            metadata[meta_key] = event.get("args", {}).get("name", "")
        if "ts" not in event or "pid" not in event or "tid" not in event:
            continue
        if event.get("ph") == "X" and "dur" in event:
            point = key(event)
            if event not in points[point]:
                points[point].append(event)
                lanes[point[:2]].append(event)
        if event.get("cat") == "HostToDevice" and ("id" in event or "id2" in event):
            if event.get("ph") == "s":
                starts[flow_key(event)].append(event)
            elif event.get("ph") == "f":
                h2d[key(event)].add(flow_key(event))
    for rows in lanes.values():
        rows.sort(key=lambda e: ns(e["ts"]))
    times = {lane: [ns(e["ts"]) for e in rows] for lane, rows in lanes.items()}
    records = []
    for row in audit:
        if row["status"] != "attributed" or row["name"] != "EVENT_WAIT":
            continue
        if int(row["step"]) not in STEPS:
            continue
        dest = row["pid"], row["tid"], int(row["start_ns"])
        source = row["source_pid"], row["source_tid"], int(row["source_ns"])
        candidates = points.get(dest, [])
        if len(candidates) != 1 or candidates[0]["name"] != "EVENT_WAIT":
            raise ValueError(f"Audit and trace disagree at {dest}")
        event = candidates[0]
        if ns(event["dur"]) != int(row["duration_ns"]):
            raise ValueError(f"Audit duration and trace disagree at {dest}")
        lane_rows = lanes.get(source[:2], [])
        cutoff = bisect_right(times.get(source[:2], []), source[2])
        ancestors = [e for e in lane_rows[:cutoff]
                     if ns(e["ts"]) <= source[2] < ns(e["ts"])+ns(e["dur"])]
        ancestors.sort(key=lambda e: (ns(e["ts"]), -ns(e["dur"])))
        source_names = sorted({e["name"] for e in points.get(source, [])})
        # Keep every explicit HostToDevice source. Multiple sources are not arbitrated.
        cann = []
        for ident in sorted(h2d.get(dest, set()), key=str):
            for start in starts.get(ident, []):
                cann.append(dict(flow_start=start, ranges=points.get(key(start), [])))
        records.append(dict(
            audit=row, device_event=event, source_names=source_names,
            cpu_ancestors=ancestors, cann_sources=cann,
            process_name=metadata.get((dest[0], "*"), ""),
            thread_name=metadata.get(dest[:2], ""),
        ))
    return records


def report_waits(rank, records):
    groups, lanes = defaultdict(list), defaultdict(list)
    for record in records:
        row = record["audit"]
        a, d = int(row["start_ns"]), int(row["duration_ns"])
        groups[(row["owner"], " | ".join(record["source_names"]) or "UNRESOLVED")].append(d)
        lanes[(row["pid"], row["tid"], record["thread_name"])].append((a, a+d))
    scale = len(STEPS)*1_000_000
    print(f"\nTP{rank}: selected EVENT_WAIT={len(records)}")
    for group, durations in sorted(groups.items(), key=lambda item: sum(item[1]), reverse=True):
        print(f"  sum={sum(durations)/scale:.6f} ms/iter calls={len(durations)} "
              f"source={group[0]} | {group[1]}")
    print("按 trace lane 的等待区间并集（不是整个设备的空闲时间）：")
    for lane, intervals in sorted(lanes.items()):
        print(f"  {lane}: union={union_ns(intervals)/scale:.6f} ms/iter "
              f"sum={sum(b-a for a,b in intervals)/scale:.6f} ms/iter")
    # Two examples from each distinct CPU source; all records are saved in JSON.
    chosen = Counter()
    for record in sorted(records, key=lambda r: int(r["audit"]["duration_ns"]), reverse=True):
        source = tuple(record["source_names"])
        if chosen[source] >= 2:
            continue
        chosen[source] += 1
        row = record["audit"]
        print(f"\n  step={row['step']} owner={row['owner']} "
              f"wait={int(row['duration_ns'])/1e6:.6f} ms source={source}")
        print("  device:", dump(record["device_event"]))
        print("  CPU chain:", " -> ".join(e["name"] for e in record["cpu_ancestors"][-10:]))
        print("  CANN flow sources:", dump(record["cann_sources"]))


def source_sections(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("AscendMoERunner", "MoERunner"):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name in (
                        "forward", "forward_impl", "_forward_impl", "_select_forward"):
                    selected.append((f"{node.name}.{child.name}", child))
        if isinstance(node, ast.FunctionDef) and node.name in ("moe_forward_shared", "_moe_forward_shared"):
            selected.append((node.name, node))
    return [(name, node.lineno, lines[node.lineno-1:node.end_lineno]) for name, node in selected]


def inspect_sources(manifests, output):
    expected = defaultdict(set)
    paths = set()
    for manifest in manifests:
        inventory = Path(manifest.get("inventory") or "/__missing_inventory__")
        if not inventory.is_file():
            print("没有 capture inventory:", inventory)
            continue
        data = load(inventory)
        for path, digest in data.get("source_hashes", {}).items():
            expected[path].add(digest)
        for target in data.get("targets", []):
            if "AscendMoERunner" in target.get("label", ""):
                paths.update(Path(source["file"]) for source in target.get("source", []))
    for root in SOURCE_ROOTS:
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    full = []
    for path in sorted(paths):
        if not path.is_file():
            print("采集时源码路径当前不存在:", path)
            continue
        try:
            sections = source_sections(path)
        except (OSError, SyntaxError, UnicodeError) as exc:
            print("源码检查失败:", path, exc)
            continue
        if not sections:
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = expected.get(str(path), set())
        match = "MATCH" if recorded == {actual} else "CHANGED" if recorded else "NOT_RECORDED"
        banner = f"{path} sha256={actual} capture_hash={match}"
        print("\nSOURCE", banner)
        full.append(banner)
        for name, first, lines in sections:
            print(f"  {name}:{first} ({len(lines)} lines; full source saved)")
            full.append("\n"+name)
            full.extend(f"{first+i:5}: {line}" for i, line in enumerate(lines))
            tokens = ("event", "stream", "synchronize", "overlap", "forward_impl", "quant_method", "prepare", "finalize")
            for i, line in enumerate(lines):
                if any(token in line.lower() for token in tokens):
                    print(f"  {first+i}: {line.strip()}")
    target = output / "moe_source.txt"
    target.write_text("\n".join(full)+"\n", encoding="utf-8")
    if not full:
        print("未找到 MoE 源码；不影响已有 trace 的等待分析。")
    print("完整相关源码:", target)


def main():
    print(f"PROFILE_DIR={PROFILE_DIR}\nOUTPUT_DIR={OUTPUT_DIR}", flush=True)
    manifests = defaultdict(list)
    for path in PROFILE_DIR.glob("serve_pid*_tp*.json"):
        if path.name.endswith(".inventory.json"):
            continue
        manifest = load(path)
        rank = int(manifest["identity"]["tp_rank"])
        if rank in RANKS:
            manifests[rank].append(manifest)
    if set(manifests) != set(RANKS) or any(len(rows) != 1 for rows in manifests.values()):
        raise ValueError("TP0/TP2 manifest 缺失或不唯一")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank in RANKS:
        manifest = manifests[rank][0]
        if manifest.get("status") != "complete" or manifest.get("errors"):
            raise ValueError(f"TP{rank}: capture 未完整结束")
        with (AUDIT_DIR / f"tp{rank}_flow_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
            audit = list(csv.DictReader(stream))
        data = load(Path(manifest["trace_dir"]) / "ASCEND_PROFILER_OUTPUT/trace_view.json")
        records = inspect_waits(data if isinstance(data, list) else data["traceEvents"], audit)
        (OUTPUT_DIR / f"tp{rank}_waits.json").write_text(dump(records)+"\n", encoding="utf-8")
        report_waits(rank, records)
    inspect_sources([manifests[rank][0] for rank in RANKS], OUTPUT_DIR)
    print("\nCPU source 表示等待任务从哪里下发，不表示它在等待哪个任务。")
    print("不会仅凭相同 Event ID 或时间接近配对 RECORD；全部原始参数保存在 tp*_waits.json。")
    print("同 lane 并集也不能与 HCCL/AICPU 时间相加；本脚本没有计算关键路径或无 profiler 的耗时。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"诊断失败：{exc}") from exc
