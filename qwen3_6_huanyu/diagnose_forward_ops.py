#!/usr/bin/env python3
"""Read an existing eager capture; compare framework self time and flow-linked NPU events.

No arguments or nonstandard packages are required. CPU scope selection uses the
launching thread in execute calls 6-9; a device event may finish after that CPU
scope. Only torch_to_npu flows establish device ownership. Task queue dequeue
events are never treated as device kernels. Full CSVs retain all linked endpoint
audit records; unmapped devices cannot be assigned to an iteration.
"""
import csv
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

PROFILE_DIR = Path("/tmp/huanyu_text_eager_tp4.Ogud1q/capture")
OUTPUT_DIR = PROFILE_DIR.parent / "forward_ops_diag"
STEPS = (6, 7, 8, 9)
RANKS = (0, 2)
SCALE = len(STEPS) * 1_000_000


def ns(value):
    return int(Decimal(str(value)) * 1000)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def lane(event):
    return str(event.get("pid")), str(event.get("tid"))


def point(event):
    return lane(event) + (ns(event["ts"]),)


def complete(event):
    a = ns(event["ts"])
    d = ns(event["dur"])
    if d < 0:
        raise ValueError("Negative duration")
    return dict(a=a, b=a+d, lane=lane(event),
                name=str(event.get("name", "")), cat=str(event.get("cat", "")))


def contains(outer, inner):
    return (outer["lane"] == inner["lane"]
            and outer["a"] <= inner["a"] <= inner["b"] <= outer["b"])


def tree(root, rows):
    """Build strictly nested ranges; reject crossing intervals instead of guessing."""
    nodes = [dict(r, children=[]) for r in rows if contains(root, r)]
    nodes.sort(key=lambda r: (
        r["a"], -r["b"], r != root and not (
            r["name"] == root["name"] and r["a"] == root["a"] and r["b"] == root["b"]),
        not r["name"].startswith("serve/"), r["name"]))
    stack = []
    for node in nodes:
        while stack and not contains(stack[-1], node):
            old = stack.pop()
            if node["a"] < old["b"]:
                raise ValueError(f"Crossing CPU ranges: {old['name']} / {node['name']}")
        if not stack and nodes and node is not nodes[0]:
            raise ValueError("CPU ranges have multiple roots")
        inherited = stack[-1]["owner"] if stack else "Forward_unclassified"
        module = node["name"].startswith("serve/module/")
        node["owner"] = node["name"].split("/")[2] if module else inherited
        node["module"] = node["name"] if module else (
            stack[-1]["module"] if stack else "")
        node["moe_stage"] = node["name"] if node["name"].startswith("serve/moe/") else (
            stack[-1]["moe_stage"] if stack else "")
        if stack:
            stack[-1]["children"].append(node)
        stack.append(node)
    if not nodes:
        raise ValueError("Empty CPU range tree")
    for node in nodes:
        node["starts"] = [c["a"] for c in node["children"]]
        node["own"] = node["b"]-node["a"]-sum(c["b"]-c["a"] for c in node["children"])
        if node["own"] < 0:
            raise ValueError("Negative CPU self time")
    if sum(n["own"] for n in nodes) != root["b"]-root["a"]:
        raise ValueError("CPU self coverage mismatch")
    return nodes[0], nodes


def context_at(roots, source):
    for step, root in roots:
        if root["lane"] != source[:2] or not root["a"] <= source[2] < root["b"]:
            continue
        node = root
        while True:
            i = bisect_right(node["starts"], source[2])-1
            if i < 0:
                break
            child = node["children"][i]
            if not child["a"] <= source[2] < child["b"]:
                break
            node = child
        return step, node["owner"], node["module"], node["moe_stage"]
    return None


def flow_key(event):
    ident = event.get("id2", event.get("id"))
    local = str(event.get("pid")) if isinstance(ident, dict) and "local" in ident else ""
    return (event.get("cat"), event.get("name"), event.get("scope"), local,
            json.dumps(ident, sort_keys=True, default=str))


def device_kind(name):
    name = name.lower()
    if name.startswith("hcom_"):
        return "HCCL_range"
    if "aicpu" in name and any(p in name for p in (
            "allreduce", "allgather", "reducescatter", "alltoall", "broadcast")):
        return "collective_AICPU"
    return "other_device_event"


def analyse(events, steps=STEPS):
    xs = [complete(e) for e in events
          if e.get("ph") == "X" and "ts" in e and "dur" in e]
    markers = [r for r in xs if r["name"].startswith("serve/")]
    cpu, roots, total, cpu_errors = defaultdict(lambda: [0, 0]), [], 0, []
    for step in steps:
        execute = [r for r in markers if r["name"] == f"serve/execute/{step}"]
        if len(execute) != 1:
            raise ValueError(f"execute/{step}: expected one range, found {len(execute)}")
        forward = [r for r in markers if r["name"] == "serve/runner/_model_forward"
                   and contains(execute[0], r)]
        if len(forward) != 1:
            raise ValueError(f"execute/{step}: forward range missing or ambiguous")
        fw = forward[0]
        scope_root, _ = tree(fw, markers)
        roots.append((step, scope_root))
        total += fw["b"]-fw["a"]
        try:
            _, nodes = tree(fw, xs)
        except ValueError as exc:
            cpu_errors.append(f"step={step}: {exc}")
            continue
        for node in nodes:
            kind = "marker_gap" if node["name"].startswith("serve/") else "recorded_range_self"
            cpu[(node["owner"], kind, node["name"])][0] += node["own"]
            cpu[(node["owner"], kind, node["name"])][1] += 1
    # Never compare a partially accounted CPU table against a complete rank.
    if cpu_errors:
        cpu.clear()

    by_point = defaultdict(dict)
    for r in xs:
        by_point[r["lane"] + (r["a"],)][(r["name"], r["b"], r["cat"])] = r
    sources, targets = defaultdict(set), defaultdict(set)
    h2d_endpoints = set()
    for e in events:
        if "ts" not in e:
            continue
        if e.get("cat") == "HostToDevice" and e.get("ph") == "f":
            h2d_endpoints.add(point(e))
        if e.get("name") != "torch_to_npu" or e.get("cat") != "async_npu":
            continue
        if "id" not in e and "id2" not in e:
            continue
        if e.get("ph") == "s":
            sources[flow_key(e)].add(point(e))
        elif e.get("ph") == "f":
            targets[point(e)].add(flow_key(e))

    audit, devices = [], defaultdict(lambda: [0, 0])
    # H2D endpoints without a torch flow remain visible in the audit.
    for dest in sorted(set(targets) | h2d_endpoints):
        keys = targets.get(dest, set())
        src = set().union(*(sources.get(k, set()) for k in keys)) if keys else set()
        candidates = list(by_point.get(dest, {}).values())
        record = dict(pid=dest[0], tid=dest[1], start_ns=dest[2],
                      status="", step="", owner="", module="", moe_stage="", name="", kind="",
                      duration_ns="", source_pid="", source_tid="", source_ns="")
        if not keys:
            record["status"] = "no_torch_flow"
        elif any(not sources.get(k) for k in keys):
            record["status"] = "missing_flow_source"
        elif len(src) != 1:
            record["status"] = "ambiguous_flow_source"
        elif len(candidates) != 1:
            record["status"] = "missing_or_ambiguous_endpoint_range"
        else:
            source = next(iter(src))
            event = candidates[0]
            record.update(name=event["name"], duration_ns=event["b"]-event["a"],
                          source_pid=source[0], source_tid=source[1], source_ns=source[2])
            host_event = (event["cat"] in ("cpu_op", "enqueue", "dequeue", "python_function")
                          or event["name"].startswith(("Enqueue@", "Dequeue@", "serve/")))
            context = context_at(roots, source)
            if host_event or dest[:2] == source[:2]:
                record["status"] = "host_endpoint_rejected"
            elif context is None:
                record["status"] = "outside_selected_forward"
            else:
                step, owner, module, moe_stage = context
                kind = device_kind(event["name"])
                record.update(status="attributed", step=step, owner=owner, module=module,
                              moe_stage=moe_stage, kind=kind)
                devices[(owner, kind, event["name"])][0] += event["b"]-event["a"]
                devices[(owner, kind, event["name"])][1] += 1
        if candidates:
            record["name"] = record["name"] or " | ".join(r["name"] for r in candidates)
        audit.append(record)
    return dict(cpu=cpu, devices=devices, audit=audit, cpu_errors=cpu_errors, total=total)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(rank, result, output, iterations=len(STEPS)):
    for key in ("cpu", "devices"):
        rows = [dict(owner=o, kind=k, name=n, sum_ms=v[0]/1e6,
                     mean_ms_per_iter=v[0]/(iterations*1_000_000), calls=v[1])
                for (o, k, n), v in result[key].items()]
        rows.sort(key=lambda row: row["sum_ms"], reverse=True)
        write_csv(output / f"tp{rank}_{key}.csv", rows,
                  ("owner", "kind", "name", "sum_ms", "mean_ms_per_iter", "calls"))
    fields = ("pid", "tid", "start_ns", "status", "step", "owner", "module", "moe_stage", "name",
              "kind", "duration_ns", "source_pid", "source_tid", "source_ns")
    write_csv(output / f"tp{rank}_flow_audit.csv", result["audit"], fields)


def main():
    print(f"PROFILE_DIR={PROFILE_DIR}\nOUTPUT_DIR={OUTPUT_DIR}", flush=True)
    manifests = defaultdict(list)
    for p in PROFILE_DIR.glob("serve_pid*_tp*.json"):
        if p.name.endswith(".inventory.json"):
            continue
        m = read(p)
        rank = int(m["identity"]["tp_rank"])
        if rank in RANKS:
            manifests[rank].append(m)
    if set(manifests) != set(RANKS) or any(len(v) != 1 for v in manifests.values()):
        raise ValueError("TP0/TP2 manifest 缺失或不唯一")
    results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank in RANKS:
        m = manifests[rank][0]
        if m.get("status") != "complete" or m.get("errors"):
            raise ValueError(f"TP{rank}: capture未完整结束")
        trace = Path(m["trace_dir"]) / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        data = read(trace)
        result = analyse(data if isinstance(data, list) else data["traceEvents"])
        results[rank] = result
        save(rank, result, OUTPUT_DIR)
        print(f"\nTP{rank}: forward_CPU_ms/iter={result['total']/SCALE:.6f}")
        print("flow endpoint audit:", dict(Counter(r["status"] for r in result["audit"])))
        for error in result["cpu_errors"]:
            print("CPU self统计未生成:", error)

    for key, heading in (("cpu", "CPU self"),
                         ("devices", "flow关联设备事件duration之和（可重叠，不是墙钟时间）")):
        print(f"\n{heading}: TP0_ms/iter TP2_ms/iter delta calls0/calls2")
        if key == "cpu" and any(r["cpu_errors"] for r in results.values()):
            print("CPU范围存在交叉，跳过跨rank比较；详见上面的错误。")
            continue
        tables = {r: results[r][key] for r in RANKS}
        def val(rank, item):
            return tables[rank].get(item, [0, 0])[0]/SCALE
        keys = set(tables[0]) | set(tables[2])
        for item in sorted(keys, key=lambda k: max(val(0, k), val(2, k)), reverse=True)[:12]:
            a, b = val(0, item), val(2, item)
            counts = [tables[r].get(item, [0, 0])[1] for r in RANKS]
            print(f"{a:.6f} {b:.6f} {a-b:+.6f} {counts[0]}/{counts[1]} "
                  f"{item[0]} | {item[1]} | {item[2]}")
        if key == "devices":
            continue
        print("\nCPU self正差最大的8项:")
        for item in sorted(keys, key=lambda k: val(0, k)-val(2, k), reverse=True)[:8]:
            a, b = val(0, item), val(2, item)
            print(f"{a:.6f} {b:.6f} {a-b:+.6f} {item[0]} | {item[1]} | {item[2]}")

    print("\nmarker_gap=serve标记内未被更细事件覆盖的主机时间，不能直接称为Python计算。")
    print("CPU self包含框架调用和等待；设备时间按CPU来源关联，不按时间相邻猜测。")
    print("HCCL_range与collective_AICPU可能重叠；不能相加作通信开销。")
    print("无法关联的端点单独保留，不能宣称已完成全部设备算子的模块归因。")
    print(f"完整CPU/设备表和flow审计CSV：{OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"诊断失败：{exc}") from exc
