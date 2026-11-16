#!/usr/bin/env python3
"""Compare TP0/TP2 forward CPU self time in an existing eager capture.

Run directly with Python 3; no arguments, environment variables, or third-party
packages are required. PROFILE_DIR and STEPS below select the existing capture.
This script only reads profiler files and prints its report. CPU self time
excludes nested serve markers, but still includes framework dispatch and waits;
it is not NPU kernel execution time.
"""

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

PROFILE_DIR = Path("/tmp/huanyu_text_eager_tp4.Ogud1q/capture")
STEPS = (6, 7, 8, 9)
def ns(x):
    return int(Decimal(str(x)) * 1000)
def read(p):
    return json.loads(p.read_text(encoding="utf-8"), parse_float=Decimal)
def inside(a, b):
    return a["lane"] == b["lane"] and a["a"] <= b["a"] <= b["b"] <= a["b"]
def union(spans):
    total, last = 0, None
    for a, b in sorted(spans):
        total += b-a if last is None or a >= last else max(0, b-last)
        last = b if last is None else max(last, b)
    return total

def main():
    print(f"PROFILE_DIR = {PROFILE_DIR}", flush=True)
    if not PROFILE_DIR.is_dir():
        raise SystemExit(f"采集目录不存在：{PROFILE_DIR}")

    files = defaultdict(list)
    for p in PROFILE_DIR.glob("serve_pid*_tp*.json"):
        if p.name.endswith(".inventory.json"): continue
        m = read(p)
        rank = int(m["identity"]["tp_rank"])
        if rank in (0,2): files[rank].append(m)
    if set(files) != {0,2} or any(len(v) != 1 for v in files.values()):
        raise SystemExit("TP0/TP2 manifest 缺失或不唯一")

    results = {}
    for rank in (0,2):
        path = Path(files[rank][0]["trace_dir"]) / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        data = read(path)
        events = data if isinstance(data, list) else data["traceEvents"]
        rows = [
            dict(name=e["name"], lane=(str(e.get("pid")),str(e.get("tid"))),
                 a=ns(e["ts"]), b=ns(e["ts"])+ns(e["dur"]))
            for e in events
            if e.get("ph") == "X" and str(e.get("name","")).startswith("serve/")
        ]
        stats, total = defaultdict(lambda: [0,0]), 0
        for step in STEPS:
            ex = [e for e in rows if e["name"] == f"serve/execute/{step}"]
            if len(ex) != 1: raise SystemExit(f"TP{rank} execute/{step} 不唯一")
            fw = [e for e in rows if e["name"] == "serve/runner/_model_forward"
                  and inside(ex[0], e)]
            if len(fw) != 1: raise SystemExit(f"TP{rank} step={step} forward不唯一")
            fw = fw[0]
            selected = [dict(e, children=[]) for e in rows if inside(fw,e)]
            selected.sort(key=lambda e: (
                e["a"], -e["b"], e["name"] != "serve/runner/_model_forward", e["name"]))
            stack = []
            subtotal = 0
            for e in selected:
                while stack and not inside(stack[-1],e):
                    old = stack.pop()
                    if e["a"] < old["b"]:
                        raise SystemExit("serve范围交叉重叠，停止self时间归因")
                if stack: stack[-1]["children"].append((e["a"],e["b"]))
                stack.append(e)
            for e in selected:
                own = e["b"]-e["a"]-union(e["children"])
                if own < 0: raise SystemExit("出现负self时间")
                stats[e["name"]][0] += own
                stats[e["name"]][1] += 1
                subtotal += own
            if subtotal != fw["b"]-fw["a"]:
                raise SystemExit("self时间未覆盖完整forward，停止归因")
            total += subtotal
        results[rank] = stats
        print(f"TP{rank} steps={STEPS} forward_CPU_mean_ms={total/len(STEPS)/1e6:.6f}")

    def category(name):
        return name.split("/")[2] if name.startswith("serve/module/") else name
    groups = {r: defaultdict(int) for r in (0,2)}
    for r in (0,2):
        for name,(duration,count) in results[r].items():
            groups[r][category(name)] += duration

    scale = len(STEPS)*1e6
    print("\ncategory: TP0_self_ms/iter TP2_self_ms/iter delta_ms/iter")
    for name in sorted(set(groups[0])|set(groups[2]),
                       key=lambda n: max(groups[0].get(n,0),groups[2].get(n,0)), reverse=True):
        a,b = groups[0].get(name,0)/scale, groups[2].get(name,0)/scale
        print(f"{name}: {a:.6f} {b:.6f} {a-b:+.6f}")

    names = set(results[0])|set(results[2])
    def value(r,n):
        return results[r].get(n,[0,0])[0]/scale
    print("\n逐模块self差异最大的10项：TP0_ms TP2_ms delta_ms calls0/calls2")
    for name in sorted(names,key=lambda n: abs(value(0,n)-value(2,n)),reverse=True)[:10]:
        a,b = value(0,name),value(2,name)
        counts = [results[r].get(name,[0,0])[1] for r in (0,2)]
        print(f"{a:.6f} {b:.6f} {a-b:+.6f} {counts[0]}/{counts[1]} {name}")
    print("\nself仅扣除嵌套serve范围，包含框架下发及等待，不是NPU纯计算时间。")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        raise SystemExit(f"读取采集文件失败：{exc}") from exc
