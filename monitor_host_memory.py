"""Sample Linux host and job memory to JSONL, independently of Ray and torch."""

import argparse
import json
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path


PROC = Path("/proc")
MEMINFO = ("MemTotal MemAvailable MemFree AnonPages Shmem Cached SReclaimable SUnreclaim "
           "Unevictable Mlocked SwapTotal SwapFree").split()
STATUS = "VmRSS RssAnon RssFile RssShmem VmLck VmPin VmSwap".split()
VM_COUNTERS = ("compact_", "allocstall", "pgscan", "pgsteal", "pswp", "oom_kill", "nr_foll_pin_")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def fields(text):
    """Read /proc or cgroup integer fields, preserving zero and omitting missing fields."""
    result = {}
    for line in (text or "").splitlines():
        words = line.split()
        if len(words) >= 2 and words[1].isdigit():
            result[words[0].rstrip(":")] = int(words[1])
    return result


def memory_group(pid_dir):
    unified = None
    for line in (read(pid_dir / "cgroup") or "").splitlines():
        _, controllers, path = line.split(":", 2)
        if "memory" in controllers.split(","):
            return "v1", path
        if not controllers:
            unified = ("v2", path)
    return unified


def unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def cgroup_dirs(group, mountinfo):
    version, group_name = group
    group_path = Path(group_name)
    # Outside this cgroup namespace even if the PID is visible in /proc.
    if ".." in group_path.parts:
        return []
    marker = "memory.current" if version == "v2" else "memory.usage_in_bytes"
    for line in (mountinfo or "").splitlines():
        before, after = line.split(" - ", 1)
        fs, _, options = after.split()[:3]
        if version == "v1" and (fs != "cgroup" or "memory" not in options.split(",")):
            continue
        if version == "v2" and fs != "cgroup2":
            continue
        root, mount = map(Path, map(unescape_mount, before.split()[3:5]))
        # Mount root may be a host path while /proc/<pid>/cgroup is namespace-relative.
        relative = group_path.relative_to(root) if group_path.is_relative_to(root) else group_path.relative_to("/")
        leaf = mount / relative
        if (leaf / marker).is_file() or (leaf == mount and (leaf / "cgroup.controllers").is_file()):
            return [leaf, *(parent for parent in leaf.parents if parent.is_relative_to(mount))]
    return []


def group_sample(path, version):
    names = ("memory.current memory.peak memory.max memory.high memory.swap.current memory.swap.max "
             "memory.events memory.events.local memory.pressure" if version == "v2" else
             "memory.usage_in_bytes memory.max_usage_in_bytes memory.limit_in_bytes memory.failcnt "
             "memory.memsw.usage_in_bytes memory.memsw.limit_in_bytes memory.oom_control")
    data = {}
    for name in names.split():
        raw = read(path / name)
        if raw is not None:
            if name in ("memory.events", "memory.events.local", "memory.oom_control"):
                data[name] = fields(raw)
            else:
                data[name] = int(raw) if raw.isdigit() else raw
    wanted = {"anon", "file", "shmem", "kernel", "slab_unreclaimable", "pagetables", "unevictable",
              "rss", "cache", "swap", "total_rss", "total_cache", "total_shmem", "total_swap"}
    data["memory.stat"] = {k: v for k, v in fields(read(path / "memory.stat")).items() if k in wanted}
    return data


def buddy_sample(raw):
    zones = {}
    for line in (raw or "").splitlines():
        match = re.fullmatch(r"\s*Node\s+(\d+),\s+zone\s+(\S+)\s+([\d\s]+)", line)
        if match:
            zones[f"{match[1]}/{match[2]}"] = [int(n) for n in match[3].split()]
    return zones


def zone_watermarks(raw):
    zones = {}
    zone = None
    for line in (raw or "").splitlines():
        if line.startswith("Node "):
            header = re.fullmatch(r"Node\s+(\d+),\s+zone\s+(\S+)", line.strip())
            zone = zones.setdefault(f"{header[1]}/{header[2]}", {}) if header else None
        elif zone is not None:
            field = re.fullmatch(r"\s*(pages free|min|low|high)\s+(\d+)\s*", line)
            if field:
                zone[field[1]] = int(field[2])
    return zones


def process_sample(group, top, pss):
    version, scope = group
    rows = []
    by_role = {}
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        candidate = memory_group(entry)
        if not candidate or candidate[0] != version:
            continue
        path = candidate[1]
        if ".." in path.split("/") or (path != scope and not path.startswith(scope.rstrip("/") + "/")):
            continue
        values = fields(read(entry / "status"))
        if "VmRSS" not in values:
            continue
        name = read(entry / "comm") or "unknown"
        try:
            with (entry / "cmdline").open("rb") as stream:
                title = stream.read(256).split(b"\0", 1)[0].decode(errors="replace").strip()
            if title.startswith(("ray::", "vllm::", "VLLM::")):
                name = title.split()[0][:128]
        except OSError:
            pass
        role = name.split(".", 1)[0]
        totals = by_role.setdefault(role, {"count": 0})
        totals["count"] += 1
        if "RssAnon" in values:
            totals["RssAnon_kB_sum"] = totals.get("RssAnon_kB_sum", 0) + values["RssAnon"]
        rows.append({"pid": int(entry.name), "name": name, **{k: values[k] for k in STATUS if k in values}})
    rows.sort(key=lambda row: row["VmRSS"], reverse=True)
    top_rows = rows[:top]
    if pss:
        for row in top_rows:
            rollup = fields(read(PROC / str(row["pid"]) / "smaps_rollup"))
            row.update({k: v for k, v in rollup.items() if k in ("Pss", "Pss_Anon", "Pss_Shmem", "Locked")})
    return by_role, top_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new JSONL file on a persistent mount")
    parser.add_argument("--pid", type=int, default=os.getpid(), help="PID identifying the job cgroup at startup")
    parser.add_argument("--interval", type=int, default=15, help="seconds between samples")
    parser.add_argument("--top", type=int, default=12, help="number of processes ranked by resident memory")
    parser.add_argument("--pss", action="store_true", help="also read smaps_rollup for the top processes (slower)")
    parser.add_argument("--once", action="store_true", help="write a single sample and exit")
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or args.interval <= 0 or args.top < 1:
        parser.error("requires Linux, --interval > 0 and --top >= 1")
    group = memory_group(PROC / str(args.pid))
    if group is None or ".." in group[1].split("/"):
        parser.error(f"cannot identify a visible memory cgroup for PID {args.pid}")
    paths = cgroup_dirs(group, read(PROC / "self/mountinfo"))
    if not paths:
        print("Memory controller files inaccessible. Recording host and process data only.", file=sys.stderr)
    metadata = {"host": socket.gethostname(), "boot_id": read(PROC / "sys/kernel/random/boot_id"),
                "page_bytes": os.sysconf("SC_PAGE_SIZE"), "scope": group}
    with args.out.open("x", encoding="utf-8") as output:
        while True:
            record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **metadata}
            record["meminfo_kB"] = {k: v for k, v in fields(read(PROC / "meminfo")).items() if k in MEMINFO}
            record["buddy_free_blocks_by_order"] = buddy_sample(read(PROC / "buddyinfo"))
            record["zone_watermarks_pages"] = zone_watermarks(read(PROC / "zoneinfo"))
            record["vmstat"] = {k: v for k, v in fields(read(PROC / "vmstat")).items() if k.startswith(VM_COUNTERS)}
            record["host_memory_psi"] = read(PROC / "pressure/memory")
            record["cgroups"] = {str(path): group_sample(path, group[0]) for path in paths}
            record["roles"], record["top_kB"] = process_sample(group, args.top, args.pss)
            print(json.dumps(record, separators=(",", ":")), file=output, flush=True)
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
