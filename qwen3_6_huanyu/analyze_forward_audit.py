"""Inspect existing Huanyu captures without executing a model or changing capture files.

Copy this file alone to the training container. torch_npu is imported lazily only
to decode its framework file format. Raw counters are used for host interval
ratios, never treated as calibrated device timestamps.
"""

import argparse
import ast
import csv
import json
import re
import sys
import warnings
from collections import Counter
from configparser import ConfigParser, Error as ConfigError
from decimal import Decimal, InvalidOperation
from pathlib import Path


MANIFEST_NAME = re.compile(r"pid(?P<pid>\d+)_tp(?P<tp>\d+)_t\d+\.json")
OUTER_MODULE = re.compile(
    r"^huanyu/(GDN|Attention|MoE|MLP)/.+\.layers\.\d+\."
    r"(linear_attn|self_attn|self_attention|mlp):[^/]+$"
)
CATEGORIES = ("GDN", "Attention", "MoE", "MLP")


def select_manifest(root, pid=None, tp_rank=None):
    if not root.is_dir():
        raise ValueError(f"采集目录不存在：{root}")
    matches = []
    for path in sorted(root.rglob("pid*_tp*_t*.json")):
        match = MANIFEST_NAME.fullmatch(path.name)
        if match and (pid is None or int(match["pid"]) == pid):
            if tp_rank is None or int(match["tp"]) == tp_rank:
                matches.append(path)
    if len(matches) != 1:
        names = "\n".join(str(p) for p in matches[:8])
        raise ValueError(f"需要唯一 manifest，找到 {len(matches)} 个；请用 --pid/--tp-rank 筛选。\n{names}")
    return matches[0]


def load_framework_events(trace):
    """Decode directly, avoiding FwkFileParser's writes to the capture logs/ directory."""
    try:
        from torch_npu.profiler.analysis.prof_bean._torch_op_bean import TorchOpEnum
        from torch_npu.profiler.analysis.prof_common_func._tlv_decoder import TLVDecoder
        from torch_npu.profiler.analysis.prof_config._fwk_file_parser_config import FwkFileParserConfig
    except ImportError as error:
        raise RuntimeError("需要在安装了对应 torch_npu 的采集环境中运行；此脚本不会安装依赖。") from error

    specs = [(tag, spec) for tag, spec in FwkFileParserConfig.FILE_BEAN_MAP.items()
             if getattr(spec.get("bean"), "__name__", "") == "TorchOpBean"]
    if len(specs) != 1 or not specs[0][1].get("is_tlv"):
        raise RuntimeError("当前 torch_npu 框架格式不受支持，停止自动解析。")
    tag, spec = specs[0]
    framework = trace / "FRAMEWORK"
    pattern = FwkFileParserConfig.FILE_DISPATCH_MAP[tag]
    files = sorted(p for p in framework.iterdir() if p.is_file() and re.match(pattern, p.name))
    if len(files) != 1:
        raise ValueError(f"需要唯一的 TorchOp 框架文件，找到 {len(files)} 个：{files}")
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        ops = TLVDecoder.decode(files[0].read_bytes(), spec["bean"], spec["struct_size"])
    events = []
    for op in ops:
        if op.name.startswith(("huanyu/", "ProfilerStep#")):
            data = op._constant_data
            events.append(dict(
                name=op.name, pid=op.pid, tid=op.tid,
                start=data[TorchOpEnum.START_NS.value], end=data[TorchOpEnum.END_NS.value],
            ))
    return events, len(ops), [str(w.message) for w in notices], files[0]


def inside(child, parent):
    return (child["pid"] == parent["pid"] and child["tid"] == parent["tid"]
            and parent["start"] <= child["start"] < child["end"] <= parent["end"])


def union_length(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def audit_iterations(manifest, events, decode_warnings=()):
    # Child implementations such as chunk_gated_delta_rule are optional branches.
    expected = {t["label"] for t in manifest.get("targets", [])
                if OUTER_MODULE.fullmatch(t.get("label", ""))}
    active = [r for r in manifest.get("iterations", []) if r.get("warmup") is False and not r.get("error")]
    identity_pid = manifest.get("identity", {}).get("pid")
    if identity_pid is not None:
        events = [e for e in events if e["pid"] == identity_pid]
    iteration_counts = Counter(r["engine_iteration"] for r in active)
    results = []
    for row in active:
        iteration = row["engine_iteration"]
        execute = [e for e in events if e["name"] == f"huanyu/execute/iteration_{iteration}"]
        forward = [e for e in events if e["name"] == "huanyu/forward"
                   and len(execute) == 1 and inside(e, execute[0])]
        steps = [e for e in events if re.fullmatch(r"ProfilerStep#\d+", e["name"])
                 and len(execute) == 1 and inside(execute[0], e)]
        record = dict(iteration=iteration, host_s=row.get("host_seconds"), execute_ranges=len(execute),
                      forward_ranges=len(forward), step_ranges=len(steps), host_fractions=None)
        results.append(record)
        if len(execute) != 1 or len(forward) != 1 or len(steps) != 1 or iteration_counts[iteration] != 1:
            record["reason"] = "缺少唯一、闭合的 execute/forward/ProfilerStep 边界"
            continue
        fwd = forward[0]
        record.update(step_id=steps[0]["name"].split("#")[1],
                      execute_raw=(execute[0]["start"], execute[0]["end"]),
                      forward_raw=(fwd["start"], fwd["end"]))
        modules = [e for e in events if e["name"] in expected and inside(e, fwd)]
        calls = Counter(e["name"] for e in modules)
        record["missing_or_repeated"] = sorted(name for name in expected if calls[name] != 1)
        if not expected or record["missing_or_repeated"] or decode_warnings:
            record["reason"] = "外层模块清单为空、调用不齐/重复或框架解码有告警；不计算比例"
            continue
        denominator = fwd["end"] - fwd["start"]
        fractions, counts, category_sum = {}, {}, 0
        for category in CATEGORIES:
            selected = [e for e in modules if OUTER_MODULE.fullmatch(e["name"])[1] == category]
            length = union_length([(e["start"], e["end"]) for e in selected])
            counts[category] = len(selected)
            fractions[category] = 100 * length / denominator
            category_sum += length
        covered = union_length([(e["start"], e["end"]) for e in modules])
        fractions["other"] = 100 * (denominator - covered) / denominator
        fractions["cross_category_overlap"] = 100 * (category_sum - covered) / denominator
        record.update(host_fractions=fractions, module_calls=counts)
    return Counter(OUTER_MODULE.fullmatch(name)[1] for name in expected), results


def print_clock_anchors(trace):
    print("\n===== 同次采集的 CANN 时钟信息 =====")
    profiles = sorted(p for p in trace.glob("PROF_*") if p.is_dir())
    if not profiles:
        print("NO_CANN_PROF_DIRECTORY")
    for prof in profiles:
        paths = [prof / "host/info.json", prof / "host/host_start.log", prof / "host/start_info",
                 *sorted(prof.glob("device_*/start_info.*"))]
        for path in paths:
            print("CLOCK_FILE", path, "exists=", path.is_file())
            if not path.is_file():
                continue
            try:
                if path.name == "host_start.log":
                    config = ConfigParser()
                    config.read(path, encoding="utf-8")
                    print("HOST_CLOCK", {k: config.get("Host", k, fallback=None)
                                         for k in ("clock_monotonic_raw", "cntvct")})
                else:
                    text = path.read_text(encoding="utf-8")
                    try:
                        data = json.loads(text)
                    except ValueError:
                        data = ast.literal_eval(text)
                    if path.name == "info.json":
                        print("CPU_FREQUENCIES", [x.get("Frequency") for x in (data.get("CPU") or [])])
                    else:
                        print("START_INFO", {k: data.get(k) for k in ("collectionTimeBegin", "clockMonotonicRaw")})
            except (OSError, ValueError, SyntaxError, AttributeError, TypeError, ConfigError) as error:
                print("CLOCK_READ_ERROR", repr(error))


def csv_coverage(path):
    count, invalid, lo, hi = 0, 0, None, None
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Task Start Time(us)", "Task Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"缺少时间字段：{reader.fieldnames}")
        for row in reader:
            try:
                start = Decimal(row["Task Start Time(us)"].strip())
                duration = Decimal(row["Task Duration(us)"].strip())
                if not start.is_finite() or not duration.is_finite() or duration <= 0:
                    raise ValueError("nonpositive or nonfinite time")
            except (ValueError, InvalidOperation, AttributeError):
                invalid += 1
                continue
            stop = start + duration
            lo, hi = start if lo is None else min(lo, start), stop if hi is None else max(hi, stop)
            count += 1
    return dict(rows=count, invalid_rows=invalid, start_us=lo, end_us=hi,
                span_s=None if lo is None else (hi - lo) / Decimal(1_000_000))


def print_device_evidence(exported, trace):
    print("\n===== 已导出设备事件的覆盖范围与关联事件 =====")
    # Never combine files from another capture under a shared export directory.
    names = {p.name for p in trace.glob("PROF_*") if p.is_dir()}

    def same_capture(path):
        return any(parent.name in names for parent in path.parents)

    csv_files = sorted(p for p in exported.rglob("op_summary*.csv") if same_capture(p))
    print("MATCHED_OP_SUMMARY_FILES", len(csv_files))
    for path in csv_files:
        try:
            print("CSV", path, csv_coverage(path))
        except (OSError, ValueError, csv.Error) as error:
            print("CSV_READ_ERROR", path, repr(error))
    timeline_files = sorted(p for p in exported.rglob("msprof*.json") if same_capture(p))
    print("TIMELINE_FILES", len(timeline_files))
    for path in timeline_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data = data.get("traceEvents", []) if isinstance(data, dict) else data
            if not isinstance(data, list) or any(not isinstance(e, dict) for e in data):
                raise ValueError("timeline 必须是事件列表")
            flows = [e for e in data if e.get("cat") == "HostToDevice"]
            print("TIMELINE", path, "events=", len(data),
                  "HostToDevice=", dict(Counter(e.get("ph") for e in flows)))
            for event in flows[:2]:
                print("FLOW_SAMPLE", {k: event.get(k) for k in ("ph", "id", "pid", "tid", "ts")})
        except (OSError, ValueError) as error:
            print("TIMELINE_READ_ERROR", path, repr(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读检查 Huanyu forward 框架范围、模块主机占比及 CANN 关联证据")
    parser.add_argument("profile_dir", type=Path, help="原始采集根目录")
    parser.add_argument("export_dir", type=Path, help="含同名 PROF_* 目录的 msprof 导出副本目录")
    parser.add_argument("--pid", type=int, help="按采集 worker PID 筛选 manifest")
    parser.add_argument("--tp-rank", type=int, help="按 TP rank 筛选 manifest")
    args = parser.parse_args(argv)
    try:
        if not args.export_dir.is_dir():
            raise ValueError(f"导出目录不存在：{args.export_dir}")
        manifest_path = select_manifest(args.profile_dir, args.pid, args.tp_rank)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("trace_dir"), str) or not manifest["trace_dir"]:
            raise ValueError("manifest 缺少 trace_dir")
        trace = Path(manifest["trace_dir"])
        if not trace.is_absolute():
            trace = manifest_path.parent / trace
        if not (trace / "FRAMEWORK").is_dir():
            raise ValueError(f"FRAMEWORK 目录不存在：{trace}")
        print("MANIFEST", manifest_path)
        print("STATE", manifest.get("status"), manifest.get("phase"),
              "FORWARD_RETURNED", manifest.get("forwards_completed"))
        print("TRACE", trace)
        print("PROFILER_INFO", [p.name for p in trace.glob("profiler_info*.json")])
        events, count, notices, source = load_framework_events(trace)
        print("FRAMEWORK_FILE", source, "RECORDS", count, "DECODE_WARNINGS", notices)
        expected, records = audit_iterations(manifest, events, notices)
        print("EXPECTED_OUTER_MODULES", dict(expected))
        print("EXPECTED_ACTIVE", manifest.get("active"), "ACTIVE_ITERATIONS", [r["iteration"] for r in records])
        print("\n===== HOST FORWARD：模块主机区间比例，包含等待和通信 =====")
        for record in records:
            print("ITERATION", {k: v for k, v in record.items() if k not in ("host_fractions", "module_calls")})
            if record["host_fractions"] is not None:
                print("MODULE_CALLS", record["module_calls"])
                print("HOST_SHARE_PCT", {k: round(v, 4) for k, v in record["host_fractions"].items()})
        print_clock_anchors(trace)
        print_device_evidence(args.export_dir, trace)
        print("\n检查输出完成；不代表设备事件完整或采集成功。")
        print("HOST_SHARE_PCT 不是纯计算占比；本脚本不伪造时钟元数据，也不按时间/行数平分设备任务。")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"[forward-audit] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
