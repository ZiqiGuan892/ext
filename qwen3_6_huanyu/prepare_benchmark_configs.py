#!/usr/bin/env python3
"""Create an eager/PIECEWISE pair or graph x prefix-cache matrix from Huanyu YAML.

For the user's vllm 0.28.0+empty / ascend 0.19.1rc2 / omni 0.28.0+npu
single-stage schema. Preserve devices, TP, memory, scheduler, media and sampling
settings. The default pair disables prefix caching; --four-way creates all four
combinations. Never edit the installed YAML.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


VARIANTS = {"baseline": (False, False), "graph_only": (True, False),
            "prefix_only": (False, True), "graph_prefix": (True, True)}


def single_stage(stages, location):
    if (not isinstance(stages, list) or len(stages) != 1
            or not isinstance(stages[0], dict) or stages[0].get("stage_id") != 0):
        raise ValueError(f"Expected exactly one stage_id=0 at {location}; refusing to guess the schema")
    if "engine_args" in stages[0]:
        raise ValueError("This helper expects flat stages, not legacy nested engine_args")
    return stages[0]


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_variants(original, variants):
    if not isinstance(original, dict) or original.get("pipeline") != "minicpmo_5_0_thinker_only":
        raise ValueError("Expected pipeline: minicpmo_5_0_thinker_only")
    single_stage(original.get("stages"), "stages")
    npu = (original.get("platforms") or {}).get("npu")
    if not isinstance(npu, dict):
        raise ValueError("Expected platforms.npu overlay from the installed deploy config")
    single_stage(npu.get("stages"), "platforms.npu.stages")
    pair = {}
    for name, (graph, prefix) in variants.items():
        config = copy.deepcopy(original)
        config["enable_prefix_caching"] = prefix
        platform = config["platforms"]["npu"]
        if "enable_prefix_caching" in platform:
            platform["enable_prefix_caching"] = prefix
        for stage in (config["stages"][0], platform["stages"][0]):
            # The platform overlay must not re-enable graph/cache accidentally.
            stage["enforce_eager"] = not graph
            compilation = stage.setdefault("compilation_config", {})
            if not isinstance(compilation, dict):
                raise ValueError("Expected a mapping for compilation_config")
            compilation["cudagraph_mode"] = "PIECEWISE" if graph else "NONE"
            stage["enable_prefix_caching"] = prefix
            extras = stage.get("engine_extras") or {}
            if any(key in extras for key in ("enforce_eager", "compilation_config", "enable_prefix_caching")):
                raise ValueError("Conflicting graph/cache settings in engine_extras; inspect them before benchmarking")
        pair[name] = config
    return pair


def build_pair(original):
    pair = build_variants(original, {"eager": (False, False), "graph": (True, False)})
    # The only A/B differences must be enforce_eager and cudagraph_mode.
    eager_normalized = copy.deepcopy(pair["eager"])
    for stage in (eager_normalized["stages"][0], eager_normalized["platforms"]["npu"]["stages"][0]):
        stage["enforce_eager"] = False
        stage["compilation_config"]["cudagraph_mode"] = "PIECEWISE"
    assert eager_normalized == pair["graph"]
    return pair


def with_topology(original, tp, devices):
    if tp is None and devices is None:
        return original
    if tp is None or devices is None or tp <= 0:
        raise ValueError("Pass both --tp (positive integer) and --devices")
    entries = [value.strip() for value in devices.split(",")]
    if (len(entries) != tp or any(not value.isdecimal() for value in entries)
            or len({int(value) for value in entries}) != tp):
        raise ValueError("--devices must list exactly --tp distinct nonnegative device indices")
    result = copy.deepcopy(original)
    for stages in (result.get("stages"), (result.get("platforms") or {}).get("npu", {}).get("stages")):
        stage = single_stage(stages, "topology override")
        if {"tensor_parallel_size", "devices"} & set(stage.get("engine_extras") or {}):
            raise ValueError("Conflicting topology in engine_extras")
        stage.update(tensor_parallel_size=tp, devices=",".join(str(int(v)) for v in entries))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--four-way", action="store_true", help="Generate baseline, graph_only, prefix_only, graph_prefix")
    parser.add_argument("--tp", type=int, help="Override TP in both base and NPU stages; requires --devices")
    parser.add_argument("--devices", help="Comma-separated device indices; requires --tp")
    args = parser.parse_args(argv)
    try:
        source = args.source.read_bytes()
        original = yaml.safe_load(source)
        # Validate the installed schema before applying optional topology changes.
        build_pair(original)
        original = with_topology(original, args.tp, args.devices)
        pair = build_variants(original, VARIANTS) if args.four_way else build_pair(original)
        targets = [args.output_dir / f"huanyu_{name}.yaml" for name in pair]
        if any(p.exists() for p in targets):
            raise ValueError("Output config already exists; choose a new output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for (name, config), target in zip(pair.items(), targets):
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
            effective = merge(config["stages"][0], config["platforms"]["npu"]["stages"][0])
            keys = ("devices", "tensor_parallel_size", "max_num_seqs", "max_model_len",
                    "max_num_batched_tokens", "gpu_memory_utilization", "enforce_eager", "compilation_config",
                    "enable_prefix_caching")
            print(f"{name}: {target.resolve()}")
            print(json.dumps({k: effective.get(k) for k in keys}, ensure_ascii=False))
        print(f"source_sha256={hashlib.sha256(source).hexdigest()}")
        print("Generated four-way graph/cache matrix" if args.four_way else "Prefix cache disabled in both configs")
        print("Config generation does not prove graph replay or cache hits: check engine logs and runtime counters.")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
