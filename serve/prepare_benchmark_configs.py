#!/usr/bin/env python3
"""Create an eager/PIECEWISE A/B pair from the installed Huanyu deploy YAML.

For the user's vllm 0.28.0+empty / ascend 0.19.1rc2 / omni 0.28.0+npu
single-stage schema. Preserve devices, TP, memory, scheduler, media and sampling
settings. Both files disable prefix caching. Never edit the installed YAML.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


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


def build_pair(original):
    if not isinstance(original, dict) or original.get("pipeline") != "minicpmo_5_0_thinker_only":
        raise ValueError("Expected pipeline: minicpmo_5_0_thinker_only")
    single_stage(original.get("stages"), "stages")
    npu = (original.get("platforms") or {}).get("npu")
    if not isinstance(npu, dict):
        raise ValueError("Expected platforms.npu overlay from the installed deploy config")
    single_stage(npu.get("stages"), "platforms.npu.stages")
    pair = {}
    for name in ("eager", "graph"):
        config = copy.deepcopy(original)
        config["enable_prefix_caching"] = False
        platform = config["platforms"]["npu"]
        if "enable_prefix_caching" in platform:
            platform["enable_prefix_caching"] = False
        for stage in (config["stages"][0], platform["stages"][0]):
            # The platform overlay must not re-enable graph/cache accidentally.
            stage["enforce_eager"] = name == "eager"
            compilation = stage.setdefault("compilation_config", {})
            if not isinstance(compilation, dict):
                raise ValueError("Expected a mapping for compilation_config")
            compilation["cudagraph_mode"] = "NONE" if name == "eager" else "PIECEWISE"
            if "enable_prefix_caching" in stage:
                stage["enable_prefix_caching"] = False
            extras = stage.get("engine_extras") or {}
            if any(key in extras for key in ("enforce_eager", "compilation_config", "enable_prefix_caching")):
                raise ValueError("Conflicting graph/cache settings in engine_extras; inspect them before benchmarking")
        pair[name] = config
    # The only A/B differences must be enforce_eager and cudagraph_mode.
    eager_normalized = copy.deepcopy(pair["eager"])
    for stage in (eager_normalized["stages"][0], eager_normalized["platforms"]["npu"]["stages"][0]):
        stage["enforce_eager"] = False
        stage["compilation_config"]["cudagraph_mode"] = "PIECEWISE"
    assert eager_normalized == pair["graph"]
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = args.source.read_bytes()
        original = yaml.safe_load(source)
        pair = build_pair(original)
        targets = [args.output_dir / f"huanyu_{name}.yaml" for name in pair]
        if any(p.exists() for p in targets):
            raise ValueError("Output config already exists; choose a new output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for (name, config), target in zip(pair.items(), targets):
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
            effective = merge(config["stages"][0], config["platforms"]["npu"]["stages"][0])
            keys = ("devices", "tensor_parallel_size", "max_num_seqs", "max_model_len",
                    "max_num_batched_tokens", "gpu_memory_utilization", "enforce_eager", "compilation_config")
            print(f"{name}: {target.resolve()}")
            print(json.dumps({k: effective.get(k) for k in keys}, ensure_ascii=False))
        print(f"source_sha256={hashlib.sha256(source).hexdigest()}; prefix cache disabled in both configs")
        print("Config generation does not prove graph replay: check the effective engine config and capture logs.")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
