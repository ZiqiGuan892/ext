#!/usr/bin/env python3
"""Pure inference with STEM's TP8 x 2 rollout topology; no training/Ray/profiler.

Uses the sibling benchmark transport and deploy-schema validator. Two replicas
run concurrently, with a separate bounded client pool for each. ROLLOUT_N is
expanded client-side into n=1 requests; aggregate timing uses ONE common clock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import yaml

import bench_throughput as bench
import prepare_benchmark_configs as deploy


MODEL = "/opt/huawei/dataset/model_gy/downloaded_models/2748"
SOURCE = "/vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml"
HOOK = "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks"
# Mirrored from run_huanyu_stem.sh: huanyu_configure_precision_mode(full).
FULL_ON = """MINICPMO_HUANYU_GDN_FP32_TRAINING MINICPMO_GDN_FP32_REFERENCE
MINICPMO_HUANYU_LAYERWISE_RELOAD MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE
MINICPMO_ISOLATE_MEDIA_ENCODERS MINICPMO_GDN_CACHE_BARRIER
MINICPMO_GDN_PREFILL_REFERENCE MINICPMO_GDN_L2NORM_REFERENCE
MINICPMO_GDN_CONV_REFERENCE MINICPMO_GDN_GATING_REFERENCE
MINICPMO_PRIVATE_HYBRID_CACHE MINICPMO_STABLE_LINEAR""".split()
FULL_OFF = """MINICPMO_HF_NORM_REFERENCE MINICPMO_HF_ROUTER_REFERENCE
MINICPMO_HF_TEXT_ROPE_REFERENCE MINICPMO_MOE_MATCH_ROLLOUT
MINICPMO_ISOLATE_FULL_ATTENTION MINICPMO_VISION_POSITION_REFERENCE
MINICPMO_NATIVE_VISION_REFERENCE MINICPMO_FSDP2_PIN_MEMORY
MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE""".split()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def make_deploy(source, c):
    result = deploy.build_pair(source)["eager"]
    # devices are LOCAL indices within each server's ASCEND_RT_VISIBLE_DEVICES.
    result = deploy.with_topology(result, c["tp"], ",".join(map(str, range(c["tp"]))))
    values = dict(max_num_seqs=c["max_num_seqs"], gpu_memory_utilization=c["gpu_memory_utilization"],
                  max_model_len=c["max_model_len"], max_num_batched_tokens=c["max_num_batched_tokens"],
                  enable_chunked_prefill=False, mm_processor_cache_gb=0,
                  data_parallel_size=1, pipeline_parallel_size=1, load_format="safetensors")
    for stage in (result["stages"][0], result["platforms"]["npu"]["stages"][0]):
        bench.require(not (set(values) & set(stage.get("engine_extras") or {})),
                      "Conflicting capacity/topology in engine_extras")
        stage.update(values)
        stage.setdefault("additional_config", {})["weight_nz_mode"] = 0
        if c["precision"] == "stem_full":
            stage["worker_extension_cls"] = (
                "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension")
        else:
            stage.pop("worker_extension_cls", None)
        stage["default_sampling_params"] = dict(
            temperature=c["temperature"], top_p=c["top_p"], top_k=c["top_k"],
            repetition_penalty=c["repetition_penalty"], max_tokens=c["output_tokens"],
            ignore_eos=c["ignore_eos"], seed=c["seed"], detokenize=True)
    return result


def prepare(args):
    c = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
         if k not in {"action", "case_dir"}}
    for key in ("tp", "max_num_seqs", "client_concurrency_per_replica", "train_batch_size", "rollout_n",
                "input_tokens", "output_tokens", "max_model_len", "max_num_batched_tokens", "num_batches"):
        bench.positive_int(c[key])
    devices = c["devices"].split(",")
    bench.require(len(devices) == 16 and all(x.isdecimal() for x in devices)
                  and len(set(map(int, devices))) == 16, "Exactly 16 distinct device IDs are required")
    bench.require(16 % c["tp"] == 0, "TP must divide 16")
    c["num_devices"], c["replicas"] = 16, 16 // c["tp"]
    c["device_groups"] = [",".join(devices[i:i+c["tp"]]) for i in range(0, 16, c["tp"])]
    c["ports"] = list(range(c["port"], c["port"] + c["replicas"]))
    bench.require(1 <= min(c["ports"]) <= max(c["ports"]) <= 65535, "Invalid ports")
    c["trajectories_per_batch"] = c["train_batch_size"] * c["rollout_n"]
    bench.require(c["trajectories_per_batch"] % c["replicas"] == 0, "Trajectories must divide evenly across replicas")
    bench.require(c["input_tokens"] + c["output_tokens"] <= c["max_model_len"], "Input + output exceeds model context")
    bench.require(c["max_num_batched_tokens"] >= c["max_model_len"], "Chunked prefill is OFF: token budget must cover context")
    bench.require(0 < c["gpu_memory_utilization"] < 1, "Invalid memory fraction")
    bench.require(c["warmup_batches"] >= 1 and c["seed"] >= 0, "Need warmup >= 1 and seed >= 0")
    bench.require(math.isfinite(c["timeout_s"]) and c["timeout_s"] > 0, "Invalid timeout")
    bench.require(math.isfinite(c["temperature"]) and c["temperature"] >= 0
                  and 0 < c["top_p"] <= 1 and (c["top_k"] == -1 or c["top_k"] >= 1)
                  and math.isfinite(c["repetition_penalty"]) and c["repetition_penalty"] > 0, "Invalid sampling settings")
    source = Path(c["source"]).read_bytes()
    config = make_deploy(yaml.safe_load(source), c)
    c["source_sha256"] = hashlib.sha256(source).hexdigest()
    bench.require(not args.case_dir.exists() or not any(args.case_dir.iterdir()), "Use a new, empty case directory")
    args.case_dir.mkdir(parents=True, exist_ok=True)
    with (args.case_dir / "deploy.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    c["deploy_sha256"] = hashlib.sha256((args.case_dir / "deploy.yaml").read_bytes()).hexdigest()
    write(args.case_dir / "case.json", c)
    print(json.dumps(c, ensure_ascii=False, indent=2))


def serve_environment(c, replica, inherited):
    env = dict(inherited)
    for name in list(env):
        if (name.startswith(("HUANYU_SERVE_PROFILE_", "HUANYU_FORWARD_PROFILE_", "HUANYU_GREEDY_"))
                or name in {"HUANYU_GEN_TIMING", "VLLM_TORCH_PROFILER_DIR", "VLLM_OMNI_EXTERNAL_MODULES",
                            "HUANYU_TEXT_AB_DIR", "HUANYU_LOGITS_AB_DIR", "HUANYU_EMBEDDING_AB_DIR",
                            "HUANYU_AUDIO_PROBE_DIR", "HUANYU_TEXT_AB_DETERMINISM", "HUANYU_SERVE_REPEAT_DETERMINISM"}):
            env.pop(name, None)
    for name in FULL_ON + FULL_OFF:
        env[name] = "1" if c["precision"] == "stem_full" and name in FULL_ON else "0"
    env.update(ASCEND_RT_VISIBLE_DEVICES=c["device_groups"][replica],
               HUANYU_PRECISION_PROFILE="full" if c["precision"] == "stem_full" else "custom",
               HUANYU_GRAPH_MODE="0", MINICPMO_HUANYU_ROLLOUT_TP=str(c["tp"]),
               HUANYU_JOB_DETERMINISM="0", VLLM_BATCH_INVARIANT="0", HCCL_DETERMINISTIC="false",
               ASCEND_LAUNCH_BLOCKING="0", NCCL_DETERMINISTIC="0", LCCL_DETERMINISTIC="0",
               TORCH_COMPILE_DISABLE="1", NPU_GRAPH_CAPTURE_DISABLE="1", VLLM_TORCH_COMPILE="0",
               HUANYU_USE_PRECOMPUTE="0", MINICPMO_USE_PRECOMPUTE="0", HUANYU_EMBED_CACHE_DIR="",
               HUANYU_STEM_TEXT_TRAIN="1", HUANYU_AQA_AUDIO_TRAIN="0", PYTHONUNBUFFERED="1")
    env["HCCL_OP_EXPANSION_MODE"] = c["hccl_op_expansion_mode"]
    if c["precision"] == "stem_full":
        env["VLLM_OMNI_EXTERNAL_MODULES"] = HOOK
        # Only expose the training plugin; never replace installed vllm-omni.
        env["PYTHONPATH"] = c["project_root"] + os.pathsep + env.get("PYTHONPATH", "")
    return env


def serve(args):
    root = args.case_dir.resolve()
    c = read(root / "case.json")
    bench.require(0 <= args.replica < c["replicas"], "Invalid replica ID")
    bench.require(hashlib.sha256((root / "deploy.yaml").read_bytes()).hexdigest() == c["deploy_sha256"],
                  "Deploy config changed after prepare")
    log = root / f"serve_{args.replica}.log"
    bench.require(not log.exists(), "Serve log already exists; use a NEW case directory for a retry")
    spec = importlib.util.find_spec("vllm_omni")
    bench.require(spec is not None, "vllm_omni is not installed")
    native_root = Path(next(iter(spec.submodule_search_locations)))
    bench.require(native_root.resolve() == Path(c["source"]).resolve().parents[1],
                  f"Imported vllm-omni {native_root} differs from deploy source; check PYTHONPATH")
    if c["precision"] == "stem_full":
        bench.require((Path(c["project_root"]) / "verl_omni_ext/models/qwen3_6_huanyu/engine_hooks.py").is_file(),
                      "Set --project-root to the working training plugin checkout")
    env = serve_environment(c, args.replica, os.environ)
    write(root / f"launch_{args.replica}.json", dict(
        vllm_omni_source=str(native_root), replica=args.replica, physical_devices=c["device_groups"][args.replica],
        precision=c["precision"], environment={k: env[k] for k in sorted(env)
        if k.startswith(("MINICPMO_", "HUANYU_", "VLLM_", "HCCL_", "ASCEND_", "NCCL_", "LCCL_"))}))
    cmd = ["vllm", "serve", c["tokenizer"], "--omni", "--host", "127.0.0.1", "--port", str(c["ports"][args.replica]),
           "--log-stats", "--trust-remote-code", "--stage-init-timeout", "1800", "--served-model-name", c["model"],
           "--deploy-config", str(root / "deploy.yaml")]
    print(f"replica={args.replica} TP={c['tp']} physical={c['device_groups'][args.replica]} source={native_root}", flush=True)
    print(f"Service log: {log}", flush=True)
    # Foreground lifetime: Ctrl-C terminates this server; never kill unrelated jobs.
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                return child.wait()


def send(c, replica, prompt, request_id, limit=None):
    a = SimpleNamespace(**{**c, "host": "127.0.0.1", "port": c["ports"][replica], "n": 1,
                           "api_key": "EMPTY", "timing_source": "none"})
    payload = bench.make_payload(a, prompt, output_tokens=limit, seed=c["seed"] + request_id)
    payload.update(top_p=c["top_p"], top_k=c["top_k"], repetition_penalty=c["repetition_penalty"])
    return {**bench.send_payload(a, payload, request_id, c["input_tokens"]), "replica": replica}


def run_batch(c, pools, prompts, start_id, sender=send):
    # ROLLOUT_N trajectories per prompt, independent request seed, HTTP n=1.
    trajectories = [(p, sample) for p in prompts for sample in range(c["rollout_n"])]
    start = time.perf_counter()
    futures = [pools[i % c["replicas"]].submit(sender, c, i % c["replicas"], p, start_id + i)
               for i, (p, _) in enumerate(trajectories)]
    results = [f.result() for f in as_completed(futures)]
    wall = time.perf_counter() - start
    return dict(wall_time_s=wall, requests=sorted(results, key=lambda x: x["request_id"]))


def summary(batches, c):
    result = bench.summarize(batches, c["num_devices"])
    # Same global denominator for EVERY replica. These contributions add exactly.
    result["replicas"] = []
    for replica in range(c["replicas"]):
        selected = [dict(wall_time_s=b["wall_time_s"], requests=[r for r in b["requests"] if r["replica"] == replica])
                    for b in batches]
        item = bench.summarize(selected, c["tp"])
        result["replicas"].append(dict(replica=replica, **item))
    return result


def run(args):
    from transformers import AutoTokenizer
    from contextlib import ExitStack

    root = args.case_dir.resolve()
    c = read(root / "case.json")
    target = root / "throughput.json"
    bench.require(not target.exists(), "Report exists; use a new case directory")
    for replica in range(c["replicas"]):
        launch = read(root / f"launch_{replica}.json")
        bench.require(launch["precision"] == c["precision"], "Launch mode differs")
        if c["precision"] == "stem_full":
            log = (root / f"serve_{replica}.log").read_text(encoding="utf-8", errors="replace")
            bench.require(log.count("[huanyu-precision] rollout installed: gdn_fp32=True") >= c["tp"],
                          f"Replica {replica}: missing {c['tp']} successful precision install messages; inspect serve log")
    tokenizer = AutoTokenizer.from_pretrained(c["tokenizer"], trust_remote_code=True)
    prompts = bench.make_prompts(tokenizer, c["replicas"] + c["train_batch_size"] * (c["warmup_batches"] + c["num_batches"]),
                                 c["input_tokens"], c["seed"], c["enable_thinking"], "distinct")
    report = dict(config=c, profiling_run=False, input_modality="text", valid=False, probe=[], warmup=[], batches=[],
                  summary=None, corpus_sha256=hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
                  timing_definition="sum of joint batch wall times; excludes probe, warmup and offline prompt preparation")
    print(f"TP={c['tp']} replicas={c['replicas']} total_NPUs=16 precision={c['precision']} eager prefix_cache=OFF", flush=True)
    print(f"prompts/batch={c['train_batch_size']} rollout_n={c['rollout_n']} trajectories/batch={c['trajectories_per_batch']} "
          f"HTTP n=1 concurrency/replica={c['client_concurrency_per_replica']}", flush=True)
    try:
        for replica in range(c["replicas"]):
            result = send(c, replica, prompts[replica], replica, min(16, c["output_tokens"]))
            report["probe"].append(result)
            bench.require(result["success"], f"Replica {replica} preflight: {result.get('error')}")
        cursor = c["replicas"]
        request_id = cursor
        with ExitStack() as stack:
            pools = [stack.enter_context(ThreadPoolExecutor(max_workers=c["client_concurrency_per_replica"]))
                     for _ in range(c["replicas"])]
            for phase, number in (("warmup", c["warmup_batches"]), ("batches", c["num_batches"])):
                for i in range(number):
                    batch = run_batch(c, pools, prompts[cursor:cursor+c["train_batch_size"]], request_id)
                    cursor += c["train_batch_size"]
                    request_id += c["trajectories_per_batch"]
                    report[phase].append(batch)
                    s = summary([batch], c)
                    print(f"{phase} {i+1}: wall={batch['wall_time_s']:.3f}s output={s['total_completion_tokens']} "
                          f"aggregate={s['throughput_tokens_per_s']} per_card={s['output_tokens_per_s_per_device']} "
                          f"errors={s['num_errors']}", flush=True)
                    bench.require(s["valid"], f"Failed requests: {[r for r in batch['requests'] if not r['success']]}")
        report["summary"] = summary(report["batches"], c)
        report["valid"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        write(target, report)
        print(f"Report: {target}", flush=True)
    return 0 if report["valid"] else 1


def show(args):
    r = read(args.case_dir / "throughput.json")
    bench.require(r["valid"] and not r["profiling_run"], "Incomplete/invalid benchmark")
    c = r["config"]
    bench.require(len(r["batches"]) == c["num_batches"], "Missing batches")
    bench.require(all(len(b["requests"]) == c["trajectories_per_batch"] for b in r["batches"]), "Missing trajectories")
    s = summary(r["batches"], c)
    bench.require(s["valid"] and math.isclose(s["throughput_tokens_per_s"], r["summary"]["throughput_tokens_per_s"]),
                  "Summary mismatch")
    print(f"正式请求={s['num_requests']} 输出={s['total_completion_tokens']} tokens 总时间={s['total_wall_clock_s']:.3f}s")
    for item in s["replicas"]:
        print(f"副本 {item['replica']} TP={c['tp']}: {item['throughput_tokens_per_s']:.3f} tok/s（共同测量窗口）")
    print(f"16卡聚合输出吞吐={s['throughput_tokens_per_s']:.3f} tok/s")
    print(f"平均单卡输出吞吐={s['output_tokens_per_s_per_device']:.3f} tok/s/卡（TP设备均摊，不是TP=1实测）")
    print(f"输入+输出/纯推理时间/16={s['total_tokens_per_s']/16:.3f} tok/s/卡（仍不是训练step吞吐）")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--project-root", default="/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new")
    p.add_argument("--tokenizer", default=MODEL)
    p.add_argument("--model", default="qwen3.8")
    p.add_argument("--devices", default=",".join(map(str, range(16))))
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--port", type=int, default=28889)
    p.add_argument("--precision", choices=("stem_full", "native"), default="stem_full")
    p.add_argument("--hccl-op-expansion-mode", default="AIV")
    for name, value in (("max-num-seqs", 8), ("client-concurrency-per-replica", 8), ("max-model-len", 12288),
                        ("max-num-batched-tokens", 12288), ("input-tokens", 2048), ("output-tokens", 3072),
                        ("train-batch-size", 4), ("rollout-n", 4), ("warmup-batches", 1), ("num-batches", 3),
                        ("top-k", 20), ("seed", 42)):
        p.add_argument("--" + name, type=int, default=value)
    for name, value in (("gpu-memory-utilization", .6), ("temperature", 1.), ("top-p", .95),
                        ("repetition-penalty", 1.05), ("timeout-s", 7200.)):
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    for name in ("serve", "run", "report"):
        child = sub.add_parser(name)
        child.add_argument("--case-dir", type=Path, required=True)
        if name == "serve":
            child.add_argument("--replica", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        return {"prepare": prepare, "serve": serve, "run": run, "report": show}[args.action](args) or 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
