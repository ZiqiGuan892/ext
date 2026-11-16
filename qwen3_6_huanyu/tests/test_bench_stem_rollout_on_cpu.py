import copy
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("stem_bench", ROOT / "bench_stem_rollout.py")
stem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stem)
sys.path.pop(0)


def source_config():
    return dict(pipeline="minicpmo_5_0_thinker_only", enable_prefix_caching=True,
                stages=[dict(stage_id=0, max_num_seqs=16, tensor_parallel_size=1, devices="0")],
                platforms=dict(npu=dict(stages=[dict(stage_id=0, tensor_parallel_size=2, devices="0,1",
                    enforce_eager=False, compilation_config=dict(cudagraph_mode="PIECEWISE"))])))


def case(tmp_path, extra=()):
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(source_config()))
    root = tmp_path / "run"
    assert stem.main(["prepare", "--case-dir", str(root), "--source", str(source), *extra]) == 0
    return root, stem.read(root / "case.json")


def result(replica, count, request_id=0):
    return dict(replica=replica, request_id=request_id, success=True, completion_tokens=count,
                prompt_tokens=20, request_latency_s=99,
                choices=[dict(ttft_s=None, tpot_s=None, max_tokens_per_chunk=0)])


def test_common_clock_aggregate_and_per_card_no_sum_of_replica_times():
    batches = [dict(wall_time_s=10, requests=[result(0, 100), result(1, 200)]),
               dict(wall_time_s=20, requests=[result(0, 300), result(1, 600)])]
    s = stem.summary(batches, dict(num_devices=16, tp=8, replicas=2))
    assert s["total_wall_clock_s"] == 30
    assert s["throughput_tokens_per_s"] == 40  # 1200 / 30, not /60 or sum of rates
    assert s["output_tokens_per_s_per_device"] == 2.5
    assert sum(r["throughput_tokens_per_s"] for r in s["replicas"]) == pytest.approx(40)
    assert s["total_tokens_per_s"] == pytest.approx(1280 / 30)


def test_failure_does_not_report_valid_global_throughput():
    b = dict(wall_time_s=10, requests=[result(0, 100), dict(replica=1, success=False, error="failed")])
    s = stem.summary([b], dict(num_devices=16, tp=8, replicas=2))
    assert not s["valid"]
    assert s["throughput_tokens_per_s"] is None
    assert s["output_tokens_per_s_per_device"] is None


def test_both_replicas_run_together_and_expand_n_without_serializing():
    barrier = threading.Barrier(2)
    seen = []
    lock = threading.Lock()

    def send(c, replica, prompt, request_id):
        barrier.wait(timeout=3)  # Would fail if replica 1 only started after 0 finished.
        with lock:
            seen.append((replica, prompt, request_id))
        return result(replica, 10, request_id)

    with ThreadPoolExecutor(max_workers=1) as a, ThreadPoolExecutor(max_workers=1) as b:
        batch = stem.run_batch(dict(rollout_n=4, replicas=2), [a, b], ["A", "B"], 7, send)
    assert sorted(x[2] for x in seen) == list(range(7, 15))
    assert sum(x[1] == "A" for x in seen) == sum(x[1] == "B" for x in seen) == 4
    assert sum(x[0] == 0 for x in seen) == sum(x[0] == 1 for x in seen) == 4
    assert batch["wall_time_s"] > 0
    assert len(batch["requests"]) == 8


def test_deploy_matches_stem_in_both_overlays(tmp_path):
    root, c = case(tmp_path)
    cfg = yaml.safe_load((root / "deploy.yaml").read_text())
    assert c["device_groups"] == ["0,1,2,3,4,5,6,7", "8,9,10,11,12,13,14,15"]
    assert c["ports"] == [28889, 28890]
    assert c["trajectories_per_batch"] == 16
    assert c["replicas"] == 2
    for stage in (cfg["stages"][0], cfg["platforms"]["npu"]["stages"][0]):
        assert stage["tensor_parallel_size"] == 8 and stage["data_parallel_size"] == 1
        assert stage["devices"] == "0,1,2,3,4,5,6,7"  # visible device IDs differ per SERVER
        assert stage["max_num_seqs"] == 8 and stage["gpu_memory_utilization"] == .6
        assert stage["max_model_len"] == stage["max_num_batched_tokens"] == 12288
        assert stage["enforce_eager"] and stage["compilation_config"]["cudagraph_mode"] == "NONE"
        assert not stage["enable_prefix_caching"] and not stage["enable_chunked_prefill"]
        assert stage["additional_config"]["weight_nz_mode"] == 0
        assert stage["worker_extension_cls"].endswith("HuanyuColocateWorkerExtension")


@pytest.mark.parametrize("extra", [["--devices", "0,0"], ["--tp", "3"], ["--input-tokens", "12288"],
                                 ["--max-num-batched-tokens", "2048"], ["--warmup-batches", "0"],
                                 ["--train-batch-size", "1", "--rollout-n", "1"], ["--top-p", "1.2"]])
def test_invalid_setup_fails_before_writing_case(tmp_path, extra):
    with pytest.raises(SystemExit):
        case(tmp_path, extra)
    assert not (tmp_path / "run/case.json").exists()


def test_no_overwrite_existing_case(tmp_path):
    root, _ = case(tmp_path)
    saved = (root / "case.json").read_bytes()
    with pytest.raises(SystemExit):
        stem.main(["prepare", "--case-dir", str(root), "--source", str(tmp_path / "source.yaml")])
    assert (root / "case.json").read_bytes() == saved


def test_precisions_and_profiler_cleanup(tmp_path):
    _, c = case(tmp_path)
    inherited = dict(HUANYU_GREEDY_CASE="old", HUANYU_SERVE_PROFILE_DIR="old", HUANYU_GEN_TIMING="1",
                     MINICPMO_GDN_PREFILL_REFERENCE="0", ASCEND_RT_VISIBLE_DEVICES="0", VLLM_OMNI_EXTERNAL_MODULES="old")
    env = stem.serve_environment(c, 1, inherited)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    assert env["VLLM_OMNI_EXTERNAL_MODULES"] == stem.HOOK
    assert all(env[k] == "1" for k in stem.FULL_ON)
    assert all(env[k] == "0" for k in stem.FULL_OFF)
    assert env["HCCL_OP_EXPANSION_MODE"] == "AIV"
    assert not any(k.startswith(("HUANYU_SERVE_PROFILE_", "HUANYU_GREEDY_")) for k in env)
    assert "HUANYU_GEN_TIMING" not in env
    assert inherited["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    c["precision"] = "native"
    env = stem.serve_environment(c, 1, inherited)
    assert "VLLM_OMNI_EXTERNAL_MODULES" not in env
    assert all(env[k] == "0" for k in stem.FULL_ON + stem.FULL_OFF)


def test_sampling_expansion_uses_n1_with_stem_sampling(tmp_path, monkeypatch):
    _, c = case(tmp_path)
    captured = []

    def transport(args, payload, request_id, expected_input_tokens):
        captured.append((args, payload, request_id, expected_input_tokens))
        return dict(success=True)

    monkeypatch.setattr(stem.bench, "send_payload", transport)
    stem.send(c, 1, "test", 7)
    a, payload, request_id, expected = captured[0]
    assert a.port == 28890 and a.timing_source == "none"
    assert payload["n"] == 1 and payload["seed"] == c["seed"] + 7
    assert payload["top_p"] == .95 and payload["top_k"] == 20 and payload["repetition_penalty"] == 1.05
    assert payload["max_tokens"] == payload["min_tokens"] == 3072
    assert request_id == 7 and expected == 2048


def test_report_rejects_incomplete_rounds(tmp_path):
    root, c = case(tmp_path)
    (root / "throughput.json").write_text(json.dumps(dict(valid=True, profiling_run=False, config=c, batches=[])))
    with pytest.raises(SystemExit):
        stem.main(["report", "--case-dir", str(root)])


def test_config_builder_leaves_source_unchanged(tmp_path):
    _, c = case(tmp_path)
    source = source_config()
    before = copy.deepcopy(source)
    stem.make_deploy(source, c)
    assert source == before


def test_end_to_end_client_excludes_probe_warmup_and_saves_both_replicas(tmp_path, monkeypatch):
    root, c = case(tmp_path)
    for replica in (0, 1):
        (root / f"launch_{replica}.json").write_text(json.dumps(dict(precision="stem_full")))
        (root / f"serve_{replica}.log").write_text("[huanyu-precision] rollout installed: gdn_fp32=True\n" * 8)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object())))
    monkeypatch.setattr(stem.bench, "make_prompts", lambda tok, count, *a: [f"prompt{i}" for i in range(count)])

    def transport(args, payload, request_id, expected_input_tokens):
        return {**result(args.port - 28889, payload["max_tokens"], request_id),
                "prompt_tokens": expected_input_tokens}

    monkeypatch.setattr(stem.bench, "send_payload", transport)
    assert stem.main(["run", "--case-dir", str(root)]) == 0
    r = stem.read(root / "throughput.json")
    assert r["valid"]
    assert len(r["probe"]) == 2 and len(r["warmup"]) == 1 and len(r["batches"]) == 3
    assert r["summary"]["total_completion_tokens"] == 147456
    assert r["summary"]["num_requests"] == 48
    assert [x["num_requests"] for x in r["summary"]["replicas"]] == [24, 24]
    assert stem.main(["report", "--case-dir", str(root)]) == 0
