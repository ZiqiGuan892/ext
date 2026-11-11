import copy
import importlib.util
import json
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench_throughput")
deploy = load("prepare_benchmark_configs")


def emit(meter, elapsed, **chunk):
    meter.feed(json.dumps(chunk), elapsed)


def finish_event(meter, count, prompt=2048):
    emit(meter, 10, choices=[], usage={"prompt_tokens": prompt, "completion_tokens": count})
    meter.feed("[DONE]", 10)


def sample(n=8, tokens=3, source="token_ids"):
    meter = bench.StreamMetrics(n, source)
    for step in range(tokens):
        emit(meter, 1 + step * 0.05, choices=[
            {"index": i, "delta": {"content": "several words"},
             "token_ids": [100 + step],
             "logprobs": {"content": [{"token": "x", "logprob": -1}]},
             "finish_reason": "length" if step == tokens - 1 else None}
            for i in range(n)
        ])
    finish_event(meter, n * tokens)
    return meter


def test_n8_latency_is_per_choice_not_divided_by_eight():
    result = sample().finish(10, 2048, 3, True)
    assert result["completion_tokens"] == 24
    assert len(result["choices"]) == 8
    assert [c["tpot_s"] for c in result["choices"]] == pytest.approx([0.05] * 8)
    # The delayed usage/footer is deliberately absent from per-choice TPOT.
    assert result["request_latency_s"] == 10


def test_first_token_is_tracked_for_each_choice_separately():
    meter = bench.StreamMetrics(2, "token_ids")
    for i, first, last in ((0, 1, 2), (1, 4, 6)):
        emit(meter, first, choices=[{"index": i, "token_ids": [1]}])
        emit(meter, last, choices=[{"index": i, "token_ids": [2], "finish_reason": "length"}])
    finish_event(meter, 4)
    result = meter.finish(10, 2048, 2, True)
    assert [c["ttft_s"] for c in result["choices"]] == [1, 4]
    assert [c["tpot_s"] for c in result["choices"]] == [1, 2]


def test_multi_token_and_reasoning_chunks_use_ids_not_text_events():
    meter = bench.StreamMetrics(1, "token_ids")
    emit(meter, 0.2, choices=[{"index": 0, "delta": {"role": "assistant"}, "token_ids": []}])
    emit(meter, 1, choices=[{"index": 0, "delta": {"reasoning_content": "thought"}, "token_ids": [1, 2]}])
    emit(meter, 3, choices=[{"index": 0, "delta": {"content": ""}, "token_ids": [3, 4, 5],
                           "finish_reason": "length"}])
    finish_event(meter, 5)
    choice = meter.finish(10, 2048, 5, True)["choices"][0]
    assert choice["tokens"] == 5
    assert choice["token_chunks"] == 2
    assert choice["content_chunks"] == 1
    assert choice["ttft_s"] == 1
    assert choice["tpot_s"] == 0.5
    assert choice["max_tokens_per_chunk"] == 3


def test_logprobs_are_an_explicit_alternative_count_source():
    result = sample(n=2, source="logprobs").finish(10, 2048, 3, True)
    assert result["completion_tokens"] == 6
    assert result["choices"][0]["tpot_s"] == pytest.approx(0.05)


def test_usage_only_does_not_invent_sequence_counts_or_latency():
    result = sample(n=2, source="none").finish(10, 2048, 3, True)
    assert result["completion_tokens"] == 6
    for c in result["choices"]:
        assert c["tokens"] is c["ttft_s"] is c["tpot_s"] is None


@pytest.mark.parametrize("failure,match", [
    ("done", "Missing.*DONE"), ("usage", "Missing final usage"),
    ("choice", "Not all"), ("input", "tokenizer/chat template mismatch"),
    ("short", "expected n\\*max_tokens"),
    ("metadata", "count=.*usage="),
])
def test_incomplete_or_inconsistent_stream_is_not_success(failure, match):
    meter = sample(n=2)
    if failure == "done":
        meter.done = False
    elif failure == "usage":
        meter.usage = None
    elif failure == "choice":
        meter.choices[1]["finish_reason"] = None
    elif failure == "input":
        meter.usage["prompt_tokens"] = 2010
    elif failure == "short":
        meter.usage["completion_tokens"] -= 1
    else:
        meter.choices[0]["tokens"] = 0
    with pytest.raises(bench.BenchmarkError, match=match):
        meter.finish(10, 2048, 3, True)


def test_per_choice_length_checked_even_when_total_is_correct():
    meter = sample(n=2)
    meter.choices[0]["tokens"] = 2
    meter.choices[1]["tokens"] = 4
    with pytest.raises(bench.BenchmarkError, match="wrong output length"):
        meter.finish(10, 2048, 3, True)


def test_stop_on_final_allowed_token_keeps_exact_length_measurement():
    meter = sample(n=1)
    meter.choices[0]["finish_reason"] = "stop"
    assert meter.finish(10, 2048, 3, True)["completion_tokens"] == 3


def test_error_inside_http_200_is_failure():
    with pytest.raises(bench.BenchmarkError, match="Engine died"):
        emit(bench.StreamMetrics(1, "token_ids"), 1, error={"message": "Engine died"})


def test_buffered_all_at_once_cannot_claim_zero_tpot():
    meter = bench.StreamMetrics(1, "token_ids")
    emit(meter, 1, choices=[{"index": 0, "token_ids": [1, 2, 3], "finish_reason": "length"}])
    finish_event(meter, 3)
    with pytest.raises(bench.BenchmarkError, match="buffered"):
        meter.finish(10, 2048, 3, True)


def test_single_output_token_has_no_tpot():
    result = sample(n=1, tokens=1).finish(10, 2048, 1, True)
    assert result["choices"][0]["tpot_s"] is None


def test_sse_handles_fragmented_utf8_comments_multiline_and_combined_events():
    wire = ': keepalive\r\nevent: message\r\ndata: {"text":\r\ndata: "中文"}\r\n\r\ndata: [DONE]\n\n'.encode()
    chunks = [wire[i:i + 1] for i in range(len(wire))]
    events = list(bench.sse_events(bench.stream_lines(chunks)))
    assert json.loads(events[0]) == {"text": "中文"}
    assert events[1] == "[DONE]"


@pytest.mark.parametrize("wire", [b'data: {}', b'data: {}\n'])
def test_truncated_sse_is_rejected(wire):
    with pytest.raises(bench.BenchmarkError, match="ended inside"):
        list(bench.sse_events(bench.stream_lines([wire])))


def args(**overrides):
    values = dict(model="test", output_tokens=3, input_tokens=2048, temperature=1.0,
                  n=2, ignore_eos=True, enable_thinking=False, timing_source="token_ids",
                  host="127.0.0.1", port=28889, api_key="EMPTY", timeout_s=10, seed=42)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("source", ["token_ids", "logprobs", "none"])
def test_payload_uses_top_level_extras_and_explicit_sampling(source):
    payload = bench.make_payload(args(timing_source=source), "hello", seed=7)
    assert "extra_body" not in payload
    assert payload["ignore_eos"] is True
    assert payload["min_tokens"] == payload["max_tokens"] == 3
    assert payload["n"] == 2 and payload["temperature"] == 1
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload.get("return_token_ids", False) is (source == "token_ids")
    assert payload.get("logprobs", False) is (source == "logprobs")


def test_aggregate_is_total_usage_over_total_wall_not_mean_batch_rates():
    # Batch 1: 24 / 2 = 12 tok/s; batch 2: 24 / 6 = 4 tok/s.
    # Aggregate is 48 / 8 = 6, NOT mean(12,4)=8, nor multiplied by n.
    result = sample().finish(10, 2048, 3, True)
    summary = bench.summarize([
        {"wall_time_s": 2, "requests": [result]}, {"wall_time_s": 6, "requests": [result]},
    ], num_devices=2)
    assert summary["throughput_tokens_per_s"] == 6
    assert summary["output_tokens_per_s_per_device"] == 3
    assert summary["total_prompt_tokens"] == 4096
    assert summary["num_successful_sequences"] == 16
    assert summary["tpot_s"]["mean"] == pytest.approx(0.05)


def test_failure_invalidates_comparison_without_removing_its_wall_time():
    good = sample(n=1).finish(10, 2048, 3, True)
    summary = bench.summarize([
        {"wall_time_s": 2, "requests": [good]},
        {"wall_time_s": 4, "requests": [{"success": False, "error": "failed"}]},
    ])
    assert summary["valid"] is False
    assert summary["throughput_tokens_per_s"] is None
    assert summary["successful_output_tokens_per_s"] == 0.5
    assert summary["num_errors"] == 1


def test_n_greater_than_one_with_greedy_is_rejected_locally():
    with pytest.raises(SystemExit):
        bench.parse_args(["--tokenizer", "dummy", "--n", "8", "--temperature", "0"])


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return list(range(7 + len(messages[-1]["content"].split())))


def test_exact_templated_input_and_reproducible_distinct_prompts():
    tokenizer = Tokenizer()
    prompts = bench.make_prompts(tokenizer, 4, 64, 42, False)
    assert len(set(prompts)) == 4
    assert prompts == bench.make_prompts(tokenizer, 4, 64, 42, False)
    assert all(len(tokenizer.apply_chat_template(bench.messages_for(p), tokenize=True,
               add_generation_prompt=True, enable_thinking=False)) == 64 for p in prompts)


@pytest.mark.parametrize("container", ["mapping", "batch", "mapping_batch", "array"])
def test_exact_2048_tokens_with_chat_template_return_containers(container):
    class Array:
        def __init__(self, ids):
            self.ids = ids

        def tolist(self):
            return [self.ids]

    class ContainerTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            if container == "mapping":
                return UserDict(input_ids=ids, attention_mask=[1] * len(ids))
            if container == "batch":
                return [ids]
            if container == "mapping_batch":
                return UserDict(input_ids=[ids], attention_mask=[[1] * len(ids)])
            return UserDict(input_ids=Array(ids))

    prompts = bench.make_prompts(ContainerTokenizer(), 2, 2048, 42, False)
    assert len(set(prompts)) == 2
    # Count independently with the flat tokenizer, not the production helper.
    assert all(len(Tokenizer().apply_chat_template(bench.messages_for(p), tokenize=True,
               add_generation_prompt=True, enable_thinking=False)) == 2048 for p in prompts)


@pytest.mark.parametrize("encoded,match", [
    ({"attention_mask": [1, 1]}, "no input_ids"),
    ([[1, 2], [3, 4]], "multiple sequences"),
    ({"input_ids": "123"}, "Expected nonempty token IDs"),
    ({"input_ids": []}, "Expected nonempty token IDs"),
    ({"input_ids": [[[]]]}, "integer token IDs"),
])
def test_invalid_template_result_cannot_be_counted_as_tokens(encoded, match):
    with pytest.raises(bench.BenchmarkError, match=match):
        bench.tokenized_chat_length(encoded)


def test_http_path_uses_token_metadata_and_closes_response(monkeypatch):
    wire = []
    for j in range(3):
        event = {"choices": [{"index": i, "token_ids": [j], "delta": {"content": "many words"},
                              "finish_reason": "length" if j == 2 else None} for i in range(2)]}
        wire.append(("data: " + json.dumps(event) + "\n\n").encode())
    wire.append(b'data: {"choices":[], "usage":{"prompt_tokens":2048,"completion_tokens":6}}\n\n')
    wire.append(b'data: [DONE]\n\n')
    exited = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            exited.append("response")

        def iter_content(self, chunk_size):
            assert chunk_size is None
            yield from wire

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            exited.append("session")

        def post(self, url, **kwargs):
            assert self.trust_env is False
            assert kwargs["json"]["return_token_ids"] is True
            return Response()

    clock = iter([0, 1, 2, 3, 4, 5, 5])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(bench.requests, "Session", Session)
    result = bench.send_request(args(), "hello", 3)
    assert result["success"] is True
    assert result["completion_tokens"] == 6
    assert result["choices"][0]["tpot_s"] == 1
    assert exited == ["response", "session"]


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_run_excludes_probe_and_warmup_and_stops_after_failure(tmp_path, monkeypatch, warmup_fails):
    arm = tmp_path / "profile.arm"
    parsed = bench.parse_args([
        "--tokenizer", "fake", "--input-tokens", "64", "--output-tokens", "3",
        "--batch-size", "2", "--n", "2", "--warmup-batches", "1", "--num-batches", "3",
        "--report", str(tmp_path / "report.json"),
        "--profile-arm-file", str(arm),
    ])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))

    def fake_send(args, prompt, request_id, output_tokens=None):
        assert arm.exists() == (request_id >= 3)
        if warmup_fails and request_id == 1:
            return {"request_id": request_id, "success": False, "error": "warmup failed"}
        limit = output_tokens if output_tokens is not None else args.output_tokens
        meter = sample(n=args.n, tokens=limit)
        meter.usage["prompt_tokens"] = args.input_tokens
        return {"request_id": request_id, **meter.finish(10, args.input_tokens, limit, True)}

    monkeypatch.setattr(bench, "send_request", fake_send)
    assert bench.run(parsed) == (1 if warmup_fails else 0)
    report = json.loads(parsed.report.read_text(encoding="utf-8"))
    assert len(report["warmup"]) == 1
    if warmup_fails:
        assert report["summary"] is None
        assert report["batches"] == []
    else:
        assert len(report["batches"]) == 3
        assert report["summary"]["num_requests"] == 6
        assert report["summary"]["total_completion_tokens"] == 36
        assert report["summary"]["total_wall_clock_s"] == sum(b["wall_time_s"] for b in report["batches"])
    assert "api_key" not in report["config"]
    assert report["profiling_run"] is True
    assert arm.exists() is not warmup_fails


def deploy_fixture():
    return {"pipeline": "minicpmo_5_0_thinker_only", "session_mode": "duplex",
            "enable_prefix_caching": True,
            "stages": [{"stage_id": 0, "max_num_seqs": 16, "tensor_parallel_size": 1,
                        "devices": "0", "max_model_len": 32768, "enforce_eager": False,
                        "default_sampling_params": {"temperature": 0.0, "max_tokens": 2048}}],
            "platforms": {"npu": {"stages": [
                {"stage_id": 0, "gpu_memory_utilization": 0.9, "tensor_parallel_size": 2,
                 "devices": "0,1", "max_num_batched_tokens": 16384,
                 "compilation_config": {"cudagraph_mode": "PIECEWISE"}},
            ]}}}


def test_ab_pair_preserves_npu_topology_and_changes_only_graph_mode():
    original = deploy_fixture()
    before = copy.deepcopy(original)
    pair = deploy.build_pair(original)
    assert original == before
    for name, config in pair.items():
        effective = deploy.merge(config["stages"][0], config["platforms"]["npu"]["stages"][0])
        assert effective["devices"] == "0,1"
        assert effective["tensor_parallel_size"] == 2
        assert effective["max_num_seqs"] == 16
        assert effective["max_num_batched_tokens"] == 16384
        assert effective["gpu_memory_utilization"] == 0.9
        assert effective["enforce_eager"] is (name == "eager")
        assert effective["compilation_config"]["cudagraph_mode"] == ("NONE" if name == "eager" else "PIECEWISE")
        assert config["enable_prefix_caching"] is False
        for stage in (config["stages"][0], config["platforms"]["npu"]["stages"][0]):
            stage.pop("enforce_eager")
            stage["compilation_config"].pop("cudagraph_mode")
    assert pair["eager"] == pair["graph"]


@pytest.mark.parametrize("problem", ["multi_stage", "legacy", "conflicting_extras"])
def test_deploy_helper_rejects_ambiguous_schema(problem):
    config = deploy_fixture()
    if problem == "multi_stage":
        config["stages"].append({"stage_id": 1})
    elif problem == "legacy":
        config["stages"][0]["engine_args"] = {}
    else:
        config["stages"][0]["engine_extras"] = {"enforce_eager": True}
    with pytest.raises(ValueError):
        deploy.build_pair(config)


def test_deploy_helper_never_overwrites_original_or_existing_outputs(tmp_path):
    source = tmp_path / "original.yaml"
    source.write_text(deploy.yaml.safe_dump(deploy_fixture()), encoding="utf-8")
    before = source.read_bytes()
    dest = tmp_path / "generated"
    deploy.main(["--source", str(source), "--output-dir", str(dest)])
    assert source.read_bytes() == before
    assert (dest / "huanyu_graph.yaml").is_file()
    with pytest.raises(SystemExit):
        deploy.main(["--source", str(source), "--output-dir", str(dest)])
