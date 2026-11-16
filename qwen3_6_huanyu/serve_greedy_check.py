#!/usr/bin/env python3
"""Five text or audio requests: observe output logits, compare eager/graph.

Standalone; no training imports, profiler, teacher forcing or precision changes.
Only synchronous single-request scheduling without speculative decoding is supported.
"""

import argparse
import ast
import base64
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import MethodType
import uuid


ENV = "HUANYU_GREEDY_CASE"
SIDE = "HUANYU_GREEDY_SIDE"
BEGIN, END = "# huanyu-greedy-check:begin", "# huanyu-greedy-check:end"
ACTIVE = ContextVar("huanyu_greedy_active", default=None)
OFF = dict(VLLM_BATCH_INVARIANT="0", HCCL_DETERMINISTIC="false",
           ASCEND_LAUNCH_BLOCKING="0", NCCL_DETERMINISTIC="0", LCCL_DETERMINISTIC="0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def source(fn):
    fn = inspect.unwrap(fn)
    path = inspect.getsourcefile(fn)
    return {"name": getattr(fn, "__qualname__", str(type(fn))), "path": path,
            "sha256": sha(Path(path).read_bytes()) if path and Path(path).is_file() else None}


def wraps_bound(original):
    signature = inspect.signature(original)
    receiver = '_huanyu_receiver'
    while receiver in signature.parameters:
        receiver = '_'+receiver
    signature = signature.replace(parameters=[inspect.Parameter(receiver, inspect.Parameter.POSITIONAL_ONLY),
                                               *signature.parameters.values()])
    def decorate(fn):
        fn = wraps(original)(fn)
        fn.__signature__ = signature
        return fn
    return decorate


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        require(len(value) == 1, "Expected one tokenized prompt")
        value = value[0]
    from numbers import Integral
    require(isinstance(value, (list, tuple)) and all(isinstance(x, Integral) and not isinstance(x, bool)
                                                   and x >= 0 for x in value), "Invalid token IDs")
    return list(map(int, value))


def audio_case(case):
    return case.get('input_modality') == 'audio'


def matches_request_id(rid, request_key):
    # AsyncOmni appends a unique suffix to the external chatcmpl request ID.
    base = 'chatcmpl-' + request_key
    return rid == base or rid.startswith(base + '-')


def case_row(case, prompt, rid):
    if audio_case(case):
        candidates = [row for row in case['rows'] if matches_request_id(rid, row['request_key'])]
        require(len(candidates) == 1, 'Audio request ID not preserved/matched: ' + str(rid))
    else:
        candidates = [row for row in case['rows'] if row['prompt_token_ids'] == prompt]
        require(len(candidates) == 1, 'Actual prompt token IDs do not match any of the five saved inputs')
    return candidates[0]


def tensor_tree(value):
    """Small audit of real encoder input/output, once per prefill, no data mutation."""
    import torch
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to('cpu', non_blocking=False).contiguous()
        return dict(kind='tensor', shape=list(cpu.shape), dtype=str(cpu.dtype), numel=cpu.numel(),
                    sha256=sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()))
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): tensor_tree(v) for k, v in sorted(value.items())}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError('Unknown audio tensor container: ' + str(type(value)))


def has_matrix(tree):
    if isinstance(tree, dict):
        if tree.get('kind') == 'tensor':
            return tree['numel'] > 0 and len(tree['shape']) >= 2
        return any(has_matrix(v) for v in tree.values())
    return isinstance(tree, list) and any(has_matrix(v) for v in tree)


def capture(logits):
    """Owned original-dtype bytes, before the sampler can modify its input."""
    import torch
    require(isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[0] == 1,
            "Expected exactly one [1, vocabulary] output row, not prompt positions or hidden states")
    require(logits.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), "Unknown logits dtype")
    cpu = logits.detach().to("cpu", non_blocking=False).contiguous().clone()
    values = cpu.double()
    require(not values.isnan().any().item() and not values.isposinf().any().item()
            and values.isfinite().any().item(), "Invalid logits: NaN, +inf or all masked")
    raw = cpu.view(torch.uint8).numpy().copy()
    return {"dtype": str(cpu.dtype), "shape": list(cpu.shape), "sha256": sha(raw.tobytes())}, raw


def decode(raw, dtype):
    import numpy as np
    if dtype == "torch.bfloat16":
        return (raw.reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    names = {"torch.float16": np.float16, "torch.float32": np.float32, "torch.float64": np.float64}
    require(dtype in names, "Unknown saved dtype: " + dtype)
    return raw.reshape(-1).view(names[dtype]).astype(np.float64)


def difference(a, b, dtype):
    import numpy as np
    require(a.shape == b.shape, "Logits byte shapes differ")
    x, y = decode(a, dtype), decode(b, dtype)
    finite = np.isfinite(x) & np.isfinite(y)
    valid = not (np.isnan(x).any() or np.isnan(y).any() or np.isposinf(x).any() or np.isposinf(y).any())
    error = np.abs(x[finite] - y[finite])
    return dict(bitwise_equal=bool(valid and np.array_equal(a, b)),
                numeric_equal=bool(valid and np.array_equal(x, y)),
                differing_values=int(np.count_nonzero(x != y)),
                masked_positions_equal=bool(np.array_equal(np.isneginf(x), np.isneginf(y))),
                finite_max_abs=float(error.max()) if error.size else 0.0,
                finite_abs_sum=float(error.sum()), finite_count=int(error.size))


def validate_sampling(params, metadata, output_tokens):
    # vLLM 0.28 normalizes greedy SamplingParams.top_k to 0 in __post_init__.
    # Both 0 and -1 disable top-k; retain the observed value in the audit record.
    top_k = get(params, "top_k")
    require(type(top_k) is int and top_k in (-1, 0), f"Unexpected greedy top_k: {top_k!r}")
    wanted = dict(temperature=0.0, top_p=1.0, top_k=top_k, n=1, repetition_penalty=1.0,
                  frequency_penalty=0.0, presence_penalty=0.0, ignore_eos=True, max_tokens=output_tokens)
    actual = {key: get(params, key) for key in wanted}
    differences = {key: {"expected": val, "actual": actual[key]}
                   for key, val in wanted.items() if actual[key] != val}
    require(not differences, "Actual SamplingParams differ from greedy case: " + json.dumps(differences, default=str))
    require(not get(params, "stop") and not get(params, "stop_token_ids"), "Unexpected explicit stop rules")
    require(get(params, "min_tokens") == output_tokens, "Expected fixed-length output")
    require(get(metadata, "all_greedy") is True and get(metadata, "all_random") is False,
            "Actual sampler is not all_greedy")
    return actual


def determinism():
    import torch
    env = {key: os.environ.get(key) for key in OFF}
    native = sys.modules.get("vllm_ascend.batch_invariant")
    registered = getattr(native, "_batch_invariant_LIB", None) is not None
    state = dict(environment=env, torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 batch_invariant_registered=registered)
    require(env == OFF and not state["torch_deterministic_algorithms"] and not registered,
            "Determinism was enabled or flags differ: " + json.dumps(state))
    return state


def layout(runner, width, vocab, rank, tp):
    if width == vocab:
        return dict(kind="full", start=0, end=vocab, width=width)
    # Never infer a TP slice merely by dividing vocab_size by TP: padding matters.
    candidates = []
    for name, module in runner.model.named_modules():
        if not name.endswith("lm_head"):
            continue
        indices = getattr(module, "shard_indices", None)
        start, end = get(indices, "org_vocab_start_index"), get(indices, "org_vocab_end_index")
        added = get(indices, "num_added_embeddings", get(module, "num_added_embeddings", 0))
        if type(start) is int and type(end) is int and 0 <= start < end <= vocab and end-start <= width and not added:
            candidates.append(dict(kind="shard", start=start, end=end, width=width))
    require(len(candidates) == 1 and tp > 1 and width < vocab,
            f"Cannot prove full vocabulary coverage: tp={rank}/{tp}, width={width}, vocab={vocab}, heads={candidates}")
    return candidates[0]


class Observer:
    def __init__(self, runner):
        import torch
        import importlib.metadata
        from vllm.distributed.parallel_state import get_tp_group
        self.runner = runner
        self.case_path = Path(os.environ[ENV]).resolve()
        self.case = read_json(self.case_path)
        self.case_hash = digest(self.case)
        self.side = os.environ[SIDE]
        require(self.side in ("eager", "graph"), "Unknown side")
        self.root = self.case_path.parent/self.side
        group = get_tp_group()
        self.rank, self.tp = group.rank_in_group, len(group.ranks)
        require(self.tp == 4, "This case requires TP=4")
        self.folder = self.root/f"tp{self.rank}"
        self.folder.mkdir(parents=True, exist_ok=True)
        require(not (self.folder/'runtime.json').exists(), "Duplicate worker/output directory; use a new case")
        self.config = runner.vllm_config
        require(get(self.config.cache_config, "enable_prefix_caching") is False, "Prefix cache must be OFF")
        require(get(self.config.scheduler_config, "async_scheduling", False) is False, "Disable async_scheduling in both groups")
        require(get(self.config, "speculative_config") is None, "Speculative decoding is unsupported")
        require(self.config.model_config.enforce_eager == (self.side == "eager"), "Effective eager setting mismatches case")
        self.vocab = int(self.config.model_config.get_vocab_size())
        self.requests, self.pending, self.sampling = {}, None, None
        self.replays, self.last_finished_replays = 0, 0
        sections = {
            "model_config": ("model", "dtype", "quantization", "max_model_len", "enforce_eager"),
            "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"),
            "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "async_scheduling", "enable_chunked_prefill"),
            "cache_config": ("enable_prefix_caching", "cache_dtype", "block_size"),
            "compilation_config": ("mode", "cudagraph_mode"),
        }
        config = {section: {key: str(get(get(self.config, section), key)) for key in keys} for section, keys in sections.items()}
        versions = {}
        for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "vllm-omni"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unknown"
        self.runtime = dict(case_hash=self.case_hash, side=self.side, tp_rank=self.rank, tp_size=self.tp,
                            host=socket.gethostname(), pid=os.getpid(), config=config, vocab_size=self.vocab,
                            determinism=determinism(), versions=versions,
                            observer_sha256=sha(Path(__file__).read_bytes()),
                            model_compute_logits=source(runner.model.compute_logits),
                            sampler=source(runner.sampler.forward), greedy_sample=source(runner.sampler.greedy_sample))
        if audio_case(self.case):
            require(callable(get(runner.model, 'embed_multimodal')), 'Audio audit requires model.embed_multimodal')
            self.runtime['audio_embedder'] = source(runner.model.embed_multimodal)
        write_json(self.folder/'runtime.json', self.runtime)
        print('[huanyu-greedy-check] SETTINGS_VERIFIED ' + json.dumps(self.runtime, ensure_ascii=False), flush=True)
        self.bind()

    def identity(self):
        batch = self.runner.input_batch
        ids = list(batch.req_ids)
        require(len(ids) == 1, "Expected one active request; do not send concurrent traffic")
        rid = ids[0]
        req = self.runner.requests[rid]
        prompt = token_ids(get(req, "prompt_token_ids"))
        prefix = token_ids(get(req, "output_token_ids"))
        row = case_row(self.case, prompt, rid)
        if rid not in self.requests:
            require(not prefix, "First observed sample already has generated tokens")
            require(not any(s['sample_id'] == row['sample_id'] for s in self.requests.values()), "Duplicate case request")
            self.requests[rid] = dict(sample_id=row['sample_id'], tokens=[], prompt_hash=digest(prompt),
                                      case_hash=self.case_hash, status="running", captured=0,
                                      replays_before=self.last_finished_replays)
            if audio_case(self.case):
                require(0 < len(prompt) <= self.case['input_token_budget'], 'Invalid/oversized expanded audio prompt IDs')
                self.requests[rid].update(prompt_token_ids=prompt, native_request_id=rid,
                                          audio_pcm_sha256=row['audio']['pcm_sha256'], audio_calls=[])
        state = self.requests[rid]
        require(state['prompt_hash'] == digest(prompt), 'Prompt changed during generation')
        require(prefix == state['tokens'], "Stale/asynchronous prefix or unobserved sampling step")
        return rid, req, state, len(prefix)

    def bind_audio(self):
        model = self.runner.model
        embed = model.embed_multimodal

        @wraps_bound(embed)
        def observe_audio(_model, *args, **kwargs):
            if ACTIVE.get() is not self:
                return embed(*args, **kwargs)
            import torch
            require(not torch.npu.is_current_stream_capturing(), 'Audio observation unexpectedly inside graph capture')
            _, _, state, step = self.identity()
            require(step == 0, 'Unexpected audio encoder invocation after decoding began')
            require('audio_features' in kwargs, 'Expected raw audio features at embed_multimodal; refusing text-only/precomputed path')
            inputs = tensor_tree({key: value for key, value in kwargs.items() if key.startswith('audio_')})
            require(has_matrix(inputs.get('audio_features')), 'No actual audio feature tensor')
            result = embed(*args, **kwargs)
            outputs = tensor_tree(result)
            require(has_matrix(outputs), 'Audio encoder returned no embedding matrix')
            state['audio_calls'].append(dict(inputs=inputs, outputs=outputs))
            return result

        model.embed_multimodal = MethodType(observe_audio, model)

    def bind(self):
        model, sampler = self.runner.model, self.runner.sampler
        compute, sample, greedy = model.compute_logits, sampler.forward, sampler.greedy_sample
        sample_sig = inspect.signature(sample)
        require({'logits', 'sampling_metadata'} <= set(sample_sig.parameters), 'Unknown sampler signature')

        @wraps_bound(compute)
        def observe_compute(_model, *args, **kwargs):
            result = compute(*args, **kwargs)
            if ACTIVE.get() is self:
                import torch
                require(not torch.npu.is_current_stream_capturing(), "Logits observation unexpectedly inside graph capture")
                rid, req, state, step = self.identity()
                require(self.pending is None, "Unconsumed model logits; unsupported runner path")
                vector = capture(result) if step < self.case['logits_steps'] else None
                self.pending = (rid, step, vector)
            return result

        @wraps_bound(greedy)
        def observe_greedy(_sampler, *args, **kwargs):
            result = greedy(*args, **kwargs)
            if self.sampling is not None:
                require(not self.sampling['greedy_called'], "Multiple greedy calls in one output position")
                self.sampling['greedy_called'] = True
            return result

        @wraps_bound(sample)
        def observe_sample(_sampler, *args, **kwargs):
            if ACTIVE.get() is not self:
                return sample(*args, **kwargs)
            import numpy as np
            rid, req, state, step = self.identity()
            if step == 0:
                determinism()
                if audio_case(self.case):
                    require(bool(state['audio_calls']), 'No audio encoder execution observed for this request')
            require(self.pending is not None and self.pending[:2] == (rid, step), "Sampler lacks matching original logits")
            vector = self.pending[2]
            self.pending = None
            bound = sample_sig.bind(*args, **kwargs)
            params = validate_sampling(get(req, 'sampling_params'), bound.arguments['sampling_metadata'], self.case['output_tokens'])
            self.sampling = {'greedy_called': False}
            try:
                result = sample(*args, **kwargs)
                selected = result.sampled_token_ids.detach().to('cpu', non_blocking=False).reshape(-1).tolist()
                require(self.sampling['greedy_called'] and len(selected) == 1, "No actual single-token greedy decision observed")
                chosen = int(selected[0])
                require(0 <= chosen < self.vocab, "Sampled token outside real vocabulary")
                if vector is not None:
                    info, raw = vector
                    shard = layout(self.runner, info['shape'][1], self.vocab, self.rank, self.tp)
                    folder = self.folder/state['sample_id']
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder/f'{step:03d}.npy'
                    with target.open('xb') as f:
                        np.save(f, raw, allow_pickle=False)
                    record = dict(step=step, sample_id=state['sample_id'], case_hash=self.case_hash,
                                  prompt_hash=state['prompt_hash'], prefix_hash=digest(state['tokens']),
                                  token_id=chosen, sampling=params, tensor=info, layout=shard,
                                  graph_replays=self.replays, file=target.name)
                    write_json(folder/f'{step:03d}.json', record)
                    state['captured'] += 1
                state['tokens'].append(chosen)
                if len(state['tokens']) == self.case['output_tokens']:
                    state.update(status='complete', graph_replays=self.replays-state['replays_before'])
                    self.last_finished_replays = self.replays
                    write_json(self.folder/(state['sample_id']+'.json'), state)
                    print(f"[huanyu-greedy-check] {self.side} tp={self.rank} {state['sample_id']} "
                          f"tokens={len(state['tokens'])} logits={state['captured']} replays={state['graph_replays']}", flush=True)
                return result
            finally:
                self.sampling = None

        model.compute_logits = MethodType(observe_compute, model)
        sampler.forward = MethodType(observe_sample, sampler)
        sampler.greedy_sample = MethodType(observe_greedy, sampler)
        if audio_case(self.case):
            self.bind_audio()


def install_runtime(cls):
    import torch
    import torch_npu
    torch.use_deterministic_algorithms(False, warn_only=False)
    # Python NPUGraph.replay shim: count actual replay calls, do not wrap model.forward.
    graph_cls = torch_npu.npu.NPUGraph
    replay = graph_cls.replay
    if not getattr(replay, '_huanyu_greedy_count', False):
        @wraps(replay)
        def counted(graph, *args, **kwargs):
            result = replay(graph, *args, **kwargs)
            state = ACTIVE.get()
            if state is not None:
                state.replays += 1
            return result
        counted._huanyu_greedy_count = True
        graph_cls.replay = counted
    for name in ('execute_model', 'sample_tokens'):
        original = getattr(cls, name, None)
        if original is None or getattr(original, '_huanyu_greedy_bound', False):
            continue
        def bind(original):
            @wraps(original)
            def wrapped(self, *args, **kwargs):
                if not os.environ.get(ENV):
                    return original(self, *args, **kwargs)
                state = getattr(self, '_huanyu_greedy_observer', None)
                if state is None:
                    state = Observer(self)
                    self._huanyu_greedy_observer = state
                token = ACTIVE.set(state)
                try:
                    return original(self, *args, **kwargs)
                except BaseException as exc:
                    write_json(state.folder/'error.json', dict(error=type(exc).__name__+': '+str(exc)))
                    raise
                finally:
                    ACTIVE.reset(token)
            wrapped._huanyu_greedy_bound = True
            return wrapped
        setattr(cls, name, bind(original))


def edit_entry(path, remove=False):
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    if BEGIN in text or END in text:
        require(text.count(BEGIN) == text.count(END) == 1 and text.index(BEGIN) < text.index(END), 'Malformed existing hook')
        start, stop = text.index(BEGIN), text.index(END)+len(END)
        if start and text[start-1] == '\n':
            start -= 1
        if text[stop:stop+1] == '\n':
            stop += 1
        text = text[:start]+text[stop:]
    elif remove:
        return
    require(any(isinstance(n, ast.ClassDef) and n.name == 'NPUARModelRunner' for n in ast.parse(text).body),
            'Expected installed NPUARModelRunner module')
    if not remove:
        text += (f'\n{BEGIN}\nimport os as _hgc_os\nif _hgc_os.environ.get({ENV!r}):\n'
                 '    import importlib.util as _hgc_util\n    import sys as _hgc_sys\n'
                 f'    _hgc_spec = _hgc_util.spec_from_file_location("_huanyu_greedy_check", {str(Path(__file__).resolve())!r})\n'
                 '    _hgc_module = _hgc_util.module_from_spec(_hgc_spec)\n'
                 '    _hgc_sys.modules["_huanyu_greedy_check"] = _hgc_module\n'
                 '    _hgc_spec.loader.exec_module(_hgc_module)\n'
                 f'    _hgc_module.install_runtime(NPUARModelRunner)\n{END}\n')
    compile(text, str(path), 'exec')
    if text.encode() != raw:
        backup = path.with_name(path.name+'.greedy-backup-'+uuid.uuid4().hex[:10])
        backup.write_bytes(raw)
        path.write_bytes(text.encode())
        print(f'Updated {path}\nBackup {backup}')


def build_case_pair(original, input_tokens, output_tokens):
    """Keep the unchunked scheduler valid for the fixed-length A/B case."""
    require(type(input_tokens) is int and input_tokens > 0, 'Input tokens must be positive')
    require(type(output_tokens) is int and output_tokens >= 100,
            'Output must contain at least 100 generated tokens')
    deploy = sibling('prepare_benchmark_configs')
    deploy.build_pair(original)
    pair = deploy.build_pair(deploy.with_topology(original, 4, '0,1,2,3'))
    context = input_tokens + output_tokens
    for config in pair.values():
        stages = (config['stages'][0], config['platforms']['npu']['stages'][0])
        effective = deploy.merge(*stages)
        budget = effective.get('max_num_batched_tokens')
        require(budget is None or (type(budget) is int and budget > 0),
                'max_num_batched_tokens must be a positive integer or null')
        # Preserve the configured budget when sufficient. Disabling chunked
        # prefill requires a budget at least as large as the entire context.
        budget = max(context, budget or 0, effective.get('max_num_seqs') or 128)
        for stage in stages:
            conflicts = {'max_model_len', 'max_num_batched_tokens', 'async_scheduling',
                         'enable_chunked_prefill'} & set(stage.get('engine_extras') or {})
            require(not conflicts, 'Conflicting scheduler settings in engine_extras: '+str(sorted(conflicts)))
            stage.update(async_scheduling=False, enable_chunked_prefill=False,
                         max_model_len=context, max_num_batched_tokens=budget)
            params = stage.setdefault('default_sampling_params', {})
            params.update(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                          frequency_penalty=0.0, presence_penalty=0.0, max_tokens=output_tokens)
    return pair


def prepare(args):
    from transformers import AutoTokenizer
    import yaml
    bench = sibling('bench_throughput')
    require(not args.case_dir.exists(), 'Choose a new case directory')
    original = yaml.safe_load(args.source.read_text(encoding='utf-8'))
    pair = build_case_pair(original, args.input_tokens, args.output_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = bench.make_prompts(tokenizer, 5, args.input_tokens, 42, False)
    rows = []
    for i, prompt in enumerate(prompts):
        messages = bench.messages_for(prompt)
        ids = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        require(len(ids) == args.input_tokens, 'Token length mismatch')
        rows.append(dict(sample_id=f'sample_{i:02d}', messages=messages, prompt_token_ids=ids))
    require(len({digest(row['prompt_token_ids']) for row in rows}) == 5, 'Expected five distinct tokenized prompts')
    args.case_dir.mkdir(parents=True)
    for side, config in pair.items():
        (args.case_dir/f'{side}.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    case = dict(format='huanyu_greedy_graph_v1', model=str(Path(args.model).resolve()),
                input_tokens=args.input_tokens, output_tokens=args.output_tokens, logits_steps=100,
                rows=rows, source_yaml_sha256=sha(args.source.read_bytes()))
    write_json(args.case_dir/'case.json', case)
    print(f'Prepared exactly 5 distinct text inputs: {args.case_dir.resolve()}')


def prepare_audio(args):
    import yaml
    require(not args.case_dir.exists(), 'Choose a new case directory')
    require(0 < args.min_audio_seconds <= args.max_audio_seconds < float('inf'), 'Invalid audio duration range')
    require(bool(args.prompt.strip()), 'An audio question is required')
    original = yaml.safe_load(args.source.read_text(encoding='utf-8'))
    pair = build_case_pair(original, args.input_token_budget, args.output_tokens)
    for config in pair.values():
        for stage in (config['stages'][0], config['platforms']['npu']['stages'][0]):
            require('mm_processor_cache_gb' not in (stage.get('engine_extras') or {}),
                    'Conflicting mm_processor_cache_gb in engine_extras')
            stage['mm_processor_cache_gb'] = 0
    audio = sibling('bench_audio_profile')
    selected, skipped = audio.select_audio(args, 5)
    rows = []
    (args.case_dir/'audio').mkdir(parents=True)
    for i, item in enumerate(selected):
        sample = f'sample_{i:02d}'
        filename = f'audio/{sample}.wav'
        data = base64.b64decode(item['data'], validate=True)
        (args.case_dir/filename).write_bytes(data)
        rows.append(dict(sample_id=sample, request_key=f'huanyu-audio-{uuid.uuid4().hex}-{i}',
                         audio_file=filename, wav_sha256=sha(data),
                         audio={k: v for k, v in item.items() if k != 'data'}))
    for side, config in pair.items():
        (args.case_dir/f'{side}.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    case = dict(format='huanyu_audio_greedy_graph_v1', input_modality='audio',
                model=str(Path(args.model).resolve()), input_token_budget=args.input_token_budget,
                output_tokens=args.output_tokens, logits_steps=100, prompt=args.prompt,
                audio_input_format=args.audio_input_format, selection_seed=args.seed,
                rows=rows, selection_skipped=skipped, source_yaml_sha256=sha(args.source.read_bytes()))
    write_json(args.case_dir/'case.json', case)
    print(f'Prepared exactly 5 distinct WAV snapshots: {args.case_dir.resolve()}', flush=True)
    for row in rows:
        print(f"{row['sample_id']}: {row['audio']['duration_s']:.3f}s {row['audio']['path']}")


def request_payload(case_dir, case, row, served_model):
    payload = dict(model=served_model, stream=False, n=1,
                   temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                   frequency_penalty=0.0, presence_penalty=0.0, max_tokens=case['output_tokens'],
                   ignore_eos=True, min_tokens=case['output_tokens'], skip_special_tokens=False,
                   stop=[], stop_token_ids=[], chat_template_kwargs=dict(enable_thinking=False))
    if not audio_case(case):
        payload['messages'] = row['messages']
        return payload
    data = (case_dir/row['audio_file']).read_bytes()
    require(sha(data) == row['wav_sha256'], 'WAV snapshot changed: '+row['sample_id'])
    encoded = base64.b64encode(data).decode('ascii')
    if case['audio_input_format'] == 'audio_url':
        part = dict(type='audio_url', audio_url=dict(url='data:audio/wav;base64,'+encoded))
    else:
        require(case['audio_input_format'] == 'input_audio', 'Unknown audio input format')
        part = dict(type='input_audio', input_audio=dict(data=encoded, format='wav'))
    payload.update(request_id=row['request_key'], modalities=['text'], messages=[
        dict(role='system', content='You are a helpful assistant.'),
        dict(role='user', content=[part, dict(type='text', text=case['prompt'])])])
    return payload


def request(args):
    import requests
    case = read_json(args.case_dir/'case.json')
    out = args.case_dir/args.side/'client.json'
    require(not out.exists(), 'Client report already exists; do not reuse a capture')
    report = dict(case_hash=digest(case), side=args.side, rows=[], complete=False)
    with requests.Session() as session:
        session.trust_env = False
        for row in case['rows']:
            payload = request_payload(args.case_dir, case, row, args.served_model)
            print(f"{args.side} {row['sample_id']} requesting {case['output_tokens']} tokens", flush=True)
            response = session.post(args.url, json=payload, timeout=(10, args.timeout))
            response.raise_for_status()
            data = response.json()
            choices, usage = data.get('choices', []), data.get('usage') or {}
            require(len(choices) == 1 and choices[0].get('finish_reason') == 'length', 'Incomplete generation: '+str(data)[:500])
            require(usage.get('completion_tokens') == case['output_tokens'], 'Server output usage does not match the case')
            capture_path = args.case_dir/args.side/'tp0'/(row['sample_id']+'.json')
            require(capture_path.is_file(), 'No matching TP0 capture: check current server side, hook and environment')
            observed = read_json(capture_path)
            require(observed['status'] == 'complete' and observed['captured'] == 100
                    and observed['case_hash'] == digest(case), 'Incomplete native capture')
            saved = dict(sample_id=row['sample_id'], usage=usage, message=choices[0]['message'])
            if audio_case(case):
                prompt = observed['prompt_token_ids']
                require(0 < len(prompt) <= case['input_token_budget'], 'Expanded audio prompt exceeds the input token budget')
                require(usage.get('prompt_tokens') == len(prompt), 'Audio usage disagrees with expanded native prompt IDs')
                require(matches_request_id(data.get('id', ''), row['request_key']), 'Server response request ID mismatch')
                require(observed['audio_calls'], 'No observed audio encoder call')
                saved.update(wav_sha256=row['wav_sha256'], request_key=row['request_key'], response_id=data['id'])
            else:
                require(usage.get('prompt_tokens') == case['input_tokens'], 'Server prompt usage does not match the case')
            report['rows'].append(saved)
            write_json(out, report)
    report['complete'] = True
    write_json(out, report)
    print(f'Completed 5 inputs: {out}')


def audio_evidence(case, row, results, client_rows):
    """Verify input alignment independently of any output/logit equality."""
    prompt = results['eager'][0]['prompt_token_ids']
    require(0 < len(prompt) <= case['input_token_budget'], 'Invalid expanded audio prompt length')
    for record in (r for records in results.values() for r in records):
        require(record['prompt_token_ids'] == prompt, 'Expanded audio prompt IDs differ across ranks/sides')
        require(record['prompt_hash'] == digest(prompt), 'Expanded audio prompt checksum differs')
        require(record['audio_pcm_sha256'] == row['audio']['pcm_sha256'], 'Audio PCM identity mismatch')
        require(matches_request_id(record['native_request_id'], row['request_key']), 'Native audio request ID mismatch')
        calls = record['audio_calls']
        require(calls and all(has_matrix(call['inputs'].get('audio_features')) and has_matrix(call['outputs'])
                              for call in calls), 'Missing actual audio feature/embedding evidence')
    for client in client_rows:
        require(client['wav_sha256'] == row['wav_sha256'] and client['request_key'] == row['request_key']
                and matches_request_id(client['response_id'], row['request_key']), 'Client audio identity mismatch')
        require(client['usage']['prompt_tokens'] == len(prompt), 'Audio prompt usage mismatch')
    features_equal = all([call['inputs'] for call in a['audio_calls']] == [call['inputs'] for call in b['audio_calls']]
                         for a, b in zip(results['eager'], results['graph']))
    embeddings_equal = all([call['outputs'] for call in a['audio_calls']] == [call['outputs'] for call in b['audio_calls']]
                           for a, b in zip(results['eager'], results['graph']))
    return prompt, dict(audio_input_verified=True, expanded_prompt_token_ids_equal=True,
                        audio_features_bitwise_equal=features_equal, audio_embeddings_bitwise_equal=embeddings_equal,
                        prompt_tokens=len(prompt), audio_duration_s=row['audio']['duration_s'])


def coverage(layouts, vocab):
    kinds = {item['kind'] for item in layouts}
    if kinds == {'full'}:
        require(all(item['start'] == 0 and item['end'] == item['width'] == vocab for item in layouts), 'Incomplete full-vocab layout')
    elif kinds == {'shard'}:
        end = 0
        for item in sorted(layouts, key=lambda x: x['start']):
            require(item['start'] == end and item['end']-item['start'] <= item['width'], 'Missing/overlapping vocabulary shard')
            end = item['end']
        require(end == vocab, 'Shards do not cover full vocabulary')
    else:
        raise ValueError('Inconsistent vocab layout across ranks')


def compare(case_dir):
    import numpy as np
    case = read_json(case_dir/'case.json')
    clients = {side: read_json(case_dir/side/'client.json') for side in ('eager', 'graph')}
    require(all(c['complete'] and c['case_hash'] == digest(case) and len(c['rows']) == 5 for c in clients.values()), 'Incomplete clients')
    runtime = {side: [read_json(case_dir/side/f'tp{r}/runtime.json') for r in range(4)] for side in clients}
    for rank in range(4):
        a, b = (runtime[side][rank] for side in clients)
        for item in (a, b):
            require(item['case_hash'] == digest(case) and item['tp_rank'] == rank and item['tp_size'] == 4, 'Runtime identity mismatch')
            require(item['determinism']['environment'] == OFF and not item['determinism']['torch_deterministic_algorithms']
                    and not item['determinism']['batch_invariant_registered'], 'Determinism not OFF')
            require(item['config']['cache_config']['enable_prefix_caching'] == 'False', 'Prefix cache not OFF')
            require(not (case_dir/item['side']/f'tp{rank}/error.json').exists(), 'Worker error exists')
        for key in ('versions', 'vocab_size', 'observer_sha256', 'model_compute_logits', 'sampler', 'greedy_sample'):
            require(a[key] == b[key], 'Runtime implementations differ: '+key)
        if audio_case(case):
            require(a['audio_embedder'] == b['audio_embedder'], 'Audio embedder implementations differ')
        ca, cb = json.loads(json.dumps(a['config'])), json.loads(json.dumps(b['config']))
        require(ca['model_config'].pop('enforce_eager') == 'True' and cb['model_config'].pop('enforce_eager') == 'False', 'Mode mismatch')
        ca.pop('compilation_config'); cb.pop('compilation_config')
        require(ca == cb, 'Effective runtime controls differ')
    findings = []
    for index, row in enumerate(case['rows']):
        sample = row['sample_id']
        results = {side: [read_json(case_dir/side/f'tp{rank}'/(sample+'.json')) for rank in range(4)] for side in clients}
        cm = [clients[side]['rows'][index] for side in clients]
        require(all(m['sample_id'] == sample for m in cm), 'Client sample order differs')
        if audio_case(case):
            prompt, input_evidence = audio_evidence(case, row, results, cm)
        else:
            prompt, input_evidence = row['prompt_token_ids'], {}
        for side, records in results.items():
            require(all(r['status'] == 'complete' and r['case_hash'] == digest(case) and r['captured'] == 100
                        and len(r['tokens']) == case['output_tokens'] for r in records), 'Incomplete capture '+sample)
            require(all(r['tokens'] == records[0]['tokens'] for r in records), 'TP sampled tokens disagree')
            require(all((r['graph_replays'] > 0 if side == 'graph' else r['graph_replays'] == 0) for r in records),
                    'Actual graph replay not verified (or eager unexpectedly replayed): '+side)
        a, b = (results[side][0]['tokens'] for side in clients)
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same_context = 100 if first is None else min(first+1, 100)
        metrics = dict(bitwise_equal=True, numeric_equal=True, differing_values=0, finite_max_abs=0.0,
                       finite_abs_sum=0.0, finite_count=0, masked_positions_equal=True)
        all_steps_exact, first_logits_difference = True, None
        graph_positions = [0]*4
        decode_graph_positions = [0]*4
        last_replays = [r['replays_before'] for r in results['graph']]
        for step in range(100):
            layouts = {side: [] for side in clients}
            for rank in range(4):
                saved = []
                for side in clients:
                    folder = case_dir/side/f'tp{rank}'/sample
                    rec = read_json(folder/f'{step:03d}.json')
                    require(rec['case_hash'] == digest(case) and rec['step'] == step and rec['sample_id'] == sample, 'Record identity mismatch')
                    require(rec['prompt_hash'] == digest(prompt)
                            and rec['prefix_hash'] == digest(results[side][rank]['tokens'][:step])
                            and rec['token_id'] == results[side][rank]['tokens'][step], 'Misaligned logits/prefix')
                    raw = np.load(folder/rec['file'], allow_pickle=False)
                    require(sha(raw.tobytes()) == rec['tensor']['sha256'], 'Logits checksum mismatch')
                    layouts[side].append(rec['layout'])
                    if side == 'graph':
                        replayed = int(rec['graph_replays'] > last_replays[rank])
                        graph_positions[rank] += replayed
                        if step > 0:
                            decode_graph_positions[rank] += replayed
                        last_replays[rank] = rec['graph_replays']
                    saved.append((rec, raw))
                (ra, xa), (rb, xb) = saved
                require(ra['layout'] == rb['layout'] and ra['tensor']['dtype'] == rb['tensor']['dtype'], 'Layout/dtype changed')
                # Padding outside the model vocabulary is not an output logit.
                width = ra['layout']['width']
                require(ra['tensor']['shape'] == rb['tensor']['shape'] == [1, width], 'Tensor shape/layout mismatch')
                require(xa.dtype == xb.dtype == np.uint8 and xa.shape == xb.shape
                        and xa.size % width == 0, 'Invalid raw tensor byte layout')
                valid_bytes = (ra['layout']['end']-ra['layout']['start'])*(xa.size//width)
                delta = difference(xa.reshape(-1)[:valid_bytes], xb.reshape(-1)[:valid_bytes], ra['tensor']['dtype'])
                if not delta['bitwise_equal']:
                    all_steps_exact = False
                    if first_logits_difference is None:
                        first_logits_difference = step
                if step < same_context:
                    for key in ('bitwise_equal', 'numeric_equal', 'masked_positions_equal'):
                        metrics[key] &= delta[key]
                    for key in ('differing_values', 'finite_abs_sum', 'finite_count'):
                        metrics[key] += delta[key]
                    metrics['finite_max_abs'] = max(metrics['finite_max_abs'], delta['finite_max_abs'])
            for side in clients:
                coverage(layouts[side], runtime[side][0]['vocab_size'])
        require(all(count > 0 for count in graph_positions), 'No graph replay observed during first 100 output positions')
        require(all(count > 0 for count in decode_graph_positions),
                'No graph replay during decode; prefill/audio encoder replay alone is insufficient')
        metrics['finite_mean_abs'] = metrics['finite_abs_sum']/metrics['finite_count'] if metrics['finite_count'] else 0.0
        result = dict(sample_id=sample, output_token_ids_equal=a == b,
                      output_text_equal=cm[0]['message'].get('content') == cm[1]['message'].get('content'),
                      output_reasoning_equal=all(cm[0]['message'].get(k) == cm[1]['message'].get(k)
                                                 for k in ('reasoning', 'reasoning_content')),
                      output_messages_equal=cm[0]['message'] == cm[1]['message'],
                      graph_positions_with_replay_per_tp=graph_positions,
                      graph_decode_positions_with_replay_per_tp=decode_graph_positions,
                      first_different_output_position=None if first is None else first+1,
                      logits_first100_bitwise_equal=all_steps_exact,
                      first_different_logits_position=None if first_logits_difference is None else first_logits_difference+1,
                      same_context_logits_positions=same_context,
                      context_diverged_positions_excluded=100-same_context,
                      same_context_metrics=metrics)
        result.update(input_evidence)
        findings.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = dict(case_hash=digest(case), valid=True, rows=findings, input_modality=case.get('input_modality', 'text'),
                  all_output_equal=all(x['output_token_ids_equal'] and x['output_text_equal']
                                       and x['output_reasoning_equal'] for x in findings),
                  all_logits_first100_bitwise_equal=all(x['logits_first100_bitwise_equal'] for x in findings),
                  note='Single run, determinism OFF. Differences do not alone prove a graph-only cause; copies perturb timing.')
    write_json(case_dir/'comparison.json', report)
    print(f"OUTPUT_EQUAL={report['all_output_equal']} LOGITS_FIRST100_BITWISE_EQUAL={report['all_logits_first100_bitwise_equal']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--input-tokens', type=int, default=2048)
    p.add_argument('--output-tokens', type=int, default=3072)
    p = sub.add_parser('prepare-audio')
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--input-token-budget', type=int, default=2048)
    p.add_argument('--output-tokens', type=int, default=3072)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--audio-dir', type=Path)
    inputs.add_argument('--audio-manifest', type=Path)
    inputs.add_argument('--audio-file', type=Path, action='append')
    p.add_argument('--min-audio-seconds', type=float, default=1)
    p.add_argument('--max-audio-seconds', type=float, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt', default='请根据音频内容回答其中的问题。')
    p.add_argument('--audio-input-format', choices=('audio_url', 'input_audio'), default='audio_url')
    p = sub.add_parser('install')
    p.add_argument('--runner-file', type=Path, required=True)
    p.add_argument('--remove', action='store_true')
    p = sub.add_parser('request')
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--side', choices=('eager', 'graph'), required=True)
    p.add_argument('--url', default='http://127.0.0.1:28889/v1/chat/completions')
    p.add_argument('--served-model', default='qwen3.8')
    p.add_argument('--timeout', type=float, default=1800)
    p = sub.add_parser('compare')
    p.add_argument('--case-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'prepare-audio':
        prepare_audio(args)
    elif args.action == 'install':
        edit_entry(args.runner_file, args.remove)
    elif args.action == 'request':
        request(args)
    else:
        compare(args.case_dir)


if __name__ == '__main__':
    main()
