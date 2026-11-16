import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('serve_greedy_check_cpu', ROOT/'serve_greedy_check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_capture_preserves_original_dtype_bytes_and_owns_storage(dtype):
    x = torch.tensor([[0.0, -0.0, 1.25, -float('inf')]], dtype=dtype)
    info, raw = check.capture(x)
    before = raw.copy()
    x[0, 2] = 9
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(check.decode(raw, info['dtype']), [0, -0.0, 1.25, -np.inf])
    assert check.sha(raw.tobytes()) == info['sha256']
    _, other = check.capture(torch.tensor([[0.0, 0.0, 1.25, -float('inf')]], dtype=dtype))
    difference = check.difference(raw, other, info['dtype'])
    assert difference['numeric_equal'] and not difference['bitwise_equal']


@pytest.mark.parametrize('x', [torch.zeros(2, 4), torch.zeros(4), torch.tensor([[float('nan'), 1.0]]),
                              torch.tensor([[float('inf'), 1.0]]), torch.tensor([[-float('inf')]])])
def test_capture_rejects_prompt_rows_and_invalid_logits(x):
    with pytest.raises(ValueError):
        check.capture(x)


def test_shard_coverage_uses_real_indices_not_tp_division():
    head = NS(shard_indices=NS(org_vocab_start_index=4, org_vocab_end_index=7))
    runner = NS(model=NS(named_modules=lambda: [('thinker.lm_head', head)]))
    result = check.layout(runner, 4, 7, 1, 2)
    assert result == dict(kind='shard', start=4, end=7, width=4)
    check.coverage([dict(kind='shard', start=0, end=4, width=4), result], 7)
    with pytest.raises(ValueError, match='Missing/overlapping'):
        check.coverage([dict(kind='shard', start=0, end=3, width=4), result], 7)
    with pytest.raises(ValueError, match='Cannot prove'):
        check.layout(NS(model=NS(named_modules=lambda: [])), 4, 7, 1, 2)


def test_hook_install_is_opt_in_idempotent_and_reversible(tmp_path):
    p = tmp_path/'runner.py'
    original = b'class NPUARModelRunner:\n    pass\n'
    p.write_bytes(original)
    check.edit_entry(p)
    installed = p.read_bytes()
    assert b'if _hgc_os.environ.get(' in installed
    check.edit_entry(p)
    assert p.read_bytes() == installed
    check.edit_entry(p, remove=True)
    assert p.read_bytes() == original


def sampling(output_tokens=3):
    return NS(temperature=0.0, top_p=1.0, top_k=-1, n=1, repetition_penalty=1.0,
              frequency_penalty=0.0, presence_penalty=0.0, ignore_eos=True,
              max_tokens=output_tokens, min_tokens=output_tokens, stop=[], stop_token_ids=[])


@pytest.mark.parametrize('top_k', [0, -1])
@pytest.mark.parametrize('modality', ['text', 'audio', 'audio_missing'])
def test_runtime_captures_only_generated_rows_before_mutation_and_counts_replay(tmp_path, monkeypatch, top_k, modality):
    for key, val in check.OFF.items():
        monkeypatch.setenv(key, val)
    case = dict(output_tokens=3, logits_steps=2, rows=[dict(sample_id='sample_00', prompt_token_ids=[1, 2])])
    rid = 'req'
    if modality != 'text':
        case.update(input_modality='audio', input_token_budget=2048)
        case['rows'][0].update(request_key='audio-test', audio=dict(pcm_sha256='pcm'))
        rid = 'chatcmpl-audio-test-native-suffix'
    check.write_json(tmp_path/'case.json', case)
    monkeypatch.setenv(check.ENV, str(tmp_path/'case.json'))
    monkeypatch.setenv(check.SIDE, 'graph')
    group_module = ModuleType('vllm.distributed.parallel_state')
    group_module.get_tp_group = lambda: NS(rank_in_group=0, ranks=[0, 1, 2, 3])
    monkeypatch.setitem(sys.modules, 'vllm.distributed.parallel_state', group_module)

    class Graph:
        def replay(self):
            return 'replayed'

    npu = NS(NPUGraph=Graph, is_current_stream_capturing=lambda: False)
    monkeypatch.setattr(torch, 'npu', npu, raising=False)
    monkeypatch.setitem(sys.modules, 'torch_npu', NS(npu=npu))

    class Model(torch.nn.Module):
        def compute_logits(self, hidden_states):
            return hidden_states

        def embed_multimodal(self, **kwargs):
            return [kwargs['audio_features'][0].T * 2]

    class Sampler(torch.nn.Module):
        @staticmethod
        def greedy_sample(logits):
            return logits.argmax(-1)

        def forward(self, logits, sampling_metadata):
            logits[0, 1] += 0.5
            return NS(sampled_token_ids=self.greedy_sample(logits).reshape(1, 1))

    config = NS(model_config=NS(enforce_eager=False, get_vocab_size=lambda: 4),
                cache_config=NS(enable_prefix_caching=False),
                scheduler_config=NS(async_scheduling=False), parallel_config=NS(),
                compilation_config=NS(), speculative_config=None)
    req = NS(prompt_token_ids=[1, 2], output_token_ids=[], sampling_params=sampling())
    req.sampling_params.top_k = top_k

    class Runner:
        def __init__(self):
            self.vllm_config = config
            self.model, self.sampler = Model(), Sampler()
            self.requests = {rid: req}
            self.input_batch = NS(req_ids=[rid])

        def execute_model(self, scheduler_output):
            if modality == 'audio' and not req.output_token_ids:
                self.model.embed_multimodal(audio_features=[torch.ones(40, 8)], audio_feature_lens=[torch.tensor([8])])
            Graph().replay()
            logits = self.model.compute_logits(torch.tensor([[0.0, 1.0, 3.0, -1.0]]))
            result = self.sampler(logits, NS(all_greedy=True, all_random=False))
            req.output_token_ids.append(int(result.sampled_token_ids[0, 0]))
            return result

    check.install_runtime(Runner)
    runner = Runner()
    if modality == 'audio_missing':
        with pytest.raises(ValueError, match='No audio encoder execution'):
            runner.execute_model(None)
        assert (tmp_path/'graph/tp0/error.json').exists()
        return
    for _ in range(3):
        runner.execute_model(None)
    assert list(inspect.signature(runner.model.compute_logits).parameters) == ['hidden_states']
    assert list(inspect.signature(runner.sampler.greedy_sample).parameters) == ['logits']
    folder = tmp_path/'graph/tp0'
    report = check.read_json(folder/'sample_00.json')
    assert report['tokens'] == [2, 2, 2] and report['captured'] == 2 and report['graph_replays'] == 3
    if modality == 'audio':
        assert report['prompt_token_ids'] == [1, 2] and report['native_request_id'] == rid
        assert len(report['audio_calls']) == 1
        assert report['audio_calls'][0]['inputs']['audio_features'][0]['shape'] == [40, 8]
        assert report['audio_calls'][0]['outputs'][0]['shape'] == [8, 40]
    raw = np.load(folder/'sample_00/000.npy')
    assert check.read_json(folder/'sample_00/000.json')['sampling']['top_k'] == top_k
    np.testing.assert_array_equal(check.decode(raw, 'torch.float32'), [0, 1, 3, -1])
    assert not (folder/'sample_00/002.json').exists()
    assert check.read_json(folder/'runtime.json')['determinism']['torch_deterministic_algorithms'] is False
    req.output_token_ids.append(0)
    with pytest.raises(ValueError, match='Stale/asynchronous'):
        runner.execute_model(None)
    assert (folder/'error.json').exists()


def test_greedy_params_fail_if_actual_runtime_sampling_changes():
    params = sampling()
    check.validate_sampling(params, NS(all_greedy=True, all_random=False), 3)
    params.temperature = 1.0
    with pytest.raises(ValueError, match='SamplingParams.*temperature.*expected.*0.0.*actual.*1.0'):
        check.validate_sampling(params, NS(all_greedy=True, all_random=False), 3)


@pytest.mark.parametrize('top_k', [0, -1])
def test_normalized_greedy_sampling_preserves_observed_top_k(top_k):
    params = sampling(3072)
    params.top_k = top_k
    observed = check.validate_sampling(params, NS(all_greedy=True, all_random=False), 3072)
    assert observed['top_k'] == top_k
    assert observed['max_tokens'] == 3072 and params.top_k == top_k


@pytest.mark.parametrize('top_k', [20, 1, None, False, '0'])
def test_greedy_does_not_accept_other_top_k_values(top_k):
    params = sampling()
    params.top_k = top_k
    with pytest.raises(ValueError, match='Unexpected greedy top_k'):
        check.validate_sampling(params, NS(all_greedy=True, all_random=False), 3)


def test_top_k_zero_does_not_bypass_real_greedy_or_length_checks():
    params = sampling()
    params.top_k = 0
    with pytest.raises(ValueError, match='not all_greedy'):
        check.validate_sampling(params, NS(all_greedy=False, all_random=True), 3)
    params.min_tokens = 0
    with pytest.raises(ValueError, match='fixed-length'):
        check.validate_sampling(params, NS(all_greedy=True, all_random=False), 3)


@pytest.fixture(scope='module', params=['text', 'audio'])
def comparison_case(tmp_path_factory, request):
    root = tmp_path_factory.mktemp('graph_logits_compare')
    case = dict(output_tokens=101, logits_steps=100, rows=[dict(sample_id=f'sample_{i:02d}', prompt_token_ids=[i+1]) for i in range(5)])
    is_audio = request.param == 'audio'
    if is_audio:
        case.update(input_modality='audio', input_token_budget=2048)
        for row in case['rows']:
            row.update(request_key=row['sample_id'], wav_sha256=row['sample_id'],
                       audio=dict(pcm_sha256=row['sample_id'], duration_s=2.0))
    check.write_json(root/'case.json', case)
    info, raw = check.capture(torch.tensor([[0.0, 1.0, 2.0, 4.0]]))
    for side in ('eager', 'graph'):
        client = dict(complete=True, case_hash=check.digest(case), rows=[dict(sample_id=r['sample_id'], message={'content': 'answer'}) for r in case['rows']])
        if is_audio:
            for saved, row in zip(client['rows'], case['rows']):
                saved.update(wav_sha256=row['wav_sha256'], request_key=row['request_key'],
                             response_id='chatcmpl-'+row['request_key'], usage=dict(prompt_tokens=1))
        check.write_json(root/side/'client.json', client)
        for rank in range(4):
            folder = root/side/f'tp{rank}'
            runtime = dict(case_hash=check.digest(case), side=side, tp_rank=rank, tp_size=4,
                determinism=dict(environment=check.OFF, torch_deterministic_algorithms=False, batch_invariant_registered=False),
                versions={}, vocab_size=4, observer_sha256='same', model_compute_logits={}, sampler={}, greedy_sample={},
                config={'cache_config': {'enable_prefix_caching': 'False'},
                        'model_config': {'enforce_eager': str(side == 'eager')}, 'compilation_config': {}})
            if is_audio:
                runtime['audio_embedder'] = {'name': 'same'}
            check.write_json(folder/'runtime.json', runtime)
            for row in case['rows']:
                sample = row['sample_id']
                captured = dict(status='complete', case_hash=check.digest(case), captured=100,
                        tokens=[3]*101, graph_replays=101 if side == 'graph' else 0, replays_before=0)
                if is_audio:
                    captured.update(prompt_token_ids=row['prompt_token_ids'], prompt_hash=check.digest(row['prompt_token_ids']),
                                    audio_pcm_sha256=row['audio']['pcm_sha256'], native_request_id='chatcmpl-'+row['request_key'],
                                    audio_calls=[dict(inputs=dict(audio_features=check.tensor_tree(torch.ones(40, 8))),
                                                      outputs=check.tensor_tree(torch.ones(2, 10)))])
                check.write_json(folder/(sample+'.json'), captured)
                (folder/sample).mkdir()
                np.save(folder/sample/'data.npy', raw)
                for step in range(100):
                    record = dict(case_hash=check.digest(case), step=step, sample_id=sample,
                        prompt_hash=check.digest(row['prompt_token_ids']), prefix_hash=check.digest([3]*step), token_id=3,
                        tensor=info, layout=dict(kind='full', start=0, end=4, width=4), file='data.npy', graph_replays=step+1 if side == 'graph' else 0)
                    check.write_json(folder/sample/f'{step:03d}.json', record)
    return root


def test_complete_comparison_and_no_false_success_on_missing_worker(comparison_case):
    root = comparison_case
    report = check.compare(root)
    assert report['all_output_equal'] and report['all_logits_first100_bitwise_equal']
    if report['input_modality'] == 'audio':
        assert all(r['audio_input_verified'] and r['audio_features_bitwise_equal']
                   and r['audio_embeddings_bitwise_equal'] for r in report['rows'])
    path = root/'graph/tp3/sample_00.json'
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            check.compare(root)
    finally:
        path.write_bytes(original)


def test_comparison_detects_single_bit_change_with_identical_output(comparison_case):
    root = comparison_case
    path = root/'graph/tp0/sample_00/000.json'
    original = path.read_bytes()
    record = json.loads(original)
    raw = np.load(path.parent/'data.npy').copy()
    raw.reshape(-1)[0] ^= 1
    np.save(path.parent/'perturbed.npy', raw)
    record['file'] = 'perturbed.npy'
    record['tensor']['sha256'] = check.sha(raw.tobytes())
    check.write_json(path, record)
    try:
        report = check.compare(root)
        assert report['all_output_equal'] and not report['all_logits_first100_bitwise_equal']
        assert report['rows'][0]['first_different_logits_position'] == 1
        assert report['rows'][0]['same_context_metrics']['finite_max_abs'] > 0
    finally:
        path.write_bytes(original)


def test_no_graph_replay_is_not_a_valid_graph_comparison(comparison_case):
    root = comparison_case
    path = root/'graph/tp0/sample_00.json'
    original = path.read_bytes()
    record = json.loads(original)
    record['graph_replays'] = 0
    check.write_json(path, record)
    try:
        with pytest.raises(ValueError, match='graph replay'):
            check.compare(root)
    finally:
        path.write_bytes(original)


def test_audio_or_prefill_only_replay_does_not_prove_decode_graph(comparison_case):
    root = comparison_case
    backups = {}
    try:
        for rank in range(4):
            for step in range(100):
                path = root/'graph'/f'tp{rank}/sample_00'/f'{step:03d}.json'
                backups[path] = path.read_bytes()
                record = check.read_json(path)
                record['graph_replays'] = 1
                check.write_json(path, record)
        with pytest.raises(ValueError, match='No graph replay during decode'):
            check.compare(root)
    finally:
        for path, data in backups.items():
            path.write_bytes(data)


def test_prefix_divergence_excludes_later_logits_from_same_context_metrics(comparison_case):
    root = comparison_case
    backups = {}
    def replace(path, value):
        backups[path] = path.read_bytes()
        check.write_json(path, value)
    try:
        for rank in range(4):
            folder = root/'graph'/f'tp{rank}'
            path = folder/'sample_00.json'
            result = check.read_json(path)
            result['tokens'][0] = 2
            replace(path, result)
            for step in range(100):
                path = folder/'sample_00'/f'{step:03d}.json'
                rec = check.read_json(path)
                rec['prefix_hash'] = check.digest(result['tokens'][:step])
                rec['token_id'] = result['tokens'][step]
                replace(path, rec)
        report = check.compare(root)
        assert not report['all_output_equal']
        first = report['rows'][0]
        assert first['first_different_output_position'] == 1
        assert first['same_context_logits_positions'] == 1
        assert first['context_diverged_positions_excluded'] == 99
    finally:
        for path, data in backups.items():
            path.write_bytes(data)


def test_client_sends_exactly_five_greedy_requests_without_probe(tmp_path, monkeypatch):
    rows = [dict(sample_id=f'sample_{i:02d}', messages=[{'role': 'user', 'content': str(i)}]) for i in range(5)]
    case = dict(rows=rows, input_tokens=2, output_tokens=100)
    check.write_json(tmp_path/'case.json', case)
    calls = []
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json, timeout):
            index = len(calls)
            assert json['temperature'] == 0 and json['top_p'] == 1 and json['top_k'] == -1 and json['n'] == 1
            assert json['repetition_penalty'] == 1 and json['stream'] is False
            calls.append(json)
            check.write_json(tmp_path/'eager/tp0'/f'sample_{index:02d}.json',
                dict(status='complete', captured=100, case_hash=check.digest(case)))
            return NS(raise_for_status=lambda: None, json=lambda: dict(
                choices=[dict(finish_reason='length', message=dict(content='answer'))],
                usage=dict(prompt_tokens=2, completion_tokens=100)))
    monkeypatch.setitem(sys.modules, 'requests', NS(Session=Session))
    check.request(NS(case_dir=tmp_path, side='eager', served_model='model', url='http://unused', timeout=10))
    assert len(calls) == 5
    assert check.read_json(tmp_path/'eager/client.json')['complete']


@pytest.mark.parametrize('input_tokens,output_tokens,budget', [
    (2048, 3072, 16384),  # Reported 32K-context / 16K-budget startup failure.
    (12288, 8192, 16384),  # Longer cases must also fit without chunking.
    (2048, 3072, None),
])
def test_case_scheduler_covers_full_context_in_both_platform_layers(input_tokens, output_tokens, budget):
    original = dict(pipeline='minicpmo_5_0_thinker_only',
                    stages=[dict(stage_id=0, max_model_len=32768,
                                 max_num_batched_tokens=32768, max_num_seqs=16)],
                    platforms=dict(npu=dict(stages=[dict(stage_id=0, max_num_batched_tokens=budget)])))
    before = copy.deepcopy(original)
    pair = check.build_case_pair(original, input_tokens, output_tokens)
    assert original == before
    for cfg in pair.values():
        for stage in (cfg['stages'][0], cfg['platforms']['npu']['stages'][0]):
            assert stage['enable_chunked_prefill'] is False
            assert stage['max_model_len'] == input_tokens + output_tokens
            assert stage['max_num_batched_tokens'] >= stage['max_model_len']
            assert stage['max_num_batched_tokens'] == max(input_tokens + output_tokens, budget or 0)
            assert stage['default_sampling_params']['max_tokens'] == output_tokens
    # Eager and graph must retain exactly the same scheduling and workload.
    eager = copy.deepcopy(pair['eager'])
    for stage in (eager['stages'][0], eager['platforms']['npu']['stages'][0]):
        stage['enforce_eager'] = False
        stage['compilation_config']['cudagraph_mode'] = 'PIECEWISE'
    assert eager == pair['graph']


def test_case_scheduler_rejects_hidden_engine_override():
    original = dict(pipeline='minicpmo_5_0_thinker_only', stages=[dict(stage_id=0)],
                    platforms=dict(npu=dict(stages=[dict(stage_id=0, engine_extras=dict(max_model_len=32768))])))
    with pytest.raises(ValueError, match='Conflicting scheduler'):
        check.build_case_pair(original, 2048, 3072)


def test_prepare_uses_exactly_five_distinct_inputs_and_both_modes(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(messages[1]['content'].split())+7))
    # Unique token contents as well as exact length, unlike a pure length mock.
    class UniqueTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            ids[0] = sum(map(ord, messages[1]['content'][:40]))
            return ids
    monkeypatch.setitem(sys.modules, 'transformers', NS(AutoTokenizer=NS(from_pretrained=lambda *a, **kw: UniqueTokenizer())))
    source = tmp_path/'deploy.yaml'
    source.write_text('pipeline: minicpmo_5_0_thinker_only\nstages:\n- stage_id: 0\nplatforms:\n  npu:\n    stages:\n    - stage_id: 0\n')
    args = NS(case_dir=tmp_path/'case', model=str(tmp_path/'model'), source=source, input_tokens=128, output_tokens=100)
    check.prepare(args)
    case = check.read_json(args.case_dir/'case.json')
    assert len(case['rows']) == 5 and case['logits_steps'] == 100
    import yaml
    for side in ('eager', 'graph'):
        cfg = yaml.safe_load((args.case_dir/(side+'.yaml')).read_text())
        for stage in (cfg['stages'][0], cfg['platforms']['npu']['stages'][0]):
            assert stage['enforce_eager'] == (side == 'eager')
            assert stage['enable_prefix_caching'] is False and stage['async_scheduling'] is False
            assert stage['tensor_parallel_size'] == 4 and stage['default_sampling_params']['temperature'] == 0
            assert stage['max_model_len'] == 228 and stage['max_num_batched_tokens'] >= 228


@pytest.fixture
def audio_case_dir(tmp_path, monkeypatch):
    import wave
    monkeypatch.syspath_prepend(str(ROOT))
    source = tmp_path/'deploy.yaml'
    source.write_text('pipeline: minicpmo_5_0_thinker_only\nstages:\n- stage_id: 0\nplatforms:\n  npu:\n    stages:\n    - stage_id: 0\n')
    audio_dir = tmp_path/'wav'
    audio_dir.mkdir()
    rows = []
    for i in range(5):
        p = audio_dir/f'{i}.wav'
        with wave.open(str(p), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(np.full(16000+i*160, i, dtype='<i2').tobytes())
        rows.append(dict(audios=[dict(wav_path=p.name)]))
    (audio_dir/'manifest.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    def forbidden_scan(*args, **kwargs):
        raise AssertionError('Do not recursively list indexed audio')
    monkeypatch.setattr(Path, 'rglob', forbidden_scan)
    check.main(['prepare-audio', '--case-dir', str(tmp_path/'case'), '--model', str(tmp_path/'model'),
                '--source', str(source), '--audio-dir', str(audio_dir), '--output-tokens', '100'])
    return tmp_path/'case'


def test_audio_prepare_snapshots_and_payload_keep_pcm_and_greedy_controls(audio_case_dir):
    import base64
    import yaml
    import wave
    root = audio_case_dir
    case = check.read_json(root/'case.json')
    assert len(case['rows']) == len({r['audio']['pcm_sha256'] for r in case['rows']}) == 5
    assert case['input_modality'] == 'audio' and 'input_tokens' not in case
    for row in case['rows']:
        snapshot = root/row['audio_file']
        with wave.open(str(snapshot), 'rb') as wav:
            assert check.sha(wav.readframes(wav.getnframes())) == row['audio']['pcm_sha256']
        payload = check.request_payload(root, case, row, 'qwen3.8')
        assert payload['request_id'] == row['request_key']
        assert payload['temperature'] == 0 and payload['min_tokens'] == payload['max_tokens'] == 100
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        part = payload['messages'][1]['content'][0]
        assert base64.b64decode(part['audio_url']['url'].split(',', 1)[1]) == snapshot.read_bytes()
    for side in ('eager', 'graph'):
        cfg = yaml.safe_load((root/(side+'.yaml')).read_text())
        for stage in (cfg['stages'][0], cfg['platforms']['npu']['stages'][0]):
            assert stage['tensor_parallel_size'] == 4 and stage['enable_prefix_caching'] is False
            assert stage['mm_processor_cache_gb'] == 0 and stage['enforce_eager'] == (side == 'eager')
            assert stage['max_model_len'] == 2148 and stage['default_sampling_params']['temperature'] == 0
    (root/row['audio_file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='WAV snapshot changed'):
        check.request_payload(root, case, row, 'qwen3.8')


def test_audio_request_mapping_does_not_confuse_identical_placeholder_tokens():
    case = dict(input_modality='audio', rows=[dict(request_key='one'), dict(request_key='two')])
    assert check.case_row(case, [8, 8], 'chatcmpl-one-native')['request_key'] == 'one'
    assert check.case_row(case, [8, 8], 'chatcmpl-two-native')['request_key'] == 'two'
    with pytest.raises(ValueError, match='request'):
        check.case_row(case, [8, 8], 'chatcmpl-unknown')
    assert not check.matches_request_id('chatcmpl-onex', 'one')


def test_audio_client_sends_five_requests_and_checks_variable_native_input_length(audio_case_dir, monkeypatch):
    root = audio_case_dir
    case = check.read_json(root/'case.json')
    calls = []
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json, timeout):
            index = len(calls)
            row = case['rows'][index]
            calls.append(json)
            prompt = [7]*(index+3)
            check.write_json(root/'eager/tp0'/(row['sample_id']+'.json'),
                             dict(status='complete', captured=100, case_hash=check.digest(case),
                                  prompt_token_ids=prompt, audio_calls=[{'observed': True}]))
            return NS(raise_for_status=lambda: None, json=lambda: dict(id='chatcmpl-'+row['request_key'],
                      choices=[dict(finish_reason='length', message=dict(content='answer'))],
                      usage=dict(prompt_tokens=len(prompt), completion_tokens=100)))
    monkeypatch.setitem(sys.modules, 'requests', NS(Session=Session))
    check.request(NS(case_dir=root, side='eager', served_model='model', url='http://unused', timeout=10))
    assert len(calls) == 5
    report = check.read_json(root/'eager/client.json')
    assert report['complete'] and [x['usage']['prompt_tokens'] for x in report['rows']] == [3, 4, 5, 6, 7]
    assert all(x['modalities'] == ['text'] and x['n'] == 1 for x in calls)


def test_audio_evidence_rejects_wrong_input_but_reports_encoder_numeric_difference():
    case = dict(input_token_budget=2048)
    row = dict(request_key='sample', wav_sha256='wav', audio=dict(pcm_sha256='pcm', duration_s=2))
    native = dict(prompt_token_ids=[1, 2], prompt_hash=check.digest([1, 2]),
                  audio_pcm_sha256='pcm', native_request_id='chatcmpl-sample-native',
                  audio_calls=[dict(inputs=dict(audio_features=check.tensor_tree(torch.ones(40, 8))),
                                    outputs=check.tensor_tree(torch.ones(2, 10)))])
    results = {side: [copy.deepcopy(native) for _ in range(4)] for side in ('eager', 'graph')}
    clients = [dict(wav_sha256='wav', request_key='sample', response_id='chatcmpl-sample', usage=dict(prompt_tokens=2))]*2
    _, evidence = check.audio_evidence(case, row, results, clients)
    assert evidence['audio_embeddings_bitwise_equal']
    results['graph'][1]['audio_calls'][0]['outputs']['sha256'] = 'different'
    _, evidence = check.audio_evidence(case, row, results, clients)
    assert evidence['audio_features_bitwise_equal'] and not evidence['audio_embeddings_bitwise_equal']
    results['graph'][1]['prompt_token_ids'][0] = 3
    with pytest.raises(ValueError, match='Expanded audio prompt IDs differ'):
        check.audio_evidence(case, row, results, clients)
