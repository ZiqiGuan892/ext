"""Padded memory bounds, sample preservation and coordinated DP refinement."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_padded_batching_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.padded_batching", ROOT / "padded_batching.py")
batching = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batching
spec.loader.exec_module(batching)


def batch(lengths, budget=10, **flags):
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    ids = torch.nested.nested_tensor_from_jagged(torch.arange(sum(lengths)), offsets)
    return dict(input_ids=ids, sample_ids=list(range(len(lengths))),
                max_token_len_per_gpu=budget, use_dynamic_bsz=True, use_remove_padding=False, **flags)


def select(data, indices):
    rows = data["input_ids"].unbind()
    result = batch([len(rows[i]) for i in indices], data["max_token_len_per_gpu"])
    result["sample_ids"] = [data["sample_ids"][i] for i in indices]
    return result


class Planner:
    """Small independent valid-token planner with the upstream helper contract."""

    def __init__(self, synchronize=False):
        self.calls = []
        self.synchronize = synchronize

    def prepare(self, data, dp_group=None, num_batches_divided_by=None,
                same_micro_num_in_dp=True, min_num_micro_batch=None, use_dynamic_bsz_balance=True):
        self.calls.append(min_num_micro_batch)
        lengths = data["input_ids"].offsets().diff().tolist()
        group_size = data.get("force_group_size", 1)
        groups = [list(range(i, i + group_size)) for i in range(0, len(lengths), group_size)]
        budget = data["max_token_len_per_gpu"] * data.get("sp_size", 1)
        if max(lengths) > budget:
            raise ValueError("single sequence exceeds budget")
        count = max((sum(lengths) + budget - 1) // budget, min_num_micro_batch or 1)
        if self.synchronize and same_micro_num_in_dp and dp_group is not None:
            value = torch.tensor([count])
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX, group=dp_group)
            count = value.item()
        divisor = num_batches_divided_by or 1
        count = ((count + divisor - 1) // divisor) * divisor
        while count <= len(groups):
            partitions = [[] for _ in range(count)]
            totals = [0] * count
            for group in sorted(groups, key=lambda g: sum(lengths[i] for i in g), reverse=True):
                target = min(range(count), key=totals.__getitem__)
                partitions[target].extend(group)
                totals[target] += sum(lengths[i] for i in group)
            if max(totals) <= budget:
                return [select(data, indices) for indices in partitions], partitions
            count += divisor
        raise ValueError("valid-token budget cannot fit groups")


IMPL = SimpleNamespace(
    tu=SimpleNamespace(get_non_tensor_data=lambda data, key, default: data.get(key, default)),
    get_device_name=lambda: "cpu",
)


@contextmanager
def active():
    token = batching._ACTIVE.set(True)
    try:
        yield
    finally:
        batching._ACTIVE.reset(token)


def run(data, planner=None, **kwargs):
    planner = planner or Planner()
    with active():
        return batching._prepare(planner.prepare, IMPL, data, **kwargs)


def test_padding_exceeds_valid_budget_and_repartition_preserves_order_and_data():
    data = batch([7, 1, 1, 1])
    original, _ = Planner().prepare(data)
    assert batching._padded_tokens(original[0]) == 28
    before = dict(data)
    micro, indices = run(data)
    assert len(micro) == 2
    assert max(map(batching._padded_tokens, micro)) <= 10
    restored = [None] * 4
    for part, positions in zip(micro, indices, strict=True):
        for value, position in zip(part["sample_ids"], positions, strict=True):
            assert restored[position] is None
            restored[position] = value
    assert restored == list(range(4))
    assert data == before


@pytest.mark.parametrize("lengths", [[4, 4], [10], [1] * 31, [10, 1, 2, 3, 4, 5], [8, 1, 1, 8, 1, 1]])
def test_bounds_and_exact_coverage(lengths):
    micro, indices = run(batch(lengths))
    assert max(map(batching._padded_tokens, micro)) <= 10
    assert sorted(i for part in indices for i in part) == list(range(len(lengths)))


def test_divisibility_minimum_count_and_force_groups():
    micro, indices = run(batch([7, 1, 1, 1, 7, 1, 1, 1], 16, force_group_size=2),
                         num_batches_divided_by=2, min_num_micro_batch=2)
    assert len(micro) % 2 == 0
    assert max(map(batching._padded_tokens, micro)) <= 16
    for part in indices:
        for i in part:
            assert i ^ 1 in part


def test_impossible_padded_group_fails_without_dropping_or_truncating():
    data = batch([7, 1, 7, 1], force_group_size=2)
    with pytest.raises(ValueError, match="indivisible group"):
        run(data)
    assert data["input_ids"].offsets().diff().tolist() == [7, 1, 7, 1]


def test_single_sequence_budget_failure_is_not_hidden():
    with pytest.raises(ValueError, match="single sequence"):
        run(batch([11, 1]))


def test_refinement_preserves_weighted_token_gradient():
    lengths = [7, 1, 1, 1]
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    reference = parameter.detach().clone().requires_grad_()
    weights = torch.arange(1, 11, dtype=torch.float64)
    rows = weights.split(lengths)
    micro, indices = run(batch(lengths))
    for part in indices:
        loss = sum(((parameter * rows[i]).sin() * rows[i]).sum() for i in part) / weights.sum()
        loss.backward()
    (((reference * weights).sin() * weights).sum() / weights.sum()).backward()
    torch.testing.assert_close(parameter.grad, reference.grad)


class VirtualDP:
    """Threaded, bounded-time collective simulator; not an HCCL integration test."""

    def __init__(self):
        self.local = threading.local()
        self.values = {}
        self.barrier = threading.Barrier(2, timeout=5, action=self.reduce)

    def reduce(self):
        values = list(self.values.values())
        reduced = torch.stack([v.clone() for v in values]).amax(dim=0)
        for value in values:
            value.copy_(reduced)

    def all_reduce(self, value, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        self.values[self.local.rank] = value
        self.barrier.wait()
        # No rank can overwrite the shared slots until both consumed the result.
        self.barrier.wait()

    def worker(self, rank, data):
        self.local.rank = rank
        return run(data, Planner(synchronize=True), dp_group="dp")


@pytest.mark.parametrize("impossible", [False, True])
def test_all_ranks_refine_together_or_fail_together(monkeypatch, impossible):
    dp = VirtualDP()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", dp.all_reduce)
    inputs = [batch([7, 1, 1, 1]), batch([2, 2, 2, 2])]
    if impossible:
        # Rank 1 cannot match rank 0's necessary second microbatch.
        inputs[1] = batch([8])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(dp.worker, rank, data) for rank, data in enumerate(inputs)]
        if impossible:
            for job in jobs:
                with pytest.raises(ValueError, match="common DP microbatch count"):
                    job.result(timeout=10)
        else:
            results = [job.result(timeout=10) for job in jobs]
            assert [len(micro) for micro, _ in results] == [2, 2]
            for micro, indices in results:
                assert max(map(batching._padded_tokens, micro)) <= 10
                assert sorted(i for part in indices for i in part) == list(range(4))


def fake_engine(monkeypatch):
    impl = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.tu, impl.get_device_name = IMPL.tu, IMPL.get_device_name
    impl.prepare_micro_batches = Planner().prepare
    exec("class FSDPEngine:\n"
         "    def forward_backward_batch(self, data, loss_function=None, forward_only=False):\n"
         "        if data.get('fail'): raise ValueError('forward failed')\n"
         "        return prepare_micro_batches(data)\n", impl.__dict__)
    monkeypatch.setitem(sys.modules, impl.__name__, impl)
    return impl


def test_install_scope_peft_idempotence_and_exception_reset(monkeypatch):
    impl = fake_engine(monkeypatch)
    huanyu = SimpleNamespace(_huanyu_scalar_output_packing=True)
    batching.install_huanyu_padded_batching(huanyu)
    first = impl.FSDPEngine.forward_backward_batch
    batching.install_huanyu_padded_batching(huanyu)
    assert impl.FSDPEngine.forward_backward_batch is first
    engine = impl.FSDPEngine()
    engine.module = SimpleNamespace(get_base_model=lambda: huanyu)
    for forward_only in (True, False):
        micro, _ = engine.forward_backward_batch(batch([7, 1, 1, 1]), forward_only=forward_only)
        assert len(micro) == 2
        assert not batching._ACTIVE.get()
    with pytest.raises(ValueError, match="forward failed"):
        engine.forward_backward_batch(batch([7, 1, 1, 1], fail=True))
    assert not batching._ACTIVE.get()
    engine.module = SimpleNamespace()
    with active():
        # An unrelated nested engine must temporarily clear the parent's scope.
        micro, _ = engine.forward_backward_batch(batch([7, 1, 1, 1]))
        assert len(micro) == 1
        assert batching._ACTIVE.get()


@pytest.mark.parametrize("flag", ["use_dynamic_bsz", "use_remove_padding", "inactive"])
def test_other_paths_delegate_unchanged(flag):
    data = batch([7, 1, 1, 1])
    if flag != "inactive":
        data[flag] = flag == "use_remove_padding"
    sentinel = object()

    def original(value, **kwargs):
        assert value is data
        return sentinel

    token = batching._ACTIVE.set(flag != "inactive")
    try:
        assert batching._prepare(original, IMPL, data) is sentinel
    finally:
        batching._ACTIVE.reset(token)


def test_unknown_engine_contract_fails_before_patching(monkeypatch):
    impl = fake_engine(monkeypatch)
    original = impl.FSDPEngine.forward_backward_batch
    impl.prepare_micro_batches = lambda data: None
    with pytest.raises(RuntimeError, match="Unsupported prepare_micro_batches"):
        batching.install_huanyu_padded_batching(SimpleNamespace(_huanyu_scalar_output_packing=True))
    assert impl.FSDPEngine.forward_backward_batch is original


def test_outer_engine_wrappers_keep_scope_and_idempotence(monkeypatch):
    impl = fake_engine(monkeypatch)
    model = SimpleNamespace(_huanyu_scalar_output_packing=True)
    batching.install_huanyu_padded_batching(model)
    original = impl.FSDPEngine.forward_backward_batch
    calls = []

    @wraps(original)
    def external_wrapper(self, *args, **kwargs):
        calls.append("begin")
        try:
            return original(self, *args, **kwargs)
        finally:
            calls.append("end")

    impl.FSDPEngine.forward_backward_batch = external_wrapper
    batching.install_huanyu_padded_batching(model)
    assert impl.FSDPEngine.forward_backward_batch is external_wrapper
    engine = impl.FSDPEngine()
    engine.module = model
    micro, _ = engine.forward_backward_batch(batch([7, 1, 1, 1]))
    assert len(micro) == 2 and calls == ["begin", "end"]


def test_planner_ignoring_minimum_fails_instead_of_looping():
    planner = Planner()

    def ignore_minimum(data, **kwargs):
        kwargs["min_num_micro_batch"] = None
        return planner.prepare(data, **kwargs)

    with active(), pytest.raises(RuntimeError, match="ignored min_num_micro_batch"):
        batching._prepare(ignore_minimum, IMPL, batch([7, 1, 1, 1]))


def test_unexpected_dp_counts_fail_collectively(monkeypatch):
    dp = VirtualDP()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", dp.all_reduce)

    def worker(rank, data):
        dp.local.rank = rank
        # Simulate an incompatible upstream helper that ignores count sync.
        return run(data, Planner(synchronize=False), dp_group="dp")

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, rank, data) for rank, data in enumerate(
            [batch([7, 1, 1, 1]), batch([7, 7, 1, 1])])]
        for job in jobs:
            with pytest.raises(RuntimeError, match="different DP microbatch counts"):
                job.result(timeout=10)
