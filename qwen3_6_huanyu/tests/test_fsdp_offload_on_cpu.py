"""CPU regression for host reads before an asynchronous offload completes.

The transfer queue is simulated; the re-padding is PyTorch's real FSDPParam
implementation. This catches the ordering failure without claiming to execute
an NPU transfer or a distributed training job.
"""

from contextlib import redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_fsdp_offload_cpu_tests", ROOT / "fsdp_offload.py")
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class SpyModel:
    def __init__(self):
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self


class FSDPSpyModel(SpyModel):
    def _get_fsdp_state(self):
        return SimpleNamespace()


class QueuedOffloadModel:
    """Model the FSDPModule._apply order using the real reset implementation."""

    def __init__(self, full_tensor, rank, world_size=16):
        self.source = full_tensor
        width = (full_tensor.numel() + world_size - 1) // world_size
        self.local_source = full_tensor[rank * width:(rank + 1) * width].clone()
        mesh = SimpleNamespace(device_type="cpu", ndim=1)
        spec = DTensorSpec(mesh, (Shard(0),), tensor_meta=TensorMeta(full_tensor.size(), (1,), full_tensor.dtype))
        tensor = DTensor(self.local_source, spec, requires_grad=False)
        # Bypass FSDP construction/collectives; only the post-conversion reset
        # needs exercising. The fields below are its actual required state.
        state = FSDPParam.__new__(FSDPParam)
        state.sharded_param = tensor
        state._module_info = SimpleNamespace(module=SimpleNamespace(weight=tensor), param_name="weight")
        state._sharded_param_data = torch.zeros(width)
        state.padded_sharded_param_size = torch.Size([width])
        state.fsdp_placement = Shard(0)
        state.pin_memory = False
        self.param = state
        self.pending = []

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        assert device.type == "cpu"
        assert dtype is None
        destination = torch.full_like(self.local_source, -555.0)
        if non_blocking:
            self.pending.append(lambda: destination.copy_(self.local_source))
        else:
            destination.copy_(self.local_source)
        self.param.sharded_param._local_tensor = destination
        # Same ordering as FSDPModule._apply: convert first, then reset/re-pad.
        self.param.reset_sharded_param()
        return self

    def finish_transfers(self):
        for transfer in self.pending:
            transfer()
        self.pending.clear()


class FSDPQueuedOffloadModel(QueuedOffloadModel):
    def _get_fsdp_state(self):
        return SimpleNamespace()


class OffloadTests(unittest.TestCase):
    def install(self, module):
        with redirect_stdout(io.StringIO()):
            self.assertIs(policy.install_huanyu_fsdp_offload(module), module)
        return module

    def test_40_over_16_only_partial_tail_is_corrupted_without_guard(self):
        full = torch.arange(40, dtype=torch.float32) + 0.25
        models = [FSDPQueuedOffloadModel(full, rank) for rank in range(16)]
        for model in models:
            model.to("cpu", non_blocking=True)
            model.finish_transfers()  # Too late to repair the CPU-to-CPU copy.
        actual = torch.cat([model.param.sharded_param._local_tensor for model in models])
        torch.testing.assert_close(actual[:39], full[:39], rtol=0, atol=0)
        self.assertEqual(actual[39].item(), -555.0)
        self.assertEqual(models[13].param.sharded_param._local_tensor.numel(), 1)

    def test_guard_preserves_all_40_values_including_partial_and_empty_ranks(self):
        full = torch.arange(40, dtype=torch.float32) + 0.25
        models = [self.install(QueuedOffloadModel(full, rank)) for rank in range(16)]
        with redirect_stdout(io.StringIO()):
            for model in models:
                # fully_shard changes the class of the same object after the
                # adapter has installed the instance method.
                model.__class__ = FSDPQueuedOffloadModel
                self.assertIs(model.to("cpu", non_blocking=True), model)
                self.assertFalse(model.pending)
        actual = torch.cat([model.param.sharded_param._local_tensor for model in models])
        torch.testing.assert_close(actual, full, rtol=0, atol=0)

    def test_even_chunks_20_over_16_are_not_affected_by_queued_transfer(self):
        full = torch.arange(20, dtype=torch.float32) + 0.5
        models = [FSDPQueuedOffloadModel(full, rank) for rank in range(16)]
        for model in models:
            model.to("cpu", non_blocking=True)
            model.finish_transfers()
        actual = torch.cat([model.param.sharded_param._local_tensor for model in models])
        torch.testing.assert_close(actual, full, rtol=0, atol=0)

    def test_non_fsdp_and_other_models_keep_original_call(self):
        marked = self.install(SpyModel())
        unmarked = FSDPSpyModel()
        for model in (marked, unmarked):
            model.to("cpu", non_blocking=True)
            self.assertEqual(model.calls, [(("cpu",), {"non_blocking": True})])

    def test_non_host_dtype_only_and_blocking_calls_are_unchanged(self):
        model = self.install(FSDPSpyModel())
        for args, kwargs in (
            (("cuda:0",), {"non_blocking": True}),
            ((torch.float64,), {"non_blocking": True}),
            (("cpu",), {"non_blocking": False}),
        ):
            model.to(*args, **kwargs)
            self.assertEqual(model.calls[-1], (args, kwargs))
        self.assertEqual(getattr(model, policy._COUNT), 0)

    def test_host_conversion_keeps_dtype_and_memory_format(self):
        model = self.install(FSDPSpyModel())
        with redirect_stdout(io.StringIO()):
            model.to("cpu", torch.float64, True, memory_format=torch.channels_last)
        self.assertEqual(model.calls[-1], ((), dict(device=torch.device("cpu"), dtype=torch.float64,
                                                   non_blocking=False, memory_format=torch.channels_last)))

    def test_tensor_overload_and_logging_are_bounded(self):
        model = self.install(FSDPSpyModel())
        with redirect_stdout(io.StringIO()) as output:
            for _ in range(3):
                model.to(torch.ones(1, dtype=torch.float64), non_blocking=True)
        self.assertEqual(output.getvalue().count("phase=executed"), 1)
        self.assertIn("reason=host_read_before_shard_repad", output.getvalue())
        self.assertEqual(getattr(model, policy._COUNT), 3)
        self.assertEqual(model.calls[-1][1]["dtype"], torch.float64)

    def test_install_is_idempotent_and_deepcopy_rebinds_original_to(self):
        model = self.install(FSDPSpyModel())
        method = model.to
        self.install(model)
        self.assertIs(model.to, method)
        clone = copy.deepcopy(model)
        with redirect_stdout(io.StringIO()):
            clone.to("cpu", non_blocking=True)
        self.assertFalse(model.calls)
        self.assertEqual(len(clone.calls), 1)


if __name__ == "__main__":
    unittest.main()
