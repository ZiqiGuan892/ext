"""Real tensor checks for Huanyu-only hybrid KV cache hooks."""

from contextlib import contextmanager, redirect_stdout
from functools import wraps
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_file("huanyu_cache_precision_cpu", ROOT / "cache_precision.py")
shared = load_file("huanyu_shared_cache_precision_cpu", ROOT / "qwen_precision/cache_precision.py")


def model_config(huanyu=True):
    audio_type = "huanyu_conformer_0_6b" if huanyu else "whisper"
    # Names deliberately disagree with the architecture: do not select by path.
    return SimpleNamespace(model="/renamed/checkpoint" if huanyu else "Qwen3.6-huanyu",
                           hf_config=SimpleNamespace(audio_config={"model_type": audio_type}))


class AttentionSpec:
    def __init__(self, block_size=4, page_size_bytes=8):
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes


class MambaSpec:
    page_size_bytes = 4


class UniformTypeKVCacheSpecs:
    def __init__(self, specs):
        self.kv_cache_specs = specs


@contextmanager
def native_apis(*, missing_zero=False, missing_private=False):
    class Runner:
        def __init__(self, huanyu=True):
            self.vllm_config = SimpleNamespace(model_config=model_config(huanyu), kv_transfer_config=None)
            self.device = torch.device("cpu")
            self.use_sparse = self.use_compress = False
            self.native_zero_calls = []
            self.native_alloc_calls = []
            self.states = {}

        def _torch_zero_attention_block_ids(self, ids):
            self.native_zero_calls.append(ids)
            return "native-zero"

        def _allocate_kv_cache_tensors(self, config):
            self.native_alloc_calls.append(config)
            return "native-allocate"

        def _iter_kv_cache_states(self, name):
            return iter(self.states[name])

        def _get_layer_kv_cache_specs(self, config):
            return self.layer_specs

    class Worker:
        def __init__(self, runner):
            self.model_runner = runner
            self.vllm_config = runner.vllm_config
            self.model_config = runner.vllm_config.model_config
            self.native_budget_calls = []

        def _scale_kv_cache_memory_for_multi_group(self, available):
            self.native_budget_calls.append(available)
            return available - 1

        def get_kv_cache_spec(self):
            return self.model_runner.layer_specs

    if missing_zero:
        del Runner._torch_zero_attention_block_ids
    if missing_private:
        del Worker._scale_kv_cache_memory_for_multi_group
    cache_interface = ModuleType("vllm.v1.kv_cache_interface")
    cache_interface.AttentionSpec = AttentionSpec
    cache_interface.MambaSpec = MambaSpec
    cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    modules = {
        "vllm_omni.worker.gpu_model_runner": SimpleNamespace(OmniGPUModelRunner=Runner),
        "vllm_ascend.worker.worker": SimpleNamespace(NPUWorker=Worker),
        "vllm_ascend.worker.model_runner_v1": SimpleNamespace(NPUModelRunner=Runner),
        "vllm.v1.core.kv_cache_utils": SimpleNamespace(
            get_kv_cache_groups=lambda config, specs: [
                SimpleNamespace(kv_cache_spec=spec, layer_names=[name]) for name, spec in specs.items()
            ]),
        "vllm.v1.kv_cache_interface": cache_interface,
    }
    with patch.dict(sys.modules, modules), patch.object(policy, "_shared", return_value=shared):
        yield Runner, Worker


class CachePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE": "1",
            "MINICPMO_PRIVATE_HYBRID_CACHE": "0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        policy._REPORTED.clear()

    def test_execution_evidence_requires_successful_huanyu_zeroing_and_reports_once(self):
        with native_apis() as (Runner, _):
            with redirect_stdout(io.StringIO()) as startup:
                policy.install_huanyu_cache_precision()
                policy.install_huanyu_cache_precision()
            self.assertEqual(startup.getvalue().count("event=installed component=logical_zero"), 1)
            self.assertNotIn("event=executed", startup.getvalue())
            for field in ("host=", "pid=", "module=vllm_omni.worker.gpu_model_runner", "scope=huanyu"):
                self.assertIn(field, startup.getvalue())
            runner = Runner()
            with redirect_stdout(io.StringIO()) as skipped:
                Runner(False)._torch_zero_attention_block_ids([0])
                runner._torch_zero_attention_block_ids([])
                runner._torch_zero_attention_block_ids([0])  # No cache yet.
            self.assertNotIn("event=executed", skipped.getvalue())
            runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=AttentionSpec(), layer_names=["a"])
            ])
            state = torch.ones((4, 2, 1, 2))
            runner.states = {"a": [state]}
            with redirect_stdout(io.StringIO()) as failed:
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    runner._torch_zero_attention_block_ids([3])
            self.assertNotIn("event=executed", failed.getvalue())
            with redirect_stdout(io.StringIO()) as executed:
                runner._torch_zero_attention_block_ids([1])
                runner._torch_zero_attention_block_ids([1])
            self.assertEqual(executed.getvalue().count("event=executed component=logical_zero"), 1)
            self.assertIn("logical_blocks=1", executed.getvalue())
            self.assertIn("states=1 logical=4 physical=2 pages_per_block=2", executed.getvalue())
            expected = torch.ones_like(state)
            expected[2:4] = 0
            torch.testing.assert_close(state, expected)

    def test_actual_audio_config_selects_mapping_and_object_without_path_guess(self):
        self.assertTrue(policy.is_huanyu_policy(SimpleNamespace(model_config=model_config())))
        self.assertTrue(policy.is_huanyu_policy({"vllm_config": {"model_config": {
            "hf_config": {"audio_config": {"model_type": "huanyu_conformer_0_6b"}}
        }}}))
        self.assertFalse(policy.is_huanyu_policy(SimpleNamespace(model_config=model_config(False))))
        self.assertFalse(policy.is_huanyu_policy({"model_config": {"model": "Qwen3.6-huanyu"}}))
        self.assertFalse(policy.is_huanyu_policy(SimpleNamespace()))

    def test_separate_kv_expand_logical_pages_and_leave_mamba_untouched(self):
        with native_apis() as (Runner, _):
            policy.install_huanyu_cache_precision()
            runner = Runner()
            runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=AttentionSpec(), layer_names=["attention"]),
                SimpleNamespace(kv_cache_spec=MambaSpec(), layer_names=["mamba"]),
            ])
            key = torch.arange(8 * 2 * 3 * 4, dtype=torch.float32).reshape(8, 2, 3, 4) + 1
            value = key + 1000
            mamba = torch.full((4, 2, 3), 17.0)
            runner.states = {"attention": [key, value], "mamba": [mamba]}
            original_key, original_value = key.clone(), value.clone()
            runner._torch_zero_attention_block_ids([1, 3])
            for original, actual in ((original_key, key), (original_value, value)):
                original[[2, 3, 6, 7]] = 0
                torch.testing.assert_close(actual, original)
            torch.testing.assert_close(mamba, torch.full_like(mamba, 17.0))
            self.assertEqual(runner.native_zero_calls, [])

    def test_combined_kv_uses_page_axis_one_and_two_page_separate_kv_does_not(self):
        with native_apis() as (Runner, _):
            policy.install_huanyu_cache_precision()
            for shape, logical, ids, page_axis in (((2, 4, 2, 1, 3), 4, [1], 1),
                                                    ((2, 2, 1, 3), 2, [1], 0)):
                with self.subTest(shape=shape):
                    runner = Runner()
                    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                        SimpleNamespace(kv_cache_spec=AttentionSpec(logical), layer_names=["a"])
                    ])
                    state = torch.ones(shape)
                    runner.states = {"a": [state]}
                    expected = state.clone()
                    pages = torch.tensor([2, 3] if page_axis == 1 else [1])
                    expected.index_fill_(page_axis, pages, 0)
                    runner._torch_zero_attention_block_ids(ids)
                    torch.testing.assert_close(state, expected)

    def test_nonhuanyu_reward_uses_original_method_even_when_path_mentions_huanyu(self):
        with native_apis() as (Runner, _):
            policy.install_huanyu_cache_precision()
            runner = Runner(False)
            self.assertEqual(runner._torch_zero_attention_block_ids([3]), "native-zero")
            self.assertEqual(runner.native_zero_calls, [[3]])

    def test_invalid_block_geometry_and_indices_raise_instead_of_touching_wrong_pages(self):
        with native_apis() as (Runner, _):
            policy.install_huanyu_cache_precision()
            runner = Runner()
            state = torch.ones((4, 2, 1, 2))
            runner.states = {"a": [state]}
            for logical, ids, message in ((3, [0], "logical/physical"),
                                         (4, [2], "outside"), (4, [-1], "outside")):
                with self.subTest(logical=logical, ids=ids):
                    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                        SimpleNamespace(kv_cache_spec=AttentionSpec(logical), layer_names=["a"])
                    ])
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner._torch_zero_attention_block_ids(ids)
                    torch.testing.assert_close(state, torch.ones_like(state))

    def test_install_idempotence_and_optout_do_not_import_or_replace_native(self):
        with native_apis() as (Runner, _):
            original = Runner._torch_zero_attention_block_ids
            with patch.dict("os.environ", {"MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE": "0"}):
                with patch.object(policy.importlib, "import_module", side_effect=AssertionError("unexpected import")):
                    self.assertFalse(policy.install_huanyu_cache_precision())
                self.assertIs(Runner._torch_zero_attention_block_ids, original)
            self.assertTrue(policy.install_huanyu_cache_precision())
            installed = Runner._torch_zero_attention_block_ids
            self.assertTrue(policy.install_huanyu_cache_precision())
            self.assertIs(Runner._torch_zero_attention_block_ids, installed)

    def test_missing_or_noncallable_enabled_zero_hook_fails_without_claiming_success(self):
        with native_apis(missing_zero=True), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "does not expose a callable"):
                policy.install_huanyu_cache_precision()
        self.assertNotIn("installed", output.getvalue())
        with native_apis() as (Runner, _), redirect_stdout(io.StringIO()) as output:
            Runner._torch_zero_attention_block_ids = None
            with self.assertRaisesRegex(RuntimeError, "does not expose a callable"):
                policy.install_huanyu_cache_precision()
        self.assertNotIn("installed", output.getvalue())

    def test_explicit_private_mode_requires_native_api(self):
        with native_apis(missing_private=True), patch.dict("os.environ", {
            "MINICPMO_PRIVATE_HYBRID_CACHE": "1",
        }):
            with self.assertRaisesRegex(RuntimeError, "explicitly requests Huanyu private buffers"):
                policy.install_huanyu_cache_precision()

    def test_private_budget_allocation_and_reward_passthrough_are_paired(self):
        with native_apis() as (Runner, Worker), patch.dict("os.environ", {
            "MINICPMO_PRIVATE_HYBRID_CACHE": "1",
        }):
            with redirect_stdout(io.StringIO()) as startup:
                policy.install_huanyu_cache_precision()
            budget_hook, allocation_hook = Worker._scale_kv_cache_memory_for_multi_group, Runner._allocate_kv_cache_tensors
            policy.install_huanyu_cache_precision()
            self.assertIs(Worker._scale_kv_cache_memory_for_multi_group, budget_hook)
            self.assertIs(Runner._allocate_kv_cache_tensors, allocation_hook)
            runner = Runner()
            runner.layer_specs = {"attention": AttentionSpec(), "mamba": MambaSpec()}
            worker = Worker(runner)
            with redirect_stdout(io.StringIO()) as budget_output:
                self.assertEqual(worker._scale_kv_cache_memory_for_multi_group(120), 80)
                self.assertEqual(worker._scale_kv_cache_memory_for_multi_group(120), 80)
            self.assertEqual(budget_output.getvalue().count("event=executed component=private_budget"), 1)
            self.assertIn("available_bytes=120 planner_bytes=80", budget_output.getvalue())
            self.assertFalse(runner.supports_standardized_shared_kv_backing)
            self.assertTrue(runner._q36_private_cache_budget_applied)
            with redirect_stdout(io.StringIO()) as allocation_output:
                result = runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
                runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
            self.assertEqual(allocation_output.getvalue().count("event=executed component=private_buffers"), 1)
            self.assertIn("layers=2 blocks=10 allocated_bytes=120", allocation_output.getvalue())
            for component in ("private_budget", "private_buffers"):
                self.assertEqual(startup.getvalue().count(f"event=installed component={component}"), 1)
            self.assertEqual({name: value.numel() for name, value in result.items()}, {"attention": 80, "mamba": 40})
            self.assertNotEqual(result["attention"].data_ptr(), result["mamba"].data_ptr())
            self.assertTrue(all(value.dtype == torch.int8 and torch.count_nonzero(value) == 0 for value in result.values()))
            with self.assertRaisesRegex(RuntimeError, "budget is 120"):
                runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=11))
            rm = Runner(False)
            rm_worker = Worker(rm)
            self.assertEqual(rm_worker._scale_kv_cache_memory_for_multi_group(120), 119)
            self.assertEqual(rm._allocate_kv_cache_tensors("config"), "native-allocate")
            self.assertEqual(rm_worker.native_budget_calls, [120])
            self.assertEqual(rm.native_alloc_calls, ["config"])

    def test_zero_no_false_evidence_from_previous_call_or_partially_failed_call(self):
        with native_apis() as (Runner, _):
            policy.install_huanyu_cache_precision()
            runner = Runner()
            runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=AttentionSpec(), layer_names=["a"])
            ])
            runner.states = {"a": [torch.ones((4, 2, 1, 2))]}
            runner._torch_zero_attention_block_ids([1])
            policy._REPORTED.clear()
            with redirect_stdout(io.StringIO()) as empty:
                runner._torch_zero_attention_block_ids([])
                runner.states = {"a": []}
                runner._torch_zero_attention_block_ids([1])
            self.assertNotIn("event=executed", empty.getvalue())
            self.assertIsNone(runner._huanyu_zero_execution)
            runner.states = {"a": [torch.ones((4, 2, 1, 2)), torch.ones((1, 2, 1, 2))]}
            with redirect_stdout(io.StringIO()) as failed:
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    runner._torch_zero_attention_block_ids([1])
            self.assertNotIn("event=executed", failed.getvalue())
            self.assertIsNone(runner._huanyu_zero_execution)

    def test_existing_whisper_wrappers_do_not_block_huanyu_and_remain_for_other_models(self):
        with native_apis() as (Runner, Worker), patch.dict("os.environ", {
            "MINICPMO_PRIVATE_HYBRID_CACHE": "1",
        }):
            calls = []

            def whisper_wrapper(original, name, flag=None):
                @wraps(original)
                def wrapped(self, *args):
                    calls.append(name)
                    return original(self, *args)
                if flag:
                    setattr(wrapped, flag, True)
                return wrapped

            Runner._torch_zero_attention_block_ids = whisper_wrapper(
                Runner._torch_zero_attention_block_ids, "zero", "_q36_logical_blocks")
            Worker._scale_kv_cache_memory_for_multi_group = whisper_wrapper(
                Worker._scale_kv_cache_memory_for_multi_group, "budget", "_q36_private_cache")
            Runner._allocate_kv_cache_tensors = whisper_wrapper(Runner._allocate_kv_cache_tensors, "allocate")
            policy.install_huanyu_cache_precision()
            runner = Runner()
            runner.layer_specs = {"a": AttentionSpec(), "m": MambaSpec()}
            runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=AttentionSpec(), layer_names=["a"])
            ])
            runner.states = {"a": [torch.ones((4, 2, 1, 2))]}
            runner._torch_zero_attention_block_ids([1])
            self.assertEqual(Worker(runner)._scale_kv_cache_memory_for_multi_group(120), 80)
            buffers = runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
            self.assertEqual(len(buffers), 2)
            self.assertEqual(calls, [])
            other = Runner(False)
            other._torch_zero_attention_block_ids([1])
            Worker(other)._scale_kv_cache_memory_for_multi_group(120)
            other._allocate_kv_cache_tensors("native")
            self.assertEqual(calls, ["zero", "budget", "allocate"])

    def test_private_partial_hook_install_fails_instead_of_accepting_unpaired_budget(self):
        with native_apis() as (_, Worker), patch.dict("os.environ", {
            "MINICPMO_PRIVATE_HYBRID_CACHE": "1",
        }), redirect_stdout(io.StringIO()) as output:
            Worker._scale_kv_cache_memory_for_multi_group._huanyu_private_cache = True
            with self.assertRaisesRegex(RuntimeError, "partially installed"):
                policy.install_huanyu_cache_precision()
        self.assertNotIn("component=private_", output.getvalue())

    def test_private_empty_and_failed_operations_do_not_report_execution(self):
        with native_apis() as (Runner, Worker), patch.dict("os.environ", {
            "MINICPMO_PRIVATE_HYBRID_CACHE": "1",
        }):
            policy.install_huanyu_cache_precision()
            runner = Runner()
            runner.layer_specs = {}
            worker = Worker(runner)
            with redirect_stdout(io.StringIO()) as empty:
                self.assertEqual(worker._scale_kv_cache_memory_for_multi_group(120), 120)
            self.assertNotIn("event=executed", empty.getvalue())
            self.assertFalse(runner._q36_private_cache_budget_applied)
            runner.layer_specs = {"a": AttentionSpec(), "m": MambaSpec()}
            with redirect_stdout(io.StringIO()) as failed:
                with self.assertRaisesRegex(RuntimeError, "matching physical-memory budget"):
                    runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
            self.assertNotIn("event=executed", failed.getvalue())
            worker._scale_kv_cache_memory_for_multi_group(120)
            with patch.object(shared, "planner_budget", side_effect=RuntimeError("planner failed")):
                with redirect_stdout(io.StringIO()) as failed:
                    with self.assertRaisesRegex(RuntimeError, "planner failed"):
                        worker._scale_kv_cache_memory_for_multi_group(120)
                self.assertNotIn("event=executed", failed.getvalue())
                self.assertFalse(runner._q36_private_cache_budget_applied)
            worker._scale_kv_cache_memory_for_multi_group(120)
            with redirect_stdout(io.StringIO()) as failed:
                with self.assertRaisesRegex(RuntimeError, "budget is 120"):
                    runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=11))
                aliased = torch.zeros(80, dtype=torch.int8)
                with patch.object(shared.torch, "zeros", return_value=aliased):
                    with self.assertRaisesRegex(RuntimeError, "unexpectedly alias"):
                        runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
                with patch.object(shared.torch, "zeros", side_effect=RuntimeError("allocation failed")):
                    with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                        runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
            self.assertNotIn("event=executed component=private_buffers", failed.getvalue())

    def test_uniform_group_planner_accounts_for_each_physical_layer(self):
        with native_apis():
            specs = {"a": AttentionSpec(page_size_bytes=8), "b": AttentionSpec(page_size_bytes=16)}
            groups = [SimpleNamespace(kv_cache_spec=UniformTypeKVCacheSpecs(specs), layer_names=["a", "b"]),
                      SimpleNamespace(kv_cache_spec=MambaSpec(), layer_names=["m"])]
            self.assertEqual(shared.planner_budget(280, groups), 240)


if __name__ == "__main__":
    unittest.main()
