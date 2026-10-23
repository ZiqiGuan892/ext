"""Real tensor checks for Huanyu-only hybrid KV cache hooks."""

from contextlib import contextmanager, redirect_stdout
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
        policy._UNAVAILABLE_REPORTED.clear()

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

    def test_missing_default_hook_reports_unavailable_once_without_claiming_success(self):
        with native_apis(missing_zero=True), redirect_stdout(io.StringIO()) as output:
            self.assertFalse(policy.install_huanyu_cache_precision())
            self.assertFalse(policy.install_huanyu_cache_precision())
        self.assertEqual(output.getvalue().count("unavailable stage=logical_zero"), 1)
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
            policy.install_huanyu_cache_precision()
            budget_hook, allocation_hook = Worker._scale_kv_cache_memory_for_multi_group, Runner._allocate_kv_cache_tensors
            policy.install_huanyu_cache_precision()
            self.assertIs(Worker._scale_kv_cache_memory_for_multi_group, budget_hook)
            self.assertIs(Runner._allocate_kv_cache_tensors, allocation_hook)
            runner = Runner()
            runner.layer_specs = {"attention": AttentionSpec(), "mamba": MambaSpec()}
            worker = Worker(runner)
            self.assertEqual(worker._scale_kv_cache_memory_for_multi_group(120), 80)
            self.assertFalse(runner.supports_standardized_shared_kv_backing)
            self.assertTrue(runner._q36_private_cache_budget_applied)
            result = runner._allocate_kv_cache_tensors(SimpleNamespace(num_blocks=10))
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


if __name__ == "__main__":
    unittest.main()
