"""CPU regressions for Huanyu-only GDN/linear bindings and execution evidence.

Native device kernels are mocked; reference outputs/gradients are compared with
Whisper's formulas. These tests do not claim to validate Ascend kernel accuracy.
"""
import importlib.util
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gdn = load("huanyu_component_gdn", ROOT / "qwen_precision/gdn_precision.py")
linear = load("huanyu_component_linear", ROOT / "qwen_precision/linear_precision.py")
whisper = load("whisper_component_gdn", ROOT.parent / "qwen3_6_whisper/gdn_precision.py")


def chunk_reference(q, k, v, g, beta, initial_state,
                    output_final_state, use_qk_l2norm_in_kernel):
    # Small differentiable recurrence; float matmul exposes unwanted autocast.
    state = initial_state
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].float().exp()[..., None, None]
        update = k[:, t].float().unsqueeze(-1) @ v[:, t].float().unsqueeze(-2)
        state = state + beta[:, t].float()[..., None, None] * update
        outputs.append((q[:, t].float().unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outputs, 1).to(v.dtype), state


class Attention:
    _q36_reference_prefill_installed = True  # Simulate a foreign class patch.
    prefix = "model.layers.0.linear_attn"  # No fragile thinker.llm substring.
    _chunk_gated_delta_rule_fused = staticmethod(lambda *a, **kw: "native-prefill")

    def forward(self, action):
        return action()


def native(attention):
    return SimpleNamespace(llm=SimpleNamespace(model=SimpleNamespace(
        layers=[SimpleNamespace(linear_attn=attention)])))


def class_dispatch_attention(ascend_module):
    """Model Ascend copying core methods, but not static helpers, onto Qwen."""
    exec(compile('''
class AscendGatedDeltaNetAttention:
    _fused_chunk_available = False
    probe_calls = 0

    @classmethod
    def _probe_fused_chunk(cls):
        cls.probe_calls += 1
        return cls._fused_chunk_available

    @staticmethod
    def _chunk_gated_delta_rule_fused(q, k, v, g, beta, initial_state, cu_seqlens, scale):
        return "native-fused"

    def _forward_core(self, q, k, v, g, beta, initial_state, cu_seqlens, scale):
        if AscendGatedDeltaNetAttention._probe_fused_chunk():
            return AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                cu_seqlens=cu_seqlens, scale=scale)
        return "native-fallback"

    def forward(self, *args, **kwargs):
        return self._forward_core(*args, **kwargs)

class QwenAttention:
    prefix = "model.layers.0.linear_attn"
    forward = AscendGatedDeltaNetAttention.forward
    _forward_core = AscendGatedDeltaNetAttention._forward_core
''', "fake_ascend_gdn.py", "exec"), vars(ascend_module))
    return ascend_module.QwenAttention


def prefill_inputs():
    torch.manual_seed(29)
    return dict(
        q=torch.randn(1, 4, 1, 2, dtype=torch.bfloat16),
        k=torch.randn(1, 4, 1, 2, dtype=torch.bfloat16),
        v=torch.randn(1, 4, 2, 3, dtype=torch.bfloat16),
        g=torch.full((1, 4, 2), -0.2),
        beta=torch.full((1, 4, 2), 0.3),
        initial_state=torch.randn(2, 2, 3, 2),
        cu_seqlens=torch.tensor([0, 2, 4]),
        scale=2 ** -0.5,
    )


class GDNTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        gdn._EXECUTED.clear()
        self.hf = ModuleType("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
        self.hf.torch_chunk_gated_delta_rule = chunk_reference
        self.ascend = ModuleType("vllm_ascend.ops.gdn")

        def native_gate(*args):
            return "native-gate"
        native_gate._q36_actor_gating = True
        self.ascend.DeviceOperator = type("DeviceOperator", (), {"fused_gdn_gating": staticmethod(native_gate)})
        self.ascend.l2norm_fwd = lambda *a, **kw: "native-norm"
        self.modules = patch.dict(sys.modules, {self.hf.__name__: self.hf, self.ascend.__name__: self.ascend})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_prefill_binding_ignores_foreign_marker_is_local_and_disables_autocast(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        attention, rm = Attention(), Attention()
        with redirect_stdout(io.StringIO()) as installed:
            gdn.install_reference_prefill(native(attention))
            first = attention.forward
            first_chunk = attention._chunk_gated_delta_rule_fused
            gdn.install_reference_prefill(native(attention))
        self.assertIs(attention.forward, first)
        self.assertIs(attention._chunk_gated_delta_rule_fused, first_chunk)
        self.assertIn("phase=already_installed", installed.getvalue())
        self.assertNotIn("phase=executed", installed.getvalue())
        self.assertEqual(rm._chunk_gated_delta_rule_fused(), "native-prefill")
        torch.manual_seed(21)
        q = torch.randn(1, 4, 1, 2, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn(1, 4, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        decay = torch.full((1, 4, 2), -0.2)
        beta = torch.full((1, 4, 2), 0.3)
        state = torch.randn(2, 2, 3, 2, requires_grad=True)
        args = (q, k, v, decay, beta, state, torch.tensor([0, 2, 4]), 2 ** -0.5)
        expected = whisper.reference_prefill(*args)
        with redirect_stdout(io.StringIO()) as executed:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = attention.forward(lambda: attention._chunk_gated_delta_rule_fused(*args))
                attention.forward(lambda: attention._chunk_gated_delta_rule_fused(*args))
                self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(executed.getvalue().count("component=gdn_prefill phase=executed"), 1)
        self.assertEqual(actual[1].dtype, torch.float32)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        grads = torch.autograd.grad(sum(x.float().square().sum() for x in actual), (q, k, v, state))
        expected_grads = torch.autograd.grad(sum(x.float().square().sum() for x in expected), (q, k, v, state))
        for got, want in zip(grads, expected_grads):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertFalse(gdn._policy_prefill.get())

    def test_class_dispatch_on_qwen_instance_runs_reference_without_changing_rm(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        cls = class_dispatch_attention(self.ascend)
        attention, another, rm = cls(), cls(), cls()
        owner = self.ascend.AscendGatedDeltaNetAttention
        self.assertFalse(hasattr(attention, "_chunk_gated_delta_rule_fused"))
        self.assertFalse(hasattr(attention, "_probe_fused_chunk"))
        model = native(attention)
        model.llm.model.layers.append(SimpleNamespace(linear_attn=another))
        with redirect_stdout(io.StringIO()) as installed:
            gdn.install_reference_prefill(model)
            first_forward, first_core = attention.forward, attention._forward_core
            first_chunk = owner._chunk_gated_delta_rule_fused
            first_probe = owner._probe_fused_chunk.__func__
            gdn.install_reference_prefill(model)
        self.assertIn("dispatch=ascend_class", installed.getvalue())
        self.assertNotIn("phase=executed", installed.getvalue())
        self.assertIs(attention.forward, first_forward)
        self.assertIs(attention._forward_core, first_core)
        self.assertIs(owner._chunk_gated_delta_rule_fused, first_chunk)
        self.assertIs(owner._probe_fused_chunk.__func__, first_probe)
        self.assertFalse(owner._fused_chunk_available)
        inputs = prefill_inputs()
        expected = whisper.reference_prefill(**inputs)
        with redirect_stdout(io.StringIO()) as executed:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                results = [attention.forward(**inputs), attention._forward_core(**inputs),
                           another._forward_core(**inputs)]
                self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(executed.getvalue().count("component=gdn_prefill phase=executed"), 1)
        self.assertEqual(owner.probe_calls, 0)
        for result in results:
            self.assertEqual(result[0].dtype, torch.bfloat16)
            self.assertEqual(result[1].dtype, torch.float32)
            for got, want in zip(result, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertFalse(gdn._policy_prefill.get())
        self.assertFalse(owner._fused_chunk_available)
        self.assertEqual(rm.forward(**inputs), "native-fallback")
        self.assertEqual(rm._forward_core(**inputs), "native-fallback")
        self.assertFalse(owner._probe_fused_chunk())
        self.assertEqual(owner.probe_calls, 3)
        self.assertEqual(owner._chunk_gated_delta_rule_fused(**inputs), "native-fused")
        owner._fused_chunk_available = True
        self.assertEqual(rm.forward(**inputs), "native-fused")

    def test_class_dispatch_failure_restores_scope_and_does_not_report_execution(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        attention = class_dispatch_attention(self.ascend)()
        gdn.install_reference_prefill(native(attention))
        with redirect_stdout(io.StringIO()) as log:
            with patch.object(gdn, "reference_prefill", side_effect=RuntimeError("reference failed")):
                for method in (attention.forward, attention._forward_core):
                    with self.assertRaisesRegex(RuntimeError, "reference failed"):
                        method(**prefill_inputs())
                    self.assertFalse(gdn._policy_prefill.get())
        self.assertNotIn("phase=executed", log.getvalue())
        self.assertFalse(self.ascend.AscendGatedDeltaNetAttention._probe_fused_chunk())

    def test_class_dispatch_missing_probe_fails_before_mutating_layer(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        attention = class_dispatch_attention(self.ascend)()
        owner = self.ascend.AscendGatedDeltaNetAttention
        original_forward, original_core = attention.forward, attention._forward_core
        original_chunk = owner._chunk_gated_delta_rule_fused
        del owner._probe_fused_chunk
        with redirect_stdout(io.StringIO()) as log:
            with self.assertRaisesRegex(RuntimeError, "prefill/probe"):
                gdn.install_reference_prefill(native(attention))
        self.assertEqual(attention.forward, original_forward)
        self.assertEqual(attention._forward_core, original_core)
        self.assertIs(owner._chunk_gated_delta_rule_fused, original_chunk)
        self.assertNotIn("phase=installed", log.getvalue())

    def test_whisper_barrier_is_preserved_when_discovering_class_prefill(self):
        os.environ.update(MINICPMO_GDN_PREFILL_REFERENCE="1", MINICPMO_GDN_CACHE_BARRIER="1")
        cls = class_dispatch_attention(self.ascend)
        attention, rm = cls(), cls()
        for layer in (attention, rm):
            layer.layer_idx = 0
            layer.kv_cache = (None, torch.zeros(1))
            whisper.install_mixed_cache_barrier(native(layer))
        barrier = attention._forward_core
        self.assertFalse(hasattr(barrier, "__wrapped__"))
        context = ModuleType("vllm.forward_context")
        context.get_forward_context = lambda: SimpleNamespace(attn_metadata={
            attention.prefix: SimpleNamespace(num_prefills=1, num_decodes=2)})
        npu = SimpleNamespace(synchronize=Mock())
        inputs = prefill_inputs()
        expected = whisper.reference_prefill(**inputs)
        with patch.dict(sys.modules, {context.__name__: context}), patch.object(torch, "npu", npu, create=True):
            with redirect_stdout(io.StringIO()) as installed:
                gdn.install_reference_prefill(native(attention))
                scoped = attention._forward_core
                gdn.install_reference_prefill(native(attention))
            self.assertIs(attention._forward_core, scoped)
            self.assertIs(scoped.__wrapped__, barrier)
            self.assertIn("dispatch=ascend_class", installed.getvalue())
            with redirect_stdout(io.StringIO()) as executed:
                for method in (attention.forward, attention._forward_core):
                    result = method(**inputs)
                    for got, want in zip(result, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertEqual(npu.synchronize.call_count, 4)
            self.assertEqual(executed.getvalue().count("component=gdn_prefill phase=executed"), 1)
            self.assertIn("mixed cache barrier executed", executed.getvalue())
            self.assertFalse(gdn._policy_prefill.get())

            # Re-discovery must also traverse Huanyu's wraps-decorated barrier.
            gdn.install_mixed_cache_barrier(native(attention))
            both_barriers = attention._forward_core
            gdn.install_reference_prefill(native(attention))
            gdn.install_mixed_cache_barrier(native(attention))
            self.assertIs(attention._forward_core, both_barriers)
            npu.synchronize.reset_mock()
            attention._forward_core(**inputs)
            self.assertEqual(npu.synchronize.call_count, 4)

            npu.synchronize.reset_mock()
            self.assertEqual(rm.forward(**inputs), "native-fallback")
            self.assertEqual(npu.synchronize.call_count, 2)
            self.assertFalse(self.ascend.AscendGatedDeltaNetAttention._fused_chunk_available)
            self.assertFalse(gdn._policy_prefill.get())

            gdn._EXECUTED.clear()
            npu.synchronize.reset_mock()
            with redirect_stdout(io.StringIO()) as failed:
                with patch.object(gdn, "reference_prefill", side_effect=RuntimeError("reference failed")):
                    for method in (attention.forward, attention._forward_core):
                        with self.assertRaisesRegex(RuntimeError, "reference failed"):
                            method(**inputs)
                        self.assertFalse(gdn._policy_prefill.get())
            self.assertEqual(npu.synchronize.call_count, 8)
            self.assertNotIn("component=gdn_prefill phase=executed", failed.getvalue())

    def test_unknown_unwrapped_closure_is_not_treated_as_whisper_barrier(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        attention = class_dispatch_attention(self.ascend)()
        original = attention._forward_core

        def unknown(self, *args, **kwargs):
            return original(*args, **kwargs)

        attention._forward_core = MethodType(unknown, attention)
        attention._q36_mixed_cache_barrier = True  # A marker alone is insufficient.
        core = attention._forward_core
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            gdn.install_reference_prefill(native(attention))
        self.assertIs(attention._forward_core, core)

    def test_whisper_prefill_then_barrier_then_huanyu_uses_huanyu_reference(self):
        os.environ.update(MINICPMO_GDN_PREFILL_REFERENCE="1", MINICPMO_GDN_CACHE_BARRIER="1")
        cls = class_dispatch_attention(self.ascend)
        upstream = ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        upstream.QwenGatedDeltaNetAttention = cls
        with patch.dict(sys.modules, {upstream.__name__: upstream}):
            whisper.install_reference_prefill()
        attention, rm = cls(), cls()
        attention.prefix = "thinker.llm.model.layers.0.linear_attn"
        attention.layer_idx = 0
        attention.kv_cache = (None, torch.zeros(1))
        whisper.install_mixed_cache_barrier(native(attention))
        barrier = attention._forward_core
        self.assertFalse(hasattr(attention.forward, "__wrapped__"))
        self.assertFalse(hasattr(barrier, "__wrapped__"))
        context = ModuleType("vllm.forward_context")
        context.get_forward_context = lambda: SimpleNamespace(attn_metadata={
            attention.prefix: SimpleNamespace(num_prefills=1, num_decodes=2)})
        npu = SimpleNamespace(synchronize=Mock())
        inputs = prefill_inputs()
        expected = gdn.reference_prefill(**inputs)
        with patch.dict(sys.modules, {context.__name__: context}), patch.object(torch, "npu", npu, create=True):
            gdn.install_reference_prefill(native(attention))
            scoped = attention._forward_core
            gdn.install_reference_prefill(native(attention))
            self.assertIs(attention._forward_core, scoped)
            self.assertIs(scoped.__wrapped__, barrier)
            with patch.object(whisper, "reference_prefill", side_effect=AssertionError("foreign reference used")):
                for method in (attention.forward, attention._forward_core):
                    result = method(**inputs)
                    for got, want in zip(result, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    self.assertFalse(gdn._policy_prefill.get())
                    self.assertFalse(whisper._policy_prefill.get())
                self.assertEqual(rm.forward(**inputs), "native-fused")
            self.assertEqual(npu.synchronize.call_count, 4)

    def test_malformed_whisper_barrier_original_fails_without_hanging(self):
        os.environ.update(MINICPMO_GDN_PREFILL_REFERENCE="1", MINICPMO_GDN_CACHE_BARRIER="1")
        cls = class_dispatch_attention(self.ascend)
        for bad_original in ("not_callable", "self_cycle"):
            with self.subTest(original=bad_original):
                attention = cls()
                whisper.install_mixed_cache_barrier(native(attention))
                core = attention._forward_core
                cells = dict(zip(core.__code__.co_freevars, core.__closure__))
                cells["original"].cell_contents = None if bad_original == "not_callable" else core
                with self.assertRaisesRegex(RuntimeError, "(?i)(prefill|wrapper|barrier|cycle)"):
                    gdn.install_reference_prefill(native(attention))
                self.assertIs(attention._forward_core, core)

    def test_gate_and_l2_run_only_inside_huanyu_scope_and_log_after_success(self):
        os.environ.update(MINICPMO_GDN_GATING_REFERENCE="1", MINICPMO_GDN_L2NORM_REFERENCE="1")
        attention, rm = Attention(), Attention()
        gdn.install_reference_prefill(native(attention))
        gate = self.ascend.DeviceOperator.fused_gdn_gating
        norm = self.ascend.l2norm_fwd
        self.assertTrue(gate._huanyu_actor_gating)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        args = (torch.randn(3), a, a.clone(), torch.randn(3))
        self.assertEqual(rm.forward(lambda: gate(*args)), "native-gate")
        self.assertEqual(rm.forward(lambda: norm(a)), "native-norm")
        with redirect_stdout(io.StringIO()) as log:
            with patch.object(gdn, "actor_gdn_gating", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    attention.forward(lambda: gate(*args))
        self.assertNotIn("phase=executed", log.getvalue())
        self.assertFalse(gdn._policy_prefill.get())
        with redirect_stdout(io.StringIO()) as log:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = attention.forward(lambda: gate(*args))
                normalized = attention.forward(lambda: norm(a))
                attention.forward(lambda: gate(*args))
        for got, want in zip(result, whisper.actor_gdn_gating(*args)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(normalized, whisper.actor_l2norm(a), rtol=0, atol=0)
        self.assertEqual(log.getvalue().count("component=gdn_gating phase=executed"), 1)
        self.assertEqual(log.getvalue().count("component=gdn_l2norm phase=executed"), 1)

    def test_missing_conv_can_retry_after_native_init_and_foreign_marker_is_not_ours(self):
        os.environ["MINICPMO_GDN_CONV_REFERENCE"] = "1"
        ops = SimpleNamespace()
        attention = Attention()
        original_forward = attention.forward
        calls = []
        value = torch.tensor([0.0134, -1.0007, 3.121], dtype=torch.float32)

        def raw(output, activation_mode, run_mode):
            calls.append((activation_mode, run_mode))
            output.copy_(F.silu(value).to(output.dtype) if activation_mode else value.to(output.dtype))
            return "native-result"
        raw._q36_conv_reference = True
        with patch.object(torch.ops, "_C_ascend", ops):
            with self.assertRaisesRegex(RuntimeError, "retry after native model initialization"):
                gdn.install_reference_prefill(native(attention))
            self.assertEqual(attention.forward, original_forward)
            ops.npu_causal_conv1d_custom = raw
            gdn.install_reference_prefill(native(attention))
            wrapper = ops.npu_causal_conv1d_custom
            gdn.install_reference_prefill(native(attention))
            self.assertIs(wrapper, ops.npu_causal_conv1d_custom)
            self.assertIs(wrapper.default, wrapper)
            output = torch.empty(3, dtype=torch.bfloat16)
            wrapper(output=output, activation_mode=1, run_mode=0)
            self.assertEqual(calls[-1], (1, 0))
            with redirect_stdout(io.StringIO()) as log:
                for mode in (0, 1, 0):
                    result = attention.forward(lambda: wrapper.default(output=output, activation_mode=1, run_mode=mode))
                    self.assertEqual(result, "native-result")
                    torch.testing.assert_close(output, F.silu(value.to(torch.bfloat16)), rtol=0, atol=0)
                    self.assertEqual(calls[-1], (0, mode))
            self.assertEqual(log.getvalue().count("phase=executed"), 2)

    def test_barrier_is_local_idempotent_and_does_not_claim_failed_execution(self):
        os.environ["MINICPMO_GDN_CACHE_BARRIER"] = "1"
        attention, rm = Attention(), Attention()
        attention.kv_cache = (None, torch.zeros(1))
        attention._forward_core = Mock(return_value="output")
        rm._forward_core = Mock(return_value="rm")
        context = ModuleType("vllm.forward_context")
        context.get_forward_context = lambda: SimpleNamespace(attn_metadata={
            attention.prefix: SimpleNamespace(num_prefills=1, num_decodes=2)})
        npu = SimpleNamespace(synchronize=Mock())
        with patch.dict(sys.modules, {context.__name__: context}), patch.object(torch, "npu", npu, create=True):
            gdn.install_mixed_cache_barrier(native(attention))
            first = attention._forward_core
            gdn.install_mixed_cache_barrier(native(attention))
            self.assertIs(first, attention._forward_core)
            with redirect_stdout(io.StringIO()) as log:
                self.assertEqual(attention._forward_core(), "output")
                self.assertEqual(attention._forward_core(), "output")
            self.assertEqual(npu.synchronize.call_count, 4)
            self.assertEqual(log.getvalue().count("component=gdn_cache_barrier phase=executed"), 1)
            self.assertEqual(rm._forward_core(), "rm")
            gdn._EXECUTED.clear()
            attention._forward_core.__wrapped__.side_effect = RuntimeError("core failed")
            with redirect_stdout(io.StringIO()) as log:
                with self.assertRaisesRegex(RuntimeError, "core failed"):
                    attention._forward_core()
            self.assertNotIn("phase=executed", log.getvalue())

    def test_requested_unsupported_api_fails_instead_of_installing_empty_coverage(self):
        os.environ["MINICPMO_GDN_PREFILL_REFERENCE"] = "1"
        with self.assertRaisesRegex(RuntimeError, "explicit native"):
            gdn.install_reference_prefill(SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "no language linear-attention"):
            gdn.install_reference_prefill(SimpleNamespace(llm=SimpleNamespace(model=SimpleNamespace(layers=[]))))
        attention = Attention()
        attention._chunk_gated_delta_rule_fused = None
        with self.assertRaisesRegex(RuntimeError, "requires _chunk"):
            gdn.install_reference_prefill(native(attention))

    def test_barrier_without_mixed_metadata_never_reports_execution_or_synchronizes(self):
        os.environ["MINICPMO_GDN_CACHE_BARRIER"] = "1"
        attention = Attention()
        attention.kv_cache = (None, torch.zeros(1))
        attention._forward_core = lambda: "output"
        context = ModuleType("vllm.forward_context")
        npu = SimpleNamespace(synchronize=Mock())
        with patch.dict(sys.modules, {context.__name__: context}), patch.object(torch, "npu", npu, create=True):
            gdn.install_mixed_cache_barrier(native(attention))
            for metadata in (None, {}, {attention.prefix: SimpleNamespace(num_prefills=1, num_decodes=0)},
                             {attention.prefix: SimpleNamespace(num_prefills=0, num_decodes=1)}):
                context.get_forward_context = lambda: SimpleNamespace(attn_metadata=metadata)
                with redirect_stdout(io.StringIO()) as log:
                    self.assertEqual(attention._forward_core(), "output")
                self.assertNotIn("phase=executed", log.getvalue())
            npu.synchronize.assert_not_called()


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Sequential(torch.nn.Linear(3, 3, dtype=torch.bfloat16))
        # Avoid registering self as a child module.
        self.model.language_model.layers = [self.model.language_model[0]]
        self.model.audio_encoder = torch.nn.Linear(3, 3, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(3, 3, dtype=torch.bfloat16)


class LinearTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MINICPMO_STABLE_LINEAR": "1"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        linear._EXECUTED.clear()

    def test_actor_foreign_marker_idempotency_local_scope_and_gradient_parity(self):
        original = F.linear
        foreign = Mock(side_effect=original)
        foreign._q36_stable_linear = True
        # Mock's dynamic attributes must not pretend to own Huanyu's marker.
        foreign._huanyu_stable_linear = False
        actor, other = Actor(), Actor()
        reference_calls = []

        def reference(x, w, b=None):
            reference_calls.append(1)
            return original(x, w, b)

        with patch.object(F, "linear", foreign), patch.object(linear, "persistent_linear", side_effect=reference), \
                patch.object(linear, "_eligible", side_effect=lambda x, w: x.dtype == w.dtype == torch.bfloat16):
            with redirect_stdout(io.StringIO()) as log:
                linear.install_actor(actor)
                wrapper = F.linear
                first = actor.model.language_model.forward
                linear.install_actor(actor)
                linear.install_actor(other)
            self.assertNotIn("phase=executed", log.getvalue())
            self.assertIs(wrapper, F.linear)
            self.assertIs(first, actor.model.language_model.forward)
            x = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
            w, b = actor.model.language_model[0].parameters()
            expected = original(x, w, b)
            with redirect_stdout(io.StringIO()) as log:
                result = actor.model.language_model(x)
                actor.model.language_model(x)
                other.model.language_model(x)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            grads = torch.autograd.grad(result.float().square().sum(), (x, w, b))
            expected_grads = torch.autograd.grad(expected.float().square().sum(), (x, w, b))
            for got, want in zip(grads, expected_grads):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertEqual(log.getvalue().count("component=stable_linear_actor phase=executed"), 1)
            count = len(reference_calls)
            actor.model.audio_encoder(x)
            F.linear(x, w, b)  # Same dtype outside the Huanyu language scope.
            self.assertEqual(len(reference_calls), count)
            self.assertEqual(linear._actor_depth.get(), 0)
            with patch.object(linear, "persistent_linear", side_effect=RuntimeError("kernel failed")):
                linear._EXECUTED.clear()
                with redirect_stdout(io.StringIO()) as log:
                    with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                        actor.model.language_model(x)
                self.assertNotIn("phase=executed", log.getvalue())
                self.assertEqual(linear._actor_depth.get(), 0)

    def test_rollout_marks_explicit_layers_not_prefix_and_preserves_rm(self):
        original = F.linear

        class Method:
            def apply(self, layer, x, bias=None):
                return "native"
        Method.apply._q36_stable_linear = True
        module = ModuleType("vllm_ascend.ops.linear")
        module.AscendUnquantizedLinearMethod = Method

        def layer():
            result = torch.nn.Linear(3, 3, dtype=torch.bfloat16)
            result.quant_method = Method()
            result.prefix = "llm.unusual_prefix"
            return result
        language = torch.nn.Sequential(layer(), layer())
        rm = layer()
        # An identical prefix on RM must not select the Huanyu path.
        rm.prefix = "thinker.llm.layer.0"
        huanyu = SimpleNamespace(thinker=SimpleNamespace(llm=language))
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch.object(linear, "persistent_linear", side_effect=original), \
                patch.object(linear, "_eligible", return_value=True):
            with redirect_stdout(io.StringIO()) as log:
                linear.install_rollout(huanyu)
                first = Method.apply
                linear.install_rollout(huanyu)
            self.assertIs(first, Method.apply)
            self.assertNotIn("phase=executed", log.getvalue())
            x = torch.randn(2, 3, dtype=torch.bfloat16)
            with redirect_stdout(io.StringIO()) as log:
                for projection in language:
                    result = projection.quant_method.apply(projection, x, projection.bias)
                    torch.testing.assert_close(result, original(x, projection.weight, projection.bias), rtol=0, atol=0)
            self.assertEqual(log.getvalue().count("component=stable_linear_rollout phase=executed"), 1)
            self.assertEqual(rm.quant_method.apply(rm, x), "native")

    def test_enabled_unsupported_models_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "explicit actor"):
            linear.install_actor()
        module = ModuleType("vllm_ascend.ops.linear")
        module.AscendUnquantizedLinearMethod = type("Method", (), {})
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "no Ascend unquantized"):
                linear.install_rollout(SimpleNamespace(llm=torch.nn.Linear(3, 3)))


if __name__ == "__main__":
    unittest.main()
