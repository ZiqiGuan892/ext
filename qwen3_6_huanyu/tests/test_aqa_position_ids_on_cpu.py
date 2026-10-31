"""AQA/STEM positions remain aligned through the trainer's jagged batch boundary."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
ENV = "HUANYU_AQA_AUDIO_TRAIN"
STEM_ENV = "HUANYU_STEM_TEXT_TRAIN"
BASE_MODULE = "verl.experimental.agent_loop.agent_loop"
TQ_MODULE = "verl.trainer.ppo.v1.agent_loop_tq"


def worker_class():
    class Worker:
        def __init__(self, positions, huanyu=True):
            self.processor = SimpleNamespace(_huanyu_training_processor=huanyu)
            self.positions = positions
            self.calls = []

        def _compute_position_ids(
            self, input_ids, attention_mask, multi_modal_inputs=None, mm_processor_kwargs=None
        ):
            self.calls.append((input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs))
            return self.positions

    return Worker


class AqaPositionIdsTests(unittest.TestCase):
    mode_flag = ENV

    def setUp(self):
        self.base = worker_class()
        self.tq = worker_class()
        modules = {}
        for path in (BASE_MODULE, TQ_MODULE):
            components = path.split(".")
            for length in range(1, len(components) + 1):
                name = ".".join(components[:length])
                module = ModuleType(name)
                module.__path__ = []
                modules.setdefault(name, module)
        modules[BASE_MODULE].AgentLoopWorker = self.base
        modules[TQ_MODULE].AgentLoopWorkerTQ = self.tq
        self.modules = modules
        missing = object()
        original_modules = {name: sys.modules.get(name, missing) for name in modules}
        sys.modules.update(modules)

        def restore_mocked_modules():
            # Jagged operations lazily import torch._dynamo/_inductor. Restoring
            # an entire sys.modules snapshot would delete those real imports,
            # causing duplicate TORCH_LIBRARY registration in subsequent tests.
            for name, original in original_modules.items():
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.addCleanup(restore_mocked_modules)
        flags = {ENV: "0", STEM_ENV: "0", self.mode_flag: "1"}
        self.env_patch = patch.dict(os.environ, flags)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        spec = importlib.util.spec_from_file_location(
            "_huanyu_aqa_position_ids_tested", ROOT / "aqa_position_ids.py"
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)

    @staticmethod
    def inputs():
        # Audio placeholders are ordinary sequential positions in this path.
        ids = torch.tensor([[10, 70, 76, 76, 71, 20, 21, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return ids, mask, positions.unsqueeze(1).repeat(1, 4, 1)

    def test_collapses_both_worker_classes_without_changing_tokens_or_metadata(self):
        self.adapter.install_huanyu_aqa_position_ids()
        for cls in (self.base, self.tq):
            ids, mask, original = self.inputs()
            media = {"audio_features": torch.zeros(1, 40, 8)}
            options = {"sampling_rate": 16000}
            worker = cls(original)
            actual = worker._compute_position_ids(ids, mask, media, options)
            self.assertEqual(actual.shape, ids.shape)
            self.assertEqual(actual.dtype, torch.long)
            torch.testing.assert_close(actual, original[:, 0])
            self.assertEqual(len(worker.calls), 1)
            for seen, expected in zip(worker.calls[0], (ids, mask, media, options)):
                self.assertIs(seen, expected)

    def test_install_and_runtime_environment_gates_preserve_frozen_and_old_paths(self):
        base_original, tq_original = self.base._compute_position_ids, self.tq._compute_position_ids
        with patch.dict(os.environ, {self.mode_flag: "0"}):
            self.adapter.install_huanyu_aqa_position_ids()
        self.assertIs(self.base._compute_position_ids, base_original)
        self.assertIs(self.tq._compute_position_ids, tq_original)
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        worker = self.tq(positions)
        for disabled in ("", "0"):
            with self.subTest(flag=disabled), patch.dict(os.environ, {self.mode_flag: disabled}):
                self.assertIs(worker._compute_position_ids(ids, mask, {}), positions)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(worker._compute_position_ids(ids, mask, {}), positions)

    def test_aqa_position_contract_is_independent_of_audio_freezing(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        for frozen in (True, False):
            worker = self.tq(positions)
            worker.processor.config = SimpleNamespace(huanyu_freeze_audio=frozen)
            with self.subTest(frozen=frozen):
                torch.testing.assert_close(
                    worker._compute_position_ids(ids, mask, {}), positions[:, 0]
                )

    def test_other_processors_are_exact_delegates_even_with_vision(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        worker = self.tq(positions, huanyu=False)
        media = {"image_grid_thw": torch.tensor([[1, 2, 2]])}
        self.assertIs(worker._compute_position_ids(ids, mask, media), positions)
        self.assertIs(worker.calls[0][2], media)

    def test_already_two_dimensional_native_positions_are_returned_unchanged(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        original = positions[:, 0]
        worker = self.tq(original)
        self.assertIs(worker._compute_position_ids(ids, mask), original)

    def test_install_is_idempotent_and_ray_backing_classes_are_patched(self):
        ray_base, ray_tq = worker_class(), worker_class()

        class ActorHandle:
            def __init__(self, cls):
                self.__ray_metadata__ = SimpleNamespace(modified_class=cls)

        self.modules[BASE_MODULE].AgentLoopWorker = ActorHandle(ray_base)
        self.modules[TQ_MODULE].AgentLoopWorkerTQ = ActorHandle(ray_tq)
        self.adapter.install_huanyu_aqa_position_ids()
        first = [cls._compute_position_ids for cls in (ray_base, ray_tq)]
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        for cls, installed in zip((ray_base, ray_tq), first):
            self.assertIs(cls._compute_position_ids, installed)
            worker = cls(positions)
            torch.testing.assert_close(
                worker._compute_position_ids(ids, mask, {}), positions[:, 0]
            )
            self.assertEqual(len(worker.calls), 1)

    def test_wrapper_keeps_local_identity_for_ray_serialization(self):
        original = self.base._compute_position_ids
        self.adapter.install_huanyu_aqa_position_ids()
        installed = self.base._compute_position_ids
        self.assertEqual(installed.__name__, original.__name__)
        self.assertEqual(installed.__module__, self.adapter.__name__)
        self.assertIn("install_huanyu_aqa_position_ids.<locals>", installed.__qualname__)
        self.assertNotEqual(installed.__qualname__, original.__qualname__)

    def test_installer_handles_inherited_methods_without_double_wrapping(self):
        class TQ(self.base):
            pass

        self.modules[TQ_MODULE].AgentLoopWorkerTQ = TQ
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        worker = TQ(positions)
        torch.testing.assert_close(worker._compute_position_ids(ids, mask, {}), positions[:, 0])
        self.assertEqual(len(worker.calls), 1)

    def test_padding_axis_differences_are_ignored_but_valid_axis_differences_fail(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        positions[:, 1:, -1] = torch.tensor([10, 20, 30])
        actual = self.tq(positions)._compute_position_ids(ids, mask, {})
        torch.testing.assert_close(actual, positions[:, 0])
        changed = positions.clone()
        changed[0, 2, 3] += 1
        with self.assertRaises(ValueError):
            self.tq(changed)._compute_position_ids(ids, mask, {})

    def test_nonempty_vision_inputs_cannot_silently_lose_multimodal_positions(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.tq(positions)._compute_position_ids(
                    ids, mask, {name: torch.ones(1, 3, dtype=torch.long)}
                )

    def test_empty_vision_fields_do_not_block_audio(self):
        self.adapter.install_huanyu_aqa_position_ids()
        ids, mask, positions = self.inputs()
        media = {
            "pixel_values": None, "image_grid_thw": torch.empty(0, 3, dtype=torch.long),
            "pixel_values_videos": [], "video_grid_thw": None,
        }
        actual = self.tq(positions)._compute_position_ids(ids, mask, media)
        torch.testing.assert_close(actual, positions[:, 0])

    def test_server_four_axis_jagged_metadata_reproduces_reported_split_error(self):
        # as_nested_tensor chooses its ragged dimension from the input; merely
        # changing _ragged_idx cannot move offsets from the axis to token lengths.
        for length in (792, 3675):  # AQA and STEM server reports
            with self.subTest(length=length):
                nested = torch.nested.as_nested_tensor(
                    [torch.arange(length).repeat(4, 1)], layout=torch.jagged
                )
                torch.testing.assert_close(nested.offsets(), torch.tensor([0, 4]))
                nested._ragged_idx = 2
                with self.assertRaisesRegex(RuntimeError, r"split_sizes=\[4\]"):
                    nested.unbind()

    def test_restoration_expands_without_a_copy_and_preserves_int64_precision(self):
        ids = torch.arange(18).reshape(3, 6)
        positions = ids + 4096
        restored = self.adapter.restore_huanyu_aqa_positions(positions, ids)
        self.assertEqual(restored.shape, (4, 3, 6))
        self.assertEqual(restored.dtype, torch.long)
        self.assertEqual(restored.untyped_storage().data_ptr(), positions.untyped_storage().data_ptr())
        for axis in restored:
            torch.testing.assert_close(axis, positions)
        self.assertIs(self.adapter.restore_huanyu_aqa_positions(restored, ids), restored)

    def test_restoration_is_gated_and_rejects_bad_batch_or_axis_shapes(self):
        ids = torch.arange(18).reshape(3, 6)
        positions = ids + 4096
        with patch.dict(os.environ, {self.mode_flag: "0"}):
            self.assertIs(self.adapter.restore_huanyu_aqa_positions(positions, ids), positions)
        self.assertIsNone(self.adapter.restore_huanyu_aqa_positions(None, ids))
        for invalid in (
            torch.zeros(3, 7), torch.zeros(4, 2, 6),
            torch.zeros(3, 4, 6), torch.zeros(3, 3, 6),
        ):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                self.adapter.restore_huanyu_aqa_positions(invalid, ids)

    def test_variable_length_jagged_selection_preserves_every_token_and_position(self):
        self.adapter.install_huanyu_aqa_position_ids()
        masks = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        tokens = torch.arange(24, dtype=torch.long).reshape(3, 8) + 2000
        positions = (masks.cumsum(-1) - 1).clamp_min(0)
        native = positions.unsqueeze(1).repeat(1, 4, 1)
        # Model offsets may be large; int64 positions must not be BF16-rounded.
        native += 2048
        collapsed = self.tq(native)._compute_position_ids(tokens, masks, {})
        position_rows = [row[mask.bool()] for row, mask in zip(collapsed, masks)]
        token_rows = [row[mask.bool()] for row, mask in zip(tokens, masks)]
        packed_positions = torch.nested.as_nested_tensor(position_rows, layout=torch.jagged)
        packed_tokens = torch.nested.as_nested_tensor(token_rows, layout=torch.jagged)
        torch.testing.assert_close(packed_positions.offsets(), torch.tensor([0, 6, 10, 17]))
        self.assertEqual(packed_positions._ragged_idx, 1)
        # The trainer selects per-sample tensors and rebuilds a jagged minibatch.
        # Select a nontrivial order, with a repeated sample, then pad for forward.
        order = [2, 0, 1, 2]
        position_items, token_items = packed_positions.unbind(), packed_tokens.unbind()
        selected_positions = torch.nested.as_nested_tensor(
            [position_items[index] for index in order], layout=torch.jagged
        )
        selected_tokens = torch.nested.as_nested_tensor(
            [token_items[index] for index in order], layout=torch.jagged
        )
        padded_positions = torch.nn.utils.rnn.pad_sequence(
            selected_positions.unbind(), batch_first=True, padding_value=-1
        )
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            selected_tokens.unbind(), batch_first=True, padding_value=-1
        )
        restored = self.adapter.restore_huanyu_aqa_positions(padded_positions, padded_tokens)
        self.assertEqual(restored.shape, (4, 4, 7))
        for output_index, input_index in enumerate(order):
            valid = masks[input_index].bool()
            size = int(valid.sum())
            torch.testing.assert_close(
                padded_positions[output_index, :size], native[input_index, 0, valid]
            )
            torch.testing.assert_close(
                padded_tokens[output_index, :size], tokens[input_index, valid]
            )
            # Reconstruct all four model-entry axes, including the separate text
            # axis used by HF linear attention, from the transported common axis.
            torch.testing.assert_close(
                restored[:, output_index, :size], native[input_index, :, valid]
            )
            self.assertTrue(torch.all(padded_positions[output_index, size:] == -1))
            self.assertTrue(torch.all(padded_tokens[output_index, size:] == -1))
        self.assertEqual(sum(len(row) for row in selected_positions.unbind()), 24)


class StemPositionIdsTests(AqaPositionIdsTests):
    """Apply the same transport/restore and vision guards with AQA disabled."""

    mode_flag = STEM_ENV


if __name__ == "__main__":
    unittest.main()
