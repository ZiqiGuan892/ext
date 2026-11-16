"""Video M-RoPE values survive TQ restoration and nested batch selection."""

import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch


WHISPER = Path(__file__).resolve().parents[2] / "qwen3_6_whisper"
PACKAGE = "_huanyu_mrope_transport_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(WHISPER)]
sys.modules[PACKAGE] = package
positions = importlib.import_module(PACKAGE + ".position_ids")


def video_positions(length, offset=0):
    tokens = torch.arange(length, dtype=torch.long)
    # Distinct text/time/height/width axes: a 1D collapse would corrupt these.
    return torch.stack((tokens, tokens // 16, (tokens // 4) % 4, tokens % 4)) + offset


class MropeTransportTests(unittest.TestCase):
    def assert_items(self, nested, expected):
        actual = nested.unbind()
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(nested._ragged_idx, 2)
        self.assertEqual(nested.values().shape[0], 4)
        for item, reference in zip(actual, expected):
            torch.testing.assert_close(item, reference, rtol=0, atol=0)

    def test_reported_split_four_failure_is_repaired_without_changing_axes(self):
        for length in (9965, 14066, 25261):
            with self.subTest(length=length):
                reference = video_positions(length)
                broken = torch.nested.as_nested_tensor([reference], layout=torch.jagged)
                torch.testing.assert_close(broken.offsets(), torch.tensor([0, 4]))
                broken._ragged_idx = 2
                with self.assertRaisesRegex(RuntimeError, r"split_sizes=\[4\]"):
                    broken.unbind()
                restored = positions.normalize_position_ids_nested(broken)
                torch.testing.assert_close(restored.offsets(), torch.tensor([0, length]))
                self.assert_items(restored, [reference])

    def test_channel_concatenation_restores_each_sample_then_reorders(self):
        items = [video_positions(37, offset) for offset in (0, 1000, 2000)]
        transported = torch.nested.as_nested_tensor(items, layout=torch.jagged)
        restored = positions.normalize_position_ids_nested(transported)
        torch.testing.assert_close(restored.offsets(), torch.tensor([0, 37, 74, 111]))
        self.assert_items(restored, items)
        selected = [restored.unbind()[i] for i in (2, 0, 2, 1)]
        repacked = torch.nested.as_nested_tensor(selected, layout=torch.jagged)
        self.assert_items(positions.normalize_position_ids_nested(repacked), selected)

    def test_variable_length_sequence_offsets_and_values_are_preserved(self):
        items = [video_positions(13), video_positions(23, 1000), video_positions(9, 2000)]
        offsets = torch.tensor([0, 13, 36, 45])
        nested = torch.nested.nested_tensor_from_jagged(
            values=torch.cat(items, dim=1), offsets=offsets, jagged_dim=2,
        )
        self.assert_items(positions.normalize_position_ids_nested(nested), items)
        strided = torch.nested.as_nested_tensor(items, layout=torch.strided)
        self.assert_items(positions.normalize_position_ids_nested(strided), items)

    def test_text_positions_are_unchanged_and_invalid_four_axis_payload_fails(self):
        text = torch.nested.as_nested_tensor([torch.arange(9), torch.arange(5)], layout=torch.jagged)
        self.assertIs(positions.normalize_position_ids_nested(text), text)
        invalid = torch.nested.as_nested_tensor([torch.ones(3, 11)], layout=torch.jagged)
        with self.assertRaisesRegex(ValueError, "four channels"):
            positions.normalize_position_ids_nested(invalid)

    def test_restore_hook_replaces_existing_alias_and_remains_idempotent(self):
        original = Mock()
        original._q36_sequence_offsets = False
        verl = ModuleType("verl")
        verl.__path__ = []
        utils = ModuleType("verl.utils")
        utils.__path__ = []
        tu = ModuleType("verl.utils.tensordict_utils")
        alias = ModuleType("verl.utils.transferqueue_utils")
        tu.maybe_fix_3d_position_ids = alias.maybe_fix_3d_position_ids = original
        verl.utils, utils.tensordict_utils = utils, tu
        modules = {m.__name__: m for m in (verl, utils, tu, alias)}
        with patch.dict(sys.modules, modules):
            positions.install_tq_position_ids_patch()
            installed = tu.maybe_fix_3d_position_ids
            self.assertIs(alias.maybe_fix_3d_position_ids, installed)
            positions.install_tq_position_ids_patch()
            self.assertIs(tu.maybe_fix_3d_position_ids, installed)
            reference = video_positions(31)
            data = {"position_ids": torch.nested.as_nested_tensor([reference], layout=torch.jagged)}
            alias.maybe_fix_3d_position_ids(data)
            self.assert_items(data["position_ids"], [reference])
            original.assert_not_called()
            ordinary = {"position_ids": torch.arange(8).reshape(1, 8)}
            installed(ordinary)
            original.assert_called_once_with(ordinary)


if __name__ == "__main__":
    unittest.main()
