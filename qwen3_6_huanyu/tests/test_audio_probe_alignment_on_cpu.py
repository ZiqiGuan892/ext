"""Batched and isolated native audio calls align only under a verified contract."""

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "huanyu_probe_alignment.py"
SPEC = importlib.util.spec_from_file_location("huanyu_probe_alignment_cpu", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def capture(partition):
    lengths = [9, 23, 16]
    meta = {"encoder_downsample": 4, "projector_downsample": 2,
            "audio_groups": [{"chunk_lengths": lengths[:2]}, {"chunk_lengths": lengths[2:]}],
            "hook_counts": {"encoder": len(partition), "projector": len(partition)}}
    arrays, offset = {}, 0
    for call, batch in enumerate(partition):
        lens = lengths[offset:offset + batch]
        frames = ((max(lens) + 3) // 4) * 4
        encoded_frames = frames // 4
        x = np.full((batch, 2, frames), -100.0)
        y = np.full((batch, encoded_frames, 3), -200.0)
        p = np.full((batch, encoded_frames // 2, 6), -300.0)
        q = np.full((batch, encoded_frames // 2, 5), -400.0)
        for index, length in enumerate(lens):
            identity = offset + index + 1
            enc_length = (length + 3) // 4
            x[index, :, :length] = identity
            y[index, :enc_length] = identity * 10
            p[index, :enc_length // 2] = identity * 100
            q[index, :enc_length // 2] = identity * 1000
        arrays.update({
            "encoder/%d/input/args/0" % call: x,
            "encoder/%d/input/args/1" % call: np.asarray(lens, dtype=np.int64),
            "encoder/%d/output/0" % call: y,
            "projector/%d/input/args/0" % call: p,
            "projector/%d/output" % call: q,
        })
        offset += batch
    return meta, arrays


class AudioProbeAlignmentTests(unittest.TestCase):
    def test_batched_and_isolated_chunks_match_with_varied_padding(self):
        batched, reason = MODULE.align_native_chunks(*capture([3]))
        self.assertIsNone(reason)
        isolated, reason = MODULE.align_native_chunks(*capture([1, 1, 1]))
        self.assertIsNone(reason)
        self.assertEqual(len(batched), 3)
        for left, right in zip(batched, isolated):
            self.assertEqual(set(left), set(right))
            for stage in left:
                np.testing.assert_array_equal(left[stage], right[stage])
        self.assertEqual([row["encoder_output"].shape[0] for row in batched], [3, 6, 4])
        self.assertEqual([row["projector_output"].shape[0] for row in batched], [1, 3, 2])

    def test_invalid_layouts_are_not_guessed(self):
        cases = []
        meta, arrays = capture([3])
        bad = copy.deepcopy(meta)
        bad["audio_groups"][0]["chunk_lengths"] = [23, 9]
        cases.append((bad, arrays, "lengths/order"))
        bad = copy.deepcopy(meta)
        bad["projector_downsample"] = 3
        cases.append((bad, arrays, "fold factor"))
        bad = dict(arrays)
        bad["encoder/0/output/0"] = bad["encoder/0/output/0"][:, :1]
        cases.append((meta, bad, "sequence shape"))
        bad = dict(arrays)
        bad["projector/0/output"] = bad["projector/0/output"][0]
        cases.append((meta, bad, "layout"))
        bad = dict(arrays)
        bad["encoder/0/input/args/1"] = np.array([9., 23., 16.])
        cases.append((meta, bad, "lengths/order"))
        bad = dict(arrays)
        bad["encoder/0/input/args/0"] = bad["encoder/0/input/args/0"][:, :, :20]
        cases.append((meta, bad, "valid lengths"))
        for current_meta, current_arrays, message in cases:
            with self.subTest(message=message):
                result, reason = MODULE.align_native_chunks(current_meta, current_arrays)
                self.assertIsNone(result)
                self.assertIn(message, reason)

    def test_metadata_and_cardinality_are_required(self):
        for field, value in (("encoder_downsample", None), ("projector_downsample", 0),
                             ("encoder_downsample", True)):
            meta, arrays = capture([3])
            meta[field] = value
            self.assertIn("positive integer", MODULE.align_native_chunks(meta, arrays)[1])
        meta, arrays = capture([1, 1])
        self.assertIn("every audio chunk", MODULE.align_native_chunks(meta, arrays)[1])
        meta, arrays = capture([3])
        meta["hook_counts"]["projector"] = 2
        self.assertIn("call counts", MODULE.align_native_chunks(meta, arrays)[1])
        meta, arrays = capture([3])
        arrays["encoder/1/output/0"] = np.ones((1, 1, 1))
        self.assertIn("hook paths", MODULE.align_native_chunks(meta, arrays)[1])


if __name__ == "__main__":
    unittest.main()
