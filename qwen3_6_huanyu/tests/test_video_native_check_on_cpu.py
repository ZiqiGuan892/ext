"""The native smoke check must reject recovered but invalid video metadata."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


PACKAGE = "_huanyu_video_native_check_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
checker = importlib.import_module(f"{PACKAGE}.check_video_contract")


class NativeVideoCheckTests(unittest.TestCase):
    def run_check(self, *, recovered=False, expanded=True, mask=True):
        # Reproduce the logged grid: 38 groups, 252 video tokens per group.
        ids = ([14] + [16] * 252 + [15]) * 38
        calls = []

        class Boundary:
            def __init__(self):
                raise AssertionError("The check must not initialize a model/engine")

            def _get_prompt_updates(self, *args):
                return "updates"

            def _bind_and_group_updates(self, updates, counts):
                return "resolved"

            def _apply_prompt_updates(self, prompt, updates):
                calls.append(prompt)
                position = SimpleNamespace(
                    offset=1 if recovered else 0,
                    length=38 * 252 if recovered else len(ids),
                    is_embed=(torch.tensor(ids) == 16) if mask else None,
                )
                return (ids if expanded else prompt), {"video": [SimpleNamespace(to_range=lambda: position)]}

        native = SimpleNamespace(
            MiniCPMO50OmniLLMMultiModalProcessor=Boundary,
            Q36_VIDEO_PLACEHOLDER="qwen-video",
        )
        tokenizer = SimpleNamespace(encode=lambda text, **kw: [1 if text == "qwen-video" else 2])
        proc = SimpleNamespace(tokenizer=tokenizer, video_token_id=16)
        result = list(checker._native_video_ranges(
            proc, native, SimpleNamespace(_name_or_path="2748"), torch.tensor([38, 28, 36]), ids,
        ))
        self.assertEqual(calls, [[1], [2], ids])
        return result

    def test_full_ranges_and_masks_are_checked_for_compact_and_preexpanded_prompts(self):
        positions = self.run_check()
        self.assertEqual(len(positions), 3)
        self.assertEqual([int(p.is_embed.sum()) for p in positions], [9576, 9576, 9576])

    def test_compact_prompt_returned_by_fallback_is_not_a_success(self):
        with self.assertRaisesRegex(AssertionError, "prompt update differs"):
            self.run_check(expanded=False)

    def test_summed_disjoint_pad_runs_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "lost temporal boundaries"):
            self.run_check(recovered=True)

    def test_missing_embedding_mask_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "lost its embedding mask"):
            self.run_check(mask=False)


if __name__ == "__main__":
    unittest.main()
