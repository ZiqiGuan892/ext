# Copyright 2026 verl-omni contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""No model packages are needed to exercise the no-RM lexical reward."""

import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "_huanyu_rule_based_reward_tested",
    Path(__file__).resolve().parents[1] / "rule_based_reward.py",
)
_REWARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWARD)


class RuleBasedRewardTests(unittest.TestCase):
    def score(self, answer, reference, **kwargs):
        return _REWARD.compute_score("omni_mix", answer, reference, **kwargs)

    def test_exact_match_accepts_omni_mix_without_default_reward_routing(self):
        result = self.score("北京", "北京")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["rule_based_score"], 1.0)
        self.assertEqual(result["rule_based_exact_match"], 1.0)
        self.assertEqual(result["rule_based_empty_answer"], 0.0)
        self.assertNotIn("acc", result)

    def test_nfkc_casefold_and_whitespace_normalization(self):
        for answer, reference in (
            ("ＡＢＣ　１２", "abc12"),
            ("Straße", "STRASSE"),
            ("北 京\n欢\t迎 你", "北京欢迎你"),
            ("Cafe\u0301", "CAFÉ"),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(self.score(answer, reference)["score"], 1.0)

    def test_punctuation_preserved_for_numbers_and_signs(self):
        self.assertEqual(self.score("1.2", "12")["score"], 0.0)
        self.assertEqual(self.score("-1", "1")["score"], 0.0)
        self.assertEqual(self.score("1,000", "1000")["rule_based_exact_match"], 0.0)
        self.assertEqual(self.score("１２．５", "12.5")["score"], 1.0)

    def test_bigram_dice_has_known_partial_chinese_score(self):
        # Reference grams: 北京 / 京欢 / 欢迎 / 迎你; answer grams: 北京 / 京欢 / 欢迎.
        result = self.score("北京欢迎", "北京欢迎你")
        self.assertAlmostEqual(result["score"], 0.8571)
        self.assertEqual(result["rule_based_exact_match"], 0.0)

    def test_single_character_reference(self):
        self.assertEqual(self.score("B", "b")["score"], 1.0)
        self.assertEqual(self.score("A", "B")["score"], 0.0)
        self.assertEqual(self.score("AB", "B")["score"], 0.0)

    def test_thinking_overlap_does_not_get_reward(self):
        result = self.score("<think>北京</think>上海", "北京")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["rule_based_empty_answer"], 0.0)

    def test_prompt_side_think_opener_and_last_close(self):
        result = self.score("thinking</think>wrong</think>北京", "北京")
        self.assertEqual(result["score"], 1.0)

    def test_thinking_mode_requires_generation_to_close_prompt_side_think(self):
        for answer in ("北京", "北京是我想到的答案", "<answer>北京</answer>"):
            with self.subTest(answer=answer):
                result = self.score(answer, "北京", require_think_end=True)
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["rule_based_empty_answer"], 1.0)
        for answer in ("reasoning</think>北京", "<think>reasoning</THINK >北京"):
            with self.subTest(answer=answer):
                result = self.score(answer, "北京", require_think_end=True)
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["rule_based_empty_answer"], 0.0)

    def test_non_thinking_mode_accepts_plain_final_answers(self):
        result = self.score("北京", "<answer>北京</answer>", require_think_end=False)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["rule_based_empty_answer"], 0.0)

    def test_think_end_setting_requires_an_actual_bool(self):
        for value in ("true", "false", 0, 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "require_think_end"):
                    self.score("北京", "北京", require_think_end=value)

    def test_think_tag_case_and_trailing_space(self):
        self.assertEqual(
            self.score("<THINK>wrong</THINK >北京", "北京")["score"], 1.0
        )

    def test_unclosed_thinking_is_empty_even_if_it_contains_answer_tags(self):
        for answer in (
            "<think>北京",
            "<think><answer>北京</answer>",
            "earlier</think>北京<think>unfinished",
            "<think><think>unfinished</think>北京",
        ):
            with self.subTest(answer=answer):
                result = self.score(answer, "北京")
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["rule_based_empty_answer"], 1.0)

    def test_answer_blocks_are_extracted_from_output_and_reference(self):
        result = self.score(
            "<think>noise</think>prefix <answer>北京</answer> suffix",
            "<answer>北京</answer>",
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(
            self.score("<answer>上海</answer><ANSWER>北京</ANSWER>", "北京")["score"],
            1.0,
        )

    def test_answer_block_in_reasoning_is_not_a_final_answer(self):
        self.assertEqual(
            self.score("<think><answer>北京</answer></think>上海", "北京")["score"],
            0.0,
        )

    def test_empty_or_unfinished_answer_block_gets_zero(self):
        for answer in (
            "",
            " \n\t ",
            "</think>",
            "<answer></answer>",
            "<answer>北京",
            "<answer>北京</answer><answer>unfinished",
            "</answer>",
        ):
            with self.subTest(answer=answer):
                result = self.score(answer, "北京")
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["rule_based_empty_answer"], 1.0)

    def test_bad_reference_raises_instead_of_silent_zero(self):
        for reference in (None, 123, ["北京"], {"answer": "北京"}, "", " \n ", "<answer></answer>"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ValueError, "ground_truth|reference answer"):
                    self.score("北京", reference)
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            _REWARD.compute_score("omni_mix", "北京")

    def test_bad_generated_response_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "solution_str"):
            self.score(None, "北京")

    def test_all_metrics_are_finite_scalars_in_unit_interval(self):
        for answer in ("北京", "北京欢迎", "上海", "", "<think>北京"):
            result = self.score(answer, "北京欢迎你")
            self.assertEqual(result["score"], result["rule_based_score"])
            for value in result.values():
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_no_console_output_or_reference_mutation(self):
        extra = {"raw_source": "aqa_real"}
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.score("北京", "北京", extra_info=extra, reward_router_address=None)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(extra, {"raw_source": "aqa_real"})


if __name__ == "__main__":
    unittest.main()
