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

"""CPU-only lexical reward for Huanyu AQA runs without a reward model.

The reward is character-bigram set Dice, as used by the existing GenRM lexical
fallback. It measures surface overlap, not semantic correctness. In particular,
paraphrases may score poorly and repeated text is not penalized by set Dice.
This entry point intentionally does not import the GenRM scorer or call an API.

Final answers are extracted after the last ``</think>`` and from the last
complete ``<answer>...</answer>`` pair, when present. Unfinished thinking or an
unfinished answer block receives zero rather than rewarding reasoning text.
For a template with a prompt-side think opener, set ``require_think_end=True``
so a response truncated before its closing tag also receives zero.
Both reference and answer use NFKC, case folding and whitespace removal.
Punctuation is retained: removing it would conflate ``1.2`` with ``12`` and
``-1`` with ``1``.
"""

import re
import unicodedata
from typing import Any


_THINK_TAG = re.compile(r"<(/?)think\s*>", flags=re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<(/?)answer\s*>", flags=re.IGNORECASE)
_ANSWER_BLOCK = re.compile(
    r"<answer\s*>(.*?)</answer\s*>", flags=re.IGNORECASE | re.DOTALL
)


def _final_answer(text: str) -> str:
    """Extract a final answer, including an implicit prompt-side think opener."""
    depth = 0
    last_think_end = 0
    for match in _THINK_TAG.finditer(text):
        if match.group(1):
            depth = max(0, depth - 1)
            last_think_end = match.end()
        else:
            depth += 1
    if depth:
        return ""

    candidate = text[last_think_end:].strip()
    answer_depth = 0
    has_answer_tag = False
    for match in _ANSWER_TAG.finditer(candidate):
        has_answer_tag = True
        answer_depth = (
            max(0, answer_depth - 1) if match.group(1) else answer_depth + 1
        )
    if answer_depth:
        return ""
    if has_answer_tag:
        blocks = list(_ANSWER_BLOCK.finditer(candidate))
        return blocks[-1].group(1).strip() if blocks else ""
    return candidate


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if not character.isspace())


def _bigram_dice(reference: str, answer: str) -> float:
    def grams(text: str) -> set[str]:
        if len(text) < 2:
            return {text} if text else set()
        return {text[index : index + 2] for index in range(len(text) - 1)}

    reference_grams, answer_grams = grams(reference), grams(answer)
    if not reference_grams or not answer_grams:
        return 0.0
    return 2.0 * len(reference_grams & answer_grams) / (
        len(reference_grams) + len(answer_grams)
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict | None = None,
    require_think_end: bool = False,
    **kwargs: Any,
) -> dict[str, float]:
    """Score against ``reward_model.ground_truth`` without an RM or data router.

    ``data_source`` and ``extra_info`` are accepted for verl's reward interface;
    AQA extraction preserves the original ``omni_mix`` source name. Missing or
    invalid references raise instead of silently producing all-zero rewards.
    ``require_think_end`` must match the prompt's thinking mode: it requires a
    closing think tag in generated text only, never in the reference answer.
    The additional metrics describe lexical scoring and are not semantic
    accuracy estimates.
    """
    del data_source, extra_info, kwargs
    if not isinstance(require_think_end, bool):
        raise TypeError("rule_based reward requires require_think_end to be a bool")
    if not isinstance(ground_truth, str):
        raise ValueError("rule_based reward requires a string ground_truth")
    reference = _normalize(_final_answer(ground_truth))
    if not reference:
        raise ValueError("rule_based reward requires a non-empty reference answer")
    if not isinstance(solution_str, str):
        raise TypeError("rule_based reward requires solution_str to be a string")

    has_think_end = any(match.group(1) for match in _THINK_TAG.finditer(solution_str))
    answer = (
        ""
        if require_think_end and not has_think_end
        else _normalize(_final_answer(solution_str))
    )
    score = round(_bigram_dice(reference, answer), 4)
    return {
        "score": score,
        "rule_based_score": score,
        "rule_based_exact_match": float(bool(answer) and answer == reference),
        "rule_based_empty_answer": float(not answer),
    }
