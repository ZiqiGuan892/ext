# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Verification-style generative reward model (GenRM) scorer for text RL.

An instruct model (e.g. Qwen3.5-35B-A3B) is served behind the reward router and
asked one closed question: *is the policy's final answer equivalent to the
reference answer?* It does not re-solve the problem and it does not score style
-- with a ground truth in hand, an open-ended 0-10 judge only adds variance.
The verdict maps to a binary reward, so the signal stays on the same scale the
rule-based scorer produced and GSPO's clip ratios keep their meaning.

Why a GenRM instead of ``verl.utils.reward_score.gsm8k``: the rule-based scorer
does a literal string match on the text after ``####``. Answers that are right
but written differently ("18", "18.0", "$18", "18 dollars") score 0, and that
false negative is indistinguishable from a real mistake in the gradient.

Wiring (see ``examples/gspo_trainer/minicpmo_5_0/run_minicpmo_5_0_thinker_gspo_npu.sh``)::

    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.genrm_verifier
    reward.custom_reward_function.name=compute_score
    reward.reward_model.enable=True
    reward.reward_model.model_path=<judge checkpoint>

``reward.custom_reward_function.path`` is not optional here.
``RewardLoopWorker.compute_score`` routes a bare ``reward_model.enable=True`` to
``compute_score_disrm``, which POSTs ``/classify`` -- a pooling endpoint a
generative checkpoint does not serve.

Environment overrides (all optional):

===========================  ==========================================================
``GENRM_MODEL_NAME``         served model name; default: ask the router ``/v1/models``
``GENRM_TEMPERATURE``        judge sampling temperature (default ``0.0``)
``GENRM_TOP_P``              judge nucleus cutoff (default ``1.0``)
``GENRM_MAX_TOKENS``         judge generation budget (default ``96``; 2026-10-13 从 1024 下调,
                             理由见 :func:`_sampling_params`)
``GENRM_FREQUENCY_PENALTY``  按出现次数累计扣 logits —— 本文件里唯一能直接打断
                             judge 的 ``0 0 0…`` 2-gram 环的旋钮 (default ``0.5``)
``GENRM_STOP_AT_FIRST_LINE`` ``0`` 关闭「判词首行结束处即停」的 ``\n`` stop
                             (default ``1``; 详见 ``VERDICT_STOP_STRINGS`` 下方的注释)
``GENRM_SEED``               per-request seed; unset by default
``GENRM_ENABLE_THINKING``    ``1`` lets the judge think before the verdict (default ``0``)
``GENRM_STRIP_THINK``        ``0`` shows the judge the policy's reasoning (default ``1``)
``GENRM_MAX_RETRIES``        HTTP attempts per request (default ``5``)
``GENRM_TIMEOUT``            per-attempt timeout in seconds (default ``600``)
``GENRM_ENDPOINT``           ``auto`` (default), ``chat`` or ``completions``; see :func:`_judge`
``GENRM_FALLBACK_ON_ERROR``  ``rule`` (default), ``zero`` or ``similarity``; see
                             :func:`_fallback_score`. **omni_mix 建议设 ``similarity``** ——
                             ``rule`` 对该 data_source 恒 NotImplementedError → 硬 0
``GENRM_LOG_RESPONSE``       ``0`` drops the judge text from reward_extra_info (default ``1``)
``GENRM_RESPONSE_MAX_CHARS`` truncation for that text (default ``512``)
``GENRM_DUMP_DIR``           sample dump directory; unset disables the dump entirely
``GENRM_DUMP_MAX``           dumps written per process (default ``50``)
``GENRM_DUMP_ERR_MAX``       extra dump quota for fallback samples past ``GENRM_DUMP_MAX`` (default ``20``)
``GENRM_DUMP_FULL_CHARS``    ``rollout_full`` length cap in the dump (default ``4000``)
``GENRM_SCORING_MODE``       ``coverage`` (default): judge outputs ``Score: k/N``, reward=k/N;
                             ``verdict``: legacy binary Verdict grading
===========================  ==========================================================

Three diagnostic messages carry the hallucination/truncation evidence that the
reward value alone cannot express. With ``MINICPMO_ROLLOUT_DIAG=1`` they go only
to JSONL files in ``MINICPMO_ROLLOUT_DIAG_DIR``; otherwise their historical console
output is retained:

``[answer-diag]``
    Per degraded sample: response length, whether ``</think>`` ever arrived (an
    answer taken from an unterminated think block is a truncated chain of thought,
    not an answer), and the longest trailing repetition.
``[judge-diag]``
    The judge side of the same story: whether it repeated itself into
    ``finish_reason=length`` instead of emitting a ``Verdict:`` line. A judge that
    collapsed is a different bug from a judge that answered off-format.
``[genrm_verifier] REWARD-DEGRADED``
    Every fallback, with a reason and a running per-reason count -- a step of hard
    zeros that came from the scorer not knowing the dataset is a different bug from
    one that came from the judge timing out, and the two need opposite fixes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from verl_omni_ext.probes import diagnostic_io as _rollout_diag

_JUDGE_TRACE_ID = ContextVar("genrm_diagnostic_trace_id", default=None)

# ── repetition_penalty：以「每个 server 自己模型的 generation_config.json」为准 ──
# 背景：verl 的 vllm_async_server.py `_get_override_generation_config()` 把
# repetition_penalty 写死成 1.0，而这个值会盖掉模型自带的 generation_config.json，
# 于是 judge（Qwen3.5-35B-A3B，文件里声明了 1.1）被压成 1.0，更容易复读。该函数在
# vLLM server 起来之前就会被调用，所以这段 patch 必须放在模块 import 期。
#
# ⚠️ 2026-10-10 两处修正：
# 1) 原来是无条件 `config["repetition_penalty"] = 1.1`，但
#    `_get_override_generation_config` 是 vLLMHttpServer 的**类方法**，actor 的
#    rollout engine 也是这个类 → judge 专属的 1.1 顺带打在 actor 上：
#      qwen36_whisper_full.log:5171 actor engine(pid=727128) 的启动参数
#      '--override_generation_config {"temperature": 1.0, "top_k": -1, "top_p": 1,
#                                     "repetition_penalty": 1.1, "max_new_tokens": 512}'
#    而 actor checkpoint 的 generation_config.json 里**根本没有** repetition_penalty
#    → RL 采样分布悄悄偏离 checkpoint 声明的配置，正是本项目最忌讳的「静默失效」。
# 2) 第一版修正想用 `MODEL_PATH` 环境变量区分 judge / actor，**这样不行**：
#    env_export.sh 的注释已经写明——WorkerDict / vLLMOmniHttpServer 都是 Ray actor，
#    env 继承自 raylet，而 raylet 是平台的 `ray start` fork 出来的；驱动脚本里的
#    `export MODEL_PATH` 只活在 driver 进程里，actor 侧读不到 → 判断会静默退化成
#    「永远是 judge」，等于没修。
#
# 现在的判据完全不依赖环境变量：**读这个 server 自己那份模型的
# generation_config.json**。
#   · 文件里声明了 repetition_penalty → 用它（judge 得 1.1，与它的 checkpoint 一致）
#   · 文件里没有（actor 的情况）      → 保留上游默认 1.0（与它的 checkpoint 一致）
# 两边都变成「checkpoint 说什么就是什么」，与 actor 侧 top_p/top_k 那处修正同一原则。
# 读不到文件 / 解析失败 → 保留上游值并在日志里说明，不猜、不写死。
# 每个 server 初始化时都会打一行，说明它读到了什么、最终用了什么，事后可核。


def _declared_repetition_penalty(local_path):
    """从模型目录的 generation_config.json 读 repetition_penalty；未声明/读不到返回 None。"""
    if not local_path:
        return None
    try:
        with open(os.path.join(local_path, "generation_config.json"), encoding="utf-8") as f:
            val = json.load(f).get("repetition_penalty")
        return float(val) if val is not None else None
    except Exception:
        return None


def _resolve_repetition_penalty(local_path, upstream_value):
    """决定这个 server 最终用哪个 repetition_penalty，返回 ``(值, 说明)``。

    判据是「该 server 自己那份 checkpoint 说了算」：``generation_config.json`` 里
    声明了就用它，没声明就保留上游值。**全程不依赖环境变量**（Ray actor 的 env 来自
    raylet，驱动脚本里的 export 传不进来，见上方注释），也不需要区分 judge/actor ——
    两类 server 各自的 checkpoint 天然给出了正确答案：

      · judge（Qwen3.5-35B-A3B）文件里有 1.1  → 1.1，与其 checkpoint 一致
      · actor（whisper）文件里没有该字段     → 保留上游 1.0，与其 checkpoint 一致

    读不到文件 / JSON 坏了也走「保留上游」，并且把原因写进日志，不猜、不写死。
    """
    declared = _declared_repetition_penalty(local_path)
    if declared is None:
        return upstream_value, (
            f"model={local_path} 的 generation_config.json 未声明 repetition_penalty"
            f"（或读不到）→ 保留上游 {upstream_value}"
            "…actor 走这支是对的；judge 若也走这支，说明 RM 目录下缺该文件"
        )
    return declared, f"model={local_path} -> {declared}（取自该模型 generation_config.json）"


try:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer as _vLLMHttpServer
    if not getattr(_vLLMHttpServer._get_override_generation_config, "_rp_patched", False):
        _orig_get_override = _vLLMHttpServer._get_override_generation_config
        _RP_DECLARED_CACHE = {}

        def _patched_override(self):
            config = _orig_get_override(self)
            local_path = getattr(getattr(self, "model_config", None), "local_path", "") or ""
            if local_path not in _RP_DECLARED_CACHE:
                _RP_DECLARED_CACHE[local_path] = _resolve_repetition_penalty(
                    local_path, config.get("repetition_penalty")
                )
            value, why = _RP_DECLARED_CACHE[local_path]
            config["repetition_penalty"] = value
            # 告警线：本补丁若被后续 import 的补丁再包一层（历史上
            # verl_omni_ext/trainer/__init__.py 就干过这事，把 1.1 写死盖回来），
            # 这里的返回值仍然会被外层改掉。发现被覆盖就大声报，别再静默失效。
            if _vLLMHttpServer._get_override_generation_config is not _patched_override:
                _rollout_diag.console_or_file(
                    "genrm_repetition_penalty_override_warning",
                    "[genrm_verifier][WARN] repetition_penalty 补丁已被后续补丁覆盖："
                    f"我算出 {value}，但最终生效值可能被外层改写（查其它 patch "
                    "_get_override_generation_config 的地方）",
                    flush=True,
                )
            _rollout_diag.console_or_file("genrm_repetition_penalty", f"[genrm_verifier] repetition_penalty: {why}")
            return config

        _patched_override._rp_patched = True
        _vLLMHttpServer._get_override_generation_config = _patched_override
        _rollout_diag.console_or_file("genrm_repetition_penalty_patch", "[genrm_verifier] patched vLLMHttpServer._get_override_generation_config "
              "(repetition_penalty 改为逐 server 读各自 checkpoint 的 generation_config.json)",
              flush=True)
except Exception as _e:
    _rollout_diag.console_or_file("genrm_repetition_penalty_patch_error", f"[genrm_verifier] failed to patch repetition_penalty: {_e}")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["compute_score", "extract_final_answer"]


# ⚠️ 判词必须**放在最前面**，不能放最后。这是 2026-10-10 夜从 16/16 个真机样本
# 里量出来的结论，不是风格偏好：
#   genrm_dump_qwen36/sample_0001..0007 与 qwen36_run_008 的 9 个 judge-diag
#   —— judge（Qwen3.5-35B-A3B）**一次都没写过 "Verdict:"**，全部 finish='length'，
#   形态固定为「先写两三句分析 → 撞进整句复读 → 512 token 烧完」：
#       "The student's response is incorrect."        ×59
#       "The user provided the answer."               ×64
#       "<answer>"                                    ×119
#       " an"                                         ×454
#   把判词放最后 = 判词所在的那一段永远排在复读后面 → 永远读不到 →
#   _parse_verdict 只能靠第 5 层散文兜底或直接落 fallback，而 omni_mix 没有规则
#   打分器（NotImplementedError）→ 整批 reward=0 → GRPO 优势=0 → 这一步白跑。
# 放最前面之后：stop 串（"Verdict: correct"/"Verdict: incorrect"）+
# include_stop_str_in_output=True 会在判词产生的**那一刻**结束生成，复读根本没机会
# 开始；即使 judge 退化到只会复读，头几个 token 也已经把判词吐出来了。
# 换句话说：**把「判词是否出现」从「取决于 judge 后面还能不能好好说话」变成
# 「取决于 judge 前 3 个 token」**。后者坏掉的话，判分链路本来就无解。
JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. You compare a student's final answer against a reference "
    "answer and report whether they agree. You never solve the problem yourself. "
    "Your reply must BEGIN with exactly one of these two lines, copied character for "
    "character, before any other word:\n"
    "Verdict: correct\n"
    "Verdict: incorrect\n"
    "Anything you write after that line must be at most two short sentences."
)

JUDGE_PROMPT_TEMPLATE = """You are grading one answer to a multimodal problem. The problem itself may have been given as an image, an audio clip or a video that you CANNOT see: a bracketed placeholder in the problem text stands for that missing media and is NOT the problem statement. Grade the student's final answer against the reference answer only.

Write your verdict line FIRST, copied exactly from these two options, with nothing before it:

Verdict: correct
Verdict: incorrect

[Problem]
{question}

[Reference answer]
{ground_truth}

[Student answer]
{answer}

Rules:
- Decide whether the student's **final** answer is semantically equivalent to the reference answer.
- Compare the semantic meaning, not the exact wording. Both answers may be in different languages or formats.
- For open-ended answers (explanations, analyses, descriptions), correct means the student captures the same key points and conclusions as the reference.
- For mathematical/formula answers, treat mathematically equal values as equivalent (1/2 vs 0.5, "18" vs "18.0").
- A student answer that is empty, cut off mid-sentence, stuck repeating itself, or about a different question is incorrect.
- Do not re-solve the problem. The reference answer is correct by definition.
- Do not discuss the input, the prompt or the placeholders, and do not restate these instructions.
- Write the verdict line first; then at most two short sentences of justification."""


# ─────────────────────────────────────────────────────────────────────────────
# 覆盖率打分（2026-10-13 起的默认模式）
#
# 为什么放弃二元 Verdict：真机 dump（滑板平衡题）实证了它的结构性缺陷——
# 参考答案是「多要点长文」（双臂定位 / 一腿抬起配重 / 躯干前倾 / 动态系统四点），
# 学生答案答对一半（①③），二元制下得 0。判对率长期 8~13%，n=4 的组里大多数整组
# 全 0 ⇒ GRPO/GSPO 优势恒 0 ⇒ 没有学习信号。改成「要点覆盖率 k/N」：
#   · 该样本会拿 ~0.5 而不是 0；
#   · 组内方差直接出现（0.25/0.5/0.75 之间也有优势差），这正是奖励稀疏的解法；
#   · 数学/单值答案的参考答案天然 N=1，自动回到严格等价判定，不会放水。
# 解析向后兼容：判官没吐 Score 行时退回 _parse_verdict（旧的 Verdict 措辞、
# 以及判官塌成 'Verdict' 复读时行为与旧版完全一致）。GENRM_SCORING_MODE=verdict
# 可整体切回旧模式做对照。
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_COVERAGE_SYSTEM_PROMPT = (
    "You are a strict grader. You compare a student's final answer against a reference "
    "answer by counting how many of the reference's key points the student's final answer "
    "covers. You never solve the problem yourself. Your reply must BEGIN with exactly one "
    "line of the form ``Score: <covered>/<total>`` (for example ``Score: 3/4`` or "
    "``Score: 0/2``), before any other word. Anything you write after that line must be at "
    "most two short sentences."
)

JUDGE_COVERAGE_PROMPT_TEMPLATE = """You are grading one answer to a multimodal problem. The problem itself may have been given as an image, an audio clip or a video that you CANNOT see: a bracketed placeholder in the problem text stands for that missing media and is NOT the problem statement. Grade the student's final answer against the reference answer only.

First identify the distinct key points of the reference answer (its claims, mechanisms, or results; the smallest natural set, typically 1-5). Then count how many of them the student's final answer covers, in meaning if not in wording.

Write your FIRST line exactly in this format, with nothing before it:

Score: <covered>/<total>

[Problem]
{question}

[Reference answer]
{ground_truth}

[Student answer]
{answer}

Rules:
- Count a key point as covered only if the student's final answer actually states it. Same meaning counts, in any language or format; mathematically equal values count.
- A single-value answer (a number, a word, a formula) has total 1.
- An empty answer, one cut off mid-sentence, one stuck repeating itself, or one about a different question is Score: 0/<total>.
- Do not re-solve the problem. The reference answer is correct by definition.
- Do not discuss the input, the prompt or the placeholders, and do not restate these instructions.
- Write the Score line first; then at most two short sentences of justification."""


# 判词的 stop 串。**只能是小写这两份，不许加大小写变体** —— 2026-10-12 03:5x–05:4x
# 用一轮真机（run#87）换来的结论，加变体会把整条奖励链打垮：
#
# 真机 judge（Qwen3.5-35B-A3B）吐的是首字母大写的 ``Verdict: Incorrect``，与小写
# 形式的 stop 表**差一个字母的大小写**，所以这两份在真机上**永远不会命中**——这不是
# bug 而是当前唯一的活路：
#   · stop 不命中 ⇒ 生成跑满 max_tokens=1024，尾巴是裸 ``Verdict`` 复读
#     （dump 实证：judge_text_chars_full=2738、repeat='Verdict'x340），
#     **代价是每条判分多烧 ~1000 个 decode token**（判官占的正是策略在等的卡）；
#   · 一旦让它命中（把大写变体加进 stop 表）⇒ vllm-omni 的 /v1/chat/completions
#     **静默忽略 include_stop_str_in_output**（不报 400，所以下面那条「服务端拒绝该
#     字段就重发」的防御分支根本不会触发）⇒ 判词行被删掉，而判词在**最前面**，
#     前面什么都不剩 ⇒ judge_text 长度 0 ⇒ _parse_verdict 拿不到证词 ⇒ 落 fallback
#     （omni_mix 没有规则评分器）⇒ 整步 reward 全 0、GSPO 优势全 0。
# 事故量化（同一批 32 卡、同一份代码，只差这张表）：
#   run#86 base  （小写两份）：[judge-diag] chars=0 → 0 条，REWARD-DEGRADED=1
#   run#87 freq05（加了大小写四份）：[judge-diag] chars=0 → 153 条，REWARD-DEGRADED=156
# 结论：**在 include_stop_str_in_output 真正生效之前，stop 表只能保持「不命中」**。
# 想省掉那 1000 个 token，正确方向是给 judge 一个小的 max_tokens 上限（判词在第一行，
# 截尾不损判定），而不是让 stop 命中。见 tests 里 test_a_matched_stop_string_costs_the_verdict。
VERDICT_STOP_STRINGS = [
    "Verdict: correct",
    "Verdict: incorrect",
]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

# The reward manager decodes the response with ``skip_special_tokens=True``
# (verl/experimental/reward_loop/reward_manager/naive.py:55). Whether the think
# delimiters survive that depends on whether the checkpoint registered them as
# special tokens, which differs across MiniCPM-o revisions -- so handle every case
# rather than asserting one.
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)


def extract_final_answer(solution_str: str, strip_think: bool = True) -> str:
    """Return the part of *solution_str* the judge should grade.

    Keeps everything after the last ``</think>``. Falls back to the text after a
    dangling ``<think>`` (generation hit the length cap mid-thought), and to the
    whole string when neither delimiter is present.

    Returning the tail rather than the whole response is deliberate: chain-of-thought
    routinely contains discarded intermediate values, and a judge that sees them
    grades the reasoning instead of the answer.
    """
    if not strip_think or not solution_str:
        return (solution_str or "").strip()

    matches = list(_THINK_CLOSE_RE.finditer(solution_str))
    if matches:
        return solution_str[matches[-1].end() :].strip()

    matches = list(_THINK_OPEN_RE.finditer(solution_str))
    if matches:
        # Unterminated think block: everything after the tag is reasoning, and the
        # answer never arrived. Hand over the tail anyway; the judge will call it
        # incorrect, which is the right verdict for a truncated generation.
        return solution_str[matches[-1].end() :].strip()

    return solution_str.strip()


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"verdict\s*[:：]\s*\*{0,2}\s*(correct|incorrect)", re.IGNORECASE)
_BARE_VERDICT_RE = re.compile(r"\b(correct|incorrect)\b", re.IGNORECASE)
# Match "correctly" / "incorrectly" as synonyms — judge may write "correctly identifies"
_CORRECTLY_RE = re.compile(r"\b(correctly|incorrectly)\b", re.IGNORECASE)
# Line that is ONLY "correct" or "incorrect" (possibly with punctuation/whitespace)
_STANDALONE_LINE_RE = re.compile(r"^\s*(correct|incorrect)\s*[.!。]?\s*$", re.IGNORECASE | re.MULTILINE)
# 紧贴在匹配词之前的否定。judge 写 "is not correct" / "does not match" 时，字面上
# 有 correct/match，语义却是反的 —— 不识别就会把错答判成 1.0（假奖励比漏判更贵）。
#
# 窗口用 `[^\n]{0,12}` 而不是 `\W{0,12}`（2026-10-10 修正）。`\W` 只吃**非单词字符**，
# 而中间那个词恰恰是字母："is not **the** correct answer" 里 `\W{0,12}` 只能吃掉
# "not" 后面那一个空格，接着 `$` 就对不上 → 匹配失败 → "not the correct" 里的
# correct 被当成正面判定。实测：
#     'is not the '      → \W: False   [^\n]: True
#     'is not an '       → \W: False   [^\n]: True
#     'is not '          → \W: True    [^\n]: True      ← 只有「否定紧贴单词」才管用
#     'is not only wrong but also '    → \W: False   [^\n]: False  ← 12 字窗口挡住，正确
# 也就是说旧写法只在「not correct」这种最贴近的形态上生效，一旦判官写成
# "is not the correct answer" / "is not a correct response" 就静默漏掉。
_NEGATED_BEFORE_RE = re.compile(
    r"(?:\bno\b|\bnot\b|\bnever\b|\bnor\b|\bcannot\b|\bcan't\b|\bcouldn't\b|\bdon't\b|\bdoesn't\b|"
    r"\bdoes\s+not\b|\bdo\s+not\b|\bfails?\s+to\b|\bunable\s+to\b|n't\b)[^\n]{0,12}$",
    re.IGNORECASE,
)

# 紧跟在匹配词**之后**的「参考答案」名词：这里的 correct/incorrect 是在**指代参考
# 答案**（"compare it with the correct answer"），不是在给判定。
# 真机证据（genrm_dump_qwen36/sample_0005.json，2026-10-10 22:52，run#9）：
#     "The student's response consists of a massive block of repeated text that does
#      not correspond to the problem or the correct answer. ... "
#     后面整段复读 "The student's response is completely unrelated to the problem."
# 判官从头到尾没给过一个正面判定，却因为这句里的 "the correct answer" 被优先级 4
# 读成 1.0 —— 一条复读垃圾拿到满分奖励。假奖励比漏判贵：它把「复读」往正梯度推。
#
# 代价（明确接受）：把「the student provided the correct answer」这种**真**判词也
# 一起丢掉，掉到优先级 5 的散文判词或 fallback。判断依据是频率：这个判官的输出里
# 指代形式（"the correct answer"）反复出现在每一段复读里，而 "provided the correct
# answer" 一次没出现过；且改成判词前置后（见 JUDGE_PROMPT_TEMPLATE）绝大多数样本
# 会在优先级 1 就解析完，这一层只是剩下的兜底。
_MENTION_AFTER_RE = re.compile(
    r"[ \t]{1,3}(?:answer|response|output|solution|option|choice|value|values|"
    r"result|results|interpretation|reasoning|explanation|statement|expected|"
    r"mathematical|numerical|final)s?[\s.,;:!?]",
    re.IGNORECASE,
)

# 指代形式和判定形式差一个系动词：
#   "compare it with the correct answer"   ← 指代（correct 是定语）
#   "the student's answer **is** the correct answer" ← 判定（correct 表语）
# 所以系动词在近处出现时，指代形式要按判定收下（再由 _flip_if_negated 处理
# "is **not** the correct answer"）。
_COPULA_BEFORE_RE = re.compile(r"(?:\b(?:is|are|was|were|be|been|seems?|appears?)\b|'s)[^\n]{0,12}$", re.I)

# 优先级 5：整句判词（judge 不用 "Verdict:" 格式时的兜底）。
#
# 真机证据（2026-10-10，genrm_dump_qwen36/sample_0001..0007.json，7/7 全部走到 fallback）：
# judge 一次都没写过 "Verdict:"，它以散文给判定，而且复读塌掉后整段里连一个
# correct/incorrect 都没有 —— 前四层全 miss：
#   "The student's response is incoherent."            ×48
#   "The provided answer is incomplete."               ×66
#   "The student's response is a repetition of the pr" ×39
#   "The student's response is completely irrelevant." ×55
#   "The expected answer is not provided by the model." ×26
# 这些是**完整且正确**的判定（那几个回答确实是复读/不切题），只是格式不同。
# 走到这一层意味着下面只剩 fallback，而 omni_mix 没有规则打分器
# （default_compute_score 抛 NotImplementedError）→ 兜底恒为 0.0。
# 于是判分塌掉和判对在指标里长得一模一样，GRPO 整组奖励全 0、优势为 0、这一步的
# 梯度被丢掉 —— 这正是「rollout 一直复读」在训练侧的样子。
#
# 这一层**只会把「降级成硬 0」变成「按判定给分」**：它排在前四层之后，凡是已经能
# 解析的文本都不受影响。取最后一次出现（同前四层的「读最后的表态」原则）。
_PHRASE_VERDICTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"\b(?:incoherent|irrelevant|incomplete|inconsistent|nonsensical|unintelligible|unrelated)\b", re.I), 0.0),
    # 「学生答案根本没交上来」也是判定，而且是**唯一**一种判官从第一句就给准的形态
    # （genrm_dump_qwen36/sample_0001.json："The student's response is empty. No answer
    # was provided to evaluate."，随后才塌成 "an an an an …"）。判它 0.0 是照实记分。
    (re.compile(r"\b(?:response|answer|output|solution)\s+(?:is|was)\s+empty\b", re.I), 0.0),
    (re.compile(r"\bno\s+(?:answer|response|output|attempt)\s+was\s+(?:provided|given|made)\b", re.I), 0.0),
    (re.compile(r"\bis\s+(?:a\s+|an\s+|the\s+)?(?:repetition|repeat|copy|restatement|garbled)\b", re.I), 0.0),
    (re.compile(r"\b(?:no|not\s+a|not\s+an)\s+valid\s+(?:answer|response|attempt)\b", re.I), 0.0),
    (re.compile(r"\b(?:fails?|failed)\s+to\s+(?:provide|answer|address|state|give|mention|match)\b", re.I), 0.0),
    (re.compile(r"\b(?:does|do)\s+not\s+(?:constitute|provide|answer|address|match|contain|explain|correspond)\b", re.I), 0.0),
    (re.compile(r"\bis\s+not\s+(?:provided|present|given)\b", re.I), 0.0),
    (re.compile(r"\bnot\s+semantically\s+equivalent\b", re.I), 0.0),
    # "is not an equivalent answer" / "is not the expected answer" 这类没有
    # correct/incorrect 字样的否定判定（优先级 4 看不到，否则就漏成 fallback → 硬 0）。
    (re.compile(
        r"\b(?:is|are|was|were|does|do)\s+not\s+(?:the\s+|a\s+|an\s+)?"
        r"(?:equivalent|expected|matching|accurate|acceptable|valid)\b",
        re.I,
    ), 0.0),
    (re.compile(
        r"\b(?:semantically\s+equivalent|equivalent\s+to\s+the\s+(?:reference|ground)|"
        r"matches\s+the\s+(?:reference|ground\s*truth)|"
        r"agrees?\s+with\s+the\s+(?:reference|ground\s*truth)|"
        r"consistent\s+with\s+the\s+(?:reference|ground\s*truth)|"
        r"captures\s+the\s+(?:same|key)\b)\b",
        re.I,
    ), 1.0),
)

# 注意这里**故意不收**「judge 自己没法判」的措辞（"cannot be evaluated"、
# "the student's response is missing from the input"、sample_0003 那种「看不懂自己被
# 喂了什么」）。那些不是判定，是 judge 侧的故障：给它们 0.0 等于把「没判」记成
# 「判错」，还把 genrm_parsed 刷成 1.0 盖住故障。它们继续走 fallback，保留
# fallback_reason 那条可 grep 的线索。


def _flip_if_negated(text: str, start: int, value: float) -> float:
    """匹配词前面有否定时翻转判定（"not correct" → 0.0，"not incorrect" → 1.0）。"""
    if _NEGATED_BEFORE_RE.search(text[:start]):
        return 1.0 - value
    return value


def _parse_verdict(judge_text: str) -> Optional[float]:
    """Map judge text to ``1.0`` / ``0.0``; ``None`` when no verdict is found.

    Reads the *last* match, not the first: an instruct model that ignores the
    output format tends to restate the task ("decide whether it is correct")
    before committing, and the commitment is at the end.

    Priority order:
    1. Explicit "Verdict: correct/incorrect" — the prompt asks for this format
    2. Last standalone line that is ONLY "correct"/"incorrect"
    3. "correctly"/"incorrectly" — judge writes "correctly identifies"
    4. Full-text \b(correct|incorrect)\b over lines that don't read like a
       restatement of the task ("wants" / "user" / "model")
    """
    if not judge_text:
        return None

    judge_text = extract_final_answer(judge_text, strip_think=True)

    # Priority 1: explicit "Verdict: correct/incorrect"
    matches = _VERDICT_RE.findall(judge_text)
    if matches:
        return 0.0 if matches[-1].lower() == "incorrect" else 1.0

    # Priority 2: last standalone line that is ONLY "correct"/"incorrect"
    standalone_matches = _STANDALONE_LINE_RE.findall(judge_text)
    if standalone_matches:
        return 0.0 if standalone_matches[-1].lower() == "incorrect" else 1.0

    # Priority 3: "correctly" / "incorrectly"
    matches = list(_CORRECTLY_RE.finditer(judge_text))
    if matches:
        last = matches[-1]
        return _flip_if_negated(judge_text, last.start(), 0.0 if last.group(1).lower() == "incorrectly" else 1.0)

    # Priority 4: \b(correct|incorrect)\b in full text, but skip lines that are
    # restating the task rather than answering it ("The user wants to know if the
    # model is correct"). The filter is deliberately narrow: "answer is" /
    # "response is" used to be in this list, and that made the single most common
    # judge sentence -- "The student's answer is incorrect." -- parse as no
    # verdict at all, which degrades the sample to the rule-based scorer and
    # turns a real 0 into an untraceable one.
    lines = judge_text.split("\n")
    filtered_matches = []
    offset = 0
    for line in lines:
        line_lower = line.lower()
        if not any(w in line_lower for w in ("wants", "user", "model")):
            for m in _BARE_VERDICT_RE.finditer(line):
                # 指代参考答案的 "the correct answer" 不算判词（见 _MENTION_AFTER_RE）。
                # 两种例外：
                #   · 前面有否定（"is **not** the correct answer"）→ 是判词，翻成 0.0；
                #   · 前面有系动词（"the student's answer **is** the correct answer"）
                #     → 是表语判定，不是指代。
                before = judge_text[: offset + m.start()]
                if _MENTION_AFTER_RE.match(line, m.end()) and not (
                    _NEGATED_BEFORE_RE.search(before) or _COPULA_BEFORE_RE.search(before)
                ):
                    continue
                filtered_matches.append((offset + m.start(), m.group(1), m.start()))
        offset += len(line) + 1
    if filtered_matches:
        pos, word, in_line = filtered_matches[-1]
        return _flip_if_negated(judge_text[:pos], in_line, 0.0 if word.lower() == "incorrect" else 1.0)

    # Priority 5: judge 用散文给判定（没有 "Verdict:" 也没有 correct/incorrect 字样）。
    # 详见 _PHRASE_VERDICTS 上的注释：这一层只把「硬 0」换成「按判定给分」。
    best: Optional[tuple[int, float]] = None
    for regex, value in _PHRASE_VERDICTS:
        for m in regex.finditer(judge_text):
            scored = _flip_if_negated(judge_text, m.start(), value)
            if best is None or m.start() >= best[0]:
                best = (m.start(), scored)
    if best is not None:
        return best[1]

    return None


# 覆盖率模式的判分行：``Score: 3/4``（大小写不敏感、容忍空格与全角斜杠后空格）。
_SCORE_RE = re.compile(r"\bscore\s*[:：]\s*\*{0,2}\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)

# ⚠️ 故意**不**采信「没有分母的裸 ``Score: 0``」（2026-10-13 权衡后否决了这条 salvage）。
# 真机上判官最常见的塌缩形态是「写完 ``Score: 0`` 就进 ``0 0 0…`` 死循环」，
# 把这个 0 采信成判定会让 genrm_parsed 虚假回升（90%→95%），把「判官塌了」粉饰成
# 「判官在判」。正确姿势是让它继续走 fallback，由 ``GENRM_FALLBACK_ON_ERROR=
# similarity`` 给一个粗粒度的语义相似度分（见 :func:`_lexical_similarity`）——
# 「Score: 0」到底是「坦率的 0」还是「判官放弃」本就无法区分，不该由解析层假装知道。


def _parse_score(judge_text: str) -> Optional[float]:
    """Parse ``Score: <covered>/<total>`` into ``covered/total``.

    和 ``_parse_verdict`` 一样取**最后一个**匹配：instruct 模型先复述任务再给结论
    是常态，结论在后面。``total`` 为 0 视为「判官没按格式给」（返回 None，走
    fallback），``covered > total`` 钳到 1.0。
    """
    if not judge_text:
        return None
    matches = _SCORE_RE.findall(judge_text)
    if not matches:
        return None
    covered, total = (int(x) for x in matches[-1])
    if total <= 0:
        return None
    return min(covered, total) / total


def _parse_reward(judge_text: str) -> Optional[float]:
    """Dispatch judge text: ``Score: k/N`` first, then the legacy Verdict wording.

    Score 行优先与模式无关：判官哪怕被切回 verdict 模式，吐了 Score 行也按
    Score 算（多给一层兼容没有坏处——两个命名空间不重叠）。反过来，覆盖率
    模式下判官退化成旧措辞（或复读）时自动落回 ``_parse_verdict``，行为与
    旧版本完全一致。
    """
    score = _parse_score(judge_text)
    if score is not None:
        return score
    return _parse_verdict(judge_text)


def _scoring_mode() -> str:
    """``coverage``（默认，要点覆盖率 k/N）或 ``verdict``（二元，旧行为对照）。"""
    return os.environ.get("GENRM_SCORING_MODE", "coverage").strip().lower()


# ---------------------------------------------------------------------------
# Router I/O
# ---------------------------------------------------------------------------

_model_name_cache: dict[str, str] = {}
_model_name_lock: Optional[asyncio.Lock] = None


def _get_model_name_lock() -> asyncio.Lock:
    global _model_name_lock
    if _model_name_lock is None:
        _model_name_lock = asyncio.Lock()
    return _model_name_lock


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _sampling_params() -> dict[str, Any]:
    """Judge sampling params. Greedy by default -- a grader should not be creative.

    ``GENRM_MAX_TOKENS`` 默认 **96**（2026-10-13 从 1024 下调）。1024 的出台背景是
    「判官在 512 里写不完思考」；但 2026-10-13 真机量出来，判官的主病态不是预算不够，
    而是**判词首行之后立刻塌进 ``0 0 0…`` 复读**——618 次调用里 410 次（66%）把 1024
    烧满，尾巴全是复读，一条都没多判出内容。首行判词（``Score: k/N``）只要 ~10 个
    token；96 给「判词行 + 两个短句」留足余地，截尾不损首行判定（塌缩的尾巴本来
    就没有信息）。

    ``GENRM_FREQUENCY_PENALTY`` 默认 **0.5**（从 0.0 上调）：repetition_penalty 只看
    「这个 token 出现过没有」（非累计），压不住 ``0``/``an`` 这类 2-gram 环；
    frequency_penalty 按出现次数累计扣 logits，是本文件里唯一能直接打断环的旋钮
    （文件内多处注释与真机 dump 均如此认定）。首行生成时所有 token 计数都是 0，
    不受该惩罚影响，所以它只动「环」，不动「判」。

    两条旋钮都有 env 覆盖；``GENRM_STOP_AT_FIRST_LINE``（默认开）还会在判词首行
    的换行处直接停止生成（见 ``VERDICT_STOP_STRINGS`` 下方的注释），三件套同时
    生效时判官单次调用从 ~1000 token 降到 ~10-20 token。
    """
    params: dict[str, Any] = {
        "temperature": _env_float("GENRM_TEMPERATURE", 0.0),
        "top_p": _env_float("GENRM_TOP_P", 1.0),
        "max_tokens": _env_int("GENRM_MAX_TOKENS", 96),
        "repetition_penalty": _env_float("GENRM_REPETITION_PENALTY", 1.1),
        "frequency_penalty": _env_float("GENRM_FREQUENCY_PENALTY", 0.5),
        "presence_penalty": _env_float("GENRM_PRESENCE_PENALTY", 0.0),
    }
    if "GENRM_SEED" in os.environ:
        params["seed"] = _env_int("GENRM_SEED", 0)
    return params


async def _resolve_model_name(session: aiohttp.ClientSession, router_address: str) -> str:
    """Return the name the judge server answers to.

    vLLM registers the served model under the path it was launched with
    (``vllm serve <model_config.local_path>``), and ``RewardModelManager`` builds that
    from ``reward.reward_model.model_path``. Those two agree for a local checkpoint
    directory but not for a remote URI that ``copy_to_local`` stages elsewhere, so ask
    the server instead of reconstructing the name -- a mismatch is a 404 per sample,
    every step, with nothing in the trainer log pointing at the cause.
    """
    override = os.environ.get("GENRM_MODEL_NAME")
    if override:
        return override

    if router_address in _model_name_cache:
        return _model_name_cache[router_address]

    async with _get_model_name_lock():
        if router_address in _model_name_cache:
            return _model_name_cache[router_address]
        async with session.get(f"http://{router_address}/v1/models") as resp:
            resp.raise_for_status()
            payload = json.loads(await resp.text())
        model_name = payload["data"][0]["id"]
        _model_name_cache[router_address] = model_name
        logger.info("[genrm_verifier] resolved judge model name %r from %s", model_name, router_address)
        return model_name


class _ChatUnsupported(Exception):
    """The judge server has no working ``/v1/chat/completions`` handler."""


# vLLM answers a request for an endpoint whose serving object was never constructed
# with 400 and this wording, not 404 -- the route is always registered, the handler
# is not. Whether vLLM-Omni's ``omni_init_app_state`` builds the chat handler in AR
# mode is not verifiable from the authoring machine (no vllm_omni there), and the
# actor rollout never exercises it: it generates from token ids, not /v1/chat.
_CHAT_UNSUPPORTED_MARKERS = ("does not support Chat Completions", "not support chat")


async def _chat_complete(session: aiohttp.ClientSession, router_address: str, payload: dict) -> tuple[str, str]:
    """POST one chat completion and return ``(content, finish_reason)``.

    ``finish_reason`` is carried out because it separates two failures that look
    identical in the text: the judge wrote a verdict we could not parse, versus the
    judge never finished -- ``length`` means it burned the whole ``GENRM_MAX_TOKENS``
    budget, which is what a judge that collapsed into repetition does.

    Retries on timeouts, connection errors and 5xx. Two 400s are handled specially
    instead of being propagated:

    * one naming ``chat_template_kwargs`` -- retried without that field, since older
      vLLM builds reject it and the judge grades fine without it;
    * one saying chat completions are unsupported -- raised as :class:`_ChatUnsupported`
      so the caller can render the template itself and use ``/v1/completions``.
    """
    url = f"http://{router_address}/v1/chat/completions"
    max_retries = _env_int("GENRM_MAX_RETRIES", 5)
    payload = dict(payload)
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            _trace_http = _rollout_diag.limited("genrm_http")
            if _trace_http:
                _rollout_diag.emit("genrm_http_request", trace_id=_JUDGE_TRACE_ID.get(),
                                   endpoint="chat", attempt=attempt, payload=payload)
            async with session.post(url, json=payload) as resp:
                if resp.status == 400:
                    body = await resp.text()
                    if any(marker.lower() in body.lower() for marker in _CHAT_UNSUPPORTED_MARKERS):
                        raise _ChatUnsupported(body[:200])
                    if "chat_template_kwargs" in payload and (
                        "chat_template_kwargs" in body or "enable_thinking" in body
                    ):
                        logger.warning(
                            "[genrm_verifier] server rejected chat_template_kwargs; retrying without it: %s",
                            body[:200],
                        )
                        payload.pop("chat_template_kwargs")
                        continue
                    if "include_stop_str_in_output" in payload and "include_stop_str_in_output" in body:
                        # 老版本 vLLM 可能不认这个字段。去掉它是**有损**的（Verdict 行
                        # 又会被 stop 串抹掉），但整个 judge 挂掉更贵：那样每条样本都
                        # 走 fallback。降级要看得见。
                        logger.warning(
                            "[genrm_verifier] server rejected include_stop_str_in_output; "
                            "retrying without it (Verdict 行会被 stop 串吃掉): %s",
                            body[:200],
                        )
                        payload.pop("include_stop_str_in_output")
                        continue
                resp.raise_for_status()
                output = json.loads(await resp.text())
            choice = output["choices"][0]
            if _trace_http:
                _rollout_diag.emit("genrm_http_response", trace_id=_JUDGE_TRACE_ID.get(), endpoint="chat", attempt=attempt,
                                   response_id=output.get("id"), choice=choice, usage=output.get("usage"))
            return (choice["message"]["content"] or ""), str(choice.get("finish_reason") or "")
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                raise
            last_exc = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_exc = e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Malformed body: the server answered but not with a completion. Retrying
            # a deterministic parse failure is pointless.
            raise RuntimeError(f"Unexpected chat completion payload from {url}: {e}") from e

        if attempt < max_retries - 1:
            await asyncio.sleep(min(2**attempt, 30))

    raise RuntimeError(f"GenRM request to {url} failed after {max_retries} attempts") from last_exc


async def _text_complete(
    session: aiohttp.ClientSession,
    router_address: str,
    payload: dict,
    tokenizer: Any,
) -> tuple[str, str]:
    """Render the chat template locally and grade through ``/v1/completions``.

    Returns ``(text, finish_reason)``; see :func:`_chat_complete` for why the
    finish reason is carried out.

    The fallback for a server without a chat handler. ``RewardLoopWorker`` already
    loads the judge's tokenizer and hands it to the scorer
    (``reward_loop.py:128-135``), so the template is available client side at no
    extra cost.
    """
    if tokenizer is None:
        raise _ChatUnsupported("no chat endpoint and no reward_model_tokenizer to render the template with")

    payload = dict(payload)
    messages = payload.pop("messages")
    template_kwargs = payload.pop("chat_template_kwargs", {}) or {}
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    except TypeError:
        # Templates that do not declare enable_thinking reject the kwarg outright.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    url = f"http://{router_address}/v1/completions"
    max_retries = _env_int("GENRM_MAX_RETRIES", 5)
    payload["prompt"] = prompt
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            _trace_http = _rollout_diag.limited("genrm_http")
            if _trace_http:
                _rollout_diag.emit("genrm_http_request", trace_id=_JUDGE_TRACE_ID.get(),
                                   endpoint="completions", attempt=attempt, payload=payload)
            async with session.post(url, json=payload) as resp:
                if resp.status == 400 and "include_stop_str_in_output" in payload:
                    body = await resp.text()
                    if "include_stop_str_in_output" in body:
                        # 见 _chat_complete 同名分支：有损降级，但比 judge 全挂便宜。
                        logger.warning(
                            "[genrm_verifier] /v1/completions rejected include_stop_str_in_output; "
                            "retrying without it (Verdict 行会被 stop 串吃掉): %s",
                            body[:200],
                        )
                        payload.pop("include_stop_str_in_output")
                        continue
                resp.raise_for_status()
                output = json.loads(await resp.text())
            choice = output["choices"][0]
            if _trace_http:
                _rollout_diag.emit("genrm_http_response", trace_id=_JUDGE_TRACE_ID.get(), endpoint="completions", attempt=attempt,
                                   response_id=output.get("id"), choice=choice, usage=output.get("usage"))
            return (choice["text"] or ""), str(choice.get("finish_reason") or "")
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                raise
            last_exc = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_exc = e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unexpected completion payload from {url}: {e}") from e

        if attempt < max_retries - 1:
            await asyncio.sleep(min(2**attempt, 30))

    raise RuntimeError(f"GenRM request to {url} failed after {max_retries} attempts") from last_exc


async def _judge(
    session: aiohttp.ClientSession,
    router_address: str,
    payload: dict,
    tokenizer: Any,
) -> tuple[str, str]:
    """Ask the judge, preferring ``/v1/chat/completions``; returns ``(text, finish_reason)``.

    ``GENRM_ENDPOINT`` pins the choice (``chat`` or ``completions``) once the answer
    is known for a given deployment, so the probe is not repeated per sample.
    """
    endpoint = os.environ.get("GENRM_ENDPOINT", "auto").strip().lower()
    if endpoint == "completions":
        return await _text_complete(session, router_address, payload, tokenizer)

    try:
        return await _chat_complete(session, router_address, payload)
    except _ChatUnsupported as e:
        if endpoint == "chat":
            raise
        logger.warning(
            "[genrm_verifier] judge has no chat endpoint (%s); rendering the template "
            "locally and retrying on /v1/completions. Set GENRM_ENDPOINT=completions to "
            "skip this probe.",
            e,
        )
        return await _text_complete(session, router_address, payload, tokenizer)


# ---------------------------------------------------------------------------
# Degradation diagnostics
# ---------------------------------------------------------------------------
#
# 这两个计数器存在的理由是「静默失败比崩溃贵」。降级路径原先只写一行 warning
# 或者干脆 `except Exception: return 0.0`，日志里既看不出降级多频繁，也看不出
# 降级之后那个 0 到底是「判错了」还是「压根没判」——而 RL 的失效形态恰恰是
# 「数字慢慢不对」：整组奖励全 0 → GRPO 优势为 0 → 这一步的梯度被丢掉。

_FALLBACK_STATS: dict[str, int] = {}
_FALLBACK_STATS_TRACE_LIMIT = 5

# judge 每次调用的 finish_reason 分布：``length`` = 把预算烧满（塌缩的典型收尾）、
# ``stop`` = 自己停下、``error`` = 请求失败。2026-10-13：618 次调用里 410 次
# ``length``，尾巴全是 ``0 0 0…`` 复读 —— 这是判分塌缩率的一手指标，
# 与 genrm_parsed / fallback 率互相独立，任何单边改善都不会把它盖住。
_JUDGE_FINISH_STATS: dict[str, int] = {}

# 复读检测。实测有两种形态，周期差了两个数量级，必须分别抓：
#   * actor：打满预算时尾部死循环「平安、平安、平安…」—— 同一行内的 2-gram 复读，
#     这是「思考没写完就撞上 cap」的指纹，而模型在复读里编出题目里根本不存在
#     的人名/年份，正是幻觉的第一现场。
#   * judge：整句重复「The user provided the text.」几百遍 —— 周期 27 字符，
#     超出了下面这个 8 字符的窗口，只能按行抓。
# 两者都只会让判分失败（fallback），而 fallback 在指标里和「判错」长得一样。
_REPEAT_TEXT_RE = re.compile(r"(.{2,8}?)\1{3,}", re.DOTALL)
_LINE_REPEAT_MIN = 3  # 连续相同的非空行 >= 3 次才算复读


def _bump(reason: str) -> int:
    _FALLBACK_STATS[reason] = _FALLBACK_STATS.get(reason, 0) + 1
    return _FALLBACK_STATS[reason]


def _stats_line() -> str:
    if not _FALLBACK_STATS:
        return "fallback_stats={}"
    return "fallback_stats=" + ",".join(f"{k}:{v}" for k, v in sorted(_FALLBACK_STATS.items()))


# ---------------------------------------------------------------------------
# 奖励链健康度报警
# ---------------------------------------------------------------------------
#
# 一句话背景：本仓库最贵的一次事故不是崩溃，而是**整批 reward 恒 0 却没人发现**。
# 2026-10-10 夜的真机形态：judge 16/16 个样本都没吐出 Verdict → 全部落 fallback →
# omni_mix 没有规则打分器（NotImplementedError）→ 每个样本 score=0.0 →
# 同一组内奖励全相等 → GSPO/GRPO 优势为 0 → 这一步的梯度被**静默**丢掉。
# 训练照跑、loss 照打、日志里只有几行 WARNING，看上去和「模型训得慢」一模一样。
#
# 单看一个样本永远发现不了这件事（判错、判分失败、真·全错，三者都是 0.0）。
# 能发现它的是**分布**：连续 N 个样本全是同一个值，或者全是 fallback。
# 所以这里按样本计数，跨过阈值就打一条显眼的横幅。
_VERDICT_TALLY: dict[str, int] = {}
_ALERT_MIN_SAMPLES = 8
_ALERTED: set[str] = set()
# 每个 worker 进程只打一次真正发给 judge 的 prompt（见 compute_score 里的打印）。
_PROMPT_LOGGED = False


def _tally(verdict: Optional[float], fb_reason: str) -> None:
    if verdict is None:
        key = f"fallback:{fb_reason or 'unknown'}"
    else:
        key = "correct" if verdict >= 0.5 else "incorrect"
    _VERDICT_TALLY[key] = _VERDICT_TALLY.get(key, 0) + 1


def _tally_line() -> str:
    if not _VERDICT_TALLY:
        return "verdict_tally={}"
    total = sum(_VERDICT_TALLY.values())
    body = ",".join(f"{k}:{v}" for k, v in sorted(_VERDICT_TALLY.items(), key=lambda kv: -kv[1]))
    return f"verdict_tally={body} (n={total})"


def _judge_finish_line() -> str:
    """本进程 judge finish_reason 的分布（可 grep 的塌缩率）。"""
    if not _JUDGE_FINISH_STATS:
        return "judge_finish_tally={}"
    body = ",".join(
        f"{k}:{v}" for k, v in sorted(_JUDGE_FINISH_STATS.items(), key=lambda kv: -kv[1])
    )
    return f"judge_finish_tally={body}"


def _reward_health_check(data_source: str, min_samples: int = 0) -> Optional[str]:
    """奖励分布看起来「整批坏掉」时打横幅；返回横幅文本（没触发则 None）。

    三条判据，都是「分布」而不是「单个样本」：
      1. ≥N 个样本全部来自 fallback           —— 判分链路没在工作（judge 挂了/没接上）；
      2. ≥N 个样本全部同一个 verdict           —— judge 退化成常量（复读/塌缩的常见结局）；
      3. ≥N 个样本全部 fallback 且都是同一原因 —— 同上，但连原因都固定（多半是配置问题）。

    只报一次（按 reason 去重）：这个函数挂在每个样本的热路径上，不刷屏。
    """
    n_needed = min_samples or _env_int("GENRM_ALERT_MIN_SAMPLES", _ALERT_MIN_SAMPLES)
    total = sum(_VERDICT_TALLY.values())
    if total < n_needed:
        return None

    kinds = list(_VERDICT_TALLY)
    alert = ""
    if all(k.startswith("fallback:") for k in kinds):
        alert = f"整批 {total} 个样本全部走了 fallback（判分链路没在工作）"
    elif len(kinds) == 1 and not kinds[0].startswith("fallback:"):
        alert = f"整批 {total} 个样本的 verdict 恒为 {kinds[0]}（judge 退化成常量）"

    if not alert:
        return None
    key = alert.split("（")[0]
    if key in _ALERTED:
        return None
    _ALERTED.add(key)

    line = (
        f"[genrm_verifier] REWARD-SUSPECT ⚠️  data_source={data_source} {alert}；"
        f"{_tally_line()} {_stats_line()} —— 同组奖励全相等时 GSPO/GRPO 优势为 0，"
        f"这一步的梯度会被丢掉（训练看着照跑，其实没在学）。"
        f"先查 judge（[judge-diag] 的 finish / 复读指纹）与 fallback_reason。"
    )
    _rollout_diag.console_or_file("genrm_reward_suspect", line)
    if not _rollout_diag.enabled():
        logger.warning(line)
    return line


def _repeat_hint(text: str, limit: int = 48) -> str:
    """文本里复读片段的指纹（先按行抓整句重复，再按字符抓短周期）；没有则返回空串。

    行级优先：整句复读的周期远超 ``_REPEAT_TEXT_RE`` 的 8 字符窗口，字符级那条
    对 judge 那种塌法完全看不见。反过来，actor 的「平安、平安…」是同一行内的
    n-gram，行级也看不见 —— 所以两条都要。
    """
    if not text:
        return ""

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    best = None  # (line, run)
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= _LINE_REPEAT_MIN and (best is None or run > best[1]):
            best = (lines[i], run)
        i = j
    if best is not None:
        return f"repeat={best[0][:limit]!r}x{best[1]}"

    hit = _REPEAT_TEXT_RE.search(text)
    if hit is None:
        return ""
    unit = hit.group(1)
    repeats = (hit.end() - hit.start()) // max(len(unit), 1)
    return f"repeat={unit[:limit]!r}x{repeats}"


def _think_closed(text: str) -> bool:
    return bool(_THINK_CLOSE_RE.search(text or ""))


def _answer_origin(solution_str: str, answer: str) -> str:
    """answer 是从 response 的哪一段取出来的。

    ``truncated_think`` 是幻觉的旗标：response 打满预算、``</think>`` 从未出现，
    于是交给 judge 的「答案」其实是被截断的推理中段。judge 只会判它 incorrect，
    但真正的问题在 actor 侧的预算，不在 judge —— 所以这一档单独标出来。
    """
    if not (solution_str or "").strip():
        return "empty_response"
    if _think_closed(solution_str):
        return "after_think"
    if _THINK_OPEN_RE.search(solution_str):
        return "truncated_think"
    return "plain"


def _answer_diag(solution_str: str, answer: str, tag: str) -> str:
    """打印一行 answer 侧诊断（截断 / 复读 / 空的指纹）并返回该行。"""
    resp = solution_str or ""
    origin = _answer_origin(resp, answer)
    parts = [
        f"[answer-diag] {tag}",
        f"resp_chars={len(resp)}",
        f"answer_chars={len(answer or '')}",
        f"answer_empty={not (answer or '').strip()}",
        f"origin={'!!' + origin if origin == 'truncated_think' else origin}",
        f"think_closed={_think_closed(resp)}",
    ]
    hint = _repeat_hint(resp)
    if hint:
        parts.append(hint)
    line = " ".join(parts)
    _rollout_diag.console_or_file("genrm_answer_diagnostic", line)
    return line


def _judge_diag(judge_text: str, judge_error: str, judge_finish: str = "") -> str:
    """打印一行 judge 侧诊断并返回该行。

    判分失败的原因绝大多数不是「判错」，而是 judge 自己的生成塌了 —— 实测
    sample_0001 的 judge_text 是「The user provided the text.」重复几百遍，
    一个 Verdict 都没吐。这几乎必然是上游喂进去的复读回答把它带塌的（垃圾进、
    垃圾出），但两者必须分别可见：只骂 judge 会去调 judge 的 prompt，只骂 actor
    又会漏掉「judge 的 token 预算也打满了」这一层 —— 那一点由 ``finish=length``
    给出，只看文本长度是猜不出来的。
    """
    text = judge_text or ""
    parts = [
        "[judge-diag]",
        f"chars={len(text)}",
        f"think_closed={_think_closed(text)}",
    ]
    if judge_finish:
        parts.append(f"finish={judge_finish!r}")
    if judge_error:
        parts.append(f"error={judge_error!r}")
    hint = _repeat_hint(text)
    if hint:
        # judge 复读是「没有 Verdict」的直接原因，标出来省一次人工翻 dump。
        parts.append("!! " + hint)
    line = " ".join(parts)
    _rollout_diag.console_or_file("genrm_judge_diagnostic", line)
    return line


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _lexical_similarity(ground_truth: str, answer: str) -> float:
    """字符 2-gram Dice 系数 —— judge 塌缩时的粗粒度语义相似度兜底。

    [2026-10-13 新增，配合 ``GENRM_FALLBACK_ON_ERROR=similarity``]
    出台背景：判官 Qwen3.5-35B-A3B 在真机上大面积「写完 ``Score: 0`` 就复读塌缩」
    （618 次调用 410 次 finish=length），一切无法解析的样本原先全部落 `rule` 兜底，
    而 omni_mix 的规则打分器直接 NotImplementedError ⇒ 每题恒 0 ⇒ 同组奖励全相等
    ⇒ GSPO 优势为 0 ⇒ 这一步白跑（真机实测 critic/score/mean ≈ 0.008）。
    用户侧的对齐预期是「不看题干、直接量 ground truth 与 rollout 答案的语义相似度」，
    本函数就是这句话的最小实现。

    为什么用字符 2-gram Dice（而不是词嵌入/编辑距离）：
      · 零依赖（stdlib）、对中英混排都成立、rollout 被打断的答案不会抛错；
      · Dice 同时度量覆盖与噪声，2-gram 对中文近似「字的搭配」，
        「张小龙」这种关键实体只要出现就能贡献交集；
      · 它只是**兜底粗分**：用于把「恒 0 死锁」变成「有梯度的高低差」，
        不用于替代 judge —— 判官能出分时永远优先判官（调用方逻辑不变）。

    已知代价（明确接受）：长参考答案 vs 短正确回答的 Dice 偏低（GT 的冗余描写
    没有对应物）；语义改写（同义不同字）量不出来。所以 fallback_reason 会记
    ``similarity``、``genrm_fallback=1.0`` 保持可见 —— 它始终是"降级信号"，
    不是"判过的分"。
    """
    def _grams(s: str) -> set:
        s = "".join(ch for ch in (s or "").lower() if not ch.isspace())
        if not s:
            return set()
        if len(s) == 1:
            return {s}
        return {s[i:i + 2] for i in range(len(s) - 1)}

    g_gt, g_ans = _grams(ground_truth), _grams(answer)
    if not g_gt or not g_ans:
        return 0.0
    inter = len(g_gt & g_ans)
    return 2.0 * inter / (len(g_gt) + len(g_ans))


def _fallback_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict]
) -> tuple[float, str]:
    """Score without the judge, for when the judge is unreachable or unparseable.

    ``GENRM_FALLBACK_ON_ERROR`` 三种取值：

    * ``rule``（默认，历史行为）：退回本仓库的规则打分器。对 gsm8k 之类的
      data_source 是合理兜底，但 **omni_mix 会直接 NotImplementedError → 恒 0.0**，
      等价于「没有兜底」。
    * ``zero``：显式退回硬 0（在组内制造恒等奖励，梯度被丢弃——仅供对照）。
    * ``similarity``（推荐 omni_mix 用）：**直接量 ground truth 与 answer 的
      字符 2-gram Dice 相似度**（见 :func:`_lexical_similarity`）。判官塌缩时
      给出一个粗但有梯度的分，把「整组恒 0 → 优势恒 0 → 这一步白跑」变成
      「组内有高低差 → 优势非零 → 至少还有可学的信号」。

    无论走哪条，调用方都会记 ``genrm_fallback=1.0``，降级率在指标里保持可见；
    ``similarity`` 也**不是判过的分**，它只是止损。

    Returns ``(score, reason)``: ``reason`` is ``""`` when the rule-based scorer
    actually answered, otherwise a short tag naming why this score is a placeholder.
    The tag travels into the sample dump, so a dump directory full of
    ``rule_failed:NotImplementedError`` looks different from one full of judge
    timeouts -- previously both were an untraceable ``0.0``.
    """
    _mode = os.environ.get("GENRM_FALLBACK_ON_ERROR", "rule").strip().lower()
    if _mode == "zero":
        n = _bump("env_zero")
        _rollout_diag.console_or_file(
            "genrm_reward_degraded",
            f"[genrm_verifier] REWARD-DEGRADED #{n} (GENRM_FALLBACK_ON_ERROR=zero) "
            f"data_source={data_source} score=0.0 {_stats_line()}",
            flush=True,
        )
        return 0.0, "env_zero"
    if _mode == "similarity":
        # 与判官同口径：相似度也只看 extract_final_answer 之后的「答案段」，
        # 否则思考段与 GT 的词汇重叠会把分数撑高（思考里充斥题干词）。
        answer = extract_final_answer(solution_str, strip_think=_env_flag("GENRM_STRIP_THINK", True))
        score = round(_lexical_similarity(ground_truth, answer), 4)
        n = _bump("similarity")
        _rollout_diag.console_or_file(
            "genrm_reward_degraded",
            f"[genrm_verifier] REWARD-DEGRADED #{n} (similarity) data_source={data_source} "
            f"score={score} gt_chars={len(ground_truth or '')} ans_chars={len(answer or '')} "
            f"{_stats_line()}",
            flush=True,
        )
        return score, "similarity"
    try:
        from verl.utils.reward_score import default_compute_score

        result = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return float(result["score"] if isinstance(result, dict) else result), ""
    except Exception as e:  # noqa: BLE001 - the fallback must not raise
        # 这里吞掉的通常不是「网络抖了一下」，而是「默认打分器根本没有这个
        # data_source 的分支」（例如 omni_mix 会抛 NotImplementedError）——
        # 那是每一步、每个样本都必然发生的确定性 0，不是偶发。计数 + 打出原因。
        reason = f"rule_failed:{type(e).__name__}"
        n = _bump(reason)
        _rollout_diag.console_or_file(
            "genrm_reward_degraded",
            f"[genrm_verifier] REWARD-DEGRADED #{n} ({reason}) data_source={data_source} "
            f"score=0.0 ex={e} {_stats_line()}",
            flush=True,
        )
        if not _rollout_diag.enabled():
            logger.warning("[genrm_verifier] rule-based fallback failed for %s: %s", data_source, e)
        return 0.0, reason


# judge 是「读文本」的：题干里的 <image>/<audio>/<video> 对它是**没有内容的**
# 一段占位文本。真机 dump 里最刺眼的一类就是音频样本 —— omni_mix 的音频题把题目
# 念在音频里，``question`` 字段本身就是 7 个字符的 ``'<audio>'``，于是 judge 收到
# 的 [Problem] 只有「<audio>」，它的反应是复述输入并纠结「题目缺失、无法评测」：
#     sample_0004: "The student's response is missing from the input, so I cannot
#                   evaluate the student's answer against the reference."
#     sample_0003: "The user provided a prompt containing a question about a chart
#                   and a long, repetitive string of text that appears to be a glitch"
# 这些是 judge 侧故障（不是判定），既拿不到 Verdict，也把 512 token 烧在复述上。
# 换成人话后，judge 至少知道「这块媒体我看不见，请只按参考答案判」。
_MODALITY_PLACEHOLDER_RE = re.compile(
    r"<\|?(?:image_pad|audio_pad|video_pad|vision_start|vision_end|image_start|image_end|"
    r"audio_start|audio_end|video_start|video_end|image|audio|video)\|?>",
    re.IGNORECASE,
)
_PLACEHOLDER_NAMES = {"image": "图片", "audio": "音频", "video": "视频"}


def _humanize_modality_placeholders(question: str) -> str:
    """把模态占位符换成人话，并说明这块媒体 judge 看不到。"""

    def _sub(m):
        tag = m.group(0).strip("<|>").lower()
        for name, cn in _PLACEHOLDER_NAMES.items():
            if tag.startswith(name):
                return f"[{cn}（判分时不可见）]"
        return "[多模态内容（判分时不可见）]"

    return _MODALITY_PLACEHOLDER_RE.sub(_sub, question)


def _question_of(extra_info: Optional[dict]) -> str:
    """Recover the problem statement the judge needs to see.

    ``omni_mix.py`` writes ``extra_info["question"]`` (the user's original question
    including placeholders like <image>/<audio>/<video>). The other keys are covers
    for datasets built elsewhere; without any of them the judge still gets the
    reference answer, which carries most of the signal.

    Placeholders are replaced with plain language (see
    ``_humanize_modality_placeholders``); when nothing but placeholders is left, the
    judge is told explicitly that the statement is not available, so it stops
    hunting for a problem statement that was never in the text.
    """
    if not extra_info:
        return ""
    raw = ""
    for key in ("question", "problem", "prompt", "query"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break
    if not raw:
        return ""

    text = _humanize_modality_placeholders(raw)
    # 只剩占位符（音频题的常态）：明确告诉 judge 题干不在文本里，免得它去复述输入。
    if not _MODALITY_PLACEHOLDER_RE.sub("", raw).strip(" \t\r\n，,。.；;：:"):
        return text + "\n（题干以媒体形式给出，判分时不可见；请只依据参考答案判断学生答案是否等价。）"
    return text


def _sanitize_for_judge(text: str) -> tuple[str, int]:
    """Drop characters the judge tokenizer cannot encode; report how many.

    2026-10-13 真机（run 15:00）：3/280 条判官请求被服务端 400 五连拒、全部落
    fallback（omni_mix 无规则评分器 ⇒ 整条奖励 0）。复现出的 400 body 是
    ``TextEncodeInput must be Union[TextInputSequence, ...]`` —— 触发条件是
    prompt 里混进**未配对代理字符**（U+D800-DFFF，探针直接复现）。来源：actor
    rollout 偶发吐出**字节级垃圾 token**（不完整 UTF-8 序列），vLLM 的增量解码器
    （tokenizers 的 DecodeStream）会把它解成含代理字符的文本 —— 与肉眼可见的
    2-gram 死环是**同一种采样退化在字节层的表现**。而 naive_router 会把判官的
    400 抹成 500（body 丢弃），worker 侧 5 次重试全部无效。

    ``encode('utf-8', 'ignore')`` 恰好只丢这些字符、正常文本逐字节不变；
    丢了多少必须报数（可观测）。
    """
    clean = text.encode("utf-8", "ignore").decode("utf-8")
    return clean, len(text) - len(clean)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    **kwargs,
) -> dict[str, Any]:
    """Grade one rollout with the generative reward model.

    Args:
        data_source: Dataset identifier, used only by the fallback scorer.
        solution_str: Policy response, already decoded with ``skip_special_tokens=True``.
        ground_truth: Reference answer from ``reward_model.ground_truth``.
        extra_info: Per-sample metadata; ``question`` is read from it.
        reward_router_address: ``host:port`` of the reward router. ``None`` means no
            reward model was launched, in which case this degrades to the rule-based
            scorer instead of failing -- handy for running the script with
            ``RM_ENABLE=0`` without also editing the reward-function overrides.
        reward_model_tokenizer: Judge tokenizer, loaded by ``RewardLoopWorker`` from
            ``reward.reward_model.model_path``. Normally unused -- the server applies
            the chat template. Only touched if the judge server turns out to serve
            ``/v1/completions`` but not ``/v1/chat/completions``, in which case the
            template is rendered here instead.

    Returns:
        dict with ``score`` (the reward, 0.0/1.0), ``acc``, ``genrm_parsed``,
        ``genrm_fallback`` and optionally ``genrm_response``. Every key lands in
        ``reward_extra_info`` and is logged; the numeric ones aggregate into
        validation metrics, the string one is skipped by ``process_validation_metrics``.
    """
    del kwargs

    ground_truth_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    answer = extract_final_answer(solution_str, strip_think=_env_flag("GENRM_STRIP_THINK", True))

    if reward_router_address is None:
        logger.warning("[genrm_verifier] no reward router address; falling back to the rule-based scorer")
        _answer_diag(solution_str, answer, "no_router")
        score, fb_reason = _fallback_score(data_source, solution_str, ground_truth_str, extra_info)
        fb_reason = fb_reason or "no_router"
        # 没有 router 这条路**同样**是「奖励链断了」，必须和正常路径一样计数并可能报警。
        # 2026-10-10 踩过一次：`RM_ENABLE=0` 跑了一整晚，判官一次没调、分数全是规则兜底，
        # 但 _tally 只挂在下面那条路上，于是 tally 一直是空的、REWARD-SUSPECT 一次没响 ——
        # 这正是「静默失败比崩溃贵」的又一个形态。
        _tally(None, fb_reason)
        _reward_health_check(data_source)
        _dump_sample(
            data_source,
            solution_str,
            ground_truth_str,
            extra_info,
            answer,
            "",
            {"score": score, "genrm_parsed": 0.0, "genrm_fallback": 1.0},
            fallback_reason=fb_reason,
        )
        return {"score": score, "acc": score, "genrm_parsed": 0.0, "genrm_fallback": 1.0}

    # 判官 prompt 的三段文本一律先剔除不可编码字符（见 _sanitize_for_judge 的
    # 事故记录）；正常批次零开销、零改动。
    question_txt, q_drop = _sanitize_for_judge(_question_of(extra_info))
    if answer:
        answer_txt, a_drop = _sanitize_for_judge(answer)
    else:
        answer_txt, a_drop = "(the model produced no answer)", 0
    gt_txt, g_drop = _sanitize_for_judge(ground_truth_str)
    dropped = q_drop + a_drop + g_drop
    if dropped:
        _rollout_diag.console_or_file(
            "genrm_prompt_sanitized",
            f"[genrm_verifier] judge prompt 剔除了 {dropped} 个不可编码字符"
            "（未配对代理，来自 rollout 的字节级垃圾 token —— 不剔除则判官 400、"
            "重试 5 次全废、奖励落 fallback）",
            flush=True,
        )
    # 判分模式：coverage（默认，判官输出 Score: k/N，reward=k/N 连续值）/
    # verdict（二元旧行为，对照用）。解析端 _parse_reward 对两种措辞都兼容。
    if _scoring_mode() == "verdict":
        system_prompt, prompt_template = JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT_TEMPLATE
    else:
        system_prompt, prompt_template = JUDGE_COVERAGE_SYSTEM_PROMPT, JUDGE_COVERAGE_PROMPT_TEMPLATE
    prompt = prompt_template.format(
        question=question_txt,
        ground_truth=gt_txt.strip(),
        answer=answer_txt,
    )
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        # Grading is a lookup, not a reasoning task, and a thinking judge costs a few
        # hundred extra decode tokens per rollout on cards the policy is waiting for.
        "chat_template_kwargs": {"enable_thinking": _env_flag("GENRM_ENABLE_THINKING", False)},
        **_sampling_params(),
    }

    # Stop sequences: only stop on explicit Verdict lines, NOT on \n\n
    # \n\n is too aggressive — judge outputs analysis paragraphs separated by \n\n
    # before reaching the Verdict line, causing premature truncation
    # （只能给这两份小写：真机 judge 吐 ``Verdict: Incorrect``，这个「不命中」是**故意**的，
    # 因为服务端会删掉命中的 stop 串而判词在第一行 —— 加变体 = 判词被删成空串 = 奖励链全垮。
    # 完整证据与事故量化见 VERDICT_STOP_STRINGS 定义处。）
    payload["stop"] = list(VERDICT_STOP_STRINGS)

    # [2026-10-13 加] 判词首行之后直接停 —— coverage 模式下的主要止损手段。
    #
    # 真机形态（qwen36_whisper_full.log：618 次 judge 调用）：判官写完 ``Score: 0``
    # 后 410 次（66%）撞进 ``0 0 0…`` 复读（``repeat='0'x510``），把 1024 预算烧满，
    # 什么都没多判出来；而 ``VERDICT_STOP_STRINGS`` 在 coverage 格式下**永远不会
    # 命中**（判官不写那个前缀）。把 ``"\n"`` 加进 stop 表后：判词首行
    # （``Score: k/N`` 或 ``Verdict: …``）一碰到换行，生成当场结束 —— 复读根本没
    # 机会开始。
    #
    # 与 run#87「加大小写变体把判词删没了」事故的本质区别（**不要删这段注释**）：
    # 那次是 **stop 串本身就是判词**（"Verdict: Incorrect"），服务端忽略
    # include_stop_str_in_output 时把它整段删掉 ⇒ 判词没了、文本空串。
    # 这里 stop 串是 ``"\n"`` —— 就算同样被删，删掉的只是一个换行符；
    # ``Score: …``/``Verdict: …`` 行本身不是 stop 串，永远保留在返回文本里 ⇒
    # 解析端拿得到。唯一代价：判官若没照格式、把判词写到第二行，会被截在首行
    # → 落 fallback（与现状「塌进复读 → 落 fallback」同价，不更差）。
    # GENRM_STOP_AT_FIRST_LINE=0 可关掉做对照。
    if _env_flag("GENRM_STOP_AT_FIRST_LINE", True):
        payload["stop"] = ["\n"] + payload["stop"]

    # 但 stop 串默认**会被从返回文本里删掉**（vllm/sampling_params.py:516、
    # v1/engine/detokenizer.py:108：`if self.stop and not include_stop_str_in_output`），
    # 而这两个 stop 串恰好就是 `_parse_verdict` 唯一认得的字符串 —— 于是 judge
    # 只要**听话**（写完 Verdict 行就停），判定串就被抹掉，parse 拿到的只剩前面的
    # 分析段，落进「没有 Verdict」的降级分支。真机证据（qwen36_whisper_full.log
    # 2026-10-10 21:02:18）：
    #     [answer-diag] no_verdict resp_chars=830 ...
    #     [judge-diag] chars=233 think_closed=False finish='stop'   ← 干净停下
    #     [genrm_verifier] REWARD-DEGRADED #1 (rule_failed:NotImplementedError) score=0.0
    # 结果是「judge 判对」和「judge 塌掉」在指标里长得一模一样 —— 而 233 字符的
    # 分析段根本没有 correct/incorrect 字样，parse 只能返回 None。
    # 打开 include_stop_str_in_output 同时满足两个目标：早停（省掉 judge 复读的
    # token）+ Verdict 行留在文本里可见可解析。字段在 chat 与 completions 两条
    # 协议里都是 vLLM 的一等公民（chat_completion/protocol.py:272 并带进
    # SamplingParams.from_optional），服务端 pydantic 是 extra="allow"，只多不少。
    # GENRM_INCLUDE_STOP_STR=0 可回到「删掉 stop 串」的旧行为做对照。
    payload["include_stop_str_in_output"] = _env_flag("GENRM_INCLUDE_STOP_STR", True)

    # Debug: log the actual payload sent to vLLM
    # frequency_penalty 一起打：这是判官侧**唯一**能压住 2-gram 环的旋钮
    # （rp 只问「token 出现过没有」，非累计，`' an'x477` 在 rp=1.1 下照样转），
    # 少打一个字段，驱动脚本就没法证明某一档真的把它设进去了。
    _sp = _sampling_params()
    _rollout_diag.console_or_file("genrm_sampling_parameters", f"[genrm_verifier] payload sampling params: scoring={_scoring_mode()}, "
          f"temperature={_sp.get('temperature')}, "
          f"repetition_penalty={_sp.get('repetition_penalty')}, "
          f"frequency_penalty={_sp.get('frequency_penalty')}, "
          f"presence_penalty={_sp.get('presence_penalty')}, "
          f"max_tokens={_sp.get('max_tokens')}, "
          f"stop={payload.get('stop')}, "
          f"include_stop_str={payload['include_stop_str_in_output']} {_tally_line()} {_judge_finish_line()}", flush=True)

    # 头一个样本把**真正发给 judge 的 prompt** 打出来一眼：判断「判词在最前面」
    # 这条改动是否生效、题干占位符有没有换成人话，都不用再去翻 dump 猜。
    # 每个进程只打一次（这是热路径），GENRM_LOG_PROMPT=0 可关。
    global _PROMPT_LOGGED
    if not _PROMPT_LOGGED and _env_flag("GENRM_LOG_PROMPT", True):
        _PROMPT_LOGGED = True
        limit = _env_int("GENRM_PROMPT_LOG_CHARS", 1200)
        _rollout_diag.console_or_file("genrm_first_prompt", f"[genrm_verifier] judge prompt (first sample of this worker):\n{prompt[:limit]}")

    judge_text = ""
    judge_finish = ""
    judge_error = ""
    timeout = aiohttp.ClientTimeout(total=_env_float("GENRM_TIMEOUT", 600.0))
    _trace_judge = _rollout_diag.limited("genrm_sample")
    _judge_trace_id = uuid4().hex if _trace_judge else None
    _trace_token = _JUDGE_TRACE_ID.set(_judge_trace_id) if _trace_judge else None
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload["model"] = await _resolve_model_name(session, reward_router_address)
            if _trace_judge:
                _rollout_diag.emit(
                    "genrm_request", trace_id=_judge_trace_id, role="reward_model", data_source=data_source,
                    sample_metadata={key: extra_info.get(key) for key in ("index", "problem_id", "uid")}
                    if isinstance(extra_info, dict) else None,
                    actor_response=_rollout_diag.text_info(solution_str),
                    payload=_rollout_diag.summarize(payload),
                )
            judge_text, judge_finish = await _judge(
                session, reward_router_address, payload, reward_model_tokenizer
            )
        verdict = _parse_reward(judge_text)
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill the step
        logger.warning("[genrm_verifier] judge call failed (%s: %s); using fallback", type(e).__name__, e)
        judge_error = f"{type(e).__name__}: {e}"[:200]
        verdict = None
    finally:
        if _trace_token is not None:
            _JUDGE_TRACE_ID.reset(_trace_token)

    # finish_reason 计数：塌缩率（length 占比）的进程级观测点，见 _JUDGE_FINISH_STATS。
    _finish_key = judge_finish or ("error" if judge_error else "unknown")
    _JUDGE_FINISH_STATS[_finish_key] = _JUDGE_FINISH_STATS.get(_finish_key, 0) + 1

    if verdict is None:
        # judge 自己出的问题（超时/无 chat 端点/没按要求吐 Verdict）与 actor 被
        # 截断是两码事，dump 里必须能分开，否则又会退回「看半天不知道为什么是 0」。
        _answer_diag(solution_str, answer, "judge_error" if judge_error else "no_verdict")
        _judge_diag(judge_text, judge_error, judge_finish)
        score, fb_reason = _fallback_score(data_source, solution_str, ground_truth_str, extra_info)
        result = {"score": score, "acc": score, "genrm_parsed": 0.0, "genrm_fallback": 1.0}
        if not fb_reason:
            fb_reason = judge_error or "unparsed_verdict"
        # 判官塌缩（finish=length）是 fallback 里最常见的病因，写进 reason 让 dump
        # 可直接 grep：``judge_collapsed:rule_failed:NotImplementedError`` 一眼分得出
        # 「判官塌了」和「判官根本没被判」。
        if judge_finish == "length" and not fb_reason.startswith("judge_collapsed"):
            fb_reason = f"judge_collapsed:{fb_reason}"
    else:
        result = {"score": verdict, "acc": verdict, "genrm_parsed": 1.0, "genrm_fallback": 0.0}
        fb_reason = ""
        # 判分拿到了 ≠ judge 健康：finish=length 说明它写完首行后仍在复读里把预算
        # 烧满（stop 未命中/未配置时的旧形态）。塌缩必须继续可见 —— 这行 diag 与
        # judge_finish_tally 是 stop / 频率惩罚这些止损被关掉时的报警线。
        if judge_finish == "length":
            _judge_diag(judge_text, "parsed_but_finish_length", judge_finish)

    # 奖励链健康度：加一次计数，必要时大声报警（见 _reward_health_check 上的注释）。
    _tally(verdict, fb_reason)
    _reward_health_check(data_source)

    if _env_flag("GENRM_LOG_RESPONSE", True):
        limit = _env_int("GENRM_RESPONSE_MAX_CHARS", 512)
        # Truncated because this string is broadcast into the batch's non_tensor_batch
        # and carried through the rest of the step for every rollout.
        result["genrm_response"] = judge_text[:limit]

    _dump_sample(
        data_source, solution_str, ground_truth_str, extra_info, answer, judge_text, result,
        fallback_reason=fb_reason, judge_finish=judge_finish,
    )

    if _trace_judge:
        _rollout_diag.emit(
            "genrm_result", trace_id=_judge_trace_id, role="reward_model", data_source=data_source,
            actor_response=_rollout_diag.text_info(solution_str),
            judge_text=_rollout_diag.text_info(judge_text), finish_reason=judge_finish,
            judge_error=judge_error, result=result, fallback_reason=fb_reason,
        )

    return result


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------

_dump_counter = 0


def _dump_sample(
    data_source, solution_str, ground_truth_str, extra_info, answer, judge_text, result,
    fallback_reason: str = "", judge_finish: str = "",
):
    """Dump one sample to GENRM_DUMP_DIR for offline inspection.

    Enabled when GENRM_DUMP_DIR is set; disabled by default. Each sample writes
    a JSON file with question / ground_truth / rollout_answer / judge_text / score,
    plus the截断/降级指纹：``answer_origin`` 为 ``truncated_think`` 说明 response
    打满预算、``</think>`` 从未出现，交出去的「答案」其实是被截断的推理中段；
    ``repeat_hint`` 是尾部死循环；``fallback_reason`` 说明这个分数是判出来的还是
    兜底兜出来的。
    """
    global _dump_counter
    import json as _json

    dump_dir = os.environ.get("GENRM_DUMP_DIR", "")
    if not dump_dir:
        return

    _dump_counter += 1
    # 正常样本只 dump 前 GENRM_DUMP_MAX 条；但**落 fallback 的样本**（判官请求
    # 失败 / 判不出 verdict）在配额之外另有 GENRM_DUMP_ERR_MAX 条配额 ——
    # 2026-10-13 的 3 条 400-失败样本恰好发生在配额用完之后，judge-error 的
    # 现场一个都没留下，只能靠探针复现反推根因。
    _max_dump = _env_int("GENRM_DUMP_MAX", 50)
    if _dump_counter <= _max_dump:
        limit = _dump_counter
    elif fallback_reason and _dump_counter <= _max_dump + _env_int("GENRM_DUMP_ERR_MAX", 20):
        limit = _dump_counter
    else:
        limit = 0
    if not limit:
        return

    resp = solution_str or ""
    # rollout_full 是要人肉看的字段，被悄悄砍掉一半就会重现「答案和全文重复」那种
    # 误判，所以长度上限可调、且把「砍过」这件事写进 dump。
    full_limit = _env_int("GENRM_DUMP_FULL_CHARS", 4000)
    os.makedirs(dump_dir, exist_ok=True)
    _dump_id = f"{_dump_counter:04d}"
    _path = os.path.join(dump_dir, f"sample_{_dump_id}.json")
    try:
        with open(_path, "w", encoding="utf-8") as f:
            _json.dump({
                # _dump_counter 是**进程级**的，而每个 reward worker 都在写同一个
                # 目录：不记 pid 的话两个 worker 的 sample_0001.json 会互相覆盖，
                # 而覆盖之后你看到的样本和日志里的样本对不上。文件里有 pid，
                # 至少能看出「这一条被谁写的 / 有没有被盖掉」。
                "pid": os.getpid(),
                "dump_index": _dump_counter,
                "data_source": str(data_source),
                "question": _question_of(extra_info)[:2000],
                "ground_truth": ground_truth_str[:2000],
                "rollout_answer": answer[:2000],
                "rollout_full": resp[:full_limit],
                "judge_text": (judge_text or "")[:full_limit],
                "judge_repeat_hint": _repeat_hint(judge_text or ""),
                "score": result.get("score"),
                "genrm_parsed": result.get("genrm_parsed"),
                "genrm_fallback": result.get("genrm_fallback"),
                # ---- 诊断字段（幻觉/截断调查用） ----
                "resp_chars": len(resp),
                "rollout_full_truncated": len(resp) > full_limit,
                "answer_chars": len(answer or ""),
                "answer_empty": not (answer or "").strip(),
                "answer_origin": _answer_origin(resp, answer),
                "think_closed": _think_closed(resp),
                "repeat_hint": _repeat_hint(resp),
                "fallback_reason": fallback_reason,
                "fallback_stats_so_far": dict(_FALLBACK_STATS),
                # 到本样本为止的判定分布：单看一个样本分不出「判错/没判/全错」，
                # 分布才分得出（恒 0 → GRPO 优势 0 → 这一步白跑）。
                "verdict_tally_so_far": dict(_VERDICT_TALLY),
                "judge_max_tokens": _env_int("GENRM_MAX_TOKENS", 96),
                "judge_text_chars_full": len(judge_text or ""),
                # ``length`` = judge 把整个 GENRM_MAX_TOKENS 预算烧完了（复读塌掉的
                # 典型收尾）；``stop`` = 它自己停下了，只是没写成要求的格式。
                # 两种情况要改的东西完全不同。
                "judge_finish_reason": judge_finish,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
