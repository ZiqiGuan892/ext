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
"""Convert self-research 6-modality ShareGPT JSON into verl RL parquet.

Six modalities are merged into one unified schema so nextqa.py/avqa.py:

    OmniVideo (video + audio)   examples/examples/omnivideo/omnivideo_100.json
    VAQA      (image + audio)   vaqa.json
    VL Real   (image)            vl_real.json
    AQA Real  (audio)           examples/examples/aqa/real_aqa_data_400.json
    VQA       (image)           examples/examples/text_vqa/vqa_400.json
    Text      (text)            examples/examples/text_vqa/text_400.json

Three things differ from NExT-QA, all handled here:

1. Open-ended answers, not multiple-choice.  The raw ``gpt`` value is kept
   verbatim and wrapped in ``<answer>...</answer>`` as ``ground_truth``; the reward
   (:mod:`omni_open_reward`) extracts and compares it it.

2. Mixed media.  Some row has video+audio, some image+audio, some image, some
   audio, some text.  All three media columns (``images``/``videos``/``audios``)
   are always present; empty rows carry ``[]````.  Placeholders ``<video>`` /
   ``<image>`` / ``<audio>`` in the user message determine which columns
   are non-empty for that row.

3. Path rewriting.  Source JSON mixes ``/opt/huawei/dataset`` absolute
   prefixes, ``/opt/huawei/dataset`` absolute prefixes, relative paths (joined
   under a media root), and ``zarr.zip::uuid`` audio members.  All are normalised
   to ``/opt/huawei/dataset/...`` before writing to parquet, so the training
   machine sees one consistent prefix.

Single-turn extraction
--------------------
Each ShareGPT record may be multi-turn. VAQA in particular chains several
human/gpt pairs sharing one image.  verl's RLHFDataset is single-turn: this
converter takes the first ``human`` turn (the question) and the first ``gpt``
turn after it (the reference answer), discarding later turns.  Media items are
sliced to the placeholders found in that first human turn: ``<image>`` ->
``images[0]``, ``<video>`` -> ``videos[0]``, ``<audio>`` -> ``audios[0]``.

Run on the training machine (paths are remote)::

    python3 examples/gspo_trainer/data_process/omni_mix.py \
        --output_dir /opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8 \
        --max_frames 8 \
        --train_ratio 0.9

Edit ``SOURCES`` below to match the real machine's layout, then run there.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DATA_SOURCE = "omni_mix"
DATASET_NAME = "OmniMix"

# -------------------------------------------------------------------------
# Sources.  Edit on the training machine to match its real layout.  Each source
# is (json_path, structure, modality, media_root, enable_path_prefix_rewrite).
#
# structure: "flat"  = list[record]
#           "nested" = list[list[record]]  (outer = groups, inner = records)
# modality: short used for extra_info.modality + ability routing
# media_root: joined to relative paths (e.g. omnivideo "videos/xxx.mp4").  Must
#              already be an /opt/huawei/dataset path.
# path_prefix_rewrite: rewrite absolute paths whose prefix matches "from" to "to".
#              Defaults to /opt/huawei/dataset -> /opt/huawei/dataset.
# -------------------------------------------------------------------------
SOURCES: list[dict[str, Any]] = [
    {
        "name": "omnivideo",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/examples/omnivideo/omnivideo_100.json",
        "structure": "flat",
        "modality": "video+audio",
        "media_root": "/opt/huawei/dataset/minstruction_hawk_guiyang/multimodal_streaming_understanding/brolydfgh/OmniVideo-100K",
        "path_prefix_rewrite": {"/home/ma-user/work/dataset": "/opt/huawei/dataset"},
    },
    {
        "name": "vaqa",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/vaqa/sharegpt_hw/vaqa_gui/gui-0116.json",
        "structure": "nested",
        "modality": "image+audio",
        "media_root": None,
        "path_prefix_rewrite": {},  # VAQA paths are already /opt/huawei/dataset
    },
    {
        "name": "vl_real",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/vl_real/vl_real.json",
        "structure": "nested",
        "modality": "image",
        "media_root": None,
        "path_prefix_rewrite": {"/home/ma-user/work/dataset": "/opt/huawei/dataset"},
    },
    {
        "name": "aqa_real",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/aqa_real/aqa_real.json",
        "structure": "flat",
        "modality": "audio",
        "media_root": None,
        "path_prefix_rewrite": {"/home/ma-user/work/dataset": "/opt/huawei/dataset"},
    },
    {
        "name": "vqa",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/examples/text_vqa/vqa_400.json",
        "structure": "flat",
        "modality": "image",
        "media_root": None,
        "path_prefix_rewrite": {"/home/ma-user/work/dataset": "/opt/huawei/dataset"},
    },
    {
        "name": "text",
        "json_path": "/opt/huawei/dataset/minstruction_hawk_guiyang/omnidata_sample/examples/text_vqa/text_400.json",
        "structure": "flat",
        "modality": "text",
        "media_root": None,
        "path_prefix_rewrite": {"/home/ma-user/work/dataset": "/opt/huawei/dataset"},
    },
]

# Unified RL system prompt.  Replaces every source's own system message so the
# rollout emits a consistent <answer>...</answer> tail that the reward can parse.
SYSTEM_PROMPT = (
    "你是一个多模态助手。请结合图像、音频、视频等信息分析问题，给出准确、详细的回答。"
)

# Video frame budget baked into every video row (see nextqa.py:179).  The launch
# script's preflight reads videos[0]["max_frames"]; this value must not exceed
# MAX_FRAMES_BUDGET there.  Kept as a function arg so the script passes --max_frames.
DEFAULT_FPS = 1.0
DEFAULT_MIN_PIXELS = 32 * 28 * 28
DEFAULT_MAX_PIXELS = 128 * 28 * 28

_PLACEHOLDER_RE = re.compile(r"<(image|video|audio)>")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def rewrite_path(path: Any, media_root: str | None, prefix_map: dict[str, str]) -> str | None:
    """Normalise one media path to an /opt/huawei/dataset absolute path.

    Handles ``zarr.zip::uuid`` members (rewrites the archive prefix only),
    absolute ``/home/ma-user/...`` paths, already-``/opt/...`` paths, and
    relative paths joined under ``media_root``.
    """
    raw = _text(path)
    if raw is None:
        return None

    # zarr member: rewrite archive, preserve uuid
    if "::" in raw:
        archive, member = raw.split("::", 1)
        return f"{rewrite_path(archive, media_root, prefix_map)}::{member}"

    if raw.startswith("/"):
        # Apply prefix rewrites (default /home/ma-user -> /opt/huawei)
        for src, dst in prefix_map.items():
            if raw.startswith(src):
                return dst + raw[len(src):]
        return raw  # already /opt or other absolute

    # relative path -> join media_root
    if media_root:
        return os.path.join(media_root, raw)
    raise ValueError(
        f"相对路径 {raw!r} 无法解析：该 source 未配置 media_root。"
        "请在 SOURCES 中为该 source 设置 media_root 指向 /opt/huawei/dataset 下的媒体根目录。"
    )


def _count_placeholders(text: str) -> dict[str, int]:
    """Count <image>/<video>/<audio> placeholders in one message's text."""
    counts: dict[str, int] = {"image": 0, "video": 0, "audio": 0}
    for m in _PLACEHOLDER_RE.finditer(text or ""):
        counts[m.group(1)] += 1
    return counts


def _extract_first_turn(conversations: list[dict]) -> tuple[str | None, str | None]:
    """Return (first_human_value, first_gpt_value_after_it) from a conversation.

    Multi-turn records (VAQA chains several pairs) are truncated to the first
    human/gpt pair for single-turn RL.  The system message, if present, is
    ignored — a unified SYSTEM_PROMPT is injected by the caller.
    """
    human_val: str | None = None
    for turn in conversations or []:
        role = (turn.get("from") or "").lower()
        value = _text(turn.get("value"))
        if role == "human" and human_val is None:
            human_val = value
        elif role == "gpt" and human_val is not None:
            return human_val, value
    return human_val, None


def build_video_entry(
    path: str,
    fps: float = DEFAULT_FPS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_frames: int = 8,
) -> dict[str, Any]:
    """One ``videos`` list item.  max_frames is baked here, not read from env."""
    return {
        "video": path,
        "fps": fps,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
    }


def build_rl_row(
    record: dict[str, Any],
    source: dict[str, Any],
    index: int,
    max_frames: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one ShareGPT record to a unified parquet row, or a drop reason."""
    conversations = record.get("conversations") or []
    human_val, gpt_val = _extract_first_turn(conversations)
    if human_val is None:
        return None, "no_human_turn"
    if gpt_val is None:
        return None, "no_gpt_turn"

    counts = _count_placeholders(human_val)
    media_root = source.get("media_root")
    prefix_map = source.get("path_prefix_rewrite") or {}

    # Slice each media list to the placeholder count and rewrite paths.
    images: list[dict[str, str]] = []
    videos: list[dict[str, Any]] = []
    audios: list[str] = []

    for raw_path in (record.get("image") or [])[: counts["image"]]:
        rp = rewrite_path(raw_path, media_root, prefix_map)
        if rp:
            images.append({"image": rp})

    for raw_path in (record.get("video") or [])[: counts["video"]]:
        rp = rewrite_path(raw_path, media_root, prefix_map)
        if rp:
            videos.append(build_video_entry(rp, max_frames=max_frames))

    for raw_path in (record.get("audio") or [])[: counts["audio"]]:
        rp = rewrite_path(raw_path, media_root, prefix_map)
        if rp:
            audios.append(rp)

    # Sanity: if a placeholder has no backing media, the row will fail at decode
    # time with a clearer error, but we flag it here as a drop.
    if counts["image"] and not images:
        return None, "missing_image"
    if counts["video"] and not videos:
        return None, "missing_video"
    if counts["audio"] and not audios:
        return None, "missing_audio"

    modality = source["modality"]
    raw_source = source["name"]
    answer = gpt_val  # keep verbatim; GenRM judge compares semantic equivalence
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human_val},
        ],
        "images": images,
        "videos": videos,
        "audios": audios,
        "ability": modality,
        "reward_model": {"style": "model", "ground_truth": answer},
        "extra_info": {
            "split": "",
            "index": index,
            "problem_id": f"{raw_source}_{index}",
            "dataset": DATASET_NAME,
            "modality": modality,
            "raw_source": raw_source,
            "raw_answer": answer,
            "question": human_val,
        },
    }, None


def load_records(json_path: str, structure: str) -> list[dict[str, Any]]:
    """Load a ShareGPT JSON, flattening nested ``list[list[record]]`` to flat."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if structure == "nested":
        # list[list[record]] -> list[record]
        flat: list[dict[str, Any]] = []
        for group in data or []:
            if isinstance(group, list):
                flat.extend(r for r in group if isinstance(r, dict))
            elif isinstance(group, dict):
                flat.append(group)
        return flat
    # flat: list[record]
    return [r for r in (data or []) if isinstance(r, dict)]


def convert_source(source: dict[str, Any], max_frames: int) -> tuple[list[dict], Counter]:
    """Load + convert one source, returning (rows, drop_reasons)."""
    json_path = source["json_path"]
    if not Path(json_path).is_file():
        print(f"[skip] {source['name']}: {json_path} 不存在，跳过该 source")
        return [], Counter({"missing_source_file": 1})

    records = load_records(json_path, source.get("structure", "flat"))
    rows: list[dict] = []
    dropped: Counter[str] = Counter()
    for idx, record in enumerate(records):
        row, reason = build_rl_row(record, source, idx, max_frames)
        if row is None:
            dropped[reason or "invalid_record"] += 1
            continue
        rows.append(row)
    return rows, dropped


def _write_parquet(rows: list[dict], output_path: Path) -> None:
    """Write parquet with use_dictionary=False (matches nextqa.py/avqa.py)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        output_path, engine="pyarrow", index=False, use_dictionary=False
    )


def convert_dataset(
    output_dir: str | Path,
    max_frames: int = 8,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert all configured SOURCES into train.parquet + validation.parquet."""
    import random

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    per_source_stats: list[dict[str, Any]] = []
    total_dropped: Counter[str] = Counter()

    for source in SOURCES:
        rows, dropped = convert_source(source, max_frames)
        all_rows.extend(rows)
        total_dropped.update(dropped)
        per_source_stats.append({
            "name": source["name"],
            "modality": source["modality"],
            "json_path": source["json_path"],
            "input": len(rows) + sum(dropped.values()),
            "kept": len(rows),
            "dropped": dict(dropped),
            "modality_breakdown": _modality_breakdown(rows),
        })

    if not all_rows:
        raise ValueError(
            "没有从任何 source 转换出行有效行。检查 SOURCES 里的 json_path 是否存在、"
            "structure 是否正确。"
        )

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_train = int(len(all_rows) * train_ratio)
    if n_train <= 0 or n_train >= len(all_rows):
        raise ValueError(
            f"train_rows={len(all_rows)} train_ratio={train_ratio} 导致 n_train={n_train}，"
            "必须满足 0 < n_train < len(all_rows)。请调整 train_ratio。"
        )
    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train:]

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)

    return {
        "sources": per_source_stats,
        "total_kept": len(all_rows),
        "total_dropped": dict(total_dropped),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "modality_breakdown": _modality_breakdown(all_rows),
        "train_path": str(train_path),
        "validation_path": str(val_path),
    }


def _modality_breakdown(rows: list[dict]) -> dict[str, int]:
    """Count rows per extra_info.modality (also == ability)."""
    counts: Counter[str] = Counter()
    for r in rows:
        counts[r["extra_info"]["modality"]] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for train.parquet and validation.parquet.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=8,
        help="Max sampled frames per video, baked into videos[0].max_frames.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of rows for training; rest goes to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the train/val split.",
    )
    args = parser.parse_args()

    stats = convert_dataset(
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()