#!/usr/bin/env python3
"""Convert STEM JSONL data into Parquet for verl's RLHFDataset.

The STEM data at ``/opt/huawei/dastaset/rl_alg_data/rl_stem_data`` is already
in verl RL schema (JSONL, one JSON object per line with the keys
``data_source`` / ``prompt`` / ``ability`` / ``reward_model`` / ``extra_info``).
verl's ``RLHFDataset`` reads **Parquet**, not JSONL, so this script does a
straightforward format conversion — no row-level transformation needed.

Why a converter (not just pointing verl at the JSONL):
  verl's ``RLHFDataset._build_messages`` does ``example[key]`` on columns read
  by ``pyarrow.parquet.read_table``. JSONL is not a supported input shape; it
  must be Parquet. The schema is already correct, so this is a 1:1 row copy.

Output columns (identical to input JSONL keys):
==================  ==================================================
``data_source``     ``"math_dapo"``; selects the reward function
``prompt``          ``[{"role": "user", "content": ...}]``
``ability``         ``"math"``
``reward_model``    ``{"style": "rule", "ground_truth": "<answer>"}``
``extra_info``      ``{split, index, source_dataset, uuid, ...}``
==================  ==================================================

No ``images`` / ``videos`` / ``audios`` column: this is the text-only line.

reward routing (no custom_reward_function):
    verl default_compute_score("math_dapo", solution_str, ground_truth)
      └─ math_dapo.compute_score → extract \\boxed{} / "Answer:" → is_equiv

Usage
-----
    python3 examples/qwen3_6_huanyu/prepare_stem_jsonl_to_parquet.py \\
        --jsonl-dir /opt/huawei/dastaset/rl_alg_data/rl_stem_data \\
        --out-dir /opt/huawei/dastaset/rl_alg_data/rl_stem_data

writes ``<out-dir>/train.parquet`` and ``<out-dir>/val.parquet`` (or
``test.parquet`` if ``--val-split test``).

Use ``--self-test`` to exercise the row validation with no files at all.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# verl RLHFDataset reads these columns; missing any is the KeyError that
# silently empties the dataset (see run_qwen36_huanyu_gspo_npu.sh §坑 #32).
REQUIRED_COLUMNS = ("prompt", "data_source", "reward_model")

# Known data_source values that verl's default_compute_score can route to.
# math_dapo → math_dapo.compute_score (\\boxed{} / "Answer:" extraction + is_equiv)
KNOWN_DATA_SOURCES = {
    "math_dapo",
    "openai/gsm8k",
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
    "math",
    "math_dapo_reasoning",
}


def validate_row(row: dict, line_no: int) -> dict:
    """Check one STEM row has the verl RL schema; raise ValueError if not.

    Raises:
        ValueError: if the row is missing a required column or has a bad shape.
    """
    for col in REQUIRED_COLUMNS:
        if col not in row:
            raise ValueError(f"line {line_no}: missing required column {col!r}")
    if not isinstance(row["prompt"], list) or not row["prompt"]:
        raise ValueError(f"line {line_no}: 'prompt' must be a non-empty list")
    if not isinstance(row["prompt"][0], dict) or "content" not in row["prompt"][0]:
        raise ValueError(f"line {line_no}: 'prompt[0]' must be a dict with 'content'")
    if not isinstance(row["data_source"], str):
        raise ValueError(f"line {line_no}: 'data_source' must be a string, got {type(row['data_source'])}")
    if row["data_source"] not in KNOWN_DATA_SOURCES:
        # Not fatal — warn but keep going. The row may still work if a custom
        # reward function handles it; only default_compute_score would raise
        # NotImplementedError.
        print(f"  WARN line {line_no}: data_source={row['data_source']!r} not in "
              f"verl default_compute_score routing table; will raise "
              f"NotImplementedError unless custom_reward_function is set",
              file=sys.stderr)
    rm = row["reward_model"]
    if not isinstance(rm, dict) or "ground_truth" not in rm:
        raise ValueError(f"line {line_no}: 'reward_model' must be a dict with 'ground_truth'")
    if not rm.get("ground_truth"):
        raise ValueError(f"line {line_no}: 'reward_model.ground_truth' is empty")
    return row


def convert_split(jsonl_dir: str, out_dir: str, split: str, out_name: Optional[str] = None) -> int:
    """Read <jsonl-dir>/<split>.jsonl, validate, write <out-dir>/<out_name>.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = os.path.join(jsonl_dir, f"{split}.jsonl")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"no JSONL for split {split!r} at {src}. "
            f"Contents of {jsonl_dir}: {sorted(os.listdir(jsonl_dir))[:20]}"
        )

    rows, bad = [], []
    n = 0
    with open(src, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(validate_row(row, n))
            except (json.JSONDecodeError, ValueError) as e:
                bad.append((n, str(e)))

    if bad:
        # Loud: a silently-shrunk dataset is the failure mode this script
        # exists to prevent (see run_qwen36_huanyu_gspo_npu.sh §坑 #32).
        print(f"  {split}: WARNING dropped {len(bad)} bad rows, first 3: {bad[:3]}")

    os.makedirs(out_dir, exist_ok=True)
    dst_name = out_name or f"{split}.parquet"
    dst = os.path.join(out_dir, dst_name)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    print(f"  {split}: read {n} lines, wrote {len(rows)} valid rows -> {dst}")

    # Sanity: confirm no multimodal columns leaked in.
    schema_cols = set(pq.read_schema(dst).names)
    mm_cols = schema_cols & {"images", "videos", "audios"}
    if mm_cols:
        print(f"  {split}: ERROR output has multimodal columns {sorted(mm_cols)}; "
              f"text-only training will break")
    return len(rows)


def self_test() -> int:
    """Row-level checks. No files, no pyarrow, no dataset — runs anywhere."""
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("\n=== row validation ===")
    good_row = {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": "What is 2+2? \\boxed{}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "4"},
        "extra_info": {"split": "train", "index": 0},
    }
    try:
        validate_row(good_row, 1)
        check("valid row passes", True)
    except ValueError as e:
        check("valid row passes", False, str(e))

    print("\n=== missing column ===")
    bad_row = {"data_source": "math_dapo", "prompt": []}
    try:
        validate_row(bad_row, 2)
        check("missing reward_model raises", False, "did not raise")
    except ValueError as e:
        check("missing reward_model raises", True)

    print("\n=== empty ground_truth ===")
    bad_row = {**good_row, "reward_model": {"style": "rule", "ground_truth": ""}}
    try:
        validate_row(bad_row, 3)
        check("empty ground_truth raises", False, "did not raise")
    except ValueError:
        check("empty ground_truth raises", True)

    print("\n=== unknown data_source warns but does not raise ===")
    import io
    import contextlib
    unknown_row = {**good_row, "data_source": "custom_stem"}
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_buf):
            validate_row(unknown_row, 4)
        check("unknown data_source does not raise", True)
        check("warns on stderr", "NotImplementedError" in stderr_buf.getvalue(), stderr_buf.getvalue())
    except ValueError:
        check("unknown data_source does not raise", False, "raised")

    print("\n=== prompt not a list ===")
    bad_row = {**good_row, "prompt": "not a list"}
    try:
        validate_row(bad_row, 5)
        check("non-list prompt raises", False, "did not raise")
    except ValueError:
        check("non-list prompt raises", True)

    print("\n=== no multimodal columns in schema ===")
    for key in ("images", "videos", "audios"):
        check(f"no {key!r}", key not in good_row)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl-dir",
                    default="/opt/huawei/dastaset/rl_alg_data/rl_stem_data",
                    help="directory containing train.jsonl and val.jsonl")
    ap.add_argument("--out-dir",
                    default="/opt/huawei/dastaset/rl_alg_data/rl_stem_data",
                    help="directory to write train.parquet and val.parquet")
    ap.add_argument("--val-split", default="val",
                    help="split name for validation (default 'val'; use 'test' for test.jsonl)")
    ap.add_argument("--val-out-name", default=None,
                    help="output parquet name for val split (default '<val-split>.parquet')")
    ap.add_argument("--self-test", action="store_true", help="run row-level checks and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    print(f"jsonl-dir = {args.jsonl_dir}")
    print(f"out-dir   = {args.out_dir}")
    n_train = convert_split(args.jsonl_dir, args.out_dir, "train")
    n_val = convert_split(args.jsonl_dir, args.out_dir, args.val_split, args.val_out_name)
    print(
        f"\nDone: {n_train} train / {n_val} val.\n"
        f"Set TRAIN_FILE={os.path.join(args.out_dir, 'train.parquet')}\n"
        f"    VAL_FILE={os.path.join(args.out_dir, (args.val_out_name or f'{args.val_split}.parquet'))}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())