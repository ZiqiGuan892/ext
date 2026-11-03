#!/usr/bin/env bash
# Terminal A: bash run_huanyu_serve_validation100.sh serve
# Terminal B: paste the printed EVAL_ROOT, then: bash run_huanyu_serve_validation100.sh review
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}"
ex="$PROJECT_ROOT/examples/qwen3_6_huanyu"
port="${HUANYU_SERVE_PORT:-28889}"
model_name="qwen3.8"

check_workers() {
    python3 - "$EVAL_ROOT/serve_console.log" <<'PY'
import json, sys
from pathlib import Path

marker = "actual scope=ab_worker_init status=SETTINGS_VERIFIED "
expected = dict(VLLM_BATCH_INVARIANT="0", HCCL_DETERMINISTIC="false", ASCEND_LAUNCH_BLOCKING="0",
                NCCL_DETERMINISTIC="0", LCCL_DETERMINISTIC="0")
ranks = set()
with Path(sys.argv[1]).open(encoding="utf-8", errors="replace") as log:
    for line in log:
        if marker in line:
            state, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1].lstrip())
            if (state["side"] == "serve" and state["mode"] == "off"
                    and state["torch_deterministic_algorithms"] is False and not state["errors"]
                    and all(state["environment"].get(k) == v for k, v in expected.items())):
                ranks.add(state["rank"])
if ranks != {0, 1, 2, 3}:
    raise SystemExit(f"Waiting for verified off-mode TP ranks 0–3; observed {ranks}. Check {sys.argv[1]} and retry.")
PY
}

export_samples() {
    python3 - "$ex" "$EVAL_ROOT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import aqa_text_ab as ab

root, _, rows = ab.load_case(sys.argv[2])
target = root / "samples"
completed = 0
for row in rows:
    sample = row["sample_id"]
    directory = target / sample
    directory.mkdir(parents=True, exist_ok=True)
    files = {"ground_truth.txt": row["ground_truth"]}
    result, status = ab._read_result(root, "serve", row)
    if status == "complete" and result["reply"].strip():
        files["response.txt"] = result["reply"]
        completed += 1
    else:
        detail = (result or {}).get("error", {}).get("message", status)
        print(f"{sample}: no answer text ({detail})", file=sys.stderr)
    for name, content in files.items():
        path = directory / name
        if path.exists():
            if path.read_bytes() != content.encode("utf-8"):
                raise SystemExit(f"Existing review file differs: {path}")
        else:
            with path.open("xb") as stream:
                stream.write(content.encode("utf-8"))
print(f"{target}: {completed}/{len(rows)} samples have answers")
sys.exit(0 if completed == len(rows) else 1)
PY
}

case "${1:-}" in
serve)
    parquet="${AQA_WAV_SOURCE_PARQUET:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}"
    wav="${AQA_WAV_EXPORT_DIR:-/opt/huawei/dataset/infra_data/data/validation_vllm}"
    checkpoint="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
    export EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/runs/huanyu_2748_val100_$(date +%Y%m%d_%H%M%S)}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
    export HUANYU_JOB_DETERMINISM=0
    if [[ ! -d "$wav" ]]; then
        python3 "$ex/export_aqa_validation_wav.py" --parquet "$parquet" --output-dir "$wav"
    fi
    python3 "$ex/aqa_text_ab.py" prepare --parquet "$parquet" --export-dir "$wav" \
        --output-dir "$EVAL_ROOT" --model-path "$checkpoint" --sample-size 100 --seed 42 \
        --prompt-mode system_audio --tp 4 --max-tokens 2048 --enable-thinking true \
        --determinism off --batch-invariant false --embedding-check false --logits-check false --forward-check false
    printf '复制到同一节点的终端 B：\nexport EVAL_ROOT=%q\n' "$EVAL_ROOT"
    exec python3 "$ex/launch_huanyu_aqa_text_ab.py" serve --case-dir "$EVAL_ROOT" --port "$port" --served-model "$model_name"
    ;;
review)
    : "${EVAL_ROOT:?请粘贴终端 A 打印的 export EVAL_ROOT=...}"
    check_workers
    curl -fsS --max-time 10 "http://127.0.0.1:$port/health" >/dev/null
    rc=0
    python3 "$ex/aqa_text_ab.py" serve --case-dir "$EVAL_ROOT" \
        --url "http://127.0.0.1:$port/v1/chat/completions" --model "$model_name" \
        --timeout-s 3600 --skip-existing --continue-on-error || rc=$?
    export_samples
    exit "$rc"
    ;;
export)
    : "${EVAL_ROOT:?请设置 EVAL_ROOT 为已有的实验目录}"
    export_samples
    ;;
*) echo "用法: bash $0 {serve|review|export}" >&2; exit 2 ;;
esac
