#!/usr/bin/env bash
# Text-only eager TP4; collect exactly 10 execute calls at Level2 in local /tmp.
set -euo pipefail
export TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
DEPLOY_SOURCE="${DEPLOY_SOURCE:-$VLLM_OMNI_ROOT/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml}"
PORT="${HUANYU_SERVE_PORT:-28889}"
SERVED_MODEL="${SERVED_MODEL:-qwen3.8}"

request() {
    local name="$1" batches="$2"
    shift 2
    curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null
    python3 "$TOOLS_DIR/bench_throughput.py" \
        --host 127.0.0.1 --port "$PORT" --model "$SERVED_MODEL" \
        --tokenizer "$MODEL_PATH" --trust-remote-code \
        --batch-size 16 --n 1 --temperature 1.0 \
        --input-tokens 2048 --output-tokens 3072 --ignore-eos \
        --no-enable-thinking --seed 42 --timing-source none \
        --warmup-batches 1 --num-batches "$batches" --num-devices 4 \
        --report "$PROFILE_RUN/$name.json" "$@"
}

case "${1:-}" in
serve)
    export PROFILE_RUN="${PROFILE_RUN:-$(mktemp -d /tmp/huanyu_text_moe_tp4.XXXXXX)}"
    export PROFILE_RUN="$(python3 -c 'import os; print(os.path.realpath(os.environ["PROFILE_RUN"]))')"
    case "$PROFILE_RUN" in /tmp/?*) ;; *) echo 'PROFILE_RUN must be a NEW local /tmp subdirectory' >&2; exit 2 ;; esac
    if [[ -e "$PROFILE_RUN/armed" || -d "$PROFILE_RUN/capture" || -e "$PROFILE_RUN/eager_tp4.yaml" ]]; then
        echo "Use a NEW PROFILE_RUN (unset PROFILE_RUN to create one): $PROFILE_RUN" >&2
        exit 2
    fi
    python3 "$TOOLS_DIR/serve_forward_profile.py" install \
        --runner-file "$VLLM_OMNI_ROOT/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    python3 "$TOOLS_DIR/serve_forward_profile.py" prepare \
        --source "$DEPLOY_SOURCE" --output "$PROFILE_RUN/eager_tp4.yaml"
    python3 - "$PROFILE_RUN/eager_tp4.yaml" <<'PY'
import sys
from pathlib import Path
import yaml
p = Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text())
for group in (cfg, cfg["platforms"]["npu"]):
    group["stages"][0].update(tensor_parallel_size=4, devices="0,1,2,3")
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
print("Text eager: TP=4 devices=0,1,2,3 prefix_cache=false; other capacity settings retained")
PY
    unset HUANYU_FORWARD_PROFILE_DIR VLLM_TORCH_PROFILER_DIR
    unset VLLM_OMNI_EXTERNAL_MODULES HUANYU_GEN_TIMING
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export HUANYU_SERVE_PROFILE_DIR="$PROFILE_RUN/capture"
    export HUANYU_SERVE_PROFILE_ARM_FILE="$PROFILE_RUN/armed"
    export HUANYU_SERVE_PROFILE_START=1 HUANYU_SERVE_PROFILE_ACTIVE=10
    export HUANYU_SERVE_PROFILE_LEVEL=2 HUANYU_SERVE_PROFILE_MOE_DETAIL=1
    export HUANYU_SERVE_PROFILE_SHAPES=0 HUANYU_SERVE_PROFILE_STACK=0
    printf 'Copy to terminal B/C:\nexport PROFILE_RUN=%q\nexport TOOLS_DIR=%q\nexport MODEL_PATH=%q\nexport HUANYU_SERVE_PORT=%q\nexport SERVED_MODEL=%q\n' \
        "$PROFILE_RUN" "$TOOLS_DIR" "$MODEL_PATH" "$PORT" "$SERVED_MODEL"
    vllm serve "$MODEL_PATH" --omni --port "$PORT" --log-stats --trust-remote-code \
        --stage-init-timeout 1800 --served-model-name "$SERVED_MODEL" \
        --deploy-config "$PROFILE_RUN/eager_tp4.yaml" 2>&1 | tee "$PROFILE_RUN/serve.log"
    ;;
request)
    : "${PROFILE_RUN:?Paste the exports printed by the serve terminal}"
    request profile_requests 1 --profile-arm-file "$PROFILE_RUN/armed"
    ;;
baseline)
    : "${PROFILE_RUN:?Paste the exports printed by the serve terminal}"
    python3 "$TOOLS_DIR/serve_forward_profile.py" check \
        --profile-dir "$PROFILE_RUN/capture" --expected-workers 4 --require-moe
    # All capture instances are done and all per-module/method wrappers restored.
    # No arm flag: do not benchmark the profiler's stop/export as normal throughput.
    request post_capture_requests 3
    ;;
status|check|analyse)
    : "${PROFILE_RUN:?Paste the exports printed by the serve terminal}"
    options=()
    if [[ "$1" != status ]]; then options+=(--require-moe); fi
    python3 "$TOOLS_DIR/serve_forward_profile.py" "$1" \
        --profile-dir "$PROFILE_RUN/capture" --expected-workers 4 "${options[@]}"
    ;;
report)
    : "${PROFILE_RUN:?Paste the exports printed by the serve terminal}"
    python3 "$TOOLS_DIR/moe_detail_report.py" --profile-dir "$PROFILE_RUN/capture" \
        --expected-workers 4 --steps "${PROFILE_STEPS:-6,7,8,9}" | tee "$PROFILE_RUN/moe_detail_report.txt"
    ;;
*) echo "Usage: bash $0 {serve|request|status|check|analyse|report|baseline}" >&2; exit 2 ;;
esac
