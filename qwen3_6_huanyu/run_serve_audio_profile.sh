#!/usr/bin/env bash
# Independent audio-input eager capture; TP=4, no training stack.
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TOOLS_DIR
MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
DEPLOY_SOURCE="${DEPLOY_SOURCE:-$VLLM_OMNI_ROOT/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml}"
PORT="${HUANYU_SERVE_PORT:-28889}"
SERVED_MODEL="${SERVED_MODEL:-qwen3.8}"

case "${1:-}" in
serve)
    export PROFILE_RUN="${PROFILE_RUN:-$TOOLS_DIR/audio_profile_tp4_$(date +%Y%m%d_%H%M%S)_$$}"
    export PROFILE_RUN="$(python3 -c 'import os; print(os.path.abspath(os.environ["PROFILE_RUN"]))')"
    if [[ -e "$PROFILE_RUN/armed" || -d "$PROFILE_RUN/capture" || -e "$PROFILE_RUN/audio_eager_tp4.yaml" ]]; then
        echo "Use a NEW PROFILE_RUN for this audio capture: $PROFILE_RUN" >&2
        exit 2
    fi
    python3 "$TOOLS_DIR/serve_forward_profile.py" install \
        --runner-file "$VLLM_OMNI_ROOT/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    python3 "$TOOLS_DIR/serve_forward_profile.py" prepare \
        --source "$DEPLOY_SOURCE" --output "$PROFILE_RUN/audio_eager_tp4.yaml"
    python3 - "$PROFILE_RUN/audio_eager_tp4.yaml" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text())
for group in (cfg, cfg["platforms"]["npu"]):
    stage = group["stages"][0]
    stage["tensor_parallel_size"] = 4
    stage["devices"] = "0,1,2,3"
    stage["mm_processor_cache_gb"] = 0
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
print("Audio eager: TP=4 devices=0,1,2,3 prefix_cache=false mm_processor_cache_gb=0")
PY
    unset HUANYU_FORWARD_PROFILE_DIR VLLM_TORCH_PROFILER_DIR
    unset VLLM_OMNI_EXTERNAL_MODULES HUANYU_GEN_TIMING
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export HUANYU_SERVE_PROFILE_DIR="$PROFILE_RUN/capture"
    export HUANYU_SERVE_PROFILE_ARM_FILE="$PROFILE_RUN/armed"
    # Capture the first real execute after arming, including media prefill.
    export HUANYU_SERVE_PROFILE_START=1
    export HUANYU_SERVE_PROFILE_ACTIVE="${HUANYU_SERVE_PROFILE_ACTIVE:-10}"
    export HUANYU_SERVE_PROFILE_LEVEL="${HUANYU_SERVE_PROFILE_LEVEL:-2}"
    export HUANYU_SERVE_PROFILE_SHAPES="${HUANYU_SERVE_PROFILE_SHAPES:-0}"
    export HUANYU_SERVE_PROFILE_STACK="${HUANYU_SERVE_PROFILE_STACK:-0}"
    printf 'Copy to terminal B/C:\nexport PROFILE_RUN=%q\nexport TOOLS_DIR=%q\nexport HUANYU_SERVE_PORT=%q\nexport SERVED_MODEL=%q\n' \
        "$PROFILE_RUN" "$TOOLS_DIR" "$PORT" "$SERVED_MODEL"
    vllm serve "$MODEL_PATH" --omni --port "$PORT" --log-stats --trust-remote-code \
        --stage-init-timeout 1800 --served-model-name "$SERVED_MODEL" \
        --deploy-config "$PROFILE_RUN/audio_eager_tp4.yaml" \
        2>&1 | tee "$PROFILE_RUN/serve.log"
    ;;
request)
    : "${PROFILE_RUN:?Paste the export PROFILE_RUN printed by the serve terminal}"
    shift
    curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null
    python3 "$TOOLS_DIR/bench_audio_profile.py" \
        --host 127.0.0.1 --port "$PORT" --model "$SERVED_MODEL" \
        --audio-dir "${AUDIO_DIR:-/opt/huawei/dataset/infra_data/data/validation_vllm}" \
        --batch-size "${BATCH_SIZE:-16}" --n 1 --temperature "${TEMPERATURE:-1.0}" \
        --output-tokens "${OUTPUT_TOKENS:-3072}" --ignore-eos --no-enable-thinking \
        --min-audio-seconds "${MIN_AUDIO_SECONDS:-1}" --max-audio-seconds "${MAX_AUDIO_SECONDS:-30}" \
        --seed "${SEED:-42}" --timing-source none --num-devices 4 \
        --warmup-batches 1 --num-batches 1 \
        --profile-arm-file "$PROFILE_RUN/armed" --report "$PROFILE_RUN/profile_audio_requests.json" "$@"
    ;;
status|check|analyse)
    : "${PROFILE_RUN:?Paste the export PROFILE_RUN printed by the serve terminal}"
    options=()
    if [[ "$1" != status ]]; then options+=(--require-audio); fi
    python3 "$TOOLS_DIR/serve_forward_profile.py" "$1" \
        --profile-dir "$PROFILE_RUN/capture" --expected-workers 4 "${options[@]}"
    ;;
*) echo "Usage: bash $0 {serve|request|status|check|analyse}" >&2; exit 2 ;;
esac
