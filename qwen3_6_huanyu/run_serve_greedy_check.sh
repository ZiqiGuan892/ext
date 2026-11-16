#!/usr/bin/env bash
# Run two separate serve processes, on the same four NPUs; no training or profiler.
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${AB_RUN:?Set AB_RUN to a NEW local /tmp case directory path}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}"
RUNNER_FILE="$VLLM_OMNI_ROOT/vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
PORT="${HUANYU_SERVE_PORT:-28889}"
SERVED_MODEL="${SERVED_MODEL:-qwen3.8}"

case "${1:-}" in
prepare|prepare-audio)
    prepare_args=()
    if [[ "$1" == prepare-audio ]]; then
        if [[ -n "${AUDIO_MANIFEST:-}" ]]; then
            prepare_args+=(--audio-manifest "$AUDIO_MANIFEST")
        else
            prepare_args+=(--audio-dir "${AUDIO_DIR:-/opt/huawei/dataset/infra_data/data/validation_vllm}")
        fi
        prepare_args+=(--input-token-budget "${INPUT_TOKEN_BUDGET:-2048}"
          --audio-input-format "${AUDIO_INPUT_FORMAT:-audio_url}"
          --min-audio-seconds "${MIN_AUDIO_SECONDS:-1}" --max-audio-seconds "${MAX_AUDIO_SECONDS:-30}"
          --seed "${AUDIO_SEED:-42}" --prompt "${AUDIO_PROMPT:-请根据音频内容回答其中的问题。}")
    else
        prepare_args+=(--input-tokens "${INPUT_TOKENS:-2048}")
    fi
    python3 "$TOOLS_DIR/serve_greedy_check.py" "$1" \
      --case-dir "$AB_RUN" \
      --model "${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}" \
      --source "$VLLM_OMNI_ROOT/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml" \
      "${prepare_args[@]}" --output-tokens "${OUTPUT_TOKENS:-3072}"
    python3 "$TOOLS_DIR/serve_greedy_check.py" install --runner-file "$RUNNER_FILE"
    printf 'Use in both terminals:\nexport AB_RUN=%q\n' "$AB_RUN"
    ;;
serve)
    side="${2:?Choose eager or graph}"
    case "$side" in eager|graph) ;; *) exit 2 ;; esac
    test -f "$AB_RUN/case.json"
    if [[ -e "$AB_RUN/$side/serve.log" ]]; then
        echo 'This side has already started; use a NEW AB_RUN for a retry.' >&2
        exit 2
    fi
    # Confirm we instrumented the source imported by this interpreter.
    python3 - "$RUNNER_FILE" <<'PY'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.find_spec('vllm_omni')
actual = Path(next(iter(spec.submodule_search_locations))) / 'platforms/npu/worker/npu_ar_model_runner.py'
assert actual.resolve() == Path(sys.argv[1]).resolve(), f'Imported source differs: {actual}'
assert '# huanyu-greedy-check:begin' in actual.read_text(), 'Run prepare/install first'
PY
    for name in ${!HUANYU_SERVE_PROFILE_@} ${!HUANYU_FORWARD_PROFILE_@}; do unset "$name"; done
    unset HUANYU_GEN_TIMING VLLM_TORCH_PROFILER_DIR VLLM_OMNI_EXTERNAL_MODULES
    unset HUANYU_TEXT_AB_DIR HUANYU_LOGITS_AB_DIR HUANYU_EMBEDDING_AB_DIR
    unset HUANYU_TEXT_AB_DETERMINISM HUANYU_SERVE_REPEAT_DETERMINISM
    unset TORCH_COMPILE_DISABLE TORCHINDUCTOR_DISABLE NPU_GRAPH_CAPTURE_DISABLE VLLM_TORCH_COMPILE
    export HUANYU_JOB_DETERMINISM=0 VLLM_BATCH_INVARIANT=0 HCCL_DETERMINISTIC=false
    export ASCEND_LAUNCH_BLOCKING=0 NCCL_DETERMINISTIC=0 LCCL_DETERMINISTIC=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export HUANYU_GREEDY_CASE="$AB_RUN/case.json" HUANYU_GREEDY_SIDE="$side"
    model_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$AB_RUN/case.json")"
    mkdir -p "$AB_RUN/$side"
    vllm serve "$model_path" --omni --port "$PORT" --log-stats --trust-remote-code \
      --stage-init-timeout 1800 --served-model-name "$SERVED_MODEL" \
      --deploy-config "$AB_RUN/$side.yaml" 2>&1 | tee "$AB_RUN/$side/serve.log"
    ;;
request)
    side="${2:?Choose eager or graph}"
    curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null
    python3 -u "$TOOLS_DIR/serve_greedy_check.py" request --case-dir "$AB_RUN" --side "$side" \
      --url "http://127.0.0.1:$PORT/v1/chat/completions" --served-model "$SERVED_MODEL" \
      --timeout "${REQUEST_TIMEOUT_S:-1800}"
    ;;
compare)
    python3 "$TOOLS_DIR/serve_greedy_check.py" compare --case-dir "$AB_RUN"
    ;;
remove-hook)
    python3 "$TOOLS_DIR/serve_greedy_check.py" install --runner-file "$RUNNER_FILE" --remove
    ;;
*) echo 'Usage: bash run_serve_greedy_check.sh prepare|prepare-audio|serve eager|request eager|serve graph|request graph|compare|remove-hook' >&2; exit 2 ;;
esac
