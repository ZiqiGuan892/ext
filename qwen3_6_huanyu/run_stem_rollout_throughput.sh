#!/usr/bin/env bash
# Pure inference: 16 cards, TP8 x 2 independent servers; no Ray setup/training.
set -euo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${BENCH_RUN:?Set BENCH_RUN to a new local /tmp directory}"
case "${1:-}" in
prepare)
    flags=()
    case "${BENCH_IGNORE_EOS:-true}" in
        true|1) flags+=(--ignore-eos) ;; false|0) flags+=(--no-ignore-eos) ;; *) exit 2 ;;
    esac
    case "${BENCH_ENABLE_THINKING:-false}" in
        true|1) flags+=(--enable-thinking) ;; false|0) flags+=(--no-enable-thinking) ;; *) exit 2 ;;
    esac
    python3 -u "$TOOLS_DIR/bench_stem_rollout.py" prepare --case-dir "$BENCH_RUN" \
      --source "${DEPLOY_SOURCE:-/vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_5_0_huanyu_thinker_only.yaml}" \
      --project-root "${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/code/verl_omni_qwen36_gzq_new}" \
      --tokenizer "${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}" \
      --model "${SERVED_MODEL:-qwen3.8}" --port "${HUANYU_SERVE_PORT:-28889}" \
      --precision "${BENCH_PRECISION:-stem_full}" \
      --devices "${BENCH_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}" \
      --tp "${ROLLOUT_TP:-8}" --max-num-seqs "${ROLLOUT_MAX_NUM_SEQS:-8}" \
      --client-concurrency-per-replica "${BENCH_CONCURRENCY_PER_REPLICA:-8}" \
      --gpu-memory-utilization "${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
      --max-model-len "${ROLLOUT_MAX_MODEL_LEN:-12288}" \
      --max-num-batched-tokens "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN:-12288}}" \
      --train-batch-size "${TRAIN_BATCH_SIZE:-4}" --rollout-n "${ROLLOUT_N:-4}" \
      --input-tokens "${BENCH_INPUT_TOKENS:-2048}" --output-tokens "${BENCH_OUTPUT_TOKENS:-3072}" \
      --temperature "${ROLLOUT_TEMPERATURE:-1.0}" --top-p "${ROLLOUT_TOP_P:-0.95}" \
      --top-k "${ROLLOUT_TOP_K:-20}" --repetition-penalty "${ROLLOUT_REPETITION_PENALTY:-1.05}" \
      --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE:-AIV}" \
      --warmup-batches "${BENCH_WARMUP_BATCHES:-1}" --num-batches "${BENCH_NUM_BATCHES:-3}" \
      --seed "${BENCH_SEED:-42}" --timeout-s "${BENCH_TIMEOUT_S:-7200}" "${flags[@]}"
    printf 'Use the SAME case in both serve terminals and the client terminal:\nexport BENCH_RUN=%q\n' "$BENCH_RUN"
    ;;
serve)
    exec python3 -u "$TOOLS_DIR/bench_stem_rollout.py" serve --case-dir "$BENCH_RUN" --replica "${2:?Choose replica 0 or 1}"
    ;;
bench)
    python3 -u "$TOOLS_DIR/bench_stem_rollout.py" run --case-dir "$BENCH_RUN" 2>&1 | tee "$BENCH_RUN/benchmark.log"
    ;;
report)
    exec python3 "$TOOLS_DIR/bench_stem_rollout.py" report --case-dir "$BENCH_RUN"
    ;;
*) echo 'Usage: bash run_stem_rollout_throughput.sh prepare|serve 0|serve 1|bench|report' >&2; exit 2 ;;
esac
