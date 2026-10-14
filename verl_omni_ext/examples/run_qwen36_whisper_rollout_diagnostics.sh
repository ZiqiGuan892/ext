#!/usr/bin/env bash
# Run this with bash, after preparing the gzq environment and Ray cluster.
# This wrapper never sources env_export.sh, installs packages, or starts/stops Ray.
set -eo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run this script with bash; do not source it.' >&2
    return 2
fi

usage() {
    cat <<'EOF'
Usage: bash run_qwen36_whisper_rollout_diagnostics.sh [--dry-run] [-- HYDRA_OVERRIDE ...]

Default: Qwen3.6-whisper + converted OC-RL 4B data, two training steps,
Actor response length 4096, independent JSONL diagnostic files.

Overrides through environment:
  CODE_ROOT                       gzq checkout root
  DIAG_DIR                        new diagnostic directory (must not exist)
  DIAG_RUN_DIR                    new training-output directory (must not exist)
  DIAG_TRAIN_FILE / DIAG_VAL_FILE  already converted RL parquet files
  DIAG_CACHE_DIR                  Qwen36 embedding cache root, containing vision/*.pt
  DIAG_MODEL_PATH                 Actor checkpoint
  DIAG_RM_MODEL_PATH              GenRM checkpoint
  DIAG_LAUNCHER                   existing *_omni_mix_genrm.sh entry point
  DIAG_STEPS                     positive training-step count (default: 2)

The four MINICPMO diagnostic settings are exported below. GDN_NATIVE=0
preserves the current GDN patch; this is observation, not a precision fix.
--dry-run performs no environment sourcing, mkdir, imports, or training.
EOF
}

dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) printf 'Unknown option: %s (put Hydra overrides after --)\n' "$1" >&2; exit 2 ;;
    esac
done

CODE_ROOT=${CODE_ROOT:-/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq}
DIAG_DIR=${DIAG_DIR:-${CODE_ROOT}/diagnostics/actor_precision_001}
DIAG_RUN_DIR=${DIAG_RUN_DIR:-${CODE_ROOT}/runs/$(basename "$DIAG_DIR")}
DIAG_TRAIN_FILE=${DIAG_TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/dataset/train.parquet}
DIAG_VAL_FILE=${DIAG_VAL_FILE:-/opt/huawei/dataset/infra_data/data/dataset/validation.parquet}
# This is a suggested cache location, not a verified server artifact. Override
# DIAG_CACHE_DIR if the completed Qwen36 precomputation is elsewhere.
DIAG_CACHE_DIR=${DIAG_CACHE_DIR:-${CODE_ROOT}/embed_cache_4b_qwen36}
DIAG_MODEL_PATH=${DIAG_MODEL_PATH:-/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl}
DIAG_RM_MODEL_PATH=${DIAG_RM_MODEL_PATH:-/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B}
DIAG_STEPS=${DIAG_STEPS:-2}
launcher_explicit=${DIAG_LAUNCHER:+1}
DIAG_LAUNCHER=${DIAG_LAUNCHER:-${CODE_ROOT}/examples/qwen3_6_whisper/run_qwen36_whisper_gspo_npu_omni_mix_genrm.sh}

if [[ ! "$DIAG_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DIAG_STEPS must be a positive integer: %s\n' "$DIAG_STEPS" >&2
    exit 2
fi
for directory in "$CODE_ROOT" "$DIAG_DIR" "$DIAG_RUN_DIR" "$DIAG_CACHE_DIR"; do
    [[ "$directory" = /* ]] || { printf 'An absolute path is required: %s\n' "$directory" >&2; exit 2; }
done

# Prefer the server's established launcher location. A package-only deployment
# can use the identical launcher pair bundled in ext/examples.
if [[ ! -f "$DIAG_LAUNCHER" && -z "$launcher_explicit" ]]; then
    fallback="${CODE_ROOT}/verl_omni_ext/examples/run_qwen36_whisper_gspo_npu_omni_mix_genrm.sh"
    if [[ -f "$fallback" ]]; then DIAG_LAUNCHER="$fallback"; fi
fi

run_env=(
    "MINICPMO_ROLLOUT_DIAG=1"
    "MINICPMO_ROLLOUT_DIAG_DIR=$DIAG_DIR"
    "MINICPMO_DIAG_NUMERICS=1"
    "MINICPMO_DIAG_GDN_NATIVE=0"
    "MINICPMO_DIAG_STRICT_MM=0"
    "TRAIN_FILE=$DIAG_TRAIN_FILE"
    "VAL_FILE=$DIAG_VAL_FILE"
    "MODEL_PATH=$DIAG_MODEL_PATH"
    "RM_MODEL_PATH=$DIAG_RM_MODEL_PATH"
    "MINICPMO_EMBED_CACHE_DIR=$DIAG_CACHE_DIR"
    "DEBUG_MODE=0"
    "DEBUG_SAMPLES=0"
    "TRAIN_BATCH_SIZE=4"
    "PPO_MINI_BATCH_SIZE=4"
    "PPO_MICRO_BATCH_SIZE_PER_GPU=1"
    "ROLLOUT_N=4"
    "MAX_PROMPT_LENGTH=3072"
    "MAX_RESPONSE_LENGTH=4096"
    "ROLLOUT_PROMPT_LENGTH=3104"
    "ROLLOUT_MAX_MODEL_LEN=8192"
    "MAX_TOKEN_LEN_PER_GPU=8192"
    "VAL_BEFORE_TRAIN=false"
    "PROJECT_NAME=oc_rl_4b_gzq"
    "EXPERIMENT_NAME=$(basename "$DIAG_DIR")"
    "DEFAULT_LOCAL_DIR=$DIAG_RUN_DIR/checkpoints"
    "TENSORBOARD_DIR=$DIAG_RUN_DIR/tensorboard"
    "GENRM_DUMP_DIR=$DIAG_RUN_DIR/genrm_dump"
    "MINICPMO_HANG_DUMP_DIR=$DIAG_RUN_DIR/hang_dump"
)

# These last-position overrides defeat the existing launcher's fixed 5120
# budgets. DEBUG_MODE=0 avoids its unconditional response-length reset to 512.
hydra_args=(
    data.train_max_samples=-1
    data.val_max_samples=-1
    data.max_response_length=4096
    actor_rollout_ref.rollout.response_length=4096
    actor_rollout_ref.rollout.prompt_length=3104
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    "trainer.total_training_steps=$DIAG_STEPS"
    trainer.total_epochs=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    "$@"
)

print_command() {
    printf 'cd %q\n' "$DIAG_RUN_DIR"
    printf 'env'
    printf ' %q' "${run_env[@]}"
    printf ' bash %q' "$DIAG_LAUNCHER"
    printf ' %q' "${hydra_args[@]}"
    printf ' > %q 2>&1\n' "$DIAG_RUN_DIR/launcher.stdout.log"
}

if (( dry_run )); then
    printf '%s\n' '# Dry run: no environment scripts, imports, directories, or training.'
    printf '%s\n' '# The real run first checks module origins, files and the cache directory.'
    print_command
    exit 0
fi

# These exports affect only this Bash process and its training children.
export MINICPMO_ROLLOUT_DIAG=1
export MINICPMO_ROLLOUT_DIAG_DIR="$DIAG_DIR"
export MINICPMO_DIAG_NUMERICS=1
export MINICPMO_DIAG_GDN_NATIVE=0
export MINICPMO_DIAG_STRICT_MM=0

for file in "$DIAG_TRAIN_FILE" "$DIAG_VAL_FILE" "$DIAG_LAUNCHER" \
    "$(dirname "$DIAG_LAUNCHER")/run_qwen36_whisper_gspo_npu.sh"; do
    [[ -f "$file" ]] || { printf 'Required file does not exist: %s\n' "$file" >&2; exit 2; }
done
for directory in "$CODE_ROOT" "$DIAG_MODEL_PATH" "$DIAG_RM_MODEL_PATH" "$DIAG_CACHE_DIR/vision"; do
    [[ -d "$directory" ]] || { printf 'Required directory does not exist: %s\n' "$directory" >&2; exit 2; }
done
shopt -s nullglob
vision_entries=("$DIAG_CACHE_DIR"/vision/*.pt)
shopt -u nullglob
if (( ${#vision_entries[@]} == 0 )); then
    printf 'No vision/*.pt cache files found in %s; media/*.png is not an embedding cache.\n' "$DIAG_CACHE_DIR" >&2
    exit 2
fi
unset vision_entries
for directory in "$DIAG_DIR" "$DIAG_RUN_DIR"; do
    [[ ! -e "$directory" ]] || { printf 'Output already exists; select a new DIAG_DIR/DIAG_RUN_DIR: %s\n' "$directory" >&2; exit 2; }
done

# Verify from /tmp so the current checkout cannot hide a wrong editable install.
# No package imports: find_spec of these top-level names checks locations only.
(
    cd /tmp
    python3 - "$CODE_ROOT" <<'PY'
from importlib.util import find_spec
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for name in ("verl_omni_ext", "verl_omni", "verl", "vllm_omni"):
    spec = find_spec(name)
    if spec is None or not spec.origin:
        raise SystemExit(f"Missing {name}; prepare the gzq environment before running this wrapper.")
    origin = Path(spec.origin).resolve()
    print(f"[module] {name} -> {origin}")
    if name == "verl_omni_ext" and not origin.is_relative_to(root):
        raise SystemExit(f"Wrong plugin copy: {origin}; expected a package inside {root}.")
    if name != "verl_omni_ext" and origin.is_relative_to(root):
        raise SystemExit(f"Expected shared dependency outside the gzq copy: {name} -> {origin}")
PY
)

mkdir -p "$(dirname "$DIAG_DIR")" "$(dirname "$DIAG_RUN_DIR")"
mkdir "$DIAG_DIR"
mkdir "$DIAG_RUN_DIR"
mkdir "$DIAG_RUN_DIR/genrm_dump" "$DIAG_RUN_DIR/hang_dump"
print_command > "$DIAG_RUN_DIR/command.sh.txt"
printf 'Training log: %s\n' "$DIAG_RUN_DIR/run_qwen36_whisper_gspo_npu.log"
printf 'Launcher output: %s\n' "$DIAG_RUN_DIR/launcher.stdout.log"
printf 'Separate diagnostics: %s/trace-<hostname>-<pid>.jsonl\n' "$DIAG_DIR"

set +e
(
    cd "$DIAG_RUN_DIR" || exit 2
    env "${run_env[@]}" bash "$DIAG_LAUNCHER" "${hydra_args[@]}"
) > "$DIAG_RUN_DIR/launcher.stdout.log" 2>&1
result=$?
set -e

shopt -s nullglob
traces=("$DIAG_DIR"/trace-*.jsonl)
shopt -u nullglob
printf 'Training exit code: %s; local diagnostic files: %s\n' "$result" "${#traces[@]}"
if (( ${#traces[@]} == 0 )); then
    printf '%s\n' 'No local JSONL found. Check worker environment propagation and remote-node directories; absence of traces is not diagnostic success.' >&2
fi
exit "$result"
