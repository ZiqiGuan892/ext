#!/usr/bin/env bash
# Single-node 16-NPU AQA training: rule reward, no RM model or RM resource pool.
# Deploy beside run_huanyu_aqa_audio_train.sh and check_aqa_audio_train.py in
# ${PROJECT_ROOT}/examples/qwen3_6_huanyu/. Prepare the environment and Ray first.
set -euo pipefail

_HUANYU_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RM_ENABLE=0
export REWARD_POOL_NPUS=0
export NUM_NPUS=16 NNODES=1 ROLLOUT_TP=4
export HUANYU_FREEZE_AUDIO="${HUANYU_FREEZE_AUDIO:-false}"

exec bash "${_HUANYU_EXAMPLE_DIR}/run_huanyu_aqa_audio_train.sh" "$@"
