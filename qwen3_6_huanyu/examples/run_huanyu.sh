#!/usr/bin/env bash
# Huanyu-only entry. Source env_export_huanyu.sh and start Ray on all nodes beforehand.
# Existing backend owns batching, sampling, GenRM and training log tee.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/hfy-model-gy/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
cd "${PROJECT_ROOT}"
source "${_HUANYU_EXAMPLE_DIR}/run_output.sh" || exit $?

# The driver and NEW Ray workers must import the same Huanyu-enabled checkout.
# Exporting PYTHONPATH only in this shell does not establish the Ray job's env.
export HUANYU_VLLM_OMNI_ROOT=/vllm-workspace/vllm-omni
_huanyu_retained_path=""
IFS=: read -r -a _huanyu_path_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_path_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|"$HUANYU_VLLM_OMNI_ROOT"|"$PWD"|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit) continue ;;
    esac
    _huanyu_retained_path+="${_huanyu_retained_path:+:}${_huanyu_path}"
done
export PYTHONPATH="${HUANYU_VLLM_OMNI_ROOT}:${PWD}${_huanyu_retained_path:+:${_huanyu_retained_path}}"
python3 -B "${_HUANYU_EXAMPLE_DIR}/check_native_runtime.py" --check
# JSON quotes are passed as part of ONE argv element for Hydra to parse; no eval.
_huanyu_ray_pythonpath="$(python3 -c 'import json, os; print(json.dumps(os.environ["PYTHONPATH"]))')"

export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens}"
export TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}"
export VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/hfy-model-gy/downloaded_models/Qwen3.5-35B-A3B}"
export FILTER_OVERLONG="${FILTER_OVERLONG:-false}"
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_AUDIO=0

echo "[huanyu] adapter=Qwen36HuanyuForTraining dataset=Qwen36HuanyuOmniRLHFDataset pipeline=qwen3_6_huanyu; freeze_audio=${HUANYU_FREEZE_AUDIO:-true} precompute=${HUANYU_USE_PRECOMPUTE:-0} length_filter=${FILTER_OVERLONG}"

# Keep HF config.architectures intact. architecture below selects only the
# verl training adapter; the RM retains its independent Qwen3.5 registration.
# The inherited vLLM pipeline already dispatches audio_config to Huanyu.
exec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/run_qwen36_huanyu_32card.sh" \
    "$@" \
    actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu \
    actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.use_remove_padding=false \
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset \
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${_huanyu_ray_pythonpath}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_VLLM_OMNI_ROOT=${HUANYU_VLLM_OMNI_ROOT}"
