#!/usr/bin/env bash
# Huanyu-only entry. Source env_export.sh and start Ray on all nodes beforehand.
# Existing backend owns batching, sampling, GenRM and training log tee.
set -euo pipefail

cd /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq

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
python3 -B verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py --check
# JSON quotes are passed as part of ONE argv element for Hydra to parse; no eval.
_huanyu_ray_pythonpath="$(python3 -c 'import json, os; print(json.dumps(os.environ["PYTHONPATH"]))')"

export MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens
export TRAIN_FILE=/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/train.parquet
export VAL_FILE=/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/validation.parquet
export RM_MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B
export FILTER_OVERLONG=false
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_AUDIO=0
export HUANYU_USE_PRECOMPUTE="${HUANYU_USE_PRECOMPUTE:-0}"
export HUANYU_EMBED_CACHE_DIR="${HUANYU_EMBED_CACHE_DIR:-}"
case "$HUANYU_USE_PRECOMPUTE" in
    0) ;;
    1)
        [[ -n "$HUANYU_EMBED_CACHE_DIR" && -f "$HUANYU_EMBED_CACHE_DIR/manifest.json" ]] || {
            echo '[huanyu] cache mode requires HUANYU_EMBED_CACHE_DIR with manifest.json; run Huanyu precompute first' >&2
            exit 2
        }
        [[ "${MINICPMO_TUNE_VISION:-0}" != 1 ]] || { echo '[huanyu] cached vision must remain frozen' >&2; exit 2; }
        ;;
    *) echo '[huanyu] HUANYU_USE_PRECOMPUTE must be 0 or 1' >&2; exit 2 ;;
esac
_huanyu_ray_cache_dir="$(python3 -c 'import json, os; print(json.dumps(os.environ["HUANYU_EMBED_CACHE_DIR"]))')"

# Budget the EXPANDED multimodal prompt. Length filtering remains disabled.
# Capture the values here: the inherited DEBUG wrapper resets MAX_PROMPT_LENGTH.
_huanyu_prompt_length=${MAX_PROMPT_LENGTH:-4096}
_huanyu_response_length=${MAX_RESPONSE_LENGTH:-2048}
_huanyu_rollout_prompt_length=${ROLLOUT_PROMPT_LENGTH:-${_huanyu_prompt_length}}
for _huanyu_length in "$_huanyu_prompt_length" "$_huanyu_response_length" "$_huanyu_rollout_prompt_length"; do
    [[ "$_huanyu_length" =~ ^[1-9][0-9]*$ ]] || { echo '[huanyu] prompt/response lengths must be positive integers' >&2; exit 2; }
done
_huanyu_sequence_length=$(( _huanyu_prompt_length + _huanyu_response_length ))
_huanyu_model_length=${ROLLOUT_MAX_MODEL_LEN:-$(( _huanyu_sequence_length + 128 ))}
_huanyu_token_budget=${MAX_TOKEN_LEN_PER_GPU:-${_huanyu_model_length}}
for _huanyu_length in "$_huanyu_model_length" "$_huanyu_token_budget"; do
    [[ "$_huanyu_length" =~ ^[1-9][0-9]*$ ]] || { echo '[huanyu] model/token budgets must be positive integers' >&2; exit 2; }
done
if (( _huanyu_rollout_prompt_length > _huanyu_prompt_length ||
      _huanyu_model_length < _huanyu_rollout_prompt_length + _huanyu_response_length ||
      _huanyu_token_budget < _huanyu_sequence_length )); then
    echo '[huanyu] require rollout prompt <= data prompt, model length >= rollout prompt + response, and token budget >= data prompt + response' >&2
    exit 2
fi
export MAX_PROMPT_LENGTH=$_huanyu_prompt_length MAX_RESPONSE_LENGTH=$_huanyu_response_length
export ROLLOUT_PROMPT_LENGTH=$_huanyu_rollout_prompt_length ROLLOUT_MAX_MODEL_LEN=$_huanyu_model_length
export MAX_TOKEN_LEN_PER_GPU=$_huanyu_token_budget

echo "[huanyu] adapter=Qwen36HuanyuForTraining dataset=Qwen36HuanyuOmniRLHFDataset pipeline=qwen3_6_huanyu; HUANYU_USE_PRECOMPUTE=${HUANYU_USE_PRECOMPUTE}; length filter off"
echo "[huanyu] expanded prompt=${_huanyu_prompt_length} rollout prompt=${_huanyu_rollout_prompt_length} response=${_huanyu_response_length} model length=${_huanyu_model_length} token budget=${_huanyu_token_budget}"

# Keep HF config.architectures intact. architecture below selects only the
# verl training adapter; the RM retains its independent Qwen3.5 registration.
# The inherited vLLM pipeline already dispatches audio_config to Huanyu.
exec bash /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/examples/qwen3_6_whisper/run_qwen36_whisper_32card.sh \
    "$@" \
    actor_rollout_ref.model.external_lib=verl_omni_ext.models.qwen3_6_huanyu \
    actor_rollout_ref.model.architecture=Qwen36HuanyuForTraining \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.use_remove_padding=false \
    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset \
    data.custom_cls.name=Qwen36HuanyuOmniRLHFDataset \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_6_huanyu \
    data.max_prompt_length="${_huanyu_prompt_length}" \
    data.max_response_length="${_huanyu_response_length}" \
    actor_rollout_ref.rollout.prompt_length="${_huanyu_rollout_prompt_length}" \
    actor_rollout_ref.rollout.response_length="${_huanyu_response_length}" \
    actor_rollout_ref.rollout.max_model_len="${_huanyu_model_length}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${_huanyu_model_length}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${_huanyu_token_budget}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${_huanyu_token_budget}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${_huanyu_token_budget}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${_huanyu_ray_pythonpath}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_VLLM_OMNI_ROOT=${HUANYU_VLLM_OMNI_ROOT}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_USE_PRECOMPUTE='${HUANYU_USE_PRECOMPUTE}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_EMBED_CACHE_DIR=${_huanyu_ray_cache_dir}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MINICPMO_USE_PRECOMPUTE='0'" \
    data.filter_overlong_prompts=false
