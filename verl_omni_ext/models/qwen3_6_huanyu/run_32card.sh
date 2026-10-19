#!/usr/bin/env bash
# Huanyu-only entry. Source env_export.sh and start Ray on all nodes beforehand.
# Existing backend owns batching, sampling, GenRM and training log tee.
set -euo pipefail

cd /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq

export MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-huanyu06b-iter2000-omni4b-hf-huanyu7tokens
export TRAIN_FILE=/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/train.parquet
export VAL_FILE=/opt/huawei/dataset/ccb_data/data/omni/omni_mix_f8_genrm_sampleall/validation.parquet
export RM_MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B
export FILTER_OVERLONG=false
export MINICPMO_USE_PRECOMPUTE=0
export MINICPMO_TUNE_AUDIO=0

echo '[huanyu] adapter=Qwen36HuanyuForTraining dataset=Qwen36HuanyuOmniRLHFDataset pipeline=qwen3_6_huanyu; online frozen audio; length filter off'

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
    data.filter_overlong_prompts=false
