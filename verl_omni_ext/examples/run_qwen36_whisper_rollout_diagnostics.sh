#!/usr/bin/env bash
# 请先自行准备服务器环境并启动 Ray，再用 bash 执行本脚本。
# 多节点需共用支持文件锁的诊断目录；同名日志会追加并沿用总限额。
# 每轮诊断请设置新的 MINICPMO_ROLLOUT_DIAG_DIR；训练日志由原入口写入当前工作目录。
set -eo pipefail

# 单文件 rollout 诊断：rollout_diagnostics.jsonl。
export MINICPMO_ROLLOUT_DIAG=1
export MINICPMO_ROLLOUT_DIAG_DIR=${MINICPMO_ROLLOUT_DIAG_DIR:-/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/diagnostics/actor_precision_001}
export MINICPMO_DIAG_NUMERICS=1
export MINICPMO_DIAG_GDN_NATIVE=0
export MINICPMO_DIAG_STRICT_MM=0
export MINICPMO_ROLLOUT_DIAG_MAX_EVENTS=4
export MINICPMO_ROLLOUT_DIAG_MAX_RECORDS=2000
export MINICPMO_ROLLOUT_DIAG_MAX_BYTES=4194304
export MINICPMO_ROLLOUT_DIAG_TEXT_CHARS=256

# Qwen36 Actor + Qwen35 GenRM，使用已转换的 4B 来源数据。
export MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.6-35B-A3B-whisper-0814-rl
export RM_MODEL_PATH=/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B
export TRAIN_FILE=/opt/huawei/dataset/infra_data/data/dataset/train.parquet
export VAL_FILE=/opt/huawei/dataset/infra_data/data/dataset/validation.parquet
# 若实际 cache 不在默认位置，请运行前自行 export 正确路径；不要指向 media 图片目录。
export MINICPMO_EMBED_CACHE_DIR=${MINICPMO_EMBED_CACHE_DIR:-/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/embed_cache_4b_qwen36}

# DEBUG_MODE=1 会把 response 改回 512；用末尾的 total_training_steps 控制短诊断。
export DEBUG_MODE=0
export DEBUG_SAMPLES=0
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_N=4
export MAX_PROMPT_LENGTH=3072
export MAX_RESPONSE_LENGTH=4096
export ROLLOUT_PROMPT_LENGTH=3104
export ROLLOUT_MAX_MODEL_LEN=8192
export MAX_TOKEN_LEN_PER_GPU=8192
export VAL_BEFORE_TRAIN=false
export PROJECT_NAME=oc_rl_4b_gzq
export EXPERIMENT_NAME=actor_precision_001
export DEFAULT_LOCAL_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_001/checkpoints
export TENSORBOARD_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_001/tensorboard
export GENRM_DUMP_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_001/genrm_dump
export MINICPMO_HANG_DUMP_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_001/hang_dump

# 保留原入口的训练流程；末尾覆盖其固定 token budget，额外 Hydra 参数直接追加。
bash /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/examples/qwen3_6_whisper/run_qwen36_whisper_gspo_npu_omni_mix_genrm.sh \
  data.train_max_samples=-1 \
  data.val_max_samples=-1 \
  data.max_response_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.prompt_length=3104 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  trainer.total_training_steps=2 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"
