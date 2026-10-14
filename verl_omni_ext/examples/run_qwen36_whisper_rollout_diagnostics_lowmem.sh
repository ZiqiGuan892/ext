#!/usr/bin/env bash
# 请先自行准备服务器环境并启动 Ray，再用 bash 执行本脚本。
# 多节点需共用支持文件锁的诊断目录；同名日志会追加并沿用总限额。
# 每轮诊断请设置新的 MINICPMO_ROLLOUT_DIAG_DIR；训练日志由原入口写入当前工作目录。
set -eo pipefail

# 保留原诊断设置，使用独立目录，避免沿用上一轮已达限额的日志。
export MINICPMO_ROLLOUT_DIAG=1
export MINICPMO_ROLLOUT_DIAG_DIR=${MINICPMO_ROLLOUT_DIAG_DIR:-/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/diagnostics/actor_precision_lowmem_001}
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
export MINICPMO_EMBED_CACHE_DIR=/opt/huawei/dataset/infra_data/data/embed_cache_qwen36

# 保留 4096 输出长度、精度、batch 和每题 4 次采样；仍只诊断 2 步。
# batch=4、n=4 对应 16 个 Actor rank 每 rank 一条序列，不再缩小。
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
export EXPERIMENT_NAME=actor_precision_lowmem_001
export DEFAULT_LOCAL_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_lowmem_001/checkpoints
export TENSORBOARD_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_lowmem_001/tensorboard
export GENRM_DUMP_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_lowmem_001/genrm_dump
export MINICPMO_HANG_DUMP_DIR=/opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/runs/actor_precision_lowmem_001/hang_dump

# 降低 Actor/RM 推理并发和引擎显存预算、权重同步缓冲及 entropy 分块。
# 关闭参数预取，启用手动 optimizer offload；offload_policy=false 避免覆盖手动卸载。
# activation offload 沿用原入口的 false，避免重新引入已有的锁页内存问题。
# 覆盖放在命令末尾，确保不会被 OmniMix 入口的固定值覆盖；额外 Hydra 参数仍可追加。
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
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  reward.reward_model.rollout.max_num_seqs=4 \
  reward.reward_model.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=false \
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=false \
  actor_rollout_ref.actor.fsdp_config.offload_policy=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.entropy_from_logits_chunk_size=128 \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=128 \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true \
  trainer.total_training_steps=2 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"
