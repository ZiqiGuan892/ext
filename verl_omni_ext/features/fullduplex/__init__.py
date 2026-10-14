"""
全双工 Omni RL 训练

跨域特性：需要 trainer（@register_trainer）+ async worker（worker_cls）协同。

组件：
  trainer.py       — @register_trainer("omni_fullduplex")，训练和推理并发
  async_worker.py  — 异步推理 worker，权重定期同步

配置方式（config.yaml）：
  trainer:
    v1:
      trainer_name: omni_fullduplex
      fullduplex:
        num_warmup_batches: 2
        parameter_sync_step: 4
  actor_rollout_ref:
    rollout:
      worker_cls: verl_omni_ext.features.fullduplex.async_worker.AsyncRolloutWorker

详见 docs/feature_fullduplex.md
"""
from . import trainer, async_worker  # noqa: F401