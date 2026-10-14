"""
自定义 policy loss

通过 verl 的 @register_policy_loss 注册到 policy loss 注册表。
verl 在运行时通过 algorithm.policy_loss 配置字段查找。

用法（config.yaml）：
  algorithm:
    policy_loss: audio_aware_ppo
"""
import logging

logger = logging.getLogger(__name__)


def _import_policy_loss_registry():
    """延迟 import verl 的 policy loss 注册表"""
    try:
        from verl.trainer.ppo.core_algos import register_policy_loss
        return register_policy_loss
    except ImportError:
        logger.warning("verl core_algos not available, policy_loss not registered")
        return None


register_policy_loss = _import_policy_loss_registry()


if register_policy_loss is not None:
    @register_policy_loss("audio_aware_ppo")
    class AudioAwarePPOLoss:
        """音频感知的 PPO loss

        在标准 PPO clip loss 基础上，对音频 token 的 clip 范围做特殊处理：
        - 文本 token：标准 clip ε=0.2
        - 音频 token：更严格的 clip ε=0.1（音频对齐更敏感）

        用法：
          algorithm:
            policy_loss: audio_aware_ppo
        """

        def __init__(self, omegaconf):
            self.clip_range = omegaconf.actor_rollout_ref.actor.clip_range
            self.audio_clip_range = omegaconf.algorithm.get("audio_clip_range", 0.1)

        def __call__(self, data, response_mask, audio_mask=None):
            import torch
            import torch.nn.functional as F

            # 标准 PPO clip loss
            pi = data.batch["old_log_probs"]  # 旧策略
            logp = data.batch["log_probs"]    # 新策略
            advantages = data.batch["advantages"]

            # ratio = exp(logp - pi)
            ratio = torch.exp(logp - pi)

            # 标准 clip
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            # 音频 token 用更严格的 clip
            if audio_mask is not None:
                audio_clipped = torch.clamp(
                    ratio, 1 - self.audio_clip_range, 1 + self.audio_clip_range
                )
                clipped_ratio = torch.where(
                    audio_mask, audio_clipped, clipped_ratio
                )

            loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages,
            )
            return F.masked_mean(loss, response_mask)