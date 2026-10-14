"""
自定义优势估计器

通过 verl 的 @register_adv_est 注册到优势估计器注册表。
verl 在运行时通过 algorithm.adv_estimator 配置字段查找。

用法（config.yaml）：
  algorithm:
    adv_estimator: gae_with_audio_penalty
"""
import logging

logger = logging.getLogger(__name__)


def _import_adv_est_registry():
    """延迟 import verl 的优势估计器注册表"""
    try:
        from verl.trainer.ppo.core_algos import register_adv_est
        return register_adv_est
    except ImportError:
        logger.warning("verl core_algos not available, adv_est not registered")
        return None


register_adv_est = _import_adv_est_registry()


if register_adv_est is not None:
    @register_adv_est("gae_with_audio_penalty")
    def gae_with_audio_penalty(data, omegaconf):
        """带音频惩罚的 GAE 优势估计

        在标准 GAE 基础上，对音频质量差的样本施加额外惩罚。

        用法：
          algorithm:
            adv_estimator: gae_with_audio_penalty

        与标准 GAE 的区别：
        - 标准 GAE：advantage = reward + γλ * next_advantage
        - 本算法：advantage = reward + γλ * next_advantage - audio_penalty

        audio_penalty 来源：reward 函数里的音频质量分（低于阈值时惩罚）
        """
        import torch
        from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_gae_advantage

        # 调用标准 GAE 计算
        adv = compute_gae_advantage(
            data=data,
            gamma=omegaconf.actor_rollout_ref.actor.gamma,
            lam=omegaconf.actor_rollout_ref.actor.lam,
        )

        # 叠加音频惩罚（如果 reward 里有音频质量信息）
        if "audio_quality" in data.batch:
            quality = data.batch["audio_quality"]
            threshold = omegaconf.algorithm.get("audio_quality_threshold", 0.5)
            penalty = torch.where(
                quality < threshold,
                omegaconf.algorithm.get("audio_penalty", -0.1),
                torch.zeros_like(quality),
            )
            adv = adv + penalty

        return adv