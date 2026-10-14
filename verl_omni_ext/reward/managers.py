"""
Reward Manager 注册

通过 verl 的 @register(name) 装饰器注册到 REWARD_MANAGER 字典。
verl 在运行时通过 reward_model.reward_manager 配置字段查找。

用法（config.yaml）：
  reward:
    reward_model:
      reward_manager: audio_quality_reward
"""
import logging

logger = logging.getLogger(__name__)


def _import_reward_registry():
    """延迟 import verl 的 reward 注册表

    verl 的 reward_loop 是实验性模块，可能没装。
    用 try/except 降级，不影响其他模块。
    """
    try:
        from verl.experimental.reward_loop.reward_manager.registry import register
        from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
        return register, RewardManagerBase
    except ImportError:
        logger.warning("verl reward_loop not available, reward managers not registered")
        return None, None


register, RewardManagerBase = _import_reward_registry()


if register is not None and RewardManagerBase is not None:
    @register("audio_quality_reward")
    class AudioQualityRewardManager(RewardManagerBase):
        """音频质量 reward 管理器

        使用音频质量指标（MCD、F0 相关性、频谱损失）计算 reward。
        适用于 omni 模型的音频输出评估。

        与文本 reward 的区别：
        - 文本 reward：基于规则或 LLM 打分
        - 音频 reward：需要信号处理（MCD、F0、频谱）
        """

        def __init__(self, config=None):
            self.config = config or {}
            self.mcd_weight = self.config.get("mcd_weight", 0.4)
            self.f0_weight = self.config.get("f0_weight", 0.3)
            self.spectral_weight = self.config.get("spectral_weight", 0.3)

        def compute_reward(self, generated_audio, target_audio, **kwargs):
            """计算音频 reward

            Args:
                generated_audio: 生成的音频
                target_audio: 目标音频

            Returns:
                float: 综合 reward 分数 (0-1)
            """
            # MCD（梅尔倒谱失真，越小越好）
            mcd_score = self._compute_mcd(generated_audio, target_audio)
            # F0 相关性（越大越好）
            f0_score = self._compute_f0_correlation(generated_audio, target_audio)
            # 频谱损失（越小越好）
            spectral_score = self._compute_spectral_loss(generated_audio, target_audio)

            reward = (
                self.mcd_weight * mcd_score
                + self.f0_weight * f0_score
                + self.spectral_weight * spectral_score
            )
            return float(reward)

        def _compute_mcd(self, gen, tgt):
            """MCD → 转换为 0-1 分数"""
            return 0.5  # 实现省略

        def _compute_f0_correlation(self, gen, tgt):
            """F0 相关性"""
            return 0.5  # 实现省略

        def _compute_spectral_loss(self, gen, tgt):
            """频谱损失 → 转换为 0-1 分数"""
            return 0.5  # 实现省略