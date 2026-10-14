"""
自定义 reward 函数

通过 config 的 custom_reward_function 路径加载：
  reward:
    custom_reward_function:
      path: pkg://verl_omni_ext.reward.functions.audio_quality
      name: compute_score

verl 通过 load_extern_object(path, name) 加载，和 data.custom_cls 同一机制。
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_score(data_source: str, solution_str: str, **kwargs) -> float:
    """默认 reward 函数

    Args:
        data_source: 数据集名
        solution_str: 模型输出
        **kwargs: 额外参数（ground_truth 等）

    Returns:
        float: reward 分数
    """
    # 示例：简单的文本匹配 reward
    ground_truth = kwargs.get("ground_truth", "")
    if not ground_truth:
        return 0.0

    # 精确匹配
    if solution_str.strip() == ground_truth.strip():
        return 1.0

    # 部分匹配
    overlap = len(set(solution_str.split()) & set(ground_truth.split()))
    total = len(set(ground_truth.split()))
    return overlap / total if total > 0 else 0.0


def audio_quality_score(data_source: str, solution_str: str, **kwargs) -> float:
    """音频质量 reward 函数

    用于 omni 模型的音频输出评估。
    可与 AudioQualityRewardManager 配合使用。
    """
    from verl_omni_ext.reward.managers import AudioQualityRewardManager

    generated_audio = kwargs.get("generated_audio")
    target_audio = kwargs.get("target_audio")

    if generated_audio is None or target_audio is None:
        logger.warning("audio_quality_score: missing audio data, returning 0")
        return 0.0

    manager = AudioQualityRewardManager()
    return manager.compute_reward(generated_audio, target_audio)


def multimodal_reward(data_source: str, solution_str: str, **kwargs) -> float:
    """多模态加权 reward

    结合文本 + 音频 + 视觉的加权 reward。
    """
    text_weight = kwargs.get("text_weight", 0.5)
    audio_weight = kwargs.get("audio_weight", 0.3)
    video_weight = kwargs.get("video_weight", 0.2)

    text_score = compute_score(data_source, solution_str, **kwargs)
    audio_score = audio_quality_score(data_source, solution_str, **kwargs)
    # video_score = ... 省略

    return text_weight * text_score + audio_weight * audio_score