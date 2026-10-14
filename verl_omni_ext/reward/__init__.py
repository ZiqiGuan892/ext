"""
Reward 扩展模块

三个扩展点：
  1. RewardManager 注册表 —— @register("xxx") 注册到 verl 的 REWARD_MANAGER 字典
  2. custom_reward_function —— config 里 path=pkg://verl_omni_ext.reward.functions.xxx
  3. GenRM verifier —— config 里 path=pkg://verl_omni_ext.reward.genrm_verifier

配置方式（config.yaml）：
  # 方式 1：注册 reward manager
  reward:
    reward_model:
      reward_manager: my_audio_reward    # @register 的名字

  # 方式 2：自定义 reward 函数路径（rule-based）
  reward:
    custom_reward_function:
      path: pkg://verl_omni_ext.reward.functions.audio_quality
      name: compute_score

  # 方式 3：GenRM（model-based reward，Qwen3.5-35B-A3B judge）
  reward:
    custom_reward_function:
      path: pkg://verl_omni_ext.reward.genrm_verifier
      name: compute_score
    reward_model:
      enable: True
      enable_resource_pool: True
      model_path: <judge checkpoint>
"""
from . import managers, functions  # noqa: F401
from .genrm_verifier import compute_score, extract_final_answer

__all__ = ["compute_score", "extract_final_answer"]