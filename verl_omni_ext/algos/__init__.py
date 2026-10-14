"""
自定义算法模块

两个 verl 注册表：
  @register_adv_est("xxx")       — 优势估计器（verl/trainer/ppo/core_algos.py:116）
  @register_policy_loss("xxx")   — policy loss（verl/trainer/ppo/core_algos.py:53）

verl-omni 的 diffusion 注册表：
  @register_diffusion_adv_est   — diffusion 优势估计器
  @register_diffusion_loss      — diffusion loss

配置方式（config.yaml）：
  algorithm:
    adv_estimator: my_gae         # @register_adv_est 的名字
    policy_loss: my_loss          # @register_policy_loss 的名字
"""
from . import adv_est, policy_loss  # noqa: F401