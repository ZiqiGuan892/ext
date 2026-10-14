"""
全双工 omni RL trainer（L1 插件，零侵入）

通过 verl 的 @register_trainer 注册表注册——不修改 verl/verl-omni 任何源码。

全双工 = 训练和推理真正并发（不是交替）：
  传统 RL：训练 → 停 → 推理生成 rollout → 停 → 训练（串行）
  全双工：  训练和推理同时进行，推理结果实时反馈到训练

verl 已有的异步基础：
  - @register_trainer("separate_async") — 训练和 rollout 分离，可部分重叠
  - @register_trainer("colocate_async") — 训练和 rollout 共置，partial rollout
  - FullyAsyncLLMServerClient — 异步推理服务客户端
  - agent_loop_tq — TransferQueue 异步数据流
  - checkpoint_manager.update_weights() — 权重同步

本 trainer 继承 OmniPPOTrainerSync，改为异步并发执行。
"""

import logging
import os

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("omni_fullduplex")
class OmniPPOTrainerFullDuplex(OmniPPOTrainerSync):
    """全双工 omni PPO trainer

    与 omni_sync 的区别：
    1. 训练和推理并发（不是串行交替）
    2. 使用 FullyAsyncLLMServerClient 进行异步推理
    3. 权重定期同步（parameter_sync_step 控制频率）
    4. 推理结果实时反馈到训练队列

    配置方式（config.yaml）：
        trainer:
          v1:
            trainer_name: omni_fullduplex   # 通过 @register_trainer 注册的名字
            fullduplex:
              num_warmup_batches: 2          # 预热：往推理队列放的 batch 数
              parameter_sync_step: 4         # 每 4 个 mini-batch 同步一次权重

    加载链路（零侵入）：
      1. main_omni.py 的 uses_v1_trainer() 返回 True（trainer_type=policy_gradient）
      2. run_ppo() 加载 V1 trainer 系统
      3. @register_trainer("omni_fullduplex") 从 config.trainer.v1.trainer_name 匹配
      4. 本 trainer 被实例化

      全程不需要改 main_omni.py 或 verl 的任何源码——
      @register_trainer 是 verl 的公开扩展点（注册表字典）。
    """

    def get_llm_client(self):
        """使用异步推理客户端（不是同步的）"""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_train_begin(self):
        """预热：往推理队列放 warmup batches，让推理先跑起来"""
        if self.config.skip.rollout_tq.enable:
            return

        num_warmup = self.config.trainer.v1.get("fullduplex", {}).get(
            "num_warmup_batches", 2
        )
        for _ in range(num_warmup):
            self._add_batch_to_generate()

        logger.info(
            f"FullDuplex: added {num_warmup} warmup batches to agent loop. "
            f"Training and inference will run concurrently."
        )

    def on_step_end(self):
        """每 parameter_sync_step 步同步一次权重到推理引擎"""
        sync_step = self.config.trainer.v1.get("fullduplex", {}).get(
            "parameter_sync_step", 4
        )

        if self.global_steps % sync_step == 0:
            with marked_timer("update_weights", self.timing_raw, color="red"):
                self.checkpoint_manager.update_weights(self.global_steps)
                logger.debug(
                    f"FullDuplex: weights synced at step {self.global_steps}"
                )

    def on_train_end(self):
        """训练结束时清理异步推理"""
        logger.info("FullDuplex: training ended, cleaning up async inference")
        # agent_loop_manager 会自动清理


# ============================================================================
# 推理侧的依赖（vllm-omni experimental/fullduplex）
# ============================================================================
#
# 训练侧零侵入了，但推理侧需要 vllm-omni 的 fullduplex 支持：
#
# vllm-omni 已有实验性全双工代码：
#   vllm_omni/experimental/fullduplex/
#     ├── __init__.py          (DuplexAdapter, DuplexRuntime, DuplexSession)
#     ├── core/runtime.py      (DuplexRuntime — 全双工推理运行时)
#     ├── core/session.py      (DuplexSession — 会话管理)
#     ├── core/adapter.py      (DuplexAdapter — 输入/输出适配)
#     ├── personaplex/         (全双工会话管理实验)
#     └── request_client.py    (异步请求客户端)
#
# 但这些是实验性的，且：
#   1. 需要适配到你的具体模型（类似 rollout 侧的 pipeline.py 适配）
#   2. vllm-omni 没有 plugin 机制——必须改源码树
#   3. 属于 rollout 侧的侵入面（见 docs/rollout_adaptation.md）
#
# 如果你的全双工只做"训练和推理并发"（不做真正的流式推理），
# 可能不需要 vllm-omni 的 fullduplex——verl 的 FullyAsyncLLMServerClient
# 已经足够支持"训练和推理并发 + 权重同步"。