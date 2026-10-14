"""
异步 rollout worker

用于全双工训练：推理 worker 独立运行，不受训练 step 阻塞。

verl 已有 FullyAsyncLLMServerClient，但某些模型需要更底层的 worker 定制：
  - 自定义权重同步策略（不完全 offload + partial update）
  - 自定义推理 batch 调度（按音频长度分桶）
  - 自定义 KV cache 管理

配置方式（config.yaml）：
  actor_rollout_ref:
    rollout:
      worker_cls: verl_omni_ext.workers.async_rollout.AsyncRolloutWorker
"""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class AsyncRolloutWorker:
    """异步 rollout worker（骨架）

    继承 verl 的 worker 基类，重写关键方法实现异步行为。

    ⚠ verl 的 worker 基类在不同版本间变化较大，
    实际使用时需要根据运行时的 verl 版本确定基类。
    本骨架展示接口契约，不做具体实现。
    """

    def __init__(self, config):
        self.config = config
        self.weight_sync_step = config.trainer.v1.get(
            "fullduplex", {}
        ).get("parameter_sync_step", 4)

        # 异步推理客户端
        self.llm_client = None  # 在 init 时设置

    def init_worker(self):
        """初始化 worker

        全双工：在 init 时就启动推理引擎，不等第一个训练 step
        """
        from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

        # 启动推理引擎（提前启动，不等训练）
        logger.info("AsyncRolloutWorker: starting inference engine early for fullduplex")
        self.llm_client = self._get_async_client()

    def update_weights(self, global_steps: int):
        """权重同步

        全双工：训练侧更新参数后，同步到推理侧。
        关键决策：完全更新 vs 增量更新
          - 完全更新：停推理 → 换权重 → 重启推理（简单但卡顿）
          - 增量更新：不停推理，热更新参数（复杂但无卡顿）

        verl 默认是完全更新（checkpoint_manager.update_weights）。
        如果要做增量更新，在这里覆写。
        """
        if global_steps % self.weight_sync_step != 0:
            return

        logger.info(f"AsyncRolloutWorker: syncing weights at step {global_steps}")
        # 默认：调用 verl 的 update_weights
        # 自定义：增量更新（如 LoRA adapter swap）
        self._default_update_weights(global_steps)

    def _get_async_client(self):
        """获取异步推理客户端"""
        # 复用 verl 的 FullyAsyncLLMServerClient
        from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
        return FullyAsyncLLMServerClient

    def _default_update_weights(self, global_steps):
        """默认权重同步：完全更新"""
        # 委托给 verl 的 checkpoint_manager
        pass

    def generate_sequences(self, prompts, **kwargs):
        """生成推理结果

        全双工：非阻塞返回，推理结果通过队列异步返回
        """
        if self.llm_client is None:
            raise RuntimeError("AsyncRolloutWorker not initialized")

        # 非阻塞提交推理请求
        return self.llm_client.submit(prompts, **kwargs)
