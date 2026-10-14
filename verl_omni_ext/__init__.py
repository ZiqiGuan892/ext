"""
verl-omni-ext: verl-omni 的 out-of-tree 扩展包

三层分治架构（按对象所有权分层）：
  L1 插件 (≥95%)       — 自己的适配代码，走 verl/verl-omni 预留的扩展点
  L2 monkey patch (~4%) — 第三方库/checkpoint remote code
  L3 gate patch (≤1%)  — verl-omni 自身、扩展点够不着的地方

加载方式：
  生产路径——启动脚本里设置环境变量（自动激活）：
    export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext

  verl-omni 会在每个 Ray worker 进程里调用 import_external_libs("verl_omni_ext")，
  执行本文件；因 env 含 verl_omni_ext，文件末尾自动调 activate()，触发入口点
  自动发现 + algos/features 副作用。

  无 env 的路径（CPU 单测 / 契约检查）——`import verl_omni_ext` 后显式调
  `verl_omni_ext.activate()`：import 子模块（如 models.common）不再连带拉起
  vllm_omni / 重量级依赖，遵守 CPU import 契约。

扩展点覆盖：
  模型适配器   — @OmniModelBase.register           → entry_points("verl_omni.models")
  推理拓扑     — @OmniRolloutPipelineBase.register → entry_points("verl_omni.models")
  训练范式     — @register_trainer                 → entry_points("verl_omni.trainers")
  reward 管理器 — @register (reward_loop)          → entry_points("verl_omni.reward")
  优势估计器   — @register_adv_est                 → import 触发（algos/）
  policy loss  — @register_policy_loss             → import 触发（algos/）
  数据集       — data.custom_cls                   → import 触发（models/）
  自定义 worker — worker_cls 配置                   → import 触发（workers/）
"""

import logging
import os
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# 所有入口点组——加新组只需在这里加一行
_ENTRY_POINT_GROUPS = [
    "verl_omni.models",     # 模型适配器 + 推理拓扑
    "verl_omni.trainers",   # 训练范式（如全双工）
    "verl_omni.reward",     # reward 管理器
]


def _load_all() -> None:
    """入口点自动发现：遍历所有入口点组，逐个 import 触发 @register 装饰器。

    每个 import 用 try/except 降级——继承 MiniCPM-o rollout adapter 的正确模式：
    如果运行环境的依赖没装上，硬失败会把已跑通的其他模块一起带下水。
    降级后只 warning，不影响其他模块。

    这取代了旧方案里手动编辑 verl_omni/pipelines/__init__.py 添加
    from .qwen35_moe import ... 的做法。
    """
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+: 用 select 遍历多个组
            groups = {group: eps.select(group=group) for group in _ENTRY_POINT_GROUPS}
        else:
            groups = {group: eps.get(group, []) for group in _ENTRY_POINT_GROUPS}
    except Exception as e:
        logger.error("Failed to enumerate entry_points: %s", e)
        groups = {}

    for group, group_eps in groups.items():
        for ep in group_eps:
            try:
                ep.load()  # import 触发 @register + 模块级 patch
                logger.info("Loaded %s: %s -> %s", group, ep.name, ep.value)
            except Exception as e:
                # 降级：某个模块加载失败不影响其他模块。logger 可能没有 handler（Ray
                # worker 里 logging 未配置会吞掉），所以用 print 兜底，让验证/契约
                # 检查能直接看到"谁被跳过、为什么"——静默失败比崩溃贵。
                logger.warning(
                    "Plugin %r in group %s unavailable (skipped): %s",
                    ep.name, group, e,
                )
                print(
                    f"[verl-omni] WARNING plugin {ep.name!r} in {group} "
                    f"unavailable (skipped): {e}",
                    flush=True,
                )


_ACTIVATED = False


def activate() -> None:
    """显式激活所有插件：入口点自动发现 + algos/features 副作用 import。

    不再在模块 import 时自动执行——这样 import 子模块（如
    ``verl_omni_ext.models.common``）不会连带拉起 vllm_omni / 重量级依赖，
    遵守 common.py 文档里声明的 CPU import 契约。幂等。

    调用时机：
    - 生产路径：启动脚本 export VERL_USE_EXTERNAL_MODULES 含 verl_omni_ext，
      本文件 import 时自动调（见下方），或 verl 的 import_external_libs 显式触发。
    - CPU 单测 / 契约检查：import verl_omni_ext 后手动调 ``verl_omni_ext.activate()``。
    """
    global _ACTIVATED
    if _ACTIVATED:
        return
    _ACTIVATED = True
    _load_all()
    # algos 通过 import 触发 @register_adv_est / @register_policy_loss。
    # features 下放跨域特性（如全双工 = trainer + worker 协同）。
    # trainer 打 L2 patch（env 注入 run_omni），让 MINICPMO_*/GENRM_* 到达 Ray worker。
    from . import algos    # noqa: F401
    from . import features  # noqa: F401
    from . import trainer  # noqa: F401

    # Diagnostic hooks belong to this test copy of ext; shared dependencies on
    # disk remain unchanged. Disabled by default and never required for training.
    if os.environ.get("MINICPMO_ROLLOUT_DIAG") == "1":
        try:
            from .probes.diagnostic_io import startup
            from .probes.rollout_trace import install

            startup()
            install()
        except Exception as exc:
            # Observability must not add noise to the trainer's tee/log stream.
            try:
                from .probes.diagnostic_io import emit

                emit("diagnostic_install_error", error_type=type(exc).__name__, error=str(exc))
            except Exception:
                pass


# 仅在 VERL_USE_EXTERNAL_MODULES 显式包含 verl_omni_ext 时自动激活（生产路径）；
# 单测/CPU/契约检查环境不设这个环境变量，import 子模块不会被污染、不会拉起
# vllm_omni。契约检查需手动调 verl_omni_ext.activate()。
if "verl_omni_ext" in os.environ.get("VERL_USE_EXTERNAL_MODULES", "").split(","):
    activate()
