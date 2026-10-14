"""
L2 monkey patch 公共基建（增强版）

所有 monkey patch 必须走本模块提供的装饰器，强制满足规范：
  (a) 打补丁前断言目标存在且签名符合预期
  (b) 记录被 patch 版本的指纹（transformers 版本区间等）
  (c) 有对应的 CPU 测试验证补丁行为
  (d) 返回 bool，启动脚本自检时断言为 True

多进程/分布式下的额外保证（本增强版新增）：
  (e) 签名指纹：patch 应用前对目标做 inspect.signature 指纹哈希。
      upstream 改函数签名时，指纹变化可被检测——不再"打上但语义已错"。
  (f) 跨进程传播证明：每个 worker 用 get_patch_state() / patch_state_line()
      上报本地 patch 状态，driver 用 assert_patch_consensus() 断言
      所有 worker 的 patch 集合、applied 状态、签名指纹一致。
      没走 external_libs 的 worker 会在这里被抓住，而不是静默缺失。
  (g) 运行时 watchdog：verify_patches_alive() 复查已打补丁的 flag 是否还在
      （防止被 reload/其他库覆盖），可挂到 step loop 定期调用。
  (h) strict 模式：VERL_OMNI_EXT_PATCH_STRICT=1 时，任何 patch 未打上
      直接 raise（fail-fast），而不是返回 False 静默继续。

适用对象（L2 的准入规则）：
  只能打第三方代码——transformers / vllm / checkpoint remote code / 模型实例。
  打 verl-omni 自己的东西说明该用 L3 gate patch 或该给上游提扩展点。
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _strict_enabled() -> bool:
    """strict 模式：patch 未打上直接 raise（fail-fast），不静默继续。

    运行时动态读取 env——进程启动后也能打开，不要求模块加载时生效。
    """
    return os.environ.get("VERL_OMNI_EXT_PATCH_STRICT", "0") == "1"

# 全局注册表：记录所有已打的补丁，供启动脚本自检
_PATCH_REGISTRY: dict[str, "PatchRecord"] = {}


class PatchRecord:
    """一条补丁的注册记录"""

    __slots__ = ("name", "target", "fingerprint", "applied", "signature_fp")

    def __init__(self, name: str, target: str, fingerprint: str, applied: bool,
                 signature_fp: str = "N/A"):
        self.name = name          # 补丁名（如 "minicpmo_auto_register_guard"）
        self.target = target      # 被打的对象（如 "transformers.AutoImageProcessor"）
        self.fingerprint = fingerprint  # 版本指纹（如 "transformers>=4.46,<4.50"）
        self.applied = applied    # 是否真的打上了
        self.signature_fp = signature_fp  # 被替换目标应用前的签名指纹（防漂移）

    def __repr__(self):
        status = "✓ applied" if self.applied else "✗ skipped"
        return (
            f"<Patch {self.name} -> {self.target} [{self.fingerprint}] "
            f"{status} sig={self.signature_fp}>"
        )


def _signature_fingerprint(obj: Any) -> str:
    """计算目标可调用对象的签名指纹（防上游改签名导致的语义漂移）。

    对 inspect.signature 的结果做 sha256，取前 16 位。
    builtin / 无法内省的对象返回 "N/A"（不参与指纹校验）。
    """
    try:
        sig = inspect.signature(obj)
        return hashlib.sha256(str(sig).encode("utf-8")).hexdigest()[:16]
    except (ValueError, TypeError):
        return "N/A"


def assert_target(
    module_path: str,
    attr_name: str,
    expected_signature: str | None = None,
) -> Any:
    """前置断言：被 patch 的目标必须存在，签名必须匹配。

    这是防止"静默失效"的第一道防线。如果上游改了名/删了函数，
    这里会直接抛异常，而不是让补丁默默打到一个不存在的东西上。

    Args:
        module_path: 如 "transformers"
        attr_name: 如 "AutoImageProcessor"
        expected_signature: 可选，期望的函数签名片段（如 "register"）

    Returns:
        被断言通过的对象

    Raises:
        AssertionError: 目标不存在或签名不匹配
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise AssertionError(
            f"Patch target module '{module_path}' not importable: {e}"
        ) from e

    target = getattr(module, attr_name, None)
    if target is None:
        raise AssertionError(
            f"Patch target '{module_path}.{attr_name}' does not exist — "
            f"upstream may have renamed or removed it. "
            f"Do NOT proceed: patching a non-existent target is a silent no-op."
        )

    if expected_signature is not None:
        # 检查是否有期望的方法/属性
        if not hasattr(target, expected_signature):
            raise AssertionError(
                f"'{module_path}.{attr_name}' has no attribute '{expected_signature}' — "
                f"upstream API changed. Patch needs updating."
            )

    return target


def _record(name: str, target: str, fingerprint: str, applied: bool,
            signature_fp: str = "N/A", reason: str = "") -> bool:
    """登记补丁状态。strict 模式下未打上直接 raise。

    统一所有 return False 的出口，保证：
      - 状态一定进 registry（无论成败）
      - strict 模式下失败即 raise（fail-fast，不静默）
    """
    _PATCH_REGISTRY[name] = PatchRecord(
        name, target, fingerprint, applied, signature_fp
    )
    if not applied and _strict_enabled():
        raise RuntimeError(
            f"[VERL_OMNI_EXT_PATCH_STRICT] Patch {name!r} not applied: {reason}"
        )
    return applied


def idempotent_patch(
    name: str,
    target_module: str,
    target_attr: str,
    fingerprint: str = "",
    expected_signature: str | None = None,
    flag_attr: str | None = None,
    probe_signature: bool = False,
):
    """装饰器：把一个函数变成幂等的 monkey patch。

    被装饰的函数接收 original（被替换的原函数/原属性）作为第一个参数，
    返回替换后的函数。

    保证：
      - 幂等：重复调用无害（用 flag_attr 标志位）
      - 前置断言：目标存在且签名匹配
      - 版本指纹：记录在 PatchRecord 里
      - 签名指纹：probe_signature=True 时对目标做 inspect.signature 哈希，
        存入 PatchRecord（跨进程一致性校验 / 漂移检测用）
      - 签名注入：probe_signature=True 时把原目标签名指纹以
        `__patch_signature__` 关键字传给 patch_fn，patch_fn 可按签名分支
      - 返回 bool：True=打上了，False=跳过了（strict 模式下 raise）
      - strict：VERL_OMNI_EXT_PATCH_STRICT=1 时未打上即 raise

    Args:
        name: 补丁名（唯一标识）
        target_module: 如 "transformers"
        target_attr: 如 "AutoImageProcessor"
        fingerprint: 版本区间，如 "transformers>=4.46,<4.50"
        expected_signature: 期望的签名片段（用于前置断言）
        flag_attr: 幂等标志位属性名（默认 "_verl_omni_ext_patched_" + name）
        probe_signature: 是否探测并记录目标签名指纹（默认 False，兼容旧用法）

    Usage:
        @idempotent_patch(
            name="minicpmo_auto_register_guard",
            target_module="transformers",
            target_attr="AutoImageProcessor",
            fingerprint="transformers>=4.46",
            expected_signature="register",
            probe_signature=True,   # 记录签名指纹 + 注入 __patch_signature__
        )
        def patch_auto_register_guard(original, __patch_signature__=None):
            # __patch_signature__ = 原目标签名指纹（如 "a1b2c3d4e5f60718"）
            # 签名变了时这里能看到并分支适配，而不是静默失效
            ...
    """
    if flag_attr is None:
        flag_attr = f"_verl_omni_ext_patched_{name}"

    def decorator(patch_fn: Callable) -> Callable:
        @functools.wraps(patch_fn)
        def wrapper(*args, **kwargs) -> bool:
            # 幂等检查
            try:
                module = importlib.import_module(target_module)
            except ImportError:
                logger.warning("Patch %r: module %s not importable, skipping", name, target_module)
                return _record(name, f"{target_module}.{target_attr}",
                               fingerprint, False, reason=f"module {target_module} not importable")

            target = getattr(module, target_attr, None)
            if target is None:
                logger.warning("Patch %r: target %s.%s not found, skipping", name, target_module, target_attr)
                return _record(name, f"{target_module}.{target_attr}",
                               fingerprint, False, reason="target not found")

            # 幂等：已打过就跳过
            if getattr(target, flag_attr, False):
                logger.debug("Patch %r: already applied, skipping", name)
                return True

            # 前置断言（弱断言：只查属性存在，参数签名靠 probe_signature）
            if expected_signature is not None:
                if not hasattr(target, expected_signature):
                    logger.error(
                        "Patch %r: %s.%s missing expected signature %r — "
                        "upstream API changed, NOT patching to avoid silent failure",
                        name, target_module, target_attr, expected_signature,
                    )
                    return _record(name, f"{target_module}.{target_attr}",
                                   fingerprint, False,
                                   reason=f"missing expected signature {expected_signature!r}")

            # 签名指纹：在替换前探测原目标（probe_signature=True 时）
            sig_fp = _signature_fingerprint(target) if probe_signature else "N/A"

            # 执行补丁
            original = getattr(target, expected_signature, target) if expected_signature else target
            try:
                if probe_signature:
                    # 把原目标签名指纹注入 patch_fn，支持按签名分支适配
                    replacement = patch_fn(original, __patch_signature__=sig_fp)
                else:
                    replacement = patch_fn(original)
                if expected_signature:
                    setattr(target, expected_signature, replacement)
                else:
                    setattr(module, target_attr, replacement)
                setattr(target, flag_attr, True)
                # flag 也打在 replacement 上：verify_patches_alive 检查的是
                # module.target_attr（已被替换为 replacement），不是原始 target。
                setattr(replacement, flag_attr, True)
                logger.info("Patch %r applied on %s.%s [%s] sig=%s",
                            name, target_module, target_attr, fingerprint, sig_fp)
                _PATCH_REGISTRY[name] = PatchRecord(
                    name, f"{target_module}.{target_attr}", fingerprint, True, sig_fp
                )
                return True
            except Exception as e:
                logger.error("Patch %r failed: %s", name, e)
                return _record(name, f"{target_module}.{target_attr}",
                               fingerprint, False, reason=f"exception: {e}")

        return wrapper

    return decorator


def self_check() -> dict[str, bool]:
    """启动脚本自检：断言所有 L2 补丁都已打上。

    在 examples/*/run_*.sh 的前置自检里调用：

        python -c "from verl_omni_ext._patchkit import self_check; \\
                    results = self_check(); \\
                    assert all(results.values()), results"

    如果有补丁没打上（目标不存在/签名变了），这里会响，
    而不是等到第一个 micro-batch 才炸。
    """
    return {name: rec.applied for name, rec in _PATCH_REGISTRY.items()}


def patch_summary() -> str:
    """人类可读的补丁状态摘要"""
    if not _PATCH_REGISTRY:
        return "No patches registered."
    lines = ["L2 monkey patch status:"]
    for name, rec in _PATCH_REGISTRY.items():
        status = "✓" if rec.applied else "✗"
        lines.append(f"  {status} {name} -> {rec.target} [{rec.fingerprint}] sig={rec.signature_fp}")
    return "\n".join(lines)


# ============================================================================
# 多进程/分布式传播证明
# ============================================================================

def get_patch_state() -> dict[str, dict[str, str | bool]]:
    """当前进程的 patch 状态快照（可 JSON 序列化）。

    Ray 分布式下每个 worker 调用本函数，把返回的 dict 上报给 driver，
    由 assert_patch_consensus() 做一致性校验。

    返回结构：
        {
          "patch_name": {
            "applied": True,
            "target": "transformers.AutoImageProcessor",
            "fingerprint": "transformers>=4.46",
            "signature_fp": "a1b2c3d4e5f60718",   # "N/A" 表示未探测
          },
          ...
        }
    """
    return {
        name: {
            "applied": rec.applied,
            "target": rec.target,
            "fingerprint": rec.fingerprint,
            "signature_fp": rec.signature_fp,
        }
        for name, rec in _PATCH_REGISTRY.items()
    }


def patch_state_line() -> str:
    """单行可解析的 patch 状态输出。

    每个 worker 打印这一行（带固定前缀），driver 用
    `grep "^PATCH_STATE " <worker_log>` 收集后做一致性校验。
    """
    return "PATCH_STATE " + json.dumps(get_patch_state(), sort_keys=True)


def assert_patch_consensus(
    states: list[dict[str, dict[str, str | bool]]],
    required: list[str] | None = None,
) -> tuple[bool, str]:
    """跨进程一致性校验：所有 worker 的 patch 状态必须一致。

    Args:
        states: 每个 worker 的 get_patch_state() 返回值组成的列表
        required: 必须存在的补丁名列表（默认取第一个 worker 的补丁集合）

    Returns:
        (ok, message): ok=True 表示所有 worker 一致且所有必需补丁已打上

    校验项：
      1. 所有 worker 的补丁名集合一致（缺 patch 的 worker 会被抓住）
      2. 每个必需补丁在所有 worker 里 applied=True
      3. 签名指纹一致（不同 worker 的 transformers 版本不同 → 指纹不同 →
         说明版本倾斜，补丁语义可能不一致）

    用法（driver / 启动脚本）：
        worker_states = collect_from_ray_workers()   # 见 patch_state_line()
        ok, msg = assert_patch_consensus(worker_states)
        assert ok, msg
    """
    if not states:
        return False, "no worker states collected"

    names = sorted(states[0].keys())
    required = required or names

    # 1. 补丁集合一致性
    for i, st in enumerate(states):
        if sorted(st.keys()) != names:
            return False, (
                f"worker {i} patch set differs: "
                f"{sorted(st.keys())} vs base {names}"
            )

    # 2. 必需补丁全部 applied
    for r in required:
        if r not in names:
            return False, f"required patch {r!r} missing from all workers"
        for i, st in enumerate(states):
            if not st[r]["applied"]:
                return False, f"worker {i} patch {r!r} NOT applied"

    # 3. 签名指纹一致性（版本倾斜检测）
    for r in required:
        fps = {st[r]["signature_fp"] for st in states}
        if len(fps) > 1:
            return False, (
                f"patch {r!r} signature fingerprint differs across workers: "
                f"{fps} — version skew, patch semantics may diverge"
            )

    return True, f"all {len(states)} workers consistent ({len(required)} patches)"


def verify_patches_alive() -> dict[str, bool]:
    """运行时 watchdog：复查已打补丁的 flag 是否还在。

    patch 打的是进程内全局对象——如果运行中被 reload（如 transformers
    被重新 import）、或被其他库覆盖，flag 会丢。本函数复查：

      - 目标对象还能找到
      - flag_attr 还在（说明补丁还在线上）

    返回 {name: alive_bool}。可挂到 trainer 的 step loop 或定时任务里：
        活着才继续，丢了立即报警而不是静默错下去。

    注意：签名指纹不在这里复查（运行中目标已被替换，指纹必然不同）；
    指纹一致性属于 assert_patch_consensus 的启动期职责。
    """
    results: dict[str, bool] = {}
    for name, rec in _PATCH_REGISTRY.items():
        if not rec.applied:
            results[name] = False
            continue
        try:
            # target 形如 "module.attr" 或 "module.attr.method"
            parts = rec.target.split(".")
            module = importlib.import_module(parts[0])
            obj = module
            for p in parts[1:]:
                obj = getattr(obj, p)
            flag = getattr(obj, f"_verl_omni_ext_patched_{name}", False)
            results[name] = bool(flag)
            if not flag:
                logger.warning("Patch %r flag missing at runtime — patch was lost/overwritten", name)
        except Exception as e:  # noqa: BLE001
            results[name] = False
            logger.warning("Patch %r target lost at runtime: %s", name, e)
    return results