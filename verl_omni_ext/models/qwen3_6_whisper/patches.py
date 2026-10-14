"""
L2: MiniCPM-o 5.0 monkey patch（2 个补丁）

两个补丁打的都是第三方代码（transformers AutoClass / 模型实例），这是 L2 的
正确用法。模块级补丁必须在任何 remote code 被 import 之前执行——由
``__init__.py`` 在包 import 期调用（``from_pretrained`` 先于
``configure_model``，见包内 CLUA.md 时序陷阱）。

本文件只放 import 期 AutoClass 补丁；实例级 forward / processor / mm_inputs
适配移到了同目录独立模块（``forward.py`` / ``processor.py`` / ``mm_inputs.py``），
由 ``thinker_adapter.py`` 在 ``configure_*`` 里直接引用，不再经由这里转发。

逻辑移植自 verl-omni 树内 ``models/transformers/minicpmo_5_0.py``，但改用
``_patchkit.idempotent_patch`` 封装：assert_target + 签名指纹 + strict 模式 +
跨进程一致性。
"""
import logging

from verl_omni_ext._patchkit import idempotent_patch

logger = logging.getLogger(__name__)

# Auto classes whose ``.register()`` the MiniCPM-o remote code may call at
# import time.  Only AutoImageProcessor is known to be broken in the released
# checkpoint; the rest are listed because processing_minicpmo.py registers
# several classes in a row and a future checkpoint revision may move the bad
# call.  Extend this tuple if `grep -n "\.register(" <ckpt>/*.py` turns up an
# AutoClass that is not here.
_GUARDED_AUTO_CLASSES = (
    "AutoImageProcessor",
    "AutoProcessor",
    "AutoFeatureExtractor",
    "AutoTokenizer",
    "AutoModel",
    "AutoModelForCausalLM",
)


# ============================================================================
# 补丁 1：AutoClass register guard（6 类）
# ============================================================================
@idempotent_patch(
    name="minicpmo_auto_register_guard",
    target_module="transformers",
    target_attr="AutoImageProcessor",
    fingerprint="transformers>=4.46",
    expected_signature="register",
    probe_signature=True,  # 记录 register 签名指纹，跨进程一致性校验用
)
def apply_minicpmo_auto_register_guard(original_register, __patch_signature__=None):
    """守卫 6 个 AutoClass 的 register 方法。

    原因：checkpoint 的 processing_minicpmo.py:478 用 str 当 config class 传
    ``AutoImageProcessor.register("MiniCPMVImageProcessor", ...)``，新版
    transformers 用 ``key.__module__`` 查 mapping 时直接
    AttributeError，连带 modeling 的 import 一起炸。

    为什么跳过是安全的：mapping 按 config class 查，str key 永远命中不了，这个
    register 本来就是死代码——变成 no-op 不改变任何运行时行为。

    register 是 staticmethod，所以替换体必须 ``staticmethod`` 包裹、签名是
    ``(config_class, *args, **kwargs)``；裸函数会让 str 落进 ``self``，守卫失效
    （这是早前 ext 版本的真实 bug，见 summary）。主目标 AutoImageProcessor 的
    替换体交给 idempotent_patch 落盘，其余 5 类在函数体内同步打（防 checkpoint
    把坏调用移到别的类上）。
    """
    _ = __patch_signature__  # 签名漂移时可在此分支适配

    import transformers

    def _make_guard(original, name):
        def _register(config_class, *args, _orig=original, _name=name, **kwargs):
            if isinstance(config_class, str):
                logger.warning(
                    "%s.register() was called with a string key %r by remote code; skipping. "
                    "The registration was already inert -- the mapping is keyed by config "
                    "class, so a str key could never be hit.",
                    _name,
                    config_class,
                )
                return None
            return _orig(config_class, *args, **kwargs)

        return _register

    primary = _make_guard(original_register, "AutoImageProcessor")
    for name in _GUARDED_AUTO_CLASSES:
        if name == "AutoImageProcessor":
            continue
        auto_cls = getattr(transformers, name, None)
        if auto_cls is None or getattr(auto_cls, _REGISTER_FLAG, False):
            continue
        auto_cls.register = staticmethod(_make_guard(auto_cls.register, name))
        setattr(auto_cls, _REGISTER_FLAG, True)

    return staticmethod(primary)


_REGISTER_FLAG = "_verl_omni_minicpmo_register_str_key_guard"


# ============================================================================
# 补丁 2：AutoModelForMultimodalLM fallback
# ============================================================================
@idempotent_patch(
    name="minicpmo_automodel_fallback",
    target_module="transformers",
    target_attr="AutoModelForMultimodalLM",
    fingerprint="transformers>=4.46",
    expected_signature="from_pretrained",
    probe_signature=True,  # 记录 from_pretrained 签名指纹，跨进程一致性校验用
)
def apply_minicpmo_automodel_fallback(original_from_pretrained, __patch_signature__=None):
    """让 ``AutoModelForMultimodalLM.from_pretrained`` fallback 到 CausalLM/AutoModel。

    原因：``omni_impl.py`` 硬编码 ``AutoModelForMultimodalLM.from_pretrained``，
    而 MiniCPM-o 的 config.json auto_map 只映射 AutoConfig/AutoModel/
    AutoModelForCausalLM，查 ``AutoModelForMultimodalLM`` 直接 ValueError——
    权重还没碰到就炸。

    只在 ``trust_remote_code`` 时重试（先 AutoModelForCausalLM 再 AutoModel，
    两者都解析到 remote code 的同一个 MiniCPMO 类），其余失败原样抛。
    ``from_pretrained`` 是 classmethod，故第一个参数必须是
    ``pretrained_model_name_or_path`` 而非 ``cls``——这样在类上赋值纯函数后，
    ``AutoModelForMultimodalLM.from_pretrained(path)`` 调用恰好把 path 落进
    第一个形参（early ext 版的 ``cls``-channeling 是侥幸工作的）。
    """
    _ = __patch_signature__

    def _from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        try:
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except (ValueError, KeyError) as primary_exc:
            if not kwargs.get("trust_remote_code", False):
                raise
            from transformers import AutoModel, AutoModelForCausalLM

            for auto_cls in (AutoModelForCausalLM, AutoModel):
                try:
                    logger.warning(
                        "AutoModelForMultimodalLM could not resolve %s (%s); retrying with %s. "
                        "Add 'AutoModelForMultimodalLM' to the checkpoint's config.json "
                        "auto_map to silence this.",
                        pretrained_model_name_or_path,
                        primary_exc,
                        auto_cls.__name__,
                    )
                    return auto_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                except (ValueError, KeyError):
                    continue
            raise

    return _from_pretrained

