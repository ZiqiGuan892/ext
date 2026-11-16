"""模块级 post-import 钩子：目标模块执行完毕后立刻运行安装函数。

vLLM general plugin 在 arg 解析 / EngineCore / worker / registry 多处、很早地
被调用；此时直接 import NPU runner 或 omni scheduler 会把重依赖拉进前端进程，
还可能撞上循环 import。改为登记钩子：模块已加载 → 立即安装；尚未加载 →
在它第一次 import 完成时安装。安装函数抛出的异常原样冒泡到 import 点
（fail-closed），不会被吞成「模块没打补丁也照样跑」。
"""

import importlib.abc
import importlib.util
import sys
import threading

_HOOKS: dict[str, list] = {}
_RESOLVING: set[str] = set()
_LOCK = threading.RLock()


class _HookedLoader(importlib.abc.Loader):
    def __init__(self, loader, fullname):
        self._loader = loader
        self._fullname = fullname

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _run(self._fullname, module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        with _LOCK:
            if fullname not in _HOOKS or fullname in _RESOLVING:
                return None
            _RESOLVING.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            with _LOCK:
                _RESOLVING.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _HookedLoader(spec.loader, fullname)
        return spec


_FINDER = _PostImportFinder()


def _run(fullname, module):
    with _LOCK:
        hooks = _HOOKS.pop(fullname, [])
    for hook in hooks:
        hook(module)


def when_imported(fullname: str, hook) -> None:
    """``hook(module)`` 在 ``fullname`` 可用时恰好运行一次。"""
    with _LOCK:
        module = sys.modules.get(fullname)
        if module is None:
            _HOOKS.setdefault(fullname, []).append(hook)
            if _FINDER not in sys.meta_path:
                sys.meta_path.insert(0, _FINDER)
            return
    hook(module)


def pending() -> list[str]:
    with _LOCK:
        return sorted(_HOOKS)
