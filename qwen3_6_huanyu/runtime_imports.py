"""Check the native Huanyu dependency in the process that actually uses it.

This file also runs directly, without importing the ext package or loading model
weights. During native imports it excludes only its own CLI directory, which
may contain a stale vllm_omni.py from an earlier deployment. It restores the
search path afterwards, never chooses another checkout, and never replaces
an already imported vLLM module.
"""

import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import socket
import sys


NATIVE_MODULE = "vllm_omni.model_executor.models.minicpmo_5_0.huanyu_audio"
REQUIRED_SYMBOLS = (
    "HuanyuFbankFeatureExtractor", "HuanyuAudioEncoder", "huanyu_project_and_fold",
)
DEFAULT_ROOT = "/vllm-workspace/vllm-omni"


@contextmanager
def _without_cli_directory():
    """Do not treat files next to this CLI as top-level dependency packages.

    ``python /path/to/runtime_imports.py`` prepends its directory before PYTHONPATH.
    An old sibling vllm_omni.py therefore wins over the correctly selected checkout.
    Package-relative imports still use __path__; only the flat CLI directory is
    excluded. Preserve the order of all actual checkout paths and restore it even
    if an import fails.
    """
    directory = Path(__file__).resolve().parent
    original = list(sys.path)
    filtered = [entry for entry in original if Path(entry or os.curdir).resolve() != directory]
    if filtered == original:
        yield
        return
    sys.path[:] = filtered
    try:
        yield
    finally:
        sys.path[:] = original


def _origin(module):
    value = getattr(module, "__file__", None)
    return Path(value).resolve() if value else None


def _details(expected_root):
    root = Path(expected_root or DEFAULT_ROOT).resolve()
    target = root / "vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py"
    package = sys.modules.get("vllm_omni")
    parent = sys.modules.get(NATIVE_MODULE.rsplit(".", 1)[0])
    return (
        f"host={socket.gethostname()} pid={os.getpid()}\n"
        f"vllm_omni_origin={_origin(package)}\n"
        f"native_origin={_origin(sys.modules.get(NATIVE_MODULE))}\n"
        f"model_package_paths={list(getattr(parent, '__path__', ()))!r}\n"
        f"expected_native_file={target} exists={target.is_file()}\n"
        f"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}"
    )


def _check_origin(module, expected_root):
    if not expected_root:
        return
    package_root = Path(expected_root).resolve() / "vllm_omni"
    origin = _origin(module)
    if origin is None or not origin.is_relative_to(package_root.resolve()):
        raise RuntimeError(
            "Huanyu vLLM-Omni source mismatch. Start a fresh process with the "
            "selected checkout first in PYTHONPATH; do not mix imported modules "
            f"from different checkouts.\n{_details(expected_root)}"
        )


def load_huanyu_native(expected_root=None):
    """Import the real implementation, with actionable per-worker diagnostics."""
    expected_root = expected_root or os.environ.get("HUANYU_VLLM_OMNI_ROOT")
    try:
        with _without_cli_directory():
            package = importlib.import_module("vllm_omni")
            _check_origin(package, expected_root)
            native = importlib.import_module(NATIVE_MODULE)
    except ModuleNotFoundError as exc:
        missing_native = bool(exc.name) and (
            exc.name == NATIVE_MODULE or NATIVE_MODULE.startswith(exc.name + ".")
        )
        reason = (
            "The selected vLLM-Omni checkout does not expose the Huanyu module. "
            "Verify the complete Huanyu-enabled checkout/mount on this node "
            "and the Ray job's PYTHONPATH."
            if missing_native else
            "Importing native Huanyu failed because a dependency is missing; "
            "inspect the chained exception before changing the model code."
        )
        raise RuntimeError(
            f"{reason}\nmissing_module={exc.name!r}\n{_details(expected_root)}"
        ) from exc
    _check_origin(native, expected_root)
    missing = [name for name in REQUIRED_SYMBOLS if not callable(getattr(native, name, None))]
    if missing:
        raise RuntimeError(
            f"Huanyu runtime API mismatch: missing callables {missing!r}\n"
            f"{_details(expected_root)}"
        )
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check imports without model weights")
    parser.add_argument("--expected-root", default=os.environ.get("HUANYU_VLLM_OMNI_ROOT", DEFAULT_ROOT))
    args = parser.parse_args()
    native = load_huanyu_native(args.expected_root)
    print(f"[huanyu-runtime] OK host={socket.gethostname()} pid={os.getpid()}", flush=True)
    print(f"[huanyu-runtime] vllm_omni={_origin(sys.modules.get('vllm_omni'))}", flush=True)
    print(f"[huanyu-runtime] native_source={_origin(native)}", flush=True)


if __name__ == "__main__":
    main()
