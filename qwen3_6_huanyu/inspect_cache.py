"""Read-only cache/checkpoint/frontend contract comparison, without encoder weights.

Run directly after preparing the same Python environment as training. This does
not check individual .pt entries or dataset coverage; precompute --verify-only
does that after the contract is accepted. Exit 0 when all non-source fields match,
2 for a mismatch. Neither manifests nor cached tensors are changed.
"""

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


if not __package__:
    _package = "_huanyu_cache_inspect_cli"
    _module = ModuleType(_package)
    _module.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package] = _module
    __package__ = _package


def inspect_cache(model_path, cache_dir, max_differences=40):
    # Read before importing native model code, and avoid the package-wide ext
    # registration hooks just as the direct precompute CLI does.
    manifest = Path(cache_dir).resolve() / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    from .cache import _manifest_matches, _source_compatible, cache_contract_report, make_cache_contract
    from .processor import build_huanyu_processor

    processor = build_huanyu_processor(
        model_path, SimpleNamespace(trust_remote_code=True), install_bridge=False, use_cache=False,
    )
    current = make_cache_contract(model_path, processor)
    print(cache_contract_report(payload, current, manifest, max_differences=max_differences), flush=True)
    return 0 if _manifest_matches(payload, current) or _source_compatible(payload, current) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", required=True)
    parser.add_argument("--max-differences", type=int, choices=range(1, 201), default=40, metavar="1..200")
    args = parser.parse_args()
    return inspect_cache(args.model_path, args.cache_dir, args.max_differences)


if __name__ == "__main__":
    raise SystemExit(main())
