"""File preparation and bounded Ray readiness checks for the two-node launcher.

Run this file directly: preparation must not import the ext plugin or load NPU
libraries before the selected vLLM-Omni checkout and environment are ready.
"""

import argparse
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time


REQUIRED_FILES = (
    "vllm_omni/__init__.py",
    "vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py",
    "vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py",
)


def check_checkout(root):
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Huanyu-enabled checkout is incomplete: {root / name}")


def code_digest(root):
    """Compare executable package content, including added/deleted source files."""
    digest = hashlib.sha256()
    package = root / "vllm_omni"
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_file():
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root):
    """Verify a full backup, including hidden files, empty dirs and links."""
    digest = hashlib.sha256()

    def visit(directory):
        for path in sorted(directory.iterdir()):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            if path.is_symlink():
                digest.update(b"link\0" + os.fsencode(os.readlink(path)) + b"\0")
            elif getattr(path, "is_junction", lambda: False)():
                raise RuntimeError(f"Cannot synchronize directory junction: {path}")
            elif path.is_dir():
                digest.update(b"dir\0")
                visit(path)
            elif path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
            else:
                raise RuntimeError(f"Cannot synchronize special file: {path}")

    visit(root)
    return digest.hexdigest()


def _check_no_nested_mounts(root):
    # is_mount alone misses same-device bind mounts on Linux. Never recurse
    # into a separate mounted data directory while replacing checkout contents.
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.is_file():
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if mount != root and mount.is_relative_to(root):
                raise RuntimeError(f"Cannot replace checkout containing nested mount: {mount}")

    def visit(directory):
        for path in directory.iterdir():
            if path.is_symlink():
                continue
            if path.is_mount() or getattr(path, "is_junction", lambda: False)():
                raise RuntimeError(f"Cannot replace checkout containing nested mount/junction: {path}")
            if path.is_dir():
                visit(path)

    visit(root)


def _clear_contents(root):
    # root is the already-resolved, verified checkout. Keep it in place and
    # unlink symlinks themselves; never resolve/delete their external targets.
    _check_no_nested_mounts(root)
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)


def _copy_replace_checkout(staging, target, backup):
    """Fallback for overlayfs EXDEV; keep a verified backup for recovery."""
    _check_no_nested_mounts(target)
    before = _tree_digest(target)
    expected = _tree_digest(staging)
    shutil.copytree(target, backup, symlinks=True)
    if _tree_digest(backup) != before or _tree_digest(target) != before:
        raise RuntimeError(f"Backup verification failed; target not changed. Inspect {backup}")
    print(f"[huanyu-job] directory rename returned EXDEV; verified backup={backup}; "
          "replacing checkout contents while keeping its root directory", flush=True)
    try:
        _clear_contents(target)
        shutil.copytree(staging, target, dirs_exist_ok=True, symlinks=True)
        check_checkout(target)
        if _tree_digest(target) != expected:
            raise RuntimeError("Copied checkout does not match the staged source")
    except BaseException as failure:
        try:
            _clear_contents(target)
            shutil.copytree(backup, target, dirs_exist_ok=True, symlinks=True)
            if _tree_digest(target) != before:
                raise RuntimeError("Restored checkout does not match the backup")
        except BaseException as recovery_failure:
            raise RuntimeError(
                f"Checkout update failed ({failure!r}); restoration also failed "
                f"({recovery_failure!r}). Training was not started. Recover from {backup}"
            ) from recovery_failure
        raise RuntimeError(
            f"Checkout update failed ({failure!r}); original contents restored. Backup retained at {backup}"
        ) from failure


def sync_checkout(source, target, ray_running=False):
    """Stage a complete copy, retain the old tree, and never overwrite live code."""
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    check_checkout(source)
    if source == target:
        print(f"[huanyu-job] source and target already resolve to {target}; copy skipped", flush=True)
        return
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("vLLM-Omni source and target must not contain each other")
    if target.is_dir():
        if ray_running:
            check_checkout(target)
        if (target / "vllm_omni").is_dir() and code_digest(source) == code_digest(target):
            print("[huanyu-job] vLLM-Omni package already matches source; copy skipped", flush=True)
            return
    if ray_running:
        raise RuntimeError("Local Ray is running with a different checkout. No code was changed; "
                           "finish that job before updating vLLM-Omni.")
    if target.is_mount():
        raise RuntimeError(f"Cannot replace mount point {target}; synchronize its backing directory first")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    backup = staging.with_name(staging.name.replace(".stage-", ".backup-", 1))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        check_checkout(staging)
        if code_digest(staging) != code_digest(source):
            raise RuntimeError("Source changed during copy; target was not replaced")
        if target.exists():
            try:
                target.rename(backup)
            except OSError as exc:
                # Container overlayfs can return EXDEV for a lower-layer
                # directory even when its parent is on the same visible FS.
                if exc.errno != errno.EXDEV or not target.is_dir():
                    raise
                _copy_replace_checkout(staging, target, backup)
                print(f"[huanyu-job] vLLM-Omni copied to {target}; previous checkout={backup}", flush=True)
                return
        try:
            staging.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
        print(f"[huanyu-job] vLLM-Omni copied to {target}; "
              f"previous checkout={backup if backup.exists() else '<none>'}", flush=True)
    finally:
        # Only remove the unique staging directory created by this invocation.
        if staging.exists():
            shutil.rmtree(staging)


def cluster_ready(nodes, expected_ips, local_ip, mode, npus=16):
    live = {node["NodeManagerAddress"]: node for node in nodes if node.get("Alive")}
    required = expected_ips if mode == "all" else [local_ip]
    return all(ip in live and live[ip].get("Resources", {}).get("NPU", 0) >= npus
               for ip in required)


def wait_head(address, timeout, interval):
    host, port = address.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=min(3, timeout)):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Ray head {address}")
            print(f"[huanyu-job] waiting for head {address}", flush=True)
            time.sleep(interval)


def wait_cluster(address, expected_ips, local_ip, mode, timeout, interval):
    import ray

    deadline = time.monotonic() + timeout
    # The shell's `timeout` also bounds ray.init, not just the polling loop.
    ray.init(address=address, log_to_driver=False)
    try:
        while True:
            nodes = ray.nodes()
            live = {node["NodeManagerAddress"]: node.get("Resources", {}).get("NPU", 0)
                    for node in nodes if node.get("Alive")}
            print(f"[huanyu-job] Ray expected={expected_ips} live_NPU={live} mode={mode}", flush=True)
            if cluster_ready(nodes, expected_ips, local_ip, mode):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Expected job nodes with 16 NPU each did not join; training not started")
            time.sleep(interval)
    finally:
        ray.shutdown()  # Disconnect this probe only.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("source")
    sync.add_argument("target")
    sync.add_argument("--ray-running", action="store_true")
    hosts = sub.add_parser("resolve")
    hosts.add_argument("hosts")
    sub.add_parser("check-devices")
    for command in ("wait-head", "wait-cluster"):
        wait = sub.add_parser(command)
        wait.add_argument("--address", required=True)
        wait.add_argument("--timeout", type=int, required=True)
        wait.add_argument("--interval", type=int, default=5)
        if command == "wait-cluster":
            wait.add_argument("--expected-ips", required=True)
            wait.add_argument("--local-ip", required=True)
            wait.add_argument("--mode", choices=("all", "local"), required=True)
    args = parser.parse_args()
    if args.command == "sync":
        sync_checkout(args.source, args.target, args.ray_running)
    elif args.command == "resolve":
        names = [name.strip() for name in args.hosts.split(",")]
        if len(names) != 2 or not all(names):
            raise ValueError("VC_WORKER_HOSTS must contain exactly two comma-separated hosts")
        ips = [socket.gethostbyname(name) for name in names]
        if len(set(ips)) != 2:
            raise ValueError("The two job hosts must resolve to different IPv4 addresses")
        print(",".join(ips))
    elif args.command == "check-devices":
        import torch
        import torch_npu  # noqa: F401

        count = torch.npu.device_count()
        if count < 16:
            raise RuntimeError(f"This node exposes only {count} NPUs; 16 required before advertising Ray resources")
        print(f"[huanyu-job] locally visible NPUs={count}", flush=True)
    elif args.command == "wait-head":
        wait_head(args.address, args.timeout, args.interval)
    else:
        wait_cluster(args.address, args.expected_ips.split(","), args.local_ip,
                     args.mode, args.timeout, args.interval)


if __name__ == "__main__":
    main()
