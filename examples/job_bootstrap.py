"""File preparation and bounded Ray readiness checks for the Huanyu launchers.

Run this file directly: preparation must not import the ext plugin or load NPU
libraries before the selected vLLM-Omni checkout and environment are ready.
"""

import argparse
from collections.abc import Mapping
import errno
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import tempfile
import time


REQUIRED_FILES = (
    "vllm_omni/__init__.py",
    "vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py",
    "vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py",
)


_SUPERPOD = "HCCL_LOGIC_SUPERPOD_ID"
_BASHRC_TOKEN = re.compile(
    r"(?P<space>[ \t\r]+)|(?P<comment>\#[^\n]*)|(?P<separator>\n|&&|\|\||;)"
    r"|(?P<operator><<-?|[&|()<>])"
    r'''|(?P<word>(?:[^\s;&|()<>'"\\]+|'[^']*'|"(?:\\[\s\S]|[^"\\])*"|\\[\s\S])+)'''
)


def _bashrc_commands(text):
    """Locate simple commands without interpreting expansions or heredoc data."""
    position, command, heredocs = 0, [], []
    delimiter = None
    while position < len(text):
        token = _BASHRC_TOKEN.match(text, position)
        if token is None:
            raise ValueError("Cannot safely scan .bashrc quoting; edit HCCL_LOGIC_SUPERPOD_ID manually")
        raw, kind = token.group(), token.lastgroup
        position = token.end()
        if kind == "separator":
            yield command
            command = []
            if raw == "\n":
                for marker, strip_tabs in heredocs:
                    while position < len(text):
                        end = text.find("\n", position)
                        end = len(text) if end == -1 else end + 1
                        line = text[position:end].rstrip("\r\n")
                        position = end
                        if (line.lstrip("\t") if strip_tabs else line) == marker:
                            break
                    else:
                        raise ValueError("Unterminated .bashrc heredoc; edit HCCL_LOGIC_SUPERPOD_ID manually")
                heredocs = []
        elif kind not in ("space", "comment"):
            command.append((raw, token.start(), token.end()))
            if delimiter is not None:
                if kind != "word":
                    raise ValueError("Unsupported .bashrc heredoc delimiter; edit HCCL_LOGIC_SUPERPOD_ID manually")
                heredocs.append((shlex.split(raw)[0], delimiter == "<<-"))
                delimiter = None
            elif raw in ("<<", "<<-"):
                delimiter = raw
    yield command


def _without_superpod(text):
    edits = []
    for words in _bashrc_commands(text):
        # Keep conditional bodies syntactically valid by replacing an otherwise
        # empty setting command with ':', including `then export ...; fi`.
        while words and words[0][0] in ("if", "then", "elif", "else", "do", "while", "until", "!", "{"):
            words = words[1:]
        if not words:
            continue
        exporting = words[0][0] == "export"
        selected = []
        for word in words:
            plain = word[0].replace("\\\r\n", "").replace("\\\n", "")
            if exporting:
                plain = (shlex.split(plain) or [""])[0]
            if plain.startswith(_SUPERPOD + "=") or (exporting and plain == _SUPERPOD):
                selected.append(word)
        if not selected:
            continue
        raw_command = text[words[0][1]:words[-1][2]]
        values = words[1:] if exporting else words
        if exporting and values and values[0][0] == "--":
            values = values[1:]
        valid = all(re.match(r"^[A-Za-z_][A-Za-z_0-9]*(?:=|$)", shlex.split(word[0])[0] if exporting else word[0])
                    for word in values)
        if not exporting:
            valid = valid and all("=" in word[0] for word in values)
        if not valid or "\n" in raw_command or "$" in raw_command or "`" in raw_command:
            line = text.count("\n", 0, words[0][1]) + 1
            raise ValueError(
                f"Cannot safely remove {_SUPERPOD} at .bashrc line {line}; "
                "use a standalone literal assignment/export and rerun (file unchanged)"
            )
        if len(selected) == len(values):
            edits.append((words[0][1], words[-1][2], ":"))
        else:
            edits.extend((start, end, "") for _, start, end in selected)
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def clear_superpod_bashrc(path):
    """Remove literal SuperPod assignments, returning the backup path or None.

    Supports simple assignments and exports, including multiple variables and
    commands separated by ';', '&&', or '||'. Comments, quoted strings, heredoc
    bodies, bytes, and line endings are preserved. Unsupported target commands
    (e.g. declare, redirects, expansions, or multiline assignments) fail before
    any write. This is deliberately not a Bash interpreter: dynamically generated
    settings (eval, sourced files, or embedded substitutions) need manual removal.
    The file is never sourced. A unique sibling backup retains the original bytes
    and mode; replacement is atomic and follows an existing .bashrc symlink.
    """
    requested = Path(path).expanduser()
    if not requested.exists():
        return None
    target = requested.resolve(strict=True)
    original = target.read_bytes()
    text = original.decode("utf-8", errors="surrogateescape")
    if _SUPERPOD not in text:
        return None
    updated = _without_superpod(text).encode("utf-8", errors="surrogateescape")
    if updated == original:
        return None
    with tempfile.NamedTemporaryFile(
        prefix=f"{requested.name}.huanyu-superpod-backup-", dir=target.parent, delete=False
    ) as handle:
        backup = Path(handle.name)
        handle.write(original)
    shutil.copymode(target, backup)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.stage-", dir=target.parent, delete=False) as handle:
            staged = Path(handle.name)
            handle.write(updated)
        shutil.copymode(target, staged)
        if target.read_bytes() != original:
            raise RuntimeError(f"{requested} changed during cleanup; not replaced. Backup: {backup}")
        os.replace(staged, target)
    finally:
        if staged is not None and staged.exists():
            staged.unlink()
    print(f"[huanyu-job] removed {_SUPERPOD} from {requested}; backup={backup}", flush=True)
    return backup


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


def resolve_hosts(hosts, expected_nodes=2):
    if not isinstance(expected_nodes, int) or expected_nodes < 1:
        raise ValueError("expected_nodes must be a positive integer")
    names = [name.strip() for name in hosts.split(",")]
    if len(names) != expected_nodes or not all(names):
        raise ValueError(f"VC_WORKER_HOSTS must contain exactly {expected_nodes} comma-separated hosts")
    ips = [socket.gethostbyname(name) for name in names]
    if len(set(ips)) != expected_nodes:
        raise ValueError("Job hosts must resolve to distinct IPv4 addresses")
    return ips


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


def wait_cluster_gcs(address, expected_ips, local_ip, mode, timeout, interval):
    """Read live registrations directly from GCS, including on headless clusters."""
    from ray._raylet import GcsClient
    from ray.exceptions import RayError

    deadline = time.monotonic() + timeout
    required = {ip: 16 for ip in (expected_ips if mode == "all" else [local_ip])}
    print(f"[huanyu-env] checking Ray via GCS address={address} mode={mode} expected_NPU={required}", flush=True)
    client = None
    while True:
        try:
            if client is None:
                client = GcsClient(address=address)
            nodes = client.get_all_node_info(timeout=min(timeout, 15))
        except (RayError, OSError) as exc:
            client = None
            print(f"[huanyu-env] waiting for GCS at {address}: {exc}", flush=True)
        else:
            live = [(node.node_manager_address, node.resources_total.get("NPU", 0))
                    for node in nodes.values() if node.state == node.ALIVE]
            observed = live if mode == "all" else [entry for entry in live if entry[0] == local_ip]
            if sorted(observed) == sorted(required.items()):
                print(f"[huanyu-env] RAY_READY mode={mode} address={address} "
                      f"nodes={len(required)} total_NPU={sum(required.values())} live_NPU={live}", flush=True)
                return
            print(f"[huanyu-env] waiting for nodes: live_NPU={live} expected_NPU={required}", flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Ray at {address} did not report expected_NPU={required}")
        time.sleep(min(interval, remaining))


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


# AQA preflight is part of this kept bootstrap entry, not a separate launcher.
SOURCE = "aqa_real"
REQUIRED_COLUMNS = ("extra_info", "prompt", "images", "videos", "audios", "reward_model")


def _audio_reference_exists(item, checked_paths):
    if isinstance(item, Mapping):
        value = next((item[key] for key in ("audio", "audio_url", "bytes") if item.get(key) is not None), None)
    else:
        value = item
    if isinstance(value, (bytes, bytearray, memoryview)):
        if not value:
            raise ValueError("empty embedded audio bytes")
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio must contain a non-empty local path/archive reference or bytes")
    archive, separator, member = value.partition("::")
    if separator and not member.strip():
        raise ValueError(f"empty zarr archive member: {value!r}")
    if archive not in checked_paths:
        if not Path(archive).is_file():
            raise ValueError(f"audio file/archive does not exist on this node: {archive}")
        checked_paths.add(archive)


def validate_aqa_row(row, checked_paths):
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or extra.get("raw_source") != SOURCE:
        raise ValueError("expected extra_info.raw_source='aqa_real'; mixed sources are not allowed")
    if row.get("images") or row.get("videos"):
        raise ValueError("AQA audio training forbids non-empty images/videos")
    if row.get("ability") not in (None, "audio"):
        raise ValueError(f"expected ability='audio', got {row['ability']!r}")
    audios = row.get("audios")
    if not isinstance(audios, list) or not audios:
        raise ValueError("audios must be a non-empty list")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty message list")
    if any(not isinstance(message, Mapping) or not isinstance(message.get("content"), str)
           or not message["content"].strip() for message in prompt):
        raise ValueError("every prompt message must contain non-empty string content")
    if not any(message.get("role") == "user" for message in prompt):
        raise ValueError("prompt must contain a user message")
    reward = row.get("reward_model")
    answer = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("reward_model.ground_truth must be a non-empty string")
    for item in audios:
        _audio_reference_exists(item, checked_paths)


def check_split(path, minimum_rows=1, batch_size=1024):
    """Read-only AQA schema/media checks, without importing training modules."""
    import pyarrow.parquet as pq

    path = Path(path).resolve(strict=True)
    if minimum_rows < 1 or batch_size < 1:
        raise ValueError("minimum_rows and batch_size must be positive")
    checked_paths = set()
    scanned = 0
    with pq.ParquetFile(path) as parquet:
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        total = parquet.metadata.num_rows
        if total < minimum_rows:
            raise ValueError(f"{path}: rows={total} is smaller than required minimum={minimum_rows}")
        columns = list(REQUIRED_COLUMNS)
        if "ability" in parquet.schema_arrow.names:
            columns.append("ability")
        print(f"[huanyu-aqa-check] scanning={path} rows={total}", flush=True)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                try:
                    validate_aqa_row(row, checked_paths)
                except ValueError as error:
                    raise ValueError(f"{path} row={scanned} (zero-based): {error}") from error
                scanned += 1
    print(f"[huanyu-aqa-check] OK rows={scanned} source={SOURCE} audio_only=true "
          f"media_files_or_archives={len(checked_paths)} path={path}", flush=True)
    return scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    aqa = sub.add_parser("check-aqa", help="validate AQA Parquet/media references before training")
    aqa.add_argument("--train", required=True)
    aqa.add_argument("--validation", required=True)
    aqa.add_argument("--train-batch-size", type=int, default=4)
    clear = sub.add_parser("clear-superpod-bashrc", help="remove persistent HCCL_LOGIC_SUPERPOD_ID assignments")
    clear.add_argument("path")
    sync = sub.add_parser("sync")
    sync.add_argument("source")
    sync.add_argument("target")
    sync.add_argument("--ray-running", action="store_true")
    hosts = sub.add_parser("resolve")
    hosts.add_argument("hosts")
    hosts.add_argument("--expected-nodes", type=int, default=2)
    sub.add_parser("check-devices")
    for command in ("wait-head", "wait-cluster"):
        wait = sub.add_parser(command)
        wait.add_argument("--address", required=True)
        wait.add_argument("--timeout", type=int, required=True)
        wait.add_argument("--interval", type=int, default=5)
        if command == "wait-cluster":
            wait.add_argument("--gcs", "--state-api", action="store_true", help="query node registrations directly from GCS")
            wait.add_argument("--expected-ips", required=True)
            wait.add_argument("--local-ip", required=True)
            wait.add_argument("--mode", choices=("all", "local"), required=True)
    args = parser.parse_args()
    if args.command == "check-aqa":
        check_split(args.train, minimum_rows=args.train_batch_size)
        check_split(args.validation)
    elif args.command == "clear-superpod-bashrc":
        clear_superpod_bashrc(args.path)
    elif args.command == "sync":
        sync_checkout(args.source, args.target, args.ray_running)
    elif args.command == "resolve":
        print(",".join(resolve_hosts(args.hosts, args.expected_nodes)))
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
        check = wait_cluster_gcs if args.gcs else wait_cluster
        check(args.address, args.expected_ips.split(","), args.local_ip,
              args.mode, args.timeout, args.interval)


if __name__ == "__main__":
    main()
