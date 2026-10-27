"""Transfer successful setup's environment changes to the invoking Bash shell.

The baseline contains hashes only. Activation files contain managed runtime
settings plus other changes, have private permissions, and are published after
all checks pass. Unchanged unrelated credentials are not copied into the file.
No third-party imports, Ray connection, or device initialization occurs here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tempfile


_SHELL_INTERNALS = {"PWD", "OLDPWD", "SHLVL", "_", "SHELLOPTS", "BASHOPTS"}
_MANAGED_PREFIXES = (
    "HUANYU_", "MINICPMO_", "Q36_", "QWEN36_", "HCCL_", "ASCEND_", "VLLM_",
    "VERL_", "TORCH_", "TORCHINDUCTOR_", "PYTORCH_", "ACL_", "ACLNN_", "ATB_",
    "RAY_", "RM_", "ROLLOUT_", "GENRM_", "VC_", "WANDB_",
)
_MANAGED_NAMES = {
    "PROJECT_ROOT", "MODEL_PATH", "TRAIN_FILE", "VAL_FILE", "DEFAULT_LOCAL_DIR",
    "TENSORBOARD_DIR", "ROLLOUT_DATA_DIR", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
    "LD_PRELOAD", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "PYTHONUNBUFFERED",
    "HYDRA_FULL_ERROR", "SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "MULTI_STREAM_MEMORY_REUSE", "TASK_QUEUE_ENABLE", "CPU_AFFINITY_CONF",
    "NUM_NPUS", "NPUS_PER_NODE", "NNODES", "REWARD_POOL_NPUS",
    "NUM_GPUS_ACTOR_ROLLOUT_REWARD", "PROJECT_NAME", "EXPERIMENT_NAME",
    "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "PPO_MICRO_BATCH_SIZE_PER_GPU",
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "MAX_TOKEN_LEN_PER_GPU",
    "TOTAL_TRAINING_STEPS", "ENABLE_THINKING", "DEBUG_MODE", "DEBUG_SAMPLES",
}


def _environment():
    return {
        key: value for key, value in os.environ.items()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        and key not in _SHELL_INTERNALS
        and not key.startswith("HUANYU_PREPARE_")
        and key != "HUANYU_PREPARED_ENV_FILE"
    }


def _digest(value):
    return hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()


def snapshot(path):
    # Do not persist inherited tokens/passwords as plaintext even temporarily.
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        json.dump({key: _digest(value) for key, value in _environment().items()}, stream, sort_keys=True)


def write_activation(snapshot_path, output, stack_limit):
    if not re.fullmatch(r"unlimited|[1-9][0-9]*", stack_limit):
        raise ValueError("Invalid soft stack limit")
    before = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if not isinstance(before, dict) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for key, value in before.items()
    ):
        raise ValueError("Invalid environment snapshot")
    current = _environment()
    project = current["PROJECT_ROOT"]
    lines = [
        "#!/usr/bin/env bash",
        "# Generated on this node. Source before starting Ray; no Ray or training is launched.",
        f"cd -- {shlex.quote(project)} || return $?",
        f"ulimit -S -s {stack_limit} || return $?",
    ]
    for key in sorted(before.keys() - current.keys()):
        lines.append(f"unset {key} || return $?")
    for key, value in sorted(current.items()):
        # Prefix assignments to `bash prepare...` do not survive in its caller.
        # Reapply managed settings even if they already existed in the baseline.
        if key in _MANAGED_NAMES or key.startswith(_MANAGED_PREFIXES) or before.get(key) != _digest(value):
            lines.append(f"export {key}={shlex.quote(value)} || return $?")
    lines += [
        f"export HUANYU_PREPARED_ENV_FILE={shlex.quote(str(output))} || return $?",
        "echo '[huanyu-env] environment activated; Ray and training must be started separately'",
    ]
    output = Path(output)
    descriptor, temporary = tempfile.mkstemp(prefix=".activate-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", errors="surrogateescape", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("path")
    write = commands.add_parser("write")
    write.add_argument("--snapshot", required=True)
    write.add_argument("--output", required=True)
    write.add_argument("--stack-limit", required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.path)
    else:
        write_activation(args.snapshot, args.output, args.stack_limit)


if __name__ == "__main__":
    main()
