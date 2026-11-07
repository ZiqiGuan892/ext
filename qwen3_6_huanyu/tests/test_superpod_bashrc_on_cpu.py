"""CPU-only coverage for persistent SuperPod cleanup before Ray startup."""

import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys

import pytest


BOOTSTRAP = (
    Path(__file__).resolve().parents[1]
    / "examples/job_bootstrap.py"
)
clear_superpod_bashrc = runpy.run_path(str(BOOTSTRAP))["clear_superpod_bashrc"]
VARIABLE = "HCCL_LOGIC_SUPERPOD_ID"


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (f"{VARIABLE}=0\n", ":\n"),
        (f"  export {VARIABLE}='rack 1' # platform setting\n", "  : # platform setting\n"),
        (f'export {VARIABLE}="rack; # literal"\n', ":\n"),
        (f'export "{VARIABLE}=rack 1"\n', ":\n"),
        (f"export '{VARIABLE}'=0\n", ":\n"),
        (f"export {VARIABLE}\n", ":\n"),
        (f"export -- {VARIABLE}=0\n", ":\n"),
        (f"export KEEP=1 {VARIABLE}=0 OTHER='two words'\n", "export KEEP=1  OTHER='two words'\n"),
        (f"KEEP=1 {VARIABLE}=0 OTHER=2\n", "KEEP=1  OTHER=2\n"),
        (f"echo before; export {VARIABLE}=0; echo after\n", "echo before; :; echo after\n"),
        (f"true && {VARIABLE}=0 || echo fallback\n", "true && : || echo fallback\n"),
        (f"if true; then export {VARIABLE}=0; fi\n", "if true; then :; fi\n"),
        (f"if true; then\n  {VARIABLE}=0\nfi\n", "if true; then\n  :\nfi\n"),
    ],
)
def test_removes_only_setting_commands(tmp_path, setting, expected):
    bashrc = tmp_path / ".bashrc"
    before = f"# unrelated configuration\n{setting}export KEEP_ME=yes\n"
    bashrc.write_bytes(before.encode())

    backup = clear_superpod_bashrc(bashrc)

    assert bashrc.read_bytes() == f"# unrelated configuration\n{expected}export KEEP_ME=yes\n".encode()
    assert backup.read_bytes() == before.encode()
    assert backup.name.startswith(".bashrc.huanyu-superpod-backup-")
    assert clear_superpod_bashrc(bashrc) is None
    assert list(tmp_path.glob(".bashrc.huanyu-superpod-backup-*")) == [backup]


def test_preserves_comments_literals_heredocs_and_unrelated_multiline_content(tmp_path):
    bashrc = tmp_path / ".bashrc"
    content = (
        f"# export {VARIABLE}=0\n"
        f"export {VARIABLE}_SUFFIX=keep\n"
        f"echo 'export {VARIABLE}=0'\n"
        f'PROMPT="first line\nexport {VARIABLE}=literal\nlast line"\n'
        "cat <<'FIRST' <<-SECOND\n"
        f"export {VARIABLE}=heredoc\nFIRST\n"
        f"\t{VARIABLE}=second\n\tSECOND\n"
        f"export {VARIABLE}=0\n"
        "alias ll='ls -al'\n"
    ).encode()
    bashrc.write_bytes(content)

    backup = clear_superpod_bashrc(bashrc)

    assert bashrc.read_bytes() == content.replace(f"export {VARIABLE}=0\nalias".encode(), b":\nalias")
    assert backup.read_bytes() == content


@pytest.mark.parametrize("content", [b"", b"export KEEP=1\n", f"# export {VARIABLE}=0\n".encode()])
def test_unchanged_file_creates_no_backup(tmp_path, content):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_bytes(content)
    before = bashrc.stat()

    assert clear_superpod_bashrc(bashrc) is None

    assert bashrc.read_bytes() == content
    assert bashrc.stat().st_mtime_ns == before.st_mtime_ns
    assert list(tmp_path.iterdir()) == [bashrc]


def test_missing_bashrc_is_noop(tmp_path):
    assert clear_superpod_bashrc(tmp_path / "missing" / ".bashrc") is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "unsupported",
    [
        f"declare -x {VARIABLE}=0\n",
        f"readonly {VARIABLE}=0\n",
        f"export {VARIABLE}=$(echo 0)\n",
        f"export {VARIABLE}=$SOME_OTHER_SETTING\n",
        f"export {VARIABLE}=0 > /dev/null\n",
        f"{VARIABLE}=0 run_something\n",
        f"export \\\n{VARIABLE}=0\n",
        f"export \\\n  {VARIABLE}=0\n",
        f"export {VARIABLE}='first\nsecond'\n",
    ],
)
def test_ambiguous_setting_fails_without_partial_changes_or_backup(tmp_path, unsupported):
    bashrc = tmp_path / ".bashrc"
    original = f"export {VARIABLE}=supported\n{unsupported}".encode()
    bashrc.write_bytes(original)

    with pytest.raises(ValueError, match="Cannot safely remove"):
        clear_superpod_bashrc(bashrc)

    assert bashrc.read_bytes() == original
    assert list(tmp_path.iterdir()) == [bashrc]


def test_preserves_mode_crlf_non_utf8_and_missing_final_newline(tmp_path):
    bashrc = tmp_path / ".bashrc"
    original = b"# legacy byte: \xff\r\nexport " + VARIABLE.encode() + b"=0\r\nexport KEEP=yes"
    bashrc.write_bytes(original)
    bashrc.chmod(0o640)
    mode = stat.S_IMODE(bashrc.stat().st_mode)

    backup = clear_superpod_bashrc(bashrc)

    assert bashrc.read_bytes() == b"# legacy byte: \xff\r\n:\r\nexport KEEP=yes"
    assert backup.read_bytes() == original
    assert stat.S_IMODE(bashrc.stat().st_mode) == mode
    assert stat.S_IMODE(backup.stat().st_mode) == mode


def test_existing_backup_is_never_overwritten(tmp_path):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_bytes(f"{VARIABLE}=first\n".encode())
    first = clear_superpod_bashrc(bashrc)
    bashrc.write_bytes(f"{VARIABLE}=second\n".encode())

    second = clear_superpod_bashrc(bashrc)

    assert first != second
    assert first.read_bytes() == f"{VARIABLE}=first\n".encode()
    assert second.read_bytes() == f"{VARIABLE}=second\n".encode()


def test_failed_replace_keeps_original_and_backup(tmp_path, monkeypatch):
    bashrc = tmp_path / ".bashrc"
    original = f"export {VARIABLE}=0\n".encode()
    bashrc.write_bytes(original)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        clear_superpod_bashrc(bashrc)

    assert bashrc.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2
    assert next(tmp_path.glob(".bashrc.huanyu-superpod-backup-*")).read_bytes() == original


def test_symlink_is_preserved(tmp_path):
    bashrc = tmp_path / ".bashrc"
    target = tmp_path / "shell-config"
    target.write_bytes(f"export {VARIABLE}=0\n".encode())
    try:
        bashrc.symlink_to(target)
    except OSError:
        pytest.skip("Host does not allow symlink creation")

    backup = clear_superpod_bashrc(bashrc)

    assert bashrc.is_symlink()
    assert target.read_bytes() == b":\n"
    assert backup.name.startswith(".bashrc.huanyu-superpod-backup-")


def test_cli_needs_only_stdlib_and_never_executes_bashrc(tmp_path):
    bashrc = tmp_path / ".bashrc"
    bashrc.write_bytes(f"exit 17\nexport {VARIABLE}=0\n".encode())

    result = subprocess.run(
        [sys.executable, "-S", str(BOOTSTRAP), "clear-superpod-bashrc", str(bashrc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert bashrc.read_bytes() == b"exit 17\n:\n"
    assert "backup=" in result.stdout
