"""Exercise directory synchronization and cluster decisions without a Ray cluster."""

import errno
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("huanyu_job_bootstrap", ROOT / "job_bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def make_checkout(path, content):
    path.mkdir(parents=True, exist_ok=True)
    for name in bootstrap.REQUIRED_FILES:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    return path


def tree_contents(path):
    """Include non-package files and empty directories when checking recovery."""
    return {
        entry.relative_to(path).as_posix(): None if entry.is_dir() else entry.read_bytes()
        for entry in path.rglob("*")
    }


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="job_bootstrap_", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = make_checkout(self.root / "source", "new")
        self.target = make_checkout(self.root / "target", "old")

    def test_replacement_preserves_old_checkout_and_copies_hidden_files(self):
        (self.source / ".new-config").write_text("hidden")
        (self.target / "user-note").write_text("do not lose")
        stale = self.target / "vllm_omni" / "obsolete.py"
        stale.write_text("old code")
        bootstrap.sync_checkout(self.source, self.target)
        self.assertEqual(bootstrap.code_digest(self.source), bootstrap.code_digest(self.target))
        self.assertFalse(stale.exists())
        self.assertEqual((self.target / ".new-config").read_text(), "hidden")
        backups = list(self.root.glob(".target.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "user-note").read_text(), "do not lose")
        self.assertTrue((backups[0] / "vllm_omni/obsolete.py").exists())
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_running_ray_different_code_fails_without_changes(self):
        before = bootstrap.code_digest(self.target)
        with self.assertRaisesRegex(RuntimeError, "Local Ray is running"):
            bootstrap.sync_checkout(self.source, self.target, ray_running=True)
        self.assertEqual(bootstrap.code_digest(self.target), before)
        self.assertFalse(list(self.root.glob(".target.*")))

    def test_running_ray_matching_code_allows_pycache_and_leaves_tree_untouched(self):
        make_checkout(self.target, "new")
        cache = self.target / "vllm_omni/__pycache__"
        cache.mkdir()
        (cache / "opaque.pyc").write_bytes(b"old bytecode")
        bootstrap.sync_checkout(self.source, self.target, ray_running=True)
        self.assertTrue(cache.exists())
        self.assertFalse(list(self.root.glob(".target.*")))

    def test_missing_source_does_not_remove_valid_target(self):
        (self.source / bootstrap.REQUIRED_FILES[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            bootstrap.sync_checkout(self.source, self.target)
        self.assertEqual((self.target / bootstrap.REQUIRED_FILES[0]).read_text(), "old")

    def test_same_directory_skips_copy_and_parent_child_is_rejected(self):
        bootstrap.sync_checkout(self.source, self.source)
        with self.assertRaisesRegex(ValueError, "must not contain"):
            bootstrap.sync_checkout(self.source, self.source / "child")
        with self.assertRaisesRegex(ValueError, "must not contain"):
            bootstrap.sync_checkout(self.source, self.root)

    def test_mount_point_rejected_without_deleting_anything(self):
        with patch.object(Path, "is_mount", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "mount point"):
                bootstrap.sync_checkout(self.source, self.target)
        self.assertEqual((self.target / bootstrap.REQUIRED_FILES[0]).read_text(), "old")

    def test_copy_failure_preserves_original_target(self):
        with patch.object(bootstrap.shutil, "copytree", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                bootstrap.sync_checkout(self.source, self.target)
        self.assertEqual((self.target / bootstrap.REQUIRED_FILES[0]).read_text(), "old")
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_switch_failure_restores_old_tree(self):
        original = Path.rename

        def fail_switch(path, destination):
            if ".stage-" in path.name:
                raise OSError("rename failed")
            return original(path, destination)

        with patch.object(Path, "rename", fail_switch):
            with self.assertRaisesRegex(OSError, "rename failed"):
                bootstrap.sync_checkout(self.source, self.target)
        self.assertEqual((self.target / bootstrap.REQUIRED_FILES[0]).read_text(), "old")
        self.assertFalse(list(self.root.glob(".target.*")))

    def fail_target_rename(self, error_number=errno.EXDEV):
        original = Path.rename

        def rename(path, destination):
            if path == self.target:
                raise OSError(error_number, "cannot rename target root")
            return original(path, destination)

        return patch.object(Path, "rename", rename)

    def test_cross_device_copy_preserves_root_and_complete_old_backup(self):
        (self.source / ".new-config").write_text("new hidden config")
        (self.target / ".old-config").write_text("old hidden config")
        (self.target / "empty-user-directory").mkdir()
        (self.target / "user-note").write_text("keep this outside the package")
        stale = self.target / "vllm_omni/obsolete.py"
        stale.write_text("old code")
        before = tree_contents(self.target)
        root_identity = (self.target.stat().st_dev, self.target.stat().st_ino)

        with self.fail_target_rename():
            bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(tree_contents(self.target), tree_contents(self.source))
        self.assertEqual((self.target.stat().st_dev, self.target.stat().st_ino), root_identity)
        self.assertFalse(stale.exists())
        backups = list(self.root.glob(".target.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(tree_contents(backups[0]), before)
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_backup_copy_failure_leaves_target_untouched(self):
        (self.target / ".old-config").write_text("keep this")
        before = tree_contents(self.target)
        original = bootstrap.shutil.copytree

        def fail_backup(source, destination, *args, **kwargs):
            if Path(source) == self.target:
                Path(destination).mkdir(parents=True, exist_ok=True)
                (Path(destination) / "partial-backup").write_text("not complete")
                raise OSError(errno.ENOSPC, "backup disk full")
            return original(source, destination, *args, **kwargs)

        with self.fail_target_rename(), patch.object(bootstrap.shutil, "copytree", fail_backup):
            with self.assertRaisesRegex(OSError, "backup disk full"):
                bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(tree_contents(self.target), before)
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_backup_verification_covers_non_package_files(self):
        (self.target / ".old-config").write_text("original settings")
        before = tree_contents(self.target)
        original = bootstrap.shutil.copytree

        def corrupt_backup(source, destination, *args, **kwargs):
            result = original(source, destination, *args, **kwargs)
            if Path(source) == self.target:
                (Path(destination) / ".old-config").write_text("corrupted settings")
            return result

        with self.fail_target_rename(), patch.object(bootstrap.shutil, "copytree", corrupt_backup):
            with self.assertRaises(RuntimeError):
                bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(tree_contents(self.target), before)
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_install_failure_restores_old_contents_and_keeps_backup(self):
        (self.target / ".old-config").write_text("recover this")
        (self.target / "vllm_omni/obsolete.py").write_text("recover old code")
        (self.target / "empty-user-directory").mkdir()
        before = tree_contents(self.target)
        root_identity = (self.target.stat().st_dev, self.target.stat().st_ino)
        original = bootstrap.shutil.copytree
        failed_install = False

        def fail_install(source, destination, *args, **kwargs):
            nonlocal failed_install
            if (not failed_install and Path(source).name.startswith(".target.stage-")
                    and Path(destination) == self.target):
                failed_install = True
                (self.target / "partially-installed-file").write_text("must be removed")
                raise OSError(errno.ENOSPC, "install disk full")
            return original(source, destination, *args, **kwargs)

        with self.fail_target_rename(), patch.object(bootstrap.shutil, "copytree", fail_install):
            with self.assertRaisesRegex(RuntimeError, "original contents restored") as error:
                bootstrap.sync_checkout(self.source, self.target)

        self.assertTrue(failed_install)
        self.assertIsInstance(error.exception.__cause__, OSError)
        self.assertEqual(error.exception.__cause__.errno, errno.ENOSPC)
        self.assertEqual(tree_contents(self.target), before)
        self.assertEqual((self.target.stat().st_dev, self.target.stat().st_ino), root_identity)
        backups = list(self.root.glob(".target.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(tree_contents(backups[0]), before)
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_install_verification_failure_restores_old_contents(self):
        (self.source / ".new-config").write_text("intended settings")
        before = tree_contents(self.target)
        original = bootstrap.shutil.copytree

        def corrupt_install(source, destination, *args, **kwargs):
            result = original(source, destination, *args, **kwargs)
            if (Path(source).name.startswith(".target.stage-")
                    and Path(destination) == self.target):
                (self.target / ".new-config").write_text("corrupted settings")
            return result

        with self.fail_target_rename(), patch.object(bootstrap.shutil, "copytree", corrupt_install):
            with self.assertRaisesRegex(RuntimeError, "original contents restored"):
                bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(tree_contents(self.target), before)
        backups = list(self.root.glob(".target.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(tree_contents(backups[0]), before)
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_restore_failure_retains_recoverable_backup(self):
        (self.target / ".old-config").write_text("recover this manually")
        before = tree_contents(self.target)
        original = bootstrap.shutil.copytree

        def fail_target_copy(source, destination, *args, **kwargs):
            if Path(destination) == self.target:
                raise OSError(errno.ENOSPC, "target disk full")
            return original(source, destination, *args, **kwargs)

        with self.fail_target_rename(), patch.object(bootstrap.shutil, "copytree", fail_target_copy):
            with self.assertRaisesRegex(RuntimeError, "restoration also failed") as error:
                bootstrap.sync_checkout(self.source, self.target)

        backups = list(self.root.glob(".target.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(tree_contents(backups[0]), before)
        self.assertIn(str(backups[0]), str(error.exception))
        self.assertFalse(list(self.root.glob(".target.stage-*")))

    def test_cross_device_nested_mount_is_rejected_before_backup_or_deletion(self):
        nested = self.target / "mounted-data"
        nested.mkdir()
        (nested / "must-survive").write_text("external data")
        before = tree_contents(self.target)
        with self.fail_target_rename(), patch.object(Path, "is_mount", lambda path: path == nested):
            with self.assertRaisesRegex(RuntimeError, "nested mount"):
                bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(tree_contents(self.target), before)
        self.assertFalse(list(self.root.glob(".target.*")))

    def test_permission_error_does_not_enter_cross_device_fallback(self):
        before = tree_contents(self.target)
        original = bootstrap.shutil.copytree

        with self.fail_target_rename(errno.EACCES), patch.object(
                bootstrap.shutil, "copytree", wraps=original) as copy:
            with self.assertRaises(OSError) as error:
                bootstrap.sync_checkout(self.source, self.target)

        self.assertEqual(error.exception.errno, errno.EACCES)
        self.assertFalse(any(Path(call.args[0]) == self.target for call in copy.call_args_list))
        self.assertEqual(tree_contents(self.target), before)
        self.assertFalse(list(self.root.glob(".target.*")))


def node(ip, npu=16, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"NPU": npu}}


class ReadinessTests(unittest.TestCase):
    def test_other_cluster_nodes_cannot_replace_missing_expected_worker(self):
        nodes = [node("10.0.0.1"), node("10.0.0.3", 32)]
        self.assertFalse(bootstrap.cluster_ready(nodes, ["10.0.0.1", "10.0.0.2"], "10.0.0.1", "all"))

    def test_both_expected_nodes_need_sixteen_live_npus(self):
        for count, alive, ready in ((16, True, True), (8, True, False), (16, False, False)):
            with self.subTest(count=count, alive=alive):
                nodes = [node("10.0.0.1"), node("10.0.0.2", count, alive)]
                self.assertEqual(bootstrap.cluster_ready(
                    nodes, ["10.0.0.1", "10.0.0.2"], "10.0.0.1", "all"), ready)

    def test_reused_worker_must_belong_to_selected_cluster(self):
        nodes = [node("10.0.0.1"), node("10.0.0.3")]
        self.assertFalse(bootstrap.cluster_ready(nodes, ["10.0.0.1", "10.0.0.2"], "10.0.0.2", "local"))
        self.assertTrue(bootstrap.cluster_ready(nodes + [node("10.0.0.2")],
                                               ["10.0.0.1", "10.0.0.2"], "10.0.0.2", "local"))


if __name__ == "__main__":
    unittest.main()
