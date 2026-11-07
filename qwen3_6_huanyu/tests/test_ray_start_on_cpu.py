"""Exercise Ray-only startup with command stubs; no Ray/NPU/model imports."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "huanyu_ray_start_fixture", Path(__file__).with_name("test_job_launcher_on_cpu.py")
)
fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture_module)


@unittest.skipUnless(fixture_module.find_bash(), "Bash is required for Ray startup")
class HuanyuRayStartTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.HuanyuJobLauncherTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.entry = self.fixture.examples / "start_huanyu_ray.sh"
        shutil.copyfile(ROOT / "examples/start_huanyu_ray.sh", self.fixture.entry)
        helper = self.fixture.examples / "job_bootstrap.py"
        helper.write_text(
            'import sys\n'
            'if sys.argv[1:] == ["wait-cluster", "--help"]:\n'
            '    print("wait-cluster --gcs")\n'
            '    raise SystemExit(0)\n' + helper.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        # The entry must work with just Ray, Bash and its bootstrap helper.
        for name in ("env_export_huanyu.sh", "check_native_runtime.py", "run_huanyu.sh"):
            (self.fixture.examples / name).unlink()
        for key in ("PROJECT_ROOT", "MODEL_PATH", "HUANYU_VLLM_SOURCE"):
            self.fixture.env.pop(key, None)

    def run_start(self, updates=None, arguments=()):
        result = self.fixture.run_launcher(updates, arguments)
        for kind in ("pip", "git", "env-source", "runtime", "train", "sleep"):
            self.assertFalse(self.fixture.events(kind), kind)
        self.assertFalse(any(e["args"][0] in ("check-devices", "sync")
                             for e in self.fixture.events("helper")))
        return result

    def clear_events(self):
        self.fixture.events_file.write_text("", encoding="utf-8")

    def test_head_starts_with_policy_and_no_extra_probes(self):
        result = self.run_start({
            "RAY_worker_maximum_startup_concurrency": "99",
            "RAY_prestart_worker_first_driver": "0", "RAY_enable_worker_prestart": "1",
            "RAY_ADDRESS": "stale-head:1234", "HUANYU_TEST_FAIL_WAIT": "wait-cluster",
            "HCCL_LOGIC_SUPERPOD_ID": "scheduler-superpod",
        })
        self.fixture.assert_success(result)
        ray = self.fixture.events("ray")
        self.assertEqual(len(ray), 1)
        self.assertIsNone(ray[0]["hccl_logic_superpod_id"])
        self.assertEqual(ray[0]["args"], [
            "start", "--head", "--port=6925", "--dashboard-host=0.0.0.0",
            "--dashboard-port=8960", '--resources={"NPU": 16}',
        ])
        self.assertEqual(ray[0]["ray_startup_env"], {
            "RAY_worker_maximum_startup_concurrency": "32",
            "RAY_prestart_worker_first_driver": "1", "RAY_enable_worker_prestart": "0",
            "HUANYU_JOB_PREFLIGHT": "0", "HUANYU_RAY_READINESS_CHECK": "0",
        })
        self.assertEqual([e["args"][0] for e in self.fixture.events("helper")], ["resolve"])
        self.assertIn("not checked", result.stdout)
        self.assertIn("export RAY_ADDRESS=192.0.2.10:6925", result.stdout)

    def test_worker_waits_for_head_then_starts_and_returns(self):
        result = self.run_start({"VC_TASK_INDEX": "1", "HCCL_LOGIC_SUPERPOD_ID": "scheduler-superpod"})
        self.fixture.assert_success(result)
        self.assertIsNone(self.fixture.events("ray")[0]["hccl_logic_superpod_id"])
        self.assertEqual(self.fixture.events("ray")[0]["args"], [
            "start", "--address=192.0.2.10:6925", '--resources={"NPU": 16}',
        ])
        events = self.fixture.events_file.read_text(encoding="utf-8")
        self.assertLess(events.index('"wait-head"'), events.index('"kind": "ray"'))
        self.assertNotIn("READY:", result.stdout)

    def test_explicit_check_uses_gcs_on_both_nodes(self):
        for rank, mode in (("0", "all"), ("1", "local")):
            with self.subTest(rank=rank):
                self.clear_events()
                result = self.run_start({"VC_TASK_INDEX": rank, "HUANYU_JOB_PREFLIGHT": "0"}, ("--check",))
                self.fixture.assert_success(result)
                calls = [e["args"] for e in self.fixture.events("helper") if e["args"][0] == "wait-cluster"]
                self.assertEqual(len(calls), 1)
                self.assertIn("--gcs", calls[0])
                self.assertEqual(calls[0][calls[0].index("--mode") + 1], mode)
                self.assertIn("READY:", result.stdout)

    def test_legacy_dashboard_helper_fails_before_starting_ray(self):
        # Use the real CLI parser, with the old flag set. argparse --help exits
        # successfully even for an incompatible version, so exit code is not enough.
        source = (ROOT / "examples/job_bootstrap.py").read_text(encoding="utf-8")
        legacy = source.replace('"--gcs", "--state-api",', '"--state-api",')
        self.assertNotEqual(source, legacy)
        (self.fixture.examples / "job_bootstrap.py").write_text(legacy, encoding="utf-8")
        for rank in ("0", "1"):
            with self.subTest(rank=rank):
                result = self.run_start({"VC_TASK_INDEX": rank}, ("--check",))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("Deploy job_bootstrap.py from the same release", result.stderr)
                self.assertNotIn("READY:", result.stdout)
                self.assertFalse(self.fixture.events_file.exists())

    def test_shipped_helper_advertises_gcs_without_importing_ray(self):
        # Exercise the shipped parser as well as the launcher's command stubs.
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             'import runpy, sys; sys.modules["ray"] = None; '
             'sys.argv = [sys.argv[1], "wait-cluster", "--help"]; '
             'runpy.run_path(sys.argv[0], run_name="__main__")',
             str(ROOT / "examples/job_bootstrap.py")],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.fixture.assert_success(result)
        self.assertIn("--gcs", result.stdout)

    def test_preflight_priority_and_legacy_alias(self):
        for updates, checked in (({"HUANYU_RAY_READINESS_CHECK": "1"}, True),
                                 ({"HUANYU_JOB_PREFLIGHT": "0", "HUANYU_RAY_READINESS_CHECK": "1"}, False)):
            with self.subTest(updates=updates):
                self.clear_events()
                result = self.run_start(updates)
                self.fixture.assert_success(result)
                self.assertEqual("READY:" in result.stdout, checked)

    def test_existing_ray_is_reused_and_checked_without_restart(self):
        result = self.run_start({"HUANYU_TEST_RAY_LIVE": "1"}, ("--check",))
        self.fixture.assert_success(result)
        self.assertFalse(self.fixture.events("ray"))
        self.assertIn("inherited startup settings are unchanged", result.stdout)
        self.assertIn("READY:", result.stdout)

    def test_failed_gcs_check_is_not_reported_as_ready(self):
        result = self.run_start({"HUANYU_TEST_FAIL_WAIT": "wait-cluster"}, ("--check",))
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertNotIn("READY:", result.stdout)
        self.assertNotIn("For training", result.stdout)
        self.assertEqual(len(self.fixture.events("ray")), 1)

    def test_failed_head_wait_never_starts_worker(self):
        result = self.run_start({"VC_TASK_INDEX": "1", "HUANYU_TEST_FAIL_WAIT": "wait-head"})
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("ray"))

    def test_failed_ray_start_never_runs_check_or_reports_success(self):
        result = self.run_start({"HUANYU_TEST_RAY_START_EXIT": "23"}, ("--check",))
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertFalse(any(e["args"][0] == "wait-cluster" for e in self.fixture.events("helper")))
        self.assertNotIn("For training", result.stdout)

    def test_single_node_infers_count_and_honors_ports(self):
        result = self.run_start({"VC_WORKER_NUM": None, "VC_WORKER_HOSTS": "head",
                                 "RAY_PORT": "7000", "RAY_DASHBOARD_PORT": "9000"}, ("--check",))
        self.fixture.assert_success(result)
        resolve = self.fixture.events("helper")[0]["args"]
        self.assertEqual(resolve[-2:], ["--expected-nodes", "1"])
        self.assertIn("--port=7000", self.fixture.events("ray")[0]["args"])
        self.assertIn("--dashboard-port=9000", self.fixture.events("ray")[0]["args"])
        self.assertIn("export RAY_ADDRESS=192.0.2.10:7000", result.stdout)

    def test_invalid_configuration_fails_before_external_commands(self):
        for updates in ({"VC_TASK_INDEX": None}, {"VC_TASK_INDEX": "2"},
                        {"VC_WORKER_HOSTS": ""}, {"VC_WORKER_NUM": "1"},
                        {"VC_WORKER_NUM": "00"}, {"RAY_PORT": "65536"},
                        {"RAY_DASHBOARD_PORT": "6925"}, {"RAY_CLUSTER_WAIT_TIMEOUT": "0"},
                        {"HUANYU_JOB_PREFLIGHT": "yes"}):
            with self.subTest(updates=updates):
                result = self.run_start(updates)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(self.fixture.events_file.exists())

    def test_unknown_option_fails_before_external_commands(self):
        result = self.run_start(arguments=("--train-only",))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events_file.exists())

    def test_busy_node_lock_prevents_second_start(self):
        result = self.run_start({"HUANYU_TEST_LOCK_BUSY": "1"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("ray"))

    def test_duplicate_local_rank_skips_and_owner_clears_device_launcher_env(self):
        result = self.run_start({"LOCAL_RANK": "1"})
        self.fixture.assert_success(result)
        self.assertFalse(self.fixture.events_file.exists())
        result = self.run_start({"LOCAL_RANK": "0", "RANK": "16", "WORLD_SIZE": "32",
                                 "TORCHELASTIC_RUN_ID": "stale-launch"})
        self.fixture.assert_success(result)
        self.assertFalse(self.fixture.events("ray")[0]["launcher_env"])

    def test_help_does_not_require_node_environment(self):
        result = self.run_start({"VC_TASK_INDEX": None, "VC_WORKER_HOSTS": None}, ("--help",))
        self.fixture.assert_success(result)
        self.assertIn("Usage:", result.stdout)
        self.assertFalse(self.fixture.events_file.exists())

    def test_sourcing_is_rejected_without_exiting_or_changing_caller_options(self):
        result = subprocess.run(
            [fixture_module.find_bash(), "-c",
             'set +e; before="$-"; source "$1"; rc=$?; '
             '[[ "$rc" == 2 && "$before" == "$-" ]] && echo CALLER_OK',
             "source-test", fixture_module.shell_path(self.fixture.entry)],
            env=self.fixture.env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.fixture.assert_success(result)
        self.assertIn("CALLER_OK", result.stdout)
        self.assertFalse(self.fixture.events_file.exists())
