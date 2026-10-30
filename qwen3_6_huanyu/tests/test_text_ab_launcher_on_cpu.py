"""Validate comparison launch plans without importing Ray, vLLM, or torch."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = MODULE_ROOT.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "huanyu_text_ab_launcher", MODULE_ROOT / "examples/launch_huanyu_aqa_text_ab.py"
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def hydra_options(plan):
    return dict(arg.lstrip("+").split("=", 1) for arg in plan["command"][2:] if "=" in arg)


def cli_option(command, key):
    return command[command.index(key) + 1]


class TextABLauncherTests(unittest.TestCase):
    def setUp(self):
        # Keep even temporary files inside the workspace on managed Windows hosts.
        base = WORKSPACE / "artifacts/text_ab_launcher_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="launch ", dir=base)
        self.temp = Path(self.temporary.name).resolve()
        self.assertEqual(os.path.commonpath((str(self.temp), str(base.resolve()))), str(base.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.project = self.temp / "project with spaces"
        self.project.mkdir()
        self.root = self.temp / "case's $(literal)"
        self.root.mkdir()
        self.model = self.temp / "model's $(literal)"
        self.model.mkdir()
        self.subset = self.root / "selected samples.parquet"
        self.subset.write_bytes(b"not parsed by launcher")
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text('{"sample_id": "row_000001"}\n', encoding="utf-8")
        self.model_config = self.model / "config.json"
        self.model_config.write_text('{"model_type": "fixture"}', encoding="utf-8")
        self.case = {
            "schema_version": 1,
            "case_id": "fixture-case",
            "model_path": str(self.model),
            "subset_parquet": str(self.subset),
            "subset_parquet_sha256": hashlib.sha256(self.subset.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "checkpoint_file_sha256": {
                "config.json": hashlib.sha256(self.model_config.read_bytes()).hexdigest(),
                "chat_template.jinja": None,
            },
            "enable_thinking": False,
            "sampling": {
                "max_tokens": 1536, "temperature": 0, "top_p": 1, "top_k": -1,
                "n": 1, "seed": 123, "min_tokens": 0, "ignore_eos": False,
                "repetition_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0,
            },
            "engine": {
                "tensor_parallel_size": 4, "prompt_length": 3072,
                "max_model_len": 4736, "max_num_batched_tokens": 4736,
                "max_num_seqs": 1, "dtype": "bfloat16",
                "gpu_memory_utilization": 0.5, "enforce_eager": True,
            },
        }
        self.write_case()

    def write_case(self):
        (self.root / "case.json").write_text(json.dumps(self.case), encoding="utf-8")

    def plan(self, mode, **kwargs):
        return LAUNCHER.build_launch(self.root, self.case, self.project, mode, **kwargs)

    def invoke(self, mode, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = LAUNCHER.main([
                mode, "--case-dir", str(self.root), "--project-root", str(self.project), *extra
            ])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_both_sides_use_case_checkpoint_budget_and_request_seed(self):
        original = copy.deepcopy(self.case)
        serve, rollout = self.plan("serve"), self.plan("rollout")
        stage = serve["deploy"]["stages"][0]
        options = hydra_options(rollout)
        for plan in (serve, rollout):
            self.assertEqual(plan["env"]["MODEL_PATH"], self.case["model_path"])
            self.assertEqual(plan["env"]["ENABLE_THINKING"], "0")
        self.assertEqual(serve["command"][2], self.case["model_path"])
        self.assertEqual(rollout["env"]["VAL_FILE"], self.case["subset_parquet"])
        self.assertEqual(rollout["env"]["TRAIN_FILE"], self.case["subset_parquet"])
        self.assertEqual(int(rollout["env"]["MAX_RESPONSE_LENGTH"]), stage["default_sampling_params"]["max_tokens"])
        self.assertEqual(int(rollout["env"]["ROLLOUT_MAX_MODEL_LEN"]), stage["max_model_len"])
        self.assertEqual(int(rollout["env"]["ROLLOUT_PROMPT_LENGTH"]), self.case["engine"]["prompt_length"])
        # A per-request seed avoids the rollout replica's rank-offset engine seed.
        self.assertEqual(int(options["actor_rollout_ref.rollout.engine_kwargs.q36_sampling.seed"]),
                         stage["default_sampling_params"]["seed"])
        self.assertEqual(options["data.apply_chat_template_kwargs.enable_thinking"], "false")
        self.assertEqual(self.case, original)
        for inherited, expected in ((None, "AIV"), ("", "AIV"), ("AI_CPU", "AI_CPU")):
            with self.subTest(hccl=inherited), patch.dict(os.environ):
                if inherited is None:
                    os.environ.pop("HCCL_OP_EXPANSION_MODE", None)
                else:
                    os.environ["HCCL_OP_EXPANSION_MODE"] = inherited
                for mode in ("serve", "rollout"):
                    self.assertEqual(self.plan(mode)["env"]["HCCL_OP_EXPANSION_MODE"], expected)

    def test_serve_cli_agrees_with_deploy_and_uses_registered_thinker_stage(self):
        plan = self.plan("serve", port=30001, served_model="aqa-test")
        stage, command = plan["deploy"]["stages"][0], plan["command"]
        self.assertEqual(plan["deploy"]["pipeline"], "qwen3_6_whisper_thinker_only")
        self.assertEqual(stage["stage_id"], 0)
        self.assertEqual(stage["engine_extras"]["model_stage"], "llm")
        self.assertEqual(stage["engine_extras"]["model_arch"], "MiniCPMO50OmniForConditionalGeneration")
        self.assertEqual(stage["engine_extras"]["worker_extension_cls"],
                         "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension")
        for key in ("dtype", "tensor_parallel_size", "max_model_len", "max_num_batched_tokens",
                    "max_num_seqs", "gpu_memory_utilization", "seed", "generation_config"):
            self.assertEqual(cli_option(command, "--" + key.replace("_", "-")), str(stage[key]))
        self.assertEqual(cli_option(command, "--port"), "30001")
        self.assertEqual(cli_option(command, "--served-model-name"), "aqa-test")
        self.assertEqual(json.loads(cli_option(command, "--additional-config")),
                         stage["engine_extras"]["additional_config"])
        self.assertEqual(stage["default_sampling_params"]["n"], 1)
        self.assertTrue(stage["enforce_eager"])
        self.assertIn("--enforce-eager", command)
        self.assertFalse(stage["enable_prefix_caching"])
        self.assertIn("--no-enable-prefix-caching", command)

    def test_validation_never_trains_and_uses_one_response_per_selected_row(self):
        plan = self.plan("rollout")
        options = hydra_options(plan)
        self.assertEqual(options["trainer.val_only"], "true")
        self.assertEqual(options["trainer.val_before_train"], "true")
        self.assertEqual(options["trainer.resume_mode"], "disable")
        self.assertEqual(options["trainer.save_freq"], "-1")
        self.assertEqual(options["data.val_batch_size"], "1")
        self.assertEqual(options["data.validation_shuffle"], "false")
        self.assertEqual(options["data.val_max_samples"], "-1")
        self.assertEqual(options["actor_rollout_ref.rollout.val_kwargs.n"], "1")
        # Four replicas require four AgentLoop workers in the called launcher;
        # val_batch_size=1, rather than a fictitious worker count of one,
        # serializes the selected validation rows.
        self.assertEqual(options["actor_rollout_ref.rollout.agent.num_workers"], "4")
        self.assertEqual(options["actor_rollout_ref.rollout.ignore_eos"], "false")
        self.assertEqual(options["actor_rollout_ref.rollout.load_format"], "safetensors")
        self.assertEqual(options["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.generation_config"], "vllm")
        self.assertEqual(plan["command"][0], "bash")
        self.assertEqual(Path(plan["command"][1]),
                         self.project / "examples/qwen3_6_huanyu/run_huanyu_aqa_audio_train.sh")

    def test_batch_invariance_reaches_both_engines_and_ray_without_enabling_genrm(self):
        self.case["engine"]["batch_invariant"] = True
        with patch.dict(os.environ, {"ASCEND_LAUNCH_BLOCKING": "0", "VLLM_BATCH_INVARIANT": "1"}):
            serve, rollout = self.plan("serve"), self.plan("rollout")
        stage = serve["deploy"]["stages"][0]
        options = hydra_options(rollout)
        self.assertEqual(stage["engine_extras"]["additional_config"]["rl_config"],
                         {"enabled": True, "enable_batch_invariant": True})
        self.assertEqual(cli_option(serve["command"], "--block-size"), "128")
        self.assertEqual(stage["block_size"], 128)
        prefix = "actor_rollout_ref.rollout.engine_kwargs.vllm_omni."
        self.assertEqual(options[prefix + "additional_config.rl_config.enabled"], "true")
        self.assertEqual(options[prefix + "additional_config.rl_config.enable_batch_invariant"], "true")
        self.assertEqual(options[prefix + "block_size"], "128")
        expected = {"HCCL_DETERMINISTIC": "true", "ASCEND_LAUNCH_BLOCKING": "1",
                    "NCCL_DETERMINISTIC": "1", "LCCL_DETERMINISTIC": "1", "VLLM_BATCH_INVARIANT": "0",
                    "HUANYU_TEXT_AB_BATCH_INVARIANT": "1"}
        for key, value in expected.items():
            self.assertEqual(serve["env"][key], value)
            self.assertEqual(rollout["env"][key], value)
            self.assertEqual(json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars." + key]), value)
        self.assertTrue(stage["enforce_eager"])
        self.assertFalse(stage["enable_chunked_prefill"])
        self.assertFalse(stage["enable_prefix_caching"])
        self.assertFalse(any("reward_model" in k and "batch_invariant" in k for k in options))

    def test_external_launch_blocking_is_preserved_and_forwarded_to_ray(self):
        self.case["engine"]["batch_invariant"] = False
        with patch.dict(os.environ, {"ASCEND_LAUNCH_BLOCKING": "1", "VLLM_BATCH_INVARIANT": "1"}):
            for mode in ("serve", "rollout"):
                plan = self.plan(mode)
                self.assertEqual(plan["env"]["ASCEND_LAUNCH_BLOCKING"], "1")
                self.assertEqual(plan["env"]["VLLM_BATCH_INVARIANT"], "0")
                if mode == "rollout":
                    self.assertEqual(json.loads(hydra_options(plan)[
                        "ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_LAUNCH_BLOCKING"]), "1")
        with patch.dict(os.environ, {"ASCEND_LAUNCH_BLOCKING": "invalid"}):
            with self.assertRaisesRegex(ValueError, "ASCEND_LAUNCH_BLOCKING"):
                self.plan("rollout")

    def test_repeat_forces_native_settings_despite_ambient_disable(self):
        self.case["serve_repeat"] = {"concurrency": 1}
        self.case["engine"]["batch_invariant"] = True
        wanted = {"VLLM_BATCH_INVARIANT": "1", "HCCL_DETERMINISTIC": "strict",
                  "ASCEND_LAUNCH_BLOCKING": "1", "NCCL_DETERMINISTIC": "1", "LCCL_DETERMINISTIC": "1",
                  "HUANYU_SERVE_REPEAT_DETERMINISM": "1"}
        with patch.dict(os.environ, {key: "0" for key in wanted}):
            plan = self.plan("serve")
        self.assertEqual({key: plan["env"][key] for key in wanted}, wanted)
        stage = plan["deploy"]["stages"][0]
        self.assertEqual(stage["engine_extras"]["additional_config"]["rl_config"],
                         {"enabled": True, "enable_batch_invariant": True})
        self.assertEqual(json.loads(cli_option(plan["command"], "--additional-config"))["rl_config"],
                         stage["engine_extras"]["additional_config"]["rl_config"])
        self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_DIR"], "")
        self.assertEqual(stage["max_num_seqs"], 1)
        self.assertFalse(stage["async_scheduling"])
        self.case["engine"]["batch_invariant"] = False
        with self.assertRaisesRegex(ValueError, "prepare a new repeat case"):
            self.plan("serve")

    def test_non_repeat_clears_worker_check_including_ray_environment(self):
        with patch.dict(os.environ, {"HUANYU_SERVE_REPEAT_DETERMINISM": "1"}):
            for mode in ("serve", "rollout"):
                plan = self.plan(mode)
                self.assertEqual(plan["env"]["HUANYU_SERVE_REPEAT_DETERMINISM"], "")
                if mode == "rollout":
                    self.assertEqual(json.loads(hydra_options(plan)[
                        "ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_SERVE_REPEAT_DETERMINISM"]), "")

    def test_text_only_repeat_records_request_in_log_before_child_starts(self):
        self.case["serve_repeat"] = {"concurrency": 1}
        self.case["engine"]["batch_invariant"] = True
        self.write_case()
        with patch.object(LAUNCHER.subprocess, "call") as call:
            status, _, error = self.invoke("serve", "--dry-run")
        self.assertEqual(status, 0, error)
        self.assertFalse((self.root / "serve_console.log").exists())
        call.assert_not_called()

        def start(argv, env):
            log = (self.root / "serve_console.log").read_text(encoding="utf-8")
            self.assertIn("[huanyu-determinism] requested", log)
            self.assertIn("HCCL_DETERMINISTIC=strict", log)
            self.assertIn("repeat_worker_check=true", log)
            self.assertNotIn("actual", log)  # Parent cannot certify worker state.
            self.assertEqual(env["HUANYU_EMBEDDING_AB_DIR"], "")
            return 0

        with patch.object(LAUNCHER.subprocess, "call", side_effect=start):
            status, _, error = self.invoke("serve")
        self.assertEqual(status, 0, error)
        snapshot = json.loads((self.root / "serve.launch.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["selected_environment"]["HUANYU_SERVE_REPEAT_DETERMINISM"], "1")

    def test_full_audit_propagates_to_ray_and_never_overwrites_text_run(self):
        audit = self.temp / "new audit"
        plan = self.plan("rollout", full_weight_audit_dir=audit)
        options = hydra_options(plan)
        for key in ("HUANYU_CHECKPOINT_VALIDATION_MODE", "HUANYU_FULL_WEIGHT_AUDIT_DIR",
                    "HUANYU_FULL_WEIGHT_AUDIT_RUN_ID", "HUANYU_TEXT_AB_DIR"):
            self.assertEqual(json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars." + key]), plan["env"][key])
            self.assertEqual(sum(arg.startswith("++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=")
                                 for arg in plan["command"]), 1)
        self.assertEqual(plan["env"]["HUANYU_TEXT_AB_DIR"], "")
        self.assertEqual(plan["env"]["HUANYU_CHECKPOINT_VALIDATION_MODE"], "sync")
        self.assertEqual(len(plan["env"]["HUANYU_FULL_WEIGHT_AUDIT_RUN_ID"]), 32)
        (self.root / "rollout.launch.json").write_text("keep text run", encoding="utf-8")
        with patch.object(LAUNCHER.subprocess, "call", return_value=1):
            status, out, err = self.invoke("rollout", "--full-weight-audit-dir", str(audit))
        self.assertEqual(status, 1)  # intentional stopping status is preserved
        self.assertTrue((audit / "full_weights.launch.json").is_file())
        self.assertEqual((self.root / "rollout.launch.json").read_text(), "keep text run")
        with self.assertRaisesRegex(ValueError, "only supported for rollout"):
            self.plan("serve", full_weight_audit_dir=audit)

    def test_stale_audio_training_and_disk_probe_environment_is_overridden(self):
        stale = {
            "MODEL_PATH": "/wrong/model", "TRAIN_FILE": "/wrong/train", "VAL_FILE": "/wrong/val",
            "HUANYU_TRAIN_AUDIO_ENCODER": "true", "HUANYU_TRAIN_AUDIO_PROJECTOR": "true",
            "HUANYU_FREEZE_AUDIO": "false", "HUANYU_USE_PRECOMPUTE": "1",
            "MINICPMO_USE_PRECOMPUTE": "1", "HUANYU_PRECISION_PROFILE": "off",
            "HUANYU_CHECKPOINT_VALIDATION_MODE": "disk", "HUANYU_WEIGHT_SYNC_AUDIT_DIR": "/old/weights",
            "HUANYU_FULL_WEIGHT_AUDIT_DIR": "/old/full", "HUANYU_FULL_WEIGHT_AUDIT_RUN_ID": "old-run",
            "HUANYU_AUDIO_PROBE_DIR": "/old/probe", "HUANYU_AUDIO_PROBE_ARM_FILE": "/old/arm",
            "HUANYU_AUDIO_PROBE_FEATURE_ID": "old-feature", "HUANYU_GRAPH_MODE": "1",
            "RAY_ADDRESS": "ray-head:6379", "DEBUG_MODE": "1", "DEBUG_SAMPLES": "10",
            "VLLM_ASCEND_ENABLE_NZ": "1", "USE_OPTIMIZED_MODEL": "1",
        }
        cleared = [key for key in stale if "PROBE" in key or key in (
            "HUANYU_CHECKPOINT_VALIDATION_MODE", "HUANYU_WEIGHT_SYNC_AUDIT_DIR",
            "HUANYU_FULL_WEIGHT_AUDIT_DIR", "HUANYU_FULL_WEIGHT_AUDIT_RUN_ID")]
        with patch.dict(os.environ, stale):
            for mode in ("serve", "rollout"):
                plan = self.plan(mode)
                env = plan["env"]
                self.assertEqual(env["HUANYU_TRAIN_AUDIO_ENCODER"], "false")
                self.assertEqual(env["HUANYU_TRAIN_AUDIO_PROJECTOR"], "false")
                self.assertEqual(env["HUANYU_FREEZE_AUDIO"], "true")
                self.assertEqual(env["HUANYU_USE_PRECOMPUTE"], "0")
                self.assertEqual(env["HUANYU_PRECISION_PROFILE"], "full")
                self.assertEqual(env["HUANYU_GRAPH_MODE"], "0")
                self.assertEqual(env["MODEL_PATH"], str(self.model))
                self.assertEqual(env["VLLM_ASCEND_ENABLE_NZ"], "0")
                self.assertEqual(env["USE_OPTIMIZED_MODEL"], "0")
                for key in cleared:
                    self.assertEqual(env[key], "", key)
                if mode == "rollout":
                    options = hydra_options(plan)
                    # Empty runtime-env values override stale existing Ray workers too.
                    for key in cleared:
                        self.assertEqual(json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars." + key]), "")
                    self.assertEqual(env["RAY_ADDRESS"], stale["RAY_ADDRESS"])
                    self.assertEqual(env["DEBUG_MODE"], "0")
                    self.assertEqual(env["HUANYU_AQA_VAL_FILE"], str(self.subset))
            self.assertEqual(os.environ["MODEL_PATH"], stale["MODEL_PATH"])

    def test_paths_remain_literal_arguments_and_hydra_quoted_strings(self):
        serve, rollout = self.plan("serve"), self.plan("rollout")
        self.assertEqual(serve["argv"][4], LAUNCHER.sys.executable)
        self.assertEqual(Path(serve["argv"][5]), Path(LAUNCHER.__file__).resolve())
        self.assertEqual(serve["argv"][6], "_serve_cli")
        self.assertEqual(serve["argv"][7:], serve["command"][1:])
        shell = serve["argv"][2]
        self.assertIn('exec "$@"', shell)
        self.assertNotIn(str(self.model), shell)
        self.assertNotIn(str(self.root), shell)
        self.assertEqual(cli_option(serve["command"], "--deploy-config"), str(self.root / "serve.deploy.yaml"))
        options = hydra_options(rollout)
        self.assertEqual(json.loads(options["trainer.validation_data_dir"]), str(self.root / "validation_dump"))
        self.assertEqual(json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_TEXT_AB_DIR"]), str(self.root))

    def test_embedding_capture_is_enabled_on_both_real_engines_and_disabled_for_weight_audit(self):
        self.case["embedding_check"] = {"enabled": True, "scope": "llm_inputs_embeds", "strict": True,
                                       "expected_tp_size": 4}
        for side in ("serve", "rollout"):
            plan = self.plan(side)
            self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_DIR"], str(self.root))
            self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_SIDE"], side)
            if side == "rollout":
                options = hydra_options(plan)
                for key in ("HUANYU_EMBEDDING_AB_DIR", "HUANYU_EMBEDDING_AB_SIDE"):
                    self.assertEqual(json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars." + key]),
                                     plan["env"][key])
        audit = self.plan("rollout", full_weight_audit_dir=self.temp / "audit")
        for key in ("HUANYU_EMBEDDING_AB_DIR", "HUANYU_EMBEDDING_AB_SIDE"):
            self.assertEqual(audit["env"][key], "")
            self.assertEqual(json.loads(hydra_options(audit)["ray_kwargs.ray_init.runtime_env.env_vars." + key]), "")

    def test_legacy_text_case_clears_stale_embedding_environment(self):
        with patch.dict(os.environ, {"HUANYU_EMBEDDING_AB_DIR": "/other-case", "HUANYU_EMBEDDING_AB_SIDE": "serve"}):
            for side in ("serve", "rollout"):
                plan = self.plan(side)
                self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_DIR"], "")
                self.assertEqual(plan["env"]["HUANYU_EMBEDDING_AB_SIDE"], "")

    def test_logits_capture_requires_embedding_identity_and_synchronous_sampling(self):
        self.case["logits_check"] = {"enabled": True, "max_steps": 2048}
        with self.assertRaisesRegex(ValueError, "embedding_check"):
            self.plan("serve")
        self.case["embedding_check"] = {"enabled": True, "scope": "llm_inputs_embeds", "strict": True,
                                       "expected_tp_size": 4}
        serve, rollout = self.plan("serve"), self.plan("rollout")
        self.assertIn("--no-async-scheduling", serve["command"])
        self.assertIs(serve["deploy"]["stages"][0]["async_scheduling"], False)
        self.assertEqual(hydra_options(rollout)["actor_rollout_ref.rollout.engine_kwargs.vllm_omni.async_scheduling"], "false")

    def test_serve_pythonpath_prefers_native_and_retains_unrelated_entries(self):
        native = "/vllm-workspace/vllm-omni"
        stale = "/opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit"
        with patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join((stale, str(self.project), native, "/extra/lib"))}):
            entries = self.plan("serve")["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[:2], [native, str(self.project)])
        self.assertNotIn(stale, entries)
        self.assertEqual(entries.count(native), 1)
        self.assertIn("/extra/lib", entries)

    def test_invalid_sampling_context_or_topology_is_rejected_before_launch(self):
        for section, key, value in (("sampling", "temperature", 0.8), ("sampling", "n", 4),
                                    ("sampling", "ignore_eos", True), ("sampling", "seed", -1),
                                    ("engine", "max_model_len", 10), ("engine", "tensor_parallel_size", 2),
                                    ("engine", "max_num_seqs", 32), ("engine", "enforce_eager", False)):
            case = copy.deepcopy(self.case)
            case[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                LAUNCHER.build_launch(self.root, case, self.project, "serve")
        with self.assertRaises(ValueError):
            self.plan("serve", port=65536)

    def test_dry_run_has_no_process_or_persistent_side_effect(self):
        self.case["model_path"] = str(self.temp / "not available on this machine")
        self.write_case()
        with patch.object(LAUNCHER.subprocess, "call") as call:
            status, output, error = self.invoke("serve", "--dry-run")
        self.assertEqual(status, 0, error)
        self.assertIn("vllm serve", output)
        call.assert_not_called()
        self.assertFalse((self.root / "serve.launch.json").exists())
        self.assertFalse((self.root / "serve.deploy.yaml").exists())

    def test_real_handoff_persists_reproducible_plan_without_ambient_secrets(self):
        secret = "fixture-secret-that-must-not-be-written"
        with patch.dict(os.environ, {"HF_TOKEN": secret}), patch.object(LAUNCHER.subprocess, "call", return_value=17) as call:
            status, _, error = self.invoke("serve")
        self.assertEqual(status, 17, error)
        call.assert_called_once()
        contents = (self.root / "serve.launch.json").read_text(encoding="utf-8")
        snapshot = json.loads(contents)
        self.assertNotIn("env", snapshot)
        self.assertNotIn(secret, contents)
        self.assertNotIn("HF_TOKEN", snapshot["selected_environment"])
        self.assertEqual(snapshot["selected_environment"]["HCCL_OP_EXPANSION_MODE"],
                         call.call_args.kwargs["env"]["HCCL_OP_EXPANSION_MODE"])
        self.assertEqual(snapshot["argv"], call.call_args.args[0])
        self.assertEqual(call.call_args.kwargs["env"]["MODEL_PATH"], str(self.model))
        self.assertEqual(json.loads((self.root / "serve.deploy.yaml").read_text()), snapshot["deploy"])
        with patch.object(LAUNCHER.subprocess, "call") as second_call:
            status, _, _ = self.invoke("serve")
        self.assertEqual(status, 1)
        second_call.assert_not_called()

    def test_rollout_handoff_preserves_shared_case_and_validation_capture_directory(self):
        with patch.object(LAUNCHER.subprocess, "call", return_value=0) as call:
            status, _, error = self.invoke("rollout")
        self.assertEqual(status, 0, error)
        self.assertEqual(call.call_args.kwargs["env"]["HUANYU_TEXT_AB_DIR"], str(self.root))
        snapshot = json.loads((self.root / "rollout.launch.json").read_text())
        self.assertEqual(snapshot["case_id"], self.case["case_id"])
        self.assertIsNone(snapshot["deploy"])
        self.assertIn("trainer.val_only=true", call.call_args.args[0])
        self.assertFalse((self.root / "serve.deploy.yaml").exists())

    def test_bad_case_or_missing_files_never_start_process(self):
        with patch.object(LAUNCHER.subprocess, "call") as call:
            self.case["schema_version"] = 99
            self.write_case()
            status, _, error = self.invoke("rollout")
            self.assertEqual(status, 1)
            self.assertIn("schema", error)
            self.case["schema_version"] = 1
            self.case["subset_parquet"] = str(self.root / "missing.parquet")
            self.write_case()
            status, _, error = self.invoke("rollout")
            self.assertEqual(status, 1)
            self.assertIn("absent", error)
        call.assert_not_called()
        self.assertFalse((self.root / "rollout.launch.json").exists())

    def test_changed_subset_manifest_or_checkpoint_config_cannot_start_comparison(self):
        for path in (self.subset, self.manifest, self.model_config):
            original = path.read_bytes()
            with self.subTest(path=path.name), patch.object(LAUNCHER.subprocess, "call") as call:
                try:
                    path.write_bytes(original + b"changed after preparation")
                    status, _, error = self.invoke("rollout")
                    self.assertEqual(status, 1)
                    self.assertIn("changed", error)
                    call.assert_not_called()
                    self.assertFalse((self.root / "rollout.launch.json").exists())
                finally:
                    path.write_bytes(original)

    def test_api_pipeline_registration_precedes_native_cli_resolution_in_same_process(self):
        pipelines, events = {}, []
        pipeline = SimpleNamespace(model_type="qwen3_6_whisper_thinker_only")
        # A cached registration module need not run its import side effect again.
        registration = SimpleNamespace(QWEN36_WHISPER_THINKER_ONLY_PIPELINE=pipeline)
        def register(value):
            events.append("register")
            pipelines[value.model_type] = value
        native_arguments = self.plan("serve")["command"][1:]
        original_argv = LAUNCHER.sys.argv
        def cli_main():
            events.append("native_cli")
            # Reproduce StageConfigFactory's lookup before worker hook imports.
            self.assertIs(pipelines["qwen3_6_whisper_thinker_only"], pipeline)
            self.assertEqual(LAUNCHER.sys.argv, ["vllm", *native_arguments])
            return 17
        modules = {
            "verl_omni_ext.models.qwen3_6_whisper.vllm_omni": registration,
            "vllm_omni.config.pipeline_registry": SimpleNamespace(register_pipeline=register),
            "vllm.entrypoints.cli.main": SimpleNamespace(main=cli_main),
        }
        def load(name):
            if name == "vllm.entrypoints.cli.main":
                self.assertIn("qwen3_6_whisper_thinker_only", pipelines)
            return modules[name]
        with patch.object(LAUNCHER.importlib, "import_module", side_effect=load), redirect_stdout(io.StringIO()):
            self.assertEqual(LAUNCHER.main(["_serve_cli", *native_arguments]), 17)
        self.assertEqual(events, ["register", "native_cli"])
        self.assertIs(LAUNCHER.sys.argv, original_argv)

    def test_api_registration_failure_never_falls_back_to_unregistered_native_cli(self):
        original_argv = LAUNCHER.sys.argv
        with patch.object(LAUNCHER.importlib, "import_module", side_effect=ImportError("missing extension")) as load:
            with self.assertRaisesRegex(ImportError, "missing extension"):
                LAUNCHER.main(["_serve_cli", "serve", str(self.model), "--omni"])
        self.assertEqual(load.call_count, 1)
        self.assertIs(LAUNCHER.sys.argv, original_argv)

    def test_internal_entry_rejects_non_serve_commands_without_imports(self):
        with patch.object(LAUNCHER.importlib, "import_module") as load:
            with self.assertRaises(ValueError):
                LAUNCHER.main(["_serve_cli", "collect-env"])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
