"""Execute the real Huanyu precision launcher policy with a Python argv stub."""

import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
spec = importlib.util.spec_from_file_location("_huanyu_graph_fixture", Path(__file__).with_name("test_graph_mode_on_cpu.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)

CORE = (
    "MINICPMO_HUANYU_GDN_FP32_TRAINING", "MINICPMO_GDN_FP32_REFERENCE",
    "MINICPMO_HUANYU_LAYERWISE_RELOAD", "MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE",
    "MINICPMO_ISOLATE_MEDIA_ENCODERS",
)
FULL = (
    "MINICPMO_GDN_CACHE_BARRIER", "MINICPMO_GDN_PREFILL_REFERENCE", "MINICPMO_GDN_L2NORM_REFERENCE",
    "MINICPMO_GDN_CONV_REFERENCE", "MINICPMO_GDN_GATING_REFERENCE", "MINICPMO_STABLE_LINEAR",
    "MINICPMO_PRIVATE_HYBRID_CACHE",
)
OPTIONAL = (
    "MINICPMO_HF_NORM_REFERENCE", "MINICPMO_HF_ROUTER_REFERENCE", "MINICPMO_HF_TEXT_ROPE_REFERENCE",
    "MINICPMO_MOE_MATCH_ROLLOUT", "MINICPMO_ISOLATE_FULL_ATTENTION",
    "MINICPMO_VISION_POSITION_REFERENCE", "MINICPMO_NATIVE_VISION_REFERENCE",
    "MINICPMO_MOE_TP_REFERENCE", "MINICPMO_ROW_TP_REFERENCE",
)
PREFIX = fixture_module.ENV_PREFIX
WORKER_KEY = "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls"
WORKER_CLASS = "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
SOURCE = 'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh" "$@" || exit $?\n'


@unittest.skipUnless(fixture_module.find_bash(), "Bash is required for launcher argv integration")
class PrecisionModeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.HuanyuGraphModeTests(methodName="test_unset_mode_defaults_to_eager_and_overrides_inherited_graph_flags")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.env.pop("NNODES", None)
        self.fixture.env.pop("HCCL_OP_EXPANSION_MODE", None)
        self.fixture.env.pop("HCCL_DETERMINISTIC", None)
        keys = CORE + FULL + OPTIONAL + ("HUANYU_PRECISION_PROFILE", "MINICPMO_FSDP2_PIN_MEMORY",
                                   "MINICPMO_HUANYU_ROLLOUT_TP", "VLLM_OMNI_EXTERNAL_MODULES",
                                   "HUANYU_USE_PRECOMPUTE", "HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC", "ROLLOUT_TP",
                                   "HUANYU_HCCL_OP_EXPANSION_MODE")
        self.fixture.capture_script.write_text(
            "import json, os, pathlib, sys\n"
            + "keys = " + repr(keys) + "\n"
            + "pathlib.Path(os.environ['HUANYU_TEST_CAPTURE']).write_text(json.dumps({"
            + "'argv':sys.argv[1:], 'env':{key:os.environ.get(key) for key in keys}}),encoding='utf-8')\n",
            encoding="utf-8",
        )

    def capture(self, body=SOURCE + 'python3 "${HUANYU_PRECISION_ARGS[@]}"\n', updates=None, arguments=()):
        result = self.fixture.run_shell(body, updates=updates, arguments=arguments)
        return self.fixture.read_capture(result)

    @staticmethod
    def overrides(capture):
        return dict(arg.lstrip("+").split("=", 1) for arg in capture["argv"] if "=" in arg)

    def test_full_defaults_enable_complete_policy_without_unrelated_references(self):
        capture = self.capture()
        for key in CORE + FULL:
            self.assertEqual(capture["env"][key], "1", key)
        for key in OPTIONAL:
            self.assertEqual(capture["env"][key], "0", key)
        self.assertEqual(capture["env"]["HUANYU_PRECISION_PROFILE"], "full")
        self.assertEqual(capture["env"]["MINICPMO_FSDP2_PIN_MEMORY"], "0")
        self.assertEqual(capture["env"]["MINICPMO_HUANYU_ROLLOUT_TP"], "4")
        self.assertEqual(capture["env"]["VLLM_OMNI_EXTERNAL_MODULES"], "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks")
        for key, value in self.overrides(capture).items():
            if not key.startswith(PREFIX):
                continue
            decoded = json.loads(value)
            self.assertIsInstance(decoded, str, key)
            expected = capture["env"][key.removeprefix(PREFIX)] or ""
            self.assertEqual(decoded, expected, key)

    def test_explicit_optout_and_optional_switches_survive_repeated_source(self):
        updates = {"HUANYU_PRECISION_PROFILE": "custom", CORE[0]: "0",
                   "MINICPMO_STABLE_LINEAR": "1", "HUANYU_USE_PRECOMPUTE": "1"}
        first = self.capture(updates=updates)
        second = self.capture(SOURCE + SOURCE + 'python3 "${HUANYU_PRECISION_ARGS[@]}"\n', updates=updates)
        self.assertEqual(first, second)
        self.assertEqual(second["env"][CORE[0]], "0")
        self.assertEqual(second["env"]["MINICPMO_STABLE_LINEAR"], "1")
        self.assertEqual(second["env"]["HUANYU_USE_PRECOMPUTE"], "1")
        self.assertEqual(second["env"]["MINICPMO_GDN_PREFILL_REFERENCE"], "0")
        self.assertEqual(self.overrides(second)[WORKER_KEY], WORKER_CLASS)

    def test_full_overrides_stale_disabled_flags_and_reports_actual_policy_once(self):
        updates = dict.fromkeys(CORE + FULL, "0")
        updates["MINICPMO_FSDP2_PIN_MEMORY"] = "1"
        result = self.fixture.run_shell(
            SOURCE + SOURCE + 'python3 "${HUANYU_PRECISION_ARGS[@]}"\n', updates=updates,
        )
        capture = self.fixture.read_capture(result)
        self.assertIn("overrides inherited values", result.stdout)
        for key in CORE + FULL:
            self.assertEqual(capture["env"][key], "1", key)
            self.assertEqual(result.stdout.count(f"effective {key}=1\n"), 1)
            self.assertEqual(json.loads(self.overrides(capture)[PREFIX + key]), "1")
        self.assertEqual(capture["env"]["MINICPMO_FSDP2_PIN_MEMORY"], "0")
        self.assertEqual(json.loads(self.overrides(capture)[PREFIX + "MINICPMO_FSDP2_PIN_MEMORY"]), "0")

    def test_full_graph_conflict_rejected_in_either_source_order(self):
        graph = 'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh" || exit $?\n'
        for body in (SOURCE + graph, graph + SOURCE):
            for value in ("1", "true"):
                with self.subTest(body=body, value=value):
                    result = self.fixture.run_shell(body + "python3\n", {"HUANYU_GRAPH_MODE": value})
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("full", result.stderr)
                    self.assertFalse(self.fixture.capture.exists())

    def test_hccl_policy_is_not_synthesized_and_late_defaults_are_forwarded(self):
        clear = "unset HCCL_OP_EXPANSION_MODE HCCL_DETERMINISTIC\n"
        capture = self.capture(clear + SOURCE + 'python3 "${HUANYU_PRECISION_ARGS[@]}"\n')
        values = self.overrides(capture)
        self.assertNotIn(PREFIX + "HCCL_OP_EXPANSION_MODE", values)
        self.assertNotIn(PREFIX + "HCCL_DETERMINISTIC", values)
        capture = self.capture(
            clear + SOURCE + 'export HCCL_OP_EXPANSION_MODE=AIV\n'
            + 'huanyu_configure_precision_mode "$@"\npython3 "${HUANYU_PRECISION_ARGS[@]}"\n'
        )
        values = self.overrides(capture)
        self.assertEqual(json.loads(values[PREFIX + "HCCL_OP_EXPANSION_MODE"]), "AIV")
        self.assertNotIn(PREFIX + "HCCL_DETERMINISTIC", values)

    def test_final_hydra_tp_override_drives_worker_tp(self):
        capture = self.capture(updates={"ROLLOUT_TP": "8"}, arguments=(
            "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
            "++actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        ))
        self.assertEqual(capture["env"]["ROLLOUT_TP"], "2")
        self.assertEqual(capture["env"]["MINICPMO_HUANYU_ROLLOUT_TP"], "2")
        self.assertEqual(json.loads(self.overrides(capture)[PREFIX + "MINICPMO_HUANYU_ROLLOUT_TP"]), "2")

    def test_cross_node_auto_replaces_inherited_and_late_site_aiv(self):
        capture = self.capture(
            SOURCE + 'export HCCL_OP_EXPANSION_MODE=AIV\n'
            + 'huanyu_configure_precision_mode "$@"\npython3 "${HUANYU_PRECISION_ARGS[@]}"\n',
            updates={"NNODES": "2", "HCCL_OP_EXPANSION_MODE": "AIV"},
        )
        self.assertEqual(capture["env"]["HCCL_OP_EXPANSION_MODE"], "AI_CPU")
        self.assertEqual(capture["env"]["HUANYU_HCCL_OP_EXPANSION_MODE"], "auto")
        self.assertEqual(json.loads(self.overrides(capture)[PREFIX + "HCCL_OP_EXPANSION_MODE"]), "AI_CPU")
        self.assertEqual(json.loads(self.overrides(capture)[PREFIX + "HUANYU_HCCL_OP_EXPANSION_MODE"]), "auto")

    def test_last_trainer_node_override_selects_actual_hccl_policy(self):
        for shell_nodes, arguments, expected in (
            ("1", ("trainer.nnodes=1", "++trainer.nnodes=2"), "AI_CPU"),
            ("2", ("+trainer.nnodes=2", "trainer.nnodes=1"), "AIV"),
        ):
            with self.subTest(arguments=arguments):
                capture = self.capture(updates={"NNODES": shell_nodes, "HCCL_OP_EXPANSION_MODE": "AIV"},
                                       arguments=arguments)
                self.assertEqual(capture["env"]["HCCL_OP_EXPANSION_MODE"], expected)
                self.assertEqual(json.loads(self.overrides(capture)[PREFIX + "HCCL_OP_EXPANSION_MODE"]), expected)

    def test_explicit_cross_node_communication_policy_wins_over_inherited_aiv(self):
        for mode in ("AI_CPU", "HOST", "HOST_TS"):
            with self.subTest(mode=mode):
                capture = self.capture(updates={"NNODES": "2", "HCCL_OP_EXPANSION_MODE": "AIV",
                                               "HUANYU_HCCL_OP_EXPANSION_MODE": mode})
                self.assertEqual(capture["env"]["HCCL_OP_EXPANSION_MODE"], mode)
                for key in ("HCCL_OP_EXPANSION_MODE", "HUANYU_HCCL_OP_EXPANSION_MODE"):
                    self.assertEqual(json.loads(self.overrides(capture)[PREFIX + key]), mode)

    def test_invalid_flags_or_tp_fail_before_python(self):
        for updates, arguments in (
            ({"HUANYU_PRECISION_PROFILE": "custom", "MINICPMO_GDN_FP32_REFERENCE": "False"}, ()),
            ({"HUANYU_PRECISION_PROFILE": "invalid"}, ()),
            ({"HUANYU_USE_PRECOMPUTE": "yes"}, ()),
            ({"MINICPMO_ROW_TP_REFERENCE": "4"}, ("actor_rollout_ref.rollout.tensor_model_parallel_size=2",)),
            ({}, ("actor_rollout_ref.rollout.tensor_model_parallel_size=0",)),
            ({"HUANYU_HCCL_OP_EXPANSION_MODE": "invalid"}, ()),
            ({"NNODES": "2", "HUANYU_HCCL_OP_EXPANSION_MODE": "AIV"}, ()),
            ({"HUANYU_HCCL_OP_EXPANSION_MODE": "AIV"}, ("trainer.nnodes=2",)),
            ({"NNODES": "0"}, ()),
            ({}, ("trainer.nnodes=invalid",)),
        ):
            with self.subTest(updates=updates, arguments=arguments):
                result = self.fixture.run_shell(SOURCE + 'python3 "${HUANYU_PRECISION_ARGS[@]}"\n', updates, arguments)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(self.fixture.capture.exists())

    def test_backend_exports_before_first_python_import(self):
        text = (EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
        first_python = re.search(r"^\s*python3 -m pip install[^\n]*\n", text, re.MULTILINE)
        self.assertIsNotNone(first_python)
        # Installation now lives in a case statement; execute the complete
        # setup prefix so the shell parses the matching esac as well.
        setup_end = text.index("\nesac\n", first_python.end()) + len("\nesac\n")
        capture = self.capture("git() { return 0; }\n" + text[:setup_end],
                               updates={"HUANYU_SKIP_EDITABLE_INSTALL": "0"})
        self.assertEqual(capture["argv"][:3], ["-m", "pip", "install"])
        for key in CORE:
            self.assertEqual(capture["env"][key], "1")
        environment = (EXAMPLES / "env_export_huanyu.sh").read_text(encoding="utf-8")
        self.assertLess(environment.index('/precision_mode.sh"'), environment.index("pip install"))

    def test_final_training_argv_overrides_stale_ray_precision_for_all_modes(self):
        text = (EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
        start = re.search(r'^huanyu_configure_precision_mode "\$@" \|\| exit \$\?', text, re.MULTILINE)
        self.assertIsNotNone(start)
        setup = (
            SOURCE + 'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            + "_CHAT_TEMPLATE_ARGS=()\n_DATASET_ARGS=()\n"
            + "TRAIN_FILE=train.parquet VAL_FILE=validation.parquet MODEL_PATH=model\n"
            + "MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 MAX_TOKEN_LEN_PER_GPU=6272\n"
            + "ROLLOUT_NUM_WORKERS=4 PROJECT_NAME=test EXPERIMENT_NAME=test\n"
            + "DEFAULT_LOCAL_DIR=checkpoints NUM_NPUS=16 NNODES=1\n"
            + "export HCCL_OP_EXPANSION_MODE=AIV\n"
        )
        for frozen, cache, rm in (("true", "1", "true"), ("false", "0", "true"), ("false", "0", "false")):
            with self.subTest(frozen=frozen, cache=cache, rm=rm):
                arguments = (
                    f"++{WORKER_KEY}=old.worker.Extension",
                    "++reward.reward_model.rollout.engine_kwargs.vllm_omni.worker_extension_cls=reward.worker.Extension",
                    f"++{PREFIX}MINICPMO_GDN_FP32_REFERENCE=0",
                    f"++{PREFIX}VLLM_OMNI_EXTERNAL_MODULES=old.whisper",
                    f"++{PREFIX}HUANYU_USE_PRECOMPUTE={1-int(cache)}",
                    f"actor_rollout_ref.model.override_config.huanyu_freeze_audio={frozen}",
                    f"reward.reward_model.enable={rm}",
                )
                capture = self.capture(setup + text[start.start():], {"HUANYU_USE_PRECOMPUTE": cache}, arguments)
                values = self.overrides(capture)
                self.assertEqual(json.loads(values[PREFIX + "MINICPMO_GDN_FP32_REFERENCE"]), "1")
                self.assertEqual(json.loads(values[PREFIX + "HUANYU_USE_PRECOMPUTE"]), cache)
                self.assertEqual(json.loads(values[PREFIX + "HCCL_OP_EXPANSION_MODE"]), "AIV")
                self.assertEqual(json.loads(values[PREFIX + "VLLM_OMNI_EXTERNAL_MODULES"]),
                                 "verl_omni_ext.models.qwen3_6_huanyu.engine_hooks")
                self.assertEqual(values["actor_rollout_ref.model.override_config.huanyu_freeze_audio"], frozen)
                self.assertEqual(values["reward.reward_model.enable"], rm)
                self.assertEqual(values[WORKER_KEY], WORKER_CLASS)
                self.assertEqual(values["reward.reward_model.rollout.engine_kwargs.vllm_omni.worker_extension_cls"],
                                 "reward.worker.Extension")
                self.assertEqual(json.loads(values[PREFIX + "HUANYU_PRECISION_PROFILE"]), "full")
                self.assertEqual(json.loads(values[PREFIX + "MINICPMO_FSDP2_PIN_MEMORY"]), "0")


if __name__ == "__main__":
    unittest.main()
