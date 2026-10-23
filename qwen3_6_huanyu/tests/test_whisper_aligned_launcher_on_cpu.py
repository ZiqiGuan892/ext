"""Exercise the Whisper comparison profile through Huanyu's real shell argv chain."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ENTRY = "run_huanyu_32card_whisper_aligned_job.sh"
_spec = importlib.util.spec_from_file_location(
    "huanyu_whisper_job_fixture", Path(__file__).with_name("test_job_launcher_on_cpu.py")
)
_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_job)


def effective_arguments(capture):
    return dict(
        arg.lstrip("+").split("=", 1)
        for arg in capture["argv"] if "=" in arg and not arg.startswith("--")
    )


# Literal expectations from the supplied Whisper launch command, excluding
# project/model paths, Huanyu-required adapter differences and unspecified
# length variables. Check final argv so later OmniMix defaults cannot undo it.
WHISPER_DEFAULTS = {
    "trainer.device": "npu",
    "data.train_batch_size": "32", "data.image_key": "images",
    "data.shuffle": "true", "data.seed": "42", "data.filter_overlong_prompts": "true",
    "data.filter_overlong_prompts_workers": "64", "data.truncation": "error",
    "actor_rollout_ref.model.model_stage": "thinker",
    "actor_rollout_ref.model.trust_remote_code": "true",
    "actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
    "actor_rollout_ref.model.lora_rank": "0",
    "actor_rollout_ref.model.enable_gradient_checkpointing": "true",
    "actor_rollout_ref.model.enable_activation_offload": "false",
    "actor_rollout_ref.actor.freeze_vision_tower": "true",
    "actor_rollout_ref.actor.trainer_type": "policy_gradient",
    "actor_rollout_ref.actor.strategy": "fsdp2",
    "actor_rollout_ref.actor.optim.lr": "1e-6",
    "actor_rollout_ref.actor.optim.weight_decay": "0.1",
    "actor_rollout_ref.actor.optim.clip_grad": "1.0",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.actor.use_dynamic_bsz": "true",
    "actor_rollout_ref.actor.use_torch_compile": "false",
    "actor_rollout_ref.actor.use_kl_loss": "false",
    "actor_rollout_ref.actor.entropy_coeff": "0",
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking": "true",
    "actor_rollout_ref.actor.entropy_from_logits_chunk_size": "512",
    "actor_rollout_ref.actor.policy_loss.loss_mode": "gspo",
    "actor_rollout_ref.actor.clip_ratio_low": "3e-4",
    "actor_rollout_ref.actor.clip_ratio_high": "4e-4",
    "actor_rollout_ref.actor.clip_ratio_c": "10.0",
    "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "true",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "false",
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch": "false",
    "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
    "actor_rollout_ref.actor.fsdp_config.use_orig_params": "true",
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile": "false",
    "actor_rollout_ref.actor.fsdp_config.offload_policy": "true",
    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": "true",
    "actor_rollout_ref.ref.strategy": "fsdp2",
    "actor_rollout_ref.ref.fsdp_config.param_offload": "true",
    "actor_rollout_ref.ref.fsdp_config.forward_prefetch": "true",
    "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": "true",
    "actor_rollout_ref.rollout.name": "vllm_omni", "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.rollout.n": "8", "actor_rollout_ref.rollout.temperature": "1.0",
    "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
    "actor_rollout_ref.rollout.top_p": "0.95", "actor_rollout_ref.rollout.top_k": "20",
    "actor_rollout_ref.rollout.repetition_penalty": "1.0",
    "actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty": "0.0",
    "actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty": "0.0",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": "2",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.65",
    "actor_rollout_ref.rollout.max_num_seqs": "4",
    "actor_rollout_ref.rollout.enforce_eager": "true",
    "actor_rollout_ref.rollout.load_format": "safetensors",
    "actor_rollout_ref.rollout.free_cache_engine": "true",
    "actor_rollout_ref.rollout.enable_prefix_caching": "false",
    "actor_rollout_ref.rollout.enable_chunked_prefill": "false",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "true",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": "4096",
    "actor_rollout_ref.rollout.agent.num_workers": "8",
    "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode": "ar",
    "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode": "0",
    "actor_rollout_ref.rollout.val_kwargs.n": "1",
    "actor_rollout_ref.rollout.val_kwargs.temperature": "0",
    "actor_rollout_ref.rollout.val_kwargs.top_p": "1.0",
    "actor_rollout_ref.rollout.val_kwargs.top_k": "-1",
    "algorithm.adv_estimator": "grpo", "algorithm.trainer_type": "policy_gradient",
    "algorithm.sample_source": "online", "algorithm.use_kl_in_reward": "false",
    "algorithm.norm_adv_by_std_in_grpo": "true",
    "reward.reward_manager.source": "register", "reward.reward_manager.name": "naive",
    "trainer.v1.trainer_mode": "omni_sync", "trainer.resume_mode": "disable",
    "trainer.val_before_train": "false", "trainer.balance_batch": "true",
    "trainer.critic_warmup": "0", "trainer.n_gpus_per_node": "16", "trainer.nnodes": "1",
    "trainer.save_freq": "50", "trainer.test_freq": "10", "trainer.total_epochs": "10",
}


@unittest.skipUnless(_job.find_bash(), "Bash is required for launcher argv integration")
class WhisperAlignedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _job.HuanyuJobLauncherTests("runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.entry = self.fixture.examples / ENTRY
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        for key in list(self.fixture.env):
            if key.startswith(("MAX_", "ROLLOUT_", "TRAIN_", "PPO_", "ACTOR_", "REF_")) or key in {
                "ACTIVATION_OFFLOAD", "FILTER_OVERLONG", "SAVE_FREQ", "RESUME_MODE", "VAL_BEFORE_TRAIN",
            }:
                self.fixture.env.pop(key)
        capture = self.fixture.project / "capture_train.py"
        keys = (
            "HUANYU_JOB_PROFILE", "HUANYU_JOB_MODE", "HUANYU_GRAPH_MODE",
            "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE", "ROLLOUT_N", "ROLLOUT_TP",
            "ACTOR_MODEL_DTYPE", "REF_MODEL_DTYPE", "FILTER_OVERLONG",
            "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "ROLLOUT_PROMPT_LENGTH",
            "ROLLOUT_MAX_MODEL_LEN", "MAX_TOKEN_LEN_PER_GPU", "ROLLOUT_REPETITION_PENALTY",
        )
        capture.write_text(capture.read_text(encoding="utf-8").replace(
            "keys = (\n", f"keys = {keys!r} + (\n", 1
        ), encoding="utf-8")
        self.use_real_training_composition()

    def use_real_training_composition(self):
        examples = self.fixture.examples
        for name in (
            "run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh", "graph_mode.sh", "precision_mode.sh",
        ):
            shutil.copyfile(EXAMPLES / name, examples / name)
        # The middle wrapper must remain in the chain: its final filter flag
        # used to defeat later profile overrides. Redirect only the filesystem
        # checkout preflight to an inert local package; execute its real logic.
        native = self.fixture.project / "native/vllm_omni"
        native.mkdir(parents=True)
        (native / "__init__.py").write_text("", encoding="utf-8")
        middle = "run_qwen36_huanyu_32card.sh"
        source = (EXAMPLES / middle).read_text(encoding="utf-8")
        source = source.replace(
            "_huanyu_vllm_omni_root=/vllm-workspace/vllm-omni",
            f'_huanyu_vllm_omni_root="{_job.shell_path(native.parent)}"', 1,
        )
        (examples / middle).write_text(source, encoding="utf-8", newline="\n")
        source = (EXAMPLES / "run_huanyu.sh").read_text(encoding="utf-8")
        start = source.index('exec bash "${PROJECT_ROOT}/examples/qwen3_6_huanyu/')
        (examples / "run_huanyu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            '_huanyu_ray_pythonpath=\'"fixture-pythonpath"\'\n' + source[start:],
            encoding="utf-8", newline="\n",
        )
        source = (EXAMPLES / "run_qwen36_huanyu_gspo_npu.sh").read_text(encoding="utf-8")
        command = re.search(r"^python3 -m verl_omni\.trainer\.main_omni\b", source, re.MULTILINE)
        workers = re.search(r"^ROLLOUT_NUM_WORKERS=.*$", source, re.MULTILINE)
        self.assertIsNotNone(command)
        self.assertIsNotNone(workers)
        (examples / "run_qwen36_huanyu_gspo_npu.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/graph_mode.sh"\n'
            'source "${PROJECT_ROOT}/examples/qwen3_6_huanyu/precision_mode.sh" "$@"\n'
            '_CHAT_TEMPLATE_ARGS=()\n_DATASET_ARGS=()\n'
            'DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints"\n'
            + workers.group(0) + "\n" + source[command.start():], encoding="utf-8", newline="\n",
        )
        python_stub = self.fixture.bin / "python3"
        python_stub.write_text(python_stub.read_text(encoding="utf-8").replace(
            '#!/usr/bin/env bash\n',
            '#!/usr/bin/env bash\n'
            'if [[ "${1:-}" == -m && "${2:-}" == verl_omni.trainer.main_omni ]]; then\n'
            '  exec "$HUANYU_TEST_PYTHON" "$HUANYU_TEST_TRAIN_CAPTURE" "$@"\nfi\n', 1,
        ), encoding="utf-8", newline="\n")

    def launch(self, updates=None, arguments=()):
        result = self.fixture.run_launcher(updates, arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.fixture.capture_file.is_file(), result.stdout + result.stderr)
        capture = json.loads(self.fixture.capture_file.read_text(encoding="utf-8"))
        return capture, effective_arguments(capture)

    def assert_args(self, actual, expected):
        for key, value in expected.items():
            with self.subTest(key=key):
                observed = actual.get(key)
                if value in ("true", "false") and observed is not None:
                    observed = observed.lower()
                self.assertEqual(observed, value, key)

    def test_default_final_configuration_matches_whisper_and_keeps_huanyu_contract(self):
        capture, actual = self.launch()
        self.assert_args(actual, WHISPER_DEFAULTS)
        self.assert_args(actual, {
            "data.max_prompt_length": "3072", "data.max_response_length": "2048",
            "actor_rollout_ref.rollout.prompt_length": "3104",
            "actor_rollout_ref.rollout.max_model_len": "5248",
            "actor_rollout_ref.rollout.max_num_batched_tokens": "5248",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "5248",
            "actor_rollout_ref.model.external_lib": "verl_omni_ext.models.qwen3_6_huanyu",
            "actor_rollout_ref.model.architecture": "Qwen36HuanyuForTraining",
            "actor_rollout_ref.model.use_remove_padding": "false",
            "data.custom_cls.path": "pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset",
            "data.custom_cls.name": "Qwen36HuanyuOmniRLHFDataset",
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name": "qwen3_6_huanyu",
            "reward.reward_model.n_gpus_per_node": "16", "reward.reward_model.nnodes": "1",
            "reward.reward_model.rollout.tensor_model_parallel_size": "4",
            "reward.reward_model.rollout.enforce_eager": "true",
            "reward.custom_reward_function.path": "pkg://verl_omni_ext.reward.genrm_verifier",
            "reward.reward_model.enable": "true",
            "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking": "true",
            "actor_rollout_ref.actor.checkpoint.save_contents": "['model','extra']",
            "actor_rollout_ref.model.path": _job.DEFAULT_MODEL_PATH,
            "reward.reward_model.model_path": _job.DEFAULT_RM_MODEL_PATH,
            "data.train_files": _job.DEFAULT_TRAIN_FILE, "data.val_files": _job.DEFAULT_VAL_FILE,
        })
        for key, value in {
            "HUANYU_JOB_MODE": "frozen_cache", "HUANYU_FREEZE_AUDIO": "true",
            "HUANYU_USE_PRECOMPUTE": "1", "MINICPMO_USE_PRECOMPUTE": "0",
            "NNODES": "1", "NUM_NPUS": "16", "REWARD_POOL_NPUS": "16",
        }.items():
            self.assertEqual(capture["env"][key], value, key)
        self.assertEqual(len(self.fixture.events("train")), 1)
        self.assertIn("--head", self.fixture.events("ray")[0]["args"])
        for kind in ("runtime", "ray", "train"):
            for event in self.fixture.events(kind):
                self.assertEqual(event["precision"]["HUANYU_PRECISION_PROFILE"], "full")
                self.assertEqual(event["precision"]["MINICPMO_HUANYU_ROLLOUT_TP"], "2")
                self.assertEqual(event["precision"]["MINICPMO_STABLE_LINEAR"], "1")

    def test_profile_survives_hostile_site_defaults(self):
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            stream.write(
                "export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=4\n"
                "export ACTOR_MODEL_DTYPE=fp32 FILTER_OVERLONG=false ROLLOUT_TP=4\n"
                "export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=512 ROLLOUT_PROMPT_LENGTH=4160\n"
                "export ROLLOUT_MAX_MODEL_LEN=8192 MAX_TOKEN_LEN_PER_GPU=8192\n"
                "export ROLLOUT_REPETITION_PENALTY=1.05\n"
            )
        capture, actual = self.launch()
        self.assert_args(actual, WHISPER_DEFAULTS)
        for key, expected in {
            "TRAIN_BATCH_SIZE": "32", "PPO_MINI_BATCH_SIZE": "16", "ROLLOUT_N": "8",
            "ACTOR_MODEL_DTYPE": "bfloat16", "FILTER_OVERLONG": "true", "ROLLOUT_TP": "2",
            "MAX_PROMPT_LENGTH": "3072", "MAX_RESPONSE_LENGTH": "2048",
            "ROLLOUT_PROMPT_LENGTH": "3104", "ROLLOUT_MAX_MODEL_LEN": "5248",
            "MAX_TOKEN_LEN_PER_GPU": "5248", "ROLLOUT_REPETITION_PENALTY": "1.0",
        }.items():
            self.assertEqual(capture["env"][key], expected, key)

    def test_explicit_env_and_hydra_comparison_overrides_survive(self):
        _, actual = self.launch({
            "TRAIN_BATCH_SIZE": "64", "PPO_MINI_BATCH_SIZE": "32", "ROLLOUT_N": "4",
            "ROLLOUT_TP": "4", "ACTOR_MODEL_DTYPE": "fp32", "FILTER_OVERLONG": "false",
            "MAX_PROMPT_LENGTH": "6144", "MAX_RESPONSE_LENGTH": "1024",
            "ROLLOUT_PROMPT_LENGTH": "6176", "ROLLOUT_MAX_MODEL_LEN": "7296",
            "MAX_TOKEN_LEN_PER_GPU": "7296",
        }, arguments=(
            "--scheduler_mount=/ignored/not-hydra", "trainer.total_training_steps=2",
            "data.filter_overlong_prompts=false",
            "actor_rollout_ref.actor.fsdp_config.offload_policy=false",
            "actor_rollout_ref.rollout.max_num_seqs=8",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.55",
            "actor_rollout_ref.model.override_config.attn_implementation=eager",
            "trainer.experiment_name=Whisper comparison run",
        ))
        self.assert_args(actual, {
            "data.train_batch_size": "64", "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
            "actor_rollout_ref.rollout.n": "4", "actor_rollout_ref.rollout.tensor_model_parallel_size": "4",
            "actor_rollout_ref.rollout.agent.num_workers": "4",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32", "data.filter_overlong_prompts": "false",
            "actor_rollout_ref.actor.fsdp_config.offload_policy": "false",
            "data.max_prompt_length": "6144", "data.max_response_length": "1024",
            "actor_rollout_ref.rollout.prompt_length": "6176",
            "actor_rollout_ref.rollout.max_model_len": "7296",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "7296",
            "actor_rollout_ref.rollout.max_num_seqs": "8",
            "actor_rollout_ref.rollout.gpu_memory_utilization": "0.55",
            "actor_rollout_ref.model.override_config.attn_implementation": "eager",
            "trainer.total_training_steps": "2", "trainer.experiment_name": "Whisper comparison run",
        })

    def test_worker_only_joins_cluster(self):
        result = self.fixture.run_launcher({"VC_TASK_INDEX": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.fixture.events("train"))
        self.assertFalse(self.fixture.capture_file.exists())
        args = self.fixture.events("ray")[0]["args"]
        self.assertIn("--address=192.0.2.10:6925", args)
        self.assertIn("--block", args)
        self.assertNotIn("--head", args)

    def test_external_entry_resolves_backend_from_project_root(self):
        temporary = tempfile.TemporaryDirectory(prefix=".whisper_entry_", dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        external = Path(temporary.name)
        self.fixture.entry = external / "comparison entry.sh"
        shutil.copyfile(EXAMPLES / ENTRY, self.fixture.entry)
        (external / "run_huanyu_32card_job.sh").write_text(
            '#!/usr/bin/env bash\necho "Wrong adjacent backend" >&2\nexit 91\n', encoding="utf-8"
        )
        _, actual = self.launch(arguments=("trainer.total_training_steps=2",))
        self.assert_args(actual, WHISPER_DEFAULTS)
        self.assertEqual(actual["trainer.total_training_steps"], "2")

    def test_invalid_lengths_batches_and_tp_stop_before_setup(self):
        for updates in (
            {"MAX_PROMPT_LENGTH": "0"}, {"MAX_RESPONSE_LENGTH": "abc"},
            {"ROLLOUT_MAX_MODEL_LEN": "4096"}, {"MAX_TOKEN_LEN_PER_GPU": "4096"},
            {"ROLLOUT_PROMPT_LENGTH": "3071"}, {"TRAIN_BATCH_SIZE": "0"},
            {"TRAIN_BATCH_SIZE": "33"}, {"PPO_MINI_BATCH_SIZE": "3"},
            {"PPO_MINI_BATCH_SIZE": "64"}, {"PPO_MICRO_BATCH_SIZE_PER_GPU": "0"},
            {"ROLLOUT_N": "0"}, {"ROLLOUT_TP": "3"}, {"ROLLOUT_TP": "0"},
            {"RM_TP": "3"}, {"MAX_PROMPT_LENGTH": "03072"},
        ):
            with self.subTest(updates=updates):
                result = self.fixture.run_launcher(updates)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ("helper", "env-source", "pip", "ray", "train"):
                    self.assertFalse(self.fixture.events(kind), kind)

    def test_graph_mode_rejected_before_ray_or_package_setup(self):
        result = self.fixture.run_launcher({"HUANYU_GRAPH_MODE": "1"})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        for kind in ("helper", "env-source", "pip", "ray", "train"):
            self.assertFalse(self.fixture.events(kind), kind)


if __name__ == "__main__":
    unittest.main()
