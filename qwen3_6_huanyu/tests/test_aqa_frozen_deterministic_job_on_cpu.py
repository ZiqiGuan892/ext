"""Exercise the frozen job through site setup, real training argv and stubbed Ray."""

import json
import shutil
import unittest

import test_aqa_job_launcher_on_cpu as aqa


ENTRY = "run_huanyu_aqa_frozen_deterministic_32card_job.sh"
EXPECTED = {
    "HUANYU_JOB_DETERMINISM": "1", "HUANYU_FREEZE_AUDIO": "true",
    "HUANYU_TRAIN_AUDIO_ENCODER": "false", "HUANYU_TRAIN_AUDIO_PROJECTOR": "false",
    "MINICPMO_TUNE_AUDIO": "0", "HUANYU_USE_PRECOMPUTE": "0",
    "HCCL_DETERMINISTIC": "strict", "ASCEND_LAUNCH_BLOCKING": "1",
    "NCCL_DETERMINISTIC": "1", "LCCL_DETERMINISTIC": "1", "VLLM_BATCH_INVARIANT": "0",
}


@unittest.skipUnless(aqa._job.find_bash(), "Bash is required")
class FrozenDeterministicJobTests(unittest.TestCase):
    def setUp(self):
        self.helper = aqa.AqaJobLauncherTests("runTest")
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.fixture = self.helper.fixture
        self.fixture.entry = self.fixture.examples / ENTRY
        shutil.copyfile(aqa.EXAMPLES / ENTRY, self.fixture.entry)
        # Both the invoking shell and site setup carry deliberately wrong values.
        self.stale = {key: "0" for key in EXPECTED}
        self.stale.update(HUANYU_TRAIN_AUDIO_ENCODER="true", HUANYU_TRAIN_AUDIO_PROJECTOR="true",
                          HUANYU_FREEZE_AUDIO="false", MINICPMO_TUNE_AUDIO="1",
                          VLLM_BATCH_INVARIANT="1", HUANYU_USE_PRECOMPUTE="1")
        with (self.fixture.examples / "env_export_huanyu.sh").open("a", encoding="utf-8") as stream:
            for key, value in self.stale.items():
                stream.write(f"export {key}={value}\n")

    def test_head_keeps_original_batch_and_length_but_freezes_audio_and_enforces_determinism(self):
        capture = self.helper.launch(self.stale, (
            "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=true",
            "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=true",
            '++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_DETERMINISTIC="false"',
            '++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_JOB_DETERMINISM="0"',
            "actor_rollout_ref.rollout.full_determinism=true",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enable_batch_invariant=true",
        ))
        actual = aqa.effective_arguments(capture)
        for key, value in EXPECTED.items():
            self.assertEqual(capture["env"][key], value, key)
            self.assertEqual(json.loads(actual[aqa.ENV_PREFIX + key]), value, key)
        for key, value in {
            "actor_rollout_ref.model.override_config.huanyu_freeze_audio": "true",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_encoder": "false",
            "actor_rollout_ref.model.override_config.huanyu_train_audio_projector": "false",
            "actor_rollout_ref.rollout.full_determinism": "false",
            "reward.reward_model.rollout.full_determinism": "false",
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enable_batch_invariant": "false",
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled": "true",
            "data.train_batch_size": "4", "actor_rollout_ref.actor.ppo_mini_batch_size": "4",
            "actor_rollout_ref.rollout.n": "4", "actor_rollout_ref.rollout.max_model_len": "5248",
        }.items():
            self.assertEqual(actual[key], value, key)
        self.assertEqual(capture["env"]["EXPERIMENT_NAME"], "huanyu_aqa_frozen_deterministic_5k_32card")
        for event in self.fixture.events("runtime") + self.fixture.events("ray"):
            for key, value in EXPECTED.items():
                self.assertEqual(event["env"][key], value, key)

    def test_reused_ray_receives_explicit_runtime_env_without_starting_ray_again(self):
        capture = self.helper.launch(dict(self.stale, HUANYU_TEST_RAY_LIVE="1"))
        actual = aqa.effective_arguments(capture)
        for key, value in EXPECTED.items():
            self.assertEqual(json.loads(actual[aqa.ENV_PREFIX + key]), value, key)
        self.assertFalse(any("start" in e["args"] for e in self.fixture.events("ray")))

    def test_worker_joins_with_identical_policy_without_submitting_training(self):
        result = self.fixture.run_launcher(dict(self.stale, VC_TASK_INDEX="1"))
        self.fixture.assert_worker_supervised_until_exit(result)
        self.assertFalse(self.fixture.events("train"))
        self.assertIn("--address=192.0.2.10:6925", self.fixture.events("ray")[0]["args"])
        for event in self.fixture.events("ray") + self.fixture.events("runtime"):
            for key, value in EXPECTED.items():
                self.assertEqual(event["env"][key], value, key)


if __name__ == "__main__":
    unittest.main()
