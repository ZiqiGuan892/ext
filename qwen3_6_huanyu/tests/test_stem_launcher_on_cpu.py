"""STEM defaults and position-transport flags reach the final Ray job argv."""

import importlib.util
import json
from pathlib import Path
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_stem_aqa_fixture", Path(__file__).with_name("test_aqa_job_launcher_on_cpu.py")
)
_aqa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aqa)


@unittest.skipUnless(_aqa._job.find_bash(), "Bash is required for launcher argv integration")
class StemLauncherTests(unittest.TestCase):
    def setUp(self):
        self.case = _aqa.AqaJobLauncherTests("runTest")
        self.addCleanup(self.case.doCleanups)
        self.case.setUp()
        fixture = self.case.fixture
        for name in ("run_huanyu_stem.sh",):
            shutil.copyfile(ROOT / "examples" / name, fixture.examples / name)
        fixture.entry = fixture.examples / "run_huanyu_stem.sh"
        data = fixture.project / "stem data"
        data.mkdir()
        row = {
            "data_source": "math_dapo", "ability": "math",
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"index": 0, "split": "train"},
        }
        for split in ("train", "val"):
            _aqa.pq.write_table(_aqa.pa.Table.from_pylist([row] * 4), data / (split + ".parquet"))
        fixture.env.update(STEM_DATA_DIR=_aqa._job.shell_path(data), PROJECT_NAME="huanyu_stem")
        for name in ("STEM_CONTEXT_LENGTH", "STEM_PROMPT_LENGTH", "VAL_BEFORE_TRAIN", "TEST_FREQ"):
            fixture.env.pop(name, None)

    def launch(self, updates=None, arguments=()):
        capture = self.case.launch(updates, arguments)
        actual = _aqa.effective_arguments(capture)
        # Both the driver and remote workers need the flag; AQA mode stays off.
        for key, value in {"HUANYU_STEM_TEXT_TRAIN": "1", "HUANYU_AQA_AUDIO_TRAIN": "0"}.items():
            self.assertEqual(capture["env"][key], value)
            self.assertEqual(json.loads(actual[_aqa.ENV_PREFIX + key]), value)
        return actual

    def assert_batching(self, actual, dynamic, sync):
        for key in ("actor.use_dynamic_bsz", "rollout.log_prob_use_dynamic_bsz", "ref.log_prob_use_dynamic_bsz"):
            self.assertEqual(actual["actor_rollout_ref." + key], dynamic)
        self.assertEqual(actual["actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch"], sync)

    def test_default_5k_skips_initial_validation_and_enables_dynamic_positions(self):
        actual = self.launch({"HUANYU_AQA_AUDIO_TRAIN": "1", "HUANYU_STEM_TEXT_TRAIN": "0"})
        self.assert_batching(actual, "true", "false")
        self.assertEqual(actual["trainer.val_before_train"], "false")
        self.assertEqual(actual["trainer.test_freq"], "10")
        self.assertEqual(actual["data.max_response_length"], "3072")

    def test_12k_preserves_dynamic_batching_and_token_budget(self):
        actual = self.launch({"STEM_CONTEXT_LENGTH": "12k"})
        self.assert_batching(actual, "true", "false")
        self.assertEqual(actual["data.max_response_length"], "10240")
        self.assertEqual(actual["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"], "12416")

    def test_user_batching_and_validation_overrides_remain_effective(self):
        actual = self.launch({"VAL_BEFORE_TRAIN": "true"}, (
            "trainer.val_before_train=false",
            "actor_rollout_ref.actor.use_dynamic_bsz=false",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false",
            "++actor_rollout_ref.model.override_config.huanyu_sync_each_microbatch=true",
        ))
        self.assert_batching(actual, "false", "true")
        self.assertEqual(actual["trainer.val_before_train"], "false")


if __name__ == "__main__":
    unittest.main()
