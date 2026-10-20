"""Console evidence must survive restrictive worker logging defaults."""

import importlib.util
import io
import logging
from contextlib import redirect_stderr
from pathlib import Path
import unittest
from uuid import uuid4


spec = importlib.util.spec_from_file_location(
    "huanyu_training_logging_test",
    Path(__file__).resolve().parents[1] / "training_logging.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.old_root_level = self.root.level
        self.old_root_handlers = self.root.handlers[:]
        self.root_output = io.StringIO()
        self.root_handler = logging.StreamHandler(self.root_output)
        self.root_handler.setLevel(logging.WARNING)
        self.root.handlers = [self.root_handler]
        self.root.setLevel(logging.WARNING)
        self.name = f"huanyu_console_test_{uuid4().hex}"
        self.loggers = []
        self.addCleanup(self.restore_logging)

    def restore_logging(self):
        for logger in self.loggers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        self.root.handlers = self.old_root_handlers
        self.root.setLevel(self.old_root_level)
        self.root_handler.close()

    def make_logger(self, suffix="audio_training"):
        logger = module.get_training_logger(f"{self.name}.{suffix}")
        if logger not in self.loggers:
            self.loggers.append(logger)
        return logger

    def test_info_reaches_stderr_with_warning_root_and_handler(self):
        output = io.StringIO()
        with redirect_stderr(output):
            logger = self.make_logger()
            logger.info("[huanyu-audio-train] enabled")
        self.assertIn("[huanyu-audio-train] enabled", output.getvalue())
        self.assertIn("pid=", output.getvalue())
        self.assertEqual(self.root_output.getvalue(), "")
        self.assertEqual(self.root.level, logging.WARNING)
        self.assertEqual(self.root.handlers, [self.root_handler])
        self.assertEqual(self.root_handler.level, logging.WARNING)

    def test_repeated_setup_does_not_duplicate_console_messages(self):
        output = io.StringIO()
        with redirect_stderr(output):
            logger = self.make_logger()
            handlers = logger.handlers[:]
            for _ in range(3):
                self.assertIs(self.make_logger(), logger)
            logger.info("[huanyu-audio-train] backward component=audio_encoder")
        self.assertEqual(logger.handlers, handlers)
        self.assertEqual(output.getvalue().count("backward component="), 1)
        self.assertEqual(self.root_output.getvalue(), "")

    def test_adapter_info_bypasses_parent_level_without_changing_it(self):
        parent = logging.getLogger(self.name)
        parent.setLevel(logging.ERROR)
        self.addCleanup(parent.setLevel, logging.NOTSET)
        output = io.StringIO()
        with redirect_stderr(output):
            logger = self.make_logger("thinker_adapter")
            logger.info("Huanyu training adapter selected: huanyu_freeze_audio=False")
        self.assertIn("huanyu_freeze_audio=False", output.getvalue())
        self.assertEqual(parent.level, logging.ERROR)
        self.assertEqual(self.root_output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
