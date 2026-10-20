"""Console visibility for the few Huanyu training lifecycle messages."""

import logging


def get_training_logger(name):
    """Keep INFO evidence visible without changing other workers' logging.

    Ray captures the handler's stderr; the launcher collects it with 2>&1.
    The caller still limits forward/backward messages to their first occurrence.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False
    for handler in logger.handlers:
        if getattr(handler, "_huanyu_training_console", False):
            handler.setLevel(logging.INFO)
            break
    else:
        handler = logging.StreamHandler()
        handler._huanyu_training_console = True
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(levelname)s:%(asctime)s:pid=%(process)d:%(message)s"
        ))
        logger.addHandler(handler)
    return logger
