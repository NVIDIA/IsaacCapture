# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The console log level of the example's programs."""

from __future__ import annotations

import logging

LOGGER_NAME = "isaaccapture.examples.agentic_remote"
_HANDLER_NAME = "agentic_remote.console"


def set_console_level(level: str) -> None:
    """Set the console log level, through Isaac Capture's logging when it is there."""
    try:
        from isaaccapture import logging_config
    except ImportError:
        logging_config = None
    if logging_config is not None:
        logging_config.set_console_level(level)
        return
    # Without Isaac Capture nothing shows this logger tree: the example's one
    # stream handler is attached here and nowhere else.
    logger = logging.getLogger(LOGGER_NAME)
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
