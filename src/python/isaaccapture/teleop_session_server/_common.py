# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pieces the Teleop Session Server's main and session processes share."""

from __future__ import annotations

import logging

from isaaccapture.logging_config import set_console_level
from isaaccapture.logging_config._core import logging_enabled

DEFAULT_TCP_PORT = 47600
DEFAULT_WS_PORT = 47601

BUSY_MESSAGE = "busy: another robot app is connected to this Teleop Session Server"
SHUTDOWN_MESSAGE = "The Teleop Session Server is shutting down"


def configure_logging(verbose: bool) -> None:
    """Debug output with ``verbose``; with isaaccapture logging off, a plain stderr handler."""
    if logging_enabled():
        if verbose:
            set_console_level("debug")
        return
    # Nothing of ours is attached when logging is off, so info records would vanish.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
