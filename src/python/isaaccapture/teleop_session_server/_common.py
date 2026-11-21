# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server's ports, refusal texts, retry delays and logging setup."""

from __future__ import annotations

import logging

from isaaccapture.logging_config import set_console_level
from isaaccapture.logging_config._core import logging_enabled

DEFAULT_TCP_PORT = 47600
DEFAULT_WS_PORT = 47601

BUSY_MESSAGE = "busy: another robot app is connected to this Teleop Session Server"
SHUTDOWN_MESSAGE = "The Teleop Session Server is shutting down"

# Seconds before the OpenXR session or a robot app's trackers are tried again after a failure.
RETRY_S = 5.0
RETRY_MAX_S = 60.0


class Backoff:
    """Retry delays: RETRY_S, doubled up to RETRY_MAX_S while the same failure repeats."""

    def __init__(self) -> None:
        self._delay_s = RETRY_S
        # The last failure and when it happened.
        self._last: tuple[str, float] | None = None

    def delay(self, failure: str, now: float) -> float:
        """Seconds until the next try after ``failure`` at ``now`` (time.monotonic())."""
        last = self._last
        # A failure that comes after a pause longer than two delays starts over.
        repeated = (
            last is not None
            and last[0] == failure
            and now - last[1] <= 2 * self._delay_s
        )
        self._delay_s = min(self._delay_s * 2, RETRY_MAX_S) if repeated else RETRY_S
        self._last = (failure, now)
        return self._delay_s


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
