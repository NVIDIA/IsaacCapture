# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""An injectable monotonic clock.

``mock_isaac_capture.TeleopSession.step()`` derives synthetic motion from
call-time elapsed time rather than a tick counter: no background ticking
thread, sleep/rate limit or configured sample-rate driver -- motion speed
must not depend on how frequently the SDK is polled. A hardcoded
``time.monotonic()`` call would make that motion -- and any adapter-owned
deadline pacing built on the same clock -- untestable
without real wall-clock sleeps; passing a :class:`Clock` lets a test
substitute a controlled source of "now" instead.
"""

from __future__ import annotations

import time
from typing import Protocol

__all__ = ["Clock", "MonotonicClock"]


class Clock(Protocol):
    # Protocol method declaration: `...` is the standard stub body, not a no-op mistake.
    def monotonic(self) -> float: ...


class MonotonicClock:
    """The real wall-clock monotonic source; the default for production
    and mock use alike outside of tests."""

    def monotonic(self) -> float:
        return time.monotonic()
