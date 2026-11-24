# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What makes a received LiveKit frame's timing usable, shared by the recorder and the server's delivery check.

A check that is weaker than the consumer it qualifies the path for would pass frames the consumer then rejects, so
both read these limits and this uncertainty budget from here.
"""

from __future__ import annotations

import os

#: A frame older than this when it arrives, or stamped further than the lead below in the future, means the
#: publisher's clock and this host's disagree grossly. This catches gross disagreement only, not millisecond error.
MAX_FRAME_AGE_S = 2.0
MAX_FRAME_LEAD_S = 0.05

BUDGET_ENV = "EGOCAP_CLOCK_UNCERTAINTY_BUDGET_MS"
# A frame at 30 fps is 33 ms; a mapping uncertain by more than a few ms would blur which camera frame matches a
# tracking sample. This default is a policy choice, not a measurement; set the environment variable to change it.
DEFAULT_BUDGET_MS = 5.0


def uncertainty_budget_ns() -> int:
    """The largest clock-mapping uncertainty a frame may be placed with, from the environment or the default."""
    raw = os.environ.get(BUDGET_ENV)
    if raw is None or not raw.strip():
        return int(DEFAULT_BUDGET_MS * 1e6)
    try:
        ms = float(raw)
    except ValueError:
        raise ValueError(f"{BUDGET_ENV} must be a number of milliseconds, got {raw!r}") from None
    if not ms > 0:
        raise ValueError(f"{BUDGET_ENV} must be positive, got {raw!r}")
    return int(ms * 1e6)


def frame_age_is_plausible(age_s: float) -> bool:
    """Whether a frame's age at arrival is within the limits a recording accepts (NaN is not)."""
    return -MAX_FRAME_LEAD_S <= age_s <= MAX_FRAME_AGE_S
