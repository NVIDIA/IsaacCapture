# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand script and its label timeline.

The two roll windows are motion; every other step is a held pose.
"""

from __future__ import annotations

from cts_common.labels import SIDECAR_SUFFIX, Defect, Step
from cts_common.labels import StepTimeline as _StepTimeline

__all__ = [
    "FLAT_WINDOWS",
    "PINCHES",
    "ROLLS",
    "SCRIPT",
    "SETTLE_FRACTION",
    "SIDECAR_SUFFIX",
    "STILL_LABELS",
    "Defect",
    "Step",
    "StepTimeline",
]

# Leading fraction of a held window ignored; measurements read the rest.
SETTLE_FRACTION = 0.40

PINCHES: dict[str, str] = {
    "pinch_index": "index",
    "pinch_middle": "middle",
    "pinch_ring": "ring",
    "pinch_little": "little",
}

FLAT_WINDOWS = ("flat_on_table_open", "flat_on_table_close")

# Roll window -> the hand that turns; the other holds still against its fingertip.
ROLLS: dict[str, str] = {"right_tip_roll": "right", "left_tip_roll": "left"}

SCRIPT: tuple[str, ...] = (
    FLAT_WINDOWS[0],
    "fist",
    *PINCHES,
    "palms_together",
    "tips_together",
    *ROLLS,
    FLAT_WINDOWS[1],
)

STILL_LABELS = frozenset(label for label in SCRIPT if label not in ROLLS)


class StepTimeline(_StepTimeline):
    still_labels = STILL_LABELS


def settled(step: Step, sample_ns: int) -> bool:
    return sample_ns >= step.start_ns + int(
        SETTLE_FRACTION * (step.end_ns - step.start_ns)
    )
