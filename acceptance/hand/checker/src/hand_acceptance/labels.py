# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand script and its label timeline.

Every step is a held pose except ``wrist_rotate``. The capture opens each window on a
timer, so the first part of a window may still hold the transition into it; window
measurements read only the trailing ``1 - SETTLE_FRACTION`` of each.
"""

from __future__ import annotations

from acceptance_common.labels import SIDECAR_SUFFIX, Defect, Step
from acceptance_common.labels import StepTimeline as _StepTimeline

__all__ = [
    "GRASPS",
    "PINCHES",
    "SCRIPT",
    "SETTLE_FRACTION",
    "SIDECAR_SUFFIX",
    "STILL_LABELS",
    "Defect",
    "Step",
    "StepTimeline",
]

SETTLE_FRACTION = 0.40

PINCHES: dict[str, str] = {
    "pinch_index": "index",
    "pinch_middle": "middle",
    "pinch_ring": "ring",
    "pinch_little": "little",
}

# The grasped fist belongs to the hand named second.
GRASPS: dict[str, tuple[str, str]] = {
    "right_grasps_left_fist": ("right", "left"),
    "left_grasps_right_fist": ("left", "right"),
}

SCRIPT: tuple[str, ...] = (
    "open_hand_open",
    "fist",
    *GRASPS,
    *PINCHES,
    "wrist_rotate",
    "open_hand_close",
)

STILL_LABELS = frozenset(label for label in SCRIPT if label != "wrist_rotate")


class StepTimeline(_StepTimeline):
    still_labels = STILL_LABELS


def settled(step: Step, sample_ns: int) -> bool:
    return sample_ns >= step.start_ns + int(
        SETTLE_FRACTION * (step.end_ns - step.start_ns)
    )
