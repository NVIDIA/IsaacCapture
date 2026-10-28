# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which results the panel puts first, and how the rest group."""

from __future__ import annotations

from cts_common.panel.status import WORST_FIRST, Group, grouped, picked, worst_first

from ..report import CheckResult, Report

__all__ = [
    "DECISIVE",
    "GROUP_TITLES",
    "WORST_FIRST",
    "Group",
    "decisive",
    "groups",
    "worst_first",
]

# Results that decide whether the rest of the report is worth reading, in priority order.
DECISIVE = (
    "coverage.payload_presence_rate",
    "coverage.validity_trend",
    "rate.interval_regularity",
    "rate.frame_gaps",
    "coordinate_frame.up_axis",
    "coordinate_frame.handedness",
    "skeleton.left_right_labelling",
    "units.position_scale_metres",
    "skeleton.bone_length_constancy",
)

GROUP_TITLES: tuple[tuple[str, str], ...] = (
    ("signal", "Schema, envelope and signal quality"),
    ("geometry", "Skeleton geometry"),
    ("posture", "Posture over the labelled windows"),
)


def groups(report: Report) -> tuple[Group, ...]:
    return grouped(report, GROUP_TITLES)


def decisive(report: Report) -> tuple[CheckResult, ...]:
    return picked(report, DECISIVE)
