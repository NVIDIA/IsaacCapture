# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which hand results the panel puts first, and the group titles."""

from __future__ import annotations

from cts_common.panel.status import Group, grouped, picked

from ..report import CheckResult, Report

# Results the panel lists first.
DECISIVE = (
    "coverage.payload_presence_rate",
    "coverage.required_joints_valid",
    "rate.interval_regularity",
    "rate.frame_gaps",
    "units.position_scale_metres",
    "quaternion.component_order",
    "coordinate_frame.handedness",
    "skeleton.joint_index_assignment",
    "segmentation.labelled_step_actually_performed",
)

GROUP_TITLES: tuple[tuple[str, str], ...] = (
    ("signal", "Schema, envelope and signal quality"),
    ("shape", "Hand shape"),
    ("placement", "Placement and frames"),
    ("windows", "Poses over the labelled windows"),
)


def groups(report: Report) -> tuple[Group, ...]:
    return grouped(report, GROUP_TITLES)


def decisive(report: Report) -> tuple[CheckResult, ...]:
    return picked(report, DECISIVE)


def table_height(report: Report) -> float | None:
    """The table surface the flat windows measured, or None when none was held."""
    for result in report.results:
        if result.name == "table.hands_same_height":
            return result.measurements.get("table_height_m")
    return None
