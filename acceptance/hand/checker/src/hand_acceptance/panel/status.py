# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which hand results the panel puts first, and the group titles."""

from __future__ import annotations

from acceptance_common.panel.status import Group, grouped, picked

from ..report import CheckResult, Report

# Validity and rate first: without them nothing downstream means anything. Then the
# conventions that leave every later number wrong while looking plausible.
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
