# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which results the panel puts first, and how the rest group.

The text report already lists all 38 checks, and moving 38 lines onto a web page buys
nothing. What it cannot do is order them by what would make a take worth discarding, so
that ordering lives here.
"""

from __future__ import annotations

from acceptance_common.panel.status import WORST_FIRST, worst_first
from acceptance_common.panel.status import Group as Gate
from acceptance_common.panel.status import grouped, picked

from ..report import CheckResult, Report

__all__ = [
    "DECISIVE",
    "GATE_TITLES",
    "WORST_FIRST",
    "Gate",
    "decisive",
    "gates",
    "worst_first",
]

# The few results that decide whether the rest of the report is worth reading at all.
# Joint validity first: a take whose joints were never valid is void whatever else it
# says. Then rate, because dropped frames make every derivative meaningless. Then the
# coordinate conventions, which are one-off facts that leave every downstream number
# wrong while looking entirely plausible.
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

# The gate numbering is the narrative for the submitter, not for whoever is looking at a
# take, so the panel groups by what the checks read and names the group after that. G1
# carries G3's signal-quality checks, which is why its title says both. A gate with no
# checks of its own is not shown at all: G0 is the submitter's attestation, and G5 and
# G6 are not answerable from a recording.
GATE_TITLES: tuple[tuple[str, str], ...] = (
    ("G1", "Schema, envelope and signal quality"),
    ("G2", "Skeleton geometry"),
    ("G4", "Posture over the labelled windows"),
)


def gates(report: Report) -> tuple[Gate, ...]:
    return grouped(report, GATE_TITLES)


def decisive(report: Report) -> tuple[CheckResult, ...]:
    return picked(report, DECISIVE)
