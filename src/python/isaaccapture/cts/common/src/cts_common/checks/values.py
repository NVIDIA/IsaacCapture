# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class Finite(Check):
    name = "values.finite"
    severity = Severity.HARD
    summary = "Positions and orientations on valid joints are finite"

    def __init__(self) -> None:
        super().__init__()
        self.nan_components = 0
        self.inf_components = 0
        self.offending_joints: set[int] = set()
        self.first_frame: int | None = None

    def _update(self, frame: Frame) -> None:
        # Invalid joints carry arbitrary poses and are skipped.
        for index, joint in frame.valid_joints():
            for value in (*joint.position, *joint.orientation):
                if math.isnan(value):
                    self.nan_components += 1
                elif math.isinf(value):
                    self.inf_components += 1
                else:
                    continue
                self.offending_joints.add(index)
                if self.first_frame is None:
                    self.first_frame = frame.sequence

    def _result(self) -> Outcome:
        total = self.nan_components + self.inf_components
        measurements = {
            "nan_components": self.nan_components,
            "inf_components": self.inf_components,
            "joints": sorted(self.offending_joints),
            "first_frame": self.first_frame,
        }
        if total == 0:
            return Outcome(Status.PASS, "all finite", measurements)
        return Outcome(
            Status.FAIL,
            f"{self.nan_components} NaN and {self.inf_components} Inf components on "
            f"{len(self.offending_joints)} valid joints, first at frame "
            f"{self.first_frame}",
            measurements,
        )


class ZeroPoseOnValidJoint(Check):
    name = "values.zero_pose_on_valid_joint"
    severity = Severity.HARD
    summary = "A joint marked valid does not sit at exactly the origin"

    # Largest fraction of valid joints whose position may be exactly zero.
    MAX_ZERO_RATE = 0.01

    def __init__(self) -> None:
        super().__init__()
        self.zero_valid = 0
        self.valid_total = 0
        self.offending_joints: set[int] = set()

    def _update(self, frame: Frame) -> None:
        for index, joint in frame.valid_joints():
            self.valid_total += 1
            if any(joint.position):
                continue
            self.zero_valid += 1
            self.offending_joints.add(index)

    def _result(self) -> Outcome:
        if self.valid_total == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no valid joints seen")
        rate = self.zero_valid / self.valid_total
        measurements = {
            "zero_position_valid_joints": self.zero_valid,
            "valid_joint_samples": self.valid_total,
            "rate": rate,
            "joints": sorted(self.offending_joints),
        }
        if rate <= self.MAX_ZERO_RATE:
            return Outcome(
                Status.PASS, f"{rate:.2%} of valid joints at origin", measurements
            )
        return Outcome(
            Status.FAIL,
            f"{rate:.1%} of valid joint samples report position (0,0,0) across "
            f"joints {sorted(self.offending_joints)}",
            measurements,
        )
