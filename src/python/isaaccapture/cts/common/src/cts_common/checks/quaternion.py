# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class UnitNormOnValidJoints(Check):
    name = "quaternion.unit_norm_on_valid_joints"
    severity = Severity.HARD
    summary = "Orientations on valid joints are unit quaternions"

    # Allowed deviation of the quaternion norm from 1; invalid joints are skipped.
    TOLERANCE = 1e-3

    def __init__(self) -> None:
        super().__init__()
        self.samples = 0
        self.violations = 0
        self.offending_joints: set[int] = set()
        self.worst_norm: float | None = None

    def _update(self, frame: Frame) -> None:
        for index, joint in frame.valid_joints():
            norm = math.sqrt(sum(c * c for c in joint.orientation))
            if not math.isfinite(norm):
                continue
            self.samples += 1
            error = abs(norm - 1.0)
            if error <= self.TOLERANCE:
                continue
            self.violations += 1
            self.offending_joints.add(index)
            if self.worst_norm is None or error > abs(self.worst_norm - 1.0):
                self.worst_norm = norm

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no valid orientations seen")
        measurements = {
            "samples": self.samples,
            "violations": self.violations,
            "joints": sorted(self.offending_joints),
            "worst_norm": self.worst_norm,
        }
        if self.violations == 0:
            return Outcome(Status.PASS, "all unit norm", measurements)
        return Outcome(
            Status.FAIL,
            f"{self.violations} of {self.samples} valid orientations off the unit "
            f"sphere on joints {sorted(self.offending_joints)}, worst norm "
            f"{self.worst_norm:.3f}",
            measurements,
        )
