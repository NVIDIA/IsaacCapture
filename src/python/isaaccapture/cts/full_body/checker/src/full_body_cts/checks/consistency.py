# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..frames import Frame
from cts_common.checks.base import Check, Outcome, Severity, Status


class AllJointPosesTracked(Check):
    name = "consistency.all_joint_poses_tracked"
    group = "signal"
    severity = Severity.ADVISORY
    summary = "The quality flag agrees with the per-joint validity flags"

    # Maximum fraction of records where the flag and the per-joint flags may disagree.
    MAX_DISAGREEMENT_RATE = 0.01

    def __init__(self) -> None:
        super().__init__()
        self.comparable = 0
        self.flag_true_but_joints_invalid = 0
        self.flag_false_but_all_valid = 0

    def _update(self, frame: Frame) -> None:
        if frame.all_joint_poses_tracked is None or frame.joints is None:
            return
        self.comparable += 1
        all_valid = all(joint.is_valid for joint in frame.joints)
        if frame.all_joint_poses_tracked and not all_valid:
            self.flag_true_but_joints_invalid += 1
        elif not frame.all_joint_poses_tracked and all_valid:
            self.flag_false_but_all_valid += 1

    def _result(self) -> Outcome:
        if self.comparable == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no comparable records")
        disagreements = (
            self.flag_true_but_joints_invalid + self.flag_false_but_all_valid
        )
        rate = disagreements / self.comparable
        measurements = {
            "comparable_records": self.comparable,
            "flag_true_but_joints_invalid": self.flag_true_but_joints_invalid,
            "flag_false_but_all_valid": self.flag_false_but_all_valid,
            "rate": rate,
        }
        if rate <= self.MAX_DISAGREEMENT_RATE:
            return Outcome(
                Status.PASS, "flag agrees with per-joint flags", measurements
            )
        return Outcome(
            Status.FAIL,
            f"all_joint_poses_tracked disagrees with the per-joint flags on "
            f"{rate:.0%} of records ({self.flag_true_but_joints_invalid} claim all "
            f"tracked while a joint is invalid)",
            measurements,
        )
