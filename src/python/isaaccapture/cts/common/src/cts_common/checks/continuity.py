# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from ..frames import Frame, JointPose
from .base import Check, Outcome, Severity, Status


class MaxJointVelocity(Check):
    name = "continuity.max_joint_velocity"
    severity = Severity.HARD
    summary = "No joint moves faster than a human limb can"
    min_frames = 2

    # Speed above which a joint movement counts as a violation, m/s.
    MAX_SPEED_MPS = 20.0

    # Intervals shorter than this are discarded as clock artefacts, seconds.
    MIN_INTERVAL_S = 0.004

    def __init__(self) -> None:
        super().__init__()
        self.previous_joints: tuple[JointPose, ...] | None = None
        self.previous_time: int | None = None
        self.samples = 0
        self.peak_speed = 0.0
        self.peak_joint: int | None = None
        self.peak_frame: int | None = None
        self.violations = 0
        self.discarded_intervals = 0

    def _update(self, frame: Frame) -> None:
        joints, time_ns = frame.joints, frame.sample_time_ns
        previous, previous_time = self.previous_joints, self.previous_time

        if joints is not None and time_ns is not None:
            self.previous_joints, self.previous_time = joints, time_ns

        if (
            joints is None
            or time_ns is None
            or previous is None
            or previous_time is None
        ):
            return
        dt = (time_ns - previous_time) / 1e9
        if dt <= 0:
            return
        if dt < self.MIN_INTERVAL_S:
            self.discarded_intervals += 1
            return

        for index, (before, after) in enumerate(zip(previous, joints)):
            # An invalid joint carries a zero pose, which would read as a teleport.
            if not (before.is_valid and after.is_valid):
                continue
            delta = [a - b for a, b in zip(after.position, before.position)]
            if not all(math.isfinite(component) for component in delta):
                continue
            speed = math.sqrt(sum(component * component for component in delta)) / dt
            self.samples += 1
            if speed > self.MAX_SPEED_MPS:
                self.violations += 1
            if speed > self.peak_speed:
                self.peak_speed = speed
                self.peak_joint = index
                self.peak_frame = frame.sequence

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no consecutive valid joint pairs")
        measurements = {
            "samples": self.samples,
            "peak_speed_mps": self.peak_speed,
            "peak_joint": self.peak_joint,
            "peak_frame": self.peak_frame,
            "violations": self.violations,
            "discarded_short_intervals": self.discarded_intervals,
        }
        if self.violations == 0:
            return Outcome(Status.PASS, f"peak {self.peak_speed:.1f} m/s", measurements)
        return Outcome(
            Status.FAIL,
            f"joint {self.peak_joint} reaches {self.peak_speed:.0f} m/s at frame "
            f"{self.peak_frame} ({self.violations} samples past "
            f"{self.MAX_SPEED_MPS:.0f} m/s)",
            measurements,
        )
