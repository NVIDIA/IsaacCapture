# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class JointsFieldPresent(Check):
    name = "schema.required_field_present.joints"
    severity = Severity.HARD
    summary = "A record carrying a pose also carries its joints"

    def __init__(self) -> None:
        super().__init__()
        self.with_payload = 0
        self.missing_joints = 0
        self.first_frame: int | None = None

    def _update(self, frame: Frame) -> None:
        if not frame.has_payload:
            return
        self.with_payload += 1
        if frame.has_joints:
            return
        self.missing_joints += 1
        if self.first_frame is None:
            self.first_frame = frame.sequence

    def _result(self) -> Outcome:
        if self.with_payload == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no record carried a pose")
        measurements = {
            "records_with_payload": self.with_payload,
            "missing_joints": self.missing_joints,
            "first_frame": self.first_frame,
        }
        # No tolerance here: the pose .fbs files state every field of the pose table is
        # present whenever the table is, so one absence is a schema violation rather than
        # a rate.
        # This is distinct from a null payload, where the data table itself is absent.
        if self.missing_joints == 0:
            return Outcome(Status.PASS, "joints present on every pose", measurements)
        return Outcome(
            Status.FAIL,
            f"{self.missing_joints} of {self.with_payload} records carry a "
            f"pose whose joints field is absent, first at frame "
            f"{self.first_frame}",
            measurements,
        )
