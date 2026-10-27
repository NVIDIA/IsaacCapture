# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class PayloadPresenceRate(Check):
    name = "coverage.payload_presence_rate"
    severity = Severity.HARD
    summary = "Most records actually carry a pose"

    # Minimum fraction of records that carry a `data` table.
    MIN_PRESENCE_RATE = 0.95

    def __init__(self) -> None:
        super().__init__()
        self.with_payload = 0

    def _update(self, frame: Frame) -> None:
        if frame.has_payload:
            self.with_payload += 1

    def _result(self) -> Outcome:
        rate = self.with_payload / self.frames_seen
        measurements = {
            "records": self.frames_seen,
            "with_payload": self.with_payload,
            "rate": rate,
        }
        if rate >= self.MIN_PRESENCE_RATE:
            return Outcome(Status.PASS, f"{rate:.1%} carry a pose", measurements)
        return Outcome(
            Status.FAIL,
            f"only {rate:.0%} of {self.frames_seen} records carry a `data` table, so "
            f"the integration is producing almost no pose data",
            measurements,
        )
