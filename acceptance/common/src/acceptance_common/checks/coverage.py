# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class PayloadPresenceRate(Check):
    name = "coverage.payload_presence_rate"
    severity = Severity.HARD
    summary = "Most records actually carry a pose"

    # A record with a timestamp and no `data` table is legal on its own -- pack_record
    # emits exactly that for an inactive device -- so this is a rate, not a per-record
    # failure. Threshold is a placeholder until real recordings exist.
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


class ValidityTrend(Check):
    name = "coverage.validity_trend"
    severity = Severity.SOFT
    summary = "Per-joint validity coverage does not decay over the session"

    # A trend, not a dropout. A limb that goes invalid for a second and comes back is a
    # transient occlusion and must pass, so the comparison is head window against tail
    # window rather than anything about the middle of the run.
    WINDOW = 120
    MAX_COVERAGE_DROP = 0.25
    min_frames = 2 * WINDOW

    def __init__(self) -> None:
        super().__init__()
        self.head: list[float] = []
        self.tail: deque[float] = deque(maxlen=self.WINDOW)

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        coverage = sum(1 for joint in frame.joints if joint.is_valid) / len(
            frame.joints
        )
        if len(self.head) < self.WINDOW:
            self.head.append(coverage)
        self.tail.append(coverage)

    def _result(self) -> Outcome:
        if len(self.head) < self.WINDOW or len(self.tail) < self.WINDOW:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"needs {2 * self.WINDOW} frames carrying joints",
            )
        start = sum(self.head) / len(self.head)
        end = sum(self.tail) / len(self.tail)
        drop = start - end
        measurements = {
            "start_coverage": start,
            "end_coverage": end,
            "drop": drop,
            "window_frames": self.WINDOW,
        }
        if drop <= self.MAX_COVERAGE_DROP:
            return Outcome(
                Status.PASS,
                f"coverage {start:.0%} -> {end:.0%}",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"validity coverage decays from {start:.0%} to {end:.0%} over the "
            f"session, so trackers are dropping out progressively",
            measurements,
        )
