# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import statistics
from typing import ClassVar

from ..frames import Frame
from .base import Check, Outcome, Severity, Status


class _IntervalCheck(Check):
    """Collects intervals between sample times."""

    def __init__(self) -> None:
        super().__init__()
        self.previous: int | None = None
        self.intervals_ns: list[int] = []

    def _update(self, frame: Frame) -> None:
        if frame.sample_time_ns is None:
            return
        if self.previous is not None:
            self.intervals_ns.append(frame.sample_time_ns - self.previous)
        self.previous = frame.sample_time_ns

    def _forward_intervals(self) -> list[int]:
        return [d for d in self.intervals_ns if d > 0]


class IntervalRegularity(_IntervalCheck):
    name = "rate.interval_regularity"
    severity = Severity.SOFT
    summary = "Frame intervals are regular around the nominal period"
    min_frames = 30

    # Largest median absolute deviation of the interval, as a fraction of the median.
    MAX_RELATIVE_MAD = 0.2

    def _result(self) -> Outcome:
        intervals = self._forward_intervals()
        if len(intervals) < 10:
            return Outcome(
                Status.INSUFFICIENT_DATA, f"{len(intervals)} forward intervals"
            )
        median = statistics.median(intervals)
        if median <= 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no positive median interval")
        mad = statistics.median([abs(d - median) for d in intervals])
        relative = mad / median
        measurements = {
            "median_interval_ms": median / 1e6,
            "nominal_rate_hz": 1e9 / median,
            "mad_ms": mad / 1e6,
            "relative_mad": relative,
            "min_interval_ms": min(intervals) / 1e6,
            "max_interval_ms": max(intervals) / 1e6,
        }
        if relative <= self.MAX_RELATIVE_MAD:
            return Outcome(
                Status.PASS,
                f"{1e9 / median:.1f} Hz, jitter {mad / 1e6:.2f} ms",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"interval jitter {mad / 1e6:.1f} ms is {relative:.0%} of the "
            f"{median / 1e6:.1f} ms median period",
            measurements,
        )


class MinimumRate(_IntervalCheck):
    name = "rate.minimum_rate"
    severity = Severity.HARD
    summary = "The median frame rate is not below the schema's minimum"
    min_frames = 30

    # Lowest median frame rate accepted, hertz; None leaves the check unanswered.
    MIN_HZ: ClassVar[float | None] = None

    def _result(self) -> Outcome:
        intervals = self._forward_intervals()
        if len(intervals) < 10:
            return Outcome(
                Status.INSUFFICIENT_DATA, f"{len(intervals)} forward intervals"
            )
        median = statistics.median(intervals)
        if median <= 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no positive median interval")
        hz = 1e9 / median
        measurements = {"median_rate_hz": hz, "min_rate_hz": self.MIN_HZ}
        if self.MIN_HZ is None:
            return Outcome(
                Status.INSUFFICIENT_DATA, "no minimum rate set", measurements
            )
        if hz >= self.MIN_HZ:
            return Outcome(Status.PASS, f"{hz:.1f} Hz", measurements)
        return Outcome(
            Status.FAIL,
            f"{hz:.1f} Hz is below the {self.MIN_HZ:g} Hz minimum",
            measurements,
        )


class FrameGaps(_IntervalCheck):
    name = "rate.frame_gaps"
    severity = Severity.SOFT
    summary = "The stream has no dropped blocks of frames"
    min_frames = 10

    # An interval longer than this multiple of the median counts as a gap.
    GAP_MULTIPLE = 5.0

    def _result(self) -> Outcome:
        intervals = self._forward_intervals()
        if len(intervals) < 5:
            return Outcome(
                Status.INSUFFICIENT_DATA, f"{len(intervals)} forward intervals"
            )
        median = statistics.median(intervals)
        if median <= 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no positive median interval")
        limit = median * self.GAP_MULTIPLE
        gaps = [d for d in intervals if d > limit]
        measurements = {
            "median_interval_ms": median / 1e6,
            "gap_threshold_ms": limit / 1e6,
            "gaps": len(gaps),
            "gap_lengths_ms": [round(d / 1e6, 1) for d in sorted(gaps, reverse=True)],
            "lost_seconds": sum(gaps) / 1e9,
        }
        if not gaps:
            return Outcome(Status.PASS, "no gaps", measurements)
        return Outcome(
            Status.FAIL,
            f"{len(gaps)} gaps beyond {limit / 1e6:.0f} ms "
            f"({', '.join(f'{d / 1e6:.0f} ms' for d in sorted(gaps, reverse=True)[:6])})",
            measurements,
        )
