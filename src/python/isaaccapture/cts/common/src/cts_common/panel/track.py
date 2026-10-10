# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulates the frames a viewer draws, via ``update(frame)`` / ``track()``."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ..frames import Frame
from ..labels import StepTimeline
from .sample import Sample, Topology, Vec3

SAMPLE_CLOCK = "sample_time_local_common_clock"
LOG_CLOCK = "log_time"


@dataclass(frozen=True, slots=True)
class Track:
    samples: tuple[Sample, ...]
    times_s: tuple[float, ...]
    clock: str
    profile: Topology

    @property
    def duration_s(self) -> float:
        return self.times_s[-1] if self.times_s else 0.0

    @property
    def joint_count(self) -> int:
        return len(self.profile.joint_names)

    @property
    def min_valid_count(self) -> int:
        return min((s.valid_count for s in self.samples), default=0)

    def joints_ever_invalid(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self.joint_count)
            if any(not s.valid[index] for s in self.samples)
        )

    def index_at(self, t_s: float) -> int:
        """The last sample at or before ``t_s``, clamped to the track."""
        if not self.times_s:
            return 0
        return max(0, bisect_right(self.times_s, t_s) - 1)

    def rate_window(self, index: int, span_s: float) -> tuple[list[float], list[float]]:
        """Instantaneous rate over the trailing ``span_s`` seconds ending at ``index``.

        Samples with no rate are left out; one NaN suppresses a whole uPlot series.
        """
        if not self.samples:
            return ([], [])
        end = min(max(index, 0), len(self.samples) - 1)
        if self.times_s[end] < span_s:
            chosen = self.samples[: bisect_right(self.times_s, span_s)]
        else:
            start = bisect_left(self.times_s, self.times_s[end] - span_s)
            chosen = self.samples[start : end + 1]
        drawable = [s for s in chosen if s.rate_hz == s.rate_hz]
        return ([s.t_s for s in drawable], [s.rate_hz for s in drawable])

    def rate_extent(self) -> tuple[float, float] | None:
        """Lowest and highest instantaneous rate in the take, or None if it has none."""
        finite = [s.rate_hz for s in self.samples if s.rate_hz == s.rate_hz]
        return (min(finite), max(finite)) if finite else None

    def validity_series(self) -> tuple[list[float], list[float]]:
        """Valid joint count against time, over the whole take."""
        return (
            [s.t_s for s in self.samples],
            [float(s.valid_count) for s in self.samples],
        )


class TrackBuilder:
    """Accumulates a ``Track``, one frame at a time; ``origin_ns`` pins time zero."""

    def __init__(
        self,
        profile: Topology,
        timeline: StepTimeline | None = None,
        origin_ns: int | None = None,
    ) -> None:
        self._timeline = timeline
        self._profile = profile
        self._count = len(profile.joint_names)
        self._samples: list[Sample] = []
        self._held: list[Vec3 | None] = [None] * self._count
        self._origin_ns = origin_ns
        self._previous_ns: int | None = None
        self._clocks: set[str] = set()

    def update(self, frame: Frame) -> None:
        stamp, clock = self._stamp(frame)
        self._clocks.add(clock)
        if self._origin_ns is None:
            self._origin_ns = stamp
        interval_ms = (
            None if self._previous_ns is None else (stamp - self._previous_ns) / 1e6
        )
        self._previous_ns = stamp

        valid = [False] * self._count
        if frame.joints is not None:
            for index, joint in enumerate(frame.joints):
                if joint.is_valid:
                    valid[index] = True
                    self._held[index] = joint.position

        self._samples.append(
            Sample(
                sequence=frame.sequence,
                t_s=(stamp - self._origin_ns) / 1e9,
                positions=tuple(self._held),
                valid=tuple(valid),
                interval_ms=interval_ms,
                step=self._step(frame),
            )
        )

    @staticmethod
    def _stamp(frame: Frame) -> tuple[int, str]:
        """The clock the rate checks use, falling back to the container's log time.

        Never substitute a local clock for a frame without a sample time.
        """
        if frame.sample_time_ns is None:
            return (frame.log_time_ns, LOG_CLOCK)
        return (frame.sample_time_ns, SAMPLE_CLOCK)

    def _step(self, frame: Frame) -> str | None:
        if self._timeline is None or frame.sample_time_ns is None:
            return None
        step = self._timeline.step_at(frame.sample_time_ns)
        return step.label if step is not None else None

    def track(self) -> Track:
        return Track(
            samples=tuple(self._samples),
            times_s=tuple(s.t_s for s in self._samples),
            clock=" and ".join(sorted(self._clocks)),
            profile=self._profile,
        )
