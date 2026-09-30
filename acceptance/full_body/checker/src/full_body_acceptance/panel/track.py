# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frames a viewer draws, accumulated the way a check accumulates.

``update(frame)`` / ``track()`` over the ``FrameSource`` protocol and nothing else, so
a live source will feed this without a line changing here. Everything in this module is
pure Python: the panel's renderer is optional, its arithmetic is not.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from acceptance_common.panel.sample import Sample, Vec3

from ..frames import NUM_JOINTS, Frame, FrameSource, SourceMetadata
from ..labels import StepTimeline
from ..profile import FULL_BODY, SkeletonProfile

SAMPLE_CLOCK = "sample_time_local_common_clock"
LOG_CLOCK = "log_time"


@dataclass(frozen=True, slots=True)
class Track:
    samples: tuple[Sample, ...]
    times_s: tuple[float, ...]
    clock: str
    profile: SkeletonProfile

    @property
    def duration_s(self) -> float:
        return self.times_s[-1] if self.times_s else 0.0

    @property
    def min_valid_count(self) -> int:
        return min((s.valid_count for s in self.samples), default=0)

    def joints_ever_invalid(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(NUM_JOINTS)
            if any(not s.valid[index] for s in self.samples)
        )

    def index_at(self, t_s: float) -> int:
        """The last sample at or before ``t_s``, clamped to the track."""
        if not self.times_s:
            return 0
        return max(0, bisect_right(self.times_s, t_s) - 1)

    def rate_window(self, index: int, span_s: float) -> tuple[list[float], list[float]]:
        """Instantaneous rate over the ``span_s`` seconds around ``index``.

        Fills, then scrolls, the way a monitor does: the window is the span ending at
        the playhead once the take has run that long, and the opening span before that,
        so a panel sitting at frame zero shows a curve rather than an empty axis.

        Trailing rather than centred, because a spike is easier to catch arriving at
        the right-hand edge than drifting through the middle.

        Samples with no rate are left out rather than carried as NaN: one NaN anywhere
        in a uPlot series suppresses the entire curve (measured, viser 1.1). The only
        interior cause is a timestamp that did not move forward, and
        ``timestamps.monotonic`` is a hard check that says so in the list.
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
    """Accumulates a ``Track``, one frame at a time."""

    def __init__(
        self,
        timeline: StepTimeline | None = None,
        profile: SkeletonProfile = FULL_BODY,
    ) -> None:
        self._timeline = timeline
        self._profile = profile
        self._samples: list[Sample] = []
        self._held: list[Vec3 | None] = [None] * NUM_JOINTS
        self._origin_ns: int | None = None
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

        valid = [False] * NUM_JOINTS
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

        A recording carries both, so naming which one was used is enough. A live source
        must not reach for a local clock here: a sample the device could not stamp has
        no time, and inventing one turns an unanswerable question into a wrong answer.
        """
        if frame.sample_time_ns is None:
            return (frame.log_time_ns, LOG_CLOCK)
        return (frame.sample_time_ns, SAMPLE_CLOCK)

    def _step(self, frame: Frame) -> str | None:
        # Windows are timed in the sample clock, so a frame without one is unlabelled.
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


class TeeSource:
    """A ``FrameSource`` pass-through that builds the track as the checks consume it.

    One pass over the recording, because a live source cannot be iterated twice and
    decoding 5757 frames costs about two seconds even when it can.
    """

    def __init__(
        self, source: FrameSource, timeline: StepTimeline | None = None
    ) -> None:
        self._source = source
        self._builder = TrackBuilder(timeline)

    @property
    def metadata(self) -> SourceMetadata:
        return self._source.metadata

    @property
    def path(self) -> Path:
        # ``report.run`` reads this through ``getattr(source, "path", None)``, so
        # letting the wrapped source's AttributeError through is how a source with no
        # path stays pathless instead of acquiring the string "None".
        return self._source.path  # type: ignore[attr-defined]

    def __iter__(self) -> Iterator[Frame]:
        for frame in self._source:
            self._builder.update(frame)
            yield frame

    def track(self) -> Track:
        return self._builder.track()
