# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One ``Track`` per hand, on one time axis, built while the checks run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from cts_common.panel.track import Track, TrackBuilder

from ..frames import Frame, FrameSource, SourceMetadata
from ..labels import StepTimeline
from ..profile import HAND, WRIST

SIDES = ("left", "right")

Vec3 = tuple[float, float, float]


class HandTee:
    """A ``FrameSource`` pass-through that splits frames into the two hands' tracks.

    Both tracks share one time zero, the recording's first frame.
    """

    def __init__(
        self, source: FrameSource, timeline: StepTimeline | None = None
    ) -> None:
        self._source = source
        self._timeline = timeline
        self._builders: dict[str, TrackBuilder] = {}
        self._origin_ns: int | None = None

    @property
    def metadata(self) -> SourceMetadata:
        return self._source.metadata

    @property
    def path(self) -> Path:
        return self._source.path  # type: ignore[attr-defined]

    @property
    def topics(self) -> tuple[str, ...]:
        return getattr(self._source, "topics", ())

    def __iter__(self) -> Iterator[Frame]:
        for frame in self._source:
            if frame.side in SIDES:
                if self._origin_ns is None:
                    self._origin_ns = (
                        frame.sample_time_ns
                        if frame.sample_time_ns is not None
                        else frame.log_time_ns
                    )
                builder = self._builders.get(frame.side)
                if builder is None:
                    builder = TrackBuilder(HAND, self._timeline, self._origin_ns)
                    self._builders[frame.side] = builder
                builder.update(frame)
            yield frame

    def tracks(self) -> dict[str, Track]:
        return {side: builder.track() for side, builder in self._builders.items()}


def centre(tracks: dict[str, Track]) -> Vec3 | None:
    """Mean wrist position over both hands' first valid wrists, for the camera."""
    wrists = []
    for track in tracks.values():
        found = next(
            (s.positions[WRIST] for s in track.samples if s.valid[WRIST]), None
        )
        if found is not None:
            wrists.append(found)
    if not wrists:
        return None
    return tuple(sum(w[axis] for w in wrists) / len(wrists) for axis in range(3))  # type: ignore[return-value]
