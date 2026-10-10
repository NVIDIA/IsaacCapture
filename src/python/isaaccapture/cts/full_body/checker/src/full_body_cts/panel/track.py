# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The full-body ``Track``, and the pass-through that builds it while the checks run."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from cts_common.panel.track import LOG_CLOCK, SAMPLE_CLOCK, Track
from cts_common.panel.track import TrackBuilder as _TrackBuilder

from ..frames import Frame, FrameSource, SourceMetadata
from ..labels import StepTimeline
from ..profile import FULL_BODY, SkeletonProfile

__all__ = ["LOG_CLOCK", "SAMPLE_CLOCK", "TeeSource", "Track", "TrackBuilder"]


class TrackBuilder(_TrackBuilder):
    def __init__(
        self,
        timeline: StepTimeline | None = None,
        profile: SkeletonProfile = FULL_BODY,
    ) -> None:
        super().__init__(profile, timeline)


class TeeSource:
    """A ``FrameSource`` pass-through that builds the track as the checks consume it."""

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
        # Let the wrapped source's AttributeError through; report.run probes this with getattr.
        return self._source.path  # type: ignore[attr-defined]

    def __iter__(self) -> Iterator[Frame]:
        for frame in self._source:
            self._builder.update(frame)
            yield frame

    def track(self) -> Track:
        return self._builder.track()
