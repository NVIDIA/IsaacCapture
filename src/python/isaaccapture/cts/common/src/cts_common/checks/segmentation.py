# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label-window checks that read only the sidecar and the sample clock."""

from __future__ import annotations

from typing import Any

from ..frames import Frame
from ..labels import StepTimeline
from .base import Attribution, Check, Outcome, Severity, Status


class LabelCheck(Check):
    """A check that reads the recording through the label windows."""

    needs_timeline = True
    required = False

    def __init__(self, timeline: StepTimeline | None = None, **_: Any) -> None:
        super().__init__()
        self.timeline = timeline


class LabelWindowsWellformed(LabelCheck):
    name = "segmentation.label_windows_wellformed"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "The label windows are ordered, non-overlapping, and named once each"

    def _update(self, frame: Frame) -> None:
        return

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels to check")
        defects = self.timeline.defects()
        between_s = self.timeline.unlabelled_between_ns / 1e9
        measurements = {
            "steps": len(self.timeline.steps),
            "unlabelled_between_s": between_s,
            "defects": [f"{d.kind}: {d.detail}" for d in defects],
        }
        if not defects:
            return Outcome(
                Status.PASS,
                f"{len(self.timeline.steps)} windows in order and non-overlapping, "
                f"with {between_s:.1f} s between them unlabelled",
                measurements,
            )
        kinds = sorted({d.kind for d in defects})
        return Outcome(
            Status.FAIL,
            f"the label windows are not well formed ({', '.join(kinds)}); every "
            f"measurement downstream would describe the wrong frames",
            measurements,
        )


class LabelAlignment(LabelCheck):
    name = "segmentation.label_alignment"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "Every labelled window is backed by the frames it describes"

    # Fraction of its own length a window may stick out past the recorded span.
    MAX_UNCOVERED_FRACTION = 0.02

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.per_window: dict[int, int] = {}
        self.unlabelled = 0
        self.labelled = 0

    def _update(self, frame: Frame) -> None:
        if frame.sample_time_ns is None:
            return
        if self.first_ns is None:
            self.first_ns = frame.sample_time_ns
        self.last_ns = frame.sample_time_ns
        if self.timeline is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None:
            self.unlabelled += 1
            return
        self.labelled += 1
        self.per_window[step.index] = self.per_window.get(step.index, 0) + 1

    def _covered(self, step) -> float:
        """Fraction of a window's own span that the recording actually spans."""
        length = step.end_ns - step.start_ns
        if length <= 0:
            return 0.0
        overlap = min(step.end_ns, self.last_ns) - max(step.start_ns, self.first_ns)
        return max(0.0, min(1.0, overlap / length))

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels to align")
        span = self.timeline.span_ns
        if span is None or self.first_ns is None or self.last_ns is None:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "no timestamped frames to align labels against",
            )

        lead_in_s = (span[0] - self.first_ns) / 1e9
        tail_s = (self.last_ns - span[1]) / 1e9
        starved, clipped = [], []
        for step in self.timeline.steps:
            if not self.per_window.get(step.index):
                starved.append(step.label)
            elif self._covered(step) < 1.0 - self.MAX_UNCOVERED_FRACTION:
                clipped.append(f"{step.label} ({self._covered(step):.0%} covered)")

        measurements = {
            "lead_in_s": lead_in_s,
            "tail_s": tail_s,
            "unlabelled_fraction": self.unlabelled
            / max(1, self.labelled + self.unlabelled),
            "unlabelled_frames": self.unlabelled,
            "windows_without_frames": starved,
            "windows_partly_outside": clipped,
        }

        if starved:
            return Outcome(
                Status.FAIL,
                f"no frames fall inside {', '.join(starved)}, so the labels do not "
                f"belong to this recording",
                measurements,
            )
        if clipped:
            return Outcome(
                Status.FAIL,
                f"the recording ran out part way through {', '.join(clipped)}, so "
                f"those windows would be measured from a fragment",
                measurements,
            )
        return Outcome(
            Status.PASS,
            f"all {len(self.timeline.steps)} windows are backed by frames, with "
            f"{lead_in_s:.1f} s of lead-in and {tail_s:.1f} s of tail unlabelled",
            measurements,
        )
