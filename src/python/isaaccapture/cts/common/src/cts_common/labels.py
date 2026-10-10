# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-step labels and the timeline the window checks read them through.

Labels live in a sidecar JSON beside the recording, timed in
``sample_time_local_common_clock``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

SIDECAR_SUFFIX = ".labels.json"


@dataclass(frozen=True, slots=True)
class Step:
    index: int
    label: str
    start_ns: int
    end_ns: int
    is_still_window: bool

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def contains(self, sample_ns: int) -> bool:
        return self.start_ns <= sample_ns < self.end_ns


@dataclass(frozen=True, slots=True)
class Defect:
    kind: str
    detail: str


@dataclass(frozen=True)
class StepTimeline:
    """Subclassed per schema to name the script's still windows."""

    # What ``is_still_window`` defaults to for a sidecar row that does not say.
    still_labels: ClassVar[frozenset[str]] = frozenset()

    steps: tuple[Step, ...]
    nominal_rate_hz: float | None = None
    source: str = ""

    @classmethod
    def load(cls, path: str | Path) -> StepTimeline:
        payload = json.loads(Path(path).read_text())
        steps = tuple(
            Step(
                index=int(raw["index"]),
                label=str(raw["label"]),
                start_ns=int(raw["start_ns"]),
                end_ns=int(raw["end_ns"]),
                is_still_window=bool(
                    raw.get("is_still_window", raw["label"] in cls.still_labels)
                ),
            )
            for raw in payload["steps"]
        )
        return cls(
            steps=steps,
            nominal_rate_hz=payload.get("nominal_rate_hz"),
            source=str(path),
        )

    @classmethod
    def beside(cls, recording: str | Path) -> StepTimeline | None:
        """Loads ``<recording>.labels.json``, or returns None when there is none."""
        path = Path(recording)
        sidecar = path.with_name(path.stem + SIDECAR_SUFFIX)
        return cls.load(sidecar) if sidecar.is_file() else None

    def step_at(self, sample_ns: int | None) -> Step | None:
        if sample_ns is None:
            return None
        for step in self.steps:
            if step.contains(sample_ns):
                return step
        return None

    def named(self, label: str) -> tuple[Step, ...]:
        return tuple(step for step in self.steps if step.label == label)

    def one(self, label: str) -> Step | None:
        found = self.named(label)
        return found[0] if len(found) == 1 else None

    @property
    def span_ns(self) -> tuple[int, int] | None:
        if not self.steps:
            return None
        return (
            min(step.start_ns for step in self.steps),
            max(step.end_ns for step in self.steps),
        )

    @property
    def unlabelled_between_ns(self) -> int:
        """Nanoseconds between consecutive windows that no window claims."""
        ordered = sorted(self.steps, key=lambda step: step.start_ns)
        return sum(
            max(0, later.start_ns - earlier.end_ns)
            for earlier, later in zip(ordered, ordered[1:])
        )

    def defects(self) -> tuple[Defect, ...]:
        """Structural faults in the labels: empty, out-of-order, overlapping, duplicate."""
        found: list[Defect] = []
        if not self.steps:
            return (Defect("empty", "the sidecar declares no steps"),)

        for step in self.steps:
            if step.end_ns <= step.start_ns:
                found.append(
                    Defect(
                        "empty_window",
                        f"step {step.index} {step.label!r} ends at or before it starts",
                    )
                )

        ordered = sorted(self.steps, key=lambda step: step.start_ns)
        if [step.index for step in ordered] != [step.index for step in self.steps]:
            found.append(
                Defect("out_of_order", "step indices do not follow the start times")
            )
        for earlier, later in zip(ordered, ordered[1:]):
            if later.start_ns < earlier.end_ns:
                found.append(
                    Defect(
                        "overlap",
                        f"{earlier.label!r} and {later.label!r} overlap by "
                        f"{(earlier.end_ns - later.start_ns) / 1e9:.2f} s",
                    )
                )

        duplicated = sorted(
            {step.label for step in self.steps if len(self.named(step.label)) > 1}
        )
        if duplicated:
            found.append(
                Defect("duplicate_label", f"more than one window named {duplicated}")
            )
        return tuple(found)
