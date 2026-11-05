# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accumulator contract: checks are incremental, via ``update(frame)`` / ``result()``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Mapping

from ..frames import Frame


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class Severity(StrEnum):
    HARD = "hard"
    SOFT = "soft"
    ADVISORY = "advisory"


class Attribution(StrEnum):
    DEVICE = "device"
    PERFORMANCE = "performance"


@dataclass(frozen=True, slots=True)
class Outcome:
    status: Status
    detail: str = ""
    measurements: Mapping[str, Any] = field(default_factory=dict)

    # Per-result override of the check's ``attribution``.
    attribution: "Attribution | None" = None


class Check(ABC):
    """Subclasses set the class-level fields and implement ``_update`` / ``_result``.

    ``name`` must match the fixture index's ``expected_failing_check`` for its defect.
    """

    name: ClassVar[str]
    # Group a result is reported under; set by each schema package.
    group: ClassVar[str]
    # HARD and SOFT failures count toward the verdict; ADVISORY failures are reported only.
    severity: ClassVar[Severity] = Severity.HARD
    # Who a failure is blamed on: DEVICE gives a fail verdict, PERFORMANCE a retake.
    attribution: ClassVar[Attribution] = Attribution.DEVICE
    summary: ClassVar[str] = ""
    min_frames: ClassVar[int] = 1

    # False when the answer depends on what the subject did; an unanswered required check
    # makes the verdict insufficient_data.
    required: ClassVar[bool] = True

    # False for a check that only reports a measurement; it returns PASS to mean "measured".
    judged: ClassVar[bool] = True

    # Names of checks whose failure makes this one's measurement meaningless; the result
    # becomes insufficient_data, never a pass.
    depends_on: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        self.frames_seen = 0

    def update(self, frame: Frame) -> None:
        self.frames_seen += 1
        self._update(frame)

    def result(self) -> Outcome:
        if self.frames_seen < self.min_frames:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"{self.frames_seen} frames, needs at least {self.min_frames}",
            )
        return self._result()

    @abstractmethod
    def _update(self, frame: Frame) -> None: ...

    @abstractmethod
    def _result(self) -> Outcome: ...
