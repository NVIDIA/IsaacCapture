# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accumulator contract.

Checks are incremental (``update(frame)`` / ``result()``) so one implementation serves a
live session, an MCAP file and a replay session.

Measurement and policy are separate: an accumulator reports a status and its
measurements, while ``severity`` is declared on the class and ``attribution`` defaults there
consumed by verdict aggregation. A threshold can then be filled in without touching a
measurement.
"""

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

    # Set only when the recording itself says who is at fault, overriding the class's
    # default. A check whose evidence can separate a device fault from a sloppy
    # performance should say which it saw rather than declare one for both.
    attribution: "Attribution | None" = None


class Check(ABC):
    """Subclasses set the class-level fields and implement ``update`` / ``_result``.

    ``name`` must match the ``expected_failing_check`` string the fixture index uses for
    the defect this check is meant to catch.
    """

    name: ClassVar[str]
    # The group a result is reported under. Set by each schema package, never here:
    # the shared checks are grouped differently by each schema that runs them.
    gate: ClassVar[str]
    severity: ClassVar[Severity] = Severity.HARD
    attribution: ClassVar[Attribution] = Attribution.DEVICE
    summary: ClassVar[str] = ""
    min_frames: ClassVar[int] = 1

    # False when the question depends on what the subject did, not on the recording
    # being adequate. A held T-pose cannot reveal a frame mismatch however long it runs,
    # so such a check returning "insufficient data" must not drag the whole recording
    # into limbo; an unanswered required check must.
    required: ClassVar[bool] = True

    # False for a check that reports a number and never judges it, which the graded posture
    # measurements deliberately do: their thresholds are meant to come from real
    # subjects, not from the fixtures used to prove the measurement correct. Such a
    # check returns PASS because there is no other way to say "measured", so without
    # this flag a reader cannot tell a value that was approved from one nobody judged.
    judged: ClassVar[bool] = True

    # Names of checks whose failure makes this one's measurement meaningless. A device
    # measurement read through motion windows cannot be trusted once those windows are
    # known to be wrong: the arm-raise window of a session performed out of order holds
    # a leg raise, and reporting that as a clipped arm is the device-versus-performance
    # confusion this whole process exists to prevent. Suppression turns such a result
    # into "cannot conclude", never into a pass.
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
