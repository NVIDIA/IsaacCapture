# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A person drives the arm: the newest human target and the human drive.

A human target is a joint target with its sample age and the owner and stop
numbers it quotes; it says nothing about where it was computed. The sample age
is how old the controller data behind a target are. A human drive is one
stretch of human motion on the Thor.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .authority import Authority
    from .clock import Clock
    from .config import RobotAppConfig
    from .interfaces import ArmPort, AuthorityState, Checker
    from .state import JointHistory

# The speed clamp as a share of limits.human.max_joint_speed.
SPEED_MARGIN = 0.98

_NOT_BUILT = "the human drive is not built yet"


@dataclass(frozen=True)
class HumanTarget:
    """What the Thor side of the human path takes, whoever made it."""

    seq: int
    owner_number: int
    stop_number: int
    source: str  # HUMAN_SOURCE_TELEOP or HUMAN_SOURCE_SYNTHETIC
    stage: str  # one of HUMAN_STAGES
    note: str
    engaged: bool
    grip_released: bool
    q: tuple[float, ...] | None  # joint target, rad
    # Age of the controller data when the target was made. None: unknown, stale,
    # or not a finite number of 0 or more.
    sample_age_s: float | None
    # The app's monotonic clock when the target reached the Thor side.
    arrived_mono: float
    round_trip_s: float | None = None


def _finite(value: object) -> bool:
    """True for a finite real number that is not a bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _age_s(target: HumanTarget, now_mono: float) -> float | None:
    """The reported sample age plus the time since arrival; None if unknown."""
    reported = target.sample_age_s
    if not _finite(reported) or reported < 0.0:
        return None
    since = now_mono - target.arrived_mono
    if not _finite(since):
        return None
    return float(reported) + max(0.0, float(since))


class HumanInput:
    """The newest human target of this robot connection. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._newest: HumanTarget | None = None

    def put(self, target: HumanTarget) -> None:
        """Keep the last target that arrived; seq is not compared."""
        with self._lock:
            self._newest = target

    def newest(self) -> HumanTarget | None:
        with self._lock:
            return self._newest

    def sample_age_s(self, now_mono: float) -> float | None:
        """The newest target's reported age plus the time since it arrived.

        None without a target and when its age is unknown.
        """
        target = self.newest()
        return None if target is None else _age_s(target, now_mono)

    def stage(self, now_mono: float, keep_s: float) -> str:
        """The newest target's stage; "" when none arrived within keep_s."""
        target = self.newest()
        if target is None:
            return ""
        since = now_mono - target.arrived_mono
        return target.stage if since <= keep_s else ""

    def clear(self) -> None:
        """Forget the newest target."""
        with self._lock:
            self._newest = None


class DriveEnd(NamedTuple):
    """How a human drive ended. Every end is a hold with the motors on."""

    # "released", "disengaged", "stopped", "stale", "view", "refused", "late",
    # "contact" or "failed".
    status: str
    reason: str  # a "failed" reason starts with "fault: "
    sent: int  # setpoints sent
    end_q: list[float] | None


class HumanDrive:
    """Runs one human drive at a time, toward the newest human target.

    A skeleton: run() refuses every drive and sends nothing.
    """

    def __init__(
        self,
        arm: ArmPort,
        authority: Authority,
        checker: Checker,
        history: JointHistory | None,
        config: RobotAppConfig,
        clock: Clock,
        human_input: HumanInput,
        view_age: Callable[[float], float | None],
        *,
        claim_idle: Callable[[], AbstractContextManager[bool]],
        note_clamped: Callable[[], None] | None = None,
        reads_paused: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        """checker is the human checker; view_age(now) is the live view's age.

        A drive runs inside claim_idle() (MoveExecutor.claim_idle), so no move
        starts meanwhile. note_clamped is called for a step the speed clamp
        cut. The drive's start read runs inside reads_paused()
        (IdleReader.paused).
        """
        self._arm = arm
        self._authority = authority
        self._checker = checker
        self._history = history
        self._config = config
        self._clock = clock
        self._human_input = human_input
        self._view_age = view_age
        self._claim_idle = claim_idle
        self._note_clamped = note_clamped
        self._reads_paused = reads_paused or contextlib.nullcontext

    def progress(self) -> tuple[str, int, int] | None:
        """(HUMAN_MOVE_ID, setpoints sent, 0) while a drive runs, else None."""
        return None

    def run(
        self, admitted: AuthorityState, should_stop: Callable[[], str | None]
    ) -> DriveEnd:
        """One drive from its start checks to its end; it never raises a stop.

        admitted holds the owner number and the stop number the drive was
        admitted with; should_stop() gives a reason once the drive must end.
        """
        return DriveEnd("refused", _NOT_BUILT, 0, None)
