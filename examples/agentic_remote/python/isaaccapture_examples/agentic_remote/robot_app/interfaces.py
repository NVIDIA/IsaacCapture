# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types the robot app modules share: enums, errors, readings, outcomes and ports."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .state import JointHistory


class Owner(StrEnum):
    """Control owner: who may move the arm right now."""

    NOBODY = "nobody"
    AGENT = "agent"
    HUMAN = "human"
    RECOVERY = "recovery"


class StopLevel(StrEnum):
    NONE = "none"
    STOP = "stop"
    FAULT = "fault"


class JawLoad(StrEnum):
    """What the jaws hold, as far as measurements can prove."""

    EMPTY = "empty"
    LOADED = "loaded"
    UNKNOWN = "unknown"


class MotionState(StrEnum):
    IDLE = "idle"
    MOVING = "moving"
    HOLD = "hold"
    REST_HOLD = "rest_hold"
    REST_UNPOWERED = "rest_unpowered"


class Refusal(Exception):
    """A check refused a setpoint, a pose or a command."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ContactStop(Refusal):
    """Motor feedback shows contact."""


class FeedbackLost(Refusal):
    """A joint gave no good reading for too long; a fault-stop must follow."""


class ReadFailed(Exception):
    """A joint or jaw read failed."""


class ArmFault(Exception):
    """The driver reported a motor or gripper fault."""


@dataclass(frozen=True)
class JointReading:
    """One measured joint value with the app-clock times its read began and ended."""

    joint: int
    value: float
    t_start: float
    t_end: float
    valid: bool = True


@dataclass(frozen=True)
class GripperOutcome:
    # "reached", "stalled", "timeout", "cancelled" or "refused" from the arm; the
    # gripper worker adds "failed" when the driver raised.
    status: str
    opening: float | None
    detail: str = ""


@dataclass(frozen=True)
class AuthorityState:
    """One consistent reading of the authority's values."""

    session_id: str
    owner: Owner
    owner_number: int
    stop_number: int
    stop_level: StopLevel = StopLevel.NONE
    fault_reason: str = ""


@dataclass(frozen=True)
class MoveOutcome:
    # "completed", "refused", "stopped", "contact", "late", "not_settled" or "failed".
    # "failed" is a driver fault or lost joint feedback: its reason starts with
    # "fault: " and a fault-stop must follow.
    status: str
    reason: str = ""
    index: int = 0  # how many setpoints were sent
    end_q: list[float] | None = None  # measured end pose, None if it was not read


@dataclass(frozen=True)
class RestOutcome:
    status: str
    reason: str = ""


@runtime_checkable
class ArmPort(Protocol):
    """The arm and gripper driver as the robot app uses it."""

    n_joints: int
    joint_names: list[str]

    def connect(self) -> None:
        """Connect to the motors. Called once at start and never again."""

    def torque_on(self) -> bool:
        """Whether the motors are powered."""

    def send_target(self, q: Sequence[float]) -> None:
        """Send one already-checked setpoint without waiting."""

    def last_target(self) -> tuple[list[float], float] | None:
        """The last setpoint sent and its send time on the app clock."""

    def read_joint(self, joint: int) -> JointReading:
        """Read one joint; raises ReadFailed."""

    def read_all(self) -> list[JointReading]:
        """Read every joint afresh, never from a cache; raises ReadFailed."""

    def hold_measured(self) -> None:
        """Command a hold at a freshly measured pose."""

    def release_torque(self) -> None:
        """Switch the motors off. Only the guarded release in rest.py may call this."""

    def release_confirmed(self) -> bool:
        """Whether the driver knows that release_torque() switched the motors off.

        False before a release, and after one that the driver only sent.
        """

    def faults(self) -> list[str]:
        """Active motor or gripper faults; empty when there are none."""

    def temperatures(self) -> list[float] | None:
        """Motor temperatures, None when the driver cannot read them."""

    def gripper_referenced(self) -> bool:
        """Whether the jaw reference taken at start is still valid."""

    def gripper_reference(self, cancel: threading.Event | None = None) -> None:
        """Reference the jaws: they close slowly to their stop; raises ArmFault.

        Once cancel is set it ends early without raising: the jaws stay
        unreferenced and are held where they are.
        """

    def gripper_read(self) -> float | None:
        """Jaw opening, 0 closed to 1 open; None if unknown or the read failed."""

    def gripper_last_target(self) -> float | None:
        """The jaw target in force, set by the reference and by each command."""

    def gripper_command(
        self, opening: float, max_effort: float, mode: str, cancel: threading.Event
    ) -> GripperOutcome:
        """Run one jaw command to its end; ends early when cancel is set.

        A close that ends "stalled" or "timeout" short of its target is left at
        the light holding stiffness, in either mode. A cancelled opening stops
        within a small step, with the jaw target where the jaws are.
        """

    def gripper_light_hold(self) -> None:
        """Re-send the current jaw target at the light holding stiffness."""


@runtime_checkable
class ContactWatch(Protocol):
    """Joint reads and the contact check for one running move."""

    def after_send(
        self, index: int, q: Sequence[float], deadline_mono: float
    ) -> list[JointReading]:
        """Do this step's joint reads.

        Raises ContactStop on contact and FeedbackLost when a joint has had no
        good reading for too long.
        """

    def hold(self) -> None:
        """After a contact: command a hold at a freshly measured pose.

        Raises ReadFailed, and sends nothing, when the pose cannot be read or
        the watch does not trust the reading.
        """

    def finish(self) -> None:
        """End the watch when the move ends."""


@runtime_checkable
class Checker(Protocol):
    """Accepts or refuses each setpoint just before it is sent; one per owner."""

    name: str  # "agent", "human" or "recovery"
    version: str

    def feed_watchdog(self, t_mono: float | None = None) -> None:
        """Record fresh evidence taken at t_mono, or now when None."""

    def halt_count(self) -> int:
        """How many halts were raised so far."""

    def halt(self, reason: str) -> int:
        """Stop a running move at its next setpoint; returns the new halt count."""

    def latch(self, reason: str) -> None:
        """Fault-stop: refuse everything until reset_latch."""

    def reset_latch(self) -> None:
        """Clear the fault-stop latch."""

    def latched(self) -> bool:
        """Whether the fault-stop latch is set."""

    def begin_motion(self, halt_count: int) -> None:
        """Start a move; raises Refusal if latched or if halt_count is out of date."""

    def end_motion(self) -> None:
        """End the move begun by begin_motion."""

    def check_setpoint(
        self, q_prev: Sequence[float], q_next: Sequence[float], dt: float
    ) -> None:
        """Check one step from q_prev to q_next taken in dt seconds; raises Refusal."""

    def check_pose(self, q: Sequence[float]) -> None:
        """Check a pose on its own; raises Refusal."""

    def check_path(
        self, q_start: Sequence[float], setpoints: Sequence[Sequence[float]], dt: float
    ) -> None:
        """Check a whole move before its first setpoint; raises Refusal."""

    def check_grasp_region(self, region: GraspRegion) -> None:
        """Check a grasp region against the Thor's bounds for one; raises Refusal."""

    def set_grasp_region(self, region: GraspRegion | None) -> None:
        """Relax the table and obstacle rules inside region; None restores them.

        Raises Refusal for a region check_grasp_region refuses.
        """

    def note_contact(self, q: Sequence[float]) -> None:
        """Remember a contact spot at this pose; does nothing if unsupported."""

    def clear_contact_spots(self) -> None:
        """Forget the contact spots collected so far."""

    def tip_blocked(self, q: Sequence[float], region: GraspRegion | None) -> str | None:
        """Why the gripper tip at q may not start a return to rest, else None.

        Blocked means below the table clearance or inside region.
        """

    def contact_watch(
        self, arm: ArmPort, q_start: Sequence[float], history: JointHistory
    ) -> ContactWatch:
        """A watch for one move that starts at q_start and adds its reads to history."""


@dataclass
class CheckerSet:
    """The three checker instances, one per control owner that can move the arm."""

    agent: Checker
    human: Checker
    recovery: Checker

    def for_owner(self, owner: Owner) -> Checker:
        """The checker of a control owner; raises ValueError for nobody."""
        owner = Owner(owner)
        if owner is Owner.NOBODY:
            raise ValueError("control owner nobody has no checker")
        return {
            Owner.AGENT: self.agent,
            Owner.HUMAN: self.human,
            Owner.RECOVERY: self.recovery,
        }[owner]

    def all(self) -> tuple[Checker, Checker, Checker]:
        return (self.agent, self.human, self.recovery)
