# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the operator console prints, as pure functions of the link messages."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

from ..protocol.confirm import PATH_IS_CLEAR
from ..robot_app.text import MAX_OUT_CHARS, MAX_WORD_CHARS, printable

if TYPE_CHECKING:
    from ..protocol.local_messages import LocalStatus
    from ..protocol.messages import Event, RobotState, Welcome

# EVENT kinds the console treats in a way of its own: robot_app.waits.BELL,
# robot_app.app.EVENT_REFUSED and EVENT_HELP_NEEDED.
KIND_BELL = "bell"
KIND_REFUSED = "refused"
KIND_HELP_NEEDED = "help_needed"

NOT_ATTACHED = "not attached to the robot link service"
NO_STATE = "no ROBOT_STATE"
# WELCOME.driver "" declares nothing; every reader treats that as the real arm.
NOT_DECLARED = "not declared (treated as real)"
_SEP = " | "


def _text(value: object, limit: int = MAX_OUT_CHARS) -> str:
    return printable(value, limit)


def _number(value: Any, digits: int) -> str | None:
    """A number of the link as text; None when it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    return f"{value:.{digits}f}"


def _seconds(value: Any, digits: int, unknown: str = "?") -> str:
    number = _number(value, digits)
    return unknown if number is None else f"{number} s"


def driver_text(driver: str) -> str:
    return _text(driver, MAX_WORD_CHARS) if driver else NOT_DECLARED


def welcome_line(welcome: Welcome) -> str:
    """The line printed after a WELCOME: which robot, session and driver."""
    calibration = _text(welcome.camera_calibration_id) or "none"
    return (
        f"WELCOME: robot {_text(welcome.robot_id)}, profile "
        f"{_text(welcome.profile_id)}, session {_text(welcome.session_id)}, "
        f"driver {driver_text(welcome.driver)}, camera calibration {calibration}"
    )


def _connection_parts(status: LocalStatus | None) -> list[str]:
    if status is None:
        return [NOT_ATTACHED]
    if status.robot_connected:
        parts = ["robot connection up"]
    else:
        why = _text(status.close_reason) or "not connected"
        parts = [f"robot connection lost ({why})"]
    clients = ", ".join(
        f"{count} {_text(role, MAX_WORD_CHARS)}"
        for role, count in sorted(status.roles.items())
    )
    parts.append(f"clients {clients or 'none'}")
    if status.stop_unconfirmed:
        parts.append("a STOP is not confirmed yet")
    return parts


def _permit_part(health: dict[str, Any]) -> str:
    permit = health.get("permit")
    if not isinstance(permit, dict):
        return "permit none"
    through = _number(permit.get("through_index"), 0) or "?"
    part = f"permit {through}, evidence {_seconds(permit.get('evidence_age_s'), 2)}"
    left = permit.get("expires_in_s")
    if _number(left, 2) is not None and left < 0:
        part += ", expired"
    return part


def status_line(
    welcome: Welcome | None,
    state: RobotState | None,
    status: LocalStatus | None,
    state_age_s: float | None,
) -> str:
    """One line: the robot connection, the clients and the newest ROBOT_STATE.

    state_age_s is how long ago this console received `state`.
    """
    parts = _connection_parts(status)
    if welcome is not None:
        parts.append(f"driver {driver_text(welcome.driver)}")
    if state is None:
        return _SEP.join([*parts, NO_STATE])
    parts.append(f"owner {_text(state.owner, MAX_WORD_CHARS)} #{state.owner_number}")
    stop = f"stop {_text(state.stop_level, MAX_WORD_CHARS)} #{state.stop_number}"
    if state.fault_reason:
        stop += f" ({_text(state.fault_reason)})"
    parts.append(stop)
    motion = _text(state.motion, MAX_WORD_CHARS)
    if state.move_id:
        motion += f" {state.move_index}/{state.move_total}"
        if state.move_stage:
            motion += f" {_text(state.move_stage, MAX_WORD_CHARS)}"
    parts.append(motion)
    jaws = f"jaws {_text(state.gripper.load, MAX_WORD_CHARS)}"
    if state.gripper.may_be_holding:
        jaws += ", may be holding"
    parts.append(jaws)
    parts.append(f"torque {'on' if state.torque_on else 'off'}")
    if state.assistance_wait_s is not None:
        parts.append(f"wait {_seconds(state.assistance_wait_s, 0)}")
    health = state.health if isinstance(state.health, dict) else {}
    parts.append(_permit_part(health))
    refusal = health.get("permit_refusal")
    if refusal:
        parts.append(f"permit refused: {_text(refusal)}")
    parts.append(f"view {_seconds(state.view_age_s, 1, unknown='none')}")
    parts.append(f"state {_seconds(state_age_s, 2, unknown='age unknown')}")
    return _SEP.join(parts)


def status_key(state: RobotState | None, status: LocalStatus | None) -> Hashable:
    """What a new status line is printed for when it changes.

    The facts of status_line() that change in steps. The ages, the move index,
    the wait's seconds and how far the permit reaches change all the time and
    are left out.
    """
    connection = tuple(_connection_parts(status))
    if state is None:
        return (connection, None)
    health = state.health if isinstance(state.health, dict) else {}
    return (
        connection,
        state.session_id,
        state.owner,
        state.owner_number,
        state.stop_level,
        state.stop_number,
        state.fault_reason,
        state.motion,
        state.move_id,
        state.move_stage,
        state.gripper.load,
        state.gripper.may_be_holding,
        state.torque_on,
        state.assistance_wait_s is None,
        isinstance(health.get("permit"), dict),
        str(health.get("permit_refusal") or ""),
        state.view_age_s is None,
    )


def help_block(event: Event, state: RobotState | None) -> list[str]:
    """The lines printed for a help request: why, the jaws, the time, what to do.

    `event` is the EVENT of kind help_needed; `state` the newest ROBOT_STATE.
    """
    data = event.data if isinstance(event.data, dict) else {}
    reason = _text(data.get("reason") or event.text) or "no reason given"
    if state is None:
        jaws = f"not known ({NO_STATE})"
    else:
        jaws = _text(state.gripper.load, MAX_WORD_CHARS)
        if state.gripper.may_be_holding:
            jaws += "; they may be holding something"
    # The wait starts with the help request: an older ROBOT_STATE shows none yet.
    left = None if state is None else state.assistance_wait_s
    if left is None:
        left = data.get("assistance_s")
    timed_return = data.get("timed_return")
    if timed_return is True:
        after = "then the arm moves to rest by itself"
    elif timed_return is False:
        after = "then the arm keeps holding"
    else:
        after = "the bells say what the arm does then"
    return [
        f"HELP NEEDED: {reason}",
        f"  jaws: {jaws}",
        f"  time left: {_seconds(left, 0, unknown='not known')}; {after}",
        "  you can type:",
        "    stop   a person's STOP ends the wait; the arm holds until someone acts",
        f'    rest   return the arm to rest now; asks for a typed "{PATH_IS_CLEAR}"',
        "  or give a new task at the agent program's terminal",
    ]
