# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the operator console prints, as pure functions of the link messages."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

from ..protocol.confirm import PATH_IS_CLEAR
from ..protocol.local_messages import ROLE_HUMAN
from ..protocol.messages import REQUEST_HANDBACK, REQUEST_HUMAN
from ..robot_app.text import (
    MAX_OUT_CHARS,
    MAX_WORD_CHARS,
    TORQUE_HOLD_UNKNOWN,
    TORQUE_HOLDING,
    printable,
    torque_text,
)

if TYPE_CHECKING:
    from ..protocol.local_messages import LocalStatus
    from ..protocol.messages import Event, RobotState, Welcome

# EVENT kinds the console treats in a way of its own: robot_app.waits.BELL,
# robot_app.app.EVENT_REFUSED, EVENT_HELP_NEEDED, EVENT_HUMAN and EVENT_OWNER.
KIND_BELL = "bell"
KIND_REFUSED = "refused"
KIND_HELP_NEEDED = "help_needed"
KIND_HUMAN = "human"
KIND_OWNER = "owner"

NOT_ATTACHED = "not attached to the robot link service"
NO_STATE = "no ROBOT_STATE"
# WELCOME.driver "" declares nothing; every reader treats that as the real arm.
NOT_DECLARED = "not declared (treated as real)"
_SEP = " | "

# A takeover makes a person the control owner; the hand-back ends it at rest.
# From the one to the other a takeover is open. The human program is the
# workstation program that reads the person's controller; ROBOT_STATE carries
# its stage and its note. A fault-stop is the robot app's hold after a fault;
# it stays latched until a person types reset.
TAKEOVER_POSSIBLE = (
    "takeover: possible now: hold right B for 1 s with the grip released"
)
NO_HUMAN_PROGRAM = "no human program is attached: start run_human.sh"
NOT_REPORTED = "this robot app does not report it"
# The status line's part about the live view where the robot app's file asks
# for none and none reports.
VIEW_NOT_REQUIRED = "live view not required"
ROBOT_LOST = "the robot connection is lost"
OPEN_FAULT = "a fault-stop is latched: type reset first"
OPEN_AT_REST = "at rest: hold right A for 1 s to give the arm back"
# In its place while the jaw load, what the robot app last knew of an object
# in the jaws, is "loaded": the hand-back needs the jaws proven empty.
OPEN_AT_REST_JAWS = (
    "at rest: the jaws are not proven empty: type stop, then open, then yes, "
    "then hold right A for 1 s"
)
OPEN_NO_CONTROLLER = (
    f"no controller data. To end it: type rest, then {PATH_IS_CLEAR}; then "
    "start the human program and hold right A, or see the README, "
    '"A takeover that cannot be ended with right A"'
)
OPEN_PERSON = "a person owns the arm"
OPEN_NOBODY = "nobody owns the arm: hold right B for 1 s to take it again"
# Followed by the robot app's reason, where it would refuse that right B.
OPEN_NOBODY_REFUSED = "nobody owns the arm: a new takeover is not possible: "
# The jaws under a person. While a takeover is open the trigger of the person's
# controller moves the jaws, once the jaw opening it asks for has reached the
# grip level: the jaw opening in force at the takeover. Openings are shares of
# the jaw travel, 1.00 fully open. ROBOT_STATE says what the trigger does in
# health["human"]["jaws"].
JAWS_FOLLOW = "jaws: follow the trigger"
JAWS_NOTHING = "jaws: the trigger moves nothing"
JAWS_BRING = (
    "jaws: trigger asks {asked}, grip level {level}: bring the trigger to the "
    "grip level"
)
# robot_app.gripper.NOT_MATCHED: why a trigger short of the grip level moves
# nothing.
_NOT_MATCHED = "the trigger has not reached the grip level"
# How the EVENT texts begin that ring more than once: a takeover that began
# or ended, and a refused takeover or hand-back.
_TWO_BELLS = ("TAKEOVER", "takeover ended")
_THREE_BELLS = (
    f"owner request {REQUEST_HUMAN!r} refused",
    f"owner request {REQUEST_HANDBACK!r} refused",
)
# Words of ROBOT_STATE: the stage with fresh controller data, the motion at
# rest with the motors on, the jaw load of jaws that may hold something, two
# control owners, and the stop level of a fault-stop.
_STAGE_READY = "ready"
_REST_HOLD = "rest_hold"
# The motion after a torque release whose driver call failed: the motors may
# be on or off.
_RELEASE_FAILED = "release_failed"
_JAWS_LOADED = "loaded"
_OWNER_HUMAN = "human"
_OWNER_NOBODY = "nobody"
_STOP_FAULT = "fault"
_NO_DIGITS = str.maketrans("", "", "0123456789")
# In the joints line, in place of the value of a joint without a valid reading.
INVALID = "invalid"


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


def _health(state: RobotState | None) -> dict[str, Any]:
    health = None if state is None else state.health
    return health if isinstance(health, dict) else {}


def _human(health: dict[str, Any]) -> dict[str, Any]:
    """health["human"]: the robot app's facts of the human path; {} without."""
    human = health.get("human")
    return human if isinstance(human, dict) else {}


def motors_text(state: RobotState) -> str:
    """What ROBOT_STATE says of the motors: one of robot_app.text's TORQUE_* texts.

    torque_on says only that the robot app has not released the motors.
    health["motors"] is what the joint readings say of the hold: "torque on
    (holding)", "torque NOT HOLDING" or "torque on, hold unknown". A robot
    app that does not report it gives "torque on". After a torque release
    that failed the motors may be on or off: "torque unknown".
    """
    motors = _health(state).get("motors")
    hold = motors.get("state") if isinstance(motors, dict) else None
    return torque_text(state.torque_on, hold, state.motion == _RELEASE_FAILED)


def _view_part(state: RobotState, health: dict[str, Any]) -> str:
    """The status line's part about the live view: the view age.

    Without a view age it is VIEW_NOT_REQUIRED where the robot app says that
    a person drives without a live view (health["human"]["live_view_required"]
    False), else "view none".
    """
    unasked = _human(health).get("live_view_required") is False
    if state.view_age_s is None and unasked:
        return VIEW_NOT_REQUIRED
    return f"view {_seconds(state.view_age_s, 1, unknown='none')}"


def _missed_deadlines(health: dict[str, Any]) -> int | None:
    """health["missed_deadlines"] as a count of 0 or more; None without one.

    A missed deadline is a setpoint of a move that the robot app sent late; it
    counts them from its start.
    """
    count = health.get("missed_deadlines")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def joints_line(state: RobotState) -> str:
    """One line: the measured joints, the last target sent and the readings' health.

    Joint values in rad in the order of the joints; INVALID for a joint
    without a valid reading. The reading age is that of the oldest reading
    when the robot app built `state`.
    """
    measured = [
        (_number(sample.value, 3) if sample.valid is True else None) or INVALID
        for sample in state.joints
    ]
    target = [_number(value, 3) or "?" for value in state.target_q or []]
    health = _health(state)
    age = _seconds(health.get("reading_age_s"), 2, unknown="unknown")
    missed = _missed_deadlines(health)
    return (
        f"joints (rad): {' '.join(measured) or 'none'}; "
        f"last target: {' '.join(target) or 'none'}; reading age {age}; "
        f"missed deadlines {'not reported' if missed is None else missed}"
    )


def _at_robot_app(state: RobotState | None) -> bool:
    """Whether ROBOT_STATE says a human program is attached at the robot app."""
    return _human(_health(state)).get("local_program") is True


def _no_human_program(status: LocalStatus | None, state: RobotState | None) -> bool:
    """Whether no human program is attached, at either place one attaches.

    At the robot link service, by its client list, or at the robot app's own
    socket on the Thor, by ROBOT_STATE. Without a LOCAL_STATUS it is not known.
    """
    if status is None or status.roles.get(ROLE_HUMAN, 0):
        return False
    return not _at_robot_app(state)


def _human_parts(
    state: RobotState, status: LocalStatus | None, health: dict[str, Any]
) -> list[str]:
    """The status line's parts about a person at the controller."""
    human = _human(health)
    parts = []
    stage = _text(state.teleop_stage, MAX_WORD_CHARS)
    wanted = bool(human.get("enabled") or human.get("takeover_open"))
    if _at_robot_app(state):
        # The service's client list does not name this one.
        parts.append("human program at the robot app")
    if wanted and _no_human_program(status, state):
        parts.append("no human program")
    elif stage:
        parts.append(f"headset {stage}")
    age = _number(state.controller_age_s, 2)
    if age is not None:
        parts.append(f"controller {age} s")
    if human.get("driving"):
        parts.append("human driving")
    if human.get("takeover_open"):
        parts.append("takeover open")
    # Under any other owner the flag is set and asks nothing of anybody.
    if human.get("needs_release") and state.owner == _OWNER_HUMAN:
        parts.append("needs release")
    if health.get("contact_hold"):
        parts.append("contact hold")
    note = _text(human.get("note") or "")
    # A note without a stage is one that nobody is sending any more.
    if note and stage:
        parts.append(f"note: {note}")
    return parts


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
    parts.append(motors_text(state))
    if state.assistance_wait_s is not None:
        parts.append(f"wait {_seconds(state.assistance_wait_s, 0)}")
    health = _health(state)
    parts.append(_permit_part(health))
    refusal = health.get("permit_refusal")
    if refusal:
        parts.append(f"permit refused: {_text(refusal)}")
    missed = _missed_deadlines(health)
    if missed:
        parts.append(f"missed deadlines {missed}")
    parts.append(_view_part(state, health))
    parts.extend(_human_parts(state, status, health))
    parts.append(f"state {_seconds(state_age_s, 2, unknown='age unknown')}")
    return _SEP.join(parts)


def status_key(state: RobotState | None, status: LocalStatus | None) -> Hashable:
    """What a new status line is printed for when it changes.

    The facts of status_line(), takeover_text() and jaws_text() that change in
    steps. The ages, the move index, the wait's seconds, how far the permit
    reaches and the numbers in the robot app's takeover reason and in the jaws
    line change all the time and are left out. Of the missed deadlines only
    whether there are any is in: the first one makes a line due, the count's
    growth does not.
    """
    connection = tuple(_connection_parts(status))
    if state is None:
        return (connection, None)
    health = _health(state)
    human = _human(health)
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
        # From each joint target until every joint was read at it the hold is
        # unknown: that comes with every move and makes no line due.
        motors_text(state).replace(TORQUE_HOLD_UNKNOWN, TORQUE_HOLDING),
        state.assistance_wait_s is None,
        isinstance(health.get("permit"), dict),
        str(health.get("permit_refusal") or ""),
        bool(_missed_deadlines(health)),
        state.view_age_s is None,
        human.get("live_view_required") is False,
        state.teleop_stage,
        bool(health.get("contact_hold")),
        bool(human.get("enabled")),
        bool(human.get("takeover_open")),
        bool(human.get("driving")),
        bool(human.get("needs_release")),
        str(human.get("note") or ""),
        str(human.get("ended") or ""),
        str(human.get("takeover") or "").translate(_NO_DIGITS),
        human.get("local_program") is True,
        (jaws_text(state, status) or "").translate(_NO_DIGITS),
    )


def takeover_text(state: RobotState | None, status: LocalStatus | None) -> str:
    """One line: whether a person can take the arm now, or how a takeover ends.

    Why none is possible is the robot app's word, health["human"]["takeover"],
    after what only the console sees: a lost robot connection, no human
    program at either place, no ROBOT_STATE.
    """
    human = _human(_health(state))
    # After the loss `state` is the last one before it: it says what was open
    # then and nothing of what is possible now.
    lost = status is not None and not status.robot_connected
    if state is not None and human.get("takeover_open"):
        needs = ROBOT_LOST if lost else _open_takeover(state, status)
        return f"takeover open: {needs}"
    if lost:
        return f"takeover: not possible: {ROBOT_LOST}"
    why = human.get("takeover")
    # With human driving off in the robot app's file a human program would
    # change nothing: the robot app's word stands before the missing program.
    off = human.get("enabled") is False and isinstance(why, str) and bool(why)
    if _no_human_program(status, state) and not off:
        why = NO_HUMAN_PROGRAM
    elif state is None:
        why = NO_STATE
    elif not isinstance(why, str):
        why = NOT_REPORTED
    elif not why:
        return TAKEOVER_POSSIBLE
    return f"takeover: not possible: {_text(why)}"


def _open_takeover(state: RobotState, status: LocalStatus | None) -> str:
    """What an open takeover needs next: by the stop, stage, motion, jaws, owner."""
    # A fault-stop refuses the hand-back and a new takeover alike.
    if state.stop_level == _STOP_FAULT:
        return OPEN_FAULT
    # The stage outlasts a human program that left by a few seconds.
    if state.teleop_stage != _STAGE_READY or _no_human_program(status, state):
        return OPEN_NO_CONTROLLER
    if state.motion == _REST_HOLD:
        # Unknown jaws keep the plain line: the hand-back reads them anew.
        loaded = state.gripper.load == _JAWS_LOADED
        return OPEN_AT_REST_JAWS if loaded else OPEN_AT_REST
    if state.owner == _OWNER_HUMAN:
        return OPEN_PERSON
    if state.owner == _OWNER_NOBODY:
        why = _human(_health(state)).get("takeover")
        if isinstance(why, str) and why:
            return f"{OPEN_NOBODY_REFUSED}{_text(why)}"
        return OPEN_NOBODY
    return f"{_text(state.owner, MAX_WORD_CHARS)} owns the arm"


def jaws_text(state: RobotState | None, status: LocalStatus | None) -> str | None:
    """One line: what the person's trigger does to the jaws; None without the fact.

    The fact is health["human"]["jaws"] of ROBOT_STATE. After a lost robot
    connection `state` is the last one before it and says nothing of now.
    """
    jaws = _human(_health(state)).get("jaws")
    if not isinstance(jaws, dict):
        return None
    if status is not None and not status.robot_connected:
        return None
    refusal = jaws.get("refusal")
    refusal = _text(refusal) if isinstance(refusal, str) else ""
    if jaws.get("matched") is True:
        # A refusal here holds the jaws wider than the trigger asks.
        return f"{JAWS_FOLLOW}; {refusal}" if refusal else JAWS_FOLLOW
    asked = _number(jaws.get("asked"), 2)
    level = _number(jaws.get("grip_level"), 2)
    if refusal == _NOT_MATCHED and asked is not None and level is not None:
        return JAWS_BRING.format(asked=asked, level=level)
    return f"{JAWS_NOTHING}: {refusal}" if refusal else JAWS_NOTHING


def event_bells(event: Event) -> int:
    """How many times the console rings for an EVENT; a headset's wearer hears it.

    Once for a bell and for the end of a human drive, twice for a takeover
    that began or ended, three times for a refused takeover or hand-back.
    """
    text = event.text if isinstance(event.text, str) else ""
    if event.kind in (KIND_BELL, KIND_HUMAN):
        return 1
    if event.kind == KIND_OWNER and text.startswith(_TWO_BELLS):
        return 2
    if event.kind == KIND_REFUSED and text.startswith(_THREE_BELLS):
        return 3
    return 0


def help_block(
    event: Event, state: RobotState | None, status: LocalStatus | None = None
) -> list[str]:
    """The lines printed for a help request: why, the jaws, the time, what to do.

    `event` is the EVENT of kind help_needed; `state` the newest ROBOT_STATE;
    `status` the newest LOCAL_STATUS, for the takeover line.
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
        f"  {takeover_text(state, status)}",
    ]
