# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The human program's logic: buttons, the new-squeeze rule and the ahead checks.

HumanLogic.step runs once per pass of the human program's loop and does no I/O.

Terms. A takeover makes a person the control owner, the one who may move the
arm; a hand-back ends it at rest. The clutch lets the arm follow the hand only
while the grip is squeezed; an engagement is one such stretch, and it starts
at the home, the arm's measured pose. A sample age is known when it is a
finite number of 0 or more. A pass is fresh when the controller feed's stage
is ready, its sample age is known and the graph reports the controller
present. The grip is squeezed on a fresh pass with the squeeze above the
threshold, and released on one with it at or below. The gates are what must
hold before the arm may follow the hand. Armed: the gates held on a pass with
the grip released, so the next squeeze is a new one, and the ROBOT_STATEs
have shown the drive of the engagement before as ended and the arm at rest:
one made earlier gives a pose the arm has left. Ahead: how far a target lies
ahead of what follows it.

At rest. After an engagement the arm counts as at rest once, for every joint,
two readings in a row that both began after the robot app's last setpoint
agree within REST_TOL_RAD. That is waited for at most REST_WAIT_S from the
engagement's end; then the squeeze counts on the newest ROBOT_STATE.

Following. The Thor is the machine on the robot; its robot app follows a joint
target at its speed clamp, a share of the human joint speed limit it names in
WELCOME. An engagement's first joint target is the home, and it stands until a
ROBOT_STATE shows the robot app's drive: one that follows it, or one that has
had this engagement's joint target for LIFT_AFTER_S and has sent no setpoint.
Such a drive waits for a low gripper tip to be lifted: the joint target then
follows the hand no further than the robot app reaches from the measured arm
in one step, so that its first setpoint is that joint target. From there on
each pass solves for a pose at most one pass of the hand speed limits from
the pose of the last joint target sent, toward the graph's target, and moves
the joint target toward that solution by at most JOINT_SPEED_SHARE of what
the clamp allows, every joint by the same share. A hand that is ahead is
caught up with; the arm stops where the hand stopped, or at once when the
grip is released.

The tip box. The robot app names in WELCOME the box its checker holds a
person's gripper tip in, and the height under which that tip counts as low.
The pose solved for stays TIP_BOX_MARGIN_M inside the box and above that
height, on each axis by itself: the tip stops at a wall and slides along it
while the hand goes on. On an axis on which the tip is outside at the
engagement's first pass it may come toward the box and go no further out,
which keeps the checker's own rule: a tip outside may only come back, a low
one only rise.

The joint box and the reach. The robot app names in WELCOME the joint box its
checker holds a person's joint target in. The joint target stops
JOINT_BOX_MARGIN_RAD inside it, and it stands while the pose solved for has no
solution: the engagement goes on, and the arm follows again once a step
toward the hand solves inside the box.

The hold-back. The joint target goes no further than HOLD_BACK_STEPS of the
robot app's steps ahead of where its own joint target can be: its last one in
the ROBOT_STATE, moved on by what its loop can have done since, at the pace
its last two ROBOT_STATEs show. Nothing is added once the next ROBOT_STATE is
overdue, STATE_DUE_PERIODS of the time from one to the next in this drive:
the robot app may stand. A loop that runs slow then costs speed, and so does
one that pauses. A joint target that is held back for NO_FOLLOW_S with no new
joint target of the robot app ends the engagement.

Buttons. A pass has known buttons when the feed gave the graph's outputs with
a known sample age; the controller's pose need not be valid. Right B sends one
STOP per stretch that it is down, on the stretch's first pass with known
buttons. The stretch ends when B is seen up on such a pass, and when two such
passes are more than 1 s apart: B may have been released unseen, so B that is
down then is a STOP again. A takeover and a hand-back need more, a press that
counts: the button was seen up on a fresh pass since the last pass that was
not fresh, and it is held on fresh passes. Two passes with known buttons more
than 1 s apart end such a press as well: the button is to be seen up before
a new one counts.

Jaws. The jaws are the gripper's fingers. A jaw opening is a share of the jaw
travel: 1.0 fully open, 0.0 closed. The trigger asks for one on every fresh
pass while the person owns the arm, the grip squeezed or not. The grip level
is the jaw opening in force at the takeover; the robot app lets the jaws
follow the trigger once the opening it asks for has reached the grip level.

Log lines. step() logs nothing: a log call can block, and right B's STOP must
not wait for one. The lines are kept until the program, which has sent the
pass's link messages by then, calls log_pass().
"""

from __future__ import annotations

import itertools
import logging
import math
import statistics
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..protocol.messages import (
    REQUEST_HANDBACK,
    REQUEST_HUMAN,
    HumanSetpoint,
    Message,
    OwnerRequest,
    Result,
    RobotState,
    Stop,
    Welcome,
)
from .types import (
    AHEAD_NOTE_CM,
    LIMITER_PASS_S,
    NOTE_AT_JOINT_LIMIT,
    NOTE_AT_LIMIT,
    NOTE_CATCHING_UP,
    NOTE_OUT_OF_REACH,
    NOTE_THOR_BEHIND,
    WALL_NAMES,
    WALL_TABLE,
    FeedStep,
    GraphPort,
    GraphStep,
    HumanSettings,
    IkStep,
    SolverPort,
)

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_logic")

# The robot app's speed clamp as a share of the human joint speed limit it
# names in WELCOME: SPEED_MARGIN of robot_app/human.py.
THOR_SPEED_MARGIN = 0.98
# The share of that clamp at which the joint target sent here moves, so that
# the robot app's own last joint target does not fall behind it.
JOINT_SPEED_SHARE = 0.9
# The human joint speed limit, rad/s, taken for a robot app that names none.
UNNAMED_JOINT_SPEED = 0.1
# How long an engagement's first joint target stands with no ROBOT_STATE that
# shows the robot app's drive following it; then the engagement ends. While
# that drive waits for a lift, the time counts from the last pass on which
# the joint target's tip rose by LIFT_RISE_M.
FOLLOW_WAIT_S = 1.0
LIFT_RISE_M = 0.0005
# While the robot app's drive waits for a lift, the joint target stays this
# far inside ik.max_ahead_rad of the measured arm: two readings of a holding
# arm differ, and beyond ik.max_ahead_rad the engagement ends.
LIFT_SLACK_RAD = 0.005
# Under the height the robot app names for a low gripper tip its checker
# takes a step only when the tip rises with it. The joint target moves there
# only where each step of the robot app lifts the tip by this much, m; the
# way to a solution is tried in these shares, and a way of more steps of the
# robot app than LIFT_STEPS_MOST is not taken.
LIFT_STEP_M = 2e-5
# The hand's pose counts as above the last joint target's tip from this many
# times ik.tol, what a solve may leave of its pose.
LIFT_ASKED_TOLS = 2.0
LIFT_SHARES = (1.0, 0.5, 0.25, 0.125)
LIFT_STEPS_MOST = 64
# A drive of the robot app that has this engagement's joint target and has
# sent no setpoint for this long waits for a lift: the joint target follows
# the hand and the note says NOTE_RAISE. Its start read, about 0.06 s before
# its first setpoint, looks the same until then.
LIFT_AFTER_S = 0.15
# Two readings of a joint that differ by no more than this show it at rest,
# rad: 5 steps of the motors' 16-bit position word over 8 pi rad. The idle
# reads of one joint are 0.06 s apart, so a joint slower than 2 degrees a
# second counts as at rest.
REST_TOL_RAD = 0.002
# The longest a new squeeze waits, from the end of the engagement before it,
# for the arm to be shown at rest.
REST_WAIT_S = 0.5
# NOTE_DRIVE_ON shows once the robot app has reported the drive of an ended
# engagement for this long.
DRIVE_ON_NOTE_S = 1.0
# A pose step whose solution is a jump is halved up to this often.
STEP_HALVINGS = 3
# The farthest the hand may be turned ahead of the last joint target's tip,
# rad. Near a half turn the shorter way round is not the hand's way for sure.
MAX_AHEAD_TURN_RAD = 2.0
# The note NOTE_CATCHING_UP gives the distance as AHEAD_NOTE_CM, then in steps
# of AHEAD_NOTE_STEP_CM, and stands until the hand is AHEAD_NOTE_SLACK_CM
# outside its step: a note that changed on every pass would make a new line
# on the console each time.
AHEAD_NOTE_STEP_CM = 10
AHEAD_NOTE_SLACK_CM = 2.0
# How far inside the walls of the robot app's box for the gripper tip the
# pose solved for stays, m. The tip the solver reckons with is the checker's:
# both are the arm model's FK. The margin covers what a solve leaves of its
# pose and how far a setpoint of the robot app lies off the way between two
# joint targets.
TIP_BOX_MARGIN_M = 0.01
# NOTE_AT_LIMIT names a wall once the hand is this far beyond it, m, with the
# tip no further than this from it; then until the hand is back inside it.
WALL_NOTE_M = 0.01
# How far inside the robot app's joint box for a person the joint target
# stops, rad: a joint driven to the edge of that box can read a few
# thousandths of a rad outside it, where a takeover is refused.
JOINT_BOX_MARGIN_RAD = 0.01
# The joint target goes at most this many of the robot app's steps, each its
# speed clamp times its setpoint period, ahead of where its joint target can be.
HOLD_BACK_STEPS = 2.0
# The robot app's loop stands when a ROBOT_STATE was made more than this many
# of its setpoint periods after its last setpoint: its joint target is then
# taken not to have moved since.
STANDING_PERIODS = 1.5
# The next ROBOT_STATE is overdue when the newest one arrived more than this
# many state periods ago; the state period is the middle one of the last
# STATE_GAPS_KEPT times from one ROBOT_STATE of this drive to the next, on
# the robot app's clock, and is not known before STATE_GAPS_MIN of them.
STATE_DUE_PERIODS = 1.5
STATE_GAPS_KEPT = 5
STATE_GAPS_MIN = 3
# The robot app's reported joint target counts as a new one when a joint of
# it moved by more than this share of one of its steps.
NEW_TARGET_SHARE = 0.05
# How long the joint target may be held back with no new joint target of the
# robot app in a ROBOT_STATE; then the engagement ends.
NO_FOLLOW_S = 1.0
# A pass that is held back adds 2 to a count, another takes 1 off, up to
# twice this. NOTE_THOR_BEHIND shows from this count on, until it is 0 again:
# one line on the console for a stretch of holding back, not one per pass.
BEHIND_NOTE_PASSES = 10

# The STOP that right B sends.
STOP_REASON = "right B"
STOP_SENDER = "controller"

# HUMAN_SETPOINT.note: what the operator does next, or why nothing moves.
NOTE_TAKE_OVER = "hold right B for 1 s with the grip released to take over"
# While right B is down with no press that can ask: none that counts, one with
# the grip squeezed on a pass of it, or one whose STOP went out with no
# ROBOT_STATE of the robot app's session of now. Holding on asks for nothing.
NOTE_RELEASE_B = (
    "release right B, then hold it for 1 s with the grip released to take over"
)
NOTE_RELEASE_FIRST = "release the grip, then squeeze it to move"
NOTE_SQUEEZE = "squeeze the grip to move"
NOTE_SQUEEZE_AGAIN = "release the grip and squeeze again: "  # then why
# While a new squeeze waits for the ROBOT_STATEs that show the drive of the
# engagement before as ended and the arm at rest; the squeeze counts from then.
NOTE_STOPPING = "wait: the arm is still stopping"
# In its place, and with the grip released too, once the robot app has
# reported that drive for DRIVE_ON_NOTE_S: no squeeze counts until it ends.
NOTE_DRIVE_ON = (
    "wait: the robot app still reports the last drive; if this stays, type stop"
)
# While the robot app's drive waits for a low gripper tip to be lifted: it
# sends nothing until the joint target's tip is back at the highest it
# measured. With a tip under the height the robot app names, and without.
NOTE_RAISE_TABLE = "raise the hand: the gripper is under the table clearance"
NOTE_RAISE = "raise the hand: the gripper is low and the robot app waits for a lift"
# Why an engagement ended whose joint target the robot app did not come after.
NOT_FOLLOWED = (
    f"the arm does not follow: no new setpoint of the robot app in {NO_FOLLOW_S:g} s"
)
NOTE_AT_REST = "at rest: hold right A for 1 s to give the arm back"
# In place of NOTE_AT_REST while right A is down with no press that can ask:
# none that counts, or one with the grip squeezed on a pass of it.
NOTE_RELEASE_A = (
    "release right A, then hold it for 1 s with the grip released to give the arm back"
)
# In place of both while the ROBOT_STATE shows the jaws, the gripper's
# fingers, as loaded: the robot app refuses that hand-back.
NOTE_AT_REST_JAWS = (
    "at rest: jaws not proven empty: at the console type stop, open, yes; "
    "then hold right A for 1 s"
)
# While a takeover is open and recovery, the return to rest, is the control
# owner: right B's STOP ends that return in a hold.
NOTE_RETURNING = "the arm returns to rest; right B is a STOP"
# While the arm follows the hand and the ROBOT_STATE says that the trigger has
# not reached the grip level: which way the trigger is to go. Pull: the jaws
# are closed further than it asks. Ease: it asks for more closing than that.
NOTE_TRIGGER_PULL = "pull the trigger to the grip level to move the jaws"
NOTE_TRIGGER_EASE = "ease the trigger back to the grip level to move the jaws"
# The trigger's two dead zones. At or below the first value it asks for fully
# open jaws: a trigger at rest need not read 0, and the robot app proves the
# jaws empty only when they are fully open. At or above the second it asks for
# closed jaws. Between them the jaw opening is 1.0 minus the trigger's value.
TRIGGER_OPEN_MAX = 0.05
TRIGGER_CLOSED_MIN = 0.95
# A note goes out as printable ASCII of at most this many characters, which
# keeps every HUMAN_SETPOINT under 600 bytes as it is sent. A longer one loses
# its middle: a refusal ends with what to do about it. The reason behind
# NOTE_SQUEEZE_AGAIN is cut by itself, so that its own first characters stay.
NOTE_MAX_CHARS = 100
NOTE_CUT = "..."
_NOTE_TAIL = 57  # as long as the longest remedy the robot app writes
_NOTE_HEAD = NOTE_MAX_CHARS - len(NOTE_CUT) - _NOTE_TAIL

# The keys of HumanLogic.counts(): engagements started, engagements ended by
# cause (a released grip, a gate, the workstation's own checks), and the button
# presses that counted with the requests they led to.
COUNT_KEYS = (
    "engagements",
    "released",
    "data_lost",
    "state_old",
    "not_owner",
    "stop",
    "needs_release",
    "no_reading",
    "no_solution",
    "jump",
    "solution_ahead",
    "hand_ahead",
    "limiter",
    "clutch",
    "a_pressed",
    "b_pressed",
    "takeover_asked",
    "handback_asked",
)
# Log lines kept for log_pass(); of more, the oldest are dropped.
LOG_LINES_KEPT = 64
# One of them: its level, its text, the text's values, the error it tells of.
_Line = tuple[int, str, tuple[Any, ...], BaseException | None]

# HUMAN_SETPOINT.stage of fresh controller data with an active right controller.
_STAGE_READY = "ready"
# ROBOT_STATE words read here.
_OWNER_HUMAN = "human"
_OWNER_RECOVERY = "recovery"  # the owner while the arm returns to rest
_STOP_LEVEL_NONE = "none"
_MOTION_REST_HOLD = "rest_hold"
_JAWS_LOADED = "loaded"  # ROBOT_STATE.gripper.load: the jaws may hold something
# health["human"]["jaws"]["refusal"] while the trigger is ignored for this.
_JAWS_NOT_MATCHED = "the trigger has not reached the grip level"
# RESULT.status of a command that goes on, and of one that was done.
_ACCEPTED = "accepted"
_COMPLETED = "completed"
# The rate limiter's disposition for a hand pose it refused to follow.
_LIMITER_REJECTED = "rejected"
# How the solver's problem text begins for a jump between two solutions.
_JUMP = "IK jump"
# The two requests of the human program, as its notes and counts name them.
_TAKEOVER = "takeover"
_HANDBACK = "handback"
# Slack for comparing times that are sums of pass periods.
_EPS = 1e-9
# How long the gates wait, once right B is up again, for a ROBOT_STATE that
# shows its STOP. A STOP that got lost is never shown.
_STOP_WAIT_S = 1.0
# How far apart two passes with known buttons may be for a stretch of right B,
# and for the hold of a button, to go on. Further apart, the button may have
# been released unseen in between.
_STRETCH_KEEP_S = 1.0


@dataclass(frozen=True)
class _Pass:
    """What one pass knows before anything is decided."""

    now_mono: float
    feed: FeedStep
    step: GraphStep | None  # the graph's result; None: the feed gave no outputs
    sample_age_s: float | None  # None unless a finite number of 0 or more
    known: bool  # the buttons are known: outputs came with a sample age
    fresh: bool
    squeezed: bool
    released: bool
    state: RobotState | None  # None: none yet, or one of an earlier session
    state_age_s: float | None
    young: bool  # the ROBOT_STATE is younger than state.max_age_s
    owns: bool  # the ROBOT_STATE shows the human as the control owner
    human: dict[str, Any]  # health["human"] of the ROBOT_STATE
    measured: tuple[float, ...] | None  # None: a joint has no valid reading
    thor_speed: float  # the robot app's speed clamp on a person's joint, rad/s
    thor_period_s: float  # the time from one of its setpoints to the next
    # The lowest and the highest corner, m, the pose solved for stays within
    # for the robot app's tip box and the height it names; None: it names
    # neither in a form that can be used.
    box: tuple[np.ndarray, np.ndarray] | None
    table: bool  # the bottom of that room is the named height, not the box's
    floor: float | None  # the height the robot app names for a low tip, m
    # The lowest and the highest corner, rad, the joint target stays within
    # for the robot app's joint box; None: it names no usable one.
    joint_box: tuple[np.ndarray, np.ndarray] | None

    @property
    def takeover_open(self) -> bool:
        """A takeover was accepted and no hand-back has ended it yet."""
        return bool(self.human.get("takeover_open"))


class _Button:
    """One button of the right controller, with the hold of its newest press.

    A press counts only when the button was seen up on a fresh pass since the
    last pass that was not fresh; its hold ends on any pass that is not fresh.
    After more than _STRETCH_KEEP_S without known buttons HumanLogic ends the
    hold the same way, ahead of the next pass's own reading.
    Only a press that counts asks for a takeover or a hand-back. Right B's
    STOP needs none: HumanLogic sends it on known buttons.
    """

    def __init__(self) -> None:
        self._seen_up = False
        self.pressed_at: float | None = None  # start of the press that counts
        self._clean = False  # the grip was released on every pass of the press
        self.asked = False  # the request of this press went out

    def take(self, fresh: bool, down: bool, released: bool, now_mono: float) -> bool:
        """Take this pass's reading; True on a press that counts."""
        if not fresh or not down:
            self._seen_up = fresh
            self.pressed_at = None
            return False
        if self.pressed_at is not None:
            self._clean = self._clean and released
            return False
        if not self._seen_up:
            return False
        self._seen_up = False
        self.pressed_at = now_mono
        self._clean = released
        self.asked = False
        return True

    def clean(self) -> bool:
        """A press that counts is on, with the grip released on each pass of it."""
        return self.pressed_at is not None and self._clean

    def held(self, now_mono: float, hold_s: float) -> bool:
        """The press lasted hold_s with the grip released and has not asked yet."""
        return (
            self.pressed_at is not None
            and self._clean
            and not self.asked
            and now_mono - self.pressed_at >= hold_s - _EPS
        )


class HumanLogic:
    """Turns each pass's controller data and ROBOT_STATE into link messages."""

    def __init__(
        self,
        graph: GraphPort,
        solver: SolverPort,
        settings: HumanSettings,
        source: str,
    ) -> None:
        """source is HUMAN_SOURCE_TELEOP or HUMAN_SOURCE_SYNTHETIC."""
        self._graph = graph
        self._solver = solver
        self._settings = settings
        self._source = source
        self._idle_period = 1.0 / settings.idle_hz
        self._period = 1.0 / settings.loop_hz
        self._armed = False
        # The grip was released on a pass with the gates held, and no gate
        # failed and no engagement was ended here since.
        self._released = False
        # The seq of the first HUMAN_SETPOINT without a joint target after the
        # last engagement, and when its pass was; None once a ROBOT_STATE has
        # shown that engagement's drive as ended, and before any engagement.
        self._end_seq: int | None = None
        self._ended_at = 0.0
        # The arm has not been shown at rest since the last engagement ended.
        self._settling = False
        # Per joint, the newest reading that began after the robot app's last
        # setpoint, as (when it began, its value), and whether it agreed with
        # the one before it.
        self._rest_marks: list[tuple[float, float] | None] = []
        self._rest_agree: list[bool] = []
        self._waits = False  # this pass's squeeze waits for those ROBOT_STATEs
        self._drive_on = False  # the note says NOTE_DRIVE_ON
        self._last: GraphStep | None = None
        self._clutch_engaged = False  # after the graph's last pass
        self._engaged = False  # the last pass sent a joint target
        self._home_q: tuple[float, ...] | None = None  # the pose of the last home
        self._q_sent: tuple[float, ...] | None = None  # this engagement's last target
        self._sent_at = 0.0  # when that target was computed
        self._began_at = 0.0  # when this engagement's first target was computed
        self._first_seq = 0  # the seq of the HUMAN_SETPOINT with that first target
        self._followed = False  # a ROBOT_STATE showed the drive following it
        self._ahead_m: float | None = None  # the hand ahead of the last target's tip
        self._ahead_cm: int | None = None  # as the note gives it; None: no such note
        # The walls the tip stands at with the hand beyond them, as the note
        # names them, in the order of the axes.
        self._walls: tuple[str, ...] = ()
        # This engagement's room for the pose solved for and for the joint
        # target: opened at its first pass to where the tip and the joints
        # were, then only narrowed. None before that pass.
        self._room: tuple[np.ndarray, np.ndarray] | None = None
        self._joint_room: tuple[np.ndarray, np.ndarray] | None = None
        # Why the joint target stands though the hand goes on: NOTE_OUT_OF_REACH
        # or NOTE_AT_JOINT_LIMIT with its joint; "" when it does not.
        self._stand = ""
        # Since when the robot app's drive waits for a lift, and the highest
        # the joint target's tip has been meanwhile, m.
        self._lift_since: float | None = None
        self._lift_top = -math.inf
        self._lift_under = False  # the measured tip is under the named height
        # The robot app's loop as its ROBOT_STATEs of this engagement show it:
        # (setpoints sent, when the last one went out) of the newest state
        # that names both, and its setpoint period over the time between two
        # setpoints, 1.0 at most.
        self._thor_mark: tuple[int, float] | None = None
        self._thor_pace = 1.0
        self._thor_target: tuple[float, ...] | None = None  # its last joint target
        # (seq, when it was made) of the newest ROBOT_STATE of this drive, and
        # the times from one to the next, on the robot app's clock.
        self._state_mark: tuple[int, float] | None = None
        self._state_gaps: deque[float] = deque(maxlen=STATE_GAPS_KEPT)
        # Since when the joint target is held back with that target unchanged.
        self._held_since: float | None = None
        self._behind = 0  # rises on a pass that was held back, falls on another
        self._behind_shown = False  # the note says NOTE_THOR_BEHIND
        self._failures = 0  # failed solves in a row
        self._ended = ""  # why the workstation ended the last engagement
        self._moved = False  # an engagement ran while this takeover was open
        self._a = _Button()
        self._b = _Button()
        # Right B on the last pass with known buttons. While it is down, the
        # STOP of its stretch went out.
        self._b_down = False
        self._a_down = False  # right A on the last pass with known buttons
        self._known_at: float | None = None  # the last pass with known buttons
        # (session id, stop number) of the ROBOT_STATE at right B's last STOP.
        self._b_stop: tuple[str, int] | None = None
        self._b_stop_at: float | None = None  # when that STOP went out
        self._every_pass = False  # HUMAN_SETPOINT on every pass since B's press
        self._asked: dict[str, str] = {}  # command id -> _TAKEOVER or _HANDBACK
        # (which request, the note, what the ROBOT_STATE showed at its RESULT)
        self._refusal: tuple[str, str, tuple[Any, ...] | None] | None = None
        self._ids = itertools.count(1)
        self._id_prefix = f"human-{uuid.uuid4().hex[:8]}"
        self._seq = 0
        self._idle_due = -math.inf
        self._counts = dict.fromkeys(COUNT_KEYS, 0)
        self._lines: deque[_Line] = deque(maxlen=LOG_LINES_KEPT)  # since log_pass()

    @property
    def armed(self) -> bool:
        """Whether the next squeeze may engage; False at the start.

        The program runs the graph stopped, with the engage not permitted,
        while this is False.
        """
        return self._armed

    @property
    def last(self) -> GraphStep | None:
        """The graph's result of the last pass; None when the feed gave none."""
        return self._last

    @property
    def ahead_m(self) -> float | None:
        """How far the hand is ahead of the last joint target's tip, m.

        None unless the last pass sent a joint target.
        """
        return self._ahead_m if self._engaged else None

    @property
    def walls(self) -> tuple[str, ...]:
        """The walls of the tip box the tip stands at with the hand beyond them.

        Empty unless the last pass sent a joint target.
        """
        return self._walls if self._engaged else ()

    @property
    def behind(self) -> bool:
        """Whether the joint target is being held back for a slow robot app."""
        return self._engaged and self._behind_shown

    @property
    def stand(self) -> str:
        """Why the joint target stands though the hand goes on; "" when it does not.

        NOTE_OUT_OF_REACH, or NOTE_AT_JOINT_LIMIT with its joint.
        """
        return self._stand if self._engaged else ""

    def tip_box(
        self, welcome: Welcome | None
    ) -> tuple[tuple[float, ...], tuple[float, ...], bool] | None:
        """The corners the pose solved for stays within under this WELCOME, m.

        With them, whether the bottom is the height the WELCOME names for a
        low tip. A side it does not bound is infinite. None: it names
        neither a box nor that height in a form that can be used.
        """
        box = _named_box(welcome)
        if box is None:
            return None
        return tuple(map(float, box[0])), tuple(map(float, box[1])), box[2]

    def joint_box(self, welcome: Welcome | None) -> bool:
        """Whether this WELCOME names a joint box the joint target is held in."""
        return _named_joint_box(welcome, self._solver.n_joints) is not None

    def joint_speed(self, welcome: Welcome | None) -> float:
        """The fastest a joint of the joint target moves under this WELCOME, rad/s."""
        span = min(self._period, _named_period(welcome, self._period))
        clamp = THOR_SPEED_MARGIN * _named_speed(welcome)
        return JOINT_SPEED_SHARE * clamp * span / self._period

    def counts(self) -> dict[str, int]:
        """How often each of COUNT_KEYS happened so far."""
        return dict(self._counts)

    def log_pass(self) -> None:
        """Log the lines of the passes since the last call, in their order.

        The program calls it once it has sent a pass's link messages.
        """
        while self._lines:
            level, text, values, error = self._lines.popleft()
            _log.log(level, text, *values, exc_info=error)

    def _later(
        self, level: int, text: str, *values: Any, error: BaseException | None = None
    ) -> None:
        """Keep one log line for log_pass(); error: the exception it tells of."""
        self._lines.append((level, text, values, error))

    def step(
        self,
        now_mono: float,
        feed: FeedStep,
        welcome: Welcome | None,
        state: RobotState | None,
        state_age_s: float | None,
        results: Sequence[Result] = (),
    ) -> list[Message]:
        """The link messages of this pass, in the order to send them.

        At most one STOP, at most one OWNER_REQUEST, then a HUMAN_SETPOINT when
        one is due. state is the newest ROBOT_STATE, state_age_s the time since
        it arrived, results the RESULTs that arrived since the last pass. An
        error is raised, unless a STOP or an OWNER_REQUEST was due before it
        came: those are returned and the error is a log line.
        """
        step = None if feed.outputs is None else self._graph.read(feed.outputs)
        self._last = step
        if step is not None:
            self._clutch_engaged = bool(step.engaged)
        p = self._pass(now_mono, feed, step, welcome, state, state_age_s)
        self._take_results(results, p)
        self._forget(p)
        out = self._buttons(p)
        try:
            # Behind right B's STOP: an error from here on does not lose it.
            request = self._request(p)
            if request is not None:
                out.append(request)
            setpoint = self._after_buttons(p)
        except Exception as error:
            # Nothing follows the hand after a pass that failed.
            if self._engaged:
                self._await_end(p)
            self._armed = False
            self._released = False
            self._engaged = False
            if not out:
                raise
            text = "the pass failed after its buttons were read"
            self._later(logging.ERROR, text, error=error)
            return out
        if setpoint is not None:
            out.append(setpoint)
        return out

    def _after_buttons(self, p: _Pass) -> HumanSetpoint | None:
        """The gates, the home and the joint target: this pass's HUMAN_SETPOINT."""
        step = p.step
        gate = self._gate(p)
        awaited = self._end_seq is not None or self._settling
        over = self._drive_over(p)
        if gate is not None:
            self._armed = False
            self._released = False
        elif p.released:
            self._released = True
            self._ended = ""
        if self._released and over:
            self._armed = True
        # A new squeeze that came before the ROBOT_STATE it counts from.
        self._waits = awaited and self._released and p.squeezed
        reported_s = p.now_mono - self._ended_at
        self._drive_on = self._end_seq is not None and reported_s > DRIVE_ON_NOTE_S

        # No home while the drive of the engagement before may still run, or
        # the arm still moves: the measured pose then lies behind the arm.
        if not self._clutch_engaged and over and p.young and p.measured is not None:
            self._home_q = tuple(self._solver.inside(p.measured))
            self._graph.set_home(self._solver.fk(self._home_q))

        q = None
        if self._armed and p.squeezed and step.engaged:
            q = self._follow(p)
        elif self._engaged:
            if gate is not None:
                self._end(*gate)
            elif p.released:
                self._counts["released"] += 1
                self._later(logging.INFO, "engagement ended: the grip was released")
            else:
                self._end("clutch", "the clutch let go of the hand")
        if self._engaged and q is None:
            self._await_end(p)
        self._engaged = q is not None
        return self._setpoint(p, gate, q)

    def _await_end(self, p: _Pass) -> None:
        """An engagement ended on this pass: its drive is to be shown as ended.

        And then the arm as at rest. Until then the logic is not armed and
        takes no home.
        """
        self._armed = False
        # The next HUMAN_SETPOINT: the first without a joint target.
        self._end_seq = self._seq + 1
        self._ended_at = p.now_mono
        self._settling = True
        self._rest_marks = [None] * self._solver.n_joints
        self._rest_agree = [False] * self._solver.n_joints

    def _drive_over(self, p: _Pass) -> bool:
        """Whether the last engagement's drive has ended and the arm is at rest.

        First a ROBOT_STATE shows the drive as ended; with none such the
        logic stays un-armed, however long. Then the ROBOT_STATEs show the
        arm at rest, or REST_WAIT_S have passed since the engagement ended.
        True also when no engagement has ended since.
        """
        if self._end_seq is not None:
            if not self._shown_ended(p):
                return False
            self._end_seq = None
        if not self._settling:
            return True
        waited_s = p.now_mono - self._ended_at
        if self._at_rest(p) or waited_s > REST_WAIT_S + _EPS:
            self._settling = False
            return True
        return False

    def _shown_ended(self, p: _Pass) -> bool:
        """Whether this ROBOT_STATE shows the last engagement's drive as ended.

        It shows no human drive, and it was made after the pass that ended
        the engagement: the newest HUMAN_SETPOINT the robot app had taken
        then is that pass's, or a later one. A robot app that names no seq
        is taken at a ROBOT_STATE that arrived after that pass. States that
        stop coming close the gates as old.
        """
        if p.state is None or p.state_age_s is None or p.human.get("driving"):
            return False
        seq = p.human.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            return seq >= self._end_seq
        return p.now_mono - p.state_age_s > self._ended_at

    def _at_rest(self, p: _Pass) -> bool:
        """Whether the ROBOT_STATEs so far show every joint at rest.

        For each joint: two readings in a row that both began after the
        robot app's last setpoint and agree within REST_TOL_RAD. A robot app
        that names no send time of a setpoint has sent none.
        """
        state = p.state
        if state is None or p.human.get("driving"):
            return False
        sent_at = _finite(state.target_t)
        after = -math.inf if sent_at is None else sent_at
        marks, agree = self._rest_marks, self._rest_agree
        for sample in state.joints:
            began = _finite(sample.t_start)
            value = _finite(sample.value)
            if not 0 <= sample.joint < len(marks) or not sample.valid:
                continue
            if began is None or value is None or not began > after:
                continue
            mark = marks[sample.joint]
            if mark is None or not mark[0] > after:
                agree[sample.joint] = False  # the first reading since the setpoint
            elif began > mark[0]:
                agree[sample.joint] = abs(value - mark[1]) <= REST_TOL_RAD
            else:
                continue  # the reading the mark was taken from
            marks[sample.joint] = (began, value)
        return all(agree)

    def _pass(
        self,
        now_mono: float,
        feed: FeedStep,
        step: GraphStep | None,
        welcome: Welcome | None,
        state: RobotState | None,
        state_age_s: float | None,
    ) -> _Pass:
        """The facts of this pass."""
        settings = self._settings
        age = _sample_age(feed.sample_age_s)
        known = step is not None and age is not None
        fresh = bool(known and step.present and feed.stage == _STAGE_READY)
        threshold = settings.retarget.squeeze_threshold
        if welcome is None or (
            state is not None and state.session_id != welcome.session_id
        ):
            state = None
        room = _named_box(welcome)
        return _Pass(
            now_mono=now_mono,
            feed=feed,
            step=step,
            sample_age_s=age,
            known=known,
            fresh=fresh,
            squeezed=fresh and bool(step.squeeze > threshold),
            released=fresh and bool(step.squeeze <= threshold),
            state=state,
            state_age_s=state_age_s,
            young=(
                state is not None
                and state_age_s is not None
                and state_age_s < settings.state_max_age_s
            ),
            owns=state is not None and state.owner == _OWNER_HUMAN,
            human=_human_facts(state),
            measured=(
                None if state is None else _measured(state, self._solver.n_joints)
            ),
            thor_speed=THOR_SPEED_MARGIN * _named_speed(welcome),
            thor_period_s=_named_period(welcome, self._period),
            box=None if room is None else (room[0], room[1]),
            table=room is not None and room[2],
            floor=_named_floor(welcome),
            joint_box=_named_joint_box(welcome, self._solver.n_joints),
        )

    def _take_results(self, results: Sequence[Result], p: _Pass) -> None:
        """Note the final RESULT of a request this logic sent."""
        for result in results:
            which = self._asked.get(result.command_id)
            if which is None or result.status == _ACCEPTED:
                continue
            del self._asked[result.command_id]
            if which == _TAKEOVER:
                self._every_pass = False
            if result.status == _COMPLETED:
                self._later(logging.INFO, "%s: %s", which, result.status)
            else:
                note = f"{which} {result.status}: {result.reason}"
                self._refusal = (which, note, _shown(p.state))
                self._later(logging.WARNING, "%s", note)

    def _forget(self, p: _Pass) -> None:
        """Drop what the newest ROBOT_STATE has overtaken."""
        if p.state is None:
            return
        if not p.owns:
            self._ended = ""
        if not p.takeover_open:
            self._moved = False
        if self._refusal is not None:
            # A refusal goes once what was asked for has come about, and once
            # the ROBOT_STATE shows another owner, STOP or motion than at its
            # RESULT: its reason may not hold any more.
            which, _, shown = self._refusal
            if (
                (which == _TAKEOVER and p.owns)
                or (which == _HANDBACK and not p.takeover_open)
                or shown != _shown(p.state)
            ):
                self._refusal = None

    def _buttons(self, p: _Pass) -> list[Message]:
        """Take this pass's buttons; right B's STOP when one is due."""
        out: list[Message] = []
        step = p.step
        known_at = self._known_at
        apart = known_at is not None and p.now_mono - known_at > _STRETCH_KEEP_S + _EPS
        if apart:
            # Too long unseen: a press that was on may have ended meanwhile. It
            # counts no more, and the button is to be seen up before one does.
            self._b.take(False, False, False, p.now_mono)
            self._a.take(False, False, False, p.now_mono)
        b_pressed = self._b.take(p.fresh, p.fresh and step.b, p.released, p.now_mono)
        a_pressed = self._a.take(p.fresh, p.fresh and step.a, p.released, p.now_mono)
        if a_pressed or b_pressed:
            self._refusal = None
        if a_pressed:
            self._counts["a_pressed"] += 1
        if b_pressed:
            self._counts["b_pressed"] += 1
            self._every_pass = True
        if p.known:
            self._a_down = bool(step.a)
            b_down = bool(step.b)
            self._known_at = p.now_mono
            if apart:
                self._b_down = False  # too long unseen: a new stretch
            if b_down and not self._b_down:
                # The first pass of a stretch; a press that counts is on one.
                state = p.state
                self._b_stop = (
                    None if state is None else (state.session_id, state.stop_number)
                )
                self._b_stop_at = p.now_mono
                out.append(Stop(reason=STOP_REASON, sender=STOP_SENDER))
                self._later(logging.INFO, "right B is down: STOP sent")
            self._b_down = b_down
        if self._every_pass and self._b.pressed_at is None:
            # B was seen up, or the data were lost with no request on its way.
            if p.fresh or _TAKEOVER not in self._asked.values():
                self._every_pass = False
        return out

    def _request(self, p: _Pass) -> OwnerRequest | None:
        """The OWNER_REQUEST a held button asks for on this pass, if any.

        Both wait for a ROBOT_STATE that shows right B's last STOP, so that
        they quote the numbers that STOP has made.
        """
        state = p.state
        if state is None or not p.young or self._stop_unseen(p, state):
            return None
        hold_s = self._settings.hold_s
        seen = self._b_stop
        if (
            self._b.held(p.now_mono, hold_s)
            and seen is not None
            and seen[0] == state.session_id
        ):
            self._b.asked = True
            return self._owner_request(state, REQUEST_HUMAN, _TAKEOVER, "B")
        if self._a.held(p.now_mono, hold_s) and p.takeover_open:
            self._a.asked = True
            return self._owner_request(state, REQUEST_HANDBACK, _HANDBACK, "A")
        return None

    def _owner_request(
        self, state: RobotState, request: str, which: str, button: str
    ) -> OwnerRequest:
        """One OWNER_REQUEST that quotes state; an older one of its kind is dropped."""
        command_id = f"{self._id_prefix}-{next(self._ids)}"
        self._asked = {key: kind for key, kind in self._asked.items() if kind != which}
        self._asked[command_id] = which
        self._counts[f"{which}_asked"] += 1
        reason = f"right {button} held {float(self._settings.hold_s)} s"
        self._later(logging.INFO, "%s: asking for the %s", reason, which)
        return OwnerRequest(
            command_id,
            state.session_id,
            state.owner_number,
            state.stop_number,
            request,
            reason,
        )

    def _gate(self, p: _Pass) -> tuple[str, str] | None:
        """The first gate that fails, as (count key, why); None when all hold."""
        state = p.state
        if not p.fresh:
            return "data_lost", "controller data lost"
        if state is None or p.state_age_s is None:
            return "state_old", "no robot state"
        if not p.young:
            return "state_old", f"the robot state is {p.state_age_s:.2f} s old"
        if state.owner != _OWNER_HUMAN or state.stop_level != _STOP_LEVEL_NONE:
            return "not_owner", "the human does not own the arm"
        if self._stop_unseen(p, state):
            return "stop", "STOP by right B"
        if p.human.get("needs_release"):
            # A squeeze refused as busy ran no drive: "ended" is an older end.
            refusal = str(p.human.get("refusal") or "")
            busy = refusal if refusal.startswith("busy: ") else ""
            why = busy or p.human.get("ended")
            return "needs_release", str(why or "the robot app asks for a released grip")
        if p.measured is None:
            return "no_reading", "a joint has no valid reading"
        return None

    def _stop_unseen(self, p: _Pass, state: RobotState) -> bool:
        """Whether ROBOT_STATE does not show right B's last STOP yet.

        While B stays down the wait has no end. Once B was seen up on a pass
        with known buttons it ends _STOP_WAIT_S after the STOP.
        """
        if not self._b_down:
            sent_at = self._b_stop_at
            if sent_at is None or p.now_mono - sent_at >= _STOP_WAIT_S - _EPS:
                return False
        seen = self._b_stop
        if seen is None:
            return True
        return seen[0] == state.session_id and state.stop_number <= seen[1]

    def _follow(self, p: _Pass) -> tuple[float, ...] | None:
        """This pass's joint target; None when the engagement ends here.

        Called only while the gates hold, so the ROBOT_STATE and the measured
        pose are there.
        """
        ik = self._settings.ik
        step = p.step
        if not self._engaged:
            self._q_sent = None
            self._followed = False
            self._began_at = p.now_mono
            # This pass's HUMAN_SETPOINT: one goes out on every pass while the
            # person owns the arm.
            self._first_seq = self._seq + 1
            self._ahead_cm = None
            self._walls = ()
            self._room = None
            self._joint_room = None
            self._stand = ""
            self._lift_since = None
            self._lift_top = -math.inf
            self._thor_mark = None
            self._thor_pace = 1.0
            self._thor_target = None
            self._state_mark = None
            self._state_gaps.clear()
            self._held_since = None
            self._behind = 0
            self._behind_shown = False
            self._failures = 0
            self._moved = True
            self._refusal = None
            self._counts["engagements"] += 1
            self._later(logging.INFO, "engagement started")
        seed = self._q_sent
        since = self._period if seed is None else max(0.0, p.now_mono - self._sent_at)
        if seed is None:
            seed = self._home_q
        if seed is None:
            seed = tuple(self._solver.inside(p.measured))
        if step.limiter == _LIMITER_REJECTED:
            self._end("limiter", "the hand moved too fast for the rate limiter")
            return None
        if step.ee_target is None or step.commanded_target is None:
            self._end("no_solution", "the retargeting graph gave no target")
            return None
        at = self._solver.fk(seed)  # the tip's pose at the last joint target
        tip = at[:3, 3]
        self._room = _room(p.box, tip, self._room)
        low, high = self._room
        hand = step.commanded_target[:3, 3]
        far_m = float(np.linalg.norm(hand - tip))
        if not far_m <= ik.max_ahead_m:
            why = (
                f"the hand is {far_m * 100.0:.0f} cm from the gripper tip, over "
                f"the {ik.max_ahead_m * 100.0:.0f} cm believed"
            )
            self._end("hand_ahead", why)
            return None
        # Where the tip may go for this hand: the hand's point, held inside
        # the tip box. The arm is behind the hand by its way to that point;
        # the rest of the hand's way lies beyond a wall, where no tip follows.
        # Neither part is longer than the hand's distance from a tip inside.
        reach = np.clip(hand, low, high)
        ahead_m = float(np.linalg.norm(reach - tip))
        turned = _axis_angle(at[:3, :3].T @ step.commanded_target[:3, :3])[1]
        if not turned <= MAX_AHEAD_TURN_RAD:
            why = (
                f"the hand is turned {math.degrees(turned):.0f} deg ahead of the "
                f"arm ({math.degrees(MAX_AHEAD_TURN_RAD):.0f} are followed)"
            )
            self._end("hand_ahead", why)
            return None
        self._ahead_m = ahead_m
        self._ahead_cm = _ahead_step(100.0 * ahead_m, self._ahead_cm)
        bottom = WALL_TABLE if p.table else WALL_NAMES[2][0]
        self._walls = _walls_at(hand - reach, tip, low, high, self._walls, bottom)
        waits = False
        if self._q_sent is not None and not self._thor_follows(p):
            waits = self._thor_waits(p)
            if p.now_mono - self._began_at > FOLLOW_WAIT_S + _EPS:
                why = f"the arm did not follow the squeeze within {FOLLOW_WAIT_S:g} s"
                self._end("solution_ahead", why)
                return None
        if not waits:
            self._lift_since = None
        elif self._lift_since is None:
            self._lift_since = p.now_mono
            self._lift_top = float(tip[2])
            # The robot app counts the measured arm's tip as low under the
            # height it names.
            measured_z = float(self._solver.fk(p.measured)[2, 3])
            self._lift_under = p.floor is not None and measured_z < p.floor
        # The robot app's drive has had this engagement's joint target for
        # longer than its start read takes and has sent nothing: it waits
        # for a low gripper tip to be lifted.
        lifting = waits and p.now_mono - self._lift_since > LIFT_AFTER_S
        rise = 0.0  # how far the hand's pose is above the last joint target's tip
        if self._q_sent is not None and not self._followed and not lifting:
            # The first target stands: the robot app measures the first step
            # of its drive from the arm, and refuses a target that has moved
            # on while it read the arm.
            q = seed
        else:
            goal = np.array(step.ee_target, dtype=np.float64)
            goal[:3, 3] = np.clip(goal[:3, 3], low, high)
            rise = float(goal[2, 3] - tip[2])
            q = self._stepped(goal, at, seed, since, p)
            if q is None:
                return None
        reference, moves = _reference(p.state, p.human, p.measured)
        held = False
        under = p.floor is not None and float(tip[2]) < p.floor
        if lifting:
            q = self._lifted_off(seed, q, reference, under, rise, p)
        elif under:
            q = self._lifted(seed, q, reference, moves, rise, p)
        if lifting:
            top = float(self._solver.fk(q)[2, 3])
            if top > self._lift_top + LIFT_RISE_M:
                self._lift_top = top
                self._began_at = p.now_mono
        if moves:
            # The hold-back: this pass's joint target goes only as far as it
            # stays within reach of where the robot app's joint target can be.
            self._watch_thor(p, reference)
            room = HOLD_BACK_STEPS * p.thor_speed * p.thor_period_s + self._credit(p)
            share = _share_within(seed, q, reference, room)
            if share < 1.0:
                held = True
                q = tuple(a + share * (b - a) for a, b in zip(seed, q))
        self._behind = (
            min(2 * BEHIND_NOTE_PASSES, self._behind + 2)
            if held
            else max(0, self._behind - 1)
        )
        if self._behind >= BEHIND_NOTE_PASSES:
            self._behind_shown = True
        elif self._behind == 0:
            self._behind_shown = False
        if not held:
            self._held_since = None
        elif self._held_since is None:
            self._held_since = p.now_mono
        elif p.now_mono - self._held_since > NO_FOLLOW_S + _EPS:
            self._end("solution_ahead", NOT_FOLLOWED)
            return None
        # The robot app's joint target may have come this much nearer since
        # the ROBOT_STATE: what is left is how far the arm does not follow.
        nearer = p.state_age_s * p.thor_speed if moves else 0.0
        ahead = max(abs(a - b) for a, b in zip(q, reference)) - nearer
        if not ahead <= ik.max_ahead_rad:
            why = f"the solution is {ahead:.3f} rad ahead of the arm"
            self._end("solution_ahead", why)
            return None
        self._q_sent = q
        self._sent_at = p.now_mono
        return q

    def _watch_thor(self, p: _Pass, target: tuple[float, ...]) -> None:
        """Take what a ROBOT_STATE of the robot app's drive says of its loop.

        Its joint target, which is a new one when a joint of it moved by more
        than NEW_TARGET_SHARE of a step. The time from one ROBOT_STATE to the
        next. The pace of its loop: its setpoint period over the time it took
        for each setpoint between this state and the one before, 1.0 at most.
        A robot app that names no send time of its joint target is taken at
        its setpoint period.
        """
        last_target = self._thor_target
        least = NEW_TARGET_SHARE * p.thor_speed * p.thor_period_s
        if last_target is None or any(
            abs(a - b) > least for a, b in zip(target, last_target)
        ):
            self._thor_target = target
            self._held_since = None
        made_at = _finite(p.state.clock_mono_s)
        seq = p.state.seq
        if made_at is not None and isinstance(seq, int) and not isinstance(seq, bool):
            seen = self._state_mark
            if seen is not None and seq > seen[0] and made_at > seen[1]:
                self._state_gaps.append((made_at - seen[1]) / (seq - seen[0]))
            if seen is None or seq != seen[0]:
                self._state_mark = (seq, made_at)
        sent_at = _finite(p.state.target_t)
        if sent_at is None:
            return
        mark = (p.state.move_index, sent_at)
        last = self._thor_mark
        if last is not None and mark[0] > last[0] and mark[1] > last[1]:
            period = (mark[1] - last[1]) / (mark[0] - last[0])
            self._thor_pace = min(1.0, p.thor_period_s / period)
        self._thor_mark = mark

    def _credit(self, p: _Pass) -> float:
        """How far the robot app's joint target is taken to have come on, rad.

        Since the ROBOT_STATE: its age at the speed clamp and the pace of the
        robot app's loop. Nothing when the state was made more than
        STANDING_PERIODS setpoint periods after its last setpoint: the loop
        stands. Nothing either when the next ROBOT_STATE is overdue: the
        robot app, or the way from it, may stand, and what it has done since
        is not known.
        """
        sent_at = _finite(p.state.target_t)
        made_at = _finite(p.state.clock_mono_s)
        if sent_at is not None and made_at is not None:
            if made_at - sent_at > STANDING_PERIODS * p.thor_period_s:
                return 0.0
        if len(self._state_gaps) >= STATE_GAPS_MIN:
            period_s = statistics.median(self._state_gaps)
            if p.state_age_s > STATE_DUE_PERIODS * period_s:
                return 0.0
        return self._thor_pace * p.state_age_s * p.thor_speed

    def _thor_waits(self, p: _Pass) -> bool:
        """Whether a ROBOT_STATE shows the robot app's drive waiting for a lift.

        Its drive runs, has this engagement's first joint target, or a later
        one, as its newest, and has sent no setpoint. A robot app that names
        no seq is not taken so.
        """
        seq = p.human.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            return False
        waits = p.human.get("driving") and p.state.move_index == 0
        return bool(waits) and seq >= self._first_seq

    def _thor_follows(self, p: _Pass) -> bool:
        """Whether a ROBOT_STATE has shown the robot app's drive following this engagement.

        Its drive runs, has sent a setpoint, had this engagement's first joint
        target, or a later one, as its newest, and its own joint target has
        come within one of its steps of that first target. A robot app that
        names no seq, or no joint target, is taken at the others.
        """
        if self._followed:
            return True
        seq = p.human.get("seq")
        named = isinstance(seq, int) and not isinstance(seq, bool)
        if not (
            p.human.get("driving")
            and p.state.move_index > 0
            and (not named or seq >= self._first_seq)
        ):
            return False
        reference, known = _reference(p.state, p.human, p.measured)
        if known:
            off = max(abs(a - b) for a, b in zip(self._q_sent, reference))
            if not off <= p.thor_speed * p.thor_period_s + _EPS:
                return False
        self._followed = True
        return True

    def _stepped(
        self,
        goal: np.ndarray,
        at: np.ndarray,
        seed: tuple[float, ...],
        since: float,
        p: _Pass,
    ) -> tuple[float, ...] | None:
        """This pass's joint target toward goal; None when the engagement ends here.

        seed again, the last joint target, while the pose solved for has no
        solution: nothing moves on a solve that failed, and the engagement
        goes on for the hand to come back. A solution that is a jump ends
        the engagement, as it did. A solution outside the robot app's joint
        box is gone toward as far as the box allows.
        """
        solved = self._solve(goal, at, seed, since)
        q = None if solved.problem else _joints(solved.q, len(seed))
        if q is None:
            problem = solved.problem or "no IK solution (no usable joint values)"
            if problem.startswith(_JUMP):
                self._end("jump", problem)
                return None
            self._failures += 1
            if self._failures >= self._settings.ik.max_failures:
                self._stand = NOTE_OUT_OF_REACH
            return seed
        self._failures = 0
        self._stand = ""
        q = self._limited(seed, q, since, p)
        self._joint_room = _room(p.joint_box, np.array(seed), self._joint_room)
        share, joint = _share_inside(seed, q, *self._joint_room)
        if joint is not None:
            self._stand = NOTE_AT_JOINT_LIMIT.format(joint=joint + 1)
            q = tuple(a + share * (b - a) for a, b in zip(seed, q))
        return q

    def _lifted(
        self,
        seed: tuple[float, ...],
        q: tuple[float, ...],
        reference: tuple[float, ...],
        known: bool,
        rise: float,
        p: _Pass,
    ) -> tuple[float, ...]:
        """The joint target toward q that only lifts a low gripper tip; else seed.

        For a last joint target whose tip is under the height the robot app
        names, once the robot app's drive has sent a setpoint. Its checker
        takes a step there only when the tip rises with it, so the joint
        target moves only where the hand's pose is above its tip by more
        than a solve may leave (rise over LIFT_ASKED_TOLS of ik.tol) and
        every step of the robot app toward it lifts the tip. The joint
        target waits for the robot app's own, reference, to be the last one
        sent, so that each of its steps ends on a joint target and none is
        cut by its speed clamp. known False: no ROBOT_STATE of that drive
        yet; the target stands.
        """
        if q == seed or not known or not self._asked_up(rise):
            return seed
        if not (p.human.get("driving") and p.state.move_index > 0):
            return seed
        if max(abs(a - b) for a, b in zip(seed, reference)) > _EPS:
            return seed
        for share in LIFT_SHARES:
            part = tuple(a + share * (b - a) for a, b in zip(seed, q))
            if self._rises(reference, part, p):
                return part
        return seed

    def _lifted_off(
        self,
        seed: tuple[float, ...],
        q: tuple[float, ...],
        arm: tuple[float, ...],
        under: bool,
        rise: float,
        p: _Pass,
    ) -> tuple[float, ...]:
        """The joint target while the robot app's drive has sent no setpoint.

        The longest part of this pass's step from seed toward q that the
        robot app reaches from the measured arm in one step: no joint
        further from the arm than one step of the joint target, and the tip
        no further than the hand's speed limit allows in one setpoint
        period. The robot app's first setpoint is then that joint target: on
        the hand's way, not cut by its speed clamp, no faster than the hand
        may be. under, the last joint target's tip is under the height the
        robot app names: only where the hand's pose is above that tip and
        the tip rises, from the last joint target and from the arm. seed
        when no part does.
        """
        if q == seed or (under and not self._asked_up(rise)):
            return seed
        step = JOINT_SPEED_SHARE * p.thor_speed * p.thor_period_s
        room = min(step, self._settings.ik.max_ahead_rad - LIFT_SLACK_RAD)
        limit = self._settings.retarget.max_linear_velocity * p.thor_period_s
        from_arm = self._solver.fk(arm)[:3, 3]
        height = float(self._solver.fk(seed)[2, 3])
        share = _share_within(seed, q, arm, room)
        for _ in LIFT_SHARES:
            if share <= 0.0:
                break
            part = tuple(a + share * (b - a) for a, b in zip(seed, q))
            tip = self._solver.fk(part)[:3, 3]
            fits = float(np.linalg.norm(tip - from_arm)) <= limit
            if fits and under:
                rises = float(tip[2]) > height + LIFT_STEP_M
                fits = rises and self._rises(arm, part, p)
            if fits:
                return part
            share *= 0.5
        return seed

    def _asked_up(self, rise: float) -> bool:
        """Whether the hand's pose is above the last joint target's tip.

        By more than a solve may leave of its pose: what a solve leaves
        upward is not a rise the hand asked for.
        """
        return rise > LIFT_ASKED_TOLS * self._settings.ik.tol

    def _rises(self, start: Sequence[float], target: Sequence[float], p: _Pass) -> bool:
        """Whether the tip rises with every step of the robot app from start to target.

        Its steps: each joint by at most its speed clamp for one setpoint
        period, as robot_app/human.py clamps them.
        """
        most = p.thor_speed * p.thor_period_s
        at = [float(value) for value in start]
        height = float(self._solver.fk(at)[2, 3])
        for _ in range(LIFT_STEPS_MOST):
            to = [a + max(-most, min(most, b - a)) for a, b in zip(at, target)]
            if to == at:
                return True
            risen = float(self._solver.fk(to)[2, 3])
            if not risen > height + LIFT_STEP_M:
                return False
            at, height = to, risen
        return False

    def _solve(
        self,
        goal: np.ndarray,
        at: np.ndarray,
        seed: tuple[float, ...],
        since: float,
    ) -> IkStep:
        """One solve on the way from the pose `at` to goal, seeded with seed.

        The pose solved for lies at most one pass of the hand speed limits
        from `at`, so the tip is no faster than those limits while it catches
        up. Near a singular pose a short step of the tip asks a joint for a
        long one: while the solution is a jump the step is halved, up to
        STEP_HALVINGS times. since is the time since the last joint target.
        """
        retarget = self._settings.retarget
        span = min(LIMITER_PASS_S, max(0.5 * LIMITER_PASS_S, since))  # as the limiter
        metres = retarget.max_linear_velocity * span
        rad = retarget.max_angular_velocity * span
        slack = self._settings.ik.tol  # what a solve may leave of its pose
        for _ in range(STEP_HALVINGS):
            solved = self._solver.solve(_toward(at, goal, metres, rad, slack), seed)
            if not solved.problem.startswith(_JUMP):
                return solved
            metres, rad = 0.5 * metres, 0.5 * rad
        return self._solver.solve(_toward(at, goal, metres, rad, slack), seed)

    def _limited(
        self, seed: tuple[float, ...], q: tuple[float, ...], since: float, p: _Pass
    ) -> tuple[float, ...]:
        """q, or the joint values on the way to it from seed that the speed allows.

        No joint moves faster than JOINT_SPEED_SHARE of the robot app's speed
        clamp, over at most one pass and one of the robot app's setpoint
        periods. Every joint moves by the same share of its way, so the
        direction in joint space is kept.
        """
        span = min(since, self._period, p.thor_period_s)
        most = JOINT_SPEED_SHARE * p.thor_speed * span
        ways = [to - start for start, to in zip(seed, q)]
        far = max(abs(way) for way in ways)
        if far <= most:
            return q
        share = most / far
        return tuple(start + share * way for start, way in zip(seed, ways))

    def _end(self, key: str, why: str) -> None:
        """End the engagement on the workstation; only a new squeeze starts one."""
        self._armed = False
        self._released = False
        self._ended = why
        self._q_sent = None
        self._failures = 0
        self._counts[key] += 1
        self._later(logging.INFO, "engagement ended: %s", why)

    def _setpoint(
        self,
        p: _Pass,
        gate: tuple[str, str] | None,
        q: tuple[float, ...] | None,
    ) -> HumanSetpoint | None:
        """This pass's HUMAN_SETPOINT; None when none is due.

        One is due on every pass while the human owns the arm and from a press
        of right B that counts on; else rates.idle_hz times per second, without
        a target.
        """
        state = p.state
        if state is None:
            return None
        every_pass = p.owns or self._every_pass
        if not every_pass and p.now_mono < self._idle_due - _EPS:
            return None
        if every_pass or self._idle_due + self._idle_period <= p.now_mono:
            self._idle_due = p.now_mono + self._idle_period
        else:
            self._idle_due += self._idle_period
        self._seq += 1
        return HumanSetpoint(
            session_id=state.session_id,
            owner_number=state.owner_number,
            stop_number=state.stop_number,
            seq=self._seq,
            source=self._source,
            stage=p.feed.stage,
            sample_age_s=p.sample_age_s,
            engaged=q is not None,
            grip_released=p.released,
            q=None if q is None else list(q),
            jaw_opening=self._jaw_opening(p),
            state_clock_mono_s=state.clock_mono_s,
            note=_cut(self._note(p, state, gate)),
        )

    def _jaw_opening(self, p: _Pass) -> float | None:
        """What the trigger asks of the jaws on this pass; None: they keep their state.

        Only a fresh pass has a trigger to read. The robot app takes a jaw
        opening from nobody but a person who owns the arm, so none is sent
        unless the ROBOT_STATE shows one.
        """
        if not (p.fresh and p.owns):
            return None
        return jaw_opening(p.step.trigger)

    def _jaw_note(self, p: _Pass) -> str:
        """What to do with the trigger while the robot app ignores it; else ""."""
        jaws = p.human.get("jaws")
        if not isinstance(jaws, dict) or jaws.get("matched"):
            return ""
        if jaws.get("refusal") != _JAWS_NOT_MATCHED:
            return ""
        asked = _finite(jaws.get("asked"))
        level = _finite(jaws.get("grip_level"))
        if asked is not None and level is not None and asked < level:
            return NOTE_TRIGGER_EASE
        return NOTE_TRIGGER_PULL

    def _note(self, p: _Pass, state: RobotState, gate: tuple[str, str] | None) -> str:
        """What the operator does next, or why the arm does not follow."""
        if p.owns and self._engaged:
            parts = []
            lifted_s = None
            if self._lift_since is not None:
                lifted_s = p.now_mono - self._lift_since
            if lifted_s is not None and lifted_s > LIFT_AFTER_S:
                parts.append(NOTE_RAISE_TABLE if self._lift_under else NOTE_RAISE)
            if self._walls:
                parts.append(NOTE_AT_LIMIT.format(walls=", ".join(self._walls)))
            if self._stand:
                parts.append(self._stand)
            if self._behind_shown:
                parts.append(NOTE_THOR_BEHIND)
            if self._ahead_cm is not None:
                parts.append(NOTE_CATCHING_UP.format(cm=self._ahead_cm))
            # Behind every note about the arm.
            return _joined(parts) if parts else self._jaw_note(p)
        if p.owns and self._ended:
            # The prefix stays whole: the reason alone is cut to what is left.
            room = NOTE_MAX_CHARS - len(NOTE_SQUEEZE_AGAIN)
            return NOTE_SQUEEZE_AGAIN + _cut(self._ended, room)
        ready = p.feed.stage == _STAGE_READY
        if ready and p.takeover_open and state.owner == _OWNER_RECOVERY:
            # Only with a ready stage: else right B may not be read.
            return NOTE_RETURNING
        if self._refusal is not None:
            return self._refusal[1]
        if not ready:
            return p.feed.note
        at_rest = state.motion == _MOTION_REST_HOLD
        if not p.owns:
            if p.takeover_open and at_rest:
                return self._give_back(state)
            # B is down with no press that can ask: its hold asks for nothing.
            # One can ask only on a ROBOT_STATE of the session its STOP quoted.
            seen = self._b_stop
            quoted = seen is not None and seen[0] == state.session_id
            waiting = self._b_down and not (self._b.clean() and quoted)
            return NOTE_RELEASE_B if waiting else NOTE_TAKE_OVER
        if gate is not None:
            key, why = gate
            if key == "needs_release":
                return NOTE_RELEASE_FIRST
            # The age in the text would change the note on every pass.
            return "no fresh robot state" if key == "state_old" else why
        if self._drive_on:
            return NOTE_DRIVE_ON
        if p.squeezed:
            # A new squeeze that waits for the drive before it to be shown as
            # ended, and the arm as at rest, counts from then on; any other
            # is to be released.
            return NOTE_STOPPING if self._waits else NOTE_RELEASE_FIRST
        # Before the arm was moved, the next step is the squeeze, also at rest.
        return self._give_back(state) if at_rest and self._moved else NOTE_SQUEEZE

    def _give_back(self, state: RobotState) -> str:
        """The note at rest with a takeover open: the hand-back's next step."""
        if state.gripper.load == _JAWS_LOADED:
            return NOTE_AT_REST_JAWS  # the jaws' opening comes first
        # A is down with no press that can ask: its hold asks for nothing.
        waiting = self._a_down and not self._a.clean()
        return NOTE_RELEASE_A if waiting else NOTE_AT_REST


def _cut(text: str, limit: int = NOTE_MAX_CHARS) -> str:
    """text as a note: printable ASCII of at most limit characters.

    Any other character becomes "?". A longer text loses its middle: its first
    _NOTE_HEAD characters stay, then NOTE_CUT, then as much of its end as
    fits. limit is at least the length of those two.
    """
    text = "".join(char if " " <= char <= "~" else "?" for char in text)
    if len(text) <= limit:
        return text
    tail = limit - _NOTE_HEAD - len(NOTE_CUT)
    return text[:_NOTE_HEAD] + NOTE_CUT + text[len(text) - tail :]


def _joined(parts: Sequence[str]) -> str:
    """The first of these texts, and each next one that still fits in a note."""
    text = parts[0]
    for part in parts[1:]:
        if len(text) + 2 + len(part) <= NOTE_MAX_CHARS:
            text += "; " + part
    return text


def _shown(state: RobotState | None) -> tuple[Any, ...] | None:
    """What a refusal is tied to: state's session, owner, STOP and motion."""
    if state is None:
        return None
    return state.session_id, state.owner_number, state.stop_number, state.motion


def _sample_age(value: Any) -> float | None:
    """A sample age as a finite number of 0 or more; None for anything else."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def jaw_opening(trigger: Any) -> float | None:
    """The jaw opening a trigger value asks for; None when it is no finite number.

    1.0 minus the value, with the two dead zones: 1.0 at or below
    TRIGGER_OPEN_MAX, 0.0 at or above TRIGGER_CLOSED_MIN.
    """
    value = _finite(trigger)
    if value is None:
        return None
    if value <= TRIGGER_OPEN_MAX:
        return 1.0
    if value >= TRIGGER_CLOSED_MIN:
        return 0.0
    return 1.0 - value


def _finite(value: Any) -> float | None:
    """value as a finite float; None for anything else."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:  # a whole number too large for a float
        return None
    return number if math.isfinite(number) else None


def _joints(values: Any, n_joints: int) -> tuple[float, ...] | None:
    """values as n_joints finite floats; None when they are not that."""
    try:
        q = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(q) != n_joints or not all(math.isfinite(value) for value in q):
        return None
    return q


def _measured(state: RobotState, n_joints: int) -> tuple[float, ...] | None:
    """The measured pose in joint order; None unless every joint has a valid reading."""
    joints = sorted(state.joints, key=lambda sample: sample.joint)
    if [sample.joint for sample in joints] != list(range(n_joints)):
        return None
    if not all(sample.valid for sample in joints):
        return None
    return _joints([sample.value for sample in joints], n_joints)


def _human_facts(state: RobotState | None) -> dict[str, Any]:
    """health["human"] of a ROBOT_STATE; empty when the robot app reports none."""
    facts = None if state is None else state.health.get("human")
    return facts if isinstance(facts, dict) else {}


def _reference(
    state: RobotState, human: dict[str, Any], measured: tuple[float, ...]
) -> tuple[tuple[float, ...], bool]:
    """What a joint target may not run ahead of, and whether that moves on.

    The robot app's last joint target once its drive for the person has sent a
    setpoint: the drive moves it on at the speed clamp. Else the measured
    pose, which stands: before that the target is the last owner's, which the
    arm may hold short of.
    """
    if human.get("driving") and state.move_index > 0 and state.target_q is not None:
        target = _joints(state.target_q, len(measured))
        if target is not None:
            return target, True
    return measured, False


def _named_speed(welcome: Welcome | None) -> float:
    """The human joint speed limit a WELCOME names, rad/s; else UNNAMED_JOINT_SPEED."""
    speed = None if welcome is None else _finite(welcome.human_max_joint_speed)
    return speed if speed is not None and speed > 0.0 else UNNAMED_JOINT_SPEED


def _named_period(welcome: Welcome | None, own_s: float) -> float:
    """The setpoint period a WELCOME names, s; else this program's own period."""
    period = None if welcome is None else _finite(welcome.setpoint_period_s)
    return period if period is not None and period > 0.0 else own_s


def _corners(named: Any, width: int, margin: float) -> np.ndarray | None:
    """A box a WELCOME names, margin inside its walls: its two corners.

    An axis narrower than twice the margin is held at its middle. None: what
    is named is not two corners of width finite numbers, the first no higher
    than the second.
    """
    if named is None:
        return None
    try:
        corners = np.array(named, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if corners.shape != (2, width) or not np.all(np.isfinite(corners)):
        return None
    if np.any(corners[0] > corners[1]):
        return None
    middle = 0.5 * (corners[0] + corners[1])
    low = np.minimum(corners[0] + margin, middle)
    high = np.maximum(corners[1] - margin, middle)
    return np.array([low, high])


def _named_floor(welcome: Welcome | None) -> float | None:
    """The height a WELCOME names under which the gripper tip is low, m."""
    return None if welcome is None else _finite(welcome.human_tip_floor)


def _named_box(welcome: Welcome | None) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Where a WELCOME lets the pose solved for be, m, and what its bottom is.

    The lowest and the highest corner: the tip box it names, and the height
    it names for a low tip as the bottom where that is higher, each
    TIP_BOX_MARGIN_M inside. True when the bottom is that height. A side
    that nothing bounds is infinite. None: it names neither in a usable form.
    """
    named = None if welcome is None else welcome.human_tip_box
    corners = _corners(named, 3, TIP_BOX_MARGIN_M)
    floor = _named_floor(welcome)
    if corners is None and floor is None:
        return None
    if corners is None:
        low, high = np.full(3, -math.inf), np.full(3, math.inf)
    else:
        low, high = corners
    table = floor is not None and bool(floor + TIP_BOX_MARGIN_M > low[2])
    if table:
        low[2] = min(floor + TIP_BOX_MARGIN_M, high[2])
    return low, high, table


def _named_joint_box(
    welcome: Welcome | None, n_joints: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """The joint box a WELCOME names, JOINT_BOX_MARGIN_RAD inside it, rad.

    None: it names none that can be used.
    """
    named = None if welcome is None else welcome.human_joint_box
    corners = _corners(named, n_joints, JOINT_BOX_MARGIN_RAD)
    return None if corners is None else (corners[0], corners[1])


def _room(
    box: tuple[np.ndarray, np.ndarray] | None,
    at: np.ndarray,
    kept: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and the highest values an engagement may go to, by axis.

    For the pose solved for, with `at` the tip of the last joint target, and
    for the joint target, with `at` that target. kept is the room of the
    pass before, None on an engagement's first pass. There the room is the
    box, opened on each axis as far as `at` is outside it. From then on it
    is only narrowed, to `at` where that has come toward the box: what a
    solve leaves outward of its pose is not kept. Without a box nothing is
    bounded.
    """
    if box is None:
        return np.full(len(at), -math.inf), np.full(len(at), math.inf)
    if kept is None:
        return np.minimum(box[0], at), np.maximum(box[1], at)
    low = np.minimum(box[0], np.maximum(kept[0], at))
    high = np.maximum(box[1], np.minimum(kept[1], at))
    return low, high


def _share_inside(
    seed: Sequence[float],
    q: Sequence[float],
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[float, int | None]:
    """The largest share, 0 to 1, of the way from seed to q inside low..high.

    With it the index of the joint that bounds it; None when the whole way
    is inside. A joint outside already may still move toward the inside.
    """
    share, bound = 1.0, None
    for joint, (start, to) in enumerate(zip(seed, q)):
        way = to - start
        if way == 0.0:
            continue
        edge = high[joint] if way > 0.0 else low[joint]
        if (to - edge) * way > 0.0:
            part = max(0.0, (edge - start) / way)
            if part < share:
                share, bound = part, joint
    return share, bound


def _walls_at(
    past: np.ndarray,
    tip: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    named: Sequence[str],
    bottom: str,
) -> tuple[str, ...]:
    """The walls NOTE_AT_LIMIT names on this pass, in the order of the axes.

    past is the hand's position less that position held inside low..high;
    tip is the tip of the last joint target; named is what the pass before
    named; bottom is the name of the lowest wall. A wall is named once the
    hand is WALL_NOTE_M beyond it with the tip within WALL_NOTE_M of it, and
    from then on while the hand is beyond it at all.
    """
    walls = []
    for axis, names in enumerate(WALL_NAMES):
        if axis == 2:
            names = (bottom, names[1])
        sides = ((-1.0, low[axis], names[0]), (1.0, high[axis], names[1]))
        for side, bound, name in sides:
            need = 0.0 if name in named else WALL_NOTE_M
            if side * past[axis] > need and abs(tip[axis] - bound) <= WALL_NOTE_M:
                walls.append(name)
    return tuple(walls)


def _share_within(
    seed: Sequence[float],
    q: Sequence[float],
    reference: Sequence[float],
    room: float,
) -> float:
    """The largest share, 0 to 1, of the way from seed to q that the room allows.

    No joint may go further than room from reference in the direction it
    moves; one that is further already may still move toward reference.
    """
    share = 1.0
    for start, to, held_to in zip(seed, q, reference):
        way = to - start
        if way != 0.0:
            edge = held_to + math.copysign(room, way)
            share = min(share, max(0.0, (edge - start) / way))
    return share


def _ahead_step(cm: float, shown: int | None) -> int | None:
    """The distance the catching-up note gives for a hand cm ahead; None: no note.

    shown is what the note of the pass before gave.
    """
    if shown is not None:
        step = AHEAD_NOTE_STEP_CM
        above = step if shown < step else shown + step
        if shown - AHEAD_NOTE_SLACK_CM <= cm < above + AHEAD_NOTE_SLACK_CM:
            return shown
    if cm < AHEAD_NOTE_CM:
        return None
    if cm < AHEAD_NOTE_STEP_CM:
        return AHEAD_NOTE_CM
    return int(cm // AHEAD_NOTE_STEP_CM) * AHEAD_NOTE_STEP_CM


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """The unit axis and the angle, 0 to pi, of a 3x3 rotation."""
    twice = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sine = 0.5 * float(np.linalg.norm(twice))
    cosine = 0.5 * (float(np.trace(rotation)) - 1.0)
    angle = math.atan2(sine, cosine)
    if sine > 1e-6:
        return twice / (2.0 * sine), angle
    if cosine > 0.0:
        return np.array([1.0, 0.0, 0.0]), angle  # no turn: any axis
    # A half turn: the axis is in the diagonal, its signs beside it.
    axis = np.sqrt(np.clip(0.5 * (np.diag(rotation) + 1.0), 0.0, 1.0))
    most = int(np.argmax(axis))
    for index in range(3):
        if index != most and rotation[most, index] + rotation[index, most] < 0.0:
            axis[index] = -axis[index]
    return axis / float(np.linalg.norm(axis)), angle


def _turn(axis: np.ndarray, angle: float) -> np.ndarray:
    """The 3x3 rotation by angle about a unit axis."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def _toward(
    at: np.ndarray, goal: np.ndarray, metres: float, rad: float, slack: float
) -> np.ndarray:
    """goal, or the pose on the way to it at most metres and rad from `at`.

    The position and the turn are bounded each by itself, as the rate limiter
    bounds them. A goal within slack of a bound is given as it is.
    """
    pose = np.array(goal, dtype=np.float64)
    way = pose[:3, 3] - at[:3, 3]
    far = float(np.linalg.norm(way))
    if far > metres + slack:
        pose[:3, 3] = at[:3, 3] + way * (metres / far)
    axis, angle = _axis_angle(at[:3, :3].T @ pose[:3, :3])
    if angle > rad + slack:
        pose[:3, :3] = at[:3, :3] @ _turn(axis, rad)
    return pose
