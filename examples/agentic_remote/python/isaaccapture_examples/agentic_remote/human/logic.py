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
the grip released, so the next squeeze is a new one. Ahead: how far a target
lies ahead of what follows it.

Buttons. A pass has known buttons when the feed gave the graph's outputs with
a known sample age; the controller's pose need not be valid. Right B sends one
STOP per stretch that it is down, on the stretch's first pass with known
buttons. The stretch ends when B is seen up on such a pass, and when two such
passes are more than 1 s apart: B may have been released unseen, so B that is
down then is a STOP again. A takeover and a hand-back need more, a press that
counts: the button was seen up on a fresh pass since the last pass that was
not fresh, and it is held on fresh passes.
"""

from __future__ import annotations

import itertools
import logging
import math
import uuid
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
from .types import FeedStep, GraphPort, GraphStep, HumanSettings, SolverPort

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_logic")

# The STOP that right B sends.
STOP_REASON = "right B"
STOP_SENDER = "controller"

# HUMAN_SETPOINT.note: what the operator does next, or why nothing moves.
NOTE_TAKE_OVER = "hold right B for 1 s with the grip released to take over"
# While right B is down with no press that counts: holding on asks for nothing.
NOTE_RELEASE_B = (
    "release right B, then hold it for 1 s with the grip released to take over"
)
NOTE_RELEASE_FIRST = "release the grip, then squeeze it to move"
NOTE_SQUEEZE = "squeeze the grip to move"
NOTE_SQUEEZE_AGAIN = "release the grip and squeeze again: "  # then why
NOTE_AT_REST = "at rest: hold right A for 1 s to give the arm back"
# A note goes out as printable ASCII of at most this many characters, which
# keeps every HUMAN_SETPOINT under 512 bytes. A longer one loses its middle:
# a refusal ends with what to do about it.
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

# HUMAN_SETPOINT.stage of fresh controller data with an active right controller.
_STAGE_READY = "ready"
# ROBOT_STATE words read here.
_OWNER_HUMAN = "human"
_STOP_LEVEL_NONE = "none"
_MOTION_REST_HOLD = "rest_hold"
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
# How far apart two passes with known buttons may be for a stretch of right B
# to go on. Further apart, B may have been released unseen in between.
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

    @property
    def takeover_open(self) -> bool:
        """A takeover was accepted and no hand-back has ended it yet."""
        return bool(self.human.get("takeover_open"))


class _Button:
    """One button of the right controller, with the hold of its newest press.

    A press counts only when the button was seen up on a fresh pass since the
    last pass that was not fresh; its hold ends on any pass that is not fresh.
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
        self._armed = False
        self._last: GraphStep | None = None
        self._clutch_engaged = False  # after the graph's last pass
        self._engaged = False  # the last pass sent a joint target
        self._home_q: tuple[float, ...] | None = None  # the pose of the last home
        self._q_sent: tuple[float, ...] | None = None  # this engagement's last target
        self._failures = 0  # failed solves in a row
        self._ended = ""  # why the workstation ended the last engagement
        self._moved = False  # an engagement ran while this takeover was open
        self._a = _Button()
        self._b = _Button()
        # Right B on the last pass with known buttons. While it is down, the
        # STOP of its stretch went out.
        self._b_down = False
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

    def counts(self) -> dict[str, int]:
        """How often each of COUNT_KEYS happened so far."""
        return dict(self._counts)

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
        error of the solver or the graph is raised, unless a STOP or an
        OWNER_REQUEST is due: those are returned and the error is logged.
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
            setpoint = self._after_buttons(p)
        except Exception:
            # Nothing follows the hand after a pass that failed.
            self._armed = False
            self._engaged = False
            if not out:
                raise
            _log.exception("the pass failed after its buttons were read")
            return out
        if setpoint is not None:
            out.append(setpoint)
        return out

    def _after_buttons(self, p: _Pass) -> HumanSetpoint | None:
        """The gates, the home and the joint target: this pass's HUMAN_SETPOINT."""
        step = p.step
        gate = self._gate(p)
        if gate is not None:
            self._armed = False
        elif p.released:
            self._armed = True
            self._ended = ""

        if not self._clutch_engaged and p.young and p.measured is not None:
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
                _log.info("engagement ended: the grip was released")
            else:
                self._end("clutch", "the clutch let go of the hand")
        self._engaged = q is not None
        return self._setpoint(p, gate, q)

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
                _log.info("%s: %s", which, result.status)
            else:
                note = f"{which} {result.status}: {result.reason}"
                self._refusal = (which, note, _shown(p.state))
                _log.warning("%s", note)

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
        """Right B's STOP and the OWNER_REQUEST of a held button."""
        out: list[Message] = []
        step = p.step
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
            b_down = bool(step.b)
            known_at, self._known_at = self._known_at, p.now_mono
            if known_at is not None and p.now_mono - known_at > _STRETCH_KEEP_S + _EPS:
                self._b_down = False  # too long unseen: a new stretch
            if b_down and not self._b_down:
                # The first pass of a stretch; a press that counts is on one.
                state = p.state
                self._b_stop = (
                    None if state is None else (state.session_id, state.stop_number)
                )
                self._b_stop_at = p.now_mono
                out.append(Stop(reason=STOP_REASON, sender=STOP_SENDER))
                _log.info("right B is down: STOP sent")
            self._b_down = b_down
        request = self._request(p)
        if request is not None:
            out.append(request)
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
        _log.info("%s: asking for the %s", reason, which)
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
            why = p.human.get("ended") or "the robot app asks for a released grip"
            return "needs_release", str(why)
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
            self._failures = 0
            self._moved = True
            self._refusal = None
            self._counts["engagements"] += 1
            _log.info("engagement started")
        seed = self._q_sent
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
        hand = step.commanded_target[:3, 3] - step.ee_target[:3, 3]
        ahead_m = float(np.linalg.norm(hand))
        if not ahead_m <= ik.max_ahead_m:
            why = f"the hand is {ahead_m * 100.0:.1f} cm ahead of the arm's target"
            self._end("hand_ahead", why)
            return None
        solved = self._solver.solve(step.ee_target, seed)
        q = None if solved.problem else _joints(solved.q, len(seed))
        if q is None:
            problem = solved.problem or "no IK solution (no usable joint values)"
            if problem.startswith(_JUMP):
                self._end("jump", problem)
                return None
            self._failures += 1
            if self._failures >= ik.max_failures:
                self._end("no_solution", problem)
                return None
            q = seed  # one failed pass sends the last joint values again
        else:
            self._failures = 0
        reference = _reference(p.state, p.human, p.measured)
        ahead = max(abs(a - b) for a, b in zip(q, reference))
        if not ahead <= ik.max_ahead_rad:
            why = f"the solution is {ahead:.3f} rad ahead of the arm"
            self._end("solution_ahead", why)
            return None
        self._q_sent = q
        return q

    def _end(self, key: str, why: str) -> None:
        """End the engagement on the workstation; only a new squeeze starts one."""
        self._armed = False
        self._ended = why
        self._q_sent = None
        self._failures = 0
        self._counts[key] += 1
        _log.info("engagement ended: %s", why)

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
            jaw_opening=None,
            state_clock_mono_s=state.clock_mono_s,
            note=_cut(self._note(p, state, gate)),
        )

    def _note(self, p: _Pass, state: RobotState, gate: tuple[str, str] | None) -> str:
        """What the operator does next, or why the arm does not follow."""
        if p.owns and self._engaged:
            return ""
        if p.owns and self._ended:
            return NOTE_SQUEEZE_AGAIN + self._ended
        if self._refusal is not None:
            return self._refusal[1]
        if p.feed.stage != _STAGE_READY:
            return p.feed.note
        at_rest = state.motion == _MOTION_REST_HOLD
        if not p.owns:
            if p.takeover_open and at_rest:
                return NOTE_AT_REST
            # B is down with no press that counts: its hold asks for nothing.
            waiting = self._b_down and self._b.pressed_at is None
            return NOTE_RELEASE_B if waiting else NOTE_TAKE_OVER
        if gate is not None:
            key, why = gate
            if key == "needs_release":
                return NOTE_RELEASE_FIRST
            # The age in the text would change the note on every pass.
            return "no fresh robot state" if key == "state_old" else why
        if p.squeezed:
            return NOTE_RELEASE_FIRST
        # Before the arm was moved, the next step is the squeeze, also at rest.
        return NOTE_AT_REST if at_rest and self._moved else NOTE_SQUEEZE


def _cut(text: str) -> str:
    """text as a note: printable ASCII of at most NOTE_MAX_CHARS characters.

    Any other character becomes "?"; a longer text loses its middle.
    """
    text = "".join(char if " " <= char <= "~" else "?" for char in text)
    if len(text) <= NOTE_MAX_CHARS:
        return text
    return text[:_NOTE_HEAD] + NOTE_CUT + text[-_NOTE_TAIL:]


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
) -> tuple[float, ...]:
    """What a solution may not run ahead of.

    The robot app's last joint target while it drives for the person, else the
    measured pose.
    """
    if human.get("driving") and state.target_q is not None:
        target = _joints(state.target_q, len(measured))
        if target is not None:
            return target
    return measured
