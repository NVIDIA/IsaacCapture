# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No torque cut away from measured rest, shown on the whole robot app.

conftest.py checks every release of every test and test_wiring.py counts the call
sites of release_torque. Here: every command in every state away from rest, seeded
random command sequences, and a scan of every call that could cut torque.

A person is in them too. A takeover makes a person the control owner and the
hand-back ends it at rest; from the one to the other a takeover is open. A human
target is a joint target from the person's hand; the human program, on the
workstation, sends each in a HUMAN_SETPOINT link message. A human drive is one
stretch of human motion toward the newest human target, while the grip of the
person's controller is squeezed.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import random
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import isaaccapture_examples.agentic_remote as agentic_remote
import pytest
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    MESSAGE_TYPES,
    Event,
    HumanSetpoint,
    Release,
    Result,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.robot_app.app import (
    NOTE_SQUEEZE_AGAIN,
    human_target,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.connection import RobotDialer
from isaaccapture_examples.agentic_remote.robot_app.human import DriveEnd
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    CheckerSet,
    JointReading,
    MotionState,
    Owner,
    ReadFailed,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm
from conftest import AppHarness, check_released_at_rest
from test_takeover import human_config, keep_human_fed

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
# A pose the agent may hold, well away from rest.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
# From RAISED at 0.2 rad/s on every joint: 6 and 30 setpoints.
LIFT = [[q + 0.004 * (index + 1) for q in RAISED] for index in range(6)]
REACH = [[q + 0.004 * (index + 1) for q in RAISED] for index in range(30)]
# From RAISED at the agent's speed limit: far enough for a contact to show.
FAST = [[q + 0.008 * (index + 1) for q in RAISED] for index in range(40)]
# The threads of a robot app, by the names app.py and gripper.py give them.
APP_THREADS = ("robot-app-dispatcher", "robot-app-motion", "gripper-worker")
# The demo values as they ship, with human driving off, and with it switched on.
SHIPPED = RobotAppConfig.default()
PERSON = human_config(SHIPPED)
# The reasons the human program gives for a takeover and for the hand-back:
# a button of the person's right controller, held down for a second.
RIGHT_B = "right B held 1.0 s"
RIGHT_A = "right A held 1.0 s"


class Done:
    """The on_done of a request made at the Thor terminal."""

    def __init__(self) -> None:
        self.results: list[Result] = []
        self._called = threading.Event()

    def __call__(self, result: Result) -> None:
        self.results.append(result)
        self._called.set()

    def wait(self) -> Result:
        assert self._called.wait(WAIT_S), "the request did not end"
        return self.results[0]


def live(h) -> bool:
    """Whether a command sent over the link can still be answered."""
    return h.app.running and h.robot_end.alive


def answered(h, *command_ids: str) -> list[Result]:
    """Each command's final RESULT, where the link can still carry one."""
    if not live(h):
        return []
    return [h.client.wait_result(one, timeout_s=WAIT_S) for one in command_ids]


def sync(h) -> None:
    """The client quotes the owner number and stop number the robot app has now."""
    if live(h):
        now = h.app.snapshot()
        h.client.wait_state(
            lambda s: (
                s.owner_number >= now.owner_number and s.stop_number >= now.stop_number
            ),
            WAIT_S,
        )


def quiet(h) -> None:
    """Motion work has ended and the dispatcher has made a pass since."""
    h.idle()
    if h.app.running:
        h.step()


def off_rest(h) -> float:
    """How far the measured pose is from rest, in rad."""
    return max(abs(at - rest) for at, rest in zip(h.arm.pose(), h.config.rest.rest_q))


def assert_torque_kept(h) -> None:
    assert h.arm.released_at == [], "torque was released"
    assert h.arm.torque_on(), "the motors are off"


def assert_released_only_at_rest(h) -> None:
    """Torque is on, or it was cut once with the arm measured at rest."""
    check_released_at_rest(h.arm, h.config)
    assert len(h.arm.released_at) <= 1
    assert h.arm.torque_on() == (h.arm.released_at == [])


# A person at the controller


_SEQ = itertools.count(1)


def human_setpoint(h, q: Sequence[float] | None = None, **fields: Any) -> HumanSetpoint:
    """A HUMAN_SETPOINT that quotes the app's numbers as they are now.

    With q the grip is squeezed toward q; without, it is released. The
    controller is ready and its data are 0.02 s old. fields replace any field.
    """
    now = h.app.snapshot()
    values: dict[str, Any] = {
        "session_id": now.session_id,
        "owner_number": now.owner_number,
        "stop_number": now.stop_number,
        "seq": next(_SEQ),
        "source": HUMAN_SOURCE_TELEOP,
        "stage": "ready",
        "sample_age_s": 0.02,
        "engaged": q is not None,
        "grip_released": q is None,
        "q": None if q is None else [float(value) for value in q],
        "jaw_opening": None,
        "state_clock_mono_s": None,
        "note": "",
    }
    values.update(fields)
    return HumanSetpoint(**values)


def feed(h, message: HumanSetpoint | None = None, view: bool = True) -> None:
    """Send a HUMAN_SETPOINT; the app has taken it on return, where it still can.

    Without a message: a ready controller with the grip released. With view a
    live VIEW_STATUS of age 0 goes with it.
    """
    if message is None:
        message = human_setpoint(h)
    h.client.send(message)
    if view:
        h.client.send(ViewStatus(live=True, age_s=0.0))
    answered(h, h.client.checker_state("probe"))  # answered after both were taken


def driving(h) -> bool:
    """Whether a human drive runs."""
    return h.app._human_drive.progress() is not None


class Fed:
    """The harness as keep_human_fed() uses it, for an app that may close.

    AppHarness.step() fails when the dispatcher makes no pass. This step()
    waits only while a human drive runs, and no longer than the app does.
    """

    def __init__(self, h: AppHarness) -> None:
        self._h = h
        self.arm = h.arm

    @property
    def client(self):
        return self._h.client

    def step(self) -> None:
        give_up = time.monotonic() + WAIT_S
        app = self._h.app
        while driving(self._h) and app.running and not app.wait_step(0.05):
            assert time.monotonic() < give_up, "the dispatcher made no pass"


class Hand:
    """A person's hand on the controller: keep_human_fed()'s setpoint.

    It speaks only while a human drive runs. Squeezed, each HUMAN_SETPOINT asks
    for `goal`; the live view reports beside it.
    """

    def __init__(self, h) -> None:
        self.h = h
        # The joint target while the grip is squeezed; None: the grip is released.
        self.goal: list[float] | None = None
        # While squeezed the target runs this far beyond the last setpoint
        # sent, on joint index 0; None: it stays at `goal`.
        self.stride: float | None = None
        # Once the arm was sent this many setpoints in all, `ending` is called;
        # without one the grip is released.
        self.ends_at: int | None = None
        self.ending: Callable[[], None] | None = None
        self.fields: dict[str, Any] = {}  # of every HUMAN_SETPOINT it makes
        self.silent = False  # the human program sends nothing
        # What the live view reports with each HUMAN_SETPOINT; None: nothing.
        self.view: ViewStatus | None = ViewStatus(live=True, age_s=0.0)
        # Clear while squeeze() waits for the app's answer: an ending may cut
        # the link that answer comes over.
        self._answered = threading.Event()
        self._answered.set()

    def __call__(self) -> HumanSetpoint | None:
        if not driving(self.h):
            return None  # a setpoint of other work: no drive waits for the hand
        assert self._answered.wait(WAIT_S), "the squeeze was not answered"
        return self.next()

    def next(self) -> HumanSetpoint | None:
        """The hand's next HUMAN_SETPOINT; the view's report is sent before it."""
        sent = self.h.arm.sent
        if self.ends_at is not None and len(sent) >= self.ends_at:
            self.ends_at = None
            (self.ending or self.release)()
        if self.view is not None:
            self.h.client.send(self.view)
        if self.silent:
            return None
        if self.goal is not None and self.stride is not None and sent:
            last = sent[-1][0]
            self.goal = [last[0] + self.stride, *last[1:]]
        return human_setpoint(self.h, self.goal, **self.fields)

    def squeeze(
        self,
        goal: Sequence[float],
        sends: int | None = None,
        ending: Callable[[], None] | None = None,
    ) -> None:
        """Squeeze toward goal; the app has taken the first engaged target on return.

        With sends, `ending` is called once that many setpoints went out, on the
        thread that sends them; without an ending the grip is released then.
        """
        self.goal = [float(value) for value in goal]
        self.ends_at = None if sends is None else len(self.h.arm.sent) + sends
        self.ending = ending
        self._answered.clear()
        try:
            feed(self.h, self.next(), view=False)
        finally:
            self._answered.set()

    def release(self) -> None:
        """Release the grip: the next HUMAN_SETPOINT says so."""
        self.goal = None

    def reset(self) -> None:
        """The grip released; the human program and the live view as at first."""
        self.goal = self.stride = self.ends_at = self.ending = None
        self.fields, self.silent = {}, False
        self.view = ViewStatus(live=True, age_s=0.0)


def take_over(h) -> None:
    """A person takes the arm over from nobody: controller data, a view, the request."""
    feed(h)
    (taken,) = answered(h, h.client.request_owner("human", reason=RIGHT_B))
    assert taken.status == "completed", taken.reason
    sync(h)


def hand_of(h) -> Hand:
    """The hand of the person who owns the arm, the grip released; it feeds the app."""
    hand = Hand(h)
    keep_human_fed(Fed(h), hand, view=False)
    feed(h)  # the app sees the grip released with the person the owner
    return hand


def nearby(h, by: float = 0.09) -> list[float]:
    """The measured pose with joint index 0 moved on by `by` rad.

    The default lies under the human step limit of 0.1 rad and is four
    setpoints away at the human speed limit.
    """
    at = h.arm.pose()
    return [at[0] + by, *at[1:]]


def squeeze_into_something(h, hand: Hand) -> None:
    """Squeeze with something in the way of joint index 0, 0.05 rad on.

    The hand keeps moving on, beyond the arm, so the drive ends at the contact.
    """
    h.arm.block_joint(0, h.arm.pose()[0] + 0.05)
    hand.stride = 0.08
    hand.squeeze(nearby(h))


def drive_ends(h) -> list[DriveEnd]:
    """How each human drive ends, from here on."""
    ended: list[DriveEnd] = []
    real = h.app._human_drive.run

    def run(admitted, should_stop) -> DriveEnd:
        ended.append(real(admitted, should_stop))
        return ended[-1]

    h.app._human_drive.run = run
    return ended


# States away from rest. Each builder returns what lets a held move go on.


def nothing_held() -> None:
    return None


def raised_by_the_agent(h) -> None:
    h.arm.set_pose(RAISED)
    h.new_task()
    lifted = h.client.run_move(LIFT, start_q=RAISED)
    assert lifted.status == "completed", lifted.reason


def pushed_off_rest(h) -> Callable[[], None]:
    """The app last measured rest; the arm has been pushed away since."""
    h.arm.set_pose(RAISED)
    return nothing_held


def idle(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    answered(h, h.client.request_owner("nobody"))
    h.client.wait_state(lambda s: s.owner == "nobody")
    return nothing_held


def moving(h) -> Callable[[], None]:
    h.arm.set_pose(RAISED)
    h.new_task()
    reached, release = h.hold_move_after(3)
    move_id = h.client.move(REACH, start_q=RAISED)
    h.client.permit(move_id, len(REACH) - 1)
    assert reached.wait(WAIT_S)
    return release.set


def returning(h) -> Callable[[], None]:
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    h.client.rest()
    assert reached.wait(WAIT_S)
    return release.set


def stopped(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    h.client.stop()
    h.client.wait_state(lambda s: s.stop_number == 1)
    return nothing_held


def fault_latched(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    h.arm.clear_faults()  # a reset would now be taken
    return nothing_held


def waiting(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    answered(h, h.client.request_owner("help_needed", reason="the agent is stuck"))
    h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    return nothing_held


def at_a_contact(h) -> Callable[[], None]:
    h.arm.set_pose(RAISED)
    h.new_task()
    h.arm.block_joint(0, RAISED[0] + 0.01)
    assert h.client.run_move(FAST, start_q=RAISED).reason == "contact"
    h.arm.block_joint(0, None)
    return nothing_held


def jaws_loaded(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    h.arm.jaw_object(0.4)
    (closed,) = answered(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert closed.status == "completed", closed.reason
    return nothing_held


def link_lost(h) -> Callable[[], None]:
    raised_by_the_agent(h)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    return nothing_held


def link_made_again(h) -> Callable[[], None]:
    link_lost(h)
    h.connect()
    return nothing_held


def started_raised(h) -> Callable[[], None]:
    """The app found the arm away from rest at its start: no jaw reference."""
    assert not h.arm.gripper_referenced()
    return nothing_held


def held_by_a_person(h) -> Callable[[], None]:
    """A person took the raised arm over, drove it on and released the grip."""
    idle(h)
    take_over(h)
    hand_of(h).squeeze(nearby(h), sends=5)
    h.idle()
    assert h.app.snapshot().owner is Owner.HUMAN and h.app.takeover_open
    assert h.app.motion_state is MotionState.HOLD
    return nothing_held


def driven_by_a_person(h) -> Callable[[], None]:
    """A human drive is held after its third setpoint; three more would follow."""
    idle(h)
    take_over(h)
    hand = hand_of(h)
    reached, release = h.hold_move_after(len(h.arm.sent) + 3)
    hand.squeeze(nearby(h), sends=6)
    assert reached.wait(WAIT_S)
    assert driving(h)
    return release.set


def stopped_under_a_person(h) -> Callable[[], None]:
    """Right B while a person owned the arm: the takeover stays open."""
    held_by_a_person(h)
    h.client.stop("right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert h.app.snapshot().owner is Owner.NOBODY and h.app.takeover_open
    return nothing_held


def at_a_contact_under_a_person(h) -> Callable[[], None]:
    """A person drove the arm into something, which is gone since."""
    idle(h)
    take_over(h)
    hand = hand_of(h)
    ended = drive_ends(h)
    squeeze_into_something(h, hand)
    h.idle()
    assert [end.status for end in ended] == ["contact"]
    h.arm.block_joint(0, None)
    hand.reset()
    return nothing_held


# The one state the robot app is started in, with the arm at RAISED.
STARTED_RAISED = "started raised, the jaws not referenced"
STATES = {
    STARTED_RAISED: started_raised,
    "pushed off rest": pushed_off_rest,
    "idle": idle,
    "moving": moving,
    "returning to rest": returning,
    "hold after STOP": stopped,
    "fault latched": fault_latched,
    "wait running": waiting,
    "contact hold": at_a_contact,
    "jaws loaded": jaws_loaded,
    "connection lost": link_lost,
    "connection made again": link_made_again,
    "the human owns the arm, holding": held_by_a_person,
    "a human drive runs": driven_by_a_person,
    "a takeover is open and nobody owns the arm": stopped_under_a_person,
    "contact hold under a person": at_a_contact_under_a_person,
}
# The states with a person in them: they need human driving switched on.
PERSON_STATES = {
    "the human owns the arm, holding",
    "a human drive runs",
    "a takeover is open and nobody owns the arm",
    "contact hold under a person",
}
# The states from which a shutdown may bring the arm to rest and release there.
MAY_RETURN = {
    "pushed off rest",
    "idle",
    "hold after STOP",
    "wait running",
    "connection lost",
    "connection made again",
    # With the grip released a confirmed shutdown is taken from a person too,
    # and while a takeover is open also from a hold at a contact.
    "the human owns the arm, holding",
    "a takeover is open and nobody owns the arm",
    "contact hold under a person",
}

# Payloads no link message decodes from; one of them is deeper than JSON may nest.
GARBAGE = (
    b"",
    b"\xff\xfe\x00",
    b"release_torque",
    b"null",
    b"[]",
    b'{"command_id": 7, "session_id": null, "stop_number": "0"}',
    b'{"reason": NaN, "sender": Infinity}',
    b'{"stop_number": 1e999}',
    b"[" * 5000,
)


class Injection:
    """Makes the next arm call of one of the app's threads raise."""

    CALLS = (
        "torque_on",
        "send_target",
        "last_target",
        "read_joint",
        "read_all",
        "hold_measured",
        "faults",
        "temperatures",
        "gripper_referenced",
        "gripper_read",
        "gripper_last_target",
        "gripper_command",
        "gripper_light_hold",
    )

    def __init__(self, arm: SimArm, thread: str) -> None:
        self.raised_in: str | None = None  # the arm call that raised
        self._thread = thread
        self._armed = True
        for name in self.CALLS:
            setattr(arm, name, self._wrap(name, getattr(arm, name)))

    def disarm(self) -> None:
        self._armed = False

    def _wrap(self, name: str, real: Callable) -> Callable:
        def call(*args, **kwargs):
            mine = threading.current_thread().name == self._thread
            if mine and self._armed and self.raised_in is None:
                self.raised_in = name
                raise RuntimeError(f"injected into {name}")
            return real(*args, **kwargs)

        return call


def commands(*sends: Callable) -> Callable:
    """An action: these commands in turn over the link, each one answered."""

    def run(h, finish: Callable[[], None]) -> None:
        for send in sends:
            answered(h, send(h))
            sync(h)

    return run


def a_move(h) -> str:
    """A MOVE from where the arm is measured, with a permit for all of it."""
    start = h.arm.pose()
    path = [[q + 0.004 * (index + 1) for q in start] for index in range(6)]
    move_id = h.client.new_id()
    h.client.permit(move_id, len(path) - 1)
    return h.client.move(path, start_q=start, command_id=move_id)


def stop_from(sender: str) -> Callable:
    def send(h) -> str:
        h.client.stop("stop", sender=sender)
        return h.client.checker_state("probe")  # answered once the STOP was taken

    return send


def refused_and_malformed(h, finish: Callable[[], None]) -> None:
    """Every link message the robot app must refuse or drop, in turn.

    With human driving switched on, the takeover among them is taken where
    nobody owns the arm and the controller data are fresh.
    """
    client = h.client
    sent = [
        client.rest(confirmed=False),
        client.open_jaws(confirmed=False),
        client.request_owner("human"),
        client.request_owner("limp"),
        client.checker_state("release_torque"),
    ]
    for old in ({"session_id": "an-earlier-run"}, {"stop_number": -1}):
        sent += [
            client.release(**old),
            client.shutdown(**old),
            client.reset(**old),
            client.rest(**old),
            client.open_jaws(**old),
            client.gripper(1.0, **old),
            client.move(LIFT, start_q=RAISED, **old),
            client.request_owner("human", **old),
            client.request_owner("handback", **old),
        ]
    client.permit("no-such-move", 10_000)
    client.permit("no-such-move", 10_000, session_id="an-earlier-run")
    for message in (
        ViewStatus(live=False, age_s=30.0),
        client.welcome,
        client.state,
        Result(sent[5], "completed"),
        Event(1, "release", "torque released at rest: the motors are off"),
        # A squeeze with no live view, and one that quotes another session.
        human_setpoint(h, nearby(h)),
        human_setpoint(h, nearby(h), session_id="an-earlier-run"),
    ):
        client.send(message)
    for msg_type in (*MESSAGE_TYPES, 0, 63, 100, 255):
        for payload in GARBAGE:  # a STOP with a bad payload is still a STOP
            h.workstation_end.send_reliable(msg_type, payload)
    answered(h, *sent, client.checker_state("probe"))


def stops_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    def broken(kind: str, text: str, data: dict) -> None:
        raise RuntimeError("listener bug")

    h.app.add_event_listener(broken)
    assert h.app.stop("typed stop") > 0
    h.app.request_stop("Ctrl+C")
    h.app.request_stop("Ctrl+C")  # a second Ctrl+C is a STOP too, nothing more
    h.step()  # the dispatcher has raised both


def rest_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    for confirmed in (False, None, True):
        done = Done()
        h.app.request_rest(confirmed, done)
        done.wait()


def release_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    done = Done()
    h.app.request_release(done)
    assert done.wait().status == "refused"


def shutdown_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    for confirmed in (False, None, True):
        done = Done()
        h.app.request_shutdown(confirmed, done)
        if done.wait().status == "completed":
            assert h.app.wait_closed(WAIT_S)


def lines_typed_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    """Every line but quit, and Ctrl+C, at a real Thor terminal on the app."""
    thor = h.thor_terminal()
    thor.ask("help", answer="help ")
    for line in ("limp", "release torque", "exit"):
        thor.ask(line, answer="unknown command")
    assert "refused" in thor.ask("release", answer="release: ")
    assert "refused" in thor.ask("rest", "no", answer="rest: ")
    thor.ask("stop", answer="[stop] STOP from thor_terminal: typed stop")
    assert "refused" in thor.ask("rest", "stop", answer="rest: ")
    stops = len(thor.written("[stop] STOP from thor_terminal: Ctrl+C"))
    thor.terminal.interrupt()
    thor.terminal.interrupt()  # a second Ctrl+C is a STOP too, nothing more
    thor.wait_line("[stop] STOP from thor_terminal: Ctrl+C", stops + 2)
    assert "refused" in thor.ask("release", answer="release: ")
    assert "refused" in thor.ask("rest", "yes", answer="rest: ")
    thor.ask("rest", "path is clear", answer="rest: ")
    thor.close()  # the terminal is closed: the app runs on


def quit_typed_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    """quit answered no, then confirmed; where nothing is asked it is no command."""
    thor = h.thor_terminal()
    for answer in ("no", "path is clear"):
        if thor.ask("quit", answer, answer="quit: ").startswith("quit: completed"):
            assert h.app.wait_closed(WAIT_S)


def reference_typed_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    """reference answered no, then confirmed, with the setting on."""
    assert h.app.allow_reference
    thor = h.thor_terminal()
    assert "refused" in thor.ask("reference", "no", answer="reference: ")
    told = thor.ask("reference", "jaws are empty", answer="reference: ")
    if told == "reference: completed":
        # With the jaws referenced and opened, a confirmed rest and quit are
        # taken again.
        assert h.arm.gripper_referenced()
        sync(h)
        answered(h, h.client.open_jaws())
        thor.ask("rest", "path is clear", answer="rest: ")
        if thor.ask("quit", "path is clear", answer="quit: ").startswith(
            "quit: completed"
        ):
            assert h.app.wait_closed(WAIT_S)
    thor.close()


def view_reports_and_goes_away(h, finish: Callable[[], None]) -> None:
    """A live view reports its age, says it is not live, then says nothing."""
    for status in (ViewStatus(live=True, age_s=0.2), ViewStatus(live=False, age_s=0.0)):
        h.client.send(status)
        answered(h, h.client.checker_state("probe"))


def shutdown_by_link(h, finish: Callable[[], None]) -> None:
    for confirmed in (False, True):
        for result in answered(h, h.client.shutdown(confirmed=confirmed)):
            if result.status == "completed":
                assert h.app.wait_closed(WAIT_S)
        sync(h)


def close(h, finish: Callable[[], None]) -> None:
    closing = threading.Thread(target=h.app.close, daemon=True)
    closing.start()
    finish()  # close() waits for the motion worker, which a held move keeps busy
    closing.join(WAIT_S)
    assert not closing.is_alive() and h.app.wait_closed(0.0)
    # Closed, the app takes nothing more; the calls must not cut torque either.
    h.app.stop("typed stop")
    h.app.request_stop("Ctrl+C")
    assert h.app.request_rest(True) is not None
    assert h.app.request_release() is not None
    assert h.app.request_shutdown(True) is not None
    h.client.release()
    h.client.shutdown()
    h.app.close()


def connection_lost(h, finish: Callable[[], None]) -> None:
    """The robot connection is cut; what the workstation sends then goes nowhere."""
    if live(h):
        h.lose_link()
    h.client.release()
    h.client.shutdown()
    h.client.stop()
    h.step()


def new_connections(h, finish: Callable[[], None], shutdown_late: bool = False) -> None:
    """A carrier offered to an app that has one, then two new connections.

    On each new one: commands with the numbers of the connection before, a
    command id of the connection before, a release and a move. With
    shutdown_late the SHUTDOWN among them is sent once the two RELEASEs
    before it have ended. A SHUTDOWN that is taken ends the action.
    """
    if live(h):
        offered, _ = loopback_pair()
        assert not h.app.attach_carrier(offered)
        offered.close()
    for _ in range(2):
        before = h.app.snapshot()
        earlier = h.client.release()
        if live(h):
            h.lose_link()
        client = h.connect()
        old = {
            "owner_number": before.owner_number,
            "stop_number": before.stop_number,
        }
        sent = [client.release(command_id=earlier), client.release(**old)]
        if shutdown_late:
            answered(h, *sent)
        shutdown = client.shutdown(**old)
        sent += [
            shutdown,
            client.rest(confirmed=False, **old),
            client.gripper(1.0, **old),
            client.move(LIFT, start_q=RAISED, **old),
            client.release(),
            a_move(h),
        ]
        if client.wait_result(shutdown, timeout_s=WAIT_S).status == "completed":
            # No RELEASE was queued or running when the SHUTDOWN came: the arm
            # is back at rest with the motors off and the app closes. What was
            # sent after the SHUTDOWN may stay unanswered.
            assert h.app.wait_closed(WAIT_S)
            return
        answered(h, *sent)
        sync(h)


def dialer(h, finish: Callable[[], None]) -> None:
    """A RobotDialer on the app: a failed dial, two connections and its close."""
    far_ends = []

    def connect():
        if not far_ends:
            far_ends.append(None)
            raise RuntimeError("cannot connect to 127.0.0.1:5000")
        near, far = loopback_pair()
        far_ends.append(far)
        return near

    dialing = RobotDialer(h.app, connect, h.clock, retry_s=0.2, poll_s=0.001)
    dialing.start()
    try:
        if live(h):
            h.lose_link()
        for _ in range(2):
            events = len(h.heard("connection"))
            # In fake time, until the dialer's next dial is due and was made.
            h.advance_until(
                lambda events=events: len(h.heard("connection")) > events, step_s=0.1
            )
            assert h.app.link_alive
            far_ends[-1].send_reliable(Release.TYPE, b"{}")
            far_ends[-1].break_link()
            h.wait_local_events("connection", events + 2)
        assert dialing.attempts == 3
    finally:
        dialing.close()
    assert not dialing.running


def error_in(thread: str) -> Callable:
    """An action: the thread's next arm call raises, then every thread gets work."""

    def run(h, finish: Callable[[], None]) -> None:
        injection = Injection(h.arm, thread)
        answered(h, h.client.gripper(0.5), h.client.open_jaws(), h.client.release())
        done = Done()
        h.app.request_rest(True, done)
        done.wait()
        finish()
        quiet(h)
        injection.disarm()

    return run


def right_b(h, finish: Callable[[], None]) -> None:
    """Right B is pressed: a STOP, and what it ended has ended."""
    h.client.stop("right B", sender="controller")
    answered(h, h.client.checker_state("probe"))  # answered once the STOP was taken
    finish()
    quiet(h)
    sync(h)


def takeover_asked(h, finish: Callable[[], None]) -> None:
    """OWNER_REQUEST "human": as it is typed, then as right B held down asks it.

    Each time the controller data and the live view are fresh, so that nothing
    but the state decides.
    """
    feed(h)
    answered(h, h.client.request_owner("human", reason="typed at the console"))
    sync(h)
    right_b(h, finish)
    feed(h)
    answered(h, h.client.request_owner("human", reason=RIGHT_B))
    sync(h)


def human_setpoints(h, finish: Callable[[], None]) -> None:
    """HUMAN_SETPOINTs of every kind, each taken before the next one is sent.

    Where a person owns the arm a squeeze among them starts a human drive; it
    has ended before the next one, unless the state holds the motion worker.
    """
    at = h.arm.pose()
    near = nearby(h, 0.03)
    now = h.app.snapshot()
    kinds: tuple[dict[str, Any], ...] = (
        {},  # ready, the grip released
        {"q": near},  # a squeeze
        {"q": near, "jaw_opening": 0.0},  # with the trigger pulled
        {"q": nearby(h, 0.5)},  # beyond the step limit
        {},
        {"q": near, "grip_released": True},  # squeezed and released at once
        {"q": near[:5]},  # a joint short
        {"engaged": True, "grip_released": False},  # no joint values
        {"q": near, "stage": "stale"},
        {"q": near, "sample_age_s": None},
        {"q": near, "sample_age_s": 30.0},
        {"q": near, "source": HUMAN_SOURCE_SYNTHETIC},
        {"q": near, "owner_number": now.owner_number + 1},
        {"q": near, "stop_number": now.stop_number - 1},
        {"q": near, "session_id": "an-earlier-run"},
        {"stage": "no_controller", "sample_age_s": None, "grip_released": False},
        {},
    )
    for kind in kinds:
        feed(h, human_setpoint(h, **kind))
        if finish is nothing_held:
            h.idle()
    # The robot app's own entry for a human target: it needs no link message.
    squeeze = human_setpoint(h, [at[0] - 0.03, *at[1:]])
    h.app.offer_human_target(human_target(squeeze, h.clock.monotonic()))
    if finish is nothing_held:
        h.idle()


def person_takes_over_and_squeezes(h, finish: Callable[[], None]) -> None:
    """Right B and its takeover, the grip released, then squeezed and held.

    Where the takeover is granted a human drive follows; nothing feeds it
    unless the state has a hand, so its controller data run stale.
    """
    right_b(h, finish)
    feed(h)
    answered(h, h.client.request_owner("human", reason=RIGHT_B))
    sync(h)
    feed(h)  # the grip is released with the person the owner
    feed(h, human_setpoint(h, nearby(h)))
    h.idle()


REFERENCE = "reference typed at the Thor terminal"
ACTIONS = {
    "refused and malformed link messages": refused_and_malformed,
    "MOVE": commands(a_move),
    "GRIPPER": commands(
        lambda h: h.client.gripper(0.3), lambda h: h.client.gripper(1.0)
    ),
    "CHECKER_STATE": commands(
        lambda h: h.client.checker_state("clear_contact_spots"),
        lambda h: h.client.checker_state("close_grasp"),
    ),
    "STOP": commands(*(stop_from(sender) for sender in ("console", "agent"))),
    "OWNER_REQUEST agent": commands(lambda h: h.client.request_owner("agent")),
    "OWNER_REQUEST rest": commands(lambda h: h.client.rest()),
    "OWNER_REQUEST nobody": commands(lambda h: h.client.request_owner("nobody")),
    "OWNER_REQUEST help_needed": commands(
        lambda h: h.client.request_owner("help_needed", reason="stuck")
    ),
    "RESET": commands(lambda h: h.client.reset()),
    "RELEASE": commands(lambda h: h.client.release()),
    "OPEN": commands(lambda h: h.client.open_jaws()),
    "SHUTDOWN": shutdown_by_link,
    "stop and Ctrl+C at the Thor terminal": stops_at_the_thor_terminal,
    "rest at the Thor terminal": rest_at_the_thor_terminal,
    "release at the Thor terminal": release_at_the_thor_terminal,
    "shutdown at the Thor terminal": shutdown_at_the_thor_terminal,
    "lines typed at the Thor terminal": lines_typed_at_the_thor_terminal,
    "quit typed at the Thor terminal": quit_typed_at_the_thor_terminal,
    REFERENCE: reference_typed_at_the_thor_terminal,
    "the live view reports and goes away": view_reports_and_goes_away,
    "close()": close,
    "the robot connection lost": connection_lost,
    "new robot connections": new_connections,
    "the dialer": dialer,
    "an error in the dispatcher": error_in(APP_THREADS[0]),
    "an error in the motion worker": error_in(APP_THREADS[1]),
    "an error in the gripper worker": error_in(APP_THREADS[2]),
    "OWNER_REQUEST human": takeover_asked,
    "OWNER_REQUEST handback": commands(
        lambda h: h.client.request_owner("handback", reason=RIGHT_A),
        lambda h: h.client.request_owner("handback", stop_number=-1),
    ),
    "HUMAN_SETPOINT": human_setpoints,
    "a person takes over and squeezes": person_takes_over_and_squeezes,
}
# The actions with a person in them: they need human driving switched on.
PERSON_ACTIONS = {
    "OWNER_REQUEST human",
    "OWNER_REQUEST handback",
    "HUMAN_SETPOINT",
    "a person takes over and squeezes",
}
SHUTDOWNS = {
    "SHUTDOWN",
    "shutdown at the Thor terminal",
    "quit typed at the Thor terminal",
}
# On a new robot connection a SHUTDOWN quotes the numbers of the connection
# before. Where the lost link changed no owner they still stand, and the
# shutdown is refused only as busy, behind the RELEASE sent before it: it is
# taken when that release has ended by then, and the app closes. Every state
# of MAY_RETURN allows for it.
RECONNECTED = "new robot connections"


def sweep() -> list:
    """Every state with every action, and the configuration the pair runs on.

    A pair with a person in it runs with human driving switched on; every
    other pair on the shipped values, where it is off.
    """
    pairs = []
    for action in ACTIONS:
        for state in STATES:
            person = state in PERSON_STATES or action in PERSON_ACTIONS
            values = PERSON if person else SHIPPED
            pairs.append(pytest.param(values, state, action, id=f"{state}-{action}"))
    return pairs


def assert_no_cut_away_from_rest(h, state: str, action: str) -> None:
    """The sweep's check, once the action in the state has ended.

    Torque was kept. Only a shutdown from a state of MAY_RETURN, and the typed
    reference with the quit after it, may have released it, at measured rest.
    """
    referenced = action == REFERENCE and state == STARTED_RAISED
    # Among new robot connections a SHUTDOWN was taken where the app has closed.
    shutdown = action in SHUTDOWNS or (action == RECONNECTED and not h.app.running)
    if (shutdown and state in MAY_RETURN) or referenced:
        # The one way to a release: the arm was first returned to measured rest.
        assert_released_only_at_rest(h)
        if referenced:
            # The reference ran, and the quit after it did return and release.
            assert len(h.arm.released_at) == 1
    else:
        assert_torque_kept(h)


@pytest.mark.parametrize(("config", "state", "action"), sweep())
def test_no_command_in_any_state_cuts_torque_away_from_rest(
    make_robot_app, state, action
):
    # The typed reference is switched on, so that its request is not refused
    # for the setting alone.
    pose = RAISED if state == STARTED_RAISED else None
    h = make_robot_app(pose=pose, allow_reference=True)
    finish = STATES[state](h)
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    assert_torque_kept(h)
    ACTIONS[action](h, finish)
    finish()
    quiet(h)
    assert_no_cut_away_from_rest(h, state, action)


@pytest.mark.parametrize(
    ("config", "state"),
    [
        pytest.param(PERSON if state in PERSON_STATES else SHIPPED, state, id=state)
        for state in STATES
    ],
)
def test_a_shutdown_after_the_release_on_a_new_connection_releases_only_at_rest(
    make_robot_app, state
):
    """New robot connections in the order the sweep rarely sees.

    The SHUTDOWN with the numbers of the connection before finds no RELEASE
    queued or running.
    """
    pose = RAISED if state == STARTED_RAISED else None
    h = make_robot_app(pose=pose, allow_reference=True)
    finish = STATES[state](h)
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    assert_torque_kept(h)
    new_connections(h, finish, shutdown_late=True)
    finish()
    quiet(h)
    assert_no_cut_away_from_rest(h, state, RECONNECTED)
    # From a state of MAY_RETURN the shutdown was taken, on the first of the two
    # connections: the arm returned to rest before the one release. From every
    # other state it was refused on both.
    taken = state in MAY_RETURN
    assert h.app.running == (not taken)
    assert len(h.arm.released_at) == (1 if taken else 0)


@pytest.mark.parametrize("thread", APP_THREADS)
def test_an_error_in_any_thread_is_a_fault_stop_and_no_shutdown_follows(
    robot_app, thread
):
    h = robot_app
    idle(h)
    assert thread in {one.name for one in threading.enumerate()}
    injection = Injection(h.arm, thread)
    # Work for the gripper worker and for the motion worker.
    answered(h, h.client.open_jaws(), h.client.release())
    quiet(h)
    injection.disarm()
    assert injection.raised_in is not None, f"{thread} made no arm call"
    h.client.wait_state(lambda s: s.stop_level == "fault")
    (refused,) = answered(h, h.client.shutdown())
    assert refused.status == "refused"
    done = Done()
    h.app.request_shutdown(True, done)
    assert done.wait().status == "refused"
    quiet(h)
    assert h.app.running
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    assert_torque_kept(h)


# Ways a return to rest can end short of proven rest


def at_setpoint(h, count: int, act: Callable[[], None]) -> None:
    """Do something on the motion worker right after setpoint number `count`."""
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(h.arm.sent) == count:
            act()

    h.arm.send_target = send_target


def stop_in_step(h) -> None:
    h.client.stop()
    h.step()  # the dispatcher has raised the STOP before the next setpoint


def refuse_setpoint(h, count: int, error: Exception) -> None:
    """Make the driver raise instead of taking setpoint number `count`."""
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        if len(h.arm.sent) == count - 1:
            raise error
        real(q)

    h.arm.send_target = send_target


def slow_read_after(h, count: int, seconds: float) -> None:
    """Make the first joint read after setpoint number `count` take this much longer."""
    real = h.arm.read_joint
    slowed = []

    def read_joint(joint: int) -> JointReading:
        if len(h.arm.sent) == count and not slowed:
            slowed.append(joint)
            h.clock.advance(seconds)
        return real(joint)

    h.arm.read_joint = read_joint


def fail_full_read(h, call: int) -> None:
    """Make the arm's full read number `call` fail."""
    real = h.arm.read_all
    calls = []

    def read_all() -> list[JointReading]:
        calls.append(1)
        if len(calls) == call:
            raise ReadFailed("joint index 3: the read failed")
        return real()

    h.arm.read_all = read_all


ENDINGS = {
    "a STOP": lambda h: at_setpoint(h, 5, lambda: stop_in_step(h)),
    "a contact": lambda h: h.arm.block_joint(0, 0.15),
    "a late setpoint": lambda h: slow_read_after(h, 10, 0.15),
    "lost joint feedback": lambda h: at_setpoint(
        h, 1, lambda: h.arm.fail_reads(1, 10_000)
    ),
    "a driver fault": lambda h: refuse_setpoint(h, 3, ArmFault("overcurrent")),
    "an error in the driver": lambda h: refuse_setpoint(h, 3, RuntimeError("bug")),
    # Held 0.08 rad short of rest: too little for a contact, too much for rest.
    "a joint short of rest": lambda h: h.arm.block_joint(0, 0.08),
    # The move reached rest; the first of the two reads that prove it fails.
    "a failed read at rest": lambda h: fail_full_read(h, 5),
}


@pytest.mark.parametrize("ending", ENDINGS)
def test_a_shutdown_whose_return_ends_short_of_proven_rest_releases_nothing(
    robot_app, ending
):
    h = robot_app
    h.arm.set_pose(RAISED)
    ENDINGS[ending](h)
    (result,) = answered(h, h.client.shutdown())
    assert result.status in ("cancelled", "failed"), result
    quiet(h)
    assert h.app.running
    assert_torque_kept(h)
    if off_rest(h) > h.config.rest.rest_tol_rad:
        # Where the arm ended, a release is refused as well.
        sync(h)
        (refused,) = answered(h, h.client.release())
        assert refused.status == "refused"
        assert_torque_kept(h)


# Seeded random sequences


class Rig:
    """A robot app on a simulated arm of its own, built inside a test."""

    def __init__(self, config, raised: bool = False) -> None:
        """raised: the arm is found away from rest once the app has started."""
        self.config = config
        self.clock = FakeClock()
        rest = list(config.rest.rest_q)
        self.arm = SimArm(n_joints=config.n_joints, clock=self.clock, q0=rest)
        limits = config.limits
        checkers = CheckerSet(
            agent=BasicChecker(limits.agent, self.clock),
            human=BasicChecker(limits.human, self.clock),
            recovery=BasicChecker(limits.recovery, self.clock, watchdog_s=None),
        )
        # (setpoints still to go, what to do then): a STOP in the middle of a move.
        self.at_setpoint: list = []
        self.blocked = False
        self.idle_steps = 0  # steps with the motors off or the connection lost
        real = self.arm.send_target

        def send_target(q: Sequence[float]) -> None:
            real(q)
            if self.at_setpoint:
                self.at_setpoint[0] -= 1
                if self.at_setpoint[0] <= 0:
                    act = self.at_setpoint[1]
                    self.at_setpoint.clear()
                    act()

        self.arm.send_target = send_target
        self.h = AppHarness(config, self.arm, checkers, self.clock).start()
        # Where human driving is switched on: a person's hand, fed to the app
        # in step with each human drive, the takeovers granted and the drives' ends.
        self.hand: Hand | None = None
        self.takeovers = 0
        self.drives: list[DriveEnd] = []
        if config.human.enabled:
            self.hand = Hand(self.h)
            keep_human_fed(Fed(self.h), self.hand, view=False)
            self.drives = drive_ends(self.h)
        if raised:
            self.arm.set_pose(RAISED)

    def spent(self) -> bool:
        """Whether this app has little left to show: closed, or long unable to move."""
        if not (self.arm.torque_on() and self.h.robot_end.alive):
            self.idle_steps += 1
        return not self.h.app.running or self.idle_steps >= 6

    def command(self, send: Callable) -> Result | None:
        results = answered(self.h, send(self.h.client))
        return results[0] if results else None

    def typed(self, request: Callable, *args) -> Result:
        done = Done()
        request(*args, done)
        return done.wait()

    def settle(self) -> None:
        self.at_setpoint.clear()
        quiet(self.h)
        sync(self.h)


def pick(options: Sequence, share: float):
    return options[min(int(share * len(options)), len(options) - 1)]


def play_task(rig: Rig, dice: list[float]) -> None:
    rig.command(lambda c: c.request_owner("agent", timed_return=dice[0] < 0.7))


def play_move(rig: Rig, dice: list[float], stop_after: int = 0) -> None:
    """A MOVE of up to 0.15 rad; some quote old numbers or lack part of a permit."""
    h, client = rig.h, rig.h.client
    start = rig.arm.pose()
    goal = [
        min(0.6, max(-0.6, at + 0.3 * (share - 0.5)))
        for at, share in zip(start, dice[:6])
    ]
    path = client.plan_straight(start, goal, 0.3)
    reach = len(path) - 1 if dice[6] < 0.7 else int(dice[6] * len(path)) - 1
    old = pick([{}, {}, {}, {}, {"owner_number": 0}, {"stop_number": 0}], dice[7])
    if stop_after:
        sender = pick(["console", "agent", "controller", "thor_terminal"], dice[7])

        def stop() -> None:
            if sender == "thor_terminal":
                h.app.stop("typed stop")
            else:
                client.stop("stop", sender=sender)
                h.step()

        rig.at_setpoint[:] = [stop_after, stop]
        reach, old = len(path) - 1, {}
    move_id = client.new_id()
    client.permit(move_id, reach)
    client.move(path, start_q=start, command_id=move_id, **old)
    answered(h, move_id)


def play_stopped_move(rig: Rig, dice: list[float]) -> None:
    play_move(rig, dice, stop_after=1 + int(dice[6] * 6))


def play_gripper(rig: Rig, dice: list[float]) -> None:
    if dice[0] < 0.4:
        rig.arm.jaw_object(pick([None, 0.3, 0.5], dice[1]))
    opening = pick([0.0, 0.0, 0.2, 1.0, 1.0], dice[2])
    mode = pick(["plain", "close_until_contact"], dice[3])
    rig.command(lambda c: c.gripper(opening, mode=mode))


def play_stop(rig: Rig, dice: list[float]) -> None:
    sender = pick(["console", "agent", "controller", "thor_terminal"], dice[0])
    if dice[1] < 0.3:
        rig.h.app.stop("typed stop")
    elif dice[1] < 0.4:
        rig.h.app.request_stop("Ctrl+C")
    else:
        rig.h.client.stop("stop", sender=sender)
        rig.command(lambda c: c.checker_state("probe"))


def play_owner(rig: Rig, dice: list[float]) -> None:
    request = pick(["help_needed", "help_needed", "help_needed", "nobody"], dice[0])
    rig.command(lambda c: c.request_owner(request, reason="stuck"))


def play_rest(rig: Rig, dice: list[float]) -> None:
    confirmed = dice[0] < 0.85
    if dice[1] < 0.3:
        rig.typed(rig.h.app.request_rest, confirmed)
    else:
        rig.command(lambda c: c.rest(confirmed=confirmed))


def play_release(rig: Rig, dice: list[float]) -> None:
    old = {} if dice[0] < 0.85 else {"stop_number": 0}
    if dice[1] < 0.3:
        rig.typed(rig.h.app.request_release)
    else:
        rig.command(lambda c: c.release(**old))


def play_shutdown(rig: Rig, dice: list[float]) -> None:
    confirmed = dice[1] < 0.85
    if dice[0] < 0.4:
        result = rig.typed(rig.h.app.request_shutdown, confirmed)
    else:
        result = rig.command(lambda c: c.shutdown(confirmed=confirmed))
    if result is not None and result.status == "completed":
        assert rig.h.app.wait_closed(WAIT_S)


def play_open(rig: Rig, dice: list[float]) -> None:
    rig.command(lambda c: c.open_jaws(confirmed=dice[0] < 0.8))


def play_reset(rig: Rig, dice: list[float]) -> None:
    rig.arm.clear_faults()
    for joint in range(rig.arm.n_joints):
        rig.arm.fail_reads(joint, 0)
    rig.command(lambda c: c.reset())


def play_checker(rig: Rig, dice: list[float]) -> None:
    action = pick(["clear_contact_spots", "close_grasp", "limp"], dice[0])
    rig.command(lambda c: c.checker_state(action))


def play_clock(rig: Rig, dice: list[float]) -> None:
    """Let time pass; it stops where the assistance wait runs out."""
    h = rig.h
    seconds = pick([0.05, 0.3, 2.0, 15.0, 60.0, 130.0], dice[0])
    waiting = live(h) and h.client.state.assistance_wait_s is not None
    if waiting and dice[1] < 0.6:
        seconds = 130.0  # long enough for the wait to run out
    expired = len(h.heard("expired"))
    while seconds > 1e-9 and len(h.heard("expired")) == expired:
        # The wait counts at most a second per dispatcher pass.
        step = min(1.0, seconds) if waiting else seconds
        h.clock.advance(step)
        h.step()
        seconds -= step
    h.idle()  # a timed return, if one began, has ended


def play_fault(rig: Rig, dice: list[float]) -> None:
    if dice[0] < 0.6:
        rig.arm.set_fault("motor 3 overheated")
    else:
        rig.arm.fail_reads(int(dice[1] * 5.99), pick([1, 2, 40], dice[2]))


def play_garbage(rig: Rig, dice: list[float]) -> None:
    msg_type = pick([*MESSAGE_TYPES, 0, 100], dice[0])
    rig.h.workstation_end.send_reliable(msg_type, pick(GARBAGE, dice[1]))
    rig.command(lambda c: c.checker_state("probe"))


def play_block(rig: Rig, dice: list[float]) -> None:
    """Put something in the way of one joint, or take it away again."""
    joint = int(dice[0] * 5.99)
    rig.blocked = not rig.blocked
    for each in range(rig.arm.n_joints):
        rig.arm.block_joint(each, None)
    if rig.blocked:
        rig.arm.block_joint(joint, rig.arm.pose()[joint] + 0.02)


def play_push(rig: Rig, dice: list[float]) -> None:
    """Someone moves the powered arm by hand: to rest, or away from it."""
    if not rig.arm.torque_on():
        return  # a limp arm lies on its stops at rest
    rest = list(rig.config.rest.rest_q)
    pose = rest if dice[0] < 0.4 else [share - 0.5 for share in dice[1:7]]
    rig.arm.set_pose(pose)


def play_link_loss(rig: Rig, dice: list[float]) -> None:
    if rig.h.robot_end.alive:
        rig.h.lose_link()


def play_new_link(rig: Rig, dice: list[float]) -> None:
    """A new robot connection, or one offered to an app that has a live one."""
    h = rig.h
    if h.app.wants_carrier:
        h.connect()
    else:
        offered, _ = loopback_pair()
        assert not h.app.attach_carrier(offered)
        offered.close()


def play_close(rig: Rig, dice: list[float]) -> None:
    rig.h.app.close()


def play_takeover(rig: Rig, dice: list[float]) -> None:
    """A takeover is asked, mostly as right B asks it: after its STOP."""
    h = rig.h
    if dice[0] < 0.7:
        h.client.stop("right B", sender="controller")
        rig.command(lambda c: c.checker_state("probe"))
    if dice[1] < 0.9:
        feed(h)  # else the controller data and the view are as they were left
    taken = rig.command(lambda c: c.request_owner("human", reason=RIGHT_B))
    if taken is not None and taken.status == "completed":
        rig.takeovers += 1


def play_handback(rig: Rig, dice: list[float]) -> None:
    rig.command(lambda c: c.request_owner("handback", reason=RIGHT_A))


def play_human_setpoint(rig: Rig, dice: list[float]) -> None:
    """One HUMAN_SETPOINT with no hand behind it: a status, a squeeze, a bad one."""
    h = rig.h
    at = rig.arm.pose()
    near = [value + 0.08 * (share - 0.5) for value, share in zip(at, dice[:6])]
    now = h.app.snapshot()
    kinds: list[dict[str, Any]] = [
        {},
        {"q": near},
        {"q": near, "grip_released": True},
        {"q": near[:5]},
        {"q": [value + 0.5 for value in at]},
        {"q": near, "stage": "stale"},
        {"q": near, "sample_age_s": None},
        {"q": near, "source": HUMAN_SOURCE_SYNTHETIC},
        {"q": near, "owner_number": now.owner_number - 1},
        {"q": near, "stop_number": now.stop_number + 1},
        {"stage": "no_controller", "sample_age_s": None, "grip_released": False},
    ]
    feed(h, human_setpoint(h, **pick(kinds, dice[6])), view=dice[7] < 0.8)


def play_human_drive(rig: Rig, dice: list[float]) -> None:
    """The grip is released, then squeezed toward a pose up to 0.09 rad away.

    Most squeezes end with a released grip after up to 21 setpoints; the others
    with a STOP, a human program that goes silent or says that the arm follows
    the hand no longer, or a live view that is not live any more.
    """
    h, hand = rig.h, rig.hand
    hand.reset()

    def stop() -> None:
        if dice[6] < 0.5:
            h.client.stop("right B", sender="controller")
        else:
            h.app.stop("typed stop")

    def silent() -> None:
        hand.silent = True

    def ended() -> None:
        # IK, the inverse kinematics, gives the human program the joint values
        # for a pose of the gripper tip. Without a solution it says that the
        # arm follows the hand no longer.
        note = NOTE_SQUEEZE_AGAIN + "no IK solution"
        hand.fields = {"engaged": False, "grip_released": False, "note": note}

    def not_live() -> None:
        hand.view = ViewStatus(live=False, age_s=0.0)

    ending = pick([None, None, None, None, stop, silent, ended, not_live], dice[7])
    at = rig.arm.pose()
    goal = [value + 0.18 * (share - 0.5) for value, share in zip(at, dice[:6])]
    feed(h)  # the app sees the grip released
    hand.squeeze(goal, 2 + int(dice[6] * 20), ending)
    h.idle()
    hand.reset()


# What a random step does, with its weight.
PLAYS = (
    (play_task, 4.0),
    (play_move, 5.0),
    (play_stopped_move, 1.0),
    (play_gripper, 2.0),
    (play_stop, 0.8),
    (play_owner, 2.0),
    (play_rest, 1.5),
    (play_release, 2.0),
    (play_shutdown, 0.4),
    (play_open, 0.7),
    (play_reset, 1.2),
    (play_checker, 0.3),
    (play_clock, 2.5),
    (play_fault, 0.5),
    (play_garbage, 0.4),
    (play_block, 0.4),
    (play_push, 0.8),
    (play_link_loss, 0.12),
    (play_new_link, 0.3),
    (play_close, 0.08),
)
# What it does as well where a person is at the controller.
PERSON_PLAYS = (
    (play_takeover, 3.0),
    (play_human_drive, 8.0),
    (play_human_setpoint, 1.0),
    (play_handback, 1.5),
)
# The only steps that may end with the motors off.
RELEASERS = (play_release, play_shutdown)
STEPS = 200


class Tally(NamedTuple):
    """What one random sequence held."""

    away: int  # steps that ended with the arm away from rest
    releases: int  # torque releases
    takeovers: int  # takeovers granted
    driven: int  # setpoints that human drives sent


def random_run(config, seed: int, plays: Sequence[tuple] = PLAYS) -> Tally:
    """Play one seeded sequence of these plays; what it held.

    After every step: each release was at measured rest, came from a release or
    a shutdown, and the motors are on wherever the arm is away from rest.
    """
    rng = random.Random(seed)
    plays, weights = zip(*plays)
    tol = config.rest.rest_tol_rad
    history: list[str] = []
    away = releases = 0
    rigs = [Rig(config)]
    try:
        for step in range(STEPS):
            rig = rigs[-1]
            (play,) = rng.choices(plays, weights)
            dice = [rng.random() for _ in range(8)]
            history.append(play.__name__)
            where = f"seed {seed}, step {step}, after {history[-6:]}"
            released = len(rig.arm.released_at)
            play(rig, dice)
            rig.settle()
            check_released_at_rest(rig.arm, config)
            if off_rest(rig.h) > tol:
                away += 1
                assert rig.arm.torque_on(), where
            assert rig.arm.torque_on() == (rig.arm.released_at == []), where
            if len(rig.arm.released_at) > released:
                assert play in RELEASERS and len(rig.arm.released_at) == 1, where
                releases += 1
            if rig.spent():
                rig.h.close()
                rigs.append(Rig(config, raised=len(rigs) % 2 == 1))
    finally:
        rigs[-1].h.close()
    takeovers = sum(rig.takeovers for rig in rigs)
    driven = sum(end.sent for rig in rigs for end in rig.drives)
    return Tally(away, releases, takeovers, driven)


@pytest.mark.parametrize(
    ("config", "plays", "seed"),
    [
        *(pytest.param(SHIPPED, PLAYS, seed, id=str(seed)) for seed in (1, 3, 5, 7, 9)),
        *(
            pytest.param(PERSON, PLAYS + PERSON_PLAYS, seed, id=f"a person-{seed}")
            for seed in (1, 3, 5, 7, 9)
        ),
    ],
)
def test_random_commands_cut_torque_only_on_request_at_measured_rest(
    config, plays, seed
):
    config = dataclasses.replace(
        config, permits=dataclasses.replace(config.permits, first_wait_s=0.02)
    )
    tally = random_run(config, seed, plays)
    # The sequence did hold the arm away from rest and did release it at rest.
    assert tally.away >= 20 and tally.releases >= 1, tally
    if config.human.enabled:
        # A person did take the arm over, more than once, and did drive it.
        assert tally.takeovers >= 3 and tally.driven >= 50, tally


# Source scan: what may cut torque, and where


PACKAGE = Path(agentic_remote.__file__).resolve().parent
# Names that cut motor torque or disable a motor: ArmPort's release, and what
# APAI's drivers and their motor controllers call it. Any use of one counts.
CUTS = frozenset(
    {
        "release_torque",
        "disconnect",
        "disable",
        "disable_all",
        "torque_off",
        "hard_estop",
        "close_bus",
    }
)
# Names harmless objects have too (the app's own stop(), a carrier's close()).
# They count only when taken from an arm, a driver or a motor bus. The driver's
# stop() makes later jaw commands vanish, and its controller's shutdown() and
# close() end the motor session.
CUTS_ON_AN_ARM = frozenset({"stop", "estop", "shutdown", "close"})
ARM_WORDS = frozenset(
    {"arm", "arms", "backend", "driver", "motor", "motors", "ctrl", "bus"}
)


class Site(NamedTuple):
    file: str  # relative to the package
    where: str  # the class and function it is in
    what: str  # the expression


# The only two places that may cut torque.
ALLOWED = [
    # ApaiArm's ArmPort.release_torque: on APAI's drivers the disconnect is what
    # switches the motors off. Nothing else in the class reaches the disconnect,
    # and release_torque itself is called only by the guarded release below.
    Site(
        "robot_app/apai_backend.py",
        "ApaiArm.release_torque",
        "self._backend.disconnect",
    ),
    # The guarded release: taken only on an operator's request, with the current
    # stop number, no fault-stop, no move able to start, the jaws proven empty
    # and two full joint reads at rest.
    Site("robot_app/rest.py", "ReturnToRest._release", "self._arm.release_torque"),
]


def words(expression: str) -> set[str]:
    """The words of an expression: {"self", "arm"} for self._arm."""
    return set("".join(c if c.isalnum() else " " for c in expression.lower()).split())


def cuts_torque(name: str, target: str) -> bool:
    """Whether taking `name` from the expression `target` can cut torque."""
    on_an_arm = not ARM_WORDS.isdisjoint(words(target))
    return name in CUTS or (name in CUTS_ON_AN_ARM and on_an_arm)


class _Finder(ast.NodeVisitor):
    """Collects the attributes that `wanted(name, target)` picks, with their scope."""

    def __init__(self, file: str, wanted: Callable[[str, str], bool]) -> None:
        self.file = file
        self.wanted = wanted
        self.scope: list[str] = []
        self.sites: list[Site] = []

    def _scoped(self, node: ast.AST) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _scoped

    def _found(self, what: str) -> None:
        self.sites.append(Site(self.file, ".".join(self.scope), what))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        target = ast.unparse(node.value)
        if self.wanted(node.attr, target):
            self._found(f"{target}.{node.attr}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # A name as a string reaches the same method through getattr().
        if isinstance(node.value, str) and self.wanted(node.value, ""):
            self._found(repr(node.value))


def sites(source: str, file: str, wanted: Callable[[str, str], bool]) -> list[Site]:
    """Every place in the source that takes a wanted name; nothing is run."""
    finder = _Finder(file, wanted)
    finder.visit(ast.parse(source))
    return finder.sites


def package_sites(wanted: Callable[[str, str], bool] = cuts_torque) -> list[Site]:
    found: list[Site] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        file = path.relative_to(PACKAGE).as_posix()
        found += sites(path.read_text(encoding="utf-8"), file, wanted)
    return found


def test_the_scan_finds_a_torque_cut_however_it_is_written():
    source = """
class Terminal:
    def quit(self):
        self._arm.release_torque()
        cut = self.app._arm.release_torque
        getattr(self._arm, "release_torque")()
        self._backend.stop()
        arm.hard_estop()
        self._ctrl.disable_all()
        self._stops.stop("typed stop", "thor_terminal")
        self._app.stop("typed stop")
        self._harness.estop("latched")
        carrier.close()
"""
    found = sites(source, "terminal.py", cuts_torque)
    assert {site.where for site in found} == {"Terminal.quit"}
    assert [site.what for site in found] == [
        "self._arm.release_torque",
        "self.app._arm.release_torque",
        "'release_torque'",
        "self._backend.stop",
        "arm.hard_estop",
        "self._ctrl.disable_all",
    ]


def test_torque_can_be_cut_at_two_places_in_the_package_and_nowhere_else():
    assert sorted(package_sites()) == ALLOWED


def test_the_app_its_terminal_and_its_entry_point_cut_no_torque_themselves():
    files = {site.file for site in package_sites()}
    assert (PACKAGE / "robot_app" / "app.py").is_file()
    for module in ("app.py", "terminal.py", "__main__.py", "connection.py"):
        assert f"robot_app/{module}" not in files
    assert not any(file.startswith(("protocol/", "testing/")) for file in files)


def test_the_guarded_release_is_asked_for_by_the_release_and_shutdown_work_only():
    # ReturnToRest.release() is reached from the app's RELEASE and SHUTDOWN
    # work and from nothing else: no wait, no stop and no close asks for it.
    def asks(name: str, target: str) -> bool:
        return name == "release" and "rest" in words(target)

    assert [(site.file, site.where) for site in package_sites(asks)] == [
        ("robot_app/app.py", "RobotApp._run_release"),
        ("robot_app/app.py", "RobotApp._run_shutdown"),
    ]


def test_the_one_release_call_stands_behind_its_guards():
    tree = ast.parse((PACKAGE / "robot_app" / "rest.py").read_text(encoding="utf-8"))
    (release,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_release"
    )

    def calls(name: str, inside: ast.AST) -> list[int]:
        return [
            node.lineno
            for node in ast.walk(inside)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]

    # No move can start: the cut is inside the executor's claim_idle() block.
    (claimed,) = (
        node
        for node in ast.walk(release)
        if isinstance(node, ast.With)
        and "claim_idle" in ast.unparse(node.items[0].context_expr)
    )
    (cut,) = calls("release_torque", claimed)
    # The jaws and two reads at rest are checked first, in that block.
    for check in ("_jaws_problem", "_rest_problem"):
        assert [line for line in calls(check, claimed) if line < cut]
    assert calls("release_torque", tree) == [cut]


def test_the_human_drives_one_send_stands_inside_the_executors_claim():
    # The guarded release cuts torque inside the executor's claim_idle() block.
    # A human drive sends inside the same claim, so neither runs during the other.
    def sends(name: str, target: str) -> bool:
        return name == "send_target"

    def claimed(source: str) -> list[bool]:
        """Per call of a method named send_target, by line: whether a claim holds it."""
        tree = ast.parse(source)
        blocks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                "claim_idle" in ast.unparse(item.context_expr) for item in node.items
            )
        ]
        inside = {id(node) for block in blocks for node in ast.walk(block)}
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "send_target"
        ]
        return [id(call) in inside for call in sorted(calls, key=lambda c: c.lineno)]

    source = (PACKAGE / "robot_app" / "human.py").read_text(encoding="utf-8")
    assert claimed(source) == [True]  # one call, inside the claim
    # The name is taken at that call and nowhere else in the file.
    taken = sites(source, "robot_app/human.py", sends)
    assert [(site.where, site.what) for site in taken] == [
        ("HumanDrive.run", "self._arm.send_target")
    ]
    # In the package the arm is sent a target by the move executor, by the human
    # drive and by the APAI backend's two holds at a measured pose only.
    callers = {site.file for site in package_sites(sends)}
    assert callers == {
        "robot_app/executor.py",
        "robot_app/human.py",
        "robot_app/apai_backend.py",
    }
    folders = ("human/", "agent/", "console/", "view/", "link_service/")
    assert not [file for file in callers if file.startswith(folders)]
    # A send beside the claim, or under another name, would be found.
    other = (
        "def run(self):\n"
        "    with self._claim_idle() as idle:\n"
        "        self._arm.send_target(q)\n"
        "    self._arm.send_target(q)\n"
        "    getattr(arm, 'send_target')(q)\n"
    )
    assert claimed(other) == [True, False]
    assert [site.what for site in sites(other, "human/feed.py", sends)] == [
        "self._arm.send_target",
        "self._arm.send_target",
        "'send_target'",
    ]


def test_a_backends_connect_is_reached_only_with_apais_claim_switched_off():
    # The real driver's connect cuts torque. With an operator's confirmation to
    # claim, APAI's driver would cut it on an arm it does not measure at rest.
    def connects(name: str, target: str) -> bool:
        return name == "connect" and "backend" in words(target)

    path = PACKAGE / "robot_app" / "apai_backend.py"
    source = path.read_text(encoding="utf-8")
    assert [(site.where, site.what) for site in package_sites(connects)] == [
        ("ApaiArm.connect", "self._backend.connect")
    ]
    (guarded,) = (
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.With)
        and "_no_confirmation_claim()" in ast.unparse(node.items[0].context_expr)
    )
    inside = sites(ast.unparse(guarded), "apai_backend.py", connects)
    assert [site.what for site in inside] == ["self._backend.connect"]


# The workstation agent side (agent/): the same rule


# The files of the agent side the scan must have read.
AGENT_FILES = (
    "agent/__init__.py",
    "agent/__main__.py",
    "agent/arm_adapter.py",
    "agent/checkahead.py",
    "agent/clear_grid.py",
    "agent/config.py",
    "agent/console_cmd.py",
    "agent/launcher.py",
    "agent/link_client.py",
    "agent/self_mask.py",
    "agent/setpoints.py",
    "agent/task_gate.py",
    "agent/camera/__init__.py",
    "agent/camera/bridge_wire.py",
    "agent/camera/frame_bridge.py",
    "agent/camera/frame_source.py",
    "agent/camera/stereo_camera.py",
)


def test_the_scan_reads_every_file_of_the_agent_side_and_finds_no_torque_cut():
    scanned = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")}
    assert set(AGENT_FILES) <= scanned
    assert [site for site in package_sites() if site.file.startswith("agent/")] == []
    # A cut written into one of these files would be found.
    source = "def end(link, raw_arm):\n    link.release_torque()\n    raw_arm.stop()\n"
    found = sites(source, "agent/launcher.py", cuts_torque)
    assert [site.what for site in found] == ["link.release_torque", "raw_arm.stop"]


def test_only_the_console_command_asks_the_robot_app_for_a_release_or_a_shutdown():
    # RELEASE and SHUTDOWN are the two link messages whose work can end in a
    # torque cut, and the robot app does that only with the arm measured at
    # rest. Under agent/ only the operator's console command sends them.
    def asks(name: str, target: str) -> bool:
        from_a_client = bool(words(target) & {"client", "link", "ws", "made"})
        return name in ("Release", "Shutdown") or (
            name in ("release", "shutdown") and from_a_client
        )

    asked = [site for site in package_sites(asks) if site.file.startswith("agent/")]
    assert asked == [
        Site("agent/console_cmd.py", "_run", "client.release"),
        Site("agent/console_cmd.py", "_run", "client.shutdown"),
    ]
    for file in AGENT_FILES:
        tree = ast.parse((PACKAGE / file).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert not names & {"Release", "Shutdown"}, file
    # The agent's own link has no such call, and the robot link service lets
    # the role "agent" send neither.
    from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
    from isaaccapture_examples.agentic_remote.link_service.service import (
        ROLE_AGENT,
        ROLE_TABLE,
    )
    from isaaccapture_examples.agentic_remote.protocol.messages import Shutdown

    assert not {"release", "shutdown", "rest", "open_jaws", "reset"} & set(
        dir(AgentLink)
    )
    assert not {Release.TYPE, Shutdown.TYPE} & ROLE_TABLE[ROLE_AGENT]


# The operator console (console/) and the live view (view/): the same rule


# The files of the console and the view the scan must have read.
CONSOLE_AND_VIEW_FILES = (
    "console_log.py",
    "console/__init__.py",
    "console/__main__.py",
    "console/console.py",
    "console/status.py",
    "view/__init__.py",
    "view/__main__.py",
    "view/model.py",
    "view/server.py",
)


def test_the_scan_reads_every_file_of_the_console_and_the_view():
    scanned = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")}
    assert set(CONSOLE_AND_VIEW_FILES) <= scanned
    # No file of the two folders is missing from the list.
    assert {file for file in scanned if file.startswith(("console/", "view/"))} <= set(
        CONSOLE_AND_VIEW_FILES
    )
    under = [site for site in package_sites() if site.file in CONSOLE_AND_VIEW_FILES]
    assert under == []
    # A cut written into one of these files would be found.
    source = "def leave(self):\n    self._arm.release_torque()\n"
    found = sites(source, "console/console.py", cuts_torque)
    assert [site.what for site in found] == ["self._arm.release_torque"]


def test_the_console_and_the_view_name_no_release_and_no_shutdown():
    # The console sends RELEASE and SHUTDOWN through the console command's
    # send_command only; the view sends VIEW_STATUS and nothing else.
    for file in CONSOLE_AND_VIEW_FILES:
        tree = ast.parse((PACKAGE / file).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert not names & {"Release", "Shutdown"}, file

    def asks(name: str, target: str) -> bool:
        from_a_client = bool(words(target) & {"client", "link", "ws", "made"})
        return name in ("release", "shutdown") and from_a_client

    asked = [site for site in package_sites(asks) if site.file.startswith("view/")]
    assert asked == []
    asked = [site for site in package_sites(asks) if site.file.startswith("console/")]
    assert asked == []


# The human path, from the controller to the motors: the human program (human/)
# and the Thor's side of it (robot_app/human.py). The same rule.


# The files of the human path the scan must have read.
HUMAN_FILES = (
    "robot_app/human.py",
    "human/__init__.py",
    "human/__main__.py",
    "human/types.py",
    "human/retarget.py",
    "human/ik.py",
    "human/logic.py",
    "human/feed.py",
)


def test_the_scan_reads_every_file_of_the_human_path_and_finds_no_torque_cut():
    scanned = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")}
    assert sorted(set(HUMAN_FILES) - scanned) == []
    # No file of the human program's folder is missing from the list.
    assert {file for file in scanned if file.startswith("human/")} <= set(HUMAN_FILES)
    assert [site for site in package_sites() if site.file in HUMAN_FILES] == []
    # A cut written into one of these files would be found: on the feed's
    # session as on an arm.
    source = "def close(self):\n    self._session.disconnect()\n    self._arm.stop()\n"
    found = sites(source, "human/feed.py", cuts_torque)
    assert [site.what for site in found] == [
        "self._session.disconnect",
        "self._arm.stop",
    ]


def test_the_human_path_names_no_release_and_no_shutdown():
    # The human program sends STOP, OWNER_REQUEST "human" and "handback" and
    # HUMAN_SETPOINT; the Thor's side of the human path sends no link message.
    # A listed file that is not there yet is the scan test's finding above.
    scanned = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")}
    on_disk = {file for file in scanned if file.startswith("human/")}
    on_disk |= scanned & set(HUMAN_FILES)
    assert "robot_app/human.py" in on_disk and "human/logic.py" in on_disk
    for file in sorted(on_disk):
        tree = ast.parse((PACKAGE / file).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert not names & {"Release", "Shutdown"}, file

    # Neither name is taken from any object here, nor held as a string.
    def asks(name: str, target: str) -> bool:
        return name in ("release", "shutdown")

    asked = [site for site in package_sites(asks) if site.file in on_disk]
    assert asked == []
    # Such a name written into one of these files would be found.
    source = "def end(self):\n    self._local.shutdown()\n    getattr(c, 'release')()\n"
    found = sites(source, "human/__main__.py", asks)
    assert [site.what for site in found] == ["self._local.shutdown", "'release'"]
    # The robot link service lets the role "human" send neither.
    from isaaccapture_examples.agentic_remote.link_service.service import (
        ROLE_HUMAN,
        ROLE_TABLE,
    )
    from isaaccapture_examples.agentic_remote.protocol.messages import Shutdown

    assert not {Release.TYPE, Shutdown.TYPE} & ROLE_TABLE[ROLE_HUMAN]


def test_the_arm_adapters_driver_methods_cut_no_torque():
    # APAI calls disconnect() and stop() on its arm at teardown and on a
    # skill stop. On the robot link the first sends nothing and the second at
    # most a STOP, which is a hold.
    from isaaccapture_examples.agentic_remote.agent.arm_adapter import RobotLinkArm

    tree = ast.parse((PACKAGE / "agent" / "arm_adapter.py").read_text("utf-8"))
    (adapter,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RobotLinkArm"
    )
    methods = {
        node.name: node for node in adapter.body if isinstance(node, ast.FunctionDef)
    }

    def link_calls(method: str) -> set[str]:
        return {
            node.func.attr
            for node in ast.walk(methods[method])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and "link" in words(ast.unparse(node.func.value))
        }

    disconnect = methods["disconnect"]
    assert not [node for node in ast.walk(disconnect) if isinstance(node, ast.Call)]
    assert link_calls("stop") == {"stop"}
    assert link_calls("close_task") == {"unpin"}
    # What APAI reads before its teardown: no park, no torque cut at either end.
    assert RobotLinkArm.shutdown_park_required is False
    assert RobotLinkArm.disconnect_preserves_drive_state is True
    assert RobotLinkArm.connect_cuts_torque is False


def _obstacle(rig) -> None:
    def switch_in(count: int) -> None:
        if count == 30:
            rig.obstacle_beyond(-10.0)  # every later setpoint is refused

    rig.checker.before_approve = switch_in


def _latched(rig) -> None:
    def latch(count: int) -> None:
        if count == 30:
            rig.checker.estopped = True

    rig.checker.before_approve = latch


def _after(rig, count: int, act: Callable[[], None]) -> None:
    """act() on another thread once the arm was sent `count` setpoints."""
    reached, release = rig.h.hold_move_after(count)
    release.set()

    def later() -> None:
        assert reached.wait(WAIT_S)
        act()

    threading.Thread(target=later, daemon=True).start()


# What can end a move of the agent, seen from the agent program. Each sets the
# ending up on a SimRig of test_arm_adapter.py before the move starts.
AGENT_ENDINGS = {
    "the check ahead refuses a setpoint": _obstacle,
    "the workstation checker is latched": _latched,
    "Ctrl+C at the agent terminal": lambda rig: _after(rig, 15, rig.adapter.stop),
    "STOP at the Thor terminal": lambda rig: _after(
        rig, 15, lambda: rig.h.app.stop("typed stop", "thor_terminal")
    ),
    "the link is lost": lambda rig: _after(rig, 15, rig.robot_end.break_link),
    "the agent program is gone": lambda rig: _after(rig, 15, rig.link.close),
    "nothing: the move completes": lambda rig: None,
}


@pytest.mark.apai
@pytest.mark.parametrize("ending", AGENT_ENDINGS)
def test_no_ending_of_an_agent_move_cuts_torque_away_from_rest(
    make_realtime_robot_app, ending
):
    """The arm adapter's permit loop against the robot app, in real time."""
    import numpy as np
    from test_arm_adapter import SimRig

    h = make_realtime_robot_app(pose=RAISED, connected=False)
    rig = SimRig(h)
    try:
        rig.own()
        goal = np.array(RAISED)
        goal[0] += 0.2  # 60 setpoints in 1.2 s
        AGENT_ENDINGS[ending](rig)
        try:
            completed = rig.move(goal, 1.2)
        except Exception:
            completed = False
        assert completed is ending.startswith("nothing")
        h.idle()
        sent = len(h.sent())
        assert (sent == 60) if completed else (0 < sent < 60)
        # What the agent program still does at the end of a task and at its
        # own end: a skill's stop, the end of the task, APAI's teardown, the
        # close of its link. None of it may move the arm or cut its torque.
        with rig.adapter.skill_scope():
            rig.adapter.stop()
        rig.adapter.close_task()
        rig.adapter.stop()  # APAI's teardown latch: a person's STOP
        rig.adapter.disconnect()
        rig.link.close()
        if h.app.running and h.robot_end is not None:
            h.idle()
        assert len(h.sent()) == sent
        assert_torque_kept(h)
        assert off_rest(h) > h.config.rest.rest_tol_rad
    finally:
        rig.close()


# What ends a human drive


class DriveEnding(NamedTuple):
    """One way a human drive ends, and what holds afterwards."""

    # The squeeze toward the goal, with what ends it set up: (h, hand, goal).
    squeeze: Callable[[AppHarness, Hand, list[float]], None]
    status: str  # of the drive's end
    owner: Owner  # the control owner afterwards
    config: RobotAppConfig = PERSON


def squeezed(
    sends: int | None = None, act: Callable[[AppHarness, Hand], None] | None = None
) -> Callable[[AppHarness, Hand, list[float]], None]:
    """A squeeze toward the goal; act(h, hand) follows once `sends` setpoints went out.

    Without an act the grip is released then; without sends it stays squeezed.
    """

    def squeeze(h, hand: Hand, goal: list[float]) -> None:
        hand.squeeze(goal, sends, None if act is None else lambda: act(h, hand))

    return squeeze


def says(**fields: Any) -> Callable[[AppHarness, Hand], None]:
    """From then on the human program's HUMAN_SETPOINTs carry these fields."""

    def act(h, hand: Hand) -> None:
        hand.fields = fields

    return act


def _human_program_gone(h, hand: Hand) -> None:
    hand.silent = True


def _view_not_live(h, hand: Hand) -> None:
    hand.view = ViewStatus(live=False, age_s=0.0)


def _view_reports_no_more(h, hand: Hand) -> None:
    hand.view = None  # the last report grows older than human.view_age_limit_s


def _hand_jumps(h, hand: Hand) -> None:
    hand.goal = nearby(h, 0.5)


def _driver_fault(h, hand: Hand, goal: list[float]) -> None:
    refuse_setpoint(h, len(h.arm.sent) + 3, ArmFault("overcurrent"))
    hand.squeeze(goal)


def _late(h, hand: Hand, goal: list[float]) -> None:
    slow_read_after(h, len(h.arm.sent) + 2, 0.15)
    hand.squeeze(goal)


def _agent_program_gone(h, hand: Hand, goal: list[float]) -> None:
    def last_words() -> None:
        # What an agent program may still send as it dies; all of it is refused.
        gone = h.client.request_owner("help_needed", reason="agent program gone")
        answered(h, gone, h.client.request_owner("nobody"), a_move(h))

    at_setpoint(h, len(h.arm.sent) + 3, last_words)
    hand.squeeze(goal, 6)


# Human driving on, with the human joint box ending 0.3 rad up joint index 0:
# the third setpoint toward the goal lies outside it.
_HUMAN_BOX = PERSON.limits.human
BOXED = dataclasses.replace(
    PERSON,
    limits=dataclasses.replace(
        PERSON.limits,
        human=dataclasses.replace(
            _HUMAN_BOX, joint_max=(0.3, *_HUMAN_BOX.joint_max[1:])
        ),
    ),
)
# What the human program says when the arm shall follow the hand no longer
# though the grip is still squeezed; its note says why. LOST: it has no
# controller data.
OVER = {"engaged": False, "grip_released": False}
LOST = {
    **OVER,
    "sample_age_s": None,
    "note": NOTE_SQUEEZE_AGAIN + "controller data lost",
}
# The rows of "what ends a human drive", and three more: controller data that
# are not ready in a human target that still says engaged, a fault the driver
# raises at a setpoint, and a late setpoint.
HUMAN_ENDINGS = {
    "the grip is released": DriveEnding(squeezed(5), "released", Owner.HUMAN),
    "right B": DriveEnding(
        squeezed(3, lambda h, hand: h.client.stop("right B", sender="controller")),
        "stopped",
        Owner.NOBODY,
    ),
    "stop typed at the console": DriveEnding(
        squeezed(3, lambda h, hand: h.client.stop("typed stop", sender="console")),
        "stopped",
        Owner.NOBODY,
    ),
    "stop typed at the Thor terminal": DriveEnding(
        squeezed(3, lambda h, hand: h.app.stop("typed stop")), "stopped", Owner.NOBODY
    ),
    "Ctrl+C at the Thor terminal": DriveEnding(
        squeezed(3, lambda h, hand: h.app.request_stop("Ctrl+C")),
        "stopped",
        Owner.NOBODY,
    ),
    # The human program has its controller data from the Teleop Session Server,
    # Isaac Capture's program that reads the headset and its controllers, in
    # SAMPLES link messages. The ways they stop, as the human program reports
    # them, and the human program itself going away:
    "the Teleop Session Server is gone": DriveEnding(
        squeezed(3, says(**LOST, stage="no_server")), "disengaged", Owner.HUMAN
    ),
    "SAMPLES are stale": DriveEnding(
        squeezed(3, says(**LOST, stage="stale")), "disengaged", Owner.HUMAN
    ),
    "the controller is not active": DriveEnding(
        squeezed(3, says(**LOST, stage="no_controller")), "disengaged", Owner.HUMAN
    ),
    "the human program is gone": DriveEnding(
        squeezed(3, _human_program_gone), "stale", Owner.HUMAN
    ),
    "controller data not ready in a target that says engaged": DriveEnding(
        squeezed(3, says(stage="stale")), "stale", Owner.HUMAN
    ),
    # The hand's target stands still for a second, so the arm does; then the
    # controller is reported not active.
    "the headset's stream stalls": DriveEnding(
        squeezed(50, says(**LOST, stage="no_controller")), "disengaged", Owner.HUMAN
    ),
    "the live view is not live": DriveEnding(
        squeezed(3, _view_not_live), "view", Owner.HUMAN
    ),
    "the live view is too old": DriveEnding(
        squeezed(3, _view_reports_no_more), "view", Owner.HUMAN
    ),
    "the robot connection is lost": DriveEnding(
        squeezed(3, lambda h, hand: h.lose_link()), "stopped", Owner.HUMAN
    ),
    "a target beyond the step limit": DriveEnding(
        squeezed(3, _hand_jumps), "refused", Owner.HUMAN
    ),
    "the human checker refuses a setpoint": DriveEnding(
        squeezed(), "refused", Owner.HUMAN, BOXED
    ),
    "no IK solution": DriveEnding(
        squeezed(3, says(**OVER, note=NOTE_SQUEEZE_AGAIN + "no IK solution")),
        "disengaged",
        Owner.HUMAN,
    ),
    "a contact": DriveEnding(
        lambda h, hand, goal: squeeze_into_something(h, hand), "contact", Owner.HUMAN
    ),
    "a motor fault": DriveEnding(
        squeezed(3, lambda h, hand: h.arm.set_fault("motor 3 overheated")),
        "stopped",
        Owner.NOBODY,
    ),
    "a driver fault at a setpoint": DriveEnding(_driver_fault, "failed", Owner.NOBODY),
    "lost joint feedback": DriveEnding(
        squeezed(2, lambda h, hand: h.arm.fail_reads(0, 10_000)),
        "failed",
        Owner.NOBODY,
    ),
    "a late setpoint": DriveEnding(_late, "late", Owner.HUMAN),
    "the agent program crashes: nothing changes": DriveEnding(
        _agent_program_gone, "released", Owner.HUMAN
    ),
}
# The ones that end in a fault-stop: a hold that is latched.
FAULT_STOPS = {"a motor fault", "a driver fault at a setpoint", "lost joint feedback"}


@pytest.mark.parametrize(
    ("config", "ending"),
    [pytest.param(case.config, name, id=name) for name, case in HUMAN_ENDINGS.items()],
)
def test_no_ending_of_a_human_drive_cuts_torque_away_from_rest(make_robot_app, ending):
    """A person drives the raised arm on the fake clock, and the drive ends."""
    case = HUMAN_ENDINGS[ending]
    h = make_robot_app()
    idle(h)
    take_over(h)
    hand = hand_of(h)
    ended = drive_ends(h)
    goal = nearby(h)
    before = len(h.arm.sent)
    case.squeeze(h, hand, goal)
    h.idle()
    (end,) = ended
    assert end.status == case.status, end
    # The arm was moving: setpoints went out before the end.
    assert end.sent > 0 and len(h.arm.sent) == before + end.sent
    quiet(h)
    # Every end is a hold with the motors on, and the robot app runs on.
    assert h.app.running and h.app.motion_state is MotionState.HOLD
    assert_torque_kept(h)
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    now = h.app.snapshot()
    assert now.owner is case.owner and h.app.takeover_open
    assert (now.stop_level is StopLevel.FAULT) == (ending in FAULT_STOPS)
    # What the person and the human program still do: the squeeze that is
    # still down, a released grip, right A away from rest, and then nothing
    # for longer than human.stage_keep_s. None of it moves the arm or cuts torque.
    sent = len(h.arm.sent)
    hand.reset()
    if end.status != "released":
        feed(h, human_setpoint(h, goal))
        h.idle()
    feed(h)
    answered(h, h.client.request_owner("handback", reason=RIGHT_A))
    h.idle()
    h.advance(h.config.human.stage_keep_s + 1.0)
    assert len(h.arm.sent) == sent
    assert_torque_kept(h)
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    assert h.app.takeover_open


@pytest.mark.parametrize("config", [PERSON], ids=["human driving on"])
def test_a_release_asked_during_a_human_drive_is_refused_and_cuts_nothing(
    make_robot_app,
):
    h = make_robot_app()  # at rest, the jaws referenced and open
    take_over(h)
    hand = hand_of(h)
    ended = drive_ends(h)
    reached, release = h.hold_move_after(2)
    # The drive stays within the rest tolerance: only the drive itself can
    # stand in the way of a release.
    tol = h.config.rest.rest_tol_rad
    hand.squeeze(nearby(h, 0.6 * tol), sends=4)
    assert reached.wait(WAIT_S)
    stop_number = h.app.snapshot().stop_number
    answers: list[str | None] = []
    try:
        assert driving(h) and off_rest(h) <= tol
        asking = threading.Thread(
            target=lambda: answers.append(h.app._rest.release(stop_number)),
            daemon=True,
        )
        asking.start()
        asking.join(WAIT_S)
        assert answers == ["a move is running"]
        assert_torque_kept(h)
        # Each of the release's two guards against a running move holds by
        # itself: the drive has the executor's claim, and the app counts it.
        with h.app._executor.claim_idle() as idle_now:
            assert not idle_now
        assert h.app._rest._move_running()
    finally:
        release.set()
    h.idle()
    assert [(end.status, end.sent) for end in ended] == [("released", 4)]
    assert_torque_kept(h)
    # Nothing else stood in the way: the jaws are proven empty and the arm is
    # measured at rest, the stop number is current and no fault-stop is latched.
    assert off_rest(h) <= tol
    now = h.app.snapshot()
    assert now.stop_number == stop_number and now.stop_level is StopLevel.NONE
    assert h.app._rest.proven_at_rest() is None
    assert_torque_kept(h)
