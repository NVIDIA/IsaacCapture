# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No torque cut away from measured rest, shown on the whole robot app.

conftest.py checks every release of every test and test_wiring.py counts the call
sites of release_torque. Here: every command in every state away from rest, seeded
random command sequences, and a scan of every call that could cut torque.
"""

from __future__ import annotations

import ast
import dataclasses
import random
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import isaaccapture_examples.agentic_remote as agentic_remote
import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MESSAGE_TYPES,
    Event,
    Result,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    CheckerSet,
    JointReading,
    ReadFailed,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm
from conftest import AppHarness, check_released_at_rest

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


STATES = {
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
}
# The states from which a shutdown may bring the arm to rest and release there.
MAY_RETURN = {
    "pushed off rest",
    "idle",
    "hold after STOP",
    "wait running",
    "connection lost",
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
    """Every link message the robot app must refuse or drop, in turn."""
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
        ]
    client.permit("no-such-move", 10_000)
    client.permit("no-such-move", 10_000, session_id="an-earlier-run")
    for message in (
        ViewStatus(live=False, age_s=30.0),
        client.welcome,
        client.state,
        Result(sent[5], "completed"),
        Event(1, "release", "torque released at rest: the motors are off"),
    ):
        client.send(message)
    for msg_type in (*MESSAGE_TYPES, 0, 63, 99, 255):
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
    thor.ask("rest", "yes", answer="rest: ")
    thor.close()  # the terminal is closed: the app runs on


def quit_typed_at_the_thor_terminal(h, finish: Callable[[], None]) -> None:
    """quit answered no, then yes; where nothing is asked the answer is no command."""
    thor = h.thor_terminal()
    for answer in ("no", "yes"):
        if thor.ask("quit", answer, answer="quit: ").startswith("quit: completed"):
            assert h.app.wait_closed(WAIT_S)


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
    "close()": close,
    "an error in the dispatcher": error_in(APP_THREADS[0]),
    "an error in the motion worker": error_in(APP_THREADS[1]),
    "an error in the gripper worker": error_in(APP_THREADS[2]),
}
SHUTDOWNS = {
    "SHUTDOWN",
    "shutdown at the Thor terminal",
    "quit typed at the Thor terminal",
}


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("state", STATES)
def test_no_command_in_any_state_cuts_torque_away_from_rest(robot_app, state, action):
    h = robot_app
    finish = STATES[state](h)
    assert off_rest(h) > 2 * h.config.rest.rest_tol_rad
    assert_torque_kept(h)
    ACTIONS[action](h, finish)
    finish()
    quiet(h)
    if action in SHUTDOWNS and state in MAY_RETURN:
        # The one way to a release: the arm was first returned to measured rest.
        assert_released_only_at_rest(h)
    else:
        assert_torque_kept(h)


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
    msg_type = pick([*MESSAGE_TYPES, 0, 99], dice[0])
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
        rig.h.robot_end.break_link()
        rig.h.wait_local_event("connection")


def play_close(rig: Rig, dice: list[float]) -> None:
    rig.h.app.close()


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
    (play_close, 0.08),
)
# The only steps that may end with the motors off.
RELEASERS = (play_release, play_shutdown)
STEPS = 200


def random_run(config, seed: int) -> tuple[int, int]:
    """Play one seeded sequence; (steps that ended away from rest, releases).

    After every step: each release was at measured rest, came from a release or
    a shutdown, and the motors are on wherever the arm is away from rest.
    """
    rng = random.Random(seed)
    plays, weights = zip(*PLAYS)
    tol = config.rest.rest_tol_rad
    history: list[str] = []
    away = releases = rigs = 0
    rig = Rig(config)
    try:
        for step in range(STEPS):
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
                rigs += 1
                rig = Rig(config, raised=rigs % 2 == 1)
    finally:
        rig.h.close()
    return away, releases


@pytest.mark.parametrize("seed", [1, 3, 5, 7, 9])
def test_random_commands_cut_torque_only_on_request_at_measured_rest(config, seed):
    config = dataclasses.replace(
        config, permits=dataclasses.replace(config.permits, first_wait_s=0.02)
    )
    away, releases = random_run(config, seed)
    # The sequence did hold the arm away from rest and did release it at rest.
    assert away >= 20 and releases >= 1, (away, releases)


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
    for module in ("app.py", "terminal.py", "__main__.py"):
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
