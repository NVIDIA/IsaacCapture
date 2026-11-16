# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The human program on the Thor: attached at the robot app's own socket.

The human program is the program through which a person drives the arm with
the headset's right controller; the Thor is the machine on the robot, where
the robot app talks to the motors. Started with --human-socket PATH the robot
app serves a Unix socket that speaks the local API, as the robot link service's
socket does at the workstation, for one client of the role "human". What that
client sends is handled as when it arrives on the robot connection, the link
between the two machines, and also while that connection is down.

A takeover makes a person the control owner, the one who may move the arm; the
hand-back ends it at rest. A human target is a joint target from the person's
hand, sent in a HUMAN_SETPOINT link message; a human drive is one stretch of
human motion.

The last two tests start the programs as processes of their own; they need
Isaac Capture and APAI and are skipped without them.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import stat
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.console.status import NO_HUMAN_PROGRAM
from isaaccapture_examples.agentic_remote.human.__main__ import load_human_settings
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalRefused,
    LocalSocketInUse,
    connect_local,
    pack_record,
)
from isaaccapture_examples.agentic_remote.link_service.service import (
    link_message_name,
)
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.local_messages import (
    ROLE_HUMAN,
    LocalHello,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HumanSetpoint,
    Reset,
    Result,
    RobotState,
    Stop,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app.app import (
    HANDED_BACK,
    HUMAN_OFF,
    LINK_LOST,
    NO_TAKEOVER,
    SECOND_HUMAN_PROGRAM,
    TAKEN_OVER,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.human_socket import (
    ONLY_HUMAN,
    HumanSocket,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from link_stack import Station, until
from repo_paths import repo_root
from test_takeover import (
    REST,
    RIGHT_A,
    RIGHT_B,
    Hand,
    drives,
    feed,
    final,
    human_config,
    human_facts,
    moved,
    setpoint,
    state_now,
)

from conftest import AppHarness

apai = pytest.mark.apai
link = pytest.mark.link

# Real seconds a test gives another thread, or a process, before it fails.
WAIT_S = 10.0
REST_WAIT_S = 30.0
CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
ROBOT_MODULE = "isaaccapture_examples.agentic_remote.robot_app"
# Where the dropped HUMAN_SETPOINTs of a second human program arrive, and
# where the one that is followed sends: the words of SECOND_HUMAN_PROGRAM.
AT_SOCKET = "at the robot app's socket"
OVER_LINK = "over the robot connection"
NO_DATA = "no fresh controller data: no HUMAN_SETPOINT has arrived"
# What the Thor terminal prints for the STOP of right B, before its number.
STOP_LINE = "[stop] STOP from controller: right B; the arm holds, stop number "
# Threads of the robot app's program and of its socket; none may outlive it.
THREADS = (
    "robot-app-",
    "robot-dialer",
    "gripper-worker",
    "thor-terminal-",
    "link-service-",
)


@pytest.fixture
def config() -> RobotAppConfig:
    """The demo values with human driving switched on.

    In place of conftest's fixture, so that the checkers of this file's apps
    are built with the human joint box that human_config() sets.
    """
    return human_config(RobotAppConfig.default())


@pytest.fixture
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("th")  # a short path, mode 0700


class Thor:
    """A robot app with a socket for the human program, and who attached there."""

    def __init__(self, h: AppHarness, folder: Path, **service_options: Any) -> None:
        self.h = h
        self.path = str(folder / "h.sock")
        assert len(self.path) < 100, f"socket path too long: {self.path}"
        options = {"status_period_s": 0.05, "poll_s": 0.01, **service_options}
        self.socket = HumanSocket(self.path, **options)
        self.socket.start()
        h.app.attach_local(self.socket.carrier, self.socket.attached)
        self.stations: list[Station] = []

    def attach(self, name: str = "human") -> Station:
        """A human program's place at the socket, with WELCOME and a ROBOT_STATE."""
        end = connect_local(self.path, ROLE_HUMAN, name, WAIT_S)
        ws = WorkstationClient(end, self.h.clock, name="human", quotable=self.current)
        station = Station(end, ws, ROLE_HUMAN)
        self.stations.append(station)
        ws.wait_welcome(WAIT_S)
        ws.wait_state(self.current, WAIT_S)
        return station

    def current(self, state: RobotState) -> bool:
        """Whether a ROBOT_STATE has the owner and stop numbers the app has now."""
        now = self.h.app.snapshot()
        return (state.owner_number, state.stop_number) == (
            now.owner_number,
            now.stop_number,
        )

    def close(self) -> None:
        for station in self.stations:
            station.close()
        self.socket.close()


@pytest.fixture
def on_thor(
    make_robot_app: Callable[..., AppHarness], folder: Path
) -> Iterator[Callable[..., tuple[AppHarness, Thor]]]:
    """Builds a robot app on the simulated arm with a socket for the human program.

    on_thor(config=None, connected=True, **service_options): the keywords of
    make_robot_app, then those of the socket's service.
    """
    made: list[Thor] = []

    def make(
        config: RobotAppConfig | None = None,
        connected: bool = True,
        **service_options: Any,
    ) -> tuple[AppHarness, Thor]:
        chosen = {} if config is None else {"config": config}
        h = make_robot_app(connected=connected, **chosen)
        made.append(Thor(h, folder, **service_options))
        return h, made[-1]

    yield make
    for thor in made:
        thor.close()


def newest_seq(h: AppHarness) -> int | None:
    """The seq of the newest human target the app has; None without one."""
    newest = h.app._human_input.newest()
    return None if newest is None else newest.seq


def taken(h: AppHarness, seq: int) -> bool:
    """Whether the app has taken the HUMAN_SETPOINT of this seq, or a later one.

    The seqs of this file's HUMAN_SETPOINTs rise, whoever sends them.
    """
    newest = newest_seq(h)
    return newest is not None and newest >= seq


def feed_at_socket(
    h: AppHarness,
    station: Station,
    message: HumanSetpoint | None = None,
    *,
    view: bool = True,
) -> HumanSetpoint:
    """Send a HUMAN_SETPOINT at the socket; the app has taken it on return.

    Without a message: the status of a ready controller with the grip
    released. With view a live VIEW_STATUS goes over the robot connection
    first: the live view is the operator's browser page at the workstation.
    """
    message = setpoint(h) if message is None else message
    if view:
        now = h.clock.monotonic()  # it is sent now, by the app's clock
        h.client.send(ViewStatus(live=True, age_s=0.0, sent_mono_s=now))
        h.step()
        h.step()
    station.ws.send(message)
    until(lambda: taken(h, message.seq), "the HUMAN_SETPOINT at the robot app")
    h.step()
    return message


def ask_at_socket(station: Station, request: str, **quoted: Any) -> Result:
    """OWNER_REQUEST from the human program at the socket; its final RESULT."""
    reason = RIGHT_B if request == "human" else RIGHT_A
    command_id = station.ws.request_owner(request, reason=reason, **quoted)
    return station.ws.wait_result(command_id, timeout_s=WAIT_S)


def keep_fed_at_socket(h: AppHarness, station: Station, hand: Hand) -> None:
    """Feed the app after every setpoint the arm is sent, in step with the drive.

    As keep_human_fed of test_takeover.py, with the hand's HUMAN_SETPOINT sent
    at the socket. The live view's VIEW_STATUS goes over the robot connection
    while there is one.
    """
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if h.app.link_alive:
            now = h.clock.monotonic()  # it is sent now, by the app's clock
            h.client.send(ViewStatus(live=True, age_s=0.0, sent_mono_s=now))
        message = hand()
        if message is not None:
            station.ws.send(message)
            end = time.monotonic() + 1.0  # one that a STOP overtook never arrives
            while not taken(h, message.seq) and time.monotonic() < end:
                time.sleep(0.001)
        h.step()

    h.arm.send_target = send_target


def person_at_socket(h: AppHarness, station: Station) -> Hand:
    """A person took the arm over at the socket and has released the grip."""
    feed_at_socket(h, station)
    result = ask_at_socket(station, "human")
    assert result.status == "completed", result.reason
    number = result.detail["owner_number"]
    station.ws.wait_state(lambda s: s.owner_number == number, WAIT_S)
    hand = Hand(h)
    keep_fed_at_socket(h, station, hand)
    feed_at_socket(h, station, hand())
    return hand


def squeeze_at_socket(
    h: AppHarness, station: Station, hand: Hand, goal: Sequence[float]
) -> None:
    """Squeeze toward goal; the app has taken the first engaged target on return."""
    hand.goal = [float(value) for value in goal]
    hand.release_at = None
    feed_at_socket(h, station, hand())


def at_socket(state: RobotState) -> bool:
    """Whether a ROBOT_STATE says a human program is attached at the socket."""
    return state.health["human"].get("local_program") is True


def left_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(THREADS)
    ]


# What the human program gets at the socket


def test_a_human_program_at_the_socket_gets_welcome_state_and_its_results(on_thor):
    h, thor = on_thor()
    # The robot app reports its socket, and that nobody is attached there.
    assert state_now(h).health["human"]["local_program"] is False
    station = thor.attach()
    assert station.ws.welcome == h.client.welcome
    assert station.ws.welcome.session_id == h.app.session_id
    assert station.end.status.roles == {"human": 1}
    # Both ends of the robot app get the ROBOT_STATE that says who is attached.
    h.advance(0.1, 0.05)
    shown = h.client.wait_state(at_socket, WAIT_S)
    assert shown.health["human"]["local_program"] is True
    station.ws.wait_state(lambda s: s.seq >= shown.seq and at_socket(s), WAIT_S)

    # A takeover asked for at the socket: its RESULT comes back there.
    feed_at_socket(h, station)
    result = ask_at_socket(station, "human")
    assert (result.status, result.detail["owner"]) == ("completed", "human")
    now = h.app.snapshot()
    assert (now.owner, now.owner_number) == (Owner.HUMAN, result.detail["owner_number"])
    assert h.app.takeover_open
    station.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
    # The robot connection gets that RESULT too, for the operator console, and
    # the EVENT, which the human program does not read and is not sent.
    assert h.client.wait_result(result.command_id, timeout_s=WAIT_S) == result
    text = TAKEN_OVER.format(number=now.owner_number)
    assert h.client.wait_event("owner", timeout_s=WAIT_S).text == text
    assert station.ws.events == [] and station.ws.unexpected == []

    # Once the program has left, the robot app says so.
    station.close()
    until(lambda: not thor.socket.attached(), "the human program detached")
    h.advance(0.1, 0.05)
    assert state_now(h).health["human"]["local_program"] is False


def test_without_the_socket_robot_state_says_nothing_of_it(make_robot_app):
    h = make_robot_app()
    assert "local_program" not in state_now(h).health["human"]
    feed(h)  # a HUMAN_SETPOINT over the robot connection is taken as before
    assert human_facts(h)["seq"] is not None
    assert h.heard("refused") == []


@pytest.mark.parametrize(("enabled", "why"), [(False, HUMAN_OFF), (True, NO_DATA)])
def test_a_request_at_the_socket_is_answered_as_one_on_the_robot_connection(
    on_thor, config, enabled, why
):
    human = dataclasses.replace(config.human, enabled=enabled)
    h, thor = on_thor(dataclasses.replace(config, human=human))
    station = thor.attach()
    refused: list[str] = []
    for request, reason in (("human", why), ("handback", NO_TAKEOVER)):
        over_link = final(h, h.client.request_owner(request, reason=RIGHT_B))
        here = ask_at_socket(station, request)
        assert (over_link.status, over_link.reason) == ("refused", reason)
        assert (here.status, here.reason) == ("refused", reason)
        refused += [f"owner request {request!r} refused: {reason}"] * 2
    # Each refusal was told at the Thor terminal, whichever way it came.
    assert [text for _, text, _ in h.heard("refused")] == refused
    # Numbers that are not the app's are refused with the app's own words.
    stale = ask_at_socket(station, "human", owner_number=99)
    over_link = final(h, h.client.request_owner("human", owner_number=99))
    assert stale.status == "refused" and stale.reason == over_link.reason
    assert ("owner number" in stale.reason) == enabled


def test_a_repeated_command_at_the_socket_is_answered_again_and_not_run_again(
    on_thor,
):
    h, thor = on_thor()
    station = thor.attach()
    for _ in range(2):
        station.ws.request_owner("handback", reason=RIGHT_A, command_id="again-1")
    until(lambda: len(station.ws.results("again-1")) == 2, "the RESULT, twice")
    first, second = station.ws.results("again-1")
    assert first == second and (first.status, first.reason) == (
        "refused",
        NO_TAKEOVER,
    )
    assert len(h.heard("refused")) == 1  # the handler ran once


# What the socket refuses


def test_a_second_human_program_and_every_other_role_are_refused(on_thor):
    h, thor = on_thor()
    thor.attach()
    with pytest.raises(LocalRefused, match="a human program is attached already"):
        connect_local(thor.path, ROLE_HUMAN, "second", WAIT_S)
    for role in ("console", "agent", "viewer", "tool"):
        with pytest.raises(LocalRefused) as refused:
            connect_local(thor.path, role, role, WAIT_S)
        assert str(refused.value) == ONLY_HUMAN
    with pytest.raises(LocalRefused, match="unknown role 'operator'"):
        connect_local(thor.path, "operator", "odd", WAIT_S)
    assert thor.socket.attached() and h.heard("refused") == []


def test_the_human_role_sends_only_its_own_link_messages(on_thor):
    h, thor = on_thor()
    station = thor.attach()
    before = h.app.snapshot()
    reset = station.ws.wait_result(station.ws.reset(), timeout_s=WAIT_S)
    name = link_message_name(Reset.TYPE)
    assert (reset.status, reset.reason) == (
        "refused",
        f"the human role may not send {name}",
    )
    for request in ("rest", "agent", "nobody"):
        result = station.ws.wait_result(
            station.ws.request_owner(request), timeout_s=WAIT_S
        )
        assert (result.status, result.reason) == (
            "refused",
            f'the human role may not send OWNER_REQUEST "{request}"',
        )
    # None of them reached the robot app.
    h.step()
    assert h.heard("refused") == [] and h.app.snapshot() == before
    assert h.arm.sent == [] and h.arm.torque_on()


# A STOP at the socket


def test_a_stop_at_the_socket_ends_a_human_drive_in_a_hold(on_thor):
    h, thor = on_thor()
    station = thor.attach()
    hand = person_at_socket(h, station)
    ended = drives(h)
    reached, release = h.hold_move_after(len(h.arm.sent) + 3)
    squeeze_at_socket(h, station, hand, moved(REST))
    # The grip is squeezed and the arm follows.
    assert reached.wait(WAIT_S), (ended, h.app._human_refusal)
    assert h.app.motion_state is MotionState.MOVING
    before = h.app.snapshot()
    station.ws.stop(reason="right B", sender="controller")
    until(
        lambda: h.app.snapshot().stop_number == before.stop_number + 1,
        "the STOP at the robot app",
    )
    release.set()
    h.idle()
    assert [end.status for end in ended] == ["stopped"]
    assert ended[0].reason.startswith("STOP from controller: right B")
    sent = len(h.arm.sent)
    now = h.app.snapshot()
    assert now.owner is Owner.NOBODY and h.app.takeover_open
    assert h.app.motion_state is MotionState.HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.heard("stop")[-1][1].startswith("STOP from controller: right B; ")
    # The human program is shown the new numbers; the grip still squeezed
    # moves nothing.
    station.ws.wait_state(lambda s: s.stop_number == now.stop_number, WAIT_S)
    feed_at_socket(h, station, hand())
    h.idle()
    assert len(h.arm.sent) == sent and len(ended) == 1


def test_a_stop_at_the_socket_arrives_with_the_robot_connection_down(on_thor):
    h, thor = on_thor()
    station = thor.attach()
    hand = person_at_socket(h, station)
    ended = drives(h)
    reached, release = h.hold_move_after(len(h.arm.sent) + 3)
    goal = moved(REST)
    squeeze_at_socket(h, station, hand, goal)
    assert reached.wait(WAIT_S), (ended, h.app._human_refusal)
    # The robot connection goes down while the arm follows the hand: the
    # drive ends in a hold and the person keeps the arm.
    h.lose_link()
    release.set()
    h.idle()
    assert [end.status for end in ended] == ["stopped"]
    assert LINK_LOST in ended[0].reason
    owned = h.app.snapshot()
    assert owned.owner is Owner.HUMAN and h.app.takeover_open
    assert h.app.motion_state is MotionState.HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    sent = len(h.arm.sent)

    # The human program still gets ROBOT_STATE, and what it sends is taken.
    seq = station.ws.state.seq
    h.advance(0.1, 0.05)
    held = station.ws.wait_state(lambda s: s.seq > seq, WAIT_S)
    assert (held.owner, held.motion, held.torque_on) == ("human", "hold", True)
    hand.release()
    feed_at_socket(h, station, hand(), view=False)
    assert human_facts(h)["needs_release"] is False
    # A new squeeze moves nothing: the live view came over the lost connection.
    squeeze_at_socket(h, station, hand, goal)
    h.idle()
    assert len(h.arm.sent) == sent
    assert [end.status for end in ended] == ["stopped", "view"]
    assert ended[1].reason.startswith("the live view is not fresh")

    # Right B: the STOP needs no robot connection.
    station.ws.stop(reason="right B", sender="controller")
    until(
        lambda: h.app.snapshot().stop_number == owned.stop_number + 1,
        "the STOP at the robot app",
    )
    now = h.app.snapshot()
    assert now.owner is Owner.NOBODY and h.app.takeover_open
    assert h.heard("stop")[-1][1].startswith("STOP from controller: right B; ")
    assert not h.app.link_alive
    assert h.arm.torque_on() and h.arm.released_at == []
    assert len(h.arm.sent) == sent
    shown = station.ws.wait_state(lambda s: s.stop_number == now.stop_number, WAIT_S)
    assert (shown.owner, shown.motion, shown.torque_on) == ("nobody", "hold", True)


def test_a_stop_at_the_socket_of_an_app_that_never_had_a_robot_connection(on_thor):
    h, thor = on_thor(connected=False)
    assert h.app.running and not h.app.link_alive
    station = thor.attach()
    assert station.ws.welcome.session_id == h.app.session_id
    station.ws.stop(reason="right B", sender="controller")
    until(lambda: h.app.snapshot().stop_number == 1, "the STOP at the robot app")
    assert h.heard("stop")[-1][1].startswith("STOP from controller: right B; ")
    station.ws.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    # A STOP with a payload that cannot be read is still a STOP.
    station.end.send_reliable(Stop.TYPE, b"not json")
    until(lambda: h.app.snapshot().stop_number == 2, "the second STOP")
    assert h.arm.torque_on() and h.arm.released_at == []


# The socket holds nothing up


def test_a_client_that_does_not_read_is_cut_off_and_holds_nothing_up(on_thor):
    h, thor = on_thor(send_timeout_s=0.2)
    deaf = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        deaf.connect(thor.path)
        hello = LocalHello(role=ROLE_HUMAN, name="deaf")
        deaf.sendall(pack_record(hello.TYPE, hello.to_payload()))
        until(thor.socket.attached, "the client attached")
        # It never reads: ROBOT_STATE after ROBOT_STATE fills its socket.
        longest = 0.0
        end = time.monotonic() + 2 * WAIT_S
        while thor.socket.attached():
            assert time.monotonic() < end, "the client was not cut off"
            began = time.monotonic()
            h.advance(0.05, 0.05)  # one ROBOT_STATE
            longest = max(longest, time.monotonic() - began)
        # No pass of the dispatcher waited for the socket.
        assert longest < 1.0
        # The robot app served its robot connection all along: a STOP is taken.
        h.client.stop()
        h.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    finally:
        deaf.close()
    # And the socket takes the next human program.
    station = thor.attach("after")
    assert station.ws.state.stop_number == 1


# One human program at a time


def test_one_human_program_at_a_time_is_followed(on_thor):
    h, thor = on_thor()
    station = thor.attach()
    keep_s = h.config.human.stage_keep_s
    # The program at the robot link service sends first: it is followed.
    feed(h, view=False)
    first = newest_seq(h)
    for _ in range(2):
        station.ws.send(setpoint(h))
    # Behind them, a request: once it was answered the two were handled.
    assert ask_at_socket(station, "handback").reason == NO_TAKEOVER
    assert first is not None and newest_seq(h) == first
    told = SECOND_HUMAN_PROGRAM.format(here=AT_SOCKET, there=OVER_LINK)
    assert [text for _, text, _ in h.heard("refused")][:-1] == [told]

    # Once that program was silent for human.stage_keep_s, the other is taken.
    h.advance(keep_s + 0.1, 1.0)
    local = feed_at_socket(h, station, view=False)
    assert newest_seq(h) == local.seq
    # And now the HUMAN_SETPOINTs of the robot connection are the second one's.
    for _ in range(2):
        feed(h, view=False)
    assert newest_seq(h) == local.seq
    told_back = SECOND_HUMAN_PROGRAM.format(here=OVER_LINK, there=AT_SOCKET)
    assert [text for _, text, _ in h.heard("refused")][-1] == told_back
    assert len(h.heard("refused")) == 3
    assert h.arm.sent == []


# The socket's file, and the program's argument


def test_the_socket_is_private_and_its_file_goes_with_it(folder):
    path = folder / "h.sock"
    served = HumanSocket(str(path))
    served.start()
    try:
        assert served.path == str(path) and not served.attached()
        mode = path.stat().st_mode
        assert stat.S_ISSOCK(mode) and stat.S_IMODE(mode) == 0o600
        # A live socket is not taken by a second robot app.
        with pytest.raises(LocalSocketInUse):
            HumanSocket(str(path)).start()
        assert path.exists()
    finally:
        served.close()
    assert not path.exists()
    assert not served.carrier.alive
    with pytest.raises(ValueError, match="needs a path"):
        HumanSocket("")
    assert left_threads() == []


@pytest.mark.parametrize(
    ("more", "problem"),
    [
        (
            ["--human-socket", "h.sock"],
            "--human-socket can be used with --connect only",
        ),
        (
            ["--selftest", "--human-socket", "h.sock"],
            "--human-socket can be used with --connect only",
        ),
        (
            ["--connect", "127.0.0.1:9", "--human-socket", " "],
            "--human-socket needs the path of the socket",
        ),
    ],
)
def test_a_human_socket_argument_that_cannot_be_used_gives_exit_code_2(
    more, problem, capsys, monkeypatch
):
    monkeypatch.setattr(robot_main, "RobotApp", None)  # no app may be built
    argv = ["--config", str(CONFIGS / "robot_app.yaml"), "--arm", "sim", *more]
    assert robot_main.main(argv) == 2
    assert capsys.readouterr().err == f"robot app: {problem}\n"


class NoLinks:
    """Stands in for LinkCarrier in the robot app's program.

    connect() fails while `error` is set; else it gives the robot end of a new
    loopback link, whose other end is in `workstation_ends`.
    """

    def __init__(self) -> None:
        self.error: str | None = "nobody listens"
        self.workstation_ends: list[Any] = []

    def connect(self, endpoint: str, timeout_s: float, *, name: str) -> Any:
        if self.error is not None:
            raise RuntimeError(self.error)
        robot_end, workstation_end = loopback_pair()
        self.workstation_ends.append(workstation_end)
        return robot_end


class ProgramRun:
    """run_connected() on the simulated arm with --human-socket, on a thread."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
        self.links = NoLinks()
        monkeypatch.setattr(robot_main, "LinkCarrier", self.links)
        self.config = RobotAppConfig.default()
        self.clock = SystemClock()
        self.arm = SimArm(
            n_joints=6, clock=self.clock, q0=list(self.config.rest.rest_q)
        )
        self.typed = TypedLines()
        self.lines: list[str] = []
        self._said = threading.Condition()
        self.code: list[int] = []
        self._thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self._thread.start()

    def _run(self, path: str) -> None:
        self.code.append(
            robot_main.run_connected(
                self.config,
                self.arm,
                robot_main.sim_checkers(self.config, self.clock),
                self.clock,
                "robot-link.invalid:1",  # never dialed: NoLinks stands in
                typed=self.typed,
                say=self._say,
                driver="sim",
                human_socket=path,
            )
        )

    def _say(self, line: str) -> None:
        with self._said:
            self.lines.append(line)
            self._said.notify_all()

    def wait_line(self, part: str) -> str:
        def found() -> list[str]:
            return [line for line in self.lines if part in line]

        with self._said:
            assert self._said.wait_for(lambda: bool(found()), WAIT_S), (
                f"no line with {part!r}; said: {self.lines}"
            )
            return found()[0]

    def result(self) -> int:
        self._thread.join(2 * WAIT_S)
        assert self.code, "run_connected() did not return"
        return self.code[0]


def test_the_program_serves_the_socket_without_a_robot_connection(monkeypatch, folder):
    path = str(folder / "h.sock")
    run = ProgramRun(monkeypatch, path)
    run.wait_line(f"robot app: the human program's socket is {path}")
    run.wait_line("robot app: no robot connection: nobody listens")
    end = connect_local(path, ROLE_HUMAN, "human", WAIT_S)
    ws = WorkstationClient(end, run.clock, name="human")
    try:
        assert ws.wait_welcome(WAIT_S).driver == "sim"
        state = ws.wait_state(at_socket, WAIT_S)
        assert (state.owner, state.motion, state.torque_on) == (
            "nobody",
            "rest_hold",
            True,
        )
        # Right B: the Thor terminal tells the STOP, with no robot connection.
        ws.stop(reason="right B", sender="controller")
        assert run.wait_line(STOP_LINE) == f"{STOP_LINE}1"
        ws.wait_state(lambda s: s.stop_number == 1, WAIT_S)
        assert run.links.workstation_ends == []
        # The robot app's end closes the socket and removes its file.
        run.typed.type("quit")
        assert run.result() == 0
        assert end.wait_dead(WAIT_S)
    finally:
        ws.close()
        end.close()
    run.wait_line("quit: completed; the robot app closes")
    assert not os.path.exists(path)
    assert len(run.arm.released_at) == 1  # the one release, at rest
    assert left_threads() == []


def test_a_socket_that_cannot_be_made_ends_the_program_before_the_arm_is_connected(
    monkeypatch, folder, capsys
):
    monkeypatch.setattr(robot_main, "LinkCarrier", NoLinks())
    path = str(folder / "h.sock")
    config, clock = RobotAppConfig.default(), SystemClock()
    arm = SimArm(n_joints=6, clock=clock, q0=list(config.rest.rest_q))
    connects: list[int] = []
    arm.connect = lambda: connects.append(1)
    held = HumanSocket(path)  # another robot app's
    held.start()
    try:
        code = robot_main.run_connected(
            config,
            arm,
            robot_main.sim_checkers(config, clock),
            clock,
            "robot-link.invalid:1",
            typed=iter(()),
            say=lambda line: None,
            driver="sim",
            human_socket=path,
        )
    finally:
        held.close()
    assert code == 2 and connects == []
    assert capsys.readouterr().err == (
        f"robot app: --human-socket {path} cannot be served: a robot link "
        f"service already runs on {path}\n"
    )


# The rig's file for the human program


def test_the_rig_human_file_names_the_thors_socket_and_a_placeholder_server():
    rig = load_human_settings(CONFIGS / "human_rebot.yaml")
    assert rig.socket_path == os.path.expanduser("~/.agentic_remote/human.sock")
    # 192.0.2.1 is a documentation address: nothing answers there.
    assert rig.teleop.endpoint == "192.0.2.1:47600"
    text = (CONFIGS / "human_rebot.yaml").read_text(encoding="utf-8")
    comments = " ".join(
        line.strip().lstrip("# ") for line in text.splitlines() if "#" in line
    )
    for part in (
        "started with --human-socket and this same path",
        "crosses between the two machines",
        "192.0.2.1 is a placeholder: write the workstation's address on the wired "
        "link in before the first run",
        "The server must listen on that address (its --listen",
    ):
        assert part in comments, part
    # The one-machine file keeps the robot link service's default socket.
    demo = load_human_settings(CONFIGS / "human.yaml")
    assert (demo.socket_path, demo.teleop.endpoint) == ("", "127.0.0.1:47600")


# The programs as processes of their own


def thor_files(folder: Path, robot: Path) -> tuple[str, str, Path, Path]:
    """(the service's socket, the robot app's, the service's file, the human file).

    The shipped files with the test's sockets: the service's with a port the
    system picks, the human program's with the robot app's socket and, as its
    Teleop Session Server, a port that this process holds and never listens
    on. The script feed dials nothing.
    """
    from test_demo1_recipe import fitted
    from test_human_program import NO_SERVER

    path, thor_socket = str(folder / "s.sock"), str(folder / "h.sock")
    service = fitted(
        CONFIGS / "link_service.yaml",
        folder,
        {"port: 47610": "port: 0", '\nsocket_path: ""': f"\nsocket_path: {path}"},
    )
    human_file = fitted(
        CONFIGS / "human.yaml",
        folder,
        {
            '\nsocket_path: ""': f"\nsocket_path: {thor_socket}",
            'endpoint: "127.0.0.1:47600"': f'endpoint: "{NO_SERVER}"',
        },
    )
    assert robot.is_file()
    return path, thor_socket, service, human_file


@link
@apai
def test_the_human_program_on_the_thor_runs_as_processes(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Demo 2's recipe with the human program at the robot app's socket.

    The robot link service, the robot app on the simulated arm with
    --human-socket, the operator console, and the human program started by
    run_human.sh on the script "lift", a synthetic controller: timed rows of
    controller data with no headset. What differs from the rig: both machines
    are this one, the service listens on a port the system picks, the console
    is started with python -m, and a local client in the role "tool" reads
    ROBOT_STATE.
    """
    demo2 = pytest.importorskip("test_demo2_programs")
    from isaaccapture_examples.agentic_remote.human import logic
    from isaaccapture_examples.agentic_remote.human.ik import Solver
    from isaaccapture_examples.agentic_remote.protocol.confirm import (
        PATH_IS_CLEAR,
        YES,
    )
    from link_stack import HOST, READY, SERVICE_MODULE, Program
    from test_console import ASK_REST, ASK_YES
    from test_demo1_programs import ConsoleProcess
    from test_demo1_recipe import Watcher

    import numpy as np

    folder = tmp_path_factory.mktemp("th")  # a socket path holds 107 bytes
    path, thor_socket, service_file, human_file = thor_files(folder, demo2.ROBOT_DEMO2)
    solver = Solver.from_apai("mock")
    programs: list[Any] = []
    try:
        service = Program(SERVICE_MODULE, "--config", str(service_file))
        programs.append(service)
        endpoint = service.wait_line(READY).removeprefix(READY).split()[0]
        assert endpoint.startswith(f"{HOST}:")
        # The robot app, as on the Thor: the argument names the socket.
        app = Program(
            ROBOT_MODULE, "--config", str(demo2.ROBOT_DEMO2), "--arm", "sim",
            "--connect", endpoint, "--human-socket", thor_socket,
        )  # fmt: skip
        programs.append(app)
        app.wait_line("robot app: session ")
        app.wait_line(f"robot app: the human program's socket is {thor_socket}")
        desk = ConsoleProcess(path)
        programs.append(desk)
        assert ", driver sim, " in desk.wait("WELCOME: ")
        demo2.console_shows(desk, " | owner nobody #0 | ", " | no human program | ")
        desk.type("state")
        desk.wait(f"takeover: not possible: {NO_HUMAN_PROGRAM}")
        watcher = Watcher(path)
        programs.append(watcher)
        rest = watcher.pose()
        rest_tip = solver.fk(solver.inside(rest))[:3, 3]

        # The human program attaches at the robot app's socket.
        person = demo2.human_script(human_file, "--controller", "synthetic:lift")
        programs.append(person)
        started = person.wait_line('human program: synthetic controller "lift"')
        assert started == demo2.LIFT_START.format(driver="sim")
        # The console: the program is attached, and not at the service.
        attached = demo2.console_shows(desk, " | human program at the robot app | ")
        assert "no human program" not in attached
        assert " | clients 1 console, 1 tool | " in attached
        # Right B, its hold, the squeeze, the lift, the release.
        stop = f"[stop] STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; "
        desk.wait(stop, 2 * WAIT_S)
        app.wait_line(STOP_LINE)
        owned = watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
        taken_over = f"[owner] {TAKEN_OVER.format(number=owned.owner_number)}"
        desk.wait(taken_over)
        owner = f" | owner human #{owned.owner_number} | "
        demo2.console_shows(
            desk, owner, " | moving ", " | human driving | ", wait_s=2 * WAIT_S
        )
        person.wait_line("human program: grip up")
        held = watcher.ws.wait_state(
            lambda s: not demo2.human(s)["driving"] and s.motion == "hold", WAIT_S
        )
        assert (held.owner, held.torque_on) == ("human", True)
        demo2.console_shows(desk, owner, " | hold | ", " | takeover open | ")
        rose = solver.fk(watcher.pose())[:3, 3] - rest_tip
        np.testing.assert_allclose(rose, [0.0, 0.0, 0.05], atol=3e-3)

        # rest and path is clear at the console; right A gives the arm back.
        desk.type("rest")
        desk.answer(ASK_REST, PATH_IS_CLEAR)
        desk.wait("rest: RESULT completed", REST_WAIT_S)
        back = f"[owner] {HANDED_BACK}"
        desk.wait(back)
        ended = watcher.ws.wait_state(
            lambda s: not demo2.human(s)["takeover_open"], WAIT_S
        )
        assert (ended.owner, ended.motion, ended.torque_on) == (
            "nobody",
            "rest_hold",
            True,
        )
        assert watcher.pose() == pytest.approx(rest, abs=0.05)
        demo2.in_order(
            list(desk.said),
            (f"takeover: not possible: {NO_HUMAN_PROGRAM}",),
            (" | human program at the robot app | ",),
            (stop,),
            (taken_over,),
            (owner, " | moving ", " | human driving | "),
            (owner, " | hold | ", " | takeover open | "),
            ("rest: RESULT completed",),
            (back,),
        )

        # The end: Ctrl+C at the human program, then the console's shutdown.
        person.signal(signal.SIGINT)
        assert person.wait_exit() == 0
        demo2.console_shows(
            desk, f" | owner nobody #{ended.owner_number} | ", " | no human program | "
        )
        desk.type("shutdown")
        desk.answer(f"{ASK_YES}shut down", YES)
        assert app.wait_exit() == 0
        assert not os.path.exists(thor_socket)
        desk.wait("ROBOT CONNECTION LOST: ")
        desk.type("quit")
        assert desk.wait_exit() == 0
        watcher.close()
        service.signal(signal.SIGINT)
        assert service.wait_exit() == 0
        assert "Traceback" not in "\n".join(person.lines + app.lines)
        for name, lines in (
            ("Thor terminal", app.lines),
            ("console", desk.said),
            ("human program", person.lines),
        ):
            demo2.transcript(name, lines)
    finally:
        for program in programs:
            program.close()


@link
@apai
def test_right_b_reaches_the_robot_app_process_with_the_robot_connection_down(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The robot app and the human program alone: nobody listens for the app.

    The robot app runs on configs/robot_app_human_check.yaml, which keeps the
    live view as the rig's file does: the STOP of right B arrives, and the
    takeover that right B asks for a second later is refused.
    """
    demo2 = pytest.importorskip("test_demo2_programs")
    from link_stack import Program

    folder = tmp_path_factory.mktemp("th")
    _, thor_socket, _, human_file = thor_files(folder, demo2.ROBOT_HUMAN_CHECK)
    # A port this process holds and never listens on: every dial is refused.
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    programs: list[Any] = []
    try:
        app = Program(
            ROBOT_MODULE, "--config", str(demo2.ROBOT_HUMAN_CHECK), "--arm", "sim",
            "--connect", f"127.0.0.1:{held.getsockname()[1]}",
            "--human-socket", thor_socket,
        )  # fmt: skip
        programs.append(app)
        app.wait_line(f"robot app: the human program's socket is {thor_socket}")
        app.wait_line("robot app: no robot connection: ")
        person = demo2.human_script(human_file, "--controller", "synthetic:lift")
        programs.append(person)
        # WELCOME and ROBOT_STATE came without a robot connection.
        started = person.wait_line('human program: synthetic controller "lift"')
        assert started == demo2.LIFT_START.format(driver="sim")
        assert app.wait_line(STOP_LINE, 2 * WAIT_S).endswith(f"{STOP_LINE}1")
        refused = app.wait_line("[refused] owner request 'human' refused: ")
        assert "the live view is not fresh: no VIEW_STATUS has arrived" in refused
        # quit at the Thor terminal: the arm lies at rest, the app closes, and
        # the human program ends when its socket does.
        app.type("quit")
        assert app.wait_exit() == 0
        assert not os.path.exists(thor_socket)
        assert person.wait_exit() == 1
        assert any(
            line.startswith("human program: the robot link service is gone")
            for line in person.lines
        )
        assert "Traceback" not in "\n".join(person.lines + app.lines)
        for name, lines in (
            ("Thor terminal", app.lines),
            ("human program", person.lines),
        ):
            demo2.transcript(name, lines)
    finally:
        for program in programs:
            program.close()
        held.close()
