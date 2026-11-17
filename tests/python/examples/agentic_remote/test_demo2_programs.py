# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The programs of demo 2 together, on one machine.

Demo 2: a task of the agent fails, a person takes the arm over, moves it,
brings it to rest and gives it back, and the agent's next task runs. A takeover
makes a person the control owner, the one who may move the arm; the hand-back
ends it at rest. The human program is the workstation program that turns the
right controller's motion into joint targets; here its controller is a
synthetic one: a script of timed rows, with no headset. The arm follows the
hand only while the grip is squeezed, and a human drive is one such stretch of
motion on the robot app's side. The live view is the browser page that shows
the camera picture; the robot app lets a person drive only while that page
acknowledges the pictures it draws.

The robot link service listens on 127.0.0.1 and a port the system picked, and
the robot app dials it over a real link. Attached to the service: the human
program's main() on a thread, the operator console (the Console class, its
lines typed by the test), the live view's main() on a thread with an HTTP
client that plays the browser tab, and, where a test needs it, the agent
program in this process on APAI's mock scene. The robot app runs in real time,
on the simulated arm or on APAI's mock arm.

The last two tests start the programs as processes of their own and run the
recipe "Run demo 2 on one machine" of the README: on the simulated arm, and
with a task of the agent that fails first, on APAI's mock arm. The file is
skipped without Isaac Capture.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import http.client
import os
import signal
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("isaaccapture.retargeters")

import isaaccapture  # noqa: E402
from isaaccapture_examples.agentic_remote import console_log  # noqa: E402
from isaaccapture_examples.agentic_remote.agent import (  # noqa: E402
    console_cmd,
    launcher,
    task_gate,
)
from isaaccapture_examples.agentic_remote.agent.link_client import (  # noqa: E402
    AgentLink,
)
from isaaccapture_examples.agentic_remote.console.console import (  # noqa: E402
    BELL_CHAR,
)
from isaaccapture_examples.agentic_remote.console.status import (  # noqa: E402
    NO_HUMAN_PROGRAM,
    TAKEOVER_POSSIBLE,
)
from isaaccapture_examples.agentic_remote.human import (  # noqa: E402
    __main__ as human_main,
)
from isaaccapture_examples.agentic_remote.human import feed as feed_module  # noqa: E402
from isaaccapture_examples.agentic_remote.human import logic  # noqa: E402
from isaaccapture_examples.agentic_remote.human.ik import Solver  # noqa: E402
from isaaccapture_examples.agentic_remote.link_service.local import (  # noqa: E402
    LocalCarrier,
    connect_local,
)
from isaaccapture_examples.agentic_remote.link_service.service import (  # noqa: E402
    AGENT_GONE,
)
from isaaccapture_examples.agentic_remote.protocol.confirm import (  # noqa: E402
    PATH_IS_CLEAR,
    YES,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (  # noqa: E402
    DRIVER_APAI_MOCK,
    DRIVER_REAL,
    DRIVER_SIM,
    REQUEST_HUMAN,
    HumanSetpoint,
    Message,
    OwnerRequest,
    RobotState,
    Stop,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app import (  # noqa: E402
    app as app_module,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (  # noqa: E402
    ConfigError,
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.gripper import (  # noqa: E402
    CLOSE_UNTIL_CONTACT,
    NOT_MATCHED,
    OPEN_EFFORT,
    PLAIN,
)
from isaaccapture_examples.agentic_remote.robot_app.human import (  # noqa: E402
    SPEED_MARGIN,
)
from isaaccapture_examples.agentic_remote.robot_app.rest import (  # noqa: E402
    EVENT_REST,
)
from isaaccapture_examples.agentic_remote.robot_app.stop import (  # noqa: E402
    STOP_EVENT,
)
from isaaccapture_examples.agentic_remote.robot_app.waits import (  # noqa: E402
    EXPIRED,
)
from isaaccapture_examples.agentic_remote.view import (  # noqa: E402
    __main__ as view_main,
)
from isaaccapture_examples.agentic_remote.view.model import NO_PICTURE  # noqa: E402
from link_stack import (  # noqa: E402
    HOST,
    READY,
    SERVICE_MODULE,
    Program,
    Stack,
    StartScript,
    Station,
)
from test_agent_end_to_end import (  # noqa: E402
    RUN_BUDGET_S,
    ObstacleGrid,
    Scene,
    raise_arm,
)
from test_console import ASK_REST, ASK_YES  # noqa: E402
from test_demo1_programs import (  # noqa: E402
    PATH,
    REST,
    VIEW_MODULE,
    ConsoleProcess,
    Desks,
    TimedDesk,
)
from test_demo1_recipe import (  # noqa: E402
    AGENT_MOCK,
    CLEAR_WARNING,
    INVENTED_WARNING,
    ROBOT_MOCK,
    START_WAIT_S,
    AgentTerminal,
    Watcher,
    fitted,
    recipe_folder,  # noqa: F401  (a fixture of that file)
)
from test_human_program import BASE, CONFIGS, HAND, NO_SERVER, UP, flat  # noqa: E402
from test_human_program import Running as HumanProgram  # noqa: E402
from test_human_program import script as script_feed  # noqa: E402
from test_human_program import shipped as shipped_file  # noqa: E402
from test_human_program import write_config as write_human_config  # noqa: E402
from test_launcher import PICK, QUICK, scene_robot_config  # noqa: E402
from test_link_programs import ROBOT_MODULE  # noqa: E402
from test_takeover import RIGHT_B, human_config  # noqa: E402
from test_view_program import Browser, StandInSource  # noqa: E402

from conftest import APAI_SRC_ENV, apai_runtime_env  # noqa: E402

apai = pytest.mark.apai
link = pytest.mark.link

# Real seconds a test gives a program before it fails.
WAIT_S = 10.0
# Real seconds a return to rest may take here.
REST_WAIT_S = 30.0
# And from the mock scene's home pose, 1.2 rad from rest at 0.1 rad/s.
RECIPE_REST_WAIT_S = 60.0
# How often the browser tab of a test acknowledges a picture.
ACK_S = 0.25
# human.view_age_limit_s of the robot apps here. The age the Thor, the machine
# on the robot, sees is that of the acknowledged picture plus the time since
# the VIEW_STATUS arrived: with the shipped 0.5 s it would swing past the limit.
VIEW_AGE_LIMIT_S = 2.0
# The view's settings here: 15 pictures and 10 VIEW_STATUS per second.
VIEW_BLOCKS: dict[str, dict[str, Any]] = {
    "picture": {"fps": 15.0},
    "status": {"period_s": 0.1, "ack_timeout_s": 1.0, "state_max_age_s": 1.0},
}
# The owners a ROBOT_STATE names.
NOBODY, HUMAN, AGENT, RECOVERY = "nobody", "human", "agent", "recovery"
# How the robot app's EVENT of a refused takeover begins; its reason follows.
TAKEOVER_REFUSED = f"owner request {REQUEST_HUMAN!r} refused: "
# The human program's report line, printed once a second: how it begins, and
# its counts while the program has ended no engagement itself (for an IK
# solution, a target too far ahead or the hand's speed). An engagement is the
# stretch in which the arm follows the squeezed hand, as that program sees it.
REPORT_START = f"{human_main.PROGRAM}: stage "
_NO_REPORT = human_main.report_line("", [], [], None, None, [], {})
NONE_ENDED = "ended: " + _NO_REPORT.partition("ended: ")[2]


def wait_for(
    read: Callable[[], Any],
    what: str,
    last: Callable[[], Any] | None = None,
    wait_s: float = WAIT_S,
) -> Any:
    """Wait in real time until read() gives something true; returns it.

    A failure says what last() gave at the end, read() itself without it.
    """
    end = time.monotonic() + wait_s
    while True:
        got = read()
        if got:
            return got
        if time.monotonic() >= end:
            seen = got if last is None else last()
            raise AssertionError(f"{what}: not within {wait_s:g} s; last: {seen!r}")
        time.sleep(0.005)


def console_shows(desk: Any, *parts: str, wait_s: float = WAIT_S) -> str:
    """Wait for a line of a console that holds every part; returns the first.

    desk keeps the console's lines in .said: a console of this process or one
    that runs as a process of its own. A failure says its last lines.
    """

    def found() -> list[str]:
        return [line for line in list(desk.said) if all(p in line for p in parts)]

    what = f"a console line with {parts}"
    return wait_for(found, what, lambda: list(desk.said)[-6:], wait_s)[0]


def console_has(
    desk: Any, shown: Callable[[RobotState], Any], what: str, wait_s: float = WAIT_S
) -> RobotState:
    """Wait until a console's newest ROBOT_STATE is one `shown` accepts.

    desk is a console of this process; a command it sends quotes that state.
    A failure says the state the console had at the end.
    """

    def has() -> RobotState | None:
        state = desk.console._state
        return state if state is not None and shown(state) else None

    what = f"the console's ROBOT_STATE with {what}"
    return wait_for(has, what, lambda: desk.console._state, wait_s)


def help_lines(said: list[str], last: str) -> list[str]:
    """The console's help block that ends with the line `last`.

    The block is the line "HELP NEEDED: ..." and the indented lines behind it.
    The console prints its status line from another thread, so one can stand
    between two lines of the block; a status line is not indented.
    """
    end = said.index(last)
    start = max(at for at in range(end) if said[at].startswith("HELP NEEDED: "))
    indented = [line for line in said[start + 1 : end + 1] if line.startswith("  ")]
    return [said[start], *indented]


class Cues:
    """Waits of a script that the test ends.

    row(name) is a script row that holds the script's clock until give(name).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch = monkeypatch
        self._given: dict[str, threading.Event] = {}

    def row(self, name: str) -> dict[str, str]:
        given = self._given.setdefault(name, threading.Event())
        word = f"cue {name}"
        # The feed's own table of what a script may wait for, for this test.
        self._patch.setitem(feed_module.WAITS, word, lambda state: given.is_set())
        return {"wait_for": word}

    def give(self, name: str) -> None:
        self._given[name].set()


class Script:
    """The rows of a synthetic controller, written in the order they happen.

    Each step is a row some seconds of the script's clock after the one before.
    The script's clock starts with the human program's first pass and stands
    while a wait does not hold.
    """

    def __init__(self, cues: Cues) -> None:
        self.rows: list[dict[str, Any]] = [dict(BASE)]
        self._cues = cues
        self._t = 0.0
        self._hand = np.array(HAND, dtype=float)

    def then(self, after_s: float, **values: Any) -> Script:
        """A row after_s after the one before; it keeps what it does not name."""
        self._t = round(self._t + after_s, 3)
        self.rows.append({"t": self._t, **values})
        return self

    def wait(self, word: str) -> Script:
        """Hold the clock until the newest ROBOT_STATE shows what the feed knows."""
        self.rows.append({"wait_for": word})
        return self

    def cue(self, name: str) -> Script:
        """Hold the clock until the test gives this cue."""
        self.rows.append(self._cues.row(name))
        return self

    def take_over(self) -> Script:
        """Right B goes down and stays down until the person owns the arm."""
        return self.then(0.3, b=True).wait("owner_human").then(0.2, b=False)

    def squeeze(self) -> Script:
        return self.then(0.4, squeeze=1.0)

    def lift(self, metres: float, seconds: float, **at_start: Any) -> Script:
        """The hand rises this far in this long, at one speed.

        at_start are values that change as the hand sets off.
        """
        self.then(0.2, **at_start)  # where the hand's way begins
        self._hand = self._hand + metres * np.array(UP)
        return self.then(seconds, pos=[float(value) for value in self._hand])

    def release(self) -> Script:
        return self.then(0.2, squeeze=0.0)

    def hand_back(self, done: str) -> Script:
        """Right A goes down once the arm holds at rest, and up at the cue."""
        return self.wait("rest_hold").then(0.3, a=True).cue(done).then(0.2, a=False)


class Tap:
    """connect_local for the human program's main(), keeping what it made.

    .carrier is the program's end of the local API; .sent are the link messages
    the program sent through it, in order.
    """

    def __init__(self) -> None:
        self.carrier: LocalCarrier | None = None
        self.sent: list[Message] = []

    def __call__(self, path: str, role: str, name: str, timeout_s: float) -> Any:
        carrier = connect_local(path, role, name, timeout_s)
        for send in ("send_reliable", "send_latest"):
            setattr(carrier, send, self._noted(getattr(carrier, send)))
        self.carrier = carrier
        return carrier

    def _noted(self, send: Callable[[int, bytes], None]) -> Callable[..., None]:
        def noted(msg_type: int, payload: bytes) -> None:
            self.sent.append(decode(msg_type, payload))
            send(msg_type, payload)

        return noted

    def of(self, kind: type) -> list[Any]:
        """Every link message of this kind the program sent so far."""
        return [message for message in list(self.sent) if isinstance(message, kind)]


class Person(HumanProgram):
    """The human program's main() on a thread, on the rows of a script."""

    def __init__(self, config: Path, rows: list[dict[str, Any]], *args: str) -> None:
        self.tap = Tap()
        super().__init__(config, *args, attach=self.tap, feed=script_feed(rows))

    def setpoints(self, **fields: Any) -> list[HumanSetpoint]:
        """The HUMAN_SETPOINTs the program sent so far that hold these values."""
        return [
            message
            for message in self.tap.of(HumanSetpoint)
            if all(getattr(message, name) == value for name, value in fields.items())
        ]


class Page:
    """A browser tab on the live view that stays in front.

    Every ACK_S it asks for the newest picture and acknowledges the one it
    got, over one connection, until close(). While paused it asks for nothing
    and acknowledges nothing, as a tab that is hidden or frozen.
    """

    def __init__(self, port: int) -> None:
        self.acknowledged = 0
        self.refused: list[int] = []  # the status codes of those not taken
        self.failed: list[BaseException] = []
        self._browser = Browser(port)
        self._paused = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._done.is_set():
                if not self._paused.is_set():
                    self._turn()
                self._done.wait(ACK_S)
        except (OSError, http.client.HTTPException) as error:
            self.failed.append(error)

    def _turn(self) -> None:
        browser = self._browser
        seq, _ = browser.frame()
        if seq is None:
            return
        code = browser.shown_raw({"seq": seq, "visible": True, "token": browser.token})
        if code == 204:
            self.acknowledged += 1
        else:
            self.refused.append(code)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def close(self) -> None:
        self._done.set()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the page did not end"
        self._browser.close()


class Stage(Desks):
    """The robot app behind the service, and the programs a test attaches.

    Desks of test_demo1_programs.py, which holds the consoles and the views,
    with a browser tab that keeps acknowledging, the human program and a
    local client in the role "tool" that reads ROBOT_STATE.
    """

    def __init__(self, stack: Stack, folder: Path, cues: Cues) -> None:
        super().__init__(stack.path, folder)
        self.stack, self.cues = stack, cues
        self.h = stack.h
        self.watcher: Station | None = None
        self._pages: list[Page] = []
        self._people: list[Person] = []

    def start(self, **service_options: Any) -> Stage:
        """Start the service, let the robot app dial it and attach the watcher."""
        self.stack.start_service(**service_options)
        self.stack.dial()
        self.watcher = self.stack.client("tool")
        self.stack.connected(self.watcher)
        return self

    def live_view(self) -> Page:
        """The live view with pictures, and a tab that acknowledges them."""
        view = self.view(source=StandInSource(), **VIEW_BLOCKS)
        page = Page(view.program.port)
        self._pages.append(page)
        return page

    def person(self, script: Script, *args: str, **blocks: Any) -> Person:
        """The human program on this script; blocks are blocks of its file."""
        config = write_human_config(self.folder, self.path, **blocks)
        person = Person(config, script.rows, *args)
        self._people.append(person)
        return person

    @property
    def state(self) -> RobotState:
        """The newest ROBOT_STATE the watcher has."""
        return self.watcher.ws.state

    def shows(
        self, shown: Callable[[RobotState], Any], wait_s: float = WAIT_S
    ) -> RobotState:
        """Wait for a ROBOT_STATE that `shown` accepts; returns it."""
        return self.watcher.ws.wait_state(shown, wait_s)

    def live(self) -> RobotState:
        """Wait until the robot app reports the age of the live view."""
        return self.shows(lambda s: s.view_age_s is not None)

    def sent(self) -> int:
        """How many joint targets the arm was sent so far."""
        return len(self.h.sent())

    def holds(self) -> None:
        """No motion work runs, and the motors are on and were never off."""
        self.h.idle()
        # APAI's mock arm notes a torque release on the spy of its harness.
        noted = getattr(self.h, "spy", self.h.arm)
        assert self.h.arm.torque_on() and noted.released_at == []

    def close(self) -> None:
        """End the human programs, the tabs, the views, the consoles, the stack.

        In that order, and each of them also when one before it failed to end.
        """
        with contextlib.ExitStack() as ending:
            ending.callback(self.stack.close)
            ending.callback(super().close)
            for page in self._pages:
                ending.callback(page.close)
            for person in self._people:
                if person.running:
                    ending.callback(person.end)


def human(state: RobotState) -> dict[str, Any]:
    """health["human"] of a ROBOT_STATE: the robot app's facts of the human path."""
    return state.health["human"]


def driving(state: RobotState, sent: int = 1) -> bool:
    """Whether a human drive runs that has sent this many joint targets."""
    return bool(human(state)["driving"]) and state.move_index >= sent


def sent_at(stage: Stage, kind: str) -> list[int]:
    """How many joint targets the arm had been sent at each event of this kind.

    Of the robot app's events from here on, counted as each one is emitted.
    """
    counts: list[int] = []

    def note(event: str, text: str, data: dict[str, Any]) -> None:
        if event == kind:
            counts.append(stage.sent())

    stage.h.app.add_event_listener(note)
    return counts


def every_state(stage: Stage) -> list[RobotState]:
    """Every ROBOT_STATE the watcher receives from here on, in order."""
    states: list[RobotState] = []

    def note(message: Message) -> None:
        if isinstance(message, RobotState):
            states.append(message)

    stage.watcher.ws.add_listener(note)
    return states


def drive_end(stage: Stage, count: int = 1) -> dict[str, Any]:
    """Wait for the robot app's EVENT number `count` of an ended human drive.

    Returns its data: the drive's status, its reason and the joint targets it
    sent. The EVENT's text is the reason in the words of DRIVE_ENDED.
    """
    heard = stage.h.heard
    wait_for(
        lambda: len(heard(app_module.EVENT_HUMAN)) >= count,
        f"EVENT {count} of an ended human drive",
        lambda: [event[:2] for event in stage.h.local_events[-6:]],
    )
    _, text, data = heard(app_module.EVENT_HUMAN)[count - 1]
    assert text == app_module.DRIVE_ENDED.format(why=data["reason"])
    return data


# How a human drive ends when its controller data stop while it runs, by who
# notices first: the drive finds them too old ("stale"), the human program
# says that it ended the engagement ("disengaged"), or the robot app finds a
# gap between two human targets ("stopped", with DATA_AWAY).
LOST_DATA_ENDS = ("stale", "disengaged", "stopped")


def check_lost_data_end(data: dict[str, Any]) -> None:
    """This ended human drive is one that lost its controller data."""
    assert data["status"] in LOST_DATA_ENDS, data
    if data["status"] == "stopped":
        assert data["reason"] == app_module.DATA_AWAY, data


def agent_move(stage: Stage) -> list[list[float]]:
    """The setpoints of a short move of the agent from where the arm holds.

    25 setpoints, 0.5 s. After a return to rest the arm lies within the rest
    tolerance of the rest pose, not on it.
    """
    start = stage.watcher.ws.measured_pose()
    return [[q + row[0] for q in start] for row in PATH]


def rest_from_the_console(desk: TimedDesk, count: int = 1) -> None:
    """Type rest and the confirmation; wait for the RESULT of this rest."""
    desk.type("rest")
    desk.wait(ASK_REST, count)
    desk.type(PATH_IS_CLEAR)
    desk.wait_long("rest: RESULT completed", REST_WAIT_S)
    assert len(desk.holding("rest: RESULT completed")) == count


@pytest.fixture(autouse=True)
def log_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    """No main() of a program attaches a log handler in a test."""
    monkeypatch.setattr(view_main, "set_console_level", lambda level: None)
    monkeypatch.setattr(console_log, "set_console_level", lambda level: None)


@pytest.fixture
def cues(monkeypatch: pytest.MonkeyPatch) -> Cues:
    return Cues(monkeypatch)


def sim_config(**human_fields: Any) -> RobotAppConfig:
    """The demo values with human driving on and a view age limit of 2 s."""
    fields = {"view_age_limit_s": VIEW_AGE_LIMIT_S, **human_fields}
    return human_config(RobotAppConfig.default(), **fields)


@pytest.fixture
def make_sim_stage(
    make_realtime_robot_app: Callable[..., Any],
    tmp_path_factory: pytest.TempPathFactory,
    cues: Cues,
) -> Iterator[Callable[..., Stage]]:
    """make_sim_stage(config=None, **service_options): a started Stage.

    The robot app runs on the simulated arm (driver "sim") in real time with
    human driving switched on; service_options are keywords of the service.
    """
    made: list[Stage] = []

    def make(config: RobotAppConfig | None = None, **service_options: Any) -> Stage:
        # A short path: a socket path holds 107 bytes at most.
        folder = tmp_path_factory.mktemp("d2")
        harness = make_realtime_robot_app(
            config=sim_config() if config is None else config, connected=False
        )
        stage = Stage(Stack(harness, folder), folder, cues)
        made.append(stage)
        return stage.start(**service_options)

    yield make
    for stage in made:
        stage.close()


@pytest.fixture
def sim_stage(make_sim_stage: Callable[..., Stage]) -> Stage:
    return make_sim_stage()


def mock_config(human_speed: float | None = None) -> RobotAppConfig:
    """The mock scene's robot app file with human driving on.

    scene_robot_config() of test_launcher.py, a view age limit of 2 s and,
    when given, another human joint speed limit in rad/s.
    """
    config = human_config(scene_robot_config(), view_age_limit_s=VIEW_AGE_LIMIT_S)
    if human_speed is None:
        return config
    slower = dataclasses.replace(config.limits.human, max_joint_speed=human_speed)
    limits = dataclasses.replace(config.limits, human=slower)
    return dataclasses.replace(config, limits=limits)


@pytest.fixture
def make_mock_stage(
    make_realtime_apai: Callable[..., Any],
    tmp_path_factory: pytest.TempPathFactory,
    cues: Cues,
) -> Iterator[Callable[..., Stage]]:
    """make_mock_stage(config=None): a started Stage on APAI's mock arm.

    The robot app declares the driver "apai-mock" and runs in real time with
    human driving switched on; the arm is raised off its joint stops, where
    APAI's checkers let it move.
    """
    made: list[Stage] = []

    def make(config: RobotAppConfig | None = None) -> Stage:
        harness = make_realtime_apai(
            config=mock_config() if config is None else config, connected=False
        )
        raise_arm(harness)
        # A short path: a socket path holds 107 bytes at most.
        folder = tmp_path_factory.mktemp("d2")
        stage = Stage(Stack(harness, folder), folder, cues)
        made.append(stage)
        return stage.start()

    yield make
    for stage in made:
        stage.close()


# The robot app on the simulated arm


@link
@apai
def test_a_stop_during_human_motion_holds_the_arm_and_keeps_the_takeover_open(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.live_view()
    agent = stage.stack.agent()
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.06, 6.0)
    script.cue("let go").release().hand_back("given back")
    person = stage.person(script)
    at_stop = sent_at(stage, STOP_EVENT)
    stage.live()
    stage.cues.give("go")
    stage.shows(lambda s: driving(s, 10))
    solver = Solver.from_apai("mock")
    rest_tip = solver.fk(solver.inside(REST))[:3, 3]

    # stop typed at the console while the grip is squeezed and the hand rises.
    desk.type("stop")
    taken = desk.wait("stop: taken.")
    assert "stop number 2, stop level stop" in taken
    stopped = stage.shows(lambda s: s.stop_number == 2 and s.motion == "hold")
    assert (stopped.owner, human(stopped)["takeover_open"]) == (NOBODY, True)
    assert not human(stopped)["driving"]
    stage.holds()
    held = stage.sent()
    # Within one period: at most the joint target that was on its way.
    assert 10 <= at_stop[-1] <= held <= at_stop[-1] + 1
    told = h.heard(STOP_EVENT)
    senders = [event[2]["sender"] for event in told]
    assert senders == [logic.STOP_SENDER, console_cmd.SENDER]
    assert told[-1][1].startswith(
        f"STOP from {console_cmd.SENDER}: {console_cmd.STOP_REASON}"
    )
    # A drive that a STOP ended has the STOP's line and no line of its own.
    assert h.heard(app_module.EVENT_HUMAN) == []
    # The hand is still squeezed and rising: nothing follows it any more.
    wait_for(
        lambda: len(person.setpoints(stop_number=2)) >= 15,
        "HUMAN_SETPOINTs that quote the STOP",
        lambda: person.tap.sent[-3:],
    )
    assert stage.sent() == held
    assert not any(sent.engaged for sent in person.setpoints(stop_number=2))
    rose = solver.fk(stage.watcher.ws.measured_pose())[:3, 3] - rest_tip
    assert 0.0 < rose[2] < 0.06

    # The agent gets no task while the takeover is open.
    refused = agent.ws.start_task()
    assert (refused.status, refused.reason) == ("refused", app_module.TAKEOVER_OPEN)
    desk.wait(app_module.TAKEOVER_OPEN)

    # The operator's rest, then right A at rest: the takeover ends.
    stage.cues.give("let go")
    person.wait_said("human program: grip up")
    rest_from_the_console(desk)
    assert h.heard(EVENT_REST)  # the kind a return to rest is told with
    stage.shows(lambda s: s.motion == "rest_hold" and s.owner == NOBODY)
    ended = stage.shows(lambda s: not human(s)["takeover_open"])
    assert (ended.owner, ended.motion) == (NOBODY, "rest_hold")
    desk.wait(f"[owner] {app_module.HANDED_BACK}")
    stage.cues.give("given back")
    requests = [request.request for request in person.tap.of(OwnerRequest)]
    assert requests == ["human", "handback"]
    assert person.tap.of(Stop) == [
        Stop(reason=logic.STOP_REASON, sender=logic.STOP_SENDER)
    ]

    # Now the agent's task runs: it takes the arm and moves it.
    granted = agent.ws.start_task()
    assert granted.status == "completed", granted.reason
    moved = agent.ws.run_move(agent_move(stage), WAIT_S)
    assert moved.status == "completed", moved.reason
    stage.holds()
    assert person.running and person.end() == 0
    print(
        f"measured: stop typed after {at_stop[-1]} joint targets, {held} sent in "
        f"all; the tip had risen {rose[2] * 100:.1f} cm"
    )


@link
@apai
def test_right_b_during_human_motion_holds_the_arm(sim_stage: Stage) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.live_view()
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.01, 1.0)
    # Right B goes down with the grip squeezed and stays down for 2.2 s, while
    # the hand rises on.
    script.lift(0.02, 2.0, b=True).then(0.2, b=False).release()
    person = stage.person(script)
    at_stop = sent_at(stage, STOP_EVENT)
    stage.live()
    stage.cues.give("go")
    stage.shows(lambda s: driving(s, 10))

    stopped = stage.shows(lambda s: s.stop_number == 2 and s.motion == "hold")
    assert (stopped.owner, stopped.stop_level) == (NOBODY, "stop")
    assert human(stopped)["takeover_open"] and not human(stopped)["driving"]
    stage.holds()
    held = stage.sent()
    # Within one period: at most the joint target that was on its way.
    assert 25 <= at_stop[1] <= held <= at_stop[1] + 1
    person.wait_said("human program: B up", 2)
    pressed = [
        at
        for at, line in zip(person.said_at, person.said)
        if line in ("human program: B down", "human program: B up")
    ]
    assert len(pressed) == 4 and pressed[3] - pressed[2] > 2.0
    # One STOP for that press, as for the first one; the robot app names the
    # controller as its sender.
    stop = Stop(reason=logic.STOP_REASON, sender=logic.STOP_SENDER)
    assert person.tap.of(Stop) == [stop, stop]
    told = h.heard(STOP_EVENT)
    assert [event[2]["sender"] for event in told] == [logic.STOP_SENDER] * 2
    line = f"[stop] STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; "
    assert len(desk.holding(line)) == 2
    # Right B was held for longer than buttons.hold_s with the grip squeezed:
    # nothing but the first takeover was asked for, and the note says why.
    requests = [request.request for request in person.tap.of(OwnerRequest)]
    assert requests == ["human"]
    assert person.setpoints(note=logic.NOTE_RELEASE_B, stop_number=2)
    desk.wait(f" | note: {logic.NOTE_RELEASE_B} | ")
    # Nobody owns the arm, the takeover stays open, and nothing has moved.
    now = stage.state
    assert (now.owner, now.stop_number) == (NOBODY, 2)
    assert human(now)["takeover_open"] and h.app.takeover_open
    assert stage.sent() == held and h.heard(app_module.EVENT_HUMAN) == []
    person.wait_said("human program: grip up")
    assert stage.sent() == held
    stage.holds()
    print(
        f"measured: right B after {at_stop[1]} joint targets, {held} sent in all; "
        f"held for {pressed[3] - pressed[2]:.2f} s with the grip squeezed"
    )


@link
@apai
def test_lost_headset_data_during_human_motion_holds_the_arm(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.live_view()
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.01, 1.0)
    # The controller goes away for 1.5 s with the grip squeezed and comes back
    # with it still squeezed; the hand rises on for a second.
    script.then(0.1, absent=True).then(1.5, absent=False).lift(0.01, 1.0)
    # A release and a new squeeze.
    script.release().squeeze().lift(0.01, 1.0).release()
    person = stage.person(script)
    stage.live()
    stage.cues.give("go")
    owned = stage.shows(lambda s: s.owner == HUMAN)
    stage.shows(lambda s: driving(s, 10))

    # The drive ends in a hold, by whichever side notices first.
    data = drive_end(stage)
    check_lost_data_end(data)
    stage.holds()
    held = stage.sent()
    assert held == data["sent"] >= 10
    desk.wait(" | headset no_controller | ")
    # The data are back with the grip still squeezed, the hand rising: the
    # person still owns the arm and nothing follows the hand. (A controller
    # that is away reads every button as up: that was the first "grip up".)
    person.wait_said("human program: right controller seen", 2)
    person.wait_said("human program: grip up", 2)
    assert stage.sent() == held
    now = stage.state
    assert (now.owner, now.owner_number) == (HUMAN, owned.owner_number)
    assert human(now)["takeover_open"] and now.stop_number == owned.stop_number
    sent = person.tap.of(HumanSetpoint)
    away = [index for index, one in enumerate(sent) if one.stage != "ready"]
    assert len(away) >= 50  # 1.5 s without the controller
    back = [one for one in sent[away[-1] + 1 :] if not one.grip_released]
    assert len(back) >= 40 and not any(one.engaged for one in back)

    # After the release, a new squeeze: the arm follows the hand again.
    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up", 3)
    stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    assert stage.sent() >= held + 40
    assert len(h.heard(app_module.EVENT_HUMAN)) == 1
    print(
        f"measured: the drive ended {data['status']} ({data['reason']}) after "
        f"{held} joint targets; {len(back)} HUMAN_SETPOINTs with the grip "
        f"squeezed moved nothing; {stage.sent() - held} joint targets after the "
        "new squeeze"
    )


@link
@apai
def test_a_human_program_that_dies_during_human_motion_holds_the_arm(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.live_view()
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.05, 10.0)
    person = stage.person(script)
    stage.live()
    stage.cues.give("go")
    owned = stage.shows(lambda s: driving(s, 25))
    # A console that attached before the human program said so then.
    before = len(desk.holding(" | no human program | "))

    # The human program's socket closes, as at the end of its process.
    person.tap.carrier.close()
    closed = time.monotonic()
    data = drive_end(stage)
    check_lost_data_end(data)
    stage.holds()
    held = stage.sent()
    assert held == data["sent"] >= 25
    # The last joint target went out within the sample age limit and one
    # period of the last human target, on the robot app's clock.
    limit = h.config.human.sample_age_limit_s + h.config.executor.period_s
    last_target = h.app._human_input.newest()
    _, last_sent = h.arm.sent[-1]
    assert last_sent - last_target.arrived_mono <= limit
    # And, with room for this machine's threads, of the socket's close.
    assert last_sent - closed <= limit + 0.25
    wait_for(lambda: not person.running, "the human program's end")
    assert person.code == 1
    # The person still owns the arm; the console says that the program is gone.
    gone = wait_for(
        lambda: desk.holding(" | no human program | ")[before:],
        "a status line that says no human program",
        lambda: desk.said[-3:],
    )
    assert f" | owner human #{owned.owner_number} | " in gone[-1]
    assert " | takeover open | " in gone[-1]
    now = stage.shows(lambda s: s.motion == "hold" and not human(s)["driving"])
    assert (now.owner, now.owner_number) == (HUMAN, owned.owner_number)
    assert human(now)["takeover_open"] and now.stop_number == owned.stop_number
    assert stage.sent() == held
    stage.holds()
    print(
        f"measured: the drive ended {data['status']} ({data['reason']}); the "
        f"last joint target went out {(last_sent - closed) * 1e3:.0f} ms after "
        f"the close, {(last_sent - last_target.arrived_mono) * 1e3:.0f} ms after "
        "the last human target"
    )


@link
@apai
def test_an_agent_program_that_dies_during_human_motion_changes_nothing(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.live_view()
    agent = stage.stack.agent()
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.03, 3.0)
    script.release()
    person = stage.person(script)
    states = every_state(stage)
    stage.live()
    stage.cues.give("go")
    owned = stage.shows(lambda s: driving(s, 20))
    solver = Solver.from_apai("mock")
    rest_tip = solver.fk(solver.inside(REST))[:3, 3]

    # The agent program's socket closes while the grip is squeezed.
    agent.kill()
    stage.watcher.end.wait_status(lambda s: not s.roles.get(AGENT), WAIT_S)
    at_kill = stage.sent()
    # The same drive goes on, and ends when the grip is released.
    stage.shows(lambda s: driving(s, at_kill + 40))
    person.wait_said("human program: grip up")
    done = stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    assert stage.sent() > 140  # the 3 s of the lift, at 50 per second
    rose = solver.fk(stage.watcher.ws.measured_pose())[:3, 3] - rest_tip
    np.testing.assert_allclose(rose, [0.0, 0.0, 0.03], atol=3e-3)
    # The person stayed the control owner; no help request, no assistance wait.
    assert (done.owner, done.owner_number) == (HUMAN, owned.owner_number)
    after = [state for state in states if state.seq >= owned.seq]
    assert len(after) > 50 and {state.owner for state in after} == {HUMAN}
    assert {state.assistance_wait_s for state in states} == {None}
    assert h.heard(app_module.EVENT_HELP_NEEDED) == []
    assert h.heard(app_module.EVENT_HUMAN) == []
    assert len(h.heard(STOP_EVENT)) == 1  # that of the takeover's right B
    assert desk.holding("HELP NEEDED") == []
    print(
        f"measured: the agent's socket closed after {at_kill} joint targets; "
        f"{stage.sent()} in all; the tip rose {rose[2] * 100:.1f} cm"
    )


@link
@apai
def test_a_lost_link_during_human_motion_holds_the_arm_and_the_person_keeps_it(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h, stack = stage.h, stage.stack
    desk = stage.console()
    stage.live_view()
    # The grip stays squeezed through the whole outage. The hand rises on, and
    # before the connection is back it returns to where the squeeze began.
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.06, 6.0)
    script.cue("down").lift(-0.06, 1.0)
    script.cue("let go").release().squeeze().lift(0.01, 1.0).release()
    person = stage.person(script)
    thor = h.thor_terminal()
    stage.live()
    stage.cues.give("go")
    # Far enough from rest for the return to rest to have a way to go.
    owned = stage.shows(lambda s: driving(s, 150))

    # The robot connection is cut and stays away.
    stack.held = True
    stack.cut()
    _, text, data = h.heard(app_module.EVENT_CONNECTION)[-1]
    assert text.startswith(f"{app_module.LINK_LOST}: the arm holds")
    assert app_module.PERSON_KEEPS_ARM in text
    assert data["owner_before"] == HUMAN
    thor.wait_line(f"[connection] {app_module.LINK_LOST}")
    desk.wait_long("ROBOT CONNECTION LOST: ", WAIT_S)
    stage.holds()
    held = stage.sent()
    assert held >= 150
    now = h.app.snapshot()
    assert (now.owner.value, now.owner_number) == (HUMAN, owned.owner_number)
    assert h.app.takeover_open and h.app.motion_state.value == "hold"
    # A drive that the lost connection ended has that line and no other.
    assert h.heard(app_module.EVENT_HUMAN) == []

    # Meanwhile rest typed at the Thor terminal returns the arm.
    thor.ask("rest", answer=app_module.PATH_CLEAR_PROMPT)
    thor.ask(PATH_IS_CLEAR, answer="[rest] ")
    answer = wait_for(
        lambda: thor.written("rest: "),
        "the answer to the rest typed at the Thor terminal",
        lambda: thor.lines[-5:],
        REST_WAIT_S,
    )
    assert answer == ["rest: completed"]
    h.idle()
    assert h.app.motion_state.value == "rest_hold"
    rested = stage.sent()
    assert rested > held
    now = h.app.snapshot()
    assert (now.owner.value, h.app.takeover_open) == (HUMAN, True)
    # The hand returns to where the squeeze began, as the arm did: its target
    # lies on the arm again, so no limit on a target that is far from the arm
    # holds the arm back, only the rule of the new squeeze.
    at_start = "hand x {:.3f} y {:.3f} z {:.3f} m".format(*HAND)
    before = len([line for line in list(person.said) if at_start in line])
    stage.cues.give("down")
    person.wait_said(at_start, before + 1)

    # The connection comes back, the grip still squeezed: nothing moves.
    stack.held = False
    stage.watcher.end.wait_status(lambda s: s.robot_connected, WAIT_S)
    desk.wait_long("robot connection up again", WAIT_S)
    # A ROBOT_STATE of the new connection: the arm at rest, and human targets
    # arriving again.
    first = stage.shows(
        lambda s: s.motion == "rest_hold" and human(s)["seq"] is not None
    )
    assert first.teleop_stage == "ready"
    later = stage.shows(lambda s: (human(s)["seq"] or 0) >= human(first)["seq"] + 40)
    assert stage.sent() == rested
    assert not human(later)["driving"] and later.motion == "rest_hold"
    # Each side keeps the rule by itself: the human program sends the squeezed
    # grip as not engaged, and the robot app waits for a released grip.
    squeezed = person.setpoints(seq=human(later)["seq"])
    assert squeezed and not squeezed[0].grip_released and not squeezed[0].engaged
    assert (later.owner, human(later)["needs_release"]) == (HUMAN, True)
    # The human program's next report line: it ended no engagement for an IK
    # solution or for a target that lay too far ahead.
    reports = len([line for line in list(person.said) if REPORT_START in line])
    report = person.wait_said(REPORT_START, reports + 1)
    assert NONE_ENDED in report, report

    # After a release and a new squeeze the arm follows the hand again.
    stage.cues.give("let go")
    person.wait_said("human program: grip up")
    assert stage.sent() == rested
    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up", 2)
    stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    assert stage.sent() >= rested + 40
    # The person kept the arm all along: one takeover, and no STOP since.
    now = h.app.snapshot()
    assert (now.owner.value, now.stop_number) == (HUMAN, owned.stop_number)
    assert [request.request for request in person.tap.of(OwnerRequest)] == ["human"]
    print(
        f"measured: the link was cut after {held} joint targets; the return to "
        f"rest sent {rested - held}; with the hand back where the squeeze began "
        f"and the grip squeezed, {human(later)['seq'] - human(first)['seq']} "
        f"HUMAN_SETPOINTs moved nothing; {stage.sent() - rested} joint targets "
        "after the new squeeze"
    )


@link
@apai
def test_the_view_going_not_live_during_human_motion_holds_the_arm(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    page = stage.live_view()
    # The grip stays squeezed, the hand rising, until the view is back.
    script = Script(stage.cues).cue("go").take_over().squeeze().lift(0.02, 4.0)
    # With the view back: the grip is released and squeezed again, and a lift.
    script.cue("view back").release().squeeze().lift(0.01, 1.0).release()
    # With the view away once more: right B is held until it is refused.
    script.cue("view away").then(0.3, b=True).cue("refused").then(0.2, b=False)
    person = stage.person(script)
    stage.live()
    stage.cues.give("go")
    owned = stage.shows(lambda s: driving(s, 20))
    not_fresh = app_module.VIEW_NOT_FRESH.removesuffix(": ")

    # The browser tab stops acknowledging the pictures.
    page.pause()
    data = drive_end(stage)
    assert data["status"] == "view" and data["reason"].startswith(not_fresh), data
    stage.holds()
    held = stage.sent()
    assert held == data["sent"] >= 20
    ended = app_module.DRIVE_ENDED.format(why=data["reason"])
    assert desk.wait(f"[human] {ended}") == f"{BELL_CHAR}[human] {ended}"
    now = stage.shows(lambda s: s.view_age_s is None and s.motion == "hold")
    assert (now.owner, now.owner_number) == (HUMAN, owned.owner_number)
    assert human(now)["takeover"].startswith(app_module.VIEW_NOT_FRESH)

    # The acknowledgements are back. The person still owns the arm: after a
    # release and a new squeeze, with no new takeover, it follows the hand.
    page.resume()
    stage.live()
    assert stage.sent() == held
    stage.cues.give("view back")
    person.wait_said("human program: grip up")
    assert stage.sent() == held
    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up", 2)
    again = stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    moved = stage.sent()
    assert moved >= held + 40
    assert (again.owner, again.owner_number) == (HUMAN, owned.owner_number)
    assert again.stop_number == owned.stop_number
    assert [request.request for request in person.tap.of(OwnerRequest)] == ["human"]
    assert len(h.heard(app_module.EVENT_HUMAN)) == 1
    assert h.heard(app_module.EVENT_REFUSED) == []

    # The tab stops again. A takeover is refused for the reason that ended the
    # drive: right B is a STOP, after which nobody owns the arm, and its hold
    # asks for the arm.
    page.pause()
    stage.shows(lambda s: s.view_age_s is None)
    stage.cues.give("view away")
    stage.shows(lambda s: s.stop_number == 2 and s.owner == NOBODY)
    _, text, refusal = h.wait_local_event(app_module.EVENT_REFUSED)
    reason = refusal["reason"]
    assert text == f"{TAKEOVER_REFUSED}{reason}"
    assert reason.startswith(app_module.VIEW_NOT_FRESH)
    assert reason.endswith(app_module.OPEN_THE_VIEW)
    desk.wait(f"{BELL_CHAR * 3}[refused] {TAKEOVER_REFUSED}{not_fresh}")
    now = stage.state
    assert (now.owner, now.stop_number) == (NOBODY, 2)
    assert human(now)["takeover_open"] and stage.sent() == moved
    stage.cues.give("refused")
    person.wait_said("human program: B up", 2)
    requests = [request.request for request in person.tap.of(OwnerRequest)]
    assert requests == ["human", "human"]
    stage.holds()
    assert page.refused == [] and page.failed == []
    print(
        f"measured: the drive ended after {held} joint targets ({data['reason']}); "
        f"{moved - held} joint targets after the view was back and the grip was "
        f"squeezed again; then the takeover was refused with: {reason}"
    )


@link
@apai
def test_a_takeover_without_a_live_view_is_refused(sim_stage: Stage) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    stage.view("--source", "none")  # a view without pictures is never live
    script = Script(stage.cues).cue("go").then(0.3, b=True).cue("refused")
    script.then(0.2, b=False)
    person = stage.person(script)
    # The robot app says so before anyone asks, and the console repeats it.
    not_fresh = app_module.VIEW_NOT_FRESH
    told = stage.shows(lambda s: human(s)["takeover"].startswith(not_fresh))
    assert human(told)["takeover"].endswith(app_module.OPEN_THE_VIEW)
    console_has(
        desk,
        lambda state: human(state)["takeover"].startswith(not_fresh),
        "why no takeover is possible",
    )
    desk.type("state")
    desk.wait(f"takeover: not possible: {not_fresh}")

    # Right B: a STOP, and after its hold the request, which is refused.
    stage.cues.give("go")
    _, text, refusal = h.wait_local_event(app_module.EVENT_REFUSED)
    reason = refusal["reason"]
    assert text == f"{TAKEOVER_REFUSED}{reason}"
    assert reason.startswith(app_module.VIEW_NOT_FRESH)
    assert reason.endswith(app_module.OPEN_THE_VIEW)
    # Three bells: the wearer of a headset cannot read.
    assert desk.wait(f"[refused] {text}") == f"{BELL_CHAR * 3}[refused] {text}"
    # The person at the controller is told through the human program's note.
    noted = stage.shows(lambda s: app_module.REFUSED in human(s)["note"])
    assert human(noted)["note"].startswith(f"takeover {app_module.REFUSED}: ")
    # Held on, right B asks for nothing more.
    wait_for(
        lambda: len(person.setpoints(note=human(noted)["note"])) >= 20,
        "HUMAN_SETPOINTs that carry the refusal",
        lambda: person.tap.sent[-3:],
    )
    requests = person.tap.of(OwnerRequest)
    assert [request.request for request in requests] == ["human"]
    assert requests[0].reason == RIGHT_B
    # Nothing changed but the STOP: nobody owns the arm and nothing moved.
    now = stage.state
    assert (now.owner, now.stop_number) == (NOBODY, 1)
    assert not human(now)["takeover_open"] and not h.app.takeover_open
    assert h.heard(app_module.EVENT_OWNER) == [] and stage.sent() == 0
    assert now.motion == "rest_hold"
    stage.holds()
    stage.cues.give("refused")
    person.wait_said("human program: B up")
    print(f"measured: the takeover was refused with: {reason}")


@link
@apai
def test_a_stop_during_the_assistance_wait_prevents_the_timed_return(
    sim_stage: Stage,
) -> None:
    stage = sim_stage
    h = stage.h
    desk = stage.console()
    agent = stage.stack.agent()
    # Right B is pressed for 0.3 s: a STOP, with no hold that asks for the arm.
    script = Script(stage.cues).cue("press").then(0.3, b=True).then(0.3, b=False)
    person = stage.person(script)
    person.wait_said("human program: right controller seen")
    states = every_state(stage)

    # The agent moves the arm off rest; then its program is gone.
    h.new_task()
    moved = agent.ws.run_move(agent_move(stage), WAIT_S)
    assert moved.status == "completed", moved.reason
    sent = stage.sent()
    agent.kill()
    desk.wait(f"HELP NEEDED: {AGENT_GONE}")
    waiting = stage.shows(lambda s: s.assistance_wait_s is not None)
    assert waiting.owner == NOBODY and waiting.motion == "hold"
    assert h.config.rest.auto_return is True  # the wait would end in a return

    # Right B during the wait.
    stage.cues.give("press")
    stopped = stage.shows(lambda s: s.stop_number == 1)
    _, text, data = h.heard(STOP_EVENT)[-1]
    assert data["sender"] == logic.STOP_SENDER and data["wait_cancelled"] is True
    assert text.startswith(f"STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; ")
    desk.wait(f"[stop] {text}")
    person.wait_said("human program: B up")
    # The wait is over for good, and no return to rest starts.
    later = stage.shows(lambda s: s.seq >= stopped.seq + 20)
    since = [state for state in states if state.seq >= stopped.seq]
    assert len(since) >= 20
    assert {state.assistance_wait_s for state in since} == {None}
    assert {state.owner for state in since} == {NOBODY}
    assert {state.motion for state in since} == {"hold"}
    assert later.stop_level == "stop" and not human(later)["takeover_open"]
    assert h.heard(EVENT_REST) == [] and h.heard(EXPIRED) == []
    assert stage.sent() == sent
    assert person.tap.of(OwnerRequest) == []
    stage.holds()
    print(
        f"measured: right B with {waiting.assistance_wait_s:.0f} s of the wait "
        "left; the STOP's line: " + text
    )


@link
@apai
def test_demo_2_with_the_typed_fallback_and_no_buttons(
    make_sim_stage: Callable[..., Stage],
) -> None:
    stage = make_sim_stage(typed_takeover=True)
    desk = stage.console()
    stage.live_view()
    # The script presses neither right A nor right B.
    script = Script(stage.cues).cue("squeeze").squeeze().lift(0.02, 2.0).release()
    person = stage.person(script)
    stage.live()
    stage.shows(lambda s: human(s)["takeover"] == "")
    solver = Solver.from_apai("mock")
    rest_tip = solver.fk(solver.inside(REST))[:3, 3]

    # stop, takeover, yes: a person owns the arm.
    desk.type("stop")
    desk.wait("stop: taken.")
    desk.type("takeover")
    desk.wait(console_cmd.ASK_TAKEOVER.rstrip())
    desk.type(YES)
    desk.wait("takeover: RESULT completed")
    owned = stage.shows(lambda s: s.owner == HUMAN)
    taken = app_module.TAKEN_OVER.format(number=owned.owner_number)
    assert desk.wait(f"[owner] {taken}") == f"{BELL_CHAR * 2}[owner] {taken}"

    # The squeeze, the lift and the release.
    stage.cues.give("squeeze")
    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up")
    stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    rose = solver.fk(stage.watcher.ws.measured_pose())[:3, 3] - rest_tip
    np.testing.assert_allclose(rose, [0.0, 0.0, 0.02], atol=3e-3)

    # rest, path is clear; then handback, yes: the takeover ends.
    rest_from_the_console(desk)
    stage.shows(lambda s: s.motion == "rest_hold" and s.owner == HUMAN)
    # The console quotes the ROBOT_STATE it has: one that shows the person as
    # the owner again. A hand-back typed on an older one is refused.
    console_has(
        desk,
        lambda state: state.motion == "rest_hold" and state.owner == HUMAN,
        "the person as the owner at rest",
    )
    desk.type("handback")
    desk.wait(console_cmd.ASK_HANDBACK.rstrip())
    desk.type(YES)
    desk.wait("handback: RESULT completed")
    back = f"[owner] {app_module.HANDED_BACK}"
    assert desk.wait(back) == f"{BELL_CHAR * 2}{back}"
    ended = stage.shows(lambda s: s.owner == NOBODY)
    assert not human(ended)["takeover_open"] and ended.motion == "rest_hold"
    desk.wait_long(f" | owner nobody #{ended.owner_number} | ", WAIT_S)
    # The human program sent no STOP and asked for nothing.
    assert person.tap.of(Stop) == [] and person.tap.of(OwnerRequest) == []
    assert not any(line.endswith((" A down", " B down")) for line in person.said)
    stage.holds()

    # Against a service whose file does not allow it, the same words print the
    # service's refusal and change nothing.
    other = make_sim_stage()
    plain = other.console()
    plain.type("stop")
    plain.wait("stop: taken.")
    before = other.shows(lambda s: s.stop_number == 1)
    for word, ask in (
        ("takeover", console_cmd.ASK_TAKEOVER),
        ("handback", console_cmd.ASK_HANDBACK),
    ):
        plain.type(word)
        plain.wait(ask.rstrip())
        plain.type(YES)
        refused = plain.wait(f"{word}: RESULT refused: ")
        assert console_cmd.NOT_ALLOWED in refused
        assert refused.endswith(f"; {console_cmd.USE_BUTTONS}")
    now = other.shows(lambda s: s.seq > before.seq + 5)
    assert (now.owner, now.owner_number) == (NOBODY, before.owner_number)
    assert (now.stop_number, now.motion) == (1, "rest_hold")
    assert not human(now)["takeover_open"] and not other.h.app.takeover_open
    # The robot app never saw the two requests.
    assert other.h.heard(app_module.EVENT_OWNER) == []
    assert other.h.heard(app_module.EVENT_REFUSED) == []
    assert other.sent() == 0
    other.holds()
    print(f"measured: the tip rose {rose[2] * 100:.1f} cm; refused with: {refused}")


# The robot app on APAI's mock arm


@pytest.fixture
def scene_stage(
    make_mock_stage: Callable[..., Stage], tmp_path: Path
) -> Iterator[tuple[Scene, Stage]]:
    """The mock scene's robot app with human driving on, and what attaches.

    The Scene of test_agent_end_to_end.py runs the agent program in this
    process, on APAI's mock scene, through the stage's service.
    """
    stage = make_mock_stage()
    with apai_runtime_env(tmp_path) as state_folder:
        scene = Scene(stage.h, state_folder)
        scene.socket_path, scene.stack = stage.path, stage.stack
        yield scene, stage


@link
@apai
def test_demo_2_the_agent_fails_a_person_takes_over_and_the_next_task_runs(
    scene_stage: tuple[Scene, Stage],
) -> None:
    import test_apai_app as apai

    scene, stage = scene_stage
    h, trace = stage.h, scene.trace
    desk = stage.console()
    stage.live_view()
    # Right B once the agent's task has failed; a lift of 3 cm in 3 s; right A
    # once the arm holds at rest.
    script = Script(stage.cues).cue("help").take_over().squeeze().lift(0.03, 3.0)
    script.release().hand_back("given back")
    person = stage.person(script)
    states = every_state(stage)
    # Who owned the arm, and whether a takeover was open, at each jaw command.
    jaws: list[tuple[str, bool]] = []
    h.spy.after_jaw_command.append(
        lambda *_: jaws.append((h.app.snapshot().owner.value, h.app.takeover_open))
    )
    solver = Solver.from_apai("mock")
    most = SPEED_MARGIN * h.config.limits.human.max_joint_speed
    most *= h.config.executor.period_s
    stage.live()
    person.wait_said("human program: right controller seen")
    stage.shows(lambda s: human(s)["takeover"] == "")

    # An obstacle where the gripper tip is at the end of the agent's first move.
    ObstacleGrid.made.clear()
    scene.grid = ObstacleGrid
    grids: dict[str, np.ndarray] = {}

    def on_attach(made: AgentLink) -> None:
        send_move = made.send_move

        def first_move(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            if not grids:
                kinematics = trace.built["harness"].kin
                tip = kinematics.fk(np.asarray(q[-1], dtype=float))[:3, 3]
                grids["clear"] = ObstacleGrid.made[-1].grid
                grids["box"] = ObstacleGrid.made[-1].with_box(tip, 0.04)
            send_move(move_id, start, tol, q, *rest)

        made.send_move = first_move

    def switch_in(number: int) -> None:
        if number == 5:  # the arm is under way; the grid's next query shows it
            ObstacleGrid.made[-1].grid = grids["box"]

    measured: dict[str, Any] = {}

    def person_takes_over() -> None:
        """Steps 1 to 10 of the takeover, between the agent's two tasks."""
        # The model ended its task after the refused move, as in the
        # functional program: the agent gave the arm back and asked for no
        # help. The console says so; nobody owns the arm and no wait runs.
        gave_back = desk.wait_long(
            "[owner] the agent gave the arm back: task ended without success: ",
            WAIT_S,
        )
        assert gave_back.endswith("; the arm holds and nobody owns it")
        assert desk.count("HELP NEEDED") == 0 and h.heard("help_needed") == []
        waiting = stage.shows(lambda s: s.owner == NOBODY and s.motion == "hold")
        assert waiting.assistance_wait_s is None
        measured["agent targets"] = stage.sent()
        held_at = h.spy.pose()
        held_tip = solver.fk(solver.inside(held_at))[:3, 3]
        # A person takes the obstacle away.
        ObstacleGrid.made[-1].grid = grids["clear"]

        # Right B: a STOP; held, it takes the arm over.
        stage.cues.give("help")
        desk.wait(f"[stop] STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; ")
        (stop,) = h.heard(STOP_EVENT)
        assert stop[2]["sender"] == logic.STOP_SENDER
        assert not stop[2]["wait_cancelled"]  # there was no wait to cancel
        assert desk.count(f"[stop] {stop[1]}") == 1
        owned = stage.shows(lambda s: s.owner == HUMAN)
        assert owned.assistance_wait_s is None and human(owned)["takeover_open"]
        taken = app_module.TAKEN_OVER.format(number=owned.owner_number)
        assert desk.wait(f"[owner] {taken}") == f"{BELL_CHAR * 2}[owner] {taken}"
        assert stage.sent() == measured["agent targets"]
        jaws_before = list(jaws)
        sent_before = len(h.spy.jaw_commands)

        # The squeeze and the lift: the arm follows the hand.
        stage.shows(lambda s: driving(s, 10))
        shown = desk.wait_long(" | human driving | ", WAIT_S)
        assert f" | owner human #{owned.owner_number} | " in shown
        assert " | moving " in shown and "/0 | " in shown
        person.wait_said("human program: grip up")
        stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
        stage.holds()
        measured["human targets"] = stage.sent() - measured["agent targets"]
        rose = solver.fk(h.spy.pose())[:3, 3] - held_tip
        np.testing.assert_allclose(rose, [0.0, 0.0, 0.03], atol=3e-3)
        measured["rose"] = rose[2]
        desk.wait_long(f" | note: {logic.NOTE_SQUEEZE} | ", WAIT_S)

        # rest and path is clear at the console; the person owns the arm again.
        rest_from_the_console(desk)
        rested = stage.shows(lambda s: s.motion == "rest_hold" and s.owner == HUMAN)
        assert human(rested)["takeover_open"]
        assert h.spy.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)

        # Right A at rest: the hand-back.
        ended = stage.shows(lambda s: not human(s)["takeover_open"])
        assert (ended.owner, ended.motion) == (NOBODY, "rest_hold")
        back = f"[owner] {app_module.HANDED_BACK}"
        assert desk.wait(back) == f"{BELL_CHAR * 2}{back}"
        # From the takeover to the hand-back the jaws were not moved. They are
        # open and the script's trigger is released, so they followed the
        # trigger; the typed rest and the hand-back each ended that with the
        # light hold, the low jaw stiffness, where the jaws were: fully open,
        # which the mock counts as position 0.
        assert jaws[len(jaws_before) :] == [(HUMAN, True)] * 2
        light_hold = (0.0, pytest.approx(apai.LIGHT))
        assert h.spy.jaw_commands[sent_before:] == [light_hold] * 2
        stage.cues.give("given back")
        # The agent program has seen the hand-back too: its task gate takes no
        # task on a ROBOT_STATE that still shows the takeover open.
        wait_for(
            lambda: not human(scene.link.state())["takeover_open"],
            "the agent program's ROBOT_STATE of the hand-back",
            lambda: human(scene.link.state()),
        )

    typed = [PICK, QUICK]
    failed: list[BaseException] = []

    def next_line(prompt: str = "") -> str:
        """The agent terminal's task line; the person acts before the second."""
        if len(typed) == 1:
            try:
                person_takes_over()
            except BaseException as error:
                failed.append(error)
                raise EOFError from error  # ends the agent program
        if not typed:
            raise EOFError
        return typed.pop(0)

    scene.on_attach = on_attach
    scene.next_line = next_line
    h.spy.after_target.append(switch_in)
    scene.answers = ["yes", "yes"]
    began = time.monotonic()
    code = scene.run("--interactive")
    took = time.monotonic() - began
    if failed:
        raise failed[0]
    assert took < RUN_BUDGET_S

    # The first task ended without success; the second one, after the
    # hand-back, ran and succeeded.
    blocked, opened = scene.reports()
    assert blocked["success"] is False, blocked["summary"]
    assert opened["success"] is True, (opened["summary"], scene.lines[-12:])
    (stopped,) = scene.results("pick_and_place")
    assert stopped["ok"] is False and "motion_blocked" not in stopped
    assert scene.results("open_gripper")[0]["ok"] is True
    assert code == launcher.EXIT_FAILED  # one task ended without success
    assert trace.owners == [
        NOBODY, AGENT, NOBODY, HUMAN, RECOVERY, HUMAN, NOBODY, AGENT, NOBODY,
    ]  # fmt: skip
    # No program was restarted: one run of each, on one session of the robot app.
    assert not scene.said(task_gate.RESTART)
    assert person.running and len(person.said) > 5
    assert desk.count("attached to the robot link service: ") == 1
    assert desk.count("WELCOME: ") == 1 and desk.count("CONNECTION LOST") == 0
    assert {state.session_id for state in states} == {h.app.session_id}
    # The motors were on throughout.
    assert {state.torque_on for state in states} == {True}
    stage.holds()

    # Every joint target of the agent was a setpoint of a MOVE under a PERMIT.
    by_owner: dict[str, list[int]] = {}
    for index, target in enumerate(trace.targets):
        by_owner.setdefault(target.owner, []).append(index)
    assert set(by_owner) == {AGENT, HUMAN, RECOVERY}
    agents = copy.copy(trace)
    agents.targets = [trace.targets[index] for index in by_owner[AGENT]]
    agents.check_every_target_was_permitted()
    assert len(by_owner[AGENT]) == measured["agent targets"] >= 5
    # Every joint target of the person lay within the speed clamp of the one
    # before it, and came after the agent's and before the return to rest's.
    sent = np.array(h.sent())
    people = by_owner[HUMAN]
    assert people == list(range(people[0], people[-1] + 1))
    assert len(people) == measured["human targets"] > 140
    steps = np.abs(sent[people] - sent[[index - 1 for index in people]]).max(axis=1)
    assert steps.max() <= most + 1e-9
    assert by_owner[AGENT][-1] < people[0] and people[-1] < by_owner[RECOVERY][0]
    # But for those two light holds, no jaw command went out under the person
    # or while the takeover was open.
    others = [command for command in jaws if command != (HUMAN, True)]
    assert others and {owner for owner, _ in others} <= {AGENT, NOBODY}
    assert not any(open_ for _, open_ in others) and len(others) == len(jaws) - 2
    # The human program: one STOP, the takeover and the hand-back.
    assert person.tap.of(Stop) == [
        Stop(reason=logic.STOP_REASON, sender=logic.STOP_SENDER)
    ]
    requests = [request.request for request in person.tap.of(OwnerRequest)]
    assert requests == ["human", "handback"]
    assert h.heard(app_module.EVENT_HUMAN) == []
    assert person.end() == 0
    print(
        f"measured: run {took:.1f} s; {measured['agent targets']} joint targets "
        f"of the agent, {measured['human targets']} of the person (largest step "
        f"{steps.max():.4f} rad of {most:.4f}), {len(by_owner[RECOVERY])} of the "
        f"return to rest; the tip rose {measured['rose'] * 100:.1f} cm; "
        f"{len(jaws)} jaw commands, two light holds of them from the takeover "
        "to the hand-back"
    )


@link
@apai
def test_a_fast_hand_is_clamped_and_the_drive_goes_on(
    make_mock_stage: Callable[..., Stage],
) -> None:
    speed = 0.3  # the human joint speed limit here, rad/s
    stage = make_mock_stage(mock_config(human_speed=speed))
    h = stage.h
    stage.live_view()
    # The squeezed hand is 1 cm higher on the next pass, and stays there.
    script = Script(stage.cues).cue("go").take_over().squeeze().then(0.3)
    jump = [float(value) for value in np.array(HAND) + 0.01 * np.array(UP)]
    script.then(0.02, pos=jump).then(3.0).release()
    person = stage.person(script)
    solver = Solver.from_apai("mock")
    start = h.spy.pose()
    start_tip = solver.fk(solver.inside(start))[:3, 3]
    stage.live()
    stage.cues.give("go")
    owned = stage.shows(lambda s: s.owner == HUMAN)
    before = owned.health["clamped"]
    first = stage.sent()

    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up")
    done = stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    stage.holds()
    # The speed clamp cut steps of this drive, and no drive ended for it.
    clamped = done.health["clamped"] - before
    assert clamped >= 1
    assert h.heard(app_module.EVENT_HUMAN) == []
    assert (done.owner, done.owner_number) == (HUMAN, owned.owner_number)
    # Each joint target lies within the clamp of the one before it.
    most = SPEED_MARGIN * speed * h.config.executor.period_s
    targets = np.array([start, *h.sent()[first:]])
    steps = np.abs(np.diff(targets, axis=0)).max(axis=1)
    assert len(steps) > 100 and steps.max() <= most + 1e-9
    at_the_clamp = int((steps >= most - 1e-9).sum())
    assert at_the_clamp == clamped
    # APAI's human checker approved every one of them.
    approved = [row[1] for row in h.spy.approved[first:]]
    assert approved == list(range(approved[0], approved[0] + len(approved)))
    # The arm caught up with the hand: the gripper tip is 1 cm higher.
    rose = solver.fk(h.spy.pose())[:3, 3] - start_tip
    np.testing.assert_allclose(rose, [0.0, 0.0, 0.01], atol=2e-3)
    assert person.tap.of(Stop) == [
        Stop(reason=logic.STOP_REASON, sender=logic.STOP_SENDER)
    ]
    print(
        f"measured: {clamped} of {len(steps)} joint targets were cut by the clamp "
        f"of {most:.5f} rad; the largest step was {steps.max():.5f} rad; the tip "
        f"rose {rose[2] * 1000:.1f} mm; ended on the workstation: "
        f"{[line for line in person.said if 'ended:' in line][-1].split('ended: ')[1]}"
    )


# The two robot app files of demo 2

ROBOT_DEMO2 = CONFIGS / "robot_app_demo2.yaml"
ROBOT_HUMAN_CHECK = CONFIGS / "robot_app_human_check.yaml"
# The human joint box of both files, which lies inside rest.region_*.
INSIDE_REGION = {
    "limits.human.joint_min": [-2.5] * 6,
    "limits.human.joint_max": [2.5] * 6,
}
# Every key in which each file differs from robot_app.yaml, with its value.
DIFFERS: dict[Path, dict[str, Any]] = {
    ROBOT_DEMO2: {
        "human.enabled": True,
        "human.require_live_view": False,
        **INSIDE_REGION,
    },
    ROBOT_HUMAN_CHECK: {
        "human.enabled": True,
        "limits.human.max_joint_speed": 0.8,
        **INSIDE_REGION,
    },
}


def test_the_demo_2_robot_files_differ_from_the_shipped_one_only_where_listed(
    tmp_path: Path,
) -> None:
    plain = shipped_file("robot_app.yaml")
    assert plain["human.enabled"] is False
    for file, differs in DIFFERS.items():
        ours = shipped_file(file.name)
        assert set(ours) == set(plain), file.name
        found = {key: ours[key] for key in ours if ours[key] != plain[key]}
        assert found == differs, file.name
        # The file begins with a comment that names each of those keys.
        text = file.read_text(encoding="utf-8")
        assert text.startswith("# SPDX-FileCopyrightText: "), file.name
        assert "SPDX-License-Identifier: Apache-2.0" in text
        head = text[: text.index("\nrobot_id:")]
        assert all(line.startswith("#") for line in head.splitlines() if line)
        assert "differs from configs/robot_app.yaml in exactly these keys" in head
        listed = [line for line in head.splitlines() if line.startswith("# - ")]
        assert len(listed) == 3, file.name
        for key in differs:
            assert key in head, (file.name, key)
        assert "/home/" not in text and "~" not in text, file.name

        # It loads, and its human joint box lies inside the recovery region,
        # as human.enabled needs.
        config = load_config(file)
        assert config.human.enabled is True
        box, rest = config.limits.human, config.rest
        assert all(low >= edge for low, edge in zip(box.joint_min, rest.region_min))
        assert all(high <= edge for high, edge in zip(box.joint_max, rest.region_max))
        # With the human joint box of robot_app.yaml it would be refused.
        wide = tmp_path / file.name
        wide.write_text(
            text.replace("-2.5, " * 5 + "-2.5]", "-3.0, " * 5 + "-3.0]", 1),
            encoding="utf-8",
        )
        changed = flat(yaml.safe_load(wide.read_text(encoding="utf-8")))
        assert changed["limits.human.joint_min"] == [-3.0] * 6
        with pytest.raises(
            ConfigError,
            match=r"human\.enabled needs limits\.human inside rest\.region_\*: joint "
            r"index 0 has human -3\.0\.\.2\.5 and region -2\.5\.\.2\.5",
        ):
            load_config(wide)

    # Without a live view only on a simulated driver; the file of the headset
    # checks keeps the live view, so the driver does not matter to it.
    demo, check = load_config(ROBOT_DEMO2), load_config(ROBOT_HUMAN_CHECK)
    assert demo.human.require_live_view is False
    assert app_module.live_view_problem(demo, DRIVER_SIM) is None
    assert app_module.live_view_problem(demo, DRIVER_APAI_MOCK) is None
    for driver in (DRIVER_REAL, ""):
        refused = app_module.live_view_problem(demo, driver)
        assert refused is not None and "only with a simulated driver" in refused
        assert app_module.live_view_problem(check, driver) is None
    assert check.human.require_live_view is True
    # The speed of the headset checks is the rig's.
    rig = load_config(CONFIGS / "robot_app_rebot.yaml")
    assert check.limits.human.max_joint_speed == rig.limits.human.max_joint_speed
    assert rig.limits.human.max_joint_speed == 0.8  # the owner's cap for a person


# The waits of this file


def test_a_wait_that_runs_out_says_what_was_last_seen() -> None:
    with pytest.raises(AssertionError, match=r"a thing: not within 0\.05 s; last: 7"):
        wait_for(lambda: 0, "a thing", lambda: 7, wait_s=0.05)
    with pytest.raises(AssertionError, match=r"last: \[\]"):
        wait_for(list, "a thing", wait_s=0.05)  # without `last`: what read() gave

    # A console's lines: the last six are named.
    said = [f"line {number} | driver sim | owner nobody #0" for number in range(8)]
    desk = types.SimpleNamespace(said=said)
    assert console_shows(desk, " | driver sim | ", "line 3 ") == said[3]
    with pytest.raises(AssertionError) as failed:
        console_shows(desk, " | driver sim | ", " | owner human ", wait_s=0.05)
    assert f"; last: {said[2:]!r}" in str(failed.value)
    assert "' | owner human '" in str(failed.value) and said[1] not in str(failed.value)

    # A console's newest ROBOT_STATE: it is named.
    desk = types.SimpleNamespace(console=types.SimpleNamespace(_state=None))
    with pytest.raises(AssertionError, match=r"with an owner: .*; last: None"):
        console_has(desk, lambda state: True, "an owner", wait_s=0.05)
    desk.console._state = "a state"
    assert console_has(desk, lambda state: state == "a state", "an owner") == "a state"
    with pytest.raises(AssertionError, match=r"with an owner: .*; last: 'a state'"):
        console_has(desk, lambda state: False, "an owner", wait_s=0.05)


def test_the_help_block_is_read_past_a_status_line_between_its_lines() -> None:
    block = ["HELP NEEDED: why", "  jaws: empty", "  you can type:", "  takeover: x"]
    status = "robot connected | driver sim | owner nobody #2 | wait 90 s"
    earlier = ["HELP NEEDED: before", "  jaws: loaded", "  takeover: y", status]
    event = "[help_needed] the agent asks for help: why"
    assert help_lines([*earlier, event, *block, status], block[-1]) == block
    # Another thread of the console printed its status line between two of them.
    said = [*earlier, event, *block[:2], status, *block[2:], status]
    assert said[said.index(block[-1]) - 3] != block[0]
    assert help_lines(said, block[-1]) == block


# The programs as processes of their own

# The human program's first line on the script "lift".
LIFT_START = (
    'human program: synthetic controller "lift"; driver {driver}; '
    "arm model mock, 6 joints"
)


# The help request of the mock agent's pick and place against the robot app's
# own mock arm: APAI gave the grasp up after its attempts closed on air. And
# the real seconds those attempts may take (about 125 s were measured).
STUCK = "stuck: I could not hold 'red object' after "
STUCK_WAIT_S = 240.0


def recipe_files(folder: Path) -> tuple[str, Path, Path, Path]:
    """The socket of a test and the shipped files that name it.

    (socket path, the service's file, the view's, the human program's): each
    the shipped file with the test's socket, the service's and the view's with
    a port the system picks, the view's with a state folder of the test. The
    human program's also names, as its Teleop Session Server, a port that this
    process holds and never listens on: the script feed dials nothing, and
    nothing else could reach the server's own port either.
    """
    path = str(folder / "s.sock")
    socket_line = {'\nsocket_path: ""': f"\nsocket_path: {path}"}
    service = fitted(
        CONFIGS / "link_service.yaml", folder, {"port: 47610": "port: 0", **socket_line}
    )
    view = fitted(
        CONFIGS / "view.yaml",
        folder,
        {
            "port: 47620": "port: 0",
            "\nstate_dir: ../.state\n": f"\nstate_dir: {folder}/state\n",
            **socket_line,
        },
    )
    endpoint = {'endpoint: "127.0.0.1:47600"': f'endpoint: "{NO_SERVER}"'}
    human_file = fitted(CONFIGS / "human.yaml", folder, {**socket_line, **endpoint})
    return path, service, view, human_file


def human_script(config: Path, *args: str) -> StartScript:
    """bash run_human.sh --config PATH ..., with the variables the script needs."""
    build = Path(isaaccapture.__file__).resolve().parents[1]
    env = {
        "AGENTIC_REMOTE_HUMAN_PYTHON": sys.executable,
        "AGENTIC_REMOTE_ISAAC_CAPTURE_PY": str(build),
        "AGENTIC_REMOTE_APAI_SRC": os.environ[APAI_SRC_ENV],
    }
    return StartScript("run_human.sh", "--config", str(config), *args, env=env)


def in_order(lines: list[str], *wanted: tuple[str, ...]) -> list[str]:
    """The first lines that hold each tuple's parts, each after the one before."""
    found: list[str] = []
    at = 0
    for parts in wanted:
        rest = [line for line in lines[at:] if all(part in line for part in parts)]
        assert rest, f"no line with {parts} after {found}; lines: {lines[at:]}"
        found.append(rest[0])
        at = lines.index(rest[0], at) + 1
    return found


def transcript(name: str, lines: list[str]) -> None:
    """Print what a program wrote, for pytest -rP: the recipe as it ran."""
    print(f"--- {name} ---")
    for line in lines:
        print(line.replace(BELL_CHAR, "<bell>"))


@link
@apai
def test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Steps 1 to 8 of the recipe. What differs from the recipe by hand:

    the service, the view and the human program run on copies of their files
    that name a port the system picks, a socket and a state folder of the test
    and a Teleop Session Server port nobody listens on; the console and the
    view are started with python -m, not by their scripts; a local client in
    the role "tool" reads ROBOT_STATE; the view's page is asked over HTTP.
    """
    folder = tmp_path_factory.mktemp("d2")  # a socket path holds 107 bytes
    path, service_file, view_file, human_file = recipe_files(folder)
    solver = Solver.from_apai("mock")
    programs: list[Any] = []
    try:
        # 1. The robot link service.
        service = Program(SERVICE_MODULE, "--config", str(service_file))
        programs.append(service)
        endpoint = service.wait_line(READY).removeprefix(READY).split()[0]
        assert endpoint.startswith(f"{HOST}:")
        # 2. The robot app on the simulated arm, with the shipped file.
        app = Program(
            ROBOT_MODULE, "--config", str(ROBOT_DEMO2), "--arm", "sim",
            "--connect", endpoint,
        )  # fmt: skip
        programs.append(app)
        app.wait_line("robot app: session ")
        # 3. The operator console; state says why no takeover is possible yet.
        desk = ConsoleProcess(path)
        programs.append(desk)
        assert ", driver sim, " in desk.wait("WELCOME: ")
        console_shows(desk, " | driver sim | ", " | owner nobody #0 | ")
        console_shows(desk, " | view none | ", " | no human program | ")
        desk.type("state")
        desk.wait(f"takeover: not possible: {NO_HUMAN_PROGRAM}")
        # 4. The live view, without a camera; this robot app does not need it.
        view = Program(VIEW_MODULE, "--config", str(view_file), "--source", "none")
        programs.append(view)
        line = view.wait_line("live view: open http://127.0.0.1:")
        browser = Browser(int(line.split("http://127.0.0.1:")[1].split("/")[0]))
        try:
            page = browser.status()
        finally:
            browser.close()
        assert page["picture"]["overlay"] == NO_PICTURE
        console_shows(desk, " 1 viewer | ", " | view none | ")
        watcher = Watcher(path)
        programs.append(watcher)
        rest = watcher.pose()
        assert rest == pytest.approx(REST, abs=1e-6)
        rest_tip = solver.fk(solver.inside(rest))[:3, 3]

        # 5. The human program on the script "lift", started with its script.
        person = human_script(human_file, "--controller", "synthetic:lift")
        programs.append(person)
        started = person.wait_line('human program: synthetic controller "lift"')
        assert started == LIFT_START.format(driver="sim")
        # 6. The script's right B, its hold, the squeeze, the lift, the release.
        stop = f"[stop] STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; "
        desk.wait(stop, 2 * WAIT_S)
        owned = watcher.ws.wait_state(lambda s: s.owner == HUMAN, WAIT_S)
        taken = f"[owner] {app_module.TAKEN_OVER.format(number=owned.owner_number)}"
        desk.wait(taken)
        owner = f" | owner human #{owned.owner_number} | "
        console_shows(
            desk, owner, " | moving ", " | human driving | ", wait_s=2 * WAIT_S
        )
        person.wait_line("human program: grip up")
        held = watcher.ws.wait_state(
            lambda s: not human(s)["driving"] and s.motion == "hold", WAIT_S
        )
        assert (held.owner, held.torque_on) == (HUMAN, True)
        console_shows(desk, owner, " | hold | ", " | takeover open | ")
        # The pose rose off rest: the gripper tip is 5 cm higher.
        rose = solver.fk(watcher.pose())[:3, 3] - rest_tip
        np.testing.assert_allclose(rose, [0.0, 0.0, 0.05], atol=3e-3)

        # 7. rest and path is clear; then the script's right A gives the arm back.
        desk.type("rest")
        desk.answer(ASK_REST, PATH_IS_CLEAR)
        desk.wait("rest: RESULT completed", REST_WAIT_S)
        back = f"[owner] {app_module.HANDED_BACK}"
        desk.wait(back)
        ended = watcher.ws.wait_state(lambda s: not human(s)["takeover_open"], WAIT_S)
        assert (ended.owner, ended.motion) == (NOBODY, "rest_hold")
        assert ended.torque_on is True
        nobody = f" | owner nobody #{ended.owner_number} | "
        console_shows(desk, nobody, " | rest_hold | ")
        assert watcher.pose() == pytest.approx(rest, abs=0.05)
        # In the order of the recipe's steps 6 and 7.
        in_order(
            list(desk.said),
            (stop,),
            (taken,),
            (owner, " | moving ", " | human driving | "),
            (owner, " | hold | ", " | takeover open | "),
            ("rest: RESULT completed",),
            (back,),
            (nobody,),
        )

        # 8. The end of every program, each by its own way out, exit code 0.
        desk.type("shutdown")
        desk.answer(f"{ASK_YES}shut down", YES)
        assert app.wait_exit() == 0
        desk.wait("ROBOT CONNECTION LOST: ")
        desk.type("quit")
        assert desk.wait_exit() == 0
        watcher.close()
        for program in (person, view, service):
            program.signal(signal.SIGINT)  # Ctrl+C at its terminal
            assert program.wait_exit() == 0
        assert "Traceback" not in "\n".join(person.lines)
        for name, lines in (
            ("1 service", service.lines),
            ("2 Thor terminal", app.lines),
            ("3 console", desk.said),
            ("4 view", view.lines),
            ("5 human program", person.lines),
        ):
            transcript(name, lines)
    finally:
        for program in programs:
            program.close()


@link
@apai
def test_demo_2_with_the_agent_failing_first_runs_as_processes(
    recipe_folder: Path,  # noqa: F811
) -> None:
    """The recipe's variant: a task of the agent ends in a help request first.

    The robot app runs on APAI's mock arm with configs/robot_app_mock.yaml and
    the agent program at a terminal of its own on configs/agent_mock.yaml, as
    in demo 1 by hand; the human program is started once the agent has asked
    for help. What differs from the recipe by hand is what differs in the test
    above, and the agent's file names the test's socket and state folder.
    """
    folder = recipe_folder
    path, service_file, view_file, human_file = recipe_files(folder)
    state = f"state_dir: {folder}/state\n"
    agent_file = fitted(
        AGENT_MOCK, folder, {"\nlink:\n": f"\nlink:\n  socket_path: {path}\n"}, state
    )
    solver = Solver.from_apai("mock")
    programs: list[Any] = []
    try:
        # Terminals 1 to 4, with the robot app on APAI's mock arm.
        service = Program(SERVICE_MODULE, "--config", str(service_file))
        programs.append(service)
        endpoint = service.wait_line(READY).removeprefix(READY).split()[0]
        app = Program(
            ROBOT_MODULE, "--config", str(ROBOT_MOCK), "--arm", "apai",
            "--driver", "mock", "--connect", endpoint,
        )  # fmt: skip
        programs.append(app)
        app.wait_line("robot app: session ")
        desk = ConsoleProcess(path)
        programs.append(desk)
        assert ", driver apai-mock, " in desk.wait("WELCOME: ")
        console_shows(desk, " | driver apai-mock | ", " | owner nobody #0 | ")
        view = Program(VIEW_MODULE, "--config", str(view_file), "--source", "none")
        programs.append(view)
        view.wait_line("live view: open http://127.0.0.1:")
        watcher = Watcher(path)
        programs.append(watcher)
        rest = watcher.pose()

        # The sixth terminal: the agent program. The mock arm of the robot app
        # has nothing between its jaws, and the agent decides from the jaw
        # reading the robot app measures: every grasp closes on air. APAI
        # tries the grasp again, gives up as stuck, and the agent asks for help.
        agent = AgentTerminal(agent_file)
        programs.append(agent)
        agent.wait(INVENTED_WARNING, START_WAIT_S)
        agent.wait(CLEAR_WARNING)
        agent.wait("ready. Ctrl+C at this terminal is STOP", START_WAIT_S)
        agent.wait("task> ")
        agent.type(PICK)
        agent.answer("Type yes to proceed", YES, START_WAIT_S)
        agent.wait(f"HELP NEEDED: {STUCK}", STUCK_WAIT_S)
        agent.wait("task> ")
        # The console shows the help request, the wait and the takeover line.
        desk.wait(f"HELP NEEDED: {STUCK}")
        desk.wait(f"  takeover: not possible: {NO_HUMAN_PROGRAM}")
        waiting = watcher.ws.wait_state(lambda s: s.assistance_wait_s is not None)
        assert (waiting.owner, waiting.torque_on) == (NOBODY, True)
        console_shows(desk, f" | owner nobody #{waiting.owner_number} | ", " | wait ")
        raised = watcher.pose()
        assert max(abs(at - was) for at, was in zip(raised, rest)) > 0.5
        raised_tip = solver.fk(solver.inside(raised))[:3, 3]

        # Terminal 5: the human program. Its right B ends the wait and takes
        # the arm over; the squeezed hand lifts the gripper 5 cm.
        person = human_script(human_file, "--controller", "synthetic:lift")
        programs.append(person)
        started = person.wait_line('human program: synthetic controller "lift"')
        assert started == LIFT_START.format(driver="apai-mock")
        stop = f"[stop] STOP from {logic.STOP_SENDER}: {logic.STOP_REASON}; "
        desk.wait(stop, 2 * WAIT_S)
        stopped = watcher.ws.wait_state(lambda s: s.stop_number == 1, WAIT_S)
        assert stopped.assistance_wait_s is None  # the STOP cancelled the wait
        owned = watcher.ws.wait_state(lambda s: s.owner == HUMAN, WAIT_S)
        assert owned.assistance_wait_s is None
        taken = f"[owner] {app_module.TAKEN_OVER.format(number=owned.owner_number)}"
        desk.wait(taken)
        owner = f" | owner human #{owned.owner_number} | "
        console_shows(
            desk, owner, " | moving ", " | human driving | ", wait_s=2 * WAIT_S
        )
        person.wait_line("human program: grip up")
        held = watcher.ws.wait_state(
            lambda s: not human(s)["driving"] and s.motion == "hold", WAIT_S
        )
        assert (held.owner, held.torque_on) == (HUMAN, True)
        rose = solver.fk(watcher.pose())[:3, 3] - raised_tip
        np.testing.assert_allclose(rose, [0.0, 0.0, 0.05], atol=3e-3)

        # rest and path is clear at the console; the script's right A.
        desk.type("rest")
        desk.answer(ASK_REST, PATH_IS_CLEAR)
        desk.wait("rest: RESULT completed", RECIPE_REST_WAIT_S)
        back = f"[owner] {app_module.HANDED_BACK}"
        desk.wait(back)
        ended = watcher.ws.wait_state(lambda s: not human(s)["takeover_open"], WAIT_S)
        assert (ended.owner, ended.motion) == (NOBODY, "rest_hold")
        assert ended.torque_on is True
        assert watcher.pose() == pytest.approx(rest, abs=0.05)
        # The console's status line shows it: the next task can be typed. (The
        # agent program refuses one typed while its newest ROBOT_STATE still
        # shows the takeover open.)
        nobody = f" | owner nobody #{ended.owner_number} | "
        console_shows(desk, nobody, " | rest_hold | ")

        # The agent's next task runs: no program was restarted.
        agent.type(QUICK)
        agent.answer("Type yes to proceed", YES, START_WAIT_S)
        agent.wait("task completed: the arm holds where it is", START_WAIT_S)
        agent.wait("success: True")
        agent.wait("task> ")
        again = ended.owner_number + 1
        desk.wait(f"[owner] new task: the agent owns the arm, owner number {again}")
        desk.wait("[owner] the agent gave the arm back: task completed")
        assert len([line for line in list(desk.said) if "WELCOME: " in line]) == 1

        # The end of every program. Before it ends the agent program asks
        # whether the arm returns to rest. On the typed phrase the robot app
        # runs its own return to rest, which ends as the functional one: rest
        # is measured, the jaws close and the motors go off.
        agent.end_input()
        agent.answer("End of the session: return the arm to rest", PATH_IS_CLEAR)
        agent.wait(
            "the arm lies at rest: the jaws are closed, the motors are off",
            RECIPE_REST_WAIT_S,
        )
        assert agent.wait_exit() == launcher.EXIT_FAILED
        assert "1 task(s) of this session ended without success" in agent.text
        parked = watcher.ws.wait_state(lambda s: s.motion == "rest_unpowered", WAIT_S)
        assert (parked.torque_on, parked.stop_level) == (False, "none")
        assert watcher.pose() == pytest.approx(rest, abs=0.05)
        desk.wait("[release] torque release")
        desk.type("shutdown")
        desk.answer(f"{ASK_YES}shut down", YES)
        assert app.wait_exit() == 0
        desk.wait("ROBOT CONNECTION LOST: ")
        desk.type("quit")
        assert desk.wait_exit() == 0
        watcher.close()
        for program in (person, view, service):
            program.signal(signal.SIGINT)  # Ctrl+C at its terminal
            assert program.wait_exit() == 0
        for name, lines in (
            ("2 Thor terminal", app.lines),
            ("3 console", desk.said),
            ("5 human program", person.lines),
            ("6 agent", agent.text.splitlines()),
        ):
            transcript(name, lines)
    finally:
        for program in programs:
            program.close()


# The trigger and the jaws under a person


@link
@apai
def test_a_person_opens_the_jaws_the_agent_left_closed_and_gives_the_arm_back(
    make_mock_stage: Callable[..., Stage],
) -> None:
    """The jaws, the gripper's fingers, under the trigger of a person's controller.

    A jaw opening is a share of the jaw travel: 1.0 fully open, 0.0 closed.
    Every HUMAN_SETPOINT carries the one the trigger asks for. The grip level
    is the opening in force at the takeover: the robot app ignores the trigger
    until the opening it asks for has reached that level.
    """
    import test_apai_app as apai

    stage = make_mock_stage()
    h = stage.h
    desk = stage.console()
    stage.live_view()
    agent = stage.stack.agent()
    script = Script(stage.cues).cue("go").take_over()
    script.cue("lift").squeeze().lift(0.01, 1.0).release()
    script.cue("pull").then(0.2, trigger=1.0)
    script.cue("let go").then(0.2, trigger=0.0).hand_back("given back")
    person = stage.person(script, "--report-s", "0.2")
    # Who owned the arm at each jaw command the driver got from here on.
    owners: list[str] = []
    before = len(h.spy.jaw_commands)
    h.spy.after_jaw_command.append(
        lambda *_: owners.append(h.app.snapshot().owner.value)
    )
    stage.live()
    person.wait_said("human program: right controller seen")

    def jaw_facts(state: RobotState) -> dict[str, Any]:
        """The robot app's facts of the jaws under the trigger; {} without them."""
        return human(state).get("jaws", {})

    def asks(number: int, opening: float, count: int = 25) -> None:
        """Wait for this many HUMAN_SETPOINTs of this owner number that ask for it."""
        wait_for(
            lambda: (
                len(person.setpoints(owner_number=number, jaw_opening=opening)) >= count
            ),
            f"{count} HUMAN_SETPOINTs that ask for a jaw opening of {opening}",
            lambda: person.tap.sent[-3:],
        )

    # The agent's task: it closes the jaws on an object half the jaw travel
    # wide and asks for help. The grip keeps the stiffness of its close: the
    # help request sends the jaws nothing.
    assert agent.ws.start_task().status == "completed"
    grip = 0.5  # the effort of the agent's jaw commands
    for opening, mode in ((1.0, PLAIN), (0.0, CLOSE_UNTIL_CONTACT)):
        moved = agent.ws.wait_result(agent.ws.gripper(opening, grip, mode=mode), WAIT_S)
        assert moved.status == "completed", moved.reason
    asked = agent.ws.request_owner("help_needed", reason="the grasp is not confirmed")
    assert agent.ws.wait_result(asked, WAIT_S).status == "completed"
    left = stage.shows(lambda s: s.owner == NOBODY and s.assistance_wait_s is not None)
    assert left.gripper.opening == pytest.approx(0.5) and left.gripper.may_be_holding
    assert "jaws" not in human(left)
    agents = len(h.spy.jaw_commands)
    assert grip > apai.LIGHT
    assert h.spy.jaw_commands[-1] == (apai.JAWS_CLOSED, pytest.approx(grip))
    assert set(owners) <= {AGENT, NOBODY}
    # The help request prompts the person at the console: its block ends
    # with the takeover line.
    assert desk.wait_long("  takeover: ", WAIT_S) == f"  {TAKEOVER_POSSIBLE}"

    # The takeover, with the trigger released: every HUMAN_SETPOINT asks for
    # fully open jaws, and the jaws stay as the agent left them.
    stage.cues.give("go")
    owned = stage.shows(lambda s: s.owner == HUMAN)
    number = owned.owner_number
    # Right B's STOP cancelled the assistance wait before the takeover.
    (stop,) = h.heard(STOP_EVENT)
    assert stop[2]["sender"] == logic.STOP_SENDER and stop[2]["wait_cancelled"]
    assert owned.assistance_wait_s is None
    # Right B's STOP sent the jaws nothing: no jaw command was running, and
    # what the agent grips stays gripped.
    assert h.spy.jaw_commands[agents:] == []
    asks(number, 1.0)
    short = stage.shows(lambda s: jaw_facts(s).get("asked") == 1.0)
    assert jaw_facts(short) == {
        "matched": False,
        "grip_level": 0.0,  # the agent's last jaw target
        "opened": False,
        "asked": 1.0,
        "refusal": NOT_MATCHED,
    }
    person.wait_said(
        f"; jaws follow the trigger no, grip level 0.00, trigger asks 1.00: "
        f"{NOT_MATCHED}"
    )
    # The grip is squeezed and the hand rises 1 cm: the arm follows, the jaws
    # do not, and the note says what the trigger is to do.
    stage.cues.give("lift")
    stage.shows(lambda s: driving(s, 10))
    person.wait_said("human program: grip up")
    stage.shows(lambda s: not human(s)["driving"] and s.motion == "hold")
    followed = person.setpoints(engaged=True)
    assert len(followed) > 40
    assert {sent.note for sent in followed} == {logic.NOTE_TRIGGER_PULL}
    assert {sent.jaw_opening for sent in followed} == {1.0}
    desk.wait_long(f" | note: {logic.NOTE_TRIGGER_PULL} | ", WAIT_S)
    assert len(h.spy.jaw_commands) == agents
    held = stage.state
    assert held.gripper.opening == pytest.approx(0.5) and held.gripper.may_be_holding
    assert not jaw_facts(held)["matched"]

    # The trigger is pulled all the way, to the grip level: it has matched.
    # The jaws hold what they hold: nothing is asked of them.
    stage.cues.give("pull")
    asks(number, 0.0)
    matched = stage.shows(lambda s: jaw_facts(s).get("asked") == 0.0)
    assert jaw_facts(matched) == {
        "matched": True,
        "grip_level": 0.0,
        "opened": False,
        "asked": 0.0,
        "refusal": "",
    }
    person.wait_said(
        "; jaws follow the trigger yes, grip level 0.00, trigger asks 0.00"
    )
    assert len(h.spy.jaw_commands) == agents
    assert stage.state.gripper.opening == pytest.approx(0.5)

    # The trigger is released: the jaws open in steps until they are fully open,
    # which proves them empty.
    began = time.monotonic()
    stage.cues.give("let go")
    opened = stage.shows(lambda s: jaw_facts(s).get("opened"))
    took = time.monotonic() - began
    assert jaw_facts(opened) == {
        "matched": True,
        "grip_level": 0.0,
        "opened": True,
        "asked": 1.0,
        "refusal": "",
    }
    gripper = opened.gripper
    assert (gripper.opening, gripper.last_target) == (pytest.approx(1.0),) * 2
    assert (gripper.may_be_holding, gripper.load) == (False, "empty")
    steps = h.spy.jaw_commands[agents:]
    # The mock counts the jaw position from 0 open: 0.5 to 0 in steps of 0.05.
    wanted = [0.5 - 0.05 * count for count in range(1, 11)]
    assert [position for position, _ in steps] == pytest.approx(wanted, abs=1e-9)
    assert [effort for _, effort in steps] == [OPEN_EFFORT] * 10
    assert owners[agents - before :] == [HUMAN] * 10
    person.wait_said(
        "; jaws follow the trigger yes, grip level 0.00, trigger asks 1.00"
    )
    # The trigger never asked for anything between its two ends.
    asked_for = {sent.jaw_opening for sent in person.setpoints(owner_number=number)}
    assert asked_for == {1.0, 0.0}

    # rest and path is clear at the console, then the script's right A at rest:
    # with the jaws proven empty the hand-back is taken.
    rest_from_the_console(desk)
    ended = stage.shows(lambda s: not human(s)["takeover_open"])
    assert (ended.owner, ended.motion) == (NOBODY, "rest_hold")
    assert "jaws" not in human(ended)
    desk.wait(f"[owner] {app_module.HANDED_BACK}")
    stage.cues.give("given back")
    assert h.spy.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    # Nothing closed the jaws again, and nobody else's trigger moves them.
    after = h.spy.jaw_commands[agents + 10 :]
    assert all(position == pytest.approx(0.0) for position, _ in after), after
    assert ended.gripper.opening == pytest.approx(1.0)
    requests = [request.request for request in person.tap.of(OwnerRequest)]
    assert requests == ["human", "handback"]
    assert person.tap.of(Stop) == [
        Stop(reason=logic.STOP_REASON, sender=logic.STOP_SENDER)
    ]
    wait_for(
        lambda: len(person.setpoints(owner_number=ended.owner_number)) >= 5,
        "HUMAN_SETPOINTs that quote the hand-back",
        lambda: person.tap.sent[-3:],
    )
    given_back = person.setpoints(owner_number=ended.owner_number)
    assert {sent.jaw_opening for sent in given_back} == {None}
    stage.holds()
    assert person.end() == 0
    print(
        f"measured: the jaws opened from 0.5 in {len(steps)} steps and {took:.2f} s; "
        f"{len(after)} jaw command(s) from there to the hand-back: {after}"
    )
