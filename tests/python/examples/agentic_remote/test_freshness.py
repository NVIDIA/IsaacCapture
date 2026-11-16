# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How old a human target and the live view are when the robot app acts on them.

A human target is a joint target from a person's hand, sent in a
HUMAN_SETPOINT link message by the human program, the program that turns
controller data into joint targets. Its sample age is how old the controller
data behind it are. The live view is the operator's browser page with the
camera picture; the view program reports its age in VIEW_STATUS link messages.
A takeover makes a person the control owner, the one who may move the arm; a
human drive is one stretch of human motion.

Both ages must hold every stretch of time up to the moment the robot app acts:
the sender's own work, the link message's way to the Thor (the machine on the
robot), its wait at the robot app. The sender says when it sent, on the robot
app's monotonic clock as it reads that clock off its newest ROBOT_STATE
(sent_mono_s), and the robot app adds the time since then. That clock reading
is never later than the true moment, so an age is never less than it is.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.human.__main__ import as_sent
from isaaccapture_examples.agentic_remote.human.logic import HumanLogic
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_SOURCE_TELEOP,
    HumanSetpoint,
    Result,
    ViewStatus,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app.app import human_target
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.human import HumanInput
from isaaccapture_examples.agentic_remote.robot_app.interfaces import Owner
from isaaccapture_examples.agentic_remote.view.model import ViewModel
from link_bench import Bench, robot_state
from link_stack import Station, until
from test_human_on_thor import Thor, ask_at_socket, feed_at_socket, taken
from test_takeover import (
    OPEN_THE_VIEW,
    REST,
    Hand,
    ask_takeover,
    drive,
    drives,
    feed,
    human_config,
    human_facts,
    keep_human_fed,
    moved,
    setpoint,
    state_now,
    take_over,
)
from test_view_model import JPEG, NOT_LIVE, T0, WALL0, Page, model_with_state, near
from test_view_model import shown as picture_shown
from test_view_model import state_at as view_state_at

from conftest import AppHarness

WAIT_S = 10.0  # real seconds a test gives another thread before it fails
# The two limits of the demo values, human.sample_age_limit_s and
# human.view_age_limit_s, and how far a datum of these tests lies from one.
SAMPLE_LIMIT_S = 0.25
VIEW_LIMIT_S = 0.5
NEAR_S = 0.002
NO_DATA = "no fresh controller data: "
VIEW_NOT_FRESH = "the live view is not fresh: "
# A HUMAN_SETPOINT of a ready controller with the grip squeezed, its data
# 0.03 s old, computed on a ROBOT_STATE whose clock_mono_s is 500.
SQUEEZE = HumanSetpoint(
    session_id="s",
    owner_number=3,
    stop_number=1,
    seq=9,
    source=HUMAN_SOURCE_TELEOP,
    stage="ready",
    sample_age_s=0.03,
    engaged=True,
    grip_released=False,
    q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    jaw_opening=None,
    state_clock_mono_s=500.0,
)


@pytest.fixture
def config() -> RobotAppConfig:
    """The demo values with human driving switched on, as test_takeover.py's."""
    return human_config(RobotAppConfig.default())


@pytest.fixture
def human_app(make_robot_app: Callable[..., AppHarness]) -> AppHarness:
    """A started robot app at rest with human driving switched on."""
    return make_robot_app()


class Way:
    """How a test's HUMAN_SETPOINTs reach the robot app.

    Over the robot connection, the link between the two machines, as from a
    human program at the workstation; or at the robot app's own socket, as
    from a human program on the Thor. The live view's VIEW_STATUS goes over
    the robot connection either way.
    """

    def __init__(self, h: AppHarness, station: Station | None) -> None:
        self.h, self.station = h, station
        self._client = h.client if station is None else station.ws

    def send(self, message: HumanSetpoint) -> None:
        """Send it; on return it may still be on its way."""
        self._client.send(message)

    def feed(self, message: HumanSetpoint | None = None, *, view: bool = True) -> None:
        """Send it, with view a live VIEW_STATUS too; the app has taken them.

        Without a message: a ready controller with the grip released.
        """
        if self.station is None:
            feed(self.h, message, view=view)
        else:
            feed_at_socket(self.h, self.station, message, view=view)

    def ask_takeover(self) -> Result:
        """The takeover's request; its final RESULT."""
        if self.station is None:
            return ask_takeover(self.h)
        return ask_at_socket(self.station, "human")

    def take_over(self) -> None:
        """A person owns the arm and has released the grip."""
        self.feed()
        result = self.ask_takeover()
        assert result.status == "completed", result.reason
        number = result.detail["owner_number"]
        self._client.wait_state(lambda s: s.owner_number == number, WAIT_S)
        self.feed()


@pytest.fixture(params=["robot connection", "socket"])
def way(
    request: pytest.FixtureRequest,
    human_app: AppHarness,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Way]:
    """Each test that takes it runs once per placement of the human program."""
    if request.param == "robot connection":
        yield Way(human_app, None)
        return
    thor = Thor(human_app, tmp_path_factory.mktemp("fr"))  # a short path
    try:
        yield Way(human_app, thor.attach())
    finally:
        thor.close()


def test_the_two_age_limits_keep_their_names_and_values():
    human = RobotAppConfig.default().human
    assert human.sample_age_limit_s == SAMPLE_LIMIT_S
    assert human.view_age_limit_s == VIEW_LIMIT_S


# The sample age of a human target


@pytest.mark.parametrize(
    "stretch", ["at the human program", "on the way", "since the ROBOT_STATE"]
)
@pytest.mark.parametrize(
    ("total_s", "fresh"),
    [(SAMPLE_LIMIT_S - NEAR_S, True), (SAMPLE_LIMIT_S + NEAR_S, False)],
    ids=["just inside the limit", "just outside the limit"],
)
def test_every_stretch_of_time_counts_toward_the_sample_age_limit(
    way, stretch, total_s, fresh
):
    h = way.h
    now = h.clock.monotonic()
    if stretch == "at the human program":
        # All of the age is in what the human program measured when it sent.
        fields = {"sample_age_s": total_s, "sent_mono_s": now}
    elif stretch == "on the way":
        # Fresh when it was sent; the rest it spent on its way to the app.
        fields = {"sample_age_s": 0.02, "sent_mono_s": now - (total_s - 0.02)}
    else:
        # A human program of before names only the ROBOT_STATE it computed
        # on: the app counts from that one's clock.
        made = now - (total_s - 0.02)
        fields = {"sample_age_s": 0.02, "sent_mono_s": None, "state_clock_mono_s": made}
    way.feed(setpoint(h, **fields))
    result = way.ask_takeover()
    if fresh:
        assert result.status == "completed", result.reason
        return
    assert result.status == "refused"
    old = rf"{NO_DATA}0\.25\d s old, the limit is 0\.25 s"
    assert re.fullmatch(old, result.reason), result.reason
    assert h.app.snapshot().owner is Owner.NOBODY and not h.app.takeover_open


def test_a_sample_260_ms_old_on_arrival_starts_no_drive(way):
    """The review's case: 0.14 s old when sent, 0.12 s on its way."""
    h = way.h
    way.take_over()
    ended = drives(h)
    made_at = h.clock.monotonic()
    late = setpoint(
        h,
        moved(REST),
        sample_age_s=0.14,
        state_clock_mono_s=made_at,
        sent_mono_s=made_at,
    )
    h.clock.advance(0.12)
    way.feed(late, view=False)
    h.idle()
    # No motor target, and the arm holds as it did, with the motors on.
    assert h.arm.sent == [] and ended == []
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.snapshot().owner is Owner.HUMAN
    state = state_now(h)
    old = rf"{NO_DATA}0\.2[67]\d s old, the limit is 0\.25 s"
    assert re.fullmatch(old, state.health["human"]["refusal"])
    assert state.controller_age_s >= 0.26
    assert state.health["human"]["round_trip_s"] >= 0.12


def test_the_wait_at_the_robot_app_counts_toward_the_sample_age(way):
    """A HUMAN_SETPOINT waits while the app's dispatcher handles another."""
    h = way.h
    way.take_over()
    ended = drives(h)
    busy, release = threading.Event(), threading.Event()
    handle = h.app._handlers[ViewStatus]

    def held(message: ViewStatus) -> None:
        busy.set()
        assert release.wait(WAIT_S)
        handle(message)

    h.app._handlers[ViewStatus] = held
    try:
        now = h.clock.monotonic()
        h.client.send(ViewStatus(live=True, age_s=0.0, sent_mono_s=now))
        assert busy.wait(WAIT_S), "the dispatcher did not take the VIEW_STATUS"
        # Sent now with data 0.14 s old; taken up 0.12 s later.
        late = setpoint(h, moved(REST), sample_age_s=0.14, sent_mono_s=now)
        way.send(late)
        h.clock.advance(0.12)
    finally:
        release.set()
    until(lambda: taken(h, late.seq), "the HUMAN_SETPOINT at the robot app")
    h.idle()
    assert h.arm.sent == [] and ended == []
    old = rf"{NO_DATA}0\.2[67]\d s old, the limit is 0\.25 s"
    assert re.fullmatch(old, human_facts(h)["refusal"])


class LateHand(Hand):
    """A hand whose HUMAN_SETPOINTs take longer to arrive after some setpoints."""

    late_from = 0  # how many setpoints the arm was sent before they are late
    late_s = 0.0  # how long a late one was on its way

    def __call__(self) -> HumanSetpoint | None:
        message = super().__call__()
        if message is None or len(self.h.arm.sent) < self.late_from:
            return message
        sent = message.sent_mono_s - self.late_s
        return dataclasses.replace(message, sent_mono_s=sent)


def test_a_drive_ends_in_a_hold_when_its_targets_take_too_long_on_the_way(
    human_app,
):
    h = human_app
    take_over(h)
    hand = LateHand(h)
    hand.late_from, hand.late_s = 3, 0.3
    keep_human_fed(h, hand)
    feed(h, hand())
    ended = drives(h)
    drive(h, hand, moved(REST), sends=6)
    # Three setpoints on fresh targets; the fourth target was 0.3 s on its way.
    [end] = ended
    assert (end.status, end.sent) == ("stale", 3)
    old = r"controller data stale \(0\.3\d\d s old, the limit is 0\.25 s\)"
    assert re.fullmatch(old, end.reason), end.reason
    # Every end of a human drive is a hold with the motors on.
    state = state_now(h)
    assert (state.motion, state.owner, state.torque_on) == ("hold", "human", True)
    assert h.arm.torque_on() and h.arm.released_at == []
    assert len(h.sent()) == 3
    told = h.wait_local_event("human")[1]
    assert told.startswith("human drive ended: controller data stale (0.3")


def late_views(h: AppHarness, late_from: int, late_s: float) -> None:
    """A live VIEW_STATUS before every setpoint the arm is sent.

    Once the arm was sent late_from setpoints, each was late_s on its way.
    """
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        late = late_s if len(h.arm.sent) >= late_from else 0.0
        sent = h.clock.monotonic() - late
        h.client.send(ViewStatus(live=True, age_s=0.1, sent_mono_s=sent))
        real(q)

    h.arm.send_target = send_target


def test_a_drive_ends_in_a_hold_when_the_views_reports_take_too_long(human_app):
    h = human_app
    take_over(h)
    hand = Hand(h)
    keep_human_fed(h, hand, view=False)
    late_views(h, late_from=3, late_s=0.45)
    feed(h, hand())
    ended = drives(h)
    drive(h, hand, moved(REST), sends=8)
    # The report that went out with the fourth setpoint was 0.55 s old.
    [end] = ended
    assert (end.status, end.sent) == ("view", 4)
    old = r"the live view is not fresh \(0\.5\d\d s old, the limit is 0\.5 s\)"
    assert re.fullmatch(old, end.reason), end.reason
    state = state_now(h)
    assert (state.motion, state.owner, state.torque_on) == ("hold", "human", True)
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_human_setpoint_that_names_no_time_has_no_known_age(way):
    h = way.h
    # Neither when it was sent nor a ROBOT_STATE it was computed on.
    way.feed(setpoint(h, sent_mono_s=None, state_clock_mono_s=None))
    unknown = f"{NO_DATA}the sample age is unknown"
    assert human_facts(h)["takeover"] == unknown
    result = way.ask_takeover()
    assert (result.status, result.reason) == ("refused", unknown)
    assert state_now(h).controller_age_s is None
    # With the time it was sent, the same data are fresh.
    way.feed()
    assert way.ask_takeover().status == "completed"


@pytest.mark.parametrize("ahead_s", [0.001, 3600.0])
def test_a_time_from_the_future_is_not_trusted(human_app, ahead_s):
    """As after a jump of the sender's clock, or of the robot app's."""
    h = human_app
    later = h.clock.monotonic() + ahead_s
    feed(h, setpoint(h, sent_mono_s=later))
    assert human_facts(h)["takeover"] == f"{NO_DATA}the sample age is unknown"
    # A ROBOT_STATE of the future, where that is all the setpoint names.
    feed(h, setpoint(h, sent_mono_s=None, state_clock_mono_s=later))
    assert human_facts(h)["takeover"] == f"{NO_DATA}the sample age is unknown"
    # Fresh controller data, and a view that says it was sent in the future.
    h.client.send(ViewStatus(live=True, age_s=0.0, sent_mono_s=later))
    feed(h, view=False)
    unknown = f"{VIEW_NOT_FRESH}the view's age is unknown{OPEN_THE_VIEW}"
    assert human_facts(h)["takeover"] == unknown
    state = state_now(h)
    assert state.view_age_s is None
    assert state.controller_age_s == pytest.approx(0.02)
    assert ask_takeover(h).status == "refused"


def test_a_time_long_ago_makes_the_datum_that_old(human_app):
    h = human_app
    long_ago = h.clock.monotonic() - 900.0
    feed(h, setpoint(h, sent_mono_s=long_ago))
    old = rf"{NO_DATA}900\.0\d\d s old, the limit is 0\.25 s"
    assert re.fullmatch(old, human_facts(h)["takeover"])
    h.client.send(ViewStatus(live=True, age_s=0.0, sent_mono_s=long_ago))
    feed(h, view=False)
    old = rf"{VIEW_NOT_FRESH}900\.0\d\d s old, the limit is 0\.5 s"
    assert re.fullmatch(old + re.escape(OPEN_THE_VIEW), human_facts(h)["takeover"])


def test_a_human_targets_age_is_the_reported_one_plus_the_time_since_it_was_sent():
    sent = dataclasses.replace(SQUEEZE, sent_mono_s=500.2)
    target = human_target(sent, 500.3)
    assert target.sample_age_s == pytest.approx(0.03 + 0.1)
    assert (target.arrived_mono, target.round_trip_s) == (500.3, pytest.approx(0.3))
    # The time since it arrived comes on top, wherever the age is asked.
    newest = HumanInput()
    newest.put(target)
    assert newest.sample_age_s(500.35) == pytest.approx(0.03 + 0.1 + 0.05)
    # Without the time it was sent: from the ROBOT_STATE it was computed on.
    assert human_target(SQUEEZE, 500.3).sample_age_s == pytest.approx(0.03 + 0.3)
    # With neither, and with a time that is no time before now: unknown.
    bare = dataclasses.replace(SQUEEZE, state_clock_mono_s=None)
    assert human_target(bare, 500.3).sample_age_s is None
    for when in (500.3001, float("inf"), float("nan")):
        told = dataclasses.replace(SQUEEZE, sent_mono_s=when)
        assert human_target(told, 500.3).sample_age_s is None, when
        ahead = dataclasses.replace(SQUEEZE, state_clock_mono_s=when)
        assert human_target(ahead, 500.3).sample_age_s is None, when
    # A time it was sent that cannot be used is not replaced by the state's.
    assert human_target(dataclasses.replace(sent, sent_mono_s=501.0), 500.3) == (
        dataclasses.replace(target, sample_age_s=None)
    )
    # An age the human program does not know stays unknown.
    for age in (None, -0.001, float("nan"), float("inf")):
        unknown = dataclasses.replace(sent, sample_age_s=age)
        assert human_target(unknown, 500.3).sample_age_s is None, age


# The human program


def test_a_human_setpoint_is_stamped_as_it_is_sent():
    # Asked for the controller data at 10.0; the ROBOT_STATE it was computed
    # on arrived at 9.98; the graph and the IK took until 10.015.
    sent = as_sent(SQUEEZE, asked_mono=10.0, state_at=9.98, now_mono=10.015)
    assert sent.sample_age_s == pytest.approx(0.03 + 0.015)
    assert sent.sent_mono_s == pytest.approx(500.0 + 0.035)
    assert dataclasses.replace(sent, sample_age_s=0.03, sent_mono_s=None) == SQUEEZE
    # An age the feed does not know stays unknown; the send time is still said.
    unknown = dataclasses.replace(SQUEEZE, sample_age_s=None)
    stamped = as_sent(unknown, 10.0, 9.98, 10.015)
    assert stamped.sample_age_s is None
    assert stamped.sent_mono_s == pytest.approx(500.035)
    # Without a ROBOT_STATE there is no clock to read the send time off.
    bare = dataclasses.replace(SQUEEZE, state_clock_mono_s=None)
    assert as_sent(bare, 10.0, None, 10.015).sent_mono_s is None
    assert as_sent(SQUEEZE, 10.0, None, 10.015).sent_mono_s is None
    # A clock that stepped back makes nothing younger and nothing later.
    back = as_sent(SQUEEZE, asked_mono=10.0, state_at=9.98, now_mono=9.0)
    assert (back.sample_age_s, back.sent_mono_s) == (0.03, 500.0)


@pytest.mark.parametrize(("down_s", "up_s"), [(0.0, 0.0), (0.004, 0.2), (0.15, 0.003)])
def test_the_sample_age_at_the_thor_is_never_less_than_the_true_one(down_s, up_s):
    """down_s and up_s: a link message's way from the Thor, and to it."""
    made = 500.0  # the Thor's clock when it made the ROBOT_STATE
    state_at = 77.0  # the human program's clock when that ROBOT_STATE arrived
    asked, sends = state_at + 0.01, state_at + 0.02
    # The feed says 0.03 s at `asked`; the pass works for 0.01 s; then it sends.
    sent = as_sent(SQUEEZE, asked, state_at, sends)
    true_age_s = 0.03 + (sends - asked) + up_s
    arrived = made + down_s + (sends - state_at) + up_s
    age = human_target(sent, arrived).sample_age_s
    # What is too much is the ROBOT_STATE's way to the human program.
    assert age == pytest.approx(true_age_s + down_s)
    assert age >= true_age_s - 1e-9


@pytest.fixture
def program(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Any, Bench, Callable[..., Any]]]:
    """test_human_program.py's helpers, a bench and a start of the human program.

    The bench is a robot link service on loopback carriers. Skipped without
    Isaac Capture: the program's retargeting graph needs it.
    """
    pytest.importorskip("isaaccapture.retargeters")
    import test_human_program as hp
    from isaaccapture_examples.agentic_remote import console_log

    # No console log level is set in a test.
    monkeypatch.setattr(console_log, "set_console_level", lambda level: None)
    folder = tmp_path_factory.mktemp("fp")
    bench = Bench(folder)
    started: list[Any] = []

    def start(*args: str, **keywords: Any) -> Any:
        config = hp.write_config(folder, bench.path)
        started.append(hp.Running(config, *args, **keywords))
        return started[-1]

    yield hp, bench, start
    for running in started:
        running.end()
    bench.close()


@pytest.mark.link
@pytest.mark.apai
def test_the_human_program_sends_the_age_and_the_time_of_the_moment_it_sends(
    program, monkeypatch
):
    hp, bench, start = program
    work_s = 0.06  # the pass's own work, as the graph and the IK take time
    step = HumanLogic.step

    def slow(self: HumanLogic, *args: Any, **keywords: Any) -> Any:
        time.sleep(work_s)
        return step(self, *args, **keywords)

    monkeypatch.setattr(HumanLogic, "step", slow)
    before = time.monotonic()
    # One ROBOT_STATE, whose clock_mono_s is 500: every setpoint names it.
    thor = hp.robot(bench, owner="agent")
    made = hp.robot_state().clock_mono_s
    running = start("--report-s", "0", feed=hp.script([hp.BASE]))
    got: list[tuple[float, HumanSetpoint]] = []
    end = time.monotonic() + WAIT_S
    while len(got) < 5:
        assert time.monotonic() < end, f"{len(got)} HUMAN_SETPOINTs: {running.said}"
        for record in thor.drain():
            message = decode(*record)
            if isinstance(message, HumanSetpoint):
                got.append((time.monotonic(), message))
        time.sleep(0.002)
    for at, message in got:
        assert message.state_clock_mono_s == made
        # The script's controller data are 0 s old when the feed gives them.
        assert work_s <= message.sample_age_s < work_s + 1.0
        # Read off that ROBOT_STATE: never later than the true moment.
        assert 0.0 <= message.sent_mono_s - made <= at - before
    # It is the time of each send, not the ROBOT_STATE's.
    (first_at, first), (last_at, last) = got[0], got[-1]
    assert last_at - first_at > 3 * work_s
    assert last.sent_mono_s - first.sent_mono_s == pytest.approx(
        last_at - first_at, abs=0.1
    )
    assert running.end() == 0


# The live view


@pytest.mark.parametrize("stretch", ["at the view", "on the way"])
@pytest.mark.parametrize(
    ("total_s", "fresh"),
    [(VIEW_LIMIT_S - NEAR_S, True), (VIEW_LIMIT_S + NEAR_S, False)],
    ids=["just inside the limit", "just outside the limit"],
)
def test_every_stretch_of_time_counts_toward_the_view_age_limit(
    human_app, stretch, total_s, fresh
):
    h = human_app
    now = h.clock.monotonic()
    if stretch == "at the view":
        report = ViewStatus(live=True, age_s=total_s, sent_mono_s=now)
    else:
        on_the_way = total_s - 0.1
        report = ViewStatus(live=True, age_s=0.1, sent_mono_s=now - on_the_way)
    h.client.send(report)
    feed(h, view=False)
    result = ask_takeover(h)
    if fresh:
        assert result.status == "completed", result.reason
        return
    assert result.status == "refused"
    old = rf"{VIEW_NOT_FRESH}0\.50\d s old, the limit is 0\.5 s"
    assert re.fullmatch(old + re.escape(OPEN_THE_VIEW), result.reason), result.reason


def test_a_view_status_delayed_on_its_way_is_as_old_as_it_is(human_app):
    """The review's case: a report that is fresh when it is received."""
    h = human_app
    before = h.app.snapshot()
    report = ViewStatus(live=True, age_s=0.3, sent_mono_s=h.clock.monotonic())
    h.clock.advance(0.3)
    h.client.send(report)
    feed(h, view=False)
    result = ask_takeover(h)
    assert result.status == "refused"
    old = rf"{VIEW_NOT_FRESH}0\.6\d\d s old, the limit is 0\.5 s"
    assert re.fullmatch(old + re.escape(OPEN_THE_VIEW), result.reason), result.reason
    assert state_now(h).view_age_s >= 0.6
    assert h.app.snapshot() == before and not h.app.takeover_open
    assert h.arm.sent == [] and h.arm.torque_on()


def test_a_view_status_that_does_not_say_when_it_was_sent_is_not_fresh(human_app):
    """As a live view program of before the field sends it."""
    h = human_app
    h.client.send(ViewStatus(live=True, age_s=0.0))
    feed(h, view=False)
    untold = (
        f"{VIEW_NOT_FRESH}the VIEW_STATUS does not say when it was sent{OPEN_THE_VIEW}"
    )
    assert human_facts(h)["takeover"] == untold
    result = ask_takeover(h)
    assert (result.status, result.reason) == ("refused", untold)
    assert state_now(h).view_age_s is None


def test_a_live_view_status_says_when_it_is_sent_on_the_thors_clock():
    model = model_with_state()  # its ROBOT_STATE arrived at T0
    made = view_state_at(WALL0).clock_mono_s
    picture_shown(Page(model))
    sent = model.status_to_send(T0)
    assert sent == ViewStatus(live=True, age_s=near(0.1), sent_mono_s=made)
    # 0.3 s later on the same ROBOT_STATE: the Thor's clock went on as well.
    later = model.status_to_send(T0 + 0.3)
    assert later == ViewStatus(
        live=True, age_s=near(0.4), sent_mono_s=pytest.approx(made + 0.3)
    )
    # The age is view_status()'s of the same moment.
    plain = model.view_status(T0 + 0.3, model.thor_wall_now(T0 + 0.3))
    assert (later.live, later.age_s) == (plain.live, plain.age_s)
    # A view that is not live names no time: its age is not asked.
    assert ViewModel().status_to_send(T0) == NOT_LIVE
    stale = model_with_state(state_max_age_s=1.0)
    picture_shown(Page(stale))
    assert stale.status_to_send(T0 + 1.0) == NOT_LIVE


def test_the_view_age_at_the_thor_is_the_pictures_age_whatever_the_link_takes(
    human_app,
):
    h = human_app
    down_s, up_s = 0.07, 0.11  # a link message's way from the Thor, and to it
    made_mono, made_wall = h.clock.monotonic(), h.clock.wall()

    def state(ago_s: float) -> Any:
        """The Thor's ROBOT_STATE of ago_s before now."""
        return dataclasses.replace(
            robot_state(),
            clock_mono_s=made_mono - ago_s,
            clock_wall_s=made_wall - ago_s,
        )

    model = ViewModel()
    model.on_message(state(60.0), T0 - 60.0)
    # At the workstation at T0, down_s after the Thor made it.
    model.on_message(state(0.0), T0)
    # A picture exposed 0.1 s before that ROBOT_STATE was made; the page draws
    # it 0.05 s after the ROBOT_STATE arrived, and the report goes out.
    model.put_picture(1, JPEG, made_wall - 0.1, T0 - 0.02, False)
    assert Page(model).draw(1, T0 + 0.05)
    report = model.status_to_send(T0 + 0.05)
    assert report.live
    # The view program cannot see down_s: it says 0.15 s and means the sum.
    assert report.age_s == near(0.15)
    h.clock.advance(down_s + 0.05 + up_s)
    h.client.send(report)
    h.step()
    h.step()
    arrived = h.app._view[1]
    since_exposure = 0.1 + (arrived - made_mono)
    assert since_exposure >= 0.1 + down_s + 0.05 + up_s - 1e-9
    assert h.app.view_age_s(arrived) == near(since_exposure)


# The link messages


def test_link_messages_of_before_are_read_and_those_of_now_stay_small():
    old = decode(ViewStatus.TYPE, b'{"live":true,"age_s":0.1}')
    assert old == ViewStatus(live=True, age_s=0.1) and old.sent_mono_s is None
    fields = json.loads(SQUEEZE.to_payload())
    assert fields.pop("sent_mono_s") is None
    payload = json.dumps(fields).encode()
    assert decode(HumanSetpoint.TYPE, payload) == SQUEEZE
    # A field the reader does not know is passed over, as sent_mono_s is by
    # a robot app of before.
    newer = decode(ViewStatus.TYPE, b'{"live":true,"age_s":0.1,"later":1}')
    assert newer == old
    # With every digit of a joint target and of the two times.
    long_q = [-(1.0 + joint / 3.0 + 1.0 / 7.0) for joint in range(6)]
    full = dataclasses.replace(
        SQUEEZE,
        q=long_q,
        sample_age_s=1.0 / 30.0,
        state_clock_mono_s=123456.0 + 1.0 / 7.0,
        sent_mono_s=123456.0 + 2.0 / 7.0,
        note="x" * 100,
    )
    assert len(full.to_payload()) < 512
