# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the human drive and the human input in
isaaccapture_examples.agentic_remote.robot_app.human.

A human target is a joint target with its sample age (how old the controller
data behind it are) and the owner and stop numbers it quotes. A human drive is
one stretch of human motion on the Thor: it runs while the person squeezes the
controller's grip. The tests run on the simulated arm and the fake clock; those
marked "apai" run on APAI's human checker and need an APAI checkout
(AGENTIC_REMOTE_APAI_SRC).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_MOVE_ID,
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_STAGES,
)
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.executor import (
    HOLD_FAILED,
    MoveExecutor,
)
from isaaccapture_examples.agentic_remote.robot_app.human import (
    SPEED_MARGIN,
    DriveEnd,
    HumanDrive,
    HumanInput,
    HumanTarget,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    Checker,
    ContactStop,
    FeedbackLost,
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.state import (
    IDLE_AFTER_TARGET_S,
    IdleReader,
    JointHistory,
)
from test_apai_backend import HOME, make_config, with_speed

try:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
except ModuleNotFoundError as missing:
    # Without an APAI checkout conftest.py skips the tests marked "apai"; the
    # others run without it.
    if missing.name != "apai":
        raise
    apai_backend = None

approx = pytest.approx

N = 6
PERIOD = 0.02
READ_S = 0.008  # the simulated arm's time for one joint read
START_READ_S = N * READ_S  # a drive begins with one read of every joint
REST = [0.0] * N
# The human limit set of the default settings.
SPEED = 1.5  # rad/s
MAX_STEP = 0.1  # rad
# The speed clamp: the largest step of one period.
STEP = SPEED_MARGIN * SPEED * PERIOD
# How far the target of a moving hand lies ahead of the last setpoint.
AHEAD = 0.08
# A target within MAX_STEP of the rest pose on every joint, as offsets.
REACH = (0.09, -0.06, 0.02)
GOAL = [*REACH, 0.0, 0.0, 0.0]
FAR = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
RAISED = [0.4, -0.3, 0.5, 0.2, -0.2, 0.1]
LOGGER = "isaaccapture.examples.agentic_remote.human_drive"
# The simulated arm or APAI's mock arm with APAI's human checker.
KINDS = ["sim", pytest.param("apai", marks=pytest.mark.apai)]
# The watchdog of APAI's checker in the test of a long hold. It runs on the
# machine's clock, so the test really waits past it.
APAI_WATCHDOG_S = 0.5
# Per joint how far a long APAI drive swings to either side of HOME, rad.
SWING = (0.5, 0.2, 0.2, 0.4, 0.4, 1.0)
# More setpoints than any test here lets a rig send.
MOST_SENDS = 5000
# A target's stage says how far the human program, the program that makes the
# targets from controller data, is with those data. These are the stages at
# which the data are not fresh; "": no stage is given.
NOT_READY = [*(stage for stage in HUMAN_STAGES if stage != "ready"), ""]


def never_stop() -> str | None:
    return None


def a_target(seq: int = 1, **fields: Any) -> HumanTarget:
    """A human target that arrived at 100 s, the grip not squeezed, not released."""
    values: dict[str, Any] = {
        "seq": seq,
        "owner_number": 1,
        "stop_number": 0,
        "source": HUMAN_SOURCE_SYNTHETIC,
        "stage": "ready",
        "note": "",
        "engaged": False,
        "grip_released": False,
        "q": None,
        "sample_age_s": 0.0,
        "arrived_mono": 100.0,
    }
    values.update(fields)
    return HumanTarget(**values)


class Rig:
    """A human drive on a connected arm whose control owner is the human.

    On the fake clock time passes only in the drive's own reads and sleeps, so
    a test feeds the drive from the arm's send: after each setpoint the arm
    was sent the live view is reported fresh, and the step given to feed()
    puts what the person does next into the human input.
    """

    def __init__(
        self,
        arm: Any,
        checker: Checker,
        config: RobotAppConfig,
        clock: FakeClock,
        patch: pytest.MonkeyPatch,
        start: Sequence[float] = REST,
    ) -> None:
        self.arm = arm
        self.checker = checker
        self.config = config
        self.clock = clock
        self.patch = patch
        self.start = [float(value) for value in start]  # where the arm is
        self.authority = Authority()
        self.authority.set_owner(Owner.HUMAN, "takeover")
        self.admitted = self.authority.snapshot()
        self.history = JointHistory(config.n_joints, config.state.history_s)
        self.executor = MoveExecutor(
            arm, self.authority, self.history, config.executor, clock
        )
        self.input = HumanInput()
        # The live view: (the age it reported, when that arrived); None: not live.
        self.view: tuple[float, float] | None = (0.0, clock.monotonic())
        self.view_fed = True  # whether a fresh view follows each setpoint
        self.clamped: list[int] = []  # one entry per note_clamped call
        self.sends: list[tuple[list[float], float]] = []  # (setpoint, when sent)
        # Called with the number of setpoints sent so far: before a send, and
        # after it the probes. Then the step of feed().
        self.before_send: Callable[[int], None] | None = None
        self.probes: list[Callable[[int], None]] = []
        self.step: Callable[[int], None] | None = None
        self.fed_from = 0  # setpoints sent before feed() was called
        self.seq = 0
        self.drive = self.new_drive()
        real = arm.send_target

        def send_target(q: Sequence[float]) -> None:
            # On the fake clock a drive that does not end would run for ever.
            assert len(self.sends) < MOST_SENDS, "the drive does not end"
            if self.before_send is not None:
                self.before_send(len(self.sends))
            real(q)
            now = clock.monotonic()
            self.sends.append(([float(value) for value in q], now))
            for probe in self.probes:
                probe(len(self.sends))
            if self.view_fed:
                self.view = (0.0, now)
            if self.step is not None:
                self.step(len(self.sends) - self.fed_from)

        patch.setattr(arm, "send_target", send_target)

    def feed(self, step: Callable[[int], None]) -> None:
        """After each setpoint from now on, call step(setpoints sent since now)."""
        self.step = step
        self.fed_from = len(self.sends)

    def new_drive(self, **changes: Any) -> HumanDrive:
        """A drive on this rig; `config` and the keywords of HumanDrive change it."""
        settings: dict[str, Any] = {
            "claim_idle": self.executor.claim_idle,
            "note_clamped": lambda: self.clamped.append(1),
        }
        settings.update(changes)
        return HumanDrive(
            self.arm,
            self.authority,
            self.checker,
            self.history,
            settings.pop("config", self.config),
            self.clock,
            self.input,
            self.view_age,
            **settings,
        )

    def view_age(self, now: float) -> float | None:
        """As RobotApp.view_age_s: the reported age plus the time since."""
        if self.view is None:
            return None
        age, arrived = self.view
        return age + max(0.0, now - arrived)

    def lose_view(self) -> None:
        """The live view is not live from now on."""
        self.view_fed = False
        self.view = None

    def wait(self, seconds: float) -> None:
        """Time passes with the arm holding; the live view goes on reporting."""
        self.clock.advance(seconds)
        if self.view_fed:
            self.view = (0.0, self.clock.monotonic())

    def place(self, q: Sequence[float]) -> None:
        """Put the simulated arm at q, holding there."""
        self.arm.set_pose(q)
        self.start = [float(value) for value in q]

    def near(self, *offsets: float) -> list[float]:
        """The start pose moved by one offset per joint; joints left out stay."""
        moved = list(self.start)
        for joint, offset in enumerate(offsets):
            moved[joint] += offset
        return moved

    def target(self, q: Sequence[float] | None = None, **fields: Any) -> HumanTarget:
        """A target that arrives now and quotes the admitted numbers.

        With q the grip is squeezed (engaged), without it neither squeezed nor
        released; fields change any of that.
        """
        self.seq += 1
        values: dict[str, Any] = {
            "owner_number": self.admitted.owner_number,
            "stop_number": self.admitted.stop_number,
            "engaged": q is not None,
            "q": None if q is None else tuple(q),
            "arrived_mono": self.clock.monotonic(),
        }
        values.update(fields)
        return a_target(self.seq, **values)

    def put(self, target: HumanTarget) -> HumanTarget:
        self.input.put(target)
        return target

    def squeeze(self, q: Sequence[float], **fields: Any) -> HumanTarget:
        """An engaged target arrives now."""
        return self.put(self.target(q, **fields))

    def release(self) -> HumanTarget:
        """The target of a released grip arrives now."""
        return self.put(self.target(None, grip_released=True))

    def grip_until(
        self, q: Sequence[float], sends: int, then: Callable[[], object] | None = None
    ) -> None:
        """Squeeze toward q now and after each setpoint, until `sends` went out.

        After that one then() is called and nothing more arrives.
        """
        self.squeeze(q)

        def step(count: int) -> None:
            if count < sends:
                self.squeeze(q)
            elif count == sends and then is not None:
                then()

        self.feed(step)

    def hold_grip(self, q: Sequence[float], sends: int) -> None:
        """Squeeze toward q for `sends` setpoints, then release the grip."""
        self.grip_until(q, sends, self.release)

    def ahead(self, goal: Sequence[float]) -> list[float]:
        """The pose at most AHEAD of the last setpoint on the way to goal."""
        at = self.sends[-1][0] if self.sends else self.start
        return [a + max(-AHEAD, min(AHEAD, g - a)) for a, g in zip(at, goal)]

    def lead(self, goal: Sequence[float], sends: int | None = None) -> None:
        """Keep the target AHEAD of the arm on the way to goal: a moving hand.

        With `sends` the grip is released once that many setpoints went out.
        """
        self.squeeze(self.ahead(goal))

        def step(count: int) -> None:
            if sends is not None and count >= sends:
                self.release()
            else:
                self.squeeze(self.ahead(goal))

        self.feed(step)

    def stall_in_reads(self, index: int, seconds: float) -> None:
        """Lose `seconds` once, in the joint reads that follow setpoint `index`."""
        real = self.arm.read_joint
        done: list[bool] = []

        def read(joint: int) -> object:
            if not done and len(self.sends) == index + 1:
                done.append(True)
                self.clock.advance(seconds)
            return real(joint)

        self.patch.setattr(self.arm, "read_joint", read)

    def run(
        self,
        should_stop: Callable[[], str | None] = never_stop,
        drive: HumanDrive | None = None,
    ) -> DriveEnd:
        return (drive or self.drive).run(self.admitted, should_stop)

    def sent(self) -> list[list[float]]:
        return [q for q, _ in self.sends]

    def times(self) -> list[float]:
        return [t for _, t in self.sends]

    def gaps(self) -> list[float]:
        times = self.times()
        return [later - earlier for earlier, later in zip(times, times[1:])]

    def steps(self) -> list[list[float]]:
        """Per setpoint how far each joint moved on from the one before it."""
        rows = [self.start, *self.sent()]
        return [[b - a for a, b in zip(q, q_next)] for q, q_next in zip(rows, rows[1:])]


def apai_rig(
    clock: FakeClock,
    patch: pytest.MonkeyPatch,
    *,
    speed: float | None = None,
    watchdog_s: float | None = None,
) -> Rig:
    """A rig on APAI's mock arm at HOME with APAI's human checker.

    speed replaces the human joint speed limit; watchdog_s the watchdog of
    APAI's checkers.
    """
    overrides = None if watchdog_s is None else {"safety": {"watchdog_s": watchdog_s}}
    profile = apai_backend.load_apai_profile("mock", overrides)
    config = make_config()
    if speed is not None:
        config = with_speed(config, "human", speed)
    arm = apai_backend.build_apai_arm(config, profile, clock)
    arm.connect()
    arm.send_target(HOME)  # the mock arm is there at once
    human = apai_backend.build_apai_checkers(config, profile, clock).human
    return Rig(arm, human, config, clock, patch, start=HOME)


@pytest.fixture
def rig(sim_arm, checkers, config, clock, monkeypatch) -> Rig:
    """A rig on the simulated arm at rest; the human checker was never fed."""
    human = config.limits.human
    assert (human.max_joint_speed, human.max_step_rad) == (SPEED, MAX_STEP)
    sim_arm.connect()
    return Rig(sim_arm, checkers.human, config, clock, monkeypatch)


@pytest.fixture
def make_rig(request, clock, monkeypatch) -> Callable[..., Rig]:
    """Builds the rig of one of KINDS; keywords go to apai_rig()."""

    def make(kind: str, **apai: Any) -> Rig:
        if kind == "apai":
            return apai_rig(clock, monkeypatch, **apai)
        return request.getfixturevalue("rig")

    return make


@pytest.fixture
def calls(rig, monkeypatch) -> list[str]:
    """What a drive asks of the human checker and of its watch, in order.

    "feed", "begin" (passed), "begin refused", "end" and the watch's "finish".
    """
    made: list[str] = []
    checker = rig.checker
    real_feed, real_begin = checker.feed_watchdog, checker.begin_motion
    real_end, real_watch = checker.end_motion, checker.contact_watch

    def feed_watchdog(t_mono: float | None = None) -> None:
        made.append("feed")
        real_feed(t_mono)

    def begin_motion(halt_count: int) -> None:
        try:
            real_begin(halt_count)
        except Refusal:
            made.append("begin refused")
            raise
        made.append("begin")

    def end_motion() -> None:
        made.append("end")
        real_end()

    def contact_watch(arm: object, q_start: object, history: object) -> object:
        watch = real_watch(arm, q_start, history)
        real_finish = watch.finish

        def finish() -> None:
            made.append("finish")
            real_finish()

        watch.finish = finish
        return watch

    monkeypatch.setattr(checker, "feed_watchdog", feed_watchdog)
    monkeypatch.setattr(checker, "begin_motion", begin_motion)
    monkeypatch.setattr(checker, "end_motion", end_motion)
    monkeypatch.setattr(checker, "contact_watch", contact_watch)
    return made


def owed(calls: list[str]) -> list[str]:
    """The calls that begin and end a motion and finish its watch."""
    return [call for call in calls if call in ("begin", "end", "finish")]


# The human input


def test_human_input_keeps_only_the_newest_and_forgets_on_clear():
    box = HumanInput()
    assert box.newest() is None
    assert box.sample_age_s(100.0) is None
    assert box.stage(100.0, 4.0) == ""
    first = a_target(1, stage="waiting_for_headset")
    second = a_target(2, stage="ready", sample_age_s=0.03)
    box.put(first)
    box.put(second)
    assert box.newest() is second
    assert box.stage(100.0, 4.0) == "ready"
    assert box.sample_age_s(100.0) == 0.03
    box.clear()
    assert box.newest() is None
    assert box.sample_age_s(100.0) is None
    assert box.stage(100.0, 4.0) == ""


def test_a_lower_seq_replaces_a_higher_one():
    # A human program that was started again counts from the beginning.
    box = HumanInput()
    box.put(a_target(4000, stage="ready"))
    restarted = a_target(3, stage="no_controller")
    box.put(restarted)
    assert box.newest() is restarted
    assert box.stage(100.0, 4.0) == "no_controller"


def test_the_sample_age_is_the_reported_age_plus_the_time_since_arrival():
    box = HumanInput()
    box.put(a_target(sample_age_s=0.04, arrived_mono=100.0))
    assert box.sample_age_s(100.0) == 0.04
    assert box.sample_age_s(100.25) == approx(0.29)


@pytest.mark.parametrize("reported", [None, -0.01, math.nan, math.inf, True, "0.1"])
def test_a_reported_age_that_is_no_time_of_0_or_more_is_unknown(reported):
    box = HumanInput()
    box.put(a_target(sample_age_s=reported))
    assert box.sample_age_s(100.0) is None
    assert box.sample_age_s(105.0) is None


def test_the_stage_is_shown_until_keep_s_after_the_last_arrival():
    box = HumanInput()
    box.put(a_target(stage="waiting_for_headset", arrived_mono=100.0))
    assert box.stage(104.0, 4.0) == "waiting_for_headset"
    assert box.stage(104.01, 4.0) == ""
    box.put(a_target(stage="ready", arrived_mono=104.01))
    assert box.stage(104.01, 4.0) == "ready"


# The start of a drive


@pytest.mark.parametrize("kind", KINDS)
def test_the_first_drive_starts_on_a_checker_that_was_never_fed(make_rig, kind):
    rig = make_rig(kind)
    # Nothing has fed the human checker: a begin is refused as things stand.
    with pytest.raises(Refusal, match=r"perception watchdog"):
        rig.checker.begin_motion(rig.checker.halt_count())
    rig.hold_grip(rig.near(0.05), sends=5)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 5)
    assert rig.sent()[-1] == rig.near(0.05)


@pytest.mark.parametrize("kind", KINDS)
def test_a_drive_after_a_hold_of_ten_seconds_starts(make_rig, kind):
    rig = make_rig(kind, watchdog_s=APAI_WATCHDOG_S)
    rig.hold_grip(rig.near(0.05), sends=5)
    assert rig.run().status == "released"
    rig.wait(10.0)  # the person keeps the grip released
    if kind == "apai":
        time.sleep(APAI_WATCHDOG_S + 0.1)
    # The feeds of the first drive are too old by now.
    with pytest.raises(Refusal, match=r"perception watchdog: last"):
        rig.checker.begin_motion(rig.checker.halt_count())
    rig.hold_grip(rig.start, sends=5)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 5)
    assert len(rig.sends) == 10
    assert rig.sent()[-1] == rig.start


def test_the_watchdog_is_fed_with_the_time_of_the_controller_data_not_with_now(rig):
    seen: list[tuple[str, float | None, float]] = []  # (call, time given, now)
    real_feed, real_begin = rig.checker.feed_watchdog, rig.checker.begin_motion

    def feed_watchdog(t_mono: float | None = None) -> None:
        seen.append(("feed", t_mono, rig.clock.monotonic()))
        real_feed(t_mono)

    def begin_motion(halt_count: int) -> None:
        seen.append(("begin", None, rig.clock.monotonic()))
        real_begin(halt_count)

    rig.patch.setattr(rig.checker, "feed_watchdog", feed_watchdog)
    rig.patch.setattr(rig.checker, "begin_motion", begin_motion)
    targets = [rig.squeeze(GOAL, sample_age_s=0.1)]
    rig.clock.advance(0.03)  # the target waits for the motion worker

    def step(count: int) -> None:
        # Nothing new arrives after the first and the third setpoint.
        if count == 2:
            targets.append(rig.squeeze(GOAL, sample_age_s=0.05))
        elif count == 4:
            rig.release()

    rig.feed(step)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 4)
    first, second = targets
    # Once before the begin, then once per new seq, each with the time the
    # controller data were taken at.
    assert [(call, given) for call, given, _ in seen] == [
        ("feed", first.arrived_mono - 0.1),
        ("begin", None),
        ("feed", second.arrived_mono - 0.05),
    ]
    assert all(given < now for call, given, now in seen if call == "feed")


def test_a_latched_checker_refuses_the_drive_and_sends_nothing(rig, calls):
    rig.checker.latch("motor fault")
    rig.squeeze(GOAL)
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == "fault-stop latched: motor fault"
    assert end.sent == 0
    assert rig.sends == []
    assert calls == ["feed", "begin refused"]
    assert not rig.checker.in_motion
    assert rig.drive.progress() is None


@pytest.mark.parametrize("kind", KINDS)
def test_a_drive_begins_on_a_checker_that_a_stop_halted(make_rig, kind):
    rig = make_rig(kind)
    # Right B is a STOP first: it halts every checker before the takeover.
    rig.authority.raise_stop("right B", "controller")
    rig.checker.halt("right B")
    rig.authority.set_owner(Owner.HUMAN, "takeover")
    rig.admitted = rig.authority.snapshot()
    assert rig.checker.halt_count() == 1
    rig.hold_grip(rig.near(0.05), sends=5)
    end = rig.run()
    # The drive quotes the halt count as it stands, so the begin is not refused
    # for the halt, and the halt refuses none of its setpoints.
    assert (end.status, end.reason) == ("released", "the grip was released")
    assert end.sent == 5
    assert rig.sent()[-1] == rig.near(0.05)


def test_a_stop_before_the_checkers_begin_ends_the_drive_stopped_not_refused(rig):
    real = rig.arm.read_all

    def read_all() -> list:
        readings = real()
        # A fault-stop raises the stop number, then latches the checkers.
        rig.authority.raise_fault("motor 3 over temperature")
        rig.checker.latch("motor 3 over temperature")
        return readings

    rig.patch.setattr(rig.arm, "read_all", read_all)
    rig.squeeze(GOAL)
    end = rig.run()
    assert end.status == "stopped"
    assert end.reason == "stop number is now 1, the drive was admitted at 0"
    assert rig.sends == []


def test_the_drive_starts_from_the_measured_pose_not_from_the_last_target(rig):
    # The arm rests against something at 0.3 rad with its last target beyond.
    rig.place([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    rig.arm.block_joint(0, 0.3)
    rig.arm.send_target([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    rig.sends.clear()
    assert rig.arm.pose()[0] == 0.3
    assert rig.arm.last_target()[0][0] == 0.5
    checked: list[list[float]] = []
    real = rig.checker.check_setpoint

    def check(q_prev: Sequence[float], q_next: Sequence[float], dt: float) -> None:
        checked.append(list(q_prev))
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.checker, "check_setpoint", check)
    # Within the step limit of the measured pose, beyond it from the last target.
    rig.hold_grip(rig.near(0.06), sends=3)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 3)
    assert checked[0] == rig.start
    assert rig.sent()[0] == approx(rig.near(STEP))
    assert rig.sent()[-1] == rig.near(0.06)


def test_a_drive_is_refused_while_the_executor_is_claimed(rig, calls):
    rig.squeeze(GOAL)
    began = rig.clock.monotonic()
    # A guarded torque release holds this claim while it checks and cuts.
    with rig.executor.claim_idle() as idle:
        assert idle is True
        end = rig.run()
        assert rig.drive.progress() is None
    assert end == DriveEnd("refused", "another move is running", 0, None)
    assert rig.sends == []
    assert rig.clock.monotonic() == began  # nothing was read either
    assert calls == []  # the checker was not touched
    rig.hold_grip(GOAL, sends=3)
    assert rig.run().status == "released"


def test_the_executor_starts_no_move_while_a_drive_runs(rig):
    outcomes: list[MoveOutcome] = []
    claims: list[bool] = []

    def meanwhile(count: int) -> None:
        if count != 3:
            return
        move = rig.executor.run(
            [rig.near(0.001)],
            rig.start,
            0.02,
            PERIOD,
            Owner.HUMAN,
            rig.checker,
            rig.admitted,
            "move-1",
        )
        outcomes.append(move)
        # What a guarded torque release finds.
        with rig.executor.claim_idle() as idle:
            claims.append(idle)

    rig.probes.append(meanwhile)
    rig.hold_grip(GOAL, sends=6)
    end = rig.run()
    assert outcomes == [MoveOutcome("refused", "another move is running")]
    assert claims == [False]
    assert (end.status, end.sent) == ("released", 6)
    assert len(rig.sends) == 6  # the executor sent nothing
    # The drive gave its claim up.
    with rig.executor.claim_idle() as idle:
        assert idle is True


def _no_target(rig: Rig) -> None:
    pass


def _grip_not_squeezed(rig: Rig) -> None:
    rig.release()


def _engagement_ended(rig: Rig) -> None:
    rig.put(rig.target(None, note="no IK solution"))


def _not_engaged_with_values(rig: Rig) -> None:
    rig.put(rig.target(rig.near(*REACH), engaged=False))


def _stage_not_ready(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH), stage="no_controller")


def _old_target(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH), sample_age_s=1.0)


def _unknown_age(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH), sample_age_s=None)


def _no_view(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH))
    rig.lose_view()


def _old_view(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH))
    rig.view = (0.6, rig.clock.monotonic())


def _stop_first(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH))
    rig.authority.raise_stop("right B", "controller")


def _other_numbers(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH), owner_number=rig.admitted.owner_number + 1)


def _short_target(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH)[:5])


def _start_read_fails(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH))
    rig.arm.fail_reads(2, 1)


def _latched(rig: Rig) -> None:
    rig.squeeze(rig.near(*REACH))
    rig.checker.latch("motor fault")


# What keeps a drive from its start read: label -> (arrange, status, reason).
NO_START_READ = {
    "no target": (_no_target, "stale", "controller data stale (no human target)"),
    "grip not squeezed": (_grip_not_squeezed, "released", "the grip was released"),
    "engagement ended": (_engagement_ended, "disengaged", "no IK solution"),
    "not engaged, with joint values": (
        _not_engaged_with_values,
        "disengaged",
        "the human program ended the engagement",
    ),
    "stage not ready": (
        _stage_not_ready,
        "stale",
        "controller data stale (stage no_controller)",
    ),
    "old target": (
        _old_target,
        "stale",
        "controller data stale (1.000 s old, the limit is 0.25 s)",
    ),
    "unknown age": (
        _unknown_age,
        "stale",
        "controller data stale (the sample age is unknown)",
    ),
    "no view": (
        _no_view,
        "view",
        "the live view is not fresh (the viewer reports no live picture)",
    ),
    "old view": (
        _old_view,
        "view",
        "the live view is not fresh (0.600 s old, the limit is 0.5 s)",
    ),
    "stop first": (
        _stop_first,
        "stopped",
        "stop number is now 1, the drive was admitted at 0",
    ),
    "other numbers": (
        _other_numbers,
        "stopped",
        "the human target quotes owner number 2 and stop number 0, the drive "
        "was admitted at 1 and 0",
    ),
    "short target": (
        _short_target,
        "refused",
        "the target has 5 values, the arm 6 joints",
    ),
}


@pytest.mark.parametrize(
    ("arrange", "status", "reason"), NO_START_READ.values(), ids=NO_START_READ.keys()
)
def test_nothing_is_fed_read_or_sent_unless_the_first_target_passes(
    rig, calls, arrange, status, reason
):
    arrange(rig)
    began = rig.clock.monotonic()
    end = rig.run()
    assert end == DriveEnd(status, reason, 0, None)
    assert rig.sends == []
    assert rig.clock.monotonic() == began  # a joint read would have cost time
    assert calls == []  # the watchdog was not fed
    assert rig.arm.torque_on()


def test_a_failed_start_read_refuses_the_drive_with_nothing_sent(rig, calls):
    rig.squeeze(GOAL)
    rig.arm.fail_reads(2, 1)
    end = rig.run()
    assert end.status == "refused"
    assert end.reason.startswith("the start pose could not be read: joint index 2")
    assert (end.sent, end.end_q, rig.sends) == (0, None, [])
    assert calls == ["feed"]


def test_a_start_read_without_a_valid_value_for_a_joint_refuses_the_drive(rig):
    real = rig.arm.read_all

    def read_all() -> list:
        readings = real()
        readings[4] = dataclasses.replace(readings[4], valid=False)
        return readings

    rig.patch.setattr(rig.arm, "read_all", read_all)
    rig.squeeze(GOAL)
    end = rig.run()
    assert (end.status, end.reason) == (
        "refused",
        "the start pose read left a joint out",
    )
    assert rig.sends == []
    assert not rig.checker.in_motion


def test_with_the_motors_off_the_drive_is_refused_with_nothing_sent_or_read(rig):
    unpowered = SimArm(n_joints=N, clock=rig.clock, q0=REST)  # never connected
    drive = HumanDrive(
        unpowered,
        rig.authority,
        rig.checker,
        None,
        rig.config,
        rig.clock,
        rig.input,
        rig.view_age,
        claim_idle=rig.executor.claim_idle,
    )
    rig.squeeze(GOAL)
    began = rig.clock.monotonic()
    end = rig.run(drive=drive)
    assert end == DriveEnd("refused", "the motors are off", 0, None)
    assert unpowered.sent == []
    assert rig.clock.monotonic() == began


def test_the_end_pose_is_the_start_reads_while_nothing_was_sent_after_it(rig):
    rig.place(RAISED)
    # Beyond the step limit: the drive ends after its start read.
    rig.squeeze(rig.near(0.3))
    end = rig.run()
    assert (end.status, end.sent) == ("refused", 0)
    assert end.end_q == RAISED
    # Once a setpoint went out the drive reads no joint for the end pose.
    rig.hold_grip(rig.near(0.05), sends=3)
    end = rig.run()
    assert (end.status, end.sent, end.end_q) == ("released", 3, None)


# Following the target


def test_a_drive_follows_the_target_one_speed_limited_step_per_period(rig):
    began = rig.clock.monotonic()
    rig.hold_grip(GOAL, sends=6)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 6)
    rows = [
        [STEP, -STEP, 0.02, 0.0, 0.0, 0.0],
        [2 * STEP, -2 * STEP, 0.02, 0.0, 0.0, 0.0],
        [3 * STEP, -0.06, 0.02, 0.0, 0.0, 0.0],
    ]
    for q, row in zip(rig.sent(), rows):
        assert q == approx(row, abs=1e-12)
    # At the target the setpoint is the target itself, period after period.
    assert rig.sent()[3:] == [GOAL] * 3
    first = began + START_READ_S
    on_time = [first + i * PERIOD for i in range(6)]
    assert rig.times() == approx(on_time, abs=1e-9)
    fastest = max(abs(moved) for step in rig.steps() for moved in step) / PERIOD
    assert fastest == approx(SPEED_MARGIN * SPEED)
    assert fastest < SPEED
    # The joint history holds the drive's reads.
    assert rig.history.age(rig.clock.monotonic()) <= START_READ_S + 6 * PERIOD


def test_each_setpoint_is_checked_against_the_one_before_it_over_one_period(rig):
    checked = []
    real = rig.checker.check_setpoint

    def check(q_prev: Sequence[float], q_next: Sequence[float], dt: float) -> None:
        # Checked just before it is sent: the setpoints before it have gone out.
        checked.append((list(q_prev), list(q_next), dt, len(rig.sends)))
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.checker, "check_setpoint", check)
    rig.hold_grip(GOAL, sends=5)
    rig.run()
    steps = zip([REST, *rig.sent()], rig.sent())
    assert checked == [(a, b, PERIOD, i) for i, (a, b) in enumerate(steps)]


def test_a_clamped_step_is_counted(rig):
    rig.hold_grip(GOAL, sends=6)
    rig.run()
    # Joint index 0 needs four steps and joint index 1 three: one count per
    # setpoint the clamp cut, not per joint.
    assert len(rig.clamped) == 3
    assert [step[0] for step in rig.steps()[:3]] == approx([STEP] * 3)
    assert STEP == approx(0.98 * 1.5 * 0.02)
    # The clamp follows the configured speed, and needs nobody to count it.
    slow = with_speed(rig.config, "human", 0.5)
    drive = rig.new_drive(config=slow, note_clamped=None)
    rig.start = rig.sent()[-1]
    rig.hold_grip(rig.near(-0.05), sends=3)
    assert rig.run(drive=drive).status == "released"
    after = [q[0] for q in rig.sent()[6:]]
    moved = [later - earlier for earlier, later in zip(after, after[1:])]
    assert moved == approx([-SPEED_MARGIN * 0.5 * PERIOD] * 2)
    assert len(rig.clamped) == 3


@pytest.mark.apai
@pytest.mark.parametrize("speed", [0.1, 0.8])
def test_a_clamped_step_passes_apais_human_checker(clock, monkeypatch, speed):
    rig = apai_rig(clock, monkeypatch, speed=speed)
    steps = 2000
    heading = [1.0] * N

    def ahead() -> list[float]:
        """AHEAD of the last setpoint on every joint, turning at SWING."""
        at = rig.sends[-1][0] if rig.sends else rig.start
        for joint, value in enumerate(at):
            if abs(value + heading[joint] * AHEAD - HOME[joint]) > SWING[joint]:
                heading[joint] = -heading[joint]
        return [value + toward * AHEAD for value, toward in zip(at, heading)]

    def step(count: int) -> None:
        if count < steps:
            rig.squeeze(ahead())
        else:
            rig.release()

    rig.squeeze(ahead())
    rig.feed(step)
    end = rig.run()
    # No step was refused: APAI's checker has no tolerance on the speed.
    assert (end.status, end.reason) == ("released", "the grip was released")
    assert end.sent == steps
    assert len(rig.clamped) == steps
    most = SPEED_MARGIN * speed * PERIOD
    for moved in rig.steps():
        assert [abs(by) for by in moved] == approx([most] * N, rel=1e-6)


def test_a_target_beyond_the_step_limit_ends_the_drive_and_sends_nothing(rig, calls):
    rig.squeeze(rig.near(0.0, 0.0, 0.15))
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == (
        "the target is 0.150 rad from the arm on joint index 2, the limit is 0.1 rad"
    )
    # Refused, not clipped: no setpoint goes toward such a target.
    assert (end.sent, rig.sends) == (0, [])
    assert owed(calls) == ["begin", "end", "finish"]


def test_a_target_that_moves_beyond_the_step_limit_ends_a_running_drive(rig):
    far = rig.near(0.09 + 0.2, -0.06, 0.02)
    rig.grip_until(GOAL, 3, lambda: rig.squeeze(far))
    end = rig.run()
    assert end.status == "refused"
    # From the last setpoint sent, which lies three steps from the start.
    assert end.reason == (
        f"the target is {0.29 - 3 * STEP:.3f} rad from the arm on joint index 0, "
        "the limit is 0.1 rad"
    )
    assert end.sent == 3
    assert rig.arm.last_target()[0] == rig.sent()[2]
    assert rig.arm.holds == []


def test_a_target_at_the_step_limit_is_followed(rig):
    rig.hold_grip(rig.near(MAX_STEP), sends=5)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 5)
    assert rig.sent()[-1] == rig.near(MAX_STEP)


@pytest.mark.parametrize("count", [5, 7, 0])
def test_a_target_with_the_wrong_number_of_joints_ends_the_drive(rig, count):
    wrong = [0.01] * count
    rig.grip_until(GOAL, 2, lambda: rig.squeeze(wrong))
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == f"the target has {count} values, the arm 6 joints"
    assert end.sent == 2


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, "0.1", True])
def test_a_target_with_a_value_that_is_no_finite_number_ends_the_drive(rig, value):
    rig.grip_until(GOAL, 2, lambda: rig.squeeze([0.01, 0.0, value, 0.0, 0.0, 0.0]))
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == "the target holds a value that is not a finite number"
    assert end.sent == 2


# What ends a drive


def test_a_released_grip_ends_the_drive_in_a_hold(rig):
    rig.hold_grip(GOAL, sends=5)
    end = rig.run()
    assert end == DriveEnd("released", "the grip was released", 5, None)
    # The last setpoint stands: the arm holds there with nothing more sent.
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()
    # The jaws were left alone and the person still owns the arm.
    assert (rig.arm.jaw_commands, rig.arm.light_holds) == ([], [])
    assert rig.authority.snapshot() == rig.admitted
    rig.clock.advance(1.0)
    assert rig.arm.pose() == approx(GOAL, abs=1e-6)


@pytest.mark.parametrize(
    ("note", "reason"),
    [
        ("no IK solution", "no IK solution"),
        ("", "the human program ended the engagement"),
    ],
)
def test_an_engagement_ended_by_the_human_program_is_not_a_released_grip(
    rig, note, reason
):
    # The grip is still squeezed: the workstation's own checks ended it.
    rig.grip_until(GOAL, 5, lambda: rig.put(rig.target(None, note=note)))
    end = rig.run()
    assert end == DriveEnd("disengaged", reason, 5, None)
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.torque_on()


def test_an_engaged_target_without_joint_values_ends_the_drive_disengaged(rig):
    rig.grip_until(GOAL, 2, lambda: rig.put(rig.target(None, engaged=True)))
    end = rig.run()
    assert (end.status, end.sent) == ("disengaged", 2)


@pytest.mark.parametrize(
    ("grip_released", "status"), [(False, "disengaged"), (True, "released")]
)
def test_a_target_that_is_not_engaged_ends_the_drive_whatever_it_carries(
    rig, grip_released, status
):
    # Fresh joint values within reach, in a target that says "not engaged".
    last = {"engaged": False, "grip_released": grip_released}
    rig.grip_until(GOAL, 3, lambda: rig.put(rig.target(GOAL, **last)))
    end = rig.run()
    assert (end.status, end.sent, len(rig.sends)) == (status, 3, 3)


def test_setpoints_that_stop_arriving_end_the_drive_within_the_age_limit(rig):
    rig.grip_until(GOAL, 5)  # the last target arrives after the fourth setpoint
    end = rig.run()
    limit = rig.config.human.sample_age_limit_s
    assert end.status == "stale"
    assert end.reason == "controller data stale (0.260 s old, the limit is 0.25 s)"
    last_arrival = rig.input.newest().arrived_mono
    assert last_arrival == rig.times()[3]
    since = rig.times()[-1] - last_arrival
    assert since <= limit + PERIOD
    # The drive went on for as long as the target was fresh.
    assert since > limit - PERIOD
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.torque_on()


@pytest.mark.parametrize(("reported", "sends"), [(0.0, 11), (0.1, 6), (0.2, 1)])
def test_a_reported_sample_age_counts_toward_the_limit(rig, reported, sends):
    # One target and none after it. The start read takes 0.048 s, and the
    # setpoints go out a period apart while reported + waited is within 0.25 s.
    rig.squeeze(GOAL, sample_age_s=reported)
    end = rig.run()
    assert (end.status, end.sent) == ("stale", sends)


@pytest.mark.parametrize("reported", [-0.01, -5.0, None, math.nan, math.inf])
def test_a_negative_sample_age_counts_as_unknown(rig, calls, reported):
    rig.squeeze(GOAL, sample_age_s=reported)
    end = rig.run()
    assert end.status == "stale"
    assert end.reason == "controller data stale (the sample age is unknown)"
    assert (end.sent, rig.sends) == (0, [])
    assert calls == []  # such a target feeds no watchdog
    # Nor in a running drive.
    rig.grip_until(GOAL, 3, lambda: rig.squeeze(GOAL, sample_age_s=reported))
    end = rig.run()
    assert (end.status, end.sent) == ("stale", 3)


@pytest.mark.parametrize(
    "stage", NOT_READY, ids=[stage or "no stage" for stage in NOT_READY]
)
def test_a_running_drive_ends_stale_when_the_stage_is_not_ready(rig, stage):
    rig.place(RAISED)
    goal = rig.near(*REACH)

    def step(count: int) -> None:
        # From the third setpoint on only the stage differs: the grip stays
        # squeezed and every target arrives fresh, with its joint values.
        if count < 3:
            rig.squeeze(goal)
        elif count < 8:
            rig.squeeze(goal, stage=stage)
        else:
            rig.release()

    rig.squeeze(goal)
    rig.feed(step)
    end = rig.run()
    reason = f"controller data stale (stage {stage or 'unknown'})"
    assert end == DriveEnd("stale", reason, 3, None)
    # Nothing went out after the pass that saw the stage.
    assert len(rig.sends) == 3
    seen = rig.input.newest()
    assert (seen.stage, seen.engaged, seen.q) == (stage, True, tuple(goal))
    # That target was one period old, far below the limit of 0.25 s.
    assert rig.input.sample_age_s(rig.clock.monotonic()) == approx(PERIOD)
    # The last setpoint stands: the arm holds away from rest with the motors on.
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()
    assert rig.arm.released_at == []
    assert rig.authority.snapshot() == rig.admitted
    rig.clock.advance(1.0)
    assert rig.arm.pose() == approx(rig.sent()[-1], abs=1e-6)
    assert max(abs(at) for at in rig.arm.pose()) > 0.3


@pytest.mark.parametrize(
    "misshapen", [GOAL[:5], [math.nan, *GOAL[1:]]], ids=["five values", "a NaN"]
)
def test_a_stage_that_is_not_ready_is_named_before_the_targets_shape(rig, misshapen):
    # Such a target is no fresh controller data, whatever its joint values are.
    rig.grip_until(GOAL, 3, lambda: rig.squeeze(misshapen, stage="stale"))
    end = rig.run()
    assert end == DriveEnd("stale", "controller data stale (stage stale)", 3, None)
    assert len(rig.sends) == 3
    assert rig.arm.torque_on()


def test_a_view_that_goes_stale_ends_the_drive(rig):
    def step(count: int) -> None:
        rig.squeeze(GOAL)
        if count == 5:
            # The last report of the live view: 5 ms old on arrival.
            rig.view_fed = False
            rig.view = (0.005, rig.clock.monotonic())

    rig.squeeze(GOAL)
    rig.feed(step)
    end = rig.run()
    limit = rig.config.human.view_age_limit_s
    assert end.status == "view"
    assert end.reason == "the live view is not fresh (0.505 s old, the limit is 0.5 s)"
    since = rig.times()[-1] - rig.times()[4]
    assert since <= limit
    # The drive went on for as long as the view was fresh.
    assert since > limit - 2 * PERIOD
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.torque_on()


def test_a_view_that_is_no_longer_live_ends_the_drive(rig):
    rig.grip_until(GOAL, 5, rig.lose_view)
    end = rig.run()
    assert end == DriveEnd(
        "view",
        "the live view is not fresh (the viewer reports no live picture)",
        5,
        None,
    )


def test_no_view_is_needed_when_the_setting_says_so(rig):
    human = dataclasses.replace(rig.config.human, require_live_view=False)
    unwatched = rig.new_drive(config=dataclasses.replace(rig.config, human=human))
    rig.lose_view()
    rig.hold_grip(GOAL, sends=5)
    end = rig.run(drive=unwatched)
    assert (end.status, end.sent) == ("released", 5)
    # With the setting as shipped the same drive does not start.
    rig.hold_grip(rig.sent()[-1], sends=5)
    end = rig.run()
    assert (end.status, end.sent) == ("view", 0)


def test_a_stop_ends_the_drive_before_the_next_setpoint(rig):
    rig.grip_until(GOAL, 5, lambda: rig.authority.raise_stop("right B", "controller"))
    end = rig.run()
    assert end.status == "stopped"
    assert end.reason == "stop number is now 1, the drive was admitted at 0"
    assert (end.sent, len(rig.sends)) == (5, 5)
    # The last setpoint stands: a hold with the motors on.
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()


def test_after_a_stop_the_checker_is_not_asked_for_another_setpoint(rig):
    asked: list[int] = []  # per check, how many setpoints had gone out
    real = rig.checker.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        asked.append(len(rig.sends))
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.checker, "check_setpoint", check)
    rig.grip_until(GOAL, 5, lambda: rig.authority.raise_stop("right B", "controller"))
    end = rig.run()
    assert (end.status, end.sent) == ("stopped", 5)
    # The stop conditions come first in a pass: no sixth setpoint was checked.
    assert asked == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("said", "reason"),
    [("STOP from console: stop", "STOP from console: stop"), ("", "stop requested")],
)
def test_the_jobs_stop_request_ends_the_drive_with_its_reason(rig, said, reason):
    def should_stop() -> str | None:
        return said if len(rig.sends) == 5 else None

    rig.hold_grip(GOAL, sends=100)
    end = rig.run(should_stop)
    assert end == DriveEnd("stopped", reason, 5, None)


def test_a_stop_during_the_checkers_call_ends_the_drive_stopped_not_refused(rig):
    real = rig.checker.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        if len(rig.sends) == 4:
            # A STOP halts the checkers before it reaches the job.
            rig.authority.raise_stop("right B", "controller")
            rig.checker.halt("STOP from controller: right B")
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.checker, "check_setpoint", check)
    rig.hold_grip(GOAL, sends=100)
    end = rig.run()
    assert end.status == "stopped"
    assert end.reason == "stop number is now 1, the drive was admitted at 0"
    assert (end.sent, len(rig.sends)) == (4, 4)


def _stop(rig: Rig) -> None:
    rig.authority.raise_stop("right B", "controller")


def _slow_check(rig: Rig) -> None:
    rig.clock.advance(0.2)


def _very_slow_check(rig: Rig) -> None:
    rig.clock.advance(0.3)


def _stage_gone(rig: Rig) -> None:
    rig.squeeze(GOAL, stage="stale")


# What can happen while the checker checks a setpoint: label -> (it, status).
DURING_THE_CHECK = {
    "a stop": (_stop, "stopped"),
    "a released grip": (Rig.release, "released"),
    "a view that is no longer live": (Rig.lose_view, "view"),
    "a check that makes the setpoint late": (_slow_check, "late"),
    "a check that outlasts the sample age": (_very_slow_check, "stale"),
    "a stage that is no longer ready": (_stage_gone, "stale"),
}


@pytest.mark.parametrize(
    ("happen", "status"), DURING_THE_CHECK.values(), ids=DURING_THE_CHECK.keys()
)
def test_what_changes_while_the_checker_checks_keeps_that_setpoint_back(
    rig, happen, status
):
    real = rig.checker.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        real(q_prev, q_next, dt)
        if len(rig.sends) == 4:
            happen(rig)

    rig.patch.setattr(rig.checker, "check_setpoint", check)
    rig.hold_grip(GOAL, sends=100)
    end = rig.run()
    # The checker accepted the fifth setpoint, and it did not go out.
    assert (end.status, end.sent, len(rig.sends)) == (status, 4, 4)


def test_an_owner_change_ends_the_drive(rig):
    rig.grip_until(
        GOAL, 5, lambda: rig.authority.set_owner(Owner.RECOVERY, "return to rest")
    )
    end = rig.run()
    assert end.status == "stopped"
    assert end.reason == "owner number is now 2, the drive was admitted at 1"
    assert (end.sent, len(rig.sends)) == (5, 5)


def test_a_drive_admitted_for_another_owner_sends_nothing(rig):
    rig.authority.set_owner(Owner.AGENT, "new task")
    rig.admitted = rig.authority.snapshot()
    rig.squeeze(GOAL)
    end = rig.run()
    assert end == DriveEnd("stopped", "control owner is agent, not human", 0, None)
    assert rig.sends == []


@pytest.mark.parametrize("number", ["owner_number", "stop_number"])
def test_a_setpoint_quoting_old_numbers_ends_the_drive(rig, number):
    old = {number: getattr(rig.admitted, number) - 1}
    rig.grip_until(GOAL, 4, lambda: rig.squeeze(GOAL, **old))
    end = rig.run()
    assert end.status == "stopped"
    assert end.reason.startswith("the human target quotes owner number ")
    assert (end.sent, len(rig.sends)) == (4, 4)


def test_a_checker_refusal_ends_the_drive_in_a_hold(rig):
    # The human joint box ends at 3 rad.
    rig.place([2.95, 0.0, 0.0, 0.0, 0.0, 0.0])
    rig.lead(rig.near(0.25))
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == (
        f"setpoint: joint index 0 at {2.95 + 2 * STEP:.4f} rad is outside the "
        "human joint limits -3.0000..3.0000"
    )
    # The one setpoint inside the box went out and stands.
    assert rig.sent() == [approx(rig.near(STEP))]
    assert end.sent == 1
    assert rig.arm.last_target()[0] == rig.sent()[0]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()
    assert rig.authority.snapshot() == rig.admitted


def test_the_table_is_the_checkers_alone_and_a_drive_asks_no_tip_of_it(
    rig, monkeypatch
):
    # A drive keeps no height of its own for the gripper tip: what its
    # checker's check_setpoint() takes is sent, whatever that checker would
    # say of the tip, and every pass sends a setpoint.
    asked: list[str] = []
    for name in ("tip_state", "tip_blocked", "tip_floor"):
        monkeypatch.setattr(
            rig.checker, name, lambda *args, what=name: asked.append(what)
        )
    assert "last_grasp_region" not in inspect.signature(HumanDrive).parameters
    rig.lead(FAR, sends=12)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 12)
    assert rig.sent() == [
        approx([STEP * n, 0.0, 0.0, 0.0, 0.0, 0.0]) for n in range(1, 13)
    ]
    # One setpoint per period from the first on: no pass sends nothing.
    times = [when for _, when in rig.sends]
    assert [b - a for a, b in zip(times, times[1:])] == [approx(PERIOD)] * 11
    assert asked == []


def test_a_contact_holds_at_the_measured_pose_and_is_not_a_fault(rig):
    rig.arm.block_joint(0, 0.05)
    rig.lead(FAR)
    end = rig.run()
    assert end.status == "contact"
    assert end.reason.startswith("contact: joint index 0 trails its setpoints by ")
    assert HOLD_FAILED not in end.reason
    assert 0 < end.sent == len(rig.sends)
    assert len(rig.arm.holds) == 1
    held, _ = rig.arm.holds[0]
    assert held[0] == approx(0.05, abs=1e-6)
    assert rig.arm.last_target()[0] == held
    assert rig.arm.torque_on()
    # Not latched: the person keeps the arm and nothing was stopped.
    assert rig.authority.snapshot() == rig.admitted
    assert not rig.checker.latched()


def test_a_hold_that_cannot_read_after_a_contact_leaves_the_last_setpoint_standing(
    rig, caplog
):
    rig.arm.block_joint(0, 0.05)
    tries: list[int] = []

    def hold() -> None:
        tries.append(1)
        raise ReadFailed("joint index 2: the read failed")

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    rig.lead(FAR)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        end = rig.run()
    assert end.status == "contact"
    assert end.reason.startswith("contact: joint index 0")
    assert end.reason.endswith(
        f"{HOLD_FAILED}joint index 2: the read failed; the last setpoint stands"
    )
    assert len(tries) == 2
    # Nothing more was sent.
    assert end.sent == len(rig.sends)
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [
        "no hold at the measured pose after a contact: joint index 2: the read failed"
    ]


def test_a_hold_that_fails_once_after_a_contact_is_tried_again(rig):
    rig.arm.block_joint(0, 0.05)
    real = rig.arm.hold_measured
    tries: list[int] = []

    def hold() -> None:
        tries.append(1)
        if len(tries) == 1:
            raise ReadFailed("joint index 2: the read failed")
        real()

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    rig.lead(FAR)
    end = rig.run()
    assert end.status == "contact"
    assert HOLD_FAILED not in end.reason
    assert (len(tries), len(rig.arm.holds)) == (2, 1)


def test_lost_joint_feedback_ends_the_drive_as_a_fault(rig):
    def lose(count: int) -> None:
        if count == 1:
            rig.arm.fail_reads(1, 10_000)

    rig.probes.append(lose)
    rig.hold_grip(GOAL, sends=10_000)
    end = rig.run()
    assert end.status == "failed"
    assert end.reason.startswith("fault: joint feedback lost: joint index 1")
    assert 0 < end.sent == len(rig.sends)
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()


def test_a_read_failure_that_escapes_the_watch_ends_the_drive_refused(rig):
    class Watch:
        finished = False

        def after_send(self, index: int, q: object, deadline_mono: float) -> list:
            if index == 3:
                raise ReadFailed("joint index 4: no answer")
            return []

        def finish(self) -> None:
            self.finished = True

    watch = Watch()
    rig.patch.setattr(rig.checker, "contact_watch", lambda arm, q, history: watch)
    rig.hold_grip(GOAL, sends=100)
    end = rig.run()
    assert end.status == "refused"
    assert end.reason == "a joint read failed: joint index 4: no answer"
    assert (end.sent, len(rig.sends)) == (4, 4)
    assert watch.finished
    assert not rig.checker.in_motion


def raise_fault() -> None:
    raise ArmFault("motor 3 over temperature")


def test_an_arm_fault_while_sending_ends_the_drive_failed(rig, calls):
    def before_send(index: int) -> None:
        if index == 4:
            raise_fault()

    rig.before_send = before_send
    rig.hold_grip(GOAL, sends=100)
    end = rig.run()
    assert end == DriveEnd("failed", "fault: motor 3 over temperature", 4, None)
    assert owed(calls) == ["begin", "end", "finish"]
    assert rig.arm.torque_on()


def test_an_arm_fault_in_the_start_read_ends_the_drive_failed(rig):
    rig.patch.setattr(rig.arm, "read_all", raise_fault)
    rig.squeeze(GOAL)
    end = rig.run()
    assert end == DriveEnd("failed", "fault: motor 3 over temperature", 0, None)
    assert rig.sends == []


def test_an_arm_fault_in_the_hold_after_a_contact_ends_the_drive_failed(rig):
    rig.arm.block_joint(0, 0.05)
    rig.patch.setattr(rig.arm, "hold_measured", raise_fault)
    rig.lead(FAR)
    end = rig.run()
    assert (end.status, end.reason) == ("failed", "fault: motor 3 over temperature")
    assert rig.arm.torque_on()


@pytest.mark.parametrize(
    "error",
    [
        FeedbackLost("joint feedback lost: joint index 1"),
        Refusal("the watch refuses the hold"),
        ContactStop("contact: joint index 3"),
    ],
    ids=["feedback lost", "refusal", "contact"],
)
def test_a_hold_after_a_contact_that_raises_no_failed_read_ends_the_drive_failed(
    rig, calls, error
):
    # A watch's hold is documented to raise ReadFailed only. Whatever else of
    # the watch's errors it raises does not leave run().
    rig.arm.block_joint(0, 0.05)
    tries: list[int] = []

    def hold() -> None:
        tries.append(1)
        raise error

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    rig.lead(FAR)
    end = rig.run()
    assert end.status == "failed"
    # A fault-stop follows, and its text names the contact and the hold.
    assert end.reason.startswith("fault: contact: joint index 0 trails its setpoints")
    assert end.reason.endswith(f"{HOLD_FAILED}{error.reason}")
    assert len(tries) == 1  # only a failed read is tried again
    # Nothing more was sent, and the checker and the watch got what they are owed.
    assert 0 < end.sent == len(rig.sends)
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()
    assert owed(calls) == ["begin", "end", "finish"]
    assert rig.drive.progress() is None
    with rig.executor.claim_idle() as idle:
        assert idle is True


def test_a_late_pass_ends_the_drive_and_is_not_made_up(rig):
    rig.stall_in_reads(5, 0.2)  # the pass overruns its deadline by 188 ms
    rig.lead(FAR, sends=12)
    end = rig.run()
    assert end.status == "late"
    assert end.reason == "setpoint 6 is 0.188 s late, over the limit of 0.1 s"
    # Nothing went out after the delay: no larger step, and no step at all.
    assert (end.sent, len(rig.sends)) == (6, 6)
    assert [step[0] for step in rig.steps()] == approx([STEP] * 6)
    assert rig.arm.last_target()[0] == rig.sent()[-1]
    assert rig.arm.holds == []
    assert rig.arm.torque_on()


def test_a_pass_that_is_late_within_the_limit_is_followed_by_the_same_step(rig):
    rig.stall_in_reads(5, 0.05)  # the pass overruns its deadline by 38 ms
    rig.lead(FAR, sends=12)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 12)
    # The setpoint after the delay lies one clamped step on, as every other,
    # and those after it keep a period's distance.
    assert [step[0] for step in rig.steps()] == approx([STEP] * 12)
    gaps = rig.gaps()
    assert gaps[5] == approx(PERIOD + 0.038, abs=1e-9)
    assert gaps[:5] + gaps[6:] == approx([PERIOD] * 10, abs=1e-9)


def test_a_wake_up_late_within_the_small_delay_is_not_made_up(rig):
    late = 0.003
    assert late < rig.config.executor.small_delay_s
    # The joint reads after setpoint 5 overrun the pass's deadline by 3 ms.
    rig.stall_in_reads(5, PERIOD - 2 * READ_S + late)
    rig.lead(FAR, sends=12)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 12)
    assert [step[0] for step in rig.steps()] == approx([STEP] * 12)
    gaps = rig.gaps()
    assert gaps[5] == approx(PERIOD + late, abs=1e-9)
    # The setpoints after it move back with it: none follows the one before
    # it sooner than a period, which would be a step above the speed clamp.
    assert gaps[:5] + gaps[6:] == approx([PERIOD] * 10, abs=1e-9)
    assert min(gaps) >= PERIOD - 1e-9


def test_a_setpoint_that_goes_out_late_takes_the_later_ones_with_it(rig):
    def before_send(index: int) -> None:
        if index == 5:
            rig.clock.advance(0.012)

    rig.before_send = before_send
    rig.lead(FAR, sends=12)
    end = rig.run()
    assert (end.status, end.sent) == ("released", 12)
    first = rig.times()[0]
    shifted = [first + i * PERIOD + (0.012 if i >= 5 else 0.0) for i in range(12)]
    assert rig.times() == approx(shifted, abs=1e-9)
    assert [step[0] for step in rig.steps()] == approx([STEP] * 12)


# What every end owes


def _released(rig: Rig) -> None:
    rig.hold_grip(rig.near(*REACH), sends=3)


def _disengaged(rig: Rig) -> None:
    rig.grip_until(rig.near(*REACH), 3, lambda: _engagement_ended(rig))


def _stopped(rig: Rig) -> None:
    rig.grip_until(rig.near(*REACH), 3, lambda: _stop(rig))


def _stale(rig: Rig) -> None:
    rig.grip_until(rig.near(*REACH), 3)


def _view(rig: Rig) -> None:
    rig.grip_until(rig.near(*REACH), 3, rig.lose_view)


def _refused(rig: Rig) -> None:
    far = rig.near(0.09, -0.06, 0.3)
    rig.grip_until(rig.near(*REACH), 3, lambda: rig.squeeze(far))


def _late(rig: Rig) -> None:
    rig.grip_until(rig.near(*REACH), 3, lambda: rig.clock.advance(0.2))


def _contact(rig: Rig) -> None:
    rig.arm.block_joint(0, rig.start[0] + 0.05)
    rig.lead(rig.near(0.6))


def _failed(rig: Rig) -> None:
    def lose(count: int) -> None:
        if count == 1:
            rig.arm.fail_reads(1, 10_000)

    rig.probes.append(lose)
    rig.hold_grip(rig.near(*REACH), sends=10_000)


# One way to each status of a drive that has begun: status -> arrange.
ENDINGS = {
    "released": _released,
    "disengaged": _disengaged,
    "stopped": _stopped,
    "stale": _stale,
    "view": _view,
    "refused": _refused,
    "late": _late,
    "contact": _contact,
    "failed": _failed,
}
# Ways to an end before the checker's begin passed: label -> (arrange, status).
NOT_BEGUN = {
    label: NO_START_READ[label][:2]
    for label in ("no target", "grip not squeezed", "old target", "no view")
} | {
    "stop first": (_stop_first, "stopped"),
    "start read fails": (_start_read_fails, "refused"),
    "begin refused": (_latched, "refused"),
}
EVERY_END = [
    *(
        pytest.param(arrange, status, True, id=status)
        for status, arrange in ENDINGS.items()
    ),
    *(
        pytest.param(arrange, status, False, id=f"not begun: {label}")
        for label, (arrange, status) in NOT_BEGUN.items()
    ),
]


def squeeze_again(rig: Rig) -> DriveEnd:
    """The next drive, once what ended the last one is put right."""
    rig.arm.block_joint(0, None)
    for joint in (1, 2):
        rig.arm.fail_reads(joint, 0)
    rig.checker.reset_latch()
    rig.probes.clear()
    rig.view_fed, rig.view = True, (0.0, rig.clock.monotonic())
    if rig.authority.owner is not Owner.HUMAN:
        rig.authority.set_owner(Owner.HUMAN, "takeover")
    rig.admitted = rig.authority.snapshot()
    rig.start = rig.arm.pose()
    rig.hold_grip(rig.near(0.01), sends=3)
    return rig.run()


@pytest.mark.parametrize(("arrange", "status", "begun"), EVERY_END)
def test_every_end_ends_the_checkers_motion_and_finishes_the_watch(
    rig, calls, arrange, status, begun
):
    arrange(rig)
    end = rig.run()
    assert end.status == status
    assert (end.sent > 0) == begun
    # After the checker's begin: its end and the watch's finish, once each
    # and in that order. Before it: neither.
    assert owed(calls) == (["begin", "end", "finish"] if begun else [])
    assert rig.drive.progress() is None
    assert not rig.checker.in_motion
    # The next drive begins.
    del calls[:]
    again = squeeze_again(rig)
    assert (again.status, again.sent) == ("released", 3)
    assert owed(calls) == ["begin", "end", "finish"]


def test_the_watch_is_finished_and_the_claim_given_up_when_the_checkers_end_raises(
    rig, calls
):
    def end_motion() -> None:
        calls.append("end")
        raise RuntimeError("checker bug")

    rig.patch.setattr(rig.checker, "end_motion", end_motion)
    rig.hold_grip(GOAL, sends=3)
    # An error that is no refusal, read failure or arm fault is the app's to turn
    # into a fault-stop.
    with pytest.raises(RuntimeError, match=r"checker bug"):
        rig.run()
    assert owed(calls) == ["begin", "end", "finish"]
    assert rig.drive.progress() is None
    with rig.executor.claim_idle() as idle:
        assert idle is True


def test_an_error_of_another_kind_leaves_the_drive_after_what_every_end_owes(
    rig, calls
):
    def before_send(index: int) -> None:
        if index == 2:
            raise RuntimeError("driver bug")

    rig.before_send = before_send
    rig.hold_grip(GOAL, sends=100)
    with pytest.raises(RuntimeError, match=r"driver bug"):
        rig.run()
    assert owed(calls) == ["begin", "end", "finish"]
    assert rig.drive.progress() is None
    assert not rig.checker.in_motion
    with rig.executor.claim_idle() as idle:
        assert idle is True
    assert rig.arm.torque_on()


@pytest.mark.parametrize("status", ENDINGS)
def test_no_ending_of_a_drive_cuts_torque(rig, status):
    rig.place(RAISED)
    ENDINGS[status](rig)
    end = rig.run()
    assert end.status == status
    assert end.sent > 0
    assert rig.arm.torque_on()
    assert rig.arm.released_at == []
    # The arm holds away from rest: a target stands, the last setpoint or,
    # after a contact, the measured pose.
    stands = rig.arm.last_target()[0]
    assert stands == (rig.arm.holds[0][0] if status == "contact" else rig.sent()[-1])
    assert max(abs(at) for at in rig.arm.pose()) > 0.3
    # The drive changed no owner and raised no stop.
    if status != "stopped":
        assert rig.authority.snapshot() == rig.admitted
    assert not rig.checker.latched()


# progress() and the idle reads


def test_progress_names_the_human_drive_and_counts_its_setpoints(rig):
    in_read, at_send = [], []
    real = rig.arm.read_all

    def read_all() -> list:
        in_read.append(rig.drive.progress())
        return real()

    rig.patch.setattr(rig.arm, "read_all", read_all)
    rig.probes.append(lambda count: at_send.append(rig.drive.progress()))
    assert rig.drive.progress() is None
    rig.hold_grip(GOAL, sends=4)
    rig.run()
    # A drive counts as running from its start read on.
    assert in_read == [(HUMAN_MOVE_ID, 0, 0)]
    # The count goes up once send_target has returned.
    assert at_send == [(HUMAN_MOVE_ID, i, 0) for i in range(4)]
    assert rig.drive.progress() is None


class Pauses:
    """A reads_paused that records when its blocks are open."""

    def __init__(self) -> None:
        self.open = 0
        self.entered = 0

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        self.open += 1
        self.entered += 1
        try:
            yield
        finally:
            self.open -= 1


def test_the_idle_reads_are_paused_from_the_start_read_to_the_last_setpoint(rig):
    pauses = Pauses()
    drive = rig.new_drive(reads_paused=pauses)
    in_read: list[int] = []
    in_send: list[int] = []
    real = rig.arm.read_all

    def read_all() -> list:
        in_read.append(pauses.open)
        return real()

    rig.patch.setattr(rig.arm, "read_all", read_all)
    rig.probes.append(lambda count: in_send.append(pauses.open))
    rig.hold_grip(GOAL, sends=4)
    assert rig.run(drive=drive).status == "released"
    assert in_read == [1]
    assert in_send == [1] * 4
    assert (pauses.open, pauses.entered) == (0, 1)
    # A drive that ends before its start read pauses nothing.
    rig.release()
    assert rig.run(drive=drive).status == "released"
    assert (pauses.open, pauses.entered) == (0, 1)


@pytest.mark.parametrize("failed_reads", [0, 1], ids=["first try", "second try"])
def test_the_hold_after_a_contact_runs_with_the_idle_reads_paused(rig, failed_reads):
    pauses = Pauses()
    drive = rig.new_drive(reads_paused=pauses)
    rig.arm.block_joint(0, 0.05)
    in_hold: list[int] = []
    real = rig.arm.hold_measured

    def hold() -> None:
        in_hold.append(pauses.open)
        if len(in_hold) <= failed_reads:
            raise ReadFailed("joint index 2: the read failed")
        real()

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    rig.lead(FAR)
    end = rig.run(drive=drive)
    assert (end.status, len(rig.arm.holds)) == ("contact", 1)
    # Every try of the hold reads the joints inside the drive's one pause.
    assert in_hold == [1] * (failed_reads + 1)
    assert (pauses.open, pauses.entered) == (0, 1)


def test_no_idle_read_begins_while_the_hold_after_a_contact_is_tried_again(rig):
    # The robot app's idle reader: it reads one joint when a read is due.
    idle = IdleReader(rig.arm, rig.history, rig.config.state, rig.clock)
    drive = rig.new_drive(reads_paused=idle.paused)
    rig.arm.block_joint(0, 0.05)
    real = rig.arm.hold_measured
    tries: list[int] = []
    in_retry: list[tuple[float, float | None, object]] = []

    def hold() -> None:
        tries.append(1)
        if len(tries) == 1:
            # A read of every joint that takes long and then fails.
            rig.clock.advance(IDLE_AFTER_TARGET_S)
            raise ReadFailed("joint index 2: the read failed")
        now = rig.clock.monotonic()
        since_setpoint = now - rig.arm.last_target()[1]
        in_retry.append((since_setpoint, idle.due_in(now), idle.step(now)))
        real()

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    rig.lead(FAR)
    end = rig.run(drive=drive)
    assert (end.status, len(rig.arm.holds)) == ("contact", 1)
    [(since_setpoint, due_in, read)] = in_retry
    # The last setpoint is old enough for an idle read, and none is due or made.
    assert since_setpoint >= IDLE_AFTER_TARGET_S
    assert (due_in, read) == (None, None)
    # After the drive the idle reads go on.
    rig.clock.advance(IDLE_AFTER_TARGET_S)
    assert idle.step(rig.clock.monotonic()) is not None


@pytest.mark.parametrize(
    ("asked", "pauses_entered"),
    [("feed_watchdog", 0), ("begin_motion", 1), ("contact_watch", 1)],
)
def test_a_contact_before_the_drive_has_a_watch_ends_it_contact_with_no_hold(
    rig, asked, pauses_entered
):
    # The checker reports a contact where the drive has no watch yet: before
    # the start read, with nothing paused, or after it. A ContactStop is a
    # Refusal, and the drive still ends "contact", not "refused".
    pauses = Pauses()
    drive = rig.new_drive(reads_paused=pauses)

    def contact(*args: object) -> None:
        raise ContactStop("contact: joint index 3")

    rig.patch.setattr(rig.checker, asked, contact)
    rig.squeeze(GOAL)
    end = rig.run(drive=drive)
    assert (end.status, end.reason) == ("contact", "contact: joint index 3")
    # Nothing was sent, so there is no hold: the target before the drive stands.
    assert (rig.sends, rig.arm.holds) == ([], [])
    assert rig.arm.torque_on()
    assert not rig.checker.in_motion
    assert (pauses.open, pauses.entered) == (0, pauses_entered)
