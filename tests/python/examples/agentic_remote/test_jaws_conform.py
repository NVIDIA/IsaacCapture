# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The jaws over the robot link do what APAI's own driver does.

The functional code is APAI's skill code on its reBot RS driver (RebotRSArm
with its gripper guard, RSGripperGuard), on the machine at the arm. The split
is the same skill code on the workstation's arm adapter (RobotLinkArm), whose
jaw commands cross the robot link to the robot app, which runs them on the
same driver class.

Each test runs one piece of APAI's skill code twice: on the driver itself, and
through the split. Both drivers stand on a stand-in motor layer that records
every command the jaw motor gets. The two recordings must be equal: the calls
that reach the driver with their arguments, the jaw motor's commands, the
readings the skill code gets back and what it is raised.

They need an APAI checkout (AGENTIC_REMOTE_APAI_SRC).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.robot_app.app import LOST_FEEDBACK_S
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    GripperOutcome,
    StopLevel,
)
from link_bench import until
from repo_paths import repo_root
from test_apai_backend import GuardTime, MotorSdk, make_config, rs_arm
from test_arm_adapter import SimRig
from test_rest_end_to_end import Thor

from conftest import AppHarness

try:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
except ModuleNotFoundError as missing:
    # Without an APAI checkout conftest.py skips every test here: all are
    # marked "apai". They must still be collected.
    if missing.name != "apai":
        raise
    apai_backend = None

pytestmark = pytest.mark.apai

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
# The jaw routines of APAI's driver, whose calls are recorded.
ROUTINES = (
    "set_gripper",
    "open_gripper",
    "close_gripper_two_stage",
    "close_gripper_torque",
)
# Where the jaws meet what they close on, rad of the 6.2 rad jaw travel: the
# wallet of the rig's recorded run (held at 0.13 of the travel), and an object
# about 5 cm wide.
WALLET_RAD = 0.81
BOX_RAD = 3.4
# An object that the jaws meet past the scout stage of the rigid grip profile.
THIN_RAD = 1.5
# Measured on the rig: a full open ends at 0.963 of the jaw travel.
RIG_FULL_OPEN = 0.963


def runtime() -> Any:
    """APAI's skill runtime class: the functional skill code."""
    from apai.skills.runtime import SkillRuntime

    return SkillRuntime


def drivers_open_effort() -> float:
    """The effort of APAI's own opening, read from its driver's open_gripper."""
    from apai.control.rebot_rs_arm import RebotRSArm

    seen: dict[str, float] = {}
    probe = types.SimpleNamespace(
        _grip_open=1.0,
        set_gripper=lambda pos, effort: seen.update(effort=effort),
    )
    RebotRSArm.open_gripper(probe)
    return seen["effort"]


class Skills:
    """As much of APAI's SkillRuntime as its jaw routines touch, around one arm.

    Any other attribute is SkillRuntime's own method, run on this object: the
    skill code under test is APAI's, not a copy.
    """

    def __init__(self, arm: Any, profile: Any) -> None:
        jaws = profile.arm_cfg.gripper
        self.arm = types.SimpleNamespace(
            raw=arm,
            harness=types.SimpleNamespace(),
            set_gripper=arm.set_gripper,
            get_state=arm.get_state,
        )
        self.cfg = types.SimpleNamespace(
            arm=profile.arm_cfg, grasp=profile.apai_cfg.grasp
        )
        self._grip_open = float(jaws.open_pos)
        self._grip_closed = float(jaws.closed_pos)
        self._gripper_closes_commanded = 0
        self._carry_attachment = None

    def __getattr__(self, name: str) -> Any:
        value = getattr(runtime(), name)
        return functools.partial(value, self) if callable(value) else value


class Side:
    """One driver on its stand-in motor layer, and the arm skill code calls.

    `arm` is the driver itself on the functional side and the arm adapter on
    the split's. calls are the driver's jaw routines as they were called,
    with every argument by name.
    """

    def __init__(self, arm: Any, driver: Any, sdk: MotorSdk, profile: Any) -> None:
        self.arm, self.driver, self.sdk = arm, driver, sdk
        self.calls: list[tuple[str, dict[str, Any]]] = []
        for name in ROUTINES:
            self._record(name)
        self.skills = Skills(arm, profile)  # it calls the recorded routines

    def _record(self, name: str) -> None:
        routine = getattr(self.driver, name)
        signature = inspect.signature(routine)

        def recorded(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.calls.append((name, dict(bound.arguments)))
            return routine(*args, **kwargs)

        setattr(self.driver, name, recorded)

    def motor(self) -> list[tuple[float, float]]:
        """(position in rad, stiffness) of every command the jaw motor got."""
        return [(q[0], kp[0]) for q, kp in self.sdk.gripper.sent]


def ran(side: Side, script: Callable[[Side], list[Any]]) -> dict[str, Any]:
    """What one run of a piece of skill code left behind on this side."""
    raised = None
    readings: list[Any] = []
    try:
        readings = script(side)
    except Exception as error:  # the functional code's own exception, or its twin
        raised = (type(error).__name__, str(error))
    return {
        "calls": side.calls,
        "motor": side.motor(),
        "readings": readings,
        "raised": raised,
    }


def same(a: Any, b: Any) -> bool:
    """Equal, with numbers compared to nine decimal places."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == pytest.approx(b, abs=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def assert_same(functional: dict[str, Any], split: dict[str, Any]) -> None:
    for what in ("raised", "calls", "readings", "motor"):
        assert same(functional[what], split[what]), (
            f"{what} differ:\n functional {functional[what]}\n split      {split[what]}"
        )


@pytest.fixture(scope="module")
def rs_profile() -> Any:
    return apai_backend.load_apai_profile("rebot_rs")


@pytest.fixture
def rig_config(rs_profile: Any) -> RobotAppConfig:
    return apai_backend.config_for_profile(make_config(), rs_profile)


@pytest.fixture
def driver_time(monkeypatch: pytest.MonkeyPatch) -> GuardTime:
    """The one clock both drivers and their gripper guards run on."""
    import apai.control.rebot_rs_arm as rebot_rs_arm

    shared = GuardTime()
    monkeypatch.setattr(rebot_rs_arm, "time", shared)
    return shared


def stand_in(
    config: RobotAppConfig,
    profile: Any,
    clock: Any,
    monkeypatch: pytest.MonkeyPatch,
    shared: GuardTime,
) -> tuple[Any, MotorSdk]:
    """An arm port on APAI's RebotRSArm over a stand-in motor layer; not connected."""
    import apai.control.rs_gripper as rs_gripper

    port, sdk = rs_arm(config, profile, clock, monkeypatch)
    monkeypatch.setattr(rs_gripper, "time", shared)
    connect = port.backend.connect

    def connected() -> None:
        connect()
        port.backend._last_q_t = shared.monotonic()

    monkeypatch.setattr(port.backend, "connect", connected)
    return port, sdk


@pytest.fixture
def functional(
    rig_config: RobotAppConfig,
    rs_profile: Any,
    monkeypatch: pytest.MonkeyPatch,
    driver_time: GuardTime,
) -> Side:
    """APAI's driver, as the skill code has it on the machine at the arm."""
    port, sdk = stand_in(
        rig_config, rs_profile, SystemClock(), monkeypatch, driver_time
    )
    port.backend.connect()
    return Side(port.backend, port.backend, sdk, rs_profile)


class Split(Side):
    """The split: the arm adapter, the robot link and the robot app on the driver."""

    def __init__(self, rig: SimRig, sdk: MotorSdk, profile: Any) -> None:
        self.rig, self.h = rig, rig.h
        super().__init__(rig.adapter, rig.h.arm.backend, sdk, profile)

    def state(self) -> Any:
        return self.rig.link.state()


def started_split(
    config: RobotAppConfig,
    profile: Any,
    monkeypatch: pytest.MonkeyPatch,
    shared: GuardTime,
) -> Iterator[Split]:
    """The robot app on APAI's driver, at rest, with the agent as control owner."""
    clock = SystemClock()
    port, sdk = stand_in(config, profile, clock, monkeypatch, shared)
    checkers = apai_backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, port, checkers, clock, connected=False, step_period_s=None)
    h.start()
    rig = SimRig(h, expected_profile_id=config.profile_id)
    try:
        rig.own()
        yield Split(rig, sdk, profile)
    finally:
        rig.close()
        h.close()


@pytest.fixture
def split(
    rig_config: RobotAppConfig,
    rs_profile: Any,
    monkeypatch: pytest.MonkeyPatch,
    driver_time: GuardTime,
) -> Iterator[Split]:
    """The robot app on APAI's driver, at rest, with the agent as control owner."""
    yield from started_split(rig_config, rs_profile, monkeypatch, driver_time)


def width(side: Side) -> float | None:
    """The jaw opening as APAI's skill code reads it: None when unknown."""
    return side.skills._gripper_width_frac()


def open_wide(side: Side) -> None:
    """The opening before a grasp and after a miss: APAI's own opening effort."""
    skills = side.skills
    skills._set_gripper(skills._grip_open, effort=drivers_open_effort())


def grasp_close(side: Side, material: str) -> None:
    """The close of APAI's grasp: its two-stage close with the grip profile."""
    from apai.grasping.force import select_profile

    side.skills._grasp_close(select_profile("", material), None)


def release(side: Side) -> None:
    """The release of APAI's place skill."""
    from apai.skills import release_episode

    release_episode.open_hand(side.skills, None)


def pick_and_place(material: str, object_rad: float) -> Callable[[Side], list[Any]]:
    def script(side: Side) -> list[Any]:
        open_wide(side)
        readings = [width(side)]
        side.sdk.gripper.stop = object_rad  # the jaws close on the object
        grasp_close(side, material)
        # After the lift: "held", "closed on nothing" and "grip verified" are
        # decided from this reading.
        readings += [width(side), side.arm.close_stages_skipped]
        side.sdk.gripper.stop = 0.0
        release(side)
        readings.append(width(side))  # the measured release
        return readings

    return script


def test_the_efforts_of_this_file_are_the_functional_codes() -> None:
    # The opening before a grasp and after a miss is called with a literal in
    # APAI's skill code: it is the driver's own opening effort.
    source = inspect.getsource(runtime())
    effort = drivers_open_effort()
    assert f"self._set_gripper(open_pos, effort={effort}" in source


@pytest.mark.parametrize(
    ("material", "object_rad"), [("soft", WALLET_RAD), ("rigid", BOX_RAD)]
)
def test_a_normal_pick_and_place_is_the_functional_one(
    functional: Side, split: Split, material: str, object_rad: float
) -> None:
    script = pick_and_place(material, object_rad)
    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)

    assert at_the_arm["raised"] is None
    names = [name for name, _ in over_the_link["calls"]]
    # The driver's own two-stage close ran, with both of its stages.
    assert names.count("close_gripper_two_stage") == 1
    opened, held, skipped, released = over_the_link["readings"]
    assert skipped == 0
    travel = functional.skills._grip_open - functional.skills._grip_closed
    assert held == pytest.approx(object_rad / travel)
    assert released > held and opened == released
    # The grip is left at the close's stiffness, never at the light hold's.
    hold_kp = functional.driver._grip_hold_kp
    closes = [kp for position, kp in over_the_link["motor"] if position < object_rad]
    assert closes and hold_kp not in [kp for _, kp in over_the_link["motor"]]
    # The robot app knows that the jaws may hold something, and does not relax
    # them when the agent's task ends.
    mark = len(split.sdk.gripper.sent)
    result = split.rig.link.request_owner("nobody", reason="task done")
    assert result.status == "completed"
    assert split.sdk.gripper.sent[mark:] == []


def test_the_grip_keeps_its_stiffness_through_the_end_of_the_task(
    split: Split,
) -> None:
    open_wide(split)
    split.sdk.gripper.stop = WALLET_RAD
    grasp_close(split, "soft")
    held = split.sdk.gripper.sent[-1]
    assert held[1] != [split.driver._grip_hold_kp]
    mark = len(split.sdk.gripper.sent)
    # The agent gives the arm back, and a STOP follows: nothing reaches the jaws.
    assert split.rig.link.request_owner("nobody", reason="done").status == "completed"
    split.rig.link.stop("operator", "console")
    assert split.h.app._worker.wait_idle(WAIT_S)
    state = split.rig.link.state()
    assert split.sdk.gripper.sent[mark:] == []
    assert state.gripper.may_be_holding is True


def test_a_close_on_nothing_is_the_functional_one(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        open_wide(side)
        grasp_close(side, "rigid")  # nothing between the jaws
        readings = [width(side), side.arm.close_stages_skipped]
        open_wide(side)  # APAI's answer to a grasp of air
        readings.append(width(side))
        return readings

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    from apai.grasping.force import select_profile

    closed, skipped, reopened = over_the_link["readings"]
    # The jaws read at the seat stage of the grip profile: APAI's air grasp.
    assert closed == pytest.approx(1.0 - select_profile("", "rigid").close_frac_stage2)
    assert (skipped, reopened) == (0, pytest.approx(1.0))


def test_a_second_close_on_seated_jaws_skips_what_the_driver_skips(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        open_wide(side)
        side.sdk.gripper.stop = THIN_RAD
        grasp_close(side, "rigid")
        first = side.arm.close_stages_skipped
        grasp_close(side, "rigid")  # the scout stage would open the jaws
        return [first, side.arm.close_stages_skipped, width(side)]

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    assert over_the_link["readings"][:2] == [0, 1]


def test_an_opening_from_closed_is_the_functional_one(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        readings = []
        side.arm.open_gripper()  # the driver's own opening
        readings.append(width(side))
        side.arm.set_gripper(side.skills._grip_closed, effort=0.8)
        side.arm.set_gripper(side.skills._grip_open, effort=0.6)
        readings.append(width(side))
        return readings

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    assert over_the_link["readings"] == [pytest.approx(1.0)] * 2
    # One ramp per opening: more than ten commands, none of them a step back.
    opening = [q for q, _ in over_the_link["motor"]]
    assert len(opening) > 20


def test_an_opening_that_ends_short_as_on_the_rig_is_taken_as_open(
    functional: Side, split: Split, rig_config: RobotAppConfig
) -> None:
    def script(side: Side) -> list[Any]:
        travel = side.skills._grip_open - side.skills._grip_closed
        side.sdk.gripper.blocked_at = RIG_FULL_OPEN * travel
        open_wide(side)
        readings = [width(side)]
        side.skills._wait_gripper_opened(6.0)  # APAI's wait after an opening
        return readings

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    assert over_the_link["raised"] is None
    assert over_the_link["readings"] == [pytest.approx(RIG_FULL_OPEN)]
    # The robot app takes these jaws as open and empty: a rest is not refused
    # for them.
    state = split.state()
    assert (state.gripper.load, state.gripper.may_be_holding) == ("empty", False)
    assert split.h.app.jaws_not_proven() is None


def test_a_blocked_opening_is_the_functional_one(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        side.sdk.gripper.blocked_at = 1.0  # something keeps the jaws from opening
        open_wide(side)
        return []

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    kind, text = over_the_link["raised"]
    assert kind == "GripperFault"
    assert text.startswith("gripper blocked while opening at 1.00 rad (target 6.20)")
    # As at the arm: the jaws hold where they stopped, no fault-stop is
    # latched, the agent keeps the arm and its next jaw command runs.
    state = split.state()
    assert (state.stop_level, state.owner) == ("none", "agent")
    assert split.rig.adapter._latched is False  # the adapter's stop latch
    split.sdk.gripper.blocked_at = None
    open_wide(split)
    assert width(split) == pytest.approx(1.0)


def test_jaws_that_cannot_be_read_refuse_an_opening_with_the_drivers_text(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        side.arm.open_gripper()
        side.arm.set_gripper(side.skills._grip_closed, effort=0.8)
        # The jaw motor answers no read from here on.
        side.sdk.gripper._mm["jaw"].missed = 10**6
        side.arm.set_gripper(side.skills._grip_open, effort=0.8)
        return []

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    for what in ("calls", "motor"):
        assert same(at_the_arm[what], over_the_link[what]), what
    assert at_the_arm["raised"][0] == over_the_link["raised"][0] == "GripperFault"
    assert "no gripper command sent" in over_the_link["raised"][1]
    assert split.state().stop_level == "none"


def test_the_close_before_a_torque_release_is_the_functional_one(
    functional: Side, split: Split
) -> None:
    def script(side: Side) -> list[Any]:
        side.arm.open_gripper()
        closed = side.arm.close_gripper_torque()  # as APAI's rest skill calls it
        return [closed, width(side)]

    at_the_arm, over_the_link = ran(functional, script), ran(split, script)
    assert_same(at_the_arm, over_the_link)
    assert over_the_link["readings"][0] is True
    # The gripper worker's close before a release is the same routine, at the
    # driver's own effort.
    (call,) = [args for name, args in functional.calls if name == ROUTINES[3]]
    ended = threading.Event()
    outcomes: list[GripperOutcome] = []
    split.rig.link.request_owner("nobody", reason="done")
    split.calls.clear()
    split.h.app._worker.close_at_rest(lambda o: (outcomes.append(o), ended.set()))
    assert ended.wait(WAIT_S)
    assert same(split.calls[0], (ROUTINES[3], call))


def test_the_typed_close_is_the_close_of_apais_close_gripper_skill(
    functional: Side, split: Split
) -> None:
    from apai.grasping.force import select_profile

    def skill(side: Side) -> list[Any]:
        open_wide(side)
        # What APAI's skill_close_gripper runs with a valid jaw reading.
        side.skills._close_two_stage(
            select_profile("", "rigid"),
            _never_past=side.arm.get_state().gripper_pos,
            _count_skips=True,
        )
        return [width(side)]

    at_the_arm = ran(functional, skill)
    open_wide(split)
    assert split.rig.link.request_owner("nobody", reason="done").status == "completed"
    typed = split.h.thor_terminal()
    typed.ask("close", answer="close: completed")
    over_the_link = {**ran(split, lambda side: []), "readings": [None]}
    for what in ("calls", "motor"):
        assert same(at_the_arm[what], over_the_link[what]), what


def test_the_typed_open_is_the_drivers_own_opening(
    functional: Side, split: Split
) -> None:
    functional.arm.open_gripper()
    assert split.rig.link.request_owner("nobody", reason="done").status == "completed"
    typed = split.h.thor_terminal()
    typed.ask("open", "yes", answer="open: completed")
    assert same(functional.calls, split.calls)
    assert same(functional.motor(), split.motor())
    assert [name for name, _ in split.calls] == ["open_gripper", "set_gripper"]


def test_the_jaw_reading_in_robot_state_is_taken_anew_while_the_arm_holds(
    split: Split,
) -> None:
    open_wide(split)
    assert width(split) == pytest.approx(1.0)
    # Something pushes the jaws: no command, yet the agent reads it within a
    # few ROBOT_STATEs, as APAI's driver reads the jaw motor at each call.
    split.sdk.gripper._mm["jaw"].value = 5.0
    travel = split.skills._grip_open - split.skills._grip_closed
    deadline = threading.Event()
    for _ in range(100):
        if width(split) == pytest.approx(5.0 / travel):
            break
        deadline.wait(0.02)
    assert width(split) == pytest.approx(5.0 / travel)
    assert split.arm.get_state().gripper_valid is True


# That reading on a motor bus that misses answers, in real time

ROBOT_RIG = repo_root() / "examples/agentic_remote/configs/robot_app_rebot.yaml"
LIBRARY_TIMEOUT_S = 1.0  # the motor library's, for a read that passes none
BUS_READ_S = 0.008  # one answered read on the rig's bus


class Bus:
    """The stand-in motors on a bus where an answer takes time, or does not come.

    A read without an answer returns after the timeout it passed, after the
    motor library's own when it passed none, and raises as the library does.
    """

    def __init__(self, sdk: MotorSdk, read_s: float = BUS_READ_S) -> None:
        self.read_s = read_s  # one answered read
        self.silent_until = 0.0  # no read that begins before this is answered
        self.jaw_misses = 0  # the next reads of the jaw motor without an answer
        self.joint_misses = 0  # the next reads of joint motors without an answer
        self.jaw_wrong: list[float] = []  # its next answers, in place of where it is
        # Of every read of the jaw motor: (begin, end, timeout in ms, answered).
        self.jaw_reads: list[tuple[float, float, int | None, bool]] = []
        self.joint_reads: list[float] = []  # when each read of a joint motor began
        self._lock = threading.Lock()
        for group in (sdk.arm, sdk.gripper):
            for motor in group._mm.values():
                motor.robstride_get_param_f32 = self._reader(
                    motor, group is sdk.gripper
                )

    def _reader(self, motor: Any, jaw: bool) -> Callable[..., float]:
        def read(param: int, timeout_ms: int | None = None) -> float:
            begin = time.monotonic()
            value = motor.value
            with self._lock:
                answered = begin >= self.silent_until
                if jaw and answered and self.jaw_misses > 0:
                    self.jaw_misses -= 1
                    answered = False
                elif jaw and answered and self.jaw_wrong:
                    value = self.jaw_wrong.pop(0)
                elif not jaw and answered and self.joint_misses > 0:
                    self.joint_misses -= 1
                    answered = False
                if not jaw:
                    self.joint_reads.append(begin)
            if answered:
                time.sleep(self.read_s)
            else:
                time.sleep(
                    LIBRARY_TIMEOUT_S if timeout_ms is None else timeout_ms / 1000.0
                )
            if jaw:
                self.jaw_reads.append((begin, time.monotonic(), timeout_ms, answered))
            if not answered:
                raise OSError("no answer")
            return value

        return read

    def unanswered(self, mark: int) -> list[tuple[float, float, int | None, bool]]:
        """The jaw reads from number `mark` on that got no answer."""
        return [read for read in list(self.jaw_reads[mark:]) if not read[3]]

    def answered(self, mark: int) -> list[tuple[float, float, int | None, bool]]:
        """The jaw reads from number `mark` on that got an answer."""
        return [read for read in list(self.jaw_reads[mark:]) if read[3]]

    def joint_reads_a_second(self, begin: float, end: float) -> float:
        """How many joint reads a second began between two times."""
        return sum(begin <= t < end for t in list(self.joint_reads)) / (end - begin)


@pytest.fixture
def rig_split(
    rs_profile: Any, monkeypatch: pytest.MonkeyPatch, driver_time: GuardTime
) -> Iterator[Split]:
    """split, with the rig file's ROBOT_STATE period and idle read period."""
    rig = load_config(ROBOT_RIG).state
    base = make_config()
    state = dataclasses.replace(
        base.state,
        publish_period_s=rig.publish_period_s,
        idle_read_period_s=rig.idle_read_period_s,
    )
    config = apai_backend.config_for_profile(
        dataclasses.replace(base, state=state), rs_profile
    )
    yield from started_split(config, rs_profile, monkeypatch, driver_time)


def holding_with_jaw_readings(split: Split, read_s: float = BUS_READ_S) -> Bus:
    """The bus under the split, once ROBOT_STATE's jaw readings come over it."""
    bus = Bus(split.sdk, read_s)
    until(lambda: len(bus.jaw_reads) >= 3, "the jaw readings of a hold")
    until(lambda: split.state().gripper.valid is True, "a jaw reading")
    return bus


def test_a_jaw_read_without_an_answer_in_a_hold_waits_as_an_idle_joint_read_does(
    rig_split: Split,
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split)
    authority = split.h.app._authority
    stops, mark = authority.stop_number, len(bus.jaw_reads)
    bus.jaw_misses = 1
    until(lambda: bus.jaw_misses == 0, "the jaw read without an answer")
    # That read is under way on the thread that serves a STOP.
    sent = time.monotonic()
    split.rig.link.stop("operator pressed stop", "console")
    until(lambda: authority.stop_number != stops, "the STOP")
    served = time.monotonic() - sent
    until(lambda: bus.unanswered(mark), "the end of that read")
    ((begin, end, timeout_ms, _),) = bus.unanswered(mark)
    # The idle joint reads' 50 ms, not the motor library's 1 s.
    assert timeout_ms == round(1000 * split.h.arm._read_timeout_s) == 50
    assert end - begin < 0.5 * LIBRARY_TIMEOUT_S
    assert served < 0.5 * LIBRARY_TIMEOUT_S
    # One read without an answer costs one ROBOT_STATE its jaw reading.
    assert authority.stop_level is not StopLevel.FAULT
    assert split.h.arm.gripper_referenced()
    until(lambda: split.state().gripper.valid is True, "the next jaw reading")


def test_a_short_bus_disturbance_in_a_hold_is_no_fault_stop_and_lost_feedback_is_one(
    rig_split: Split,
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split)
    authority = split.h.app._authority
    # 150 ms without an answer: a joint read fails, and so would a jaw read.
    bus.silent_until = time.monotonic() + 0.15
    time.sleep(1.5)
    state = authority.snapshot()
    assert state.stop_level is not StopLevel.FAULT, state.fault_reason
    assert split.h.arm.gripper_referenced()
    assert split.state().gripper.valid is True
    # Joints that stay silent are a fault-stop, as before.
    bus.silent_until = math.inf
    until(lambda: authority.stop_level is StopLevel.FAULT, "the fault-stop")
    bus.silent_until = 0.0
    assert authority.snapshot().fault_reason.startswith("joint feedback lost for ")


def test_bad_jaw_answers_in_a_hold_do_not_cost_the_jaws_their_reference(
    rig_split: Split,
) -> None:
    split = rig_split
    open_wide(split)
    split.sdk.gripper.stop = BOX_RAD  # the jaws close on an object
    grasp_close(split, "rigid")
    travel = split.skills._grip_open - split.skills._grip_closed
    assert width(split) == pytest.approx(BOX_RAD / travel, abs=0.01)
    bus = holding_with_jaw_readings(split)
    # Three answers in a row that are no position, as the rig's jaw motor
    # gave one (+2.3e18): to a jaw command they are a lost reference.
    bus.jaw_wrong = [2.3e18] * 3
    until(lambda: not bus.jaw_wrong, "the three answers")
    time.sleep(0.3)
    assert split.h.arm.gripper_referenced()
    assert width(split) == pytest.approx(BOX_RAD / travel, abs=0.01)
    # The object can still be let go.
    split.sdk.gripper.stop = 0.0
    release(split)
    assert split.sdk.gripper.at() == pytest.approx(travel, abs=0.01)


# A jaw motor that stays silent in a hold: each jaw reading costs its 50 ms
# timeout, and the joints must keep the bus.

SILENT = 10**6  # reads without an answer: until the test ends


def state_period(split: Split) -> float:
    """The ROBOT_STATE period of the split's robot app, in s."""
    return split.h.app._config.state.publish_period_s


@pytest.mark.parametrize("read_s", [0.002, BUS_READ_S])
def test_jaw_motor_silent_for_good_in_a_hold(rig_split: Split, read_s: float) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split, read_s)
    authority = split.h.app._authority
    began = time.monotonic()
    time.sleep(0.6)
    healthy = bus.joint_reads_a_second(began, time.monotonic())
    # Control: one joint answer that does not come while the jaw motor answers.
    bus.joint_misses = 1
    time.sleep(0.5)
    control = authority.snapshot()
    assert control.stop_level is not StopLevel.FAULT, control.fault_reason

    bus.jaw_misses = SILENT
    silent_at = time.monotonic()
    time.sleep(1.2)
    silent = bus.joint_reads_a_second(silent_at + 0.4, time.monotonic())
    # The joints are read nearly as often as before: not once per jaw timeout.
    assert silent > 0.5 * healthy, (silent, healthy)
    # The agent still gets the arm's state, with the jaws unknown.
    for _ in range(10):
        assert split.arm.get_state().gripper_valid is False
        time.sleep(0.037)
    quiet = authority.snapshot()
    assert quiet.stop_level is not StopLevel.FAULT, quiet.fault_reason
    assert split.h.arm.gripper_referenced()

    # One joint answer that does not come, three times over: no lost feedback.
    for _ in range(3):
        bus.joint_misses = 1
        time.sleep(0.4)
    missed = authority.snapshot()
    assert missed.stop_level is not StopLevel.FAULT, missed.fault_reason
    # A STOP waits for one jaw timeout at most.
    stops, sent = authority.stop_number, time.monotonic()
    split.rig.link.stop("operator pressed stop", "console")
    until(lambda: authority.stop_number != stops, "the STOP")
    assert time.monotonic() - sent < 0.5 * LIBRARY_TIMEOUT_S


def test_jaw_motor_silent_for_half_a_second_then_one_missed_joint_answer(
    rig_split: Split,
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split, 0.002)
    authority = split.h.app._authority
    bus.jaw_misses = SILENT
    time.sleep(0.45)
    bus.joint_misses = 1
    time.sleep(0.1)
    bus.jaw_misses = 0  # the jaw motor answers again
    time.sleep(0.6)
    state = authority.snapshot()
    assert state.stop_level is not StopLevel.FAULT, state.fault_reason
    assert split.state().gripper.valid is True


def test_a_healthy_jaw_motor_is_read_every_robot_state_period_in_a_hold(
    rig_split: Split,
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split)
    period = state_period(split)
    stamps: list[float | None] = []  # the jaw reading's time in each ROBOT_STATE
    split.rig.link.add_state_listener(lambda state: stamps.append(state.gripper.t))
    mark, began = len(bus.jaw_reads), time.monotonic()
    time.sleep(1.0)
    reads, took = list(bus.jaw_reads[mark:]), time.monotonic() - began
    seen = list(stamps)
    assert all(answered for _, _, _, answered in reads)
    assert len(reads) >= 0.7 * took / period
    begins = [begin for begin, _, _, _ in reads]
    assert max(b - a for a, b in zip(begins, begins[1:])) < 2.5 * period
    # Each ROBOT_STATE carries a reading, and nearly each a new one.
    assert None not in seen
    assert len(set(seen)) >= 0.7 * took / period

    # One answer that does not come costs one reading: the next is not put off.
    mark = len(bus.jaw_reads)
    bus.jaw_misses = 1
    until(lambda: bus.unanswered(mark), "the end of the read without an answer")
    ((_, ended, _, _),) = bus.unanswered(mark)
    until(lambda: bus.answered(mark), "the next jaw reading")
    assert bus.answered(mark)[0][0] - ended < 2.0 * period


def test_the_first_jaw_answer_after_an_outage_brings_the_reading_back(
    rig_split: Split,
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split)
    app, period = split.h.app, state_period(split)
    mark = len(bus.jaw_reads)
    bus.jaw_misses = SILENT
    time.sleep(0.6)
    # Two readings in a row, then one per five periods: not one per timeout.
    assert 2 <= len(bus.unanswered(mark)) <= 5
    assert app._jaw_state.gripper_state(split.h.arm).valid is False

    mark = len(bus.jaw_reads)
    bus.jaw_misses = 0  # the jaw motor answers again
    back = time.monotonic()
    until(
        lambda: app._jaw_state.gripper_state(split.h.arm).valid,
        "the jaw reading after the outage",
    )
    # Within one wait of five periods, and a read that began before the answer.
    assert time.monotonic() - back < 5.0 * period + 0.15
    # From then on there is one reading per ROBOT_STATE period again.
    until(lambda: bus.answered(mark), "the end of that read")
    mark, began = len(bus.jaw_reads), time.monotonic()
    time.sleep(0.5)
    took = time.monotonic() - began
    assert len(bus.answered(mark)) >= 0.7 * took / period
    assert bus.unanswered(mark) == []
    assert split.state().gripper.valid is True
    assert app._authority.stop_level is not StopLevel.FAULT


@pytest.mark.parametrize("jaw_silent", [False, True])
def test_silence_of_every_joint_motor_still_latches_joint_feedback_lost(
    rig_split: Split, jaw_silent: bool
) -> None:
    split = rig_split
    bus = holding_with_jaw_readings(split)
    authority = split.h.app._authority
    if jaw_silent:
        bus.jaw_misses = SILENT
        time.sleep(0.6)
    assert authority.stop_level is not StopLevel.FAULT
    bus.joint_misses = SILENT
    began = time.monotonic()
    until(lambda: authority.stop_level is StopLevel.FAULT, "the fault-stop")
    latched = time.monotonic() - began
    time.sleep(0.3)
    bus.joint_misses = 0
    assert authority.snapshot().fault_reason.startswith("joint feedback lost for ")
    # The limit, and the one read without an answer under way when it is passed.
    assert latched < LOST_FEEDBACK_S + 0.15, latched
    # Once a joint read has failed, no jaw reading takes the bus from the joints.
    assert [read for read in list(bus.jaw_reads) if read[0] > began + 0.1] == []


# The rest close measured on the rig: with nothing between the jaws the
# driver's close before a torque release stalls 0.12 rad short of the closed
# stop, more than the driver's reach of 0.1 rad from its target.
REST_STALL_RAD = 0.12
REST = [0.0] * 6


class RigRest:
    """The robot app on the rig's robot file as it stands, and an agent on it.

    The driver is APAI's over the stand-in motors; the motors can be switched
    on again. The agent is the control owner. The arm profile is laid out in
    `folder` as the one the experiment's launcher runs: the rig's robot file
    is taken on no other.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, profile: Any, folder: Path
    ) -> None:
        home = [float(value) for value in profile.arm_cfg.home_q]
        self.thor = Thor(monkeypatch, home, folder=folder)
        self.h, self.motors = self.thor.h, self.thor.motors
        self.config = self.thor.config
        jaws = profile.arm_cfg.gripper
        self.travel = abs(float(jaws.open_pos) - float(jaws.closed_pos))
        self.stall = REST_STALL_RAD / self.travel  # as an opening
        self.rig = SimRig(
            self.h,
            expected_profile_id=self.config.profile_id,
            calibration_id=self.config.camera_calibration_id,
        )
        self.typed = self.h.thor_terminal()
        self.rig.own()

    def rest_tool_closes(self, stall_rad: float = REST_STALL_RAD) -> Any:
        """The close of APAI's rest tool stalls at stall_rad; the state after."""
        jaws = self.motors.gripper
        self.rig.adapter.open_gripper()
        jaws.stop = stall_rad
        mark = len(jaws.sent)
        assert self.rig.adapter.close_gripper_torque() is True
        # The driver took the stall as contact and went to its light hold.
        assert jaws.at() == stall_rad and jaws.light_after(mark)

        def shown() -> bool:
            state = self.rig.link.state().gripper
            return state.valid and state.opening == pytest.approx(
                stall_rad / self.travel
            )

        until(shown, "the jaw reading after the close")
        state = self.rig.link.state()
        assert state.gripper.last_target == 0.0
        return state

    def off(self) -> bool:
        state = self.rig.link.state()
        return state.torque_on is False and state.motion == "rest_unpowered"

    def close(self) -> None:
        self.rig.close()
        self.thor.close()
        tolerance = self.config.rest.rest_tol_rad
        for pose in self.motors.cuts():
            assert pose == pytest.approx(REST, abs=tolerance), "torque cut off rest"


@pytest.fixture
def rig_rest(
    monkeypatch: pytest.MonkeyPatch, rs_profile: Any, tmp_path: Path
) -> Iterator[RigRest]:
    made = RigRest(monkeypatch, rs_profile, tmp_path)
    try:
        yield made
    finally:
        made.close()


def test_on_the_rig_file_a_rest_close_that_stalls_short_of_the_stop_is_closed_on_nothing(
    rig_rest: RigRest,
) -> None:
    r, jaws = rig_rest, rig_rest.config.gripper
    # Beyond the driver's reach of the closed target, below the closed line.
    assert jaws.target_tol < r.stall < jaws.closed_frac
    state = r.rest_tool_closes()
    assert (state.gripper.load, state.gripper.may_be_holding) == ("empty", False)
    assert r.h.app.jaws_not_proven() is None
    # The rest tool's end: the motors go off at measured rest.
    r.rig.adapter.disconnect()
    until(r.off, "rest with the motors off")
    assert r.rig.link.request_owner("nobody", reason="done").status == "completed"
    # On again: a new driver, whose jaws are referenced where they stalled.
    assert r.typed.ask("on", "yes", answer="on: ") == "on: completed"
    until(lambda: r.rig.link.state().torque_on, "the motors on again")
    state = r.rig.link.state()
    assert len(r.thor.drivers) == 2 and state.gripper.referenced
    assert (state.gripper.load, state.stop_level) == ("empty", "none")
    # The typed rest closes the jaws anew and switches the motors off.
    told = r.typed.ask("rest", "path is clear", answer="rest: ")
    assert told == "rest: completed"
    until(r.off, "rest with the motors off")
    assert r.typed.written("rest was refused") == []
    assert not r.h.app._jaw_state.word_given
    assert r.motors.names() == ["close_gripper_torque", "disconnect"] * 2
    assert r.motors.gripper.at() == REST_STALL_RAD


@pytest.mark.parametrize("command", ["rest", "release", "quit"])
def test_on_the_rig_file_nothing_is_asked_about_jaws_a_rest_closed_short_of_the_stop(
    rig_rest: RigRest, command: str
) -> None:
    r = rig_rest
    r.rest_tool_closes()
    # The motors stayed on, and the agent's task ended.
    assert r.rig.link.request_owner("nobody", reason="done").status == "completed"
    typed = [command, "path is clear"] if command == "rest" else [command]
    told = r.typed.ask(*typed, answer=f"{command}: ")
    assert told.startswith(f"{command}: completed")
    assert r.typed.written(f"{command} was refused") == []
    assert not r.h.app._jaw_state.word_given
    until(lambda: r.motors.disconnects == 1, "the torque cut")
    assert r.motors.names() == ["close_gripper_torque"] * 2 + ["disconnect"]


# Just above the closed line (0.04 of 6.2 rad is 0.248 rad), and the wallet.
@pytest.mark.parametrize("stall_rad", [0.28, WALLET_RAD])
def test_on_the_rig_file_jaws_a_rest_closed_above_the_closed_line_are_still_asked_about(
    rig_rest: RigRest, stall_rad: float
) -> None:
    r = rig_rest
    assert stall_rad / r.travel > r.config.gripper.closed_frac
    jaws = r.rest_tool_closes(stall_rad).gripper
    assert (jaws.load, jaws.may_be_holding) == ("loaded", True)
    assert r.rig.link.request_owner("nobody", reason="done").status == "completed"
    told = r.typed.ask("rest", "path is clear", answer="rest: ")
    assert told.startswith("rest: refused: the jaws are not proven empty (loaded)")
    r.typed.wait_line("rest was refused because the jaws are not proven")
    assert r.motors.disconnects == 0 and r.rig.link.state().torque_on


def test_the_adapters_jaw_routines_are_the_drivers(rs_profile: Any) -> None:
    from apai.control.rebot_rs_arm import RebotRSArm
    from isaaccapture_examples.agentic_remote.agent.arm_adapter import RobotLinkArm

    # Every jaw routine of the driver, with its parameters and their defaults.
    for name in ROUTINES:
        at_the_arm = inspect.signature(getattr(RebotRSArm, name))
        over_the_link = inspect.signature(getattr(RobotLinkArm, name))
        assert str(over_the_link) == str(at_the_arm), name
    assert messages.GRIPPER_MODES == (
        messages.JAW_PLAIN,
        messages.JAW_CLOSE_UNTIL_CONTACT,
        messages.JAW_OPEN,
        messages.JAW_CLOSE_TORQUE,
    )


def test_the_robot_apps_jaw_numbers_are_apais(
    rig_config: RobotAppConfig, rs_profile: Any
) -> None:
    from apai.control import rs_gripper
    from apai.control.robstride import GRIP_REACH_RAD

    jaws = rs_profile.arm_cfg.gripper
    travel = abs(float(jaws.open_pos) - float(jaws.closed_pos))
    gripper = rig_config.gripper
    # "Open": where APAI's guarded opening may end. "Closed": below this APAI
    # takes the jaws as closed on nothing. "At the target": APAI's reach.
    assert gripper.full_open_tol == pytest.approx(rs_gripper.OPEN_SHORT_RAD / travel)
    assert gripper.closed_frac == rs_profile.apai_cfg.grasp.air_grasp_frac
    assert gripper.target_tol == pytest.approx(GRIP_REACH_RAD / travel)
    assert apai_backend.JAW_TARGET_TOL_RAD == GRIP_REACH_RAD
    # The rig's full open lies inside it.
    assert 1.0 - RIG_FULL_OPEN < gripper.full_open_tol
