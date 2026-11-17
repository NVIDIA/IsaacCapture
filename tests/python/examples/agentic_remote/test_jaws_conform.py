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

import functools
import inspect
import threading
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.interfaces import GripperOutcome
from test_apai_backend import GuardTime, MotorSdk, make_config, rs_arm
from test_arm_adapter import SimRig

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


@pytest.fixture
def split(
    rig_config: RobotAppConfig,
    rs_profile: Any,
    monkeypatch: pytest.MonkeyPatch,
    driver_time: GuardTime,
) -> Iterator[Split]:
    """The robot app on APAI's driver, at rest, with the agent as control owner."""
    clock = SystemClock()
    port, sdk = stand_in(rig_config, rs_profile, clock, monkeypatch, driver_time)
    checkers = apai_backend.build_apai_checkers(rig_config, rs_profile, clock)
    h = AppHarness(
        rig_config, port, checkers, clock, connected=False, step_period_s=None
    )
    h.start()
    rig = SimRig(h, expected_profile_id=rig_config.profile_id)
    try:
        rig.own()
        yield Split(rig, sdk, rs_profile)
    finally:
        rig.close()
        h.close()


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
