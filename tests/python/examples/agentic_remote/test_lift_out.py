# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A takeover with the gripper tip under the table clearance, and the lift out.

The table clearance is the height over the table the gripper tip keeps outside
a grasp: rest.table_clearance_m of the robot app's file and APAI's own
safety.table_clearance. The last grasp region is the column the agent's last
grasp was allowed to go down in. A tip under the clearance or inside that
region is low. A takeover makes a person the control owner; a human drive is
one stretch of human motion toward human targets, the joint targets of
HUMAN_SETPOINT link messages. A typed rest is the operator's return to rest.
While the tip is low a person can only lift it: no target is followed that
puts the tip more than 1 mm under the highest it was measured since the
takeover. An arm settles low when it comes to rest under its joint target.

On APAI's mock arm and its checkers, as test_apai_app.py sets them up; the
last test is on the simulated arm's checkers.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    GraspRegion,
    HumanSetpoint,
)
from isaaccapture_examples.agentic_remote.robot_app.app import (
    DRIVE_ENDED,
    LOW_TIP,
    OUTSIDE_HUMAN_LIMITS,
    TAKEN_OVER,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.human import (
    LIFT,
    TIP_TOL_M,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    CheckerSet,
    MotionState,
    Owner,
    Refusal,
    TipState,
)

import test_apai_app as on_apai
from test_apai_app import (
    ABOVE,
    BELOW,
    HOME,
    REST,
    ApaiHarness,
    Hand,
    final,
    moved,
    run,
    statuses,
    straight,
)
from test_takeover import (
    MOTION_WORKER,
    ask_takeover,
    feed,
    human_config,
    setpoint,
    state_now,
    take_over,
)

apai = pytest.mark.apai

FLOOR_M = 0.02  # the table clearance of the mock profile and of the app's file
# What the recovery checker says of the tip at BELOW; a typed rest's refusal.
UNDER_CLEARANCE = "the gripper tip is at z=0.011 m, below the table clearance 0.020 m"
IN_REGION = "the gripper tip is inside the last grasp region"
# BELOW with the tip 4 mm lower, and ABOVE with the tip 7 mm lower: still above
# the table clearance.
LOWER = moved(BELOW, 1, 1.69)
SUNK = moved(ABOVE, 1, 1.64)
# ABOVE with the tip 7 mm higher.
LIFTED = moved(ABOVE, 1, 1.60)
# The first joint turns the arm about a vertical axis: the tip slides along the
# table at one height, here 2 cm and 8 cm.
SLID_LOW = moved(BELOW, 0, 0.05)
SLID = moved(ABOVE, 0, 0.2)
# Inside the human joint box of person_config(): the tip 1 mm over the table,
# every link over the table plane, and two links of the gripper under 1 cm.
LINK_AT_THE_TABLE = [-0.6716, 2.0581, 0.1814, -0.1053, -1.3877, -1.5782]
LINK_COMPLAINT = r"link/joint 7 at z=0\.008 would hit the table"
# How far a sagging arm settles under each joint target on the second joint,
# rad: about 3 mm at the gripper tip.
SAG = 0.008


def person_config(profile, **human_limits: object) -> RobotAppConfig:
    """test_apai_app's configuration with human driving on, at the rig's speed.

    At 0.1 rad/s one setpoint moves the gripper tip by less than 1 mm.
    human_limits replace further fields of limits.human.
    """
    config = human_config(on_apai.make_config(profile))
    human = dataclasses.replace(
        config.limits.human, **{"max_joint_speed": 0.1, **human_limits}
    )
    limits = dataclasses.replace(config.limits, human=human)
    return dataclasses.replace(config, limits=limits)


@pytest.fixture(scope="module")
def profile():
    return on_apai.apai_backend.load_apai_profile("mock")


@pytest.fixture
def make_app(profile, clock: FakeClock) -> Iterator[Callable[..., ApaiHarness]]:
    """Builds and starts a robot app on APAI's mock arm with human driving on.

    make_app(config=...); after the test each app is closed and
    test_apai_app's check_driver_use() is applied to it.
    """
    made: list[ApaiHarness] = []

    def make(config: RobotAppConfig | None = None) -> ApaiHarness:
        config = person_config(profile) if config is None else config
        harness = ApaiHarness(config, profile, clock)
        made.append(harness)
        harness.start()
        return harness

    yield make
    for harness in made:
        harness.close()
        on_apai.check_driver_use(harness)


@pytest.fixture
def app(make_app: Callable[..., ApaiHarness]) -> ApaiHarness:
    return make_app()


def tip(profile, q: Sequence[float]) -> np.ndarray:
    """Where APAI's arm model has the gripper tip at q, m."""
    return profile.kinematics.fk(np.array(q, dtype=float))[:3, 3]


def tip_z(profile, q: Sequence[float]) -> float:
    return float(tip(profile, q)[2])


def low_in_a_grasp(h: ApaiHarness, profile) -> GraspRegion:
    """The agent goes down for a grasp, stays low and asks for help.

    The arm holds at BELOW, its tip 1 cm over the table inside the grasp
    region, and nobody owns it. Returns the region.
    """
    x, y, _ = tip(profile, BELOW)
    region = GraspRegion((float(x), float(y)), 0.05, 0.0, None)
    h.spy.place(ABOVE)
    h.new_task()
    down = run(h, straight(ABOVE, BELOW, 10), ABOVE, grasp_region=region)
    assert down.status == "completed", down.reason
    stuck = final(h, h.client.request_owner("help_needed", reason="grasp failed"))
    assert stuck.status == "completed", stuck.reason
    number = stuck.detail["owner_number"]
    h.client.wait_state(lambda s: s.owner == "nobody" and s.owner_number == number)
    assert h.spy.pose() == pytest.approx(BELOW)
    return region


def left_low(h: ApaiHarness, profile) -> None:
    """The arm holds at BELOW with no grasp region on file; nobody owns it."""
    h.hold_at(BELOW)


def ended(h: ApaiHarness, why: str) -> str:
    """The EVENT text of a human drive that ended for this reason."""
    return DRIVE_ENDED.format(why=why)


def holds_for_the_person(h: ApaiHarness, at: Sequence[float]) -> None:
    """The arm holds at this pose with its motors on; the person owns it."""
    state = state_now(h)
    assert (state.owner, state.stop_level, state.motion) == ("human", "none", "hold")
    assert state.torque_on and h.arm.torque_on() and h.app.takeover_open
    assert h.spy.pose() == pytest.approx(at, abs=1e-9)
    assert "disconnect" not in h.spy.calls


class TurningHand(Hand):
    """A hand that goes on to a second goal without letting go of the grip."""

    then: list[float] | None = None

    def __call__(self) -> HumanSetpoint:
        if self.goal is not None and self.then is not None:
            last = self.h.sent()[-1]
            if max(abs(to - at) for at, to in zip(last, self.goal)) < 1e-9:
                self.goal, self.then = self.then, None
        return super().__call__()


class SinkingHand(Hand):
    """A hand that can also go down bit by bit, as a hand that pauses does.

    While it sinks, each target lies `by` rad per joint from the last joint
    target the driver took. The second joint lowers the gripper tip by about
    0.37 mm per mrad. It lets go once it has given its targets.
    """

    by: Sequence[float] = (0.0,) * 6
    left = 0  # targets it still gives while it sinks

    def __call__(self) -> HumanSetpoint:
        if self.left <= 0:
            return super().__call__()
        self.left -= 1
        return setpoint(self.h, self.on_from(self.h.sent()[-1]))

    def on_from(self, last: Sequence[float]) -> list[float]:
        return [at + by for at, by in zip(last, self.by)]

    def sink(self, by: Sequence[float], targets: int) -> None:
        """Let go, then squeeze and sink; the human drive has ended on return."""
        h = self.h
        self.goal = None
        feed(h)
        self.by, self.left = by, targets - 1
        feed(h, setpoint(h, self.on_from(h.spy.pose())))
        h.idle()


def down(rad: float) -> list[float]:
    """The second joint this far on: the gripper tip goes straight down."""
    return [0.0, rad, 0.0, 0.0, 0.0, 0.0]


class PacedHand:
    """A hand as the human program sends it: one HUMAN_SETPOINT per pass.

    A pass is one period of a human drive, whether it sent a joint target or
    not. While the hand squeezes, its target is the pose the squeeze began at
    with the second joint `rise` rad further up at every pass; then it lets go.
    """

    def __init__(self, h: ApaiHarness) -> None:
        self.h = h
        self.at: list[float] = []
        self.rise = 0.0
        self.left = 0  # passes it still squeezes
        sleep = h.clock.sleep

        def fed(seconds: float) -> None:
            sleep(seconds)
            worker = threading.current_thread().name == MOTION_WORKER
            if worker and h.app._human_drive.progress() is not None:
                feed(h, self())

        h.clock.sleep = fed

    def __call__(self) -> HumanSetpoint:
        if self.left <= 0:
            return setpoint(self.h)
        self.left -= 1
        self.at[1] -= self.rise
        return setpoint(self.h, self.at)

    def squeeze(self, rise: float, passes: int) -> None:
        """Let go, then squeeze for `passes`; the human drive has ended on return."""
        h = self.h
        self.left = 0
        feed(h)
        self.at = h.spy.pose()  # where the human program takes the hand's home
        self.rise, self.left = rise, passes
        feed(h, self())
        h.idle()


def settles_low(h: ApaiHarness) -> None:
    """From here on the mock arm settles SAG under every joint target it takes."""
    real = h.spy._real["send_joint_target"]

    def sagging(q: np.ndarray) -> None:
        settled = np.array(q, dtype=float)
        settled[1] += SAG
        real(settled)

    h.spy._real["send_joint_target"] = sagging


# The takeover


@apai
@pytest.mark.parametrize(
    "low", [low_in_a_grasp, left_low], ids=["after a grasp", "with no grasp region"]
)
def test_a_takeover_is_granted_with_the_gripper_tip_under_the_table_clearance(
    app: ApaiHarness, profile, low: Callable[[ApaiHarness, object], object]
) -> None:
    h = app
    low(h, profile)
    assert tip_z(profile, BELOW) < FLOOR_M
    # The human checker refuses the measured pose, and only for the low tip.
    human = h.checkers.human
    with pytest.raises(Refusal, match="below table clearance"):
        human.check_pose(BELOW)
    human.check_low_pose(BELOW)
    sent, jaws = len(h.sent()), len(h.spy.jaw_commands)
    before = h.app.snapshot()
    assert before.owner is Owner.NOBODY

    result = take_over(h)
    now = h.app.snapshot()
    assert now.owner is Owner.HUMAN and h.app.takeover_open
    assert now.owner_number == before.owner_number + 1
    assert result.detail == {"owner": "human", "owner_number": now.owner_number}
    # The operator is told what the person can do there.
    told = TAKEN_OVER.format(number=now.owner_number) + LOW_TIP
    assert h.heard("owner")[-1][1] == told
    assert LOW_TIP == "; the gripper tip is low: only a lift is taken until it is clear"
    # The takeover itself moves nothing.
    assert (len(h.sent()), len(h.spy.jaw_commands)) == (sent, jaws)
    assert h.app.motion_state is MotionState.HOLD
    holds_for_the_person(h, BELOW)


@apai
def test_a_takeover_above_the_clearance_is_told_as_before(
    app: ApaiHarness, profile
) -> None:
    h = app
    h.hold_at(HOME)
    result = take_over(h)
    number = result.detail["owner_number"]
    assert h.heard("owner")[-1][1] == TAKEN_OVER.format(number=number)


@apai
def test_a_takeover_inside_the_last_grasp_region_is_told_as_one_with_a_low_tip(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    # Above the table clearance, in the column of the agent's last grasp.
    h.hold_at(ABOVE)
    human = h.checkers.human
    assert human.tip_state(ABOVE, None).low is None
    assert human.tip_state(ABOVE, region).low == IN_REGION
    result = take_over(h)
    number = result.detail["owner_number"]
    assert h.heard("owner")[-1][1] == TAKEN_OVER.format(number=number) + LOW_TIP


def beside_the_workspace(profile) -> tuple[RobotAppConfig, list[float], str]:
    pose = moved(BELOW, 0, -0.9)
    assert tip_z(profile, pose) < FLOOR_M and tip(profile, pose)[1] > 0.3
    return person_config(profile), pose, r"TCP \[.*\] outside workspace \[.*\]"


def under_the_table_plane(profile) -> tuple[RobotAppConfig, list[float], str]:
    # The mock profile's workspace box ends 1 cm under the table plane.
    pose = moved(BELOW, 1, 1.75)
    assert tip_z(profile, pose) < -0.01
    return person_config(profile), pose, r"TCP \[.*\] outside workspace \[.*\]"


def outside_the_human_joint_box(profile) -> tuple[RobotAppConfig, list[float], str]:
    pose = moved(BELOW, 0, 0.25)
    assert tip_z(profile, pose) < FLOOR_M and abs(tip(profile, pose)[1]) < 0.3
    box = on_apai.make_config(profile).rest
    config = person_config(
        profile,
        joint_min=(-0.2, *box.region_min[1:]),
        joint_max=(0.2, *box.region_max[1:]),
    )
    reason = (
        r"joint 1 target 0\.250 rad is outside the human joint limits -0\.200\.\.0\.200"
    )
    return config, pose, reason


def a_link_at_the_table(profile) -> tuple[RobotAppConfig, list[float], str]:
    pose = LINK_AT_THE_TABLE
    at = tip(profile, pose)
    assert 0.0 < at[2] < FLOOR_M
    # The links between the base and the gripper tip: none under the table
    # plane, the lowest under 1 cm and more than 3 cm from the tip.
    links = profile.kinematics.link_positions(np.array(pose))[1:-1]
    lowest = min(links, key=lambda link: link[2])
    assert 0.0 < lowest[2] < 0.01 and np.linalg.norm(lowest - at) > 0.03
    return person_config(profile), pose, LINK_COMPLAINT


@apai
@pytest.mark.parametrize(
    "other",
    [
        beside_the_workspace,
        under_the_table_plane,
        outside_the_human_joint_box,
        a_link_at_the_table,
    ],
    ids=[
        "beside the workspace",
        "under the table plane",
        "outside the joint box",
        "a link at the table",
    ],
)
def test_a_low_tip_with_another_complaint_of_the_human_checker_is_still_refused(
    make_app: Callable[..., ApaiHarness],
    profile,
    other: Callable[[object], tuple[RobotAppConfig, list[float], str]],
) -> None:
    config, pose, complaint = other(profile)
    h = make_app(config)
    h.hold_at(pose)
    human = h.checkers.human
    # The tip is low, and that is not all the human checker has against the pose.
    assert human.tip_state(pose, None).low is not None
    with pytest.raises(Refusal, match=complaint):
        human.check_low_pose(pose)
    before = h.app.snapshot()
    feed(h)  # fresh controller data and a live view: neither is the reason
    refused = ask_takeover(h)
    assert refused.status == "refused"
    # The text of every takeover the human checker refuses.
    assert re.fullmatch(
        re.escape(OUTSIDE_HUMAN_LIMITS) + complaint + "; type rest", refused.reason
    ), refused.reason
    assert h.app.snapshot() == before and not h.app.takeover_open
    assert h.sent() == [] and h.arm.torque_on()


# The lift out


@apai
def test_the_typed_rest_stays_refused_under_the_table_clearance(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    sent = len(h.sent())
    rest = final(h, h.client.rest())
    assert statuses(h, rest.command_id) == ["accepted", "refused"]
    assert rest.reason == UNDER_CLEARANCE
    assert len(h.sent()) == sent
    holds_for_the_person(h, BELOW)


@apai
def test_a_person_lifts_the_gripper_out_step_by_step_and_then_rest_is_taken(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    before = len(h.sent())

    hand.squeeze(ABOVE)
    assert h.heard("human") == []
    lift = h.sent()[before:]
    heights = [tip_z(profile, q) for q in (BELOW, *lift)]
    # One speed-limited setpoint after the other, each higher than the last.
    assert len(lift) > 20
    rises = [after - at for at, after in zip(heights, heights[1:])]
    assert all(0.0 < rise < 0.001 for rise in rises)
    assert heights[0] < FLOOR_M < heights[-1]
    holds_for_the_person(h, ABOVE)

    # Above the clearance the tip is still in the column of the last grasp region.
    assert h.checkers.recovery.tip_blocked(ABOVE, region) == IN_REGION
    rest = final(h, h.client.rest())
    assert (rest.status, rest.reason) == ("refused", IN_REGION)
    holds_for_the_person(h, ABOVE)

    # Up and out of the region, the grip released: the typed rest is taken.
    hand.squeeze(HOME)
    assert h.heard("human") == []
    assert h.checkers.recovery.tip_blocked(HOME, region) is None
    holds_for_the_person(h, HOME)
    hand.put_down()
    rest = final(h, h.client.rest())  # typed with "path is clear"
    assert rest.status == "completed", rest.reason
    assert h.spy.pose() == pytest.approx(REST)
    assert h.app.snapshot().owner is Owner.HUMAN and h.arm.torque_on()


@apai
def test_a_target_that_lowers_the_tip_is_refused_and_the_arm_holds(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    before = len(h.sent())

    hand.squeeze(LOWER)
    (event,) = h.heard("human")
    assert LIFT == "lift the gripper: "
    why = f"{LIFT}{UNDER_CLEARANCE}, and the target would lower it to z=0.007 m"
    assert event[1] == ended(h, why)
    assert event[2] == {"status": "refused", "reason": why, "sent": 0}
    # Nothing went out: the arm holds where it was, with its motors on.
    assert h.sent()[before:] == []
    holds_for_the_person(h, BELOW)
    # It was this rule, not the human checker: its harness was not asked.
    assert h.checkers.human.harness.violations == []

    # The same hand, lifting: the person can still get out.
    hand.squeeze(ABOVE)
    assert len(h.heard("human")) == 1
    holds_for_the_person(h, ABOVE)


@apai
def test_inside_the_last_grasp_region_the_tip_may_not_be_lowered_either(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(ABOVE)
    human = h.checkers.human
    state = human.tip_state(ABOVE, region)
    assert state.z > FLOOR_M and state.low == IN_REGION

    # 1 mm of tolerance: a target 0.4 mm lower is followed.
    close = moved(ABOVE, 1, 1.621)
    assert 0.0 < tip_z(profile, ABOVE) - tip_z(profile, close) < TIP_TOL_M
    hand.squeeze(close)
    assert h.heard("human") == []
    holds_for_the_person(h, close)

    # A target 7 mm lower is not, though the human checker would take it: the
    # tip would stay above the table clearance.
    assert tip_z(profile, SUNK) > FLOOR_M
    assert human.harness.vet_step(np.array(close), np.array(SUNK), 1.0) is None
    before = len(h.sent())
    hand.squeeze(SUNK)
    (event,) = h.heard("human")
    assert event[1] == ended(
        h, f"{LIFT}{IN_REGION}, and the target would lower it to z=0.025 m"
    )
    assert (event[2]["status"], event[2]["sent"]) == ("refused", 0)
    assert h.sent()[before:] == []
    holds_for_the_person(h, close)


@apai
def test_the_tip_is_held_against_where_it_is_measured_not_where_the_drive_began(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    Hand(h).squeeze(ABOVE)
    # One squeeze: up by 7 mm, then back toward where it began.
    hand = TurningHand(h)
    hand.then = list(ABOVE)
    before = len(h.sent())
    hand.squeeze(LIFTED)
    (event,) = h.heard("human")
    assert event[2]["status"] == "refused"
    assert event[2]["reason"].startswith(f"{LIFT}{IN_REGION}, and the target would")
    # The lift was followed to its end and nothing after it.
    lift = h.sent()[before:]
    assert event[2]["sent"] == len(lift) > 0
    assert lift[-1] == pytest.approx(LIFTED, abs=1e-9)
    holds_for_the_person(h, LIFTED)


@apai
@pytest.mark.parametrize(
    ("by", "followed"),
    [(0.0019, 1), (0.0006, 4)],
    ids=["0.7 mm per target", "0.2 mm per target"],
)
def test_a_hand_that_goes_down_bit_by_bit_is_followed_for_1_mm_in_all(
    app: ApaiHarness, profile, by: float, followed: int
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = SinkingHand(h)
    hand.squeeze(ABOVE)
    before = len(h.sent())
    hand.sink(down(by), targets=40)
    went = h.sent()[before:]
    heights = [tip_z(profile, q) for q in (ABOVE, *went)]
    # Each target lay less than the tolerance under the tip as it was measured
    # then, and the human checker has nothing against the one after the last.
    drops = [at - after for at, after in zip(heights, heights[1:])]
    assert all(0.0 < drop < TIP_TOL_M for drop in drops)
    refused = [at + more for at, more in zip(went[-1], down(by))]
    human = h.checkers.human
    assert human.harness.vet_step(np.array(went[-1]), np.array(refused), 1.0) is None
    (event,) = h.heard("human")
    assert (event[2]["status"], event[2]["sent"]) == ("refused", followed)
    assert len(went) == followed
    told = f"{LIFT}{IN_REGION}, and the target would lower it to z=0.03"
    assert event[2]["reason"].startswith(told)
    # What counts is the highest the tip was measured: in all it went down by
    # less than the tolerance.
    assert 0.0 < heights[0] - heights[-1] < TIP_TOL_M
    assert heights[0] - tip_z(profile, refused) > TIP_TOL_M
    holds_for_the_person(h, went[-1])
    assert human.harness.violations == []


@apai
def test_a_slide_that_sinks_is_ended_once_the_tip_is_1_mm_down(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = SinkingHand(h)
    hand.squeeze(ABOVE)
    before = len(h.sent())
    # Sideways 0.7 mm per target and 0.2 mm down, as a hand that is not level.
    hand.sink([0.0019, 0.0005, 0.0, 0.0, 0.0, 0.0], targets=40)
    went = h.sent()[before:]
    (event,) = h.heard("human")
    assert (event[2]["status"], event[2]["sent"]) == ("refused", len(went))
    assert event[2]["reason"].startswith(f"{LIFT}{IN_REGION}, and the target would")
    assert 3 <= len(went) <= 6
    start, end = tip(profile, ABOVE), tip(profile, went[-1])
    assert 0.0 < start[2] - end[2] < TIP_TOL_M
    assert np.linalg.norm(end[:2] - start[:2]) > 0.002
    holds_for_the_person(h, went[-1])


@apai
def test_between_the_two_table_clearances_a_hand_that_goes_down_is_ended_too(
    make_app: Callable[..., ApaiHarness], profile
) -> None:
    # rest.table_clearance_m of the app's file above APAI's own: low is under
    # the higher one, and APAI's table rule begins under the lower.
    config = person_config(profile)
    rest = dataclasses.replace(config.rest, table_clearance_m=0.05)
    h = make_app(dataclasses.replace(config, rest=rest))
    h.hold_at(ABOVE)
    take_over(h)
    assert h.heard("owner")[-1][1].endswith(LOW_TIP)
    hand = SinkingHand(h)
    before = len(h.sent())
    hand.sink(down(0.0019), targets=40)
    (event,) = h.heard("human")
    low = "the gripper tip is at z=0.032 m, below the table clearance 0.050 m"
    why = f"{LIFT}{low}, and the target would lower it to z=0.031 m"
    assert event[2] == {"status": "refused", "reason": why, "sent": 1}
    (only,) = h.sent()[before:]
    assert 0.0 < tip_z(profile, ABOVE) - tip_z(profile, only) < TIP_TOL_M
    holds_for_the_person(h, only)


@apai
def test_no_number_of_squeezes_lowers_the_tip_by_more_than_the_tolerance(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = SinkingHand(h)
    hand.squeeze(ABOVE)
    # One target 0.7 mm lower, then the grip is released: it is followed.
    hand.sink(down(0.0019), targets=1)
    assert h.heard("human") == []
    low = h.spy.pose()
    assert 0.0 < tip_z(profile, ABOVE) - tip_z(profile, low) < TIP_TOL_M
    # The same squeeze again and again: each begins at the measured pose, and
    # the tip was higher since the takeover than its target is.
    for squeezes in range(1, 4):
        before = len(h.sent())
        hand.sink(down(0.0019), targets=1)
        events = h.heard("human")
        assert len(events) == squeezes
        assert (events[-1][2]["status"], events[-1][2]["sent"]) == ("refused", 0)
        told = f"{LIFT}{IN_REGION}, and the target would lower it to z=0.03"
        assert events[-1][2]["reason"].startswith(told)
        assert h.sent()[before:] == []
        holds_for_the_person(h, low)
    # A lift is followed as before.
    hand.squeeze(LIFTED)
    assert len(h.heard("human")) == 3
    holds_for_the_person(h, LIFTED)


@apai
def test_the_highest_tip_counts_anew_once_the_tip_was_clear(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(LIFTED)
    human = h.checkers.human
    # Sideways out of the region, down by 7 mm there, and back in at that
    # height: 7 mm under where the tip was in the region before.
    out = moved(LIFTED, 0, 0.2)
    assert human.tip_state(out, region).low is None
    for goal in (out, SLID, ABOVE):
        hand.squeeze(goal)
        assert h.heard("human") == []
        holds_for_the_person(h, goal)
    assert human.tip_state(ABOVE, region).low == IN_REGION
    # From there on it is held against where it came in.
    hand.squeeze(SUNK)
    (event,) = h.heard("human")
    assert (event[2]["status"], event[2]["sent"]) == ("refused", 0)
    holds_for_the_person(h, ABOVE)


@apai
def test_the_highest_tip_counts_anew_from_each_takeover(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(LIFTED)
    # Right B: nobody owns the arm. Someone puts it 7 mm lower by hand.
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    h.hold_at(ABOVE)
    take_over(h)
    before = len(h.sent())
    hand.squeeze(SLID)
    assert h.heard("human") == []
    assert len(h.sent()) > before
    holds_for_the_person(h, SLID)


@apai
def test_squeezes_on_an_arm_that_settles_low_do_not_sink_it_further(
    app: ApaiHarness, profile
) -> None:
    """The review's case: four squeezes of one target, each 0.4 mm up."""
    h = app
    low_in_a_grasp(h, profile)
    settles_low(h)
    take_over(h)
    hand = PacedHand(h)
    poses, sent = [h.spy.pose()], []
    for _ in range(4):
        before = len(h.sent())
        hand.squeeze(rise=0.001, passes=1)
        sent.append(len(h.sent()) - before)
        poses.append(h.spy.pose())
    heights = [tip_z(profile, pose) for pose in poses]
    # The first squeeze begins at the measured pose, where the motors held
    # the arm: its target is sent, and the arm settles under it once.
    assert sent[0] == 1 and 0.002 < heights[0] - heights[1] < 0.003
    # From then on the tip was higher since the takeover than a target is.
    # Nothing is sent and the tip stays; no drive ends refused, since it was
    # the arm that settled and not the hand that went down.
    assert sent[1:] == [0, 0, 0]
    assert poses[2:] == [poses[1]] * 3
    assert h.heard("human") == []
    state = state_now(h)
    assert (state.owner, state.motion, state.torque_on) == ("human", "hold", True)
    assert h.checkers.human.harness.violations == []


@apai
def test_a_lift_from_where_the_arm_settled_is_followed_and_not_ended(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    settles_low(h)
    take_over(h)
    hand = PacedHand(h)
    # A still hand: the arm settles under the targets of the first squeeze.
    hand.squeeze(rise=0.0, passes=3)
    top = tip_z(profile, BELOW)  # the highest the tip was measured
    settled = h.spy.pose()
    assert top - tip_z(profile, settled) > 2 * TIP_TOL_M
    before = len(h.sent())
    reads = h.spy.calls.count("read_positions_for_monitor")
    hand.squeeze(rise=0.0, passes=5)
    assert h.sent()[before:] == [] and h.spy.pose() == settled
    # A pass that sends nothing reads the joints as any other does: the
    # contact watch must not lose its feedback while the drive waits.
    assert h.spy.calls.count("read_positions_for_monitor") - reads >= 5 * 6

    # The hand goes up 0.55 mm per pass. For three passes its target is more
    # than the tolerance under that top, and nothing is sent; then the arm
    # follows.
    passes = 30
    hand.squeeze(rise=0.0015, passes=passes)
    went = h.sent()[before:]
    assert h.heard("human") == []
    assert len(went) == passes - 3
    # The first step begins where the arm is measured, as every drive's does.
    step = 0.98 * 0.1 * 0.02
    assert went[0] == pytest.approx(moved(settled, 1, settled[1] - step), abs=1e-9)
    assert went[-1] == pytest.approx(
        moved(settled, 1, settled[1] - passes * 0.0015), abs=1e-9
    )
    heights = [tip_z(profile, q) for q in (settled, *went)]
    assert all(after > at for at, after in zip(heights, heights[1:]))
    assert tip_z(profile, h.spy.pose()) > top
    state = state_now(h)
    assert (state.owner, state.motion, state.torque_on) == ("human", "hold", True)


class ProgramHand:
    """The human program's graph, solver and logic in place of PacedHand.

    On configs/human.yaml, whose arm model is the mock arm's. One pass per
    period: while a human drive runs, in step with it; else from passes().
    """

    def __init__(self, h: ApaiHarness, rows: list[dict]) -> None:
        pytest.importorskip("isaaccapture.retargeters")
        from isaaccapture_examples.agentic_remote.human import __main__ as program
        from isaaccapture_examples.agentic_remote.human.feed import ScriptFeed
        from isaaccapture_examples.agentic_remote.human.ik import Solver
        from isaaccapture_examples.agentic_remote.human.logic import HumanLogic
        from isaaccapture_examples.agentic_remote.human.retarget import HumanGraph
        from repo_paths import repo_root

        configs = repo_root() / "examples" / "agentic_remote" / "configs"
        self.h, self._as_sent = h, program.as_sent
        self.settings = program.load_human_settings(configs / "human.yaml")
        ik = self.settings.ik
        self.solver = Solver.from_apai(
            ik.apai_arm, max_iter=ik.max_iter, tol=ik.tol, max_jump_rad=ik.max_jump_rad
        )
        self.graph = HumanGraph(self.settings.retarget)
        self.feed = ScriptFeed(self.graph, rows)
        self.logic = HumanLogic(self.graph, self.solver, self.settings, "teleop")
        self.sent: list[HumanSetpoint] = []  # the HUMAN_SETPOINTs it made
        sleep = h.clock.sleep

        def fed(seconds: float) -> None:
            sleep(seconds)
            worker = threading.current_thread().name == MOTION_WORKER
            if worker and h.app._human_drive.progress() is not None:
                made = self._pass()
                if made is not None:
                    feed(h, made)

        h.clock.sleep = fed

    def _pass(self) -> HumanSetpoint | None:
        h, logic, graph = self.h, self.logic, self.graph
        now = h.clock.monotonic()
        events = graph.events(stopped=not logic.armed)
        fed = self.feed.step(graph.external_inputs(logic.armed), events, now)
        state = h.app._publisher.build(now)
        messages = logic.step(now, fed, h.client.welcome, state, 0.02, ())
        logic.log_pass()
        made = None
        for message in messages:
            if isinstance(message, HumanSetpoint):
                made = self._as_sent(message, now, now, now)
                self.sent.append(made)
        return made

    def passes(self, seconds: float) -> None:
        """The script for this long on the app's clock; each drive runs to its end."""
        end = self.h.clock.monotonic() + seconds
        while self.h.clock.monotonic() < end:
            self.h.clock.advance(0.02)
            made = self._pass()
            if made is not None:
                feed(self.h, made)
            if made is not None and made.engaged:
                self.h.idle()

    def notes(self) -> list[str]:
        """Each note it sent, once per stretch of passes that said it."""
        said: list[str] = []
        for made in self.sent:
            if not said or said[-1] != made.note:
                said.append(made.note)
        return said


@apai
def test_the_human_program_lifts_a_low_gripper_from_where_the_arm_settled(
    make_app: Callable[..., ApaiHarness], profile
) -> None:
    # A person takes over after a failed grasp with the gripper at the table.
    # A first squeeze with a still hand: the arm settles 3 mm under its
    # target. On every squeeze after it the Thor's drive sends nothing until
    # the joint target's tip is back at the highest it measured; the human
    # program's joint target follows the hand meanwhile where the hand
    # rises, so a hand that rises lifts the gripper out and one that goes
    # sideways moves nothing. At the rig's 0.8 rad/s, where one setpoint
    # lifts the tip that far.
    from test_takeover import drives

    h = make_app(person_config(profile, max_joint_speed=0.8))
    low_in_a_grasp(h, profile)
    settles_low(h)
    take_over(h)
    ended = drives(h)
    hand_at = np.array([0.0, 1.2, -0.3])
    up, right = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    rows = [
        {"t": 0.0, "pos": list(hand_at), "quat": [0, 0, 0, 1]},
        {"t": 0.1, "squeeze": 1.0},  # a still hand: the arm settles
        {"t": 1.2, "squeeze": 0.0},
        {"t": 1.8, "squeeze": 1.0},  # still again: the Thor's drive waits
        {"t": 3.4, "squeeze": 0.0},
        {"t": 4.0, "squeeze": 1.0},  # 2 cm to the right at one height: waits too
        {"t": 4.3},
        {"t": 4.7, "pos": list(hand_at + 0.02 * right)},
        {"t": 5.6, "squeeze": 0.0},
        {"t": 5.8, "pos": list(hand_at)},
        {"t": 6.2, "squeeze": 1.0},  # the lift: 3 cm up in 3 s
        {"t": 6.7},
        {"t": 9.7, "pos": list(hand_at + 0.03 * up)},
        {"t": 10.7, "squeeze": 0.0},
    ]
    hand = ProgramHand(h, rows)
    top = tip_z(profile, h.spy.pose())
    hand.passes(1.6)
    at_rest = list(h.spy.pose())
    settled = tip_z(profile, at_rest)
    assert top < FLOOR_M and 0.002 < top - settled < 0.004
    assert [end.status for end in ended] == ["released"]

    # The second squeeze, with a still hand: the Thor's drive sends nothing,
    # the note says what to do, and after 1 s the engagement ends as one the
    # arm did not follow.
    before = len(h.sent())
    hand.passes(2.2)
    assert len(h.sent()) == before and len(ended) == 2 and ended[1].sent == 0
    raise_the_hand = "raise the hand: the gripper is under the table clearance"
    not_followed = (
        "release the grip and squeeze again: the arm did not follow the squeeze "
        "within 1 s"
    )
    said = hand.notes()
    assert said[-3:] == [raise_the_hand, not_followed, "squeeze the grip to move"]
    assert hand.logic.counts()["solution_ahead"] == 1

    # The third squeeze, with the hand going to the right at one height:
    # nothing is sent either. The arm does not start where the hand did not
    # rise.
    hand.passes(2.2)
    assert len(h.sent()) == before and len(ended) == 3 and ended[2].sent == 0
    assert hand.logic.counts()["solution_ahead"] == 2
    assert tip_z(profile, h.spy.pose()) == settled

    # The fourth squeeze, with the hand rising: the arm is lifted out, above
    # the table clearance, and the drive goes on until the grip is released.
    hand.passes(5.5)
    statuses = ["released", "disengaged", "disengaged", "released"]
    assert [end.status for end in ended] == statuses
    assert ended[3].sent > 100
    lifted = tip_z(profile, h.spy.pose())
    assert lifted > FLOOR_M and lifted - settled == pytest.approx(0.03, abs=0.004)
    assert hand.logic.counts()["solution_ahead"] == 2
    # Its first setpoint is on the hand's way, straight up, and inside the
    # hand's speed limit for one setpoint period.
    first = tip(profile, h.sent()[before]) - tip(profile, at_rest)
    most = hand.settings.retarget.max_linear_velocity * 0.02
    assert 0.0 < first[2] <= most and float(np.linalg.norm(first[:2])) < 0.0005
    # The Thor told of two drives' ends: the two that sent nothing.
    told = [what for _, what, _ in h.heard("human")]
    assert len(told) == 2
    assert all("the arm did not follow the squeeze within 1 s" in what for what in told)
    said = hand.notes()
    assert raise_the_hand in said[said.index(not_followed) :]
    state = state_now(h)
    assert (state.owner, state.motion, state.torque_on) == ("human", "hold", True)
    assert h.checkers.human.harness.violations == []
    print(
        f"\ntip z {1000 * top:.1f} mm, settled at {1000 * settled:.1f} mm, lifted to "
        f"{1000 * lifted:.1f} mm; drives {[(end.status, end.sent) for end in ended]}"
    )


@apai
def test_a_step_that_dips_on_the_way_to_a_higher_target_is_refused_too(
    make_app: Callable[..., ApaiHarness], profile
) -> None:
    # At 1.2 rad/s the speed clamp cuts a joint's step at 23.5 mrad.
    h = make_app(person_config(profile, max_joint_speed=1.2))
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(ABOVE)
    # The target's tip is higher. The third joint, which raises the tip, has
    # the farthest to go and is cut; the two that lower it are not.
    goal = [0.0, ABOVE[1] + 0.02, ABOVE[2] + 0.05, -0.02, ABOVE[4], 0.0]
    assert tip_z(profile, goal) > tip_z(profile, ABOVE) + 0.003
    before = len(h.sent())
    hand.squeeze(goal)
    (event,) = h.heard("human")
    assert (event[2]["status"], event[2]["sent"]) == ("refused", 0)
    assert event[2]["reason"].startswith(
        f"{LIFT}{IN_REGION}, and the step toward the target would lower it to z=0.0"
    )
    assert h.sent()[before:] == []
    holds_for_the_person(h, ABOVE)


@apai
def test_steps_that_dip_bit_by_bit_are_followed_for_1_mm_in_all(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(ABOVE)
    # The target of the test above, at the rig's speed: a step is 2 mrad per
    # joint and lowers the tip by a fraction of a millimetre.
    goal = [0.0, ABOVE[1] + 0.02, ABOVE[2] + 0.05, -0.02, ABOVE[4], 0.0]
    before = len(h.sent())
    hand.squeeze(goal)
    went = h.sent()[before:]
    heights = [tip_z(profile, q) for q in (ABOVE, *went)]
    drops = [at - after for at, after in zip(heights, heights[1:])]
    assert len(went) > 1 and all(0.0 < drop < TIP_TOL_M for drop in drops)
    (event,) = h.heard("human")
    assert (event[2]["status"], event[2]["sent"]) == ("refused", len(went))
    assert event[2]["reason"].startswith(
        f"{LIFT}{IN_REGION}, and the step toward the target would lower it to z=0.03"
    )
    assert 0.0 < heights[0] - heights[-1] < TIP_TOL_M
    holds_for_the_person(h, went[-1])


@apai
def test_a_slide_under_the_table_clearance_is_refused_by_the_human_checker(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    before = len(h.sent())
    # A slide along the table lowers the tip by nothing, so the lift rule
    # takes it. APAI's human checker does not: under its table clearance a
    # step that moves sideways must rise.
    assert tip_z(profile, SLID_LOW) == pytest.approx(tip_z(profile, BELOW), abs=1e-9)
    hand.squeeze(SLID_LOW)
    (event,) = h.heard("human")
    assert event[2]["status"] == "refused" and event[2]["sent"] == 0
    assert not event[2]["reason"].startswith(LIFT)
    assert "below table clearance 0.020 outside the grasp" in event[2]["reason"]
    assert h.sent()[before:] == []
    holds_for_the_person(h, BELOW)


@apai
def test_a_slide_above_the_clearance_leaves_the_region_and_then_all_is_as_before(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    take_over(h)
    hand = Hand(h)
    hand.squeeze(ABOVE)
    human = h.checkers.human
    assert human.tip_state(ABOVE, region).low == IN_REGION

    # Sideways at one height inside the region: nothing is lowered.
    assert tip_z(profile, SLID) == pytest.approx(tip_z(profile, ABOVE), abs=1e-9)
    hand.squeeze(SLID)
    assert h.heard("human") == []
    holds_for_the_person(h, SLID)
    assert human.tip_state(SLID, region).low is None

    # Above the clearance and outside the region a person may lower the tip.
    down = moved(SLID, 1, 1.64)
    assert FLOOR_M < tip_z(profile, down) < tip_z(profile, SLID) - 0.005
    hand.squeeze(down)
    assert h.heard("human") == []
    holds_for_the_person(h, down)


# The checkers


@apai
def test_apais_checker_tells_a_low_tip_from_its_other_complaints(
    profile, clock: FakeClock
) -> None:
    config = person_config(profile)
    human = on_apai.apai_backend.build_apai_checkers(config, profile, clock).human
    with pytest.raises(Refusal, match="below table clearance"):
        human.check_pose(BELOW)
    human.check_low_pose(BELOW)
    # Every other complaint is the one check_pose() has.
    for pose in (moved(BELOW, 0, -0.9), moved(BELOW, 1, 1.75), moved(HOME, 0, 0.9)):
        with pytest.raises(Refusal) as plain:
            human.check_pose(pose)
        with pytest.raises(Refusal) as low:
            human.check_low_pose(pose)
        assert low.value.reason == plain.value.reason
        assert "below table clearance" not in low.value.reason
    # Only the tip is let under the clearance: a link there is still refused.
    with pytest.raises(Refusal, match="below table clearance"):
        human.check_pose(LINK_AT_THE_TABLE)
    with pytest.raises(Refusal, match=LINK_COMPLAINT):
        human.check_low_pose(LINK_AT_THE_TABLE)
    with pytest.raises(Refusal, match="a joint pose needs 6 values"):
        human.check_low_pose(BELOW[:5])
    # A pose with nothing against it passes both.
    human.check_pose(HOME)
    human.check_low_pose(HOME)
    human.latch("a fault")
    with pytest.raises(Refusal, match="e-stop latched"):
        human.check_low_pose(BELOW)


@apai
def test_apais_checker_says_how_high_the_tip_is_and_why_it_is_low(
    profile, clock: FakeClock
) -> None:
    config = person_config(profile)
    human = on_apai.apai_backend.build_apai_checkers(config, profile, clock).human
    x, y, z = tip(profile, BELOW)
    region = GraspRegion((float(x), float(y)), 0.05, 0.0, None)
    assert human.tip_state(BELOW, None) == TipState(pytest.approx(z), UNDER_CLEARANCE)
    # Where the typed rest is refused, and with its words.
    assert human.tip_blocked(BELOW, None) == UNDER_CLEARANCE
    assert human.tip_state(BELOW, region).low == UNDER_CLEARANCE
    above = human.tip_state(ABOVE, None)
    assert above == TipState(pytest.approx(tip_z(profile, ABOVE)), None)
    assert human.tip_state(ABOVE, region).low == IN_REGION
    assert human.tip_state(HOME, region).low is None
    with pytest.raises(Refusal, match="a joint pose needs 6 values"):
        human.tip_state(BELOW[:5], None)


@apai
@pytest.mark.parametrize(
    ("rest_clearance", "apai_clearance", "floor"),
    [(0.0, 0.02, "0.020"), (0.05, 0.02, "0.050"), (0.02, 0.05, "0.050")],
    ids=["APAI's is higher", "the app's is higher", "APAI's is much higher"],
)
def test_low_is_under_the_higher_of_the_two_table_clearances(
    clock: FakeClock, rest_clearance: float, apai_clearance: float, floor: str
) -> None:
    overrides = {"safety": {"table_clearance": apai_clearance}}
    profile = on_apai.apai_backend.load_apai_profile("mock", overrides)
    config = person_config(profile)
    rest = dataclasses.replace(config.rest, table_clearance_m=rest_clearance)
    config = dataclasses.replace(config, rest=rest)
    human = on_apai.apai_backend.build_apai_checkers(config, profile, clock).human
    # ABOVE has the tip 3.3 cm over the table, BELOW 1.1 cm.
    low_at_above = float(floor) > 0.04
    assert (human.tip_state(ABOVE, None).low is not None) == low_at_above
    assert human.tip_state(BELOW, None).low == (
        f"the gripper tip is at z=0.011 m, below the table clearance {floor} m"
    )
    # The typed rest keeps the clearance of the app's file.
    assert (human.tip_blocked(BELOW, None) is not None) == (rest_clearance > 0.011)


def test_the_simulated_arms_checkers_know_no_table(
    checkers: CheckerSet, config: RobotAppConfig
) -> None:
    human = checkers.human
    region = GraspRegion((0.3, 0.0), 0.05, 0.0, None)
    pose = [0.1] * config.n_joints
    # No kinematics, so no tip: the lift rule never applies on this arm.
    human.set_tip_hook(lambda q, region: "gripper tip below the table clearance")
    assert human.tip_blocked(pose, region) is not None
    assert human.tip_state(pose, region) is None
    # check_low_pose() has no table rule to leave out: it is check_pose().
    human.check_low_pose(pose)
    outside = [config.limits.human.joint_max[0] + 0.1, *pose[1:]]
    with pytest.raises(Refusal) as plain:
        human.check_pose(outside)
    with pytest.raises(Refusal) as low:
        human.check_low_pose(outside)
    assert low.value.reason == plain.value.reason
    human.latch("a fault")
    with pytest.raises(Refusal, match="fault-stop latched"):
        human.check_low_pose(pose)
