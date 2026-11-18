# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A person's drive close to the table, and a takeover with the gripper tip low.

A checker's table clearance is the height over the table plane under which it
takes only a step that rises. A person's checker has its own,
limits.human.table_clearance_m: 0.0 here, as on the rig, so a person's gripper
tip moves freely down to the table plane. The agent's checker keeps APAI's
safety.table_clearance, and a return to rest starts only above
rest.table_clearance_m of the robot app's file: 2 cm each. A tip under a
checker's clearance is low for it. The last grasp region is the column the
agent's last grasp was allowed to go down in; a return to rest does not start
inside it. A takeover makes a person the control owner; a human drive is one
stretch of human motion toward human targets, the joint targets of
HUMAN_SETPOINT link messages. A typed rest is the operator's return to rest.
An arm settles low when it comes to rest under its joint target. The floor is
where the human program stops a person's tip: TIP_FLOOR_MARGIN_M over their
table clearance. The speed clamp's step is the most a joint moves in one
setpoint of a human drive.

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
from isaaccapture_examples.agentic_remote.human.logic import JOINT_SPEED_SHARE
from isaaccapture_examples.agentic_remote.human.types import TIP_FLOOR_MARGIN_M
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion
from isaaccapture_examples.agentic_remote.robot_app.app import (
    DRIVE_ENDED,
    LOW_TIP,
    OUTSIDE_HUMAN_LIMITS,
    TAKEN_OVER,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.human import SPEED_MARGIN
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
    drives,
    feed,
    human_config,
    setpoint,
    state_now,
    take_over,
)

apai = pytest.mark.apai

PERSON_CLEARANCE_M = 0.0  # limits.human.table_clearance_m of person_config()
# The table clearance of the mock profile and rest.table_clearance_m.
CLEARANCE_M = 0.02
# What the recovery checker says of the tip at BELOW; a typed rest's refusal.
UNDER_CLEARANCE = "the gripper tip is at z=0.011 m, below the table clearance 0.020 m"
IN_REGION = "the gripper tip is inside the last grasp region"
# How APAI's harness refuses a person's step that ends under the table plane.
UNDER_THE_PLANE = r"below table clearance 0\.000 outside the grasp exemption zone"
# BELOW with the tip 7 mm and 2 mm over the table plane, and 4 mm under it.
LOWER = moved(BELOW, 1, 1.69)
LOWEST = moved(BELOW, 1, 1.705)
UNDER = moved(BELOW, 1, 1.72)
# ABOVE with the tip 7 mm lower: still above 2 cm.
SUNK = moved(ABOVE, 1, 1.64)
# The first joint turns the arm about a vertical axis: the tip slides over the
# table at one height, here 1 cm and 2 mm.
SLID_LOW = moved(BELOW, 0, 0.05)
SLID_LOWEST = moved(LOWEST, 0, 0.05)
# Inside the human joint box of person_config(): the tip 1 mm over the table,
# every link over the table plane, and two links of the gripper under 1 cm.
LINK_AT_THE_TABLE = [-0.6716, 2.0581, 0.1814, -0.1053, -1.3877, -1.5782]
LINK_COMPLAINT = r"link/joint 7 at z=0\.008 would hit the table"
# How far a sagging arm settles under each joint target on the second joint,
# rad: about 3 mm at the gripper tip.
SAG = 0.008
SLOW = 0.1  # rad/s: one setpoint moves the gripper tip by less than 1 mm
# Slides of the gripper tip over the table, m, in the robot's base axes (x
# forward, y left), and how finely a slide's way is solved for joint values.
SLIDES = {
    "away from the base": (0.10, 0.0, 0.0),
    "toward the base": (-0.12, 0.0, 0.0),
    "toward the base and to the right": (-0.10, -0.15, 0.0),
}
FINE_M = 0.00025
# Of this many joint targets in a row one does not reach the robot app.
LOST_EVERY = 10


def person_config(profile, **human_limits: object) -> RobotAppConfig:
    """test_apai_app's configuration with human driving on and the rig's table.

    A person's table clearance is 0.0 and their joint speed SLOW.
    human_limits replace further fields of limits.human.
    """
    config = human_config(on_apai.make_config(profile))
    fields = {
        "max_joint_speed": SLOW,
        "table_clearance_m": PERSON_CLEARANCE_M,
        **human_limits,
    }
    human = dataclasses.replace(config.limits.human, **fields)
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


def ended(why: str) -> str:
    """The EVENT text of a human drive that ended for this reason."""
    return DRIVE_ENDED.format(why=why)


def holds_for_the_person(h: ApaiHarness, at: Sequence[float]) -> None:
    """The arm holds at this pose with its motors on; the person owns it."""
    state = state_now(h)
    assert (state.owner, state.stop_level, state.motion) == ("human", "none", "hold")
    assert state.torque_on and h.arm.torque_on() and h.app.takeover_open
    assert h.spy.pose() == pytest.approx(at, abs=1e-9)
    assert "disconnect" not in h.spy.calls


def told_plainly(h: ApaiHarness, result) -> None:
    """The takeover's line is the one of any takeover: it names no low tip."""
    number = result.detail["owner_number"]
    assert h.heard("owner")[-1][1] == TAKEN_OVER.format(number=number)


class PacedHand:
    """A hand as the human program sends it: one HUMAN_SETPOINT per pass.

    A pass is one period of a human drive. While the hand squeezes, each pass
    has the next of its joint targets; None stands for one that is lost on its
    way to the robot app. After the last the hand lets go.
    """

    def __init__(self, h: ApaiHarness) -> None:
        self.h = h
        self.targets: list[list[float] | None] = []
        sleep = h.clock.sleep

        def fed(seconds: float) -> None:
            sleep(seconds)
            worker = threading.current_thread().name == MOTION_WORKER
            if worker and h.app._human_drive.progress() is not None:
                self.one_pass()

        h.clock.sleep = fed

    def one_pass(self) -> None:
        """This pass's HUMAN_SETPOINT reaches the robot app, unless it is lost."""
        if not self.targets:
            feed(self.h)
            return
        q = self.targets.pop(0)
        if q is not None:
            feed(self.h, setpoint(self.h, q))

    def follow(self, targets: Sequence[Sequence[float] | None]) -> None:
        """Let go, then squeeze through targets; the human drive has ended on return."""
        h = self.h
        self.targets = []
        feed(h)
        self.targets = [None if q is None else [float(v) for v in q] for q in targets]
        self.one_pass()
        h.idle()

    def squeeze(self, rise: float, passes: int) -> None:
        """follow() from where the arm is, the second joint `rise` rad further up per pass."""
        at = self.h.spy.pose()  # where the human program takes the hand's home
        targets = []
        for _ in range(passes):
            at[1] -= rise
            targets.append(list(at))
        self.follow(targets)


def clamp_step(config: RobotAppConfig) -> float:
    """The most a joint moves in one setpoint of a person's drive, rad."""
    human = config.limits.human
    return SPEED_MARGIN * human.max_joint_speed * config.executor.period_s


def slide_on_the_floor(
    profile, config: RobotAppConfig, way: Sequence[float]
) -> list[list[float]]:
    """Joint targets, one per pass, that slide the tip `way` over the table and back.

    The tip keeps the height the human program stops it at, TIP_FLOOR_MARGIN_M
    over a person's table clearance, and the gripper its angle. The fastest
    joint moves at most JOINT_SPEED_SHARE of the speed clamp's step per pass,
    as the human program's joint target does. The first is the pose the slide
    begins at.
    """
    kinematics = profile.kinematics
    q = np.array(BELOW, dtype=float)
    begin = np.array(kinematics.fk(q), dtype=float)
    begin[2, 3] = config.limits.human.table_clearance_m + TIP_FLOOR_MARGIN_M
    count = round(float(np.linalg.norm(way)) / FINE_M)
    fine = []
    for index in range(count + 1):
        pose = begin.copy()
        pose[:3, 3] += np.array(way, dtype=float) * index / count
        solved = kinematics.ik(pose, q, tol=1e-7)
        assert solved.success, f"no joint values for the tip at {pose[:3, 3]}"
        q = solved.q
        fine.append(q)
    fine += fine[-2::-1]
    pace = JOINT_SPEED_SHARE * clamp_step(config)
    targets, at = [fine[0]], 0
    while at < len(fine) - 1:
        at += 1
        while at + 1 < len(fine) and np.abs(fine[at + 1] - targets[-1]).max() <= pace:
            at += 1
        assert np.abs(fine[at] - targets[-1]).max() <= pace
        targets.append(fine[at])
    return [target.tolist() for target in targets]


def settles_low(h: ApaiHarness) -> None:
    """From here on the mock arm settles SAG under every joint target it takes."""
    real = h.spy._real["send_joint_target"]

    def sagging(q: np.ndarray) -> None:
        settled = np.array(q, dtype=float)
        settled[1] += SAG
        real(settled)

    h.spy._real["send_joint_target"] = sagging


# A person's drive close to the table


@apai
@pytest.mark.parametrize(
    "low", [low_in_a_grasp, left_low], ids=["after a grasp", "with no grasp region"]
)
def test_a_squeeze_at_1_cm_moves_sideways_and_down_to_the_table(
    app: ApaiHarness, profile, low: Callable[[ApaiHarness, object], object]
) -> None:
    h = app
    low(h, profile)
    assert PERSON_CLEARANCE_M < tip_z(profile, BELOW) < CLEARANCE_M
    # 1 cm over the table is no low tip for a person: the pose is taken as
    # any other, and the takeover is told as any other.
    human = h.checkers.human
    human.check_pose(BELOW)
    assert human.tip_state(BELOW, None).low is None
    told_plainly(h, take_over(h))
    hand = Hand(h)
    before = len(h.sent())

    # Sideways at 1 cm, straight down to 2 mm, sideways there, and up again.
    assert tip_z(profile, SLID_LOW) == pytest.approx(tip_z(profile, BELOW), abs=1e-9)
    assert 0.001 < tip_z(profile, SLID_LOWEST) < 0.003
    for goal in (SLID_LOW, SLID_LOWEST, LOWEST, ABOVE):
        hand.squeeze(goal)
        assert h.heard("human") == []
        holds_for_the_person(h, goal)
    # The first squeeze went out at once: its drive began with a setpoint.
    went = h.sent()[before:]
    assert went[0] == pytest.approx(
        moved(BELOW, 0, SPEED_MARGIN * SLOW * 0.02), abs=1e-12
    )
    heights = [tip_z(profile, q) for q in went]
    assert min(heights) == pytest.approx(tip_z(profile, LOWEST), abs=1e-9)
    assert len([z for z in heights if z < 0.003]) > 20
    # APAI's human checker took every setpoint.
    assert human.harness.violations == []


@apai
def test_a_target_under_the_table_plane_is_refused_by_apais_checker_and_the_arm_holds(
    app: ApaiHarness, profile
) -> None:
    h = app
    h.hold_at(LOWEST)
    told_plainly(h, take_over(h))
    hand = Hand(h)
    before = len(h.sent())
    human = h.checkers.human

    assert tip_z(profile, UNDER) < PERSON_CLEARANCE_M
    hand.squeeze(UNDER)
    (event,) = h.heard("human")
    went = h.sent()[before:]
    heights = [tip_z(profile, q) for q in went]
    # The steps down to the table plane were followed; the one that would
    # end under it was not.
    assert len(went) == 2 and all(z >= PERSON_CLEARANCE_M for z in heights)
    following = moved(went[-1], 1, went[-1][1] + SPEED_MARGIN * SLOW * 0.02)
    assert tip_z(profile, following) < PERSON_CLEARANCE_M
    assert event[2]["status"] == "refused" and event[2]["sent"] == len(went)
    assert re.fullmatch(r"TCP z=-0\.000 " + UNDER_THE_PLANE, event[2]["reason"])
    assert event[1] == ended(event[2]["reason"])
    # It was APAI's checker that refused, and the arm holds over the table.
    assert len(human.harness.violations) == 1
    holds_for_the_person(h, went[-1])

    # Sideways and up from there are taken as before.
    for goal in (moved(went[-1], 0, 0.05), ABOVE):
        hand.squeeze(goal)
        assert len(h.heard("human")) == 1
        holds_for_the_person(h, goal)


@apai
def test_at_the_speed_cap_no_setpoint_of_a_person_ends_under_the_table_plane(
    make_app: Callable[..., ApaiHarness], profile
) -> None:
    # At the mock profile's cap of 1.2 rad/s the speed clamp cuts a joint's
    # step at 23.5 mrad, which lowers the tip by 9 mm.
    h = make_app(person_config(profile, max_joint_speed=1.2))
    h.hold_at(ABOVE)
    take_over(h)
    hand = Hand(h)
    before = len(h.sent())
    hand.squeeze(LOWEST)
    assert h.heard("human") == []
    holds_for_the_person(h, LOWEST)
    down = h.sent()[before:]
    assert len(down) == 4  # three steps at the clamp and the rest

    hand.squeeze(ABOVE)
    before = len(h.sent())
    hand.squeeze(UNDER)
    (event,) = h.heard("human")
    went = h.sent()[before:]
    assert event[2]["status"] == "refused" and event[2]["sent"] == len(went) == 3
    assert re.search(UNDER_THE_PLANE, event[2]["reason"])
    # The fourth step would end under the table plane: the arm holds 7 mm
    # over it, where the third left it.
    heights = [tip_z(profile, q) for q in went]
    assert all(z > PERSON_CLEARANCE_M for z in heights) and heights[-1] < 0.008
    following = moved(went[-1], 1, went[-1][1] + SPEED_MARGIN * 1.2 * 0.02)
    assert following[1] < UNDER[1] and tip_z(profile, following) < PERSON_CLEARANCE_M
    holds_for_the_person(h, went[-1])


@apai
def test_inside_the_last_grasp_region_a_person_lowers_the_tip_like_anywhere(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    # Above 2 cm, in the column of the agent's last grasp.
    h.hold_at(ABOVE)
    human = h.checkers.human
    assert human.tip_state(ABOVE, region).low == IN_REGION
    told_plainly(h, take_over(h))
    hand = Hand(h)
    before = len(h.sent())
    # 7 mm down inside the region, and on down to 2 mm over the table.
    for goal in (SUNK, LOWEST):
        hand.squeeze(goal)
        assert h.heard("human") == []
        holds_for_the_person(h, goal)
    heights = [tip_z(profile, q) for q in (ABOVE, *h.sent()[before:])]
    assert all(after < at for at, after in zip(heights, heights[1:]))
    assert human.harness.violations == []


@apai
def test_an_arm_that_settles_under_its_targets_is_followed_on_every_pass(
    app: ApaiHarness, profile
) -> None:
    h = app
    low_in_a_grasp(h, profile)
    settles_low(h)
    take_over(h)
    ends = drives(h)
    hand = PacedHand(h)
    # A still hand: every pass sends its setpoint, and the arm settles 3 mm
    # under the target.
    before = len(h.sent())
    hand.squeeze(rise=0.0, passes=5)
    assert len(h.sent()) - before == 5
    settled = h.spy.pose()
    sag = tip_z(profile, BELOW) - tip_z(profile, settled)
    assert 0.002 < sag < 0.004 and tip_z(profile, settled) > PERSON_CLEARANCE_M

    # A hand that rises 0.55 mm per pass is followed from its first pass.
    passes = 30
    before = len(h.sent())
    hand.squeeze(rise=0.0015, passes=passes)
    went = h.sent()[before:]
    assert len(went) == passes
    # The first step begins where the arm is measured, as every drive's does.
    assert 0.0015 < SPEED_MARGIN * SLOW * 0.02
    assert went[0] == pytest.approx(moved(settled, 1, settled[1] - 0.0015), abs=1e-9)
    heights = [tip_z(profile, q) for q in (settled, *went)]
    assert all(after > at for at, after in zip(heights, heights[1:]))
    assert [(end.status, end.sent) for end in ends] == [
        ("released", 5),
        ("released", 30),
    ]
    assert h.heard("human") == []
    state = state_now(h)
    assert (state.owner, state.motion, state.torque_on) == ("human", "hold", True)
    assert h.checkers.human.harness.violations == []


@apai
@pytest.mark.parametrize("slide", list(SLIDES))
def test_a_slide_on_the_floor_stays_there_when_joint_targets_are_lost_on_the_way(
    make_app: Callable[..., ApaiHarness], profile, slide: str
) -> None:
    # A person's joints as fast as the mock profile lets them be.
    fastest = profile.apai_cfg.safety.max_joint_vel
    config = person_config(profile, max_joint_speed=fastest)
    h = make_app(config)
    targets = slide_on_the_floor(profile, config, SLIDES[slide])
    floor = PERSON_CLEARANCE_M + TIP_FLOOR_MARGIN_M
    heights = [tip_z(profile, q) for q in targets]
    assert heights == pytest.approx([floor] * len(targets), abs=1e-5)
    h.hold_at(targets[0])
    take_over(h)
    ends = drives(h)
    hand = PacedHand(h)
    before, cut = len(h.sent()), state_now(h).health["clamped"]

    # The robot app misses one joint target in LOST_EVERY: the next lies
    # further from its last setpoint than the speed clamp's step.
    arriving = [
        None if index % LOST_EVERY == 0 else q
        for index, q in enumerate(targets[1:], start=1)
    ]
    assert arriving.count(None) >= 3
    hand.follow(arriving)
    went = h.sent()[before:]
    assert state_now(h).health["clamped"] > cut
    # APAI's checker took every setpoint, and the drive ran to the release.
    assert [end.status for end in ends] == ["released"]
    assert h.heard("human") == []
    assert h.checkers.human.harness.violations == []
    # No setpoint has the tip more than 0.1 mm under the lowest joint target's.
    assert min(tip_z(profile, q) for q in went) > min(heights) - 1e-4
    holds_for_the_person(h, went[-1])


# The takeover with the tip under a person's table clearance


@apai
def test_a_takeover_with_the_tip_under_the_table_plane_is_granted_and_lifts(
    app: ApaiHarness, profile
) -> None:
    h = app
    h.hold_at(UNDER)
    assert -0.005 < tip_z(profile, UNDER) < PERSON_CLEARANCE_M
    # The human checker refuses the measured pose, and only for the low tip.
    human = h.checkers.human
    with pytest.raises(Refusal, match=r"below table clearance 0\.000"):
        human.check_pose(UNDER)
    human.check_low_pose(UNDER)
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
    holds_for_the_person(h, UNDER)

    # A slide under the table plane is refused by APAI's checker: there a
    # step must rise.
    hand = Hand(h)
    hand.squeeze(moved(UNDER, 0, 0.05))
    (event,) = h.heard("human")
    assert event[2]["status"] == "refused" and event[2]["sent"] == 0
    assert re.search(UNDER_THE_PLANE, event[2]["reason"])
    assert h.sent()[sent:] == []
    holds_for_the_person(h, UNDER)

    # A lift is followed, one speed-limited setpoint after the other, each
    # higher than the last, and over the table the tip is free again.
    hand.squeeze(BELOW)
    assert len(h.heard("human")) == 1
    lift = h.sent()[sent:]
    heights = [tip_z(profile, q) for q in (UNDER, *lift)]
    assert len(lift) > 15
    assert all(0.0 < after - at < 0.001 for at, after in zip(heights, heights[1:]))
    assert heights[0] < PERSON_CLEARANCE_M < heights[-1]
    holds_for_the_person(h, BELOW)
    hand.squeeze(SLID_LOW)
    assert len(h.heard("human")) == 1
    holds_for_the_person(h, SLID_LOW)


def beside_the_workspace(profile) -> tuple[RobotAppConfig, list[float], str]:
    pose = moved(UNDER, 0, -0.9)
    assert tip_z(profile, pose) < PERSON_CLEARANCE_M and tip(profile, pose)[1] > 0.3
    return person_config(profile), pose, r"TCP \[.*\] outside workspace \[.*\]"


def under_the_workspace(profile) -> tuple[RobotAppConfig, list[float], str]:
    # The mock profile's workspace box ends 1 cm under the table plane.
    pose = moved(BELOW, 1, 1.75)
    assert tip_z(profile, pose) < -0.01
    return person_config(profile), pose, r"TCP \[.*\] outside workspace \[.*\]"


def outside_the_human_joint_box(profile) -> tuple[RobotAppConfig, list[float], str]:
    pose = moved(UNDER, 0, 0.25)
    assert tip_z(profile, pose) < PERSON_CLEARANCE_M
    assert abs(tip(profile, pose)[1]) < 0.3
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
    assert PERSON_CLEARANCE_M < at[2] < CLEARANCE_M
    # The links between the base and the gripper tip: none under the table
    # plane, the lowest under 1 cm and more than 3 cm from the tip.
    links = profile.kinematics.link_positions(np.array(pose))[1:-1]
    lowest = min(links, key=lambda link: link[2])
    assert 0.0 < lowest[2] < 0.01 and np.linalg.norm(lowest - at) > 0.03
    return person_config(profile), pose, LINK_COMPLAINT


@apai
@pytest.mark.parametrize(
    ("other", "low"),
    [
        (beside_the_workspace, True),
        (under_the_workspace, True),
        (outside_the_human_joint_box, True),
        (a_link_at_the_table, False),
    ],
    ids=[
        "beside the workspace",
        "under the workspace",
        "outside the joint box",
        "a link at the table",
    ],
)
def test_a_takeover_the_human_checker_has_another_complaint_against_is_refused(
    make_app: Callable[..., ApaiHarness],
    profile,
    other: Callable[[object], tuple[RobotAppConfig, list[float], str]],
    low: bool,
) -> None:
    config, pose, complaint = other(profile)
    h = make_app(config)
    h.hold_at(pose)
    human = h.checkers.human
    # Where the tip is low, that is not all the human checker has against
    # the pose. A link between the base and the tip keeps 1 cm over the
    # table plane, whatever a person's table clearance.
    assert (human.tip_state(pose, None).low is not None) == low
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


# The typed rest, the agent and WELCOME


@apai
def test_the_typed_rest_stays_refused_under_2_cm_and_inside_the_last_grasp_region(
    app: ApaiHarness, profile
) -> None:
    h = app
    region = low_in_a_grasp(h, profile)
    take_over(h)
    sent = len(h.sent())
    rest = final(h, h.client.rest())
    assert statuses(h, rest.command_id) == ["accepted", "refused"]
    assert rest.reason == UNDER_CLEARANCE
    assert len(h.sent()) == sent
    holds_for_the_person(h, BELOW)
    hand = Hand(h)

    # A person lifts the gripper above 2 cm. There the tip is still in the
    # column of the last grasp region.
    hand.squeeze(ABOVE)
    assert h.heard("human") == []
    assert tip_z(profile, ABOVE) > CLEARANCE_M
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
def test_the_agent_and_a_return_to_rest_keep_their_table_and_their_box(
    make_app: Callable[..., ApaiHarness], profile
) -> None:
    box = ([0.1, -0.3, -0.01], [0.5, 0.3, 0.55])  # the mock profile's
    wide = ([0.1, -0.5, -0.01], [0.55, 0.5, 0.7])  # the rig's for a person
    config = person_config(profile, workspace_min=wide[0], workspace_max=wide[1])
    h = make_app(config)
    human, agent, recovery = h.checkers.human, h.checkers.agent, h.checkers.recovery
    # A person: the table plane and their own box, as WELCOME names them.
    welcome = h.client.welcome
    assert human.tip_floor() == welcome.human_tip_floor == PERSON_CLEARANCE_M
    assert human.tip_box() == wide
    assert welcome.human_tip_box == [wide[0], wide[1]]
    human.check_pose(BELOW)
    # The agent and a return to rest: 2 cm and the profile's box.
    for other in (agent, recovery):
        assert other.harness.limits.table_clearance == CLEARANCE_M
        assert other.tip_floor() == CLEARANCE_M
        assert other.tip_box() == box
        with pytest.raises(Refusal, match=r"below table clearance 0\.020"):
            other.check_pose(BELOW)
    assert recovery.tip_blocked(BELOW, None) == UNDER_CLEARANCE
    assert recovery.tip_blocked(ABOVE, None) is None
    # The agent's path toward the table is refused where a person's is taken.
    lowered = straight(ABOVE, BELOW, 10)
    human.check_path(ABOVE, lowered, 10.0)
    with pytest.raises(Refusal, match=r"setpoint \d: TCP z=0\.0\d\d below table"):
        agent.check_path(ABOVE, lowered, 10.0)


# The checkers


@apai
def test_apais_checker_tells_a_low_tip_from_its_other_complaints(
    profile, clock: FakeClock
) -> None:
    config = person_config(profile)
    human = on_apai.apai_backend.build_apai_checkers(config, profile, clock).human
    with pytest.raises(Refusal, match=r"below table clearance 0\.000"):
        human.check_pose(UNDER)
    human.check_low_pose(UNDER)
    # Every other complaint is the one check_pose() has.
    for pose in (moved(UNDER, 0, -0.9), moved(BELOW, 1, 1.75), moved(HOME, 0, 0.9)):
        with pytest.raises(Refusal) as plain:
            human.check_pose(pose)
        with pytest.raises(Refusal) as low:
            human.check_low_pose(pose)
        assert low.value.reason == plain.value.reason
        assert "below table clearance" not in low.value.reason
    # Only the tip is let under a clearance: a link at the table is refused.
    for check in (human.check_pose, human.check_low_pose):
        with pytest.raises(Refusal, match=LINK_COMPLAINT):
            check(LINK_AT_THE_TABLE)
    with pytest.raises(Refusal, match="a joint pose needs 6 values"):
        human.check_low_pose(UNDER[:5])
    # A pose with nothing against it passes both.
    for pose in (HOME, BELOW, LOWEST):
        human.check_pose(pose)
        human.check_low_pose(pose)
    human.latch("a fault")
    with pytest.raises(Refusal, match="e-stop latched"):
        human.check_low_pose(UNDER)


@apai
def test_apais_checker_says_how_high_the_tip_is_and_why_it_is_low(
    profile, clock: FakeClock
) -> None:
    config = person_config(profile)
    checkers = on_apai.apai_backend.build_apai_checkers(config, profile, clock)
    human, recovery = checkers.human, checkers.recovery
    x, y, z = tip(profile, BELOW)
    region = GraspRegion((float(x), float(y)), 0.05, 0.0, None)
    # A person's checker: low is under the table plane, or inside the region
    # it is asked about.
    assert human.tip_state(BELOW, None) == TipState(pytest.approx(z), None)
    under = human.tip_state(UNDER, None)
    assert under == TipState(
        pytest.approx(tip_z(profile, UNDER)),
        "the gripper tip is at z=-0.004 m, below the table clearance 0.000 m",
    )
    assert human.tip_state(UNDER, region).low == under.low
    assert human.tip_state(ABOVE, region).low == IN_REGION
    assert human.tip_state(HOME, region).low is None
    # The recovery checker: where the typed rest is refused, and its words.
    assert recovery.tip_state(BELOW, None) == TipState(
        pytest.approx(z), UNDER_CLEARANCE
    )
    assert recovery.tip_blocked(BELOW, None) == UNDER_CLEARANCE
    assert recovery.tip_blocked(ABOVE, region) == IN_REGION
    assert recovery.tip_blocked(ABOVE, None) is None
    with pytest.raises(Refusal, match="a joint pose needs 6 values"):
        human.tip_state(BELOW[:5], None)


@apai
@pytest.mark.parametrize(
    ("rest_clearance", "apai_clearance", "floor"),
    [(0.0, 0.02, "0.020"), (0.05, 0.02, "0.050"), (0.02, 0.05, "0.050")],
    ids=["APAI's is higher", "the app's is higher", "APAI's is much higher"],
)
def test_without_a_clearance_of_their_own_a_person_has_the_higher_of_the_other_two(
    clock: FakeClock, rest_clearance: float, apai_clearance: float, floor: str
) -> None:
    # limits.human.table_clearance_m left out: the human harness keeps APAI's
    # safety.table_clearance, and low is under the higher of that and
    # rest.table_clearance_m.
    overrides = {"safety": {"table_clearance": apai_clearance}}
    profile = on_apai.apai_backend.load_apai_profile("mock", overrides)
    config = person_config(profile, table_clearance_m=None)
    rest = dataclasses.replace(config.rest, table_clearance_m=rest_clearance)
    config = dataclasses.replace(config, rest=rest)
    checkers = on_apai.apai_backend.build_apai_checkers(config, profile, clock)
    human = checkers.human
    assert human.harness.limits.table_clearance == apai_clearance
    assert human.tip_floor() == pytest.approx(float(floor))
    # ABOVE has the tip 3.3 cm over the table, BELOW 1.1 cm.
    low_at_above = float(floor) > 0.04
    assert (human.tip_state(ABOVE, None).low is not None) == low_at_above
    assert human.tip_state(BELOW, None).low == (
        f"the gripper tip is at z=0.011 m, below the table clearance {floor} m"
    )
    # A named clearance is a person's whatever the other two are; the typed
    # rest keeps the clearance of the app's file either way.
    named = person_config(profile, table_clearance_m=0.005)
    named = dataclasses.replace(named, rest=rest)
    own = on_apai.apai_backend.build_apai_checkers(named, profile, clock)
    assert own.human.harness.limits.table_clearance == 0.005
    assert own.human.tip_floor() == pytest.approx(0.005)
    assert own.human.tip_state(BELOW, None).low is None
    for built in (checkers, own):
        blocked = built.recovery.tip_blocked(BELOW, None) is not None
        assert blocked == (rest_clearance > 0.011)
        assert built.agent.harness.limits.table_clearance == apai_clearance


def test_the_simulated_arms_checkers_know_no_table(
    checkers: CheckerSet, config: RobotAppConfig
) -> None:
    human = checkers.human
    region = GraspRegion((0.3, 0.0), 0.05, 0.0, None)
    pose = [0.1] * config.n_joints
    # No kinematics, so no tip: no takeover is told as one with a low tip.
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
