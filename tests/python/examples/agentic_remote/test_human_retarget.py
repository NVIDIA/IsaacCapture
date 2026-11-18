# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""human/retarget.py: the retargeting graph of the human program.

Isaac Capture's nodes run here directly, pass by pass, on ControllerSnapshot
objects: no headset and no Teleop Session Server. A pass is one run of the
graph, 20 ms after the one before unless a test gives another time; a late
pass comes later than that. The stage space is the space the controller poses
are given in: x to the person's right, y up, z toward the person. The robot's
base axes are x forward, y left, z up. The home is the pose an engagement
starts from. The file is skipped without Isaac Capture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaaccapture.retargeters")
pytest.importorskip("isaaccapture.schema")

from isaaccapture.retargeting_engine.interface import (  # noqa: E402
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    GraphTime,
    TensorGroup,
)
from isaaccapture.schema import (  # noqa: E402
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    Point,
    Pose,
    Quaternion,
)
from isaaccapture_examples.agentic_remote.human.retarget import (  # noqa: E402
    EE_KEY,
    LIMITER_PASS_S,
    PERMITTED_LEAF,
    TO_BASE_LEAF,
    HumanGraph,
)
from isaaccapture_examples.agentic_remote.human.types import (  # noqa: E402
    GraphStep,
    RetargetSettings,
)

PASS_S = 0.02  # the human program runs 50 passes per second
PASS_NS = 20_000_000
# The poses arrive as float32.
POSE_TOL = 1e-6
PATH_TOL = 1e-5  # after some tens of passes

X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 1.0)
NO_TURN = (0.0, 0.0, 0.0, 1.0)
HAND = (0.1, 1.2, -0.3)  # where the hand is in the stage space, m
# Hand moves in the stage space, as unit vectors.
UP = (0.0, 1.0, 0.0)
AWAY = (0.0, 0.0, -1.0)  # away from the person
LEFT = (-1.0, 0.0, 0.0)  # to the person's left
DOWN = (0.0, -1.0, 0.0)
TOWARD = (0.0, 0.0, 1.0)  # toward the person
RIGHT = (1.0, 0.0, 0.0)  # to the person's right
# Two poses of the arm, as joint values in rad, away from its joint limits.
RAISED_Q = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
MID_Q = (0.3, 0.8, 1.0, 0.2, -0.4, 0.1)


def turn(axis: Sequence[float], angle: float) -> np.ndarray:
    """The 3x3 rotation by angle, in rad, about a unit axis."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def quat(axis: Sequence[float], angle: float) -> tuple[float, float, float, float]:
    """The same rotation as a quaternion [x, y, z, w]."""
    half = math.sin(angle / 2)
    return (axis[0] * half, axis[1] * half, axis[2] * half, math.cos(angle / 2))


def pose(position: Sequence[float], rotation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def shifted(matrix: np.ndarray, by: Sequence[float]) -> np.ndarray:
    moved = matrix.copy()
    moved[:3, 3] += by
    return moved


def along(start: Sequence[float], direction: Sequence[float], distance: float):
    return tuple(s + distance * d for s, d in zip(start, direction))


# A home that is turned about every axis, so that a wrong quaternion order shows.
HOME = pose((0.30, 0.05, 0.22), turn(Z, 0.3) @ turn(Y, -0.4) @ turn(X, 0.2))


def snapshot(
    position: Sequence[float],
    orientation: Sequence[float] = NO_TURN,
    *,
    squeeze: float = 0.0,
    trigger: float = 0.0,
    a: bool = False,
    b: bool = False,
    valid: bool = True,
    aim: Sequence[float] | None = None,
    aim_valid: bool = True,
) -> ControllerSnapshot:
    """One sample of the right controller; aim: the aim pose's own position."""
    turned = Quaternion(*orientation)
    grip = ControllerPose(Pose(Point(*position), turned), valid)
    aimed = ControllerPose(
        Pose(Point(*(position if aim is None else aim)), turned), aim_valid
    )
    buttons = ControllerInputState(a, b, False, False, 0.0, 0.0, squeeze, trigger)
    return ControllerSnapshot(grip, aimed, buttons)


class Bench:
    """A HumanGraph run pass by pass, in the order the human program runs it."""

    def __init__(
        self,
        settings: RetargetSettings | None = None,
        home: np.ndarray | None = HOME,
    ) -> None:
        self.graph = HumanGraph(settings or RetargetSettings(), home)
        self.now_ns = 0
        self.events = ExecutionEvents()  # those of the last pass

    def step(
        self,
        position: Sequence[float] | None = HAND,
        orientation: Sequence[float] = NO_TURN,
        *,
        permitted: bool = True,
        stopped: bool = False,
        pass_ns: int = PASS_NS,
        **controller: object,
    ) -> GraphStep:
        """One pass; position None: the right controller is not active.

        pass_ns: the time since the pass before this one.
        """
        self.events = self.graph.events(stopped=stopped)
        right = None
        if position is not None:
            right = snapshot(position, orientation, **controller)
        self.now_ns += pass_ns
        outputs = self.graph.pipeline.execute_pipeline(
            {
                **self.graph.controller_inputs(right),
                **self.graph.external_inputs(permitted),
            },
            context=ComputeContext(
                graph_time=GraphTime(self.now_ns, self.now_ns),
                execution_events=self.events,
            ),
        )
        return self.graph.read(outputs)

    def engage(self, position: Sequence[float] = HAND) -> GraphStep:
        """A pass with the grip released, then one with it squeezed."""
        self.step(position)
        return self.step(position, squeeze=1.0)

    def walk(
        self,
        start: Sequence[float],
        direction: Sequence[float],
        distance: float,
        passes: int,
    ) -> GraphStep:
        """Move the squeezed hand in a straight line, in equal steps."""
        for i in range(1, passes + 1):
            position = along(start, direction, distance * i / passes)
            step = self.step(position, squeeze=1.0)
        return step


def assert_both_at(step: GraphStep, expected: np.ndarray, tol: float = POSE_TOL):
    """Both targets, before and after the rate limiter, are at one pose."""
    for name in ("commanded_target", "ee_target"):
        target = getattr(step, name)
        assert target is not None, name
        assert target.shape == (4, 4) and target.dtype == np.float64, name
        np.testing.assert_allclose(target, expected, atol=tol, err_msg=name)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the rotation from one 4x4 pose to another, rad."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return math.acos(max(-1.0, min(1.0, cos)))


# The clutch and the re-base


@pytest.mark.parametrize(
    "hand, orientation",
    [
        (HAND, NO_TURN),
        ((-0.4, 0.9, 0.25), quat((0.6, 0.0, 0.8), 1.1)),
    ],
)
def test_a_squeeze_engages_and_both_targets_start_at_the_home(hand, orientation):
    bench = Bench()
    released = bench.step(hand, orientation)
    assert released.present and not released.engaged
    assert_both_at(released, HOME)

    engaged = bench.step(hand, orientation, squeeze=1.0)
    assert engaged.engaged
    assert_both_at(engaged, HOME)
    assert engaged.limiter == "latched"


def test_without_a_home_the_first_home_is_the_identity():
    bench = Bench(home=None)
    assert_both_at(bench.engage(), np.eye(4))


def test_the_limited_target_is_at_the_home_on_the_engage_pass_after_the_home_moved_20_cm():
    bench = Bench()
    bench.engage()
    assert bench.events.reset  # the engage pass followed a released one
    lifted = bench.walk(HAND, UP, 0.01, 10)
    assert not bench.events.reset  # no pass that follows an engaged one
    assert_both_at(lifted, shifted(HOME, (0.0, 0.0, 0.01)), PATH_TOL)
    released = bench.step(HAND)
    assert not released.engaged and not bench.events.reset

    # The arm was moved meanwhile: the next engagement starts 20 cm away.
    moved = shifted(HOME, (0.0, 0.0, 0.20))
    bench.graph.set_home(moved)
    engaged = bench.step(HAND, squeeze=1.0)
    assert bench.events.reset
    assert engaged.engaged
    assert engaged.limiter == "latched"
    assert_both_at(engaged, moved)

    follows = bench.step(along(HAND, UP, 0.001), squeeze=1.0)
    assert not bench.events.reset
    assert follows.limiter == "passed"
    assert_both_at(follows, shifted(moved, (0.0, 0.0, 0.001)), PATH_TOL)


def test_a_released_grip_holds_and_the_next_squeeze_starts_from_the_new_home():
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    lifted = shifted(HOME, (0.0, 0.0, 0.01))

    # The grip opens while the hand moves on: the targets stay.
    released = bench.step(along(HAND, UP, 0.05))
    assert not released.engaged
    assert_both_at(released, lifted, PATH_TOL)

    new_home = pose((0.25, -0.10, 0.30), turn(Y, 0.5))
    elsewhere = (0.3, 1.0, -0.5)
    for _ in range(3):
        bench.graph.set_home(new_home)
        waiting = bench.step(elsewhere)
        assert not waiting.engaged
        assert_both_at(waiting, new_home)

    engaged = bench.step(elsewhere, squeeze=1.0)
    assert engaged.engaged
    assert_both_at(engaged, new_home)
    follows = bench.walk(elsewhere, UP, 0.005, 5)
    assert_both_at(follows, shifted(new_home, (0.0, 0.0, 0.005)), PATH_TOL)


# The hand's motion in the robot's base axes


def test_the_target_follows_the_hand_in_the_robots_base_axes():
    bench = Bench()
    bench.engage()
    # 1 mm per pass is 0.05 m/s: under the rate limiter's 0.10 m/s.
    up = bench.walk(HAND, UP, 0.01, 10)
    assert up.limiter == "passed"
    assert_both_at(up, shifted(HOME, (0.0, 0.0, 0.01)), PATH_TOL)

    hand = along(HAND, UP, 0.01)
    away = bench.walk(hand, AWAY, 0.01, 10)
    assert_both_at(away, shifted(HOME, (0.01, 0.0, 0.01)), PATH_TOL)

    hand = along(hand, AWAY, 0.01)
    left = bench.walk(hand, LEFT, 0.01, 10)
    assert_both_at(left, shifted(HOME, (0.01, 0.01, 0.01)), PATH_TOL)

    # The hand turns about the stage's vertical axis: the target turns about
    # the base's vertical axis, on the spot. 0.004 rad per pass is 0.2 rad/s.
    hand = along(hand, LEFT, 0.01)
    for i in range(1, 11):
        turned = bench.step(hand, quat(Y, 0.004 * i), squeeze=1.0)
    assert turned.limiter == "passed"
    expected = shifted(HOME, (0.01, 0.01, 0.01))
    expected[:3, :3] = turn(Z, 0.04) @ HOME[:3, :3]
    assert_both_at(turned, expected, PATH_TOL)


def test_a_stage_yaw_turns_the_hands_motion():
    # The stage is turned a quarter about its vertical axis: what the default
    # takes for "away from the person" now points to the robot's left.
    bench = Bench(RetargetSettings(stage_yaw_deg=90.0))
    bench.engage()
    away = bench.walk(HAND, AWAY, 0.01, 10)
    assert_both_at(away, shifted(HOME, (0.0, 0.01, 0.0)), PATH_TOL)
    hand = along(HAND, AWAY, 0.01)
    up = bench.walk(hand, UP, 0.01, 10)
    assert_both_at(up, shifted(HOME, (0.0, 0.01, 0.01)), PATH_TOL)
    hand = along(hand, UP, 0.01)
    left = bench.walk(hand, LEFT, 0.01, 10)
    assert_both_at(left, shifted(HOME, (-0.01, 0.01, 0.01)), PATH_TOL)

    # The other way round.
    bench = Bench(RetargetSettings(stage_yaw_deg=-90.0))
    bench.engage()
    away = bench.walk(HAND, AWAY, 0.01, 10)
    assert_both_at(away, shifted(HOME, (0.0, -0.01, 0.0)), PATH_TOL)


def test_the_position_scale_scales_the_hands_motion_and_not_its_turn():
    bench = Bench(RetargetSettings(position_scale=0.5))
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    hand = along(HAND, UP, 0.01)
    for i in range(1, 6):
        turned = bench.step(hand, quat(Y, 0.004 * i), squeeze=1.0)
    expected = shifted(HOME, (0.0, 0.0, 0.005))
    expected[:3, :3] = turn(Z, 0.02) @ HOME[:3, :3]
    assert_both_at(turned, expected, PATH_TOL)


def test_a_controller_to_base_that_is_no_rotation_is_refused():
    mirrored = ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    stretched = ((0.0, 0.0, -2.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    for rows in (mirrored, stretched, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        with pytest.raises(ValueError, match=r"retarget\.controller_to_base"):
            HumanGraph(RetargetSettings(controller_to_base=rows))
    # A rotation typed with 5 decimals is taken.
    typed = tuple(tuple(round(float(v), 5) for v in row) for row in turn(Y, 0.3))
    HumanGraph(RetargetSettings(controller_to_base=typed))


# What gates and ends an engagement


def test_an_engage_is_withheld_until_it_is_permitted():
    bench = Bench()
    bench.step(HAND)
    # Squeezed and not permitted, while the hand moves 5 cm.
    for i in range(1, 6):
        withheld = bench.step(along(HAND, UP, 0.01 * i), squeeze=1.0, permitted=False)
        assert not withheld.engaged
        assert_both_at(withheld, HOME)

    hand = along(HAND, UP, 0.05)
    engaged = bench.step(hand, squeeze=1.0)
    assert engaged.engaged
    assert_both_at(engaged, HOME)  # the 5 cm are not made up

    # A withheld permission delays an engage; it does not end an engagement.
    kept = bench.step(along(hand, UP, 0.001), squeeze=1.0, permitted=False)
    assert kept.engaged
    assert_both_at(kept, shifted(HOME, (0.0, 0.0, 0.001)), PATH_TOL)


def test_the_stopped_state_ends_an_engagement():
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    lifted = shifted(HOME, (0.0, 0.0, 0.01))
    assert bench.events.execution_state is ExecutionState.RUNNING

    # Stopped, with the grip still squeezed and the hand moving on.
    hand = along(HAND, UP, 0.03)
    stopped = bench.step(hand, squeeze=1.0, stopped=True)
    assert bench.events.execution_state is ExecutionState.STOPPED
    assert not stopped.engaged
    assert_both_at(stopped, lifted, PATH_TOL)
    for _ in range(3):
        assert not bench.step(hand, squeeze=1.0, stopped=True).engaged

    # The clutch alone engages again on a squeeze that was never released,
    # from the home. The new-squeeze rule is the logic's, not the graph's.
    again = bench.step(hand, squeeze=1.0)
    assert again.engaged
    assert_both_at(again, HOME)


def test_a_squeeze_engages_nothing_while_stopped():
    bench = Bench()
    bench.step(HAND, stopped=True)
    for i in range(1, 4):
        step = bench.step(along(HAND, UP, 0.01 * i), squeeze=1.0, stopped=True)
        assert not step.engaged
        assert_both_at(step, HOME)


@pytest.mark.parametrize("lost", ["not active", "pose not valid"])
def test_a_missing_controller_disengages_and_holds(lost):
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    lifted = shifted(HOME, (0.0, 0.0, 0.01))

    if lost == "not active":
        gone = bench.step(None)
        assert (gone.squeeze, gone.trigger, gone.a, gone.b) == (0.0, 0.0, False, False)
    else:
        gone = bench.step(along(HAND, UP, 0.05), squeeze=1.0, valid=False)
    assert not gone.present and gone.hand is None
    assert not gone.engaged
    assert_both_at(gone, lifted, PATH_TOL)

    # The data come back with the grip still squeezed and the hand 4 cm
    # further: the targets start from the home again, with no jump.
    new_home = shifted(HOME, (0.0, 0.0, 0.012))
    bench.graph.set_home(new_home)
    back = bench.step(along(HAND, UP, 0.05), squeeze=1.0)
    assert back.present and back.engaged
    assert_both_at(back, new_home)


# The rate limiter


def test_a_fast_hand_is_rate_limited():
    settings = RetargetSettings()
    bench = Bench(settings)
    bench.engage()
    most = settings.max_linear_velocity * PASS_S  # 2 mm per pass
    # 1 cm per pass is 0.5 m/s: above the limit, below the rejection.
    for i in range(1, 6):
        step = bench.step(along(HAND, UP, 0.01 * i), squeeze=1.0)
        assert step.engaged
        assert step.limiter == "clamped"
        np.testing.assert_allclose(
            step.commanded_target, shifted(HOME, (0.0, 0.0, 0.01 * i)), atol=PATH_TOL
        )
        np.testing.assert_allclose(
            step.ee_target, shifted(HOME, (0.0, 0.0, most * i)), atol=PATH_TOL
        )

    # 10 cm in one pass is 5 m/s: the limited target does not move at all.
    before = step.ee_target
    hand = along(HAND, UP, 0.15)
    thrown = bench.step(hand, squeeze=1.0)
    assert thrown.limiter == "rejected"
    np.testing.assert_allclose(thrown.ee_target, before, atol=POSE_TOL)
    np.testing.assert_allclose(
        thrown.commanded_target, shifted(HOME, (0.0, 0.0, 0.15)), atol=PATH_TOL
    )

    # A turn of 0.05 rad per pass is 2.5 rad/s: limited to 0.4 rad/s.
    bench = Bench(settings)
    bench.engage()
    turned = bench.step(HAND, quat(Y, 0.05), squeeze=1.0)
    assert turned.limiter == "clamped"
    assert angle_between(HOME, turned.commanded_target) == pytest.approx(0.05, abs=1e-5)
    assert angle_between(HOME, turned.ee_target) == pytest.approx(
        settings.max_angular_velocity * PASS_S, abs=1e-5
    )


def test_the_rate_limiters_speeds_are_the_settings():
    settings = RetargetSettings(
        max_linear_velocity=0.01,
        reject_linear_velocity=None,
        reject_angular_velocity=None,
    )
    bench = Bench(settings)
    bench.engage()
    # 10 cm in one pass, and no rejection is set: clamped to 0.2 mm.
    step = bench.step(along(HAND, UP, 0.10), squeeze=1.0)
    assert step.limiter == "clamped"
    np.testing.assert_allclose(
        step.ee_target, shifted(HOME, (0.0, 0.0, 0.01 * PASS_S)), atol=POSE_TOL
    )
    np.testing.assert_allclose(
        step.commanded_target, shifted(HOME, (0.0, 0.0, 0.10)), atol=PATH_TOL
    )


def test_with_the_rig_file_the_target_goes_25_cm_and_turns_1_rad_in_a_second():
    from isaaccapture_examples.agentic_remote.human import __main__ as program

    configs = Path(program.__file__).resolve().parents[4] / "configs"
    settings = program.load_human_settings(configs / "human_rebot.yaml").retarget
    bounds = (settings.max_linear_velocity, settings.max_angular_velocity)
    rejects = (settings.reject_linear_velocity, settings.reject_angular_velocity)
    assert (bounds, rejects) == ((0.25, 1.0), (2.0, 10.0))
    bench = Bench(settings)
    bench.engage()
    # A hand at 0.15 m/s, 3 mm a pass, is under the bound: the target is at it.
    step = bench.step(along(HAND, UP, 0.003), squeeze=1.0)
    assert step.limiter == "passed"
    np.testing.assert_allclose(
        step.ee_target, shifted(HOME, (0.0, 0.0, 0.003)), atol=PATH_TOL
    )
    # One at 1 m/s, 2 cm a pass, is ahead: the target comes at 5 mm a pass.
    for i in range(1, 6):
        step = bench.step(along(HAND, UP, 0.003 + 0.02 * i), squeeze=1.0)
        assert step.engaged and step.limiter == "clamped"
        np.testing.assert_allclose(
            step.ee_target, shifted(HOME, (0.0, 0.0, 0.003 + 0.005 * i)), atol=PATH_TOL
        )
    ahead = step.commanded_target[2, 3] - step.ee_target[2, 3]
    assert ahead == pytest.approx(5 * 0.015, abs=PATH_TOL)
    # The hand stops: the target goes on to it at that speed and stands there.
    hand = along(HAND, UP, 0.103)
    for i in range(1, 18):
        step = bench.step(hand, squeeze=1.0)
        gone = min(0.103, 0.028 + 0.005 * i)
        np.testing.assert_allclose(
            step.ee_target, shifted(HOME, (0.0, 0.0, gone)), atol=PATH_TOL
        )
    assert step.limiter == "passed"
    # 5 cm in one pass is 2.5 m/s: over the speed that is not followed at all.
    thrown = bench.step(along(HAND, UP, 0.153), squeeze=1.0)
    assert thrown.limiter == "rejected"
    np.testing.assert_allclose(thrown.ee_target, step.ee_target, atol=POSE_TOL)

    # A turn of 0.05 rad a pass is 2.5 rad/s: the target turns 0.02 rad a pass.
    bench = Bench(settings)
    bench.engage()
    turned = bench.step(HAND, quat(Y, 0.05), squeeze=1.0)
    assert turned.limiter == "clamped"
    assert angle_between(HOME, turned.ee_target) == pytest.approx(0.02, abs=1e-5)


@pytest.mark.parametrize("late_ms", [40, 100, 1000])
def test_a_late_pass_is_given_the_motion_of_one_pass(late_ms):
    settings = RetargetSettings()
    assert PASS_S == LIMITER_PASS_S
    most = settings.max_linear_velocity * PASS_S  # 2 mm
    most_turn = settings.max_angular_velocity * PASS_S  # 0.008 rad
    bench = Bench(settings)
    bench.engage()

    # Since the pass before, the hand rose 3 cm and turned 0.1 rad: more than
    # one pass allows, and less than the rate limiter rejects in one pass.
    hand, turned = along(HAND, UP, 0.03), quat(Y, 0.1)
    late = bench.step(hand, turned, squeeze=1.0, pass_ns=late_ms * 1_000_000)
    assert late.engaged
    assert late.limiter == "clamped"
    np.testing.assert_allclose(
        late.commanded_target[:3, 3], HOME[:3, 3] + (0.0, 0.0, 0.03), atol=PATH_TOL
    )
    assert angle_between(HOME, late.commanded_target) == pytest.approx(0.1, abs=1e-5)
    np.testing.assert_allclose(
        late.ee_target[:3, 3], HOME[:3, 3] + (0.0, 0.0, most), atol=PATH_TOL
    )
    assert angle_between(HOME, late.ee_target) == pytest.approx(most_turn, abs=1e-5)

    # The lost time is not made up on the pass that follows.
    follows = bench.step(hand, turned, squeeze=1.0)
    assert follows.limiter == "clamped"
    np.testing.assert_allclose(
        follows.commanded_target, late.commanded_target, atol=POSE_TOL
    )
    np.testing.assert_allclose(
        follows.ee_target[:3, 3], HOME[:3, 3] + (0.0, 0.0, 2 * most), atol=PATH_TOL
    )
    assert angle_between(HOME, follows.ee_target) == pytest.approx(
        2 * most_turn, abs=1e-5
    )


def test_a_short_pass_is_given_its_own_time_and_a_stopped_clock_one_pass():
    settings = RetargetSettings()
    most = settings.max_linear_velocity * PASS_S  # 2 mm
    # The hand rose 1.5 cm: at most 1.5 m/s, over 10 ms, below the rejection.
    hand = along(HAND, UP, 0.015)
    # A short pass comes less than 20 ms after the one before. Down to 10 ms,
    # half a pass, it counts for its own time: 15 ms is three quarters of a
    # pass. Less than 10 ms is the next test's. 0 ms, a clock that did not
    # move since the pass before: a whole pass.
    for pass_ms, rose in ((15, 0.75 * most), (10, most / 2), (0, most)):
        bench = Bench(settings)
        bench.engage()
        step = bench.step(hand, squeeze=1.0, pass_ns=pass_ms * 1_000_000)
        assert step.limiter == "clamped", pass_ms
        np.testing.assert_allclose(
            step.ee_target,
            shifted(HOME, (0.0, 0.0, rose)),
            atol=PATH_TOL,
            err_msg=f"{pass_ms} ms",
        )
        np.testing.assert_allclose(
            step.commanded_target,
            shifted(HOME, (0.0, 0.0, 0.015)),
            atol=PATH_TOL,
            err_msg=f"{pass_ms} ms",
        )


@pytest.mark.parametrize("moved", ["the hand rose", "the hand turned"])
def test_a_pass_right_after_the_one_before_is_not_rejected(moved):
    settings = RetargetSettings()
    counted_s = 0.010  # the least time a pass counts for: half a pass
    bench = Bench(settings)
    bench.engage()

    # 1 ms after the pass before, with one step of the controller data in
    # between. They come about every 11 ms, so 4.4 mm is a hand at 0.4 m/s and
    # 0.011 rad a turn at 1 rad/s. Over 1 ms they would read as 4.4 m/s and
    # 11 rad/s, above the rejection.
    if moved == "the hand rose":
        step = bench.step(along(HAND, UP, 0.0044), squeeze=1.0, pass_ns=1_000_000)
        commanded = shifted(HOME, (0.0, 0.0, 0.0044))
        limited = shifted(HOME, (0.0, 0.0, settings.max_linear_velocity * counted_s))
    else:
        # About the stage's vertical axis: the base's vertical axis, on the spot.
        step = bench.step(HAND, quat(Y, 0.011), squeeze=1.0, pass_ns=1_000_000)
        commanded = pose(HOME[:3, 3], turn(Z, 0.011) @ HOME[:3, :3])
        most_turn = settings.max_angular_velocity * counted_s  # 0.004 rad
        limited = pose(HOME[:3, 3], turn(Z, most_turn) @ HOME[:3, :3])

    assert step.engaged
    assert step.limiter == "clamped"
    np.testing.assert_allclose(step.commanded_target, commanded, atol=PATH_TOL)
    np.testing.assert_allclose(step.ee_target, limited, atol=PATH_TOL)


@pytest.mark.apai
@pytest.mark.parametrize("late_ms", [20, 100])
def test_a_late_pass_at_both_limiter_speeds_is_no_jump_of_the_ik(late_ms):
    """The graph with the solver of human/ik.py, on APAI's mock profile.

    Inverse kinematics (IK) goes from a pose of the gripper tip to joint
    values; a jump is a solution far from the joint values of the pass before.
    """
    from isaaccapture_examples.agentic_remote.human.ik import Solver

    settings = RetargetSettings()
    most = settings.max_linear_velocity * PASS_S
    most_turn = settings.max_angular_velocity * PASS_S
    to_base = np.asarray(settings.controller_to_base)
    solver = Solver.from_apai("mock")
    for start in (RAISED_Q, MID_Q):
        home = solver.fk(start)
        for direction in (UP, AWAY, LEFT, DOWN, TOWARD, RIGHT):
            bench = Bench(settings, home)
            before = bench.engage().ee_target
            q, hand, angle = start, HAND, 0.0
            for i in range(1, 21):
                pass_ns = late_ms * 1_000_000 if i == 10 else PASS_NS
                # 0.11 m/s and 0.5 rad/s: just above both speeds of the limiter.
                hand = along(hand, direction, 0.11 * pass_ns * 1e-9)
                angle += 0.5 * pass_ns * 1e-9
                step = bench.step(hand, quat(Y, angle), squeeze=1.0, pass_ns=pass_ns)
                where = (start, direction, i)
                assert step.limiter == "clamped", where
                # The clutch's target is where the hand is; the limited target
                # moved by one pass at most.
                np.testing.assert_allclose(
                    step.commanded_target[:3, 3],
                    home[:3, 3] + to_base @ np.subtract(hand, HAND),
                    atol=PATH_TOL,
                    err_msg=str(where),
                )
                moved = np.linalg.norm(step.ee_target[:3, 3] - before[:3, 3])
                assert moved <= most + 1e-6, where
                assert angle_between(before, step.ee_target) <= most_turn + 1e-4, where
                before = step.ee_target
                solved = solver.solve(step.ee_target, q)
                assert solved.problem == "", (*where, solved.problem)
                q = solved.q


# The controller's own values


def test_the_buttons_the_squeeze_and_the_trigger_are_read():
    bench = Bench()
    idle = bench.step(HAND)
    assert (idle.a, idle.b, idle.squeeze, idle.trigger) == (False, False, 0.0, 0.0)

    a = bench.step(HAND, a=True)
    assert (a.a, a.b) == (True, False)
    b = bench.step(HAND, b=True)
    assert (b.a, b.b) == (False, True)

    grip = bench.step(HAND, squeeze=0.75)
    assert (grip.squeeze, grip.trigger) == (0.75, 0.0)
    assert grip.engaged
    trigger = bench.step(HAND, trigger=0.25)
    assert (trigger.squeeze, trigger.trigger) == (0.0, 0.25)
    assert not trigger.engaged  # the trigger moves nothing


def test_the_buttons_are_read_while_the_controllers_pose_is_not_valid():
    bench = Bench()
    bench.step(HAND)
    # An active controller that is not tracked: its buttons still count.
    step = bench.step(HAND, valid=False, a=True, b=True, squeeze=0.75, trigger=0.25)
    assert not step.present and step.hand is None
    assert (step.a, step.b) == (True, True)
    assert (step.squeeze, step.trigger) == (0.75, 0.25)
    assert not step.engaged  # a squeeze without a pose moves nothing
    assert_both_at(step, HOME)


# A squeeze or a trigger that is not a finite number, the other value finite
# and the grip squeezed; and whether the clutch itself stays engaged on it: a
# squeeze of NaN or of -inf is not above the threshold.
NOT_FINITE = [
    ({"squeeze": math.nan, "trigger": 0.25}, False),
    ({"squeeze": math.inf, "trigger": 0.25}, True),
    ({"squeeze": -math.inf, "trigger": 0.25}, False),
    ({"squeeze": 1.0, "trigger": math.nan}, True),
    ({"squeeze": 1.0, "trigger": math.inf}, True),
    ({"squeeze": 1.0, "trigger": -math.inf}, True),
]
NOT_FINITE_IDS = [
    "squeeze nan",
    "squeeze inf",
    "squeeze -inf",
    "trigger nan",
    "trigger inf",
    "trigger -inf",
]


def assert_no_controller_data(step: GraphStep) -> None:
    """Nothing of the controller but its buttons, and no target."""
    assert step.present is False
    assert step.engaged is False
    assert (step.squeeze, step.trigger) == (0.0, 0.0)
    assert step.hand is None
    assert step.ee_target is None
    assert step.commanded_target is None


@pytest.mark.parametrize(
    "values", [values for values, _ in NOT_FINITE], ids=NOT_FINITE_IDS
)
def test_a_squeeze_or_trigger_that_is_not_finite_is_no_controller_data(values):
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    hand = along(HAND, UP, 0.011)

    # Right B is the person's STOP: the buttons are still read.
    for a, b in ((False, True), (True, False), (False, False)):
        step = bench.step(hand, a=a, b=b, **values)
        assert_no_controller_data(step)
        assert (step.a, step.b) == (a, b)

    # The same while the controller's pose is not valid.
    step = bench.step(hand, valid=False, a=True, b=True, **values)
    assert_no_controller_data(step)
    assert (step.a, step.b) == (True, True)


@pytest.mark.parametrize("values, held", NOT_FINITE, ids=NOT_FINITE_IDS)
def test_the_pass_after_a_value_that_is_not_finite_is_as_the_clutch_itself_is(
    values, held
):
    """read() changes nothing in the clutch; held: it stayed engaged itself."""
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    assert_no_controller_data(bench.step(along(HAND, UP, 0.011), **values))

    # Finite again, the squeeze above the threshold, the hand 1 mm further,
    # and no home set in between.
    back = bench.step(along(HAND, UP, 0.012), squeeze=1.0)
    assert back.present and back.engaged
    if held:
        # No reset and no home: both targets go on from where the clutch was,
        # with the hand's motion over the pass without data in them.
        assert not bench.events.reset
        assert back.limiter == "passed"
        assert_both_at(back, shifted(HOME, (0.0, 0.0, 0.012)), PATH_TOL)
    else:
        # The clutch let go on the pass without data: a new engagement, at
        # the home.
        assert bench.events.reset
        assert back.limiter == "latched"
        assert_both_at(back, HOME)


@pytest.mark.parametrize("values, held", NOT_FINITE, ids=NOT_FINITE_IDS)
def test_a_home_set_after_a_value_that_is_not_finite_starts_a_new_engagement(
    values, held
):
    bench = Bench()
    bench.engage()
    bench.walk(HAND, UP, 0.01, 10)
    assert_no_controller_data(bench.step(along(HAND, UP, 0.011), **values))
    # The pass was read as not engaged. The next one would carry a reset only
    # where the clutch itself let go as well.
    assert bench.graph.events(stopped=False).reset is (not held)

    # A home set after a pass read as not engaged, also while the clutch itself
    # still holds the hand: the clutch lets go, and the next pass engages there.
    new_home = shifted(HOME, (0.0, 0.0, 0.20))
    bench.graph.set_home(new_home)
    assert bench.graph.events(stopped=False).reset is True
    back = bench.step(along(HAND, UP, 0.012), squeeze=1.0)
    assert bench.events.reset
    assert back.present and back.engaged
    assert back.limiter == "latched"
    assert_both_at(back, new_home)


def test_the_squeeze_engages_above_the_threshold_of_the_settings():
    bench = Bench(RetargetSettings(squeeze_threshold=0.8))
    bench.step(HAND)
    assert not bench.step(HAND, squeeze=0.75).engaged
    assert bench.step(HAND, squeeze=0.875).engaged


def test_the_hand_is_the_controllers_position_before_the_rotation():
    bench = Bench()
    bench.engage()
    step = bench.walk(HAND, UP, 0.01, 10)
    hand = along(HAND, UP, 0.01)
    assert step.hand is not None
    assert step.hand.shape == (3,) and step.hand.dtype == np.float64
    np.testing.assert_allclose(step.hand, hand, atol=POSE_TOL)  # stage: y up
    assert_both_at(step, shifted(HOME, (0.0, 0.0, 0.01)), PATH_TOL)  # base: z up

    gone = bench.step(None)
    assert gone.hand is None and not gone.present


def test_with_the_aim_pose_the_hand_and_the_clutch_use_the_aim_position():
    bench = Bench(RetargetSettings(controller_pose="aim"))
    aim = (0.12, 1.25, -0.4)
    bench.step(HAND, aim=aim)
    bench.step(HAND, aim=aim, squeeze=1.0)
    # The grip pose stands still; the aim pose rises by 5 mm.
    for i in range(1, 6):
        step = bench.step(HAND, aim=along(aim, UP, 0.001 * i), squeeze=1.0)
    np.testing.assert_allclose(step.hand, along(aim, UP, 0.005), atol=POSE_TOL)
    assert_both_at(step, shifted(HOME, (0.0, 0.0, 0.005)), PATH_TOL)

    lost = bench.step(HAND, aim=aim, aim_valid=False, squeeze=1.0)
    assert not lost.present and lost.hand is None and not lost.engaged


# What a controller feed needs of the graph


def test_the_graph_has_three_leaves_and_a_step_needs_all_of_them():
    graph = HumanGraph(RetargetSettings())
    leaves = {leaf.name for leaf in graph.pipeline.get_leaf_nodes()}
    assert leaves == {HumanGraph.LEAF, TO_BASE_LEAF, PERMITTED_LEAF}
    assert HumanGraph.LEAF == "controllers"

    external = graph.external_inputs(True)
    assert set(external) == {TO_BASE_LEAF, PERMITTED_LEAF}
    to_base = np.asarray(external[TO_BASE_LEAF]["value"][0])
    assert to_base.shape == (4, 4) and to_base.dtype == np.float32
    expected = np.eye(4)
    expected[:3, :3] = RetargetSettings().controller_to_base
    np.testing.assert_array_equal(to_base, expected)
    assert external[PERMITTED_LEAF]["value"][0] is True
    assert graph.external_inputs(False)[PERMITTED_LEAF]["value"][0] is False

    controllers = graph.controller_inputs(snapshot(HAND))
    assert set(controllers) == {HumanGraph.LEAF}
    names = set(controllers[HumanGraph.LEAF])
    assert names == {"deviceio_controller_left", "deviceio_controller_right"}
    assert controllers[HumanGraph.LEAF]["deviceio_controller_left"][0] is None

    context = ComputeContext(
        graph_time=GraphTime(PASS_NS, PASS_NS),
        execution_events=graph.events(stopped=False),
    )
    for missing in (TO_BASE_LEAF, PERMITTED_LEAF, HumanGraph.LEAF):
        inputs = {**controllers, **external}
        del inputs[missing]
        with pytest.raises(ValueError):
            graph.pipeline.execute_pipeline(inputs, context=context)
    assert graph.pipeline.execute_pipeline({**controllers, **external}, context=context)


def test_a_pose_the_graph_has_not_written_is_read_as_none():
    bench = Bench()
    outputs = dict(
        bench.graph.pipeline.execute_pipeline(
            {
                **bench.graph.controller_inputs(snapshot(HAND)),
                **bench.graph.external_inputs(True),
            },
            context=ComputeContext(
                graph_time=GraphTime(PASS_NS, PASS_NS),
                execution_events=bench.graph.events(stopped=False),
            ),
        )
    )
    assert_both_at(bench.graph.read(outputs), HOME)
    # The rate limiter writes no pose before it has seen a usable one.
    outputs[EE_KEY] = TensorGroup(bench.graph.pipeline.output_types()[EE_KEY])
    step = bench.graph.read(outputs)
    assert step.ee_target is None
    np.testing.assert_allclose(step.commanded_target, HOME, atol=POSE_TOL)
