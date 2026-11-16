# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""human/ik.py: the solver of the human program on APAI's arm model.

Inverse kinematics (IK) goes from a pose of the gripper tip to joint values,
forward kinematics (FK) the other way. The seed is where a solve starts. The
rest pose is all zeros; there the second and third joint lie on their lower
stop. The inside pose of a pose is that pose moved inside the margin the IK
keeps from the joint limits. A pass is one period of the human program, 20 ms.
The tests need an APAI checkout (AGENTIC_REMOTE_APAI_SRC).
"""

from __future__ import annotations

import functools
import inspect
import itertools
import math
import os
import re
import statistics
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.human.ik import (
    APAI_SRC_ENV,
    IK_MARGIN_RAD,
    Solver,
)
from isaaccapture_examples.agentic_remote.human.types import (
    IkSettings,
    IkStep,
    RetargetSettings,
)

pytestmark = pytest.mark.apai

PASS_S = 0.02
REST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RAISED = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
MID = (0.3, 0.8, 1.0, 0.2, -0.4, 0.1)
JUST_ABOVE_REST = (0.0, 0.15, 0.15, 0.0, 0.0, 0.0)
# A pose the solver reaches from RAISED only with a random restart.
ACROSS = (2.12, 1.82, 1.49, 0.95, -1.38, 1.26)
ARMS = ("mock", "rebot_rs")  # both use the model of the reBot RS arm

# The robot's base axes: x forward, y left, z up.
FORWARD = (1.0, 0.0, 0.0)
BACK = (-1.0, 0.0, 0.0)
LEFT = (0.0, 1.0, 0.0)
RIGHT = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)
DIRECTIONS = (FORWARD, BACK, LEFT, RIGHT, UP, DOWN)

NO_SOLUTION = re.compile(r"no IK solution \(error \S+ after 50 iterations\)")


@functools.cache
def kinematics_of(arm: str) -> Any:
    """APAI's Kinematics of an arm profile, loaded once."""
    from isaaccapture_examples.agentic_remote.robot_app.apai_backend import (
        load_apai_profile,
    )

    return load_apai_profile(arm).kinematics


@pytest.fixture
def kinematics() -> Any:
    return kinematics_of("mock")


@pytest.fixture
def solver(kinematics: Any) -> Solver:
    return Solver(kinematics)


class Recording:
    """A kinematics that notes the keywords of every ik() call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def ik(self, target: np.ndarray, **keywords: Any) -> Any:
        self.calls.append(keywords)
        return self._inner.ik(target, **keywords)


def turn(axis: Sequence[float], angle: float) -> np.ndarray:
    """The 3x3 rotation by angle, in rad, about a unit axis."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def moved(
    pose: np.ndarray,
    direction: Sequence[float],
    distance: float,
    axis: Sequence[float] | None = None,
    angle: float = 0.0,
) -> np.ndarray:
    """The pose shifted along a base axis and turned about a base axis."""
    target = pose.copy()
    target[:3, 3] += distance * np.asarray(direction)
    if axis is not None:
        target[:3, :3] = turn(axis, angle) @ pose[:3, :3]
    return target


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far two poses are apart: metres, and rad of rotation."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return (
        float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        math.acos(max(-1.0, min(1.0, cos))),
    )


def step_of(q: Sequence[float], seed: Sequence[float]) -> float:
    """The largest joint difference, rad."""
    return max(abs(a - b) for a, b in zip(q, seed))


def lift(solver: Solver, passes: int, per_pass_m: float) -> list[IkStep]:
    """Raise the target from the inside pose of rest; the seed is the last solution."""
    q = solver.inside(REST)
    start = solver.fk(q)
    steps = []
    for i in range(1, passes + 1):
        step = solver.solve(moved(start, UP, per_pass_m * i), q)
        steps.append(step)
        if step.q is not None:
            q = step.q
    return steps


# The solver on APAI's mock profile


def test_from_apai_builds_the_solver_on_the_profiles_kinematics(monkeypatch):
    source = os.environ[APAI_SRC_ENV]
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != source])
    solver = Solver.from_apai("mock", max_iter=7, tol=1e-3, max_jump_rad=0.2)
    assert sys.path[0] == source  # as the robot app puts it there
    assert (solver.max_iter, solver.tol, solver.max_jump_rad) == (7, 1e-3, 0.2)
    assert solver.n_joints == 6
    np.testing.assert_array_equal(solver.fk(MID), kinematics_of("mock").fk(MID))

    shipped = IkSettings()
    default = Solver.from_apai(shipped.apai_arm)
    assert (default.max_iter, default.tol, default.max_jump_rad) == (
        shipped.max_iter,
        shipped.tol,
        shipped.max_jump_rad,
    )


def test_fk_gives_the_pose_of_the_gripper_tip_in_the_base_axes(solver):
    pose = solver.fk(REST)
    assert pose.shape == (4, 4) and pose.dtype == np.float64
    np.testing.assert_array_equal(pose[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)
    # At rest the gripper tip is in front of the base and above it.
    assert 0.25 < pose[0, 3] < 0.35 and abs(pose[1, 3]) < 1e-6 and pose[2, 3] > 0.15
    # A positive turn of the first joint swings the tip to the right, -y.
    swung = solver.fk((0.3, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert -0.10 < swung[1, 3] < -0.08
    assert swung[2, 3] == pytest.approx(pose[2, 3], abs=1e-9)


@pytest.mark.parametrize("q", [RAISED, MID, JUST_ABOVE_REST])
def test_fk_then_ik_gives_the_pose_back(solver, q):
    target = solver.fk(q)
    # Seeded 0.02 rad away on every joint, alternately above and below.
    seed = [value + 0.02 * (-1) ** i for i, value in enumerate(q)]
    step = solver.solve(target, seed)
    assert step.problem == ""
    assert isinstance(step.q, tuple) and len(step.q) == solver.n_joints
    assert all(type(value) is float for value in step.q)
    metres, rad = pose_error(solver.fk(step.q), target)
    assert metres < 1e-4 and rad < 1e-4  # the solver's tolerance
    assert step_of(step.q, q) < 2e-3


@pytest.mark.parametrize("start", [RAISED, MID])
def test_a_small_step_of_the_target_is_a_small_step_of_the_joints(solver, start):
    pose = solver.fk(start)
    for direction in DIRECTIONS:
        target = moved(pose, direction, 0.002)
        step = solver.solve(target, start)
        assert step.problem == "", direction
        assert 1e-3 < step_of(step.q, start) < 0.02, direction
        metres, rad = pose_error(solver.fk(step.q), target)
        assert metres < 1e-4 and rad < 1e-4, direction


def test_inside_moves_a_pose_off_the_joint_stops_by_the_iks_margin(kinematics, solver):
    # The margin is the one APAI's IK keeps by default.
    margin = inspect.signature(type(kinematics).ik).parameters["limit_margin"]
    assert margin.default == IK_MARGIN_RAD == 0.025

    assert solver.inside(REST) == (0.0, 0.025, 0.025, 0.0, 0.0, 0.0)
    assert solver.inside(RAISED) == RAISED
    low, high = kinematics.joint_limits
    assert solver.inside(high) == pytest.approx(tuple(np.asarray(high) - 0.025))
    assert solver.inside(low) == pytest.approx(tuple(np.asarray(low) + 0.025))
    inside = solver.inside(np.asarray(REST))
    assert isinstance(inside, tuple) and all(type(value) is float for value in inside)


def test_a_pose_with_another_number_of_joints_is_refused(solver):
    with pytest.raises(ValueError, match="6 joints, got 5 values"):
        solver.fk(REST[:5])
    with pytest.raises(ValueError, match="6 joints, got 7 values"):
        solver.inside((*REST, 0.0))
    with pytest.raises(ValueError, match="6 joints, got 5 values"):
        solver.solve(solver.fk(RAISED), RAISED[:5])


# The two refusals


def test_a_target_out_of_reach_has_no_solution_and_gives_no_joints(solver):
    target = moved(solver.fk(RAISED), FORWARD, 2.0)
    step = solver.solve(target, RAISED)
    assert step.q is None
    assert NO_SOLUTION.fullmatch(step.problem), step.problem


def test_a_jump_is_refused_and_gives_no_joints(kinematics, solver):
    # A pose the arm reaches by turning its first joint by 0.1 rad.
    target = solver.fk((0.1, *RAISED[1:]))
    step = solver.solve(target, RAISED)
    assert step.q is None
    assert step.problem == "IK jump of 0.100 rad on joint index 0, the limit is 0.05"

    # The same solve is taken where the limit allows it.
    wide = Solver(kinematics, max_jump_rad=0.2).solve(target, RAISED)
    assert wide.problem == ""
    assert wide.q == pytest.approx((0.1, *RAISED[1:]), abs=1e-3)
    # A step right at the limit is no jump.
    near = solver.solve(solver.fk((0.049, *RAISED[1:])), RAISED)
    assert near.problem == "" and step_of(near.q, RAISED) <= 0.05


def test_no_random_restart_is_used(kinematics):
    recording = Recording(kinematics)
    solver = Solver(recording)
    target = kinematics.fk(np.asarray(ACROSS))

    # APAI's own default finds the pose with one of its four random restarts.
    restarted = kinematics.ik(target, q_init=np.asarray(RAISED), max_iter=50, tol=1e-4)
    assert restarted.success
    # The solver starts from the seed and from nothing else.
    step = solver.solve(target, RAISED)
    assert step.q is None
    assert NO_SOLUTION.fullmatch(step.problem), step.problem

    (call,) = recording.calls
    assert call["retries"] == 0
    assert (call["max_iter"], call["tol"]) == (50, 1e-4)
    np.testing.assert_array_equal(call["q_init"], RAISED)

    other = Solver(recording, max_iter=7, tol=1e-3)
    other.solve(target, MID)
    assert recording.calls[-1]["retries"] == 0
    assert (recording.calls[-1]["max_iter"], recording.calls[-1]["tol"]) == (7, 1e-3)
    np.testing.assert_array_equal(recording.calls[-1]["q_init"], MID)


# The rest pose


def test_the_rest_pose_has_no_solution_and_its_inside_pose_has(solver):
    # The second and third joint lie on their stop, inside the IK's margin.
    step = solver.solve(solver.fk(REST), REST)
    assert step.q is None
    assert NO_SOLUTION.fullmatch(step.problem), step.problem

    inside = solver.inside(REST)
    again = solver.solve(solver.fk(inside), inside)
    assert again.problem == "" and again.q == inside
    # The inside pose is some millimetres from rest at the gripper tip, and a
    # step the jump limit allows.
    metres, _ = pose_error(solver.fk(inside), solver.fk(REST))
    assert 0.004 < metres < 0.008
    assert step_of(inside, REST) == IK_MARGIN_RAD < IkSettings().max_jump_rad


@pytest.mark.parametrize("arm", ARMS)
def test_a_lift_from_the_rest_pose_solves_on_every_pass(arm):
    kinematics = kinematics_of(arm)
    solver = Solver(kinematics)
    steps = lift(solver, 200, 0.0005)
    assert [step.problem for step in steps if step.problem] == []

    low, high = (np.asarray(limit) for limit in kinematics.joint_limits)
    start = solver.fk(solver.inside(REST))
    q = solver.inside(REST)
    for i, step in enumerate(steps, start=1):
        assert step_of(step.q, q) < 0.01, i  # far below the jump limit
        metres, rad = pose_error(solver.fk(step.q), moved(start, UP, 0.0005 * i))
        assert metres < 1e-4 and rad < 1e-4, i
        # Every solution keeps the IK's margin from the joint limits.
        values = np.asarray(step.q)
        assert np.all(values >= low + IK_MARGIN_RAD - 1e-12), i
        assert np.all(values <= high - IK_MARGIN_RAD + 1e-12), i
        q = step.q
    assert solver.fk(q)[2, 3] - start[2, 3] == pytest.approx(0.10, abs=1e-4)


@pytest.mark.parametrize("arm", ARMS)
def test_at_the_rest_pose_forward_back_and_down_have_no_solution(arm):
    solver = Solver(kinematics_of(arm))
    inside = solver.inside(REST)
    start = solver.fk(inside)
    for direction in (FORWARD, BACK, DOWN):
        step = solver.solve(moved(start, direction, 0.002), inside)
        assert step.q is None, direction
        assert NO_SOLUTION.fullmatch(step.problem), (direction, step.problem)
    # Up has one: from rest, lift the gripper first.
    assert solver.solve(moved(start, UP, 0.002), inside).problem == ""


# The rate limiter's speeds against the jump limit


def test_a_pass_at_both_limiter_speeds_is_not_a_jump(solver):
    speeds = RetargetSettings()
    metres = speeds.max_linear_velocity * PASS_S  # 2 mm
    rad = speeds.max_angular_velocity * PASS_S  # 0.008 rad
    assert (metres, rad) == pytest.approx((0.002, 0.008))
    limit = IkSettings().max_jump_rad
    assert solver.max_jump_rad == limit

    above_rest = lift(solver, 200, 0.0005)[-1].q  # 10 cm above rest
    for start in (RAISED, above_rest):
        pose = solver.fk(start)
        for direction, axis in itertools.product(DIRECTIONS, DIRECTIONS):
            step = solver.solve(moved(pose, direction, metres, axis, rad), start)
            assert step.problem == "", (start, direction, axis)
            assert step_of(step.q, start) < limit, (start, direction, axis)

    # At the inside pose of rest two joints are near their stops: such a pass
    # has a solution under the limit or none, and is never a jump.
    inside = solver.inside(REST)
    pose = solver.fk(inside)
    for direction, axis in itertools.product(DIRECTIONS, DIRECTIONS):
        step = solver.solve(moved(pose, direction, metres, axis, rad), inside)
        if step.q is None:
            assert NO_SOLUTION.fullmatch(step.problem), (direction, axis, step.problem)
        else:
            assert step_of(step.q, inside) < limit, (direction, axis)
    # A lift at the limiter's full linear speed has a solution.
    lifted = solver.solve(moved(pose, UP, metres), inside)
    assert lifted.problem == "" and step_of(lifted.q, inside) < limit


def joint_speed_per_hand_speed(
    solver: Solver, q: Sequence[float]
) -> tuple[float, float]:
    """The largest joint speed, rad/s, for 1 m/s and for 1 rad/s at the gripper tip.

    From the pseudo-inverse of the arm's Jacobian at q, over 4000 directions.
    The Jacobian is taken from differences of fk(), not from the IK.
    """
    q = np.asarray(q, dtype=np.float64)
    pose = solver.fk(q)
    columns = []
    for i in range(solver.n_joints):
        nudged = q.copy()
        nudged[i] += 1e-6
        other = solver.fk(nudged)
        spin = other[:3, :3] @ pose[:3, :3].T
        angular = (
            spin[2, 1] - spin[1, 2],
            spin[0, 2] - spin[2, 0],
            spin[1, 0] - spin[0, 1],
        )
        columns.append(
            np.concatenate([other[:3, 3] - pose[:3, 3], np.asarray(angular) / 2]) / 1e-6
        )
    inverse = np.linalg.pinv(np.array(columns).T)
    directions = np.random.default_rng(1).normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return (
        float(np.max(np.abs(inverse[:, :3] @ directions.T))),
        float(np.max(np.abs(inverse[:, 3:] @ directions.T))),
    )


def test_the_solve_time_and_the_joint_speed_per_hand_speed_are_printed(solver):
    def timed(target: np.ndarray, seed: Sequence[float], count: int) -> list[float]:
        times = []
        for _ in range(count):
            began = time.perf_counter()
            solver.solve(target, seed)
            times.append((time.perf_counter() - began) * 1e3)
        return times

    near = moved(solver.fk(RAISED), UP, 0.002)
    far = moved(solver.fk(RAISED), FORWARD, 2.0)
    assert solver.solve(near, RAISED).problem == ""
    assert solver.solve(far, RAISED).q is None
    print()
    for name, target, count in (
        ("a solve that converges", near, 200),
        ("a solve that fails", far, 50),
    ):
        times = timed(target, RAISED, count)
        slow = [value for value in times if value > 5.0]
        print(
            f"{name}: middle {statistics.median(times):.3f} ms, longest "
            f"{max(times):.3f} ms; above 5 ms: {len(slow)} of {count}"
        )

    speeds, limit = RetargetSettings(), IkSettings().max_jump_rad
    poses = {
        "raised": RAISED,
        "mid": MID,
        "just above rest": JUST_ABOVE_REST,
        "10 cm above rest": lift(solver, 200, 0.0005)[-1].q,
        "the inside pose of rest": solver.inside(REST),
    }
    per_pass = {}
    for name, q in poses.items():
        linear, angular = joint_speed_per_hand_speed(solver, q)
        per_pass[name] = (
            speeds.max_linear_velocity * linear + speeds.max_angular_velocity * angular
        ) * PASS_S
        side = "under" if per_pass[name] < limit else "ABOVE"
        print(
            f"{name}: {linear:.2f} rad/s per m/s, {angular:.2f} rad/s per rad/s; "
            f"at both limiter speeds {per_pass[name]:.4f} rad per pass "
            f"({side} the jump limit {limit:g})"
        )
    # At these poses the shipped limiter speeds stay under the jump limit, in
    # every direction. Raising a speed needs the jump limit raised with it.
    for name in ("raised", "mid", "just above rest", "10 cm above rest"):
        assert per_pass[name] < limit, name
