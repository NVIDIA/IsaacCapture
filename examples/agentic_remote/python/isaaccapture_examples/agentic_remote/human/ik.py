# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The inverse kinematics of the human program: APAI's, without random restarts.

Inverse kinematics (IK) goes from a pose of the gripper tip to joint values;
forward kinematics (FK) goes the other way. A jump is a solution far from the
joint values the solve started from, its seed. APAI is imported only when a
solver is built from one of its arm profiles.

A pose that has no solution lies out of reach. APAI's solve then gives the
joint values of its last iteration, which near the edge of reach can lie
anywhere, and one error for position and turn together: neither is used.
Solver.nearest goes one step from the seed toward such a pose on the arm
model's FK alone, the tip's position first and the wrist's turn second.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .types import IkStep

# The margin APAI's Kinematics.ik keeps from the model's joint limits, rad.
IK_MARGIN_RAD = 0.025

# The src folder of an APAI checkout, as the other programs name it.
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"

# Solver.nearest. What 1 rad of the wrist's turn weighs against 1 m of the
# tip's way, m: the position comes first. Computed on the reBot arm's model
# for a hand 20 to 50 cm beyond its reach: the wrist then gives way by up to
# 0.27 rad from the hand's orientation. With 0.05 the tip comes 4 to 22 cm
# nearer and the wrist gives way by up to 1.03 rad.
NEAR_TURN_WEIGHT_M = 0.1
# The damping of its least-squares step, m per rad: at the edge of reach a
# way of the tip that no joint makes moves no joint far.
NEAR_DAMPING = 0.02
# The joint step its Jacobian, the tip's motion per motion of each joint, is
# measured with on the FK, rad.
NEAR_PROBE_RAD = 1e-6
# The shares of its step that are tried, and the least by which one must
# bring the pose nearer, m with a turn weighed as above. Less is no progress:
# the arm has come as near as it can.
NEAR_SHARES = (1.0, 0.5, 0.25, 0.125)
NEAR_GAIN_M = 2e-6
# How its problem text begins.
NO_NEARER = "no nearer pose"


class Solver:
    """FK and IK on APAI's arm model, one solve per pose, seeded by the caller.

    kinematics is an APAI Kinematics: n, fk(), clamp() and ik().
    """

    def __init__(
        self,
        kinematics: Any,
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
        max_jump_rad: float = 0.05,
    ) -> None:
        self._kinematics = kinematics
        self.n_joints = int(kinematics.n)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_jump_rad = float(max_jump_rad)

    @classmethod
    def from_apai(cls, arm: str, **settings: Any) -> Solver:
        """A solver on the kinematics of an APAI arm profile, such as "mock".

        settings are the keywords of Solver. An APAI that cannot be imported
        raises ImportError.
        """
        source = os.environ.get(APAI_SRC_ENV)
        if source and source not in sys.path:
            sys.path.insert(0, source)
        # The loader the Thor side uses, so both sides solve on one model.
        from ..robot_app.apai_backend import load_apai_profile

        return cls(load_apai_profile(arm).kinematics, **settings)

    def _joints(self, q: Sequence[float]) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.shape != (self.n_joints,):
            raise ValueError(
                f"the arm model has {self.n_joints} joints, got {values.size} values"
            )
        return values

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """The 4x4 pose of the gripper tip at q, in the robot's base axes."""
        return np.array(self._kinematics.fk(self._joints(q)), dtype=np.float64)

    def inside(self, q: Sequence[float]) -> tuple[float, ...]:
        """q moved inside the margin the IK keeps from the joint limits.

        It is the nearest pose the IK can give: a solve from a pose on a joint
        stop, as the rest pose is, never comes back to that pose.
        """
        clamped = self._kinematics.clamp(self._joints(q), IK_MARGIN_RAD)
        return tuple(float(value) for value in clamped)

    def solve(
        self,
        base_T_ee: np.ndarray,  # noqa: N803
        q_seed: Sequence[float],
    ) -> IkStep:
        """Joint values for a pose of the gripper tip, starting from q_seed only.

        No joints come back from a solve that did not converge or that
        jumped; the problem says which.
        """
        seed = self._joints(q_seed)
        result = self._kinematics.ik(
            np.asarray(base_T_ee, dtype=np.float64),
            q_init=seed,
            retries=0,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        if not result.success:
            return IkStep(
                None,
                f"no IK solution (error {float(result.error):.3g} after "
                f"{int(result.iterations)} iterations)",
            )
        q = np.asarray(result.q, dtype=np.float64).reshape(-1)
        jumps = np.abs(q - seed)
        index = int(np.argmax(jumps))
        # Written so that a value that is not a number counts as a jump.
        if not jumps[index] <= self.max_jump_rad:
            return IkStep(
                None,
                f"IK jump of {float(jumps[index]):.3f} rad on joint index {index}, "
                f"the limit is {self.max_jump_rad:g}",
            )
        return IkStep(tuple(float(value) for value in q), "")

    def nearest(
        self,
        base_T_ee: np.ndarray,  # noqa: N803
        q_seed: Sequence[float],
        low: Sequence[float] | None = None,
        high: Sequence[float] | None = None,
    ) -> IkStep:
        """Joint values one step from q_seed toward a pose with no solution.

        One damped least-squares step on the FK: the tip's position weighs
        1, the wrist's turn NEAR_TURN_WEIGHT_M. No joint leaves the margin
        the IK keeps from the joint limits, nor low..high, rad, when given;
        one outside at q_seed goes no further out. No joint moves further
        than max_jump_rad. No joints come back unless the pose at them is
        nearer by NEAR_GAIN_M than the pose at q_seed, and their tip no
        further from the pose's position: the wrist's turn costs the tip
        nothing.
        """
        seed = self._joints(q_seed)
        pose = np.asarray(base_T_ee, dtype=np.float64)
        bounds = self._kinematics.joint_limits
        floor = np.asarray(bounds[0], dtype=np.float64) + IK_MARGIN_RAD
        ceiling = np.asarray(bounds[1], dtype=np.float64) - IK_MARGIN_RAD
        if low is not None:
            floor = np.maximum(floor, self._joints(low))
        if high is not None:
            ceiling = np.minimum(ceiling, self._joints(high))
        step = nearest_step(self.fk, pose, seed, floor, ceiling, self.max_jump_rad)
        if step is None:
            return IkStep(None, f"{NO_NEARER} than the one of the last joint target")
        return IkStep(tuple(float(value) for value in step), "")


def nearest_step(
    fk: Callable[[Sequence[float]], np.ndarray],
    pose: np.ndarray,
    seed: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    most_rad: float,
) -> np.ndarray | None:
    """The joint values of Solver.nearest on any FK; None: none is nearer.

    low and high bound each joint, rad; most_rad bounds the step of each.
    """
    weights = np.array([1.0, 1.0, 1.0] + [NEAR_TURN_WEIGHT_M] * 3)
    at = fk(seed)
    gap = weights * _gap(at, pose)
    jacobian = np.empty((6, len(seed)))
    for joint in range(len(seed)):
        probe = seed.copy()
        probe[joint] += NEAR_PROBE_RAD
        jacobian[:, joint] = weights * _gap(at, fk(probe)) / NEAR_PROBE_RAD
    # A joint outside its bounds at the seed may come back; none goes out.
    step = _bounded_step(
        jacobian, gap, np.minimum(low - seed, 0.0), np.maximum(high - seed, 0.0)
    )
    far = float(np.max(np.abs(step)))
    if not math.isfinite(far) or far == 0.0:
        return None
    if far > most_rad:
        step *= most_rad / far
    off, away = float(np.linalg.norm(gap)), float(np.linalg.norm(gap[:3]))
    for share in NEAR_SHARES:
        q = seed + share * step
        left = weights * _gap(fk(q), pose)
        nearer = float(np.linalg.norm(left)) <= off - NEAR_GAIN_M
        if nearer and float(np.linalg.norm(left[:3])) <= away + 1e-9:
            return q
    return None


def _bounded_step(
    jacobian: np.ndarray, gap: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """The damped least-squares joint step for gap, each joint in low..high.

    A joint whose step would pass a bound is held at the bound, and the
    others are solved again for what is left of the gap.
    """
    step = np.zeros(jacobian.shape[1])
    free = np.ones(jacobian.shape[1], dtype=bool)
    damped = NEAR_DAMPING**2 * np.eye(jacobian.shape[0])
    while free.any():
        left = gap - jacobian[:, ~free] @ step[~free]
        part = jacobian[:, free]
        step[free] = part.T @ np.linalg.solve(part @ part.T + damped, left)
        over = free & ((step < low) | (step > high))
        if not over.any():
            break
        step[over] = np.clip(step[over], low[over], high[over])
        free &= ~over
    return step


def _gap(at: np.ndarray, to: np.ndarray) -> np.ndarray:
    """The way from one 4x4 pose to another, in the base axes.

    The tip's way, m, then the turn as its axis times its angle, rad. A half
    turn has no shorter way round and counts as no turn.
    """
    turn = to[:3, :3] @ at[:3, :3].T
    twice = np.array(
        [turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]]
    )
    sine = 0.5 * float(np.linalg.norm(twice))
    cosine = 0.5 * (float(np.trace(turn)) - 1.0)
    if sine > 1e-9:
        twice = twice * (math.atan2(sine, cosine) / sine)
    return np.concatenate([to[:3, 3] - at[:3, 3], 0.5 * twice])
