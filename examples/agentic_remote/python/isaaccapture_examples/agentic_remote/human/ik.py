# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The inverse kinematics of the human program: APAI's, without random restarts.

Inverse kinematics (IK) goes from a pose of the gripper tip to joint values;
forward kinematics (FK) goes the other way. A jump is a solution far from the
joint values the solve started from, its seed. APAI is imported only when a
solver is built from one of its arm profiles.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from .types import IkStep

# The margin APAI's Kinematics.ik keeps from the model's joint limits, rad.
IK_MARGIN_RAD = 0.025

# The src folder of an APAI checkout, as the other programs name it.
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"


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
