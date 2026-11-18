# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types the modules of the human program share: settings, results and ports.

Only numpy and the standard library are imported here. The clutch lets the arm
follow the hand only while the grip is squeezed; an engagement is one such
stretch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import numpy as np

# The most time the rate limiter counts for one pass, s: the period of the
# human program's passes. A late pass moves the target no further than a pass
# on time, so what follows the graph never gets a larger step. The least it
# counts is half of this: a rates.loop_hz (the human program's passes per
# second) above 100 would raise both hand speed limits in proportion.
LIMITER_PASS_S = 0.02

# What HUMAN_SETPOINT.note and the report line say while the arm follows a
# hand that is AHEAD_NOTE_CM or more ahead of the tip of the last joint target.
NOTE_CATCHING_UP = "hand {cm} cm ahead: the arm is catching up"
AHEAD_NOTE_CM = 5
# What they say while the gripper tip stands at a wall of the robot app's box
# for it with the hand beyond that wall; walls names each such wall.
NOTE_AT_LIMIT = "at the workspace limit: {walls}"
# The walls by the robot's base axes, x forward, y left, z up: (low, high).
WALL_NAMES = (("back", "front"), ("right", "left"), ("bottom", "top"))
# What they say while the joint target is held back for a robot app whose own
# joint target comes on slower than its speed clamp allows.
NOTE_THOR_BEHIND = "the robot app is behind: the arm follows slower"


@dataclass(frozen=True)
class RetargetSettings:
    """The `retarget` block of human.yaml."""

    controller_pose: str = "grip"  # "grip" or "aim"
    position_scale: float = 1.0
    squeeze_threshold: float = 0.5
    max_linear_velocity: float = 0.10  # m/s
    max_angular_velocity: float = 0.4  # rad/s
    reject_linear_velocity: float | None = 2.0
    reject_angular_velocity: float | None = 10.0
    # The rotation from the stage space the controller poses are given in to
    # the robot's base axes, by rows.
    controller_to_base: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    # A turn about the stage's vertical axis, applied first.
    stage_yaw_deg: float = 0.0


@dataclass(frozen=True)
class IkSettings:
    """The `ik` block of human.yaml."""

    apai_arm: str = "mock"
    max_iter: int = 50
    tol: float = 1e-4
    max_jump_rad: float = 0.05
    max_ahead_rad: float = 0.08
    max_ahead_m: float = 0.5
    max_failures: int = 5


@dataclass(frozen=True)
class TeleopSettings:
    """The `teleop` block of human.yaml."""

    endpoint: str = "127.0.0.1:47600"
    tick_rate_hz: float = 90.0
    max_tick_age_s: float = 0.15
    connect_timeout_s: float = 3.0
    retry_s: float = 5.0  # at least 2.0
    attempts: int = 720  # failed attempts in a row


@dataclass(frozen=True)
class HumanSettings:
    """human.yaml as a whole."""

    socket_path: str = ""
    log_level: str = "info"
    teleop: TeleopSettings = TeleopSettings()
    retarget: RetargetSettings = RetargetSettings()
    ik: IkSettings = IkSettings()
    hold_s: float = 1.0  # buttons.hold_s
    loop_hz: float = 50.0  # rates.loop_hz
    idle_hz: float = 20.0  # rates.idle_hz
    state_max_age_s: float = 0.2  # state.max_age_s


@dataclass(frozen=True, eq=False)
class GraphStep:
    """What one pass of the retargeting graph gives.

    Two steps are equal only when they are the same object: compare their
    fields, the arrays with np.array_equal.
    """

    # The right controller is active, its pose is valid and its squeeze and
    # trigger are finite numbers.
    present: bool
    squeeze: float
    trigger: float  # read for the report line only; it moves nothing
    a: bool  # right A (primary_click)
    b: bool  # right B (secondary_click)
    # SO101ClutchRetargeter.is_engaged after this pass; False on a pass whose
    # squeeze or trigger is not a finite number.
    engaged: bool
    # 4x4 float64, the gripper tip in the robot's base axes, after the rate
    # limiter. None on a pass whose squeeze or trigger is not a finite number.
    ee_target: np.ndarray | None
    # 4x4 float64, the clutch's output, before the rate limiter. None on such
    # a pass too.
    commanded_target: np.ndarray | None
    limiter: str  # the rate limiter's disposition, in lower case
    # 3 float64: the right controller's position in the stage space, in metres,
    # before controller_to_base (the pose that controller_pose names). None
    # when not present.
    hand: np.ndarray | None = None


@dataclass(frozen=True)
class IkStep:
    """One inverse kinematics solve."""

    q: tuple[float, ...] | None
    problem: str  # "" with q; else why there is none


class FeedStep(NamedTuple):
    """What a controller feed gives for one pass."""

    outputs: Mapping[str, Any] | None  # the graph's outputs; None: none this pass
    sample_age_s: float | None  # None: stale or absent
    stage: str  # one of HUMAN_STAGES
    note: str = ""  # the feed's own text, such as a connection error


class ControllerFeed(Protocol):
    """Where the human program gets controller data."""

    source: str  # HUMAN_SOURCE_TELEOP or HUMAN_SOURCE_SYNTHETIC

    def open(self) -> None:
        """Start the feed."""

    def step(
        self, external_inputs: dict[str, Any], events: Any, now_mono: float
    ) -> FeedStep:
        """Run one pass of the graph on the newest controller data."""

    def close(self) -> None:
        """End the feed."""


class GraphPort(Protocol):
    """What HumanLogic needs of HumanGraph."""

    def set_home(self, base_T_ee: np.ndarray) -> None:
        """Set the pose the next engagement starts from."""

    def read(self, outputs: Mapping[str, Any]) -> GraphStep:
        """The graph's outputs of one pass as a GraphStep."""


class SolverPort(Protocol):
    """What HumanLogic needs of Solver."""

    n_joints: int

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """The 4x4 pose of the gripper tip at q."""

    def inside(self, q: Sequence[float]) -> tuple[float, ...]:
        """q moved inside the margin the solver keeps from the joint limits."""

    def solve(self, base_T_ee: np.ndarray, q_seed: Sequence[float]) -> IkStep:
        """Joint values for a pose of the gripper tip, seeded with q_seed."""
