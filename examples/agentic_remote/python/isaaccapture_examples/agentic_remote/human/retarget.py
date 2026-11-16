# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The retargeting graph of the human program: Isaac Capture's nodes, wired.

A retargeter is an Isaac Capture node that turns device data into a command;
a retargeting graph is such nodes connected. This one takes the right
controller, rotates its poses from the stage space (the OpenXR reference space
they are given in: metres, y up) into the robot's base axes (x forward, y
left, z up), and runs the clutch and the rate limiter on them. The clutch lets
the target of the gripper tip follow the hand only while the grip is squeezed;
an engagement is one such stretch. Each engagement starts at the home, the pose
last given to set_home, wherever the hand is. The rate limiter bounds the speed
of that target and how far it moves in one pass.

The graph holds poses only: no arm, no joints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from isaaccapture.retargeters.rate_limiter import (
    EE_POSE_KEY,
    EE_POSE_STATUS_KEY,
    EePoseRateLimiter,
    EePoseRateLimiterDisposition,
    EePoseRateLimiterStatusIndex,
    RateLimiterConfig,
)
from isaaccapture.retargeters.SO101.clutch_retargeter import SO101ClutchRetargeter
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeting_engine.interface import (
    ExecutionEvents,
    ExecutionState,
    OutputCombiner,
    RetargeterIO,
    TensorGroup,
    TensorGroupType,
    ValueInput,
)
from isaaccapture.retargeting_engine.tensor_types import (
    BoolType,
    ControllerInputIndex,
    TransformMatrix,
)

from .types import GraphStep, RetargetSettings

# The two leaves every step must be given besides the controllers.
TO_BASE_LEAF = "controller_to_base"
PERMITTED_LEAF = "engage_permitted"
# The type of the clutch's engage permission, as the clutch spells it.
PERMITTED_TYPE = TensorGroupType(
    SO101ClutchRetargeter.ENGAGE_PERMITTED_INPUT, [BoolType("permitted")]
)

# The outputs of the pipeline; only read() reads them.
COMMANDED_KEY = "commanded_target"  # the clutch's pose
EE_KEY = "ee_target"  # the rate limiter's pose
LIMITER_KEY = "limiter"  # the rate limiter's status
CONTROLLER_KEY = "controller"  # the right controller, before the rotation

# The most time the rate limiter counts for one pass, s: the period of the
# human program's passes. A late pass moves the target no further than a pass
# on time, so what follows the graph never gets a larger step.
LIMITER_PASS_S = 0.02

# As the clutch checks its home: wide enough for a matrix typed with 5 decimals.
_ROTATION_ATOL = 1e-4
_MIN_QUAT_NORM = 1e-6
_PRESSED = 0.5  # a button arrives as 0.0 or 1.0


def _to_base(settings: RetargetSettings) -> np.ndarray:
    """The 4x4 float32 transform from the stage space to the robot's base axes."""
    rotation = np.asarray(settings.controller_to_base, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("retarget.controller_to_base must be 3 rows of 3 numbers")
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL)
        or abs(float(np.linalg.det(rotation)) - 1.0) > _ROTATION_ATOL
    ):
        raise ValueError(
            "retarget.controller_to_base must be a rotation (its rows at right "
            f"angles, of length 1, not mirrored); got {settings.controller_to_base!r}"
        )
    yaw = math.radians(float(settings.stage_yaw_deg))
    if not math.isfinite(yaw):
        raise ValueError("retarget.stage_yaw_deg must be a finite number")
    cos, sin = math.cos(yaw), math.sin(yaw)
    # A turn about the stage's vertical axis, y; it is applied first.
    about_y = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation @ about_y
    return transform


def _matrix(group: Any) -> np.ndarray | None:
    """A pose [x, y, z, qx, qy, qz, qw] as a 4x4 float64 matrix; None: no pose."""
    if group.is_none:
        return None
    try:
        written = group[0]
    except ValueError:  # the node has written none yet
        return None
    pose = np.array(np.from_dlpack(written), dtype=np.float64)
    norm = float(np.linalg.norm(pose[3:7]))
    if not np.all(np.isfinite(pose)) or norm < _MIN_QUAT_NORM:
        return None
    x, y, z, w = pose[3:7] / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


class HumanGraph:
    """The controllers, the rotation, the clutch and the rate limiter.

    One pass: events(), then the pipeline, then read(). The pipeline is given
    external_inputs() and the controllers: controller_inputs() for one
    snapshot, or a TeleopSession polls them from its controller tracker.
    """

    LEAF = "controllers"  # the name of the controllers' leaf

    def __init__(
        self,
        settings: RetargetSettings,
        home_base_T_ee: np.ndarray | None = None,  # noqa: N803
    ) -> None:
        """home_base_T_ee: the first home, 4x4; None is the identity."""
        self._to_base = _to_base(settings)
        if settings.controller_pose == "aim":
            self._hand = (
                ControllerInputIndex.AIM_POSITION,
                ControllerInputIndex.AIM_IS_VALID,
            )
        else:
            self._hand = (
                ControllerInputIndex.GRIP_POSITION,
                ControllerInputIndex.GRIP_IS_VALID,
            )
        self._controllers = ControllersSource(name=self.LEAF)
        to_base = ValueInput(TO_BASE_LEAF, TransformMatrix())
        permitted = ValueInput(PERMITTED_LEAF, PERMITTED_TYPE)
        # The clutch applies the hand's motion as it comes, so its controller
        # must be in the robot's base axes already.
        rotated = self._controllers.transformed(to_base.output(ValueInput.VALUE))
        self._clutch = SO101ClutchRetargeter(
            name="clutch",
            home_base_T_ee=np.eye(4) if home_base_T_ee is None else home_base_T_ee,
            input_device=ControllersSource.RIGHT,
            controller_pose=settings.controller_pose,
            position_scale=settings.position_scale,
            squeeze_threshold=settings.squeeze_threshold,
        )
        # The clutch's measured-pose input stays unwired: it carries the
        # position only. set_home() gives position and orientation.
        commanded = self._clutch.connect(
            {
                ControllersSource.RIGHT: rotated.output(ControllersSource.RIGHT),
                SO101ClutchRetargeter.ENGAGE_PERMITTED_INPUT: permitted.output(
                    ValueInput.VALUE
                ),
            }
        )
        limiter = EePoseRateLimiter(
            name="rate_limiter",
            config=RateLimiterConfig(
                max_linear_velocity=settings.max_linear_velocity,
                max_angular_velocity=settings.max_angular_velocity,
                reject_linear_velocity=settings.reject_linear_velocity,
                reject_angular_velocity=settings.reject_angular_velocity,
                # Also the time of a pass whose clock did not move.
                nominal_dt=LIMITER_PASS_S,
                max_dt=LIMITER_PASS_S,
            ),
        )
        limited = limiter.connect({EE_POSE_KEY: commanded.output(EE_POSE_KEY)})
        # For TeleopSessionConfig(pipeline=...) and for execute_pipeline().
        self.pipeline = OutputCombiner(
            {
                COMMANDED_KEY: commanded.output(EE_POSE_KEY),
                EE_KEY: limited.output(EE_POSE_KEY),
                LIMITER_KEY: limited.output(EE_POSE_STATUS_KEY),
                CONTROLLER_KEY: self._controllers.output(ControllersSource.RIGHT),
            }
        )

    def external_inputs(self, engage_permitted: bool) -> dict[str, RetargeterIO]:
        """The two leaves besides the controllers; every step needs both."""
        to_base = TensorGroup(TransformMatrix())
        to_base[0] = self._to_base
        permitted = TensorGroup(PERMITTED_TYPE)
        permitted[0] = bool(engage_permitted)
        return {
            TO_BASE_LEAF: {ValueInput.VALUE: to_base},
            PERMITTED_LEAF: {ValueInput.VALUE: permitted},
        }

    def controller_inputs(self, right: Any | None) -> dict[str, RetargeterIO]:
        """The controllers' leaf for one ControllerSnapshot of the right controller.

        None: the right controller is not active. The left one never is.
        """
        inputs: RetargeterIO = {}
        for name, group_type in self._controllers.input_spec().items():
            group = TensorGroup(group_type)
            group[0] = right if "right" in name else None
            inputs[name] = group
        return {self.LEAF: inputs}

    def events(self, *, stopped: bool) -> ExecutionEvents:
        """The execution events of the next pass.

        That pass carries a reset unless the clutch itself is still engaged:
        engaged after the pass before, with no set_home since. read() may have
        given the pass before as not engaged. On a reset the clutch starts
        from the home and the rate limiter forgets its last pose, so an
        engagement begins at the home for both. With stopped the clutch never
        engages and an engagement ends.
        """
        return ExecutionEvents(
            reset=not self._clutch.is_engaged,
            execution_state=(
                ExecutionState.STOPPED if stopped else ExecutionState.RUNNING
            ),
        )

    def set_home(self, base_T_ee: np.ndarray) -> None:  # noqa: N803
        """Set the pose the next engagement starts from.

        Not after a pass read() gave as engaged: both targets would step to
        this pose. A clutch that itself still holds the hand lets go.
        """
        self._clutch.set_home_base_T_ee(base_T_ee)

    def read(self, outputs: Mapping[str, Any]) -> GraphStep:
        """The pipeline's outputs of the pass just run, as a GraphStep.

        A squeeze or trigger that is not a finite number is no controller
        data: not present, not engaged, both values 0.0, no hand, no target;
        the buttons are still read. The clutch itself is not changed: it may
        still hold the hand; set_home ends that.
        """
        controller = outputs[CONTROLLER_KEY]
        position, valid = self._hand
        active = not controller.is_none

        # Read while the controller is active, also when its pose is not valid.
        def value(index: Any) -> float:
            return float(controller[index]) if active else 0.0

        squeeze = value(ControllerInputIndex.SQUEEZE_VALUE)
        trigger = value(ControllerInputIndex.TRIGGER_VALUE)
        usable = math.isfinite(squeeze) and math.isfinite(trigger)
        present = active and usable and bool(controller[valid])
        hand = None
        if present:
            hand = np.array(np.from_dlpack(controller[position]), dtype=np.float64)

        status = outputs[LIMITER_KEY]
        disposition = EePoseRateLimiterDisposition(
            int(status[EePoseRateLimiterStatusIndex.DISPOSITION])
        )
        return GraphStep(
            present=present,
            squeeze=squeeze if usable else 0.0,
            trigger=trigger if usable else 0.0,
            a=value(ControllerInputIndex.PRIMARY_CLICK) > _PRESSED,
            b=value(ControllerInputIndex.SECONDARY_CLICK) > _PRESSED,
            engaged=usable and self._clutch.is_engaged,
            ee_target=_matrix(outputs[EE_KEY]) if usable else None,
            commanded_target=_matrix(outputs[COMMANDED_KEY]) if usable else None,
            limiter=disposition.name.lower(),
            hand=hand,
        )
