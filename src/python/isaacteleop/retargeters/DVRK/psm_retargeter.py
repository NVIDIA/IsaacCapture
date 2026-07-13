# SPDX-FileCopyrightText: Copyright (c) 2026 Chris von Csefalvay (HCLTech)
# SPDX-License-Identifier: Apache-2.0

"""XR-controller retargeters for a dVRK Patient Side Manipulator (PSM).

The PSM has six task-space arm joints and two mechanically coupled jaw joints.
This module owns the XR-facing contracts only:

* :class:`DVRKPSMClutchRetargeter` emits a clipped, absolute 7-D pose in a
  caller-selected reference frame only while the XR squeeze (deadman) is
  held.  It captures the controller origin at engagement, so the target does
  not jump when the operator re-clutches.  It wraps the simulator-independent
  :class:`DVRKPSMCartesianClutchStateMachine` shared with other consumers.
* :class:`DVRKPSMGripperRetargeter` maps the analog trigger to the paired PSM
  jaw targets in radians.  It wraps :class:`DVRKPSMJawIntentStateMachine`, a
  simulator-independent input-state kernel that preserves jaw intent across
  arm clutch and tracking transitions.

Differential IK deliberately remains in the simulator integration, where the
live articulation Jacobian, current joint state, and robot-specific limits are
available.  A simulator consumer can use this module's outputs with a six-DOF
DLS controller targeting its PSM tool-tip link.

Frame contract
--------------
The controller stream, home pose, and workspace must use one shared reference
frame.  For one PSM a caller may choose its base frame.  A bimanual integration
can instead keep both streams in a shared world frame, then transform each
target into that arm's base frame immediately before DLS IK.  This keeps one XR
stream valid for two independently placed PSM bases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.execution_events import ExecutionState
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    DLDataType,
    NDArrayType,
)

from .control import (
    DEFAULT_CLUTCH_THRESHOLD as _DEFAULT_CLUTCH_THRESHOLD,
    DEFAULT_JAW_CLOSED as _DEFAULT_JAW_CLOSED,
    DEFAULT_JAW_OPEN as _DEFAULT_JAW_OPEN,
    DEFAULT_OPENING_INTENT_DURATION_S as _DEFAULT_OPENING_INTENT_DURATION_S,
    DEFAULT_TRIGGER_DEADZONE as _TRIGGER_DEADZONE,
    DEFAULT_WORKSPACE_LOWER as _DEFAULT_WORKSPACE_LOWER,
    DEFAULT_WORKSPACE_UPPER as _DEFAULT_WORKSPACE_UPPER,
    DVRKPSMCartesianClutchConfig,
    DVRKPSMCartesianClutchStateMachine,
    DVRKPSMJawIntentConfig,
    DVRKPSMJawIntentStateMachine,
    rotation_matrix_to_quat_xyzw,
)


@dataclass(frozen=True)
class DVRKPSMClutchConfig:
    """Configuration for an absolute-pose PSM clutch.

    ``home_reference_T_ee`` and all workspace coordinates are expressed in one
    shared command reference frame.  The owning simulator must reset the PSM
    to the same physical home pose before the first engagement.  The
    ``orientation_offset`` is a scalar-last calibration rotation conjugated
    around the controller's relative rotation; this preserves the configured
    tool orientation on the squeeze-latching frame.
    """

    home_reference_T_ee: np.ndarray
    input_device: str = ControllersSource.RIGHT
    workspace_lower: tuple[float, float, float] = _DEFAULT_WORKSPACE_LOWER
    workspace_upper: tuple[float, float, float] = _DEFAULT_WORKSPACE_UPPER
    translation_scale: float = 1.0
    orientation_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clutch_threshold: float = _DEFAULT_CLUTCH_THRESHOLD


class DVRKPSMClutchRetargeter(BaseRetargeter):
    """Emit a workspace-bounded absolute PSM tool-tip target with clutch rebasing.

    The Quest controller's analog squeeze is the deadman clutch.  The first
    valid squeezed frame after entering ``RUNNING`` latches the controller
    origin.  The tool target then follows controller displacement from that
    origin.  Releasing squeeze preserves the last target and re-arms the
    origin, so a later squeeze re-clutches at the current PSM target without
    teleporting.  The trigger is deliberately not read here: the paired-jaw
    retargeter owns that relative, intent-latched gripper command.
    """

    OUTPUT_POSE = "ee_pose"
    """Output group carrying the 7-D PSM tool-tip pose command."""

    def __init__(self, config: DVRKPSMClutchConfig, name: str) -> None:
        self._input_device = config.input_device
        transform = np.asarray(config.home_reference_T_ee, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("home_reference_T_ee must be a finite (4, 4) matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError(
                "home_reference_T_ee must be a homogeneous transform with "
                "bottom row [0, 0, 0, 1]"
            )
        home_position = transform[:3, 3].copy()
        home_orientation = rotation_matrix_to_quat_xyzw(transform[:3, :3])
        if home_orientation is None:
            raise ValueError(
                "home_reference_T_ee must contain a proper rotation matrix"
            )
        self._clutch_state = DVRKPSMCartesianClutchStateMachine(
            DVRKPSMCartesianClutchConfig(
                home_position=tuple(home_position),
                home_orientation=tuple(home_orientation),
                workspace_lower=config.workspace_lower,
                workspace_upper=config.workspace_upper,
                translation_scale=config.translation_scale,
                orientation_offset=config.orientation_offset,
                clutch_threshold=config.clutch_threshold,
            )
        )
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Require the optional controller input for the selected PSM side."""
        return {self._input_device: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        """Emit one 7-D ``[x, y, z, qx, qy, qz, qw]`` absolute pose."""
        return {
            self.OUTPUT_POSE: TensorGroupType(
                self.OUTPUT_POSE,
                [
                    NDArrayType(
                        "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def reset_target_state(self) -> None:
        """Restore the configured home and require a fresh clutch origin.

        Simulator integrations should call this after an environment reset or
        while discarding inactive-session outputs.  The next valid squeezed
        sample then captures its current controller pose and emits the exact
        configured home, preventing pre-start controller motion from becoming
        a first-frame target jump.
        """
        self._clutch_state.reset()

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Adapt one controller tensor sample to the shared clutch kernel."""
        if context.execution_events.reset:
            self.reset_target_state()

        output = outputs[self.OUTPUT_POSE]
        controller = inputs[self._input_device]
        if controller.is_none:
            output[0] = self._clutch_state.step(
                controller_position=None,
                controller_orientation=None,
                squeeze=None,
                tracking_valid=False,
                session_active=(
                    context.execution_events.execution_state == ExecutionState.RUNNING
                ),
            )
            return
        output[0] = self._clutch_state.step(
            controller_position=controller[ControllerInputIndex.GRIP_POSITION],
            controller_orientation=controller[ControllerInputIndex.GRIP_ORIENTATION],
            squeeze=controller[ControllerInputIndex.SQUEEZE_VALUE],
            tracking_valid=bool(controller[ControllerInputIndex.GRIP_IS_VALID]),
            session_active=(
                context.execution_events.execution_state == ExecutionState.RUNNING
            ),
        )


@dataclass(frozen=True)
class DVRKPSMGripperConfig:
    """Configuration for one intent-latched pair of native PSM jaws."""

    input_device: str = ControllersSource.RIGHT
    jaw_open: tuple[float, float] = _DEFAULT_JAW_OPEN
    jaw_closed: tuple[float, float] = _DEFAULT_JAW_CLOSED
    initial_closedness: float = 0.0
    clutch_threshold: float = _DEFAULT_CLUTCH_THRESHOLD
    trigger_deadband: float = _TRIGGER_DEADZONE
    opening_intent_duration_s: float = _DEFAULT_OPENING_INTENT_DURATION_S


class DVRKPSMGripperRetargeter(BaseRetargeter):
    """Map an analog trigger to the two coupled dVRK PSM jaw targets.

    The I4H PSM model uses ``psm_tool_gripper1_joint`` and
    ``psm_tool_gripper2_joint``.  The emitted vector is ordered exactly as that
    pair: trigger travel toward pressed closes the jaws and travel toward
    released opens them, bounded by ``[-0.50, +0.50]`` and
    ``[-0.09, +0.09]`` rad by default.  The trigger reference is captured at
    squeeze engagement, so arm unclutch, tracking loss, and re-clutch hold the
    exact last safe target.  See :class:`DVRKPSMJawIntentStateMachine` for the
    deliberate-opening guard.
    """

    OUTPUT_JAW_TARGETS = "jaw_targets"
    """Output group carrying the two PSM gripper-joint targets in radians."""

    def __init__(self, config: DVRKPSMGripperConfig, name: str) -> None:
        self._input_device = config.input_device
        self._jaw_intent = DVRKPSMJawIntentStateMachine(
            DVRKPSMJawIntentConfig(
                jaw_open=config.jaw_open,
                jaw_closed=config.jaw_closed,
                initial_closedness=config.initial_closedness,
                clutch_threshold=config.clutch_threshold,
                trigger_deadband=config.trigger_deadband,
                opening_intent_duration_s=config.opening_intent_duration_s,
            )
        )
        self._last_graph_time_ns: int | None = None
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Require the optional selected controller with an analog trigger."""
        return {self._input_device: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        """Emit the two ordered PSM gripper-joint targets in radians."""
        return {
            self.OUTPUT_JAW_TARGETS: TensorGroupType(
                self.OUTPUT_JAW_TARGETS,
                [
                    NDArrayType(
                        "jaws", shape=(2,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def reset_target_state(self) -> None:
        """Restore configured initial jaw intent and re-arm trigger capture."""
        self._jaw_intent.reset()
        self._last_graph_time_ns = None

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Advance the shared jaw-intent kernel from one graph-time sample."""
        if context.execution_events.reset:
            self.reset_target_state()

        graph_time_ns = int(context.graph_time.real_time_ns)
        if self._last_graph_time_ns is None:
            dt_seconds = 0.0
            self._last_graph_time_ns = graph_time_ns
        elif graph_time_ns <= self._last_graph_time_ns:
            # A repeated or regressed clock sample contributes no intent time.
            # Keep the prior high-water mark so the next normal sample cannot
            # count the regressed interval a second time.
            dt_seconds = 0.0
        else:
            dt_seconds = (graph_time_ns - self._last_graph_time_ns) / 1_000_000_000.0
            self._last_graph_time_ns = graph_time_ns

        output = outputs[self.OUTPUT_JAW_TARGETS]
        controller = inputs[self._input_device]
        if controller.is_none:
            output[0] = self._jaw_intent.step(
                trigger=None,
                squeeze=None,
                tracking_valid=False,
                session_active=(
                    context.execution_events.execution_state == ExecutionState.RUNNING
                ),
                dt_seconds=dt_seconds,
            )
            return

        trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
        squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
        output[0] = self._jaw_intent.step(
            trigger=trigger,
            squeeze=squeeze,
            tracking_valid=bool(controller[ControllerInputIndex.GRIP_IS_VALID]),
            session_active=(
                context.execution_events.execution_state == ExecutionState.RUNNING
            ),
            dt_seconds=dt_seconds,
        )
