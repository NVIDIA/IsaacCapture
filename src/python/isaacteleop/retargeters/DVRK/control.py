# SPDX-FileCopyrightText: Copyright (c) 2026 Chris von Csefalvay (HCLTech)
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent control state for a dVRK PSM.

This module intentionally depends only on NumPy. Isaac Teleop retargeters and
lightweight simulator consumers can therefore share the exact same Cartesian
clutch and jaw-intent semantics without importing one another's runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_JAW_OPEN = (-0.50, 0.50)
DEFAULT_JAW_CLOSED = (-0.09, 0.09)
DEFAULT_CLUTCH_THRESHOLD = 0.50
DEFAULT_TRIGGER_DEADZONE = 0.05
DEFAULT_OPENING_INTENT_DURATION_S = 0.10

# Conservative I4H PSM-base-frame defaults.  They guard a target stream; they
# do not replace consumer-side articulation limits or collision checking.
DEFAULT_HOME_POSITION = (0.0, 0.0, 0.18)
DEFAULT_WORKSPACE_LOWER = (-0.16, -0.14, 0.06)
DEFAULT_WORKSPACE_UPPER = (0.16, 0.14, 0.28)
IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
MIN_QUATERNION_NORM = 1e-6


def _as_vector(values: object, *, length: int, name: str) -> np.ndarray:
    """Return a finite vector, accepting NumPy and DLPack producers."""
    if hasattr(values, "__dlpack__"):
        values = np.from_dlpack(values)  # type: ignore[arg-type]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return vector.copy()


def normalise_quaternion_xyzw(quaternion: object) -> np.ndarray | None:
    """Return a finite unit scalar-last quaternion, or ``None`` if unsafe."""
    try:
        vector = _as_vector(quaternion, length=4, name="quaternion")
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < MIN_QUATERNION_NORM:
        return None
    return vector / norm


def quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of two scalar-last quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def quat_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of a scalar-last unit quaternion."""
    return np.array(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
        dtype=np.float64,
    )


def rotation_matrix_to_quat_xyzw(rotation: object) -> np.ndarray | None:
    """Convert a proper 3-D rotation matrix to a unit scalar-last quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return None
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) <= 0.0
    ):
        return None
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            )
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            )
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            )
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    return normalise_quaternion_xyzw(quaternion)


def rebased_position(
    controller_position: np.ndarray,
    controller_origin: np.ndarray,
    home_position: np.ndarray,
    translation_scale: float,
) -> np.ndarray:
    """Apply an absolute controller delta to a pose captured at engagement."""
    return home_position + translation_scale * (controller_position - controller_origin)


def clip_workspace(
    position: np.ndarray, workspace_lower: np.ndarray, workspace_upper: np.ndarray
) -> np.ndarray:
    """Clip a Cartesian target to configured reference-frame bounds."""
    return np.minimum(np.maximum(position, workspace_lower), workspace_upper)


@dataclass(frozen=True)
class DVRKPSMCartesianClutchConfig:
    """Configuration for simulator-independent clutch-rebased Cartesian pose.

    Home, controller samples, and workspace bounds must share one reference
    frame.  Orientations are scalar-last ``xyzw`` quaternions.  The calibration
    ``orientation_offset`` is conjugated around controller-relative rotation;
    it does not change the configured home orientation at engagement.
    """

    home_position: tuple[float, float, float] = DEFAULT_HOME_POSITION
    home_orientation: tuple[float, float, float, float] = IDENTITY_QUATERNION_XYZW
    workspace_lower: tuple[float, float, float] = DEFAULT_WORKSPACE_LOWER
    workspace_upper: tuple[float, float, float] = DEFAULT_WORKSPACE_UPPER
    translation_scale: float = 1.0
    orientation_offset: tuple[float, float, float, float] = IDENTITY_QUATERNION_XYZW
    clutch_threshold: float = DEFAULT_CLUTCH_THRESHOLD


class DVRKPSMCartesianClutchStateMachine:
    """Produce a bounded absolute tool pose from one controller clutch stream.

    The first valid squeezed sample captures a controller origin and emits the
    exact held pose.  Subsequent samples apply absolute translation and
    orientation deltas from that origin.  Unclutch, tracking loss, invalid
    samples, and inactive sessions hold the exact last pose and require a fresh
    origin on the next engagement.  Differential IK remains a consumer concern.
    """

    def __init__(self, config: DVRKPSMCartesianClutchConfig | None = None) -> None:
        self._config = config or DVRKPSMCartesianClutchConfig()
        self._configured_home = _as_vector(
            self._config.home_position, length=3, name="home_position"
        )
        home_orientation = normalise_quaternion_xyzw(self._config.home_orientation)
        if home_orientation is None:
            raise ValueError(
                "home_orientation must be a non-zero finite xyzw quaternion"
            )
        self._configured_home_orientation = home_orientation
        self._workspace_lower = _as_vector(
            self._config.workspace_lower, length=3, name="workspace_lower"
        )
        self._workspace_upper = _as_vector(
            self._config.workspace_upper, length=3, name="workspace_upper"
        )
        if np.any(self._workspace_lower > self._workspace_upper):
            raise ValueError("workspace_lower must not exceed workspace_upper")
        if not np.array_equal(
            self._configured_home,
            clip_workspace(
                self._configured_home,
                self._workspace_lower,
                self._workspace_upper,
            ),
        ):
            raise ValueError("home_position must lie within the configured workspace")
        if (
            not np.isfinite(self._config.translation_scale)
            or self._config.translation_scale <= 0.0
        ):
            raise ValueError("translation_scale must be finite and positive")
        self._translation_scale = float(self._config.translation_scale)
        orientation_offset = normalise_quaternion_xyzw(self._config.orientation_offset)
        if orientation_offset is None:
            raise ValueError(
                "orientation_offset must be a non-zero finite xyzw quaternion"
            )
        self._orientation_offset = orientation_offset
        if not np.isfinite(self._config.clutch_threshold) or not (
            0.0 <= self._config.clutch_threshold <= 1.0
        ):
            raise ValueError("clutch_threshold must be finite and in [0, 1]")

        self._last_pose = np.concatenate(
            [self._configured_home, self._configured_home_orientation]
        ).astype(np.float32)
        self._engaged = False
        self._controller_origin: np.ndarray | None = None
        self._controller_orientation_origin: np.ndarray | None = None
        self._pose_at_engagement: np.ndarray | None = None

    @property
    def pose(self) -> np.ndarray:
        """Return a copy of the exact currently held absolute pose."""
        return self._last_pose.copy()

    @property
    def engaged(self) -> bool:
        """Whether a valid controller origin is currently captured."""
        return self._engaged

    def reset(self) -> np.ndarray:
        """Restore configured home and require a fresh controller origin."""
        self._last_pose = np.concatenate(
            [self._configured_home, self._configured_home_orientation]
        ).astype(np.float32)
        self._disengage()
        return self.pose

    def _disengage(self) -> None:
        self._engaged = False
        self._controller_origin = None
        self._controller_orientation_origin = None
        self._pose_at_engagement = None

    def step(
        self,
        *,
        controller_position: object | None,
        controller_orientation: object | None,
        squeeze: float | None,
        tracking_valid: bool,
        session_active: bool,
    ) -> np.ndarray:
        """Advance one controller sample and return the absolute held pose."""
        try:
            squeeze_value = float(squeeze) if squeeze is not None else None
        except (TypeError, ValueError):
            squeeze_value = None
        if (
            not session_active
            or not tracking_valid
            or squeeze_value is None
            or not np.isfinite(squeeze_value)
            or squeeze_value < self._config.clutch_threshold
        ):
            self._disengage()
            return self.pose

        try:
            position = _as_vector(
                controller_position, length=3, name="controller_position"
            )
        except (TypeError, ValueError):
            self._disengage()
            return self.pose
        orientation = normalise_quaternion_xyzw(controller_orientation)
        if orientation is None:
            self._disengage()
            return self.pose

        if not self._engaged:
            self._engaged = True
            self._controller_origin = position
            self._controller_orientation_origin = orientation
            self._pose_at_engagement = self._last_pose.astype(np.float64)
            return self.pose

        assert self._controller_origin is not None
        assert self._controller_orientation_origin is not None
        assert self._pose_at_engagement is not None
        target_position = clip_workspace(
            rebased_position(
                position,
                self._controller_origin,
                self._pose_at_engagement[:3],
                self._translation_scale,
            ),
            self._workspace_lower,
            self._workspace_upper,
        )
        controller_relative_orientation = quat_mul_xyzw(
            quat_conjugate_xyzw(self._controller_orientation_origin), orientation
        )
        calibrated_relative_orientation = quat_mul_xyzw(
            quat_mul_xyzw(self._orientation_offset, controller_relative_orientation),
            quat_conjugate_xyzw(self._orientation_offset),
        )
        target_orientation = normalise_quaternion_xyzw(
            quat_mul_xyzw(
                self._pose_at_engagement[3:7], calibrated_relative_orientation
            )
        )
        if target_orientation is None:
            self._disengage()
            return self.pose

        self._last_pose = np.concatenate([target_position, target_orientation]).astype(
            np.float32
        )
        return self.pose


def closedness_to_jaw_targets(
    closedness: float, jaw_open: np.ndarray, jaw_closed: np.ndarray
) -> np.ndarray:
    """Interpolate paired jaw targets between configured physical endpoints."""
    return jaw_open + float(np.clip(closedness, 0.0, 1.0)) * (jaw_closed - jaw_open)


def _as_jaw_vector(values: object, *, name: str) -> np.ndarray:
    """Return a finite two-element jaw vector."""
    return _as_vector(values, length=2, name=name)


@dataclass(frozen=True)
class DVRKPSMJawIntentConfig:
    """Configuration for :class:`DVRKPSMJawIntentStateMachine`.

    ``jaw_open`` and ``jaw_closed`` are the physical articulation endpoints;
    every emitted target is interpolated between them.  ``initial_closedness``
    selects the exact reset target on that segment (zero is open, one is
    closed).  The squeeze threshold is intentionally shared with the arm
    clutch so releasing the arm deadman never changes the jaw command.

    Trigger motion is relative to the value captured on each squeeze
    engagement.  This avoids a jump when the operator re-clutches with the
    index trigger at a different position.  Positive trigger travel closes
    immediately, but negative travel must exceed ``trigger_deadband`` and
    remain present for ``opening_intent_duration_s`` before it opens the jaws.
    That short time gate rejects the common natural-unclutch sequence in which
    the index trigger relaxes one or two samples before squeeze.  The pending
    opening is cancelled if squeeze or tracking disappears.
    """

    jaw_open: tuple[float, float] = DEFAULT_JAW_OPEN
    jaw_closed: tuple[float, float] = DEFAULT_JAW_CLOSED
    initial_closedness: float = 0.0
    clutch_threshold: float = DEFAULT_CLUTCH_THRESHOLD
    trigger_deadband: float = DEFAULT_TRIGGER_DEADZONE
    opening_intent_duration_s: float = DEFAULT_OPENING_INTENT_DURATION_S


class DVRKPSMJawIntentStateMachine:
    """Turn trigger/clutch samples into bounded, latched PSM jaw targets.

    Call :meth:`step` with elapsed seconds to keep opening-intent timing
    independent of render or physics frame rate.  Missing/invalid tracking, an
    inactive session, or a released squeeze freezes the exact last target and
    re-arms trigger-reference capture for the next engagement.

    The state machine never inspects contacts and never creates an attachment;
    it only represents operator jaw intent.
    """

    _DIRECTION_EPSILON = 1e-9

    def __init__(self, config: DVRKPSMJawIntentConfig | None = None) -> None:
        self._config = config or DVRKPSMJawIntentConfig()
        self._jaw_open = _as_jaw_vector(self._config.jaw_open, name="jaw_open")
        self._jaw_closed = _as_jaw_vector(self._config.jaw_closed, name="jaw_closed")
        if not (
            self._jaw_open[0] < self._jaw_closed[0] <= 0.0
            and self._jaw_open[1] > self._jaw_closed[1] >= 0.0
        ):
            raise ValueError(
                "dVRK jaw endpoints must move inward toward zero in "
                "[gripper1, gripper2] order"
            )
        if not np.isfinite(self._config.initial_closedness) or not (
            0.0 <= self._config.initial_closedness <= 1.0
        ):
            raise ValueError("initial_closedness must be finite and in [0, 1]")
        if not np.isfinite(self._config.clutch_threshold) or not (
            0.0 <= self._config.clutch_threshold <= 1.0
        ):
            raise ValueError("clutch_threshold must be finite and in [0, 1]")
        if not np.isfinite(self._config.trigger_deadband) or not (
            0.0 <= self._config.trigger_deadband < 1.0
        ):
            raise ValueError("trigger_deadband must be finite and in [0, 1)")
        if not np.isfinite(self._config.opening_intent_duration_s) or (
            self._config.opening_intent_duration_s < 0.0
        ):
            raise ValueError(
                "opening_intent_duration_s must be finite and non-negative"
            )

        self._initial_closedness = float(self._config.initial_closedness)
        self._closedness = self._initial_closedness
        self._targets = closedness_to_jaw_targets(
            self._closedness, self._jaw_open, self._jaw_closed
        ).astype(np.float32)
        self._engaged = False
        self._interaction_started = False
        self._last_trigger: float | None = None
        self._opening_anchor_trigger: float | None = None
        self._opening_elapsed_s = 0.0
        self._opening_active = False

    @property
    def targets(self) -> np.ndarray:
        """Return a copy of the exact currently latched jaw targets."""
        return self._targets.copy()

    @property
    def closedness(self) -> float:
        """Return current scalar jaw closedness in ``[0, 1]``."""
        return self._closedness

    def reset(self) -> np.ndarray:
        """Restore the configured initial target and clear engagement state."""
        self._closedness = self._initial_closedness
        self._targets = closedness_to_jaw_targets(
            self._closedness, self._jaw_open, self._jaw_closed
        ).astype(np.float32)
        self._disengage()
        return self.targets

    def _disengage(self) -> None:
        """Hold target while requiring fresh trigger-reference capture."""
        self._engaged = False
        self._interaction_started = False
        self._last_trigger = None
        self._cancel_opening()

    def _cancel_opening(self) -> None:
        self._opening_anchor_trigger = None
        self._opening_elapsed_s = 0.0
        self._opening_active = False

    def _set_closedness(self, value: float) -> None:
        self._closedness = float(np.clip(value, 0.0, 1.0))
        self._targets = closedness_to_jaw_targets(
            self._closedness, self._jaw_open, self._jaw_closed
        ).astype(np.float32)

    def step(
        self,
        *,
        trigger: float | None,
        squeeze: float | None,
        tracking_valid: bool,
        session_active: bool,
        dt_seconds: float,
    ) -> np.ndarray:
        """Advance one input sample and return the exact jaw target to emit.

        ``dt_seconds`` is elapsed wall/graph time since the previous sample.
        A non-finite or negative duration is rejected because silently counting
        it could bypass the deliberate-opening guard.
        """
        try:
            elapsed_seconds = float(dt_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError("dt_seconds must be finite and non-negative") from error
        if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("dt_seconds must be finite and non-negative")

        try:
            trigger_value = float(trigger) if trigger is not None else None
            squeeze_value = float(squeeze) if squeeze is not None else None
        except (TypeError, ValueError):
            trigger_value = None
            squeeze_value = None
        if (
            not session_active
            or not tracking_valid
            or trigger_value is None
            or squeeze_value is None
            or not np.isfinite(trigger_value)
            or not np.isfinite(squeeze_value)
            or squeeze_value < self._config.clutch_threshold
        ):
            # This also cancels a not-yet-committed opening, which is the key
            # protection for trigger-up immediately followed by squeeze-up.
            self._disengage()
            return self.targets

        trigger_value = float(np.clip(trigger_value, 0.0, 1.0))
        if not self._engaged:
            self._engaged = True
            self._interaction_started = False
            self._last_trigger = trigger_value
            self._cancel_opening()
            return self.targets

        assert self._last_trigger is not None
        trigger_delta = trigger_value - self._last_trigger

        if not self._interaction_started:
            # Reference capture makes re-clutch a no-op until the operator
            # moves the index trigger far enough to be unambiguous.
            if trigger_delta >= self._config.trigger_deadband:
                self._interaction_started = True
                self._cancel_opening()
                self._set_closedness(self._closedness + trigger_delta)
                self._last_trigger = trigger_value
                return self.targets
            if trigger_delta <= -self._config.trigger_deadband:
                if self._opening_anchor_trigger is None:
                    self._opening_anchor_trigger = self._last_trigger
                self._opening_elapsed_s += elapsed_seconds
                if self._opening_elapsed_s >= self._config.opening_intent_duration_s:
                    opening_delta = trigger_value - self._opening_anchor_trigger
                    self._interaction_started = True
                    self._opening_active = True
                    self._set_closedness(self._closedness + opening_delta)
                    self._last_trigger = trigger_value
                return self.targets

            self._cancel_opening()
            return self.targets

        if trigger_delta > self._DIRECTION_EPSILON:
            # Closing is immediately safe but remains physically bounded by
            # the configured endpoint interpolation.
            self._cancel_opening()
            self._set_closedness(self._closedness + trigger_delta)
            self._last_trigger = trigger_value
            return self.targets

        if trigger_delta < -self._DIRECTION_EPSILON:
            if self._opening_active:
                self._set_closedness(self._closedness + trigger_delta)
                self._last_trigger = trigger_value
                return self.targets
            if self._opening_anchor_trigger is None:
                self._opening_anchor_trigger = self._last_trigger
            if (
                trigger_value
                > self._opening_anchor_trigger - self._config.trigger_deadband
            ):
                self._cancel_opening()
                return self.targets
            self._opening_elapsed_s += elapsed_seconds
            if self._opening_elapsed_s >= self._config.opening_intent_duration_s:
                opening_delta = trigger_value - self._opening_anchor_trigger
                self._opening_active = True
                self._set_closedness(self._closedness + opening_delta)
                self._last_trigger = trigger_value
            return self.targets

        # A held-lower trigger still constitutes sustained opening intent even
        # though no new analogue delta arrived in this particular sample.
        if self._opening_anchor_trigger is not None and not self._opening_active:
            self._opening_elapsed_s += elapsed_seconds
            if self._opening_elapsed_s >= self._config.opening_intent_duration_s:
                opening_delta = trigger_value - self._opening_anchor_trigger
                self._opening_active = True
                self._set_closedness(self._closedness + opening_delta)
                self._last_trigger = trigger_value
        return self.targets


__all__ = [
    "DVRKPSMCartesianClutchConfig",
    "DVRKPSMCartesianClutchStateMachine",
    "DVRKPSMJawIntentConfig",
    "DVRKPSMJawIntentStateMachine",
]
