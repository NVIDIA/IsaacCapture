# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for the dVRK PSM clutch and paired-jaw retargeters."""

import math

import numpy as np
import pytest
import isaacteleop.retargeters as retargeters

from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaacteleop.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
)
from isaacteleop.retargeters import (
    DVRKPSMClutchConfig,
    DVRKPSMClutchRetargeter,
    DVRKPSMGripperConfig,
    DVRKPSMGripperRetargeter,
)
from isaacteleop.retargeters.DVRK.control import (
    DVRKPSMCartesianClutchConfig,
    DVRKPSMCartesianClutchStateMachine,
    DVRKPSMJawIntentConfig,
    DVRKPSMJawIntentStateMachine,
    closedness_to_jaw_targets,
    normalise_quaternion_xyzw,
    quat_mul_xyzw,
    rebased_position,
)

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_JAW_OPEN = np.array([-0.50, 0.50], dtype=np.float64)
_JAW_CLOSED = np.array([-0.09, 0.09], dtype=np.float64)
_POSE_OUTPUT = DVRKPSMClutchRetargeter.OUTPUT_POSE
_JAW_OUTPUT = DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS


def test_public_package_surface_contains_only_supported_dvrk_types() -> None:
    supported = {
        "DVRKPSMClutchConfig",
        "DVRKPSMClutchRetargeter",
        "DVRKPSMGripperConfig",
        "DVRKPSMGripperRetargeter",
    }
    internal = {
        "DVRKPSMCartesianClutchConfig",
        "DVRKPSMCartesianClutchStateMachine",
        "DVRKPSMJawIntentConfig",
        "DVRKPSMJawIntentStateMachine",
    }

    assert supported <= set(retargeters.__all__)
    assert internal.isdisjoint(retargeters.__all__)
    assert all(hasattr(retargeters, name) for name in supported)
    assert all(not hasattr(retargeters, name) for name in internal)
    assert DVRKPSMClutchRetargeter.OUTPUT_POSE == "ee_pose"
    assert DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS == "jaw_targets"


def test_clutch_config_requires_an_explicit_home_transform() -> None:
    with pytest.raises(TypeError, match="home_reference_T_ee"):
        DVRKPSMClutchConfig()  # type: ignore[call-arg]


def _make_context(
    *,
    reset: bool = False,
    state: ExecutionState = ExecutionState.RUNNING,
    real_time_s: float = 0.0,
) -> ComputeContext:
    """Build one retargeter compute context."""
    time_ns = round(real_time_s * 1_000_000_000)
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=time_ns, real_time_ns=time_ns),
        execution_events=ExecutionEvents(reset=reset, execution_state=state),
    )


def _build_io(retargeter):
    """Construct empty input/output tensor groups from a retargeter contract."""
    inputs = {}
    for key, value in retargeter.input_spec().items():
        inputs[key] = (
            OptionalTensorGroup(value)
            if isinstance(value, OptionalTensorGroupType)
            else TensorGroup(value)
        )
    outputs = {}
    for key, value in retargeter.output_spec().items():
        outputs[key] = (
            OptionalTensorGroup(value)
            if isinstance(value, OptionalTensorGroupType)
            else TensorGroup(value)
        )
    return inputs, outputs


def _make_controller(
    *,
    grip_pos=(0.0, 0.0, 0.0),
    grip_ori=_IDENTITY_QUAT,
    trigger: float = 0.0,
    squeeze: float = 1.0,
    grip_is_valid: bool = True,
) -> TensorGroup:
    """Build a present XR controller sample for a retargeter frame."""
    controller = TensorGroup(ControllerInput())
    controller[ControllerInputIndex.GRIP_POSITION] = np.asarray(
        grip_pos, dtype=np.float32
    )
    controller[ControllerInputIndex.GRIP_ORIENTATION] = np.asarray(
        grip_ori, dtype=np.float32
    )
    controller[ControllerInputIndex.GRIP_IS_VALID] = grip_is_valid
    controller[ControllerInputIndex.AIM_ORIENTATION] = _IDENTITY_QUAT
    controller[ControllerInputIndex.AIM_IS_VALID] = True
    controller[ControllerInputIndex.SQUEEZE_VALUE] = float(squeeze)
    controller[ControllerInputIndex.TRIGGER_VALUE] = float(trigger)
    return controller


def _make_home_transform(position, rotation: np.ndarray | None = None) -> np.ndarray:
    """Build a PSM base-to-tool home transform with identity orientation."""
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _quat_xyzw_z(angle: float) -> np.ndarray:
    """Build a scalar-last quaternion for a rotation about the Z axis."""
    return np.array((0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)))


def _quat_xyzw_x(angle: float) -> np.ndarray:
    """Build a scalar-last quaternion for a rotation about the X axis."""
    return np.array((math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)))


def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Build a 3-D Z-axis rotation matrix for a configured tool home."""
    return np.array(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def _read_vector(outputs, key: str) -> np.ndarray:
    """Read an emitted DLPack vector into a float64 array for comparisons."""
    return np.asarray(np.from_dlpack(outputs[key][0]), dtype=np.float64)


class TestDVRKPSMGripperMath:
    """Pure trigger-to-paired-jaw mapping checks."""

    def test_trigger_endpoints_match_i4h_psm_joint_targets(self):
        """Released/open and pressed/closed commands land on I4H jaw endpoints."""
        np.testing.assert_allclose(
            closedness_to_jaw_targets(0.0, _JAW_OPEN, _JAW_CLOSED),
            _JAW_OPEN,
        )
        np.testing.assert_allclose(
            closedness_to_jaw_targets(1.0, _JAW_OPEN, _JAW_CLOSED),
            _JAW_CLOSED,
        )

    def test_mapping_is_continuous_and_mirror_coupled(self):
        """A half trigger produces a bounded pair that moves symmetrically toward zero."""
        targets = closedness_to_jaw_targets(0.5, _JAW_OPEN, _JAW_CLOSED)
        assert _JAW_OPEN[0] < targets[0] < _JAW_CLOSED[0]
        assert _JAW_CLOSED[1] < targets[1] < _JAW_OPEN[1]
        assert targets[0] == pytest.approx(-targets[1], abs=1e-8)


class TestDVRKPSMGripperRetargeter:
    """End-to-end paired-jaw commands through BaseRetargeter.compute."""

    @staticmethod
    def _compute(retargeter, *, time_s: float, **controller_kwargs) -> np.ndarray:
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(**controller_kwargs)
        retargeter.compute(inputs, outputs, _make_context(real_time_s=time_s))
        return _read_vector(outputs, _JAW_OUTPUT)

    def test_engagement_captures_trigger_then_full_travel_closes_both_jaws(self):
        """Engagement is no-op; deliberate press emits bounded articulation-order targets."""
        retargeter = DVRKPSMGripperRetargeter(DVRKPSMGripperConfig(), name="jaws")
        engaged = self._compute(retargeter, time_s=0.00, trigger=0.0, squeeze=1.0)
        np.testing.assert_allclose(engaged, _JAW_OPEN, atol=1e-6)

        closed = self._compute(retargeter, time_s=0.02, trigger=1.0, squeeze=1.0)
        np.testing.assert_allclose(closed, _JAW_CLOSED, atol=1e-6)

    def test_configured_initial_hold_can_use_a_validated_tighter_endpoint(self):
        """A task can start at its physical grasp cap without an initial trigger jump."""
        validated_closed = (-0.032, 0.032)
        retargeter = DVRKPSMGripperRetargeter(
            DVRKPSMGripperConfig(
                jaw_closed=validated_closed,
                initial_closedness=1.0,
            ),
            name="jaws",
        )
        initial = self._compute(retargeter, time_s=0.00, trigger=0.63, squeeze=1.0)
        np.testing.assert_allclose(initial, validated_closed, atol=1e-7)

        # A different trigger position after re-clutch remains a no-op until
        # deliberate post-clutch travel occurs.
        self._compute(retargeter, time_s=0.02, trigger=0.10, squeeze=0.0)
        reclutched = self._compute(retargeter, time_s=0.04, trigger=0.10, squeeze=1.0)
        np.testing.assert_array_equal(reclutched, initial)

    def test_explicit_target_reset_restores_configured_hold_before_reengagement(self):
        """An environment reset restores the configured initial jaw target."""
        configured_hold = (-0.030, 0.030)
        retargeter = DVRKPSMGripperRetargeter(
            DVRKPSMGripperConfig(
                jaw_open=(-0.25, 0.25),
                jaw_closed=configured_hold,
                initial_closedness=1.0,
            ),
            name="left_jaws",
        )
        initial = self._compute(retargeter, time_s=0.00, trigger=1.0)
        np.testing.assert_allclose(initial, configured_hold, atol=1e-8)
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.05, trigger=0.0), initial
        )
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.10, trigger=0.0), initial
        )
        opened = self._compute(retargeter, time_s=0.151, trigger=0.0)
        assert not np.array_equal(opened, initial)

        retargeter.reset_target_state()
        reset_engagement = self._compute(retargeter, time_s=0.16, trigger=0.0)
        np.testing.assert_allclose(reset_engagement, configured_hold, atol=1e-8)

    def test_wrapper_times_deliberate_opening_from_graph_time(self):
        """Opening guard uses elapsed graph time rather than an assumed frame rate."""
        retargeter = DVRKPSMGripperRetargeter(
            DVRKPSMGripperConfig(initial_closedness=1.0), name="jaws"
        )
        closed = self._compute(retargeter, time_s=0.000, trigger=1.0)
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.020, trigger=0.4), closed
        )
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.099, trigger=0.4), closed
        )
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.100, trigger=0.4), closed
        )
        opened = self._compute(retargeter, time_s=0.121, trigger=0.4)
        assert not np.array_equal(opened, closed)

    def test_regressed_graph_time_cannot_bypass_opening_duration(self):
        """A clock regression contributes zero time and does not move the high-water mark."""
        retargeter = DVRKPSMGripperRetargeter(
            DVRKPSMGripperConfig(initial_closedness=1.0), name="jaws"
        )
        closed = self._compute(retargeter, time_s=1.00, trigger=1.0)
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=0.50, trigger=0.4), closed
        )
        np.testing.assert_array_equal(
            self._compute(retargeter, time_s=1.05, trigger=0.4), closed
        )
        opened = self._compute(retargeter, time_s=1.10, trigger=0.4)
        assert not np.array_equal(opened, closed)

    def test_dropped_frame_holds_and_reset_restores_configured_initial_target(self):
        """Tracking loss holds exactly; reset restores the configured initial target."""
        retargeter = DVRKPSMGripperRetargeter(DVRKPSMGripperConfig(), name="jaws")
        self._compute(retargeter, time_s=0.00, trigger=0.0)
        closed = self._compute(retargeter, time_s=0.02, trigger=1.0)

        dropped_inputs, dropped_outputs = _build_io(retargeter)
        retargeter.compute(
            dropped_inputs, dropped_outputs, _make_context(real_time_s=0.04)
        )
        np.testing.assert_allclose(
            _read_vector(dropped_outputs, _JAW_OUTPUT),
            closed,
            atol=1e-6,
        )

        reset_inputs, reset_outputs = _build_io(retargeter)
        retargeter.compute(
            reset_inputs,
            reset_outputs,
            _make_context(reset=True, real_time_s=0.06),
        )
        np.testing.assert_allclose(
            _read_vector(reset_outputs, _JAW_OUTPUT), _JAW_OPEN, atol=1e-6
        )

    def test_nonfinite_trigger_holds_last_safe_jaw_target(self):
        """A malformed analog sample cannot emit NaN joint targets."""
        retargeter = DVRKPSMGripperRetargeter(DVRKPSMGripperConfig(), name="jaws")
        self._compute(retargeter, time_s=0.00, trigger=0.0)
        closed = self._compute(retargeter, time_s=0.02, trigger=1.0)

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(trigger=math.nan)
        retargeter.compute(inputs, outputs, _make_context(real_time_s=0.04))
        emitted = _read_vector(outputs, _JAW_OUTPUT)
        assert np.all(np.isfinite(emitted))
        np.testing.assert_allclose(emitted, closed, atol=1e-6)

    def test_stopped_session_holds_the_last_jaw_target(self):
        """A session stop freezes jaws as well as the Cartesian PSM target."""
        retargeter = DVRKPSMGripperRetargeter(DVRKPSMGripperConfig(), name="jaws")
        self._compute(retargeter, time_s=0.00, trigger=0.0)
        closed = self._compute(retargeter, time_s=0.02, trigger=1.0)

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(trigger=0.0)
        retargeter.compute(
            inputs,
            outputs,
            _make_context(state=ExecutionState.STOPPED, real_time_s=0.04),
        )
        np.testing.assert_allclose(
            _read_vector(outputs, _JAW_OUTPUT), closed, atol=1e-6
        )


class TestDVRKPSMJawIntentStateMachine:
    """Pure jaw-intent behaviour shared by device adapters."""

    @staticmethod
    def _kernel(*, initial_closedness: float = 0.65):
        return DVRKPSMJawIntentStateMachine(
            DVRKPSMJawIntentConfig(
                jaw_open=tuple(_JAW_OPEN),
                jaw_closed=tuple(_JAW_CLOSED),
                initial_closedness=initial_closedness,
                trigger_deadband=0.05,
                opening_intent_duration_s=0.08,
            )
        )

    @staticmethod
    def _step(
        kernel,
        *,
        trigger: float | None = 0.8,
        squeeze: float | None = 1.0,
        valid: bool = True,
        active: bool = True,
        dt: float = 0.02,
    ) -> np.ndarray:
        return kernel.step(
            trigger=trigger,
            squeeze=squeeze,
            tracking_valid=valid,
            session_active=active,
            dt_seconds=dt,
        )

    def test_initial_target_is_exact_and_first_engagement_is_no_op(self):
        kernel = self._kernel(initial_closedness=0.65)
        expected = closedness_to_jaw_targets(0.65, _JAW_OPEN, _JAW_CLOSED)
        np.testing.assert_allclose(kernel.targets, expected, atol=1e-6)
        np.testing.assert_allclose(
            self._step(kernel, trigger=0.73), expected, atol=1e-6
        )

    @pytest.mark.parametrize(
        ("jaw_open", "jaw_closed"),
        (
            ((-0.50, 0.50), (-0.60, 0.09)),
            ((-0.50, 0.50), (-0.09, 0.60)),
            ((0.10, 0.50), (0.05, 0.09)),
            ((-0.50, -0.10), (-0.09, -0.05)),
        ),
    )
    def test_invalid_dvrk_jaw_endpoint_direction_is_rejected(
        self, jaw_open, jaw_closed
    ):
        with pytest.raises(ValueError, match="must move inward toward zero"):
            DVRKPSMJawIntentStateMachine(
                DVRKPSMJawIntentConfig(jaw_open=jaw_open, jaw_closed=jaw_closed)
            )

    @pytest.mark.parametrize("field", ("trigger", "squeeze"))
    def test_malformed_analog_sample_holds_and_rearms(self, field):
        kernel = self._kernel(initial_closedness=0.8)
        held = self._step(kernel, trigger=0.4)
        sample = {field: "not-a-number"}
        np.testing.assert_array_equal(self._step(kernel, **sample), held)
        np.testing.assert_array_equal(self._step(kernel, trigger=0.9), held)

    @pytest.mark.parametrize(
        "transition",
        (
            {"squeeze": 0.0},
            {"valid": False},
            {"active": False},
            {"trigger": None},
        ),
    )
    def test_unclutch_tracking_loss_and_inactive_session_hold_exact_target(
        self, transition
    ):
        kernel = self._kernel(initial_closedness=0.2)
        self._step(kernel, trigger=0.2)
        moved = self._step(kernel, trigger=0.7)
        sample = {"trigger": 0.0, **transition}
        held = self._step(kernel, **sample)
        np.testing.assert_array_equal(held, moved)

    def test_reclutch_captures_new_reference_and_requires_fresh_interaction(self):
        kernel = self._kernel(initial_closedness=0.7)
        initial = self._step(kernel, trigger=0.9)
        self._step(kernel, trigger=0.2, squeeze=0.0)

        # Re-clutching at a very different index-trigger position is a no-op.
        np.testing.assert_array_equal(self._step(kernel, trigger=0.2), initial)
        np.testing.assert_array_equal(self._step(kernel, trigger=0.23), initial)

        # Deliberate travel beyond the deadband resumes relative control.
        moved = self._step(kernel, trigger=0.4)
        assert kernel.closedness == pytest.approx(0.9)
        assert not np.array_equal(moved, initial)

    def test_short_trigger_release_then_squeeze_release_never_opens(self):
        kernel = self._kernel(initial_closedness=1.0)
        closed = self._step(kernel, trigger=1.0)

        # Index release starts first, as it commonly does in a natural
        # unclutch, but remains shorter than the deliberate-opening duration.
        np.testing.assert_array_equal(
            self._step(kernel, trigger=0.2, squeeze=1.0, dt=0.02), closed
        )
        np.testing.assert_array_equal(
            self._step(kernel, trigger=0.2, squeeze=0.0, dt=0.02), closed
        )

        # Re-clutch at the relaxed trigger position also preserves the hold.
        np.testing.assert_array_equal(
            self._step(kernel, trigger=0.2, squeeze=1.0, dt=0.50), closed
        )

    def test_deliberate_opening_uses_observed_time_not_frame_count(self):
        for time_steps in ((0.02, 0.02, 0.04), (0.079, 0.001)):
            kernel = self._kernel(initial_closedness=1.0)
            closed = self._step(kernel, trigger=1.0, dt=0.0)

            # A long gap before opening intent is first observed contributes
            # no duration: the trigger state during that gap is unknown.
            np.testing.assert_array_equal(
                self._step(kernel, trigger=0.4, dt=10.0), closed
            )
            for dt in time_steps[:-1]:
                np.testing.assert_array_equal(
                    self._step(kernel, trigger=0.4, dt=dt), closed
                )
            opened = self._step(kernel, trigger=0.4, dt=time_steps[-1])
            assert kernel.closedness == pytest.approx(0.4)
            assert not np.array_equal(opened, closed)

    def test_reversing_after_closing_starts_a_fresh_observed_opening_timer(self):
        kernel = self._kernel(initial_closedness=0.5)
        self._step(kernel, trigger=0.4, dt=0.0)
        closed = self._step(kernel, trigger=0.8, dt=0.02)
        assert kernel.closedness == pytest.approx(0.9)

        # This exercises the post-interaction direction-reversal path.  The
        # first lower sample starts, but cannot satisfy, the time gate.
        np.testing.assert_array_equal(self._step(kernel, trigger=0.6, dt=10.0), closed)
        opened = self._step(kernel, trigger=0.6, dt=0.08)
        assert kernel.closedness == pytest.approx(0.7)
        assert not np.array_equal(opened, closed)

    def test_returning_to_anchor_cancels_pending_opening_after_closing(self):
        kernel = self._kernel(initial_closedness=0.5)
        self._step(kernel, trigger=0.4, dt=0.0)
        closed = self._step(kernel, trigger=0.8, dt=0.02)
        self._step(kernel, trigger=0.6, dt=0.0)

        # Returning to the pre-opening anchor cannot mature a zero-motion
        # intent, regardless of how long it remains there.
        np.testing.assert_array_equal(self._step(kernel, trigger=0.8, dt=1.0), closed)
        np.testing.assert_array_equal(self._step(kernel, trigger=0.6, dt=0.0), closed)

        opened = self._step(kernel, trigger=0.6, dt=0.08)
        assert kernel.closedness == pytest.approx(0.7)
        assert not np.array_equal(opened, closed)

    def test_closing_is_immediate_but_capped_at_configured_endpoint(self):
        kernel = self._kernel(initial_closedness=0.4)
        self._step(kernel, trigger=0.0)
        kernel.step(
            trigger=4.0,
            squeeze=1.0,
            tracking_valid=True,
            session_active=True,
            dt_seconds=0.01,
        )
        assert kernel.closedness == 1.0
        np.testing.assert_allclose(kernel.targets, _JAW_CLOSED, atol=1e-6)

    def test_reset_restores_initial_target_and_rearms_reference_capture(self):
        kernel = self._kernel(initial_closedness=0.35)
        self._step(kernel, trigger=0.0)
        self._step(kernel, trigger=0.6)
        reset = kernel.reset()
        expected = closedness_to_jaw_targets(0.35, _JAW_OPEN, _JAW_CLOSED)
        np.testing.assert_allclose(reset, expected, atol=1e-6)
        np.testing.assert_allclose(self._step(kernel, trigger=0.9), expected, atol=1e-6)

    @pytest.mark.parametrize(
        ("config", "field"),
        (
            (
                DVRKPSMJawIntentConfig(
                    jaw_open=(-1e40, 1e40),
                    jaw_closed=(-0.09, 0.09),
                ),
                "jaw_open",
            ),
            (
                DVRKPSMJawIntentConfig(
                    jaw_open=(-0.50, 0.50),
                    jaw_closed=(-1e40, 1e40),
                ),
                "jaw_closed",
            ),
        ),
    )
    def test_finite_jaw_endpoints_outside_float32_range_are_rejected(
        self, config, field
    ):
        with pytest.raises(
            ValueError, match=f"{field} must lie within the finite float32 range"
        ):
            DVRKPSMJawIntentStateMachine(config)


class TestDVRKPSMClutchMath:
    """Pure absolute-pose clutch calculations."""

    def test_large_finite_quaternion_normalises_without_overflow(self):
        quaternion = normalise_quaternion_xyzw((1e308, 1e308, 1e308, 1e308))
        assert quaternion is not None
        np.testing.assert_allclose(quaternion, (0.5, 0.5, 0.5, 0.5), atol=1e-12)

    def test_rebased_position_applies_scaled_controller_delta(self):
        """The latching frame is home and later controller motion is scaled from it."""
        home = np.array([0.02, 0.00, 0.18], dtype=np.float64)
        origin = np.array([0.31, -0.12, 0.44], dtype=np.float64)
        delta = np.array([0.04, -0.01, 0.03], dtype=np.float64)
        np.testing.assert_allclose(rebased_position(origin, origin, home, 1.5), home)
        np.testing.assert_allclose(
            rebased_position(origin + delta, origin, home, 1.5), home + 1.5 * delta
        )


class TestDVRKPSMCartesianClutchStateMachine:
    """Pure Cartesian clutch behaviour shared by every simulator consumer."""

    @staticmethod
    def _kernel(
        *,
        home_position=(0.02, 0.00, 0.18),
        home_orientation=(0.0, 0.0, 0.0, 1.0),
    ) -> DVRKPSMCartesianClutchStateMachine:
        return DVRKPSMCartesianClutchStateMachine(
            DVRKPSMCartesianClutchConfig(
                home_position=home_position,
                home_orientation=home_orientation,
                workspace_lower=(-0.10, -0.10, 0.08),
                workspace_upper=(0.10, 0.10, 0.24),
                translation_scale=1.0,
            )
        )

    @staticmethod
    def _step(
        kernel,
        *,
        position=(0.30, -0.20, 0.50),
        orientation=_IDENTITY_QUAT,
        squeeze: float | None = 1.0,
        valid: bool = True,
        active: bool = True,
    ) -> np.ndarray:
        return kernel.step(
            controller_position=position,
            controller_orientation=orientation,
            squeeze=squeeze,
            tracking_valid=valid,
            session_active=active,
        )

    def test_engagement_and_reclutch_are_no_jump_absolute_deltas(self):
        kernel = self._kernel()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        home = kernel.pose

        np.testing.assert_array_equal(self._step(kernel, position=origin), home)
        moved = self._step(
            kernel,
            position=origin + (0.03, 0.01, -0.02),
            orientation=_quat_xyzw_z(0.30),
        )
        np.testing.assert_allclose(moved[:3], home[:3] + (0.03, 0.01, -0.02), atol=1e-6)
        np.testing.assert_allclose(moved[3:], _quat_xyzw_z(0.30), atol=1e-6)

        # Repositioning while unclutched cannot move either pose component.
        np.testing.assert_array_equal(
            self._step(
                kernel,
                position=(9.0, 9.0, 9.0),
                orientation=_quat_xyzw_z(-1.0),
                squeeze=0.0,
            ),
            moved,
        )
        new_origin = np.array((-0.8, 0.4, 1.2), dtype=np.float64)
        np.testing.assert_array_equal(
            self._step(
                kernel,
                position=new_origin,
                orientation=_quat_xyzw_z(-0.6),
            ),
            moved,
        )
        resumed = self._step(
            kernel,
            position=new_origin + (0.01, -0.02, 0.01),
            orientation=_quat_xyzw_z(-0.4),
        )
        np.testing.assert_allclose(
            resumed[:3], moved[:3] + (0.01, -0.02, 0.01), atol=1e-6
        )
        expected_orientation = quat_mul_xyzw(
            moved[3:].astype(np.float64), _quat_xyzw_z(0.2)
        )
        np.testing.assert_allclose(resumed[3:], expected_orientation, atol=1e-6)

    def test_non_commuting_orientation_offset_maps_z_rotation_to_negative_y(self):
        quarter_turn_about_x = _quat_xyzw_x(math.pi / 2.0)
        kernel = DVRKPSMCartesianClutchStateMachine(
            DVRKPSMCartesianClutchConfig(
                home_position=(0.02, 0.00, 0.18),
                workspace_lower=(-0.10, -0.10, 0.08),
                workspace_upper=(0.10, 0.10, 0.24),
                orientation_offset=tuple(quarter_turn_about_x),
            )
        )
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        self._step(kernel, position=origin, orientation=_IDENTITY_QUAT)

        angle = 0.4
        pose = self._step(
            kernel,
            position=origin,
            orientation=_quat_xyzw_z(angle),
        )

        # Rx(pi/2) Rz(angle) Rx(-pi/2) is a rotation of -angle about Y.
        expected = np.array(
            (0.0, -math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(pose[3:], expected, atol=1e-6)

    @pytest.mark.parametrize(
        ("config", "field"),
        (
            (
                DVRKPSMCartesianClutchConfig(
                    home_position=(1e40, 0.0, 0.18),
                ),
                "home_position",
            ),
            (
                DVRKPSMCartesianClutchConfig(
                    workspace_lower=(-1e40, -0.14, 0.06),
                ),
                "workspace_lower",
            ),
            (
                DVRKPSMCartesianClutchConfig(
                    workspace_upper=(1e40, 0.14, 0.28),
                ),
                "workspace_upper",
            ),
        ),
    )
    def test_finite_pose_configuration_outside_float32_range_is_rejected(
        self, config, field
    ):
        with pytest.raises(
            ValueError, match=f"{field} must lie within the finite float32 range"
        ):
            DVRKPSMCartesianClutchStateMachine(config)

    @pytest.mark.parametrize(
        ("bad_position", "bad_orientation"),
        (
            ((math.nan, 0.0, 0.0), _IDENTITY_QUAT),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 1.0)),
        ),
    )
    def test_invalid_pose_holds_exactly_and_requires_fresh_origin(
        self, bad_position, bad_orientation
    ):
        kernel = self._kernel()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        self._step(kernel, position=origin)
        moved = self._step(kernel, position=origin + (0.02, 0.0, 0.0))

        invalid = self._step(kernel, position=bad_position, orientation=bad_orientation)
        np.testing.assert_array_equal(invalid, moved)
        assert not kernel.engaged

        # Recovery only captures a new origin; stale deltas cannot leak across
        # the invalid interval.
        np.testing.assert_array_equal(
            self._step(kernel, position=(1.0, 1.0, 1.0)), moved
        )

    @pytest.mark.parametrize(
        "transition",
        (
            {"squeeze": 0.0},
            {"valid": False},
            {"active": False},
        ),
    )
    def test_unclutch_tracking_loss_and_inactive_session_hold_exact_pose(
        self, transition
    ):
        kernel = self._kernel()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        self._step(kernel, position=origin)
        moved = self._step(
            kernel,
            position=origin + (0.02, -0.01, 0.01),
            orientation=_quat_xyzw_z(0.2),
        )
        held = self._step(kernel, position=(8.0, 8.0, 8.0), **transition)
        np.testing.assert_array_equal(held, moved)
        assert not kernel.engaged

    def test_reset_is_deterministic_and_rearms_origin_capture(self):
        home_orientation = _quat_xyzw_z(0.35)
        kernel = self._kernel(home_orientation=tuple(home_orientation))
        home = kernel.pose
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        self._step(kernel, position=origin)
        self._step(kernel, position=origin + (0.03, 0.0, 0.0))
        np.testing.assert_array_equal(kernel.reset(), home)
        assert not kernel.engaged
        np.testing.assert_array_equal(
            self._step(kernel, position=(1.0, 1.0, 1.0)), home
        )

    def test_left_and_right_instances_are_isolated(self):
        left = self._kernel(home_position=(-0.04, 0.00, 0.18))
        right = self._kernel(home_position=(0.04, 0.00, 0.18))
        left_origin = np.array((-0.30, 0.10, 0.40), dtype=np.float64)
        right_origin = np.array((0.30, 0.10, 0.40), dtype=np.float64)
        self._step(left, position=left_origin)
        self._step(right, position=right_origin)
        right_home = right.pose

        left_moved = self._step(left, position=left_origin + (0.03, 0.0, 0.0))
        np.testing.assert_allclose(left_moved[:3], (-0.01, 0.00, 0.18), atol=1e-6)
        np.testing.assert_array_equal(right.pose, right_home)

        right_moved = self._step(
            right,
            position=right_origin + (0.0, -0.02, 0.01),
            orientation=_quat_xyzw_z(-0.2),
        )
        np.testing.assert_allclose(right_moved[:3], (0.04, -0.02, 0.19), atol=1e-6)
        np.testing.assert_array_equal(left.pose, left_moved)


class TestDVRKPSMClutchRetargeter:
    """End-to-end PSM clutch output and resilience checks."""

    @staticmethod
    def _retargeter() -> DVRKPSMClutchRetargeter:
        config = DVRKPSMClutchConfig(
            home_reference_T_ee=_make_home_transform((0.02, 0.00, 0.18)),
            workspace_lower=(-0.10, -0.10, 0.08),
            workspace_upper=(0.10, 0.10, 0.24),
            orientation_offset=(0.0, 0.0, 0.0, 1.0),
        )
        return DVRKPSMClutchRetargeter(config, name="ee_pose")

    def test_output_contract_is_absolute_7d_pose(self):
        """The DLS integration receives one position-plus-xyzw target vector."""
        pose_type = self._retargeter().output_spec()[_POSE_OUTPUT].types[0]
        assert pose_type.shape == (7,)

    def test_wrapper_matches_shared_kernel_on_fixed_controller_trace(self):
        """Retargeting-engine adaptation cannot diverge from the reusable kernel."""
        home_position = (0.02, 0.00, 0.18)
        workspace_lower = (-0.10, -0.10, 0.08)
        workspace_upper = (0.10, 0.10, 0.24)
        orientation_offset = tuple(_quat_xyzw_x(0.5))
        wrapper = DVRKPSMClutchRetargeter(
            DVRKPSMClutchConfig(
                home_reference_T_ee=_make_home_transform(home_position),
                workspace_lower=workspace_lower,
                workspace_upper=workspace_upper,
                translation_scale=1.25,
                orientation_offset=orientation_offset,
            ),
            name="ee_pose",
        )
        kernel = DVRKPSMCartesianClutchStateMachine(
            DVRKPSMCartesianClutchConfig(
                home_position=home_position,
                workspace_lower=workspace_lower,
                workspace_upper=workspace_upper,
                translation_scale=1.25,
                orientation_offset=orientation_offset,
            )
        )

        origin_orientation = _quat_xyzw_z(-0.4)
        moved_orientation = quat_mul_xyzw(origin_orientation, _quat_xyzw_z(0.3))
        recovered_orientation = _quat_xyzw_z(0.7)
        trace = (
            {
                "position": (0.30, -0.20, 0.50),
                "orientation": origin_orientation,
            },
            {
                "position": (0.33, -0.19, 0.48),
                "orientation": moved_orientation,
            },
            {
                "position": (math.nan, 0.0, 0.0),
                "orientation": _IDENTITY_QUAT,
            },
            {
                "position": (-0.6, 0.4, 1.0),
                "orientation": recovered_orientation,
            },
            {
                "position": (-0.59, 0.38, 1.01),
                "orientation": quat_mul_xyzw(recovered_orientation, _quat_xyzw_z(-0.2)),
            },
            {
                "position": (8.0, 8.0, 8.0),
                "orientation": _quat_xyzw_z(1.0),
                "squeeze": 0.0,
            },
            {
                "position": (0.2, 0.3, 0.4),
                "orientation": _quat_xyzw_z(-0.2),
                "active": False,
            },
            {
                "position": (0.9, 0.8, 0.7),
                "orientation": _quat_xyzw_z(0.6),
                "reset": True,
            },
        )

        for index, sample in enumerate(trace):
            position = np.asarray(sample["position"], dtype=np.float32)
            orientation = np.asarray(sample["orientation"], dtype=np.float32)
            squeeze = float(sample.get("squeeze", 1.0))
            active = bool(sample.get("active", True))
            reset = bool(sample.get("reset", False))

            inputs, outputs = _build_io(wrapper)
            inputs[ControllersSource.RIGHT] = _make_controller(
                grip_pos=position,
                grip_ori=orientation,
                squeeze=squeeze,
            )
            wrapper.compute(
                inputs,
                outputs,
                _make_context(
                    reset=reset,
                    state=(
                        ExecutionState.RUNNING if active else ExecutionState.STOPPED
                    ),
                ),
            )
            if reset:
                kernel.reset()
            expected = kernel.step(
                controller_position=position,
                controller_orientation=orientation,
                squeeze=squeeze,
                tracking_valid=True,
                session_active=active,
            )
            np.testing.assert_array_equal(
                _read_vector(outputs, _POSE_OUTPUT),
                expected.astype(np.float64),
                err_msg=f"trace sample {index}",
            )

    def test_out_of_workspace_configured_home_is_rejected(self):
        """A reset pose outside the target guard would violate no-jump engagement."""
        with pytest.raises(ValueError, match="must lie within"):
            DVRKPSMClutchRetargeter(
                DVRKPSMClutchConfig(
                    home_reference_T_ee=_make_home_transform((0.20, 0.0, 0.18)),
                    workspace_lower=(-0.10, -0.10, 0.08),
                    workspace_upper=(0.10, 0.10, 0.24),
                ),
                name="ee_pose",
            )

    def test_non_homogeneous_configured_home_is_rejected(self):
        """A finite 4x4 array is not a pose unless its projective row is canonical."""
        transform = _make_home_transform((0.02, 0.0, 0.18))
        transform[3] = (0.0, 0.0, 0.25, 1.0)
        with pytest.raises(ValueError, match="homogeneous transform"):
            DVRKPSMClutchRetargeter(
                DVRKPSMClutchConfig(home_reference_T_ee=transform),
                name="ee_pose",
            )

    def test_engage_motion_workspace_clamp_and_reclutch_are_no_jump(self):
        """Engage, clamp, stop, and re-engage preserve a bounded continuous target."""
        retargeter = self._retargeter()
        controller_origin = np.array([0.30, -0.20, 0.50], dtype=np.float64)

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=controller_origin)
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            (0.02, 0.00, 0.18),
            atol=1e-6,
        )

        # This position would exceed every configured bound without the Cartesian guard.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=controller_origin + np.array((1.0, -1.0, 1.0))
        )
        retargeter.compute(inputs, outputs, _make_context())
        held = _read_vector(outputs, _POSE_OUTPUT)
        np.testing.assert_allclose(held[:3], (0.10, -0.10, 0.24), atol=1e-6)

        # Repositioning the hand while stopped cannot move the virtual tool.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=(9.0, 9.0, 9.0))
        retargeter.compute(inputs, outputs, _make_context(state=ExecutionState.STOPPED))
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

        # The next running frame merely latches its new controller origin; it does not jump.
        next_origin = np.array((1.0, 1.0, 1.0), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=next_origin)
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

        # A small fresh delta now starts from the previous target rather than reset home.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=next_origin + np.array((-0.02, 0.01, -0.01))
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            (0.08, -0.09, 0.23),
            atol=1e-6,
        )

    def test_continuous_squeeze_uses_controller_delta_not_integrated_delta(self):
        """Two frames while held follow absolute displacement from one clutch origin."""
        retargeter = self._retargeter()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        home = np.array((0.02, 0.00, 0.18), dtype=np.float64)

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=origin, squeeze=1.0)
        retargeter.compute(inputs, outputs, _make_context())

        first_delta = np.array((0.03, 0.01, -0.01), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=origin + first_delta, squeeze=1.0
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            home + first_delta,
            atol=1e-6,
        )

        # This is an absolute controller location relative to ``origin``;
        # the first delta must not be applied a second time.
        second_delta = np.array((0.04, -0.02, 0.02), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=origin + second_delta, squeeze=1.0
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            home + second_delta,
            atol=1e-6,
        )

    def test_squeeze_is_deadman_clutch_and_reengagement_is_no_jump(self):
        """Release freezes the PSM; squeeze again latches a fresh origin at the held pose."""
        retargeter = self._retargeter()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)

        # Merely tracking a controller does not command motion before squeeze.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=origin, squeeze=0.0)
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            (0.02, 0.00, 0.18),
            atol=1e-6,
        )
        assert not retargeter._clutch_state.engaged

        # Squeeze latches, then a hand displacement moves the virtual tool.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=origin, squeeze=1.0)
        retargeter.compute(inputs, outputs, _make_context())
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=origin + (0.03, 0.0, 0.0), squeeze=1.0
        )
        retargeter.compute(inputs, outputs, _make_context())
        held = _read_vector(outputs, _POSE_OUTPUT)
        np.testing.assert_allclose(held[:3], (0.05, 0.00, 0.18), atol=1e-6)

        # Releasing while moving cannot change the target.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=(9.0, 9.0, 9.0), squeeze=0.0
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

        # Squeezing at the new hand pose only latches it.  Motion resumes from
        # the held PSM target, not from the configured reset home.
        new_origin = np.array((1.0, 1.0, 1.0), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=new_origin, squeeze=1.0
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

    def test_orientation_is_relative_to_squeeze_latch(self):
        """Squeeze holds the configured orientation, then follows controller-relative rotation."""
        home_angle = 0.4
        controller_angle = -0.8
        home_rotation = _rotation_matrix_z(home_angle)
        retargeter = DVRKPSMClutchRetargeter(
            DVRKPSMClutchConfig(
                home_reference_T_ee=_make_home_transform(
                    (0.02, 0.00, 0.18), home_rotation
                ),
                workspace_lower=(-0.10, -0.10, 0.08),
                workspace_upper=(0.10, 0.10, 0.24),
            ),
            name="ee_pose",
        )
        origin_orientation = _quat_xyzw_z(controller_angle)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=(0.30, -0.20, 0.50),
            grip_ori=origin_orientation,
            squeeze=1.0,
        )
        retargeter.compute(inputs, outputs, _make_context())
        home_quaternion = _quat_xyzw_z(home_angle)
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[3:], home_quaternion, atol=1e-6
        )

        relative_rotation = _quat_xyzw_z(0.3)
        current_orientation = quat_mul_xyzw(origin_orientation, relative_rotation)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=(0.30, -0.20, 0.50),
            grip_ori=current_orientation,
            squeeze=1.0,
        )
        retargeter.compute(inputs, outputs, _make_context())
        expected = quat_mul_xyzw(home_quaternion, relative_rotation)
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[3:], expected, atol=1e-6
        )

    def test_invalid_or_dropped_tracking_holds_a_finite_last_pose(self):
        """A bad validity flag, NaN position, or absent sample never poisons downstream IK."""
        retargeter = self._retargeter()
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=(0.3, 0.2, 0.5))
        retargeter.compute(inputs, outputs, _make_context())
        initial = _read_vector(outputs, _POSE_OUTPUT)

        for controller in (
            _make_controller(grip_pos=(9.0, 9.0, 9.0), grip_is_valid=False),
            _make_controller(grip_pos=(math.nan, 0.0, 0.0)),
        ):
            inputs, outputs = _build_io(retargeter)
            inputs[ControllersSource.RIGHT] = controller
            retargeter.compute(inputs, outputs, _make_context())
            emitted = _read_vector(outputs, _POSE_OUTPUT)
            assert np.all(np.isfinite(emitted))
            np.testing.assert_allclose(emitted, initial, atol=1e-6)

        inputs, outputs = _build_io(retargeter)
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT), initial, atol=1e-6
        )

    def test_malformed_pose_rearms_clutch_before_the_next_valid_frame(self):
        """A malformed valid-flagged sample cannot leave a stale clutch origin armed."""
        retargeter = self._retargeter()
        origin = np.array((0.30, -0.20, 0.50), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=origin)
        retargeter.compute(inputs, outputs, _make_context())
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=origin + (0.03, 0.0, 0.0)
        )
        retargeter.compute(inputs, outputs, _make_context())
        held = _read_vector(outputs, _POSE_OUTPUT)

        # The source claims tracking is valid but sends malformed coordinates.
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=(math.nan, 0.0, 0.0)
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

        # A distant recovered controller pose only establishes a fresh origin.
        recovered_origin = np.array((1.0, 1.0, 1.0), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=recovered_origin)
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(_read_vector(outputs, _POSE_OUTPUT), held, atol=1e-6)

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=recovered_origin + (0.01, 0.0, 0.0)
        )
        retargeter.compute(inputs, outputs, _make_context())
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            held[:3] + (0.01, 0.0, 0.0),
            atol=1e-6,
        )

    def test_reset_restores_configured_home(self):
        """An episode reset discards a prior re-clutch target and starts from configured home."""
        retargeter = self._retargeter()
        origin = np.array((0.3, 0.2, 0.5), dtype=np.float64)
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=origin)
        retargeter.compute(inputs, outputs, _make_context())
        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(
            grip_pos=origin + (0.03, 0.0, 0.0)
        )
        retargeter.compute(inputs, outputs, _make_context())

        inputs, outputs = _build_io(retargeter)
        inputs[ControllersSource.RIGHT] = _make_controller(grip_pos=(1.0, 1.0, 1.0))
        retargeter.compute(inputs, outputs, _make_context(reset=True))
        np.testing.assert_allclose(
            _read_vector(outputs, _POSE_OUTPUT)[:3],
            (0.02, 0.00, 0.18),
            atol=1e-6,
        )
