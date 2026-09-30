# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free public-interface tests for :class:`EePoseJumpGuard`."""

from isaaccapture.retargeters import (
    EE_POSE_JUMP_GUARD_STATUS_KEY,
    EePoseJumpGuard,
    EePoseJumpGuardConfig,
    EePoseJumpGuardDisposition,
    EePoseJumpGuardState,
    EePoseJumpGuardStatusIndex,
)
from isaaccapture.retargeters.ee_pose_jump_guard import EE_POSE_KEY
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
import numpy as np
import pytest

_IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _build_io(guard):
    inputs = {}
    for name, group_type in guard.input_spec().items():
        inputs[name] = (
            OptionalTensorGroup(group_type)
            if isinstance(group_type, OptionalTensorGroupType)
            else TensorGroup(group_type)
        )
    outputs = {}
    for name, group_type in guard.output_spec().items():
        outputs[name] = (
            OptionalTensorGroup(group_type)
            if isinstance(group_type, OptionalTensorGroupType)
            else TensorGroup(group_type)
        )
    return inputs, outputs


def _context(*, reset: bool = False, time_ns: int = 0) -> ComputeContext:
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=time_ns, real_time_ns=time_ns),
        execution_events=ExecutionEvents(
            reset=reset,
            execution_state=ExecutionState.RUNNING,
        ),
    )


def _pose(outputs) -> np.ndarray:
    return np.asarray(np.from_dlpack(outputs[EE_POSE_KEY][0]), dtype=np.float64)


def _set_pose(
    guard,
    inputs,
    position,
    orientation=_IDENTITY_QUATERNION,
) -> None:
    optional_type = guard.input_spec()[EE_POSE_KEY]
    group = TensorGroup(optional_type.inner_type)
    group[0] = np.concatenate(
        [
            np.asarray(position, dtype=np.float32),
            np.asarray(orientation, dtype=np.float32),
        ]
    )
    inputs[EE_POSE_KEY] = group


def _status(outputs) -> dict[str, float | int]:
    status = outputs[EE_POSE_JUMP_GUARD_STATUS_KEY]
    return {
        "version": int(status[EePoseJumpGuardStatusIndex.VERSION]),
        "state": int(status[EePoseJumpGuardStatusIndex.STATE]),
        "disposition": int(status[EePoseJumpGuardStatusIndex.DISPOSITION]),
        "sample_dt_s": float(status[EePoseJumpGuardStatusIndex.SAMPLE_DT_S]),
        "effective_dt_s": float(status[EePoseJumpGuardStatusIndex.EFFECTIVE_DT_S]),
        "decision_threshold_m": float(
            status[EePoseJumpGuardStatusIndex.DECISION_THRESHOLD_M]
        ),
        "frame_step_m": float(status[EePoseJumpGuardStatusIndex.FRAME_STEP_M]),
        "frame_speed_m_s": float(status[EePoseJumpGuardStatusIndex.FRAME_SPEED_M_S]),
        "reference_distance_m": float(
            status[EePoseJumpGuardStatusIndex.REFERENCE_DISTANCE_M]
        ),
        "recovery_streak": int(status[EePoseJumpGuardStatusIndex.RECOVERY_STREAK]),
        "trigger_step_m": float(status[EePoseJumpGuardStatusIndex.TRIGGER_STEP_M]),
        "max_held_distance_m": float(
            status[EePoseJumpGuardStatusIndex.MAX_HELD_DISTANCE_M]
        ),
        "max_held_speed_m_s": float(
            status[EePoseJumpGuardStatusIndex.MAX_HELD_SPEED_M_S]
        ),
        "hold_frame_count": int(status[EePoseJumpGuardStatusIndex.HOLD_FRAME_COUNT]),
    }


class TestEePoseJumpGuard:
    def _guard(self, **config) -> EePoseJumpGuard:
        return EePoseJumpGuard(
            name="ee_jump_guard",
            config=EePoseJumpGuardConfig(**config),
        )

    @pytest.mark.parametrize(
        "config",
        [
            {"linear_velocity_m_s": 0.0},
            {"linear_distance_floor_m": float("nan")},
            {"nominal_dt_s": 0.0},
            {"min_sample_dt_s": 0.02, "nominal_dt_s": 0.01},
            {"nominal_dt_s": 0.06, "max_sample_dt_s": 0.05},
            {"recovery_distance_m": 0.0},
            {"recovery_consecutive_samples": 0},
        ],
    )
    def test_invalid_config_raises(self, config):
        with pytest.raises(ValueError):
            EePoseJumpGuardConfig(**config)

    def test_missing_and_invalid_before_latch_emit_none(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)

        guard.compute(inputs, outputs, _context())
        assert outputs[EE_POSE_KEY].is_none
        assert _status(outputs)["disposition"] == int(
            EePoseJumpGuardDisposition.HELD_NO_INPUT
        )

        _set_pose(guard, inputs, [0.0, 0.0, 0.0], np.zeros(4))
        guard.compute(inputs, outputs, _context(time_ns=10_000_000))
        assert outputs[EE_POSE_KEY].is_none
        assert _status(outputs)["disposition"] == int(
            EePoseJumpGuardDisposition.HELD_INVALID
        )

    def test_first_frame_latches_unchanged(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.2, 0.1, 0.3])

        guard.compute(inputs, outputs, _context())

        np.testing.assert_allclose(_pose(outputs)[:3], [0.2, 0.1, 0.3], atol=1e-6)
        status = _status(outputs)
        assert status["version"] == EePoseJumpGuard.STATUS_VERSION
        assert status["state"] == int(EePoseJumpGuardState.TRACKING)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.LATCHED)

    @pytest.mark.parametrize("step_m", [0.0216, 0.0368, 0.0426])
    def test_observed_tianji_small_steps_pass_unchanged(self, step_m):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())

        _set_pose(guard, inputs, [step_m, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=8_000_000))

        np.testing.assert_allclose(_pose(outputs)[:3], [step_m, 0.0, 0.0], atol=1e-6)
        status = _status(outputs)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.PASSED)
        assert status["decision_threshold_m"] == pytest.approx(0.08)
        assert status["frame_step_m"] == pytest.approx(step_m, abs=1e-6)

    @pytest.mark.parametrize("step_m", [0.1858, 0.61, 1.3715])
    def test_observed_large_steps_are_held(self, step_m):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())

        _set_pose(guard, inputs, [step_m, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=8_000_000))

        np.testing.assert_allclose(_pose(outputs)[:3], [0.0, 0.0, 0.0])
        status = _status(outputs)
        assert status["state"] == int(EePoseJumpGuardState.HOLDING)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.REJECTED)
        assert status["trigger_step_m"] == pytest.approx(step_m, abs=1e-6)
        assert status["hold_frame_count"] == 1

    @pytest.mark.parametrize(
        ("time_ns", "step_m", "expected_disposition"),
        [
            (50_000_000, 0.14, EePoseJumpGuardDisposition.PASSED),
            (50_000_000, 0.16, EePoseJumpGuardDisposition.REJECTED),
            (500_000_000, 0.14, EePoseJumpGuardDisposition.PASSED),
            (500_000_000, 0.16, EePoseJumpGuardDisposition.REJECTED),
        ],
    )
    def test_cadence_envelope_caps_at_fifty_ms(
        self,
        time_ns,
        step_m,
        expected_disposition,
    ):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())

        _set_pose(guard, inputs, [step_m, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=time_ns))

        status = _status(outputs)
        assert status["disposition"] == int(expected_disposition)
        assert status["decision_threshold_m"] == pytest.approx(0.15)

    @pytest.mark.parametrize("next_time_ns", [100_000_000, 50_000_000])
    def test_nonadvancing_timestamp_uses_nominal_cadence(self, next_time_ns):
        guard = self._guard(nominal_dt_s=0.01)
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=100_000_000))

        _set_pose(guard, inputs, [0.05, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=next_time_ns))

        status = _status(outputs)
        assert status["sample_dt_s"] == pytest.approx(0.01)
        assert status["effective_dt_s"] == pytest.approx(0.01)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.PASSED)

    def test_recovery_requires_three_consecutive_samples_near_reference(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())
        _set_pose(guard, inputs, [0.61, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=10_000_000))

        for frame, x_position in ((2, 0.02), (3, 0.03)):
            _set_pose(guard, inputs, [x_position, 0.0, 0.0])
            guard.compute(inputs, outputs, _context(time_ns=frame * 10_000_000))
            np.testing.assert_allclose(_pose(outputs)[:3], [0.0, 0.0, 0.0])
            status = _status(outputs)
            assert status["state"] == int(EePoseJumpGuardState.RECOVERING)
            assert status["recovery_streak"] == frame - 1

        _set_pose(guard, inputs, [0.04, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=40_000_000))

        np.testing.assert_allclose(_pose(outputs)[:3], [0.04, 0.0, 0.0], atol=1e-6)
        status = _status(outputs)
        assert status["state"] == int(EePoseJumpGuardState.TRACKING)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.RECOVERED)
        assert status["recovery_streak"] == 3
        assert status["trigger_step_m"] == pytest.approx(0.61, abs=1e-6)

    def test_recovery_bounce_resets_streak(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())
        _set_pose(guard, inputs, [0.61, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=10_000_000))

        for frame, x_position in ((2, 0.02), (3, 0.61), (4, 0.03)):
            _set_pose(guard, inputs, [x_position, 0.0, 0.0])
            guard.compute(inputs, outputs, _context(time_ns=frame * 10_000_000))

        status = _status(outputs)
        assert status["state"] == int(EePoseJumpGuardState.RECOVERING)
        assert status["recovery_streak"] == 1

    def test_persistent_distant_stream_is_never_adopted(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())

        for frame in range(1, 200):
            _set_pose(guard, inputs, [0.61, 0.0, 0.0])
            guard.compute(inputs, outputs, _context(time_ns=frame * 10_000_000))
            np.testing.assert_allclose(_pose(outputs)[:3], [0.0, 0.0, 0.0])

        status = _status(outputs)
        assert status["state"] == int(EePoseJumpGuardState.HOLDING)
        assert status["hold_frame_count"] == 199
        assert status["max_held_distance_m"] == pytest.approx(0.61, abs=1e-6)

    def test_reset_relatches_distant_pose(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.0, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())
        _set_pose(guard, inputs, [0.61, 0.0, 0.0])
        guard.compute(inputs, outputs, _context(time_ns=10_000_000))

        guard.compute(
            inputs,
            outputs,
            _context(reset=True, time_ns=20_000_000),
        )

        np.testing.assert_allclose(_pose(outputs)[:3], [0.61, 0.0, 0.0], atol=1e-6)
        status = _status(outputs)
        assert status["state"] == int(EePoseJumpGuardState.TRACKING)
        assert status["disposition"] == int(EePoseJumpGuardDisposition.LATCHED)

    def test_missing_and_invalid_inputs_hold_accepted_pose(self):
        guard = self._guard()
        inputs, outputs = _build_io(guard)
        _set_pose(guard, inputs, [0.1, 0.0, 0.0])
        guard.compute(inputs, outputs, _context())

        missing_inputs, missing_outputs = _build_io(guard)
        guard.compute(
            missing_inputs,
            missing_outputs,
            _context(time_ns=10_000_000),
        )
        np.testing.assert_allclose(
            _pose(missing_outputs)[:3], [0.1, 0.0, 0.0], atol=1e-6
        )
        assert _status(missing_outputs)["disposition"] == int(
            EePoseJumpGuardDisposition.HELD_NO_INPUT
        )

        _set_pose(guard, inputs, [0.2, 0.0, 0.0], np.zeros(4))
        guard.compute(inputs, outputs, _context(time_ns=20_000_000))
        np.testing.assert_allclose(_pose(outputs)[:3], [0.1, 0.0, 0.0], atol=1e-6)
        assert _status(outputs)["disposition"] == int(
            EePoseJumpGuardDisposition.HELD_INVALID
        )
