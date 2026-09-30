# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass-or-hold discontinuity guard for absolute end-effector poses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from isaaccapture.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    FloatType,
    IntType,
    NDArrayType,
)

EE_POSE_KEY = "ee_pose"
EE_POSE_JUMP_GUARD_STATUS_KEY = "ee_pose_jump_guard_status"

_MIN_QUAT_NORM = 1e-6


class EePoseJumpGuardState(IntEnum):
    """Persistent state of :class:`EePoseJumpGuard`."""

    UNINITIALIZED = 0
    TRACKING = 1
    HOLDING = 2
    RECOVERING = 3


class EePoseJumpGuardDisposition(IntEnum):
    """Authoritative decision made for one jump-guard output frame."""

    LATCHED = 0
    PASSED = 1
    REJECTED = 2
    HELD = 3
    RECOVERING = 4
    RECOVERED = 5
    HELD_NO_INPUT = 6
    HELD_INVALID = 7


class EePoseJumpGuardStatusIndex(IntEnum):
    """Element indices of :data:`EE_POSE_JUMP_GUARD_STATUS_KEY`."""

    VERSION = 0
    STATE = 1
    DISPOSITION = 2
    SAMPLE_DT_S = 3
    EFFECTIVE_DT_S = 4
    DECISION_THRESHOLD_M = 5
    FRAME_STEP_M = 6
    FRAME_SPEED_M_S = 7
    REFERENCE_DISTANCE_M = 8
    RECOVERY_STREAK = 9
    TRIGGER_STEP_M = 10
    MAX_HELD_DISTANCE_M = 11
    MAX_HELD_SPEED_M_S = 12
    HOLD_FRAME_COUNT = 13


@dataclass
class EePoseJumpGuardConfig:
    """Configuration for cadence-aware EE-pose discontinuity rejection."""

    linear_velocity_m_s: float = 3.0
    linear_distance_floor_m: float = 0.08
    nominal_dt_s: float = 1.0 / 120.0
    min_sample_dt_s: float = 1e-4
    max_sample_dt_s: float = 0.05
    recovery_distance_m: float = 0.08
    recovery_consecutive_samples: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "linear_velocity_m_s",
            "linear_distance_floor_m",
            "nominal_dt_s",
            "min_sample_dt_s",
            "max_sample_dt_s",
            "recovery_distance_m",
        ):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0")
        if not (self.min_sample_dt_s <= self.nominal_dt_s <= self.max_sample_dt_s):
            raise ValueError(
                "require 0 < min_sample_dt_s <= nominal_dt_s <= max_sample_dt_s"
            )
        if self.recovery_consecutive_samples < 1:
            raise ValueError("recovery_consecutive_samples must be >= 1")


def _frame_dts(
    previous_ns: int | None,
    current_ns: int,
    nominal_s: float,
    minimum_s: float,
    maximum_s: float,
) -> tuple[float, float]:
    """Return raw and bounded inter-sample times for one guard decision."""
    if previous_ns is None:
        raw_s = nominal_s
    else:
        delta_s = (current_ns - previous_ns) * 1e-9
        raw_s = delta_s if delta_s > 0.0 else nominal_s
    return raw_s, min(max(raw_s, minimum_s), maximum_s)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray | None:
    """Return a normalized quaternion, or ``None`` when invalid."""
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < _MIN_QUAT_NORM:
        return None
    return quaternion / norm


class EePoseJumpGuard(BaseRetargeter):
    """Pass normal EE poses unchanged and hold linear tracking discontinuities."""

    STATUS_VERSION = 1

    def __init__(
        self,
        name: str,
        config: EePoseJumpGuardConfig | None = None,
    ) -> None:
        """Initialize a pass-or-hold EE-pose jump guard."""
        self._cfg = config if config is not None else EePoseJumpGuardConfig()
        super().__init__(name=name)
        self._state = EePoseJumpGuardState.UNINITIALIZED
        self._accepted_pose: np.ndarray | None = None
        self._last_input_pos: np.ndarray | None = None
        self._last_compute_time_ns: int | None = None
        self._recovery_streak = 0
        self._trigger_step_m = 0.0
        self._max_held_distance_m = 0.0
        self._max_held_speed_m_s = 0.0
        self._hold_frame_count = 0

    def input_spec(self) -> RetargeterIOType:
        """Require an optional absolute 7-D ``ee_pose``."""
        return {
            EE_POSE_KEY: OptionalType(
                TensorGroupType(
                    EE_POSE_KEY,
                    [
                        NDArrayType(
                            "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                        )
                    ],
                )
            )
        }

    def output_spec(self) -> RetargeterIOType:
        """Output the governed pose and frame-correlated guard status."""
        return {
            EE_POSE_KEY: OptionalType(
                TensorGroupType(
                    EE_POSE_KEY,
                    [
                        NDArrayType(
                            "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                        )
                    ],
                )
            ),
            EE_POSE_JUMP_GUARD_STATUS_KEY: TensorGroupType(
                EE_POSE_JUMP_GUARD_STATUS_KEY,
                [
                    IntType("version"),
                    IntType("state"),
                    IntType("disposition"),
                    FloatType("sample_dt_s"),
                    FloatType("effective_dt_s"),
                    FloatType("decision_threshold_m"),
                    FloatType("frame_step_m"),
                    FloatType("frame_speed_m_s"),
                    FloatType("reference_distance_m"),
                    IntType("recovery_streak"),
                    FloatType("trigger_step_m"),
                    FloatType("max_held_distance_m"),
                    FloatType("max_held_speed_m_s"),
                    IntType("hold_frame_count"),
                ],
            ),
        }

    def _clear(self) -> None:
        self._state = EePoseJumpGuardState.UNINITIALIZED
        self._accepted_pose = None
        self._last_input_pos = None
        self._last_compute_time_ns = None
        self._recovery_streak = 0
        self._trigger_step_m = 0.0
        self._max_held_distance_m = 0.0
        self._max_held_speed_m_s = 0.0
        self._hold_frame_count = 0

    def _write_status(
        self,
        outputs: RetargeterIO,
        disposition: EePoseJumpGuardDisposition,
        *,
        sample_dt_s: float,
        effective_dt_s: float,
        decision_threshold_m: float,
        frame_step_m: float = 0.0,
        frame_speed_m_s: float = 0.0,
        reference_distance_m: float = 0.0,
    ) -> None:
        status = outputs[EE_POSE_JUMP_GUARD_STATUS_KEY]
        status[EePoseJumpGuardStatusIndex.VERSION] = self.STATUS_VERSION
        status[EePoseJumpGuardStatusIndex.STATE] = int(self._state)
        status[EePoseJumpGuardStatusIndex.DISPOSITION] = int(disposition)
        status[EePoseJumpGuardStatusIndex.SAMPLE_DT_S] = sample_dt_s
        status[EePoseJumpGuardStatusIndex.EFFECTIVE_DT_S] = effective_dt_s
        status[EePoseJumpGuardStatusIndex.DECISION_THRESHOLD_M] = decision_threshold_m
        status[EePoseJumpGuardStatusIndex.FRAME_STEP_M] = frame_step_m
        status[EePoseJumpGuardStatusIndex.FRAME_SPEED_M_S] = frame_speed_m_s
        status[EePoseJumpGuardStatusIndex.REFERENCE_DISTANCE_M] = reference_distance_m
        status[EePoseJumpGuardStatusIndex.RECOVERY_STREAK] = self._recovery_streak
        status[EePoseJumpGuardStatusIndex.TRIGGER_STEP_M] = self._trigger_step_m
        status[EePoseJumpGuardStatusIndex.MAX_HELD_DISTANCE_M] = (
            self._max_held_distance_m
        )
        status[EePoseJumpGuardStatusIndex.MAX_HELD_SPEED_M_S] = self._max_held_speed_m_s
        status[EePoseJumpGuardStatusIndex.HOLD_FRAME_COUNT] = self._hold_frame_count

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Emit an accepted pose unchanged or hold the frozen accepted reference."""
        now_ns = int(context.graph_time.real_time_ns)
        if context.execution_events.reset:
            self._clear()

        sample_dt_s, effective_dt_s = _frame_dts(
            self._last_compute_time_ns,
            now_ns,
            self._cfg.nominal_dt_s,
            self._cfg.min_sample_dt_s,
            self._cfg.max_sample_dt_s,
        )
        self._last_compute_time_ns = now_ns
        decision_threshold_m = max(
            self._cfg.linear_distance_floor_m,
            self._cfg.linear_velocity_m_s * effective_dt_s,
        )
        out = outputs[EE_POSE_KEY]
        inp = inputs[EE_POSE_KEY]

        if inp.is_none:
            if self._accepted_pose is not None:
                out[0] = self._accepted_pose.astype(np.float32)
            else:
                out.set_none()
            self._write_status(
                outputs,
                EePoseJumpGuardDisposition.HELD_NO_INPUT,
                sample_dt_s=sample_dt_s,
                effective_dt_s=effective_dt_s,
                decision_threshold_m=decision_threshold_m,
            )
            return

        pose = np.asarray(np.from_dlpack(inp[0]), dtype=np.float64)
        orientation = _normalize_quaternion(pose[3:7])
        if not np.all(np.isfinite(pose[:3])) or orientation is None:
            if self._accepted_pose is not None:
                out[0] = self._accepted_pose.astype(np.float32)
            else:
                out.set_none()
            self._write_status(
                outputs,
                EePoseJumpGuardDisposition.HELD_INVALID,
                sample_dt_s=sample_dt_s,
                effective_dt_s=effective_dt_s,
                decision_threshold_m=decision_threshold_m,
            )
            return

        normalized_pose = np.concatenate([pose[:3], orientation])
        if self._accepted_pose is None:
            self._accepted_pose = normalized_pose
            self._last_input_pos = pose[:3].copy()
            self._state = EePoseJumpGuardState.TRACKING
            out[0] = normalized_pose.astype(np.float32)
            self._write_status(
                outputs,
                EePoseJumpGuardDisposition.LATCHED,
                sample_dt_s=sample_dt_s,
                effective_dt_s=effective_dt_s,
                decision_threshold_m=decision_threshold_m,
            )
            return

        frame_step_m = (
            float(np.linalg.norm(pose[:3] - self._last_input_pos))
            if self._last_input_pos is not None
            else 0.0
        )
        frame_speed_m_s = frame_step_m / effective_dt_s
        reference_distance_m = float(np.linalg.norm(pose[:3] - self._accepted_pose[:3]))
        self._last_input_pos = pose[:3].copy()

        if self._state == EePoseJumpGuardState.TRACKING:
            if frame_step_m <= decision_threshold_m:
                self._accepted_pose = normalized_pose
                out[0] = normalized_pose.astype(np.float32)
                self._write_status(
                    outputs,
                    EePoseJumpGuardDisposition.PASSED,
                    sample_dt_s=sample_dt_s,
                    effective_dt_s=effective_dt_s,
                    decision_threshold_m=decision_threshold_m,
                    frame_step_m=frame_step_m,
                    frame_speed_m_s=frame_speed_m_s,
                    reference_distance_m=reference_distance_m,
                )
                return

            self._state = EePoseJumpGuardState.HOLDING
            self._trigger_step_m = frame_step_m
            self._max_held_distance_m = reference_distance_m
            self._max_held_speed_m_s = frame_speed_m_s
            self._hold_frame_count = 1
            self._recovery_streak = 0
            out[0] = self._accepted_pose.astype(np.float32)
            self._write_status(
                outputs,
                EePoseJumpGuardDisposition.REJECTED,
                sample_dt_s=sample_dt_s,
                effective_dt_s=effective_dt_s,
                decision_threshold_m=decision_threshold_m,
                frame_step_m=frame_step_m,
                frame_speed_m_s=frame_speed_m_s,
                reference_distance_m=reference_distance_m,
            )
            return

        self._hold_frame_count += 1
        self._max_held_distance_m = max(self._max_held_distance_m, reference_distance_m)
        self._max_held_speed_m_s = max(self._max_held_speed_m_s, frame_speed_m_s)
        out[0] = self._accepted_pose.astype(np.float32)

        if reference_distance_m <= self._cfg.recovery_distance_m:
            self._state = EePoseJumpGuardState.RECOVERING
            self._recovery_streak += 1
            if self._recovery_streak >= self._cfg.recovery_consecutive_samples:
                self._accepted_pose = normalized_pose
                self._state = EePoseJumpGuardState.TRACKING
                out[0] = normalized_pose.astype(np.float32)
                self._write_status(
                    outputs,
                    EePoseJumpGuardDisposition.RECOVERED,
                    sample_dt_s=sample_dt_s,
                    effective_dt_s=effective_dt_s,
                    decision_threshold_m=decision_threshold_m,
                    frame_step_m=frame_step_m,
                    frame_speed_m_s=frame_speed_m_s,
                    reference_distance_m=reference_distance_m,
                )
                self._recovery_streak = 0
                self._trigger_step_m = 0.0
                self._max_held_distance_m = 0.0
                self._max_held_speed_m_s = 0.0
                self._hold_frame_count = 0
                return

            self._write_status(
                outputs,
                EePoseJumpGuardDisposition.RECOVERING,
                sample_dt_s=sample_dt_s,
                effective_dt_s=effective_dt_s,
                decision_threshold_m=decision_threshold_m,
                frame_step_m=frame_step_m,
                frame_speed_m_s=frame_speed_m_s,
                reference_distance_m=reference_distance_m,
            )
            return

        self._state = EePoseJumpGuardState.HOLDING
        self._recovery_streak = 0
        self._write_status(
            outputs,
            EePoseJumpGuardDisposition.HELD,
            sample_dt_s=sample_dt_s,
            effective_dt_s=effective_dt_s,
            decision_threshold_m=decision_threshold_m,
            frame_step_m=frame_step_m,
            frame_speed_m_s=frame_speed_m_s,
            reference_distance_m=reference_distance_m,
        )
