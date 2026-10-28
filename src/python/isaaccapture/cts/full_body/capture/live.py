# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frames and trigger presses out of a running ``TeleopSession``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaccapture.deviceio import TrackerVendor
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    FullBodySource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types.indices import ControllerInputIndex

from full_body_cts.frames import Frame, JointPose

# Trigger hysteresis thresholds: pressed at or above DOWN, released at or below UP.
TRIGGER_DOWN = 0.6
TRIGGER_UP = 0.3


@dataclass(frozen=True, slots=True)
class LiveStep:
    """One ``session.step()``: the record it wrote, if it wrote one, and the press."""

    frame: Frame | None
    pressed: bool


def build_pipeline(
    vendor: TrackerVendor | None = None,
) -> tuple[OutputCombiner, FullBodySource]:
    """The body and controllers channels; ``vendor=None`` selects the default ``body.pico-xr`` backend."""
    controllers = ControllersSource(name="controllers")
    body = FullBodySource(name="full_body", vendor=vendor)
    pipeline = OutputCombiner(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "full_body": body.output(FullBodySource.FULL_BODY),
        }
    )
    return pipeline, body


class LiveFrameSource:
    """Steps the session and reports what came back."""

    def __init__(self, session: Any, body: FullBodySource) -> None:
        self._session = session
        self._tracker = body.get_tracker()
        self._records = 0
        self._down = False

    @property
    def records(self) -> int:
        """Records written so far, which is the index the next one will carry."""
        return self._records

    def step(self) -> LiveStep:
        result = self._session.step()
        # An empty pose means no record was written; count records, not steps.
        pose = self._tracker.get_body_pose(self._session.deviceio_session)
        frame = None
        if pose is not None:
            frame = _frame(self._records, pose)
            self._records += 1
        return LiveStep(frame=frame, pressed=self._edge(result))

    def _edge(self, result: Any) -> bool:
        pulled = max(
            _trigger(result["controller_left"]),
            _trigger(result["controller_right"]),
        )
        if self._down:
            self._down = pulled > TRIGGER_UP
            return False
        self._down = pulled >= TRIGGER_DOWN
        return self._down


def _trigger(controller: Any) -> float:
    if controller.is_none:
        return 0.0
    return float(controller[ControllerInputIndex.TRIGGER_VALUE])


def _frame(sequence: int, pose: Any) -> Frame:
    """One live pose as a checker ``Frame``; every time field is absent or zero.

    Never substitute a local clock for the device timestamps.
    """
    joints = pose.joints
    decoded = None
    if joints is not None:
        # Copy out of the strided views that alias the serialised buffer.
        decoded = tuple(
            JointPose(
                position=(float(p[0]), float(p[1]), float(p[2])),
                orientation=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                is_valid=bool(valid),
            )
            for p, q, valid in zip(
                joints.positions, joints.orientations, joints.is_valid
            )
        )
    return Frame(
        sequence=sequence,
        log_time_ns=0,
        publish_time_ns=0,
        has_payload=True,
        all_joint_poses_tracked=bool(pose.all_joint_poses_tracked),
        joints=decoded,
    )
