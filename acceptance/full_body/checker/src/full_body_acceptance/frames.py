# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decoded record, and the source protocol every check reads through.

A ``Frame`` is deliberately close to ``core.FullBodyPoseRecord`` rather than to the
retargeting tensor: the tensor drops the three ``DeviceDataTimestamp`` fields,
``all_joint_poses_tracked`` and record presence, so envelope checks cannot run off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from acceptance_common.frames import FrameSource, SourceMetadata

__all__ = [
    "NUM_JOINTS",
    "SCHEMA_NAME",
    "Frame",
    "FrameSource",
    "JointPose",
    "SourceMetadata",
]

NUM_JOINTS = 24

SCHEMA_NAME = "core.FullBodyPoseRecord"


@dataclass(frozen=True, slots=True)
class JointPose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    is_valid: bool


@dataclass(frozen=True, slots=True)
class Frame:
    sequence: int
    log_time_ns: int
    publish_time_ns: int
    has_payload: bool
    available_time_ns: int | None = None
    sample_time_ns: int | None = None
    device_time_ns: int | None = None
    all_joint_poses_tracked: bool | None = None
    joints: tuple[JointPose, ...] | None = None

    @property
    def has_joints(self) -> bool:
        """False both when the payload is absent and when the ``joints`` field is not set.

        ``BodyJoints`` is a struct holding ``[BodyJointPose:24]``, so the length needs no
        checking; whether the field is present on the table is the only real question.
        """
        return self.joints is not None

    def valid_joints(self) -> Iterator[tuple[int, JointPose]]:
        if self.joints is None:
            return
        for index, joint in enumerate(self.joints):
            if joint.is_valid:
                yield index, joint
