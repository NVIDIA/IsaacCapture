# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decoded ``core.FullBodyPoseRecord``, and the source protocol every check reads through."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from cts_common.frames import FrameSource, SourceMetadata

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
        """False when the payload is absent or the ``joints`` field is not set."""
        return self.joints is not None

    def valid_joints(self) -> Iterator[tuple[int, JointPose]]:
        if self.joints is None:
            return
        for index, joint in enumerate(self.joints):
            if joint.is_valid:
                yield index, joint
