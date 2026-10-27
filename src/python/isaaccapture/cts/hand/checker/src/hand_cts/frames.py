# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decoded hand record.

Both hands arrive interleaved in one source; ``side`` comes from the channel topic.
Joint positions are placed STAGE coordinates; ``local`` re-expresses them in the WRIST frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

from cts_common.frames import FrameSource, SourceMetadata
from cts_common.vectors import rotate_by_inverse, sub

from .profile import WRIST

__all__ = [
    "NUM_JOINTS",
    "SCHEMA_NAME",
    "SIDES",
    "Frame",
    "FrameSource",
    "JointPose",
    "SourceMetadata",
]

NUM_JOINTS = 26

SCHEMA_NAME = "core.HandPoseRecord"

SIDES = ("left", "right")

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class JointPose:
    position: Vec3
    orientation: tuple[float, float, float, float]
    is_valid: bool
    radius: float = 0.0


@dataclass(frozen=True)
class Frame:
    sequence: int
    log_time_ns: int
    publish_time_ns: int
    has_payload: bool
    side: str | None = None
    available_time_ns: int | None = None
    sample_time_ns: int | None = None
    device_time_ns: int | None = None
    joints: tuple[JointPose, ...] | None = None

    @property
    def has_joints(self) -> bool:
        return self.joints is not None

    def valid_joints(self) -> Iterator[tuple[int, JointPose]]:
        if self.joints is None:
            return
        for index, joint in enumerate(self.joints):
            if joint.is_valid:
                yield index, joint

    def position(self, index: int) -> Vec3 | None:
        """STAGE position of a valid joint, else None."""
        if self.joints is None or not self.joints[index].is_valid:
            return None
        return self.joints[index].position

    @cached_property
    def local(self) -> tuple[Vec3 | None, ...] | None:
        """Valid joint positions in the WRIST frame; None when the wrist is invalid.

        Reads the wrist quaternion as xyzw.
        """
        if self.joints is None or not self.joints[WRIST].is_valid:
            return None
        wrist = self.joints[WRIST]
        return tuple(
            rotate_by_inverse(wrist.orientation, sub(joint.position, wrist.position))
            if joint.is_valid
            else None
            for joint in self.joints
        )
