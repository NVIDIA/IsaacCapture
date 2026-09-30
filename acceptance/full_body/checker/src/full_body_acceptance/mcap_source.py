# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads full-body records out of an MCAP file, in file order, by schema name.

See ``acceptance_common.mcap_reader`` for why both of those matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from acceptance_common import mcap_reader

from ._schema import FullBodyPoseRecord, Point, Pose, Quaternion
from .frames import NUM_JOINTS, SCHEMA_NAME, Frame, JointPose, SourceMetadata


def decode_record(
    data: bytes, sequence: int, log_time: int, publish_time: int
) -> Frame:
    record = FullBodyPoseRecord.GetRootAs(data, 0)

    timestamp = record.Timestamp()
    available = sample = device = None
    if timestamp is not None:
        available = timestamp.AvailableTimeLocalCommonClock()
        sample = timestamp.SampleTimeLocalCommonClock()
        device = timestamp.SampleTimeRawDeviceClock()

    pose = record.Data()
    joints = None
    all_tracked = None
    if pose is not None:
        all_tracked = bool(pose.AllJointPosesTracked())
        body = pose.Joints()
        if body is not None:
            # Nested-struct accessors fill a caller-supplied view rather than returning
            # one, so these three are reused across the loop.
            pose_view, point_view, quat_view = Pose(), Point(), Quaternion()
            decoded = []
            for i in range(NUM_JOINTS):
                joint = body.Joints(i)
                joint_pose = joint.Pose(pose_view)
                p = joint_pose.Position(point_view)
                q = joint_pose.Orientation(quat_view)
                decoded.append(
                    JointPose(
                        position=(p.X(), p.Y(), p.Z()),
                        orientation=(q.X(), q.Y(), q.Z(), q.W()),
                        is_valid=bool(joint.IsValid()),
                    )
                )
            joints = tuple(decoded)

    return Frame(
        sequence=sequence,
        log_time_ns=log_time,
        publish_time_ns=publish_time,
        has_payload=pose is not None,
        available_time_ns=available,
        sample_time_ns=sample,
        device_time_ns=device,
        all_joint_poses_tracked=all_tracked,
        joints=joints,
    )


class McapFrameSource:
    def __init__(self, path: str | Path, schema_name: str = SCHEMA_NAME) -> None:
        self._path = Path(path)
        self._schema_name = schema_name
        self._metadata = mcap_reader.scan(self._path, schema_name)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def __iter__(self) -> Iterator[Frame]:
        if not self._metadata.channel_found:
            return
        for _, record in mcap_reader.messages(self._path, self._schema_name):
            yield decode_record(
                record.data,
                record.sequence,
                record.log_time,
                record.publish_time,
            )
