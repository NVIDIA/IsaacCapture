# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads both hands' records out of an MCAP file, in file order, by schema name."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

from cts_common import mcap_reader

from ._schema import HandPoseRecord
from .frames import NUM_JOINTS, SCHEMA_NAME, SIDES, Frame, JointPose, SourceMetadata

# core.HandJointPose: Pose{Point xyz, Quaternion xyzw}, bool is_valid, 3 pad, float radius.
_JOINT = struct.Struct("<7f?3xf")


def side_of(topic: str) -> str | None:
    """``hands/left_hand`` -> ``left``. The prefix is the recording's source name."""
    leaf = topic.rsplit("/", 1)[-1]
    for side in SIDES:
        if leaf.startswith(side):
            return side
    return None


def decode_record(
    data: bytes,
    sequence: int,
    log_time: int,
    publish_time: int,
    side: str | None,
) -> Frame:
    record = HandPoseRecord.GetRootAs(data, 0)

    timestamp = record.Timestamp()
    available = sample = device = None
    if timestamp is not None:
        available = timestamp.AvailableTimeLocalCommonClock()
        sample = timestamp.SampleTimeLocalCommonClock()
        device = timestamp.SampleTimeRawDeviceClock()

    pose = record.Data()
    joints = None
    if pose is not None:
        hand = pose.Joints()
        if hand is not None:
            start = hand._tab.Pos
            raw = bytes(hand._tab.Bytes[start : start + NUM_JOINTS * _JOINT.size])
            joints = tuple(
                JointPose(
                    position=(px, py, pz),
                    orientation=(qx, qy, qz, qw),
                    is_valid=bool(valid),
                    radius=radius,
                )
                for px, py, pz, qx, qy, qz, qw, valid, radius in _JOINT.iter_unpack(raw)
            )

    return Frame(
        sequence=sequence,
        log_time_ns=log_time,
        publish_time_ns=publish_time,
        has_payload=pose is not None,
        side=side,
        available_time_ns=available,
        sample_time_ns=sample,
        device_time_ns=device,
        joints=joints,
    )


class McapFrameSource:
    def __init__(self, path: str | Path, schema_name: str = SCHEMA_NAME) -> None:
        self._path = Path(path)
        self._schema_name = schema_name
        self._metadata = mcap_reader.scan(self._path, schema_name)
        self.topics = mcap_reader.topics(self._path, schema_name)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def __iter__(self) -> Iterator[Frame]:
        if not self._metadata.channel_found:
            return
        for topic, record in mcap_reader.messages(self._path, self._schema_name):
            yield decode_record(
                record.data,
                record.sequence,
                record.log_time,
                record.publish_time,
                side_of(topic),
            )
