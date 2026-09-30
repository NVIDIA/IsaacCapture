# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlatBuffer encoding and MCAP writing that mirror the C++ recorder.

Pinned to ``live_hand_tracker_impl.cpp`` and ``recording_traits.hpp``:

  * schema ``core.HandPoseRecord``, encoding ``flatbuffer``, data = the .bfbs bytes
  * topics ``hands/left_hand`` then ``hands/right_hand``, one record each per update,
    both carrying the same timestamp
  * timestamp (update_time, update_time, xr_time); logTime = publishTime = available
  * an inactive hand is a record with a timestamp and no ``data``
  * profile ``teleop``, no compression
"""

from __future__ import annotations

from typing import Iterable

import flatbuffers
import toolchain
from mcap.writer import CompressionType, Writer

toolchain.ensure()

from core import DeviceDataTimestamp, HandJoints, HandPose, HandPoseRecord  # noqa: E402

from script import DEVICE_CLOCK_OFFSET_NS, HandSample  # noqa: E402

SCHEMA_NAME = "core.HandPoseRecord"
ENCODING = "flatbuffer"
BASE_NAME = "hands"
TOPICS = {"left": f"{BASE_NAME}/left_hand", "right": f"{BASE_NAME}/right_hand"}
PROFILE = "teleop"


def load_bfbs() -> bytes:
    with open(toolchain.BFBS_PATH, "rb") as fh:
        return fh.read()


def encode(sample: HandSample, available_ns: int) -> bytes:
    b = flatbuffers.Builder(1100)
    data = None
    if sample.joints is not None:
        HandPose.Start(b)
        js = sample.joints
        joints = HandJoints.CreateHandJoints(
            b,
            [j.position[0] for j in js],
            [j.position[1] for j in js],
            [j.position[2] for j in js],
            [j.orientation[0] for j in js],
            [j.orientation[1] for j in js],
            [j.orientation[2] for j in js],
            [j.orientation[3] for j in js],
            [bool(j.valid) for j in js],
            [j.radius for j in js],
        )
        HandPose.AddJoints(b, joints)
        data = HandPose.End(b)
    HandPoseRecord.Start(b)
    HandPoseRecord.AddTimestamp(
        b,
        DeviceDataTimestamp.CreateDeviceDataTimestamp(
            b, available_ns, sample.sample_ns, sample.sample_ns + DEVICE_CLOCK_OFFSET_NS
        ),
    )
    if data is not None:
        HandPoseRecord.AddData(b, data)
    b.Finish(HandPoseRecord.End(b))
    return bytes(b.Output())


def write_mcap(
    path: str,
    ticks: Iterable[tuple[int, HandSample, HandSample]],
    bfbs: bytes,
    topics: dict[str, str] = TOPICS,
) -> int:
    """Writes every tick; returns the number of records."""
    count = 0
    with open(path, "wb") as fh:
        writer = Writer(fh, compression=CompressionType.NONE)
        writer.start(profile=PROFILE, library="isaacteleop-synthetic-fixtures")
        schema = writer.register_schema(name=SCHEMA_NAME, encoding=ENCODING, data=bfbs)
        channels = {
            side: writer.register_channel(
                topic=topic, message_encoding=ENCODING, schema_id=schema
            )
            for side, topic in topics.items()
        }
        sequence = {side: 0 for side in channels}
        for _, left, right in ticks:
            for sample in (left, right):
                available = sample.sample_ns
                writer.add_message(
                    channel_id=channels[sample.side],
                    log_time=available,
                    publish_time=available,
                    sequence=sequence[sample.side],
                    data=encode(sample, available),
                )
                sequence[sample.side] += 1
                count += 1
        writer.finish()
    return count
