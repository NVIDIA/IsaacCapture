# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test data built here rather than added to the shared fixture set."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import flatbuffers
from mcap.writer import CompressionType, Writer

import full_body_cts._schema  # noqa: F401  puts generated/ on sys.path
from core import BodyJoints, DeviceDataTimestamp, FullBodyPose, FullBodyPoseRecord
from full_body_cts.frames import NUM_JOINTS, Frame, JointPose
from full_body_cts.profile import FULL_BODY

SCHEMA_NAME = "core.FullBodyPoseRecord"
TOPIC = "full_body/full_body"
PROFILE = "teleop"
PERIOD_NS = 16_666_667
CLOCK_BASE_NS = 1_000_000_000_000
DEVICE_EPOCH_NS = 55_000_000_000_000

IDENTITY = (0.0, 0.0, 0.0, 1.0)

# Child position in the parent's frame, metres, for a ~1.75 m adult in a T-pose. Forward is -Z.
REST_OFFSETS = {
    "PELVIS": (0.000, 0.980, 0.000),
    "LEFT_HIP": (-0.090, -0.020, 0.000),
    "RIGHT_HIP": (0.090, -0.020, 0.000),
    "SPINE1": (0.000, 0.100, 0.000),
    "LEFT_KNEE": (0.000, -0.420, 0.000),
    "RIGHT_KNEE": (0.000, -0.420, 0.000),
    "SPINE2": (0.000, 0.120, 0.000),
    "LEFT_ANKLE": (0.000, -0.410, 0.000),
    "RIGHT_ANKLE": (0.000, -0.410, 0.000),
    "SPINE3": (0.000, 0.130, 0.000),
    "LEFT_FOOT": (0.000, -0.070, -0.120),
    "RIGHT_FOOT": (0.000, -0.070, -0.120),
    "NECK": (0.000, 0.180, 0.000),
    "LEFT_COLLAR": (-0.040, 0.140, 0.000),
    "RIGHT_COLLAR": (0.040, 0.140, 0.000),
    "HEAD": (0.000, 0.120, 0.000),
    "LEFT_SHOULDER": (-0.130, 0.020, 0.000),
    "RIGHT_SHOULDER": (0.130, 0.020, 0.000),
    "LEFT_ELBOW": (-0.285, 0.000, 0.000),
    "RIGHT_ELBOW": (0.285, 0.000, 0.000),
    "LEFT_WRIST": (-0.255, 0.000, 0.000),
    "RIGHT_WRIST": (0.255, 0.000, 0.000),
    "LEFT_HAND": (-0.090, 0.000, 0.000),
    "RIGHT_HAND": (0.090, 0.000, 0.000),
}


def rest_positions(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """World positions of the T-pose, all rotations identity."""
    world: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * NUM_JOINTS
    for index, name in enumerate(FULL_BODY.joint_names):
        local = REST_OFFSETS[name]
        parent = FULL_BODY.parents[index]
        base = world[parent] if parent >= 0 else offset
        world[index] = tuple(base[axis] + local[axis] for axis in range(3))
    return world


def rest_skeleton(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    valid_joints: int = NUM_JOINTS,
) -> tuple[JointPose, ...]:
    return tuple(
        joint(position=position, orientation=IDENTITY, is_valid=index < valid_joints)
        for index, position in enumerate(rest_positions(offset))
    )


def joint(
    position: tuple[float, float, float] = (0.1, 1.0, 0.2),
    orientation: tuple[float, float, float, float] = IDENTITY,
    is_valid: bool = True,
) -> JointPose:
    return JointPose(position=position, orientation=orientation, is_valid=is_valid)


def frame(
    sequence: int = 0,
    *,
    joints: tuple[JointPose, ...] | None = None,
    has_payload: bool = True,
    include_joints: bool = True,
    valid_joints: int = NUM_JOINTS,
    sample_ns: int | None = None,
    available_ns: int | None = None,
    device_ns: int | None = None,
    all_tracked: bool | None = True,
) -> Frame:
    """A plausible frame; every argument exists so a test can spoil exactly one thing."""
    sample = CLOCK_BASE_NS + sequence * PERIOD_NS if sample_ns is None else sample_ns
    available = sample + 2_000_000 if available_ns is None else available_ns
    device = DEVICE_EPOCH_NS + sequence * PERIOD_NS if device_ns is None else device_ns
    if joints is None and has_payload and include_joints:
        joints = rest_skeleton(valid_joints=valid_joints)
    if not include_joints:
        joints = None
    return Frame(
        sequence=sequence,
        log_time_ns=available,
        publish_time_ns=available,
        has_payload=has_payload,
        available_time_ns=available,
        sample_time_ns=sample,
        device_time_ns=device,
        all_joint_poses_tracked=all_tracked if has_payload else None,
        joints=joints if has_payload else None,
    )


def moving_frames(
    count: int, speed_mps: float = 1.0, period_ns: int = PERIOD_NS
) -> list[Frame]:
    """Frames whose joints translate at a steady speed, for the continuity check."""
    step = speed_mps * period_ns / 1e9
    out = []
    for i in range(count):
        base = frame(
            i,
            joints=rest_skeleton(offset=(step * i, 0.0, 0.0)),
            sample_ns=CLOCK_BASE_NS + i * period_ns,
        )
        out.append(base)
    return out


def frames(count: int, **kwargs) -> list[Frame]:
    return [frame(i, **kwargs) for i in range(count)]


def with_joint(base: Frame, index: int, replacement: JointPose) -> Frame:
    assert base.joints is not None
    joints = list(base.joints)
    joints[index] = replacement
    return replace(base, joints=tuple(joints))


@dataclass
class StubSource:
    """A FrameSource over frames held in memory."""

    frame_list: list[Frame] = field(default_factory=list)
    channel_found: bool = True

    @property
    def metadata(self):
        from full_body_cts.frames import SourceMetadata

        return SourceMetadata(
            schema_name=SCHEMA_NAME if self.channel_found else None,
            schema_encoding="flatbuffer" if self.channel_found else None,
            schema_data=b"\x00" if self.channel_found else None,
            message_encoding="flatbuffer" if self.channel_found else None,
            topic=TOPIC if self.channel_found else None,
            profile=PROFILE,
            channel_found=self.channel_found,
        )

    def __iter__(self):
        return iter(self.frame_list)


def encode(frame_: Frame) -> bytes:
    """Encodes a Frame back to wire bytes, matching pack_record()'s field layout."""
    builder = flatbuffers.Builder(1400)

    data_offset = None
    if frame_.has_payload:
        FullBodyPose.Start(builder)
        if frame_.joints is not None:
            joints = frame_.joints
            offset = BodyJoints.CreateBodyJoints(
                builder,
                [j.position[0] for j in joints],
                [j.position[1] for j in joints],
                [j.position[2] for j in joints],
                [j.orientation[0] for j in joints],
                [j.orientation[1] for j in joints],
                [j.orientation[2] for j in joints],
                [j.orientation[3] for j in joints],
                [bool(j.is_valid) for j in joints],
            )
            FullBodyPose.AddJoints(builder, offset)
        FullBodyPose.AddAllJointPosesTracked(
            builder, bool(frame_.all_joint_poses_tracked)
        )
        data_offset = FullBodyPose.End(builder)

    FullBodyPoseRecord.Start(builder)
    timestamp = DeviceDataTimestamp.CreateDeviceDataTimestamp(
        builder,
        frame_.available_time_ns or 0,
        frame_.sample_time_ns or 0,
        frame_.device_time_ns or 0,
    )
    FullBodyPoseRecord.AddTimestamp(builder, timestamp)
    if data_offset is not None:
        FullBodyPoseRecord.AddData(builder, data_offset)
    builder.Finish(FullBodyPoseRecord.End(builder))
    return bytes(builder.Output())


def bfbs_bytes() -> bytes:
    path = Path(__file__).resolve().parents[1] / "build" / "bfbs" / "full_body.bfbs"
    return path.read_bytes()


def write_recording(
    path: Path,
    frame_list: list[Frame],
    *,
    schema_name: str = SCHEMA_NAME,
    topic: str = TOPIC,
) -> Path:
    """Writes an MCAP the checker cannot tell from a real one by envelope."""
    with path.open("wb") as handle:
        writer = Writer(handle, compression=CompressionType.NONE)
        writer.start(profile=PROFILE, library="full-body-cts-tests")
        schema_id = writer.register_schema(
            name=schema_name, encoding="flatbuffer", data=bfbs_bytes()
        )
        channel_id = writer.register_channel(
            topic=topic, message_encoding="flatbuffer", schema_id=schema_id
        )
        for sequence, item in enumerate(frame_list):
            writer.add_message(
                channel_id=channel_id,
                log_time=item.log_time_ns,
                publish_time=item.publish_time_ns,
                sequence=sequence,
                data=encode(item),
            )
        writer.finish()
    return path


def unit_quaternion(angle_rad: float, axis: tuple[float, float, float]) -> tuple:
    half = angle_rad / 2.0
    scale = math.sin(half)
    return (axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half))


def qmul(a: tuple, b: tuple) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qrot(q: tuple, v: tuple) -> tuple:
    x, y, z, w = q
    cross1 = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    cross2 = (
        y * cross1[2] - z * cross1[1],
        z * cross1[0] - x * cross1[2],
        x * cross1[1] - y * cross1[0],
    )
    return tuple(v[i] + 2.0 * w * cross1[i] + 2.0 * cross2[i] for i in range(3))


def posed_skeleton(
    local_rotations: dict[str, tuple] | None = None,
) -> tuple[JointPose, ...]:
    """Forward kinematics, so positions and orientations describe the same frame."""
    local_rotations = local_rotations or {}
    world_q: list[tuple] = [IDENTITY] * NUM_JOINTS
    world_p: list[tuple] = [(0.0, 0.0, 0.0)] * NUM_JOINTS
    for index, name in enumerate(FULL_BODY.joint_names):
        parent = FULL_BODY.parents[index]
        local = local_rotations.get(name, IDENTITY)
        offset = REST_OFFSETS[name]
        if parent < 0:
            world_q[index] = local
            world_p[index] = offset
        else:
            world_q[index] = qmul(world_q[parent], local)
            world_p[index] = tuple(
                world_p[parent][axis] + qrot(world_q[parent], offset)[axis]
                for axis in range(3)
            )
    return tuple(
        joint(position=world_p[i], orientation=world_q[i], is_valid=True)
        for i in range(NUM_JOINTS)
    )


def waving_frames(count: int = 300, joint_name: str = "LEFT_SHOULDER") -> list[Frame]:
    """One joint swings through 80 degrees, which is what the frame checks need."""
    out = []
    for i in range(count):
        angle = math.radians(80.0) * math.sin(2.0 * math.pi * i / 120.0)
        pose = posed_skeleton({joint_name: unit_quaternion(angle, (0.0, 0.0, 1.0))})
        out.append(frame(i, joints=pose))
    return out


def mirrored(frames_: list[Frame]) -> list[Frame]:
    """Negates x on positions and conjugates the matching quaternion components."""
    out = []
    for item in frames_:
        out.append(
            replace(
                item,
                joints=tuple(
                    joint(
                        (-j.position[0], j.position[1], j.position[2]),
                        (
                            j.orientation[0],
                            -j.orientation[1],
                            -j.orientation[2],
                            j.orientation[3],
                        ),
                        j.is_valid,
                    )
                    for j in item.joints
                ),
            )
        )
    return out


def swap_left_right(frames_: list[Frame]) -> list[Frame]:
    out = []
    for item in frames_:
        joints = list(item.joints)
        for left, right in FULL_BODY.lateral_pairs():
            joints[left], joints[right] = joints[right], joints[left]
        out.append(replace(item, joints=tuple(joints)))
    return out


def swap_indices(frames_: list[Frame], first: str, second: str) -> list[Frame]:
    a, b = FULL_BODY.index(first), FULL_BODY.index(second)
    out = []
    for item in frames_:
        joints = list(item.joints)
        joints[a], joints[b] = joints[b], joints[a]
        out.append(replace(item, joints=tuple(joints)))
    return out


def arms_down_frames(count: int = 300) -> list[Frame]:
    """An A-pose held for the whole session, elbows hanging toward the pelvis."""
    down_left = unit_quaternion(math.radians(-75.0), (0.0, 0.0, 1.0))
    down_right = unit_quaternion(math.radians(75.0), (0.0, 0.0, 1.0))
    pose = posed_skeleton({"LEFT_SHOULDER": down_left, "RIGHT_SHOULDER": down_right})
    return [frame(i, joints=pose) for i in range(count)]
