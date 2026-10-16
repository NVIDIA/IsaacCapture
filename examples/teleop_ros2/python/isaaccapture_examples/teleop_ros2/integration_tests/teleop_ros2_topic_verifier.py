#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify teleop_ros2_node.py publishes expected topics for one mode."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from functools import partial

import msgpack
import rclpy
from ..constants import (
    HAND_RETARGETERS,
    HAND_TRACKING_PROVIDERS,
    LEFT_SHARPA_WAVE_JOINT_NAMES,
    LEFT_WUJI_HAND_JOINT_NAMES,
    RIGHT_SHARPA_WAVE_JOINT_NAMES,
    RIGHT_WUJI_HAND_JOINT_NAMES,
    TELEOP_MODES,
)
from geometry_msgs.msg import PoseStamped, TwistStamped
from isaaccapture.retargeting_engine.tensor_types.indices import (
    BodyJointIndex,
    HandJointIndex,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ByteMultiArray
from teleop_ros2_interfaces.msg import NamedPoseArray
from tf2_msgs.msg import TFMessage

_EXPECTED_TF_FRAMES = {"right_wrist", "left_wrist", "head"}
_EXPECTED_BODY_JOINT_NAMES = [joint.name for joint in BodyJointIndex]
_HAND_POSE_NAMES = [
    HandJointIndex(i).name
    for i in range(HandJointIndex.WRIST, HandJointIndex.LITTLE_TIP + 1)
]
_EXPECTED_HAND_POSE_NAMES = [
    f"{side}_{name}" for side in ("left", "right") for name in _HAND_POSE_NAMES
]


def _is_finite_sequence(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _byte_multi_array_to_bytes(msg: ByteMultiArray) -> bytes:
    payload = bytearray()
    for value in msg.data:
        if isinstance(value, bytes):
            payload.extend(value)
        else:
            payload.append((int(value) + 256) % 256)
    return bytes(payload)


def _unpack_msgpack(msg: ByteMultiArray) -> dict:
    return msgpack.unpackb(
        _byte_multi_array_to_bytes(msg), raw=False, strict_map_key=False
    )


def _assert_named_pose_array(msg: NamedPoseArray, expected_names: list[str]) -> None:
    if msg.header.frame_id != "world":
        raise ValueError(f"unexpected frame_id {msg.header.frame_id!r}")
    names = list(msg.name)
    if names != expected_names:
        raise ValueError("named pose entries do not match the expected order")
    if len(set(names)) != len(names):
        raise ValueError("named pose entries are not unique")
    if len(msg.pose) != len(names) or len(msg.is_valid) != len(names):
        raise ValueError("named pose arrays differ in length")

    for pose in msg.pose:
        position = (pose.position.x, pose.position.y, pose.position.z)
        orientation = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not _is_finite_sequence(position):
            raise ValueError("named pose position contains non-finite values")
        if not _is_finite_sequence(orientation):
            raise ValueError("named pose orientation contains non-finite values")


def _assert_nonzero_pose_positions(msg: NamedPoseArray, label: str) -> None:
    if not any(
        abs(value) > 1e-6
        for pose in msg.pose
        for value in (pose.position.x, pose.position.y, pose.position.z)
    ):
        raise ValueError(f"{label} positions are all zero")


def _assert_noncollinear_positions(
    positions: list[tuple[float, float, float]], label: str
) -> None:
    if len(positions) < 3:
        raise ValueError(f"{label} needs at least three valid positions")

    anchor = positions[0]
    vectors = [
        (
            position[0] - anchor[0],
            position[1] - anchor[1],
            position[2] - anchor[2],
        )
        for position in positions[1:]
    ]
    for index, left in enumerate(vectors):
        left_norm_sq = sum(value * value for value in left)
        for right in vectors[index + 1 :]:
            right_norm_sq = sum(value * value for value in right)
            norm_product = left_norm_sq * right_norm_sq
            if norm_product <= 1e-12:
                continue
            cross = (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            cross_norm_sq = sum(value * value for value in cross)
            if cross_norm_sq > norm_product * 1e-8:
                return

    raise ValueError(f"{label} positions are collinear")


def _assert_ee_poses_array(
    msg: NamedPoseArray,
    *,
    mode: str,
    hand_tracking_provider: str,
) -> None:
    _assert_named_pose_array(msg, ["left", "right"])
    if not all(bool(is_valid) for is_valid in msg.is_valid):
        raise ValueError("EE pose array contains an invalid entry")
    _assert_nonzero_pose_positions(msg, "EE pose")

    if mode != "controller_teleop" or hand_tracking_provider == "wuji":
        expected_xy = ((-0.25, 1.10), (0.25, 1.10))
    elif hand_tracking_provider == "manus":
        expected_xy = (
            (-0.1896684528, 1.255536544),
            (0.2103315472, 1.144463456),
        )
    else:
        expected_xy = ((-0.20, 1.20), (0.20, 1.20))
    for side, pose, (expected_x, expected_y) in zip(
        ("left", "right"), msg.pose, expected_xy, strict=True
    ):
        if not math.isclose(pose.position.x, expected_x, abs_tol=1e-5):
            raise ValueError(
                f"{side} EE pose x={pose.position.x} does not match "
                f"the {hand_tracking_provider} provider fixture value {expected_x}"
            )
        if not math.isclose(pose.position.y, expected_y, abs_tol=1e-5):
            raise ValueError(
                f"{side} EE pose y={pose.position.y} does not match "
                f"the {hand_tracking_provider} provider fixture value {expected_y}"
            )


def _assert_hand_pose_array(msg: NamedPoseArray) -> None:
    _assert_named_pose_array(msg, _EXPECTED_HAND_POSE_NAMES)
    if not any(bool(is_valid) for is_valid in msg.is_valid):
        raise ValueError("all hand joint poses are invalid")
    _assert_nonzero_pose_positions(msg, "hand joint pose")
    poses_per_hand = len(_HAND_POSE_NAMES)
    for side_index, side in enumerate(("left", "right")):
        start = side_index * poses_per_hand
        stop = start + poses_per_hand
        positions = [
            (pose.position.x, pose.position.y, pose.position.z)
            for pose, is_valid in zip(
                msg.pose[start:stop], msg.is_valid[start:stop], strict=True
            )
            if bool(is_valid)
        ]
        _assert_noncollinear_positions(positions, f"{side} hand joint")


def _assert_pose_stamped(
    msg: PoseStamped, *, require_nonzero_position: bool = False
) -> None:
    if msg.header.frame_id != "world":
        raise ValueError(f"unexpected frame_id {msg.header.frame_id!r}")
    position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    orientation = (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    )
    if not _is_finite_sequence(position):
        raise ValueError("PoseStamped position contains non-finite values")
    if not _is_finite_sequence(orientation):
        raise ValueError("PoseStamped orientation contains non-finite values")
    if require_nonzero_position and not any(abs(value) > 1e-6 for value in position):
        raise ValueError("PoseStamped position is all zero")


def _assert_joint_state(
    msg: JointState,
    *,
    expected_names: list[str] | None = None,
    require_nonzero: bool = False,
) -> None:
    if msg.header.frame_id != "world":
        raise ValueError(f"unexpected frame_id {msg.header.frame_id!r}")
    names = list(msg.name)
    if not names:
        raise ValueError("JointState names are empty")
    if len(set(names)) != len(names):
        raise ValueError("JointState names are not unique")
    if expected_names is not None and names != expected_names:
        raise ValueError("JointState names do not match the expected order")
    if len(msg.position) != len(names):
        raise ValueError("JointState names and positions differ in length")
    if not _is_finite_sequence(msg.position):
        raise ValueError("JointState positions contain non-finite values")
    if require_nonzero and not any(abs(float(value)) > 1e-6 for value in msg.position):
        raise ValueError("JointState positions are all zero")


def _assert_twist(msg: TwistStamped) -> None:
    if msg.header.frame_id != "world":
        raise ValueError(f"unexpected frame_id {msg.header.frame_id!r}")
    values = (
        msg.twist.linear.x,
        msg.twist.linear.y,
        msg.twist.linear.z,
        msg.twist.angular.x,
        msg.twist.angular.y,
        msg.twist.angular.z,
    )
    if not _is_finite_sequence(values):
        raise ValueError("TwistStamped contains non-finite values")
    if not any(abs(float(value)) > 1e-6 for value in values):
        raise ValueError("TwistStamped is all zero")


def _assert_controller_payload(msg: ByteMultiArray) -> None:
    payload = _unpack_msgpack(msg)
    for side in ("left", "right"):
        if payload[f"{side}_is_active"] is not True:
            raise ValueError(f"{side} controller is not active")
        for field in ("aim_position", "grip_position"):
            values = payload[f"{side}_{field}"]
            if len(values) != 3:
                raise ValueError(f"{side}_{field} must have 3 values")
            if not _is_finite_sequence(values):
                raise ValueError(f"{side}_{field} contains non-finite values")
            if not any(abs(float(value)) > 1e-6 for value in values):
                raise ValueError(f"{side}_{field} is all zero")
        for field in ("aim_orientation", "grip_orientation"):
            values = payload[f"{side}_{field}"]
            if len(values) != 4:
                raise ValueError(f"{side}_{field} must have 4 values")
            if not _is_finite_sequence(values):
                raise ValueError(f"{side}_{field} contains non-finite values")
        thumbstick = payload[f"{side}_thumbstick"]
        if len(thumbstick) != 2:
            raise ValueError(f"{side}_thumbstick must have 2 values")
        if not _is_finite_sequence(thumbstick):
            raise ValueError(f"{side}_thumbstick contains non-finite values")
        if not 0.0 <= float(payload[f"{side}_trigger_value"]) <= 1.0:
            raise ValueError(f"{side}_trigger_value must be in [0, 1]")
        if not 0.0 <= float(payload[f"{side}_squeeze_value"]) <= 1.0:
            raise ValueError(f"{side}_squeeze_value must be in [0, 1]")


def _assert_full_body_payload(msg: ByteMultiArray) -> None:
    payload = _unpack_msgpack(msg)
    if payload["joint_names"] != _EXPECTED_BODY_JOINT_NAMES:
        raise ValueError("full-body joint names do not match the expected order")
    if len(payload["joint_positions"]) != 24:
        raise ValueError("expected 24 full-body joint positions")
    if len(payload["joint_orientations"]) != 24:
        raise ValueError("expected 24 full-body joint orientations")
    if len(payload["joint_valid"]) != 24:
        raise ValueError("expected 24 full-body joint valid flags")
    if not any(bool(value) for value in payload["joint_valid"]):
        raise ValueError("all full-body joints are invalid")
    for values in payload["joint_positions"]:
        if len(values) != 3:
            raise ValueError("full-body joint position must have 3 values")
        if not _is_finite_sequence(values):
            raise ValueError("full-body joint position contains non-finite values")
    for values in payload["joint_orientations"]:
        if len(values) != 4:
            raise ValueError("full-body joint orientation must have 4 values")
        if not _is_finite_sequence(values):
            raise ValueError("full-body joint orientation contains non-finite values")
    valid_positions = [
        tuple(float(value) for value in position)
        for position, is_valid in zip(
            payload["joint_positions"], payload["joint_valid"], strict=True
        )
        if bool(is_valid)
    ]
    _assert_noncollinear_positions(valid_positions, "full-body joint")


def _finger_joint_validator(hand_retargeter: str) -> Callable:
    if hand_retargeter == "pink_ik":
        expected_names = LEFT_SHARPA_WAVE_JOINT_NAMES + RIGHT_SHARPA_WAVE_JOINT_NAMES
        samples: list[list[float]] = []

        def _validate_pink_ik(msg: JointState) -> None:
            _assert_joint_state(
                msg,
                expected_names=expected_names,
                require_nonzero=True,
            )
            samples.append([float(value) for value in msg.position])
            if len(samples) < 3:
                raise ValueError("waiting for three pink_ik joint samples")
            if not any(
                abs(value - initial) > 1e-4
                for sample in samples[1:]
                for value, initial in zip(sample, samples[0], strict=True)
            ):
                raise ValueError("pink_ik finger angles did not change over time")

        return _validate_pink_ik
    if hand_retargeter == "wuji":
        return partial(
            _assert_joint_state,
            expected_names=(LEFT_WUJI_HAND_JOINT_NAMES + RIGHT_WUJI_HAND_JOINT_NAMES),
            require_nonzero=True,
        )
    return _assert_joint_state


class TopicVerifier(Node):
    """Small ROS 2 node that waits for mode-specific verified messages."""

    def __init__(
        self,
        mode: str,
        hand_retargeter: str,
        hand_tracking_provider: str,
    ) -> None:
        super().__init__("teleop_ros2_topic_verifier")
        self._pending: set[str] = set()
        self._errors: dict[str, str] = {}
        self._seen_tf_frames: set[str] = set()

        for name, topic, msg_type, validator in self._expected_subscriptions(
            mode, hand_retargeter, hand_tracking_provider
        ):
            self._pending.add(name)
            self.create_subscription(
                msg_type, topic, self._make_callback(name, validator), 10
            )

        if mode in ("controller_teleop", "hand_teleop"):
            self._pending.add("tf")
            self.create_subscription(TFMessage, "/tf", self._tf_callback, 100)

    @property
    def pending(self) -> set[str]:
        return set(self._pending)

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def _make_callback(self, name: str, validator: Callable) -> Callable:
        def _callback(msg) -> None:
            if name not in self._pending:
                return
            try:
                validator(msg)
            except ValueError as exc:
                self._errors[name] = str(exc)
                return
            self._errors.pop(name, None)
            self._pending.discard(name)

        return _callback

    def _tf_callback(self, msg: TFMessage) -> None:
        if "tf" not in self._pending:
            return
        for transform in msg.transforms:
            if transform.header.frame_id == "world":
                self._seen_tf_frames.add(transform.child_frame_id)
        missing = _EXPECTED_TF_FRAMES - self._seen_tf_frames
        if missing:
            self._errors["tf"] = f"missing TF frames: {sorted(missing)}"
            return
        self._errors.pop("tf", None)
        self._pending.discard("tf")

    def _expected_subscriptions(
        self,
        mode: str,
        hand_retargeter: str,
        hand_tracking_provider: str,
    ) -> list[tuple[str, str, type, Callable]]:
        ee_validator = partial(
            _assert_ee_poses_array,
            mode=mode,
            hand_tracking_provider=hand_tracking_provider,
        )
        if mode == "controller_teleop":
            subscriptions = [
                (
                    "ee_poses",
                    "xr_teleop/ee_poses",
                    NamedPoseArray,
                    ee_validator,
                ),
                ("root_twist", "xr_teleop/root_twist", TwistStamped, _assert_twist),
                ("root_pose", "xr_teleop/root_pose", PoseStamped, _assert_pose_stamped),
                ("head_pose", "xr_teleop/head_pose", PoseStamped, _assert_pose_stamped),
                (
                    "finger_joints",
                    "xr_teleop/finger_joints",
                    JointState,
                    _finger_joint_validator(hand_retargeter),
                ),
                (
                    "controller_data",
                    "xr_teleop/controller_data",
                    ByteMultiArray,
                    _assert_controller_payload,
                ),
            ]
            if hand_retargeter in ("dexpilot", "pink_ik", "wuji"):
                subscriptions.insert(
                    0,
                    (
                        "hand",
                        "xr_teleop/hand",
                        NamedPoseArray,
                        _assert_hand_pose_array,
                    ),
                )
            return subscriptions
        if mode == "hand_teleop":
            return [
                (
                    "hand",
                    "xr_teleop/hand",
                    NamedPoseArray,
                    _assert_hand_pose_array,
                ),
                (
                    "ee_poses",
                    "xr_teleop/ee_poses",
                    NamedPoseArray,
                    ee_validator,
                ),
                ("root_twist", "xr_teleop/root_twist", TwistStamped, _assert_twist),
                ("root_pose", "xr_teleop/root_pose", PoseStamped, _assert_pose_stamped),
                ("head_pose", "xr_teleop/head_pose", PoseStamped, _assert_pose_stamped),
                (
                    "finger_joints",
                    "xr_teleop/finger_joints",
                    JointState,
                    _finger_joint_validator(hand_retargeter),
                ),
            ]
        if mode == "controller_raw":
            return [
                (
                    "controller_data",
                    "xr_teleop/controller_data",
                    ByteMultiArray,
                    _assert_controller_payload,
                ),
            ]
        if mode == "full_body":
            return [
                (
                    "full_body",
                    "xr_teleop/full_body",
                    ByteMultiArray,
                    _assert_full_body_payload,
                ),
                (
                    "controller_data",
                    "xr_teleop/controller_data",
                    ByteMultiArray,
                    _assert_controller_payload,
                ),
            ]
        raise ValueError(f"Unsupported mode: {mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TELEOP_MODES, required=True)
    parser.add_argument(
        "--hand-retargeter",
        choices=HAND_RETARGETERS,
        default="mode_default",
    )
    parser.add_argument(
        "--hand-tracking-provider",
        choices=HAND_TRACKING_PROVIDERS,
        default="native",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init()
    verifier = TopicVerifier(
        args.mode,
        args.hand_retargeter,
        args.hand_tracking_provider,
    )
    try:
        deadline = time.monotonic() + args.timeout
        while verifier.pending and time.monotonic() < deadline:
            rclpy.spin_once(verifier, timeout_sec=0.1)

        if verifier.pending:
            print(
                f"Timed out waiting for verified {args.mode} topics: {sorted(verifier.pending)}",
                file=sys.stderr,
            )
            for name, error in sorted(verifier.errors.items()):
                print(f"  {name}: {error}", file=sys.stderr)
            return 1

        print(
            "Verified teleop_ros2 topics for "
            f"mode {args.mode}, hand retargeter {args.hand_retargeter}, "
            f"and hand-tracking provider {args.hand_tracking_provider}"
        )
        return 0
    finally:
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
