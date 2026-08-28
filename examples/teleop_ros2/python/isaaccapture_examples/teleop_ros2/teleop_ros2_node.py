#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teleop ROS2 Reference Node.

Publishes teleoperation data over ROS2 topics using isaaccapture TeleopSession.
The `mode` parameter selects the teleoperation scenario and which topics are
published:

  - controller_teleop (default): ee_poses (from controller aim poses for native
                       or MANUS input, with MANUS calibration when selected; from
                       provider wrist poses for Wuji input),
                       root_twist, root_pose, finger_joints
                       (retargeted TriHand angles), controller_data, head_pose,
                       and TF transforms for left/right wrists and head
  - hand_teleop: ee_poses (from hand tracking wrists), hand (named left and
                 right joint poses),
                 finger_joints (retargeted Sharpa or Wuji joint angles),
                 root_twist/root_pose (from foot pedal locomotion), head_pose,
                 and TF transforms for left/right wrists and head
  - controller_raw: controller_data only
  - full_body: full_body and controller_data

Topic names (remappable via ROS 2 remapping):
  - xr_teleop/hand (NamedPoseArray): named left and right hand joint poses
  - xr_teleop/ee_poses (NamedPoseArray): named left and right EE poses
  - xr_teleop/root_twist (TwistStamped): root velocity command
  - xr_teleop/root_pose (PoseStamped): root pose command (height only)
  - xr_teleop/head_pose (PoseStamped): head pose
  - xr_teleop/controller_data (ByteMultiArray): msgpack-encoded controller data
  - xr_teleop/full_body (ByteMultiArray): msgpack-encoded full body tracking data
  - xr_teleop/finger_joints (JointState): retargeted finger joint angles

TF frames published in hand_teleop and controller_teleop modes (configurable via parameters):
  - world_frame -> right_wrist_frame
  - world_frame -> left_wrist_frame
  - world_frame -> head_frame
  - (With ee_poses_frame=head, world_frame -> head_frame -> left/right_wrist_frame)
"""

import os
import time

import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    TwistStamped,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ByteMultiArray
from teleop_ros2_interfaces.msg import NamedPoseArray
from tf2_ros import TransformBroadcaster

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.cloudxr.oob_teleop_env import TELEOP_CLIENT_ROUTE_ENV
from isaaccapture.teleop_session_manager import SessionMode, TeleopSession
from .messages import (
    build_controller_msg,
    build_ee_output_from_controllers,
    build_ee_output_from_hands,
    build_finger_joints_msg,
    build_full_body_msg,
    build_hand_msg,
    build_head_output,
    build_root_command_output,
)
from .teleop_profiles import (
    PublishType,
    SessionResult,
    resolve_teleop_profile_spec,
    validate_session_result,
)
from .node_parameters import (
    NodeParameters,
    create_node_parameters,
)
from .session_config import build_session_config


class TeleopRos2Node(Node):
    """ROS 2 node that publishes teleop data."""

    def __init__(self) -> None:
        super().__init__("teleop_ros2_node")
        self._params: NodeParameters = create_node_parameters(self)
        self._profile_spec = resolve_teleop_profile_spec(
            self._params.mode,
            self._params.resolved_hand_retargeter,
            self._params.hand_tracking_provider,
        )
        if self._profile_spec.apply_manus_controller_mount_offset:
            self.get_logger().info(
                "Applying MANUS controller mount offset after pose transform."
            )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._create_publishers()
        self._config = build_session_config(self._params)

    def _create_publishers(self) -> None:
        self._pub_hand = self.create_publisher(NamedPoseArray, "xr_teleop/hand", 10)
        self._pub_ee_poses = self.create_publisher(
            NamedPoseArray, "xr_teleop/ee_poses", 10
        )
        self._pub_root_twist = self.create_publisher(
            TwistStamped, "xr_teleop/root_twist", 10
        )
        self._pub_root_pose = self.create_publisher(
            PoseStamped, "xr_teleop/root_pose", 10
        )
        self._pub_controller = self.create_publisher(
            ByteMultiArray, "xr_teleop/controller_data", 10
        )
        self._pub_full_body = self.create_publisher(
            ByteMultiArray, "xr_teleop/full_body", 10
        )
        self._pub_finger_joints = self.create_publisher(
            JointState, "xr_teleop/finger_joints", 10
        )
        self._pub_head = self.create_publisher(PoseStamped, "xr_teleop/head_pose", 10)

    def _publish_ee_poses_from_controllers(self, result: SessionResult, now) -> None:
        ee_poses_msg, wrist_tfs = build_ee_output_from_controllers(
            result["controller_left"],
            result["controller_right"],
            now,
            self._params.world_frame,
            self._params.left_wrist_frame,
            self._params.right_wrist_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
            self._profile_spec.apply_manus_controller_mount_offset,
            head=result["head"],
            head_frame=self._params.head_frame,
            ee_poses_frame=self._params.ee_poses_frame,
        )
        self._pub_ee_poses.publish(ee_poses_msg)
        if wrist_tfs:
            self._tf_broadcaster.sendTransform(wrist_tfs)

    def _publish_controller_payload(self, result: SessionResult) -> None:
        maybe_controller_msg = build_controller_msg(
            result["controller_left"],
            result["controller_right"],
        )
        if maybe_controller_msg is not None:
            self._pub_controller.publish(maybe_controller_msg)

    def _publish_finger_joints(self, result: SessionResult, now) -> None:
        maybe_finger_joints_msg = build_finger_joints_msg(
            result["finger_joints_left"],
            result["finger_joints_right"],
            now,
            self._params.world_frame,
        )
        if maybe_finger_joints_msg is not None:
            self._pub_finger_joints.publish(maybe_finger_joints_msg)

    def _publish_full_body_payload(self, result: SessionResult) -> None:
        maybe_body_msg = build_full_body_msg(result["full_body"])
        if maybe_body_msg is not None:
            self._pub_full_body.publish(maybe_body_msg)

    def _publish_hand_poses(self, result: SessionResult, now) -> None:
        hand_msg = build_hand_msg(
            result["hand_left"],
            result["hand_right"],
            now,
            self._params.world_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
        )
        self._pub_hand.publish(hand_msg)

    def _publish_ee_poses_from_hands(self, result: SessionResult, now) -> None:
        ee_poses_msg, wrist_tfs = build_ee_output_from_hands(
            result["hand_left"],
            result["hand_right"],
            now,
            self._params.world_frame,
            self._params.left_wrist_frame,
            self._params.right_wrist_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
            head=result["head"],
            head_frame=self._params.head_frame,
            ee_poses_frame=self._params.ee_poses_frame,
        )
        self._pub_ee_poses.publish(ee_poses_msg)
        if wrist_tfs:
            self._tf_broadcaster.sendTransform(wrist_tfs)

    def _publish_head(self, result: SessionResult, now) -> None:
        maybe_head_output = build_head_output(
            result["head"],
            now,
            self._params.world_frame,
            self._params.head_frame,
            self._params.transform_rotation,
            self._params.transform_translation,
        )
        if maybe_head_output is None:
            return

        head_msg, head_tf = maybe_head_output
        self._pub_head.publish(head_msg)
        self._tf_broadcaster.sendTransform(head_tf)

    def _publish_root_command(self, result: SessionResult, now) -> None:
        maybe_root_output = build_root_command_output(
            result["root_command"],
            now,
            self._params.world_frame,
        )
        if maybe_root_output is None:
            return

        twist_msg, pose_msg = maybe_root_output
        self._pub_root_twist.publish(twist_msg)
        self._pub_root_pose.publish(pose_msg)

    def _run_session_loop(self, launcher: CloudXRLauncher | None = None) -> int:
        while rclpy.ok():
            # Confirm the runtime/WSS proxy is alive before every session
            # attempt. This also guards the no-client retry path below: each
            # retry is a new iteration here, which never reaches the inner
            # per-step check, so a dead runtime surfaces as an error instead of
            # an infinite retry.
            if launcher is not None:
                launcher.health_check()
            try:
                with TeleopSession(self._config) as session:
                    self.get_logger().info("TeleopSession started successfully")
                    while rclpy.ok():
                        # Detect a mid-session runtime death promptly while a
                        # client is actively streaming (the outer-loop check
                        # only runs between session attempts).
                        if launcher is not None:
                            launcher.health_check()

                        result = validate_session_result(
                            session.step(),
                            self._profile_spec,
                        )

                        # Keep ROS time and other callbacks updated in this
                        # manual loop so stamped messages progress with /clock.
                        rclpy.spin_once(self, timeout_sec=0.0)

                        now = self.get_clock().now().to_msg()

                        if (
                            PublishType.EE_FROM_HANDS
                            in self._profile_spec.publish_types
                        ):
                            self._publish_ee_poses_from_hands(result, now)
                        if (
                            PublishType.EE_FROM_CONTROLLERS
                            in self._profile_spec.publish_types
                        ):
                            self._publish_ee_poses_from_controllers(result, now)
                        if PublishType.HAND_POSES in self._profile_spec.publish_types:
                            self._publish_hand_poses(result, now)
                        if PublishType.ROOT_COMMAND in self._profile_spec.publish_types:
                            self._publish_root_command(result, now)
                        if (
                            PublishType.FINGER_JOINTS
                            in self._profile_spec.publish_types
                        ):
                            self._publish_finger_joints(result, now)
                        if PublishType.HEAD in self._profile_spec.publish_types:
                            self._publish_head(result, now)
                        if (
                            PublishType.CONTROLLER_PAYLOAD
                            in self._profile_spec.publish_types
                        ):
                            self._publish_controller_payload(result)
                        if (
                            PublishType.FULL_BODY_PAYLOAD
                            in self._profile_spec.publish_types
                        ):
                            self._publish_full_body_payload(result)

                        time.sleep(self._params.sleep_period_s)
            except RuntimeError as e:
                if "Failed to get OpenXR system" not in str(e):
                    raise
                # The CloudXR runtime is up but no headset/WebXR client has
                # connected yet, so xrGetSystem reports no HMD. Keep the
                # runtime alive (launcher stays open) and retry the session.
                self.get_logger().warn(
                    f"No XR client connected ({e}), retrying in 2s..."
                )
                time.sleep(2.0)

        return 0

    def run(self) -> int:
        # MCAP replay reads recorded tracker data and needs no live runtime.
        #
        # run_embedded: this node is the container's entrypoint, so there is no
        # separate service to attach to and nothing else to outlive.  It owns
        # the runtime and stops it on exit, and fails where one is already
        # serving — host_client below configures a WSS proxy, which only the
        # process that starts it can do.
        if self._params.session_mode != SessionMode.LIVE:
            return self._run_session_loop()

        os.environ[TELEOP_CLIENT_ROUTE_ENV] = self._params.cloudxr_params.client_route
        with CloudXRLauncher(
            install_dir=self._params.cloudxr_params.install_dir,
            env_config=self._params.cloudxr_params.env_config,
            accept_eula=self._params.cloudxr_params.accept_eula,
            setup_oob=self._params.cloudxr_params.setup_oob,
            usb_local=self._params.cloudxr_params.usb_local,
            host_client=True,
            run_embedded=True,
        ) as launcher:
            self.get_logger().info(
                "CloudXR runtime and WSS proxy started "
                f"(WSS log: {launcher.wss_log_path})"
            )
            return self._run_session_loop(launcher)


def main() -> int:
    rclpy.init()
    node = None
    try:
        node = TeleopRos2Node()
        return node.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
