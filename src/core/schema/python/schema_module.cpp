// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python module entry point for Isaac Teleop schema bindings.

#include <pybind11/pybind11.h>

// Include binding definitions.
#include "controller_bindings.h"
#include "full_body_bindings.h"
#include "hand_bindings.h"
#include "haptic_command_bindings.h"
#include "head_bindings.h"
#include "joint_se3_pose_bindings.h"
#include "joint_state_bindings.h"
#include "keyboard_bindings.h"
#include "message_channel_bindings.h"
#include "oak_bindings.h"
#include "oglo_tactile_bindings.h"
#include "pedals_bindings.h"
#include "plugin_device_status_bindings.h"
#include "pose_bindings.h"
#include "se3_tracker_bindings.h"
#include "timestamp_bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(_schema, m)
{
    m.doc() = "Isaac Teleop Schema - FlatBuffer message types for teleoperation";

    // Bind timestamp types (DeviceDataTimestamp struct) — must come before all other types.
    core::bind_timestamp(m);

    // Bind pose types (Point, Quaternion, Pose structs).
    core::bind_pose(m);

    // Bind head types (HeadPose table).
    core::bind_head(m);

    // Bind hand types (HandJointPose, HandJoints structs, HandPose table).
    core::bind_hand(m);

    // Bind controller types (ControllerInputState, ControllerPose structs, ControllerSnapshot table, Hand enum).
    core::bind_controller(m);

    // Bind pedals types (Generic3AxisPedalOutput table).
    core::bind_pedals(m);

    // Bind keyboard types (KeyboardOutput table) for raw keyboard press state.
    core::bind_keyboard(m);

    // Bind OGLO tactile glove types (OgloGloveSample table).
    core::bind_oglo_tactile(m);

    // Bind joint-state types (JointState, JointStateOutput tables) for generic joint-space devices.
    core::bind_joint_state(m);

    // Bind joint SE(3) pose types (JointName enum, JointSe3Pose struct, JointSe3PoseOutput table)
    // for sparse tracked-joint sources.
    core::bind_joint_se3_pose(m);

    // Bind SE3 tracker types (Se3TrackerPose table) for generic 6-DoF pose sources.
    core::bind_se3_tracker(m);

    // Bind message channel types (MessageChannelMessages table).
    core::bind_message_channel(m);

    // Bind plugin device monitoring types.
    core::bind_plugin_device_status(m);

    // Bind vendor-neutral HapticCommand table.
    core::bind_haptic_command(m);

    // Bind OAK types (StreamType enum, FrameMetadataOak table).
    core::bind_oak(m);

    // Bind full body types (BodyJointPose, BodyJoints structs, FullBodyPose table).
    core::bind_full_body(m);
}
