// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_pack.hpp"

#include <openxr/openxr.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

//! OpenXR hand joints; all of them are filled while a hand is active.
inline constexpr std::size_t kOpenXrHandJointCount = XR_HAND_JOINT_COUNT_EXT;

//! Radius the injected joints carry, in metres. The exoskeleton reports poses, not joint sizes.
inline constexpr float kHandJointRadius = 0.01f;

//! Flags of a joint whose pose is filled; the TRACKED bits are added once the wrist is tracked.
inline constexpr XrSpaceLocationFlags kPoseValidFlags =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
inline constexpr XrSpaceLocationFlags kPoseTrackedFlags =
    kPoseValidFlags | XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

//! One hand's skeleton, expressed in the OpenXR wrist frame: the wrist is the identity pose and every
//! other joint is an anatomical landmark placed from the glove's articulation. All 26 joints are
//! filled, because OpenXR requires every joint of an active hand to be valid.
struct HandSkeleton
{
    std::array<XrHandJointLocationEXT, kOpenXrHandJointCount> joints{};

    //! Joints filled, for the startup log.
    std::size_t filled = 0;

    //! Mechanism root frame the articulation was read from; empty when this hand is absent.
    std::string wrist_frame;

    bool tracked() const
    {
        return !wrist_frame.empty();
    }
};

//! Reads one hand's anatomy from the glove's fingertip chords in @p msg. A skeleton whose
//! @c wrist_frame is empty means the mechanism was absent, i.e. that glove is not streaming.
HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg, Hand hand);

//! Composes @p rotation onto @p pose: the rotation is applied to the pose's orientation and to its
//! position, i.e. the pose is re-expressed one frame further out.
XrPosef rotate_pose(const XrQuaternionf& rotation, const XrPosef& pose);

//! Inverse of a rigid pose, for the wrist-relative step.
XrPosef invert_pose(const XrPosef& pose);

//! Rotation whose -Z axis is @p forward (OpenXR's joint convention: -Z runs along the bone) and whose
//! roll is pinned by @p up_hint.
XrQuaternionf orientation_along(const XrVector3f& forward, const XrVector3f& up_hint);

//! Spherical interpolation between two unit quaternions, for smoothing a tracked wrist.
XrQuaternionf slerp_quaternions(const XrQuaternionf& from, const XrQuaternionf& to, float t);

//! Hamilton product, and conjugate (the inverse for a unit quaternion).
XrQuaternionf multiply_quaternions(const XrQuaternionf& a, const XrQuaternionf& b);
XrQuaternionf conjugate_quaternion(const XrQuaternionf& q);

} // namespace io_exo_glove_ros2
} // namespace plugins
