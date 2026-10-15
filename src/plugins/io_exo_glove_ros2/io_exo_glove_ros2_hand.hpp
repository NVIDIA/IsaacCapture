// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io_exo_glove_ros2_pack.hpp"

#include <openxr/openxr.h>
#include <tf2_msgs/msg/tf_message.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

//! OpenXR hand joints, including the optional ones this bridge leaves untracked.
inline constexpr std::size_t kOpenXrHandJointCount = XR_HAND_JOINT_COUNT_EXT;

//! Radius the injected joints carry, in metres. The exoskeleton reports poses, not joint sizes.
inline constexpr float kHandJointRadius = 0.01f;

//! Flags of a joint whose pose is filled; the TRACKED bits are added once the wrist is tracked.
inline constexpr XrSpaceLocationFlags kPoseValidFlags =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
inline constexpr XrSpaceLocationFlags kPoseTrackedFlags =
    kPoseValidFlags | XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

//! Joints the exoskeleton carries per hand: exactly the 21 OpenXR hand joints the conformance
//! suite judges. PALM and the four non-thumb metacarpals are optional in OpenXR, so a 21-joint
//! glove can pass, and the exoskeleton's own kinematic chain stops at the fingertips.
inline constexpr std::size_t kHandJointSlotCount = 21;

//! One OpenXR joint and the exoskeleton frames that can carry it, best candidate first. The
//! exoskeleton publishes a frame per skeletal link plus one per fingertip, so a slot resolves by
//! exact ``child_frame_id`` match; an empty candidate ends the list.
struct HandJointSlot
{
    XrHandJointEXT joint;
    std::array<const char*, 3> frames;
};

//! Slots for one hand, in OpenXR joint order. Static storage: the returned array only borrows names.
std::array<HandJointSlot, kHandJointSlotCount> make_hand_joint_slots(Hand hand);

//! One hand's skeleton as read from one TF message.
struct HandSkeleton
{
    //! Wrist-relative joints, indexed by ``XrHandJointEXT``. Every entry starts invalid, and the
    //! slots the message does not carry stay that way: absence is not a pose at the origin.
    std::array<XrHandJointLocationEXT, kOpenXrHandJointCount> joints{};

    //! Slots resolved from this message, including the wrist.
    std::size_t filled = 0;

    //! Glove frame the wrist came from (the rig's anchor); empty when this hand is absent.
    std::string wrist_frame;

    //! Anchor -> wrist offset, in the anchor's own axes. A rig constant: the wrist is a fixed point on
    //! the glove, so the caller freezes the first one it sees and passes it back, which keeps the
    //! wrist still while the fingers move. Filled when @c wrist_offset_valid.
    XrVector3f wrist_offset{ 0.0f, 0.0f, 0.0f };
    bool wrist_offset_valid = false;

    //! Wrist pose in the glove's own frame, for the one-time alignment onto the tracked wrist.
    XrPosef wrist_pose{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };

    //! Frame each slot resolved to, for the one-time startup log; empty when unresolved.
    std::array<std::string, kHandJointSlotCount> resolved;

    bool tracked() const
    {
        return !wrist_frame.empty();
    }
};

//! Reads one hand's skeleton out of @p msg, expressed relative to that hand's wrist (the wrist itself
//! is the identity pose, and every orientation is built from the bones so it carries OpenXR's joint
//! axes rather than the rig's). Returning a skeleton whose @c wrist_frame is empty means the hand is
//! absent: the exoskeleton publishes it only while the glove streams.
//!
//! @p frozen_wrist_offset is the anchor -> wrist offset a previous frame reported; pass it to keep
//! the wrist at a fixed point on the glove. With no value the offset is derived from this frame's
//! fingertip geometry and reported back in @c HandSkeleton::wrist_offset, for the caller to freeze.
HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg,
                                 Hand hand,
                                 const std::optional<XrVector3f>& frozen_wrist_offset = std::nullopt);

//! Composes @p rotation onto @p pose: the rotation is applied to the pose's orientation and to its
//! position, i.e. the pose is re-expressed one frame further out.
XrPosef rotate_pose(const XrQuaternionf& rotation, const XrPosef& pose);

//! Inverse of a rigid pose, for the wrist-relative step.
XrPosef invert_pose(const XrPosef& pose);

//! Spherical interpolation between two unit quaternions, for smoothing a tracked wrist.
XrQuaternionf slerp_quaternions(const XrQuaternionf& from, const XrQuaternionf& to, float t);

//! Hamilton product, and conjugate (the inverse for a unit quaternion).
XrQuaternionf multiply_quaternions(const XrQuaternionf& a, const XrQuaternionf& b);
XrQuaternionf conjugate_quaternion(const XrQuaternionf& q);

} // namespace io_exo_glove_ros2
} // namespace plugins
