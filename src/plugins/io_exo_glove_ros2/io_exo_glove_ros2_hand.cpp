// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_hand.hpp"

#include <oxr_utils/math.hpp>

#include <cmath>
#include <cstddef>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! Frame candidates per OpenXR joint, best first; "" ends the list. The exoskeleton publishes one
//! frame per skeletal link (``link_<Side>Skeleton<Part>``), one per fingertip
//! (``<side>_<finger>_tip``) and the hand root (``joint_<Side>SkeletonBase``), so a slot resolves by
//! exact ``child_frame_id`` match. The finger chains are ``<Part>1..4`` while OpenXR names only
//! PROXIMAL/INTERMEDIATE/DISTAL distinctly plus a fingertip, so the three links come from 1..3 and
//! the tip from the fingertip frame, which states what it is. The exoskeleton names the little
//! finger "pinky" in its frames and "LITTLE" in OpenXR, so that one tip accepts both spellings.
//!
//! If a rig numbers its links differently, this table is the only thing to change; the plugin logs
//! every slot it resolved, so a take's log shows what each joint actually bound to.
//! Palm length the synthesised wrist sits behind the glove's knuckle line, in metres. An adult palm
//! is 8-11 cm; the conformance suite only needs a plausible one to measure proportions against.
constexpr float kGlovePalmLength = 0.09f;

constexpr HandJointSlot kLeftSlots[kHandJointSlotCount] = {
    { XR_HAND_JOINT_WRIST_EXT, { "joint_LeftSkeletonBase", "left_hand", "joint_left_hand" } },
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, { "link_LeftSkeletonThumbBase", "", "" } },
    { XR_HAND_JOINT_THUMB_PROXIMAL_EXT, { "link_LeftSkeletonThumb1", "", "" } },
    { XR_HAND_JOINT_THUMB_DISTAL_EXT, { "link_LeftSkeletonThumb2", "", "" } },
    { XR_HAND_JOINT_THUMB_TIP_EXT, { "left_thumb_tip", "", "" } },
    { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, { "link_LeftSkeletonIndex1", "", "" } },
    { XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, { "link_LeftSkeletonIndex2", "", "" } },
    { XR_HAND_JOINT_INDEX_DISTAL_EXT, { "link_LeftSkeletonIndex3", "", "" } },
    { XR_HAND_JOINT_INDEX_TIP_EXT, { "left_index_tip", "", "" } },
    { XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, { "link_LeftSkeletonMiddle1", "", "" } },
    { XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT, { "link_LeftSkeletonMiddle2", "", "" } },
    { XR_HAND_JOINT_MIDDLE_DISTAL_EXT, { "link_LeftSkeletonMiddle3", "", "" } },
    { XR_HAND_JOINT_MIDDLE_TIP_EXT, { "left_middle_tip", "", "" } },
    { XR_HAND_JOINT_RING_PROXIMAL_EXT, { "link_LeftSkeletonRing1", "", "" } },
    { XR_HAND_JOINT_RING_INTERMEDIATE_EXT, { "link_LeftSkeletonRing2", "", "" } },
    { XR_HAND_JOINT_RING_DISTAL_EXT, { "link_LeftSkeletonRing3", "", "" } },
    { XR_HAND_JOINT_RING_TIP_EXT, { "left_ring_tip", "", "" } },
    { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, { "link_LeftSkeletonPinky1", "", "" } },
    { XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT, { "link_LeftSkeletonPinky2", "", "" } },
    { XR_HAND_JOINT_LITTLE_DISTAL_EXT, { "link_LeftSkeletonPinky3", "", "" } },
    { XR_HAND_JOINT_LITTLE_TIP_EXT, { "left_pinky_tip", "left_little_tip", "" } },
};

constexpr HandJointSlot kRightSlots[kHandJointSlotCount] = {
    { XR_HAND_JOINT_WRIST_EXT, { "joint_RightSkeletonBase", "right_hand", "joint_right_hand" } },
    { XR_HAND_JOINT_THUMB_METACARPAL_EXT, { "link_RightSkeletonThumbBase", "", "" } },
    { XR_HAND_JOINT_THUMB_PROXIMAL_EXT, { "link_RightSkeletonThumb1", "", "" } },
    { XR_HAND_JOINT_THUMB_DISTAL_EXT, { "link_RightSkeletonThumb2", "", "" } },
    { XR_HAND_JOINT_THUMB_TIP_EXT, { "right_thumb_tip", "", "" } },
    { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, { "link_RightSkeletonIndex1", "", "" } },
    { XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, { "link_RightSkeletonIndex2", "", "" } },
    { XR_HAND_JOINT_INDEX_DISTAL_EXT, { "link_RightSkeletonIndex3", "", "" } },
    { XR_HAND_JOINT_INDEX_TIP_EXT, { "right_index_tip", "", "" } },
    { XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, { "link_RightSkeletonMiddle1", "", "" } },
    { XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT, { "link_RightSkeletonMiddle2", "", "" } },
    { XR_HAND_JOINT_MIDDLE_DISTAL_EXT, { "link_RightSkeletonMiddle3", "", "" } },
    { XR_HAND_JOINT_MIDDLE_TIP_EXT, { "right_middle_tip", "", "" } },
    { XR_HAND_JOINT_RING_PROXIMAL_EXT, { "link_RightSkeletonRing1", "", "" } },
    { XR_HAND_JOINT_RING_INTERMEDIATE_EXT, { "link_RightSkeletonRing2", "", "" } },
    { XR_HAND_JOINT_RING_DISTAL_EXT, { "link_RightSkeletonRing3", "", "" } },
    { XR_HAND_JOINT_RING_TIP_EXT, { "right_ring_tip", "", "" } },
    { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, { "link_RightSkeletonPinky1", "", "" } },
    { XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT, { "link_RightSkeletonPinky2", "", "" } },
    { XR_HAND_JOINT_LITTLE_DISTAL_EXT, { "link_RightSkeletonPinky3", "", "" } },
    { XR_HAND_JOINT_LITTLE_TIP_EXT, { "right_pinky_tip", "right_little_tip", "" } },
};

//! First transform matching any of @p frames, honouring the candidate order.
const geometry_msgs::msg::TransformStamped* find_frame(const tf2_msgs::msg::TFMessage& msg,
                                                       const std::array<const char*, 3>& frames)
{
    for (const char* wanted : frames)
    {
        if (wanted == nullptr || *wanted == '\0')
        {
            break;
        }
        for (const auto& transform : msg.transforms)
        {
            if (transform.child_frame_id == wanted)
            {
                return &transform;
            }
        }
    }
    return nullptr;
}

XrPosef to_pose(const geometry_msgs::msg::Transform& transform)
{
    XrPosef pose{};
    pose.orientation.x = static_cast<float>(transform.rotation.x);
    pose.orientation.y = static_cast<float>(transform.rotation.y);
    pose.orientation.z = static_cast<float>(transform.rotation.z);
    pose.orientation.w = static_cast<float>(transform.rotation.w);
    pose.position.x = static_cast<float>(transform.translation.x);
    pose.position.y = static_cast<float>(transform.translation.y);
    pose.position.z = static_cast<float>(transform.translation.z);
    return pose;
}

//! An untracked-but-valid joint, the way WujiGlovePlugin and the other push-device plugins fill
//! theirs; the TRACKED bits are added by the caller once the wrist source is actually tracked.
XrHandJointLocationEXT make_valid_joint(const XrPosef& pose)
{
    XrHandJointLocationEXT joint{};
    joint.pose = pose;
    joint.radius = kHandJointRadius;
    joint.locationFlags = kPoseValidFlags;
    return joint;
}

} // namespace

std::array<HandJointSlot, kHandJointSlotCount> make_hand_joint_slots(Hand hand)
{
    std::array<HandJointSlot, kHandJointSlotCount> slots{};
    const HandJointSlot* source = (hand == Hand::Left) ? kLeftSlots : kRightSlots;
    for (std::size_t i = 0; i < kHandJointSlotCount; ++i)
    {
        slots[i] = source[i];
    }
    return slots;
}

XrQuaternionf multiply_quaternions(const XrQuaternionf& a, const XrQuaternionf& b)
{
    // Same convention as oxr_utils::multiply_poses, which is defined for poses.
    XrQuaternionf result{};
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

XrQuaternionf conjugate_quaternion(const XrQuaternionf& q)
{
    return XrQuaternionf{ -q.x, -q.y, -q.z, q.w };
}

XrPosef rotate_pose(const XrQuaternionf& rotation, const XrPosef& pose)
{
    // A pose is a rigid transform; rotating it is composing a pure rotation in front of it.
    const XrPosef pure_rotation{ rotation, XrVector3f{ 0.0f, 0.0f, 0.0f } };
    return oxr_utils::multiply_poses(pure_rotation, pose);
}

XrPosef invert_pose(const XrPosef& pose)
{
    XrPosef inverse{};
    inverse.orientation = conjugate_quaternion(pose.orientation);
    const XrPosef pure_rotation{ inverse.orientation, XrVector3f{ 0.0f, 0.0f, 0.0f } };
    const XrPosef negated_position{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f },
                                    XrVector3f{ -pose.position.x, -pose.position.y, -pose.position.z } };
    return oxr_utils::multiply_poses(pure_rotation, negated_position);
}

HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    HandSkeleton skeleton;
    const auto slots = make_hand_joint_slots(hand);

    // The glove's static root frame. It carries the hand's articulation but sits at the knuckle line,
    // so it is an anchor for "is this hand in the message", not the wrist itself.
    const auto* anchor = find_frame(msg, slots[0].frames);
    if (anchor == nullptr)
    {
        return skeleton;
    }
    skeleton.wrist_frame = anchor->child_frame_id;
    skeleton.resolved[0] = skeleton.wrist_frame;

    // Every slot the message carries, still in the glove's own frame.
    std::array<XrPosef, kHandJointSlotCount> poses{};
    std::array<bool, kHandJointSlotCount> present{};
    for (std::size_t i = 0; i < kHandJointSlotCount; ++i)
    {
        const auto* frame = find_frame(msg, slots[i].frames);
        if (frame == nullptr)
        {
            continue; // Sparse rig: a link the exoskeleton does not carry stays untracked.
        }
        poses[i] = to_pose(frame->transform);
        present[i] = true;
        skeleton.resolved[i] = frame->child_frame_id;
    }

    const auto slot_of = [&slots](XrHandJointEXT joint)
    {
        for (std::size_t i = 0; i < kHandJointSlotCount; ++i)
        {
            if (slots[i].joint == joint)
            {
                return i;
            }
        }
        return kHandJointSlotCount;
    };
    const auto mean_position = [&poses, &present, &slot_of](
                                   const XrHandJointEXT* joints, std::size_t count, XrVector3f& out, std::size_t& seen)
    {
        out = XrVector3f{ 0.0f, 0.0f, 0.0f };
        seen = 0;
        for (std::size_t j = 0; j < count; ++j)
        {
            const std::size_t slot = slot_of(joints[j]);
            if (slot >= kHandJointSlotCount || !present[slot])
            {
                continue;
            }
            out.x += poses[slot].position.x;
            out.y += poses[slot].position.y;
            out.z += poses[slot].position.z;
            ++seen;
        }
        if (seen > 0)
        {
            out.x /= static_cast<float>(seen);
            out.y /= static_cast<float>(seen);
            out.z /= static_cast<float>(seen);
        }
    };

    // The wrist OpenXR asks for is the anatomical one: the conformance suite measures the palm as
    // WRIST -> MIDDLE_PROXIMAL and expects a middle finger roughly 0.6-1.5 palms long. The glove's
    // root sits on the knuckle line, where that distance is ~0 and every proportion blows up, so the
    // wrist is placed one palm length back along the direction the fingertips point.
    skeleton.wrist_pose = poses[0];
    {
        const XrHandJointEXT knuckles[] = { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT,
                                            XR_HAND_JOINT_RING_PROXIMAL_EXT, XR_HAND_JOINT_LITTLE_PROXIMAL_EXT };
        const XrHandJointEXT fingertips[] = { XR_HAND_JOINT_THUMB_TIP_EXT, XR_HAND_JOINT_INDEX_TIP_EXT,
                                              XR_HAND_JOINT_MIDDLE_TIP_EXT, XR_HAND_JOINT_RING_TIP_EXT,
                                              XR_HAND_JOINT_LITTLE_TIP_EXT };
        XrVector3f knuckle_center{};
        XrVector3f fingertip_center{};
        std::size_t knuckles_seen = 0;
        std::size_t fingertips_seen = 0;
        mean_position(knuckles, 4, knuckle_center, knuckles_seen);
        mean_position(fingertips, 5, fingertip_center, fingertips_seen);
        if (knuckles_seen > 0 && fingertips_seen > 0)
        {
            XrVector3f axis{ fingertip_center.x - knuckle_center.x, fingertip_center.y - knuckle_center.y,
                             fingertip_center.z - knuckle_center.z };
            const float length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
            if (length > 1e-4f)
            {
                axis.x /= length;
                axis.y /= length;
                axis.z /= length;
                skeleton.wrist_pose.position = XrVector3f{ knuckle_center.x - kGlovePalmLength * axis.x,
                                                           knuckle_center.y - kGlovePalmLength * axis.y,
                                                           knuckle_center.z - kGlovePalmLength * axis.z };
            }
        }
    }

    // The WRIST joint is the origin of the frame every other joint is expressed in.
    skeleton.joints[XR_HAND_JOINT_WRIST_EXT] =
        make_valid_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, XrVector3f{ 0.0f, 0.0f, 0.0f } });
    ++skeleton.filled;

    const XrPosef wrist_inverse = invert_pose(skeleton.wrist_pose);
    for (std::size_t i = 1; i < kHandJointSlotCount; ++i)
    {
        if (!present[i])
        {
            continue;
        }
        skeleton.joints[slots[i].joint] = make_valid_joint(oxr_utils::multiply_poses(wrist_inverse, poses[i]));
        ++skeleton.filled;
    }

    return skeleton;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
