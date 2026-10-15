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

XrQuaternionf slerp_quaternions(const XrQuaternionf& from, const XrQuaternionf& to, float t)
{
    if (!(t > 0.0f))
    {
        return from;
    }
    if (!(t < 1.0f))
    {
        return to;
    }
    float dot = from.x * to.x + from.y * to.y + from.z * to.z + from.w * to.w;
    XrQuaternionf target = to;
    if (dot < 0.0f)
    {
        // Take the short way round.
        target = XrQuaternionf{ -to.x, -to.y, -to.z, -to.w };
        dot = -dot;
    }
    if (dot > 0.9995f)
    {
        // Nearly parallel: interpolate linearly and renormalise, which avoids dividing by sin(0).
        XrQuaternionf blended{ from.x + t * (target.x - from.x), from.y + t * (target.y - from.y),
                               from.z + t * (target.z - from.z), from.w + t * (target.w - from.w) };
        const float norm =
            std::sqrt(blended.x * blended.x + blended.y * blended.y + blended.z * blended.z + blended.w * blended.w);
        if (norm > 1e-6f)
        {
            blended.x /= norm;
            blended.y /= norm;
            blended.z /= norm;
            blended.w /= norm;
        }
        return blended;
    }
    const float theta = std::acos(dot);
    const float sin_theta = std::sin(theta);
    const float a = std::sin((1.0f - t) * theta) / sin_theta;
    const float b = std::sin(t * theta) / sin_theta;
    return XrQuaternionf{ a * from.x + b * target.x, a * from.y + b * target.y, a * from.z + b * target.z,
                          a * from.w + b * target.w };
}

XrQuaternionf conjugate_quaternion(const XrQuaternionf& q)
{
    return XrQuaternionf{ -q.x, -q.y, -q.z, q.w };
}

//! Rotation whose -Z axis is @p forward (OpenXR joint convention: -Z runs along the bone) and whose
//! roll is pinned by @p up_hint.
XrQuaternionf orientation_along(const XrVector3f& forward, const XrVector3f& up_hint)
{
    const float length = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (!(length > 1e-6f))
    {
        return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    const XrVector3f z_axis{ -forward.x / length, -forward.y / length, -forward.z / length };

    // up_hint may be parallel to the bone on a degenerate frame; fall back to any perpendicular.
    XrVector3f up = up_hint;
    const float dot = up.x * z_axis.x + up.y * z_axis.y + up.z * z_axis.z;
    up.x -= dot * z_axis.x;
    up.y -= dot * z_axis.y;
    up.z -= dot * z_axis.z;
    float up_length = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
    if (!(up_length > 1e-6f))
    {
        up = std::fabs(z_axis.x) < 0.9f ? XrVector3f{ 1.0f, 0.0f, 0.0f } : XrVector3f{ 0.0f, 1.0f, 0.0f };
        const float fallback_dot = up.x * z_axis.x + up.y * z_axis.y + up.z * z_axis.z;
        up.x -= fallback_dot * z_axis.x;
        up.y -= fallback_dot * z_axis.y;
        up.z -= fallback_dot * z_axis.z;
        up_length = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
    }
    up.x /= up_length;
    up.y /= up_length;
    up.z /= up_length;

    // x = up x z, y = z x x
    const XrVector3f x_axis{ up.y * z_axis.z - up.z * z_axis.y, up.z * z_axis.x - up.x * z_axis.z,
                             up.x * z_axis.y - up.y * z_axis.x };
    const XrVector3f y_axis{ z_axis.y * x_axis.z - z_axis.z * x_axis.y, z_axis.z * x_axis.x - z_axis.x * x_axis.z,
                             z_axis.x * x_axis.y - z_axis.y * x_axis.x };

    // Rotation matrix (columns x, y, z) -> quaternion.
    const float m00 = x_axis.x, m01 = y_axis.x, m02 = z_axis.x;
    const float m10 = x_axis.y, m11 = y_axis.y, m12 = z_axis.y;
    const float m20 = x_axis.z, m21 = y_axis.z, m22 = z_axis.z;
    XrQuaternionf q{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f)
    {
        const float scale = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * scale;
        q.x = (m21 - m12) / scale;
        q.y = (m02 - m20) / scale;
        q.z = (m10 - m01) / scale;
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float scale = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / scale;
        q.x = 0.25f * scale;
        q.y = (m01 + m10) / scale;
        q.z = (m02 + m20) / scale;
    }
    else if (m11 > m22)
    {
        const float scale = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / scale;
        q.x = (m01 + m10) / scale;
        q.y = 0.25f * scale;
        q.z = (m12 + m21) / scale;
    }
    else
    {
        const float scale = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / scale;
        q.x = (m02 + m20) / scale;
        q.y = (m12 + m21) / scale;
        q.z = 0.25f * scale;
    }
    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm > 1e-6f)
    {
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
        q.w /= norm;
    }
    return q;
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

HandSkeleton build_hand_skeleton(const tf2_msgs::msg::TFMessage& msg,
                                 Hand hand,
                                 const std::optional<XrVector3f>& frozen_wrist_offset)
{
    HandSkeleton skeleton;
    const auto slots = make_hand_joint_slots(hand);

    // The glove's static root frame. It carries the hand's articulation but sits at the knuckle line,
    // so it anchors "is this hand in the message" and gives the wrist offset its axes.
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
    // wrist sits one palm length back along the direction the fingertips point.
    //
    // That direction moves while the fingers move, so the offset is a rig constant derived once and
    // then frozen by the caller: re-deriving it every frame would slide the wrist -- and with it the
    // whole hand -- as the performer closes a fist.
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

    skeleton.wrist_pose = poses[0];
    XrVector3f offset{ 0.0f, 0.0f, 0.0f };
    if (frozen_wrist_offset.has_value())
    {
        offset = *frozen_wrist_offset;
        skeleton.wrist_offset = offset;
        skeleton.wrist_offset_valid = true;
    }
    else if (knuckles_seen > 0 && fingertips_seen > 0)
    {
        const XrVector3f axis{ fingertip_center.x - knuckle_center.x, fingertip_center.y - knuckle_center.y,
                               fingertip_center.z - knuckle_center.z };
        const float length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
        if (length > 1e-4f)
        {
            offset = XrVector3f{ knuckle_center.x - kGlovePalmLength * axis.x / length - poses[0].position.x,
                                 knuckle_center.y - kGlovePalmLength * axis.y / length - poses[0].position.y,
                                 knuckle_center.z - kGlovePalmLength * axis.z / length - poses[0].position.z };
            skeleton.wrist_offset = offset;
            skeleton.wrist_offset_valid = true;
        }
    }
    skeleton.wrist_pose.position =
        XrVector3f{ poses[0].position.x + offset.x, poses[0].position.y + offset.y, poses[0].position.z + offset.z };

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

    // Orientations carry OpenXR's joint axes, not the rig's: -Z runs along the bone, with the roll
    // pinned to the palm normal so the axes do not spin about the finger.
    const auto relative_position = [&skeleton, &slot_of, &present](XrHandJointEXT joint, XrVector3f& out)
    {
        const std::size_t slot = slot_of(joint);
        if (slot >= kHandJointSlotCount || !present[slot])
        {
            return false;
        }
        out = skeleton.joints[joint].pose.position;
        return true;
    };

    XrVector3f palm_normal{ 0.0f, 0.0f, 0.0f };
    {
        XrVector3f index_knuckle{};
        XrVector3f little_knuckle{};
        XrVector3f knuckle_mean{};
        XrVector3f tip_mean{};
        const bool have_across = relative_position(XR_HAND_JOINT_INDEX_PROXIMAL_EXT, index_knuckle) &&
                                 relative_position(XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, little_knuckle);
        const bool have_forward = relative_position(XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, knuckle_mean) &&
                                  relative_position(XR_HAND_JOINT_MIDDLE_DISTAL_EXT, tip_mean);
        if (have_across && have_forward)
        {
            const XrVector3f across{ index_knuckle.x - little_knuckle.x, index_knuckle.y - little_knuckle.y,
                                     index_knuckle.z - little_knuckle.z };
            const XrVector3f forward{ tip_mean.x - knuckle_mean.x, tip_mean.y - knuckle_mean.y,
                                      tip_mean.z - knuckle_mean.z };
            palm_normal =
                XrVector3f{ across.y * forward.z - across.z * forward.y, across.z * forward.x - across.x * forward.z,
                            across.x * forward.y - across.y * forward.x };
        }
    }

    const XrHandJointEXT chains[][4] = {
        { XR_HAND_JOINT_THUMB_METACARPAL_EXT, XR_HAND_JOINT_THUMB_PROXIMAL_EXT, XR_HAND_JOINT_THUMB_DISTAL_EXT,
          XR_HAND_JOINT_THUMB_TIP_EXT },
        { XR_HAND_JOINT_INDEX_PROXIMAL_EXT, XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, XR_HAND_JOINT_INDEX_DISTAL_EXT,
          XR_HAND_JOINT_INDEX_TIP_EXT },
        { XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT, XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT, XR_HAND_JOINT_MIDDLE_DISTAL_EXT,
          XR_HAND_JOINT_MIDDLE_TIP_EXT },
        { XR_HAND_JOINT_RING_PROXIMAL_EXT, XR_HAND_JOINT_RING_INTERMEDIATE_EXT, XR_HAND_JOINT_RING_DISTAL_EXT,
          XR_HAND_JOINT_RING_TIP_EXT },
        { XR_HAND_JOINT_LITTLE_PROXIMAL_EXT, XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT, XR_HAND_JOINT_LITTLE_DISTAL_EXT,
          XR_HAND_JOINT_LITTLE_TIP_EXT },
    };
    for (const auto& chain : chains)
    {
        for (std::size_t k = 0; k < 4; ++k)
        {
            const std::size_t slot = slot_of(chain[k]);
            if (slot >= kHandJointSlotCount || !present[slot])
            {
                continue;
            }
            // Segments run joint k -> k+1; the tip keeps the direction of the segment that ends on it.
            const XrHandJointEXT from = chain[k == 3 ? 2 : k];
            const XrHandJointEXT to = chain[k == 3 ? 3 : k + 1];
            XrVector3f from_position{};
            XrVector3f to_position{};
            if (!relative_position(from, from_position) || !relative_position(to, to_position))
            {
                continue;
            }
            const XrVector3f forward{ to_position.x - from_position.x, to_position.y - from_position.y,
                                      to_position.z - from_position.z };
            skeleton.joints[chain[k]].pose.orientation = orientation_along(forward, palm_normal);
        }
    }

    return skeleton;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
