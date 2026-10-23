// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! A bend is clamped here: a linkage glitch should not fold the anatomy through itself.
constexpr float kMaxBend = 1.4f;

//! Dorsal direction of the anatomy's rest frame, i.e. the back of the hand (OpenXR's +Y).
constexpr XrVector3f kDorsal{ 0.0f, 1.0f, 0.0f };

const char* side_prefix(Hand hand)
{
    return hand == Hand::Left ? "Left" : "Right";
}

const char* finger_tip_frame(Hand hand, std::size_t finger)
{
    static const char* kLeft[] = { "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip",
                                   "left_pinky_tip" };
    static const char* kRight[] = { "right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip",
                                    "right_pinky_tip" };
    return hand == Hand::Left ? kLeft[finger] : kRight[finger];
}

const geometry_msgs::msg::TransformStamped* find_frame(const tf2_msgs::msg::TFMessage& msg, const std::string& name)
{
    if (name.empty())
    {
        return nullptr;
    }
    for (const auto& transform : msg.transforms)
    {
        if (transform.child_frame_id == name)
        {
            return &transform;
        }
    }
    return nullptr;
}

XrVector3f to_vector(const geometry_msgs::msg::Transform& transform)
{
    return XrVector3f{ static_cast<float>(transform.translation.x), static_cast<float>(transform.translation.y),
                       static_cast<float>(transform.translation.z) };
}

XrVector3f subtract(const XrVector3f& a, const XrVector3f& b)
{
    return XrVector3f{ a.x - b.x, a.y - b.y, a.z - b.z };
}

float dot(const XrVector3f& a, const XrVector3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

XrVector3f cross(const XrVector3f& a, const XrVector3f& b)
{
    return XrVector3f{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

bool normalize(XrVector3f& v)
{
    const float length = std::sqrt(dot(v, v));
    if (!(length > 1e-6f))
    {
        return false;
    }
    v.x /= length;
    v.y /= length;
    v.z /= length;
    return true;
}

XrVector3f normalized(const XrVector3f& v)
{
    XrVector3f out = v;
    normalize(out);
    return out;
}

//! Rotates @p v by the unit quaternion @p q (R v = q v q*).
XrVector3f rotate_vector(const XrQuaternionf& q, const XrVector3f& v)
{
    const XrVector3f u{ q.x, q.y, q.z };
    const float w = q.w;
    const XrVector3f t{ 2.0f * (u.y * v.z - u.z * v.y), 2.0f * (u.z * v.x - u.x * v.z), 2.0f * (u.x * v.y - u.y * v.x) };
    return XrVector3f{ v.x + w * t.x + (u.y * t.z - u.z * t.y), v.y + w * t.y + (u.z * t.x - u.x * t.z),
                       v.z + w * t.z + (u.x * t.y - u.y * t.x) };
}

//! Signed bend from segment @p previous to segment @p next, positive towards the palm (that is,
//! towards -@p dorsal). Degenerate segments leave the bend at zero.
float bend_between(const XrVector3f& previous, const XrVector3f& next, const XrVector3f& dorsal)
{
    XrVector3f from = previous;
    XrVector3f to = next;
    if (!normalize(from) || !normalize(to))
    {
        return 0.0f;
    }
    return std::clamp(std::atan2(-dot(to, dorsal), dot(to, from)), -kMaxBend, kMaxBend);
}

XrQuaternionf axis_angle(const XrVector3f& axis, float angle)
{
    XrQuaternionf q{};
    const float half = 0.5f * angle;
    const float sine = std::sin(half);
    q.x = axis.x * sine;
    q.y = axis.y * sine;
    q.z = axis.z * sine;
    q.w = std::cos(half);
    return q;
}

XrHandJointLocationEXT make_joint(const XrPosef& pose)
{
    XrHandJointLocationEXT joint{};
    joint.pose = pose;
    joint.radius = kHandJointRadius;
    joint.locationFlags = kPoseValidFlags;
    return joint;
}

} // namespace

const char* mechanism_root_frame(Hand hand)
{
    return hand == Hand::Left ? "link_LeftSkeletonBase" : "link_RightSkeletonBase";
}

HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    HandAngles angles;

    // The root frame anchors "is this glove streaming"; the links themselves carry the angles. The
    // driver publishes both `link_<Side>SkeletonBase` and `joint_<Side>SkeletonBase`; accept either.
    const geometry_msgs::msg::TransformStamped* root = find_frame(msg, mechanism_root_frame(hand));
    if (root == nullptr)
    {
        root = find_frame(msg, std::string("joint_") + side_prefix(hand) + "SkeletonBase");
    }
    if (root == nullptr)
    {
        return angles;
    }

    // Mechanism points per finger, in chain order: F1..F4 and the fingertip.
    std::array<std::array<XrVector3f, 5>, 5> points{};
    std::array<bool, 5> present{};
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        bool complete = true;
        for (std::size_t link = 0; link < kRigFingers[finger].links.size(); ++link)
        {
            const char* suffix = kRigFingers[finger].links[link];
            const std::string name = (suffix[0] == '\0') ?
                                         std::string(finger_tip_frame(hand, finger)) :
                                         std::string("link_") + side_prefix(hand) + "Skeleton" + suffix;
            const auto* frame = find_frame(msg, name);
            if (frame == nullptr)
            {
                complete = false;
                continue;
            }
            points[finger][link] = to_vector(frame->transform);
            if (link == 0)
            {
                angles.metacarpal_frame[finger] = frame->child_frame_id;
            }
        }
        present[finger] = complete;
    }
    if (!present[1] || !present[4])
    {
        // The two outer fingers define the palm; without them nothing can be oriented.
        return angles;
    }

    // Palm orientation of the *mechanism*: across runs from the little towards the index knuckle, and
    // distal is the metacarpal direction (knuckle to the joint behind it), which the bends do not
    // rotate -- taking it from the fingertips instead would tilt the dorsal as the fingers close and
    // skew every angle read from it. Dorsal is their cross product, the same right-handed convention
    // the anatomy uses. Only directions cross over, never lengths, so the mechanism's long linkage
    // arms cannot leak into the injected hand.
    XrVector3f across = normalized(subtract(points[1][0], points[4][0]));
    XrVector3f distal{ 0.0f, 0.0f, 0.0f };
    std::size_t distal_count = 0;
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        if (!present[finger])
        {
            continue;
        }
        distal.x += points[finger][1].x - points[finger][0].x;
        distal.y += points[finger][1].y - points[finger][0].y;
        distal.z += points[finger][1].z - points[finger][0].z;
        ++distal_count;
    }
    distal.x /= static_cast<float>(distal_count);
    distal.y /= static_cast<float>(distal_count);
    distal.z /= static_cast<float>(distal_count);
    if (!normalize(distal) || !normalize(across))
    {
        return angles;
    }
    XrVector3f dorsal = normalized(cross(across, distal));
    if (hand == Hand::Right)
    {
        // The right glove's knuckle row runs the other way, so the same cross product lands on the
        // palm side; flip it here and "dorsal" means the back of the hand on both hands, which is what
        // makes a positive bend read as a curl towards the palm either way.
        dorsal = XrVector3f{ -dorsal.x, -dorsal.y, -dorsal.z };
    }
    if (!normalize(dorsal))
    {
        return angles;
    }

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        if (!present[finger])
        {
            continue;
        }
        FingerAngles& out = angles.fingers[finger];

        // Knuckle spread: how far the first phalanx leaves the palm's distal axis towards the index.
        const XrVector3f first_segment = normalized(subtract(points[finger][1], points[finger][0]));
        out.spread = std::clamp(std::atan2(dot(first_segment, across), dot(first_segment, distal)), -kMaxBend, kMaxBend);

        // Bends at the three joints after the knuckle; the thumb's chain starts one joint earlier, at
        // its CMC, so its reading keeps that first bend as well.
        std::size_t slot = 0;
        for (std::size_t link = 1; link + 1 < 5 && slot < out.bend.size(); ++link)
        {
            out.bend[slot] = bend_between(subtract(points[finger][link], points[finger][link - 1]),
                                          subtract(points[finger][link + 1], points[finger][link]), dorsal);
            ++slot;
        }
    }

    angles.valid = true;
    return angles;
}

HandSkeleton build_anatomical_skeleton(const HandAngles& angles, Hand hand)
{
    HandSkeleton skeleton;
    if (!angles.valid)
    {
        return skeleton;
    }
    skeleton.wrist_frame = mechanism_root_frame(hand);

    // The OpenXR wrist frame is the anatomy's rest frame: -Z along the bones, +Y the back of the
    // hand, +X from the little finger towards the index. The wrist is the origin and the palm half a
    // palm length towards the knuckles, both at the rest orientation: every joint below therefore
    // depends only on the glove's articulation, never on when or where a mechanism frame arrived.
    skeleton.joints[XR_HAND_JOINT_WRIST_EXT] =
        make_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, XrVector3f{ 0.0f, 0.0f, 0.0f } });
    skeleton.joints[XR_HAND_JOINT_PALM_EXT] =
        make_joint(XrPosef{ XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f }, XrVector3f{ 0.0f, 0.0f, -kPalmCentreFromWrist } });
    skeleton.filled = 2;

    // The vendor's human model is authored for the left hand; the right hand mirrors it in X. Spread
    // is the one rotation that changes sign with the mirror: the index sits on the other side.
    const float mirror = (hand == Hand::Left) ? 1.0f : -1.0f;

    for (std::size_t finger = 0; finger < kHumanFingerCount; ++finger)
    {
        const HumanFinger& anatomy = kHumanFingers[finger];
        const FingerAngles& articulation = angles.fingers[finger];
        const bool is_thumb = finger == 0;

        XrQuaternionf rotation{ 0.0f, 0.0f, 0.0f, 1.0f };
        XrVector3f position{ 0.0f, 0.0f, 0.0f };
        XrVector3f forward{ 0.0f, 0.0f, -1.0f };

        for (std::size_t landmark = 0; landmark < anatomy.count; ++landmark)
        {
            // Bending happens *at* a joint, about the current bone's axes: flexion about +X (curl
            // towards the palm, i.e. -Y) and spread about +Y (towards the index).
            if (is_thumb)
            {
                if (landmark == 0)
                {
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[0]));
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 0.0f, 1.0f, 0.0f }, -articulation.spread * mirror));
                }
                else if (landmark <= 2)
                {
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[landmark]));
                }
            }
            else if (landmark == 1)
            {
                rotation = multiply_quaternions(
                    rotation, axis_angle(XrVector3f{ 0.0f, 1.0f, 0.0f }, -articulation.spread * mirror));
                rotation =
                    multiply_quaternions(rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[0]));
            }
            else if (landmark <= 3)
            {
                rotation = multiply_quaternions(
                    rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[landmark - 1]));
            }

            const XrVector3f offset{ anatomy.landmarks[landmark].offset.x * mirror,
                                     anatomy.landmarks[landmark].offset.y, anatomy.landmarks[landmark].offset.z };
            const XrVector3f bone = rotate_vector(rotation, offset);
            position = XrVector3f{ position.x + bone.x, position.y + bone.y, position.z + bone.z };

            if (landmark + 1 < anatomy.count)
            {
                forward =
                    XrVector3f{ anatomy.landmarks[landmark + 1].offset.x * mirror,
                                anatomy.landmarks[landmark + 1].offset.y, anatomy.landmarks[landmark + 1].offset.z };
            }
            const XrVector3f forward_world = normalized(rotate_vector(rotation, forward));
            const XrVector3f dorsal_world = rotate_vector(rotation, kDorsal);
            const XrQuaternionf orientation = orientation_along(forward_world, dorsal_world);

            skeleton.joints[anatomy.landmarks[landmark].joint] = make_joint(XrPosef{ orientation, position });
            ++skeleton.filled;
        }
    }

    return skeleton;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
