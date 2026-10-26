// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! Revolute joints read per finger, in chain order: the knuckle spread F1 and the bends F2..F4.
//! The thumb's are Thumb1..Thumb4 (ThumbBase is its fixed mount).
constexpr std::size_t kRigSlotCount = 4;

//! Which hand a table row belongs to.
constexpr std::size_t hand_index(Hand hand)
{
    return hand == Hand::Left ? 0 : 1;
}

//! One revolute joint of the vendor's ``blender_human_skeleton_v5.urdf`` finger chain.
struct RigJointSpec
{
    //! URDF ``<origin rpy>``: the fixed rotation from the parent frame to this joint's zero pose,
    //! radians, applied as Rz(yaw) * Ry(pitch) * Rx(roll) like URDF does.
    std::array<float, 3> rest_rpy;
    //! URDF ``<axis xyz>``: the revolute axis in the joint frame (unit length in the URDF).
    std::array<float, 3> axis;
    //! Absolute joint angle of the open hand, radians: the calibration offset subtracted before the
    //! anatomy is driven. See the derivation above the table.
    float zero;
    //! +1 when flexion increases the URDF angle, -1 when it decreases it. Slot 0 is the knuckle
    //! spread, which is not a flexion axis, so it uses the hand sign instead (+ left, - right).
    float sign;
};

//! Rest rotation, revolute axis, open-hand zero and flexion sign for the four joints per finger that
//! drive the anatomy: the knuckle (F1 -> spread) and the three bends (F2 -> MCP, F3 -> PIP,
//! F4 -> DIP). For the thumb those are Thumb1 (spread) and Thumb2..Thumb4 (CMC, MCP, IP).
//!
//! Derivation (performed by ``io_exo_glove_ros2_kinematics_tables.py`` on a 7251-message recording
//! of both hands opening and closing):
//!
//!  - The v5 URDF is
//!    ``.../io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/blender_human_skeleton_v5.urdf``.
//!    Every finger joint is ``revolute`` with an ``<origin xyz rpy>`` and an ``<axis xyz>``, and the
//!    driver publishes both frames in ``base_link``. Therefore
//!    ``q_child = q_parent * q_rest(rpy) * q_axis(theta)`` and, with the published quaternions,
//!    ``R_rel = conj(q_parent) * q_child`` and ``theta = signed_angle(conj(q_rest) * R_rel, axis)``
//!    is the joint's absolute angle relative to the URDF zero pose. ``read_hand_angles`` computes it
//!    that way; no frame position, frame ordering or first-frame capture is involved.
//!  - Openness is measured along the finger: from each knuckle (the first bend joint) to the
//!    published fingertip. The recording's longest steady open plateau gives the zero, and its
//!    longest steady closed plateau gives the sign, at each plateau's medoid frame so a drifting
//!    pose cannot bias either. The previous table instead maximised ``dot(tip - MCP, dorsal)``,
//!    which is the palm normal, so it picked the most flexed frames and most of its zeros were
//!    fists: the injected hand then bent as the glove opened.
//!  - ``sign`` is +1 when the joint angle grows from the verified open pose towards the fist, -1
//!    when it shrinks. Both hands need the same mix (MCP +, PIP -, DIP +), including the right
//!    index, whose chain matches the other right fingers rather than mirroring the left one. Slot 0
//!    is the spread, which is not a flexion axis: its zero is the open-hand angle and its sign is
//!    the hand sign (+ left, - right).
//!  - The derivation checks that the open zero places each model fingertip one finger-length ahead
//!    of its MCP (thumb 4.9 cm, index 8.2 cm, middle 8.9 cm, ring 8.3 cm, little 6.9 cm) and that
//!    every mapped MCP/PIP/DIP reads positive at the fist on both hands.
constexpr RigJointSpec kRigJoints[2][kRigFingerCount][kRigSlotCount] = {
    // left
    {
        {
            // thumb
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.00000f, 0.00000f, 1.00000f }, -0.28837f, 1.0f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.66229f, 1.0f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.93764f, -1.0f },
            { { 2.53070f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 1.15219f, 1.0f },
        },
        {
            // index
            { { 0.00000f, 0.26180f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.27738f, 1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.04983f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.86301f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.13192f, 1.0f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.02953f, 1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.15469f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.61346f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.03451f, 1.0f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.04644f, 1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.14160f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.66886f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.05541f, 1.0f },
        },
        {
            // little
            { { 0.00000f, -0.17453f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.05504f, 1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.48432f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -2.35083f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.85100f, 1.0f },
        },
    },
    // right
    {
        {
            // thumb
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.00000f, 0.00000f, 1.00000f }, 0.37937f, -1.0f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.38808f, 1.0f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.21146f, -1.0f },
            { { 2.53070f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.06686f, 1.0f },
        },
        {
            // index
            { { 0.00000f, -0.26180f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.19091f, -1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.03796f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.67321f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.04263f, 1.0f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.01183f, -1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.45659f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.25074f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.38185f, 1.0f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.05478f, -1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.03685f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.60046f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -0.05488f, 1.0f },
        },
        {
            // little
            { { 0.00000f, 0.17453f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.23324f, -1.0f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.39179f, 1.0f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -2.18509f, -1.0f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.42326f, 1.0f },
        },
    },
};

//! Spread is clamped symmetrically: the metacarpal abduction/adduction range, about +/-25 deg.
constexpr float kMaxSpread = 0.436332f;

//! Human bend limits, radians, per bend slot (MCP, PIP, DIP): 0..100 deg, 0..110 deg, 0..110 deg.
//! Negative values are extension and are clamped to the open-hand zero.
constexpr std::array<float, 3> kFingerBendLimit = { 1.745329f, 1.919862f, 1.919862f };

//! Thumb limits, radians, per bend slot (CMC, MCP, IP): 0..60 deg, 0..60 deg, 0..80 deg. "As
//! appropriate": the rig's thumb is a four-joint chain (spread plus three bends) mapped onto the
//! anatomy's three thumb bends.
constexpr std::array<float, 3> kThumbBendLimit = { 1.047198f, 1.047198f, 1.396263f };

//! Dorsal direction of the anatomy's rest frame, i.e. the back of the hand (OpenXR's +Y).
constexpr XrVector3f kDorsal{ 0.0f, 1.0f, 0.0f };

const char* side_prefix(Hand hand)
{
    return hand == Hand::Left ? "Left" : "Right";
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

float dot(const XrVector3f& a, const XrVector3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
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

//! The URDF's fixed-axis rpy convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
XrQuaternionf quaternion_from_rpy(const std::array<float, 3>& rpy)
{
    return multiply_quaternions(multiply_quaternions(axis_angle(XrVector3f{ 0.0f, 0.0f, 1.0f }, rpy[2]),
                                                     axis_angle(XrVector3f{ 0.0f, 1.0f, 0.0f }, rpy[1])),
                                axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, rpy[0]));
}

//! A published frame's rotation as a unit quaternion, defensively normalised.
XrQuaternionf frame_rotation(const geometry_msgs::msg::TransformStamped& frame)
{
    XrQuaternionf q{ static_cast<float>(frame.transform.rotation.x), static_cast<float>(frame.transform.rotation.y),
                     static_cast<float>(frame.transform.rotation.z), static_cast<float>(frame.transform.rotation.w) };
    const float length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (!(length > 1e-6f))
    {
        return XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }
    q.x /= length;
    q.y /= length;
    q.z /= length;
    q.w /= length;
    return q;
}

//! Signed rotation of @p q about @p axis, in radians: the same ``2 * atan2(dot(v, axis), w)`` with the
//! quaternion's w made non-negative, so the result is the shortest signed angle in (-pi, pi].
float signed_angle_about(const XrQuaternionf& q, const std::array<float, 3>& axis)
{
    XrVector3f v{ q.x, q.y, q.z };
    float w = q.w;
    if (w < 0.0f)
    {
        v = XrVector3f{ -v.x, -v.y, -v.z };
        w = -w;
    }
    return 2.0f * std::atan2(dot(v, XrVector3f{ axis[0], axis[1], axis[2] }), w);
}

//! Absolute mechanism angles read from one message, before calibration: [finger][F1..F4] radians
//! relative to the URDF zero pose. Internal, but used by the local harness to check the formula
//! against known synthetic angles.
struct RawHandAngles
{
    std::array<std::array<float, kRigSlotCount>, kRigFingerCount> joint{};
    std::array<bool, kRigFingerCount> complete{};
    bool valid = false;
};

RawHandAngles read_raw_hand_angles(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    RawHandAngles raw;

    // The root frame anchors "is this glove streaming"; every finger's first joint hangs off it,
    // except the thumb's, which hangs off the mount read below. The driver publishes both
    // `link_<Side>SkeletonBase` and `joint_<Side>SkeletonBase`; accept either.
    const geometry_msgs::msg::TransformStamped* root = find_frame(msg, mechanism_root_frame(hand));
    if (root == nullptr)
    {
        root = find_frame(msg, std::string("joint_") + side_prefix(hand) + "SkeletonBase");
    }
    if (root == nullptr)
    {
        return raw;
    }

    const XrQuaternionf root_rotation = frame_rotation(*root);
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        bool complete = true;
        XrQuaternionf parent_rotation = root_rotation;
        // The thumb's first joint hangs off its fixed mount, not the mechanism root; the mount's own
        // published frame is its parent, so Thumb1's angle is independent of the mount.
        if (kRigFingers[finger].parent[0] != '\0')
        {
            const std::string parent_name =
                std::string("link_") + side_prefix(hand) + "Skeleton" + kRigFingers[finger].parent;
            const geometry_msgs::msg::TransformStamped* parent = find_frame(msg, parent_name);
            if (parent == nullptr)
            {
                complete = false;
            }
            else
            {
                parent_rotation = frame_rotation(*parent);
            }
        }
        for (std::size_t slot = 0; complete && slot < kRigSlotCount; ++slot)
        {
            const std::string name =
                std::string("link_") + side_prefix(hand) + "Skeleton" + kRigFingers[finger].links[slot];
            const geometry_msgs::msg::TransformStamped* frame = find_frame(msg, name);
            if (frame == nullptr)
            {
                // A sparse stream leaves the rest of this finger at its zero, but the hand can still be
                // injected from the fingers that are complete.
                complete = false;
                break;
            }

            const XrQuaternionf child_rotation = frame_rotation(*frame);
            const RigJointSpec& spec = kRigJoints[hand_index(hand)][finger][slot];
            const XrQuaternionf relative = multiply_quaternions(conjugate_quaternion(parent_rotation), child_rotation);
            const XrQuaternionf residual =
                multiply_quaternions(conjugate_quaternion(quaternion_from_rpy(spec.rest_rpy)), relative);
            raw.joint[finger][slot] = signed_angle_about(residual, spec.axis);
            parent_rotation = child_rotation;
        }
        raw.complete[finger] = complete;
        raw.valid = raw.valid || complete;
    }
    return raw;
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

    const RawHandAngles raw = read_raw_hand_angles(msg, hand);
    if (!raw.valid)
    {
        return angles;
    }

    const std::size_t side = hand_index(hand);

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        if (!raw.complete[finger])
        {
            continue;
        }

        FingerAngles& out = angles.fingers[finger];

        // Knuckle spread: the angle of the spread joint (F1, or Thumb1), mirrored into "positive
        // towards the index". Its table sign is the hand sign (+ left, - right).
        const RigJointSpec& knuckle = kRigJoints[side][finger][0];
        out.spread = std::clamp(knuckle.sign * (raw.joint[finger][0] - knuckle.zero), -kMaxSpread, kMaxSpread);

        // MCP/PIP/DIP (the thumb's CMC/MCP/IP): calibrated to the open hand and signed so flexion is
        // positive, then clamped to the human range.
        const std::array<float, 3>& limit = (finger == 0) ? kThumbBendLimit : kFingerBendLimit;
        for (std::size_t slot = 1; slot < kRigSlotCount; ++slot)
        {
            const RigJointSpec& spec = kRigJoints[side][finger][slot];
            out.bend[slot - 1] = std::clamp(spec.sign * (raw.joint[finger][slot] - spec.zero), 0.0f, limit[slot - 1]);
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

        // The first joint of the chain sits on landmark 1 for a finger (the MCP) and on landmark 0
        // for the thumb (the CMC); the tip landmark carries no joint.
        const std::size_t first_joint = is_thumb ? 0 : 1;

        for (std::size_t landmark = 0; landmark < anatomy.count; ++landmark)
        {
            // Place the landmark first. A joint rotates the segment *after* its own landmark, so
            // flexing the MCP cannot move the MCP itself.
            const XrVector3f offset{ anatomy.landmarks[landmark].offset.x * mirror,
                                     anatomy.landmarks[landmark].offset.y, anatomy.landmarks[landmark].offset.z };
            const XrVector3f bone = rotate_vector(rotation, offset);
            position = XrVector3f{ position.x + bone.x, position.y + bone.y, position.z + bone.z };

            // Then bend the following bone: flexion about +X (curl towards the palm, i.e. -Y), with
            // the knuckle spread about +Y (towards the index) at the same landmark.
            if (landmark >= first_joint && landmark + 1 < anatomy.count)
            {
                const std::size_t slot = landmark - first_joint;
                if (slot == 0)
                {
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 0.0f, 1.0f, 0.0f }, -articulation.spread * mirror));
                }
                rotation =
                    multiply_quaternions(rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[slot]));
            }

            if (landmark + 1 < anatomy.count)
            {
                forward =
                    XrVector3f{ anatomy.landmarks[landmark + 1].offset.x * mirror,
                                anatomy.landmarks[landmark + 1].offset.y, anatomy.landmarks[landmark + 1].offset.z };
            }
            // The tip has no following bone, so it keeps the last bone's direction.
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
