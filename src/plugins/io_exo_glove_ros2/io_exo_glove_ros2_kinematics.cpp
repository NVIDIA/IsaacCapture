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

//! Revolute joints read per finger, in chain order: F1..F4. The thumb has a fifth joint (Thumb4)
//! beyond the last anatomy joint the skeleton builder drives, so only ThumbBase..Thumb3 are read.
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
    //! spread and uses the hand sign instead (+ left, - right); see the note above the table.
    float sign;
};

//! Rest rotation, revolute axis, open-hand zero and flexion sign for the four joints per finger that
//! drive the anatomy: the knuckle (F1 -> spread) and the three bends (F2 -> MCP, F3 -> PIP,
//! F4 -> DIP). For the thumb those are ThumbBase, Thumb1, Thumb2 and Thumb3.
//!
//! Derivation (validated against a 58.7 s recording of both hands opening and closing):
//!
//!  - The v5 URDF is
//!    ``.../io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/blender_human_skeleton_v5.urdf``.
//!    Every finger joint is ``revolute`` with an ``<origin xyz rpy>`` and an ``<axis xyz>``, and the
//!    driver publishes both frames in ``base_link``. Therefore
//!    ``q_child = q_parent * q_rest(rpy) * q_axis(theta)`` and, with the published quaternions,
//!    ``R_rel = conj(q_parent) * q_child`` and ``theta = signed_angle(conj(q_rest) * R_rel, axis)``
//!    is the joint's absolute angle relative to the URDF zero pose. ``read_hand_angles`` computes it
//!    that way; no frame position, frame ordering or first-frame capture is involved.
//!  - ``zero`` is the open-hand angle of that joint, in degrees, read from
//!    ``raw_tf_both.jsonl``: for each finger, the frame where its fingertip was furthest from the
//!    palm (largest ``dot(tip - F1, dorsal)``, with dorsal taken from the knuckle row and the F1->F2
//!    direction), then every joint's angle at that frame. All 7251 recorded TF messages were used.
//!    The open-hand zeros, F1/F2/F3/F4 (degrees):
//!    - left thumb: -31.11 / -24.17 / 50.48 / -130.03
//!    - left index: -17.95 / -64.99 / -101.56 / -60.32
//!    - left middle: 5.69 / -66.71 / -104.02 / -59.16
//!    - left ring: 1.69 / -64.73 / -107.26 / -76.79
//!    - left little: 7.08 / -59.58 / -113.52 / -67.71
//!    - right thumb: 15.03 / 2.09 / 48.85 / -115.31
//!    - right index: 11.46 / 59.93 / -121.90 / 26.62
//!    - right middle: -5.97 / -73.54 / -98.84 / -74.99
//!    - right ring: -8.05 / -68.01 / -107.51 / -78.74
//!    - right little: -13.43 / -61.02 / -110.70 / -79.34
//!
//!    A joint's zero is therefore a constant calibration: two sessions of the same rig produce
//!    identical anatomical output for identical frame quaternions.
//!  - ``sign`` turns the absolute angle into the anatomy's flexion-positive convention: +1 when the
//!    joint angle grows from the most-open to the most-closed frame of the recording, -1 when it
//!    shrinks. The validated mirror is left + / right - for the index MCP, but the rig's linkage does
//!    not mirror every joint: one sign per hand would clamp the right middle/ring/little MCPs and two
//!    thumb joints to zero, so the measured sign is stored per joint. Slot 0 (spread) is not a
//!    flexion axis and always uses the hand sign.
constexpr RigJointSpec kRigJoints[2][kRigFingerCount][kRigSlotCount] = {
    // left
    {
        {
            // thumb
            { { 0.00000f, 0.00000f, 0.99442f }, { 1.00000f, 0.00000f, 0.00000f }, -0.54289f, 1.00000f },
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.00000f, 0.00000f, 1.00000f }, -0.42177f, -1.00000f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.88109f, -1.00000f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -2.26941f, 1.00000f },
        },
        {
            // index
            { { 0.00000f, 0.26180f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.31324f, 1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.13436f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.77253f, -1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.05282f, 1.00000f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.09928f, 1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.16431f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.81542f, 1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.03251f, 1.00000f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.02944f, 1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.12981f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.87196f, 1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.34022f, 1.00000f },
        },
        {
            // little
            { { 0.00000f, -0.17453f, 0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.12354f, 1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.03989f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.98132f, -1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.18170f, 1.00000f },
        },
    },
    // right
    {
        {
            // thumb
            { { 3.14160f, 0.00000f, 2.14720f }, { 1.00000f, 0.00000f, 0.00000f }, 0.26231f, -1.00000f },
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.00000f, 0.00000f, 1.00000f }, 0.03640f, 1.00000f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.85258f, -1.00000f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -2.01247f, 1.00000f },
        },
        {
            // index
            { { 0.00000f, -0.26180f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, 0.19998f, -1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 1.04597f, -1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -2.12763f, 1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, 0.46464f, -1.00000f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.10419f, -1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.28344f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.72500f, 1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.30885f, 1.00000f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.14056f, -1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.18702f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.87643f, 1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.37421f, 1.00000f },
        },
        {
            // little
            { { 0.00000f, 0.17453f, -0.11990f }, { 0.00000f, 0.00000f, 1.00000f }, -0.23439f, -1.00000f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.06498f, 1.00000f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.93203f, -1.00000f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.00000f, 0.00000f, 0.00000f }, -1.38471f, 1.00000f },
        },
    },
};

//! Spread is clamped symmetrically: the metacarpal abduction/adduction range, about +/-25 deg.
constexpr float kMaxSpread = 0.436332f;

//! Human bend limits, radians, per bend slot (MCP, PIP, DIP): 0..100 deg, 0..110 deg, 0..110 deg.
//! Negative values are extension and are clamped to the joint's URDF zero, which is the open hand.
constexpr std::array<float, 3> kFingerBendLimit = { 1.745329f, 1.919862f, 1.919862f };

//! Thumb limits, radians, per bend slot (CMC, MCP, IP): 0..60 deg, 0..60 deg, 0..80 deg. "As
//! appropriate": the rig's thumb is a five-link chain mapped onto the anatomy's three thumb bends.
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

    // The root frame anchors both "is this glove streaming" and the parent of every F1 joint. The
    // driver publishes both `link_<Side>SkeletonBase` and `joint_<Side>SkeletonBase`; accept either.
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
        for (std::size_t slot = 0; slot < kRigSlotCount; ++slot)
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

        // Knuckle spread: the angle of the F1 joint, mirrored into "positive towards the index". Its
        // table sign is the hand sign (+ left, - right); it is not a flexion axis.
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

        angles.metacarpal_frame[finger] =
            std::string("link_") + side_prefix(hand) + "Skeleton" + kRigFingers[finger].links[0];
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
