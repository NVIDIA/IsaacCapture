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

//! One revolute joint of the vendor's ``blender_human_skeleton_v5.urdf`` that still drives the
//! anatomy after flexion moved to the fingertip: the knuckle spread (F1, or Thumb1 for the thumb's
//! lift) and the thumb's revolute mount. The bends themselves are not read as angles any more.
struct SpreadJointSpec
{
    //! URDF ``<origin rpy>``: the fixed rotation from the parent frame to this joint's zero pose,
    //! radians, applied as Rz(yaw) * Ry(pitch) * Rx(roll) like URDF does.
    std::array<float, 3> rest_rpy;
    //! URDF ``<axis xyz>``: the revolute axis in the joint frame (unit length in the URDF).
    std::array<float, 3> axis;
    //! Absolute joint angle of the open hand, radians: the calibration offset subtracted before the
    //! anatomy is driven. This is the open-plateau zero measured on the vendor recordings; it is a
    //! rig constant, not a per-session calibration.
    float zero;
    //! +1 when the calibrated angle grows with a positive mechanism angle, -1 when it shrinks. For
    //! the fingers it is the measured spread sign: the v5 first joint turns the finger towards -X in
    //! the mechanism base frame, which is ulnar on the left (+X towards the index) and radial on the
    //! right.
    float sign;
};

//! Spread/lift and mount constants. Only slot 0 of the old four-joint table survives: the flexion
//! now comes from the tip chord, and the spread, the thumb's lift and its mount are the only DOFs
//! that still need the linkage's joint angles.
//!
//! ``blender_human_skeleton_v5.urdf`` is
//! ``.../io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/``. Every finger
//! joint is revolute with an ``<origin rpy>`` and an ``<axis xyz>`` and the driver publishes both
//! frames in ``base_link``, so ``R_rel = conj(q_parent) * q_child`` and
//! ``theta = signed_angle(conj(q_rest) * R_rel, axis)`` is the joint's absolute angle relative to
//! the URDF zero pose. The zeros are the recordings' open-plateau medians; the left ThumbBase rest
//! is the URDF's (0, 0, 0.99442) plus the mount recording's -0.26114 rad roll, because the left
//! mount's URDF rest lacks the right's pi roll and would otherwise read its open pose near +pi.
constexpr SpreadJointSpec kSpreadJoints[2][kRigFingerCount] = {
    // left
    {
        { { 1.57080f, 1.57080f, 0.00000f }, { 0.0f, 0.0f, 1.0f }, -0.67447f, -1.0f }, // thumb lift
        { { 0.00000f, 0.26180f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.27738f, -1.0f }, // index
        { { 0.00000f, 0.00000f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.02953f, -1.0f }, // middle
        { { 0.00000f, 0.00000f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.04644f, -1.0f }, // ring
        { { 0.00000f, -0.17453f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.05504f, -1.0f }, // little
    },
    // right
    {
        { { 1.57080f, 1.57080f, 0.00000f }, { 0.0f, 0.0f, 1.0f }, 0.44692f, +1.0f }, // thumb lift
        { { 0.00000f, -0.26180f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.19091f, +1.0f }, // index
        { { 0.00000f, 0.00000f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.01183f, +1.0f }, // middle
        { { 0.00000f, 0.00000f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.05478f, +1.0f }, // ring
        { { 0.00000f, 0.17453f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.23324f, +1.0f }, // little
    },
};

//! ThumbBase is the thumb's mount, not a flexion joint: the mechanism's own revolute axis, its open
//! zero and its sign. The left rest carries the pi-roll correction described above.
constexpr SpreadJointSpec kThumbBase[2] = {
    { { -0.26114f, 0.00000f, 0.99442f }, { 1.0f, 0.0f, 0.0f }, 0.00083f, +1.0f }, // left
    { { 3.14160f, 0.00000f, 2.14720f }, { 1.0f, 0.0f, 0.0f }, 0.19332f, -1.0f }, // right
};

//! Spread is clamped symmetrically: the metacarpal abduction/adduction range, about +/-25 deg. The
//! thumb's lift uses the same symmetric clamp.
constexpr float kMaxSpread = 0.436332f;

//! ThumbBase is the thumb's mount lift; its useful anatomical range is a quarter turn either way.
constexpr float kMaxThumbBase = 1.570796f;

//! Dead band at the open end, radians (5 deg): a tip chord a little longer than the straight chain,
//! or a joint's share of a nearly open bend, may extend this far instead of being flattened onto
//! zero, and no further. It keeps a slightly hyperextended open hand from reading as flexed.
constexpr float kDeadBand = 0.0872665f;

const char* side_prefix(Hand hand)
{
    return hand == Hand::Left ? "Left" : "Right";
}

const char* side_name(Hand hand)
{
    return hand == Hand::Left ? "left" : "right";
}

//! Which hand a table row belongs to.
constexpr std::size_t hand_index(Hand hand)
{
    return hand == Hand::Left ? 0 : 1;
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

float length_of(const XrVector3f& v)
{
    return std::sqrt(dot(v, v));
}

XrVector3f cross(const XrVector3f& a, const XrVector3f& b)
{
    return XrVector3f{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

bool normalize(XrVector3f& v)
{
    const float length = length_of(v);
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

XrVector3f frame_translation(const geometry_msgs::msg::TransformStamped& frame)
{
    return XrVector3f{ static_cast<float>(frame.transform.translation.x),
                       static_cast<float>(frame.transform.translation.y),
                       static_cast<float>(frame.transform.translation.z) };
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

const char* mechanism_root_frame(Hand hand)
{
    return hand == Hand::Left ? "link_LeftSkeletonBase" : "link_RightSkeletonBase";
}

//! Absolute spread/lift angles read from one message, before the open zero and sign are applied.
struct RawSpread
{
    std::array<float, kRigFingerCount> joint{};
    float thumb_base = 0.0f;
    bool valid = false;
};

RawSpread read_raw_spread(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    RawSpread raw;

    // The root frame anchors "is this glove streaming"; every finger's first joint hangs off it,
    // except the thumb's, which hangs off the mount read below. The driver publishes both
    // `link_<Side>SkeletonBase` and `joint_<Side>SkeletonBase`; accept either.
    const geometry_msgs::msg::TransformStamped* root = find_frame(msg, mechanism_root_frame(hand));
    if (root == nullptr)
    {
        root = find_frame(msg, std::string("joint_") + side_prefix(hand) + "SkeletonBase");
    }
    const geometry_msgs::msg::TransformStamped* thumb_base =
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonThumbBase");
    if (root == nullptr || thumb_base == nullptr)
    {
        return raw;
    }

    const XrQuaternionf root_rotation = frame_rotation(*root);
    const SpreadJointSpec& base_spec = kThumbBase[hand_index(hand)];
    const XrQuaternionf base_relative =
        multiply_quaternions(conjugate_quaternion(root_rotation), frame_rotation(*thumb_base));
    const XrQuaternionf base_residual =
        multiply_quaternions(conjugate_quaternion(quaternion_from_rpy(base_spec.rest_rpy)), base_relative);
    raw.thumb_base = signed_angle_about(base_residual, base_spec.axis);

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        XrQuaternionf parent_rotation = root_rotation;
        // The thumb's first joint hangs off its revolute mount; the mount's published frame is its
        // parent, so Thumb1's angle is independent of the mount angle read above.
        if (kRigFingers[finger].parent[0] != '\0')
        {
            parent_rotation = frame_rotation(*thumb_base);
        }
        const std::string name = std::string("link_") + side_prefix(hand) + "Skeleton" + kRigFingers[finger].links[0];
        const geometry_msgs::msg::TransformStamped* first = find_frame(msg, name);
        if (first == nullptr)
        {
            // A partial hand is not a valid pose: the missing finger would otherwise stay at its
            // open spread and be injected as an open finger.
            return raw;
        }
        const SpreadJointSpec& spec = kSpreadJoints[hand_index(hand)][finger];
        const XrQuaternionf relative =
            multiply_quaternions(conjugate_quaternion(parent_rotation), frame_rotation(*first));
        const XrQuaternionf residual =
            multiply_quaternions(conjugate_quaternion(quaternion_from_rpy(spec.rest_rpy)), relative);
        raw.joint[finger] = signed_angle_about(residual, spec.axis);
    }

    raw.valid = true;
    return raw;
}

float chord_at(const BendModel& model, float total_bend)
{
    const std::array<float, 3> joint = distribute_bend(model, total_bend);
    float cumulative = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    for (std::size_t i = 0; i < 3; ++i)
    {
        cumulative += joint[i];
        x += model.segment_m[i] * std::cos(cumulative);
        y += model.segment_m[i] * std::sin(cumulative);
    }
    return std::sqrt(x * x + y * y);
}

XrHandJointLocationEXT make_joint(const XrPosef& pose)
{
    XrHandJointLocationEXT joint{};
    joint.pose = pose;
    joint.radius = kHandJointRadius;
    joint.locationFlags = kPoseValidFlags;
    return joint;
}

bool same_pose(const XrPosef& a, const XrPosef& b)
{
    // Bit-identical, as the conformance check downstream compares it: every component, not a
    // tolerance. A floating-point difference of one ulp is a live source.
    return a.position.x == b.position.x && a.position.y == b.position.y && a.position.z == b.position.z &&
           a.orientation.x == b.orientation.x && a.orientation.y == b.orientation.y &&
           a.orientation.z == b.orientation.z && a.orientation.w == b.orientation.w;
}

} // namespace

BendModel bend_model(std::size_t finger)
{
    const HumanFinger& anatomy = kHumanFingers[finger];
    // The thumb's chord starts at the mechanism's ThumbBase and spans the CMC->MCP bone too; a
    // finger's chord starts at its MCP, so its three bones are the PIP, DIP and tip offsets.
    const std::size_t first = (finger == 0) ? 1 : 2;

    BendModel model{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        model.segment_m[i] = length_of(anatomy.landmarks[first + i].offset);
    }
    model.weights = kFingerBendWeights;
    model.max_rad = (finger == 0) ? kThumbBendMaxRad : kFingerBendMaxRad;
    return model;
}

std::array<float, 3> distribute_bend(const BendModel& model, float total_bend)
{
    std::array<float, 3> joint{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        joint[i] = std::min(total_bend * model.weights[i], model.max_rad[i]);
    }
    return joint;
}

float solve_total_bend(const BendModel& model, float chord_m)
{
    const float straight = model.segment_m[0] + model.segment_m[1] + model.segment_m[2];
    if (!(chord_m < straight))
    {
        return 0.0f;
    }

    // The largest total bend that still means something: the point at which the last joint reaches
    // its own maximum. Past it every joint is clamped and the chord cannot shrink further.
    float max_total = 0.0f;
    for (std::size_t i = 0; i < 3; ++i)
    {
        max_total = std::max(max_total, model.max_rad[i] / model.weights[i]);
    }
    if (chord_m <= chord_at(model, max_total))
    {
        return max_total;
    }

    // chord_at() is strictly decreasing over [0, max_total] for these chains, so a plain bisection
    // finds the one total bend whose tip lands on the measured chord.
    float low = 0.0f;
    float high = max_total;
    for (int iteration = 0; iteration < 60; ++iteration)
    {
        const float middle = 0.5f * (low + high);
        if (chord_at(model, middle) > chord_m)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    return 0.5f * (low + high);
}

HandChords read_tip_chords(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    HandChords chords;

    // The palm frame needs the knuckle row and the four knuckles' spread axes. Its dorsal axis is
    // the mean of those axes: the URDF's spread axis is the palm normal, and unlike the mechanism
    // root it is live in every message and does not depend on the hand's room pose.
    const geometry_msgs::msg::TransformStamped* const knuckles[kRigFingerCount - 1] = {
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonIndex1"),
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonMiddle1"),
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonRing1"),
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonPinky1"),
    };
    const geometry_msgs::msg::TransformStamped* thumb_base =
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonThumbBase");
    if (thumb_base == nullptr)
    {
        return chords;
    }
    for (const geometry_msgs::msg::TransformStamped* knuckle : knuckles)
    {
        if (knuckle == nullptr)
        {
            return chords;
        }
    }

    // All of these translations are the frame's pose in its parent; subtracting them is only
    // meaningful when the driver publishes them in the same parent frame.
    const std::string& frame_id = knuckles[0]->header.frame_id;
    if (knuckles[1]->header.frame_id != frame_id || knuckles[2]->header.frame_id != frame_id ||
        knuckles[3]->header.frame_id != frame_id || thumb_base->header.frame_id != frame_id)
    {
        return chords;
    }

    XrVector3f dorsal{ 0.0f, 0.0f, 0.0f };
    for (const geometry_msgs::msg::TransformStamped* knuckle : knuckles)
    {
        const XrVector3f axis = rotate_vector(frame_rotation(*knuckle), XrVector3f{ 0.0f, 0.0f, 1.0f });
        dorsal.x += axis.x;
        dorsal.y += axis.y;
        dorsal.z += axis.z;
    }
    if (!normalize(dorsal))
    {
        return chords;
    }

    // Anatomical across: little side -> index side on both hands. The right hand is the X mirror of
    // the left, so its raw index-little vector points the other way and is flipped here; without
    // that the cross product would put "palmward" on the back of the right hand.
    const XrVector3f index = frame_translation(*knuckles[0]);
    const XrVector3f little = frame_translation(*knuckles[3]);
    XrVector3f across{ index.x - little.x, index.y - little.y, index.z - little.z };
    if (hand == Hand::Right)
    {
        across.x = -across.x;
        across.y = -across.y;
        across.z = -across.z;
    }
    // Orthonormalise: the knuckle row is not exactly perpendicular to the mean spread axis.
    const float along_dorsal = dot(across, dorsal);
    across.x -= along_dorsal * dorsal.x;
    across.y -= along_dorsal * dorsal.y;
    across.z -= along_dorsal * dorsal.z;
    if (!normalize(across))
    {
        return chords;
    }
    const XrVector3f distal = normalized(cross(dorsal, across));

    for (std::size_t finger = 0; finger < kHumanFingerCount; ++finger)
    {
        const std::string tip_name = std::string(side_name(hand)) + "_" + kRigFingers[finger].tip + "_tip";
        const geometry_msgs::msg::TransformStamped* tip = find_frame(msg, tip_name);
        if (tip == nullptr || tip->header.frame_id != frame_id)
        {
            return chords;
        }

        const XrVector3f mcp = (finger == 0) ? frame_translation(*thumb_base) : frame_translation(*knuckles[finger - 1]);
        const XrVector3f tip_position = frame_translation(*tip);
        const XrVector3f v{ tip_position.x - mcp.x, tip_position.y - mcp.y, tip_position.z - mcp.z };
        const float chord = length_of(v);
        if (!(chord > 1e-4f))
        {
            return chords;
        }

        TipChord& out = chords.fingers[finger];
        out.chord_m = chord;
        // Positive towards the palm: dorsal is the back of the hand, so the palm side is -dorsal.
        out.palmward_rad = std::atan2(-dot(v, dorsal), dot(v, distal));
        out.valid = true;
    }

    chords.valid = true;
    return chords;
}

HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg, Hand hand)
{
    HandAngles angles;

    const HandChords chords = read_tip_chords(msg, hand);
    if (!chords.valid)
    {
        return angles;
    }
    const RawSpread raw = read_raw_spread(msg, hand);
    if (!raw.valid)
    {
        return angles;
    }

    const std::size_t side = hand_index(hand);
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        FingerAngles& out = angles.fingers[finger];

        // The knuckle spread (or the thumb's lift) still comes from the linkage: it is not flexion
        // and the tip chord does not carry it. Re-zero it on the open hand and sign it anatomically.
        const SpreadJointSpec& spread_spec = kSpreadJoints[side][finger];
        out.spread = std::clamp(spread_spec.sign * (raw.joint[finger] - spread_spec.zero), -kMaxSpread, kMaxSpread);

        // Flexion: one total bend whose chain tip lands on the measured chord, split over the three
        // joints. The palmward angle decides the sign, so a tip on the palm side always curls the
        // hand towards the palm and no linkage convention can invert it.
        const BendModel model = bend_model(finger);
        const float total_bend = solve_total_bend(model, chords.fingers[finger].chord_m);
        const std::array<float, 3> bend = distribute_bend(model, total_bend);
        const float sign = chords.fingers[finger].palmward_rad >= 0.0f ? 1.0f : -1.0f;
        for (std::size_t joint = 0; joint < 3; ++joint)
        {
            out.bend[joint] = std::clamp(sign * bend[joint], -kDeadBand, model.max_rad[joint]);
        }
        angles.chord_m[finger] = chords.fingers[finger].chord_m;
        angles.palmward_rad[finger] = chords.fingers[finger].palmward_rad;

        if (finger == 0)
        {
            // The thumb's mount is a fourth revolute joint; without it the thumb's carry angle does
            // not reach the anatomy. It is a lift, so it is clamped to the anatomical range.
            const SpreadJointSpec& base = kThumbBase[side];
            out.base = std::clamp(base.sign * (raw.thumb_base - base.zero), -kMaxThumbBase, kMaxThumbBase);
        }
    }

    angles.valid = true;
    return angles;
}

float middle_joint_span(const HandSkeleton& skeleton)
{
    const XrVector3f& proximal = skeleton.joints[XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT].pose.position;
    const XrVector3f& tip = skeleton.joints[XR_HAND_JOINT_MIDDLE_TIP_EXT].pose.position;
    const XrVector3f delta{ tip.x - proximal.x, tip.y - proximal.y, tip.z - proximal.z };
    return std::sqrt(dot(delta, delta));
}

FrozenWristGuard::Action FrozenWristGuard::frozen_action(bool controller_available) const
{
    return controller_available ? Action::SwitchToController : Action::Suppress;
}

FrozenWristGuard::Result FrozenWristGuard::update(const XrPosef& pose,
                                                  const std::array<float, kHumanFingerCount>& chords_m,
                                                  bool controller_available,
                                                  double now_s)
{
    Result result;

    if (!m_has_pose)
    {
        m_pose = pose;
        m_chords = chords_m;
        m_has_pose = true;
        return result;
    }

    if (!same_pose(pose, m_pose))
    {
        const bool was_frozen = m_frozen;
        m_pose = pose;
        m_chords = chords_m;
        m_frozen = false;
        m_has_duplicate = false;
        m_first_duplicate_s = 0.0;
        result.thawed = was_frozen;
        return result;
    }

    float moved = 0.0f;
    for (std::size_t finger = 0; finger < kHumanFingerCount; ++finger)
    {
        moved = std::max(moved, std::fabs(chords_m[finger] - m_chords[finger]));
    }
    m_chords = chords_m;

    if (m_frozen)
    {
        result.action = frozen_action(controller_available);
        return result;
    }
    if (!(moved > kChordMotionM))
    {
        // The wrist repeats but the fingers are still: there is no evidence of a stall, and the
        // frame is harmless (the downstream check only looks at frames where the fingers moved).
        return result;
    }
    if (!m_has_duplicate || !(now_s >= m_first_duplicate_s))
    {
        m_has_duplicate = true;
        m_first_duplicate_s = now_s;
        return result;
    }
    if (now_s - m_first_duplicate_s >= kFrozenHoldS)
    {
        m_frozen = true;
        result.action = frozen_action(controller_available);
        result.newly_frozen = true;
    }
    return result;
}

void FrozenWristGuard::reset()
{
    m_has_pose = false;
    m_frozen = false;
    m_has_duplicate = false;
    m_first_duplicate_s = 0.0;
    m_chords = {};
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

    // The anatomy is the left hand; the right hand is its mirror in X, applied once at the end. The
    // articulation is anatomical for both hands (positive spread towards the index, positive bends
    // towards the palm), so the same chain serves both.
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
            // Place the landmark first. A joint rotates the segment *after* its own landmark, so
            // flexing the MCP cannot move the MCP itself.
            const XrVector3f bone = rotate_vector(rotation, anatomy.landmarks[landmark].offset);
            position = XrVector3f{ position.x + bone.x, position.y + bone.y, position.z + bone.z };

            // Orient the bone that starts at this landmark. The CMC carries the metacarpal: the
            // A-pose fan for a finger, the thumb's own frame plus its revolute mount and Thumb1 lift
            // for the thumb. The knuckle then adds the articulation, and PIP/DIP (IP for the thumb)
            // bend the following bones.
            if (landmark == 0)
            {
                if (is_thumb)
                {
                    // R_base * R_lift * rest, the order the derivation script uses: the lift
                    // axis is fixed in the mount's frame, so the mount rotation carries it.
                    rotation = quaternion_from_rpy(kThumbRestRpy);
                    rotation = multiply_quaternions(
                        axis_angle(
                            XrVector3f{ kThumbLiftAxis[0], kThumbLiftAxis[1], kThumbLiftAxis[2] }, articulation.spread),
                        rotation);
                    rotation = multiply_quaternions(
                        axis_angle(
                            XrVector3f{ kThumbBaseAxis[0], kThumbBaseAxis[1], kThumbBaseAxis[2] }, articulation.base),
                        rotation);
                    rotation = multiply_quaternions(
                        axis_angle(XrVector3f{ kThumbFlexAxis[0], kThumbFlexAxis[1], kThumbFlexAxis[2] },
                                   -articulation.bend[0]),
                        rotation);
                }
                else
                {
                    rotation = quaternion_from_rpy(kFingerFanRpy[finger - 1]);
                }
            }
            else if (landmark == 1)
            {
                if (is_thumb)
                {
                    // Thumb2 at the MCP: flexion in the palm plane.
                    rotation = multiply_quaternions(
                        axis_angle(XrVector3f{ kThumbFlexAxis[0], kThumbFlexAxis[1], kThumbFlexAxis[2] },
                                   -articulation.bend[1]),
                        rotation);
                }
                else
                {
                    // The fan only carries the metacarpal; the phalanx restarts in the wrist frame so
                    // the fingers are not splayed, then abducts about the palm normal and flexes
                    // about +X (curl towards the palm, i.e. -Y).
                    rotation = XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
                    rotation =
                        multiply_quaternions(axis_angle(XrVector3f{ 0.0f, 1.0f, 0.0f }, -articulation.spread), rotation);
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[0]));
                }
            }
            else if (landmark + 1 < anatomy.count)
            {
                if (is_thumb)
                {
                    // Thumb3 at the IP: the third flexion in the palm plane.
                    rotation = multiply_quaternions(
                        axis_angle(XrVector3f{ kThumbFlexAxis[0], kThumbFlexAxis[1], kThumbFlexAxis[2] },
                                   -articulation.bend[2]),
                        rotation);
                }
                else
                {
                    rotation = multiply_quaternions(
                        rotation, axis_angle(XrVector3f{ 1.0f, 0.0f, 0.0f }, -articulation.bend[landmark - 1]));
                }
            }

            if (landmark + 1 < anatomy.count)
            {
                forward = anatomy.landmarks[landmark + 1].offset;
            }
            // The tip has no following bone, so it keeps the last bone's direction.
            const XrVector3f forward_world = normalized(rotate_vector(rotation, forward));
            const XrVector3f dorsal_world = rotate_vector(rotation, XrVector3f{ 0.0f, 1.0f, 0.0f });
            const XrQuaternionf orientation = orientation_along(forward_world, dorsal_world);

            skeleton.joints[anatomy.landmarks[landmark].joint] = make_joint(XrPosef{ orientation, position });
            ++skeleton.filled;
        }
    }

    if (hand == Hand::Right)
    {
        // M * R * M for the x mirror: positions flip x; a quaternion (x, y, z, w) becomes
        // (x, -y, -z, w). The bones stay on -Z and the palm normal stays +Y.
        for (XrHandJointLocationEXT& joint : skeleton.joints)
        {
            joint.pose.position.x = -joint.pose.position.x;
            joint.pose.orientation = XrQuaternionf{ joint.pose.orientation.x, -joint.pose.orientation.y,
                                                    -joint.pose.orientation.z, joint.pose.orientation.w };
        }
    }

    return skeleton;
}

} // namespace io_exo_glove_ros2
} // namespace plugins
