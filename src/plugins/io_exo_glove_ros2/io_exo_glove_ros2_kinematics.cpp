// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

//! Revolute joints read per finger, in chain order: the knuckle spread F1 and the bends F2..F4.
//! The thumb's are Thumb1..Thumb4 (ThumbBase is its fixed mount).
constexpr std::size_t kRigSlotCount = 4;

//! Index of ThumbBase in the kJointZeroOffsetCount order: after the five fingers' four slots.
constexpr std::size_t kThumbBaseZeroIndex = kJointZeroOffsetCount - 1;

//! One joint's config offset, or zero when no offsets were configured.
float zero_offset_at(const JointZeroOffsets* offsets, std::size_t index)
{
    return offsets != nullptr && offsets->set ? offsets->radians[index] : 0.0f;
}

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
    //! +1 when the calibrated angle grows with a positive mechanism angle, -1 when it shrinks.
    //! For the fingers' slot 0 this is the *measured* spread sign, not a hand guess: the v5 first
    //! joint turns the finger towards -X in the mechanism base frame, which is ulnar on the left
    //! (+X towards the index) and radial on the right.
    float sign;
    //! Absolute linkage angle at the fist reference, radians, pre-sign and pre-zero: the medoid of
    //! the recording's closed plateau. ``sign * (fist - zero)`` is the travel the gain divides by.
    float fist;
    //! Human anatomical maximum this joint should reach at the fist, radians. The calibrated travel
    //! is mapped onto it, because the linkage moves roughly half as far as the finger. For slot 0
    //! this is the symmetric spread clamp and the gain is not applied; see @c kMaxSpread.
    float human_max;
};

//! Rest rotation, revolute axis, open-hand zero, sign, fist reference and human maximum for the four
//! joints per finger that drive the anatomy: the knuckle (F1 -> spread) and the three bends (F2 ->
//! MCP, F3 -> PIP, F4 -> DIP). For the thumb those are Thumb1 (lift) and Thumb2..Thumb4 (CMC, MCP,
//! IP); its mount, ThumbBase, has its own spec in @c kThumbBase.
//!
//! Derivation (performed by ``io_exo_glove_ros2_kinematics_tables.py``; the thumb block from the
//! 2500-message thumb-only recording, the finger rows retained from the earlier 7251-message
//! all-gesture derivation):
//!
//!  - The v5 URDF is
//!    ``.../io_exotrans2hand_project_zenoh_22.04_x86_v2.1.4/src/exo_tf_pkg/description/blender_human_skeleton_v5.urdf``.
//!    Every finger joint is ``revolute`` with an ``<origin xyz rpy>`` and an ``<axis xyz>``, and the
//!    driver publishes both frames in ``base_link``. Therefore
//!    ``q_child = q_parent * q_rest(rpy) * q_axis(theta)`` and, with the published quaternions,
//!    ``R_rel = conj(q_parent) * q_child`` and ``theta = signed_angle(conj(q_rest) * R_rel, axis)``
//!    is the joint's absolute angle relative to the URDF zero pose. ``read_hand_angles`` computes it
//!    that way; no frame position, frame ordering or first-frame capture is involved.
//!  - Every joint angle is unwrapped across messages: the formula above returns the shortest signed
//!    angle in (-pi, pi], which jumps by 2*pi when the joint crosses the boundary. The left
//!    ThumbBase does exactly that (its rest has no pi roll, unlike the right's), so before the
//!    unwrapper it reports 359.8 deg of travel instead of the mount's real 26.6 deg and the thumb
//!    model moves opposite the vendor tip. ``JointAngleUnwrapper`` adds +-2*pi until the reading is
//!    within pi of the previous sample; the first sample initialises.
//!  - Openness is measured along the finger: from each knuckle (the first bend joint) to the
//!    published fingertip. The recording's longest steady open plateau gives the zero, and its
//!    longest steady closed plateau gives the sign and the fist reference, at each plateau's medoid
//!    frame so a drifting pose cannot bias either. The thumb has no finger length to read, so its
//!    plateaus come from the vendor ``<side>_thumb_tip`` frame's displacement along (index MCP -
//!    thumb CMC), which grows monotonically as the thumb curls; the script asserts that monotonicity
//!    after unwrapping and fails if a joint is flipped or turns back.
//!  - ``sign`` is +1 when the joint angle grows from the verified open pose towards the fist, -1
//!    when it shrinks. For the four fingers' slot 0 it is instead derived from the mechanism itself:
//!    the v5 first joint rotates about its local +Z, the finger extends +Y, and R_z(+theta) sends it
//!    towards -X; that is away from the index on the left but towards it on the right, so the sign is
//!    -1 on the left and +1 on the right for every finger. The script measures this again on the
//!    recording against the vendor's ``<side>_<finger>_tip`` frames.
//!  - The left ThumbBase rest rotation is a rig constant, not a per-session calibration. The v5
//!    table gives the right mount (3.1416, 0, 2.1472) but the left only (0, 0, 0.99442): the left is
//!    a half turn from the physical rest, so with the physical rest its open reading sits near +pi
//!    and wraps. The script solves for the roll that makes the open plateau read zero and bakes it:
//!    ``(-0.26114, 0, 0.99442)``, the URDF rest plus the pi roll and the open plateau's median
//!    -14.96 deg offset (mod 2*pi). With it the left mount's zero is 0.05 deg and its travel is
//!    26.6 deg; the right keeps its URDF rest.
//!  - The gain calibration is ``d / d_fist * human_max``, where ``d = sign * (raw - zero)`` and
//!    ``d_fist = sign * (fist - zero)``. The linkage angle is not a joint angle: a real fist reads
//!    about 55 deg at the MCP, 25-29 at the PIP and 45 at the DIP, roughly half the human range, so
//!    copying it 1:1 never closes the injected hand. ``human_max`` is the anatomical reference --
//!    finger MCP 85, PIP 95, DIP 70 deg; thumb CMC 35, MCP 55, IP 75 deg -- and the script checks
//!    that the recorded fist maps onto it on both hands. The spread keeps its symmetric clamp
//!    (@c kMaxSpread): the recording's spread excursion is not the full anatomical range, and the
//!    thumb's mount keeps @c kMaxThumbBase.
//!  - The derivation checks that the open zero places each model fingertip one finger-length ahead
//!    of its MCP and that every mapped MCP/PIP/DIP reads positive at the fist on both hands.
constexpr RigJointSpec kRigJoints[2][kRigFingerCount][kRigSlotCount] = {
    // left
    {
        {
            // thumb (thumb-only recording: Thumb1 lift 8.1 deg, CMC 50.5, MCP 71.1, IP 158.6 deg travel)
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.0f, 0.0f, 1.0f }, -0.67447f, -1.0f, -0.81633f, 0.43633f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.88166f, -1.0f, -0.00030f, 0.61087f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -2.35199f, +1.0f, -1.11137f, 0.95993f },
            { { 2.53070f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 1.81010f, -1.0f, -0.95807f, 1.30900f },
        },
        {
            // index
            { { 0.00000f, 0.26180f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.27738f, -1.0f, -0.36191f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.04983f, +1.0f, 1.00309f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.86301f, -1.0f, -2.29023f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.13192f, +1.0f, 0.65156f, 1.22173f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.02953f, -1.0f, -0.31307f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.15469f, +1.0f, 0.90542f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.61346f, -1.0f, -2.11625f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.03451f, +1.0f, 0.63469f, 1.22173f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.04644f, -1.0f, -0.31401f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.14160f, +1.0f, 0.95443f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.66886f, -1.0f, -2.18084f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.05541f, +1.0f, 0.64552f, 1.22173f },
        },
        {
            // little
            { { 0.00000f, -0.17453f, 0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.05504f, -1.0f, -0.25993f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.48432f, +1.0f, 0.85920f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -2.35083f, -1.0f, -2.47473f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.85100f, +1.0f, 0.93649f, 1.22173f },
        },
    },
    // right
    {
        {
            // thumb (thumb-only recording: Thumb1 lift 14.6 deg, CMC 58.9, MCP 74.9, IP 146.9 deg travel)
            { { 1.57080f, 1.57080f, 0.00000f }, { 0.0f, 0.0f, 1.0f }, 0.44692f, +1.0f, 0.70200f, 0.43633f },
            { { -1.30900f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.99325f, -1.0f, -0.03413f, 0.61087f },
            { { -0.81433f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -2.21701f, +1.0f, -0.91002f, 0.95993f },
            { { 2.53070f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 1.63599f, -1.0f, -0.92806f, 1.30900f },
        },
        {
            // index
            { { 0.00000f, -0.26180f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.19091f, +1.0f, 0.05781f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.03796f, +1.0f, 1.02662f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.67321f, -1.0f, -2.14679f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.04263f, +1.0f, 0.51657f, 1.22173f },
        },
        {
            // middle
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, 0.01183f, +1.0f, 0.14118f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.45659f, +1.0f, 0.88632f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.25074f, -1.0f, -1.98571f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.38185f, +1.0f, 0.51500f, 1.22173f },
        },
        {
            // ring
            { { 0.00000f, 0.00000f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.05478f, +1.0f, 0.11295f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.03685f, +1.0f, 0.90422f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -1.60046f, -1.0f, -2.10111f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -0.05488f, +1.0f, 0.63009f, 1.22173f },
        },
        {
            // little
            { { 0.00000f, 0.17453f, -0.11990f }, { 0.0f, 0.0f, 1.0f }, -0.23324f, +1.0f, 0.06695f, 0.43633f },
            { { -0.75647f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.39179f, +1.0f, 0.97634f, 1.48353f },
            { { 0.46849f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, -2.18509f, -1.0f, -2.51505f, 1.65806f },
            { { 1.22170f, 0.00000f, 0.00000f }, { 1.0f, 0.0f, 0.0f }, 0.42326f, +1.0f, 1.10365f, 1.22173f },
        },
    },
};

//! ThumbBase is the thumb's mount, not a flexion joint: the table records its fist reference for the
//! same shape as @c kRigJoints, but its own symmetric clamp in @c kMaxThumbBase is kept.
//!
//! The left rest is the URDF's (0, 0, 0.99442) plus the thumb recording's open-plateau median roll of
//! -0.26114 rad (-14.96 deg). The URDF left mount has no pi roll (the right's is 3.1416, 0, 2.1472),
//! so without it the left open reading sits near +pi, wraps, and the script reports 359.8 deg of
//! travel. With it the open zero is 0.00083 rad (0.05 deg) and the mount travels 26.6 deg. This is a
//! rig constant, not a per-session calibration; the right keeps its URDF rest.
constexpr RigJointSpec kThumbBase[2] = {
    { { -0.26114f, 0.00000f, 0.99442f }, { 1.0f, 0.0f, 0.0f }, 0.00083f, +1.0f, 0.46478f, 1.57080f }, // left
    { { 3.14160f, 0.00000f, 2.14720f }, { 1.0f, 0.0f, 0.0f }, 0.19332f, -1.0f, -0.08974f, 1.57080f }, // right
};

//! Spread is clamped symmetrically: the metacarpal abduction/adduction range, about +/-25 deg. The
//! table's slot-0 human_max repeats this value, but the gain is not applied to the spread.
constexpr float kMaxSpread = 0.436332f;

//! ThumbBase is the thumb's mount lift; its useful anatomical range is a quarter turn either way.
constexpr float kMaxThumbBase = 1.570796f;

//! Dead band at the open end, radians (5 deg): the open plateau drifts by 3-5 deg, so flexion may
//! extend this far below the zero instead of being flattened onto it, and no further.
constexpr float kDeadBand = 0.0872665f;

//! Smallest calibrated travel to the fist the gain can divide by, radians. Below it the joint keeps
//! the old 1:1 ``sign * (raw - zero)`` mapping rather than amplifying a near-zero denominator.
constexpr float kMinFistTravel = 0.05f;

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

//! Half a turn, and the tolerance used to keep a new reading within half a turn of the previous one.
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

//! Absolute mechanism angles read from one message, before calibration: [finger][F1..F4] radians
//! relative to the URDF zero pose. Internal, but used by the local harness to check the formula
//! against known synthetic angles.
struct RawHandAngles
{
    std::array<std::array<float, kRigSlotCount>, kRigFingerCount> joint{};
    std::array<bool, kRigFingerCount> complete{};
    //! ThumbBase's absolute angle, and whether its published frame was present.
    float thumb_base = 0.0f;
    bool thumb_base_valid = false;
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

    // ThumbBase is a revolute joint in its own right (and is in /io_esk/joint_data), not a fixed
    // mount: read its angle against the root with the v5 rest rotation and axis. Its published frame
    // is also Thumb1's parent, so the two angles stay independent.
    const geometry_msgs::msg::TransformStamped* thumb_base_frame =
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonThumbBase");
    if (thumb_base_frame != nullptr)
    {
        const RigJointSpec& base_spec = kThumbBase[hand_index(hand)];
        const XrQuaternionf relative =
            multiply_quaternions(conjugate_quaternion(root_rotation), frame_rotation(*thumb_base_frame));
        const XrQuaternionf residual =
            multiply_quaternions(conjugate_quaternion(quaternion_from_rpy(base_spec.rest_rpy)), relative);
        raw.thumb_base = signed_angle_about(residual, base_spec.axis);
        raw.thumb_base_valid = true;
    }

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        bool complete = true;
        XrQuaternionf parent_rotation = root_rotation;
        // The thumb's first joint hangs off its revolute mount; the mount's published frame is its
        // parent, so Thumb1's angle is independent of the mount angle read above.
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
                // An incomplete finger would keep its angles at zero, which the anatomy reads as an
                // open finger. Mark it incomplete and let read_hand_angles invalidate the whole hand:
                // a partial hand is not a valid pose, and half a fist must not be injected as an open
                // hand.
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

//! Advances @p unwrapper over one complete raw reading, so every joint is continuous across messages.
void unwrap_raw_angles(Hand hand, RawHandAngles& raw, JointAngleUnwrapper* unwrapper)
{
    if (unwrapper == nullptr)
    {
        return;
    }
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        for (std::size_t slot = 0; slot < kRigSlotCount; ++slot)
        {
            raw.joint[finger][slot] = unwrapper->unwrap(hand, finger, slot, raw.joint[finger][slot]);
        }
    }
    if (raw.thumb_base_valid)
    {
        raw.thumb_base = unwrapper->unwrap_thumb_base(hand, raw.thumb_base);
    }
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

float JointAngleUnwrapper::unwrap(Hand hand, std::size_t finger, std::size_t slot, float value)
{
    const std::size_t side = hand_index(hand);
    if (!m_joint_initialized[side][finger][slot])
    {
        m_joint_initialized[side][finger][slot] = true;
        m_joint[side][finger][slot] = value;
        return value;
    }
    const float previous = m_joint[side][finger][slot];
    while (value - previous > kPi)
    {
        value -= kTwoPi;
    }
    while (value - previous < -kPi)
    {
        value += kTwoPi;
    }
    m_joint[side][finger][slot] = value;
    return value;
}

float JointAngleUnwrapper::unwrap_thumb_base(Hand hand, float value)
{
    const std::size_t side = hand_index(hand);
    if (!m_thumb_base_initialized[side])
    {
        m_thumb_base_initialized[side] = true;
        m_thumb_base[side] = value;
        return value;
    }
    const float previous = m_thumb_base[side];
    while (value - previous > kPi)
    {
        value -= kTwoPi;
    }
    while (value - previous < -kPi)
    {
        value += kTwoPi;
    }
    m_thumb_base[side] = value;
    return value;
}

void JointAngleUnwrapper::reset()
{
    m_joint_initialized = {};
    m_thumb_base_initialized = {};
}

const char* mechanism_root_frame(Hand hand)
{
    return hand == Hand::Left ? "link_LeftSkeletonBase" : "link_RightSkeletonBase";
}

HandAngles read_hand_angles(const tf2_msgs::msg::TFMessage& msg,
                            Hand hand,
                            JointAngleUnwrapper* unwrapper,
                            bool flip_bends,
                            const JointZeroOffsets* zero_offsets)
{
    HandAngles angles;

    RawHandAngles raw = read_raw_hand_angles(msg, hand);
    if (!raw.valid)
    {
        return angles;
    }

    // A hand missing any finger is invalid, not a valid hand with one open finger: the missing
    // finger's angles would stay at zero, the anatomy would extend it, and the injector would keep a
    // hand whose other fingers are a fist. build_anatomical_skeleton returns an untracked skeleton
    // for an invalid hand, so push_hand drops the injector instead.
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        if (!raw.complete[finger])
        {
            return angles;
        }
    }

    // Make every joint continuous before it is calibrated: the quaternion formula returns the
    // shortest signed angle, so a joint crossing +-pi would otherwise jump a full turn and the sign,
    // gain and zero below would all be read off a wrapped series. The unwrapper is advanced only for
    // a complete hand, so a frame missing a finger never moves the reference.
    unwrap_raw_angles(hand, raw, unwrapper);

    const std::size_t side = hand_index(hand);

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        FingerAngles& out = angles.fingers[finger];
        const bool is_thumb = finger == 0;

        // Knuckle spread: for a finger, the angle of the spread joint (F1) made positive towards the
        // index; for the thumb, Thumb1's lift made positive out of the palm. The sign is derived from
        // the mechanism (left -1, right +1 for the fingers) and measured on the recording. A
        // configured zero offset shifts the open reading of this joint too.
        const RigJointSpec& knuckle = kRigJoints[side][finger][0];
        const float knuckle_zero = knuckle.zero + zero_offset_at(zero_offsets, finger * kRigSlotCount);
        out.spread = std::clamp(knuckle.sign * (raw.joint[finger][0] - knuckle_zero), -kMaxSpread, kMaxSpread);

        // MCP/PIP/DIP (the thumb's CMC/MCP/IP): the linkage angle is not the joint angle. Calibrate
        // it to the open hand, sign it so flexion is positive, then map its open-to-fist travel onto
        // the anatomical maximum: d / d_fist * human_max. A joint whose fist travel is too small to
        // divide by keeps the old 1:1 mapping. The dead band lets the open plateau's drift read a few
        // degrees of extension, and no more.
        for (std::size_t slot = 1; slot < kRigSlotCount; ++slot)
        {
            const RigJointSpec& spec = kRigJoints[side][finger][slot];
            // A configured offset re-zeroes the joint (the measured open pose becomes zero); the
            // sign, fist reference and human range stay the baked ones. The fist travel shrinks by
            // the same amount because it is measured from the (new) zero.
            const float zero = spec.zero + zero_offset_at(zero_offsets, finger * kRigSlotCount + slot);
            // The runtime bend-direction correction multiplies the signed raw delta by -1 *before*
            // the gain mapping and the clamp: negating the calibrated bend afterwards would instead
            // spend the open-side dead band and lose the fist's travel.
            const float d = (flip_bends ? -1.0f : 1.0f) * spec.sign * (raw.joint[finger][slot] - zero);
            const float d_fist = spec.sign * (spec.fist - zero);
            const float flexion = (std::fabs(d_fist) >= kMinFistTravel) ? d / d_fist * spec.human_max : d;
            out.bend[slot - 1] = std::clamp(flexion, -kDeadBand, spec.human_max);
        }

        if (is_thumb)
        {
            // The thumb's mount is a fourth revolute joint; without it the thumb's carry angle does
            // not reach the anatomy. It is a lift, so it is clamped to the anatomical range.
            const RigJointSpec& base = kThumbBase[side];
            const float base_zero = base.zero + zero_offset_at(zero_offsets, kThumbBaseZeroIndex);
            out.base = raw.thumb_base_valid ?
                           std::clamp(base.sign * (raw.thumb_base - base_zero), -kMaxThumbBase, kMaxThumbBase) :
                           0.0f;
        }
    }

    angles.valid = true;
    return angles;
}

bool read_linkage_angles(const tf2_msgs::msg::TFMessage& msg,
                         Hand hand,
                         JointAngleUnwrapper* unwrapper,
                         std::array<float, kJointZeroOffsetCount>& out)
{
    RawHandAngles raw = read_raw_hand_angles(msg, hand);
    if (!raw.valid)
    {
        return false;
    }
    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        if (!raw.complete[finger])
        {
            return false;
        }
    }
    // A complete thumb implies its ThumbBase mount frame; refuse rather than report a zero mount.
    if (!raw.thumb_base_valid)
    {
        return false;
    }

    // The same unwrapper contract read_hand_angles uses: continuity first, then calibration.
    unwrap_raw_angles(hand, raw, unwrapper);

    for (std::size_t finger = 0; finger < kRigFingerCount; ++finger)
    {
        for (std::size_t slot = 0; slot < kRigSlotCount; ++slot)
        {
            out[finger * kRigSlotCount + slot] = raw.joint[finger][slot];
        }
    }
    out[kThumbBaseZeroIndex] = raw.thumb_base;
    return true;
}

float baked_joint_zero(Hand hand, std::size_t index)
{
    const std::size_t side = hand_index(hand);
    if (index < kRigFingerCount * kRigSlotCount)
    {
        return kRigJoints[side][index / kRigSlotCount][index % kRigSlotCount].zero;
    }
    return index == kThumbBaseZeroIndex ? kThumbBase[side].zero : 0.0f;
}

float linkage_window_drift_rad(const LinkageSamples& samples)
{
    if (samples.empty())
    {
        return 0.0f;
    }
    float drift = 0.0f;
    for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
    {
        float lowest = samples.front()[joint];
        float highest = lowest;
        for (const auto& sample : samples)
        {
            lowest = std::min(lowest, sample[joint]);
            highest = std::max(highest, sample[joint]);
        }
        drift = std::max(drift, highest - lowest);
    }
    return drift;
}

bool linkage_window_still(const LinkageSamples& samples, float max_drift_rad)
{
    return linkage_window_drift_rad(samples) < max_drift_rad;
}

std::array<float, kJointZeroOffsetCount> average_linkage_angles(const LinkageSamples& samples)
{
    std::array<float, kJointZeroOffsetCount> mean{};
    if (samples.empty())
    {
        return mean;
    }
    std::array<double, kJointZeroOffsetCount> sum{};
    for (const auto& sample : samples)
    {
        for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
        {
            sum[joint] += sample[joint];
        }
    }
    const double count = static_cast<double>(samples.size());
    for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
    {
        mean[joint] = static_cast<float>(sum[joint] / count);
    }
    return mean;
}

JointZeroOffsets joint_zero_offsets_from_open_pose(Hand hand, const std::array<float, kJointZeroOffsetCount>& open_radians)
{
    JointZeroOffsets offsets;
    offsets.set = true;
    for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
    {
        offsets.radians[joint] = open_radians[joint] - baked_joint_zero(hand, joint);
    }
    return offsets;
}

JointZeroOffsets parse_joint_zero_offsets(const std::string& text, const std::string& what)
{
    JointZeroOffsets offsets;

    std::string separated = text;
    std::replace(separated.begin(), separated.end(), ',', ' ');

    std::istringstream stream(separated);
    std::vector<double> values;
    double value = 0.0;
    while (stream >> value)
    {
        values.push_back(value);
    }
    // A token the stream could not parse is neither an empty value nor a count error; name it as
    // such instead of silently accepting the numbers before it.
    if (!stream.eof())
    {
        throw std::runtime_error(what + " must be " + std::to_string(kJointZeroOffsetCount) +
                                 " numbers of degrees separated by spaces or commas, not '" + text + "'");
    }
    if (values.empty())
    {
        return offsets; // Absent or empty: keep the baked table.
    }
    if (values.size() != kJointZeroOffsetCount)
    {
        throw std::runtime_error(what + " must have " + std::to_string(kJointZeroOffsetCount) +
                                 " values (5 fingers x 4 joints plus ThumbBase), not " + std::to_string(values.size()));
    }
    for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
    {
        if (!std::isfinite(values[joint]))
        {
            throw std::runtime_error(what + " has a non-finite value at position " + std::to_string(joint + 1));
        }
        offsets.radians[joint] = static_cast<float>(values[joint] * (3.14159265358979323846 / 180.0));
    }
    offsets.set = true;
    return offsets;
}

std::string format_joint_zero_offsets_deg(const JointZeroOffsets& offsets)
{
    std::string text;
    char field[32];
    for (std::size_t joint = 0; joint < kJointZeroOffsetCount; ++joint)
    {
        float degrees = offsets.radians[joint] * (180.0f / 3.14159265358979323846f);
        // -0.00 in a value meant to be pasted into a config file reads as an error.
        if (std::fabs(degrees) < 0.005f)
        {
            degrees = 0.0f;
        }
        std::snprintf(field, sizeof(field), "%.2f", degrees);
        if (!text.empty())
        {
            text += ' ';
        }
        text += field;
    }
    return text;
}

float middle_joint_span(const HandSkeleton& skeleton)
{
    const XrVector3f& proximal = skeleton.joints[XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT].pose.position;
    const XrVector3f& tip = skeleton.joints[XR_HAND_JOINT_MIDDLE_TIP_EXT].pose.position;
    const XrVector3f delta{ tip.x - proximal.x, tip.y - proximal.y, tip.z - proximal.z };
    return std::sqrt(dot(delta, delta));
}

bool mechanism_middle_span(const tf2_msgs::msg::TFMessage& msg, Hand hand, float& span_m)
{
    const std::string side = hand == Hand::Left ? "left" : "right";
    const geometry_msgs::msg::TransformStamped* knuckle =
        find_frame(msg, std::string("link_") + side_prefix(hand) + "SkeletonMiddle1");
    const geometry_msgs::msg::TransformStamped* tip = find_frame(msg, side + "_middle_tip");
    if (knuckle == nullptr || tip == nullptr)
    {
        return false;
    }
    // Both translations are that frame's pose in its parent frame; comparing them is only meaningful
    // when the driver publishes the mechanism link and the fingertip in the same frame.
    if (knuckle->header.frame_id != tip->header.frame_id)
    {
        return false;
    }
    const auto& a = knuckle->transform.translation;
    const auto& b = tip->transform.translation;
    const XrVector3f delta{ static_cast<float>(b.x - a.x), static_cast<float>(b.y - a.y), static_cast<float>(b.z - a.z) };
    span_m = std::sqrt(dot(delta, delta));
    return true;
}

BendDirectionChecker::Change BendDirectionChecker::sample(float model_span_m, float glove_span_m, double now_s)
{
    if (!std::isfinite(model_span_m) || !std::isfinite(glove_span_m))
    {
        return Change::None;
    }
    m_model[m_next] = model_span_m;
    m_glove[m_next] = glove_span_m;
    m_next = (m_next + 1) % kBendDirectionWindow;
    if (m_count < kBendDirectionWindow)
    {
        ++m_count;
    }
    if (m_count < kBendDirectionMinSamples)
    {
        return Change::None;
    }
    if (m_evaluated && now_s - m_last_evaluation_s < kBendDirectionPeriodS)
    {
        return Change::None;
    }
    m_evaluated = true;
    m_last_evaluation_s = now_s;

    // A still hand carries no direction: both series are flat and a correlation would be noise.
    float model_min = m_model[0];
    float model_max = m_model[0];
    float glove_min = m_glove[0];
    float glove_max = m_glove[0];
    for (std::size_t i = 1; i < m_count; ++i)
    {
        model_min = std::min(model_min, m_model[i]);
        model_max = std::max(model_max, m_model[i]);
        glove_min = std::min(glove_min, m_glove[i]);
        glove_max = std::max(glove_max, m_glove[i]);
    }
    if (!(model_max - model_min > kBendDirectionSwingM) && !(glove_max - glove_min > kBendDirectionSwingM))
    {
        return Change::None;
    }

    // Pearson correlation of the two windowed scalars. Double accumulation keeps a 64-sample window
    // well-conditioned even though the spans differ by only a few centimetres.
    double sum_model = 0.0;
    double sum_glove = 0.0;
    double sum_model_sq = 0.0;
    double sum_glove_sq = 0.0;
    double sum_product = 0.0;
    for (std::size_t i = 0; i < m_count; ++i)
    {
        const double model = m_model[i];
        const double glove = m_glove[i];
        sum_model += model;
        sum_glove += glove;
        sum_model_sq += model * model;
        sum_glove_sq += glove * glove;
        sum_product += model * glove;
    }
    const double count = static_cast<double>(m_count);
    const double covariance = count * sum_product - sum_model * sum_glove;
    const double model_variance = count * sum_model_sq - sum_model * sum_model;
    const double glove_variance = count * sum_glove_sq - sum_glove * sum_glove;
    if (!(model_variance > 0.0) || !(glove_variance > 0.0))
    {
        return Change::None;
    }
    const double correlation = covariance / std::sqrt(model_variance * glove_variance);
    if (!std::isfinite(correlation))
    {
        return Change::None;
    }
    m_correlation = static_cast<float>(correlation);

    if (correlation < -kBendDirectionCorrelation && !m_flipped)
    {
        m_flipped = true;
        return Change::Flipped;
    }
    if (correlation > kBendDirectionCorrelation && m_flipped)
    {
        m_flipped = false;
        return Change::Unflipped;
    }
    return Change::None;
}

void BendDirectionChecker::clear_window()
{
    m_count = 0;
    m_next = 0;
    m_evaluated = false;
    m_last_evaluation_s = 0.0;
    m_correlation = 0.0f;
    // m_flipped is deliberately kept: a glove that drops out for a moment must not forget a decision
    // that only a new correlation may reverse.
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
            const XrVector3f dorsal_world = rotate_vector(rotation, kDorsal);
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
