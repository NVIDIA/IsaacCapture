// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_hand.hpp"

#include "io_exo_glove_ros2_kinematics.hpp"

#include <oxr_utils/math.hpp>

#include <cmath>
#include <cstddef>
#include <string>

namespace plugins
{
namespace io_exo_glove_ros2
{

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
    // The mechanism supplies the fingertip chords and the spread; the anatomy supplies the
    // landmarks. See io_exo_glove_ros2_kinematics.hpp for why the linkage cannot be injected
    // directly and why the chord cannot invert the hand.
    return build_anatomical_skeleton(read_hand_angles(msg, hand), hand);
}

} // namespace io_exo_glove_ros2
} // namespace plugins
