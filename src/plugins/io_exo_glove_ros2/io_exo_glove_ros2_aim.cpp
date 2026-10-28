// SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io_exo_glove_ros2_aim.hpp"

#include "io_exo_glove_ros2_hand.hpp"

#include <oxr_utils/math.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace plugins
{
namespace io_exo_glove_ros2
{

namespace
{

constexpr XrQuaternionf kIdentityOrientation{ 0.0f, 0.0f, 0.0f, 1.0f };

float length_of(const XrVector3f& vector)
{
    return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

XrVector3f scaled(const XrVector3f& vector, float scale)
{
    return XrVector3f{ vector.x * scale, vector.y * scale, vector.z * scale };
}

//! @p direction rotated by @p orientation, via the same pose composition the plugin uses elsewhere.
XrVector3f rotated_direction(const XrQuaternionf& orientation, const XrVector3f& direction)
{
    return rotate_pose(orientation, XrPosef{ kIdentityOrientation, direction }).position;
}

XrQuaternionf normalized(const XrQuaternionf& quaternion)
{
    const float norm = std::sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                                 quaternion.z * quaternion.z + quaternion.w * quaternion.w);
    if (!(norm > 1e-6f))
    {
        return kIdentityOrientation;
    }
    return XrQuaternionf{ quaternion.x / norm, quaternion.y / norm, quaternion.z / norm, quaternion.w / norm };
}

//! -0.0000 in a value meant to be pasted into a config file reads as an error; round tiny values to
//! a clean zero before the fixed-point conversion does it for us.
float cleaned_for_display(float value, float epsilon)
{
    return std::fabs(value) < epsilon ? 0.0f : value;
}

} // namespace

XrPosef wrist_from_aim(const XrPosef& aim, float behind_m, float height_m, bool flip_dorsal)
{
    // The hand is flat and palm down during calibration: world +Y (session space is Y-up) is the
    // back of the hand, whatever the controller's own roll happens to be. The height offset is in
    // that same world up, so flipping the dorsal hint must not move the wrist.
    const XrVector3f up{ 0.0f, 1.0f, 0.0f };
    const XrVector3f dorsal_hint = flip_dorsal ? XrVector3f{ 0.0f, -1.0f, 0.0f } : up;
    XrVector3f forward = rotated_direction(aim.orientation, XrVector3f{ 0.0f, 0.0f, -1.0f });
    const float forward_length = length_of(forward);
    forward = forward_length > 1e-6f ? scaled(forward, 1.0f / forward_length) : XrVector3f{ 0.0f, 0.0f, -1.0f };

    XrPosef wrist{};
    wrist.position = XrVector3f{ aim.position.x - forward.x * behind_m - up.x * height_m,
                                 aim.position.y - forward.y * behind_m - up.y * height_m,
                                 aim.position.z - forward.z * behind_m - up.z * height_m };
    // -Z along the fingers and +Y dorsal: the convention the injected skeleton uses. A flipped hint
    // flips +X and +Y about -Z, i.e. rolls the wrist 180 degrees about the finger axis.
    wrist.orientation = orientation_along(forward, dorsal_hint);
    return wrist;
}

XrPosef aim_to_wrist(const XrPosef& aim, const XrPosef& wrist)
{
    return oxr_utils::multiply_poses(invert_pose(aim), wrist);
}

XrPosef average_aim_pose(const std::vector<XrPosef>& samples)
{
    if (samples.empty())
    {
        return XrPosef{ kIdentityOrientation, XrVector3f{ 0.0f, 0.0f, 0.0f } };
    }

    // Accumulate the position in double: a second of samples at 200 Hz otherwise loses more than a
    // micrometre to float rounding, which is the order of the drift the calibration checks for.
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    XrQuaternionf orientation = normalized(samples.front().orientation);
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        x += samples[i].position.x;
        y += samples[i].position.y;
        z += samples[i].position.z;
        if (i != 0)
        {
            // Running spherical mean: each sample pulls the mean 1/(i+1) of the way towards itself.
            orientation = slerp_quaternions(orientation, normalized(samples[i].orientation), 1.0f / float(i + 1));
        }
    }

    XrPosef mean{};
    const double count = static_cast<double>(samples.size());
    mean.position =
        XrVector3f{ static_cast<float>(x / count), static_cast<float>(y / count), static_cast<float>(z / count) };
    mean.orientation = normalized(orientation);
    return mean;
}

AimDrift aim_window_drift(const std::vector<XrPosef>& samples)
{
    AimDrift drift;
    if (samples.empty())
    {
        return drift;
    }

    const XrPosef& reference = samples.front();
    const XrQuaternionf reference_inverse = conjugate_quaternion(normalized(reference.orientation));
    for (const XrPosef& sample : samples)
    {
        const XrVector3f delta{ sample.position.x - reference.position.x, sample.position.y - reference.position.y,
                                sample.position.z - reference.position.z };
        drift.position_m = std::max(drift.position_m, length_of(delta));

        // Angle of the relative rotation, from its vector part rather than acos(): accurate for the
        // small excursions a held controller actually makes.
        const XrQuaternionf relative = multiply_quaternions(reference_inverse, normalized(sample.orientation));
        const float sin_half = std::sqrt(relative.x * relative.x + relative.y * relative.y + relative.z * relative.z);
        const float angle = 2.0f * std::atan2(sin_half, std::fabs(relative.w));
        drift.angle_rad = std::max(drift.angle_rad, angle);
    }
    return drift;
}

bool aim_window_still(const std::vector<XrPosef>& samples, float max_drift_m, float max_drift_rad)
{
    const AimDrift drift = aim_window_drift(samples);
    // The limits are inclusive bounds on "still": a window that reached 5 mm or 2 deg has moved,
    // which also rejects a sample window whose drift lands exactly on the bound.
    return drift.position_m < max_drift_m && drift.angle_rad < max_drift_rad;
}

std::string format_aim_to_wrist(const XrPosef& aim_to_wrist)
{
    // Position in metres to 0.1 mm, quaternion to 1e-6: enough to reconstruct the wrist well inside
    // the calibration's own tolerances, while staying readable in the config file.
    char text[160];
    std::snprintf(
        text, sizeof(text), "%.4f %.4f %.4f %.6f %.6f %.6f %.6f", cleaned_for_display(aim_to_wrist.position.x, 5e-5f),
        cleaned_for_display(aim_to_wrist.position.y, 5e-5f), cleaned_for_display(aim_to_wrist.position.z, 5e-5f),
        cleaned_for_display(aim_to_wrist.orientation.x, 5e-7f), cleaned_for_display(aim_to_wrist.orientation.y, 5e-7f),
        cleaned_for_display(aim_to_wrist.orientation.z, 5e-7f), cleaned_for_display(aim_to_wrist.orientation.w, 5e-7f));
    return std::string(text);
}

} // namespace io_exo_glove_ros2
} // namespace plugins
