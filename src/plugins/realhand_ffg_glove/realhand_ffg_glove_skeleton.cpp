// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_skeleton.hpp"

#include <oxr_utils/math.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace plugins::realhand_ffg_glove
{
namespace
{
constexpr auto kValid = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
constexpr auto kTracked = XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
constexpr float kPi = std::numbers::pi_v<float>;

template <std::size_t N>
std::array<float, N> read_values(const YAML::Node& node, const std::string& label)
{
    if (!node.IsSequence() || node.size() != N)
    {
        throw std::invalid_argument(label + " must contain " + std::to_string(N) + " finite values");
    }
    std::array<float, N> angles{};
    for (std::size_t i = 0; i < angles.size(); ++i)
    {
        angles[i] = node[i].as<float>();
        if (!std::isfinite(angles[i]))
        {
            throw std::invalid_argument(label + " contains a non-finite angle");
        }
    }
    return angles;
}

std::array<float, kNumSensors> read_angles(const YAML::Node& node, const std::string& label)
{
    return read_values<kNumSensors>(node, label);
}

ThumbSensorMapping read_thumb_mapping(const YAML::Node& node)
{
    if (!node.IsMap() || node["format"].as<std::string>("") != "realhand-ffg-thumb-quadratic-v1")
    {
        throw std::invalid_argument("Invalid thumb sensor mapping format");
    }
    ThumbSensorMapping mapping{ read_values<4>(node["input_origin"], "thumb origin"),
                                read_values<4>(node["input_scale"], "thumb scale"),
                                read_values<4>(node["input_min"], "thumb minimum"),
                                read_values<4>(node["input_max"], "thumb maximum"),
                                {},
                                read_values<3>(node["base_position"], "thumb base"),
                                read_values<3>(node["lengths"], "thumb lengths") };
    const auto coefficients = node["coefficients"];
    if (!coefficients.IsSequence() || coefficients.size() != 4)
    {
        throw std::invalid_argument("Thumb mapping must have four coefficient rows");
    }
    bool has_motion = false;
    for (std::size_t i = 0; i < 4; ++i)
    {
        mapping.coefficients[i] = read_values<15>(coefficients[i], "thumb coefficients");
        if (mapping.scale[i] < 0.10f - 1e-6f || mapping.scale[i] > kPi || mapping.minimum[i] > 0.0f ||
            mapping.maximum[i] < 0.0f || mapping.maximum[i] - mapping.minimum[i] >= kPi ||
            std::max(std::abs(mapping.minimum[i]), std::abs(mapping.maximum[i])) > mapping.scale[i] + 1e-5f)
        {
            throw std::invalid_argument("Thumb mapping has invalid input ranges");
        }
        has_motion = has_motion || mapping.maximum[i] - mapping.minimum[i] >= 0.05f;
        for (float coefficient : mapping.coefficients[i])
        {
            if (std::abs(coefficient) > 1.76f)
            {
                throw std::invalid_argument("Thumb coefficient exceeds the bounded fit range");
            }
        }
    }
    if (!has_motion)
    {
        throw std::invalid_argument("Thumb mapping has insufficient sensor motion");
    }
    constexpr float base[3] = { 0.025f, -0.005f, -0.020f };
    constexpr float lengths[3] = { 0.035f, 0.030f, 0.026f };
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (std::abs(mapping.base[i] - base[i]) > 0.0081f || mapping.lengths[i] < lengths[i] * 0.849f ||
            mapping.lengths[i] > lengths[i] * 1.151f)
        {
            throw std::invalid_argument("Thumb geometry exceeds the bounded fit range");
        }
    }
    return mapping;
}

float angle_delta(float value, float origin)
{
    return std::remainder(std::remainder(value, 2.0f * kPi) - std::remainder(origin, 2.0f * kPi), 2.0f * kPi);
}

float response(float value, float origin, float endpoint)
{
    const float span = angle_delta(endpoint, origin);
    if (std::abs(span) < 0.05f)
    {
        return 0.0f;
    }
    // Unwrap around the middle of the calibrated arc, not the open boundary.
    // Otherwise continuing past a >pi/2 span can wrap back to "open".
    const float center = span * 0.5f;
    const float delta = center + std::remainder(angle_delta(value, origin) - center, 2.0f * kPi);
    return std::clamp(delta / span, 0.0f, 1.0f);
}

std::array<double, 2> regularized_thumb_flexion(double mcp, double ip, const std::array<float, 3>& lengths)
{
    if (mcp == 0.0)
    {
        return { mcp, ip };
    }
    const auto deficit = [&](double c, double d)
    {
        const double a = std::sin(c * 0.5), b = std::sin(d * 0.5), e = std::sin((c + d) * 0.5);
        return 4.0 * (lengths[0] * double(lengths[1]) * a * a + lengths[1] * double(lengths[2]) * b * b +
                      lengths[0] * double(lengths[2]) * e * e);
    };
    // Tip contact alone leaves MCP/IP posture ambiguous. This soft posture prior is not a measured joint limit.
    // Preserve reach and independent digits rather than concentrating the bend in MCP with a straight IP.
    const double target = deficit(mcp, ip);
    double selected = mcp / std::pow(1.0 + std::pow(mcp / 1.1, 8.0), 0.125);
    constexpr double max_ip = 1.8;
    if (deficit(selected, max_ip) < target)
    {
        double low = selected, high = mcp;
        for (int i = 0; i < 40; ++i)
        {
            const double mid = (low + high) * 0.5;
            if (deficit(mid, max_ip) < target)
                low = mid;
            else
                high = mid;
        }
        selected = high;
    }
    double low = ip, high = max_ip;
    for (int i = 0; i < 40; ++i)
    {
        const double mid = (low + high) * 0.5;
        if (deficit(selected, mid) < target)
            low = mid;
        else
            high = mid;
    }
    return { selected, (low + high) * 0.5 };
}

double thumb_chord_angle(double mcp, double ip, const std::array<float, 3>& lengths)
{
    return std::atan2(lengths[1] * std::sin(mcp) + lengths[2] * std::sin(mcp + ip),
                      lengths[0] + lengths[1] * std::cos(mcp) + lengths[2] * std::cos(mcp + ip));
}

XrPosef rotate(float angle, int axis)
{
    XrPosef pose{ { 0.0f, 0.0f, 0.0f, std::cos(angle * 0.5f) }, {} };
    if (axis == 0)
    {
        pose.orientation.x = std::sin(angle * 0.5f);
    }
    else if (axis == 1)
    {
        pose.orientation.y = std::sin(angle * 0.5f);
    }
    else
    {
        pose.orientation.z = std::sin(angle * 0.5f);
    }
    return pose;
}

XrPosef advance(const XrPosef& pose, float length)
{
    return oxr_utils::multiply_poses(pose, { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -length } });
}

void set_joint(HandJoints& joints, std::size_t index, const XrPosef& pose, float radius)
{
    joints[index] = { kValid, pose, radius };
}
} // namespace

HumanHandSkeleton::HumanHandSkeleton(const std::string& calibration_path, float hand_scale) : hand_scale_(hand_scale)
{
    if (!std::isfinite(hand_scale_) || hand_scale_ < 0.5f || hand_scale_ > 2.0f)
    {
        throw std::invalid_argument("hand scale must be finite and between 0.5 and 2.0");
    }
    if (calibration_path.empty())
    {
        return;
    }
    const auto data = YAML::LoadFile(calibration_path);
    if (!data.IsMap() || data["format"].as<std::string>("") != "realhand-ffg-glove-human-calibration-v1")
    {
        throw std::invalid_argument("Expected device-side human calibration; run realhand_ffg_glove_calibration.py");
    }
    const auto hands = data["hands"];
    if (!hands.IsMap())
    {
        throw std::invalid_argument("human calibration must contain a hands mapping");
    }
    for (std::size_t side = 0; side < 2; ++side)
    {
        const std::string name = side == 0 ? "left" : "right";
        const auto node = hands[name];
        if (!node)
        {
            continue;
        }
        calibrations_[side] =
            HumanHandCalibration{ read_angles(node["open"], name + ".open"), read_angles(node["fist"], name + ".fist"),
                                  read_angles(node["thumb_curl"], name + ".thumb_curl"),
                                  read_angles(node["thumb_opposition"], name + ".thumb_opposition"),
                                  read_angles(node["spread"], name + ".spread") };
        if (node["thumb_mapping"])
        {
            calibrations_[side]->thumb_mapping = read_thumb_mapping(node["thumb_mapping"]);
        }
        const auto& calibration = *calibrations_[side];
        const auto require_span = [&](std::size_t sensor, const auto& endpoint)
        {
            if (std::abs(angle_delta(endpoint[sensor], calibration.open[sensor])) < 0.05f)
            {
                throw std::invalid_argument(name + " sensor_" + std::to_string(sensor) +
                                            " has insufficient calibration motion; capture the poses again");
            }
        };
        // A multivariate fit uses the full captured ranges, not one legacy pose/channel pair.
        if (!calibration.thumb_mapping)
        {
            require_span(0, calibration.thumb_opposition);
            require_span(2, calibration.thumb_curl);
            require_span(4, calibration.thumb_curl);
        }
        for (std::size_t finger = 0; finger < 4; ++finger)
        {
            require_span(6 + 4 * finger, calibration.fist);
            require_span(8 + 4 * finger, calibration.fist);
        }
    }
    if (!calibrations_[0] && !calibrations_[1])
    {
        throw std::invalid_argument("human calibration has neither left nor right hand measurements");
    }
}

bool HumanHandSkeleton::calibrated(HandSide side) const
{
    return calibrations_[side == HandSide::Left ? 0 : 1].has_value();
}

std::optional<HandJoints> HumanHandSkeleton::reconstruct(const std::array<float, kNumSensors>& sensors, HandSide side) const
{
    const auto& calibration = calibrations_[side == HandSide::Left ? 0 : 1];
    if (!calibration || !std::all_of(sensors.begin(), sensors.end(), [](float x) { return std::isfinite(x); }))
    {
        return std::nullopt;
    }
    const auto& c = *calibration;
    const float mirror = side == HandSide::Left ? 1.0f : -1.0f;
    const float scale = hand_scale_;
    HandJoints joints{};
    const XrPosef identity{ { 0.0f, 0.0f, 0.0f, 1.0f }, {} };
    set_joint(joints, XR_HAND_JOINT_WRIST_EXT, identity, 0.020f * scale);
    set_joint(joints, XR_HAND_JOINT_PALM_EXT, { identity.orientation, { 0.0f, 0.0f, -0.045f * scale } }, 0.030f * scale);

    // SDK l20_config.py: 5+4*f is spread, 6+4*f MCP flexion, 8+4*f distal flexion.
    // It does not independently resolve PIP; infer it from DIP (1.5:1).
    constexpr float lateral[4] = { 0.033f, 0.011f, -0.011f, -0.031f };
    constexpr float knuckle[4] = { 0.070f, 0.075f, 0.070f, 0.060f };
    constexpr float lengths[4][3] = {
        { 0.045f, 0.026f, 0.018f }, { 0.050f, 0.030f, 0.019f }, { 0.046f, 0.027f, 0.018f }, { 0.034f, 0.020f, 0.016f }
    };
    constexpr float spread_limits[4] = { 0.25f, 0.08f, 0.12f, 0.30f };
    for (std::size_t finger = 0; finger < 4; ++finger)
    {
        const std::size_t sensor = 5 + 4 * finger;
        const std::size_t joint = XR_HAND_JOINT_INDEX_METACARPAL_EXT + 5 * finger;
        const float spread_span = angle_delta(c.spread[sensor], c.open[sensor]);
        // A contact can reach either lateral extreme: use its magnitude, never its sign.
        // Raw lateral deltas are mirrored between gloves, separately from skeleton anatomy.
        // Normalize them before applying the anatomical mirror; thumb/flexion channels are unchanged.
        const float spread =
            std::clamp(mirror * angle_delta(sensors[sensor], c.open[sensor]) / std::max(std::abs(spread_span), 0.10f),
                       -1.0f, 1.0f) *
            spread_limits[finger];
        const float base_flex = response(sensors[sensor + 1], c.open[sensor + 1], c.fist[sensor + 1]) * 1.5f;
        const float distal_flex = response(sensors[sensor + 3], c.open[sensor + 3], c.fist[sensor + 3]) * 1.1f;
        XrPosef pose{ identity.orientation, { mirror * lateral[finger] * scale, 0.0f, -0.015f * scale } };
        pose = oxr_utils::multiply_poses(pose, rotate(mirror * spread, 1));
        pose = advance(pose, (knuckle[finger] - 0.015f) * scale);
        // Fan the neutral metacarpals toward the wrist; preserve knuckle poses and bone lengths.
        const float fan_angle = std::atan2(-mirror * lateral[finger], knuckle[finger]);
        auto metacarpal = oxr_utils::multiply_poses(pose, rotate(fan_angle, 1));
        metacarpal = advance(metacarpal, -(knuckle[finger] - 0.015f) * scale);
        set_joint(joints, joint, metacarpal, 0.010f * scale);
        pose = oxr_utils::multiply_poses(pose, rotate(-base_flex, 0));
        set_joint(joints, joint + 1, pose, 0.009f * scale);
        pose = advance(pose, lengths[finger][0] * scale);
        pose = oxr_utils::multiply_poses(pose, rotate(-1.5f * distal_flex, 0));
        set_joint(joints, joint + 2, pose, 0.008f * scale);
        pose = advance(pose, lengths[finger][1] * scale);
        pose = oxr_utils::multiply_poses(pose, rotate(-distal_flex, 0));
        set_joint(joints, joint + 3, pose, 0.007f * scale);
        pose = advance(pose, lengths[finger][2] * scale);
        set_joint(joints, joint + 4, pose, 0.006f * scale);
    }

    // Thumb abduction/roll/MCP/IP use SDK channels 0/1/2/4. No other digit is consulted.
    const float opposition = response(sensors[0], c.open[0], c.thumb_opposition[0]);
    const float roll = response(sensors[1], c.open[1], c.thumb_opposition[1]);
    std::array<float, 4> angles{ -0.85f + 1.20f * opposition, -0.70f * roll,
                                 response(sensors[2], c.open[2], c.thumb_curl[2]) * 1.1f,
                                 response(sensors[4], c.open[4], c.thumb_curl[4]) * 1.2f };
    std::array<float, 3> base{ 0.025f, -0.005f, -0.020f }, thumb_lengths{ 0.035f, 0.030f, 0.026f };
    if (c.thumb_mapping)
    {
        const auto& mapping = *c.thumb_mapping;
        constexpr std::size_t channels[4] = { 0, 1, 2, 4 };
        std::array<float, 15> features{ 1.0f };
        for (std::size_t i = 0; i < 4; ++i)
        {
            const float center = (mapping.minimum[i] + mapping.maximum[i]) * 0.5f;
            const float delta =
                center + std::remainder(angle_delta(sensors[channels[i]], mapping.origin[i]) - center, 2.0f * kPi);
            features[i + 1] = std::clamp(delta, mapping.minimum[i], mapping.maximum[i]) / mapping.scale[i];
        }
        std::size_t index = 5;
        for (std::size_t first = 1; first <= 4; ++first)
        {
            for (std::size_t second = first; second <= 4; ++second)
            {
                features[index++] = features[first] * features[second];
            }
        }
        // One continuous thumb-only polynomial: no pose lookup, live contact target, or other-digit input.
        constexpr float lower[4] = { -1.25f, -1.2f, 0.0f, 0.0f }, upper[4] = { 1.0f, 1.2f, 1.8f, 1.8f };
        for (std::size_t axis = 0; axis < 4; ++axis)
        {
            float angle = 0.0f;
            for (std::size_t feature = 0; feature < features.size(); ++feature)
            {
                angle += mapping.coefficients[axis][feature] * features[feature];
            }
            angles[axis] = std::clamp(angle, lower[axis], upper[axis]);
        }
        base = mapping.base;
        thumb_lengths = mapping.lengths;
    }
    const float y = -thumb_lengths[1] * std::sin(angles[2]) - thumb_lengths[2] * std::sin(angles[2] + angles[3]);
    const float z =
        -thumb_lengths[0] - thumb_lengths[1] * std::cos(angles[2]) - thumb_lengths[2] * std::cos(angles[2] + angles[3]);
    const auto direction = oxr_utils::multiply_poses(
                               oxr_utils::multiply_poses(rotate(mirror * angles[0], 1), rotate(mirror * angles[1], 2)),
                               { identity.orientation, { 0.0f, y, z } })
                               .position;
    const auto flexion = regularized_thumb_flexion(angles[2], angles[3], thumb_lengths);
    const float elevation = std::atan2(direction.y, -direction.z);
    const float azimuth = std::atan2(-mirror * direction.x, std::hypot(direction.y, direction.z));
    const float chord = static_cast<float>(thumb_chord_angle(flexion[0], flexion[1], thumb_lengths));
    // Construct the chain in its chord/radial plane; do not flip it and hide the bend with root compensation.
    // This is a thumb-only bone-posture prior, not a measured extra DOF or an anatomical joint limit.
    XrPosef thumb{ identity.orientation, { mirror * base[0] * scale, base[1] * scale, base[2] * scale } };
    thumb = oxr_utils::multiply_poses(thumb, rotate(elevation, 0));
    thumb = oxr_utils::multiply_poses(thumb, rotate(mirror * kPi * 0.5f, 2));
    thumb = oxr_utils::multiply_poses(thumb, rotate(azimuth - chord, 0));
    set_joint(joints, XR_HAND_JOINT_THUMB_METACARPAL_EXT, thumb, 0.012f * scale);
    thumb = advance(thumb, thumb_lengths[0] * scale);
    thumb = oxr_utils::multiply_poses(thumb, rotate(static_cast<float>(flexion[0]), 0));
    set_joint(joints, XR_HAND_JOINT_THUMB_PROXIMAL_EXT, thumb, 0.010f * scale);
    thumb = advance(thumb, thumb_lengths[1] * scale);
    thumb = oxr_utils::multiply_poses(thumb, rotate(static_cast<float>(flexion[1]), 0));
    set_joint(joints, XR_HAND_JOINT_THUMB_DISTAL_EXT, thumb, 0.009f * scale);
    thumb = advance(thumb, thumb_lengths[2] * scale);
    set_joint(joints, XR_HAND_JOINT_THUMB_TIP_EXT, thumb, 0.007f * scale);
    return joints;
}

bool WristPoseContinuity::accept(const plugin_utils::WristSample& wrist, std::int64_t time_ns)
{
    const auto& pose = wrist.pose;
    const auto& p = pose.position;
    const auto& q = pose.orientation;
    const float norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    // VALID may describe a cached pose; controller publication needs live tracking.
    if (!wrist.valid || !wrist.tracked || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(norm) || std::abs(norm - 1.0f) > 0.01f)
    {
        previous_.reset();
        stable_samples_ = 0;
        recovering_ = true;
        return false;
    }
    if (!previous_)
    {
        previous_ = pose;
        previous_time_ns_ = time_ns;
        return !recovering_;
    }
    const auto before = *previous_;
    const double dt = (static_cast<double>(time_ns) - previous_time_ns_) / 1e9;
    previous_ = pose;
    previous_time_ns_ = time_ns;
    const float x = p.x - before.position.x, y = p.y - before.position.y, z = p.z - before.position.z;
    const float distance = std::sqrt(x * x + y * y + z * z);
    const auto& b = before.orientation;
    const float dot = std::abs(q.x * b.x + q.y * b.y + q.z * b.z + q.w * b.w);
    const float angle = 2.0f * std::acos(std::clamp(dot, 0.0f, 1.0f));
    const bool discontinuity =
        dt <= 0.0 || dt > 0.2 ||
        (dt < 0.05 && ((distance > 0.10f && distance / dt > 8.0) || (angle > 1.0f && angle / dt > 20.0)));
    if (discontinuity)
    {
        recovering_ = true;
        stable_samples_ = 0;
        return false;
    }
    if (recovering_)
    {
        stable_samples_ = distance / dt < 3.0 && angle / dt < 10.0 ? stable_samples_ + 1 : 0;
        if (stable_samples_ < 3)
        {
            return false;
        }
        recovering_ = false;
    }
    // Never smooth, clamp, or relabel a discontinuity as a tracked pose.
    return true;
}

HandJoints place_hand(HandJoints joints, const XrPosef& wrist, bool wrist_valid, bool wrist_tracked)
{
    for (auto& joint : joints)
    {
        joint.locationFlags &= ~kTracked;
        if (wrist_valid)
        {
            joint.pose = oxr_utils::multiply_poses(wrist, joint.pose);
            if (wrist_tracked)
            {
                joint.locationFlags |= kTracked;
            }
        }
    }
    return joints;
}
} // namespace plugins::realhand_ffg_glove
