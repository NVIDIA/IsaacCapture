// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <revohuman_plugin/pose_math.hpp>
#include <revohuman_plugin/tactile.hpp>

#include <cmath>
#include <cstdint>

namespace revohuman_test
{

using plugins::revohuman::kHandJointCount;
using plugins::revohuman::Pose;
using plugins::revohuman::ShapeSpec;

//! A rotation of angle_rad about axis.
inline std::array<double, 4> axis_angle(std::array<double, 3> axis, double angle_rad)
{
    const auto n = plugins::revohuman::normalize(axis);
    const double s = std::sin(angle_rad / 2.0);
    return { n[0] * s, n[1] * s, n[2] * s, std::cos(angle_rad / 2.0) };
}

inline Pose bind_pose(std::uint32_t joint)
{
    Pose p;
    p.p = { 0.004 * joint, 0.011 * (joint % 5), -0.002 * joint };
    p.q = axis_angle({ 1.0, 2.0, 3.0 }, 0.05 * joint);
    return p;
}

//! A valid rigid shape: taxel i sits on joint 2 + i % 24, weight 1 on that joint.
inline ShapeSpec rigid_shape(std::size_t taxels)
{
    ShapeSpec spec;
    for (std::uint32_t j = 0; j < kHandJointCount; ++j)
    {
        spec.bind_poses.push_back(bind_pose(j));
        spec.joint_ids.push_back(static_cast<core::HandJoint>(j));
    }
    for (std::size_t i = 0; i < taxels; ++i)
    {
        const auto joint = static_cast<std::uint32_t>(2 + i % 24);
        const auto& b = spec.bind_poses[joint];
        const double off = 0.001 * static_cast<double>(i % 7);
        Pose taxel = b;
        taxel.p = { b.p[0] + off, b.p[1] - off, b.p[2] + 0.005 };
        spec.taxel_bind_poses.push_back(taxel);
        spec.joint_indices.push_back({ joint, joint, joint, joint });
        spec.joint_weights.push_back({ 1.0f, 0.0f, 0.0f, 0.0f });
    }
    return spec;
}

} // namespace revohuman_test
