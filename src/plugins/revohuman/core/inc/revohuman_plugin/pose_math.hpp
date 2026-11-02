// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <string_view>

namespace plugins
{
namespace revohuman
{

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>; // x, y, z, w
using Mat3 = std::array<std::array<double, 3>, 3>;

Quat quat_mul(const Quat& a, const Quat& b);
Quat quat_conj(const Quat& q);
Quat quat_normalize(const Quat& q);
Vec3 quat_rotate(const Quat& q, const Vec3& v);
Mat3 quat_to_mat(const Quat& q);
//! Shortest-arc spherical interpolation, alpha in [0, 1].
Quat quat_slerp(const Quat& a, const Quat& b, double alpha);

double dot(const Vec3& a, const Vec3& b);
double norm(const Vec3& v);
Vec3 normalize(const Vec3& v);

//! a * b: b expressed in a's frame, mapped to a's parent.
Pose compose(const Pose& a, const Pose& b);
Pose inverse(const Pose& a);
Vec3 transform_point(const Pose& a, const Vec3& v);
Pose interpolate(const Pose& a, const Pose& b, double alpha);

//! A pose written as "px,py,pz,qx,qy,qz,qw" (meters, quaternion x y z w), for command line flags. The
//! quaternion is normalized. Throws std::invalid_argument naming @p flag for anything but seven finite
//! numbers with a non-zero quaternion.
Pose parse_pose(std::string_view text, std::string_view flag);

//! The 26 OpenXR joints in the stage, from the WRIST pose and the WRIST-frame joint poses.
std::array<Pose, kHandJointCount> joints_in_stage(const Pose& wrist_stage, const HandKinematics& kinematics);

} // namespace revohuman
} // namespace plugins
