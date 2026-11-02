// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_shapes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/pose_math.hpp>

#include <cmath>
#include <stdexcept>

using namespace plugins::revohuman;
using Catch::Approx;

namespace
{

double det(const Mat3& m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

Vec3 mat_mul(const Mat3& m, const Vec3& v)
{
    return { m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
             m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] };
}

void require_vec(const Vec3& a, const Vec3& b, double eps = 1e-12)
{
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(a[i] == Approx(b[i]).margin(eps));
    }
}

//! Same rotation: q and -q are equal.
void require_same_rotation(const Quat& a, const Quat& b, double eps = 1e-12)
{
    const double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    REQUIRE(std::abs(d) == Approx(1.0).margin(eps));
}

Pose sample_pose()
{
    Pose p;
    p.p = { 0.3, -1.2, 0.7 };
    p.q = revohuman_test::axis_angle({ 0.2, -0.5, 1.0 }, 1.1);
    return p;
}

} // namespace

TEST_CASE("quat_to_mat is a proper rotation that agrees with quat_rotate", "[revohuman][pose]")
{
    const Pose p = sample_pose();
    const Mat3 m = quat_to_mat(p.q);
    REQUIRE(det(m) == Approx(1.0));
    const Vec3 v{ 0.4, 0.1, -0.9 };
    require_vec(mat_mul(m, v), quat_rotate(p.q, v));
}

TEST_CASE("compose and inverse are consistent", "[revohuman][pose]")
{
    const Pose a = sample_pose();
    Pose b;
    b.p = { -0.05, 0.02, 0.11 };
    b.q = revohuman_test::axis_angle({ 1.0, 0.0, 0.3 }, -0.7);

    const Pose id = compose(a, inverse(a));
    require_vec(id.p, { 0, 0, 0 });
    require_same_rotation(id.q, { 0, 0, 0, 1 });

    const Vec3 v{ 0.01, -0.2, 0.3 };
    require_vec(transform_point(compose(a, b), v), transform_point(a, transform_point(b, v)));
}

TEST_CASE("slerp hits its endpoints and takes the short arc", "[revohuman][pose]")
{
    const Quat a = revohuman_test::axis_angle({ 0, 0, 1 }, 0.2);
    const Quat b = revohuman_test::axis_angle({ 0, 0, 1 }, 1.0);
    require_same_rotation(quat_slerp(a, b, 0.0), a);
    require_same_rotation(quat_slerp(a, b, 1.0), b);
    require_same_rotation(quat_slerp(a, b, 0.5), revohuman_test::axis_angle({ 0, 0, 1 }, 0.6));
    const Quat neg_b{ -b[0], -b[1], -b[2], -b[3] };
    require_same_rotation(quat_slerp(a, neg_b, 0.5), revohuman_test::axis_angle({ 0, 0, 1 }, 0.6));
}

TEST_CASE("joints in the stage chain the wrist pose and the kinematics", "[revohuman][pose]")
{
    const Pose wrist = sample_pose();

    HandKinematics kin;
    for (std::uint32_t j = 0; j < kHandJointCount; ++j)
    {
        kin.joints[j] = revohuman_test::bind_pose(j);
    }
    const auto joints = joints_in_stage(wrist, kin);
    for (std::uint32_t j = 0; j < kHandJointCount; ++j)
    {
        const Vec3 local{ 0.01, 0.0, 0.0 };
        require_vec(transform_point(joints[j], local), transform_point(wrist, transform_point(kin.joints[j], local)));
    }
}

TEST_CASE("a pose flag is seven numbers with a non-zero quaternion", "[revohuman][pose]")
{
    const Pose pose = parse_pose("0.01,-0.1,0.08,0,0,0,2", "--aim-to-wrist-left");
    REQUIRE(pose.p[0] == Approx(0.01));
    REQUIRE(pose.p[1] == Approx(-0.1));
    REQUIRE(pose.p[2] == Approx(0.08));
    // Normalized.
    REQUIRE(pose.q[3] == Approx(1.0));

    for (const char* bad : { "", "0,0,0,0,0,0", "0,0,0,0,0,0,1,0", "0,0,0,0,0,0,x", "0,0,0,0,0,,1", " 0,0,0,0,0,0,1",
                             "0,0,0,0,0,0,1 ", "nan,0,0,0,0,0,1", "0,0,inf,0,0,0,1", "0,0,0,0,0,0,0" })
    {
        INFO("'" << bad << "'");
        REQUIRE_THROWS_AS(parse_pose(bad, "--aim-to-wrist-left"), std::invalid_argument);
    }
}

TEST_CASE("wrist offset flags remain finite in OpenXR precision", "[revohuman][pose]")
{
    const Pose pose = parse_pose("0,0,0,1e308,1e308,1e308,1e308", "--aim-to-wrist-left");
    for (double v : pose.q)
        REQUIRE(v == Approx(0.5));
    REQUIRE_THROWS_AS(parse_pose("1e100,0,0,0,0,0,1", "--aim-to-wrist-left"), std::invalid_argument);
}
