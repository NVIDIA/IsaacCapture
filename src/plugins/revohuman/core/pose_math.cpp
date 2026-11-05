// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/pose_math.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace revohuman
{

Quat quat_mul(const Quat& a, const Quat& b)
{
    return { a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
             a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
             a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] };
}

Quat quat_conj(const Quat& q)
{
    return { -q[0], -q[1], -q[2], q[3] };
}

Quat quat_normalize(const Quat& q)
{
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    return { q[0] / n, q[1] / n, q[2] / n, q[3] / n };
}

Vec3 quat_rotate(const Quat& q, const Vec3& v)
{
    // v' = v + 2w (u x v) + 2 u x (u x v), u = (x, y, z).
    const Vec3 u{ q[0], q[1], q[2] };
    const Vec3 t{ 2.0 * (u[1] * v[2] - u[2] * v[1]), 2.0 * (u[2] * v[0] - u[0] * v[2]),
                  2.0 * (u[0] * v[1] - u[1] * v[0]) };
    return { v[0] + q[3] * t[0] + (u[1] * t[2] - u[2] * t[1]), v[1] + q[3] * t[1] + (u[2] * t[0] - u[0] * t[2]),
             v[2] + q[3] * t[2] + (u[0] * t[1] - u[1] * t[0]) };
}

Mat3 quat_to_mat(const Quat& q)
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return { { { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
               { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
               { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) } } };
}

Quat quat_slerp(const Quat& a, const Quat& b_in, double alpha)
{
    Quat b = b_in;
    double c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (c < 0.0)
    {
        b = { -b[0], -b[1], -b[2], -b[3] };
        c = -c;
    }
    double wa = 1.0 - alpha;
    double wb = alpha;
    if (c < 0.9995)
    {
        const double theta = std::acos(c);
        const double s = std::sin(theta);
        wa = std::sin((1.0 - alpha) * theta) / s;
        wb = std::sin(alpha * theta) / s;
    }
    return quat_normalize({ wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3] });
}

double dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const Vec3& v)
{
    return std::sqrt(dot(v, v));
}

Vec3 normalize(const Vec3& v)
{
    const double n = norm(v);
    return { v[0] / n, v[1] / n, v[2] / n };
}

Pose compose(const Pose& a, const Pose& b)
{
    const Vec3 r = quat_rotate(a.q, b.p);
    return Pose{ { a.p[0] + r[0], a.p[1] + r[1], a.p[2] + r[2] }, quat_normalize(quat_mul(a.q, b.q)) };
}

Pose inverse(const Pose& a)
{
    const Quat qi = quat_conj(a.q);
    const Vec3 r = quat_rotate(qi, a.p);
    return Pose{ { -r[0], -r[1], -r[2] }, qi };
}

Vec3 transform_point(const Pose& a, const Vec3& v)
{
    const Vec3 r = quat_rotate(a.q, v);
    return { a.p[0] + r[0], a.p[1] + r[1], a.p[2] + r[2] };
}

Pose interpolate(const Pose& a, const Pose& b, double alpha)
{
    Pose out;
    for (int i = 0; i < 3; ++i)
    {
        out.p[i] = a.p[i] + alpha * (b.p[i] - a.p[i]);
    }
    out.q = quat_slerp(a.q, b.q, alpha);
    return out;
}

Pose parse_pose(std::string_view text, std::string_view flag)
{
    const auto fail = [&](const std::string& why)
    {
        return std::invalid_argument(std::string(flag) + " needs px,py,pz,qx,qy,qz,qw (" + why + "), got '" +
                                     std::string(text) + "'");
    };
    std::array<double, 7> values{};
    std::size_t count = 0;
    std::size_t begin = 0;
    while (true)
    {
        const std::size_t end = std::min(text.find(',', begin), text.size());
        if (count == values.size())
        {
            throw fail("more than 7 numbers");
        }
        const std::string field(text.substr(begin, end - begin));
        std::size_t used = 0;
        double value = 0.0;
        // std::stod skips leading blanks; a field must be the number alone.
        if (!field.empty() && !std::isspace(static_cast<unsigned char>(field.front())))
        {
            try
            {
                value = std::stod(field, &used);
            }
            catch (const std::exception&)
            {
                used = 0;
            }
        }
        if (used == 0 || used != field.size())
        {
            throw fail("'" + field + "' is not a number");
        }
        if (!std::isfinite(value))
        {
            throw fail("values must be finite");
        }
        values[count++] = value;
        if (end == text.size())
        {
            break;
        }
        begin = end + 1;
    }
    if (count != values.size())
    {
        throw fail(std::to_string(count) + " numbers");
    }
    Pose pose;
    pose.p = { values[0], values[1], values[2] };
    for (double v : pose.p)
        if (std::abs(v) > std::numeric_limits<float>::max())
            throw fail("position exceeds OpenXR float range");
    const Quat q = { values[3], values[4], values[5], values[6] };
    // Scale before squaring: finite CLI values can still overflow the norm.
    const double scale = std::max({ std::abs(q[0]), std::abs(q[1]), std::abs(q[2]), std::abs(q[3]) });
    if (scale == 0)
        throw fail("the quaternion is zero");
    Quat scaled{ q[0] / scale, q[1] / scale, q[2] / scale, q[3] / scale };
    const double n =
        std::sqrt(scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2] + scaled[3] * scaled[3]);
    if (scale <= 1e-6 / n)
        throw fail("the quaternion is zero");
    pose.q = { scaled[0] / n, scaled[1] / n, scaled[2] / n, scaled[3] / n };
    return pose;
}

std::array<Pose, kHandJointCount> joints_in_stage(const Pose& wrist_stage, const HandKinematics& kinematics)
{
    std::array<Pose, kHandJointCount> out{};
    for (std::size_t i = 0; i < kHandJointCount; ++i)
    {
        out[i] = compose(wrist_stage, kinematics.joints[i]);
    }
    return out;
}

} // namespace revohuman
} // namespace plugins
