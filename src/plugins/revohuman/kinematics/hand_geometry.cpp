// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Keep fitting and reviewed geometry in the plugin; the external C SDK is an unmodified dependency.
#include "hand_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace plugins
{
namespace revohuman
{
namespace
{
using V = std::array<double, 3>;
using M = std::array<double, 9>;
constexpr M kIdentity{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
constexpr V kDorsal{ -1, 0, 0 };
struct Height
{
    int driver;
    V inward;
    double amount;
};
struct PointCorrection
{
    Height lateral;
    Height pad;
};
struct Model
{
    M wrist;
    PointCorrection correction[26];
};
#include "openxr_model.inc"

struct FingerMap
{
    int base_angle;
    int anchor_angle;
    int proximal_point;
    int intermediate_point;
    int distal_point;
    int tip_point;
    int placeholder_point;
    int orientation_angle;
};
// SDK angle slots and OpenXR output slots are independent index spaces. A non-thumb proximal point is
// oriented by its MCP link (base + 2), which carries it; the thumb's keeps its reviewed link.
constexpr std::array<FingerMap, 5> kFingerMap = { {
    { 0, 3, 7, 8, 9, 10, 6, 2 },
    { 4, 7, 12, 13, 14, 15, 11, 6 },
    { 8, 11, 17, 18, 19, 20, 16, 10 },
    { 12, 15, 22, 23, 24, 25, 21, 14 },
    { 16, 20, 2, 3, 4, 5, -1, 19 },
} };

V add(V a, V b)
{
    for (int i = 0; i < 3; ++i)
        a[i] += b[i];
    return a;
}
V scale(V a, double s)
{
    for (double& v : a)
        v *= s;
    return a;
}
V sub(V a, V b)
{
    return add(a, scale(b, -1));
}
double dot(V a, V b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
V cross(V a, V b)
{
    return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}
V unit(V a)
{
    double n = std::sqrt(dot(a, a));
    return scale(a, n > 1e-12 ? 1 / n : std::numeric_limits<double>::quiet_NaN());
}
V mv(M a, V b)
{
    V o{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            o[i] += a[3 * i + j] * b[j];
    return o;
}
M mul(M a, M b)
{
    M o{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return o;
}
M transpose(M a)
{
    return { a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8] };
}
M frame(V z, V y)
{
    z = unit(z);
    y = sub(y, scale(z, dot(y, z)));
    if (dot(y, y) < 1e-14)
        y = cross(z, std::abs(z[0]) < .8 ? V{ 1, 0, 0 } : V{ 0, 1, 0 });
    y = unit(y);
    V x = unit(cross(y, z));
    y = unit(cross(z, x));
    return { x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2] };
}
std::array<double, 4> quaternion(M a)
{
    std::array<double, 4> q{};
    double trace = a[0] + a[4] + a[8];
    if (trace > 0)
    {
        double s = 2 * std::sqrt(trace + 1);
        q = { (a[7] - a[5]) / s, (a[2] - a[6]) / s, (a[3] - a[1]) / s, s / 4 };
    }
    else
    {
        int i = a[4] > a[0] ? 1 : 0;
        if (a[8] > a[3 * i + i])
            i = 2;
        int j = (i + 1) % 3, k = (i + 2) % 3;
        double s = 2 * std::sqrt(1 + a[3 * i + i] - a[3 * j + j] - a[3 * k + k]);
        q[i] = s / 4;
        q[j] = (a[3 * i + j] + a[3 * j + i]) / s;
        q[k] = (a[3 * i + k] + a[3 * k + i]) / s;
        q[3] = (a[3 * k + j] - a[3 * j + k]) / s;
    }
    double n = 0;
    for (double v : q)
        n += v * v;
    n = std::sqrt(n);
    for (double& v : q)
        v /= n;
    return q;
}
const Model& model(RhHand hand)
{
    if (hand == RH_HAND_LEFT)
        return kModel0;
    if (hand == RH_HAND_RIGHT)
        return kModel1;
    throw std::invalid_argument("hand must be left or right");
}
bool finite(V p)
{
    return std::all_of(p.begin(), p.end(), [](double x) { return std::isfinite(x); });
}
void write_point(HandPoint& dst, V p, M r, bool mapped)
{
    auto q = quaternion(r);
    dst = {};
    dst.radius = .01f;
    dst.orientation[3] = 1;
    bool pf = finite(p), qf = std::all_of(q.begin(), q.end(), [](double x) { return std::isfinite(x); });
    if (pf)
        std::copy(p.begin(), p.end(), dst.position);
    if (qf)
        std::copy(q.begin(), q.end(), dst.orientation);
    dst.flags = mapped ? ((pf ? kHandPositionValid : 0u) | (qf ? kHandOrientationValid : 0u)) : 0u;
}
// Palm-relative links, TIPs and tracker origin to OpenXR points, with the reviewed corrections.
void assemble(RhHand hand,
              const Model& m,
              const std::array<V, 21>& p,
              const std::array<M, 21>& r,
              const std::array<V, 5>& tips,
              const std::array<V, 5>& tip_directions,
              V tracker,
              HandPoint out[26])
{
    std::array<V, 26> points{};
    std::array<M, 26> frames{};
    points[0] = tracker;
    frames[0] = frames[1] = m.wrist;
    for (int f = 0; f < 5; ++f)
    {
        const auto& mapping = kFingerMap[f];
        const int b = mapping.base_angle;
        if (mapping.placeholder_point >= 0)
        {
            // Placeholder offsets are along the OpenXR wrist axes from the palm_link origin, not from the shifted
            // WRIST; mirror the lateral X axis.
            V placeholder{ kPlaceholder[f][0], kPlaceholder[f][1], kPlaceholder[f][2] };
            if (hand == RH_HAND_RIGHT)
                placeholder[0] = -placeholder[0];
            points[mapping.placeholder_point] = mv(m.wrist, placeholder);
        }
        points[mapping.proximal_point] = p[mapping.anchor_angle];
        points[mapping.intermediate_point] = p[b + 1];
        points[mapping.distal_point] = p[b];
        points[mapping.tip_point] = tips[f];
    }
    // Each output slot owns both corrections, including explicit zero values.
    for (int i = 0; i < 26; ++i)
    {
        for (const Height* shift : { &m.correction[i].lateral, &m.correction[i].pad })
        {
            if (shift->amount != 0)
                points[i] = add(points[i], scale(mv(shift->driver < 0 ? kIdentity : r[shift->driver], shift->inward),
                                                 shift->amount));
        }
    }
    // A non-thumb proximal point is placed, pad correction included, in the frame of the MPR link (the anchor),
    // but the finger bends about the MCP origin 19.5 mm further out (22.5 mm from the corrected point). Carry it
    // with the MCP flexion so that it is rigid on the MCP link, like the PIP point it aims at: the bone keeps its
    // length, turns by the encoder angle, and the taxels bound to it stay on the SDK's taxels. MCP and MPR agree at
    // zero flexion, so the bind pose does not move; the knuckle point does, 21.7 mm at 1.2 rad.
    for (const auto& mapping : kFingerMap)
    {
        if (mapping.placeholder_point < 0)
            continue;
        const int mcp = mapping.base_angle + 2;
        const M flexion = mul(r[mcp], transpose(r[mapping.anchor_angle]));
        V& proximal = points[mapping.proximal_point];
        proximal = add(p[mcp], mv(flexion, sub(proximal, p[mcp])));
    }
    for (int f = 0; f < 5; ++f)
    {
        const auto& mapping = kFingerMap[f];
        const int b = mapping.base_angle;
        const int proximal = mapping.proximal_point, intermediate = mapping.intermediate_point,
                  distal = mapping.distal_point, tip = mapping.tip_point;
        if (mapping.placeholder_point >= 0)
            frames[mapping.placeholder_point] = m.wrist;
        frames[proximal] = frame(sub(points[proximal], points[intermediate]), mv(r[mapping.orientation_angle], kDorsal));
        frames[intermediate] = frame(sub(points[intermediate], points[distal]), mv(r[b + 1], kDorsal));
        frames[distal] = frames[tip] = frame(scale(tip_directions[f], -1), mv(r[b], kDorsal));
    }
    HandPoint result[26]{};
    M inv = transpose(m.wrist);
    for (int i = 0; i < 26; ++i)
    {
        bool mapped = i != 6 && i != 11 && i != 16 && i != 21;
        // Every pose, the placeholders included, is relative to WRIST (point 1).
        write_point(result[i], mv(inv, sub(points[i], points[1])), mul(inv, frames[i]), mapped);
    }
    std::copy(std::begin(result), std::end(result), out);
}
} // namespace

const char* hand_points_mapping_id()
{
    return kMappingId;
}
const char* hand_points_reviewed_sha256(const char* model_name)
{
    if (!model_name)
        return nullptr;
    for (const auto& reviewed : kReviewedModels)
    {
        if (std::strcmp(reviewed.name, model_name) == 0)
            return reviewed.sha256;
    }
    return nullptr;
}
// wrist_from_sdk_points() has only palm_link, so the WRIST corrections must not name a driving link.
static_assert(kModel0.correction[1].lateral.driver < 0 && kModel0.correction[1].pad.driver < 0 &&
                  kModel1.correction[1].lateral.driver < 0 && kModel1.correction[1].pad.driver < 0,
              "WRIST corrections must stay fixed in palm_link");
RhStatus wrist_from_sdk_points(RhHand hand, const LinkPose sdk[27], LinkPose* wrist)
{
    if (!sdk || !wrist || (hand != RH_HAND_LEFT && hand != RH_HAND_RIGHT))
        return RH_ERR_INVALID_ARG;
    M palm_rotation{};
    std::copy_n(sdk[26].r, 9, palm_rotation.begin());
    const M rotation = mul(palm_rotation, model(hand).wrist);
    V origin{};
    std::copy_n(sdk[26].p, 3, origin.begin());
    // WRIST corrections are fixed in palm_link (see the static_assert above). Share this origin with taxels.
    for (const Height* shift : { &model(hand).correction[1].lateral, &model(hand).correction[1].pad })
        origin = add(origin, mv(palm_rotation, scale(shift->inward, shift->amount)));
    std::copy(origin.begin(), origin.end(), wrist->p);
    std::copy(rotation.begin(), rotation.end(), wrist->r);
    return RH_OK;
}
int openxr_joint_of_sdk_link(std::uint32_t link)
{
    for (const auto& mapping : kFingerMap)
    {
        const auto b = static_cast<std::uint32_t>(mapping.base_angle);
        if (link == b)
            return mapping.distal_point;
        if (link == b + 1)
            return mapping.intermediate_point;
        // A non-thumb proximal point is carried by its MCP link (see assemble()); the thumb's MCP link has
        // no point of its own.
        if (link == b + 2 && mapping.placeholder_point >= 0)
            return mapping.proximal_point;
    }
    return -1;
}
RhStatus hand_points_from_sdk_points(RhHand hand, const LinkPose sdk[27], HandPoint out[26])
{
    if (!sdk || !out || (hand != RH_HAND_LEFT && hand != RH_HAND_RIGHT))
        return RH_ERR_INVALID_ARG;
    // Re-express tracker_link poses relative to palm_link (point 26); the mount is not applied again.
    V palm{};
    M palm_rotation{};
    std::copy_n(sdk[26].p, 3, palm.begin());
    std::copy_n(sdk[26].r, 9, palm_rotation.begin());
    const M to_palm = transpose(palm_rotation);
    const auto position = [&](const LinkPose& pose)
    {
        V v{};
        std::copy_n(pose.p, 3, v.begin());
        return mv(to_palm, sub(v, palm));
    };
    std::array<V, 21> p{};
    std::array<M, 21> r{};
    for (int i = 0; i < 21; ++i)
    {
        M link_rotation{};
        std::copy_n(sdk[i].r, 9, link_rotation.begin());
        p[i] = position(sdk[i]);
        r[i] = mul(to_palm, link_rotation);
    }
    std::array<V, 5> tips{};
    std::array<V, 5> tip_directions{};
    for (int f = 0; f < 5; ++f)
    {
        // SDK TIPs follow the finger order of kFingerMap and are rigid on that finger's DIP link.
        tips[f] = position(sdk[21 + f]);
        tip_directions[f] = unit(sub(tips[f], p[kFingerMap[f].base_angle]));
    }
    assemble(hand, model(hand), p, r, tips, tip_directions, mv(to_palm, scale(palm, -1)), out);
    return RH_OK;
}

} // namespace revohuman
} // namespace plugins
