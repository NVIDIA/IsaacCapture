// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/hand_model.hpp"

#include "inc/revohuman_plugin/sdk.hpp"

#include <log_bridge/logger.hpp>
#include <schema/serialized.hpp>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_metadata_validation.hpp>
#include <schema/tracker_metadata.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <revohuman_fk.h>
#include <stdexcept>
#include <utility>

namespace plugins
{
namespace revohuman
{
namespace
{

static_assert(RH_JOINT_COUNT == kGloveJointCount);
static_assert(RH_TACTILE_MAX_POINTS == kMaxTaxels);
// A new point set or taxel layout version may reorder points or change their rules: refit the mapping first.
static_assert(RH_HAND_POINT_COUNT == 27 && RH_HAND_POINTS_VERSION == 1);
static_assert(RH_TAXEL_LAYOUT_VERSION == 1);

using V = std::array<double, 3>;
using M = std::array<double, 9>; // Row-major, as the SDK writes rotations.

// Link axes point their -X to the back of the hand on every SDK model.
constexpr V kDorsal{ -1, 0, 0 };
// WRIST axes, a relabeling of palm_link: OpenXR X = -palm Y, Y = -palm X, Z = -palm Z. Do not aim WRIST at a
// finger: every pose and taxel is expressed in this one fixed basis.
constexpr M kWristAxes{ 0, -1, 0, -1, 0, 0, 0, 0, -1 };
const core::Pose kIdentityPose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 1));

// Per finger (index, middle, ring, little, thumb): the SDK link of the DIP, the link that orients the proximal
// point and the OpenXR proximal point (the next three follow it).
struct Finger
{
    int base, orientation, proximal;
};
constexpr std::array<Finger, 5> kFingers = { { { 0, 2, 7 }, { 4, 6, 12 }, { 8, 10, 17 }, { 12, 14, 22 }, { 16, 19, 2 } } };
constexpr std::size_t kThumb = kFingers.size() - 1;
// The placeholders are the non-thumb metacarpals, the OpenXR joint before each proximal: the joints
// kMappedHandMask leaves out.
static_assert(
    []
    {
        std::uint32_t placeholders = 0;
        for (std::size_t i = 0; i < kThumb; ++i)
            placeholders |= 1u << (kFingers[i].proximal - 1);
        return placeholders == (((1u << kHandJointCount) - 1u) & ~kMappedHandMask);
    }());
// Placeholder positions in the PALM basis.
constexpr std::array<V, kThumb> kPlaceholderAt = {
    { { .018, .003, -.055 }, { .005, .002, -.055 }, { -.008, .003, -.055 }, { -.022, .005, -.05 } }
};

#include "hand_fit.inc"

bool new_geometry(const std::string& sha)
{
    return std::find(kNewGeometryHashes.begin(), kNewGeometryHashes.end(), sha) != kNewGeometryHashes.end();
}

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
// a / |a|, or NaN for a vector too short to have a direction.
template <std::size_t N>
std::array<double, N> unit(std::array<double, N> a)
{
    double n = 0;
    for (double v : a)
        n += v * v;
    n = std::sqrt(n);
    for (double& v : a)
        v = n > 1e-12 ? v / n : std::numeric_limits<double>::quiet_NaN();
    return a;
}
V mv(const M& a, V b)
{
    V o{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            o[i] += a[3 * i + j] * b[j];
    return o;
}
M mul(const M& a, const M& b)
{
    M o{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return o;
}
M transpose(const M& a)
{
    return { a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8] };
}
// Rotation whose Z is z and whose Y is y made orthogonal to it.
M frame(V z, V y)
{
    z = unit(z);
    y = sub(y, scale(z, dot(y, z)));
    if (dot(y, y) < 1e-14)
        y = cross(z, std::abs(z[0]) < .8 ? V{ 1, 0, 0 } : V{ 0, 1, 0 });
    y = unit(y);
    const V x = unit(cross(y, z));
    y = unit(cross(z, x));
    return { x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2] };
}
std::array<double, 4> quaternion(const M& a)
{
    std::array<double, 4> q{};
    const double trace = a[0] + a[4] + a[8];
    if (trace > 0)
    {
        const double s = 2 * std::sqrt(trace + 1);
        q = { (a[7] - a[5]) / s, (a[2] - a[6]) / s, (a[3] - a[1]) / s, s / 4 };
    }
    else
    {
        int i = a[4] > a[0] ? 1 : 0;
        if (a[8] > a[3 * i + i])
            i = 2;
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const double s = 2 * std::sqrt(1 + a[3 * i + i] - a[3 * j + j] - a[3 * k + k]);
        q[i] = s / 4;
        q[j] = (a[3 * i + j] + a[3 * j + i]) / s;
        q[k] = (a[3 * i + k] + a[3 * k + i]) / s;
        q[3] = (a[3 * k + j] - a[3 * j + k]) / s;
    }
    return unit(q);
}
bool finite(const auto& values)
{
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
V mirrored(V v, bool right)
{
    if (right)
        v[1] = -v[1];
    return v;
}
V position_of(const RhLinkPose& pose)
{
    return { pose.p[0], pose.p[1], pose.p[2] };
}
M rotation_of(const RhLinkPose& pose)
{
    M r{};
    std::copy_n(pose.r, 9, r.begin());
    return r;
}

// The OpenXR points placed by the fitted offsets (hand_fit.inc) in the frames of the SDK link poses, in the WRIST
// frame and oriented from those poses. Invalid SDK quantities are NaN, stay NaN through the arithmetic and keep the
// identity pose; the placeholders are never valid and keep their fitted poses only when @p retain_unmapped is set.
HandJointPoses map_points(Hand hand, const RhHandPoints& sdk, bool retain_unmapped)
{
    const bool right = hand == Hand::Right;
    // The SDK points in palm_link: the joint links, then the TIPs in kFingers order, rigid on the finger's DIP link.
    const V palm = position_of(sdk.points[26]);
    const M to_palm = transpose(rotation_of(sdk.points[26]));
    std::array<V, RH_HAND_POINT_COUNT> p{};
    std::array<M, kGloveJointCount> r{};
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = mv(to_palm, sub(position_of(sdk.points[i]), palm));
    for (std::size_t i = 0; i < r.size(); ++i)
        r[i] = mul(to_palm, rotation_of(sdk.points[i]));
    // The PALM axes in palm_link; the right hand mirrors palm Y and OpenXR X.
    M palm_basis = kNewPalmBasis;
    if (right)
        for (int i : { 0, 4, 5, 6 })
            palm_basis[i] = -palm_basis[i];

    std::array<V, kHandJointCount> points{};
    std::array<M, kHandJointCount> frames{};
    points[0] = add(mv(to_palm, scale(palm, -1)), kNewPalmPad); // Offset from the tracker origin, not palm_link.
    points[1] = kNewWristPad;
    frames[0] = palm_basis;
    frames[1] = kWristAxes;
    for (std::size_t i = 0; i < kFingers.size(); ++i)
    {
        const Finger& f = kFingers[i];
        const FingerFit& fit = kNewFingers[i];
        const M& driver = r[i == kThumb ? 19 : f.base + 3];
        const auto shifted = [&](const V& at, const V& offset) { return add(at, mv(driver, mirrored(offset, right))); };
        const V pad = shifted(p[i == kThumb ? 20 : f.base + 3], fit.proximal_pad);
        points[f.proximal] =
            i == kThumb ? pad : add(p[f.base + 2], mv(mul(r[f.base + 2], transpose(driver)), sub(pad, p[f.base + 2])));
        points[f.proximal + 1] = shifted(p[f.base + 1], fit.side);
        points[f.proximal + 2] = shifted(p[f.base], fit.side);
        points[f.proximal + 3] = shifted(p[21 + i], fit.side);
        const V tip_dir = unit(sub(p[21 + i], p[f.base]));
        if (i != kThumb)
        {
            V at = kPlaceholderAt[i];
            if (right)
                at[0] = -at[0];
            points[f.proximal - 1] = add(mv(palm_basis, at), mirrored(fit.placeholder_pad, right));
            frames[f.proximal - 1] = palm_basis;
        }
        frames[f.proximal + 2] = frames[f.proximal + 3] = frame(scale(tip_dir, -1), mv(r[f.base], kDorsal));
    }
    for (const Finger& f : kFingers)
    {
        frames[f.proximal] = frame(sub(points[f.proximal], points[f.proximal + 1]), mv(r[f.orientation], kDorsal));
        frames[f.proximal + 1] = frame(sub(points[f.proximal + 1], points[f.proximal + 2]), mv(r[f.base + 1], kDorsal));
    }
    HandJointPoses out;
    out.joints.fill(kIdentityPose);
    const M inv = transpose(frames[1]);
    for (std::size_t i = 0; i < kHandJointCount; ++i)
    {
        const V at = mv(inv, sub(points[i], points[1]));
        const auto q = quaternion(mul(inv, frames[i]));
        const std::uint32_t mapped = kMappedHandMask & (1u << i);
        if (finite(at) && (mapped || retain_unmapped))
        {
            out.joints[i].mutable_position() = core::Point(at[0], at[1], at[2]);
            out.position_valid_mask |= mapped;
        }
        if (finite(q) && (mapped || retain_unmapped))
        {
            out.joints[i].mutable_orientation() = core::Quaternion(q[0], q[1], q[2], q[3]);
            out.orientation_valid_mask |= mapped;
        }
    }
    return out;
}

// The OpenXR point that moves with SDK link @p link, or -1 (MPR links, thumb MCP/CMR/CMP).
int joint_of_link(std::uint32_t link)
{
    for (std::size_t i = 0; i < kFingers.size(); ++i)
    {
        const Finger& f = kFingers[i];
        if (link == static_cast<std::uint32_t>(f.base))
            return f.proximal + 2;
        if (link == static_cast<std::uint32_t>(f.base + 1))
            return f.proximal + 1;
        if (link == static_cast<std::uint32_t>(f.base + 2) && i != kThumb)
            return f.proximal;
    }
    return -1;
}

} // namespace

HandJointPoses fit_hand_points(const std::string& model_sha256,
                               Hand hand,
                               const RhHandPoints& points,
                               bool retain_unmapped_positions)
{
    if (!new_geometry(model_sha256))
        throw std::runtime_error("no fitted table for hand model " + model_sha256);
    if (points.struct_size != sizeof(RhHandPoints) || points.points_version != RH_HAND_POINTS_VERSION ||
        points.hand != to_rh(hand))
        throw std::runtime_error("incompatible SDK hand point set");
    return map_points(hand, points, retain_unmapped_positions);
}

HandModel HandModel::for_product(const std::string& product_code)
{
    RhHandModel model = 0;
    if (rh_variant_hand_model(product_code.c_str(), &model) != RH_OK)
        throw std::runtime_error("the C SDK has no hand model for product '" + product_code + "': " + sdk_error());
    RhHandModelInfo info = RH_STRUCT_INIT(RhHandModelInfo);
    if (rh_hand_model_info(model, &info) != RH_OK || info.points_version != RH_HAND_POINTS_VERSION)
        throw std::runtime_error("the C SDK gives no hand point set for product '" + product_code + "'");
    const std::string name = sdk_text(info.name);
    const std::string sha = sdk_text(info.urdf_sha256);
    if (!new_geometry(sha))
        throw std::runtime_error("the plugin is not fitted to hand model '" + name + "' (URDF SHA-256 " + sha +
                                 "); use the C SDK release that install.sh pins");
    return HandModel(model, "sdk-points:" + name + ":new-geometry-v1-" + sha);
}

HandJointPoses HandModel::joint_poses(Hand hand,
                                      const std::array<double, kGloveJointCount>& joints_rad,
                                      std::uint32_t joints_valid_mask) const
{
    RhHandPoints sdk = RH_STRUCT_INIT(RhHandPoints);
    if (rh_hand_points(sdk_model_, to_rh(hand), joints_rad.data(), joints_valid_mask, &sdk) != RH_OK)
        throw std::runtime_error("the C SDK cannot compute the hand points: " + sdk_error());
    return map_points(hand, sdk, false);
}

std::vector<std::uint8_t> HandModel::tactile_metadata(Hand hand, std::uint32_t taxel_count) const
{
    const RhHand rh = to_rh(hand);
    const std::array<double, kGloveJointCount> zero{};
    RhHandPoints sdk = RH_STRUCT_INIT(RhHandPoints);
    if (rh_hand_points(sdk_model_, rh, zero.data(), (1u << kGloveJointCount) - 1u, &sdk) != RH_OK)
        throw std::runtime_error("the C SDK cannot compute the hand points: " + sdk_error());
    auto layout = std::make_unique<RhTaxelLayout>(); // About 19 KB: off the stack.
    layout->struct_size = sizeof(RhTaxelLayout);
    if (rh_hand_model_taxel_layout(sdk_model_, rh, taxel_count, layout.get()) != RH_OK)
        throw std::runtime_error("the C SDK has no taxel layout for " + std::to_string(taxel_count) +
                                 " taxels: " + sdk_error());
    isaaccapture::Logger::get("isaaccapture.plugins.revohuman.HandModel")
        ->debug("{}: taxel layout {}", hand_name(hand), sdk_text(layout->source));

    // WRIST in tracker_link, as map_points() places it.
    const V wrist = add(position_of(sdk.points[26]), mv(rotation_of(sdk.points[26]), kNewWristPad));
    const M to_wrist = transpose(mul(rotation_of(sdk.points[26]), kWristAxes));

    core::LBST lbs;
    core::HandSkeletonRefT skeleton;
    const HandJointPoses bind = map_points(hand, sdk, true);
    lbs.segment_bind_poses.assign(bind.joints.begin(), bind.joints.end()); // quaternion() normalizes.
    for (std::size_t i = 0; i < kHandJointCount; ++i)
        skeleton.joint_ids.push_back(static_cast<core::HandJoint>(i));
    for (std::uint32_t i = 0; i < layout->count; ++i)
    {
        const RhTaxel& taxel = layout->taxels[i];
        const int joint = taxel.link < kGloveJointCount ? joint_of_link(taxel.link) : -1;
        if (joint < 0)
            throw std::runtime_error("the C SDK puts taxel " + std::to_string(i) + " on link " +
                                     std::to_string(taxel.link) + ", which no OpenXR joint follows");
        const RhLinkPose& link = sdk.points[taxel.link];
        const M link_rotation = rotation_of(link);
        V at = mv(link_rotation, { taxel.p[0], taxel.p[1], taxel.p[2] });
        for (int k = 0; k < 3; ++k)
            at[k] += link.p[k] - wrist[k];
        const V normal = unit(mv(to_wrist, mv(link_rotation, { taxel.n[0], taxel.n[1], taxel.n[2] })));
        // Shortest-arc rotation taking +Z to the outward normal; a half turn about X for -Z.
        const std::array<double, 4> q = 1 + normal[2] >= 1e-12 ?
                                            unit(std::array<double, 4>{ -normal[1], normal[0], 0, 1 + normal[2] }) :
                                            std::array<double, 4>{ 1, 0, 0, 0 };
        const V p = mv(to_wrist, at);
        lbs.taxel_bind_poses.emplace_back(core::Point(p[0], p[1], p[2]), core::Quaternion(q[0], q[1], q[2], q[3]));
        const std::array<std::uint32_t, 4> segments{ static_cast<std::uint32_t>(joint), 0, 0, 0 };
        const std::array<float, 4> weights{ 1, 0, 0, 0 };
        lbs.taxel_segment_indices.emplace_back(flatbuffers::span<const std::uint32_t, 4>(segments));
        lbs.taxel_segment_weights.emplace_back(flatbuffers::span<const float, 4>(weights));
    }
    core::TactileMetadataT metadata;
    metadata.geometry_mapping.Set(std::move(lbs));
    metadata.segment_source.Set(std::move(skeleton));
    const auto packed = core::pack<core::TactileMetadata>(metadata);
    if (packed.buffer().size() > core::TRACKER_METADATA_MAX_SIZE)
        throw std::runtime_error("TactileMetadata exceeds the tracker metadata cap");
    core::verify_tactile_metadata(*packed);
    return { packed.buffer().begin(), packed.buffer().end() };
}

} // namespace revohuman
} // namespace plugins
