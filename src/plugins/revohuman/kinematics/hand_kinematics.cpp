// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/hand_kinematics.hpp"

#include "hand_geometry.hpp"

#include <revohuman_plugin/tactile.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#if REVOHUMAN_HAS_SDK_POINTS || REVOHUMAN_HAS_SDK_VARIANT_MODEL || REVOHUMAN_HAS_SDK_TAXEL_LAYOUT
#    include <revohuman_fk.h>
#endif

namespace plugins
{
namespace revohuman
{
namespace
{

#if REVOHUMAN_HAS_SDK_POINTS
RhHand sdk_hand(Hand hand)
{
    return hand == Hand::Left ? RH_HAND_LEFT : RH_HAND_RIGHT;
}

#endif

#if REVOHUMAN_HAS_SDK_VARIANT_MODEL || REVOHUMAN_HAS_SDK_TAXEL_LAYOUT
// The SDK's error text is "E<code> <ClassName>: <detail>". Keep the code and class.
std::string sdk_error_text()
{
    char buf[512];
    if (rh_last_error(buf, sizeof(buf)) <= 0)
        return "E" + std::to_string(rh_last_error_code());
    const std::string text(buf);
    return text.substr(0, text.find(':'));
}
#endif

#if REVOHUMAN_HAS_SDK_POINTS
static_assert(RH_HAND_POINT_COUNT == kSdkPointCount);
// A new points version may reorder points or change TIP rules: review the mapping before accepting it.
static_assert(RH_HAND_POINTS_VERSION == kSdkPointsVersion);

RhHandModel sdk_model(HandModel model)
{
    return model == HandModel::P ? RH_HAND_MODEL_P : RH_HAND_MODEL_DV1;
}

template <std::size_t N>
std::string field(const char (&text)[N])
{
    return std::string(text, std::find(text, text + N, '\0'));
}

using SdkLinks = std::array<LinkPose, kSdkPointCount>;

// The SDK applies the mask itself and writes NaN for every invalid quantity.
RhStatus sdk_points(HandModel model,
                    Hand hand,
                    const std::array<double, kGloveJointCount>& angles,
                    std::uint32_t valid_mask,
                    SdkLinks& links)
{
    RhHandPoints sdk = RH_STRUCT_INIT(RhHandPoints);
    const auto status = rh_hand_points(sdk_model(model), sdk_hand(hand), angles.data(), valid_mask, &sdk);
    if (status != RH_OK)
        return status;
    for (std::size_t i = 0; i < links.size(); ++i)
    {
        std::copy_n(sdk.points[i].p, 3, links[i].p);
        std::copy_n(sdk.points[i].r, 9, links[i].r);
    }
    return RH_OK;
}
#endif

class HandKinematicsSolver final : public KinematicsProvider
{
public:
    explicit HandKinematicsSolver(HandModel model) : model_(model)
    {
#if REVOHUMAN_HAS_SDK_POINTS
        RhHandModelInfo info = RH_STRUCT_INIT(RhHandModelInfo);
        if (rh_hand_model_info(sdk_model(model_), &info) != RH_OK)
            throw std::runtime_error("the C SDK does not provide the selected hand model");
        if (info.points_version != kSdkPointsVersion)
            throw std::runtime_error("SDK hand point set version " + std::to_string(info.points_version) +
                                     " differs from the reviewed mapping; use a C SDK with points version " +
                                     std::to_string(kSdkPointsVersion));
        // The reviewed corrections require the same mechanical model and numeric joint order.
        const std::string name = field(info.name);
        const char* reviewed = hand_points_reviewed_sha256(name.c_str());
        const std::string sha = field(info.urdf_sha256);
        if (reviewed == nullptr || sha != reviewed)
            throw std::runtime_error("SDK hand model '" + name +
                                     "' differs from the reviewed models; use a matching C SDK");
        model_id_ = "sdk-points:" + name + ":" + hand_points_mapping_id() + "-" + sha;
#else
        throw std::runtime_error("SDK hand point API unavailable in this build; use a C SDK with rh_hand_points");
#endif
    }

    std::string model_id() const override
    {
        return model_id_;
    }

    std::optional<HandKinematics> solve(Hand hand,
                                        const std::array<double, kGloveJointCount>& angles,
                                        std::uint32_t valid_mask) const override
    {
        static_assert(kHandPointCount == kHandJointCount);
        static_assert(RH_JOINT_COUNT == kGloveJointCount);
        std::array<HandPoint, kHandPointCount> points{};
        if (calculate(hand, angles, valid_mask, points.data()) != RH_OK)
        {
            return std::nullopt;
        }
        HandKinematics out;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const auto& point = points[i];
            auto& pose = out.joints[i];
            const bool finite_position = std::all_of(
                std::begin(point.position), std::end(point.position), [](double v) { return std::isfinite(v); });
            double norm2 = 0.0;
            for (double v : point.orientation)
            {
                norm2 += v * v;
            }
            if ((point.flags & kHandPositionValid) && finite_position && std::isfinite(point.radius) &&
                point.radius > 0.0f)
            {
                std::copy(std::begin(point.position), std::end(point.position), pose.p.begin());
                out.position_valid_mask |= 1u << i;
            }
            if ((point.flags & kHandOrientationValid) && std::isfinite(norm2) && norm2 > 1e-12)
            {
                for (std::size_t k = 0; k < pose.q.size(); ++k)
                {
                    pose.q[k] = point.orientation[k] / std::sqrt(norm2);
                }
                out.orientation_valid_mask |= 1u << i;
            }
            out.radius[i] = std::isfinite(point.radius) && point.radius > 0.0f ? point.radius : 0.0f;
        }
        out.position_valid_mask &= kMappedHandMask;
        out.orientation_valid_mask &= kMappedHandMask;
        return out;
    }

private:
    RhStatus calculate(Hand hand,
                       const std::array<double, kGloveJointCount>& angles,
                       std::uint32_t valid_mask,
                       HandPoint* points) const
    {
#if REVOHUMAN_HAS_SDK_POINTS
        SdkLinks links{};
        const auto status = sdk_points(model_, hand, angles, valid_mask, links);
        if (status != RH_OK)
            return status;
        return hand_points_from_sdk_points(sdk_hand(hand), links.data(), points);
#else
        static_cast<void>(hand);
        static_cast<void>(angles);
        static_cast<void>(valid_mask);
        static_cast<void>(points);
        return RH_ERR_INVALID_ARG;
#endif
    }
    HandModel model_;
    std::string model_id_;
};

#if REVOHUMAN_HAS_SDK_POINTS && REVOHUMAN_HAS_SDK_TAXEL_LAYOUT
// A new layout version may reorder taxels or change how positions and normals are taken: review first.
static_assert(RH_TAXEL_LAYOUT_VERSION == 1);
static_assert(RH_TACTILE_MAX_POINTS == kMaxTaxels);

constexpr std::uint32_t kAllGloveJoints = (1u << kGloveJointCount) - 1u;

Vec3 rotate(const double (&r)[9], const double (&v)[3])
{
    return { r[0] * v[0] + r[1] * v[1] + r[2] * v[2], r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
             r[6] * v[0] + r[7] * v[1] + r[8] * v[2] };
}

Vec3 rotate_back(const double (&r)[9], const Vec3& v)
{
    return { r[0] * v[0] + r[3] * v[1] + r[6] * v[2], r[1] * v[0] + r[4] * v[1] + r[7] * v[2],
             r[2] * v[0] + r[5] * v[1] + r[8] * v[2] };
}

// Shortest-arc rotation taking +Z to the unit vector @p n; a half turn about X for -Z.
Quat outward_z(const Vec3& n)
{
    const double w = 1 + n[2];
    if (w < 1e-12)
        return { 1, 0, 0, 0 };
    return quat_normalize({ -n[1], n[0], 0, w });
}

// The taxels at @p angles with the SDK points they were computed from (tracker_link frame).
SdkTaxels taxels_at(HandModel model, Hand hand, const std::array<double, kGloveJointCount>& angles, SdkLinks& links)
{
    // Same model checks as the poses the shape is skinned with.
    const HandKinematicsSolver checked(model);
    static_cast<void>(checked);
    if (sdk_points(model, hand, angles, kAllGloveJoints, links) != RH_OK)
        throw std::runtime_error("the C SDK cannot compute the hand points: " + sdk_error_text());
    LinkPose wrist{};
    if (wrist_from_sdk_points(sdk_hand(hand), links.data(), &wrist) != RH_OK)
        throw std::runtime_error("cannot place the WRIST in the SDK hand points");
    // About 19 KB: keep it off the stack.
    auto layout = std::make_unique<RhTaxelLayout>();
    layout->struct_size = sizeof(RhTaxelLayout);
    if (rh_hand_model_taxel_layout(sdk_model(model), sdk_hand(hand), RH_TACTILE_POINTS_TYPE_B, layout.get()) != RH_OK)
        throw std::runtime_error("the C SDK has no taxel layout for this glove: " + sdk_error_text());
    if (layout->layout_version != RH_TAXEL_LAYOUT_VERSION)
        throw std::runtime_error("the C SDK gives taxel layout version " + std::to_string(layout->layout_version) +
                                 ", which differs from the reviewed version " + std::to_string(RH_TAXEL_LAYOUT_VERSION));
    if (layout->count != RH_TACTILE_POINTS_TYPE_B)
        throw std::runtime_error("the C SDK gives " + std::to_string(layout->count) + " taxels instead of " +
                                 std::to_string(RH_TACTILE_POINTS_TYPE_B));
    SdkTaxels out;
    out.source = field(layout->source);
    for (std::uint32_t i = 0; i < layout->count; ++i)
    {
        const auto& taxel = layout->taxels[i];
        if (taxel.link >= kGloveJointCount)
            throw std::runtime_error("the C SDK puts taxel " + std::to_string(i) + " on unknown link " +
                                     std::to_string(taxel.link));
        const auto& link = links[taxel.link];
        Vec3 at = rotate(link.r, taxel.p);
        for (int k = 0; k < 3; ++k)
            at[k] += link.p[k] - wrist.p[k];
        out.positions.push_back(rotate_back(wrist.r, at));
        out.normals.push_back(normalize(rotate_back(wrist.r, rotate(link.r, taxel.n))));
        out.links.push_back(taxel.link);
    }
    return out;
}
#endif

} // namespace

HandModel hand_model_of_product(const std::string& product_code)
{
#if REVOHUMAN_HAS_SDK_VARIANT_MODEL
    RhHandModel model = 0;
    if (rh_variant_hand_model(product_code.c_str(), &model) != RH_OK)
        throw std::runtime_error("the C SDK has no hand model for product '" + product_code + "': " + sdk_error_text());
    if (model == RH_HAND_MODEL_DV1)
        return HandModel::Dv1;
    if (model == RH_HAND_MODEL_P)
        return HandModel::P;
    throw std::runtime_error("the C SDK gives product '" + product_code + "' hand model " + std::to_string(model) +
                             ", which this build does not know");
#else
    static_cast<void>(product_code);
    throw std::runtime_error(
        "this build's C SDK cannot tell the hand model of a product; use a C SDK with rh_variant_hand_model");
#endif
}

bool sdk_has_product_hand_models()
{
    return REVOHUMAN_HAS_SDK_VARIANT_MODEL;
}

std::shared_ptr<const KinematicsProvider> make_hand_kinematics(HandModel model)
{
    return std::make_shared<HandKinematicsSolver>(model);
}

SdkTaxels sdk_taxels(HandModel model, Hand hand, const std::array<double, kGloveJointCount>& angles)
{
#if REVOHUMAN_HAS_SDK_POINTS && REVOHUMAN_HAS_SDK_TAXEL_LAYOUT
    SdkLinks links{};
    return taxels_at(model, hand, angles, links);
#else
    static_cast<void>(hand);
    static_cast<void>(angles);
    // Same refusal as the poses for a build without points.
    make_hand_kinematics(model);
    throw std::runtime_error(
        "this build's C SDK has no taxel layout; use a C SDK with rh_hand_model_taxel_layout (0.1.3 or later)");
#endif
}

SdkTactileShape make_sdk_tactile_shape(HandModel model, Hand hand)
{
#if REVOHUMAN_HAS_SDK_POINTS && REVOHUMAN_HAS_SDK_TAXEL_LAYOUT
    SdkLinks links{};
    const SdkTaxels taxels = taxels_at(model, hand, {}, links);
    // Bind every OpenXR joint as the mapping places it at all-zero angles, placeholders included.
    std::array<HandPoint, kHandPointCount> points{};
    if (hand_points_from_sdk_points(sdk_hand(hand), links.data(), points.data()) != RH_OK)
        throw std::runtime_error("cannot map the SDK hand points to OpenXR joints");
    ShapeSpec spec;
    for (std::size_t j = 0; j < points.size(); ++j)
    {
        Pose bind;
        std::copy(std::begin(points[j].position), std::end(points[j].position), bind.p.begin());
        bind.q = quat_normalize(
            { points[j].orientation[0], points[j].orientation[1], points[j].orientation[2], points[j].orientation[3] });
        spec.bind_poses.push_back(bind);
        spec.joint_ids.push_back(static_cast<core::HandJoint>(j));
    }
    for (std::size_t i = 0; i < taxels.positions.size(); ++i)
    {
        const int joint = openxr_joint_of_sdk_link(taxels.links[i]);
        if (joint < 0)
            throw std::runtime_error("the C SDK puts taxel " + std::to_string(i) + " on link " +
                                     std::to_string(taxels.links[i]) + ", which no OpenXR joint follows");
        spec.taxel_bind_poses.push_back({ taxels.positions[i], outward_z(taxels.normals[i]) });
        spec.joint_indices.push_back({ static_cast<std::uint32_t>(joint), 0, 0, 0 });
        spec.joint_weights.push_back({ 1, 0, 0, 0 });
    }
    return { build_shape_buffer(spec), taxels.source };
#else
    static_cast<void>(hand);
    make_hand_kinematics(model);
    throw std::runtime_error(
        "this build's C SDK has no taxel layout; use a C SDK with rh_hand_model_taxel_layout (0.1.3 or later)");
#endif
}

} // namespace revohuman
} // namespace plugins
