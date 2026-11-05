// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/hand_kinematics.hpp>
#include <revohuman_plugin/pipeline.hpp>
#include <revohuman_plugin/pose_math.hpp>
#include <revohuman_plugin/tactile.hpp>

#if REVOHUMAN_TEST_SDK_POINTS && REVOHUMAN_TEST_SDK_TAXEL_LAYOUT
#    include <revohuman_fk.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace plugins::revohuman;
using Catch::Approx;

#if REVOHUMAN_TEST_SDK_POINTS
namespace
{
constexpr std::uint32_t kAllAngles = (1u << kGloveJointCount) - 1;
constexpr std::array<int, 5> kDistal{ 9, 14, 19, 24, 4 };
constexpr std::array<double, 5> kLengths{ .025370264322333556, .025370258547203105, .025370262040019596,
                                          .025198549138189495, .03058276137814423 };
// Non-thumb fingers: glove angle slot of the MCP flexion, and the OpenXR proximal and intermediate joints.
constexpr std::array<std::size_t, 4> kMcpSlots{ 2, 6, 10, 14 };
constexpr std::array<int, 4> kProximal{ 7, 12, 17, 22 };
constexpr std::array<int, 4> kIntermediate{ 8, 13, 18, 23 };

Vec3 delta(const Vec3& a, const Vec3& b)
{
    return { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

} // namespace
#endif

#if REVOHUMAN_TEST_SDK_POINTS
TEST_CASE("reviewed hand tips follow each DIP for both hands", "[revohuman][geometry]")
{
    const auto provider = make_hand_kinematics(HandModel::Dv1);
    CHECK(provider->model_id().rfind("sdk-points:dv1:reviewed-openxr-symmetric-v6-", 0) == 0);
    for (const auto hand : { Hand::Left, Hand::Right })
    {
        std::array<double, kGloveJointCount> angles{};
        const auto initial = provider->solve(hand, angles, kAllAngles);
        REQUIRE(initial);
        REQUIRE(initial->position_valid_mask == kMappedHandMask);
        REQUIRE(initial->orientation_valid_mask == kMappedHandMask);
        for (int finger = 0; finger < 5; ++finger)
        {
            angles.fill(0);
            angles[4 * finger] = .6;
            const auto moved = provider->solve(hand, angles, kAllAngles);
            REQUIRE(moved);
            const auto distal = kDistal[finger];
            const auto tip = distal + 1;
            CHECK(norm(delta(moved->joints[distal].p, initial->joints[distal].p)) < 1e-12);
            CHECK(norm(delta(moved->joints[tip].p, initial->joints[tip].p)) > .001);
            const auto segment = delta(moved->joints[tip].p, moved->joints[distal].p);
            CHECK(norm(segment) == Approx(kLengths[finger]).margin(1e-12));
            const auto forward = quat_rotate(moved->joints[tip].q, { 0, 0, -1 });
            CHECK(dot(normalize(segment), forward) == Approx(1).margin(1e-12));
            double quaternion_dot = 0;
            for (int i = 0; i < 4; ++i)
                quaternion_dot += initial->joints[tip].q[i] * moved->joints[tip].q[i];
            CHECK(2 * std::acos(std::min(1.0, std::abs(quaternion_dot))) == Approx(.6).margin(1e-12));
            for (int other = 0; other < 5; ++other)
                if (other != finger)
                    CHECK(norm(delta(moved->joints[kDistal[other] + 1].p, initial->joints[kDistal[other] + 1].p)) < 1e-12);
            const auto invalid = provider->solve(hand, angles, kAllAngles & ~(1u << (4 * finger)));
            REQUIRE(invalid);
            CHECK((invalid->position_valid_mask & (1u << tip)) == 0);
            CHECK((invalid->orientation_valid_mask & (1u << tip)) == 0);
        }
        for (std::size_t slot = 0; slot < angles.size(); ++slot)
        {
            angles.fill(0);
            angles[slot] = .4;
            const auto moved = provider->solve(hand, angles, kAllAngles);
            REQUIRE(moved);
            CHECK(moved->position_valid_mask == kMappedHandMask);
            double change = 0;
            for (std::size_t j = 0; j < kHandJointCount; ++j)
                if (kMappedHandMask & (1u << j))
                {
                    change += norm(delta(moved->joints[j].p, initial->joints[j].p));
                    for (int k = 0; k < 4; ++k)
                        change += std::abs(moved->joints[j].q[k] - initial->joints[j].q[k]);
                }
            CHECK(change > 1e-6);
            for (int fixed : { 0, 1, 6, 11, 16, 21 })
            {
                CHECK(norm(delta(moved->joints[fixed].p, initial->joints[fixed].p)) < 1e-12);
                CHECK(moved->joints[fixed].q == initial->joints[fixed].q);
            }
        }
    }
}

TEST_CASE("the corrected WRIST lies on the neutral palm plane for both products", "[revohuman][geometry]")
{
    // Palm-link +X toward the skin, expressed in the reviewed OpenXR WRIST basis.
    const Vec3 inward{ 0, -.9996343707598901, .027039319434451838 };
    for (const auto model : { HandModel::Dv1, HandModel::P })
    {
        const auto provider = make_hand_kinematics(model);
        for (const auto hand : { Hand::Left, Hand::Right })
        {
            const auto solved = provider->solve(hand, {}, kAllAngles);
            REQUIRE(solved);
            CHECK(norm(solved->joints[1].p) < 1e-12);
            CHECK(std::abs(solved->joints[1].q[0]) < 1e-12);
            CHECK(std::abs(solved->joints[1].q[1]) < 1e-12);
            CHECK(std::abs(solved->joints[1].q[2]) < 1e-12);
            CHECK(solved->joints[1].q[3] == Approx(1).margin(1e-12));
            CHECK(std::abs(dot(solved->joints[0].p, inward)) < 1e-12);
            for (const int proximal : { 12, 17 })
                CHECK(std::abs(dot(solved->joints[proximal].p, inward)) < .000501);
        }
    }
}

TEST_CASE("a non-thumb proximal joint is rigid on the MCP link", "[revohuman][geometry]")
{
    // The bone to the intermediate joint keeps its length and turns by the encoder angle. P takes the
    // corrections fitted on DV1, whose MCP axes it misses by 2.5e-4 rad, so it is rigid to a few microns.
    for (const auto& [model, tolerance] : { std::pair{ HandModel::Dv1, 1e-9 }, std::pair{ HandModel::P, 1e-5 } })
    {
        const auto provider = make_hand_kinematics(model);
        for (const auto hand : { Hand::Left, Hand::Right })
        {
            CAPTURE(provider->model_id(), hand_name(hand));
            std::array<double, kGloveJointCount> angles{};
            const auto initial = provider->solve(hand, angles, kAllAngles);
            REQUIRE(initial);
            for (std::size_t finger = 0; finger < kMcpSlots.size(); ++finger)
            {
                const int proximal = kProximal[finger], intermediate = kIntermediate[finger];
                const double length = norm(delta(initial->joints[proximal].p, initial->joints[intermediate].p));
                for (const double bend : { .35, .7, 1.2 })
                {
                    CAPTURE(finger, bend);
                    angles.fill(0);
                    angles[kMcpSlots[finger]] = bend;
                    const auto moved = provider->solve(hand, angles, kAllAngles);
                    REQUIRE(moved);
                    CHECK(std::abs(norm(delta(moved->joints[proximal].p, moved->joints[intermediate].p)) - length) <
                          tolerance);
                    double quaternion_dot = 0;
                    for (int i = 0; i < 4; ++i)
                        quaternion_dot += initial->joints[proximal].q[i] * moved->joints[proximal].q[i];
                    CHECK(std::abs(2 * std::acos(std::min(1.0, std::abs(quaternion_dot))) - bend) < 10 * tolerance);
                    const auto invalid = provider->solve(hand, angles, kAllAngles & ~(1u << kMcpSlots[finger]));
                    REQUIRE(invalid);
                    CHECK((invalid->position_valid_mask & (1u << proximal)) == 0);
                }
            }
        }
    }
}

TEST_CASE("hand fitting remains mirrored between left and right", "[revohuman][geometry]")
{
    for (const auto model : { HandModel::Dv1, HandModel::P })
    {
        const auto provider = make_hand_kinematics(model);
        for (const double bend : { 0., .35, .7 })
        {
            std::array<double, kGloveJointCount> angles{};
            angles.fill(bend);
            const auto left = provider->solve(Hand::Left, angles, kAllAngles);
            const auto right = provider->solve(Hand::Right, angles, kAllAngles);
            REQUIRE(left);
            REQUIRE(right);
            CHECK(left->position_valid_mask == right->position_valid_mask);
            CHECK(left->orientation_valid_mask == right->orientation_valid_mask);
            for (std::size_t j = 0; j < kHandJointCount; ++j)
            {
                CAPTURE(provider->model_id(), bend, j);
                const auto& l = left->joints[j];
                const auto& r = right->joints[j];
                CHECK(norm(delta({ -l.p[0], l.p[1], l.p[2] }, r.p)) < 1e-12);
                const Quat mirrored{ l.q[0], -l.q[1], -l.q[2], l.q[3] };
                double qdot = 0;
                for (std::size_t k = 0; k < 4; ++k)
                    qdot += mirrored[k] * r.q[k];
                CHECK(std::abs(qdot) == Approx(1).margin(1e-12));
            }
        }
    }
}

TEST_CASE("hand model P has poses but no taxels", "[revohuman][geometry]")
{
    // P1 gloves have no tactile sensors, so the SDK has no taxel layout for their hand model.
    const auto provider = make_hand_kinematics(HandModel::P);
    CHECK(provider->model_id() ==
          "sdk-points:p:reviewed-openxr-symmetric-v6-b400ebf2df5abcbf1d8c88fd66ded8ec73cd5c35bb62cf28c92eb153e007edc2");
    for (const auto hand : { Hand::Left, Hand::Right })
    {
        CAPTURE(hand_name(hand));
        for (const double bend : { 0.0, .5 })
        {
            std::array<double, kGloveJointCount> angles{};
            angles.fill(bend);
            const auto solved = provider->solve(hand, angles, kAllAngles);
            REQUIRE(solved);
            CHECK(solved->position_valid_mask == kMappedHandMask);
            CHECK(solved->orientation_valid_mask == kMappedHandMask);
        }
    }
    CHECK_THROWS_AS(make_sdk_tactile_shape(HandModel::P, Hand::Left), std::runtime_error);
    CHECK_THROWS_AS(sdk_taxels(HandModel::P, Hand::Left, {}), std::runtime_error);
}

#else
TEST_CASE("live hand poses fail at startup without the SDK point API", "[revohuman][geometry]")
{
    CHECK_THROWS_AS(make_hand_kinematics(HandModel::Dv1), std::runtime_error);
    CHECK_THROWS_AS(make_hand_kinematics(HandModel::P), std::runtime_error);
    CHECK_THROWS_AS(make_sdk_tactile_shape(HandModel::P, Hand::Left), std::runtime_error);
}
#endif

#if REVOHUMAN_TEST_SDK_POINTS && REVOHUMAN_TEST_SDK_TAXEL_LAYOUT
namespace
{
constexpr std::size_t kLayoutTaxels = RH_TACTILE_POINTS_TYPE_B;
// TactileMetadata stores poses as float32, so what is read back from it carries float rounding.
constexpr double kFloatPosition = 1e-7; // m
constexpr double kFloatDirection = 1e-6;

double angle_between(const Vec3& a, const Vec3& b)
{
    return std::acos(std::clamp(dot(normalize(a), normalize(b)), -1.0, 1.0));
}

double det(const Vec3& a, const Vec3& b, const Vec3& c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

//! Largest position and normal-angle error of the skinned taxels against the SDK's at @p angles.
std::pair<double, double> skin_error(const LoadedShape& shape,
                                     const KinematicsProvider& provider,
                                     Hand hand,
                                     const std::array<double, kGloveJointCount>& angles)
{
    const auto kinematics = provider.solve(hand, angles, kAllAngles);
    REQUIRE(kinematics);
    REQUIRE(shape.has_valid_orientations(*kinematics));
    const auto expected = sdk_taxels(HandModel::Dv1, hand, angles);
    const auto positions = shape.skin_positions(*kinematics);
    const auto normals = shape.current_normals(*kinematics);
    double position = 0, normal = 0;
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        position = std::max(position, norm(delta(positions[i], expected.positions[i])));
        normal = std::max(normal, angle_between(normals[i], expected.normals[i]));
    }
    return { position, normal };
}
} // namespace

TEST_CASE("the TactileMetadata comes from the SDK taxel layout and matches it at the bind pose",
          "[revohuman][tactile][geometry]")
{
    const auto provider = make_hand_kinematics(HandModel::Dv1);
    // The OpenXR joint each SDK link's taxels ride on: DIP, PIP and (non-thumb) MCP links.
    const std::array<int, kGloveJointCount> joint_of_link{ 9,  8,  7,  -1, 14, 13, 12, -1, 19, 18, 17,
                                                           -1, 24, 23, 22, -1, 4,  3,  -1, -1, -1 };
    for (const auto hand : { Hand::Left, Hand::Right })
    {
        CAPTURE(hand_name(hand));
        const auto built = make_sdk_tactile_shape(HandModel::Dv1, hand);
        CHECK(built.layout_source.rfind("cuff-surface-v", 0) == 0);
        const LoadedShape shape(built.metadata);
        REQUIRE(shape.taxel_count() == kLayoutTaxels);
        CHECK(shape.bytes().size() <= shape_size_bound(kHandJointCount, kLayoutTaxels));
        const auto& lbs = *shape.shape().geometry_mapping_as_LBS();
        const auto* ids = shape.shape().segment_source_as_HandSkeletonRef()->joint_ids();
        REQUIRE(ids->size() == kHandJointCount);
        const std::array<double, kGloveJointCount> zero{};
        const auto bind = provider->solve(hand, zero, kAllAngles);
        REQUIRE(bind);
        for (std::size_t j = 0; j < kHandJointCount; ++j)
        {
            CAPTURE(j);
            CHECK(ids->Get(j) == j);
            if (!(kMappedHandMask & (1u << j)))
                continue;
            const auto& pose = *lbs.segment_bind_poses()->Get(j);
            CHECK(norm(delta({ pose.position().x(), pose.position().y(), pose.position().z() }, bind->joints[j].p)) <
                  kFloatPosition);
            const Quat q{ pose.orientation().x(), pose.orientation().y(), pose.orientation().z(), pose.orientation().w() };
            double qdot = 0;
            for (int k = 0; k < 4; ++k)
                qdot += q[k] * bind->joints[j].q[k];
            CHECK(1 - std::abs(qdot) < kFloatDirection);
        }
        const auto taxels = sdk_taxels(HandModel::Dv1, hand, zero);
        REQUIRE(taxels.positions.size() == kLayoutTaxels);
        CHECK(taxels.source == built.layout_source);
        for (std::size_t i = 0; i < kLayoutTaxels; ++i)
        {
            CAPTURE(i, taxels.links[i]);
            const auto* segments = lbs.taxel_segment_indices()->Get(i)->values();
            const auto* weights = lbs.taxel_segment_weights()->Get(i)->values();
            CHECK(static_cast<int>(ids->Get(segments->Get(0))) == joint_of_link.at(taxels.links[i]));
            CHECK(weights->Get(0) == 1.0f);
            for (unsigned k = 1; k < weights->size(); ++k)
                CHECK(weights->Get(k) == 0.0f);
        }
        const auto positions = shape.skin_positions(*bind);
        const auto normals = shape.current_normals(*bind);
        for (std::size_t i = 0; i < kLayoutTaxels; ++i)
        {
            CAPTURE(i);
            CHECK(norm(delta(positions[i], taxels.positions[i])) < kFloatPosition);
            CHECK(norm(delta(normals[i], taxels.normals[i])) < kFloatDirection);
        }
    }
}

TEST_CASE("the SDK taxels are the layout on rh_hand_model_fk, in the WRIST frame", "[revohuman][tactile][geometry]")
{
    // Compare in the corrected WRIST origin before checking rotation-invariant lengths and inner products.
    std::array<double, kGloveJointCount> angles{};
    for (std::size_t i = 0; i < angles.size(); ++i)
        angles[i] = .3 + .02 * static_cast<double>(i);
    for (const auto hand : { Hand::Left, Hand::Right })
    {
        CAPTURE(hand_name(hand));
        const RhHand sdk_hand = hand == Hand::Left ? RH_HAND_LEFT : RH_HAND_RIGHT;
        std::array<RhLinkPose, RH_JOINT_COUNT> links{};
        REQUIRE(rh_hand_model_fk(RH_HAND_MODEL_DV1, sdk_hand, angles.data(), nullptr, links.data()) == RH_OK);
        auto layout = std::make_unique<RhTaxelLayout>();
        layout->struct_size = sizeof(RhTaxelLayout);
        REQUIRE(rh_hand_model_taxel_layout(RH_HAND_MODEL_DV1, sdk_hand, kLayoutTaxels, layout.get()) == RH_OK);
        std::vector<Vec3> palm_p, palm_n;
        for (std::size_t i = 0; i < kLayoutTaxels; ++i)
        {
            const auto& t = layout->taxels[i];
            const auto& l = links[t.link];
            Vec3 p{}, n{};
            for (int r = 0; r < 3; ++r)
            {
                p[r] = l.p[r];
                for (int c = 0; c < 3; ++c)
                {
                    p[r] += l.r[3 * r + c] * t.p[c];
                    n[r] += l.r[3 * r + c] * t.n[c];
                }
            }
            p[0] -= .01259782883756545; // Reviewed wrist depth in palm_link coordinates.
            palm_p.push_back(p);
            palm_n.push_back(n);
        }
        const auto taxels = sdk_taxels(HandModel::Dv1, hand, angles);
        REQUIRE(taxels.positions.size() == kLayoutTaxels);
        const auto& p = taxels.positions;
        for (std::size_t i = 0; i < kLayoutTaxels; ++i)
        {
            CAPTURE(i);
            const auto j = (i + 1) % kLayoutTaxels, k = (i + 97) % kLayoutTaxels;
            CHECK(taxels.links[i] == layout->taxels[i].link);
            CHECK(std::abs(norm(taxels.normals[i]) - 1) < 1e-12);
            CHECK(std::abs(norm(p[i]) - norm(palm_p[i])) < 1e-12);
            CHECK(std::abs(dot(p[i], p[j]) - dot(palm_p[i], palm_p[j])) < 1e-12);
            CHECK(std::abs(dot(taxels.normals[i], p[i]) - dot(palm_n[i], palm_p[i])) < 1e-12);
            CHECK(std::abs(det(p[i], p[j], p[k]) - det(palm_p[i], palm_p[j], palm_p[k])) < 1e-15);
        }
    }
}

TEST_CASE("skinned taxels follow the SDK's when the fingers bend", "[revohuman][tactile][geometry]")
{
    // Taxels ride rigidly on the OpenXR joint of their link, and each of those joints is rigid on that
    // link: DIP, PIP and thumb joints, and the non-thumb proximal joints carried with the MCP flexion.
    const auto provider = make_hand_kinematics(HandModel::Dv1);
    for (const auto hand : { Hand::Left, Hand::Right })
    {
        CAPTURE(hand_name(hand));
        const LoadedShape shape(make_sdk_tactile_shape(HandModel::Dv1, hand).metadata);
        for (const double bend : { .35, .7, 1.2 })
        {
            CAPTURE(bend);
            std::array<double, kGloveJointCount> angles{};
            for (std::size_t slot = 0; slot < kGloveJointCount; ++slot)
            {
                CAPTURE(slot);
                angles.fill(0);
                angles[slot] = bend;
                const auto [position, normal] = skin_error(shape, *provider, hand, angles);
                CHECK(position < 1e-6);
                CHECK(normal < 1e-6);
            }
            angles.fill(bend);
            const auto [position, normal] = skin_error(shape, *provider, hand, angles);
            CHECK(position < 1e-6);
            CHECK(normal < 1e-6);
        }
    }
}
#else
TEST_CASE("tactile has no metadata without the SDK taxel layout", "[revohuman][tactile][geometry]")
{
    CHECK_THROWS_AS(make_sdk_tactile_shape(HandModel::Dv1, Hand::Left), std::runtime_error);
    CHECK_THROWS_AS(sdk_taxels(HandModel::Dv1, Hand::Left, {}), std::runtime_error);
}
#endif

#if REVOHUMAN_TEST_SDK_VARIANT_MODEL
TEST_CASE("the SDK gives each product's hand model, with no default", "[revohuman][geometry]")
{
    CHECK(sdk_has_product_hand_models());
    CHECK(hand_model_of_product("P1") == HandModel::P);
    CHECK(hand_model_of_product("PTV1") == HandModel::Dv1);
    for (const char* code : { "", "auto", "P9" })
    {
        CAPTURE(code);
        CHECK_THROWS_AS(hand_model_of_product(code), std::runtime_error);
    }
}
#else
TEST_CASE("no product has a hand model without the SDK's table", "[revohuman][geometry]")
{
    CHECK_FALSE(sdk_has_product_hand_models());
    CHECK_THROWS_AS(hand_model_of_product("PTV1"), std::runtime_error);
}
#endif

TEST_CASE("manifest declares every per-hand status output", "[revohuman][status]")
{
    std::ifstream file(REVOHUMAN_MANIFEST);
    REQUIRE(file.good());
    const std::string yaml{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
    const std::regex paths("path: \\\"([^\\\"]+)\\\"");
    std::set<std::string> actual;
    std::size_t count = 0;
    for (auto it = std::sregex_iterator(yaml.begin(), yaml.end(), paths); it != std::sregex_iterator(); ++it)
    {
        actual.insert((*it)[1]);
        ++count;
    }
    std::set<std::string> expected;
    for (const std::string side : { "left", "right" })
        for (const std::string prefix : { "/hand/", "/glove/revohuman_", "/tactile/revohuman_" })
            expected.insert(prefix + side);
    CHECK(actual == expected);
    CHECK(count == expected.size());
}
