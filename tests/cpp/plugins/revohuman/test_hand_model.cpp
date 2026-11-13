// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/hand_model.hpp>
#include <schema/tactile_metadata_generated.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <revohuman_fk.h>
#include <stdexcept>
#include <string>

using namespace plugins::revohuman;
using Catch::Approx;

namespace
{

constexpr std::uint32_t kAllJoints = (1u << kGloveJointCount) - 1u;

std::array<double, kGloveJointCount> bent()
{
    std::array<double, kGloveJointCount> q{};
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = 0.05 * static_cast<double>(i % 7);
    return q;
}

std::array<double, 3> position(const core::Pose& pose)
{
    return { pose.position().x(), pose.position().y(), pose.position().z() };
}

std::array<double, 4> orientation(const core::Pose& pose)
{
    return { pose.orientation().x(), pose.orientation().y(), pose.orientation().z(), pose.orientation().w() };
}

} // namespace

TEST_CASE("hand model: products the SDK knows map onto the fitted models", "[unit]")
{
    CHECK(HandModel::for_product("PTV1").id().find(":dv1:") != std::string::npos);
    CHECK(HandModel::for_product("P1").id().find(":p:") != std::string::npos);
    CHECK_THROWS_AS(HandModel::for_product("auto"), std::runtime_error);
}

TEST_CASE("hand model: WRIST is the origin and only mapped joints are valid", "[unit]")
{
    for (const char* product : { "PTV1", "P1" })
    {
        const auto poses = HandModel::for_product(product).joint_poses(Hand::Left, bent(), kAllJoints);
        CHECK(poses.position_valid_mask == kMappedHandMask);
        CHECK(poses.orientation_valid_mask == kMappedHandMask);
        for (double v : position(poses.joints[1]))
            CHECK(v == Approx(0).margin(1e-12));
        CHECK(std::abs(orientation(poses.joints[1])[3]) == Approx(1).margin(1e-12));
        for (const core::Pose& joint : poses.joints)
        {
            const auto p = position(joint);
            const auto q = orientation(joint);
            CHECK(std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) == Approx(1).margin(1e-6));
            CHECK(std::hypot(p[0], p[1], p[2]) < 0.25);
        }
    }
}

TEST_CASE("hand model: the right hand mirrors the left across OpenXR X", "[unit]")
{
    const auto model = HandModel::for_product("PTV1");
    const auto left = model.joint_poses(Hand::Left, bent(), kAllJoints);
    const auto right = model.joint_poses(Hand::Right, bent(), kAllJoints);
    for (std::size_t j = 0; j < kHandJointCount; ++j)
    {
        const auto l = position(left.joints[j]);
        const auto r = position(right.joints[j]);
        CHECK(r[0] == Approx(-l[0]).margin(1e-7));
        CHECK(r[1] == Approx(l[1]).margin(1e-7));
        CHECK(r[2] == Approx(l[2]).margin(1e-7));
    }
}

TEST_CASE("hand model: an invalid joint invalidates what it moves", "[unit]")
{
    // SDK joint 0 is the index DIP: the index distal (9) and tip (10) lose their orientation.
    const auto poses = HandModel::for_product("PTV1").joint_poses(Hand::Left, bent(), kAllJoints & ~1u);
    CHECK((poses.orientation_valid_mask & (1u << 9)) == 0);
    CHECK((poses.orientation_valid_mask & (1u << 10)) == 0);
    CHECK((poses.orientation_valid_mask & (1u << 12)) != 0);
}

TEST_CASE("hand model: TactileMetadata binds every taxel rigidly to one joint", "[unit]")
{
    const auto bytes = HandModel::for_product("PTV1").tactile_metadata(Hand::Right, 241);
    const auto* metadata = flatbuffers::GetRoot<core::TactileMetadata>(bytes.data());
    const auto* lbs = metadata->geometry_mapping_as_LBS();
    REQUIRE(lbs != nullptr);
    CHECK(lbs->segment_bind_poses()->size() == kHandJointCount);
    CHECK(lbs->taxel_bind_poses()->size() == 241);
    for (const auto* weights : *lbs->taxel_segment_weights())
        CHECK(weights->values()->Get(0) == 1.0f);

    CHECK_THROWS_AS(HandModel::for_product("PTV1").tactile_metadata(Hand::Right, 240), std::runtime_error);
    CHECK_THROWS_AS(HandModel::for_product("P1").tactile_metadata(Hand::Right, 241), std::runtime_error);
}

namespace
{
#include "new_geometry_fixture.inc"

// The URDF hash the SDK reports for a product's hand model, as the plugin reads it.
std::string model_sha(const char* product)
{
    RhHandModel model{};
    REQUIRE(rh_variant_hand_model(product, &model) == RH_OK);
    RhHandModelInfo info = RH_STRUCT_INIT(RhHandModelInfo);
    REQUIRE(rh_hand_model_info(model, &info) == RH_OK);
    return info.urdf_sha256;
}

RhHandPoints new_neutral()
{
    RhHandPoints raw = RH_STRUCT_INIT(RhHandPoints);
    raw.points_version = RH_HAND_POINTS_VERSION;
    raw.hand = RH_HAND_LEFT;
    raw.position_valid_mask = raw.orientation_valid_mask = (1u << RH_HAND_POINT_COUNT) - 1u;
    for (std::size_t i = 0; i < kNewNeutral.size(); ++i)
    {
        std::copy(kNewNeutral[i].begin(), kNewNeutral[i].end(), raw.points[i].p);
        raw.points[i].r[0] = raw.points[i].r[4] = raw.points[i].r[8] = 1;
    }
    return raw;
}
} // namespace

TEST_CASE("hand fit: model identity and raw point contract", "[unit]")
{
    auto raw = new_neutral();
    CHECK_THROWS_AS(fit_hand_points("unknown", Hand::Left, raw), std::runtime_error);
    CHECK_THROWS_AS(fit_hand_points(model_sha("PTV1"), Hand::Right, raw), std::runtime_error);
    raw.points_version++;
    CHECK_THROWS_AS(fit_hand_points(model_sha("PTV1"), Hand::Left, raw), std::runtime_error);
}

TEST_CASE("hand fit: the wrist depth and the thumb's own fit are applied", "[unit]")
{
    const auto raw = new_neutral();
    for (const std::string& hash : { model_sha("P1"), model_sha("PTV1") })
    {
        const auto out = fit_hand_points(hash, Hand::Left, raw);
        CHECK(out.position_valid_mask == kMappedHandMask);
        CHECK(out.orientation_valid_mask == kMappedHandMask);
        for (double x : position(out.joints[1]))
            CHECK(x == Approx(0).margin(2e-8));
        CHECK(orientation(out.joints[1])[3] == Approx(1).margin(2e-8));
        // Tracker origin with its own +X pad, then re-expressed relative to the shifted WRIST.
        CHECK(position(out.joints[0])[0] == Approx(.0023415).margin(2e-8));
        CHECK(position(out.joints[0])[1] == Approx(0).margin(2e-8));
        CHECK(position(out.joints[0])[2] == Approx(-.08614770283).margin(2e-8));
        CHECK(position(out.joints[2])[1] == Approx(-.01523045813683108).margin(2e-8));
        CHECK(position(out.joints[2])[0] == Approx(.0380000196899724).margin(2e-8));
        CHECK(position(out.joints[2])[2] == Approx(-.0403310513376026).margin(2e-8));
        for (int slot : { 6, 11, 16, 21 })
            CHECK(position(out.joints[slot]) == std::array<double, 3>{});
        const auto inspected = fit_hand_points(hash, Hand::Left, raw, true);
        CHECK(inspected.position_valid_mask == kMappedHandMask);
        CHECK(std::abs(position(inspected.joints[6])[2]) > .01);
    }
}

TEST_CASE("hand fit: both hands mirror and rotating the tracker axes preserves the output", "[unit]")
{
    auto left = new_neutral(), right = left, moved = left;
    right.hand = RH_HAND_RIGHT;
    for (int i = 0; i < RH_HAND_POINT_COUNT; ++i)
    {
        right.points[i].p[1] *= -1;
        // Tracker origin is implicit zero; rotate its axes without adding a world-space translation.
        moved.points[i].p[0] = -left.points[i].p[1];
        moved.points[i].p[1] = left.points[i].p[0];
        const double rotation[9]{ 0, -1, 0, 1, 0, 0, 0, 0, 1 };
        std::copy_n(rotation, 9, moved.points[i].r);
    }
    const auto a = fit_hand_points(model_sha("PTV1"), Hand::Left, left, true);
    const auto b = fit_hand_points(model_sha("PTV1"), Hand::Right, right, true);
    const auto c = fit_hand_points(model_sha("PTV1"), Hand::Left, moved, true);
    for (std::size_t i = 0; i < kHandJointCount; ++i)
        for (int k = 0; k < 3; ++k)
        {
            CHECK(position(b.joints[i])[k] == Approx(position(a.joints[i])[k] * (k == 0 ? -1 : 1)).margin(2e-8));
            CHECK(position(c.joints[i])[k] == Approx(position(a.joints[i])[k]).margin(2e-8));
        }
}

TEST_CASE("hand fit: SDK tip movement reaches the output and invalid poses stay invalid", "[unit]")
{
    auto raw = new_neutral();
    const auto before = fit_hand_points(model_sha("PTV1"), Hand::Left, raw);
    // Simulate a terminal joint moving its rigid tip: the fitter must not rebuild it along PIP to DIP.
    raw.points[21].p[0] += .006;
    raw.points[21].p[2] -= .004;
    auto after = fit_hand_points(model_sha("PTV1"), Hand::Left, raw);
    CHECK(position(after.joints[10])[1] - position(before.joints[10])[1] == Approx(-.006).margin(2e-8));
    CHECK(position(after.joints[10])[2] - position(before.joints[10])[2] == Approx(.004).margin(2e-8));
    raw.points[21].p[0] = std::numeric_limits<double>::quiet_NaN();
    after = fit_hand_points(model_sha("PTV1"), Hand::Left, raw);
    CHECK((after.position_valid_mask & (1u << 10)) == 0);
    CHECK((after.position_valid_mask & (1u << 15)) != 0);
}

TEST_CASE("hand fit: proximal remains MCP-carried and thumb height follows CMR", "[unit]")
{
    auto raw = new_neutral();
    const auto before = fit_hand_points(model_sha("PTV1"), Hand::Left, raw);
    const double turn[9]{ 0, -1, 0, 1, 0, 0, 0, 0, 1 };
    std::copy_n(turn, 9, raw.points[2].r);
    std::copy_n(turn, 9, raw.points[19].r);
    const auto after = fit_hand_points(model_sha("PTV1"), Hand::Left, raw);
    CHECK(std::hypot(position(after.joints[7])[0] - position(before.joints[7])[0],
                     position(after.joints[7])[1] - position(before.joints[7])[1]) > .01);
    CHECK(position(after.joints[2])[0] - position(before.joints[2])[0] == Approx(-.010).margin(2e-8));
    CHECK(position(after.joints[2])[1] - position(before.joints[2])[1] == Approx(.010).margin(2e-8));
    CHECK(position(after.joints[2])[2] == Approx(position(before.joints[2])[2]).margin(2e-8));
}

TEST_CASE("hand fit: SDK runtime and raw-pose fitting use the same model parameters", "[unit]")
{
    for (const char* product : { "P1", "PTV1" })
    {
        const auto model = HandModel::for_product(product);
        RhHandModel sdk_model{};
        REQUIRE(rh_variant_hand_model(product, &sdk_model) == RH_OK);
        RhHandModelInfo info = RH_STRUCT_INIT(RhHandModelInfo);
        REQUIRE(rh_hand_model_info(sdk_model, &info) == RH_OK);
        for (Hand hand : kHands)
        {
            for (const auto& q : { std::array<double, kGloveJointCount>{}, bent() })
            {
                RhHandPoints raw = RH_STRUCT_INIT(RhHandPoints);
                REQUIRE(rh_hand_points(sdk_model, hand == Hand::Right ? RH_HAND_RIGHT : RH_HAND_LEFT, q.data(),
                                       kAllJoints, &raw) == RH_OK);
                const auto expected = fit_hand_points(info.urdf_sha256, hand, raw);
                const auto actual = model.joint_poses(hand, q, kAllJoints);
                CHECK(actual.position_valid_mask == expected.position_valid_mask);
                CHECK(actual.orientation_valid_mask == expected.orientation_valid_mask);
                for (std::size_t i = 0; i < kHandJointCount; ++i)
                {
                    CHECK(position(actual.joints[i]) == position(expected.joints[i]));
                    CHECK(orientation(actual.joints[i]) == orientation(expected.joints[i]));
                }
            }
        }
    }
}
