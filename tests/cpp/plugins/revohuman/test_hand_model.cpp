// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hand_points.hpp"

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
#include <utility>

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
    CHECK_NOTHROW(HandModel("PTV1"));
    CHECK_NOTHROW(HandModel("P1"));
    CHECK_THROWS_AS(HandModel("auto"), std::runtime_error);
}

TEST_CASE("hand points: the SDK names its points in SdkPoint order", "[unit]")
{
    const std::array<std::pair<SdkPoint, const char*>, RH_HAND_POINT_COUNT> expected = { {
        { kIndexDip, "index_DIP_Link" },   { kIndexPip, "index_PIP_Link" },   { kIndexMcp, "index_MCP_Link" },
        { kIndexMpr, "index_MPR_Link" },   { kMiddleDip, "middle_DIP_Link" }, { kMiddlePip, "middle_PIP_Link" },
        { kMiddleMcp, "middle_MCP_Link" }, { kMiddleMpr, "middle_MPR_Link" }, { kRingDip, "ring_DIP_Link" },
        { kRingPip, "ring_PIP_Link" },     { kRingMcp, "ring_MCP_Link" },     { kRingMpr, "ring_MPR_Link" },
        { kLittleDip, "little_DIP_Link" }, { kLittlePip, "little_PIP_Link" }, { kLittleMcp, "little_MCP_Link" },
        { kLittleMpr, "little_MPR_Link" }, { kThumbDip, "thumb_DIP_Link" },   { kThumbPip, "thumb_PIP_Link" },
        { kThumbMcp, "thumb_MCP_Link" },   { kThumbCmr, "thumb_CMR_Link" },   { kThumbCmp, "thumb_CMP_Link" },
        { kIndexTip, "index_TIP" },        { kMiddleTip, "middle_TIP" },      { kRingTip, "ring_TIP" },
        { kLittleTip, "little_TIP" },      { kThumbTip, "thumb_TIP" },        { kPalmLink, "palm_link" },
    } };
    for (const auto& [point, name] : expected)
    {
        char buf[64];
        REQUIRE(rh_hand_point_name(RH_HAND_LEFT, point, buf, sizeof(buf)) > 0);
        CHECK(std::string(buf) == std::string("left_") + name);
    }
}

TEST_CASE("hand model: WRIST is the origin and only mapped joints are valid", "[unit]")
{
    for (const char* product : { "PTV1", "P1" })
    {
        const auto poses = HandModel(product).joint_poses(Hand::Left, bent(), kAllJoints);
        CHECK(poses.position_valid_mask == kMappedHandMask);
        CHECK(poses.orientation_valid_mask == kMappedHandMask);
        for (double v : position(poses.joints[core::HandJoint_WRIST]))
            CHECK(v == Approx(0).margin(1e-12));
        CHECK(std::abs(orientation(poses.joints[core::HandJoint_WRIST])[3]) == Approx(1).margin(1e-12));
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
    const HandModel model("PTV1");
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
    // The index DIP moves the index distal and tip: they lose their orientation.
    const auto poses = HandModel("PTV1").joint_poses(Hand::Left, bent(), kAllJoints & ~(1u << kIndexDip));
    CHECK((poses.orientation_valid_mask & (1u << core::HandJoint_INDEX_DISTAL)) == 0);
    CHECK((poses.orientation_valid_mask & (1u << core::HandJoint_INDEX_TIP)) == 0);
    CHECK((poses.orientation_valid_mask & (1u << core::HandJoint_MIDDLE_PROXIMAL)) != 0);
}

TEST_CASE("hand model: TactileMetadata binds every taxel rigidly to one joint", "[unit]")
{
    const auto bytes = HandModel("PTV1").tactile_metadata(Hand::Right, 241);
    const auto* metadata = flatbuffers::GetRoot<core::TactileMetadata>(bytes.data());
    const auto* lbs = metadata->geometry_mapping_as_LBS();
    REQUIRE(lbs != nullptr);
    CHECK(lbs->segment_bind_poses()->size() == kHandJointCount);
    CHECK(lbs->taxel_bind_poses()->size() == 241);
    for (const auto* weights : *lbs->taxel_segment_weights())
        CHECK(weights->values()->Get(0) == 1.0f);

    CHECK_THROWS_AS(HandModel("PTV1").tactile_metadata(Hand::Right, 240), std::runtime_error);
    CHECK_THROWS_AS(HandModel("P1").tactile_metadata(Hand::Right, 241), std::runtime_error);
}

namespace
{
#include "ptv1_neutral_points.inc"

RhHandPoints neutral_points()
{
    RhHandPoints raw = RH_STRUCT_INIT(RhHandPoints);
    raw.points_version = RH_HAND_POINTS_VERSION;
    raw.hand = RH_HAND_LEFT;
    raw.position_valid_mask = raw.orientation_valid_mask = (1u << RH_HAND_POINT_COUNT) - 1u;
    for (std::size_t i = 0; i < kNeutralPoints.size(); ++i)
    {
        std::copy(kNeutralPoints[i].begin(), kNeutralPoints[i].end(), raw.points[i].p);
        raw.points[i].r[0] = raw.points[i].r[4] = raw.points[i].r[8] = 1;
    }
    return raw;
}
} // namespace

TEST_CASE("hand fit: raw point contract", "[unit]")
{
    auto raw = neutral_points();
    CHECK_THROWS_AS(map_hand_points(Hand::Right, raw), std::runtime_error);
    raw.points_version++;
    CHECK_THROWS_AS(map_hand_points(Hand::Left, raw), std::runtime_error);
}

TEST_CASE("hand fit: the wrist depth and the thumb's own fit are applied", "[unit]")
{
    const auto out = map_hand_points(Hand::Left, neutral_points());
    CHECK(out.position_valid_mask == kMappedHandMask);
    CHECK(out.orientation_valid_mask == kMappedHandMask);
    for (double x : position(out.joints[core::HandJoint_WRIST]))
        CHECK(x == Approx(0).margin(2e-8));
    CHECK(orientation(out.joints[core::HandJoint_WRIST])[3] == Approx(1).margin(2e-8));
    // Tracker origin with its own +X pad, then re-expressed relative to the shifted WRIST.
    const auto palm = position(out.joints[core::HandJoint_PALM]);
    CHECK(palm[0] == Approx(.0023415).margin(2e-8));
    CHECK(palm[1] == Approx(0).margin(2e-8));
    CHECK(palm[2] == Approx(-.08614770283).margin(2e-8));
    const auto thumb = position(out.joints[core::HandJoint_THUMB_METACARPAL]);
    CHECK(thumb[1] == Approx(-.01523045813683108).margin(2e-8));
    CHECK(thumb[0] == Approx(.0380000196899724).margin(2e-8));
    CHECK(thumb[2] == Approx(-.0403310513376026).margin(2e-8));
    for (core::HandJoint placeholder : { core::HandJoint_INDEX_METACARPAL, core::HandJoint_MIDDLE_METACARPAL,
                                         core::HandJoint_RING_METACARPAL, core::HandJoint_LITTLE_METACARPAL })
        CHECK(position(out.joints[placeholder]) == std::array<double, 3>{});
    const auto inspected = map_hand_points(Hand::Left, neutral_points(), true);
    CHECK(inspected.position_valid_mask == kMappedHandMask);
    CHECK(std::abs(position(inspected.joints[core::HandJoint_INDEX_METACARPAL])[2]) > .01);
}

TEST_CASE("hand fit: both hands mirror and rotating the tracker axes preserves the output", "[unit]")
{
    auto left = neutral_points(), right = left, moved = left;
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
    const auto a = map_hand_points(Hand::Left, left, true);
    const auto b = map_hand_points(Hand::Right, right, true);
    const auto c = map_hand_points(Hand::Left, moved, true);
    for (std::size_t i = 0; i < kHandJointCount; ++i)
        for (int k = 0; k < 3; ++k)
        {
            CHECK(position(b.joints[i])[k] == Approx(position(a.joints[i])[k] * (k == 0 ? -1 : 1)).margin(2e-8));
            CHECK(position(c.joints[i])[k] == Approx(position(a.joints[i])[k]).margin(2e-8));
        }
}

TEST_CASE("hand fit: SDK tip movement reaches the output and invalid poses stay invalid", "[unit]")
{
    auto raw = neutral_points();
    const auto before = map_hand_points(Hand::Left, raw);
    // Simulate a terminal joint moving its rigid tip: the fitter must not rebuild it along PIP to DIP.
    raw.points[kIndexTip].p[0] += .006;
    raw.points[kIndexTip].p[2] -= .004;
    auto after = map_hand_points(Hand::Left, raw);
    const auto tip_before = position(before.joints[core::HandJoint_INDEX_TIP]);
    const auto tip_after = position(after.joints[core::HandJoint_INDEX_TIP]);
    CHECK(tip_after[1] - tip_before[1] == Approx(-.006).margin(2e-8));
    CHECK(tip_after[2] - tip_before[2] == Approx(.004).margin(2e-8));
    raw.points[kIndexTip].p[0] = std::numeric_limits<double>::quiet_NaN();
    after = map_hand_points(Hand::Left, raw);
    CHECK((after.position_valid_mask & (1u << core::HandJoint_INDEX_TIP)) == 0);
    CHECK((after.position_valid_mask & (1u << core::HandJoint_MIDDLE_TIP)) != 0);
}

TEST_CASE("hand fit: proximal remains MCP-carried and thumb height follows CMR", "[unit]")
{
    auto raw = neutral_points();
    const auto before = map_hand_points(Hand::Left, raw);
    const double turn[9]{ 0, -1, 0, 1, 0, 0, 0, 0, 1 };
    std::copy_n(turn, 9, raw.points[kIndexMcp].r);
    std::copy_n(turn, 9, raw.points[kThumbCmr].r);
    const auto after = map_hand_points(Hand::Left, raw);
    const auto index_before = position(before.joints[core::HandJoint_INDEX_PROXIMAL]);
    const auto index_after = position(after.joints[core::HandJoint_INDEX_PROXIMAL]);
    CHECK(std::hypot(index_after[0] - index_before[0], index_after[1] - index_before[1]) > .01);
    const auto thumb_before = position(before.joints[core::HandJoint_THUMB_METACARPAL]);
    const auto thumb_after = position(after.joints[core::HandJoint_THUMB_METACARPAL]);
    CHECK(thumb_after[0] - thumb_before[0] == Approx(-.010).margin(2e-8));
    CHECK(thumb_after[1] - thumb_before[1] == Approx(.010).margin(2e-8));
    CHECK(thumb_after[2] == Approx(thumb_before[2]).margin(2e-8));
}
