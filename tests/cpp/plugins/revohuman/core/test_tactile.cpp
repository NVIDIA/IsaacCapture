// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_shapes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <flatbuffers/flatbuffers.h>
#include <revohuman_plugin/pose_math.hpp>
#include <revohuman_plugin/sha256.hpp>
#include <revohuman_plugin/tactile.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace plugins::revohuman;
using Catch::Approx;
using revohuman_test::rigid_shape;

namespace
{
std::string sha_of(const std::vector<std::uint8_t>& bytes)
{
    return sha256_hex(bytes.data(), bytes.size());
}

std::vector<std::string> problems(const ShapeSpec& spec)
{
    const auto bytes = build_shape_buffer(spec);
    return validate_shape(*flatbuffers::GetRoot<core::TactileMetadata>(bytes.data()));
}

LoadedShape load(const ShapeSpec& spec)
{
    return LoadedShape(build_shape_buffer(spec));
}

HandKinematics bind_kinematics(const ShapeSpec& spec)
{
    HandKinematics kin;
    for (std::size_t j = 0; j < spec.joint_ids.size(); ++j)
        kin.joints[spec.joint_ids[j]] = spec.bind_poses[j];
    return kin;
}
} // namespace

TEST_CASE("rigid metadata fits the tracker cap for every glove layout", "[revohuman][tactile]")
{
    for (std::size_t n : { 1u, 24u, 211u, 241u, 300u, 350u })
    {
        const auto spec = rigid_shape(n);
        const auto loaded = load(spec);
        INFO("taxels " << n << " bytes " << loaded.bytes().size());
        REQUIRE(problems(spec).empty());
        REQUIRE(loaded.taxel_count() == n);
        REQUIRE(loaded.bytes().size() <= shape_size_bound(kHandJointCount, n));
        REQUIRE(loaded.bytes().size() <= kShapeMaxBytes);
    }
    REQUIRE_THROWS(load(rigid_shape(kMaxTaxels + 1)));
}

TEST_CASE("metadata loading names the buffer by its sha256 and verifies it", "[revohuman][tactile]")
{
    const auto bytes = build_shape_buffer(rigid_shape(24));
    REQUIRE(LoadedShape(bytes).sha256() == sha_of(bytes));
    auto corrupt = bytes;
    corrupt[3] = 0x7F;
    REQUIRE_THROWS(LoadedShape(corrupt));
    auto oversized = bytes;
    oversized.resize(kShapeMaxBytes + 1);
    REQUIRE_THROWS(LoadedShape(oversized));
}

TEST_CASE("metadata rejects invalid poses, references and weights", "[revohuman][tactile]")
{
    auto s = rigid_shape(24);
    SECTION("non-unit taxel orientation")
    {
        s.taxel_bind_poses[3].q = { 0, 0, 0, 2 };
    }
    SECTION("NaN taxel orientation")
    {
        s.taxel_bind_poses[3].q[0] = std::numeric_limits<double>::quiet_NaN();
    }
    SECTION("non-finite taxel position")
    {
        s.taxel_bind_poses[0].p[1] = std::numeric_limits<double>::infinity();
    }
    SECTION("non-unit segment orientation")
    {
        s.bind_poses[4].q = { 0, 0, 0, 2 };
    }
    SECTION("weights do not sum to one")
    {
        s.joint_weights[5] = { .7f, .2f, 0, 0 };
    }
    SECTION("negative weight")
    {
        s.joint_weights[5] = { 1.5f, -.5f, 0, 0 };
    }
    SECTION("non-finite weight")
    {
        s.joint_weights[5][0] = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION("zero weights")
    {
        s.joint_weights[5] = { 0, 0, 0, 0 };
    }
    SECTION("invalid segment index in weighted slot")
    {
        s.joint_indices[5][0] = 26;
    }
    SECTION("invalid segment index in zero-weight slot")
    {
        s.joint_indices[5][1] = 26;
    }
    SECTION("duplicate joint identifier")
    {
        s.joint_ids[3] = s.joint_ids[0];
    }
    SECTION("joint count sentinel is not a joint")
    {
        s.joint_ids[3] = core::HandJoint_NUM_JOINTS;
    }
    SECTION("different segment and joint-id lengths")
    {
        s.joint_ids.pop_back();
    }
    SECTION("different taxel vector lengths")
    {
        s.taxel_bind_poses.pop_back();
    }
    SECTION("empty layout")
    {
        s = {};
    }
    REQUIRE_FALSE(problems(s).empty());
    REQUIRE_THROWS(load(s));
}

TEST_CASE("skinning at bind returns taxel origins and outward Z axes", "[revohuman][tactile]")
{
    const auto spec = rigid_shape(241);
    const auto shape = load(spec);
    const auto kin = bind_kinematics(spec);
    const auto pos = shape.skin_positions(kin);
    const auto nrm = shape.current_normals(kin);
    for (std::size_t i = 0; i < pos.size(); ++i)
    {
        const auto normal = quat_rotate(spec.taxel_bind_poses[i].q, { 0, 0, 1 });
        for (int c = 0; c < 3; ++c)
        {
            REQUIRE(pos[i][c] == Approx(spec.taxel_bind_poses[i].p[c]).margin(1e-6));
            REQUIRE(nrm[i][c] == Approx(normal[c]).margin(1e-6));
        }
    }
}

TEST_CASE("skinning follows a rigid motion of the hand", "[revohuman][tactile]")
{
    const auto spec = rigid_shape(211);
    const auto shape = load(spec);
    const Pose motion{ { .1, -.2, .05 }, revohuman_test::axis_angle({ .3, 1, -.4 }, .9) };
    auto kin = bind_kinematics(spec);
    for (auto& joint : kin.joints)
        joint = compose(motion, joint);
    const auto pos = shape.skin_positions(kin);
    const auto nrm = shape.current_normals(kin);
    for (std::size_t i = 0; i < pos.size(); ++i)
    {
        const auto p = transform_point(motion, spec.taxel_bind_poses[i].p);
        const auto n = quat_rotate(quat_mul(motion.q, spec.taxel_bind_poses[i].q), { 0, 0, 1 });
        for (int c = 0; c < 3; ++c)
        {
            REQUIRE(pos[i][c] == Approx(p[c]).margin(1e-6));
            REQUIRE(nrm[i][c] == Approx(n[c]).margin(1e-6));
        }
    }
}

TEST_CASE("subset segment slots resolve through reordered joint_ids", "[revohuman][tactile]")
{
    ShapeSpec spec;
    spec.joint_ids = { core::HandJoint_INDEX_TIP, core::HandJoint_WRIST };
    spec.bind_poses.resize(2);
    spec.taxel_bind_poses = { Pose{ { .01, .02, .03 }, { 0, 0, 0, 1 } } };
    spec.joint_indices = { { 0, 1, 1, 1 } };
    spec.joint_weights = { { 1, 0, 0, 0 } };
    const auto loaded = load(spec);
    auto kin = bind_kinematics(spec);
    kin.joints[0].p = { 9, 9, 9 }; // Segment 0 does not mean OpenXR PALM.
    kin.joints[core::HandJoint_INDEX_TIP].p = { .1, .2, .3 };
    kin.joints[core::HandJoint_INDEX_TIP].q = revohuman_test::axis_angle({ 1, 0, 0 }, 1.0);
    kin.orientation_valid_mask = 1u << core::HandJoint_INDEX_TIP;
    REQUIRE(loaded.has_valid_orientations(kin));
    const auto p = transform_point(kin.joints[core::HandJoint_INDEX_TIP], spec.taxel_bind_poses[0].p);
    const auto n = quat_rotate(kin.joints[core::HandJoint_INDEX_TIP].q, { 0, 0, 1 });
    for (int c = 0; c < 3; ++c)
    {
        REQUIRE(loaded.skin_positions(kin)[0][c] == Approx(p[c]).margin(1e-6));
        REQUIRE(loaded.current_normals(kin)[0][c] == Approx(n[c]).margin(1e-6));
    }
    kin.orientation_valid_mask = 1;
    REQUIRE_FALSE(loaded.has_valid_orientations(kin));
}

TEST_CASE("normal force is the SDK reading on positive local Z", "[revohuman][tactile]")
{
    for (std::uint32_t value = 0; value <= 0xFFFF; ++value)
    {
        const auto forces = taxel_forces({ static_cast<std::uint16_t>(value) });
        REQUIRE(forces[0][0] == 0);
        REQUIRE(forces[0][1] == 0);
        REQUIRE(forces[0][2] == static_cast<float>(value));
        REQUIRE(force_to_counts(forces[0]) == static_cast<long>(value));
    }
    REQUIRE(taxel_forces({}).empty());
}

TEST_CASE("only weighted joint orientations gate the layout", "[revohuman][tactile]")
{
    const auto spec = rigid_shape(1);
    const auto shape = load(spec);
    auto kin = bind_kinematics(spec);
    REQUIRE_FALSE(shape.has_valid_orientations(kin));
    const auto joint = spec.joint_ids[spec.joint_indices[0][0]];
    kin.orientation_valid_mask = 1u << joint;
    REQUIRE(shape.has_valid_orientations(kin));
    kin.joints[0].q = { 0, 0, 0, 0 };
    REQUIRE(shape.has_valid_orientations(kin));
    kin.joints[joint].q[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(shape.has_valid_orientations(kin));
}

TEST_CASE("loader refuses a blended taxel the schema permits", "[revohuman][tactile]")
{
    auto spec = rigid_shape(24);
    spec.joint_indices[5] = { 7, 8, 0, 0 };
    spec.joint_weights[5] = { .75f, .25f, 0, 0 };
    REQUIRE(problems(spec).empty());
    REQUIRE_THROWS_WITH(load(spec), Catch::Matchers::ContainsSubstring("only rigid binding"));
}
