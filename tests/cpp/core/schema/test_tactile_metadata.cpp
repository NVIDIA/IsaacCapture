// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/metadata_validation.hpp>
#include <schema/serialized.hpp>
#include <schema/tactile_metadata_generated.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#define VT(field) (field + 2) * 2
static_assert(core::TactileMetadata::VT_GEOMETRY_MAPPING_TYPE == VT(0));
static_assert(core::TactileMetadata::VT_GEOMETRY_MAPPING == VT(1));
static_assert(core::TactileMetadata::VT_SEGMENT_SOURCE_TYPE == VT(2));
static_assert(core::TactileMetadata::VT_SEGMENT_SOURCE == VT(3));
static_assert(core::HandSkeletonRef::VT_JOINT_IDS == VT(0));
static_assert(core::LBS::VT_SEGMENT_BIND_POSES == VT(0));
static_assert(core::LBS::VT_TAXEL_BIND_POSES == VT(1));
static_assert(core::LBS::VT_TAXEL_SEGMENT_INDICES == VT(2));
static_assert(core::LBS::VT_TAXEL_SEGMENT_WEIGHTS == VT(3));
static_assert(sizeof(core::UInt4) == 4 * sizeof(uint32_t));
static_assert(sizeof(core::Float4) == 4 * sizeof(float));

namespace
{

core::TactileMetadataT make_metadata()
{
    core::HandSkeletonRefT segment_source;
    segment_source.joint_ids = { core::HandJoint_WRIST, core::HandJoint_INDEX_TIP };
    core::LBST lbs;
    lbs.segment_bind_poses = { core::Pose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 1)),
                               core::Pose(core::Point(0, 0.1f, 0), core::Quaternion(0, 0, 0, 1)) };
    lbs.taxel_bind_poses = { core::Pose(core::Point(0, 0.09f, 0.01f), core::Quaternion(1, 0, 0, 0)) };
    const std::array<uint32_t, 4> indices = { 1, 0, 0, 0 };
    const std::array<float, 4> weights = { 0.75f, 0.25f, 0.0f, 0.0f };
    lbs.taxel_segment_indices.emplace_back(flatbuffers::span<const uint32_t, 4>(indices.data(), 4));
    lbs.taxel_segment_weights.emplace_back(flatbuffers::span<const float, 4>(weights.data(), 4));
    core::TactileMetadataT native;
    native.geometry_mapping.Set(std::move(lbs));
    native.segment_source.Set(std::move(segment_source));
    return native;
}

} // namespace

TEST_CASE("TactileMetadata round-trips union tags and segment-local indices", "[unit][tactile_metadata]")
{
    const auto metadata = core::pack<core::TactileMetadata>(make_metadata());
    CHECK(metadata->geometry_mapping_type() == core::AlgorithmSource_LBS);
    CHECK(metadata->segment_source_type() == core::SegmentSource_HandSkeletonRef);
    const auto* segment_source = metadata->segment_source_as_HandSkeletonRef();
    const auto* lbs = metadata->geometry_mapping_as_LBS();
    REQUIRE(segment_source);
    REQUIRE(lbs);
    REQUIRE(segment_source->joint_ids()->size() == 2);
    CHECK(segment_source->joint_ids()->Get(1) == core::HandJoint_INDEX_TIP);
    CHECK(lbs->taxel_segment_indices()->Get(0)->values()->Get(0) == 1);
    CHECK(lbs->taxel_segment_weights()->Get(0)->values()->Get(1) == 0.25f);
    CHECK(lbs->segment_bind_poses()->Get(1)->position().y() == 0.1f);
    CHECK(lbs->taxel_bind_poses()->Get(0)->orientation().x() == 1.0f);
}

TEST_CASE("TactileMetadata verifies and retains nested union views", "[unit][tactile_metadata]")
{
    auto metadata = core::pack<core::TactileMetadata>(make_metadata());
    const auto bytes = metadata.buffer();
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK(std::string(core::TactileMetadata::GetFullyQualifiedName()) == "core.TactileMetadata");
    auto lbs = metadata.narrow(metadata->geometry_mapping_as_LBS());
    metadata.reset();
    REQUIRE(lbs);
    CHECK(lbs->taxel_segment_weights()->Get(0)->values()->Get(0) == 0.75f);
}

TEST_CASE("TactileMetadata semantic verification accepts valid metadata", "[unit][tactile_metadata]")
{
    const auto metadata = core::pack<core::TactileMetadata>(make_metadata());
    CHECK_NOTHROW(core::verify_metadata(*metadata));
    CHECK_NOTHROW(core::verify_metadata<core::TactileMetadata>(*metadata));
    CHECK_NOTHROW(core::verify_metadata(*metadata, "tactile/right"));
    CHECK_NOTHROW(core::verify_metadata<core::TactileMetadata>(*metadata, "tactile/right"));
}

TEST_CASE("TactileMetadata semantic verification rejects inconsistent lengths", "[unit][tactile_metadata]")
{
    auto native = make_metadata();
    auto& lbs = *native.geometry_mapping.AsLBS();
    std::string reason;
    SECTION("taxel bind poses")
    {
        lbs.taxel_bind_poses.push_back(lbs.taxel_bind_poses.front());
        reason = "taxel_segment_indices: expected 2 rows to match taxel_bind_poses, got 1";
    }
    SECTION("taxel segment indices")
    {
        lbs.taxel_segment_indices.clear();
        reason = "taxel_segment_indices: expected 1 rows to match taxel_bind_poses, got 0";
    }
    SECTION("taxel segment weights")
    {
        lbs.taxel_segment_weights.clear();
        reason = "taxel_segment_weights: expected 1 rows to match taxel_bind_poses, got 0";
    }
    SECTION("joint identifiers and segment bind poses")
    {
        native.segment_source.AsHandSkeletonRef()->joint_ids.pop_back();
        reason = "segment_bind_poses: expected 1 poses to match segment_source.joint_ids, got 2";
    }
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK_THROWS_WITH(core::verify_metadata(*metadata), "TactileMetadata." + reason);
    CHECK_THROWS_WITH(core::verify_metadata(*metadata, "tactile/right"), "tactile/right: TactileMetadata." + reason);
}

TEST_CASE("TactileMetadata semantic verification rejects invalid or repeated joints", "[unit][tactile_metadata]")
{
    auto native = make_metadata();
    auto& joint_ids = native.segment_source.AsHandSkeletonRef()->joint_ids;
    std::string reason = "invalid HandJoint identifier";
    SECTION("sentinel")
    {
        joint_ids[1] = core::HandJoint_NUM_JOINTS;
    }
    SECTION("unknown enum value")
    {
        joint_ids[1] = static_cast<core::HandJoint>(255);
    }
    SECTION("duplicate identifier")
    {
        joint_ids[1] = joint_ids[0];
        reason = "duplicate HandJoint identifier";
    }
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK_THROWS_WITH(core::verify_metadata(*metadata),
                      Catch::Matchers::ContainsSubstring("TactileMetadata.joint_ids[1]: " + reason));
}

TEST_CASE("TactileMetadata semantic verification rejects out-of-range segment indices", "[unit][tactile_metadata]")
{
    auto native = make_metadata();
    auto& lbs = *native.geometry_mapping.AsLBS();
    std::array<uint32_t, 4> indices = { 1, 0, 0, 0 };
    std::string slot;
    SECTION("nonzero-weight slot")
    {
        indices[0] = 2;
        slot = "0";
    }
    SECTION("zero-weight slot")
    {
        indices[3] = 2;
        slot = "3";
    }
    lbs.taxel_segment_indices[0] = core::UInt4(flatbuffers::span<const uint32_t, 4>(indices.data(), 4));
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK_THROWS_WITH(core::verify_metadata(*metadata), "TactileMetadata.taxel_segment_indices[0][" + slot +
                                                            "]: segment index 2 is outside segment_bind_poses (size 2)");
}

TEST_CASE("TactileMetadata semantic verification checks each taxel's weights", "[unit][tactile_metadata]")
{
    auto native = make_metadata();
    auto& lbs = *native.geometry_mapping.AsLBS();
    std::array<float, 4> weights = { 0.75f, 0.25f, 0.0f, 0.0f };
    bool valid = false;
    std::string reason = "taxel_segment_weights[1]: weights must sum to one";
    SECTION("negative weight with sum one")
    {
        weights = { 1.25f, -0.25f, 0.0f, 0.0f };
        reason = "taxel_segment_weights[1][1]: weights must be finite and nonnegative";
    }
    SECTION("sum below one")
    {
        weights = { 0.25f, 0.25f, 0.0f, 0.0f };
    }
    SECTION("sum above one")
    {
        weights = { 0.75f, 0.75f, 0.0f, 0.0f };
    }
    SECTION("NaN")
    {
        weights[3] = std::numeric_limits<float>::quiet_NaN();
        reason = "taxel_segment_weights[1][3]: weights must be finite and nonnegative";
    }
    SECTION("infinity")
    {
        weights[3] = std::numeric_limits<float>::infinity();
        reason = "taxel_segment_weights[1][3]: weights must be finite and nonnegative";
    }
    SECTION("rounding within tolerance")
    {
        weights = { 0.75f, 0.250005f, 0.0f, 0.0f };
        valid = true;
    }
    SECTION("rounding outside tolerance")
    {
        weights = { 0.75f, 0.25002f, 0.0f, 0.0f };
    }
    // Place the changed weights after a valid taxel to exercise the whole vector.
    lbs.taxel_bind_poses.push_back(lbs.taxel_bind_poses.front());
    lbs.taxel_segment_indices.push_back(lbs.taxel_segment_indices.front());
    lbs.taxel_segment_weights.emplace_back(flatbuffers::span<const float, 4>(weights.data(), 4));
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    if (valid)
    {
        CHECK_NOTHROW(core::verify_metadata(*metadata));
        CHECK_NOTHROW(core::verify_metadata<core::TactileMetadata>(*metadata));
    }
    else
    {
        CHECK_THROWS_WITH(
            core::verify_metadata(*metadata), Catch::Matchers::ContainsSubstring("TactileMetadata." + reason));
        CHECK_THROWS_WITH(core::verify_metadata<core::TactileMetadata>(*metadata),
                          Catch::Matchers::ContainsSubstring("TactileMetadata." + reason));
    }
}

TEST_CASE("TactileMetadata semantic verification rejects nonfinite bind poses", "[unit][tactile_metadata]")
{
    const bool segment = GENERATE(true, false);
    const auto component = GENERATE(0, 1, 2, 3, 4, 5, 6);
    const auto value = GENERATE(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity());
    const std::string field = segment ? "segment_bind_poses" : "taxel_bind_poses";
    CAPTURE(field, component, value);
    auto native = make_metadata();
    auto& lbs = *native.geometry_mapping.AsLBS();
    auto& poses = segment ? lbs.segment_bind_poses : lbs.taxel_bind_poses;
    std::array<float, 7> row = { 0, 0, 0, 0, 0, 0, 1 };
    row[component] = value;
    poses.back() = core::Pose(core::Point(row[0], row[1], row[2]), core::Quaternion(row[3], row[4], row[5], row[6]));
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK_THROWS_WITH(
        core::verify_metadata(*metadata),
        Catch::Matchers::ContainsSubstring(field + "[" + std::to_string(poses.size() - 1) + "]: poses must be finite"));
}

TEST_CASE("TactileMetadata semantic verification checks bind quaternion norms", "[unit][tactile_metadata]")
{
    const bool segment = GENERATE(true, false);
    const std::string field = segment ? "segment_bind_poses" : "taxel_bind_poses";
    CAPTURE(field);
    auto native = make_metadata();
    auto& lbs = *native.geometry_mapping.AsLBS();
    auto& poses = segment ? lbs.segment_bind_poses : lbs.taxel_bind_poses;
    core::Quaternion orientation;
    bool valid = false;
    SECTION("zero quaternion")
    {
    }
    SECTION("scaled quaternion")
    {
        orientation = core::Quaternion(0, 0, 0, 2);
    }
    SECTION("finite components with overflowing squared norm")
    {
        orientation = core::Quaternion(std::numeric_limits<float>::max(), 0, 0, 0);
    }
    SECTION("rotation with all components nonzero")
    {
        orientation = core::Quaternion(0.5f, 0.5f, 0.5f, 0.5f);
        valid = true;
    }
    SECTION("negative unit quaternion")
    {
        orientation = core::Quaternion(0, 0, 0, -1);
        valid = true;
    }
    SECTION("rounding within squared-norm tolerance")
    {
        orientation = core::Quaternion(0, 0, 0, std::sqrt(1.0f + 0.5e-4f));
        valid = true;
    }
    SECTION("rounding outside squared-norm tolerance")
    {
        orientation = core::Quaternion(0, 0, 0, std::sqrt(1.0f + 2e-4f));
    }
    poses.back() = core::Pose(poses.back().position(), orientation);
    const auto metadata = core::pack<core::TactileMetadata>(native);
    flatbuffers::Verifier verifier(metadata.buffer().data(), metadata.buffer().size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    if (valid)
    {
        CHECK_NOTHROW(core::verify_metadata(*metadata));
    }
    else
    {
        CHECK_THROWS_WITH(core::verify_metadata(*metadata),
                          Catch::Matchers::ContainsSubstring(field + "[" + std::to_string(poses.size() - 1) +
                                                             "]: orientations must be unit quaternions"));
    }
}
