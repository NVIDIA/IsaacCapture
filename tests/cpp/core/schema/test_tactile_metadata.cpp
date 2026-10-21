// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/serialized.hpp>
#include <schema/tactile_metadata_generated.h>

#include <array>
#include <cstdint>
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
