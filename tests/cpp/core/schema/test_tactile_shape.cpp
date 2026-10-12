// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated TactileShape FlatBuffer message (internal only).

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <openxr/openxr.h>
#include <schema/serialized.hpp>
#include <schema/tactile_shape_generated.h>
#include <schema/timestamp_generated.h>

#include <memory>

// VT_<FIELD> == (id + 2) * 2.
#define VT(field) (field + 2) * 2
static_assert(core::TactileShape::VT_JOINT_COUNT == VT(0));
static_assert(core::TactileShape::VT_JOINT_BIND_POSES == VT(1));
static_assert(core::TactileShape::VT_JOINT_PARENTS == VT(2));
static_assert(core::TactileShape::VT_TAXEL_COUNT == VT(3));
static_assert(core::TactileShape::VT_TAXEL_POSITIONS == VT(4));
static_assert(core::TactileShape::VT_TAXEL_NORMALS == VT(5));
static_assert(core::TactileShape::VT_TAXEL_TANGENTS == VT(6));
static_assert(core::TactileShape::VT_TAXEL_JOINT_INDICES == VT(7));
static_assert(core::TactileShape::VT_TAXEL_JOINT_WEIGHTS == VT(8));
static_assert(core::TactileShape::VT_DEVICE_ID == VT(9));
static_assert(core::TactileShape::VT_SHAPE_VERSION == VT(10));
static_assert(core::TactileShapeRecord::VT_DATA == VT(0));
static_assert(core::TactileShapeRecord::VT_TIMESTAMP == VT(1));

// Producers copy OpenXR arrays straight into the taxel vectors.
static_assert(sizeof(core::Point) == sizeof(XrVector3f));
static_assert(sizeof(core::TaxelJointWeights) == sizeof(XrVector4f));
static_assert(sizeof(core::TaxelJointIndices) == 4 * sizeof(XrHandJointEXT));

namespace
{

constexpr uint32_t kRootParent = 0x7FFFFFFF;

// Two joints (wrist -> index tip) and one taxel skinned to both.
core::TactileShapeT make_shape()
{
    core::TactileShapeT native;
    native.joint_bind_poses = { core::Pose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 1)),
                                core::Pose(core::Point(0, 0.1f, 0), core::Quaternion(0, 0, 0, 1)) };
    native.joint_parents = { kRootParent, 0 };
    native.joint_count = 2;

    core::TaxelJointIndices indices;
    indices.mutable_joints()->Mutate(0, 1);
    indices.mutable_joints()->Mutate(1, 0);
    core::TaxelJointWeights weights;
    weights.mutable_weights()->Mutate(0, 0.75f);
    weights.mutable_weights()->Mutate(1, 0.25f);

    native.taxel_positions = { core::Point(0, 0.09f, 0.01f) };
    native.taxel_normals = { core::Point(0, 0, 1) };
    native.taxel_tangents = { core::Point(1, 0, 0) };
    native.taxel_joint_indices = { indices };
    native.taxel_joint_weights = { weights };
    native.taxel_count = 1;
    native.device_id = "glove_right";
    native.shape_version = 7;
    return native;
}

} // namespace

TEST_CASE("TactileShape round-trips skeleton, taxels and skinning", "[unit][tactile_shape]")
{
    const auto shape = core::pack<core::TactileShape>(make_shape());

    REQUIRE(shape);
    CHECK(shape->joint_count() == 2);
    CHECK(shape->joint_parents()->Get(0) == kRootParent);
    CHECK(shape->joint_parents()->Get(1) == 0);
    CHECK(shape->joint_bind_poses()->Get(1)->position().y() == 0.1f);
    CHECK(shape->taxel_count() == 1);
    CHECK(shape->taxel_normals()->Get(0)->z() == 1.0f);
    CHECK(shape->taxel_joint_indices()->Get(0)->joints()->Get(0) == 1);
    CHECK(shape->taxel_joint_weights()->Get(0)->weights()->Get(1) == 0.25f);
    CHECK(shape->device_id()->str() == "glove_right");
    CHECK(shape->shape_version() == 7);
}

TEST_CASE("TactileShapeRecord verifies and unpacks", "[unit][tactile_shape]")
{
    core::TactileShapeRecordT record;
    record.data = std::make_shared<core::TactileShapeT>(make_shape());
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileShapeRecord::Pack(builder, &record));

    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(core::VerifyTactileShapeRecordBuffer(verifier));

    core::TactileShapeRecordT unpacked;
    core::GetTactileShapeRecord(builder.GetBufferPointer())->UnPackTo(&unpacked);
    REQUIRE(unpacked.data);
    CHECK(unpacked.data->taxel_joint_weights.size() == 1);
    CHECK(unpacked.data->taxel_joint_weights[0].weights()->Get(0) == 0.75f);
    CHECK(unpacked.timestamp->sample_time_raw_device_clock() == 3);
}
