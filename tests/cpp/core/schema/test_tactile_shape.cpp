// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated TactileShape FlatBuffer message.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <openxr/openxr.h>
#include <schema/serialized.hpp>
#include <schema/tactile_shape_generated.h>

#include <array>
#include <cstdint>
#include <string>

// VT_<FIELD> == (id + 2) * 2.
#define VT(field) (field + 2) * 2
static_assert(core::TactileShape::VT_JOINTS == VT(0));
static_assert(core::TactileShape::VT_TAXELS == VT(1));

// Producers copy OpenXR values straight into the structs.
static_assert(sizeof(core::Point) == sizeof(XrVector3f));
static_assert(sizeof(core::TactileJoint) == sizeof(XrPosef) + sizeof(XrHandJointEXT));
static_assert(sizeof(core::TactileTaxel) == 3 * sizeof(XrVector3f) + 4 * sizeof(XrHandJointEXT) + sizeof(XrVector4f));

namespace
{

constexpr uint32_t kRootParent = 0x7FFFFFFF;

// Two joints (wrist -> index tip) and one taxel skinned to both.
core::TactileShapeT make_shape()
{
    core::TactileShapeT native;
    native.joints = { core::TactileJoint(core::Pose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 1)), kRootParent),
                      core::TactileJoint(core::Pose(core::Point(0, 0.1f, 0), core::Quaternion(0, 0, 0, 1)), 0) };

    const std::array<uint32_t, 4> indices = { 1, 0, 0, 0 };
    const std::array<float, 4> weights = { 0.75f, 0.25f, 0.0f, 0.0f };
    native.taxels = { core::TactileTaxel(core::Point(0, 0.09f, 0.01f), core::Point(0, 0, 1), core::Point(1, 0, 0),
                                         ::flatbuffers::span<const uint32_t, 4>(indices.data(), 4),
                                         ::flatbuffers::span<const float, 4>(weights.data(), 4)) };
    return native;
}

} // namespace

TEST_CASE("TactileShape round-trips skeleton, taxels and skinning", "[unit][tactile_shape]")
{
    const auto shape = core::pack<core::TactileShape>(make_shape());

    REQUIRE(shape);
    REQUIRE(shape->joints()->size() == 2);
    CHECK(shape->joints()->Get(0)->parent() == kRootParent);
    CHECK(shape->joints()->Get(1)->parent() == 0);
    CHECK(shape->joints()->Get(1)->bind_pose().position().y() == 0.1f);
    REQUIRE(shape->taxels()->size() == 1);
    const auto* taxel = shape->taxels()->Get(0);
    CHECK(taxel->normal().z() == 1.0f);
    CHECK(taxel->joint_indices()->Get(0) == 1);
    CHECK(taxel->joint_weights()->Get(1) == 0.25f);
}

TEST_CASE("TactileShape is its own root and verifies", "[unit][tactile_shape]")
{
    const auto shape = core::pack<core::TactileShape>(make_shape());
    const auto bytes = shape.buffer();

    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(core::VerifyTactileShapeBuffer(verifier));
    CHECK(std::string(core::TactileShape::GetFullyQualifiedName()) == "core.TactileShape");

    core::TactileShapeT unpacked;
    core::GetTactileShape(bytes.data())->UnPackTo(&unpacked);
    REQUIRE(unpacked.taxels.size() == 1);
    CHECK(unpacked.taxels[0].joint_weights()->Get(0) == 0.75f);
}
