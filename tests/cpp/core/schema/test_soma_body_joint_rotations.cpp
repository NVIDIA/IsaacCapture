// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_body_joint_rotations_generated.h>
#include <schema/timestamp_generated.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#define VT(field) (field + 2) * 2
static_assert(core::SomaBodyJointRotations::VT_JOINT_ROTATIONS == VT(0));
static_assert(core::SomaBodyJointRotations::VT_GLOBAL_TRANSLATION == VT(1));
static_assert(core::SomaBodyJointRotations::VT_GLOBAL_TRANSLATION_IS_VALID == VT(2));
static_assert(core::SomaBodyJointRotationsRecord::VT_DATA == VT(0));
static_assert(core::SomaBodyJointRotationsRecord::VT_TIMESTAMP == VT(1));

#define TYPE(field) decltype(std::declval<core::SomaBodyJointRotations>().field())
static_assert(std::is_same_v<TYPE(joint_rotations), const core::SomaBodyJointRotationArray*>);
static_assert(std::is_same_v<TYPE(global_translation), const core::Point*>);
static_assert(std::is_same_v<TYPE(global_translation_is_valid), bool>);

static_assert(std::is_trivially_copyable_v<core::SomaJointRotation>);
static_assert(std::is_trivially_copyable_v<core::SomaBodyJointRotationArray>);
static_assert(sizeof(core::SomaBodyJointRotationArray) == 77 * sizeof(core::SomaJointRotation));

static_assert(core::SomaBodyJoint_HIPS == 0);
static_assert(core::SomaBodyJoint_LEFT_SHOULDER == 11);
static_assert(core::SomaBodyJoint_RIGHT_SHOULDER == 39);
static_assert(core::SomaBodyJoint_LEFT_LEG == 67);
static_assert(core::SomaBodyJoint_RIGHT_TOE_END == 76);
static_assert(core::SomaBodyJoint_NUM_JOINTS == 77);

TEST_CASE("SOMA body rotation array has the public layout", "[soma_body_joint_rotations][struct]")
{
    core::SomaBodyJointRotationArray rotations;

    REQUIRE(rotations.values()->size() == static_cast<size_t>(core::SomaBodyJoint_NUM_JOINTS));
    CHECK((*rotations.values())[core::SomaBodyJoint_HIPS]->rotation().x() == 0.0f);
    CHECK_FALSE((*rotations.values())[core::SomaBodyJoint_RIGHT_TOE_END]->is_valid());

    const core::SomaJointRotation hips(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true);
    rotations.mutable_values()->Mutate(core::SomaBodyJoint_HIPS, hips);

    CHECK((*rotations.values())[core::SomaBodyJoint_HIPS]->rotation().z() == Catch::Approx(0.6f));
    CHECK((*rotations.values())[core::SomaBodyJoint_HIPS]->rotation().w() == Catch::Approx(0.8f));
    CHECK((*rotations.values())[core::SomaBodyJoint_HIPS]->is_valid());
}

TEST_CASE("SOMA body joint rotations round trip through FlatBuffers", "[soma_body_joint_rotations][flatbuffers]")
{
    core::SomaBodyJointRotationsT pose;
    pose.joint_rotations = std::make_shared<core::SomaBodyJointRotationArray>();
    pose.joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJoint_HEAD, core::SomaJointRotation(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true));
    pose.global_translation = std::make_shared<core::Point>(1.0f, 2.0f, 3.0f);
    pose.global_translation_is_valid = true;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointRotations::Pack(builder, &pose));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointRotations>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_rotations() != nullptr);
    REQUIRE(decoded->global_translation() != nullptr);
    const auto* head = (*decoded->joint_rotations()->values())[core::SomaBodyJoint_HEAD];
    CHECK(head->rotation().z() == Catch::Approx(0.6f));
    CHECK(head->rotation().w() == Catch::Approx(0.8f));
    CHECK(head->is_valid());
    CHECK(decoded->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->global_translation_is_valid());
}

TEST_CASE("SOMA body record preserves quaternion components and validity", "[soma_body_joint_rotations][flatbuffers]")
{
    const float scale = 1.0f / std::sqrt(30.0f);
    core::SomaBodyJointRotationsRecordT record;
    record.data = std::make_shared<core::SomaBodyJointRotationsT>();
    record.data->joint_rotations = std::make_shared<core::SomaBodyJointRotationArray>();
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJoint_HIPS,
        core::SomaJointRotation(core::Quaternion(scale, -2.0f * scale, 3.0f * scale, -4.0f * scale), true));
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJoint_RIGHT_TOE_END,
        core::SomaJointRotation(core::Quaternion(-scale, 2.0f * scale, -3.0f * scale, 4.0f * scale), true));
    record.data->global_translation = std::make_shared<core::Point>(1.0f, -2.0f, 3.0f);
    record.data->global_translation_is_valid = false;
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(100, 200, 300);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointRotationsRecord::Pack(builder, &record));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaBodyJointRotationsRecord>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointRotationsRecord>(builder.GetBufferPointer());
    REQUIRE(decoded->data() != nullptr);
    REQUIRE(decoded->timestamp() != nullptr);
    REQUIRE(decoded->data()->joint_rotations() != nullptr);
    REQUIRE(decoded->data()->global_translation() != nullptr);
    const auto* values = decoded->data()->joint_rotations()->values();
    const auto* hips = (*values)[core::SomaBodyJoint_HIPS];
    const auto* toe = (*values)[core::SomaBodyJoint_RIGHT_TOE_END];
    CHECK(hips->rotation().x() == Catch::Approx(scale));
    CHECK(hips->rotation().y() == Catch::Approx(-2.0f * scale));
    CHECK(hips->rotation().z() == Catch::Approx(3.0f * scale));
    CHECK(hips->rotation().w() == Catch::Approx(-4.0f * scale));
    CHECK(toe->rotation().x() == Catch::Approx(-scale));
    CHECK(toe->rotation().y() == Catch::Approx(2.0f * scale));
    CHECK(toe->rotation().z() == Catch::Approx(-3.0f * scale));
    CHECK(toe->rotation().w() == Catch::Approx(4.0f * scale));
    CHECK(hips->is_valid());
    CHECK(toe->is_valid());
    CHECK_FALSE((*values)[core::SomaBodyJoint_HEAD]->is_valid());
    CHECK(decoded->data()->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->data()->global_translation()->y() == Catch::Approx(-2.0f));
    CHECK(decoded->data()->global_translation()->z() == Catch::Approx(3.0f));
    CHECK_FALSE(decoded->data()->global_translation_is_valid());
    CHECK(decoded->timestamp()->available_time_local_common_clock() == 100);
    CHECK(decoded->timestamp()->sample_time_local_common_clock() == 200);
    CHECK(decoded->timestamp()->sample_time_raw_device_clock() == 300);
}
