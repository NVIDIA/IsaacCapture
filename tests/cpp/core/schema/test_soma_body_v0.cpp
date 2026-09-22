// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_body_v0_generated.h>
#include <schema/timestamp_generated.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#define VT(field) (field + 2) * 2
static_assert(core::SomaBodyJointRotationsV0::VT_JOINT_ROTATIONS == VT(0));
static_assert(core::SomaBodyJointRotationsV0::VT_GLOBAL_TRANSLATION == VT(1));
static_assert(core::SomaBodyJointRotationsV0::VT_GLOBAL_TRANSLATION_IS_VALID == VT(2));
static_assert(core::SomaBodyJointRotationsV0Record::VT_DATA == VT(0));
static_assert(core::SomaBodyJointRotationsV0Record::VT_TIMESTAMP == VT(1));

#define TYPE(field) decltype(std::declval<core::SomaBodyJointRotationsV0>().field())
static_assert(std::is_same_v<TYPE(joint_rotations), const core::SomaBodyJointRotationArrayV0*>);
static_assert(std::is_same_v<TYPE(global_translation), const core::Point*>);
static_assert(std::is_same_v<TYPE(global_translation_is_valid), bool>);

static_assert(std::is_trivially_copyable_v<core::SomaJointRotationV0>);
static_assert(std::is_trivially_copyable_v<core::SomaBodyJointRotationArrayV0>);
static_assert(sizeof(core::SomaBodyJointRotationArrayV0) == 77 * sizeof(core::SomaJointRotationV0));

static_assert(core::SomaBodyJointV0_HIPS == 0);
static_assert(core::SomaBodyJointV0_LEFT_SHOULDER == 11);
static_assert(core::SomaBodyJointV0_RIGHT_SHOULDER == 39);
static_assert(core::SomaBodyJointV0_LEFT_LEG == 67);
static_assert(core::SomaBodyJointV0_RIGHT_TOE_END == 76);
static_assert(core::SomaBodyJointV0_NUM_JOINTS == 77);

TEST_CASE("SOMA body v0 rotation array has the public layout", "[soma_body_v0][struct]")
{
    core::SomaBodyJointRotationArrayV0 rotations;

    REQUIRE(rotations.values()->size() == static_cast<size_t>(core::SomaBodyJointV0_NUM_JOINTS));
    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->rotation().x() == 0.0f);
    CHECK_FALSE((*rotations.values())[core::SomaBodyJointV0_RIGHT_TOE_END]->is_valid());

    const core::SomaJointRotationV0 hips(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true);
    rotations.mutable_values()->Mutate(core::SomaBodyJointV0_HIPS, hips);

    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->rotation().z() == Catch::Approx(0.6f));
    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->rotation().w() == Catch::Approx(0.8f));
    CHECK((*rotations.values())[core::SomaBodyJointV0_HIPS]->is_valid());
}

TEST_CASE("SOMA body joint rotations v0 round trip through FlatBuffers", "[soma_body_v0][flatbuffers]")
{
    core::SomaBodyJointRotationsV0T pose;
    pose.joint_rotations = std::make_shared<core::SomaBodyJointRotationArrayV0>();
    pose.joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJointV0_HEAD, core::SomaJointRotationV0(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true));
    pose.global_translation = std::make_shared<core::Point>(1.0f, 2.0f, 3.0f);
    pose.global_translation_is_valid = true;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointRotationsV0::Pack(builder, &pose));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointRotationsV0>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_rotations() != nullptr);
    REQUIRE(decoded->global_translation() != nullptr);
    const auto* head = (*decoded->joint_rotations()->values())[core::SomaBodyJointV0_HEAD];
    CHECK(head->rotation().z() == Catch::Approx(0.6f));
    CHECK(head->rotation().w() == Catch::Approx(0.8f));
    CHECK(head->is_valid());
    CHECK(decoded->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->global_translation_is_valid());
}

TEST_CASE("SOMA body record v0 preserves quaternion components and validity", "[soma_body_v0][flatbuffers]")
{
    const float scale = 1.0f / std::sqrt(30.0f);
    core::SomaBodyJointRotationsV0RecordT record;
    record.data = std::make_shared<core::SomaBodyJointRotationsV0T>();
    record.data->joint_rotations = std::make_shared<core::SomaBodyJointRotationArrayV0>();
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJointV0_HIPS,
        core::SomaJointRotationV0(core::Quaternion(scale, -2.0f * scale, 3.0f * scale, -4.0f * scale), true));
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaBodyJointV0_RIGHT_TOE_END,
        core::SomaJointRotationV0(core::Quaternion(-scale, 2.0f * scale, -3.0f * scale, 4.0f * scale), true));
    record.data->global_translation = std::make_shared<core::Point>(1.0f, -2.0f, 3.0f);
    record.data->global_translation_is_valid = false;
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(100, 200, 300);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointRotationsV0Record::Pack(builder, &record));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaBodyJointRotationsV0Record>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointRotationsV0Record>(builder.GetBufferPointer());
    REQUIRE(decoded->data() != nullptr);
    REQUIRE(decoded->timestamp() != nullptr);
    REQUIRE(decoded->data()->joint_rotations() != nullptr);
    REQUIRE(decoded->data()->global_translation() != nullptr);
    const auto* values = decoded->data()->joint_rotations()->values();
    const auto* hips = (*values)[core::SomaBodyJointV0_HIPS];
    const auto* toe = (*values)[core::SomaBodyJointV0_RIGHT_TOE_END];
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
    CHECK_FALSE((*values)[core::SomaBodyJointV0_HEAD]->is_valid());
    CHECK(decoded->data()->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->data()->global_translation()->y() == Catch::Approx(-2.0f));
    CHECK(decoded->data()->global_translation()->z() == Catch::Approx(3.0f));
    CHECK_FALSE(decoded->data()->global_translation_is_valid());
    CHECK(decoded->timestamp()->available_time_local_common_clock() == 100);
    CHECK(decoded->timestamp()->sample_time_local_common_clock() == 200);
    CHECK(decoded->timestamp()->sample_time_raw_device_clock() == 300);
}
