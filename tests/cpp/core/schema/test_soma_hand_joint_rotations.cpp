// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_hand_joint_rotations_generated.h>
#include <schema/timestamp_generated.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#define VT(field) (field + 2) * 2
static_assert(core::SomaHandJointRotations::VT_JOINT_ROTATIONS == VT(0));
static_assert(core::SomaHandJointRotations::VT_GLOBAL_TRANSLATION == VT(1));
static_assert(core::SomaHandJointRotations::VT_GLOBAL_TRANSLATION_IS_VALID == VT(2));
static_assert(core::SomaHandJointRotations::VT_HANDEDNESS == VT(3));
static_assert(core::SomaHandJointRotationsRecord::VT_DATA == VT(0));
static_assert(core::SomaHandJointRotationsRecord::VT_TIMESTAMP == VT(1));

#define TYPE(field) decltype(std::declval<core::SomaHandJointRotations>().field())
static_assert(std::is_same_v<TYPE(joint_rotations), const core::SomaHandJointRotationArray*>);
static_assert(std::is_same_v<TYPE(global_translation), const core::Point*>);
static_assert(std::is_same_v<TYPE(global_translation_is_valid), bool>);
static_assert(std::is_same_v<TYPE(handedness), core::SomaHandedness>);

static_assert(std::is_trivially_copyable_v<core::SomaJointRotation>);
static_assert(std::is_trivially_copyable_v<core::SomaHandJointRotationArray>);
static_assert(sizeof(core::SomaHandJointRotationArray) == 25 * sizeof(core::SomaJointRotation));

static_assert(core::SomaHandJoint_WRIST == 0);
static_assert(core::SomaHandJoint_INDEX1 == 5);
static_assert(core::SomaHandJoint_MIDDLE1 == 10);
static_assert(core::SomaHandJoint_RING1 == 15);
static_assert(core::SomaHandJoint_PINKY_END == 24);
static_assert(core::SomaHandJoint_NUM_JOINTS == 25);

TEST_CASE("SOMA hand rotation array has the public layout", "[soma_hand_joint_rotations][struct]")
{
    core::SomaHandJointRotationArray rotations;

    REQUIRE(rotations.values()->size() == static_cast<size_t>(core::SomaHandJoint_NUM_JOINTS));
    CHECK((*rotations.values())[core::SomaHandJoint_WRIST]->rotation().x() == 0.0f);
    CHECK_FALSE((*rotations.values())[core::SomaHandJoint_PINKY_END]->is_valid());

    const core::SomaJointRotation wrist(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true);
    rotations.mutable_values()->Mutate(core::SomaHandJoint_WRIST, wrist);

    CHECK((*rotations.values())[core::SomaHandJoint_WRIST]->rotation().z() == Catch::Approx(0.6f));
    CHECK((*rotations.values())[core::SomaHandJoint_WRIST]->rotation().w() == Catch::Approx(0.8f));
    CHECK((*rotations.values())[core::SomaHandJoint_WRIST]->is_valid());
}

TEST_CASE("SOMA hand joint rotations round trip through FlatBuffers", "[soma_hand_joint_rotations][flatbuffers]")
{
    core::SomaHandJointRotationsT pose;
    pose.joint_rotations = std::make_shared<core::SomaHandJointRotationArray>();
    pose.joint_rotations->mutable_values()->Mutate(
        core::SomaHandJoint_INDEX_END, core::SomaJointRotation(core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f), true));
    pose.global_translation = std::make_shared<core::Point>(1.0f, 2.0f, 3.0f);
    pose.global_translation_is_valid = true;
    pose.handedness = core::SomaHandedness_RIGHT;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaHandJointRotations::Pack(builder, &pose));

    const auto* decoded = flatbuffers::GetRoot<core::SomaHandJointRotations>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_rotations() != nullptr);
    REQUIRE(decoded->global_translation() != nullptr);
    const auto* index_end = (*decoded->joint_rotations()->values())[core::SomaHandJoint_INDEX_END];
    CHECK(index_end->rotation().z() == Catch::Approx(0.6f));
    CHECK(index_end->rotation().w() == Catch::Approx(0.8f));
    CHECK(index_end->is_valid());
    CHECK(decoded->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->global_translation_is_valid());
    CHECK(decoded->handedness() == core::SomaHandedness_RIGHT);
}

TEST_CASE("SOMA hand record preserves quaternion components and validity", "[soma_hand_joint_rotations][flatbuffers]")
{
    const float scale = 1.0f / std::sqrt(30.0f);
    core::SomaHandJointRotationsRecordT record;
    record.data = std::make_shared<core::SomaHandJointRotationsT>();
    record.data->joint_rotations = std::make_shared<core::SomaHandJointRotationArray>();
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaHandJoint_WRIST,
        core::SomaJointRotation(core::Quaternion(scale, -2.0f * scale, 3.0f * scale, -4.0f * scale), true));
    record.data->joint_rotations->mutable_values()->Mutate(
        core::SomaHandJoint_PINKY_END,
        core::SomaJointRotation(core::Quaternion(-scale, 2.0f * scale, -3.0f * scale, 4.0f * scale), true));
    record.data->global_translation = std::make_shared<core::Point>(1.0f, -2.0f, 3.0f);
    record.data->global_translation_is_valid = false;
    record.data->handedness = core::SomaHandedness_LEFT;
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(100, 200, 300);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaHandJointRotationsRecord::Pack(builder, &record));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaHandJointRotationsRecord>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaHandJointRotationsRecord>(builder.GetBufferPointer());
    REQUIRE(decoded->data() != nullptr);
    REQUIRE(decoded->timestamp() != nullptr);
    REQUIRE(decoded->data()->joint_rotations() != nullptr);
    REQUIRE(decoded->data()->global_translation() != nullptr);
    const auto* values = decoded->data()->joint_rotations()->values();
    const auto* wrist = (*values)[core::SomaHandJoint_WRIST];
    const auto* pinky = (*values)[core::SomaHandJoint_PINKY_END];
    CHECK(wrist->rotation().x() == Catch::Approx(scale));
    CHECK(wrist->rotation().y() == Catch::Approx(-2.0f * scale));
    CHECK(wrist->rotation().z() == Catch::Approx(3.0f * scale));
    CHECK(wrist->rotation().w() == Catch::Approx(-4.0f * scale));
    CHECK(pinky->rotation().x() == Catch::Approx(-scale));
    CHECK(pinky->rotation().y() == Catch::Approx(2.0f * scale));
    CHECK(pinky->rotation().z() == Catch::Approx(-3.0f * scale));
    CHECK(pinky->rotation().w() == Catch::Approx(4.0f * scale));
    CHECK(wrist->is_valid());
    CHECK(pinky->is_valid());
    CHECK_FALSE((*values)[core::SomaHandJoint_INDEX1]->is_valid());
    CHECK(decoded->data()->global_translation()->x() == Catch::Approx(1.0f));
    CHECK(decoded->data()->global_translation()->y() == Catch::Approx(-2.0f));
    CHECK(decoded->data()->global_translation()->z() == Catch::Approx(3.0f));
    CHECK_FALSE(decoded->data()->global_translation_is_valid());
    CHECK(decoded->data()->handedness() == core::SomaHandedness_LEFT);
    CHECK(decoded->timestamp()->available_time_local_common_clock() == 100);
    CHECK(decoded->timestamp()->sample_time_local_common_clock() == 200);
    CHECK(decoded->timestamp()->sample_time_raw_device_clock() == 300);
}
