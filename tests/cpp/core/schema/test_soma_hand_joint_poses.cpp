// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_hand_joint_poses_generated.h>

#include <type_traits>

namespace
{

constexpr auto vt(uint16_t field_id)
{
    return static_cast<flatbuffers::voffset_t>((field_id + 2) * 2);
}

} // namespace

static_assert(core::SomaHandJointPoses::VT_JOINT_POSES == vt(0));
static_assert(core::SomaHandJointPoses::VT_HANDEDNESS == vt(1));
static_assert(core::SomaHandJointPosesRecord::VT_DATA == vt(0));
static_assert(core::SomaHandJointPosesRecord::VT_TIMESTAMP == vt(1));
static_assert(std::is_trivially_copyable_v<core::SomaHandJointPose>);
static_assert(sizeof(core::SomaHandJointPose) == 32);

TEST_CASE("SOMA hand joint poses round trip as a sparse keyed vector", "[soma_hand_joint_poses][flatbuffers]")
{
    core::SomaHandJointPosesT pose;
    pose.joint_poses.emplace_back(core::SomaHandJoint_INDEX_END,
                                  core::Pose(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f)));
    pose.handedness = core::SomaHandedness_LEFT;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaHandJointPoses::Pack(builder, &pose));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaHandJointPoses>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaHandJointPoses>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_poses() != nullptr);
    REQUIRE(decoded->joint_poses()->size() == 1);
    const auto* index_end = decoded->joint_poses()->LookupByKey(core::SomaHandJoint_INDEX_END);
    REQUIRE(index_end != nullptr);
    CHECK(index_end->joint() == core::SomaHandJoint_INDEX_END);
    CHECK(index_end->pose().position().x() == Catch::Approx(1.0f));
    CHECK(index_end->pose().orientation().w() == Catch::Approx(0.8f));
    CHECK(decoded->joint_poses()->LookupByKey(core::SomaHandJoint_WRIST) == nullptr);
    CHECK(decoded->handedness() == core::SomaHandedness_LEFT);
}
