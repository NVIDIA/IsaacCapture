// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/soma_hand_joint_poses_v0_generated.h>

#include <memory>

namespace
{

constexpr auto vt(uint16_t field_id)
{
    return static_cast<flatbuffers::voffset_t>((field_id + 2) * 2);
}

} // namespace

static_assert(core::SomaHandJointPosesV0::VT_JOINT_POSES == vt(0));
static_assert(core::SomaHandJointPosesV0::VT_HANDEDNESS == vt(1));
static_assert(core::SomaHandJointPosesV0Record::VT_DATA == vt(0));
static_assert(core::SomaHandJointPosesV0Record::VT_TIMESTAMP == vt(1));
static_assert(sizeof(core::SomaHandJointPoseArrayV0) == 25 * sizeof(core::SomaHandJointPoseV0));

TEST_CASE("SOMA hand joint poses v0 round trip through FlatBuffers", "[soma_hand_joint_poses_v0][flatbuffers]")
{
    core::SomaHandJointPosesV0T pose;
    pose.joint_poses = std::make_shared<core::SomaHandJointPoseArrayV0>();
    pose.joint_poses->mutable_values()->Mutate(
        core::SomaHandJointV0_INDEX_END,
        core::SomaHandJointPoseV0(
            core::Pose(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f)), true));
    pose.handedness = core::SomaHandednessV0_LEFT;

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaHandJointPosesV0::Pack(builder, &pose));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaHandJointPosesV0>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaHandJointPosesV0>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_poses() != nullptr);
    const auto* index_end = (*decoded->joint_poses()->values())[core::SomaHandJointV0_INDEX_END];
    CHECK(index_end->pose().position().x() == Catch::Approx(1.0f));
    CHECK(index_end->pose().orientation().w() == Catch::Approx(0.8f));
    CHECK(index_end->is_valid());
    CHECK(decoded->handedness() == core::SomaHandednessV0_LEFT);
}
