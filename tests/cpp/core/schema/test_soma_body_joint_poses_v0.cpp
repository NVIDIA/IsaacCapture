// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <schema/soma_body_joint_poses_v0_generated.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace
{

constexpr flatbuffers::voffset_t vt(flatbuffers::voffset_t field)
{
    return (field + 2) * 2;
}

static_assert(core::SomaBodyJointPosesV0::VT_JOINT_POSES == vt(0));
static_assert(core::SomaBodyJointPosesV0Record::VT_DATA == vt(0));
static_assert(core::SomaBodyJointPosesV0Record::VT_TIMESTAMP == vt(1));
static_assert(std::is_trivially_copyable_v<core::SomaBodyJointPoseArrayV0>);
static_assert(sizeof(core::SomaBodyJointPoseArrayV0) == 77 * sizeof(core::SomaBodyJointPoseV0));

TEST_CASE("SOMA evaluated body joint poses v0 round trip", "[soma_body_joint_poses_v0][flatbuffers]")
{
    core::SomaBodyJointPosesV0T pose;
    pose.joint_poses = std::make_shared<core::SomaBodyJointPoseArrayV0>();
    pose.joint_poses->mutable_values()->Mutate(
        core::SomaBodyJointV0_HEAD,
        core::SomaBodyJointPoseV0(
            core::Pose(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f)), true));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointPosesV0::Pack(builder, &pose));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaBodyJointPosesV0>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointPosesV0>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_poses() != nullptr);
    const auto* head = (*decoded->joint_poses()->values())[core::SomaBodyJointV0_HEAD];
    CHECK(head->pose().position().x() == 1.0f);
    CHECK(head->pose().position().y() == 2.0f);
    CHECK(head->pose().position().z() == 3.0f);
    CHECK(head->pose().orientation().z() == 0.6f);
    CHECK(head->pose().orientation().w() == 0.8f);
    CHECK(head->is_valid());
}

} // namespace
