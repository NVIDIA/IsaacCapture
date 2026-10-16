// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <schema/soma_body_joint_poses_generated.h>

#include <type_traits>

namespace
{

constexpr flatbuffers::voffset_t vt(flatbuffers::voffset_t field)
{
    return (field + 2) * 2;
}

static_assert(core::SomaBodyJointPoses::VT_JOINT_POSES == vt(0));
static_assert(core::SomaBodyJointPosesRecord::VT_DATA == vt(0));
static_assert(core::SomaBodyJointPosesRecord::VT_TIMESTAMP == vt(1));
static_assert(std::is_trivially_copyable_v<core::SomaBodyJointPose>);
static_assert(sizeof(core::SomaBodyJointPose) == 32);

TEST_CASE("SOMA evaluated body joint poses round trip as a sparse keyed vector", "[soma_body_joint_poses][flatbuffers]")
{
    core::SomaBodyJointPosesT pose;
    pose.joint_poses.emplace_back(
        core::SomaBodyJoint_HEAD, core::Pose(core::Point(1.0f, 2.0f, 3.0f), core::Quaternion(0.0f, 0.0f, 0.6f, 0.8f)));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SomaBodyJointPoses::Pack(builder, &pose));
    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SomaBodyJointPoses>(nullptr));

    const auto* decoded = flatbuffers::GetRoot<core::SomaBodyJointPoses>(builder.GetBufferPointer());
    REQUIRE(decoded->joint_poses() != nullptr);
    REQUIRE(decoded->joint_poses()->size() == 1);
    const auto* head = decoded->joint_poses()->LookupByKey(core::SomaBodyJoint_HEAD);
    REQUIRE(head != nullptr);
    CHECK(head->joint() == core::SomaBodyJoint_HEAD);
    CHECK(head->pose().position().x() == 1.0f);
    CHECK(head->pose().position().y() == 2.0f);
    CHECK(head->pose().position().z() == 3.0f);
    CHECK(head->pose().orientation().z() == 0.6f);
    CHECK(head->pose().orientation().w() == 0.8f);
    CHECK(decoded->joint_poses()->LookupByKey(core::SomaBodyJoint_HIPS) == nullptr);
}

} // namespace
