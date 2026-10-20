// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <schema/profile_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

TEST_CASE("Semantic profiles retain typed payloads and content references", "[unit][schema]")
{
    core::ContentDescriptorT resource;
    resource.logical_id = "robot";
    resource.media_type = "application/xml";
    resource.version = "7";
    resource.uris = { "package://robot/model.urdf" };
    resource.sha256 = std::vector<uint8_t>(32, 0x2a);
    resource.byte_size = 128;
    resource.inline_data = { '<', '/', '>' };

    core::SemanticProfileT profile;
    profile.profile_id = "org.example.robot.binding";
    profile.version_major = 2;
    profile.version_minor = 3;
    profile.handling = core::ProfileHandling_REQUIRED;
    profile.payload_schema = "example.RobotBinding";
    profile.payload = { 1, 2, 3 };
    profile.resources.push_back(std::make_shared<core::ContentDescriptorT>(resource));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::SemanticProfile::Pack(builder, &profile));

    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(verifier.VerifyBuffer<core::SemanticProfile>(nullptr));
    const auto* encoded = flatbuffers::GetRoot<core::SemanticProfile>(builder.GetBufferPointer());
    CHECK(encoded->profile_id()->str() == "org.example.robot.binding");
    CHECK(encoded->handling() == core::ProfileHandling_REQUIRED);
    REQUIRE(encoded->resources()->size() == 1);
    CHECK(encoded->resources()->Get(0)->sha256()->size() == 32);
}
