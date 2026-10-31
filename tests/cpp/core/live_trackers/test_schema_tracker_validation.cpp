// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <live_trackers/schema_tracker.hpp>
#include <schema/haptic_command_generated.h>

#include <cstdint>
#include <vector>

TEST_CASE("schema tracker accepts only verified FlatBuffer payloads", "[live_trackers][security]")
{
    core::HapticCommandT command;
    command.endpoint = "left";
    command.values = { 0.25f, 0.75f };

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::HapticCommand::Pack(builder, &command));
    std::vector<uint8_t> valid(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());

    const auto* decoded = core::detail::verified_flatbuffer_root<core::HapticCommand>(valid);
    REQUIRE(decoded != nullptr);
    REQUIRE(decoded->endpoint() != nullptr);
    CHECK(decoded->endpoint()->str() == "left");

    SECTION("empty payload")
    {
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>({}) == nullptr);
    }

    SECTION("truncated payload")
    {
        valid.resize(valid.size() / 2);
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>(valid) == nullptr);
    }

    SECTION("invalid root offset")
    {
        std::vector<uint8_t> invalid(8, UINT8_MAX);
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>(invalid) == nullptr);
    }
}
