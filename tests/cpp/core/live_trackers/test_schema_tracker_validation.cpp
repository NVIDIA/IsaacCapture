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

    SECTION("an empty payload is rejected")
    {
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>({}) == nullptr);
    }

    SECTION("a truncated payload is rejected")
    {
        valid.resize(valid.size() / 2);
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>(valid) == nullptr);
    }

    SECTION("an invalid root offset is rejected")
    {
        std::vector<uint8_t> invalid(8, UINT8_MAX);
        CHECK(core::detail::verified_flatbuffer_root<core::HapticCommand>(invalid) == nullptr);
    }
}

TEST_CASE("schema tracker bounds runtime-provided sample layouts", "[live_trackers][security]")
{
    constexpr size_t maximum_size = 1024;

    CHECK(core::detail::is_valid_schema_sample_layout(512, 512, maximum_size));
    CHECK(core::detail::is_valid_schema_sample_layout(512, 768, maximum_size));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(0, 512, maximum_size));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(512, 511, maximum_size));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(1025, 1025, maximum_size));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(512, 1025, maximum_size));
}
