// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <live_trackers/schema_tracker.hpp>
#include <schema/haptic_command_generated.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace
{

std::vector<uint8_t> encode_command(std::string_view endpoint)
{
    core::HapticCommandT command;
    command.endpoint = endpoint;
    command.values = { 0.25f, 0.75f };

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::HapticCommand::Pack(builder, &command));
    return { builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize() };
}

core::SchemaTrackerBase::SampleResult command_sample(std::string_view endpoint)
{
    core::SchemaTrackerBase::SampleResult sample;
    sample.buffer = encode_command(endpoint);
    return sample;
}

core::SchemaTrackerBase::SampleResult invalid_sample()
{
    core::SchemaTrackerBase::SampleResult sample;
    sample.buffer.assign(8, UINT8_MAX);
    return sample;
}

std::string endpoint(const core::Serialized<core::HapticCommand>& command)
{
    const auto* value = command.get();
    return value != nullptr && value->endpoint() != nullptr ? value->endpoint()->str() : std::string{};
}

} // namespace

TEST_CASE("schema tracker accepts only verified FlatBuffer payloads", "[live_trackers][security]")
{
    std::vector<uint8_t> valid = encode_command("left");

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

TEST_CASE("schema tracker publishes only verified state transitions", "[live_trackers][security]")
{
    core::Serialized<core::HapticCommand> published;

    SECTION("invalid input cannot initialize an empty output")
    {
        std::vector<core::SchemaTrackerBase::SampleResult> raw{ invalid_sample() };
        auto batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, true);
        REQUIRE(batch.samples.empty());
        CHECK(batch.rejected_sample_count == 1);

        core::detail::publish_verified_flatbuffer_batch(std::move(batch), published);
        CHECK_FALSE(published);
    }

    SECTION("invalid input cannot replace the last-known-good output")
    {
        std::vector<core::SchemaTrackerBase::SampleResult> raw{ command_sample("left") };
        auto valid_batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, true);
        core::detail::publish_verified_flatbuffer_batch(std::move(valid_batch), published);
        const auto last_known_good = published;

        raw = { invalid_sample() };
        auto invalid_batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, true);
        core::detail::publish_verified_flatbuffer_batch(std::move(invalid_batch), published);

        CHECK(published.get() == last_known_good.get());
        CHECK(endpoint(published) == "left");
    }

    SECTION("the final verified sample wins even when malformed samples are interleaved")
    {
        std::vector<core::SchemaTrackerBase::SampleResult> raw{ command_sample("left"), invalid_sample(),
                                                                command_sample("right") };
        auto batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, true);
        REQUIRE(batch.samples.size() == 2);
        CHECK(batch.rejected_sample_count == 1);

        core::detail::publish_verified_flatbuffer_batch(std::move(batch), published);
        CHECK(endpoint(published) == "right");
    }

    SECTION("collection loss clears the last-known-good output")
    {
        std::vector<core::SchemaTrackerBase::SampleResult> raw{ command_sample("left") };
        auto valid_batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, true);
        core::detail::publish_verified_flatbuffer_batch(std::move(valid_batch), published);
        REQUIRE(published);

        raw.clear();
        auto absent_batch = core::detail::verify_flatbuffer_samples<core::HapticCommand>(raw, false);
        core::detail::publish_verified_flatbuffer_batch(std::move(absent_batch), published);
        CHECK_FALSE(published);
    }
}

TEST_CASE("schema tracker bounds runtime-provided sample layouts", "[live_trackers][security]")
{
    constexpr size_t maximum_sample_size = 1024;
    constexpr size_t maximum_sample_stride = 4096;

    CHECK(core::detail::is_valid_schema_sample_layout(512, 512, maximum_sample_size, maximum_sample_stride));
    CHECK(core::detail::is_valid_schema_sample_layout(512, 2048, maximum_sample_size, maximum_sample_stride));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(0, 512, maximum_sample_size, maximum_sample_stride));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(512, 511, maximum_sample_size, maximum_sample_stride));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(1025, 1025, maximum_sample_size, maximum_sample_stride));
    CHECK_FALSE(core::detail::is_valid_schema_sample_layout(512, 4097, maximum_sample_size, maximum_sample_stride));
}
