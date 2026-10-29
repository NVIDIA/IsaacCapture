// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <pusherio/plugin_session.hpp>
#include <pusherio/schema_pusher.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace
{

struct CapturedState
{
    core::SchemaPusherConfig config;
    std::vector<uint8_t> payload;
    int64_t local_time_ns{ 0 };
    int64_t device_time_ns{ 0 };
    bool session_destroyed{ false };
    bool channel_destroyed{ false };
    bool channel_destroyed_before_session{ false };
};

class CapturingChannel final : public core::ISchemaPushChannel
{
public:
    explicit CapturingChannel(std::shared_ptr<CapturedState> state) : state_(std::move(state))
    {
    }

    ~CapturingChannel() override
    {
        state_->channel_destroyed = true;
    }

    const core::SchemaPusherConfig& config() const override
    {
        return state_->config;
    }

    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns) override
    {
        state_->payload.assign(buffer, buffer + size);
        state_->local_time_ns = sample_time_local_common_clock_ns;
        state_->device_time_ns = sample_time_raw_device_clock_ns;
    }

private:
    std::shared_ptr<CapturedState> state_;
};

class CapturingSession final : public core::IPluginSession
{
public:
    explicit CapturingSession(std::shared_ptr<CapturedState> state) : state_(std::move(state))
    {
    }

    ~CapturingSession() override
    {
        state_->channel_destroyed_before_session = state_->channel_destroyed;
        state_->session_destroyed = true;
    }

    std::unique_ptr<core::IPluginPullChannel> create_pull_channel() override
    {
        return nullptr;
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        state_->config = config;
        return std::make_unique<CapturingChannel>(state_);
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT) override
    {
        return nullptr;
    }

private:
    std::shared_ptr<CapturedState> state_;
};

core::SchemaPusherConfig make_config()
{
    return core::SchemaPusherConfig{ .collection_id = "pedals",
                                     .max_flatbuffer_size = 16,
                                     .tensor_identifier = "pedal_state",
                                     .localized_name = "Pedal state",
                                     .app_name = "PusherTest" };
}

} // namespace

TEST_CASE("SchemaPusher delegates samples to its channel", "[pusherio][unit]")
{
    auto state = std::make_shared<CapturedState>();
    core::PluginSessionHandle session = std::make_shared<CapturingSession>(state);
    core::SchemaPusher pusher(session->create_schema_push_channel(make_config()));
    const std::vector<uint8_t> payload{ 1, 2, 3 };

    pusher.push_buffer(payload.data(), payload.size(), 100, 80);

    REQUIRE(state->config.collection_id == "pedals");
    REQUIRE(state->config.tensor_identifier == "pedal_state");
    REQUIRE(state->payload == payload);
    REQUIRE(state->local_time_ns == 100);
    REQUIRE(state->device_time_ns == 80);
}

TEST_CASE("SchemaPusher leaves session ownership with its caller", "[pusherio][unit]")
{
    auto state = std::make_shared<CapturedState>();
    auto session = std::make_shared<CapturingSession>(state);

    {
        core::SchemaPusher pusher(session->create_schema_push_channel(make_config()));
        REQUIRE(session.use_count() == 1);
        REQUIRE_FALSE(state->session_destroyed);
        REQUIRE_FALSE(state->channel_destroyed);
    }

    REQUIRE(state->channel_destroyed);
    REQUIRE_FALSE(state->session_destroyed);

    session.reset();
    REQUIRE(state->session_destroyed);
    REQUIRE(state->channel_destroyed_before_session);
}

TEST_CASE("SchemaPusher validates transport-independent buffer metadata", "[pusherio][unit]")
{
    auto state = std::make_shared<CapturedState>();
    auto session = std::make_shared<CapturingSession>(state);
    core::SchemaPusher pusher(session->create_schema_push_channel(make_config()));
    const std::vector<uint8_t> oversized(17, 0);

    REQUIRE_THROWS_AS(pusher.push_buffer(oversized.data(), oversized.size(), 100, 80), std::runtime_error);
    REQUIRE_THROWS_AS(pusher.push_buffer(nullptr, 1, 100, 80), std::invalid_argument);
    REQUIRE_THROWS_AS(pusher.push_buffer(nullptr, 0, 100, -1), std::runtime_error);
}
