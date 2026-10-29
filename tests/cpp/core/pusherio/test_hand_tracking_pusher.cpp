// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <pusherio/hand_tracking_pusher.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace
{

struct CapturedHandState
{
    const XrHandJointLocationEXT* joints{ nullptr };
    int64_t sample_time_ns{ 0 };
    bool channel_closed{ false };
};

class CapturingHandChannel final : public core::IHandTrackingPushChannel
{
public:
    explicit CapturingHandChannel(std::shared_ptr<CapturedHandState> state) : state_(std::move(state))
    {
    }

    ~CapturingHandChannel() override
    {
        state_->channel_closed = true;
    }

    void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns) override
    {
        state_->joints = joint_locations;
        state_->sample_time_ns = sample_time_local_common_clock_ns;
    }

private:
    std::shared_ptr<CapturedHandState> state_;
};

} // namespace

TEST_CASE("HandTrackingPusher delegates samples to its channel", "[pusherio][unit]")
{
    auto state = std::make_shared<CapturedHandState>();
    core::HandTrackingPusher pusher(std::make_unique<CapturingHandChannel>(state));
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};

    pusher.push(joints.data(), 1234);

    REQUIRE(state->joints == joints.data());
    REQUIRE(state->sample_time_ns == 1234);
}

TEST_CASE("HandTrackingPusher rejects an invalid channel or sample", "[pusherio][unit]")
{
    REQUIRE_THROWS_AS(core::HandTrackingPusher(nullptr), std::invalid_argument);

    auto state = std::make_shared<CapturedHandState>();
    core::HandTrackingPusher pusher(std::make_unique<CapturingHandChannel>(state));
    REQUIRE_THROWS_AS(pusher.push(nullptr, 1234), std::invalid_argument);
}

TEST_CASE("HandTrackingPusher closes its logical hand stream on destruction", "[pusherio][unit]")
{
    auto state = std::make_shared<CapturedHandState>();
    {
        core::HandTrackingPusher pusher(std::make_unique<CapturingHandChannel>(state));
        REQUIRE_FALSE(state->channel_closed);
    }
    REQUIRE(state->channel_closed);
}
