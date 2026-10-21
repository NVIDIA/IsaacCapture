// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <deviceio_base/controller_tracker_base.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <pusherio/plugin_session.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{

struct PullState
{
    int update_count{ 0 };
    int64_t last_update_time_ns{ 0 };
    bool tracker_resolved{ false };
    bool wrist_queried{ false };
    bool pull_channel_destroyed{ false };
    bool session_destroyed{ false };
    bool pull_destroyed_before_session{ false };
};

class FakeControllerTrackerImpl final : public core::IControllerTrackerImpl
{
public:
    explicit FakeControllerTrackerImpl(std::shared_ptr<PullState> state) : state_(std::move(state))
    {
    }

    void update(int64_t monotonic_time_ns) override
    {
        ++state_->update_count;
        state_->last_update_time_ns = monotonic_time_ns;
    }

    const core::Serialized<core::ControllerSnapshot>& get_left_controller() const override
    {
        return left_;
    }

    const core::Serialized<core::ControllerSnapshot>& get_right_controller() const override
    {
        return right_;
    }

    void apply_left_haptic_feedback(float, float, float) const override
    {
    }

    void apply_right_haptic_feedback(float, float, float) const override
    {
    }

private:
    std::shared_ptr<PullState> state_;
    core::Serialized<core::ControllerSnapshot> left_;
    core::Serialized<core::ControllerSnapshot> right_;
};

class FakeWristTrackingSource final : public core::IWristTrackingSource
{
public:
    explicit FakeWristTrackingSource(std::shared_ptr<PullState> state) : state_(std::move(state))
    {
    }

    core::WristTrackingSample query(bool is_left, int64_t sample_time_local_common_clock_ns) override
    {
        state_->wrist_queried = is_left && sample_time_local_common_clock_ns == 4242;
        core::WristTrackingSample sample;
        sample.pose.position.x = 1.0f;
        sample.valid = true;
        sample.tracked = true;
        return sample;
    }

private:
    std::shared_ptr<PullState> state_;
};

class FakePullChannel final : public core::IPluginPullChannel
{
public:
    FakePullChannel(std::shared_ptr<PullState> state, std::shared_ptr<core::ControllerTracker> controller_tracker)
        : state_(std::move(state)), controller_tracker_(std::move(controller_tracker)), controller_impl_(state_)
    {
    }

    ~FakePullChannel() override
    {
        state_->pull_channel_destroyed = true;
    }

    void update() override
    {
        controller_impl_.update(1234);
    }

    const core::ITrackerImpl& get_tracker_impl(const core::ITracker& tracker) const override
    {
        if (&tracker != controller_tracker_.get())
        {
            throw std::runtime_error("Tracker implementation not found");
        }
        state_->tracker_resolved = true;
        return controller_impl_;
    }

    std::unique_ptr<core::IWristTrackingSource> create_wrist_tracking_source(const core::WristTrackingSourceConfig&) override
    {
        return std::make_unique<FakeWristTrackingSource>(state_);
    }

private:
    std::shared_ptr<PullState> state_;
    std::shared_ptr<core::ControllerTracker> controller_tracker_;
    FakeControllerTrackerImpl controller_impl_;
};

class FakePluginSession final : public core::IPluginSession
{
public:
    FakePluginSession(std::shared_ptr<PullState> state, std::shared_ptr<core::ControllerTracker> controller_tracker)
        : state_(std::move(state)), controller_tracker_(std::move(controller_tracker))
    {
    }

    ~FakePluginSession() override
    {
        state_->pull_destroyed_before_session = state_->pull_channel_destroyed;
        state_->session_destroyed = true;
    }

    std::unique_ptr<core::IPluginPullChannel> create_pull_channel() override
    {
        return std::make_unique<FakePullChannel>(state_, controller_tracker_);
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig&) override
    {
        return nullptr;
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT) override
    {
        return nullptr;
    }

private:
    std::shared_ptr<PullState> state_;
    std::shared_ptr<core::ControllerTracker> controller_tracker_;
};

} // namespace

TEST_CASE("Plugin pull channel supports typed trackers and optional sources", "[pusherio][unit]")
{
    auto state = std::make_shared<PullState>();
    auto controller_tracker = std::make_shared<core::ControllerTracker>();
    core::PluginSessionHandle session = std::make_shared<FakePluginSession>(state, controller_tracker);

    {
        auto pull_channel = session->create_pull_channel();
        pull_channel->update();

        const auto& left = controller_tracker->get_left_controller(*pull_channel);
        REQUIRE_FALSE(left);
        REQUIRE(state->tracker_resolved);
        REQUIRE(state->update_count == 1);
        REQUIRE(state->last_update_time_ns == 1234);

        auto wrist_source = pull_channel->create_wrist_tracking_source({});
        const core::WristTrackingSample wrist = wrist_source->query(true, 4242);
        REQUIRE(state->wrist_queried);
        REQUIRE(wrist.valid);
        REQUIRE(wrist.tracked);
        REQUIRE(wrist.pose.position.x == 1.0f);
    }

    REQUIRE(state->pull_channel_destroyed);
    REQUIRE_FALSE(state->session_destroyed);
    session.reset();
    REQUIRE(state->session_destroyed);
    REQUIRE(state->pull_destroyed_before_session);
}
