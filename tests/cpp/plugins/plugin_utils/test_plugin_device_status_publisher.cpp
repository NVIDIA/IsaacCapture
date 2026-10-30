// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <plugin_utils/plugin_device_status_publisher.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct PublishedSnapshot
{
    uint32_t schema_version;
    int64_t report_time_ns;
    int64_t local_time_ns;
    int64_t raw_time_ns;
    std::vector<plugin_utils::PluginDeviceStatusEntry> devices;
};

class CapturingChannel final : public core::ISchemaPushChannel
{
public:
    CapturingChannel(core::SchemaPusherConfig config, std::vector<PublishedSnapshot>& published)
        : config_(std::move(config)), published_(published)
    {
    }

    const core::SchemaPusherConfig& config() const override
    {
        return config_;
    }

    void push_buffer(const uint8_t* buffer, size_t, int64_t local_time_ns, int64_t raw_time_ns) override
    {
        const auto* snapshot = flatbuffers::GetRoot<core::PluginDeviceStatusSnapshot>(buffer);
        PublishedSnapshot captured{
            .schema_version = snapshot->schema_version(),
            .report_time_ns = snapshot->report_time_ns(),
            .local_time_ns = local_time_ns,
            .raw_time_ns = raw_time_ns,
        };
        if (snapshot->devices() != nullptr)
        {
            captured.devices.reserve(snapshot->devices()->size());
            for (const auto* device : *snapshot->devices())
            {
                captured.devices.push_back(plugin_utils::PluginDeviceStatusEntry{
                    .path = device->path() != nullptr ? device->path()->str() : std::string{},
                    .state = device->state(),
                    .reason = device->reason(),
                    .error = device->error() != nullptr ? device->error()->str() : std::string{},
                });
            }
        }
        published_.push_back(std::move(captured));
    }

private:
    core::SchemaPusherConfig config_;
    std::vector<PublishedSnapshot>& published_;
};

class CapturingSession final : public core::IPluginSession
{
public:
    explicit CapturingSession(std::vector<PublishedSnapshot>& published) : published_(published)
    {
    }

    core::IPluginPullChannel& pull_channel() override
    {
        throw std::logic_error("CapturingSession has no pull channel");
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        return std::make_unique<CapturingChannel>(config, published_);
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT) override
    {
        throw std::logic_error("CapturingSession has no hand tracking push");
    }

private:
    std::vector<PublishedSnapshot>& published_;
};

plugin_utils::PluginDeviceStatusEntry connected(std::string path)
{
    return plugin_utils::PluginDeviceStatusEntry{
        .path = std::move(path),
        .state = core::PluginDeviceState_CONNECTED,
        .reason = core::PluginDeviceReason_NONE,
    };
}

} // namespace

TEST_CASE("Plugin device status publisher exposes the monitoring contract", "[plugin_utils][device_status]")
{
    static_assert(plugin_utils::PluginDeviceStatusPublisher::SCHEMA_VERSION == 1);
    static_assert(plugin_utils::PluginDeviceStatusPublisher::HEARTBEAT_PERIOD_NS == 1'000'000'000);
    static_assert(plugin_utils::PluginDeviceStatusPublisher::STALE_TIMEOUT_NS == 3'000'000'000);

    std::vector<PublishedSnapshot> published;
    CapturingSession session(published);
    plugin_utils::PluginDeviceStatusPublisher publisher(session, "/plugin/example");

    CHECK(publisher.collection_id() == "/plugin/example/device_status");
    CHECK(publisher.get_required_extensions() == core::SchemaPusher::get_required_extensions());
}

TEST_CASE("Plugin device status publisher sends changes and heartbeats", "[plugin_utils][device_status]")
{
    std::vector<PublishedSnapshot> published;
    CapturingSession session(published);
    plugin_utils::PluginDeviceStatusPublisher publisher(session, "/plugin/example");
    const auto left = connected("/input/left");
    const auto right = connected("/input/right");

    CHECK(publisher.publish_if_changed({ right, left }, 10));
    REQUIRE(published.size() == 1);
    CHECK(published[0].schema_version == 1);
    CHECK(published[0].report_time_ns == 10);
    CHECK(published[0].local_time_ns == 10);
    CHECK(published[0].raw_time_ns == 10);
    REQUIRE(published[0].devices.size() == 2);
    CHECK(published[0].devices[0].path == "/input/left");
    CHECK(published[0].devices[1].path == "/input/right");

    CHECK_FALSE(publisher.publish_if_changed({ left, right }, 20));
    CHECK(published.size() == 1);

    auto failed = right;
    failed.state = core::PluginDeviceState_FAILED;
    failed.reason = core::PluginDeviceReason_DEVICE_ERROR;
    failed.error = "read failed";
    CHECK(publisher.publish_if_changed({ left, failed }, 30));
    CHECK(published.size() == 2);

    const int64_t heartbeat_time = 30 + plugin_utils::PluginDeviceStatusPublisher::HEARTBEAT_PERIOD_NS;
    CHECK(publisher.publish_if_changed({ failed, left }, heartbeat_time));
    REQUIRE(published.size() == 3);
    CHECK(published.back().report_time_ns == heartbeat_time);
}

TEST_CASE("Plugin device status publisher validates input before transport", "[plugin_utils][device_status]")
{
    std::vector<PublishedSnapshot> published;
    CapturingSession session(published);

    CHECK_THROWS_AS(plugin_utils::PluginDeviceStatusPublisher(session, ""), std::invalid_argument);
    const std::string longest_root(XR_MAX_TENSOR_IDENTIFIER_SIZE - 1 - std::string("/device_status").size(), 'a');
    CHECK_NOTHROW(plugin_utils::PluginDeviceStatusPublisher(session, longest_root));
    CHECK_THROWS_AS(plugin_utils::PluginDeviceStatusPublisher(session, longest_root + "a"), std::invalid_argument);

    plugin_utils::PluginDeviceStatusPublisher publisher(session, "/plugin/example");
    CHECK_THROWS_AS(publisher.publish_if_changed({ connected("") }, 1), std::invalid_argument);
    CHECK_THROWS_AS(publisher.publish_if_changed({ connected("/same"), connected("/same") }, 1), std::invalid_argument);
    CHECK(published.empty());

    CHECK(publisher.publish_if_changed({}, 2));
    CHECK_THROWS_AS(publisher.publish_if_changed({}, 1), std::invalid_argument);
    CHECK(published.size() == 1);
}
