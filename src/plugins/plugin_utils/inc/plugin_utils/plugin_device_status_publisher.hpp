// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_trackers/plugin_device_status_tracker.hpp>
#include <pusherio/plugin_session.hpp>
#include <pusherio/schema_pusher.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace plugin_utils
{

struct PluginDeviceStatusEntry
{
    std::string path;
    core::PluginDeviceState state = core::PluginDeviceState_UNKNOWN;
    core::PluginDeviceReason reason = core::PluginDeviceReason_NONE;
    std::string error;

    bool operator==(const PluginDeviceStatusEntry&) const = default;
};

class PluginDeviceStatusPublisher
{
public:
    static constexpr uint32_t SCHEMA_VERSION = 1;
    static constexpr int64_t HEARTBEAT_PERIOD_NS = 1'000'000'000;
    static constexpr int64_t STALE_TIMEOUT_NS = 3'000'000'000;

    using PushBufferFunction = std::function<void(const uint8_t*, size_t, int64_t, int64_t)>;

    PluginDeviceStatusPublisher(const core::OpenXRSessionHandles& handles, const std::string& plugin_root_id);

    // The session must have been created with PluginSessionRequirements::schema_push.
    PluginDeviceStatusPublisher(core::IPluginSession& session, const std::string& plugin_root_id);

    // An injected transport keeps scheduling and encoding testable without an OpenXR runtime.
    PluginDeviceStatusPublisher(const std::string& plugin_root_id, PushBufferFunction push_buffer);

    PluginDeviceStatusPublisher(const PluginDeviceStatusPublisher&) = delete;
    PluginDeviceStatusPublisher& operator=(const PluginDeviceStatusPublisher&) = delete;
    PluginDeviceStatusPublisher(PluginDeviceStatusPublisher&&) = delete;
    PluginDeviceStatusPublisher& operator=(PluginDeviceStatusPublisher&&) = delete;

    static std::vector<std::string> get_required_extensions();

    const std::string& collection_id() const
    {
        return collection_id_;
    }

    bool publish_if_changed(const std::vector<PluginDeviceStatusEntry>& entries, int64_t now_ns);

private:
    static std::string make_collection_id(const std::string& plugin_root_id);
    static std::vector<PluginDeviceStatusEntry> validate_and_canonicalize(
        const std::vector<PluginDeviceStatusEntry>& entries);
    void push_snapshot(const std::vector<PluginDeviceStatusEntry>& entries, int64_t now_ns);

    std::string collection_id_;
    std::unique_ptr<core::SchemaPusher> pusher_;
    PushBufferFunction push_buffer_;
    std::vector<PluginDeviceStatusEntry> latest_entries_;
    int64_t last_publish_ns_ = 0;
    bool has_published_ = false;
};

} // namespace plugin_utils
