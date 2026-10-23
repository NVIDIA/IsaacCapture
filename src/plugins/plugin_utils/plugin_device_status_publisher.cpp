// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/plugin_utils/plugin_device_status_publisher.hpp"

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace plugin_utils
{

namespace
{

core::SchemaPusherConfig make_pusher_config(const std::string& collection_id)
{
    return core::SchemaPusherConfig{
        .collection_id = collection_id,
        .max_flatbuffer_size = core::PluginDeviceStatusTracker::DEFAULT_MAX_FLATBUFFER_SIZE,
        .tensor_identifier = std::string(core::PluginDeviceStatusTracker::TENSOR_IDENTIFIER),
        .localized_name = "Plugin Device Status",
    };
}

} // namespace

PluginDeviceStatusPublisher::PluginDeviceStatusPublisher(const core::OpenXRSessionHandles& handles,
                                                         const std::string& plugin_root_id)
    : collection_id_(make_collection_id(plugin_root_id)),
      pusher_(std::make_unique<core::SchemaPusher>(handles, make_pusher_config(collection_id_)))
{
}

PluginDeviceStatusPublisher::PluginDeviceStatusPublisher(core::IPluginSession& session, const std::string& plugin_root_id)
    : collection_id_(make_collection_id(plugin_root_id)),
      pusher_(std::make_unique<core::SchemaPusher>(session.create_schema_push_channel(make_pusher_config(collection_id_))))
{
}

PluginDeviceStatusPublisher::PluginDeviceStatusPublisher(const std::string& plugin_root_id, PushBufferFunction push_buffer)
    : collection_id_(make_collection_id(plugin_root_id)), push_buffer_(std::move(push_buffer))
{
    if (!push_buffer_)
    {
        throw std::invalid_argument("PluginDeviceStatusPublisher requires a push transport");
    }
}

std::vector<std::string> PluginDeviceStatusPublisher::get_required_extensions()
{
    return core::SchemaPusher::get_required_extensions();
}

bool PluginDeviceStatusPublisher::publish_if_changed(const std::vector<PluginDeviceStatusEntry>& entries, int64_t now_ns)
{
    auto canonical_entries = validate_and_canonicalize(entries);
    if (has_published_ && now_ns < last_publish_ns_)
    {
        throw std::invalid_argument("PluginDeviceStatusPublisher now_ns must be monotonic");
    }

    const bool changed = !has_published_ || canonical_entries != latest_entries_;
    const bool heartbeat_due = has_published_ && now_ns - last_publish_ns_ >= HEARTBEAT_PERIOD_NS;
    if (!changed && !heartbeat_due)
    {
        return false;
    }

    push_snapshot(canonical_entries, now_ns);
    latest_entries_ = std::move(canonical_entries);
    last_publish_ns_ = now_ns;
    has_published_ = true;
    return true;
}

std::string PluginDeviceStatusPublisher::make_collection_id(const std::string& plugin_root_id)
{
    if (plugin_root_id.empty())
    {
        throw std::invalid_argument("PluginDeviceStatusPublisher plugin_root_id must not be empty");
    }
    std::string collection_id = plugin_root_id + "/device_status";
    if (collection_id.size() >= XR_MAX_TENSOR_IDENTIFIER_SIZE)
    {
        throw std::invalid_argument("PluginDeviceStatusPublisher collection ID exceeds the OpenXR limit of " +
                                    std::to_string(XR_MAX_TENSOR_IDENTIFIER_SIZE - 1) + " bytes");
    }
    return collection_id;
}

std::vector<PluginDeviceStatusEntry> PluginDeviceStatusPublisher::validate_and_canonicalize(
    const std::vector<PluginDeviceStatusEntry>& entries)
{
    auto canonical_entries = entries;
    std::sort(canonical_entries.begin(), canonical_entries.end(),
              [](const PluginDeviceStatusEntry& left, const PluginDeviceStatusEntry& right)
              { return left.path < right.path; });

    for (size_t index = 0; index < canonical_entries.size(); ++index)
    {
        if (canonical_entries[index].path.empty())
        {
            throw std::invalid_argument("PluginDeviceStatusPublisher device paths must not be empty");
        }
        if (index > 0 && canonical_entries[index - 1].path == canonical_entries[index].path)
        {
            throw std::invalid_argument("PluginDeviceStatusPublisher device paths must be unique: " +
                                        canonical_entries[index].path);
        }
    }
    return canonical_entries;
}

void PluginDeviceStatusPublisher::push_snapshot(const std::vector<PluginDeviceStatusEntry>& entries, int64_t now_ns)
{
    core::PluginDeviceStatusSnapshotT snapshot;
    snapshot.schema_version = SCHEMA_VERSION;
    snapshot.report_time_ns = now_ns;
    snapshot.devices.reserve(entries.size());

    for (const auto& entry : entries)
    {
        auto device = std::make_shared<core::PluginDeviceStatusT>();
        device->path = entry.path;
        device->state = entry.state;
        device->reason = entry.reason;
        device->error = entry.error;
        snapshot.devices.push_back(std::move(device));
    }

    flatbuffers::FlatBufferBuilder builder(core::PluginDeviceStatusTracker::DEFAULT_MAX_FLATBUFFER_SIZE);
    builder.Finish(core::PluginDeviceStatusSnapshot::Pack(builder, &snapshot));
    if (builder.GetSize() > core::PluginDeviceStatusTracker::DEFAULT_MAX_FLATBUFFER_SIZE)
    {
        throw std::length_error("PluginDeviceStatusPublisher snapshot exceeds the tracker maximum size");
    }

    if (pusher_)
    {
        pusher_->push_buffer(builder.GetBufferPointer(), builder.GetSize(), now_ns, now_ns);
    }
    else
    {
        push_buffer_(builder.GetBufferPointer(), builder.GetSize(), now_ns, now_ns);
    }
}

} // namespace plugin_utils
