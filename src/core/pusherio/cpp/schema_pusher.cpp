// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pusherio/schema_pusher.hpp"

#include "inc/pusherio/openxr_schema_push_channel.hpp"

#include <stdexcept>
#include <utility>

namespace core
{

namespace
{

const SchemaPusherConfig& require_channel_config(const std::unique_ptr<ISchemaPushChannel>& channel)
{
    if (!channel)
    {
        throw std::invalid_argument("SchemaPusher requires a schema push channel");
    }
    return channel->config();
}

} // namespace

SchemaPusher::SchemaPusher(const OpenXRSessionHandles& handles, SchemaPusherConfig config)
    : SchemaPusher(make_openxr_schema_push_channel(handles, std::move(config)))
{
}

SchemaPusher::SchemaPusher(std::unique_ptr<ISchemaPushChannel> channel)
    : m_config(require_channel_config(channel)), m_channel(std::move(channel))
{
}

SchemaPusher::~SchemaPusher() = default;

void SchemaPusher::push_buffer(const uint8_t* buffer,
                               size_t size,
                               int64_t sample_time_local_common_clock_ns,
                               int64_t sample_time_raw_device_clock_ns)
{
    if (size > m_config.max_flatbuffer_size)
    {
        throw std::runtime_error("Serialized data size (" + std::to_string(size) +
                                 " bytes) exceeds max_flatbuffer_size (" +
                                 std::to_string(m_config.max_flatbuffer_size) + " bytes)");
    }
    if (sample_time_raw_device_clock_ns < 0)
    {
        throw std::runtime_error("push_buffer: sample_time_raw_device_clock_ns is negative (" +
                                 std::to_string(sample_time_raw_device_clock_ns) + ")");
    }
    if (buffer == nullptr && size != 0)
    {
        throw std::invalid_argument("push_buffer: buffer is null for a non-empty sample");
    }

    m_channel->push_buffer(buffer, size, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns);
}

const SchemaPusherConfig& SchemaPusher::config() const
{
    return m_config;
}

} // namespace core
