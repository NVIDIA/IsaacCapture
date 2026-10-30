// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pusherio/schema_pusher.hpp"

#include "inc/pusherio/openxr_schema_push_channel.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace core
{

std::vector<std::string> SchemaPusher::get_required_extensions()
{
    return OpenXRSchemaPushChannel::get_required_extensions();
}

SchemaPusher::SchemaPusher(const OpenXRSessionHandles& handles, SchemaPusherConfig config)
    : SchemaPusher(std::make_unique<OpenXRSchemaPushChannel>(handles, std::move(config)))
{
}

SchemaPusher::SchemaPusher(std::unique_ptr<ISchemaPushChannel> channel) : m_channel(std::move(channel))
{
    if (!m_channel)
    {
        throw std::invalid_argument("SchemaPusher requires a schema push channel");
    }
}

SchemaPusher::~SchemaPusher() = default;

void SchemaPusher::push_buffer(const uint8_t* buffer,
                               size_t size,
                               int64_t sample_time_local_common_clock_ns,
                               int64_t sample_time_raw_device_clock_ns)
{
    const size_t max_flatbuffer_size = m_channel->config().max_flatbuffer_size;
    if (size > max_flatbuffer_size)
    {
        throw std::runtime_error("Serialized data size (" + std::to_string(size) +
                                 " bytes) exceeds max_flatbuffer_size (" + std::to_string(max_flatbuffer_size) +
                                 " bytes)");
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
    return m_channel->config();
}

} // namespace core
