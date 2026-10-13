// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pusherio/schema_metadata_pusher.hpp"

#include <XR_NVX1_tensor_data.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

namespace
{

std::vector<uint8_t> checked_metadata(std::span<const uint8_t> metadata)
{
    if (metadata.empty() || metadata.size() > TRACKER_METADATA_SAMPLE_SIZE)
    {
        throw std::invalid_argument("SchemaMetadataPusher: metadata size " + std::to_string(metadata.size()) +
                                    " is outside 1.." + std::to_string(TRACKER_METADATA_SAMPLE_SIZE));
    }
    return std::vector<uint8_t>(metadata.begin(), metadata.end());
}

SchemaPusherConfig metadata_pusher_config(const std::string& collection_id)
{
    std::string metadata_collection_id = tracker_metadata_collection_id(collection_id);
    // SchemaPusher truncates longer identifiers, and the reader would then look for another name.
    if (metadata_collection_id.size() >= XR_MAX_TENSOR_IDENTIFIER_SIZE)
    {
        throw std::invalid_argument("SchemaMetadataPusher: collection id '" + metadata_collection_id +
                                    "' does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE");
    }
    return SchemaPusherConfig{ .collection_id = std::move(metadata_collection_id),
                               .max_flatbuffer_size = TRACKER_METADATA_SAMPLE_SIZE,
                               .tensor_identifier = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER),
                               .localized_name = "Tracker metadata" };
}

} // namespace

SchemaMetadataPusher::SchemaMetadataPusher(const OpenXRSessionHandles& handles,
                                           const std::string& collection_id,
                                           std::span<const uint8_t> metadata)
    : m_metadata(checked_metadata(metadata)), m_pusher(handles, metadata_pusher_config(collection_id))
{
}

void SchemaMetadataPusher::push(int64_t sample_time_local_common_clock_ns)
{
    m_pusher.push_buffer(
        m_metadata.data(), m_metadata.size(), sample_time_local_common_clock_ns, sample_time_local_common_clock_ns);
}

} // namespace core
