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

std::vector<Serialized<TrackerMetadataChunk>> split_checked(std::span<const uint8_t> metadata, size_t max_chunk_size)
{
    if (max_chunk_size > TRACKER_METADATA_CHUNK_SIZE)
    {
        throw std::invalid_argument("SchemaMetadataPusher: max_chunk_size " + std::to_string(max_chunk_size) +
                                    " exceeds TRACKER_METADATA_CHUNK_SIZE, which consumers reject");
    }
    return split_tracker_metadata(metadata, max_chunk_size);
}

SchemaPusherConfig metadata_pusher_config(const std::string& collection_id, size_t max_chunk_size)
{
    std::string metadata_collection_id = tracker_metadata_collection_id(collection_id);
    // SchemaPusher truncates longer identifiers, and the reader would then look for another name.
    if (metadata_collection_id.size() >= XR_MAX_TENSOR_IDENTIFIER_SIZE)
    {
        throw std::invalid_argument("SchemaMetadataPusher: collection id '" + metadata_collection_id +
                                    "' does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE");
    }
    return SchemaPusherConfig{ .collection_id = std::move(metadata_collection_id),
                               .max_flatbuffer_size = max_chunk_size,
                               .tensor_identifier = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER),
                               .localized_name = "Tracker metadata" };
}

} // namespace

SchemaMetadataPusher::SchemaMetadataPusher(const OpenXRSessionHandles& handles,
                                           const std::string& collection_id,
                                           std::span<const uint8_t> metadata,
                                           size_t max_chunk_size)
    : m_chunks(split_checked(metadata, max_chunk_size)),
      m_pusher(handles, metadata_pusher_config(collection_id, max_chunk_size))
{
}

void SchemaMetadataPusher::push_next(int64_t sample_time_local_common_clock_ns, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        const std::span<const uint8_t> bytes = m_chunks[m_next].buffer();
        m_pusher.push_buffer(
            bytes.data(), bytes.size(), sample_time_local_common_clock_ns, sample_time_local_common_clock_ns);
        m_next = (m_next + 1) % m_chunks.size();
    }
}

} // namespace core
