// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_pusher.hpp"

#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/serialized.hpp>
#include <schema/tracker_metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Publishes a tracker's metadata as a loop of chunks, a few per frame.
 *
 * Pairs with a generated tracker whose manifest entry sets `metadata`. The chunks go to
 * tracker_metadata_collection_id(collection_id), next to the data collection. Call push_next()
 * every frame for as long as the data collection exists: a consumer that connects late, or
 * drops samples, completes the metadata on a later pass of the loop.
 *
 * The metadata is fixed for the life of this object. Different metadata needs a new data
 * collection too, or consumers that already hold the old one withhold the data.
 */
class SchemaMetadataPusher
{
public:
    static std::vector<std::string> get_required_extensions()
    {
        return SchemaPusher::get_required_extensions();
    }

    /*!
     * @param collection_id The data collection the metadata describes.
     * @param metadata The encoded metadata table, e.g. `pack<TactileShape>(native).buffer()`.
     * @param max_chunk_size Encoded chunk size; at most TRACKER_METADATA_CHUNK_SIZE, which is
     *        what consumers accept.
     * @throws std::invalid_argument on an empty or oversized buffer, or a bad chunk size.
     */
    SchemaMetadataPusher(const OpenXRSessionHandles& handles,
                         const std::string& collection_id,
                         std::span<const uint8_t> metadata,
                         size_t max_chunk_size = TRACKER_METADATA_CHUNK_SIZE);

    /*!
     * @brief Pushes the next `count` chunks of the loop.
     * @param sample_time_local_common_clock_ns Push time in system monotonic nanoseconds.
     */
    void push_next(int64_t sample_time_local_common_clock_ns, size_t count = 1);

    //! Frames of push_next(_, 1) one pass over the metadata takes.
    size_t chunk_count() const
    {
        return m_chunks.size();
    }

private:
    std::vector<Serialized<TrackerMetadataChunk>> m_chunks;
    SchemaPusher m_pusher;
    size_t m_next = 0;
};

} // namespace core
