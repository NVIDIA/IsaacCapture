// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_pusher.hpp"

#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/tracker_metadata.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Publishes a tracker's metadata as samples of its metadata collection.
 *
 * Pairs with a generated tracker whose manifest entry sets `metadata`. The samples go to
 * tracker_metadata_collection_id(collection_id), next to the data collection. Call push()
 * periodically for as long as the data collection exists, so a consumer that connects late, or
 * drops samples, still receives the metadata.
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
     * @throws std::invalid_argument when it is empty or larger than TRACKER_METADATA_SAMPLE_SIZE.
     */
    SchemaMetadataPusher(const OpenXRSessionHandles& handles,
                         const std::string& collection_id,
                         std::span<const uint8_t> metadata);

    /*!
     * @brief Pushes the metadata once.
     * @param sample_time_local_common_clock_ns Push time in system monotonic nanoseconds.
     */
    void push(int64_t sample_time_local_common_clock_ns);

private:
    std::vector<uint8_t> m_metadata;
    SchemaPusher m_pusher;
};

} // namespace core
