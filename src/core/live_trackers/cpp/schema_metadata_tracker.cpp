// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/live_trackers/schema_metadata_tracker.hpp"

#include <schema/tracker_metadata.hpp>

#include <XR_NVX1_tensor_data.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

SchemaTrackerConfig schema_metadata_tracker_config(const std::string& collection_id)
{
    std::string metadata_collection_id = tracker_metadata_collection_id(collection_id);
    // A pusher truncates a longer name, so the reader would wait for a collection that never appears.
    if (metadata_collection_id.size() >= XR_MAX_TENSOR_IDENTIFIER_SIZE)
    {
        throw std::invalid_argument("metadata collection id '" + metadata_collection_id +
                                    "' does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE");
    }
    return SchemaTrackerConfig{ .collection_id = std::move(metadata_collection_id),
                                .max_flatbuffer_size = TRACKER_METADATA_MAX_SIZE,
                                .tensor_identifier = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER),
                                .localized_name = "Tracker metadata" };
}

bool same_tracker_metadata(std::span<const uint8_t> a, std::span<const uint8_t> b)
{
    if (a.size() > b.size())
    {
        std::swap(a, b);
    }
    return std::equal(a.begin(), a.end(), b.begin()) &&
           std::all_of(b.begin() + a.size(), b.end(), [](uint8_t byte) { return byte == 0; });
}

} // namespace core
