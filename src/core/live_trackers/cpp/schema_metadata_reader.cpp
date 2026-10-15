// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/live_trackers/schema_metadata_reader.hpp"

#include <XR_NVX1_tensor_data.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace core
{

namespace
{

SchemaTrackerConfig metadata_reader_config(const std::string& collection_id)
{
    std::string metadata_collection_id = tracker_metadata_collection_id(collection_id);
    // The pusher truncates longer identifiers, so this reader would look for another name.
    if (metadata_collection_id.size() >= XR_MAX_TENSOR_IDENTIFIER_SIZE)
    {
        throw std::invalid_argument("SchemaMetadataReader: collection id '" + metadata_collection_id +
                                    "' does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE");
    }
    return SchemaTrackerConfig{ .collection_id = std::move(metadata_collection_id),
                                .max_flatbuffer_size = TRACKER_METADATA_SAMPLE_SIZE,
                                .tensor_identifier = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER),
                                .localized_name = "Tracker metadata" };
}

} // namespace

SchemaMetadataReaderBase::SchemaMetadataReaderBase(const OpenXRSessionHandles& handles, const std::string& collection_id)
    : m_reader(handles, metadata_reader_config(collection_id))
{
}

std::vector<uint8_t> SchemaMetadataReaderBase::drain()
{
    m_samples.clear();
    m_reader.read_all_samples(m_samples);
    // Every sample carries the whole table, padded to the sample size with zeros the
    // FlatBuffer does not reach; the latest one is enough.
    return m_samples.empty() ? std::vector<uint8_t>() : std::move(m_samples.back().buffer);
}

} // namespace core
