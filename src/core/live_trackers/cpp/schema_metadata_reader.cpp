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
                                .max_flatbuffer_size = TRACKER_METADATA_CHUNK_SIZE,
                                .tensor_identifier = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER),
                                .localized_name = "Tracker metadata" };
}

} // namespace

SchemaMetadataReaderBase::SchemaMetadataReaderBase(const OpenXRSessionHandles& handles, const std::string& collection_id)
    : m_reader(handles, metadata_reader_config(collection_id))
{
}

SchemaMetadataReaderBase::DrainResult SchemaMetadataReaderBase::drain()
{
    m_samples.clear();
    m_reader.read_all_samples(m_samples);

    DrainResult result;
    for (const auto& sample : m_samples)
    {
        flatbuffers::Verifier verifier(sample.buffer.data(), sample.buffer.size());
        if (!verifier.VerifyBuffer<TrackerMetadataChunk>(nullptr))
        {
            if (!m_warned_rejection)
            {
                m_logger->warn("dropping a sample that is not a TrackerMetadataChunk");
                m_warned_rejection = true;
            }
            continue;
        }

        const auto& chunk = *flatbuffers::GetRoot<TrackerMetadataChunk>(sample.buffer.data());
        const auto status = m_assembler.add(chunk);
        if (status != TrackerMetadataAssembler::Status::Rejected)
        {
            result.last_observed_hash = chunk.metadata_hash();
        }
        switch (status)
        {
        case TrackerMetadataAssembler::Status::Pending:
            break;
        case TrackerMetadataAssembler::Status::Complete:
            result.completed = m_assembler.take();
            m_warned_rejection = false;
            break;
        case TrackerMetadataAssembler::Status::Rejected:
            if (!m_warned_rejection)
            {
                m_logger->warn("dropping a metadata chunk: {}", m_assembler.rejection());
                m_warned_rejection = true;
            }
            break;
        }
    }
    return result;
}

void SchemaMetadataReaderBase::restart()
{
    m_assembler.reset();
}

} // namespace core
