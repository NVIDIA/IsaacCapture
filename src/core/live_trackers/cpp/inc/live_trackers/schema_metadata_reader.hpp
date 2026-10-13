// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_tracker_base.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/serialized.hpp>
#include <schema/tracker_metadata.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Untyped half of SchemaMetadataReader: reads the metadata collection and reassembles
 *        its chunks.
 */
class SchemaMetadataReaderBase
{
public:
    /*!
     * @param collection_id The data collection; the chunks are read from
     *        tracker_metadata_collection_id(collection_id).
     * @throws std::invalid_argument when that name does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE.
     */
    SchemaMetadataReaderBase(const OpenXRSessionHandles& handles, const std::string& collection_id);

protected:
    struct DrainResult
    {
        std::vector<uint8_t> completed;
        std::optional<uint64_t> last_observed_hash;
    };

    //! Drains pending chunks and reports the hash of the last well-formed chunk it saw.
    DrainResult drain();

    //! Drops the assembly in progress, so the next buffer comes only from chunks read after this.
    void restart();

    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.core.SchemaMetadataReader");

private:
    SchemaTrackerBase m_reader;
    TrackerMetadataAssembler m_assembler;
    std::vector<SchemaTrackerBase::SampleResult> m_samples;
    bool m_warned_rejection = false;
};

/*!
 * @brief Live metadata of a tracker whose manifest entry sets `metadata`.
 *
 * The first complete metadata is adopted, recorded once, and kept for the life of this reader.
 * After that, each complete pass of the producer's chunk loop is checked against it: the data
 * may be published and recorded only while verified(). Losing the data collection clears
 * verified() until the producer that comes back publishes the same metadata again.
 */
template <typename MetadataT>
class SchemaMetadataReader : public SchemaMetadataReaderBase
{
public:
    using Channels = McapTrackerChannels<MetadataT>;

    //! @param mcap_channels Non-owning; channel 0 receives the adopted metadata. Null when not recording.
    SchemaMetadataReader(const OpenXRSessionHandles& handles, const std::string& collection_id, Channels* mcap_channels)
        : SchemaMetadataReaderBase(handles, collection_id), m_mcap_channels(mcap_channels)
    {
    }

    //! Call once per tick, before reading the data collection.
    void update(int64_t monotonic_time_ns)
    {
        DrainResult result = drain();
        const bool ended_with_unexpected_hash =
            m_metadata && result.last_observed_hash && *result.last_observed_hash != m_metadata_hash;
        if (ended_with_unexpected_hash)
        {
            withhold_for_mismatch();
        }
        std::vector<uint8_t> completed = std::move(result.completed);
        if (completed.empty())
        {
            return;
        }

        // A complete candidate supersedes the previous verification result. Restore it only
        // after the candidate is both a valid table and the metadata already adopted here.
        m_verified = false;
        flatbuffers::Verifier verifier(completed.data(), completed.size());
        if (!verifier.VerifyBuffer<MetadataT>(nullptr))
        {
            m_logger->warn("reassembled metadata is not a valid {}", MetadataT::GetFullyQualifiedName());
            return;
        }

        if (!m_metadata)
        {
            m_metadata = Serialized<MetadataT>::adopt(std::move(completed));
            m_metadata_hash = tracker_metadata_hash(m_metadata.buffer());
            m_verified = true;
            if (m_mcap_channels != nullptr)
            {
                m_mcap_channels->write(0, m_metadata, monotonic_time_ns);
            }
            m_logger->info("adopted {} B of {}", m_metadata.buffer().size(), MetadataT::GetFullyQualifiedName());
            return;
        }

        const std::span<const uint8_t> adopted = m_metadata.buffer();
        if (ended_with_unexpected_hash || !std::equal(adopted.begin(), adopted.end(), completed.begin(), completed.end()))
        {
            withhold_for_mismatch();
        }
        else
        {
            m_verified = true;
        }
    }

    //! Reports whether the data collection was present this tick.
    void set_data_present(bool present)
    {
        if (!present)
        {
            m_verified = false;
            restart();
        }
    }

    bool verified() const
    {
        return m_verified;
    }

    //! Empty until the first complete metadata, then fixed.
    const Serialized<MetadataT>& get() const
    {
        return m_metadata;
    }

private:
    void withhold_for_mismatch()
    {
        m_verified = false;
        if (!m_warned_mismatch)
        {
            m_logger->error("producer now publishes a different {} than the one adopted; its data is withheld",
                            MetadataT::GetFullyQualifiedName());
            m_warned_mismatch = true;
        }
    }

    Channels* m_mcap_channels;
    Serialized<MetadataT> m_metadata;
    uint64_t m_metadata_hash = 0;
    bool m_verified = false;
    bool m_warned_mismatch = false;
};

} // namespace core
