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
#include <span>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Untyped half of SchemaMetadataReader: reads the metadata collection.
 */
class SchemaMetadataReaderBase
{
public:
    /*!
     * @param collection_id The data collection; the metadata is read from
     *        tracker_metadata_collection_id(collection_id).
     * @throws std::invalid_argument when that name does not fit XR_MAX_TENSOR_IDENTIFIER_SIZE.
     */
    SchemaMetadataReaderBase(const OpenXRSessionHandles& handles, const std::string& collection_id);

protected:
    //! Drains the pending samples. Returns the latest metadata buffer, or an empty one.
    std::vector<uint8_t> drain();

    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.core.SchemaMetadataReader");

private:
    SchemaTrackerBase m_reader;
    std::vector<SchemaTrackerBase::SampleResult> m_samples;
};

/*!
 * @brief Live metadata of a tracker whose manifest entry sets `metadata`.
 *
 * The first metadata received is adopted, recorded once, and kept for the life of this reader.
 * After that, each metadata sample is checked against it: the data may be published and
 * recorded only while verified(). Losing the data collection clears verified() until the
 * producer that comes back publishes the same metadata again.
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
        std::vector<uint8_t> latest = drain();
        if (latest.empty())
        {
            return;
        }

        // A complete candidate supersedes the previous verification result. Restore it only
        // after the candidate is both a valid table and the metadata already adopted here.
        m_verified = false;
        flatbuffers::Verifier verifier(latest.data(), latest.size());
        if (!verifier.VerifyBuffer<MetadataT>(nullptr))
        {
            m_logger->warn("received metadata is not a valid {}", MetadataT::GetFullyQualifiedName());
            return;
        }

        if (!m_metadata)
        {
            m_metadata = Serialized<MetadataT>::adopt(std::move(latest));
            m_verified = true;
            if (m_mcap_channels != nullptr)
            {
                m_mcap_channels->write(0, m_metadata, monotonic_time_ns);
            }
            m_logger->info("adopted {} B of {}", m_metadata.buffer().size(), MetadataT::GetFullyQualifiedName());
            return;
        }

        const std::span<const uint8_t> adopted = m_metadata.buffer();
        m_verified = std::equal(adopted.begin(), adopted.end(), latest.begin(), latest.end());
        if (!m_verified && !m_warned_mismatch)
        {
            m_logger->error("producer now publishes a different {} than the one adopted; its data is withheld",
                            MetadataT::GetFullyQualifiedName());
            m_warned_mismatch = true;
        }
    }

    //! Reports whether the data collection was present this tick.
    void set_data_present(bool present)
    {
        if (!present)
        {
            m_verified = false;
        }
    }

    bool verified() const
    {
        return m_verified;
    }

    //! Empty until the first metadata arrives, then fixed.
    const Serialized<MetadataT>& get() const
    {
        return m_metadata;
    }

private:
    Channels* m_mcap_channels;
    Serialized<MetadataT> m_metadata;
    bool m_verified = false;
    bool m_warned_mismatch = false;
};

} // namespace core
