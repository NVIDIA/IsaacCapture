// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_tracker_base.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/serialized.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace core
{

/*!
 * @brief Reader config for the metadata of the data collection `collection_id`.
 * @throws std::invalid_argument when the metadata collection's name does not fit
 *         XR_MAX_TENSOR_IDENTIFIER_SIZE.
 */
SchemaTrackerConfig tracker_metadata_reader_config(const std::string& collection_id);

//! Whether two metadata samples hold the same table. Each producer zero-pads its samples to a
//! size of its choosing, so the padding is not compared.
bool same_tracker_metadata(std::span<const uint8_t> a, std::span<const uint8_t> b);

/*!
 * @brief The metadata of a generated tracker whose manifest entry sets `metadata`.
 *
 * The first metadata received is adopted, recorded, and kept for the life of this reader. The
 * tracker's data may be published and recorded while the producer's metadata is confirmed: from a
 * sample equal to the adopted one until the metadata collection goes away. A sample that differs
 * stops the data for the rest of the session.
 */
template <typename MetadataT>
class SchemaMetadataReader
{
public:
    using Channels = McapTrackerChannels<MetadataT>;

    //! @param mcap_channels Non-owning; channel 0 receives the adopted metadata. Null when not recording.
    SchemaMetadataReader(const OpenXRSessionHandles& handles, const std::string& collection_id, Channels* mcap_channels)
        : m_reader(handles, tracker_metadata_reader_config(collection_id)), m_mcap_channels(mcap_channels)
    {
    }

    /*!
     * @brief Reads the pending metadata. Call once per tick, before reading the data.
     * @return Whether the data may be published and recorded this tick.
     * @throws std::runtime_error when a sample is not a valid MetadataT.
     */
    bool update()
    {
        if (m_mismatched)
        {
            return false;
        }

        std::vector<SchemaTrackerBase::SampleResult> samples;
        const bool present = m_reader.read_all_samples(samples);
        for (auto& sample : samples)
        {
            if (!accept(sample))
            {
                return false;
            }
        }

        if (!present)
        {
            // The producer is gone; whichever comes next confirms its own metadata.
            m_confirmed = false;
        }
        else if (!samples.empty())
        {
            m_confirmed = true;
        }
        return m_confirmed;
    }

    //! Empty until the first metadata arrives, then fixed.
    const Serialized<MetadataT>& get() const
    {
        return m_metadata;
    }

private:
    //! Adopts the first sample. Returns false, for good, once a sample differs from it.
    bool accept(SchemaTrackerBase::SampleResult& sample)
    {
        const std::string& collection_id = m_reader.config().collection_id;
        flatbuffers::Verifier verifier(sample.buffer.data(), sample.buffer.size());
        if (!verifier.VerifyBuffer<MetadataT>(nullptr))
        {
            throw std::runtime_error(collection_id + ": metadata sample is not a valid " +
                                     MetadataT::GetFullyQualifiedName());
        }

        if (!m_metadata)
        {
            const int64_t log_time_ns = sample.timestamp.available_time_local_common_clock();
            m_metadata = Serialized<MetadataT>::adopt(std::move(sample.buffer));
            if (m_mcap_channels != nullptr)
            {
                m_mcap_channels->write(0, m_metadata, log_time_ns);
            }
            m_logger->info("{}: adopted {} B of {}", collection_id, m_metadata.buffer().size(),
                           MetadataT::GetFullyQualifiedName());
            return true;
        }

        if (same_tracker_metadata(m_metadata.buffer(), sample.buffer))
        {
            return true;
        }

        m_mismatched = true;
        m_confirmed = false;
        m_logger->error("{}: received a {} that differs from the one adopted; the data stays withheld this session",
                        collection_id, MetadataT::GetFullyQualifiedName());
        return false;
    }

    SchemaTrackerBase m_reader;
    Channels* m_mcap_channels;
    Serialized<MetadataT> m_metadata;
    bool m_confirmed = false;
    bool m_mismatched = false;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.core.SchemaMetadataReader");
};

} // namespace core
