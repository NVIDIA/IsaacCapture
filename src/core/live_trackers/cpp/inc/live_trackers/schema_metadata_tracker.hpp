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
 * @brief Config of the SchemaMetadataTracker for the data collection `collection_id`.
 * @throws std::invalid_argument when the metadata collection's name does not fit
 *         XR_MAX_TENSOR_IDENTIFIER_SIZE.
 */
SchemaTrackerConfig schema_metadata_tracker_config(const std::string& collection_id);

//! Whether two metadata samples hold the same table. Each producer zero-pads its samples to a
//! size of its choosing, so the padding is not compared.
bool same_tracker_metadata(std::span<const uint8_t> a, std::span<const uint8_t> b);

/*!
 * @brief Reads the metadata of a generated tracker whose manifest entry sets `metadata`.
 *
 * The first metadata received is adopted, recorded, and kept for the life of this tracker. The
 * tracker's data may be published and recorded while the producer's metadata is confirmed: from a
 * sample equal to the adopted one until the metadata collection goes away. A sample that differs
 * stops the data for the rest of the session.
 */
template <typename MetadataT>
class SchemaMetadataTracker : public SchemaTrackerBase
{
public:
    using Channels = McapTrackerChannels<MetadataT>;

    /*!
     * @param collection_id The data collection; see schema_metadata_tracker_config().
     * @param mcap_channels Non-owning; channel 0 receives the adopted metadata. Null when not recording.
     */
    SchemaMetadataTracker(const OpenXRSessionHandles& handles, const std::string& collection_id, Channels* mcap_channels)
        : SchemaTrackerBase(handles, schema_metadata_tracker_config(collection_id)), mcap_channels_(mcap_channels)
    {
    }

    /*!
     * @brief Reads the pending metadata. Call once per tick, before reading the data.
     * @return Whether the data may be published and recorded this tick.
     * @throws std::runtime_error when a sample is not a valid MetadataT.
     */
    bool update()
    {
        if (mismatched_)
        {
            return false;
        }

        std::vector<SampleResult> samples;
        const bool present = read_all_samples(samples);
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
            confirmed_ = false;
        }
        else if (!samples.empty())
        {
            confirmed_ = true;
        }
        return confirmed_;
    }

    //! Empty until the first metadata arrives, then fixed.
    const Serialized<MetadataT>& get() const
    {
        return metadata_;
    }

private:
    //! Adopts the first sample. Returns false, for good, once a sample differs from it.
    bool accept(SampleResult& sample)
    {
        const std::string& collection_id = config().collection_id;
        flatbuffers::Verifier verifier(sample.buffer.data(), sample.buffer.size());
        if (!verifier.VerifyBuffer<MetadataT>(nullptr))
        {
            throw std::runtime_error(collection_id + ": metadata sample is not a valid " +
                                     MetadataT::GetFullyQualifiedName());
        }

        if (!metadata_)
        {
            const int64_t log_time_ns = sample.timestamp.available_time_local_common_clock();
            metadata_ = Serialized<MetadataT>::adopt(std::move(sample.buffer));
            if (mcap_channels_ != nullptr)
            {
                mcap_channels_->write(0, metadata_, log_time_ns);
            }
            logger_->info(
                "{}: adopted {} B of {}", collection_id, metadata_.buffer().size(), MetadataT::GetFullyQualifiedName());
            return true;
        }

        if (same_tracker_metadata(metadata_.buffer(), sample.buffer))
        {
            return true;
        }

        mismatched_ = true;
        logger_->error("{}: received a {} that differs from the one adopted; the data stays withheld this session",
                       collection_id, MetadataT::GetFullyQualifiedName());
        return false;
    }

    Channels* mcap_channels_;
    Serialized<MetadataT> metadata_;
    bool confirmed_ = false;
    bool mismatched_ = false;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.SchemaMetadataTracker");
};

} // namespace core
