// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_tracker_base.hpp"

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <mcap/tracker_channels.hpp>
#include <schema/serialized.hpp>

#include <memory>
#include <optional>

namespace core
{

namespace detail
{

template <typename DataTableT>
const DataTableT* verified_flatbuffer_root(const std::vector<uint8_t>& buffer)
{
    if (buffer.empty())
    {
        return nullptr;
    }

    // Tensor bytes cross a runtime/process trust boundary. GetRoot() only computes
    // a pointer, so verify the complete typed buffer before any field access or adoption.
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!verifier.VerifyBuffer<DataTableT>(nullptr))
    {
        return nullptr;
    }
    return flatbuffers::GetRoot<DataTableT>(buffer.data());
}

} // namespace detail

/**
 * @brief Typed SchemaTracker that optionally records to MCAP.
 *
 * Wraps SchemaTrackerBase with FlatBuffer type knowledge so that each sample
 * read from the tensor can be automatically written to an MCAP channel.
 *
 * @tparam RecordT    FlatBuffer record wrapper (e.g. Generic3AxisPedalOutputRecord).
 * @tparam DataTableT FlatBuffer data table (e.g. Generic3AxisPedalOutput). This is both
 *                    the wire type and the type published to consumers.
 */
template <typename RecordT, typename DataTableT>
class SchemaTracker : public SchemaTrackerBase
{
public:
    using NativeDataT = typename DataTableT::NativeTableType;
    using Channels = McapTrackerChannels<RecordT>;

    /**
     * @param mcap_channels Non-owning pointer to the MCAP channel writer. Must outlive
     *        this SchemaTracker. Owned by the live tracker impl that also owns this
     *        SchemaTracker instance. Null when recording is disabled.
     * @param mcap_channel_index 0-based sub-channel index within mcap_channels
     *        used for per-sample recording.
     * @param mcap_channel_tracked_index If set, an additional write of only the final
     *        sample per update() call is made to this sub-channel index within the
     *        same mcap_channels. Unset to disable.
     */
    SchemaTracker(const OpenXRSessionHandles& handles,
                  SchemaTrackerConfig config,
                  Channels* mcap_channels = nullptr,
                  size_t mcap_channel_index = 0,
                  std::optional<size_t> mcap_channel_tracked_index = std::nullopt)
        : SchemaTrackerBase(handles, std::move(config)),
          mcap_channels_(mcap_channels),
          mcap_channel_index_(mcap_channel_index),
          mcap_channel_tracked_index_(mcap_channel_tracked_index)
    {
    }

    /**
     * @brief Read all pending samples; write each to MCAP if channels are set.
     *
     * The wire already carries `DataTableT`, which is exactly what consumers read, so
     * the final sample is published by taking ownership of its buffer -- no unpack and
     * no re-encode. Recording is the only reason to materialise a native, so samples are
     * unpacked solely when MCAP channels are attached.
     *
     * @param out Receives the final sample of this tick when any were read; left
     *            untouched when the collection is present but produced nothing (the
     *            last-known sample is retained); emptied when the collection is absent.
     * @throws std::runtime_error On critical OpenXR/tensor API failures propagated
     *         from SchemaTrackerBase.
     * @note Missing collection, temporary collection loss, and "no new sample"
     *       are treated as common non-fatal conditions and do not throw.
     */
    void update(Serialized<DataTableT>& out)
    {
        samples_.clear();
        bool present = read_all_samples(samples_);

        if (samples_.empty())
        {
            if (!present)
            {
                out.reset();
            }
            return;
        }

        SampleResult* latest_valid_sample = nullptr;
        Serialized<RecordT> last_record;
        for (auto& sample : samples_)
        {
            const auto* fb = detail::verified_flatbuffer_root<DataTableT>(sample.buffer);
            if (fb == nullptr)
            {
                continue;
            }

            latest_valid_sample = &sample;
            if (!mcap_channels_)
            {
                continue;
            }

            NativeDataT latest;
            fb->UnPackTo(&latest);

            last_record = pack_record<RecordT>(&latest, sample.timestamp);
            mcap_channels_->write(mcap_channel_index_, last_record);
        }

        if (mcap_channels_)
        {
            // The tracked channel marks the final sample of the tick, which is the record
            // just written: same payload, same timestamp, so the bytes are identical and
            // this is a second write rather than a second encode. Empty when every sample
            // failed to resolve a root, the one case where there is no final sample.
            if (mcap_channel_tracked_index_ && last_record)
            {
                mcap_channels_->write(*mcap_channel_tracked_index_, last_record);
            }
        }

        if (latest_valid_sample != nullptr)
        {
            // Adopt the final valid sample's bytes rather than copying them. Invalid
            // samples must not replace the last-known-good value held by the consumer.
            out = Serialized<DataTableT>::adopt(std::move(latest_valid_sample->buffer));
        }
    }

private:
    Channels* mcap_channels_;
    size_t mcap_channel_index_;
    std::optional<size_t> mcap_channel_tracked_index_;
    std::vector<SampleResult> samples_;
};

} // namespace core
