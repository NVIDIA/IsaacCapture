// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_tracker_base.hpp"

#include <flatbuffers/flatbuffers.h>
#include <mcap/tracker_channels.hpp>
#include <schema/serialized.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace core
{

/**
 * @brief Typed SchemaTracker that optionally records to MCAP.
 *
 * Wraps SchemaTrackerBase with FlatBuffer type knowledge so that each sample
 * read from the tensor can be automatically written to an MCAP channel.
 *
 * @tparam RecordT    Root written to MCAP: a record wrapper (e.g. Generic3AxisPedalOutputRecord),
 *                    or DataTableT itself for a table recorded without one.
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

        if (mcap_channels_)
        {
            Serialized<RecordT> last_record;
            int64_t last_log_time = 0;
            for (const auto& sample : samples_)
            {
                Serialized<RecordT> record = to_record(sample);
                if (!record)
                {
                    continue;
                }
                last_record = std::move(record);
                last_log_time = sample.timestamp.available_time_local_common_clock();
                mcap_channels_->write(mcap_channel_index_, last_record, last_log_time);
            }

            // The tracked channel marks the final sample of the tick, which is the record
            // just written: same payload, same timestamp, so the bytes are identical and
            // this is a second write rather than a second encode. Empty when every sample
            // failed to resolve a root, the one case where there is no final sample.
            if (mcap_channel_tracked_index_ && last_record)
            {
                mcap_channels_->write(*mcap_channel_tracked_index_, last_record, last_log_time);
            }
        }

        // Adopt the final sample's bytes rather than copying them: the wire type is the
        // published type. Each tick owns its own buffer, so a consumer still holding last
        // tick's handle keeps last tick's values.
        out = Serialized<DataTableT>::adopt(std::move(samples_.back().buffer));
    }

private:
    //! What MCAP stores for one sample: its Record, or the sample's own bytes when RecordT is
    //! DataTableT. Empty when the sample has no root.
    static Serialized<RecordT> to_record(const SampleResult& sample)
    {
        if constexpr (std::is_same_v<RecordT, DataTableT>)
        {
            return Serialized<RecordT>::adopt(std::vector<uint8_t>(sample.buffer));
        }
        else
        {
            auto fb = flatbuffers::GetRoot<DataTableT>(sample.buffer.data());
            if (!fb)
            {
                return Serialized<RecordT>();
            }

            NativeDataT latest;
            fb->UnPackTo(&latest);
            return pack_record<RecordT>(&latest, sample.timestamp);
        }
    }

    Channels* mcap_channels_;
    size_t mcap_channel_index_;
    std::optional<size_t> mcap_channel_tracked_index_;
    std::vector<SampleResult> samples_;
};

} // namespace core
