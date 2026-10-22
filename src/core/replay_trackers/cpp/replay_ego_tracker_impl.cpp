// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_ego_tracker_impl.hpp"

#include <schema/orbbec_ego_camera_bfbs_generated.h>
#include <schema/orbbec_ego_imu_bfbs_generated.h>

#include <stdexcept>
#include <utility>

namespace core
{

ReplayEgoFrameMetadataTrackerImpl::ReplayEgoFrameMetadataTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                                     std::string_view base_name,
                                                                     const std::vector<std::string>& channels,
                                                                     const RecordedSchemas& recorded)
    : viewers_(std::make_unique<McapTrackerViewers<OrbbecEgoFrameMetadataRecord>>(
          std::move(reader), base_name, channels, recorded)),
      streams_(channels.size()),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayEgoFrameMetadataTrackerImpl")),
      warned_no_data_(channels.size(), false)
{
}

void ReplayEgoFrameMetadataTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (size_t index = 0; index < streams_.size(); ++index)
    {
        auto record = viewers_->read(index);
        streams_[index] = record ? record.narrow(record->data()) : Serialized<OrbbecEgoFrameMetadata>{};
        if (record)
        {
            warned_no_data_[index] = false;
        }
        else if (!warned_no_data_[index])
        {
            logger_->warn("no data (EOF or gap), stream_index={}", index);
            warned_no_data_[index] = true;
        }
    }
}

const Serialized<OrbbecEgoFrameMetadata>& ReplayEgoFrameMetadataTrackerImpl::get_stream_data(size_t index) const
{
    if (index >= streams_.size())
        throw std::out_of_range("OrbbecEgoFrameMetadataTracker: invalid stream index");
    return streams_[index];
}

ReplayEgoImuTrackerImpl::ReplayEgoImuTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                                                 std::string_view base_name,
                                                 const std::vector<std::string>& channels,
                                                 const RecordedSchemas& recorded)
    : viewers_(std::make_unique<McapTrackerViewers<OrbbecEgoImuBatchRecord>>(
          std::move(reader), base_name, channels, recorded)),
      streams_(channels.size()),
      logger_(isaaccapture::Logger::get("isaaccapture.core.ReplayEgoImuTrackerImpl")),
      warned_no_data_(channels.size(), false)
{
}

void ReplayEgoImuTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    for (size_t index = 0; index < streams_.size(); ++index)
    {
        auto record = viewers_->read(index);
        streams_[index] = record ? record.narrow(record->data()) : Serialized<OrbbecEgoImuBatch>{};
        if (record)
        {
            warned_no_data_[index] = false;
        }
        else if (!warned_no_data_[index])
        {
            logger_->warn("no data (EOF or gap), stream_index={}", index);
            warned_no_data_[index] = true;
        }
    }
}

const Serialized<OrbbecEgoImuBatch>& ReplayEgoImuTrackerImpl::get_stream_data(size_t index) const
{
    if (index >= streams_.size())
        throw std::out_of_range("OrbbecEgoImuTracker: invalid stream index");
    return streams_[index];
}

} // namespace core
